"""bench_detector.py -- inference cost of the detector of Section 6.3.1 for n = 1, 2, 3 context windows on the GPU
(Section 6.4.1, Tables 6.6 and 6.8, Figure 6.3). Run as one notebook cell on Kaggle (Tesla T4), version 6 of the
benchmark notebook, TensorFlow 2.20.0.

The cost of a forward pass does not depend on the values of the weights, so the backbone carries its ImageNet weights
and the proposal network and the head their initial values. The 19 frames are synthetic (smoothed noise) with the
sizes of the 19 CADP test videos after resizing to a shorter side of 600 pixels. Three forms are timed per setting:
  eager      the inference path of evaluate_cadp.py as written (detect_frame), 3 warm-up frames, 19 timed
  compiled   the same two Keras models wrapped in tf.function, 19 warm-up frames, 5 x 19 timed
  optimized  a head that pools all regions of a batch in one crop per context window and applies the dense layers
             in one matrix product (the weights copied from the head as written), 19 warm-up, 5 x 19 timed
Each timed frame is split into backbone and proposal network, proposal decoding and suppression (NumPy, CPU), head,
and final regression and suppression. Before timing, the optimized head is compared with the head as written on the
proposals of three frames (DETBENCH_EQ, largest absolute difference of the outputs).
Output: one line "DETBENCH {json}" per setting and form, and "DETBENCH_ALL {json}".
"""
try:
    import tensorflow
except Exception:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tensorflow[and-cuda]'])
SIZES = [(600, 750)] + [(600, 800)] * 6 + [(600, 820)] * 2 + [(600, 900), (600, 905)] + [(600, 1066)] * 8
import os, sys, json, time, math, glob, platform
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer

class ContextualRoiPooling(Layer):

    def __init__(self, pool_size, num_rois, n_contexts=3, step_stride=2, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = int(pool_size)
        self.num_rois = int(num_rois)
        self.n_contexts = int(n_contexts)
        self.step_stride = float(step_stride)
        self.nb_channels = None

    def build(self, input_shape):
        self.nb_channels = input_shape[0][-1]
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels)

    def call(self, x, mask=None):
        assert len(x) == 2, 'Expected [feature_map, rois]'
        img, rois = (x[0], x[1])
        shape = tf.shape(img)
        H = tf.cast(shape[1], tf.float32)
        W = tf.cast(shape[2], tf.float32)
        roi_outputs = []
        for roi_idx in range(self.num_rois):
            x0 = rois[0, roi_idx, 0]
            y0 = rois[0, roi_idx, 1]
            w0 = rois[0, roi_idx, 2]
            h0 = rois[0, roi_idx, 3]
            context_crops = []
            for i in range(self.n_contexts):
                pad = i * self.step_stride
                x1 = x0 - pad
                y1 = y0 - pad
                x2 = x0 + w0 + pad
                y2 = y0 + h0 + pad
                x1 = tf.maximum(0.0, tf.cast(x1, tf.float32))
                y1 = tf.maximum(0.0, tf.cast(y1, tf.float32))
                x2 = tf.minimum(W, tf.cast(x2, tf.float32))
                y2 = tf.minimum(H, tf.cast(y2, tf.float32))
                norm_box = tf.stack([y1 / H, x1 / W, y2 / H, x2 / W])
                norm_box = tf.reshape(norm_box, (1, 4))
                crop = tf.image.crop_and_resize(image=img, boxes=norm_box, box_indices=[0], crop_size=(self.pool_size, self.pool_size), method='bilinear')
                context_crops.append(crop[0])
            stacked = tf.stack(context_crops, axis=0)
            fused = tf.reduce_max(stacked, axis=0)
            roi_outputs.append(fused)
        out = tf.stack(roi_outputs, axis=0)
        out = tf.reshape(out, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'pool_size': self.pool_size, 'num_rois': self.num_rois, 'n_contexts': self.n_contexts, 'step_stride': self.step_stride})
        return cfg

class Config:
    anchor_scales = [64, 128, 256]
    anchor_ratios = [[1, 1], [1, 2], [2, 1]]
    im_size = 600
    rpn_stride = 16
    num_rois = 32
    rpn_min_overlap, rpn_max_overlap = (0.3, 0.7)
    classifier_min_overlap, classifier_max_overlap = (0.1, 0.5)
    class_mapping = {'vehicle': 0, 'two_wheeler': 1, 'person': 2, 'bg': 3}
    std_scaling = 4.0
    classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
    use_horizontal_flips = True
    rpn_num_regions = 256
    rpn_nms_thresh = 0.7
    freeze_batchnorm = True
C = Config()
NUM_ANCHORS = len(C.anchor_scales) * len(C.anchor_ratios)
NUM_CLASSES = len(C.class_mapping)

def make_anchors(fh, fw):
    xs = (np.arange(fw) + 0.5) * C.rpn_stride
    ys = (np.arange(fh) + 0.5) * C.rpn_stride
    cx, cy = np.meshgrid(xs, ys)
    anchors = []
    for s in C.anchor_scales:
        for rw, rh in C.anchor_ratios:
            aw, ah = (s * rw / math.sqrt(rw * rh), s * rh / math.sqrt(rw * rh))
            anchors.append(np.stack([cx - aw / 2, cy - ah / 2, cx + aw / 2, cy + ah / 2], axis=-1))
    return np.stack(anchors, axis=2)

def build_models(n_contexts, step_stride=2):
    backbone = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(None, None, 3))
    c4 = backbone.get_layer('conv4_block6_out').output
    feat = Model(backbone.input, c4, name='backbone')
    if C.freeze_batchnorm:
        for layer in feat.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
    img_in = layers.Input(shape=(None, None, 3))
    fmap = feat(img_in)
    x = layers.Conv2D(512, 3, padding='same', activation='relu', name='rpn_conv')(fmap)
    rpn_cls = layers.Conv2D(NUM_ANCHORS, 1, activation='sigmoid', name='rpn_out_class')(x)
    rpn_reg = layers.Conv2D(NUM_ANCHORS * 4, 1, activation='linear', name='rpn_out_regress')(x)
    model_rpn = Model(img_in, [rpn_cls, rpn_reg, fmap], name='rpn')
    feat_in = layers.Input(shape=(None, None, 1024))
    rois_in = layers.Input(shape=(C.num_rois, 4))
    pooled = ContextualRoiPooling(pool_size=7, num_rois=C.num_rois, n_contexts=n_contexts, step_stride=step_stride)([feat_in, rois_in])
    out = layers.TimeDistributed(layers.Flatten())(pooled)
    out = layers.TimeDistributed(layers.Dense(2048, activation='relu'))(out)
    out = layers.TimeDistributed(layers.Dropout(0.5))(out)
    out_cls = layers.TimeDistributed(layers.Dense(NUM_CLASSES, activation='softmax'), name='dense_class')(out)
    out_reg = layers.TimeDistributed(layers.Dense(4 * (NUM_CLASSES - 1), activation='linear'), name='dense_regress')(out)
    model_cls = Model([feat_in, rois_in], [out_cls, out_reg], name='head')
    return (model_rpn, model_cls)

def rpn_to_rois(rpn_cls, rpn_reg, fh, fw, top_n=300):
    anchors = make_anchors(fh, fw).reshape(-1, 4) / C.rpn_stride
    scores = rpn_cls.reshape(-1)
    t = rpn_reg.reshape(-1, 4) / C.std_scaling
    aw, ah = (anchors[:, 2] - anchors[:, 0], anchors[:, 3] - anchors[:, 1])
    acx, acy = (anchors[:, 0] + aw / 2, anchors[:, 1] + ah / 2)
    cx, cy = (t[:, 0] * aw + acx, t[:, 1] * ah + acy)
    w, h = (np.exp(np.clip(t[:, 2], -4, 4)) * aw, np.exp(np.clip(t[:, 3], -4, 4)) * ah)
    x1, y1 = (np.maximum(0, cx - w / 2), np.maximum(0, cy - h / 2))
    x2, y2 = (np.minimum(fw - 1, cx + w / 2), np.minimum(fh - 1, cy + h / 2))
    ok = (x2 > x1) & (y2 > y1)
    x1, y1, x2, y2, scores = (x1[ok], y1[ok], x2[ok], y2[ok], scores[ok])
    keep = nms(np.stack([x1, y1, x2, y2], 1), scores, C.rpn_nms_thresh, top_n)
    boxes = np.stack([x1[keep], y1[keep], x2[keep] - x1[keep], y2[keep] - y1[keep]], 1)
    return (boxes, scores[keep])

def nms(boxes, scores, thresh, max_boxes):
    order = scores.argsort()[::-1]
    keep = []
    x1, y1, x2, y2 = (boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
    areas = (x2 - x1) * (y2 - y1)
    while len(order) and len(keep) < max_boxes:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-06)
        order = order[1:][iou <= thresh]
    return np.array(keep, dtype=int)
INV_CLASS = {v: k for k, v in C.class_mapping.items()}

def apply_regr(x, y, w, h, tx, ty, tw, th):
    try:
        cx, cy = (x + w / 2.0, y + h / 2.0)
        cx1, cy1 = (tx * w + cx, ty * h + cy)
        w1, h1 = (math.exp(tw) * w, math.exp(th) * h)
        return (int(round(cx1 - w1 / 2.0)), int(round(cy1 - h1 / 2.0)), int(round(w1)), int(round(h1)))
    except (ValueError, OverflowError):
        return (x, y, w, h)

def detect_frame(model_rpn, model_cls, img, score_thresh=0.0, nms_thresh=0.5):
    import tensorflow as tf
    x = img[None]
    p_cls, p_reg, fmap = model_rpn(x, training=False)
    rois, _ = rpn_to_rois(p_cls.numpy()[0], p_reg.numpy()[0], fmap.shape[1], fmap.shape[2], top_n=300)
    by_class = {}
    for i in range(0, len(rois), C.num_rois):
        batch = rois[i:i + C.num_rois]
        n_real = len(batch)
        if n_real < C.num_rois:
            batch = np.concatenate([batch, np.repeat(batch[:1], C.num_rois - n_real, 0)])
        pc, pr = model_cls([fmap, tf.convert_to_tensor(batch[None], tf.float32)], training=False)
        pc = pc.numpy()[0]
        pr = pr.numpy()[0]
        for j in range(n_real):
            cls = int(pc[j].argmax())
            score = float(pc[j].max())
            if INV_CLASS[cls] == 'bg' or score < score_thresh:
                continue
            x_, y_, w_, h_ = batch[j]
            tx, ty, tw, th = pr[j, 4 * cls:4 * (cls + 1)]
            sx, sy, sw, sh = C.classifier_regr_std
            x_, y_, w_, h_ = apply_regr(x_, y_, w_, h_, tx / sx, ty / sy, tw / sw, th / sh)
            st = C.rpn_stride
            by_class.setdefault(INV_CLASS[cls], []).append([st * x_, st * y_, st * (x_ + w_), st * (y_ + h_), score])
    dets = []
    for label, lst in by_class.items():
        arr = np.array(lst, dtype=np.float64)
        keep = nms(arr[:, :4], arr[:, 4], nms_thresh, 300)
        for k in keep:
            dets.append((arr[k, :4].tolist(), label, float(arr[k, 4])))
    return dets

class ContextualRoiPoolingVec(ContextualRoiPooling):

    def call(self, x, mask=None):
        img, rois = (x[0], x[1])
        shape = tf.shape(img)
        H = tf.cast(shape[1], tf.float32)
        W = tf.cast(shape[2], tf.float32)
        r = tf.cast(rois[0], tf.float32)
        x0, y0, w0, h0 = (r[:, 0], r[:, 1], r[:, 2], r[:, 3])
        idx = tf.zeros([self.num_rois], tf.int32)
        crops = []
        for i in range(self.n_contexts):
            pad = i * self.step_stride
            x1 = tf.maximum(0.0, x0 - pad)
            y1 = tf.maximum(0.0, y0 - pad)
            x2 = tf.minimum(W, x0 + w0 + pad)
            y2 = tf.minimum(H, y0 + h0 + pad)
            boxes = tf.stack([y1 / H, x1 / W, y2 / H, x2 / W], axis=1)
            crops.append(tf.image.crop_and_resize(image=img, boxes=boxes, box_indices=idx, crop_size=(self.pool_size, self.pool_size), method='bilinear'))
        fused = tf.reduce_max(tf.stack(crops, axis=0), axis=0)
        return tf.reshape(fused, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

def build_head_vec(n_contexts, step_stride=2):
    feat_in = layers.Input(shape=(None, None, 1024))
    rois_in = layers.Input(shape=(C.num_rois, 4))
    pooled = ContextualRoiPoolingVec(pool_size=7, num_rois=C.num_rois, n_contexts=n_contexts, step_stride=step_stride)([feat_in, rois_in])
    out = layers.Reshape((C.num_rois, 7 * 7 * 1024))(pooled)
    out = layers.Dense(2048, activation='relu')(out)
    out = layers.Dropout(0.5)(out)
    out_cls = layers.Dense(NUM_CLASSES, activation='softmax', name='dense_class')(out)
    out_reg = layers.Dense(4 * (NUM_CLASSES - 1), activation='linear', name='dense_regress')(out)
    return Model([feat_in, rois_in], [out_cls, out_reg], name='head_fused')

def detect_frame_timed(rpn_fn, cls_fn, img, score_thresh=0.0, nms_thresh=0.5):
    t0 = time.perf_counter()
    p_cls, p_reg, fmap = rpn_fn(tf.convert_to_tensor(img[None], tf.float32))
    p_cls = p_cls.numpy()
    p_reg = p_reg.numpy()
    fh, fw = int(fmap.shape[1]), int(fmap.shape[2])
    t1 = time.perf_counter()
    rois, _ = rpn_to_rois(p_cls[0], p_reg[0], fh, fw, top_n=300)
    t2 = time.perf_counter()
    outs = []
    for i in range(0, len(rois), C.num_rois):
        batch = rois[i:i + C.num_rois]
        n_real = len(batch)
        if n_real < C.num_rois:
            batch = np.concatenate([batch, np.repeat(batch[:1], C.num_rois - n_real, 0)])
        pc, pr = cls_fn(fmap, tf.convert_to_tensor(batch[None], tf.float32))
        outs.append((batch, n_real, pc.numpy()[0], pr.numpy()[0]))
    t3 = time.perf_counter()
    by_class = {}
    for batch, n_real, pc, pr in outs:
        for j in range(n_real):
            cls = int(pc[j].argmax())
            score = float(pc[j].max())
            if INV_CLASS[cls] == 'bg' or score < score_thresh:
                continue
            x_, y_, w_, h_ = batch[j]
            tx, ty, tw, th = pr[j, 4 * cls:4 * (cls + 1)]
            sx, sy, sw, sh = C.classifier_regr_std
            x_, y_, w_, h_ = apply_regr(x_, y_, w_, h_, tx / sx, ty / sy, tw / sw, th / sh)
            st = C.rpn_stride
            by_class.setdefault(INV_CLASS[cls], []).append([st * x_, st * y_, st * (x_ + w_), st * (y_ + h_), score])
    dets = []
    for label, lst in by_class.items():
        arr = np.array(lst, dtype=np.float64)
        keep = nms(arr[:, :4], arr[:, 4], nms_thresh, 300)
        for k in keep:
            dets.append((arr[k, :4].tolist(), label, float(arr[k, 4])))
    t4 = time.perf_counter()
    return dets, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)

def run_bench():
    import cv2, traceback
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    rng = np.random.default_rng(0)
    imgs = []
    for (h, w) in SIZES:
        im = rng.integers(0, 256, size=(h // 8 + 1, w // 8 + 1, 3)).astype("uint8")
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC).astype("float32")
        im -= np.array([103.939, 116.779, 123.68])
        imgs.append(im)
    dev = tf.config.experimental.get_device_details(gpus[0]).get("device_name", "GPU") if gpus else platform.processor()
    print("frames", len(imgs), "device", dev, "tensorflow", tf.__version__, flush=True)
    allres = {"device": dev, "tensorflow": tf.__version__, "frames": len(imgs), "settings": {}}
    for n in (1, 2, 3):
        tf.keras.backend.clear_session()
        model_rpn, model_cls = build_models(n, 2)
        params = int(model_rpn.count_params() + model_cls.count_params())
        eager = (lambda x: model_rpn(x, training=False), lambda f, r: model_cls([f, r], training=False))
        comp = (tf.function(eager[0], input_signature=[tf.TensorSpec([1, None, None, 3], tf.float32)]),
                tf.function(eager[1], input_signature=[tf.TensorSpec([1, None, None, 1024], tf.float32),
                                                       tf.TensorSpec([1, C.num_rois, 4], tf.float32)]))
        head_vec = build_head_vec(n, 2)
        head_vec.set_weights(model_cls.get_weights())
        vec_cls = tf.function(lambda f, r: head_vec([f, r], training=False),
                              input_signature=[tf.TensorSpec([1, None, None, 1024], tf.float32),
                                               tf.TensorSpec([1, C.num_rois, 4], tf.float32)])
        vec = (comp[0], vec_cls)
        try:
            md = 0.0
            for im in imgs[:3]:
                p_cls, p_reg, fmap = comp[0](tf.convert_to_tensor(im[None], tf.float32))
                rois, _ = rpn_to_rois(p_cls.numpy()[0], p_reg.numpy()[0], int(fmap.shape[1]), int(fmap.shape[2]), top_n=300)
                for i in range(0, len(rois), C.num_rois):
                    b = rois[i:i + C.num_rois]
                    if len(b) < C.num_rois:
                        b = np.concatenate([b, np.repeat(b[:1], C.num_rois - len(b), 0)])
                    bt = tf.convert_to_tensor(b[None], tf.float32)
                    a1, a2 = comp[1](fmap, bt)
                    v1, v2 = vec_cls(fmap, bt)
                    md = max(md, float(np.abs(a1.numpy() - v1.numpy()).max()), float(np.abs(a2.numpy() - v2.numpy()).max()))
            print("DETBENCH_EQ " + json.dumps({"n_contexts": n, "max_abs_diff": md}), flush=True)
        except Exception as e:
            traceback.print_exc()
            print("DETBENCH_ERR eq", n, repr(e)[:500], flush=True)
        for mode, (rf, cf), warm, reps in (("eager", eager, 3, 1), ("compiled", comp, 19, 5), ("optimized", vec, 19, 5)):
            try:
                tw = time.perf_counter()
                for im in imgs[:warm]:
                    detect_frame_timed(rf, cf, im)
                warm_s = time.perf_counter() - tw
                T, S, nd = [], [], []
                for _ in range(reps):
                    for im in imgs:
                        t0 = time.perf_counter()
                        d, st = detect_frame_timed(rf, cf, im)
                        T.append((time.perf_counter() - t0) * 1000.0)
                        S.append(st)
                        nd.append(len(d))
                T = np.array(T)
                S = np.array(S) * 1000.0
                res = {"n_contexts": n, "mode": mode, "ms_mean": float(T.mean()), "ms_median": float(np.median(T)),
                       "ms_p90": float(np.percentile(T, 90)), "fps": float(1000.0 / T.mean()), "timed": int(len(T)),
                       "params": params, "dets_mean": float(np.mean(nd)), "warmup_s": float(warm_s),
                       "stage_ms_mean": dict(zip(["rpn", "proposals", "head", "post"], [float(v) for v in S.mean(0)]))}
                allres["settings"][str(n) + "_" + mode] = res
                print("DETBENCH " + json.dumps(res), flush=True)
            except Exception as e:
                traceback.print_exc()
                print("DETBENCH_ERR", n, mode, repr(e)[:500], flush=True)
    print("DETBENCH_ALL " + json.dumps(allres), flush=True)

run_bench()
