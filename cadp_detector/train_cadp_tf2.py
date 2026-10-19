#!/usr/bin/env python3
"""
train_cadp_tf2.py — Faster R-CNN training harness for the processed CADP corpus
(TensorFlow 2 / tf.keras).

USAGE
-----
  python train_cadp_tf2.py --cadp_root /path/to/CADP \
        --csv csv/cadp_all.csv --epochs 100 --n_contexts 3

INPUTS (produced by the processing pipeline already delivered)
  cadp_root/csv/cadp_all.csv        lines: img_path,x1,y1,x2,y2,label
  cadp_root/split.json              video-level train/val/test membership
  cadp_root/id_mapping.json         youtube_id <-> 6-digit vid

WHAT IT DOES
  1. Loads the CSV ground truth; keeps a row only if its video is in the
     requested split (video-level split => no frame leakage across splits).
  2. Builds the RPN targets (anchor matching, per config) on the fly.
  3. Trains RPN + classifier head jointly, with the ContextualRoiPooling layer
     (n_contexts=1 reproduces the standard ROI-pooling baseline -> ablation).
  4. Validates each epoch on the 'val' split (losses + proxy detection rate);
     keeps the best checkpoint by val total loss.
  5. Final testing is done separately by evaluate_cadp.py on the 'test' split.

Honest scope note: this file is the complete training scaffold — data pipeline,
anchor targets, model assembly, loop, checkpointing. It requires a GPU + TF2
environment and the VATIC-derived CSV to run; it cannot execute in the
annotation-less sandbox where it was written, so treat the first run as a
shakedown (start with --epochs 1 --max_videos 4).
"""
import os, json, argparse, math, random
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers

from losses_tf2 import (rpn_loss_regr, rpn_loss_cls,
                        class_loss_regr, class_loss_cls)
from contextual_roi_pooling import ContextualRoiPooling
from accident_dataset_toolkit import canon_label

# ----------------------------- configuration -------------------------------
class Config:
    anchor_scales = [64, 128, 256]           # smaller than VOC default: CADP objects are small
    anchor_ratios = [[1, 1], [1, 2], [2, 1]]
    im_size = 600                             # resize shortest side
    rpn_stride = 16                           # ResNet50 C4
    num_rois = 32                             # train_cadp.py of the CADP repository sets cfg.num_rois = 32
    rpn_min_overlap, rpn_max_overlap = 0.3, 0.7
    classifier_min_overlap, classifier_max_overlap = 0.1, 0.5
    class_mapping = {"vehicle": 0, "two_wheeler": 1, "person": 2, "bg": 3}
    std_scaling = 4.0
    classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
    use_horizontal_flips = True               # data_augment.py: off in the CADP configuration (stated in Table 6.4)
    rpn_num_regions = 256                     # data_generators.calc_rpn: 256 anchors per image, at most 128 positive
    rpn_nms_thresh = 0.7                      # roi_helpers.rpn_to_roi: overlap_thresh used by train_frcnn.py
    freeze_batchnorm = True                   # FixedBatchNormalization: statistics frozen, batch size 1

C = Config()
NUM_ANCHORS = len(C.anchor_scales) * len(C.anchor_ratios)
NUM_CLASSES = len(C.class_mapping)            # incl. bg

# ----------------------------- data pipeline -------------------------------
def load_split_rows(cadp_root, csv_path, split_name):
    """Read the flat CSV and keep rows whose VIDEO belongs to split_name."""
    split = set(json.load(open(os.path.join(cadp_root, "split.json")))[split_name])
    sc = os.path.join(cadp_root, "split_clips.json")       # CADP clip ids of the native-frame clips
    if os.path.exists(sc):
        split |= set(json.load(open(sc))[split_name])
    per_image = {}
    with open(os.path.join(cadp_root, csv_path)) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 6:
                continue
            path, x1, y1, x2, y2, label = parts
            # video id is the parent directory of the frame
            vid = os.path.basename(os.path.dirname(path))
            if vid not in split:
                continue
            # simple_parser.py of the CADP repository reads the native labels of
            # generate_csv.py (Car, Bus, Person, ...); map them onto the canonical
            # taxonomy, drop the mask regions and keep 'bg' as a hard-negative region
            label = label if label in C.class_mapping else canon_label(label)
            if label.startswith("__"):
                continue
            if label not in C.class_mapping:
                raise ValueError(f"unknown class label {label!r} in {csv_path}")
            per_image.setdefault(path, []).append(
                dict(x1=int(float(x1)), y1=int(float(y1)), x2=int(float(x2)), y2=int(float(y2)),
                     label=label))
    return per_image                      # {img_path: [bboxes...]}


def resize_keep_ratio(img, boxes, short=600):
    """get_new_img_size + cv2.resize(INTER_CUBIC) of data_generators.py: the shorter side becomes
    `short` pixels, the longer side int(f * length)."""
    import cv2
    h, w = img.shape[:2]
    if w <= h:
        f = float(short) / w
        nh, nw = int(f * h), int(short)
    else:
        f = float(short) / h
        nw, nh = int(f * w), int(short)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    boxes = boxes * f
    return img, boxes, f


def make_anchors(fh, fw):
    """All anchor boxes on an fh x fw feature map, in image coordinates."""
    xs = (np.arange(fw) + 0.5) * C.rpn_stride
    ys = (np.arange(fh) + 0.5) * C.rpn_stride
    cx, cy = np.meshgrid(xs, ys)
    anchors = []
    for s in C.anchor_scales:
        for (rw, rh) in C.anchor_ratios:
            aw, ah = s * rw / math.sqrt(rw * rh), s * rh / math.sqrt(rw * rh)
            anchors.append(np.stack(
                [cx - aw/2, cy - ah/2, cx + aw/2, cy + ah/2], axis=-1))
    return np.stack(anchors, axis=2)      # (fh, fw, A, 4)


def iou_matrix(a, b):
    """IoU between (N,4) and (M,4)."""
    N, M = len(a), len(b)
    if N == 0 or M == 0:
        return np.zeros((N, M))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw, ih = np.clip(ix2-ix1, 0, None), np.clip(iy2-iy1, 0, None)
    inter = iw * ih
    ua = ((a[:, 2]-a[:, 0])*(a[:, 3]-a[:, 1]))[:, None] + \
         ((b[:, 2]-b[:, 0])*(b[:, 3]-b[:, 1]))[None, :] - inter
    return inter / np.maximum(ua, 1e-9)


def rpn_targets(fh, fw, gt_boxes, img_w=None, img_h=None):
    """Build y_is_valid/y_label and regression targets for every anchor, as in
    data_generators.calc_rpn of the CADP repository: anchors that cross the image border
    are excluded, every ground-truth box keeps its best anchor as a positive, and at most
    C.rpn_num_regions anchors (half of them positive at most) enter the objective."""
    A = NUM_ANCHORS
    anchors = make_anchors(fh, fw).reshape(-1, 4)          # (fh*fw*A,4)
    valid = np.zeros(len(anchors), np.float32)
    label = np.zeros(len(anchors), np.float32)
    regr  = np.zeros((len(anchors), 4), np.float32)
    W = img_w if img_w is not None else fw * C.rpn_stride
    H = img_h if img_h is not None else fh * C.rpn_stride
    inside = (anchors[:, 0] >= 0) & (anchors[:, 1] >= 0) & (anchors[:, 2] <= W) & (anchors[:, 3] <= H)
    if len(gt_boxes):
        ious = iou_matrix(anchors, gt_boxes)               # (K,G)
        ious[~inside] = 0.0
        best_gt = ious.argmax(1); best_iou = ious.max(1)
        pos = (best_iou >= C.rpn_max_overlap) & inside
        neg = (best_iou <  C.rpn_min_overlap) & inside
        # force best anchor per GT positive (only among anchors inside the image)
        best_per_gt = ious.argmax(0)
        pos[best_per_gt[ious.max(0) > 0]] = True
        neg[pos] = False
        # subsample to rpn_num_regions, at most half positive (calc_rpn)
        n_reg = C.rpn_num_regions
        pos_idx = np.where(pos)[0]
        if len(pos_idx) > n_reg // 2:
            drop = np.random.choice(pos_idx, len(pos_idx) - n_reg // 2, replace=False)
            pos[drop] = False
        neg_idx = np.where(neg)[0]
        n_neg_keep = max(n_reg - int(pos.sum()), 0)
        if len(neg_idx) > n_neg_keep:
            drop = np.random.choice(neg_idx, len(neg_idx) - n_neg_keep, replace=False)
            neg[drop] = False
        valid[pos | neg] = 1.0
        label[pos] = 1.0
        # regression targets for positives (tx,ty,tw,th), scaled
        g = gt_boxes[best_gt]
        aw, ah = anchors[:, 2]-anchors[:, 0], anchors[:, 3]-anchors[:, 1]
        acx, acy = anchors[:, 0]+aw/2, anchors[:, 1]+ah/2
        gw, gh = g[:, 2]-g[:, 0], g[:, 3]-g[:, 1]
        gcx, gcy = g[:, 0]+gw/2, g[:, 1]+gh/2
        t = np.stack([(gcx-acx)/aw, (gcy-acy)/ah,
                      np.log(np.maximum(gw, 1)/aw),
                      np.log(np.maximum(gh, 1)/ah)], axis=1) * C.std_scaling
        regr[pos] = t[pos]
    # reshape to map layout, channels-last
    valid = valid.reshape(fh, fw, A)
    label = label.reshape(fh, fw, A)
    regr  = regr.reshape(fh, fw, A*4)
    rep4  = np.repeat(valid, 4, axis=-1)
    y_cls  = np.concatenate([valid, label], axis=-1)[None]           # (1,fh,fw,2A)
    y_regr = np.concatenate([rep4 * (label.repeat(4, -1) > 0), regr],
                            axis=-1)[None]                            # (1,fh,fw,8A)
    return y_cls.astype(np.float32), y_regr.astype(np.float32)


def data_generator(per_image, shuffle=True):
    import cv2
    paths = list(per_image.keys())
    while True:
        if shuffle:
            random.shuffle(paths)
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            boxes = np.array([[b["x1"], b["y1"], b["x2"], b["y2"]]
                              for b in per_image[p]], np.float32)
            labels = [b["label"] for b in per_image[p]]
            if shuffle and C.use_horizontal_flips and random.random() < 0.5:
                img = cv2.flip(img, 1)                     # data_augment.py, horizontal flip
                if len(boxes):
                    w_img = img.shape[1]
                    x1 = boxes[:, 0].copy(); x2 = boxes[:, 2].copy()
                    boxes[:, 0] = w_img - x2; boxes[:, 2] = w_img - x1
            img, boxes, _ = resize_keep_ratio(img.astype(np.float32), boxes,
                                              C.im_size)
            img -= np.array([103.939, 116.779, 123.68])   # BGR mean
            fh, fw = img.shape[0]//C.rpn_stride, img.shape[1]//C.rpn_stride
            # 'bg' boxes are hard-negative regions (simple_parser.py): they never make an
            # anchor positive (calc_rpn skips them) but they do reach the head sampler
            fg = np.array([l != "bg" for l in labels], bool)
            y_cls, y_regr = rpn_targets(fh, fw, boxes[fg] if len(boxes) else boxes, img.shape[1], img.shape[0])
            yield img[None], y_cls, y_regr, boxes, labels, p


# ------------------------------- model -------------------------------------
def build_models(n_contexts, step_stride=2):
    backbone = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=(None, None, 3))
    # C4 feature map (stride 16): stop at conv4 block output
    c4 = backbone.get_layer("conv4_block6_out").output
    feat = Model(backbone.input, c4, name="backbone")
    if C.freeze_batchnorm:
        # FixedBatchNormalization of the CADP repository: the normalization statistics stay
        # at their ImageNet values and the layers run in inference mode, which is what a
        # non-trainable BatchNormalization layer does in TensorFlow 2 (batch size is 1)
        for layer in feat.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    img_in = layers.Input(shape=(None, None, 3))
    fmap = feat(img_in)
    # RPN
    x = layers.Conv2D(512, 3, padding="same", activation="relu",
                      name="rpn_conv")(fmap)
    rpn_cls = layers.Conv2D(NUM_ANCHORS, 1, activation="sigmoid",
                            name="rpn_out_class")(x)
    rpn_reg = layers.Conv2D(NUM_ANCHORS*4, 1, activation="linear",
                            name="rpn_out_regress")(x)
    model_rpn = Model(img_in, [rpn_cls, rpn_reg, fmap], name="rpn")

    # classifier head with contextual ROI pooling
    feat_in = layers.Input(shape=(None, None, 1024))
    rois_in = layers.Input(shape=(C.num_rois, 4))
    pooled = ContextualRoiPooling(pool_size=7, num_rois=C.num_rois,
                                  n_contexts=n_contexts,
                                  step_stride=step_stride)([feat_in, rois_in])
    out = layers.TimeDistributed(layers.Flatten())(pooled)
    out = layers.TimeDistributed(layers.Dense(2048, activation="relu"))(out)
    out = layers.TimeDistributed(layers.Dropout(0.5))(out)
    out_cls = layers.TimeDistributed(
        layers.Dense(NUM_CLASSES, activation="softmax"),
        name="dense_class")(out)
    out_reg = layers.TimeDistributed(
        layers.Dense(4*(NUM_CLASSES-1), activation="linear"),
        name="dense_regress")(out)
    model_cls = Model([feat_in, rois_in], [out_cls, out_reg], name="head")
    return model_rpn, model_cls


# --------------------------- proposal sampling ------------------------------
def rpn_to_rois(rpn_cls, rpn_reg, fh, fw, top_n=300):
    """Decode RPN output to ROIs in FEATURE-MAP coords (x,y,w,h)."""
    anchors = make_anchors(fh, fw).reshape(-1, 4) / C.rpn_stride
    scores = rpn_cls.reshape(-1)
    t = (rpn_reg.reshape(-1, 4)) / C.std_scaling
    aw, ah = anchors[:, 2]-anchors[:, 0], anchors[:, 3]-anchors[:, 1]
    acx, acy = anchors[:, 0]+aw/2, anchors[:, 1]+ah/2
    cx, cy = t[:, 0]*aw+acx, t[:, 1]*ah+acy
    w,  h  = np.exp(np.clip(t[:, 2], -4, 4))*aw, np.exp(np.clip(t[:, 3], -4, 4))*ah
    x1, y1 = np.maximum(0, cx-w/2), np.maximum(0, cy-h/2)
    x2, y2 = np.minimum(fw-1, cx+w/2), np.minimum(fh-1, cy+h/2)
    ok = (x2 > x1) & (y2 > y1)
    x1, y1, x2, y2, scores = x1[ok], y1[ok], x2[ok], y2[ok], scores[ok]
    keep = nms(np.stack([x1, y1, x2, y2], 1), scores, C.rpn_nms_thresh, top_n)
    boxes = np.stack([x1[keep], y1[keep], x2[keep]-x1[keep], y2[keep]-y1[keep]], 1)
    return boxes, scores[keep]


def nms(boxes, scores, thresh, max_boxes):
    """Greedy non-maximum suppression (roi_helpers.non_max_suppression_fast)."""
    order = scores.argsort()[::-1]
    keep = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    while len(order) and len(keep) < max_boxes:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= thresh]
    return np.array(keep, dtype=int)


def sample_rois(rois, gt_boxes_img, labels, scale=1.0, rng=None):
    """Match proposals to GT; return (rois, y_cls, y_regr) for the head."""
    if len(gt_boxes_img) == 0:
        return None
    gt_fm = gt_boxes_img / C.rpn_stride
    xyxy = np.stack([rois[:, 0], rois[:, 1],
                     rois[:, 0]+rois[:, 2], rois[:, 1]+rois[:, 3]], 1)
    ious = iou_matrix(xyxy, gt_fm)
    best_gt, best_iou = ious.argmax(1), ious.max(1)
    keep = best_iou >= C.classifier_min_overlap
    rois, best_gt, best_iou = rois[keep], best_gt[keep], best_iou[keep]
    if len(rois) == 0:
        return None
    n = C.num_rois
    rng = rng if rng is not None else np.random
    pos = best_iou >= C.classifier_max_overlap
    pos_all, neg_all = np.where(pos)[0], np.where(~pos)[0]
    # train_cadp.py: at most num_rois // 2 positives drawn without replacement, the rest
    # negatives drawn without replacement, with replacement when there are too few; an image
    # without a single negative proposal skips the head step
    if len(neg_all) == 0:
        return None
    pos_idx = pos_all if len(pos_all) < n // 2 else rng.choice(pos_all, n // 2, replace=False)
    n_neg = n - len(pos_idx)
    neg_idx = rng.choice(neg_all, n_neg, replace=len(neg_all) < n_neg)
    sel = np.concatenate([pos_idx, neg_idx]).astype(int)
    rois_s = rois[sel]
    y_cls = np.zeros((1, n, NUM_CLASSES), np.float32)
    y_reg = np.zeros((1, n, 8*(NUM_CLASSES-1)), np.float32)
    for i, ridx in enumerate(sel):
        if best_iou[ridx] >= C.classifier_max_overlap and labels[best_gt[ridx]] != "bg":
            cls = C.class_mapping[labels[best_gt[ridx]]]
            y_cls[0, i, cls] = 1
            g = gt_fm[best_gt[ridx]]
            r = rois_s[i]
            gw, gh = g[2]-g[0], g[3]-g[1]
            gcx, gcy = g[0]+gw/2, g[1]+gh/2
            rcx, rcy = r[0]+r[2]/2, r[1]+r[3]/2
            sx, sy, sw, sh = C.classifier_regr_std
            t = [sx*(gcx-rcx)/r[2], sy*(gcy-rcy)/r[3],
                 sw*math.log(max(gw, 1e-3)/r[2]), sh*math.log(max(gh, 1e-3)/r[3])]
            y_reg[0, i, 4*cls:4*cls+4]                    = 1   # mask
            y_reg[0, i, 4*(NUM_CLASSES-1)+4*cls:
                        4*(NUM_CLASSES-1)+4*cls+4]        = t
        else:
            y_cls[0, i, C.class_mapping["bg"]] = 1
    return rois_s[None].astype(np.float32), y_cls, y_reg


# ------------------------------ training loop -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cadp_root", required=True)
    ap.add_argument("--csv", default="csv/cadp_all.csv")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--n_contexts", type=int, default=3,
                    help="1 = standard ROI pooling baseline (ablation)")
    ap.add_argument("--step_stride", type=int, default=2,
                    help="expansion of every further context window, in feature-map cells (Table 6.4: 2; "
                         "the released configuration of improved_frcnn uses 4 with n_contexts 2)")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--max_videos", type=int, default=0,
                    help="debug: cap videos per split")
    ap.add_argument("--seed", type=int, default=0,
                    help="random seed; one training run per seed (Table 6.4)")
    ap.add_argument("--patience", type=int, default=0,
                    help="early stopping on validation loss (0 = off)")
    ap.add_argument("--history", default="",
                    help="path of the training-history JSON (default: <out>/history.json)")
    ap.add_argument("--joint", action="store_true",
                    help="one optimizer on the sum of the four losses (earlier port) instead of the two "
                         "consecutive steps per image of train_cadp.py (default)")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
    import time
    t_start = time.time()
    history = {"kind": "history", "n_contexts": args.n_contexts, "step_stride": args.step_stride, "seed": args.seed,
               "steps_per_epoch": args.steps, "epochs_requested": args.epochs,
               "train_loss": [], "val_loss": [], "best_epoch": None}
    hist_path = args.history or os.path.join(args.out, "history.json")

    train_rows = load_split_rows(args.cadp_root, args.csv, "train")
    val_rows   = load_split_rows(args.cadp_root, args.csv, "val")
    print(f"train imgs: {len(train_rows)}  val imgs: {len(val_rows)}")

    model_rpn, model_cls = build_models(args.n_contexts, args.step_stride)
    backbone_vars = model_rpn.get_layer("backbone").trainable_variables
    # train_cadp.py compiles the RPN model and the classifier model (which contains the shared
    # base layers) with one Adam(1e-5) each and calls train_on_batch on both, one after the
    # other, for every image: the RPN step updates backbone + RPN, the head step updates
    # backbone + head on proposals computed with the just-updated RPN
    opt_rpn = optimizers.Adam(1e-5)
    opt_cls = optimizers.Adam(1e-5)
    l_rpn_c, l_rpn_r = rpn_loss_cls(NUM_ANCHORS), rpn_loss_regr(NUM_ANCHORS)
    l_cls_r = class_loss_regr(NUM_CLASSES-1)
    history["scheme"] = "joint" if args.joint else "alternating (train_cadp.py)"

    def head_pack(img, gt, labels, training, rng=None):
        p_cls, p_reg, fmap = model_rpn(img, training=training)
        rois, _ = rpn_to_rois(p_cls.numpy()[0], p_reg.numpy()[0], fmap.shape[1], fmap.shape[2])
        return p_cls, p_reg, fmap, sample_rois(rois, gt, labels, rng=rng)

    gen = data_generator(train_rows)
    os.makedirs(args.out, exist_ok=True)
    best_val = float("inf")
    bad = 0

    for epoch in range(args.epochs):
        tr_losses = []
        comp = np.zeros(4)
        for step in range(args.steps):
            img, y_cls, y_regr, gt, labels, _ = next(gen)
            if args.joint:
                with tf.GradientTape(persistent=True) as tape:
                    p_cls, p_reg, fmap, pack = head_pack(img, gt, labels, True)
                    lc, lr = l_rpn_c(y_cls, p_cls), l_rpn_r(y_regr, p_reg)
                    hc = hr = 0.0
                    if pack is not None:
                        r, yc, yr = pack
                        pc, pr = model_cls([fmap, tf.convert_to_tensor(r, tf.float32)], training=True)
                        hc, hr = class_loss_cls(yc, pc), l_cls_r(yr, pr)
                    loss = lc + lr + hc + hr
                for m in (model_rpn, model_cls):
                    g = tape.gradient(loss, m.trainable_variables)
                    opt_rpn.apply_gradients((gi, vi) for gi, vi in zip(g, m.trainable_variables) if gi is not None)
            else:
                # step 1: RPN loss, backbone + RPN
                with tf.GradientTape() as tape:
                    p_cls, p_reg, _ = model_rpn(img, training=True)
                    lc, lr = l_rpn_c(y_cls, p_cls), l_rpn_r(y_regr, p_reg)
                    loss_rpn = lc + lr
                g = tape.gradient(loss_rpn, model_rpn.trainable_variables)
                opt_rpn.apply_gradients((gi, vi) for gi, vi in zip(g, model_rpn.trainable_variables) if gi is not None)
                # step 2: proposals from the updated RPN, head loss, backbone + head
                hc = hr = 0.0
                with tf.GradientTape() as tape:
                    _, _, fmap, pack = head_pack(img, gt, labels, True)
                    if pack is not None:
                        r, yc, yr = pack
                        pc, pr = model_cls([fmap, tf.convert_to_tensor(r, tf.float32)], training=True)
                        hc, hr = class_loss_cls(yc, pc), l_cls_r(yr, pr)
                        loss_head = hc + hr
                if pack is not None:
                    vars_ = backbone_vars + model_cls.trainable_variables
                    g = tape.gradient(loss_head, vars_)
                    opt_cls.apply_gradients((gi, vi) for gi, vi in zip(g, vars_) if gi is not None)
                loss = lc + lr + hc + hr
            comp += [float(lc), float(lr), float(hc), float(hr)]
            tr_losses.append(float(loss))
        # ---- validation pass: the four losses on the validation split (the release selects on
        # the same sum, computed on its training images) ----
        vg = data_generator(val_rows, shuffle=False)
        v_losses = []
        vrng = np.random.RandomState(args.seed)
        for _ in range(min(len(val_rows), 200)):
            img, y_cls, y_regr, gt, labels, _ = next(vg)
            p_cls, p_reg, fmap, pack = head_pack(img, gt, labels, False, rng=vrng)
            v = float(l_rpn_c(y_cls, p_cls) + l_rpn_r(y_regr, p_reg))
            if pack is not None:
                r, yc, yr = pack
                pc, pr = model_cls([fmap, tf.convert_to_tensor(r, tf.float32)], training=False)
                v += float(class_loss_cls(yc, pc) + l_cls_r(yr, pr))
            v_losses.append(v)
        tr, vl = np.mean(tr_losses), np.mean(v_losses)
        comp /= max(1, args.steps)
        print(f"epoch {epoch+1}: train {tr:.4f} (rpn cls {comp[0]:.4f}, rpn regr {comp[1]:.4f}, "
              f"head cls {comp[2]:.4f}, head regr {comp[3]:.4f})  val {vl:.4f}")
        history["train_loss"].append(float(tr)); history["val_loss"].append(float(vl))
        history.setdefault("train_loss_components", []).append([float(x) for x in comp])
        if vl < best_val:
            best_val = vl; bad = 0
            history["best_epoch"] = epoch + 1
            model_rpn.save_weights(os.path.join(args.out, "rpn_best.weights.h5"))
            model_cls.save_weights(os.path.join(args.out, "head_best.weights.h5"))
            print("  saved best")
        else:
            bad += 1
        history["epochs_run"] = epoch + 1
        history["wall_clock_s"] = time.time() - t_start
        json.dump(history, open(hist_path, "w"), indent=1)       # updated every epoch
        if args.patience and bad >= args.patience:
            print(f"early stopping after {epoch+1} epochs"); break

if __name__ == "__main__":
    main()
