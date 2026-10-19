"""
contextual_roi_pooling.py  (TF2 / tf.keras)
===========================================
Corrected, working implementation of a *contextual* ROI-pooling layer for the
Faster R-CNN detector used in the thesis.

Idea
----
Standard ROI pooling extracts a fixed-size feature map from exactly the region
proposed by the RPN. For accident detection this is limiting: a collision is
defined by the *interaction between* objects, so the evidence often lies just
outside the tight box (the other vehicle, the point of impact, debris). This
layer therefore pools several progressively larger context windows around each
ROI and fuses them with a channel-wise maxout, letting the classifier see the
box *and* its surroundings while keeping the output shape unchanged.

Relationship to the base layer
------------------------------
It subclasses the project's RoiPoolingConv but replaces the single-crop `call`
with a multi-context crop-and-fuse. Output shape is identical to standard ROI
pooling — (1, num_rois, pool_size, pool_size, channels) — so it is a drop-in
replacement in the classifier head.

Key corrections over the original draft
--------------------------------------
1. Bounds are clamped with TensorFlow ops (tf.maximum/tf.minimum), NOT Python
   `if (t < 0) is not None`, which was always True and silently zeroed x,y,w,h.
2. Width/height clamp against the correct NHWC dims (W = shape[2], H = shape[1]),
   not the batch dim.
3. Uses tf.image.crop_and_resize with NORMALISED boxes — the robust, graph-safe
   way to crop dynamic regions — instead of Python slicing with symbolic ints.
4. tf.image.resize (TF2) instead of the removed tf.image.resize_images.
5. Maxout keeps the batch/roi structure explicit via tf.stack + tf.reduce_max.
6. TF2/tf.keras imports; no keras.engine.topology.
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer


class ContextualRoiPooling(Layer):
    """
    Contextual ROI pooling with maxout fusion over n_contexts dilated windows.

    Args:
        pool_size:   output side length (e.g. 7 -> 7x7).
        num_rois:    number of ROIs processed per call.
        n_contexts:  number of context windows per ROI (>=1). 1 == standard ROI pool.
        step_stride: per-context expansion, in *feature-map* units, added on every
                     side at context i (context i grows the box by 2*i*step_stride).

    Inputs (channels-last / NHWC):
        feature_map: (1, H, W, C)
        rois:        (1, num_rois, 4) as (x, y, w, h) in feature-map coordinates.

    Output:
        (1, num_rois, pool_size, pool_size, C)
    """

    def __init__(self, pool_size, num_rois, n_contexts=3, step_stride=2, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = int(pool_size)
        self.num_rois = int(num_rois)
        self.n_contexts = int(n_contexts)
        self.step_stride = float(step_stride)
        self.nb_channels = None

    def build(self, input_shape):
        # input_shape is a list: [feature_map_shape, rois_shape]
        self.nb_channels = input_shape[0][-1]
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return (None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels)

    def call(self, x, mask=None):
        assert len(x) == 2, "Expected [feature_map, rois]"
        img, rois = x[0], x[1]

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
                # expand the box by `pad` on every side
                x1 = x0 - pad
                y1 = y0 - pad
                x2 = x0 + w0 + pad
                y2 = y0 + h0 + pad

                # clamp to image bounds (in feature-map units) with TF ops
                x1 = tf.maximum(0.0, tf.cast(x1, tf.float32))
                y1 = tf.maximum(0.0, tf.cast(y1, tf.float32))
                x2 = tf.minimum(W, tf.cast(x2, tf.float32))
                y2 = tf.minimum(H, tf.cast(y2, tf.float32))

                # normalised box for crop_and_resize: [y1, x1, y2, x2] in [0,1]
                norm_box = tf.stack([y1 / H, x1 / W, y2 / H, x2 / W])
                norm_box = tf.reshape(norm_box, (1, 4))

                crop = tf.image.crop_and_resize(
                    image=img,                       # (1, H, W, C)
                    boxes=norm_box,                  # (1, 4)
                    box_indices=[0],                 # crop from batch element 0
                    crop_size=(self.pool_size, self.pool_size),
                    method="bilinear",
                )                                    # -> (1, pool, pool, C)
                context_crops.append(crop[0])        # (pool, pool, C)

            # maxout fusion across the context windows
            stacked = tf.stack(context_crops, axis=0)      # (n_contexts, pool, pool, C)
            fused = tf.reduce_max(stacked, axis=0)         # (pool, pool, C)
            roi_outputs.append(fused)

        out = tf.stack(roi_outputs, axis=0)                # (num_rois, pool, pool, C)
        out = tf.reshape(
            out, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels)
        )
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "pool_size": self.pool_size,
            "num_rois": self.num_rois,
            "n_contexts": self.n_contexts,
            "step_stride": self.step_stride,
        })
        return cfg


if __name__ == "__main__":
    # Smoke test: random feature map + ROIs, check output shape and that
    # different ROIs produce different pooled features (i.e. context actually varies).
    import numpy as np
    H, W, C = 38, 50, 512           # typical ResNet C4 feature-map size for ~600px input
    pool, num_rois, n_ctx = 7, 4, 3
    fmap = tf.constant(np.random.rand(1, H, W, C).astype("float32"))
    rois = tf.constant(np.array([[[5, 5, 10, 12],
                                  [20, 10, 8, 8],
                                  [0, 0, 6, 6],
                                  [30, 20, 15, 10]]], dtype="float32"))
    layer = ContextualRoiPooling(pool_size=pool, num_rois=num_rois,
                                 n_contexts=n_ctx, step_stride=2)
    layer.build([(1, H, W, C), (1, num_rois, 4)])
    y = layer([fmap, rois])
    print("output shape:", y.shape)                    # (1, 4, 7, 7, 512)
    print("distinct ROI means:",
          [float(tf.reduce_mean(y[0, r]).numpy()) for r in range(num_rois)])
