"""
losses_tf2.py — Faster R-CNN losses, ported to TensorFlow 2 / tf.keras.

Port notes (vs the original TF1/Keras-2.1 repo):
  * K.image_dim_ordering() removed -> channels-last (NHWC) assumed throughout.
  * keras.objectives -> tf.keras.losses.
  * All ops are tf.* / K.* functions that exist in TF2's keras backend.
Loss definitions are mathematically identical to the originals:
  RPN regression   : smooth-L1 over positive anchors, normalised by count.
  RPN classification: binary cross-entropy over valid anchors.
  Head regression  : smooth-L1 over the true class's coordinates.
  Head classification: categorical cross-entropy.
"""
import tensorflow as tf
from tensorflow.keras import backend as K

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0
lambda_cls_regr = 1.0
lambda_cls_class = 1.0
epsilon = 1e-4


def rpn_loss_regr(num_anchors):
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        # y_true channels: [is_valid(4A) | targets(4A)] along last axis (NHWC)
        x = y_true[:, :, :, 4 * num_anchors:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)
        return lambda_rpn_regr * K.sum(
            y_true[:, :, :, :4 * num_anchors] *
            (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))
        ) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])
    return rpn_loss_regr_fixed_num


def rpn_loss_cls(num_anchors):
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        # y_true channels: [is_valid(A) | label(A)]
        return lambda_rpn_class * K.sum(
            y_true[:, :, :, :num_anchors] *
            K.binary_crossentropy(y_true[:, :, :, num_anchors:], y_pred)
        ) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
    return rpn_loss_cls_fixed_num


def class_loss_regr(num_classes):
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:, :, 4 * num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
        return lambda_cls_regr * K.sum(
            y_true[:, :, :4 * num_classes] *
            (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))
        ) / K.sum(epsilon + y_true[:, :, :4 * num_classes])
    return class_loss_regr_fixed_num


def class_loss_cls(y_true, y_pred):
    return lambda_cls_class * K.mean(
        tf.keras.losses.categorical_crossentropy(y_true[0, :, :], y_pred[0, :, :]))
