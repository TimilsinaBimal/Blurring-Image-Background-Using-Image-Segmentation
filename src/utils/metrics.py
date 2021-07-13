import tensorflow.keras.backend as K
import tensorflow as tf


def iou_score(y_true, y_pred, smooth=0.005):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()
    intersection = K.sum(K.abs(y_true * y_pred), axis=[0, 1, 2])
    union = K.sum(y_true, [0, 1, 2]) + K.sum(y_pred, [0, 1, 2])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth))
    return round(iou.numpy(), 2)
