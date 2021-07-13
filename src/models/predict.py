import tensorflow as tf
from visualization.image import display
from utils.metrics import iou_score


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(model, image, mask, iou=0):
    original = image[tf.newaxis, ...]
    prediction = model.predict(original)
    predicted_mask = create_mask(prediction)
    iou = iou_score(mask, predicted_mask)
    display([image, mask, predicted_mask], iou)
    return iou


def predict(model, dataset, steps):
    return model.evaluate(dataset, steps=steps)
