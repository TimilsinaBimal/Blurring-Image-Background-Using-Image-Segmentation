import tensorflow as tf
from src.visualization.display_image import display


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(model, image, mask):
    prediction = model.predict(image[tf.newaxis, ...])
    display([image, mask, create_mask(prediction)])


def predict(model, dataset):
    model.evaluate(dataset)
