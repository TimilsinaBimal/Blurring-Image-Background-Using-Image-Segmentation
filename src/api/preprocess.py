import requests
import json
import numpy as np
import tensorflow as tf
import os
from pathlib import Path
from models.train import train_segnet

BASE_URL = "http://127.0.0.1:5000/result/"
IMAGE_HEIGHT = IMAGE_WIDTH = 224


def get_image_data(base_url):
    response = requests.get(base_url)
    json_data = json.loads(response.content.decode())
    image = json_data["original_image"]
    image = np.array(image)
    return image


def prepare_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=3, dtype=tf.float32)
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return img[np.newaxis, ...]


def preprocess_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def predict(img):
    model = train_segnet()
    checkpoint = tf.train.Checkpoint(model)
    checkpoint_path = "models/SegNet/"
    checkpoint.restore(checkpoint_path)
    predicted_mask = model.predict(img)
    return preprocess_mask(predicted_mask)


def blur_image(original_image, predicted_mask):
    pass
