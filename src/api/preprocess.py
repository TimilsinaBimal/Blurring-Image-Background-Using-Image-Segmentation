import requests
import json
import numpy as np
import tensorflow as tf

BASE_URL = "127.0.0.1:5500/result/"
IMAGE_HEIGHT = IMAGE_WIDTH = 224


def prepare_image(img):
    img = tf.io.decode_image(img, channels=3, dtype=tf.float32)
    img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
    return img[np.newaxis, ...]


def get_image_data(base_url):
    response = requests.get(base_url)
    json_data = json.loads(response.content.decode())
    image = json_data["original_image"]
    image = list(image)
    image = np.array(image)
    return image


def get_mask(image, model):
    image = prepare_image(image)

    return image
