import requests
import json
import numpy as np
import tensorflow as tf
from model import train_segnet
from blur import resize, create_img

BASE_URL = "https://backgroundblur.herokuapp.com/image/"
IMAGE_HEIGHT = IMAGE_WIDTH = 224


def get_image_url(base_url):
    response = requests.get(base_url)
    json_data = json.loads(response.content.decode())
    filename = json_data["filename"]
    image_url = BASE_URL + str(filename)
    return image_url


def prepare_image(image_url):
    img_content = requests.get(image_url).content
    img = tf.io.decode_image(img_content, channels=3, dtype=tf.float32)
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
    checkpoint.restore(checkpoint_path).expect_partial()
    predicted_mask = model.predict(img)
    return preprocess_mask(predicted_mask)


def blur_image(original_image, predicted_mask):
    original_image = resize(img_path=original_image, url=True)
    predicted_mask = resize(predicted_mask)
    return create_img(original_image, predicted_mask)


def get_results(predicted=False, blurred=False):
    image_url = get_image_url(BASE_URL)
    original_image = prepare_image(image_url)
    predicted_mask = predict(original_image)
    img = predicted_mask * 255
    img = tf.cast(img, dtype=tf.dtypes.uint8)
    img = tf.image.encode_jpeg(
        img, quality=100)
    predicted_img_path = 'results/predicted_mask.jpg'
    tf.io.write_file(predicted_img_path, img)

    if predicted:
        return predicted_img_path
    original_image_path, predicted_image_path, blurred_image_path = blur_image(
        image_url, predicted_img_path)
    if blurred:
        return blurred_image_path

    return original_image_path, predicted_image_path, blurred_image_path
