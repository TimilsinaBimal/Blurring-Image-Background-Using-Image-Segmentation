import requests
import os
import warnings
import cv2
import numpy as np
warnings.simplefilter("ignore")
# blur image


def make_blur(img):
    ksize = (10, 10)
    blur = cv2.blur(img, ksize)
    return blur


# resize image
def resize(img_path, url=False):
    batch = []
    height = 224
    width = 224
    if url:
        img = requests.get(img_path).content
        img = np.asarray(bytearray(img), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(img_path, 1)

    img = cv2.resize(img, (height, width))
    img = img / 255.
    batch.append(img)
    batch = np.array(batch)
    return batch


# replace mask with the BGR colours


def mask_blur(original_img, blur_img, predicted_img):
    # original image channel values
    blue_channel_ori = original_img[:, :, 0]
    green_channel_ori = original_img[:, :, 1]
    red_channel_ori = original_img[:, :, 2]

    # blur image channel values
    blue_channel_blr = blur_img[:, :, 0]
    green_channel_blr = blur_img[:, :, 1]
    red_channel_blr = blur_img[:, :, 2]

    # predicted image channel values
    blue_channel_pre = predicted_img[:, :, 0]
    green_channel_pre = predicted_img[:, :, 1]
    red_channel_pre = predicted_img[:, :, 2]

    blue = []
    green = []
    red = []

    background_blur_image = np.zeros([224, 224, 3])

    for i in range(3):
        if i == 0:
            img = blue_channel_blr
            msk = blue_channel_pre
            ori = blue_channel_ori
        if i == 1:
            img = green_channel_blr
            msk = green_channel_pre
            ori = green_channel_ori
        if i == 2:
            img = red_channel_blr
            msk = red_channel_pre
            ori = red_channel_ori

        if i == 0:
            new = blue
        if i == 1:
            new = green
        if i == 2:
            new = red

        img = img.reshape(1, -1)[0]
        msk = msk.reshape(1, -1)[0]
        ori = ori.reshape(1, -1)[0]

        for k, m, o in zip(img, msk, ori):
            if int(m*255.) < 50:
                new.append(k)
            else:
                new.append(o)

        if i == 0:
            blue = np.array(blue).reshape(224, 224)
            background_blur_image[:, :, 0] = blue
        if i == 1:
            green = np.array(green).reshape(224, 224)
            background_blur_image[:, :, 1] = green
        if i == 2:
            red = np.array(red).reshape(224, 224)
            background_blur_image[:, :, 2] = red

    return background_blur_image

# creates blurred background and stores inside results folder


def create_img(original_img, predicted_img):
    blur_img = make_blur(original_img[0])  # original blurred image

    # final background blurred image
    img = mask_blur(original_img[0], blur_img, predicted_img[0])

    try:
        os.mkdir("results")
    except:
        pass
    original_image_path = "results/original_image.jpg"
    predicted_image_path = "results/predicted_mask.jpg"
    blurred_image_path = "results/blur_image.jpg"

    cv2.imwrite(original_image_path, original_img[0] * 255.)
    cv2.imwrite(predicted_image_path, predicted_img[0] * 255.)
    cv2.imwrite(blurred_image_path, img * 255.)
    return original_image_path, predicted_image_path, blurred_image_path
