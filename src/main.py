import matplotlib.pyplot as plt
from data.augmentation import DataAugmentation
from visualization.graphs import plot_accuracy_vs_epoch, plot_loss_vs_epoch
from blur.blur import create_img, resize
from visualization.image import display
import tensorflow as tf
from models.train import (
    train_segnet,
    train_deeplab,
    train_pspnet,
    train_unet,
    train,
    train_segmentation
)
from models.predict import show_predictions, create_mask, predict
from data.prepare import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
import os
import random
from utils.metrics import iou_score


def prepare_image(image_path, mask=False):
    img = tf.io.read_file(image_path)
    if mask:
        img = tf.io.decode_image(img, channels=1, dtype=tf.float32)
    else:
        img = tf.io.decode_image(img, channels=3, dtype=tf.float32)
    img = tf.image.resize(img, [224, 224])
    return img[np.newaxis, ...]


def train_model():
    ROOT_DIR = Path(__file__).parent.parent
    print("Enter the model you want to train:")
    print("1: SegNet")
    print("2: UNet")
    print("3: PSPNet")
    print("4: DeepLab")
    print("5: Segmentation")
    model_selection = int(input("Your Choice: "))
    if model_selection == 1:
        model = train_segnet()

    elif model_selection == 2:
        model = train_unet()

    elif model_selection == 3:
        model = train_pspnet()

    elif model_selection == 4:
        model = train_deeplab()

    elif model_selection == 5:
        model = train_segmentation()

    df_file = model.__class__.__name__
    history = train(model, ROOT_DIR)
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(ROOT_DIR, "reports",
              f"{df_file}_history.csv"), index=False)


def predict_model(show=True):
    BEST_MODEL = "segnet"
    ROOT_DIR = Path(__file__).parent.parent

    if BEST_MODEL == "unet":
        model = train_unet()
        model_path = os.path.join(ROOT_DIR, "models/UNet/")
    elif BEST_MODEL == "segnet":
        model = train_segnet()
        model_path = os.path.join(ROOT_DIR, "models/SegNet/")
    elif BEST_MODEL == "pspnet":
        model = train_pspnet()
        model_path = os.path.join(ROOT_DIR, "models/PSPNet/")

    checkpoint = tf.train.Checkpoint(model)
    checkpoint.restore(model_path).expect_partial()

    if not show:
        return model

    dataset = Dataset(root_dir=ROOT_DIR,
                      batch_size=1, validation=True)

    train_dataset, val_dataset, test_dataset = dataset.make()
    steps = dataset.TEST_DATA_LEN

    print("Please Choose:")
    print("1: Get Accuracy and Loss of test dataset.")
    print("2: Show Results:")
    choice = int(input("Your Choice: "))

    if choice == 1:
        loss, accuracy = predict(model, test_dataset, steps)
        print(f"Loss: {loss}, Accuracy: {accuracy}")
    elif choice == 2:
        ious = []
        for image in train_dataset.take(10):
            iou = show_predictions(model, image[0][0], image[1][0])
            ious.append(iou)
        print(f"Average IOU: {sum(ious)/ len(ious)}")


def blur_image():
    model = predict_model(show=False)
    image_path = "src/test/test.JPG"
    img = prepare_image(image_path)
    res = model.predict(img)
    res = create_mask(res)
    img = res * 255
    img = tf.cast(img, dtype=tf.dtypes.uint8)
    img = tf.image.encode_jpeg(
        img, quality=100)
    predicted_img_path = 'results/predicted_mask.jpg'
    tf.io.write_file(predicted_img_path, img)
    original_image = resize(image_path)
    predicted_mask = resize(predicted_img_path)
    create_img(original_image, predicted_mask)


def show_batch_blur():
    size = 10
    base_image_dir = 'data/raw/images'
    base_mask_dir = 'data/raw/masks'
    images_dir = os.listdir(base_image_dir)
    masks_dir = os.listdir(base_mask_dir)
    images = []
    masks = []
    model = predict_model(show=False)
    for _ in range(size):
        randn = random.choice(range(len(images_dir)))
        images.append(os.path.join(base_image_dir, images_dir[randn]))
        masks.append(os.path.join(base_mask_dir, masks_dir[randn]))
    for idx in range(len(images)):
        image_path = images[idx]
        true_mask_path = masks[idx]
        img = prepare_image(image_path)
        res = model.predict(img)
        res = create_mask(res)
        mask = prepare_image(true_mask_path, mask=True)
        iou = iou_score(mask[0], res)
        img = res * 255
        img = tf.cast(img, dtype=tf.dtypes.uint8)
        img = tf.image.encode_jpeg(
            img, quality=100)
        predicted_mask_path = f'results/predicted_mask_{idx}.jpg'
        tf.io.write_file(predicted_mask_path, img)
        original_image = resize(image_path)
        predicted_mask = resize(predicted_mask_path)
        true_mask = resize(true_mask_path)
        create_img(original_image, predicted_mask, true_mask, iou)


def test():
    df_path = "reports/SegNet_history.csv"
    df = pd.read_csv(df_path)
    plot_loss_vs_epoch(df)
    plot_accuracy_vs_epoch(df)


def augment():
    file_path = "src/test/test.JPG"
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=3, dtype=tf.float32)
    img = tf.image.resize(img, [224, 224])
    augment = DataAugmentation(
        rotation_range=20,  zoom_range=(.5, 0.5), delta=0.5, saturation_factor=0.3, subset_size=0.20)
    print(type(img.numpy()))
    image, mask = augment.adjust_light(img.numpy(), img.numpy())
    fig = plt.figure(frameon=False)
    fig.set_size_inches(5, 5)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect='auto')
    plt.show()


if __name__ == "__main__":
    print("Select one Option:")
    print("1. Train Model")
    print("2. Predict Model")
    print("3. Save Blurred Image")
    print("4. Blur Batch Image")
    print("5. PLOT GRAPH")
    choice = int(input("Choice: "))
    if choice == 1:
        train_model()

    if choice == 2:
        predict_model()
    if choice == 3:
        blur_image()
    # FOR TESTING PURPOSES ONLY
    if choice == 4:
        show_batch_blur()
    if choice == 5:
        test()
