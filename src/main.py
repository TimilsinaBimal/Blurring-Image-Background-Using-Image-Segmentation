import os
import pandas as pd
from pathlib import Path
from data.prepare import Dataset
from models.predict import show_predictions, create_mask, predict
from models.train import (
    train_segnet,
    train_deeplab,
    train_pspnet,
    train_unet,
    train,
    train_segmentation
)
import tensorflow as tf
from visualization.image import display
from blur.blur import prepare_image, create_img
from visualization.graphs import plot_accuracy_vs_epoch, plot_loss_vs_epoch
from data.augmentation import DataAugmentation
import matplotlib.pyplot as plt


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
    BEST_MODEL = "pspnet"
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
        for image in train_dataset.take(10):
            show_predictions(model, image[0][0], image[1][0])


def blur_image():
    model = predict_model(show=False)
    image_path = "src/test/test.JPG"
    img = prepare_image(image_path)
    res = model.predict(img)
    res = create_mask(res)


def test():
    df_path = "reports/DeepLab_history.csv"
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
    print("4. Test")
    print("5. Augment")
    choice = int(input("Choice: "))
    if choice == 1:
        train_model()

    if choice == 2:
        predict_model()
    if choice == 3:
        blur_image()
    # FOR TESTING PURPOSES ONLY
    if choice == 4:
        test()
    if choice == 5:
        augment()
