import os
import pandas as pd
from pathlib import Path
from data.prepare import Dataset
from models.models import SegNet, PSPNet, UNet, DeepLab, ImageSegmentation
from models.predict import show_predictions, create_mask, predict
from models.train import (
    train_segnet,
    train_deeplab,
    train_pspnet,
    train_unet,
    train,
    train_segmentation
)
from visualization.image import display


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


def predict_model():
    print("Please Choose:")
    print("1: Get Accuracy and Loss of test dataset.")
    print("2. Get accuracy and Loss of test dataset with some results:")
    choice = int(input("Your Choice: "))

    BEST_MODEL = "segnet"
    ROOT_DIR = Path(__file__).parent.parent

    if BEST_MODEL == "unet":
        model = train_unet()
        model_path = os.path.join(ROOT_DIR, "models/unet/")
    elif BEST_MODEL == "segnet":
        model = train_segnet()
        model_path = os.path.join(ROOT_DIR, "models/segnet/")
    elif BEST_MODEL == "pspnet":
        model = train_pspnet()
        model_path = os.path.join(ROOT_DIR, "models/pspnet/")

    model.load_weights(model_path)

    dataset = Dataset(root_dir=ROOT_DIR,
                      batch_size=1, validation=True)

    _, _, test_dataset = dataset.make()

    if choice == 1:
        predict(model, test_dataset)
    elif choice == 2:
        loss, accuracy = predict(model, test_dataset)
        for image in test_dataset.take(10):
            show_predictions(model, image[0][0], image[1][0])


def test():
    ROOT_DIR = Path(__file__).parent.parent
    dataset = Dataset(root_dir=ROOT_DIR,
                      batch_size=1, validation=True)

    _, _, test_dataset = dataset.make()

    for image in test_dataset.take(10):
        display([image[0][0], image[1][0]])


if __name__ == "__main__":
    print("Select one Option:")
    print("1. Train Model")
    print("2. Predict Model")
    print("3. Test")
    choice = int(input("Choice: "))
    if choice == 1:
        train_model()

    if choice == 2:
        predict_model()

    # FOR TESTING PURPOSES ONLY
    if choice == 3:
        test()
