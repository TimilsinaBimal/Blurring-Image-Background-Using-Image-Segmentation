import os
import pandas as pd
from pathlib import Path
from models.train_model import (
    train_segnet,
    train_deeplab,
    train_pspnet,
    train_unet,
    train,
    train_segmentation
)


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


if __name__ == "__main__":
    print("Choose:")
    print("1. Train Model")
    choice = int(input("Choice: "))
    if choice == 1:
        train_model()
