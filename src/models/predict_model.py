import os
from pathlib import Path
import tensorflow as tf
from src.data.make_dataset import Dataset
from src.visualization.display_image import display
from src.models.models import UNet, SegNet


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(model, image, mask):
    prediction = model.predict(image[tf.newaxis, ...])
    display([image, mask, create_mask(prediction)])


def predict(model, dataset):
    return model.evaluate(dataset, steps=100)


if __name__ == '__main__':
    print("Please Choose:")
    print("1: Get Accuracy and Loss of test dataset.")
    print("2. Get accuracy and Loss of test dataset with some results:")
    choice = int(input("Your Choice: "))

    BEST_MODEL = "unet"
    ROOT_DIR = Path(__file__).parent.parent.parent

    if BEST_MODEL == "unet":
        model = UNet()
        model_path = os.path.join(ROOT_DIR, "models/unet/")
    elif BEST_MODEL == "segnet":
        model = SegNet()
        model_path = os.path.join(ROOT_DIR, "models/segnet/")

    model.load_weights(model_path)

    dataset = Dataset(root_dir=ROOT_DIR,
                      batch_size=1, validation=True)

    train_dataset, validation_dataset, test_dataset = dataset.make()

    if choice == 1:
        predict()
    elif choice == 2:
        loss, accuracy = predict()
        for image in test_dataset.take(10):
            show_predictions(model, image[0][0], image[1][0])
