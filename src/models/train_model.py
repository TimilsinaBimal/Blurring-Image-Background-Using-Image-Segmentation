import os
from pathlib import Path
import pandas as pd
import tensorflow as tf
from src.models.models import SegNet, UNet
from src.data.make_dataset import Dataset


def checkpoints() -> list:
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ROOT_DIR, "models/unet/"), monitor='val_accuracy', verbose=1, save_best_only=True,
        save_weights_only=True, mode='max', save_freq='epoch'
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=3, verbose=1,
        mode='max', restore_best_weights=True
    )
    return [model_checkpoint, early_stopping]


def train(model_selection):
    if model_selection == 1:
        model = SegNet()
    elif model_selection == 2:
        model = UNet()

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Get the dataset for training
    dataset = Dataset(root_dir=ROOT_DIR,
                      batch_size=1, validation=True)

    train_dataset, validation_dataset, test_dataset = dataset.make()

    # Train the Model

    history = model.fit(train_dataset, steps_per_epoch=200, epochs=20, validation_data=(
        validation_dataset), validation_steps=200, callbacks=checkpoints(), verbose=1)

    return history, model


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.parent
    print("Enter the model you want to train:")
    print("1: SegNet")
    print("2: UNet")
    model_selection = int(input("Your Choice: "))
    history, model = train(model_selection)
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(ROOT_DIR, "reports/history.csv"), index=False)
