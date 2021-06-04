import os
from pathlib import Path
import pandas as pd
import tensorflow as tf
from src.models.models import DeepLab, PSPNet, SegNet, UNet
from src.data.make_dataset import Dataset


def custom_callbacks(model) -> list:
    model_name = model.__class__.__name__
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ROOT_DIR, f"models/{model_name}/"), monitor='val_accuracy', verbose=1, save_best_only=True,
        save_weights_only=False, mode='max', save_freq='epoch'
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=3, verbose=1,
        mode='max', restore_best_weights=True
    )
    return [model_checkpoint, early_stopping]


def train_segnet():
    model = SegNet()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def train_unet():
    model = UNet()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.99),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def train_pspnet():
    model = PSPNet()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.99),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def train_deeplab():
    model = DeepLab()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.99),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def train(model):
    # Get the dataset for training
    dataset = Dataset(root_dir=ROOT_DIR,
                      batch_size=1, validation=True)

    train_dataset, validation_dataset, test_dataset = dataset.make()

    # Train the Model
    TRAIN_DATA_LEN = dataset.TRAIN_DATA_LEN
    EPOCHS = 20
    VALIDATION_LEN = dataset.VAL_DATA_LEN
    callbacks = custom_callbacks(model)
    history = model.fit(train_dataset, steps_per_epoch=TRAIN_DATA_LEN, epochs=EPOCHS,  validation_data=(
        validation_dataset), validation_steps=VALIDATION_LEN, callbacks=callbacks, verbose=1)

    return history


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.parent
    print("Enter the model you want to train:")
    print("1: SegNet")
    print("2: UNet")
    print("3: PSPNet")
    print("4: DeepLab")
    model_selection = int(input("Your Choice: "))
    if model_selection == 1:
        model = train_segnet()
    elif model_selection == 2:
        model = train_unet()
    elif model_selection == 3:
        model = train_pspnet()
    elif model_selection == 4:
        model = train_deeplab()
    df_file = model.__class__.__name__
    history = train(model)
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(ROOT_DIR, f"{df_file}_history.csv"), index=False)
