import os
import tensorflow as tf
from models.models import DeepLab, ImageSegmentation, PSPNet, SegNet, UNet
from data.prepare import Dataset


def custom_callbacks(model, ROOT_DIR) -> list:
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


def train_segmentation():
    model = ImageSegmentation()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.99),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def train(model, ROOT_DIR):
    # Get the dataset for training
    dataset = Dataset(root_dir=ROOT_DIR,
                      batch_size=1, validation=True)

    train_dataset, validation_dataset, test_dataset = dataset.make()

    # Train the Model
    BATCH_SIZE = dataset.BATCH_SIZE
    TRAIN_DATA_LEN = dataset.TRAIN_DATA_LEN
    EPOCHS = 20
    VALIDATION_LEN = dataset.VAL_DATA_LEN
    steps_per_epoch = TRAIN_DATA_LEN // BATCH_SIZE
    validation_steps = VALIDATION_LEN // BATCH_SIZE

    callbacks = custom_callbacks(model, ROOT_DIR)
    history = model.fit(train_dataset, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,  validation_data=(
        validation_dataset), validation_steps=validation_steps, callbacks=callbacks, verbose=1)

    return history
