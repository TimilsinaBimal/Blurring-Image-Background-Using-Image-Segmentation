import os
from pathlib import Path
import tensorflow as tf
from src.models.segNet import SegNet
from src.data.make_dataset import Dataset
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def checkpoints():
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(ROOT_DIR, "models/segnet/"), monitor='val_accuracy', verbose=1, save_best_only=True,
        save_weights_only=True, mode='max', save_freq='epoch'
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=3, verbose=1,
        mode='max', restore_best_weights=True
    )
    return [model_checkpoint, early_stopping]


def train():
    # Get Model and Compile it
    model = SegNet()
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


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent.parent.parent
    train()
