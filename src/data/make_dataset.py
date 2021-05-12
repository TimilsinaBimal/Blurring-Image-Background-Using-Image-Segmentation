import os
import tensorflow as tf
from sklearn.model_selection import train_test_split


class Dataset:
    """
        Preprocess the dataset from raw dataset to train model.
        ARGUMENTS
        ---
        base_dir: Base data directory path
        batch_size: Batch size to generate
        image_size: [default (224,224,3)] Size of image
        repeat: (default=True) Whether to repeat data while training
    """

    def __init__(self, base_dir, batch_size, image_size=(224, 224, 3), BUFFER_SIZE=100, validation=False):
        self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.CHANNELS = image_size
        self.ROOT_DIR = base_dir
        self.DATA_DIR = "data/raw/"
        self.BATCH_SIZE = batch_size
        self.BUFFER_SIZE = BUFFER_SIZE
        self.validation = validation
        self.BASE_IMAGE_DIR = os.path.join(
            self.ROOT_DIR, self.DATA_DIR, "images/")
        self.BASE_MASK_DIR = os.path.join(
            self.ROOT_DIR, self.DATA_DIR, "masks")

    def get_file(self, _path):
        for (_, _, files) in os.walk(_path):
            return files

    def split_data(self, test_size=0.25, val_size=0.15):
        train_images, test_images, train_masks, test_masks = train_test_split(
            self.get_file(self.BASE_IMAGE_DIR), self.get_file(self.BASE_MASK_DIR), test_size=val_size)

        if self.validation:
            train_images, val_images, train_masks, val_masks = train_test_split(
                train_images, train_masks, test_size=test_size)

            return train_images, val_images, test_images, train_masks, val_masks, test_masks

        return train_images, test_images, train_masks, test_masks

    def prepare_image(self, file_path, masks=False):
        img = tf.io.read_file(file_path)
        if masks:
            img = tf.io.decode_image(img, channels=1, dtype=tf.float32)
        else:
            img = tf.io.decode_image(img, channels=3, dtype=tf.float32)

        img = tf.image.resize(img, [self.IMAGE_HEIGHT, self.IMAGE_WIDTH])
        return img

    def prepare_dataset(self, images, masks):
        for idx in range(len(images)):
            image_path = os.path.join(
                self.BASE_IMAGE_DIR, images[idx].decode())
            img = self.prepare_image(image_path)

            mask_path = os.path.join(self.BASE_MASK_DIR, masks[idx].decode())
            mask = self.prepare_image(mask_path, masks=True)
            yield img, mask

    def make_dataset(self, images, masks, training=True):
        dataset = tf.data.Dataset.from_generator(
            self.prepare_dataset,
            args=[images, masks],
            output_signature=(
                tf.TensorSpec(
                    shape=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.CHANNELS), dtype=tf.float32),
                tf.TensorSpec(shape=(self.IMAGE_HEIGHT,
                              self.IMAGE_WIDTH, 1), dtype=tf.float32)
            )
        )
        if training:
            dataset = dataset.shuffle(
                self.BUFFER_SIZE).repeat().batch(self.BATCH_SIZE)
        else:
            dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def make(self, training=True, validation=False):
        if self.validation:
            train_images, val_images, test_images, train_masks, val_masks, test_masks = self.split_data()
        else:
            train_images, test_images, train_masks, test_masks = self.split_data()

        if training:
            dataset = self.make_dataset(train_images, train_masks)
        elif validation:
            dataset = self.make_dataset(
                val_images, val_masks, training=False)
        else:
            dataset = self.make_dataset(
                test_images, test_masks, training=False)

        return dataset
