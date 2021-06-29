import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data.augmentation import DataAugmentation


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

    def __init__(self, root_dir, batch_size, image_size=(224, 224, 3), BUFFER_SIZE=50, validation=False):
        self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.CHANNELS = image_size
        self.ROOT_DIR = root_dir
        self.DATA_DIR = "data/raw/"
        self.BATCH_SIZE = batch_size
        self.BUFFER_SIZE = BUFFER_SIZE
        self.validation = validation
        self.BASE_IMAGE_DIR = os.path.join(
            self.ROOT_DIR, self.DATA_DIR, "images/")
        self.BASE_MASK_DIR = os.path.join(
            self.ROOT_DIR, self.DATA_DIR, "masks")
        self.TRAIN_DATA_LEN = 0
        self.TEST_DATA_LEN = 0
        self.VAL_DATA_LEN = 0
        self.test = []

    def get_file(self, _path):
        for (_, _, files) in os.walk(_path):
            return files

    def split_data(self, test_size=0.15, val_size=0.15):
        train_images, test_images, train_masks, test_masks = train_test_split(
            self.get_file(self.BASE_IMAGE_DIR), self.get_file(self.BASE_MASK_DIR), test_size=val_size, random_state=42)
        self.TRAIN_DATA_LEN = len(train_images)
        self.TEST_DATA_LEN = len(test_images)
        self.test += test_images

        if self.validation:
            train_images, val_images, train_masks, val_masks = train_test_split(
                train_images, train_masks, test_size=test_size, random_state=42)
            # train_images, val_images, train_masks, val_masks = train_images[
            #     :50], val_images[:10], train_masks[:50], val_masks[:10]
            self.TRAIN_DATA_LEN = len(train_images)
            self.TEST_DATA_LEN = len(test_images)
            self.VAL_DATA_LEN = len(val_images)

            return train_images, val_images, test_images, train_masks, val_masks, test_masks

        return train_images, test_images, train_masks, test_masks

    def augment(self, image, mask):
        augment = DataAugmentation(
            rotation_range=20,  zoom_range=(0.5, 0.5), delta=0.5, saturation_factor=0.3, subset_size=0.20)
        image, mask = augment.apply(image.numpy(), mask.numpy())
        return image, mask

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
            if idx >= self.TRAIN_DATA_LEN:
                img, mask = self.augment(img, mask)
            yield img, mask

    def make_dataset(self, images, masks, training=True):
        if training:
            images = images[::] + images[:len(images)]
            masks = masks[::] + masks[:len(masks)]
            self.TRAIN_DATA_LEN = len(images)
        dataset = tf.data.Dataset.from_generator(
            self.prepare_dataset,
            args=[images, masks],
            output_types=(tf.float32, tf.float32),
            output_shapes=(
                (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.CHANNELS),
                (self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1)
            )
        )
        if training:
            dataset = dataset.shuffle(
                self.BUFFER_SIZE).repeat().batch(self.BATCH_SIZE)
        else:
            dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def make(self):
        if self.validation:
            train_images, val_images, test_images, train_masks, val_masks, test_masks = self.split_data()
            validation_dataset = self.make_dataset(
                val_images, val_masks, training=False)
        else:
            train_images, test_images, train_masks, test_masks = self.split_data()

        train_dataset = self.make_dataset(
            train_images, train_masks, training=True)

        test_dataset = self.make_dataset(
            test_images, test_masks, training=False)

        if self.validation:
            return train_dataset, validation_dataset, test_dataset
        else:
            return train_dataset, test_dataset
