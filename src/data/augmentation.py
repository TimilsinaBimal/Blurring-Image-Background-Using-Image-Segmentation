import tensorflow as tf
from tensorflow.keras.preprocessing.image import random_rotation, random_shift, random_zoom, random_shear


class DataAugmentation:
    def __init__(self, rotation_range=0, vertical_shift=0, horizontal_shift=0, zoom_range=(0, 0), shear_intensity=0):
        self.rotation_range = rotation_range
        self.vertical_shift = vertical_shift
        self.horizontal_shift = horizontal_shift
        self.zoom_range = zoom_range
        self.shear_intensity = shear_intensity

    def rotation(self, image, mask):
        image = random_rotation(
            image, rg=self.rotation_range, row_axis=0, col_axis=1, channel_axis=2)
        mask = random_rotation(
            mask, rg=self.rotation_range, row_axis=0, col_axis=1, channel_axis=2)
        return image, mask

    def vertical_shift(self, image, mask):
        image = random_shift(image, 0, self.vertical_shift,
                             row_axis=0, col_axis=1, channel_axis=2)
        mask = random_shift(mask, 0, self.vertical_shift,
                            row_axis=0, col_axis=1, channel_axis=2)
        return image, mask

    def horizontal_shift(self, image, mask):
        image = random_shift(image, self.horizontal_shift,
                             0, row_axis=0, col_axis=1, channel_axis=2)
        mask = random_shift(mask, self.horizontal_shift, 0,
                            row_axis=0, col_axis=1, channel_axis=2)
        return image, mask

    def zoom(self, image, mask):
        image = random_zoom(image, self.zoom_range, row_axis=0,
                            col_axis=1, channel_axis=2)
        mask = random_zoom(mask, self.zoom_range, row_axis=0,
                           col_axis=1, channel_axis=2)
        return image, mask

    def shear(self, image, mask):
        image = random_shear(image, self.shear_intensity,
                             row_axis=0, col_axis=1, channel_axis=2)
        mask = random_shear(mask, self.shear_intensity,
                            row_axis=0, col_axis=1, channel_axis=2)
        return image, mask

    @tf.function
    def apply(self, dataset):
        if tf.random.uniform((), minval=0, maxval=1) < 0.2:
            dataset = dataset.map(
                self.rotation, num_parallel_calls=tf.data.AUTOTUNE)
        if tf.random.uniform((), minval=0, maxval=1) < 0.2:
            dataset = dataset.map(
                self.vertical_shift, num_parallel_calls=tf.data.AUTOTUNE)
        if tf.random.uniform((), minval=0, maxval=1) < 0.2:
            dataset = dataset.map(
                self.horizontal_shift, num_parallel_calls=tf.data.AUTOTUNE)
        if tf.random.uniform((), minval=0, maxval=1) < 0.2:
            dataset = dataset.map(
                self.zoom, num_parallel_calls=tf.data.AUTOTUNE)
        if tf.random.uniform((), minval=0, maxval=1) < 0.2:
            dataset = dataset.map(
                self.rotation, num_parallel_calls=tf.data.AUTOTUNE)
        return dataset
