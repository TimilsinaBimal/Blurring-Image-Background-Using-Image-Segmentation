import tensorflow as tf
from tensorflow.keras.preprocessing.image import random_rotation, random_zoom


class DataAugmentation:
    def __init__(self, rotation_range=0, zoom_range=(0, 0), delta=0, saturation_factor=0, subset_size=0.2):
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.subset_size = subset_size
        self.saturation_factor = saturation_factor
        self.delta = delta

    def rotation(self, image, mask):
        image = random_rotation(
            image, rg=self.rotation_range, row_axis=0, col_axis=1, channel_axis=2)
        mask = random_rotation(
            mask, rg=self.rotation_range, row_axis=0, col_axis=1, channel_axis=2)
        return image, mask

    def flip_left_right(self, image, mask):
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
        return image, mask

    def flip_up_down(self, image, mask):
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
        return image, mask

    def zoom(self, image, mask):
        image = random_zoom(image, self.zoom_range, row_axis=0,
                            col_axis=1, channel_axis=2)
        mask = random_zoom(mask, self.zoom_range, row_axis=0,
                           col_axis=1, channel_axis=2)
        return image, mask

    def adjust_light(self, image, mask):
        randm = tf.random.uniform(())
        if randm < 0.25:
            image = tf.image.adjust_brightness(image, self.delta)
            mask = tf.image.adjust_brightness(mask, self.delta)

        if 0.25 < randm < 0.50:
            image = tf.image.adjust_contrast(image, self.delta)
            mask = tf.image.adjust_contrast(mask, self.delta)

        if 0.5 < randm < 0.75:
            image = tf.image.adjust_saturation(image, self.saturation_factor)
            mask = tf.image.adjust_saturation(mask, self.saturation_factor)

        if 0.75 < randm < 1:
            image = tf.image.adjust_hue(image, self.delta)
            mask = tf.image.adjust_hue(mask, self.delta)

        return image, mask

    def apply(self, image, mask):
        randm = tf.random.uniform(())
        if randm < self.subset_size:
            image, mask = self.rotation(image, mask)

        if self.subset_size < randm < self.subset_size*2:
            image, mask = self.flip_left_right(image, mask)

        if self.subset_size*2 < randm < self.subset_size*3:
            image, mask = self.flip_top_down(image, mask)

        if self.subset_size*3 < randm < self.subset_size*4:
            image, mask = self.adjust_light(image, mask)

        if self.subset_size*4 < randm < self.subset_size*5:
            image, mask = self.zoom(image, mask)

        return image, mask
