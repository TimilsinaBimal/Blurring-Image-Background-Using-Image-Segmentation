import tensorflow as tf
from tensorflow.keras import layers, models


class CNNBlock(layers.Layer):
    def __init__(self, out_channels, kernel_size=3, padding="SAME"):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, padding=padding)
        self.batchNorm = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.batchNorm(x, training=training)
        x = tf.nn.relu(x)
        return x


class BlockLayer(layers.Layer):
    def __init__(self, out_channels, kernel_size=3, padding="SAME", no_layers=3):
        super(BlockLayer, self).__init__()
        self.no_layers = no_layers
        self.cnn1 = CNNBlock(out_channels[0])
        self.cnn2 = CNNBlock(out_channels[1])
        if self.no_layers == 3:
            self.cnn3 = CNNBlock(out_channels[2])
        self.upsampling = layers.UpSampling2D()
        self.add = layers.Add()

    def call(self, input_tensor, training=False, intermediate_op=None):
        x = self.upsampling(input_tensor)
        x = self.add([x, intermediate_op])
        x = self.cnn1(x)
        x = self.cnn2(x)
        if self.no_layers == 3:
            x = self.cnn3(x)
        return x
