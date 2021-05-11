import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16


def VGG_Model():
    vgg = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    output_layers = [
        'block1_conv2',
        'block2_conv2',
        'block3_conv3',
        'block4_conv3',
        'block5_conv3',
        'block5_pool',
    ]
    output = [vgg.get_layer(layer).output for layer in output_layers]

    model = models.Model(inputs=vgg.input, outputs=output)
    model.trainable = False
    return model


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


class SegNet(models.Model):
    def __init__(self):
        super(SegNet, self).__init__()
        self.encoder = VGG_Model()
        self.block1 = BlockLayer([512, 512, 512])
        self.block2 = BlockLayer([256, 256, 256])
        self.block3 = BlockLayer([128, 128, 128])
        self.block4 = BlockLayer([64, 64], no_layers=2)
        self.block5 = BlockLayer([64, 2], no_layers=2)
        self.softmax = layers.Softmax()

    def call(self, input_tensor, training=False):
        x = self.encoder(input_tensor)
        x, a1, a2, a3, a4, a5 = reversed(x)
        x = self.block1(x, intermediate_op=a1)
        x = self.block2(x, intermediate_op=a2)
        x = self.block3(x, intermediate_op=a3)
        x = self.block4(x, intermediate_op=a4)
        x = self.block5(x, intermediate_op=a5)
        x = self.softmax(x)
        return x

    def model(self):
        x = layers.Input(shape=(224, 224, 3))
        return models.Model(inputs=[x], outputs=self.call(x))
