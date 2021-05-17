import tensorflow as tf
from src.utils.layers import CNNBlock, BlockLayer
from tensorflow.keras import models, layers
from tensorflow.keras.applications import VGG16


def VGG_Model(include_last=True):
    vgg = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    if include_last:
        output_layers = [
            'block1_conv2',
            'block2_conv2',
            'block3_conv3',
            'block4_conv3',
            'block5_conv3',
            'block5_pool',
        ]
    else:
        output_layers = [
            'block1_conv2',
            'block2_conv2',
            'block3_conv3',
            'block4_conv3',
            'block4_pool',
        ]
    output = [vgg.get_layer(layer).output for layer in output_layers]

    model = models.Model(inputs=vgg.input, outputs=output)
    model.trainable = False
    return model


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


def UNet(out_channels=2, features=[64, 128, 256, 512, 1024]):
    input = layers.Input(shape=(224, 224, 3))
    x = VGG_Model(include_last=False)(input)
    skip_connections = x[:-1]
    x = x[-1]
    x = CNNBlock(features[-1]*2)(x)
    x = CNNBlock(features[-1]*2)(x)

    skip_connections = skip_connections[::-1]

    for idx, feature in enumerate(reversed(features)):
        x = layers.Conv2DTranspose(
            feature, kernel_size=2, strides=2, padding="same")(x)
        skip_connection = skip_connections[idx]
        crop_amount = (skip_connection.shape[1] - x.shape[1]) // 2
        if x.shape != skip_connection.shape:
            skip_connection = tf.keras.layers.Cropping2D(
                cropping=crop_amount)(skip_connection)

        x = layers.Concatenate()([x, skip_connection])
        x = CNNBlock(feature)(x)
        x = CNNBlock(feature)(x)

    x = layers.Conv2D(out_channels, kernel_size=1)(x)
    x = layers.Softmax()(x)
    model = models.Model(inputs=input, outputs=x)
    return model
