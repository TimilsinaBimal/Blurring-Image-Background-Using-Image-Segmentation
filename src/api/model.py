import tensorflow as tf
from layers import BlockLayer
from tensorflow.keras import models, layers
from tensorflow.keras.applications.vgg16 import VGG16


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


def train_segnet():
    model = SegNet()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model
