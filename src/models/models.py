import tensorflow as tf
from utils import CNNBlock, BlockLayer
from tensorflow.keras import models, layers
from tensorflow.keras.applications import VGG16, ResNet50


# TODO Make models compatible with keras tuner for hyperparameter tuning for models such as Segnet,PSPNet and ImageSegmentation

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


def ResNet():
    resnet = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    outputs = resnet.get_layer("conv2_block3_out").output
    model = models.Model(inputs=resnet.inputs, outputs=outputs)
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


class UNet(models.Model):
    def __init__(self, out_channels=2, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.out_channels = out_channels
        self.features = features
        self.vgg_model = VGG_Model(include_last=False)
        self.block1_1 = CNNBlock(self.features[-1]*2)
        self.block1_2 = CNNBlock(self.features[-1]*2)
        self.conv = layers.Conv2D(self.out_channels, kernel_size=1)
        self.softmax = layers.Softmax()
        for idx, feature in enumerate(reversed(self.features)):
            vars(self)[f"concat_{idx}"] = layers.Concatenate()
            vars(self)[f'cnn_block_{idx}_1'] = CNNBlock(feature)
            vars(self)[f'cnn_block_{idx}_2'] = CNNBlock(feature)
            vars(self)[f'upsample_{idx}'] = layers.Conv2DTranspose(
                feature, kernel_size=2, strides=2, padding="same")

    def call(self, input_tensor):
        x = self.vgg_model(input_tensor)
        skip_connections = x[:-1]
        x = x[-1]
        x = self.block1_1(x)
        x = self.block1_2(x)

        skip_connections = skip_connections[::-1]

        for idx in range(len(self.features)):
            x = vars(self)[f'upsample_{idx}'](x)
            x = vars(self)[f"concat_{idx}"]([x, skip_connections[idx]])
            x = vars(self)[f'cnn_block_{idx}_1'](x)
            x = vars(self)[f'cnn_block_{idx}_2'](x)

        x = self.conv(x)
        x = self.softmax(x)
        return x

    def model(self):
        x = layers.Input(shape=(224, 224, 3))
        return models.Model(inputs=[x], outputs=self.call(x))


class PSPNet(models.Model):
    def __init__(self, out_channels=2, kernel_size=[1, 2, 3, 4], features=[1024, 512]):
        super(PSPNet, self).__init__()
        self.vgg = VGG_Model(include_last=False)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv1 = layers.Conv2D(256, kernel_size=3, padding='same')
        self.pooling = layers.MaxPooling2D(pool_size=2, padding='same')
        for idx in range(1, 5):
            vars(self)[f'conv3_{idx}'] = layers.Conv2D(
                features[0], kernel_size=self.kernel_size[idx-1], padding='same')
            vars(self)[f'conv4_{idx}'] = layers.Conv2D(
                features[1], kernel_size=self.kernel_size[idx-1], padding='same')
        self.upsample = layers.Conv2DTranspose(
            256, kernel_size=2, padding='same')
        self.upsample_2 = layers.Conv2DTranspose(
            256, kernel_size=2, strides=4, padding='same')
        self.upsample_3 = layers.Conv2DTranspose(
            256, kernel_size=2, strides=4, padding='same')
        self.final_conv = layers.Conv2D(
            self.out_channels, kernel_size=3, padding='same')
        self.softmax = layers.Softmax()
        self.concat = layers.Concatenate()

    def call(self, input_tensor):
        x = self.vgg(input_tensor)
        x = x[-1]
        int_layers = [x]
        for idx in range(1, 5):
            op = vars(self)[f'conv3_{idx}'](x)
            op = vars(self)[f"conv4_{idx}"](op)
            int_layers.append(self.upsample(op))
        x = self.concat(int_layers)
        x = self.upsample_2(x)
        x = self.conv1(x)
        x = self.upsample_3(x)
        x = self.final_conv(x)
        x = self.softmax(x)
        return x

    def model(self):
        x = layers.Input(shape=(224, 224, 3))
        return models.Model(inputs=[x], outputs=self.call(x))


class DeepLab(models.Model):
    def __init__(self, out_channels=2):
        super(DeepLab, self).__init__()
        self.out_channels = out_channels
        self.resnet = ResNet()
        self.one_1_conv = layers.Conv2D(128, kernel_size=1, padding="same")

        strides = [1, 4, 8, 12]
        kernels = [1, 3, 3, 3]
        for idx in range(1, 5):
            vars(self)[f'conv_3_{idx}'] = layers.Conv2D(
                512, kernel_size=kernels[idx-1], strides=strides[idx-1], padding="same")

        self.conv_up_4 = layers.Conv2DTranspose(
            128, kernel_size=2, strides=4)
        for idx in range(1, 4):
            vars(self)[f'conv_{idx}'] = CNNBlock(128)
            vars(self)[f'conv_up_{idx}'] = layers.Conv2DTranspose(
                128, kernel_size=2, strides=2)

        for idx in range(1, 3):
            vars(self)[f'concat_{idx}'] = layers.Concatenate()
            vars(self)[f'pool_{idx}'] = layers.MaxPooling2D(padding="same")

        self.final_layer = layers.Conv2D(
            self.out_channels, kernel_size=1, padding='same')
        self.softmax = layers.Softmax()
        self.zero_padding = layers.ZeroPadding2D(padding=1)

    def call(self, input_tensor):
        x = self.resnet(input_tensor)
        add = tf.identity(x)

        # Intermediate Layers
        concat_layers = []
        layer_1 = self.conv_3_1(x)
        concat_layers.append(layer_1)

        layer_2 = self.conv_3_2(x)
        layer_2 = self.pool_2(layer_2)
        concat_layers.append(layer_2)

        layer_3 = self.conv_3_3(x)
        concat_layers.append(layer_3)

        layer_4 = self.conv_3_4(x)
        layer_4 = self.zero_padding(layer_4)
        concat_layers.append(layer_4)

        x = self.concat_1(concat_layers[1:])
        # UPScale by 4
        x = self.conv_up_4(x)
        x = self.conv_1(x)
        # upsample by 2
        x = self.conv_up_1(x)
        # 1x1 conv
        add = self.one_1_conv(add)
        x = self.concat_2([add, x, concat_layers[0]])
        x = self.conv_up_2(x)
        x = self.conv_2(x)

        x = self.conv_up_3(x)
        x = self.conv_3(x)
        x = self.final_layer(x)
        x = self.softmax(x)
        return x

    def model(self):
        x = layers.Input(shape=(224, 224, 3))
        return models.Model(inputs=x, outputs=self.call(x))


# TODO Change Architecture of Image Segmentation Model to perform better on dataset
class ImageSegmentation(models.Model):
    def __init__(self, out_channels=2):
        super(ImageSegmentation, self).__init__()
        self.out_channels = out_channels
        self.vgg = VGG_Model(include_last=True)
        kernel_size = [1, 2, 3, 4]
        strides = [1, 1, 2, 2]
        filters = [512, 256, 128, 64]
        self.att_concat = layers.Concatenate()
        self.att_upsample = layers.Conv2DTranspose(
            2048, kernel_size=2, strides=2, padding='same')
        for idx in range(1, 5):
            if idx > 2:
                vars(self)[f'att_upsample_{idx}'] = layers.Conv2DTranspose(
                    512, kernel_size=4, strides=1, padding='valid')
            vars(self)[f'att_conv_{idx}'] = layers.Conv2D(
                1024, kernel_size=2, strides=1, padding='same')
            vars(self)[f'conv_0_{idx}'] = layers.Conv2D(
                1024, kernel_size=kernel_size[idx-1], strides=strides[idx-1], padding='same')
            vars(self)[f'concat_{idx}'] = layers.Concatenate()
            vars(self)[f'upsample_{idx}'] = layers.Conv2DTranspose(
                filters[idx-1], kernel_size=2, strides=2, padding='same')
            vars(self)[f'conv_1_{idx}'] = CNNBlock(filters[idx-1])
            vars(self)[f'conv_2_{idx}'] = CNNBlock(filters[idx-1])

        self.pool_1 = layers.MaxPooling2D(padding='same')
        self.conv11 = layers.Conv2D(512, kernel_size=1, padding='same')
        self.up_by_3 = layers.Conv2DTranspose(
            512, kernel_size=4, strides=1, padding='valid')

        self.final_layer = layers.Conv2D(
            self.out_channels, kernel_size=1, padding='same')
        self.softmax = layers.Softmax()

    def call(self, input_tensor):
        x = self.vgg(input_tensor)
        identity_layers = x[:-1]
        x = x[-1]

        # Attrous Block
        attrs_layers = []
        for idx in range(1, 5):
            temp_layer = vars(self)[f'conv_0_{idx}'](x)
            if idx > 2:
                temp_layer = vars(self)[f'att_upsample_{idx}'](temp_layer)

            temp_layer = vars(self)[f'att_conv_{idx}'](temp_layer)
            attrs_layers.append(temp_layer)

        conv1 = self.conv11(x)
        attrs_layers.append(conv1)
        x = self.att_concat(attrs_layers)
        x = self.att_upsample(x)
        identity_layers = identity_layers[::-1]
        for idx in range(1, 5):
            x = vars(self)[f'concat_{idx}']([x, identity_layers[idx-1]])
            x = vars(self)[f'conv_1_{idx}'](x)
            x = vars(self)[f'conv_2_{idx}'](x)
            x = vars(self)[f"upsample_{idx}"](x)

        x = self.final_layer(x)
        x = self.softmax(x)
        return x

    def model(self):
        inputs = layers.Input(shape=(224, 224, 3))
        return models.Model(inputs=inputs, outputs=self.call(inputs))
