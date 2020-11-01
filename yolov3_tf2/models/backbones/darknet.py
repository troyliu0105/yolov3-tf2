import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Concatenate,
    Conv2D,
    Input,
    Lambda,
    LeakyReLU,
    MaxPool2D,
    UpSampling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2


def darknet_conv(x, filters, size, strides=1, batch_norm=True):
    if strides == 1:
        padding = 'same'
    else:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    x = Conv2D(filters=filters, kernel_size=size,
               strides=strides, padding=padding,
               use_bias=not batch_norm, kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def darknet_residual(x, filters):
    prev = x
    x = darknet_conv(x, filters // 2, 1)
    x = darknet_conv(x, filters, 3)
    x = Add()([prev, x])
    return x


def darknet_block(x, filters, blocks):
    x = darknet_conv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = darknet_residual(x, filters)
    return x


def yolo_conv(filters, name=None):
    def _yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = darknet_conv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = darknet_conv(x, filters, 1)
        x = darknet_conv(x, filters * 2, 3)
        x = darknet_conv(x, filters, 1)
        x = darknet_conv(x, filters * 2, 3)
        x = darknet_conv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)

    return _yolo_conv


def yolo_conv_tiny(filters, name=None):
    def _yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = darknet_conv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = darknet_conv(x, filters, 1)

        return Model(inputs, x, name=name)(x_in)

    return _yolo_conv


def yolo_output(filters, anchors, classes, name=None):
    def _yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = darknet_conv(x, filters * 2, 3)
        x = darknet_conv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)

    return _yolo_output


def darknet(name=None):
    def _darknet():
        x = inputs = Input([None, None, 3])
        x = darknet_conv(x, 32, 3)
        x = darknet_block(x, 64, 1)
        x = darknet_block(x, 128, 2)  # skip connection
        x = x_36 = darknet_block(x, 256, 8)  # skip connection
        x = x_61 = darknet_block(x, 512, 8)
        x = darknet_block(x, 1024, 4)
        return tf.keras.Model(inputs, (x_36, x_61, x), name=name)

    return _darknet(), yolo_conv, yolo_output


def darknet_tiny(name=None):
    def _darknet_tiny():
        x = inputs = Input([None, None, 3])
        x = darknet_conv(x, 16, 3)
        x = MaxPool2D(2, 2, 'same')(x)
        x = darknet_conv(x, 32, 3)
        x = MaxPool2D(2, 2, 'same')(x)
        x = darknet_conv(x, 64, 3)
        x = MaxPool2D(2, 2, 'same')(x)
        x = darknet_conv(x, 128, 3)
        x = MaxPool2D(2, 2, 'same')(x)
        x = x_8 = darknet_conv(x, 256, 3)  # skip connection
        x = MaxPool2D(2, 2, 'same')(x)
        x = darknet_conv(x, 512, 3)
        x = MaxPool2D(2, 1, 'same')(x)
        x = darknet_conv(x, 1024, 3)
        return tf.keras.Model(inputs, (x_8, x), name=name)

    return _darknet_tiny(), yolo_conv_tiny, yolo_output
