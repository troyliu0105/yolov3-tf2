import tensorflow as tf
from tensorflow.keras import Input, layers

__all__ = ['mobilenet_v1', 'mobilenet_v2', 'mobilenet_v3_large']


def _conv_unit(x, filters, kernel, stride, batchnorm=True, act='relu6'):
    channel_axis = 1 if tf.keras.backend.image_data_format() == 'channels_first' else -1

    if act == 'relu':
        act = layers.ReLU()
    elif act == 'relu6':
        act = layers.ReLU(6)
    elif act == 'hard_swish':
        def hard_swish(x):
            def hard_sigmoid(x):
                return layers.ReLU(6.)(x + 3.) * (1. / 6.)

            return layers.Multiply()([hard_sigmoid(x), x])

        act = hard_swish
    else:
        raise ValueError(f'unknown activation: {act}')

    x = layers.Conv2D(filters, kernel_size=(kernel, kernel),
                      strides=(stride, stride),
                      padding='same',
                      use_bias=not batchnorm,
                      kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
    if batchnorm:
        x = layers.BatchNormalization(axis=channel_axis, epsilon=1e-3, momentum=0.999)(x)
        x = act(x)
    return x


def _yolo_conv(filters, name=None):
    def _conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs
            x = _conv_unit(x, filters, 1, 1)
            x = layers.UpSampling2D(2)(x)
            x = layers.Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
        x = _conv_unit(x, filters, 1, 1)
        x = _conv_unit(x, filters * 2, 3, 1)
        x = _conv_unit(x, filters, 1, 1)
        x = _conv_unit(x, filters * 2, 3, 1)
        x = _conv_unit(x, filters, 1, 1)
        return tf.keras.Model(inputs, x, name=name)(x_in)

    return _conv


def _yolo_output(filters, anchors, classes, name=None):
    def _output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = _conv_unit(x, filters * 2, 1, 1)
        x = layers.Conv2D(anchors * (classes + 5),
                          kernel_size=(1, 1),
                          padding='same',
                          use_bias=True,
                          kernel_regularizer=tf.keras.regularizers.l2(0.0005))(x)
        x = layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)

    return _output


def mobilenet_v1(name=None):
    from tensorflow.keras.applications.mobilenet import MobileNet

    def _mobilenet_v1():
        net = MobileNet(include_top=False)
        layer_names = ['conv_pw_5_relu', 'conv_pw_11_relu', 'conv_pw_13_relu']
        output = [net.get_layer(n).output for n in layer_names]
        backbone = tf.keras.Model(net.input, output, name=name)
        return backbone

    return _mobilenet_v1(), _yolo_conv, _yolo_output


def mobilenet_v2(name=None):
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

    def _mobilenet_2():
        net = MobileNetV2(include_top=False)
        layer_names = ['block_6_expand_relu', 'block_13_expand_relu', 'block_16_expand_relu']
        output = [net.get_layer(n).output for n in layer_names]
        backbone = tf.keras.Model(net.input, output, name=name)
        return backbone

    return _mobilenet_2(), _yolo_conv, _yolo_output


def mobilenet_v3_large(name=None):
    from tensorflow.keras.applications import MobileNetV3Large

    def _mobilenet_3():
        net = MobileNetV3Large(include_top=False)
        layer_names = ['multiply_1', 'multiply_10', 'multiply_19']
        output = [net.get_layer(n).output for n in layer_names]
        backbone = tf.keras.Model(net.input, output, name=name)
        return backbone

    def _yolo_conv_m3_large(filters, name=None):
        def _conv(x_in):
            if isinstance(x_in, tuple):
                inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
                x, x_skip = inputs
                x = _conv_unit(x, filters, 1, 1, act='hard_swish')
                x = layers.UpSampling2D(2)(x)
                x = layers.Concatenate()([x, x_skip])
            else:
                x = inputs = Input(x_in.shape[1:])
            x = _conv_unit(x, filters, 1, 1, act='hard_swish')
            x = _conv_unit(x, filters * 2, 3, 1, act='hard_swish')
            x = _conv_unit(x, filters, 1, 1, act='hard_swish')
            x = _conv_unit(x, filters * 2, 3, 1, act='hard_swish')
            x = _conv_unit(x, filters, 1, 1, act='hard_swish')
            return tf.keras.Model(inputs, x, name=name)(x_in)

        return _conv

    def _yolo_output_m3_large(filters, anchors, classes, name=None):
        def _output(x_in):
            x = inputs = Input(x_in.shape[1:])
            x = _conv_unit(x, filters * 2, 3, 1, act='hard_swish')
            x = _conv_unit(x, anchors * (classes + 5), 1, 1, batchnorm=False)
            x = layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)
            return tf.keras.Model(inputs, x, name=name)(x_in)

        return _output

    return _mobilenet_3(), _yolo_conv_m3_large, _yolo_output_m3_large
