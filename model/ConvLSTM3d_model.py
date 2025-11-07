import tensorflow as tf
from RNN_cells.ConvLSTM3d_cell import ConvLSTM
import numpy as np

dtype = tf.float32

conv_activation_function = tf.nn.relu

convlstm_activation_function = tf.tanh
convlstm__initializer = tf.keras.initializers.GlorotUniform(seed=1997)
convlstm_feature_num = 64
convlstm_kernel_size = [3, 3]

# Global layer cache for reusing layers
_layer_cache = {}

def conv3d(scope, inputs, output_channels, kernel_shape, strides, activation_function=None, padding='same', is_training=False, initializer=None):
    if initializer is None:
        initializer = tf.keras.initializers.GlorotUniform(seed=1997)

    # Use layer caching for weight reuse
    if scope not in _layer_cache:
        _layer_cache[scope] = tf.keras.layers.Conv3D(
            output_channels, kernel_shape, strides, padding=padding,
            activation=activation_function, kernel_initializer=initializer,
            name=scope
        )
    return _layer_cache[scope](inputs)

def up_conv3d(scope, inputs, output_channels, kernel_shape, strides, activation_function=None, padding='same', is_training=False, initializer=None):
    if initializer is None:
        initializer = tf.keras.initializers.GlorotUniform(seed=1997)

    # Use layer caching for weight reuse
    if scope not in _layer_cache:
        _layer_cache[scope] = tf.keras.layers.Conv3DTranspose(
            output_channels, kernel_shape, strides, padding=padding,
            activation=activation_function, kernel_initializer=initializer,
            name=scope
        )
    return _layer_cache[scope](inputs)

def generator(input, input_length, output_length, is_training=False):
    batch_size = tf.shape(input)[0]

    channel = 16

    label = input[:, input_length:]

    one = tf.ones_like(label)
    MASK = tf.where(label < 15/80, one, label)
    MASK = tf.where((label >= 15 / 80)&(label < 35 / 80), one*3, MASK)
    MASK = tf.where((label >= 35 / 80)&(label < 45 / 80), one*8, MASK)
    MASK = tf.where(label >= 45 / 80, one*15, MASK)

    input = tf.transpose(input[:, :input_length], [0, 4, 2, 3, 1])
    u3d_input = []
    for i in range(input_length):
        input_ = tf.expand_dims(input[:, :, :, :, i], axis=-1)
        spatial_features = spatial_extractor(input_, is_training)
        u3d_input.append(spatial_features)
    u3d_input = tf.transpose(u3d_input, [1, 0, 2, 3, 4, 5])

    input_shape = u3d_input.get_shape().as_list()[2:5]

    ConvLSTM1 = ConvLSTM('ConvLSTM1', batch_size, input_shape, convlstm_feature_num, convlstm_kernel_size, convlstm_activation_function, convlstm__initializer)
    ConvLSTM1_rnn = tf.keras.layers.RNN(ConvLSTM1, return_sequences=True, return_state=True)
    ConvLSTM1_outputs, ConvLSTM1_state_c, ConvLSTM1_state_h = ConvLSTM1_rnn(u3d_input)
    ConvLSTM1_state = (ConvLSTM1_state_c, ConvLSTM1_state_h)

    ConvLSTM2 = ConvLSTM('ConvLSTM2', batch_size, input_shape, convlstm_feature_num, convlstm_kernel_size, convlstm_activation_function, convlstm__initializer)
    ConvLSTM2_rnn = tf.keras.layers.RNN(ConvLSTM2, return_sequences=False, return_state=True)
    _, ConvLSTM2_state_c, ConvLSTM2_state_h = ConvLSTM2_rnn(ConvLSTM1_outputs)
    ConvLSTM2_state = (ConvLSTM2_state_c, ConvLSTM2_state_h)

    ConvLSTM3 = ConvLSTM('ConvLSTM3', batch_size, input_shape, convlstm_feature_num, convlstm_kernel_size, convlstm_activation_function, convlstm__initializer)
    forecasting_input = tf.zeros(shape=[batch_size, output_length] + input_shape + [1], dtype=dtype)
    ConvLSTM3_rnn = tf.keras.layers.RNN(ConvLSTM3, return_sequences=True, return_state=False)
    ConvLSTM3_outputs = ConvLSTM3_rnn(forecasting_input, initial_state=ConvLSTM2_state)

    ConvLSTM4 = ConvLSTM('ConvLSTM4', batch_size, input_shape, convlstm_feature_num, convlstm_kernel_size, convlstm_activation_function, convlstm__initializer)
    ConvLSTM4_rnn = tf.keras.layers.RNN(ConvLSTM4, return_sequences=True, return_state=False)
    ConvLSTM4_outputs = ConvLSTM4_rnn(ConvLSTM3_outputs, initial_state=ConvLSTM1_state)

    out = []
    for i in range(output_length):
        out_ = decoder(ConvLSTM4_outputs[:,i], is_training)
        out.append(out_)

    out = tf.stack(out)
    out = tf.transpose(out, [1, 0, 3, 4, 2])
    loss1 = tf.abs(out - label)*MASK
    loss2 = tf.square(out - label)*MASK
    loss = tf.reduce_mean(loss1) + tf.reduce_mean(loss2)

    return [out, label, loss]


def spatial_extractor(input, is_training):
    batch_size = tf.shape(input)[0]
    altitude = input.get_shape().as_list()[1]
    input_shape = input.get_shape().as_list()[2:4]
    d1 = conv3d('spatial_extractor/d1', input, 32, [3, 3, 3], [1, 1, 1], activation_function=conv_activation_function, is_training=is_training) #(16,120,120,32)
    d2 = conv3d('spatial_extractor/d2', d1, 64, [3, 3, 3], [2, 2, 2], activation_function=conv_activation_function, is_training=is_training) #(8,60,60,64)
    d3 = conv3d('spatial_extractor/d3', d2, 64, [3, 3, 3], [1, 1, 1], activation_function=conv_activation_function, is_training=is_training) #(8,60,60,64)
    d4 = conv3d('spatial_extractor/d4', d3, 64, [3, 3, 3], [2, 1, 1], activation_function=conv_activation_function, is_training=is_training) #(4,60,60,64)
    return d4

def decoder(input, is_training):
    #(b,4,60,60,64)
    u1 = up_conv3d('decoder/u1', input, 64, [3,3, 3], [2, 2, 2], activation_function=conv_activation_function, is_training=is_training) #(b,8,120,120,64)
    u2 = conv3d('decoder/u2', u1, 64, [1, 3, 3], [1, 1, 1], activation_function=conv_activation_function, is_training=is_training) #(b,8,120,120,64)
    u3 = up_conv3d('decoder/u3', u2, 64, [3,3, 3], [2, 1, 1], activation_function=conv_activation_function, is_training=is_training) #(b,16,120,120,64)
    out = conv3d('decoder/out', u3, 1, [1, 1, 1], [1, 1, 1], activation_function=conv_activation_function, is_training=is_training) #(b,16,120,120,1)
    out = tf.squeeze(out, axis=-1)
    return out