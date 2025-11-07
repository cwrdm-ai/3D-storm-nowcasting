import tensorflow as tf
from collections import namedtuple

# Define LSTMStateTuple as a namedtuple for TF 2.0
LSTMStateTuple = namedtuple('LSTMStateTuple', ['c', 'h'])


class ConvLSTM(tf.keras.layers.AbstractRNNCell):
    def __init__(self, scope, batch_size, shape, feature_num, kernel_size, activation_function=tf.tanh, initializer=None, regularizer=None, collection=None, state_is_tuple=True, dtype=tf.float32, **kwargs):
        super(ConvLSTM, self).__init__(**kwargs)
        self.scope = scope
        self.batch_size = batch_size
        self.shape = shape
        self.feature_num = feature_num
        self.kernel_size = kernel_size
        self.activation_function = activation_function
        self.initializer = initializer if initializer is not None else tf.keras.initializers.GlorotUniform(seed=1997)
        self.regularizer = regularizer
        self.collection = collection
        self.state_is_tuple = state_is_tuple
        self.dtype_ = dtype
        self.c_size = tf.TensorShape(self.shape + [self.feature_num])
        self.h_size = tf.TensorShape(self.shape + [self.feature_num])

        # Create the conv3d layer
        self.conv_layer = None

    @property
    def state_size(self):
        return LSTMStateTuple(self.c_size, self.h_size) if self.state_is_tuple else 2 * self.feature_num

    @property
    def output_size(self):
        return self.h_size

    # zero state
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if batch_size is None:
            batch_size = self.batch_size
        if dtype is None:
            dtype = self.dtype_
        c = tf.zeros(shape=[batch_size] + self.shape + [self.feature_num], dtype=dtype)
        h = tf.zeros(shape=[batch_size] + self.shape + [self.feature_num], dtype=dtype)
        state = LSTMStateTuple(c, h)
        return state

    def zero_state(self, batch_size=None, dtype=None):
        return self.get_initial_state(batch_size=batch_size, dtype=dtype)

    def build(self, input_shape):
        # Build the conv3d layer
        output_channel = self.feature_num * 4
        self.conv_layer = tf.keras.layers.Conv3D(
            output_channel,
            kernel_size=[2, 3, 3],
            strides=[1, 1, 1],
            padding='same',
            activation=None,
            kernel_initializer=self.initializer,
            name='concat'
        )
        self.built = True

    def call(self, inputs, states):
        if self.state_is_tuple:
            c, h = states
        else:
            c, h = tf.split(value=states, num_or_size_splits=2, axis=-1)

        inputs = tf.concat([inputs, h], axis=-1)
        concat = self.conv_layer(inputs)
        i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=-1)
        i = tf.sigmoid(i)
        f = tf.sigmoid(f)
        o = tf.sigmoid(o)
        new_c = tf.multiply(c, f) + tf.multiply(i, tf.tanh(j))
        new_h = tf.multiply(self.activation_function(new_c), o)

        if self.state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = tf.concat([new_c, new_h], axis=-1)

        return new_h, new_state
