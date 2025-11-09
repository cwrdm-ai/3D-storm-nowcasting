import tensorflow as tf
from tensorflow.keras import layers


class ConvLSTM(layers.Layer):
    """3D Convolutional LSTM Cell for TensorFlow 2.x"""

    def __init__(self, name, batch_size, shape, feature_num, kernel_size,
                 activation_function=tf.nn.tanh,
                 initializer=None,
                 regularizer=None,
                 state_is_tuple=True,
                 **kwargs):
        super(ConvLSTM, self).__init__(name=name, **kwargs)
        self.batch_size = batch_size
        self.shape = shape
        self.feature_num = feature_num
        self.kernel_size = kernel_size
        self.activation_function = activation_function
        self.initializer = initializer if initializer is not None else tf.keras.initializers.GlorotUniform(seed=1997)
        self.regularizer = regularizer
        self.state_is_tuple = state_is_tuple
        self.c_size = tf.TensorShape(self.shape + [self.feature_num])
        self.h_size = tf.TensorShape(self.shape + [self.feature_num])

        # Define the convolution layer
        self.conv_layer = None

    def build(self, input_shape):
        """Build the layer by creating the convolutional layer"""
        # Create the 3D convolution layer for gate computations
        output_channel = self.feature_num * 4
        self.conv_layer = layers.Conv3D(
            filters=output_channel,
            kernel_size=[2, 3, 3],
            strides=[1, 1, 1],
            padding='same',
            activation=None,
            kernel_initializer=self.initializer,
            name='concat_conv'
        )
        super(ConvLSTM, self).build(input_shape)

    @property
    def state_size(self):
        """Return the state size"""
        return (self.c_size, self.h_size) if self.state_is_tuple else self.c_size

    @property
    def output_size(self):
        """Return the output size"""
        return self.h_size

    def get_initial_state(self, batch_size=None):
        """Get initial zero state"""
        if batch_size is None:
            batch_size = self.batch_size
        c = tf.zeros(shape=[batch_size] + self.shape + [self.feature_num], dtype=tf.float32)
        h = tf.zeros(shape=[batch_size] + self.shape + [self.feature_num], dtype=tf.float32)
        return (c, h) if self.state_is_tuple else c

    def call(self, inputs, states, training=None):
        """Forward pass of the ConvLSTM cell

        Args:
            inputs: Input tensor [batch, depth, height, width, channels]
            states: Tuple of (c, h) state tensors
            training: Boolean for training mode

        Returns:
            output: Output tensor [batch, depth, height, width, feature_num]
            new_state: Tuple of new (c, h) states
        """
        if self.state_is_tuple:
            c, h = states
        else:
            c, h = tf.split(value=states, num_or_size_splits=2, axis=-1)

        # Concatenate input and previous hidden state
        concat_input = tf.concat([inputs, h], axis=-1)

        # Apply convolution to get all gates at once
        concat_output = self.conv_layer(concat_input)

        # Split into individual gates
        i, j, f, o = tf.split(value=concat_output, num_or_size_splits=4, axis=-1)

        # Apply gate activations
        i = tf.sigmoid(i)  # Input gate
        f = tf.sigmoid(f)  # Forget gate
        o = tf.sigmoid(o)  # Output gate

        # Update cell state and hidden state
        new_c = tf.multiply(c, f) + tf.multiply(i, tf.tanh(j))
        new_h = tf.multiply(self.activation_function(new_c), o)

        # Return output and new state
        if self.state_is_tuple:
            new_state = (new_c, new_h)
        else:
            new_state = tf.concat([new_c, new_h], axis=-1)

        return new_h, new_state

    def get_config(self):
        """Return the config for serialization"""
        config = super(ConvLSTM, self).get_config()
        config.update({
            'batch_size': self.batch_size,
            'shape': self.shape,
            'feature_num': self.feature_num,
            'kernel_size': self.kernel_size,
            'state_is_tuple': self.state_is_tuple,
        })
        return config
