import tensorflow as tf
from tensorflow.keras import layers, Model
from RNN_cells.ConvLSTM3d_cell import ConvLSTM
import numpy as np

dtype = tf.float32

conv_activation_function = tf.nn.relu

convlstm_activation_function = tf.nn.tanh
convlstm__initializer = tf.keras.initializers.GlorotUniform(seed=1997)
convlstm_feature_num = 64
convlstm_kernel_size = [3, 3]


class SpatialExtractor(layers.Layer):
    """Spatial feature extractor using 3D convolutions"""

    def __init__(self, name='spatial_extractor', **kwargs):
        super(SpatialExtractor, self).__init__(name=name, **kwargs)
        self.d1 = layers.Conv3D(32, [3, 3, 3], [1, 1, 1], padding='same',
                                activation=conv_activation_function,
                                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1997),
                                name='d1')
        self.d2 = layers.Conv3D(64, [3, 3, 3], [2, 2, 2], padding='same',
                                activation=conv_activation_function,
                                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1997),
                                name='d2')
        self.d3 = layers.Conv3D(64, [3, 3, 3], [1, 1, 1], padding='same',
                                activation=conv_activation_function,
                                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1997),
                                name='d3')
        self.d4 = layers.Conv3D(64, [3, 3, 3], [2, 1, 1], padding='same',
                                activation=conv_activation_function,
                                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1997),
                                name='d4')

    def call(self, inputs, training=None):
        """Forward pass

        Args:
            inputs: Input tensor [batch, depth, height, width, 1]
            training: Boolean for training mode

        Returns:
            Extracted spatial features [batch, depth/4, height/2, width/2, 64]
        """
        x = self.d1(inputs)  # (16, 120, 120, 32)
        x = self.d2(x)       # (8, 60, 60, 64)
        x = self.d3(x)       # (8, 60, 60, 64)
        x = self.d4(x)       # (4, 60, 60, 64)
        return x


class Decoder(layers.Layer):
    """Decoder to upsample spatial features back to original resolution"""

    def __init__(self, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.u1 = layers.Conv3DTranspose(64, [3, 3, 3], [2, 2, 2], padding='same',
                                         activation=conv_activation_function,
                                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1997),
                                         name='u1')
        self.u2 = layers.Conv3D(64, [1, 3, 3], [1, 1, 1], padding='same',
                                activation=conv_activation_function,
                                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1997),
                                name='u2')
        self.u3 = layers.Conv3DTranspose(64, [3, 3, 3], [2, 1, 1], padding='same',
                                         activation=conv_activation_function,
                                         kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1997),
                                         name='u3')
        self.out = layers.Conv3D(1, [1, 1, 1], [1, 1, 1], padding='same',
                                 activation=conv_activation_function,
                                 kernel_initializer=tf.keras.initializers.GlorotUniform(seed=1997),
                                 name='out')

    def call(self, inputs, training=None):
        """Forward pass

        Args:
            inputs: Input tensor [batch, depth/4, height/2, width/2, 64]
            training: Boolean for training mode

        Returns:
            Decoded output [batch, depth, height, width]
        """
        x = self.u1(inputs)  # (b, 8, 120, 120, 64)
        x = self.u2(x)       # (b, 8, 120, 120, 64)
        x = self.u3(x)       # (b, 16, 120, 120, 64)
        x = self.out(x)      # (b, 16, 120, 120, 1)
        x = tf.squeeze(x, axis=-1)
        return x


class ConvLSTM3DGenerator(Model):
    """3D ConvLSTM Nowcasting Model"""

    def __init__(self, input_length=8, output_length=12, name='convlstm3d_generator', **kwargs):
        super(ConvLSTM3DGenerator, self).__init__(name=name, **kwargs)
        self.input_length = input_length
        self.output_length = output_length

        # Spatial extractor
        self.spatial_extractor = SpatialExtractor()

        # Decoder
        self.decoder = Decoder()

        # ConvLSTM cells (will be initialized in build)
        self.convlstm1 = None
        self.convlstm2 = None
        self.convlstm3 = None
        self.convlstm4 = None

    def build(self, input_shape):
        """Build the model layers"""
        batch_size = input_shape[0] if input_shape[0] is not None else 1

        # Shape after spatial extractor: [4, 60, 60]
        convlstm_shape = [4, 60, 60]

        # Initialize ConvLSTM cells
        self.convlstm1 = ConvLSTM('ConvLSTM1', batch_size, convlstm_shape,
                                  convlstm_feature_num, convlstm_kernel_size,
                                  convlstm_activation_function, convlstm__initializer)
        self.convlstm2 = ConvLSTM('ConvLSTM2', batch_size, convlstm_shape,
                                  convlstm_feature_num, convlstm_kernel_size,
                                  convlstm_activation_function, convlstm__initializer)
        self.convlstm3 = ConvLSTM('ConvLSTM3', batch_size, convlstm_shape,
                                  convlstm_feature_num, convlstm_kernel_size,
                                  convlstm_activation_function, convlstm__initializer)
        self.convlstm4 = ConvLSTM('ConvLSTM4', batch_size, convlstm_shape,
                                  convlstm_feature_num, convlstm_kernel_size,
                                  convlstm_activation_function, convlstm__initializer)

        super(ConvLSTM3DGenerator, self).build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass

        Args:
            inputs: Input tensor [batch, total_time, depth, height, width, 1]
                    where total_time = input_length + output_length
            training: Boolean for training mode

        Returns:
            predictions: Predicted frames [batch, output_length, depth, height, width]
            labels: Ground truth frames [batch, output_length, depth, height, width]
            loss: Scalar loss value
        """
        batch_size = tf.shape(inputs)[0]

        # Split input into input sequence and labels
        input_sequence = inputs[:, :self.input_length]
        label = inputs[:, self.input_length:]

        # Transpose to (batch, depth, height, width, time)
        input_sequence = tf.transpose(input_sequence, [0, 4, 2, 3, 1])

        # Extract spatial features for each timestep
        u3d_input = []
        for i in range(self.input_length):
            input_frame = tf.expand_dims(input_sequence[:, :, :, :, i], axis=-1)
            spatial_features = self.spatial_extractor(input_frame, training=training)
            u3d_input.append(spatial_features)

        # Stack and transpose to [batch, time, depth, height, width, channels]
        u3d_input = tf.stack(u3d_input, axis=1)

        # Process through ConvLSTM layers
        # ConvLSTM1
        convlstm1_state = self.convlstm1.get_initial_state(batch_size)
        convlstm1_outputs = []
        for t in range(self.input_length):
            output, convlstm1_state = self.convlstm1(u3d_input[:, t], convlstm1_state, training=training)
            convlstm1_outputs.append(output)
        convlstm1_outputs = tf.stack(convlstm1_outputs, axis=1)

        # ConvLSTM2
        convlstm2_state = self.convlstm2.get_initial_state(batch_size)
        for t in range(self.input_length):
            _, convlstm2_state = self.convlstm2(convlstm1_outputs[:, t], convlstm2_state, training=training)

        # ConvLSTM3 - forecasting
        input_shape = u3d_input[:, 0].shape
        forecasting_input = tf.zeros(shape=[batch_size] + input_shape[1:].as_list(), dtype=dtype)

        convlstm3_outputs = []
        for t in range(self.output_length):
            output, convlstm2_state = self.convlstm3(forecasting_input, convlstm2_state, training=training)
            convlstm3_outputs.append(output)
        convlstm3_outputs = tf.stack(convlstm3_outputs, axis=1)

        # ConvLSTM4
        convlstm4_outputs = []
        for t in range(self.output_length):
            output, convlstm1_state = self.convlstm4(convlstm3_outputs[:, t], convlstm1_state, training=training)
            convlstm4_outputs.append(output)

        # Decode each output timestep
        predictions = []
        for t in range(self.output_length):
            decoded = self.decoder(convlstm4_outputs[t], training=training)
            predictions.append(decoded)

        # Stack predictions [batch, time, depth, height, width]
        predictions = tf.stack(predictions, axis=1)

        # Compute loss
        loss = self.compute_loss(predictions, label)

        return predictions, label, loss

    def compute_loss(self, predictions, labels):
        """Compute weighted loss

        Args:
            predictions: Predicted frames [batch, time, depth, height, width]
            labels: Ground truth frames [batch, time, depth, height, width]

        Returns:
            Scalar loss value
        """
        # Create reflectivity-based mask
        one = tf.ones_like(labels)
        MASK = tf.where(labels < 15/80, one, labels)
        MASK = tf.where((labels >= 15/80) & (labels < 35/80), one * 3, MASK)
        MASK = tf.where((labels >= 35/80) & (labels < 45/80), one * 8, MASK)
        MASK = tf.where(labels >= 45/80, one * 15, MASK)

        # Compute weighted L1 and L2 loss
        loss1 = tf.abs(predictions - labels) * MASK
        loss2 = tf.square(predictions - labels) * MASK
        loss = tf.reduce_mean(loss1) + tf.reduce_mean(loss2)

        return loss


def generator(input, input_length, output_length, is_training=False, reuse=False):
    """Functional interface for backward compatibility

    Args:
        input: Input tensor [batch, total_time, depth, height, width, 1]
        input_length: Number of input frames
        output_length: Number of output frames
        is_training: Boolean for training mode
        reuse: Ignored (for TF 1.x compatibility)

    Returns:
        List of [predictions, labels, loss]
    """
    # Create or reuse model
    if not hasattr(generator, 'model'):
        generator.model = ConvLSTM3DGenerator(input_length, output_length)

    # Forward pass
    predictions, labels, loss = generator.model(input, training=is_training)

    return [predictions, labels, loss]
