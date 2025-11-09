import tensorflow as tf


def csi_score(input, label, downvalue, upvalue):
    """Compute Critical Success Index (CSI) score

    Args:
        input: Predicted values [batch, time, depth, height, width]
        label: Ground truth values [batch, time, depth, height, width]
        downvalue: Lower threshold for classification
        upvalue: Upper threshold for classification

    Returns:
        CSI score for each batch and timestep [batch, time]
    """
    ground_truth = tf.bitwise.bitwise_and(
        tf.cast(tf.greater_equal(label, downvalue), dtype=tf.int32),
        tf.cast(tf.less_equal(label, upvalue), dtype=tf.int32)
    )
    prediction = tf.bitwise.bitwise_and(
        tf.cast(tf.greater_equal(input, downvalue), dtype=tf.int32),
        tf.cast(tf.less_equal(input, upvalue), dtype=tf.int32)
    )

    x = tf.multiply(ground_truth, prediction)  # True Positives
    y = tf.multiply(ground_truth, tf.subtract(1, prediction))  # False Negatives
    z = tf.multiply(prediction, tf.subtract(1, ground_truth))  # False Positives

    x_ = tf.cast(tf.reduce_sum(x, axis=[2, 3, 4]), dtype=tf.float32)
    y_ = tf.cast(tf.reduce_sum(y, axis=[2, 3, 4]), dtype=tf.float32)
    z_ = tf.cast(tf.reduce_sum(z, axis=[2, 3, 4]), dtype=tf.float32)

    # CSI = TP / (TP + FN + FP)
    CSI = tf.math.divide_no_nan(x_, x_ + y_ + z_)
    return CSI


def csi_score_tf(input, label, downvalue, upvalue):
    """TensorFlow 2.x compatible CSI score computation

    Args:
        input: Predicted values [batch, time, depth, height, width]
        label: Ground truth values [batch, time, depth, height, width]
        downvalue: Lower threshold for classification
        upvalue: Upper threshold for classification

    Returns:
        CSI score for each batch and timestep [batch, time]
    """
    # Create binary masks
    ground_truth = tf.cast(
        tf.logical_and(
            tf.greater_equal(label, downvalue),
            tf.less_equal(label, upvalue)
        ),
        dtype=tf.float32
    )

    prediction = tf.cast(
        tf.logical_and(
            tf.greater_equal(input, downvalue),
            tf.less_equal(input, upvalue)
        ),
        dtype=tf.float32
    )

    # True Positives, False Negatives, False Positives
    tp = ground_truth * prediction
    fn = ground_truth * (1.0 - prediction)
    fp = prediction * (1.0 - ground_truth)

    # Sum over spatial dimensions [depth, height, width]
    tp_sum = tf.reduce_sum(tp, axis=[2, 3, 4])
    fn_sum = tf.reduce_sum(fn, axis=[2, 3, 4])
    fp_sum = tf.reduce_sum(fp, axis=[2, 3, 4])

    # CSI = TP / (TP + FN + FP)
    csi = tf.math.divide_no_nan(tp_sum, tp_sum + fn_sum + fp_sum)

    return csi