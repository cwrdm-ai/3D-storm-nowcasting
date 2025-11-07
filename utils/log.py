import logging
import tensorflow as tf


def setting_log(log_save_path):
    logger = logging.getLogger("log")
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_save_path, mode='a')
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def imgs_summaries(imgs_seq, name_scope, max_outputs, step=None):
    # TF 2.0 uses eager execution, so summaries are written differently
    # This function is kept for backward compatibility but may need a summary writer
    for i in range(imgs_seq.shape[0]):
        tf.summary.image(name_scope + '/img' + str(i + 1), imgs_seq[i:i+1], max_outputs=max_outputs, step=step)


def variable_summaries(var, step=None):
    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(var.name + '_mean', mean, step=step)
    tf.summary.scalar(var.name + '_stddev', stddev, step=step)
    tf.summary.scalar(var.name + '_max', tf.reduce_max(var), step=step)
    tf.summary.scalar(var.name + '_min', tf.reduce_min(var), step=step)