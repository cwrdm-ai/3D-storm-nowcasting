__author__ = 'Nengli Sun'

import os
import math
import configparser
import numpy as np
import shutil
import tensorflow as tf
try:
    from tfdeterminism import patch
    patch()
except ImportError:
    print("tfdeterminism not available for TF 2.0, using tf.random.set_seed for reproducibility")
import random
from datetime import datetime
from utils import log,  evaluation_functions
from model import ConvLSTM3d_model
import warnings
warnings.filterwarnings("ignore")

temporal_weights = np.arange(1,13)

def delete_file(path):
    jpg_files = os.listdir(path)
    for f in jpg_files:
        file_path = os.path.join(path, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

conf = configparser.ConfigParser()
conf.read("./configs.ini")

training_dataset_path = 'D:/gridrad/train8/'
validation_dataset_path = 'D:/gridrad/val8/'
test_dataset_path = 'D:/gridrad/test8/'

train_data_num = len(os.listdir('D:/gridrad/train7/'))
valid_data_num = len(os.listdir('D:/gridrad/val7/'))
test_data_num = len(os.listdir('D:/gridrad/test7/'))

running_log_path = conf.get('logfile_paths', 'running_log_path')

model_saving_path = conf.get('saver_configs', 'model_saving_path')
model_name = conf.get('saver_configs', 'model_name')
model_keep_num = int(conf.get('saver_configs', 'model_keep_num'))

generator_ini_learning_rate = float(conf.get('training_parameters', 'generator_ini_learning_rate'))
img_height = int(conf.get('training_parameters', 'img_height'))
img_width = int(conf.get('training_parameters', 'img_width'))
img_size = (img_height,img_width)
input_length = int(conf.get('training_parameters', 'input_length'))
output_length = int(conf.get('training_parameters', 'output_length'))
batch_size = int(conf.get('training_parameters', 'batch_size'))
val_batch_size = int(conf.get('training_parameters', 'val_batch_size'))
test_batch_size = int(conf.get('training_parameters', 'test_batch_size'))
training_steps = int(conf.get('training_parameters', 'training_steps'))
imgs_sum_steps = int(conf.get('training_parameters', 'imgs_sum_steps'))

patience = 20

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

def seed_tensorflow(seed=1997):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


img_shape = (20,120,120,16)
def _parse_function(example_proto):
    features = {"img_raw": tf.io.FixedLenFeature((), tf.string),
              "name": tf.io.FixedLenFeature((), tf.string)}
    parsed_features = tf.io.parse_single_example(example_proto, features)
    img_str = parsed_features["img_raw"]
    name_str = parsed_features["name"]
    img = tf.io.decode_raw(img_str,tf.float16)
    img = tf.reshape(img,img_shape)
    img = tf.cast(img, tf.float32)/80
    name = tf.io.decode_raw(name_str,tf.int64)
    return img, name

def get_dataset_from_tfrecords(tfrecords_pattern,is_train_dataset,batch_size=1, threads=18, shuffle_buffer_size=1, cycle_length=1):

    if is_train_dataset:
        files = tf.data.Dataset.list_files(tfrecords_pattern, shuffle=True)
    else:
        files = tf.data.Dataset.list_files(tfrecords_pattern)

    dataset = files.interleave(map_func=tf.data.TFRecordDataset, cycle_length=cycle_length)
    dataset = dataset.map(_parse_function, num_parallel_calls=threads)

    if is_train_dataset:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size).batch(batch_size, drop_remainder=True).repeat(100).prefetch(buffer_size=batch_size)
    else:
        dataset = dataset.batch(batch_size, drop_remainder=False).repeat(100)
    return dataset


@tf.function
def train_step(model_input, optimizer):
    with tf.GradientTape() as tape:
        _, _, loss = ConvLSTM3d_model.generator(model_input, input_length, output_length, is_training=True)

    # Get all trainable variables
    trainable_vars = tape.watched_variables()
    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))
    return loss

@tf.function
def val_step(model_input):
    prediction, label, valloss = ConvLSTM3d_model.generator(model_input, input_length, output_length, is_training=False)
    CSI45 = evaluation_functions.csi_score(prediction, label, downvalue=45 / 80, upvalue=1)
    CSI = evaluation_functions.csi_score(prediction, label, downvalue=35 / 80, upvalue=1)
    return prediction, label, valloss, CSI45, CSI

def main(argv=None):
    seed_tensorflow(seed=1997)
    update_generator = True
    if update_generator:
        delete_file(model_saving_path)
    if (os.path.isfile(running_log_path)) and update_generator:
        os.remove(running_log_path)

    logger = log.setting_log(running_log_path)

    tra_dataset = get_dataset_from_tfrecords(training_dataset_path + '*.tfrecords', is_train_dataset = True, batch_size=batch_size, shuffle_buffer_size=500)
    val_dataset = get_dataset_from_tfrecords(validation_dataset_path + '*.tfrecords', is_train_dataset = False, batch_size=val_batch_size, shuffle_buffer_size=500)
    test_dataset = get_dataset_from_tfrecords(test_dataset_path + '*.tfrecords', is_train_dataset = False, batch_size=test_batch_size, shuffle_buffer_size=500)

    if update_generator:
        eval_dataset = val_dataset
    else:
        eval_dataset = test_dataset

    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # Create checkpoint
    checkpoint = tf.train.Checkpoint(optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, model_saving_path, max_to_keep=model_keep_num
    )

    validation_sum_steps = math.floor(train_data_num / batch_size)

    logger.info("------------------- Time: " + datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + " loading pretrain models -------------------")
    pre_tra_steps = 0

    # Restore checkpoint if exists
    latest_checkpoint = checkpoint_manager.latest_checkpoint
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)
        # Extract step number from checkpoint name
        try:
            pre_tra_steps = int(latest_checkpoint.split('ckpt-')[1])
        except:
            pre_tra_steps = 0
        logger.info("Found pretrain models, reading..... " + latest_checkpoint + "   pretrain steps: " + str(pre_tra_steps))
    else:
        logger.info("Not found pretrain models, restart training..... ")

    logger.info("------------------- Time: " + datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + " training parameters -------------------")
    logger.info("train_data_num: " + str(train_data_num))
    logger.info("valid_data_num: " + str(valid_data_num))
    logger.info("test_data_num: " + str(test_data_num))
    logger.info("img_height: " + str(img_height))
    logger.info("img_width: " + str(img_width))
    logger.info("input_length: " + str(input_length))
    logger.info("output_length: " + str(output_length))
    logger.info("training_steps: " + str(training_steps))
    logger.info("imgs_sum_steps: " + str(imgs_sum_steps))
    logger.info("validation_sum_steps: " + str(validation_sum_steps))
    logger.info("batch_size: " + str(batch_size))
    logger.info("val_batch_size: " + str(val_batch_size))
    logger.info("test_batch_size: " + str(test_batch_size))
    logger.info("optimizer: " + str(tf.keras.optimizers.Adam))
    logger.info("generator_ini_learning_rate: " + str(generator_ini_learning_rate))
    logger.info("------------------- Time: " + datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + " model parameters num -------------------")
    logger.info("------------------- Time: " + datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + " start training -------------------")

    count = 0
    best = -1
    model_step = pre_tra_steps

    if update_generator:
        # Training mode
        tra_iter = iter(tra_dataset)
        val_iter = iter(val_dataset)

        for step in range(training_steps):
            # Get training batch
            try:
                train_input, _ = next(tra_iter)
            except StopIteration:
                tra_iter = iter(tra_dataset)
                train_input, _ = next(tra_iter)

            # Training step
            if update_generator and ((step + 1) % 1 == 0):
                gen_train_loss = train_step(train_input, optimizer)

            if (step + 1 + pre_tra_steps) % imgs_sum_steps == 0:
                logger.info(datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + " training steps: %d, training loss: %g" % (step + 1 + pre_tra_steps, gen_train_loss.numpy()))

            if (step + 1 + pre_tra_steps) % validation_sum_steps == 0:
                val_loss = 0
                csi = []

                batch_val_step = math.ceil(valid_data_num / val_batch_size) - 1

                for i in range(batch_val_step):
                    try:
                        val_input, _ = next(val_iter)
                    except StopIteration:
                        val_iter = iter(val_dataset)
                        val_input, _ = next(val_iter)

                    _, _, val_loss_, _, csi_ = val_step(val_input)
                    val_loss += val_loss_.numpy() * val_batch_size
                    csi.append(csi_.numpy())

                try:
                    val_input, _ = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_dataset)
                    val_input, _ = next(val_iter)

                _, _, val_loss_, _, csi_ = val_step(val_input)
                val_loss += val_loss_.numpy() * (valid_data_num - batch_val_step * val_batch_size)
                csi.append(csi_.numpy())

                csi = np.concatenate(csi, axis=0)
                csi = np.nanmean(csi, axis=0)

                val_loss = val_loss / valid_data_num

                wcsi = np.sum(temporal_weights * csi) / np.sum(temporal_weights)
                csi = list(csi)
                logger.info(csi)
                csi = np.nanmean(csi)
                logger.info("------------------- Training steps: %d, validation loss: %g, CSI: %g, wCSI: %g" % (step + 1 + pre_tra_steps, val_loss, csi, wcsi) + " -------------------")

                if wcsi <= best:
                    count += 1
                    logger.info('eval wcsi is not improved for {} epoch'.format(count) + '\n')
                else:
                    count = 0
                    logger.info('eval valpred_loss is improved from {:.5f} to {:.5f}, saving model'.format(best, wcsi) + '\n')
                    checkpoint_manager.save(checkpoint_number=step + 1 + pre_tra_steps)
                    best = wcsi

                if count == patience:
                    print('early stopping reached, best score is {:5f}'.format(best))
                    logger.info('early stopping reached, best score is {:5f}'.format(best) + '\n')
                    model_step = step + 1 + pre_tra_steps
                    break

    else:
        # Testing mode
        test_loss = 0
        csi45 = []
        csi = []

        batch_test_step = math.ceil(test_data_num / test_batch_size) - 1
        test_iter = iter(test_dataset)

        for i in range(batch_test_step):
            try:
                test_input, _ = next(test_iter)
            except StopIteration:
                test_iter = iter(test_dataset)
                test_input, _ = next(test_iter)

            prediction_imgs_, label_imgs_, test_loss_, csi45_, csi_ = val_step(test_input)

            test_loss += test_loss_.numpy() * test_batch_size
            csi45.append(csi45_.numpy())
            csi.append(csi_.numpy())

        try:
            test_input, _ = next(test_iter)
        except StopIteration:
            test_iter = iter(test_dataset)
            test_input, _ = next(test_iter)

        prediction_imgs_, label_imgs_, test_loss_, csi45_, csi_ = val_step(test_input)
        test_loss += test_loss_.numpy() * (test_data_num - batch_test_step * test_batch_size)
        csi45.append(csi45_.numpy())
        csi.append(csi_.numpy())

        csi45 = np.concatenate(csi45, axis=0)
        csi = np.concatenate(csi, axis=0)

        csi45 = np.nanmean(csi45, axis=0)
        csi = np.nanmean(csi, axis=0)

        test_loss = test_loss / test_data_num

        wcsi45 = np.sum(temporal_weights * csi45) / np.sum(temporal_weights)
        wcsi = np.sum(temporal_weights * csi) / np.sum(temporal_weights)

        csi45 = list(csi45)
        csi = list(csi)

        logger.info(csi45)
        logger.info('--------------------------------------------------------------------------')
        logger.info(csi)

        csi45 = np.nanmean(csi45)
        csi = np.nanmean(csi)

        logger.info(
            "------------------- Training steps: %d, test loss: %g, CSI45: %g, CSI: %g, wCSI45: %g, wCSI: %g" % (
            model_step, test_loss, csi45, csi, wcsi45, wcsi) + " -------------------")

        logger.info("------------------- Time: " + datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + " end testing -------------------" + '\n')



if __name__ == '__main__':
    main()