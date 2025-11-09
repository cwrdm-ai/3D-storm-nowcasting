__author__ = 'Nengli Sun'

import os
import math
import configparser
import numpy as np
import shutil
import tensorflow as tf
import random
from datetime import datetime
from utils import log, evaluation_functions
from model import ConvLSTM3d_model
import warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
def seed_tensorflow(seed=1997):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

temporal_weights = np.arange(1, 13)


def delete_file(path):
    """Delete all files in a directory"""
    if not os.path.exists(path):
        return
    jpg_files = os.listdir(path)
    for f in jpg_files:
        file_path = os.path.join(path, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def mkdir(path):
    """Create directory if it doesn't exist"""
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


# Load configuration
conf = configparser.ConfigParser()
conf.read("./configs_kochi.ini")

training_dataset_path = conf.get('dataset_paths', 'training_dataset_path')
validation_dataset_path = conf.get('dataset_paths', 'validation_dataset_path')
test_dataset_path = conf.get('dataset_paths', 'test_dataset_path')

running_log_path = conf.get('logfile_paths', 'running_log_path')

model_saving_path = conf.get('saver_configs', 'model_saving_path')
model_name = conf.get('saver_configs', 'model_name')
model_keep_num = int(conf.get('saver_configs', 'model_keep_num'))

generator_ini_learning_rate = float(conf.get('training_parameters', 'generator_ini_learning_rate'))
img_height = int(conf.get('training_parameters', 'img_height'))
img_width = int(conf.get('training_parameters', 'img_width'))
img_depth = int(conf.get('training_parameters', 'img_depth'))
input_length = int(conf.get('training_parameters', 'input_length'))
output_length = int(conf.get('training_parameters', 'output_length'))
batch_size = int(conf.get('training_parameters', 'batch_size'))
val_batch_size = int(conf.get('training_parameters', 'val_batch_size'))
test_batch_size = int(conf.get('training_parameters', 'test_batch_size'))
training_steps = int(conf.get('training_parameters', 'training_steps'))
imgs_sum_steps = int(conf.get('training_parameters', 'imgs_sum_steps'))
validation_sum_steps = int(conf.get('training_parameters', 'validation_sum_steps'))

patience = 20

# Create directories
mkdir(model_saving_path)
mkdir(os.path.dirname(running_log_path))

img_shape = (input_length + output_length, img_depth, img_height, img_width, 1)


def _parse_function(example_proto):
    """Parse TFRecord example"""
    features = {
        "img_raw": tf.io.FixedLenFeature((), tf.string),
        "name": tf.io.FixedLenFeature((), tf.string)
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)
    img_str = parsed_features["img_raw"]
    name_str = parsed_features["name"]

    img = tf.io.decode_raw(img_str, tf.float16)
    img = tf.reshape(img, img_shape)
    img = tf.cast(img, tf.float32) / 80.0

    name = tf.io.decode_raw(name_str, tf.int64)
    return img, name


def get_dataset_from_tfrecords(tfrecords_pattern, is_train_dataset, batch_size=1,
                                threads=18, shuffle_buffer_size=1, cycle_length=1):
    """Create dataset from TFRecord files"""
    if is_train_dataset:
        files = tf.data.Dataset.list_files(tfrecords_pattern, shuffle=True)
    else:
        files = tf.data.Dataset.list_files(tfrecords_pattern, shuffle=False)

    dataset = files.interleave(
        map_func=tf.data.TFRecordDataset,
        cycle_length=cycle_length,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.map(_parse_function, num_parallel_calls=threads)

    if is_train_dataset:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size) \
                         .batch(batch_size, drop_remainder=True) \
                         .repeat() \
                         .prefetch(buffer_size=tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size, drop_remainder=False) \
                         .repeat()
    return dataset


class ConvLSTM3DTrainer:
    """Trainer class for ConvLSTM3D model"""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        # Create model
        self.model = ConvLSTM3d_model.ConvLSTM3DGenerator(
            input_length=input_length,
            output_length=output_length
        )

        # Create optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=generator_ini_learning_rate)

        # Create checkpoint manager
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            model=self.model
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=model_saving_path,
            max_to_keep=model_keep_num
        )

        # Load checkpoint if exists
        self.pre_tra_steps = 0
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            # Extract step number from checkpoint path
            try:
                self.pre_tra_steps = int(self.checkpoint_manager.latest_checkpoint.split('-')[-1])
            except:
                self.pre_tra_steps = 0
            self.logger.info(f"Restored from {self.checkpoint_manager.latest_checkpoint}, step {self.pre_tra_steps}")
        else:
            self.logger.info("Initializing from scratch.")

    @tf.function
    def train_step(self, inputs):
        """Single training step"""
        with tf.GradientTape() as tape:
            predictions, labels, loss = self.model(inputs, training=True)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    @tf.function
    def val_step(self, inputs):
        """Single validation step"""
        predictions, labels, loss = self.model(inputs, training=False)

        # Compute CSI
        csi = evaluation_functions.csi_score_tf(predictions, labels,
                                                downvalue=35/80, upvalue=1)
        csi45 = evaluation_functions.csi_score_tf(predictions, labels,
                                                  downvalue=45/80, upvalue=1)

        return loss, csi, csi45, predictions, labels

    def train(self, train_dataset, val_dataset, training_steps, validation_steps):
        """Training loop"""
        best_wcsi = -1
        count = 0

        train_iter = iter(train_dataset)

        for step in range(training_steps):
            # Training step
            inputs, _ = next(train_iter)
            loss = self.train_step(inputs)

            # Log training loss
            if (step + 1 + self.pre_tra_steps) % imgs_sum_steps == 0:
                self.logger.info(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                    f"training steps: {step + 1 + self.pre_tra_steps}, "
                    f"training loss: {loss.numpy():.6f}"
                )

            # Validation
            if (step + 1 + self.pre_tra_steps) % validation_steps == 0:
                val_loss, wcsi = self.validate(val_dataset)

                self.logger.info(
                    f"------------------- Training steps: {step + 1 + self.pre_tra_steps}, "
                    f"validation loss: {val_loss:.6f}, wCSI: {wcsi:.6f} -------------------"
                )

                # Check for improvement
                if wcsi <= best_wcsi:
                    count += 1
                    self.logger.info(f'eval wcsi is not improved for {count} epoch\n')
                else:
                    count = 0
                    self.logger.info(
                        f'eval wcsi is improved from {best_wcsi:.5f} to {wcsi:.5f}, '
                        f'saving model\n'
                    )
                    self.checkpoint_manager.save(checkpoint_number=step + 1 + self.pre_tra_steps)
                    best_wcsi = wcsi

                # Early stopping
                if count >= patience:
                    self.logger.info(f'early stopping reached, best score is {best_wcsi:.5f}\n')
                    break

        return step + 1 + self.pre_tra_steps

    def validate(self, val_dataset):
        """Run validation"""
        val_loss_total = 0
        csi_list = []
        num_samples = 0

        # Count validation samples
        try:
            # Try to get number of samples from directory
            val_files = tf.io.gfile.glob(validation_dataset_path + '*.tfrecords')
            num_val_files = len(val_files)
            batch_val_steps = max(1, num_val_files // val_batch_size)
        except:
            batch_val_steps = 100  # Default fallback

        val_iter = iter(val_dataset)

        for i in range(batch_val_steps):
            try:
                inputs, _ = next(val_iter)
                loss, csi, _, _, _ = self.val_step(inputs)

                current_batch_size = tf.shape(inputs)[0].numpy()
                val_loss_total += loss.numpy() * current_batch_size
                csi_list.append(csi.numpy())
                num_samples += current_batch_size
            except tf.errors.OutOfRangeError:
                break

        if num_samples == 0:
            return float('inf'), -1

        # Compute average metrics
        val_loss = val_loss_total / num_samples
        csi_array = np.concatenate(csi_list, axis=0)
        csi_mean = np.nanmean(csi_array, axis=0)
        wcsi = np.sum(temporal_weights * csi_mean) / np.sum(temporal_weights)

        self.logger.info(f"CSI per timestep: {list(csi_mean)}")
        self.logger.info(f"Mean CSI: {np.nanmean(csi_mean):.6f}")

        return val_loss, wcsi

    def test(self, test_dataset):
        """Run testing"""
        test_loss_total = 0
        csi_list = []
        csi45_list = []
        num_samples = 0

        # Count test samples
        try:
            test_files = tf.io.gfile.glob(test_dataset_path + '*.tfrecords')
            num_test_files = len(test_files)
            batch_test_steps = max(1, num_test_files // test_batch_size)
        except:
            batch_test_steps = 100  # Default fallback

        test_iter = iter(test_dataset)

        for i in range(batch_test_steps):
            try:
                inputs, _ = next(test_iter)
                loss, csi, csi45, predictions, labels = self.val_step(inputs)

                current_batch_size = tf.shape(inputs)[0].numpy()
                test_loss_total += loss.numpy() * current_batch_size
                csi_list.append(csi.numpy())
                csi45_list.append(csi45.numpy())
                num_samples += current_batch_size
            except tf.errors.OutOfRangeError:
                break

        if num_samples == 0:
            self.logger.info("No test samples found!")
            return

        # Compute average metrics
        test_loss = test_loss_total / num_samples
        csi_array = np.concatenate(csi_list, axis=0)
        csi45_array = np.concatenate(csi45_list, axis=0)

        csi_mean = np.nanmean(csi_array, axis=0)
        csi45_mean = np.nanmean(csi45_array, axis=0)

        wcsi = np.sum(temporal_weights * csi_mean) / np.sum(temporal_weights)
        wcsi45 = np.sum(temporal_weights * csi45_mean) / np.sum(temporal_weights)

        self.logger.info(f"CSI45 per timestep: {list(csi45_mean)}")
        self.logger.info('--------------------------------------------------------------------------')
        self.logger.info(f"CSI per timestep: {list(csi_mean)}")

        self.logger.info(
            f"------------------- Test loss: {test_loss:.6f}, "
            f"CSI45: {np.nanmean(csi45_mean):.6f}, CSI: {np.nanmean(csi_mean):.6f}, "
            f"wCSI45: {wcsi45:.6f}, wCSI: {wcsi:.6f} -------------------"
        )


def main():
    """Main function"""
    seed_tensorflow(seed=1997)

    # Setup
    update_generator = True

    if update_generator and os.path.exists(model_saving_path):
        # Only delete if explicitly requested
        # delete_file(model_saving_path)
        pass

    if os.path.isfile(running_log_path) and update_generator:
        # Only delete if explicitly requested
        # os.remove(running_log_path)
        pass

    logger = log.setting_log(running_log_path)

    logger.info(
        f"------------------- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"begin running -------------------"
    )

    # Count data samples
    try:
        train_files = tf.io.gfile.glob(training_dataset_path + '*.tfrecords')
        val_files = tf.io.gfile.glob(validation_dataset_path + '*.tfrecords')
        test_files = tf.io.gfile.glob(test_dataset_path + '*.tfrecords')
        train_data_num = len(train_files)
        valid_data_num = len(val_files)
        test_data_num = len(test_files)
    except:
        train_data_num = 1000
        valid_data_num = 200
        test_data_num = 200

    # Create datasets
    train_dataset = get_dataset_from_tfrecords(
        training_dataset_path + '*.tfrecords',
        is_train_dataset=True,
        batch_size=batch_size,
        shuffle_buffer_size=500
    )

    val_dataset = get_dataset_from_tfrecords(
        validation_dataset_path + '*.tfrecords',
        is_train_dataset=False,
        batch_size=val_batch_size,
        shuffle_buffer_size=500
    )

    test_dataset = get_dataset_from_tfrecords(
        test_dataset_path + '*.tfrecords',
        is_train_dataset=False,
        batch_size=test_batch_size,
        shuffle_buffer_size=500
    )

    # Log parameters
    logger.info(
        f"------------------- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"training parameters -------------------"
    )
    logger.info(f"train_data_num: {train_data_num}")
    logger.info(f"valid_data_num: {valid_data_num}")
    logger.info(f"test_data_num: {test_data_num}")
    logger.info(f"img_height: {img_height}")
    logger.info(f"img_width: {img_width}")
    logger.info(f"img_depth: {img_depth}")
    logger.info(f"input_length: {input_length}")
    logger.info(f"output_length: {output_length}")
    logger.info(f"training_steps: {training_steps}")
    logger.info(f"imgs_sum_steps: {imgs_sum_steps}")
    logger.info(f"validation_sum_steps: {validation_sum_steps}")
    logger.info(f"batch_size: {batch_size}")
    logger.info(f"val_batch_size: {val_batch_size}")
    logger.info(f"test_batch_size: {test_batch_size}")
    logger.info(f"optimizer: Adam")
    logger.info(f"generator_ini_learning_rate: {generator_ini_learning_rate}")

    # Create trainer
    trainer = ConvLSTM3DTrainer(conf, logger)

    logger.info(
        f"------------------- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
        f"start training -------------------"
    )

    if update_generator:
        # Training mode
        model_step = trainer.train(
            train_dataset,
            val_dataset,
            training_steps,
            validation_sum_steps
        )
        logger.info(f"Training completed at step {model_step}")
    else:
        # Testing mode
        logger.info(
            f"------------------- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"start testing -------------------"
        )
        trainer.test(test_dataset)
        logger.info(
            f"------------------- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
            f"end testing -------------------\n"
        )


if __name__ == '__main__':
    main()
