# TensorFlow 2.x Migration Guide

This document describes the migration from TensorFlow 1.x to TensorFlow 2.x for the 3D ConvLSTM radar nowcasting project.

## Overview

The codebase has been updated to use TensorFlow 2.x (>=2.4.0) for improved performance, better APIs, and easier debugging with eager execution by default.

## Major Changes

### 1. **ConvLSTM Cell** (`RNN_cells/ConvLSTM3d_cell.py`)

**Before (TF 1.x):**
```python
class ConvLSTM(tf.contrib.rnn.RNNCell):
    def __init__(self, scope, ...):
        # Used tf.contrib APIs
        self.initializer = tf.contrib.layers.xavier_initializer()

    def __call__(self, input, state, scope=None):
        with tf.variable_scope(scope, reuse=self.reuse):
            # Manual variable scope management
```

**After (TF 2.x):**
```python
class ConvLSTM(layers.Layer):
    def __init__(self, name, ...):
        super(ConvLSTM, self).__init__(name=name, **kwargs)
        self.initializer = tf.keras.initializers.GlorotUniform(seed=1997)

    def call(self, inputs, states, training=None):
        # Automatic variable management
        # Uses Keras Layer API
```

**Key Changes:**
- Inherits from `tf.keras.layers.Layer` instead of `tf.contrib.rnn.RNNCell`
- Uses `build()` method for layer initialization
- Replaced `tf.contrib.layers.xavier_initializer()` with `tf.keras.initializers.GlorotUniform()`
- Replaced `tf.layers.conv3d` with `tf.keras.layers.Conv3D`
- Removed manual variable scope management

### 2. **Model Architecture** (`model/ConvLSTM3d_model.py`)

**Before (TF 1.x):**
```python
def generator(input, input_length, output_length, is_training=False, reuse=tf.AUTO_REUSE):
    with tf.variable_scope("generator", reuse=reuse):
        # Manual graph building
        conv = tf.layers.conv3d(...)
        # Used tf.nn.dynamic_rnn
```

**After (TF 2.x):**
```python
class ConvLSTM3DGenerator(Model):
    def __init__(self, input_length=8, output_length=12, **kwargs):
        super(ConvLSTM3DGenerator, self).__init__(name=name, **kwargs)
        # Initialize layers

    def call(self, inputs, training=None):
        # Forward pass with automatic differentiation
```

**Key Changes:**
- Model is now a `tf.keras.Model` subclass
- Uses Keras Functional/Sequential API for layer composition
- Automatic graph building with eager execution
- Replaced `tf.nn.dynamic_rnn` with manual iteration over timesteps
- Created separate `SpatialExtractor` and `Decoder` layer classes
- Added `compute_loss()` method for loss calculation

### 3. **Training Script** (`train_test_convlstm3d.py`)

**Before (TF 1.x):**
```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training loop
    for step in range(training_steps):
        sess.run(train_op)
        loss_val = sess.run(loss)

    # Checkpoint saving
    saver = tf.train.Saver()
    saver.save(sess, checkpoint_path)
```

**After (TF 2.x):**
```python
class ConvLSTM3DTrainer:
    def __init__(self, config, logger):
        self.model = ConvLSTM3DGenerator(...)
        self.optimizer = tf.keras.optimizers.Adam(...)

    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            predictions, labels, loss = self.model(inputs, training=True)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(...)
        return loss

    def train(self, train_dataset, val_dataset, ...):
        for step in range(training_steps):
            inputs, _ = next(train_iter)
            loss = self.train_step(inputs)
```

**Key Changes:**
- Removed `tf.Session()` - uses eager execution
- Created `ConvLSTM3DTrainer` class for organized training
- Uses `tf.GradientTape` for automatic differentiation
- Replaced `tf.train.Saver` with `tf.train.Checkpoint` and `CheckpointManager`
- Replaced `tf.placeholder` with direct tensor inputs
- Updated dataset iteration (removed `make_one_shot_iterator()`)
- Removed `tfdeterminism` dependency
- Uses `@tf.function` decorator for graph optimization

### 4. **Dataset Handling**

**Before (TF 1.x):**
```python
dataset = tf.data.TFRecordDataset(...)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

# In session
data = sess.run(next_element)
```

**After (TF 2.x):**
```python
dataset = tf.data.TFRecordDataset(...)
dataset = dataset.batch(...).prefetch(tf.data.AUTOTUNE)

# Direct iteration (eager execution)
for data in dataset:
    # Process data

# Or using iterator
iterator = iter(dataset)
data = next(iterator)
```

**Key Changes:**
- Direct iteration over datasets
- Used `tf.data.AUTOTUNE` for performance
- Updated parsing functions: `tf.parse_single_example` → `tf.io.parse_single_example`
- Updated decoding: `tf.decode_raw` → `tf.io.decode_raw`

### 5. **Evaluation Functions** (`utils/evaluation_functions.py`)

**Added TF 2.x compatible function:**
```python
def csi_score_tf(input, label, downvalue, upvalue):
    # Uses tf.logical_and instead of tf.bitwise operations
    # Uses tf.math.divide_no_nan for safer division
```

**Key Changes:**
- Added new `csi_score_tf()` function for TF 2.x
- Uses `tf.logical_and` for boolean operations
- Uses `tf.math.divide_no_nan()` to avoid division by zero

### 6. **Random Seed Setting**

**Before (TF 1.x):**
```python
tf.set_random_seed(seed)
```

**After (TF 2.x):**
```python
tf.random.set_seed(seed)
```

## Installation

### Option 1: Fresh Install

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install TensorFlow 2.x
pip install tensorflow>=2.4.0

# Install other requirements
pip install -r requirements.txt

# For preprocessing
pip install -r preprocessing/requirements.txt
```

### Option 2: Upgrade Existing Installation

```bash
# Uninstall TensorFlow 1.x
pip uninstall tensorflow

# Install TensorFlow 2.x
pip install tensorflow>=2.4.0

# Update other packages if needed
pip install --upgrade -r requirements.txt
```

### GPU Support

For GPU acceleration with TensorFlow 2.x:

```bash
# Install TensorFlow with GPU support
pip install tensorflow>=2.4.0

# Ensure you have:
# - CUDA Toolkit 11.x
# - cuDNN 8.x
# - Compatible NVIDIA GPU
```

Verify GPU availability:
```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

## Checkpoint Compatibility

**Important:** TensorFlow 2.x checkpoints are **not compatible** with TensorFlow 1.x checkpoints.

### Migrating Old Checkpoints

If you have trained models from TF 1.x, you'll need to:

1. **Option A:** Retrain the model with TF 2.x
2. **Option B:** Use TensorFlow's checkpoint conversion utilities (not guaranteed to work)

```python
# This is complex and may not work for all models
# It's recommended to retrain instead
```

## Configuration Changes

The configuration file (`configs_kochi.ini`) remains the same format, but note:

- Checkpoint paths will store TF 2.x format checkpoints
- Old TF 1.x checkpoints won't be loaded automatically

## Performance Improvements

TensorFlow 2.x provides several advantages:

1. **Eager Execution**: Easier debugging with Python-style execution
2. **@tf.function**: Automatic graph optimization for performance
3. **Better GPU Utilization**: Improved memory management
4. **Keras Integration**: Cleaner, more intuitive API
5. **Mixed Precision Training**: Easy to enable for faster training (optional)

### Enabling Mixed Precision (Optional)

For faster training on modern GPUs:

```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

## Testing the Migration

1. **Verify Installation:**
```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

2. **Test Model Creation:**
```python
from model.ConvLSTM3d_model import ConvLSTM3DGenerator
model = ConvLSTM3DGenerator(input_length=8, output_length=12)
print("Model created successfully!")
```

3. **Test Training Script:**
```bash
python train_test_convlstm3d.py
```

## Common Issues and Solutions

### Issue 1: Import Errors

**Error:** `ModuleNotFoundError: No module named 'tensorflow.contrib'`

**Solution:** This is expected. The code no longer uses `tf.contrib`. Make sure you're using the updated code.

### Issue 2: Session-related Errors

**Error:** `AttributeError: module 'tensorflow' has no attribute 'Session'`

**Solution:** TF 2.x doesn't use sessions. The new code uses eager execution.

### Issue 3: Checkpoint Loading Fails

**Error:** Checkpoint format incompatible

**Solution:** Delete old checkpoints and retrain, or ensure you're using the correct checkpoint path for TF 2.x format.

### Issue 4: Shape Mismatches

**Error:** Tensor shape errors during training

**Solution:** Check that your TFRecord files match the expected format:
- Shape: `(batch, time, depth, height, width, channels)`
- Time = 20 (8 input + 12 output)
- Depth = 16, Height = 120, Width = 120, Channels = 1

## File Changes Summary

| File | Status | Major Changes |
|------|--------|---------------|
| `RNN_cells/ConvLSTM3d_cell.py` | ✅ Updated | Keras Layer API, removed tf.contrib |
| `model/ConvLSTM3d_model.py` | ✅ Updated | Keras Model API, layer classes |
| `train_test_convlstm3d.py` | ✅ Updated | Eager execution, GradientTape, Checkpoint |
| `utils/evaluation_functions.py` | ✅ Updated | Added TF 2.x compatible CSI function |
| `preprocessing/create_tfrecords.py` | ✅ Updated | TF 2.x I/O APIs |
| `preprocessing/requirements.txt` | ✅ Updated | TensorFlow 2.x dependency |
| `requirements.txt` | ✅ New | Root requirements file |
| `configs_kochi.ini` | ✔️ No change | Configuration format unchanged |

## Additional Resources

- [TensorFlow 2.x Migration Guide (Official)](https://www.tensorflow.org/guide/migrate)
- [Keras API Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [tf.function Best Practices](https://www.tensorflow.org/guide/function)

## Support

For issues related to this migration, please:

1. Check this migration guide
2. Verify your TensorFlow version: `pip show tensorflow`
3. Ensure all dependencies are updated
4. Check the TensorFlow 2.x official documentation

## Changelog

- **2025-11-09**: Initial migration from TensorFlow 1.14 to TensorFlow 2.x
  - Updated all core model files
  - Migrated training script to eager execution
  - Updated preprocessing pipeline
  - Created migration documentation
