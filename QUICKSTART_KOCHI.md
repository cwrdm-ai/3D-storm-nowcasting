# Quick Start Guide: Kochi Radar Nowcasting with 3D ConvLSTM

This guide provides a simplified workflow to get you started quickly with training the 3D ConvLSTM model for Kochi radar nowcasting.

## Prerequisites

- Python 3.7+
- Kochi radar NetCDF files
- ~16 GB RAM (for preprocessing)
- GPU recommended for training (but not required for preprocessing)

---

## Step 1: Install Dependencies (5 minutes)

```bash
# Install preprocessing requirements
cd preprocessing
pip install -r requirements.txt

# Verify installation
python -c "import pyart; print('Py-ART installed successfully')"
```

---

## Step 2: Organize Your Data (2 minutes)

Create the following directory structure:

```bash
mkdir -p data/kochi_radar/raw_netcdf
```

Place your Kochi radar NetCDF files in `data/kochi_radar/raw_netcdf/`:

```
data/kochi_radar/raw_netcdf/
├── NETCDF5_kochi_weather_2021_06_30_20_00_00.nc
├── NETCDF5_kochi_weather_2021_06_30_20_05_00.nc
├── NETCDF5_kochi_weather_2021_06_30_20_10_00.nc
└── ... (at least 100+ files recommended)
```

**Note**: Files should be sorted chronologically with consistent 5-minute intervals.

---

## Step 3: Generate TFRecord Files (30-60 minutes)

Run the preprocessing pipeline:

```bash
cd preprocessing

python create_tfrecords.py \
  --input_dir ../data/kochi_radar/raw_netcdf/ \
  --output_dir ../data/kochi_radar/ \
  --input_length 8 \
  --output_length 12 \
  --apply_clutter_removal \
  --clutter_method gabella
```

**What this does**:
1. Reads all NetCDF files
2. Converts PPI → CAPPI (Cartesian grid)
3. Applies Gabella clutter removal
4. Creates sequences of 8 input + 12 output frames
5. Splits into train/val/test sets (70/15/15)
6. Generates TFRecord files

**Expected output**:
```
data/kochi_radar/
├── train/kochi_radar_train.tfrecord
├── val/kochi_radar_val.tfrecord
└── test/kochi_radar_test.tfrecord
```

**Progress check**:
```bash
ls -lh ../data/kochi_radar/train/
# Should show a .tfrecord file (size: 100MB-10GB depending on data)
```

---

## Step 4: Configure Training (2 minutes)

Edit `configs_kochi.ini` to update paths:

```ini
[dataset_paths]
training_dataset_path = ./data/kochi_radar/train/
validation_dataset_path = ./data/kochi_radar/val/
test_dataset_path = ./data/kochi_radar/test/
```

**For 30-minute forecast** (instead of 60 minutes), change:

```ini
[training_parameters]
output_length = 6  # 6 frames × 5 min = 30 minutes
```

---

## Step 5: Train the Model (hours to days)

```bash
cd ..
python train_test_convlstm3d.py
```

**Monitor training**:
```bash
# In another terminal
tail -f logs/kochi_radar_training_log.txt
```

**Expected output**:
- Model checkpoints saved in `pretraining_models/kochi_radar/`
- Training logs in `logs/kochi_radar_training_log.txt`
- Best model selected based on validation CSI score

**Training time estimates**:
- CPU: ~2-5 days (not recommended)
- GPU (GTX 1080 Ti): ~12-24 hours
- GPU (RTX 3090): ~6-12 hours

---

## Step 6: Evaluate Results

After training completes:

```bash
# Check test performance
grep "Test CSI" logs/kochi_radar_training_log.txt

# View generated forecast animations
ls gif/
# Should contain: ground_truth_0-1h.gif, nowcasts_0-1h.gif
```

**CSI Interpretation**:
- CSI > 0.6: Excellent
- CSI 0.4-0.6: Good
- CSI 0.2-0.4: Fair
- CSI < 0.2: Poor

---

## Typical Workflow for 30-Minute Nowcasting

### Input Specification

For **30-minute nowcasting**:

```python
# Temporal setup
time_interval = 5 minutes
input_frames = 8      # 8 × 5 = 40 minutes of past data
output_frames = 6     # 6 × 5 = 30 minutes of future forecast

# Spatial setup
grid_size = 120 × 120 pixels
grid_extent = 300 km × 300 km
resolution = 2.5 km/pixel

# Vertical setup
altitude_levels = 16
altitude_range = 0.5 to 15.5 km
vertical_resolution = 1 km
```

### Data Format

**Input to model**:
```
Shape: (batch, 8, 120, 120, 16, 1)
       - 8 time steps (t-40, t-35, t-30, t-25, t-20, t-15, t-10, t-5)
       - 120×120 spatial grid
       - 16 altitude levels
       - 1 channel (reflectivity)

Values: Normalized reflectivity [0, 1]
        0 = 0 dBZ (no echo)
        1 = 80 dBZ (extreme reflectivity)
```

**Output from model**:
```
Shape: (batch, 6, 120, 120, 16, 1)
       - 6 time steps (t+5, t+10, t+15, t+20, t+25, t+30)
       - Same spatial/vertical grid

Values: Normalized reflectivity [0, 1]
```

---

## Understanding Your Data

### What is CAPPI?

**CAPPI** (Constant Altitude PPI) represents radar reflectivity at fixed heights above sea level, rather than at fixed elevation angles.

**Why CAPPI for nowcasting?**
- Uniform vertical grid (required for 3D convolution)
- Height is physically meaningful (e.g., 5 km altitude)
- Easier to track storm vertical structure
- Better for machine learning

### Coordinate Transformation

Your raw Kochi data:
```
Format: PPI (Plan Position Indicator)
Coordinates: (sweep, azimuth, range)
Dimensions: (10 elevations, 360 azimuths, 1600 range gates)
```

After preprocessing:
```
Format: CAPPI (Constant Altitude PPI)
Coordinates: (x, y, z)
Dimensions: (120 x-points, 120 y-points, 16 z-levels)
```

The preprocessing script handles this transformation using Py-ART's gridding algorithms.

---

## Clutter Removal

### What is Clutter?

**Clutter** = non-meteorological echoes:
- Ground clutter (buildings, terrain)
- Sea clutter (ocean waves)
- Biological targets (birds, insects)
- Anomalous propagation

### Gabella Filter (Recommended)

The Gabella texture filter identifies clutter based on spatial variability:

```
High texture = Clutter (removed)
Low texture = Precipitation (kept)
```

**When to adjust**:
- **Too much data removed**: Increase `gabella_texture_threshold` to 25-30
- **Too much clutter remains**: Decrease `gabella_texture_threshold` to 15-18

Edit in `configs_kochi.ini`:
```ini
gabella_texture_threshold = 20.0  # Default
```

---

## Troubleshooting

### Problem: "Not enough data for training"

**Cause**: Need at least 100+ NetCDF files for minimal training.

**Solution**:
```bash
# Check number of files
ls -1 data/kochi_radar/raw_netcdf/*.nc | wc -l
# Should be >= 100 (minimum), 500+ recommended
```

### Problem: "All CAPPI values are NaN"

**Cause**: Grid extent exceeds radar range.

**Solution**: Reduce grid extent in preprocessing:
```python
# In create_tfrecords.py or kochi_radar_preprocessing.py
grid_limits=((-100, 100), (-100, 100), (0.5, 10.0))  # Smaller: 200×200 km
```

### Problem: "Training loss not decreasing"

**Possible causes**:
1. Learning rate too high/low
2. Not enough training data
3. Data quality issues

**Solutions**:
```bash
# Check data statistics
python -c "
import tensorflow as tf
dataset = tf.data.TFRecordDataset('data/kochi_radar/train/kochi_radar_train.tfrecord')
print(f'Number of samples: {sum(1 for _ in dataset)}')
"

# Visualize a sample
cd preprocessing
python example_usage.py  # Uncomment visualization examples
```

### Problem: "Out of memory during preprocessing"

**Solution**: Reduce grid size temporarily:
```python
# In preprocessing script
preprocessor = KochiRadarPreprocessor(
    grid_shape=(80, 80, 12),  # Smaller grid
    grid_limits=((-100, 100), (-100, 100), (0.5, 11.5))
)
```

---

## Performance Expectations

### Typical CSI Scores

Based on similar nowcasting models:

| Lead Time | Expected CSI (after training) |
|-----------|-------------------------------|
| **5 min** | 0.70 - 0.85 |
| **10 min** | 0.65 - 0.80 |
| **15 min** | 0.55 - 0.70 |
| **20 min** | 0.45 - 0.60 |
| **25 min** | 0.40 - 0.55 |
| **30 min** | 0.35 - 0.50 |

**Note**: Scores depend heavily on:
- Data quality
- Training data size
- Climate/weather regime
- Intensity thresholds

---

## Next Steps

After successful training:

1. **Evaluate on Test Set**:
   ```bash
   python train_test_convlstm3d.py --mode test
   ```

2. **Generate Visualizations**:
   ```bash
   # Forecast animations saved in gif/
   # Compare ground_truth_0-1h.gif vs nowcasts_0-1h.gif
   ```

3. **Fine-tune Hyperparameters**:
   - Learning rate
   - Batch size
   - Architecture depth
   - Loss function weights

4. **Operational Deployment**:
   - Create real-time inference pipeline
   - Set up automated data fetching
   - Implement forecast dissemination

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `preprocessing/kochi_radar_preprocessing.py` | Core preprocessing (PPI→CAPPI, clutter removal) |
| `preprocessing/create_tfrecords.py` | TFRecord generation |
| `preprocessing/example_usage.py` | Usage examples |
| `configs_kochi.ini` | Kochi-specific configuration |
| `train_test_convlstm3d.py` | Training script |
| `model/ConvLSTM3d_model.py` | Model architecture |

---

## Additional Resources

- **Detailed Documentation**: `preprocessing/README_PREPROCESSING.md`
- **Py-ART Docs**: https://arm-doe.github.io/pyart/
- **Model Paper**: See `README.md` for citations

---

## Getting Help

1. Check logs: `logs/kochi_radar_training_log.txt`
2. Review preprocessing README: `preprocessing/README_PREPROCESSING.md`
3. Visualize data: `preprocessing/example_usage.py`
4. Verify data format: Check TFRecord file sizes

---

**Good luck with your nowcasting project!** 🌧️⚡

---

**Last Updated**: 2025-11-09
**For**: Kochi C-band Radar Nowcasting (30-minute forecast)
