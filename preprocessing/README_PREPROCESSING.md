# Kochi Radar Data Preprocessing for 3D ConvLSTM Nowcasting

This directory contains the preprocessing pipeline for converting raw Kochi C-band radar data into the format required for training the 3D ConvLSTM nowcasting model.

## Overview

The preprocessing pipeline converts volumetric radar data from polar coordinates (PPI format) to Cartesian coordinates (CAPPI format), applies quality control and clutter removal, and generates TFRecord files for model training.

### Pipeline Flow

```
Raw NetCDF Files (PPI)
         ↓
  Read & Parse Data
         ↓
  Clutter Removal (Gabella/Fuzzy Logic)
         ↓
  Grid to Cartesian CAPPI
         ↓
  Normalize Reflectivity (0-1)
         ↓
  Create Temporal Sequences
         ↓
  Generate TFRecord Files
         ↓
  Train/Val/Test Split
         ↓
  Ready for Model Training
```

---

## Table of Contents

1. [Understanding the Data](#understanding-the-data)
2. [Coordinate Transformation Theory](#coordinate-transformation-theory)
3. [CAPPI Generation](#cappi-generation)
4. [Clutter Removal Methods](#clutter-removal-methods)
5. [Installation](#installation)
6. [Usage](#usage)
7. [File Descriptions](#file-descriptions)
8. [Configuration](#configuration)
9. [Troubleshooting](#troubleshooting)

---

## 1. Understanding the Data

### Kochi Radar Specifications

| Parameter | Value |
|-----------|-------|
| **Location** | 9.926°N, 76.262°E |
| **Altitude** | 30 m above MSL |
| **Band** | C-band (5.35 GHz, ~5.6 cm wavelength) |
| **Elevations** | 10 sweeps: 0.2°, 1°, 2°, 3°, 4.5°, 6°, 9°, 12°, 16°, 21° |
| **Azimuth Resolution** | 1° (360 rays per sweep) |
| **Range Gates** | 1600 gates |
| **Gate Size** | 250 m |
| **First Gate Range** | 500 m |
| **Maximum Range** | ~400 km |
| **Volume Scan Time** | ~30 seconds |

### Input Data Format

**NetCDF Structure:**
```
Dimensions:
  - sweep: 10 (elevation angles)
  - radial: 360 (azimuth angles)
  - bin: 1600 (range gates)

Variables:
  - Z(sweep, radial, bin): Reflectivity [dBZ]
  - V(sweep, radial, bin): Radial velocity [m/s]
  - W(sweep, radial, bin): Spectrum width [m/s]
  - ZDR(sweep, radial, bin): Differential reflectivity [dB]
  - PHIDP(sweep, radial, bin): Differential phase [degrees]
  - RHOHV(sweep, radial, bin): Correlation coefficient [-]
```

### Output Data Format

**Model Input Requirements:**
```
Shape: (batch, time, height, width, depth, channels)
       (4, 8, 120, 120, 16, 1)

- Batch: 4 samples per batch
- Time: 8 input frames (40 minutes at 5-min intervals)
- Height: 120 grid points (~300 km × 300 km domain)
- Width: 120 grid points
- Depth: 16 altitude levels (0.5 to 15.5 km)
- Channels: 1 (normalized reflectivity)
```

**Model Output:**
```
Shape: (batch, time, height, width, depth, channels)
       (4, 12, 120, 120, 16, 1)

- Time: 12 forecast frames (60 minutes at 5-min intervals)
```

---

## 2. Coordinate Transformation Theory

### Polar to Cartesian Conversion

Radar data is collected in **spherical polar coordinates** (range, azimuth, elevation) and must be converted to **Cartesian coordinates** (x, y, z) for gridding.

#### Step 1: Range to Height Conversion (4/3 Earth Radius Model)

The radar beam follows a curved path due to atmospheric refraction. Using the standard atmosphere model:

```
z = √(r² + Rₑ² + 2rRₑsin(θₑ)) - Rₑ
```

Where:
- `z`: Height above radar (m)
- `r`: Slant range (m)
- `θₑ`: Elevation angle (radians)
- `Rₑ`: Effective Earth radius = 4/3 × 6371 km = 8495 km

#### Step 2: Arc Length Calculation

```
s = Rₑ × arcsin(r×cos(θₑ) / (Rₑ + z))
```

Where:
- `s`: Arc length along Earth's surface (m)

#### Step 3: Cartesian Coordinates

```
x = s × sin(θₐ)
y = s × cos(θₐ)
```

Where:
- `θₐ`: Azimuth angle (radians, measured clockwise from North)
- `x`: East-West distance (m, positive = East)
- `y`: North-South distance (m, positive = North)

#### Step 4: Geographic Coordinates

Convert (x, y) to (latitude, longitude):

```
ρ = √(x² + y²)
c = ρ / R

lat = arcsin(cos(c)×sin(lat₀) + y×sin(c)×cos(lat₀)/ρ)
lon = lon₀ + arctan2(x×sin(c), ρ×cos(lat₀)×cos(c) - y×sin(lat₀)×sin(c))
```

Where:
- `(lat₀, lon₀)`: Radar location (9.926°N, 76.262°E)
- `R`: Earth radius = 6371 km

### Why Use Py-ART?

While these equations can be implemented manually, **Py-ART** (Python ARM Radar Toolkit) handles:
- Coordinate transformations with atmospheric corrections
- Quality control and data masking
- Interpolation to Cartesian grids
- Missing data handling
- Doppler dealiasing (if needed)

---

## 3. CAPPI Generation

### What is CAPPI?

**CAPPI** (Constant Altitude Plan Position Indicator) displays radar data at a constant altitude above sea level, unlike PPI which shows data at constant elevation angles.

### Why CAPPI for 3D ConvLSTM?

1. **Uniform vertical grid**: Essential for 3D convolutions
2. **Height-based features**: Better represents atmospheric structure
3. **Consistent across scans**: Elevation angles vary with range
4. **Physical interpretation**: Altitude is more meaningful than elevation angle

### CAPPI Interpolation Methods

**Barnes Weighting** (default in Py-ART):
```
w(r) = exp(-r² / (2κ²))
```

Where:
- `r`: Distance from grid point to radar gate
- `κ`: Smoothing parameter (related to data density)

**Cressman Weighting**:
```
w(r) = (R² - r²) / (R² + r²)  for r < R
w(r) = 0                       for r ≥ R
```

Where:
- `R`: Radius of influence

### Grid Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Grid Size** | 120 × 120 × 16 | (nx, ny, nz) |
| **Horizontal Extent** | -150 to +150 km | 300 km × 300 km domain |
| **Vertical Extent** | 0.5 to 15.5 km | 16 altitude levels |
| **Vertical Spacing** | 1 km | Uniform spacing |
| **Horizontal Resolution** | 2.5 km | 300 km / 120 = 2.5 km/pixel |

**CAPPI Altitude Levels** (km):
```
[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]
```

---

## 4. Clutter Removal Methods

Radar clutter (ground clutter, sea clutter, biological scatterers) must be removed to isolate precipitation echoes.

### Method 1: Gabella Texture Filter

**Principle**: Clutter has high spatial texture (variability) compared to precipitation.

**Algorithm**:
1. Calculate local texture (standard deviation) in a 5×5 window:
   ```
   Texture = √(E[Z²] - E[Z]²)
   ```

2. Classify as clutter if:
   ```
   Texture > threshold (default: 20 dBZ)
   ```

**Advantages**:
- Single-polarization method (works with Z only)
- Effective for ground clutter
- Fast computation

**Limitations**:
- May remove convective cores with high texture
- Threshold tuning required

### Method 2: Fuzzy Logic Classification

**Principle**: Combine multiple radar variables to classify precipitation vs. clutter.

**Dual-Polarization Variables**:

| Variable | Precipitation | Clutter |
|----------|---------------|---------|
| **RHOHV** | > 0.95 | < 0.80 |
| **ZDR** | 0-3 dB | < 0 or > 5 dB |
| **PHIDP** | Smooth | Noisy |
| **Texture(Z)** | Low | High |

**Membership Functions**:
```
P(precipitation|RHOHV) = {
  1.0  if RHOHV > 0.95
  0.5  if RHOHV = 0.85
  0.0  if RHOHV < 0.75
}
```

**Combined Score**:
```
Score = w₁×P(RHOHV) + w₂×P(ZDR) + w₃×P(Texture)
Classify as precipitation if Score > 0.6
```

**Advantages**:
- More robust than single-variable methods
- Uses full dual-pol capabilities
- Adaptive to different clutter types

**Limitations**:
- Requires dual-pol data
- More computationally intensive

### Recommended Approach

For Kochi radar (dual-polarization):
1. **Primary**: Gabella texture filter (fast, effective)
2. **Optional**: Fuzzy logic for critical applications

---

## 5. Installation

### Step 1: Install Dependencies

```bash
cd preprocessing
pip install -r requirements.txt
```

### Step 2: Verify Py-ART Installation

```bash
python -c "import pyart; print(f'Py-ART version: {pyart.__version__}')"
```

### Step 3: Test Preprocessing Module

```bash
python kochi_radar_preprocessing.py
```

---

## 6. Usage

### Quick Start

#### Option 1: Command-Line Interface

```bash
python create_tfrecords.py \
  --input_dir /path/to/radar/netcdf/files/ \
  --output_dir ./data/kochi_radar/ \
  --input_length 8 \
  --output_length 12 \
  --stride 1 \
  --apply_clutter_removal \
  --clutter_method gabella \
  --train_ratio 0.7 \
  --val_ratio 0.15
```

**Arguments**:
- `--input_dir`: Directory with NetCDF radar files
- `--output_dir`: Output directory for TFRecord files
- `--input_length`: Number of input frames (default: 8)
- `--output_length`: Number of forecast frames (default: 12)
- `--stride`: Stride between sequences (default: 1)
- `--apply_clutter_removal`: Enable clutter removal
- `--clutter_method`: Method (gabella/fuzzy/both)
- `--train_ratio`: Training set ratio (default: 0.7)
- `--val_ratio`: Validation set ratio (default: 0.15)

#### Option 2: Python Script

```python
from kochi_radar_preprocessing import KochiRadarPreprocessor
from create_tfrecords import TFRecordGenerator

# Initialize preprocessor
preprocessor = KochiRadarPreprocessor(
    grid_shape=(120, 120, 16),
    grid_limits=((-150, 150), (-150, 150), (0.5, 15.5))
)

# Process single file
data, grid = preprocessor.process_single_file(
    'radar_file.nc',
    apply_clutter_removal=True,
    clutter_method='gabella'
)

# Create TFRecords
tfrecord_gen = TFRecordGenerator(input_length=8, output_length=12)
sequences, timestamps = tfrecord_gen.create_sequences_from_radar_files(
    file_list, preprocessor, stride=1
)
```

### Step-by-Step Workflow

#### Step 1: Organize Raw Data

```
data/
└── kochi_radar/
    └── raw_netcdf/
        ├── NETCDF5_kochi_weather_2021_06_30_20_00_00.nc
        ├── NETCDF5_kochi_weather_2021_06_30_20_05_00.nc
        ├── NETCDF5_kochi_weather_2021_06_30_20_10_00.nc
        └── ...
```

#### Step 2: Run Preprocessing

```bash
python create_tfrecords.py \
  --input_dir ./data/kochi_radar/raw_netcdf/ \
  --output_dir ./data/kochi_radar/ \
  --apply_clutter_removal
```

#### Step 3: Verify Output

```
data/kochi_radar/
├── train/
│   └── kochi_radar_train.tfrecord
├── val/
│   └── kochi_radar_val.tfrecord
└── test/
    └── kochi_radar_test.tfrecord
```

#### Step 4: Update Configuration

Edit `configs_kochi.ini`:
```ini
[dataset_paths]
training_dataset_path = ./data/kochi_radar/train/
validation_dataset_path = ./data/kochi_radar/val/
test_dataset_path = ./data/kochi_radar/test/
```

#### Step 5: Train Model

```bash
cd ..
python train_test_convlstm3d.py --config configs_kochi.ini
```

---

## 7. File Descriptions

| File | Description |
|------|-------------|
| `kochi_radar_preprocessing.py` | Core preprocessing module with CAPPI generation and clutter removal |
| `create_tfrecords.py` | TFRecord generation script |
| `example_usage.py` | Example scripts demonstrating usage |
| `requirements.txt` | Python dependencies |
| `README_PREPROCESSING.md` | This file |

---

## 8. Configuration

### Grid Parameters

Adjust grid size and extent in `configs_kochi.ini`:

```ini
[kochi_radar_params]
grid_nx = 120
grid_ny = 120
grid_nz = 16
grid_x_min = -150  # km
grid_x_max = 150
grid_y_min = -150
grid_y_max = 150
grid_z_min = 0.5   # km
grid_z_max = 15.5
```

**Considerations**:
- **Larger grid**: Better coverage, more memory, slower processing
- **Smaller grid**: Faster processing, less coverage
- **Vertical levels**: Trade-off between resolution and memory

### Temporal Parameters

```ini
time_interval_minutes = 5
input_length = 8   # 40 minutes of input
output_length = 12  # 60 minutes of forecast
```

**For 30-Minute Forecast**:
```ini
output_length = 6  # 6 frames × 5 min = 30 minutes
```

### Clutter Removal

```ini
apply_clutter_removal = True
clutter_removal_method = gabella
gabella_texture_threshold = 20.0
gabella_texture_window_size = 5
```

**Tuning**:
- **Higher threshold** (e.g., 25): Less aggressive, preserves convective cores
- **Lower threshold** (e.g., 15): More aggressive, may remove valid precipitation

---

## 9. Troubleshooting

### Common Issues

#### Issue 1: "No module named 'pyart'"

**Solution**:
```bash
pip install arm-pyart
```

If this fails, try:
```bash
conda install -c conda-forge arm_pyart
```

#### Issue 2: "NetCDF file format not recognized"

**Cause**: Kochi radar uses custom NetCDF3 format.

**Solution**: The preprocessor handles this with custom parsing. Ensure `netCDF4` is installed:
```bash
pip install netCDF4
```

#### Issue 3: "Memory error during gridding"

**Cause**: Grid too large or too many files processed simultaneously.

**Solutions**:
1. Reduce grid size:
   ```python
   grid_shape=(100, 100, 12)  # Instead of (120, 120, 16)
   ```

2. Process in batches:
   ```python
   for batch in file_batches:
       sequences = process_batch(batch)
   ```

#### Issue 4: "All CAPPI data is NaN"

**Causes**:
- Grid limits exceed radar range
- Elevation angles too low for upper altitudes
- Too aggressive clutter removal

**Solutions**:
1. Check grid limits match radar range:
   ```python
   grid_limits=((-100, 100), (-100, 100), (0.5, 10.0))  # Smaller domain
   ```

2. Disable clutter removal for testing:
   ```python
   apply_clutter_removal=False
   ```

3. Visualize intermediate results:
   ```python
   import matplotlib.pyplot as plt
   plt.imshow(cappi_data[:, :, 8].T, origin='lower')
   plt.colorbar()
   plt.show()
   ```

#### Issue 5: "TFRecord file is empty"

**Cause**: Not enough files for creating sequences.

**Solution**: Need at least `input_length + output_length = 20` files:
```bash
ls -1 *.nc | wc -l  # Should be >= 20
```

---

## Data Requirements

### Minimum Dataset Size

For meaningful training:
- **Training**: ~500-1000 sequences (minimum)
- **Validation**: ~100-200 sequences
- **Test**: ~100-200 sequences

**Total files needed**: ~800-1400 NetCDF files

### Recommended Dataset Size

For robust performance:
- **Training**: ~2000-5000 sequences
- **Validation**: ~300-500 sequences
- **Test**: ~300-500 sequences

**Total files needed**: ~2600-6000 NetCDF files

### Data Collection Strategy

**Seasonal Coverage**: Include data from different seasons:
- Monsoon (June-September): Heavy convective precipitation
- Pre-monsoon (March-May): Isolated thunderstorms
- Post-monsoon (October-November): Moderate rainfall
- Winter (December-February): Light precipitation

**Diurnal Coverage**: Include different times of day:
- Morning (06:00-12:00)
- Afternoon (12:00-18:00)
- Evening (18:00-00:00)
- Night (00:00-06:00)

---

## Performance Optimization

### Speed Up Preprocessing

1. **Parallel Processing**:
   ```python
   from multiprocessing import Pool
   with Pool(processes=8) as pool:
       results = pool.map(preprocessor.process_single_file, file_list)
   ```

2. **Reduce Grid Resolution**:
   - Use 100×100 instead of 120×120 for testing
   - Reduce altitude levels from 16 to 12

3. **Skip Clutter Removal** (for testing):
   ```python
   apply_clutter_removal=False
   ```

### Memory Optimization

1. **Process in Batches**:
   ```python
   batch_size = 100
   for i in range(0, len(file_list), batch_size):
       batch = file_list[i:i+batch_size]
       process_batch(batch)
   ```

2. **Use float16** (if precision allows):
   ```python
   sequence_data = sequence_data.astype(np.float16)
   ```

---

## Quality Control Checks

### Visual Inspection

```python
# Plot CAPPI to check for artifacts
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cappi_data[:, :, 5].T, origin='lower', cmap='pyart_NWSRef')
axes[0].set_title('Low Level (3 km)')
axes[1].imshow(cappi_data[:, :, 10].T, origin='lower', cmap='pyart_NWSRef')
axes[1].set_title('Mid Level (7 km)')
axes[2].imshow(cappi_data[:, :, 14].T, origin='lower', cmap='pyart_NWSRef')
axes[2].set_title('Upper Level (11 km)')
plt.show()
```

### Statistical Checks

```python
# Check data range and coverage
print(f"Min: {np.nanmin(cappi_data)}, Max: {np.nanmax(cappi_data)}")
print(f"Valid data: {np.sum(~np.isnan(cappi_data)) / cappi_data.size * 100:.1f}%")
print(f"Mean reflectivity: {np.nanmean(cappi_data):.2f} dBZ")
```

---

## Next Steps

After preprocessing:

1. **Train Model**: Use `train_test_convlstm3d.py` with Kochi data
2. **Tune Hyperparameters**: Adjust learning rate, batch size, etc.
3. **Evaluate Performance**: Use CSI metrics at different lead times
4. **Optimize for Kochi Climate**: Consider monsoon-specific tuning

---

## References

1. **Py-ART Documentation**: https://arm-doe.github.io/pyart/
2. **Gabella et al. (2002)**: "Identification of ground clutter and anomalous propagation"
3. **Gourley et al. (2007)**: "Data quality of the Meteo-France C-band polarimetric radar"
4. **Lakshmanan et al. (2007)**: "An automated technique to quality control radar reflectivity data"

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#9-troubleshooting) section
2. Review example scripts in `example_usage.py`
3. Consult Py-ART documentation
4. Open an issue on GitHub

---

**Last Updated**: 2025-11-09
**Author**: AI Assistant for Kochi Radar Nowcasting Project
