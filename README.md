# 3D-storm-nowcasting
This is a TensorFlow implementation of 3D-ConvLSTM, a three-dimensional gridded radar echo extrapolation model for convective storm nowcasting as described in the following paper:

* [Three-Dimensional Gridded Radar Echo Extrapolation for Convective Storm Nowcasting Based on 3D-ConvLSTM Model](http://https://www.mdpi.com/2072-4292/14/17/4256), by Nengli Sun, Zeming Zhou, Qian Li, and Jinrui Jing.

If you use this method or this code in your research, please cite as:

```
@Article{rs14174256,
AUTHOR = {Sun, Nengli and Zhou, Zeming and Li, Qian and Jing, Jinrui},
TITLE = {Three-Dimensional Gridded Radar Echo Extrapolation for Convective Storm Nowcasting Based on 3D-ConvLSTM Model},
JOURNAL = {Remote Sensing},
VOLUME = {14},
YEAR = {2022},
NUMBER = {17},
ARTICLE-NUMBER = {4256},
DOI = {10.3390/rs14174256}
}
```

## Dependencies

**⚠️ UPDATED TO TENSORFLOW 2.x**: This codebase has been migrated from TensorFlow 1.14 to TensorFlow 2.x. See `TF2_MIGRATION_GUIDE.md` for details.

### Current Requirements
- Python >= 3.7
- **TensorFlow >= 2.4.0** (CPU or GPU version)
- NumPy >= 1.19.0
- Other dependencies in `requirements.txt`

### Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# For Kochi radar preprocessing
pip install -r preprocessing/requirements.txt
```

### GPU Support (Optional)

For GPU acceleration, ensure you have:
- CUDA Toolkit 11.x
- cuDNN 8.x
- Compatible NVIDIA GPU

Verify GPU availability:
```python
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
```

## Quick Start

### For Kochi Radar Nowcasting

See [`QUICKSTART_KOCHI.md`](QUICKSTART_KOCHI.md) for a complete guide to:
1. Install dependencies
2. Preprocess Kochi radar data
3. Train the 3D ConvLSTM model
4. Evaluate results

### Migration from TensorFlow 1.x

If you have an existing installation or trained models from TensorFlow 1.x:
- See [`TF2_MIGRATION_GUIDE.md`](TF2_MIGRATION_GUIDE.md) for migration instructions
- **Note**: TF 1.x checkpoints are not compatible with TF 2.x - you'll need to retrain

## References and Github Links
[1] Mustafa, M.A. A data-driven learning approach to image registration. University of Nottingham. 2016.("https://github.com/Mustafa3946/Lucas-Kanade-3D-Optical-Flow")

[2] Ayzel, G.; Heistermann, M.; Winterrath, T. Optical flow models as an open benchmark for radar-based precipitation nowcasting (rainymotion v0.1). Geoscientific Model Development 2019, 12, 1387-1402.("https://github.com/hydrogo/rainymotion")

[3] Pulkkinen, S.; Nerini, D.; Pérez Hortal, A.A.; Velasco-Forero, C.; Seed, A.; Germann, U.; Foresti, L. Pysteps: An open-source python library for probabilistic precipitation nowcasting (v1.0). Geoscientific Model Development 2019, 12, 4185-4219.("https://github.com/pySTEPS/pysteps")

[4] Wang, Y.; Long, M.; Wang, J.; Gao, Z.; Yu, P.S. Predrnn: Recurrent neural networks for predictive learning using spatiotemporal lstms. Advances in Neural Information Processing Systems, 2017; pp. 879–888.("https://github.com/Yunbo426/predrnn-pp")

[5] Shi, X.; Chen, Z.; Wang, H.; Yeung, D.Y.; Wong, W.K.; Woo, W.-c. Convolutional LSTM network: A machine learning approach for precipitation nowcasting. Advances in Neural Information Processing Systems, 2015; pp. 802–810.
