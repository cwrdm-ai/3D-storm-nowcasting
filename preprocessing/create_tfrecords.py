"""
TFRecord Generator for Kochi Radar Data

This module creates TFRecord files from preprocessed Kochi radar data
for training the 3D ConvLSTM nowcasting model.

The TFRecord format is required by the train_test_convlstm3d.py script.

Input: Preprocessed CAPPI data (nx=120, ny=120, nz=16)
Output: TFRecord files with sequences (input=8 frames, output=12 frames)

Author: AI Assistant
Date: 2025-11-09
"""

import os
import glob
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from tqdm import tqdm
import argparse
from kochi_radar_preprocessing import KochiRadarPreprocessor


class TFRecordGenerator:
    """
    Generate TFRecord files from preprocessed radar data for 3D ConvLSTM training.

    Parameters
    ----------
    input_length : int
        Number of input time steps (default: 8)
    output_length : int
        Number of output/forecast time steps (default: 12)
    total_length : int
        Total sequence length (input + output, default: 20)
    grid_shape : tuple
        (nx, ny, nz) grid dimensions (default: 120x120x16)
    """

    def __init__(self, input_length=8, output_length=12, grid_shape=(120, 120, 16)):
        self.input_length = input_length
        self.output_length = output_length
        self.total_length = input_length + output_length
        self.grid_shape = grid_shape

        print(f"\nTFRecord Generator initialized:")
        print(f"  Input length: {input_length} frames")
        print(f"  Output length: {output_length} frames")
        print(f"  Total sequence length: {self.total_length} frames")
        print(f"  Grid shape: {grid_shape}")


    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    def _float_feature(self, value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


    def _int64_feature(self, value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


    def serialize_example(self, sequence_data, timestamp=None):
        """
        Serialize a single sequence example to TFRecord format.

        Parameters
        ----------
        sequence_data : np.ndarray
            4D array of shape (total_length, nx, ny, nz)
        timestamp : str
            Timestamp of the sequence start (optional)

        Returns
        -------
        serialized : bytes
            Serialized example
        """
        # Ensure data is float32
        sequence_data = sequence_data.astype(np.float32)

        # Serialize the numpy array
        serialized_array = sequence_data.tobytes()

        # Create feature dictionary
        feature = {
            'data': self._bytes_feature(serialized_array),
            'height': self._int64_feature(self.grid_shape[0]),
            'width': self._int64_feature(self.grid_shape[1]),
            'depth': self._int64_feature(self.grid_shape[2]),
            'time_steps': self._int64_feature(self.total_length),
        }

        # Add timestamp if provided
        if timestamp is not None:
            feature['timestamp'] = self._bytes_feature(timestamp.encode('utf-8'))

        # Create Example protocol buffer
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

        return example_proto.SerializeToString()


    def create_tfrecord_from_sequences(self, sequences, output_filepath, timestamps=None):
        """
        Create a TFRecord file from multiple sequences.

        Parameters
        ----------
        sequences : list of np.ndarray
            List of sequence arrays, each of shape (total_length, nx, ny, nz)
        output_filepath : str
            Path to output TFRecord file
        timestamps : list of str
            List of timestamps for each sequence (optional)
        """
        print(f"\nCreating TFRecord: {output_filepath}")
        print(f"  Number of sequences: {len(sequences)}")

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

        # Write TFRecord
        with tf.io.TFRecordWriter(output_filepath) as writer:
            for i, sequence in enumerate(tqdm(sequences, desc="Writing sequences")):
                # Get timestamp if available
                timestamp = timestamps[i] if timestamps is not None else None

                # Serialize and write
                serialized = self.serialize_example(sequence, timestamp)
                writer.write(serialized)

        print(f"  TFRecord created successfully: {output_filepath}")
        print(f"  File size: {os.path.getsize(output_filepath) / 1024 / 1024:.2f} MB")


    def create_sequences_from_radar_files(self, file_list, preprocessor,
                                          stride=1, apply_clutter_removal=True):
        """
        Create overlapping sequences from a list of radar files.

        Parameters
        ----------
        file_list : list
            List of radar file paths (sorted by time)
        preprocessor : KochiRadarPreprocessor
            Initialized preprocessor object
        stride : int
            Stride between consecutive sequences (default: 1)
        apply_clutter_removal : bool
            Whether to apply clutter removal

        Returns
        -------
        sequences : list of np.ndarray
            List of sequences, each of shape (total_length, nx, ny, nz)
        timestamps : list of str
            List of timestamps for each sequence
        """
        print(f"\nCreating sequences from {len(file_list)} radar files...")
        print(f"  Sequence length: {self.total_length}")
        print(f"  Stride: {stride}")

        # First, preprocess all files
        print("\n[Step 1/2] Preprocessing all radar files...")
        processed_data_list = []
        timestamp_list = []

        for i, filepath in enumerate(tqdm(file_list, desc="Processing files")):
            try:
                # Process file
                processed_data, _ = preprocessor.process_single_file(
                    filepath, apply_clutter_removal=apply_clutter_removal
                )
                processed_data_list.append(processed_data)

                # Extract timestamp from filename
                # Example: NETCDF5_kochi_weather_2021_06_30_20_07_14.nc
                timestamp = self._extract_timestamp_from_filename(filepath)
                timestamp_list.append(timestamp)

            except Exception as e:
                print(f"\nERROR processing {os.path.basename(filepath)}: {str(e)}")
                continue

        print(f"\n  Successfully processed {len(processed_data_list)} files")

        # Create overlapping sequences
        print("\n[Step 2/2] Creating sequences...")
        sequences = []
        seq_timestamps = []

        num_sequences = (len(processed_data_list) - self.total_length) // stride + 1

        for i in range(0, len(processed_data_list) - self.total_length + 1, stride):
            # Extract sequence
            sequence = processed_data_list[i:i + self.total_length]
            sequence = np.stack(sequence, axis=0)  # (total_length, nx, ny, nz)

            sequences.append(sequence)
            seq_timestamps.append(timestamp_list[i])

        print(f"  Created {len(sequences)} sequences")

        return sequences, seq_timestamps


    def _extract_timestamp_from_filename(self, filepath):
        """
        Extract timestamp from Kochi radar filename.

        Example filename: NETCDF5_kochi_weather_2021_06_30_20_07_14.nc
        Extract: 2021-06-30 20:07:14
        """
        filename = os.path.basename(filepath)

        try:
            # Split by underscores
            parts = filename.split('_')

            # Find date and time parts
            # Format: YYYY_MM_DD_HH_MM_SS
            if len(parts) >= 6:
                year = parts[-6]
                month = parts[-5]
                day = parts[-4]
                hour = parts[-3]
                minute = parts[-2]
                second = parts[-1].replace('.nc', '')

                timestamp_str = f"{year}-{month}-{day} {hour}:{minute}:{second}"
                return timestamp_str
            else:
                return "unknown"

        except Exception as e:
            print(f"  Warning: Could not parse timestamp from {filename}")
            return "unknown"


    def split_train_val_test(self, sequences, timestamps,
                             train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split sequences into train, validation, and test sets.

        Parameters
        ----------
        sequences : list of np.ndarray
            List of sequences
        timestamps : list of str
            List of timestamps
        train_ratio : float
            Fraction for training set (default: 0.7)
        val_ratio : float
            Fraction for validation set (default: 0.15)
        test_ratio : float
            Fraction for test set (default: 0.15)

        Returns
        -------
        splits : dict
            Dictionary with keys 'train', 'val', 'test', each containing
            {'sequences': list, 'timestamps': list}
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        n_total = len(sequences)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        print(f"\nSplitting {n_total} sequences:")
        print(f"  Train: {n_train} ({train_ratio*100:.1f}%)")
        print(f"  Val: {n_val} ({val_ratio*100:.1f}%)")
        print(f"  Test: {n_total - n_train - n_val} ({test_ratio*100:.1f}%)")

        # Split
        train_sequences = sequences[:n_train]
        train_timestamps = timestamps[:n_train]

        val_sequences = sequences[n_train:n_train + n_val]
        val_timestamps = timestamps[n_train:n_train + n_val]

        test_sequences = sequences[n_train + n_val:]
        test_timestamps = timestamps[n_train + n_val:]

        return {
            'train': {'sequences': train_sequences, 'timestamps': train_timestamps},
            'val': {'sequences': val_sequences, 'timestamps': val_timestamps},
            'test': {'sequences': test_sequences, 'timestamps': test_timestamps}
        }


def main():
    """
    Main function to create TFRecord files from Kochi radar data.
    """
    parser = argparse.ArgumentParser(description='Create TFRecords from Kochi radar data')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing NetCDF radar files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for TFRecord files')
    parser.add_argument('--input_length', type=int, default=8,
                       help='Number of input time steps (default: 8)')
    parser.add_argument('--output_length', type=int, default=12,
                       help='Number of output time steps (default: 12)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride between sequences (default: 1)')
    parser.add_argument('--apply_clutter_removal', action='store_true',
                       help='Apply clutter removal (Gabella filter)')
    parser.add_argument('--clutter_method', type=str, default='gabella',
                       choices=['gabella', 'fuzzy', 'both'],
                       help='Clutter removal method (default: gabella)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')

    args = parser.parse_args()

    print("="*70)
    print("KOCHI RADAR TFRECORD GENERATOR")
    print("="*70)
    print(f"\nInput directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Input length: {args.input_length}")
    print(f"Output length: {args.output_length}")
    print(f"Stride: {args.stride}")
    print(f"Clutter removal: {args.apply_clutter_removal} ({args.clutter_method})")

    # Find all NetCDF files
    file_pattern = os.path.join(args.input_dir, "*.nc")
    file_list = sorted(glob.glob(file_pattern))

    if len(file_list) == 0:
        print(f"\nERROR: No NetCDF files found in {args.input_dir}")
        return

    print(f"\nFound {len(file_list)} NetCDF files")
    print(f"  First file: {os.path.basename(file_list[0])}")
    print(f"  Last file: {os.path.basename(file_list[-1])}")

    # Initialize preprocessor
    preprocessor = KochiRadarPreprocessor(
        grid_shape=(120, 120, 16),
        grid_limits=((-150, 150), (-150, 150), (0.5, 15.5))
    )

    # Initialize TFRecord generator
    tfrecord_gen = TFRecordGenerator(
        input_length=args.input_length,
        output_length=args.output_length,
        grid_shape=(120, 120, 16)
    )

    # Create sequences
    sequences, timestamps = tfrecord_gen.create_sequences_from_radar_files(
        file_list, preprocessor,
        stride=args.stride,
        apply_clutter_removal=args.apply_clutter_removal
    )

    if len(sequences) == 0:
        print("\nERROR: No sequences created. Check your data.")
        return

    # Split into train/val/test
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    splits = tfrecord_gen.split_train_val_test(
        sequences, timestamps,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio
    )

    # Create output directories
    train_dir = os.path.join(args.output_dir, 'train')
    val_dir = os.path.join(args.output_dir, 'val')
    test_dir = os.path.join(args.output_dir, 'test')

    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    # Create TFRecord files
    print("\n" + "="*70)
    print("CREATING TFRECORD FILES")
    print("="*70)

    # Training set
    tfrecord_gen.create_tfrecord_from_sequences(
        splits['train']['sequences'],
        os.path.join(train_dir, 'kochi_radar_train.tfrecord'),
        splits['train']['timestamps']
    )

    # Validation set
    tfrecord_gen.create_tfrecord_from_sequences(
        splits['val']['sequences'],
        os.path.join(val_dir, 'kochi_radar_val.tfrecord'),
        splits['val']['timestamps']
    )

    # Test set
    tfrecord_gen.create_tfrecord_from_sequences(
        splits['test']['sequences'],
        os.path.join(test_dir, 'kochi_radar_test.tfrecord'),
        splits['test']['timestamps']
    )

    print("\n" + "="*70)
    print("TFRECORD GENERATION COMPLETE")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  Train: {os.path.join(train_dir, 'kochi_radar_train.tfrecord')}")
    print(f"  Val: {os.path.join(val_dir, 'kochi_radar_val.tfrecord')}")
    print(f"  Test: {os.path.join(test_dir, 'kochi_radar_test.tfrecord')}")


if __name__ == "__main__":
    main()
