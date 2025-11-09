"""
Example Usage Script for Kochi Radar Preprocessing

This script demonstrates how to:
1. Preprocess a single Kochi radar NetCDF file
2. Create temporal sequences from multiple files
3. Generate TFRecord files for training
4. Visualize the preprocessed data

Author: AI Assistant
Date: 2025-11-09
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from kochi_radar_preprocessing import KochiRadarPreprocessor
from create_tfrecords import TFRecordGenerator


def example_1_process_single_file():
    """
    Example 1: Process a single radar file and visualize the CAPPI.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Process a Single Radar File")
    print("="*70)

    # Initialize preprocessor
    preprocessor = KochiRadarPreprocessor(
        grid_shape=(120, 120, 16),
        grid_limits=((-150, 150), (-150, 150), (0.5, 15.5))
    )

    # Example file path (update this to your actual file)
    filepath = "path/to/NETCDF5_kochi_weather_2021_06_30_20_07_14.nc"

    if not os.path.exists(filepath):
        print(f"\nWARNING: Example file not found: {filepath}")
        print("Please update the filepath in this script.")
        return

    # Process the file
    processed_data, grid = preprocessor.process_single_file(
        filepath,
        apply_clutter_removal=True,
        clutter_method='gabella'
    )

    # Visualize CAPPI at different altitudes
    visualize_cappi(processed_data, preprocessor.cappi_altitudes)

    print("\nExample 1 complete!")


def example_2_create_temporal_sequence():
    """
    Example 2: Create a temporal sequence from multiple radar files.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Create Temporal Sequence")
    print("="*70)

    # Initialize preprocessor
    preprocessor = KochiRadarPreprocessor(
        grid_shape=(120, 120, 16),
        grid_limits=((-150, 150), (-150, 150), (0.5, 15.5))
    )

    # Get list of radar files (update this path)
    data_dir = "path/to/radar_files/"
    file_list = sorted(glob.glob(os.path.join(data_dir, "*.nc")))

    if len(file_list) < 8:
        print(f"\nWARNING: Found only {len(file_list)} files in {data_dir}")
        print("Need at least 8 files for a sequence.")
        return

    print(f"\nFound {len(file_list)} radar files")

    # Create sequence (8 time steps)
    sequence = preprocessor.create_temporal_sequence(
        file_list[:8],
        sequence_length=8,
        apply_clutter_removal=True
    )

    print(f"\nSequence shape: {sequence.shape}")
    print(f"  Time steps: {sequence.shape[0]}")
    print(f"  Spatial dimensions: {sequence.shape[1]} x {sequence.shape[2]}")
    print(f"  Altitude levels: {sequence.shape[3]}")

    # Visualize temporal evolution
    visualize_temporal_sequence(sequence, preprocessor.cappi_altitudes)

    print("\nExample 2 complete!")


def example_3_generate_tfrecords():
    """
    Example 3: Generate TFRecord files for training.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Generate TFRecord Files")
    print("="*70)

    # Paths (update these)
    input_dir = "path/to/radar_files/"
    output_dir = "./data/kochi_radar/"

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"\nWARNING: Input directory not found: {input_dir}")
        print("Please update the input_dir in this script.")
        return

    # Initialize preprocessor
    preprocessor = KochiRadarPreprocessor(
        grid_shape=(120, 120, 16),
        grid_limits=((-150, 150), (-150, 150), (0.5, 15.5))
    )

    # Initialize TFRecord generator
    tfrecord_gen = TFRecordGenerator(
        input_length=8,
        output_length=12,
        grid_shape=(120, 120, 16)
    )

    # Get file list
    file_list = sorted(glob.glob(os.path.join(input_dir, "*.nc")))
    print(f"\nFound {len(file_list)} radar files")

    if len(file_list) < 20:
        print(f"WARNING: Need at least 20 files for meaningful train/val/test split")
        return

    # Create sequences
    sequences, timestamps = tfrecord_gen.create_sequences_from_radar_files(
        file_list,
        preprocessor,
        stride=1,
        apply_clutter_removal=True
    )

    # Split into train/val/test
    splits = tfrecord_gen.split_train_val_test(
        sequences, timestamps,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )

    # Create TFRecord files
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)

    tfrecord_gen.create_tfrecord_from_sequences(
        splits['train']['sequences'],
        os.path.join(output_dir, 'train', 'kochi_radar_train.tfrecord'),
        splits['train']['timestamps']
    )

    tfrecord_gen.create_tfrecord_from_sequences(
        splits['val']['sequences'],
        os.path.join(output_dir, 'val', 'kochi_radar_val.tfrecord'),
        splits['val']['timestamps']
    )

    tfrecord_gen.create_tfrecord_from_sequences(
        splits['test']['sequences'],
        os.path.join(output_dir, 'test', 'kochi_radar_test.tfrecord'),
        splits['test']['timestamps']
    )

    print("\nExample 3 complete!")
    print(f"\nTFRecord files created in: {output_dir}")


def visualize_cappi(cappi_data, altitudes):
    """
    Visualize CAPPI at different altitude levels.

    Parameters
    ----------
    cappi_data : np.ndarray
        3D array (nx, ny, nz) of normalized reflectivity
    altitudes : np.ndarray
        Altitude levels in km
    """
    # Select a few altitude levels to visualize
    altitude_indices = [0, 4, 8, 12, 15]  # ~0.5, 4.5, 8.5, 12.5, 15.5 km

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle('CAPPI at Different Altitudes', fontsize=16, fontweight='bold')

    for i, idx in enumerate(altitude_indices):
        if idx < cappi_data.shape[2]:
            # Get CAPPI slice
            cappi_slice = cappi_data[:, :, idx]

            # Plot
            im = axes[i].imshow(cappi_slice.T, origin='lower', cmap='pyart_NWSRef',
                               vmin=0, vmax=1, aspect='equal')
            axes[i].set_title(f'Altitude: {altitudes[idx]:.1f} km', fontweight='bold')
            axes[i].set_xlabel('X (grid points)')
            axes[i].set_ylabel('Y (grid points)')
            axes[i].grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8)
    cbar.set_label('Normalized Reflectivity (0-1)', fontsize=12)

    plt.tight_layout()
    plt.savefig('cappi_visualization.png', dpi=150, bbox_inches='tight')
    print("\nCAPPI visualization saved to: cappi_visualization.png")
    plt.show()


def visualize_temporal_sequence(sequence, altitudes, altitude_idx=8):
    """
    Visualize temporal evolution of CAPPI at a specific altitude.

    Parameters
    ----------
    sequence : np.ndarray
        4D array (time, nx, ny, nz) of normalized reflectivity
    altitudes : np.ndarray
        Altitude levels in km
    altitude_idx : int
        Index of altitude level to visualize (default: 8 -> ~8.5 km)
    """
    n_timesteps = sequence.shape[0]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Temporal Evolution at {altitudes[altitude_idx]:.1f} km Altitude',
                 fontsize=16, fontweight='bold')

    axes = axes.flatten()

    for t in range(n_timesteps):
        # Get CAPPI slice at this time and altitude
        cappi_slice = sequence[t, :, :, altitude_idx]

        # Plot
        im = axes[t].imshow(cappi_slice.T, origin='lower', cmap='pyart_NWSRef',
                           vmin=0, vmax=1, aspect='equal')
        axes[t].set_title(f'Time Step {t+1}', fontweight='bold')
        axes[t].set_xlabel('X (grid points)')
        axes[t].set_ylabel('Y (grid points)')
        axes[t].grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes, orientation='horizontal', pad=0.1, shrink=0.8)
    cbar.set_label('Normalized Reflectivity (0-1)', fontsize=12)

    plt.tight_layout()
    plt.savefig('temporal_sequence_visualization.png', dpi=150, bbox_inches='tight')
    print("\nTemporal sequence visualization saved to: temporal_sequence_visualization.png")
    plt.show()


def main():
    """
    Main function to run examples.
    """
    print("\n" + "="*70)
    print("KOCHI RADAR PREPROCESSING - EXAMPLE USAGE")
    print("="*70)

    print("\nAvailable examples:")
    print("  1. Process a single radar file and visualize CAPPI")
    print("  2. Create temporal sequence from multiple files")
    print("  3. Generate TFRecord files for training")
    print("\nNote: Update file paths in this script before running!")

    # Run examples (uncomment to execute)
    # example_1_process_single_file()
    # example_2_create_temporal_sequence()
    # example_3_generate_tfrecords()

    print("\nTo run these examples:")
    print("  1. Update the file paths in this script")
    print("  2. Uncomment the example functions at the bottom")
    print("  3. Run: python example_usage.py")


if __name__ == "__main__":
    main()
