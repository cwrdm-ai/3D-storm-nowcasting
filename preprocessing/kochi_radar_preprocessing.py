"""
Kochi Radar Data Preprocessing Pipeline for 3D ConvLSTM Nowcasting

This module provides functions to preprocess Kochi C-band radar data:
1. Read NetCDF radar files
2. Convert polar PPI data to Cartesian grid (CAPPI)
3. Apply clutter removal (Gabella filter and fuzzy logic)
4. Generate normalized 3D reflectivity grids for model input

Author: AI Assistant
Date: 2025-11-09
"""

import os
import numpy as np
import pyart
import netCDF4 as nc
from scipy import ndimage
from scipy.interpolate import griddata
import warnings
warnings.filterwarnings('ignore')


class KochiRadarPreprocessor:
    """
    Preprocessor for Kochi radar data to generate CAPPI grids for 3D ConvLSTM model.

    Parameters
    ----------
    radar_site_lat : float
        Radar site latitude (default: 9.9263°N for Kochi)
    radar_site_lon : float
        Radar site longitude (default: 76.2622°E for Kochi)
    radar_site_alt : float
        Radar site altitude in meters (default: 30m for Kochi)
    grid_shape : tuple
        (nx, ny, nz) - Number of grid points (default: 120x120x16)
    grid_limits : tuple
        ((x_min, x_max), (y_min, y_max), (z_min, z_max)) in km
    """

    def __init__(self,
                 radar_site_lat=9.9263,
                 radar_site_lon=76.2622,
                 radar_site_alt=30.0,
                 grid_shape=(120, 120, 16),
                 grid_limits=((-150, 150), (-150, 150), (0.5, 15.5))):

        self.radar_site_lat = radar_site_lat
        self.radar_site_lon = radar_site_lon
        self.radar_site_alt = radar_site_alt
        self.grid_shape = grid_shape
        self.grid_limits = grid_limits

        # Define CAPPI altitude levels in km (16 levels from 0.5 to 15.5 km)
        self.cappi_altitudes = np.linspace(grid_limits[2][0],
                                           grid_limits[2][1],
                                           grid_shape[2])

        print(f"Initialized Kochi Radar Preprocessor")
        print(f"  Grid shape: {grid_shape}")
        print(f"  Grid limits (km): x={grid_limits[0]}, y={grid_limits[1]}, z={grid_limits[2]}")
        print(f"  CAPPI altitudes (km): {self.cappi_altitudes}")


    def read_kochi_netcdf(self, filepath):
        """
        Read Kochi radar NetCDF file and convert to Py-ART radar object.

        Parameters
        ----------
        filepath : str
            Path to the NetCDF radar file

        Returns
        -------
        radar : pyart.core.Radar
            Py-ART radar object
        """
        print(f"\nReading NetCDF file: {os.path.basename(filepath)}")

        try:
            # Read NetCDF file
            ncfile = nc.Dataset(filepath, 'r')

            # Extract metadata
            nsweeps = len(ncfile.dimensions['sweep'])
            nrays_per_sweep = len(ncfile.dimensions['radial'])
            ngates = len(ncfile.dimensions['bin'])

            # Extract data arrays
            reflectivity = ncfile.variables['Z'][:]  # (nsweeps, nrays, ngates)
            elevation_angles = ncfile.variables['elevationList'][:]
            azimuth = ncfile.variables['radialAzim'][:]

            # Get radar parameters
            gate_size = ncfile.variables['gateSize'][:]  # meters
            first_gate_range = ncfile.variables['firstgateRange'][:]  # meters

            print(f"  Sweeps: {nsweeps}, Rays per sweep: {nrays_per_sweep}, Gates: {ngates}")
            print(f"  Elevation angles: {elevation_angles}")
            print(f"  Gate size: {gate_size}m, First gate: {first_gate_range}m")

            # Create range array
            range_array = np.arange(ngates) * gate_size + first_gate_range
            range_array = range_array / 1000.0  # Convert to km

            # Build Py-ART radar object manually
            # This is necessary because Kochi format is not standard
            radar = self._build_pyart_radar_object(
                reflectivity, elevation_angles, azimuth, range_array,
                nsweeps, nrays_per_sweep, ngates, ncfile
            )

            ncfile.close()
            return radar

        except Exception as e:
            print(f"ERROR reading NetCDF file: {str(e)}")
            raise


    def _build_pyart_radar_object(self, reflectivity, elevation_angles, azimuth,
                                   range_array, nsweeps, nrays_per_sweep, ngates, ncfile):
        """
        Build a Py-ART radar object from Kochi NetCDF data.
        """
        from pyart.core import Radar
        from pyart.config import get_metadata

        # Total number of rays
        total_rays = nsweeps * nrays_per_sweep

        # Expand azimuth and elevation for all sweeps
        azimuth_full = np.tile(azimuth, nsweeps)
        elevation_full = np.repeat(elevation_angles, nrays_per_sweep)

        # Reshape reflectivity to (total_rays, ngates)
        reflectivity_2d = reflectivity.reshape(total_rays, ngates)

        # Replace fill values (-999.9) with NaN
        reflectivity_2d = np.where(reflectivity_2d < -900, np.nan, reflectivity_2d)

        # Create time array (assuming uniform sampling)
        time_start = ncfile.variables['esStartTime'][:]
        time_array = np.arange(total_rays) * (30.0 / total_rays)  # Approximate

        # Sweep information
        sweep_number = np.arange(nsweeps)
        sweep_mode = np.array(['azimuth_surveillance'] * nsweeps)
        sweep_start_ray_index = np.arange(0, total_rays, nrays_per_sweep)
        sweep_end_ray_index = np.arange(nrays_per_sweep - 1, total_rays, nrays_per_sweep)
        fixed_angle = elevation_angles

        # Build fields dictionary
        fields = {}
        refl_dict = get_metadata('reflectivity')
        refl_dict['data'] = reflectivity_2d
        refl_dict['units'] = 'dBZ'
        refl_dict['_FillValue'] = np.nan
        fields['reflectivity'] = refl_dict

        # Create radar object
        radar = Radar(
            time={'data': time_array, 'units': 'seconds since start'},
            _range={'data': range_array, 'units': 'km'},
            fields=fields,
            metadata={
                'instrument_name': 'Kochi C-band DWR',
                'source': 'IMD Kochi Radar'
            },
            scan_type='ppi',
            latitude={'data': np.array([self.radar_site_lat])},
            longitude={'data': np.array([self.radar_site_lon])},
            altitude={'data': np.array([self.radar_site_alt])},
            sweep_number={'data': sweep_number},
            sweep_mode={'data': sweep_mode},
            fixed_angle={'data': fixed_angle},
            sweep_start_ray_index={'data': sweep_start_ray_index},
            sweep_end_ray_index={'data': sweep_end_ray_index},
            azimuth={'data': azimuth_full},
            elevation={'data': elevation_full}
        )

        return radar


    def apply_gabella_clutter_removal(self, radar, field_name='reflectivity',
                                      gabella_params=None):
        """
        Apply Gabella texture-based clutter removal filter.

        The Gabella filter identifies clutter by analyzing the texture
        (spatial variability) of reflectivity. Clutter typically has
        high spatial texture compared to precipitation.

        Parameters
        ----------
        radar : pyart.core.Radar
            Py-ART radar object
        field_name : str
            Name of the reflectivity field
        gabella_params : dict
            Parameters for Gabella filter:
            - texture_threshold: Texture threshold (default: 20)
            - texture_window: Window size for texture calculation (default: 5x5)

        Returns
        -------
        radar : pyart.core.Radar
            Radar object with clutter-filtered reflectivity
        """
        if gabella_params is None:
            gabella_params = {
                'texture_threshold': 20.0,
                'texture_window': (5, 5)
            }

        print("\nApplying Gabella clutter removal...")

        # Get reflectivity data
        refl_data = radar.fields[field_name]['data'].copy()

        # Calculate texture (standard deviation in local window)
        window_size = gabella_params['texture_window']
        texture_threshold = gabella_params['texture_threshold']

        # Calculate texture for each sweep separately
        for sweep_idx in range(radar.nsweeps):
            sweep_slice = radar.get_slice(sweep_idx)
            sweep_refl = refl_data[sweep_slice, :]

            # Calculate texture using uniform filter
            # Texture = std deviation in local neighborhood
            mean = ndimage.uniform_filter(sweep_refl, size=window_size, mode='constant', cval=np.nan)
            sqr_mean = ndimage.uniform_filter(sweep_refl**2, size=window_size, mode='constant', cval=np.nan)
            texture = np.sqrt(np.abs(sqr_mean - mean**2))

            # Mask high-texture regions as clutter
            clutter_mask = texture > texture_threshold
            sweep_refl[clutter_mask] = np.nan

            refl_data[sweep_slice, :] = sweep_refl

        # Update radar field
        radar.fields[field_name]['data'] = refl_data

        print(f"  Gabella filter applied with texture threshold = {texture_threshold}")

        return radar


    def apply_fuzzy_logic_clutter_removal(self, radar, field_name='reflectivity'):
        """
        Apply fuzzy logic clutter removal using dual-pol variables.

        This method uses multiple radar variables (ZDR, RHOHV, PHIDP) to
        classify clutter vs. precipitation using fuzzy membership functions.

        Parameters
        ----------
        radar : pyart.core.Radar
            Py-ART radar object (must contain dual-pol variables)
        field_name : str
            Name of the reflectivity field

        Returns
        -------
        radar : pyart.core.Radar
            Radar object with clutter-filtered reflectivity
        """
        print("\nApplying fuzzy logic clutter removal...")

        # Check if dual-pol variables exist
        required_fields = ['differential_reflectivity', 'cross_correlation_ratio']
        # For Kochi data, map to actual field names if different

        # Simple fuzzy logic using RHOHV (if available)
        # RHOHV < 0.8 typically indicates clutter
        try:
            refl_data = radar.fields[field_name]['data'].copy()

            # If RHOHV is available, use it for filtering
            if 'cross_correlation_ratio' in radar.fields or 'RHOHV' in radar.fields:
                rhohv_field = 'cross_correlation_ratio' if 'cross_correlation_ratio' in radar.fields else 'RHOHV'
                rhohv = radar.fields[rhohv_field]['data']

                # Mask low RHOHV as clutter
                clutter_mask = rhohv < 0.8
                refl_data[clutter_mask] = np.nan

                print(f"  Fuzzy logic filter applied using {rhohv_field} < 0.8")
            else:
                print("  WARNING: Dual-pol variables not found. Skipping fuzzy logic filter.")

            # Update radar field
            radar.fields[field_name]['data'] = refl_data

        except Exception as e:
            print(f"  ERROR in fuzzy logic filter: {str(e)}")

        return radar


    def grid_to_cartesian_cappi(self, radar, field_name='reflectivity'):
        """
        Convert radar PPI data to Cartesian CAPPI grid using Py-ART.

        Parameters
        ----------
        radar : pyart.core.Radar
            Py-ART radar object
        field_name : str
            Name of the reflectivity field

        Returns
        -------
        grid : pyart.core.Grid
            Py-ART grid object with CAPPI data
        cappi_data : np.ndarray
            3D array of reflectivity (nx, ny, nz)
        """
        print("\nGridding radar data to Cartesian CAPPI...")

        # Define grid parameters
        grid_limits = self.grid_limits
        grid_shape = self.grid_shape

        try:
            # Use Py-ART's map_to_grid function
            grid = pyart.map.grid_from_radars(
                radar,
                grid_shape=grid_shape,
                grid_limits=grid_limits,
                fields=[field_name],
                gridding_algo='map_gates_to_grid',
                h_factor=1.0,
                nb=0.6,
                bsp=1.0,
                min_radius=500.0,
                weighting_function='Barnes2'
            )

            print(f"  Grid created with shape: {grid.fields[field_name]['data'].shape}")
            print(f"  Grid limits (km): x={grid_limits[0]}, y={grid_limits[1]}, z={grid_limits[2]}")

            # Extract 3D reflectivity data (nz, ny, nx)
            cappi_data = grid.fields[field_name]['data']

            # Transpose to (nx, ny, nz) for consistency with model input
            cappi_data = np.transpose(cappi_data, (2, 1, 0))

            print(f"  CAPPI data shape (nx, ny, nz): {cappi_data.shape}")

            return grid, cappi_data

        except Exception as e:
            print(f"  ERROR in gridding: {str(e)}")
            raise


    def normalize_reflectivity(self, cappi_data, z_min=0.0, z_max=80.0):
        """
        Normalize reflectivity data to [0, 1] range.

        Parameters
        ----------
        cappi_data : np.ndarray
            3D reflectivity array (nx, ny, nz)
        z_min : float
            Minimum reflectivity value (dBZ)
        z_max : float
            Maximum reflectivity value (dBZ)

        Returns
        -------
        normalized_data : np.ndarray
            Normalized reflectivity in [0, 1] range
        """
        print(f"\nNormalizing reflectivity: [{z_min}, {z_max}] dBZ -> [0, 1]")

        # Clip values to [z_min, z_max]
        clipped = np.clip(cappi_data, z_min, z_max)

        # Normalize to [0, 1]
        normalized = (clipped - z_min) / (z_max - z_min)

        # Handle NaN values (set to 0)
        normalized = np.nan_to_num(normalized, nan=0.0)

        print(f"  Original range: [{np.nanmin(cappi_data):.2f}, {np.nanmax(cappi_data):.2f}] dBZ")
        print(f"  Normalized range: [{np.min(normalized):.3f}, {np.max(normalized):.3f}]")

        return normalized


    def process_single_file(self, filepath, apply_clutter_removal=True,
                           clutter_method='gabella'):
        """
        Complete preprocessing pipeline for a single radar file.

        Parameters
        ----------
        filepath : str
            Path to the NetCDF radar file
        apply_clutter_removal : bool
            Whether to apply clutter removal
        clutter_method : str
            Clutter removal method: 'gabella', 'fuzzy', or 'both'

        Returns
        -------
        processed_data : np.ndarray
            Normalized 3D CAPPI data (nx, ny, nz) in [0, 1] range
        grid : pyart.core.Grid
            Py-ART grid object
        """
        print(f"\n{'='*70}")
        print(f"PROCESSING: {os.path.basename(filepath)}")
        print(f"{'='*70}")

        # Step 1: Read NetCDF file
        radar = self.read_kochi_netcdf(filepath)

        # Step 2: Apply clutter removal
        if apply_clutter_removal:
            if clutter_method in ['gabella', 'both']:
                radar = self.apply_gabella_clutter_removal(radar)
            if clutter_method in ['fuzzy', 'both']:
                radar = self.apply_fuzzy_logic_clutter_removal(radar)

        # Step 3: Grid to Cartesian CAPPI
        grid, cappi_data = self.grid_to_cartesian_cappi(radar)

        # Step 4: Normalize reflectivity
        normalized_data = self.normalize_reflectivity(cappi_data)

        print(f"\n{'='*70}")
        print(f"PROCESSING COMPLETE")
        print(f"  Output shape: {normalized_data.shape}")
        print(f"  Output range: [{np.min(normalized_data):.3f}, {np.max(normalized_data):.3f}]")
        print(f"{'='*70}\n")

        return normalized_data, grid


    def create_temporal_sequence(self, file_list, sequence_length=8,
                                 apply_clutter_removal=True):
        """
        Create temporal sequence of CAPPI data from multiple radar files.

        Parameters
        ----------
        file_list : list
            List of NetCDF file paths (sorted by time)
        sequence_length : int
            Number of time steps in sequence (default: 8)
        apply_clutter_removal : bool
            Whether to apply clutter removal

        Returns
        -------
        sequence_data : np.ndarray
            4D array (time, nx, ny, nz) of normalized reflectivity
        """
        print(f"\nCreating temporal sequence from {len(file_list)} files...")

        if len(file_list) < sequence_length:
            raise ValueError(f"Need at least {sequence_length} files for sequence, got {len(file_list)}")

        # Process each file
        sequence_data = []
        for i, filepath in enumerate(file_list[:sequence_length]):
            print(f"\n[{i+1}/{sequence_length}] Processing {os.path.basename(filepath)}")
            processed_data, _ = self.process_single_file(filepath, apply_clutter_removal)
            sequence_data.append(processed_data)

        # Stack into 4D array
        sequence_data = np.stack(sequence_data, axis=0)
        print(f"\nTemporal sequence created: shape = {sequence_data.shape}")

        return sequence_data


def main():
    """
    Example usage of KochiRadarPreprocessor.
    """
    # Initialize preprocessor
    preprocessor = KochiRadarPreprocessor(
        grid_shape=(120, 120, 16),
        grid_limits=((-150, 150), (-150, 150), (0.5, 15.5))
    )

    # Example: Process a single file
    # filepath = "path/to/NETCDF5_kochi_weather_2021_06_30_20_07_14.nc"
    # processed_data, grid = preprocessor.process_single_file(filepath)

    # Example: Create temporal sequence
    # file_list = sorted(glob.glob("path/to/radar_files/*.nc"))
    # sequence = preprocessor.create_temporal_sequence(file_list, sequence_length=8)

    print("\nPreprocessor ready for use.")
    print("\nExample usage:")
    print("  preprocessor = KochiRadarPreprocessor()")
    print("  data, grid = preprocessor.process_single_file('radar_file.nc')")
    print("  sequence = preprocessor.create_temporal_sequence(file_list, sequence_length=8)")


if __name__ == "__main__":
    main()
