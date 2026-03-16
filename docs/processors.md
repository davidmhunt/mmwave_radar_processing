# Processors Documentation

This document details the various processor classes available in the `mmwave_radar_processing` package. These processors are responsible for transforming raw ADC data into meaningful radar products such as range profiles, Doppler maps, and point clouds.

## Processor Status & Compatibility

The following table summarizes the current status of each processor, its compatibility with the visualization GUI, and planned features.

| Processor Name | GUI View | Planned / Missing Features |
| :--- | :--- | :--- |
| **RangeProcessor** | `range_response_view` | None |
| **RangeDopplerProcessor** | `range_doppler_view` | None |
| **RangeDopplerDetector2D** | `range_doppler_detector_view` | None |
| **RangeDopplerDetectorSequential** | `range_doppler_detector_view` | None |
| **RangeDopplerGroundDetector** | `range_doppler_detector_view` | Active |
| **RangeDetector** | `range_detector_view` | Active |
| **RangeAngleProcessor** | `range_angle_view` | None |
| **RangeAngleProcessorDBSEnhanced** | `range_angle_view` | None |
| **DopplerAzimuthProcessor** | `doppler_azimuth_view` | Investigate scaling factor in zoom FFT. |
| **MicroDopplerProcessor** | `micro_doppler_view` | None |
| **PointCloudGenerator** | `point_cloud_view` | None |
| **Altimeter** | `altitude_view` | None |
| **VelocityEstimator** | None | Robust filtering, Standard array elevation support, ODS point-based estimation. |
| **SyntheticArrayBeamformer** | None | Azimuth selection for interpolation, Calibration improvements. |
| **StripMapSARProcessor** | None | None |
| **VirtualArrayReformatter** | None (Helper) | None |

---

## Creating a New Processor

All processors should inherit from the base `_Processor` class defined in `_processor.py`. This ensures a consistent interface for configuration and processing.

### Steps to Implement:

1.  **Inherit**: Create a new class inheriting from `_Processor`.
2.  **Initialize**: Implement `__init__` to accept `ConfigManager` and other parameters. Call `super().__init__(config_manager)`.
    *   Initialize any child processors (e.g., specific detectors) here.
    *   Set up initial state variables (e.g., history buffers).
3.  **Configure**: Implement the `configure()` method to set up internal state (bins, windows, pre-computed tables) based on the configuration.
    *   This is called automatically by `__init__` in the base class, but can be recalled if config changes.
    *   Example: Compute frequency axes, window functions, or calibration vectors.
4.  **Process**: Implement the `process(self, adc_cube, **kwargs)` method. This is the main entry point that takes an ADC cube and returns the processed output.
    *   Arguments should match the keys in `processor_params.yaml`.
    *   Should return a NumPy array or a dictionary of arrays.
5.  **Reset (Optional)**: Implement `reset()` to clear history or internal state if necessary.
6.  **Update History (Optional)**: Use `self.update_history(estimated, ground_truth)` if you want to track performance over time (stored in `self.history_estimated` and `self.history_gt`).

### Example Template:

```python
from mmwave_radar_processing.processors._processor import _Processor
import numpy as np

class MyNewProcessor(_Processor):
    def __init__(self, config_manager, my_param=10, **kwargs):
        self.my_param = my_param
        super().__init__(config_manager)

    def configure(self):
        # Setup bins, windows, etc.
        self.range_bins = np.linspace(0, self.config_manager.range_max_m, 100)

    def process(self, adc_cube: np.ndarray, **kwargs) -> np.ndarray:
        # Perform processing
        result = np.abs(np.fft.fft(adc_cube, axis=1))
        return result
```

---

## Processor Details

### RangeProcessor
**File**: `processors/range_resp.py`

Computes the range profile (1D FFT) of the radar data. It supports both a coarse FFT for the full range and a Zoom FFT for high-resolution inspection of specific range windows.

*   **Essential Functions**:
    *   `coarse_fft(adc_cube)`: Computes the standard range FFT.
    *   `zoom_fft(adc_cube, range_start_m, range_stop_m)`: Computes a high-resolution FFT over a specific range interval.
    *   `find_peaks(...)`: Identifies significant peaks in the range response.

*   **Parameters**:
    *   `__init__`:
        *   `config_manager` (ConfigManager): Radar configuration manager.
        *   `**kwargs`: Additional keyword arguments.
    *   `process`:
        *   `adc_cube` (np.ndarray): (num rx antennas) x (num adc samples) x (num chirps) ADC cube.
        *   `chirp_idx` (int, default=0): Index of the chirp to use for processing.
        *   `**kwargs`: Additional keyword arguments.

*   **Usage**:
    ```python
    processor = RangeProcessor(config_manager)
    range_profile = processor.process(adc_cube)
    ```

### RangeDopplerProcessor
**File**: `processors/range_doppler_resp.py`

Generates a Range-Doppler map (2D FFT) to visualize targets in terms of their distance and relative velocity.

*   **Essential Functions**:
    *   `process(adc_cube)`: Computes the 2D FFT (Range vs. Doppler).

*   **Parameters**:
    *   `__init__`:
        *   `config_manager` (ConfigManager): Radar configuration manager.
        *   `**kwargs`: Additional keyword arguments.
    *   `process`:
        *   `adc_cube` (np.ndarray): (num rx antennas) x (num adc samples) x (num chirps) ADC cube.
        *   `rx_idx` (int, default=0): Antenna index to return. Use -1 for all antennas.
        *   `return_magnitude` (bool, default=True): If True, returns magnitude; otherwise complex values.
        *   `**kwargs`: Additional keyword arguments.

*   **Usage**:
    ```python
    processor = RangeDopplerProcessor(config_manager)
    rd_map = processor.process(adc_cube)
    processor = RangeDopplerProcessor(config_manager)
    rd_map = processor.process(adc_cube)
    ```

### Range-Doppler Detection Package
**Module**: `processors.range_doppler_detection`

This package contains a hierarchy of processors designed to detect targets in the Range-Doppler domain. They all inherit from the abstract base class `RangeDopplerDetector`, which itself inherits from `RangeDopplerProcessor`.

#### Class Hierarchy
*   **`RangeDopplerDetector` (Abstract Base)**:
    *   Inherits from `RangeDopplerProcessor`.
    *   Computes the standard Range-Doppler response (Raw and Magnitude).
    *   Defines the abstract `_detect()` method that subclasses must implement.
    *   Provides helper methods like `_map_detections_to_bins` to convert bin indices to physical units.
*   **`RangeDopplerDetector2D`**:
    *   Implements `_detect()` using a 2D CFAR detector (e.g., CA-CFAR, OS-CFAR 2D) applied directly to the Range-Doppler map.
*   **`RangeDopplerDetectorSequential`**:
    *   Implements `_detect()` by first applying a 1D Range CFAR to finding range bins of interest, and then applying a 1D Velocity CFAR on the Doppler slices of those bins.
*   **`RangeDopplerDetectorGroundDetector`**:
    *   specialized detector that incorporates altitude estimation (via `Altimeter`) to limit the search space to the ground surface.

#### Registry
The package includes a registry system to dynamically load and configure detectors.
*   **Registry File**: `processors/range_doppler_detection/registry.py`
*   **Usage**: Wrappers like `PointCloudGenerator` use this registry to instantiate the configured detector type string (e.g., "range_doppler_detector_2d") from the configuration file.

#### Usage
These detectors are typically used as components within larger processors like `PointCloudGenerator`, but can be used independently.

```python
from mmwave_radar_processing.processors.range_doppler_detection.range_doppler_detector_2d import RangeDopplerDetector2D

processor = RangeDopplerDetector2D(config_manager, cfar_type="ca_cfar_2d")
detections = processor.process(adc_cube)
# detections shape: (N, 2) -> [range_idx, doppler_idx]
```
**File**: `processors/range_angle_resp.py`

Computes a Range-Angle map (2D FFT), often referred to as a Range-Azimuth map. It resolves targets in range and spatial angle.

*   **Essential Functions**:
    *   `process(adc_cube)`: Computes the 2D FFT (Range vs. Angle).

*   **Parameters**:
    *   `__init__`:
        *   `config_manager` (ConfigManager): Radar configuration manager.
        *   `num_angle_bins` (int, default=64): Number of angle bins for the response.
        *   `**kwargs`: Additional keyword arguments.
    *   `process`:
        *   `adc_cube` (np.ndarray): (num rx antennas) x (num adc samples) x (num chirps) ADC cube.
        *   `chirp_idx` (int, default=0): Index of the chirp to use.
        *   `rx_antennas` (np.ndarray | list, default=[]): List of specific RX antennas to use.
        *   `**kwargs`: Additional keyword arguments.

*   **Usage**:
    ```python
    processor = RangeAngleProcessor(config_manager)
    ra_map = processor.process(adc_cube)
    ```

### RangeAngleProcessorDBSEnhanced
**File**: `processors/range_angle_resp_dbs_enhanced.py`

Extends `RangeAngleProcessor` to provide a Range-Angle map enhanced with Doppler Beam Sharpening (DBS). This significantly improves the angular resolution of the radar by leveraging the platform's motion.

*   **Essential Functions**:
    *   `process(adc_cube, velocity_ned, ...)`: Computes the 2D FFT, calculates the 3D windowed FFT in velocity, and performs Doppler Beam Sharpening if the platform velocity is sufficient.

*   **Parameters**:
    *   `__init__`:
        *   `config_manager` (ConfigManager): Radar configuration manager.
        *   `num_angle_bins_range_angle_response` (int, default=64): Number of angle bins for the standard response.
        *   `num_angle_bins_dbs_enhanced_response` (int, default=64): Number of angle bins for the enhanced response.
        *   `min_x_y_vel_dbs` (float, default=0.25): Minimum velocity required to trigger DBS.
        *   `**kwargs`: Additional keyword arguments.
    *   `process`:
        *   `adc_cube` (np.ndarray): (num rx antennas) x (num adc samples) x (num chirps) ADC cube.
        *   `velocity_ned` (np.ndarray): (3,) array containing the [north, east, down] velocity of the platform.
        *   `rx_antennas` (np.ndarray | list, default=[]): Specific RX antennas to use.
        *   `**kwargs`: Additional keyword arguments.

*   **Usage**:
    ```python
    processor = RangeAngleProcessorDBSEnhanced(config_manager, min_x_y_vel_dbs=0.5)
    ra_map_dbs = processor.process(adc_cube, velocity_ned=np.array([1.0, 0.0, 0.0]))
    ```

### DopplerAzimuthProcessor
**File**: `processors/doppler_azimuth_resp.py`

Computes the Doppler-Azimuth response. This is typically done for a specific range bin or summed across range bins. It helps in resolving targets with the same range but different velocities and angles. Supports precise (Zoom) FFT for velocity.

*   **Essential Functions**:
    *   `process(adc_cube, range_window=...)`: Computes the response, optionally filtering for a specific range window.
    *   `precise_doppler_azimuth_fft(...)`: Uses Zoom FFT for higher velocity resolution.

*   **Parameters**:
    *   `__init__`:
        *   `config_manager` (ConfigManager): Radar configuration manager.
        *   `num_angle_bins` (int, default=64): Number of angle bins.
        *   `valid_angle_range` (np.ndarray, default=[-60, 60] deg): Valid angle range in radians.
        *   `min_zoom_fft_vel_span` (float, default=0.1): Minimum velocity span for Zoom FFT.
        *   `**kwargs`: Additional keyword arguments.
    *   `process`:
        *   `adc_cube` (np.ndarray): (num rx antennas) x (num adc samples) x (num chirps) ADC cube.
        *   `rx_antennas` (np.ndarray | list, default=[]): Specific RX antennas to use.
        *   `range_window` (np.ndarray | list, default=[]): Min/Max range to filter [min, max].
        *   `shift_angle` (bool, default=True): Whether to shift the angle axis.
        *   `use_precise_fft` (bool, default=False): Use Zoom FFT for velocity.
        *   `precise_vel_range` (np.ndarray | list, default=[-0.25, 0.25]): Velocity range for Zoom FFT.
        *   `**kwargs`: Additional keyword arguments.

*   **Usage**:
    ```python
    processor = DopplerAzimuthProcessor(config_manager)
    da_map = processor.process(adc_cube, range_window=[1.0, 5.0])
    ```

### MicroDopplerProcessor
**File**: `processors/micro_doppler_resp.py`

Generates a micro-Doppler spectrogram (Velocity vs. Time) by stacking Doppler responses over multiple frames. Useful for classifying targets based on their motion signatures (e.g., walking humans vs. drones).

*   **Essential Functions**:
    *   `process(adc_cube)`: Updates the internal history and returns the current spectrogram.

*   **Parameters**:
    *   `__init__`:
        *   `config_manager` (ConfigManager): Radar configuration manager.
        *   `target_ranges` (np.ndarray | list, default=[0, 1.0]): Range window [min, max] to monitor.
        *   `num_frames_history` (int, default=20): Number of frames to keep in history.
        *   `**kwargs`: Additional keyword arguments.
    *   `process`:
        *   `adc_cube` (np.ndarray): (num rx antennas) x (num adc samples) x (num chirps) ADC cube.
        *   `rx_idx` (int, default=0): RX antenna index to use.
        *   `**kwargs`: Additional keyword arguments.

*   **Usage**:
    ```python
    processor = MicroDopplerProcessor(config_manager)
    spectrogram = processor.process(adc_cube)
    ```

### PointCloudGenerator
**File**: `processors/point_cloud_generator.py`

Implements a standard radar signal processing pipeline to generate a 3D point cloud from raw ADC data. This typically involves range processing, Doppler processing, CFAR detection, and angle estimation.

*   **Essential Functions**:
    *   `process(adc_cube)`: Executes the full processing chain and returns the point cloud.

*   **Parameters**:
    *   `__init__`:
        *   `config_manager` (ConfigManager): Radar configuration manager.
        *   `**kwargs`: Additional keyword arguments.
    *   `process`:
        *   `adc_cube` (np.ndarray): (num rx antennas) x (num adc samples) x (num chirps) ADC cube.
        *   `**kwargs`: Additional keyword arguments.

*   **Usage**:
    ```python
    processor = PointCloudGenerator(config_manager, detector_type="range_doppler_detector_2d", detector_params={...})
    point_cloud = processor.process(adc_cube)
    ```
*   **Changes to make**

*   **Implementation Notes/Plan**


### Altimeter
**File**: `processors/altimeter.py`

A specialized processor that extends `RangeProcessor` to estimate the altitude (range to ground) of a platform. It uses a coarse FFT to find the ground peak and optionally a Zoom FFT to refine the estimate.

*   **Essential Functions**:
    *   `process(adc_cube)`: Returns the estimated altitude in meters.
    *   `find_ground_peak(...)`: Logic to select the correct peak corresponding to the ground.

*   **Parameters**:
    *   `__init__`:
        *   `config_manager` (ConfigManager): Radar configuration manager.
        *   `min_altitude_m` (float): Minimum valid altitude.
        *   `zoom_search_region_m` (float): Width of search region for Zoom FFT.
        *   `altitude_search_limit_m` (float): Max search deviation from current altitude.
        *   `range_bias` (float, default=0.0): Bias correction for altitude.
        *   `**kwargs`: Additional keyword arguments.
    *   `process`:
        *   `adc_cube` (np.ndarray): (num rx antennas) x (num adc samples) x (num chirps) ADC cube.
        *   `precise_est_enabled` (bool, default=True): Enable Zoom FFT for precision.
        *   `**kwargs`: Additional keyword arguments.

*   **Usage**:
    ```python
    processor = Altimeter(config_manager, min_altitude_m=0.5, ...)
    altitude = processor.process(adc_cube)
    ```

### VelocityEstimator
**File**: `processors/velocity_estimator.py`

Extends `DopplerAzimuthProcessor` to estimate the ego-velocity of the radar platform (vx, vy, vz). It detects peaks in the Doppler-Azimuth response and uses RANSAC or least-squares regression to fit a velocity model.

*   **Essential Functions**:
    *   `process(adc_cube)`: Updates velocity estimates.
    *   `estimate_ego_vx_velocity()`: Estimates forward velocity from zero-azimuth/elevation peaks.

*   **Parameters**:
    *   `__init__`:
        *   `config_manager` (ConfigManager): Radar configuration manager.
        *   `lower_range_bound` (float): Lower range bound relative to altitude.
        *   `upper_range_bound` (float): Upper range bound relative to altitude.
        *   `precise_vel_bound` (float, default=0.25): Velocity bound for precise FFT.
        *   `valid_angle_range` (np.ndarray, default=[-70, 70] deg): Valid angle range.
        *   `peak_threshold_dB` (float, default=30.0): Threshold for peak detection.
        *   `x_measurement_only` (bool, default=False): Estimate only X velocity.
        *   `min_R2_threshold` (float, default=0.6): Min R2 score for valid estimate.
        *   `min_inlier_percent` (float, default=0.75): Min inlier percent for RANSAC.
        *   `**kwargs`: Additional keyword arguments.
    *   `process`:
        *   `adc_cube` (np.ndarray, default=empty): ADC cube data.
        *   `points` (np.ndarray, default=empty): Point cloud data (alternative input).
        *   `altitude` (float, default=0.0): Current altitude.
        *   `enable_precise_responses` (bool, default=False): Enable precise FFT responses.
        *   `**kwargs`: Additional keyword arguments.

*   **Usage**:
    ```python
    processor = VelocityEstimator(config_manager, ...)
    processor.process(adc_cube)
    vx, vy, vz = processor.current_velocity_estimate
    ```

### SyntheticArrayBeamformerProcessor
**File**: `processors/simple_synthetic_array_beamformer_processor_multiFrame.py`

Implements Synthetic Aperture Radar (SAR) beamforming by coherently combining data across multiple frames as the radar moves. This creates a large synthetic aperture for high angular resolution.

*   **Essential Functions**:
    *   `compute_synthetic_response(array_geometry)`: Computes the beamformed response on a defined 3D grid.
    *   `perform_array_calibration()`: Self-calibrates the array geometry using strong targets.

*   **Parameters**:
    *   `__init__`:
        *   `config_manager` (ConfigManager): Radar configuration manager.
        *   `receiver_idx` (int, default=0): Receiver index for synthetic array.
        *   `chirp_cfg_idx` (int, default=0): Chirp config index.
        *   `num_frames` (int, default=2): Number of frames for synthetic aperture.
        *   `stride` (int, default=1): Frame stride.
        *   `az_angle_bins_rad` (np.ndarray, default=linspace(-30, 30, 60)): Azimuth bins.
        *   `el_angle_bins_rad` (np.ndarray, default=[0]): Elevation bins.
        *   `min_vel` (np.ndarray, default=[0.17, 0, 0]): Min velocity for SAR.
        *   `max_vel` (np.ndarray, default=[0.25, 0.05, 0.05]): Max velocity for SAR.
        *   `max_vel_stdev` (np.ndarray, default=[0.1, 0.1, 0.1]): Max velocity stdev.
        *   `enable_calibration` (bool, default=False): Enable array calibration.
        *   `num_calibration_iters` (int, default=1): Calibration iterations.
        *   `interpolated_grid_resolution_m` (float, default=0.1): Output grid resolution.
        *   `**kwargs`: Additional keyword arguments.
    *   `compute_synthetic_response`:
        *   `array_geometry` (np.ndarray): Array geometry indexed by [frame, (x,y,z), chirp].

*   **Usage**:
    ```python
    processor = SyntheticArrayBeamformerProcessor(config_manager, ...)
    # Note: Requires external management of array geometry and history
    response = processor.compute_synthetic_response(geometry)
    ```

### StripMapSARProcessor
**File**: `processors/strip_map_SAR_processor.py`

Implements a Strip-Map SAR processor. It processes data assuming a linear flight path to generate a 2D image of the ground.

*   **Essential Functions**:
    *   `process(adc_cube, vel_m_per_s, ...)`: Computes the SAR image based on current velocity and height.

*   **Parameters**:
    *   `__init__`:
        *   `config_manager` (ConfigManager): Radar configuration manager.
        *   `az_angle_range_rad` (np.ndarray | list, default=[-30, 30] deg): Azimuth angle range.
        *   `**kwargs`: Additional keyword arguments.
    *   `process`:
        *   `adc_cube` (np.ndarray): ADC cube data.
        *   `vel_m_per_s` (float): Platform velocity in m/s.
        *   `sensor_height_m` (float, default=0.24): Height of sensor above ground.
        *   `rx_index` (int, default=0): RX antenna index.
        *   `max_SAR_distance` (float, default=1.5): Max distance for SAR processing.
        *   `**kwargs`: Additional keyword arguments.

*   **Usage**:
    ```python
    processor = StripMapSARProcessor(config_manager)
    sar_image = processor.process(adc_cube, vel_m_per_s=5.0)
    ```

### VirtualArrayReformatter
**File**: `processors/virtual_array_reformater.py`

A helper processor that reformats the raw ADC cube to align chirps for virtual array (MIMO) processing. It organizes data so that virtual antennas appear as additional channels.

*   **Essential Functions**:
    *   `process(adc_cube)`: Returns the reformatted ADC cube.

*   **Parameters**:
    *   `__init__`:
        *   `config_manager` (ConfigManager): Radar configuration manager.
        *   `**kwargs`: Additional keyword arguments.
    *   `process`:
        *   `adc_cube` (np.ndarray): ADC cube data.
        *   `**kwargs`: Additional keyword arguments.

*   **Usage**:
    ```python
    reformatter = VirtualArrayReformatter(config_manager)
    virtual_cube = reformatter.process(raw_adc_cube)
    ```