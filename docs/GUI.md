# mmWave Radar GUI Documentation

This document details the architecture, usage, and extension of the mmWave Radar GUI.

## 1. GUI Architecture Overview

The GUI follows a **Model-View-Controller (MVC)** pattern to ensure separation of concerns between data handling, processing logic, and visualization.

- **Model**:
    - **DatasetModel**: Wraps `CpslDS` to provide frame-wise data access (ADC cubes) and dataset metadata.
    - **ConfigModel**: Wraps `ConfigManager` to load and expose radar configuration (range/velocity/angle bins, antenna layout).
- **Controller**:
    - **mmWaveRadarProcessorController**: The central hub that coordinates data flow. It owns the models and the processor instances. It runs the processing loop (driven by a `QTimer` for playback), feeds data to processors, and emits signals to update views.
- **View**:
    - **Views**: Located in `mmwave_radar_processing/visualization/views/`. These are `pyqtgraph` widgets that receive processed data and render it. They are passive and do not contain processing logic.
    - **MainWindow**: The top-level window containing the control panel and the view grid.

## 2. Processor Registry

The processor registry (`mmwave_radar_processing/visualization/backends/processor_registry.py`) is the central configuration that maps unique keys to processor classes and their corresponding view classes.

### Processor Status Table

| Processor Key | Processor Class | View Class | Status | Comments |
| :--- | :--- | :--- | :--- | :--- |
| `range_doppler_resp` | `RangeDopplerProcessor` | `RangeDopplerView` | Active | |
| `range_resp` | `RangeProcessor` | `RangeResponseView` | Active | |
| `range_angle_resp` | `RangeAngleProcessor` | `RangeAngleView` | Active | |
| `micro_doppler_resp` | `MicroDopplerProcessor` | `MicroDopplerView` | Active | Requires history buffer |
| `doppler_azimuth_resp` | `DopplerAzimuthProcessor` | `DopplerAzimuthView` | Active | |
| `altimeter` | `Altimeter` | `AltitudeView` | Active | |
| `velocity_estimator` | `VelocityEstimator` | `VelocityView` | Planned | |
| `strip_map_SAR` | `StripMapSARProcessor` | `SarView` | Planned | |
| `range_doppler_detector_2d` | `RangeDopplerDetector2D` | `RangeDopplerDetectorView` | Active | |
| `range_doppler_detector_sequential` | `RangeDopplerDetectorSequential` | `RangeDopplerDetectorView` | Active | |
| `range_doppler_ground_detector` | `RangeDopplerGroundDetector` | `RangeDopplerDetectorView` | Active | |
| `range_detector` | `RangeDetector` | `RangeDetectorView` | Active | |
| `point_cloud_generator` | `PointCloudGenerator` | `PointCloudView` | Active | |

### Registry Description

Each entry in the registry is a `ProcessorSpec` object containing:
- `key`: Unique identifier string.
- `display_name`: Human-readable name for the UI.
- `processor_cls`: The Python class for the signal processor.
- `view_cls`: The Python class for the GUI view widget.
- `required_inputs`: Data required (e.g., "adc_cube").
- `output_schema`: Description of output format.
- `enabled`: Boolean flag to enable/disable in the GUI.
- `num_frames_history`: Number of previous frames required (e.g., for Micro-Doppler).

### Adding a New View/Processor
 
1.  **Implement Processor**: Create your processor class in `mmwave_radar_processing/processors/`. It must implement a `process(adc_cube, **kwargs)` method.
2.  **Implement View**: Create your view class in `mmwave_radar_processing/visualization/views/` inheriting from `BaseView`. Implement `update_view(payload)`.
    *   The `payload` dictionary will contain keys defined in your registry entry.
3.  **Register**: Add a new entry to the `registry` dictionary in `mmwave_radar_processing/visualization/backends/processor_registry.py` with your new classes and specifications.
    *   **Crucial Step**: Define `view_keys` in `ProcessorSpec`. These strings must match attributes of your processor instance. The `ViewController` will automatically extract these attributes after `process()` completes and pack them into the `payload` dict passed to your view.
    *   Example: `view_keys=["range_bins", "doppler_bins", "heatmap"]`. Your processor must have `self.range_bins`, `self.doppler_bins`, and `self.heatmap`.

## 3. Backend Overview

### `mmWaveRadarProcessorController`
Located in `mmwave_radar_processing/visualization/backends/mmwave_radar_processor_controller.py`.

#### 1. What it does
Acts as the central coordinator. It manages the `DatasetModel` and `ConfigModel`, handles the playback timer, and delegates processor management and execution to the `ViewController`.

#### 2. Key Functions
-   `load_dataset(dataset_path, params)`: Initializes the `DatasetModel`.
-   `load_config(config_path)`: Initializes the `ConfigModel` and triggers processor initialization via `ViewController`.
-   `process_next_frame(frame_idx)`:
    1.  Fetches ADC cube.
    2.  Updates history buffer.
    3.  Calls `view_controller.process_frame()`.
-   `start()` / `stop()`: Controls the playback `QTimer`.

### `ViewController`
Located in `mmwave_radar_processing/visualization/backends/view_controller.py`.

#### 1. What it does
Manages the lifecycle and execution of signal processors. It decouples the processing logic from the main controller.

#### 2. Key Functions
-   `initialize_processors(config_manager, processor_params)`: Instantiates processors based on the registry and configuration.
-   `process_frame(adc_cube, history_buffer, processor_params)`:
    1.  Iterates through enabled processors.
    2.  Checks history requirements.
    3.  Calls `processor.process()`.
    4.  Dynamically constructs the payload using `ProcessorSpec.view_keys`.
    5.  Emits `view_update` signal.

#### 3. Dynamic Payload Generation
The `ViewController` uses the `view_keys` list in `ProcessorSpec` to automatically extract attributes from the processor instance and include them in the payload sent to the view. This allows views to request specific metadata (e.g., `range_bins`, `vel_bins`) without hardcoding them in the controller.

## 4. GUI Windows

## 4. GUI Windows

### `MainWindow`
Located in `mmwave_radar_processing/visualization/gui/main_window.py`.

-   **Responsibility**: Top-level application window. Manages the layout of the Control Panel (left) and the Processor View Panel (right).
-   **Key Functions**:
    -   `_init_ui()`: Sets up the layout and connects signals between the controller, control panel, and view panel.

### `ProcessorViewPanel`
Located in `mmwave_radar_processing/visualization/gui/processor_view_panel.py`.

-   **Responsibility**: Manages the grid of views.
-   **Layout**: A 2x2 grid where each cell contains a dropdown menu to select the active view for that cell.
-   **Key Functions**:
    -   `_on_dropdown_changed(row, col, text)`: Handles switching views in a grid cell.
    -   `handle_view_update(key, payload)`: Updates the view widget if it is currently active in any of the grid cells.

### `ControlPanel`
Located in `mmwave_radar_processing/visualization/gui/control_panel.py`.

-   **Responsibility**: User interface for configuration.
-   **Key Functions**:
    -   `_browse_dataset()`, `_browse_config()`: File dialogs for selection.
    -   `_emit_view_toggle()`: Notifies main window when view checkboxes change.
    -   `_emit_db_mode()`: Toggles dB vs Linear scale rendering.

## 5. Processed Views

All views inherit from `BaseView` in `mmwave_radar_processing/visualization/views/base_view.py`.

### `BaseView`
-   **Responsibility**: Abstract base class providing common interface.
-   **Key Functions**:
    -   `set_data(payload)`: Public API called by MainWindow. Stores payload and calls `update_view`.
    -   `update_view(payload)`: Abstract method. Must be implemented by subclasses to render data.
    -   `set_db_mode(enabled)`: Toggles log-scale rendering and triggers a re-draw.

### Implemented Views
1.  **RangeDopplerView**: Displays Range-Doppler heatmap. Expects `data` (2D array), `range_bins`, `vel_bins`.
2.  **RangeResponseView**: Displays 1D Range profile. Expects `data` (1D array), `range_bins`.
3.  **RangeAngleView**: Displays Range-Angle heatmap. Expects `data` (2D array), `range_bins`, `angle_bins`.
4.  **MicroDopplerView**: Displays Time-Velocity spectrogram. Expects `data` (2D array), `time_bins`, `vel_bins`.
5.  **DopplerAzimuthView**: Displays Velocity-Angle heatmap. Expects `data` (2D array), `vel_bins`, `angle_bins`.
6.  **RangeDopplerDetectorView**: Displays Range-Doppler heatmap with overlaid detections. Expects `data` (2D array), `range_bins`, `vel_bins`, `dets`.
7.  **RangeDetectorView**: Displays Range profile with overlaid thresholds and detections. Expects `data` (1D array), `range_bins`, `thresholds`, `dets`.
9.  **AltitudeView**: Displays coarse FFT with overlaid estimated altitude. Expects `coarse_fft` (1D array), `range_bins`, `current_altitude_corrected_m`.

### Creating a New View
1.  Create a file `my_new_view.py` in `views/`.
2.  Class `MyNewView(BaseView)`.
3.  In `__init__`, setup your `pyqtgraph` items (PlotWidget, ImageItem, etc.).
4.  Implement `update_view(self, payload)`:
    -   Extract data: `data = payload['data']`.
    -   Handle dB conversion if `self.convert_to_db` is True.
    -   Update your graph widget (e.g., `self.image.setImage(data)`).

## 6. Configuration Files

Configuration files are located in `gui_configs/`.

### `dataset_params.yaml`
Contains default paths and settings for loading datasets.
-   **Usage**: Loaded by `launch_mmwave_viewer.py` to set initial defaults.
-   **Structure**:
    -   `dataset`: Paths to `dataset_path`, `radar_adc_folder`, etc.
    -   `config`: Default radar config file `name`, `array_geometry`, `array_direction`.

**Example `dataset_params.yaml`:**
```yaml
dataset:
  # Path to the root directory of the dataset
  dataset_path: dev_resources/CPSL_RadVel_ods_10Hz_1_sample
  # Subfolder containing ADC data
  radar_adc_folder: radar_0_adc
  # Other sensor folders (optional)
  camera_folder: camera
  lidar_folder: lidar

config:
  # Default radar configuration file to load
  name: 6843_RadVel_ods_20Hz.cfg
  # Antenna array geometry (e.g., 'ods', 'planar')
  array_geometry: ods
  # Array mounting direction
  array_direction: down
```

### `processor_params.yaml`
Contains runtime parameters for each processor.
-   **Usage**: Loaded by the Controller. Parameters are passed as `kwargs` to the `process()` method of the corresponding processor.
-   **Structure**: Keyed by processor registry key (e.g., `range_doppler_resp`).

**Example `processor_params.yaml`:**
```yaml
processors:
  range_angle_resp:
    num_angle_bins: 64
    rx_antennas: [0,3,4,7]
    chirp_idx: 0
    
  range_doppler_resp:
    rx_idx: 0
    
  micro_doppler_resp:
    target_ranges: [3.0, 3.7]
    num_frames_history: 20
    rx_idx: 0
    
  doppler_azimuth_resp:
    num_angle_bins: 64
    valid_angle_range: [-1.047, 1.047]
    min_zoom_fft_vel_span: 0.1
    rx_antennas: [4,5,8,9]
    range_window: [0.9, 2.0]
    shift_angle: false
    use_precise_fft: false
    precise_vel_range: [-0.25, 0.25]
    
  range_resp:
    chirp_idx: 0
    
  range_detector:
    cfar_type: "os_cfar_1d"
    cfar_params:
      num_train: 5
      num_guard: 3
      rho: 0.5
      alpha: 2
      
  altimeter:
    min_altitude_m: 0.5
    zoom_search_region_m: 1.0
    altitude_search_limit_m: 2.0
    range_bias: 0.0
      
  range_doppler_detector_2d:
    cfar_type: "os_cfar_2d"
    cfar_params:
      num_train: [7,7]
      num_guard: [4,4]
      rho: 0.7
      alpha: 5
      
  range_doppler_detector_sequential:
    rng_cfar_type: "os_cfar_1d"
    rng_cfar_params:
      num_train: 5
      num_guard: 3
      rho: 0.5
      alpha: 2
    vel_cfar_type: "os_cfar_1d"
    vel_cfar_params:
      num_train: 4
      num_guard: 2
      rho: 0.8
      alpha: 3
  range_doppler_ground_detector:
    vel_cfar_type: "os_cfar_1d"
    vel_cfar_params:
      num_train: 5
      num_guard: 2
      rho: 0.7
      alpha: 3
    altimeter_params:
      min_altitude_m: 25.0e-2
      zoom_search_region_m: 20.0e-2
      altitude_search_limit_m: 50e-2
      range_bias: 0.0
      precise_est_enabled: false
  point_cloud_generator:
    detector_type: "range_doppler_detector_2d"
    detector_params:
      cfar_type: "os_cfar_2d"
      cfar_params:
        num_train: [7,7]
        num_guard: [4,4]
        rho: 0.7
        alpha: 5
    az_antenna_idxs: [0,3,4,7]
    el_antenna_idxs: [9,8,5,4]
    shift_az_resp: true
    shift_el_resp: false
```

## 8. Video Export

The GUI includes a pipeline to export the dataset visualization as an MP4 movie. This section describes the architecture and implementation details.

### 8.1. Architecture

The video export is orchestrated by a dedicated class that interfaces with the main controller to drive the playback and capture frames.

*   **UI Entry Point** (`mmwave_radar_processing/visualization/gui/control_panel.py`):
    *   **"Export Dataset Movie" Button**: Located in the Dataset control group.
    *   **Signal**: Emits `export_movie_requested(str)` with the target file path.
    *   **Logic**: Opens a `QFileDialog` to select the destination, ensuring the `.mp4` extension.

*   **Controller Integration** (`mmwave_radar_processing/visualization/backends/mmwave_radar_processor_controller.py`):
    *   **`export_movie(output_path, target_widget)`**: The main entry point in the controller.
    *   It instantiates the `VideoExporter` with the controller itself (access to data/logic) and the target capture widget (the `MainWindow`).

*   **Video Exporter** (`mmwave_radar_processing/visualization/backends/video_exporter.py`):
    *   **Purpose**: A standalone class to keeping the export logic separate from the main controller.
    *   **Dependencies**: Uses `imageio` with the `libx264` codec. requires `ffmpeg` installed on the system.

### 8.2. Export Loop Logic

The `VideoExporter.export()` method follows this sequence:

1.  **Preparation**:
    *   Validates the output path.
    *   Saves the current frame index (`controller.last_processed_frame`) to restore it later.
    *   Initializes the `imageio` writer.

2.  **Frame Iteration**:
    *   Loops from `0` to `total_frames`.
    *   **Update State**: Calls `controller.process_next_frame(i)` to update all internal data models and view logic.
    *   **Flush Events**: Calls `QApplication.processEvents()` to ensure the Qt event loop processes the paint events, updating the visual widgets. **This is critical**; without it, the captured frames would be stale or incomplete.
    *   **Capture**: Calls `target_widget.grab()` (a Qt method) to render the widget into a `QPixmap`.
    *   **Convert**: Converts the pixmap to a `QImage`, ensures `RGBA8888` format, and extracts the raw bytes into a NumPy array (discarding the alpha channel if not needed).
    *   **Write**: Appends the NumPy array to the video writer.

3.  **Cleanup**:
    *   Closes the video writer.
    *   Restores the timeline to the original frame index so the user doesn't lose their place.

### 8.3. Implementation Notes for New Projects

If implementing this in another repository:
1.  **Decouple**: Create a separate `VideoExporter` class. Do not put the export loop inside your main window or controller class to avoid stiffle code.
2.  **Re-use Logic**: Your exporter should drive the *existing* frame update logic (e.g., `controller.process_next_frame`). Do not duplicate data loading or processing logic inside the exporter.
3.  **Process Events**: Always call `QApplication.processEvents()` after triggering a frame update and before grabbing the screen.
4.  **Target Widget**: Capture the top-level window if you want the full context (sliders, control panels), or a specific sub-widget (e.g., just the plot area) if you want a cleaner video.

## 9. Verification & Tests

 The GUI logic and views are tested in `tests/verify_gui_logic.py`. These tests verify that views correctly handle data payloads and render without crashing.

 *   **`test_range_angle_view`**: Verifies 2D heatmap rendering and dB scaling for Range-Angle.
 *   **`test_micro_doppler_view`**: Verifies spectrogram rendering.
 *   **`test_doppler_azimuth_view`**: Verifies Doppler-Azimuth heatmap.
 *   **`test_range_doppler_view`**: Verifies Range-Doppler heatmap.
 *   **`test_range_response_view`**: Verifies 1D plot rendering.
 *   **`test_range_doppler_detector_2d_view`**: Verifies heatmap + scatter plot for detections.
 *   **`test_range_detector_view`**: Verifies signal plot + threshold line + detection scatter.
 *   **`test_point_cloud_view`**: Verifies 3D scatter plot setup and axis rendering.
 *   **`test_altitude_view`**: Verifies altitude line overlay on range profile.
 *   **`test_processor_view_panel_caching`**: Verifies that the UI efficiently caches data for hidden views and updates them only when revealed.

 ## 7. Launch Scripts

### `launch_mmwave_viewer.py`
Located in `scripts/launch_mmwave_viewer.py`.

-   **Usage**: Entry point for the application.
-   **Command**:
    ```bash
    poetry run python scripts/launch_mmwave_viewer.py [arguments]
    ```
-   **Arguments**:
    -   `--dataset-path`: Override default dataset path.
    -   `--config-name`: Override default config file.
    -   `--dataset-params`: Path to custom dataset params YAML.
    -   `--processor-params`: Path to custom processor params YAML.
    -   `--log-level`: Set logging verbosity (INFO, DEBUG).

---

## Implementation phases
1. ~~Scaffolding: directory creation, entry script, base controller, base view, registry with implemented items.~~
2. ~~Models: dataset/config wrappers, source adapters, basic error handling.~~
3. ~~Views (initial five responses) with pyqtgraph widgets and simple controls.~~
4. ~~Controller wiring: playback/live loops, processor dispatch, signal plumbing, status updates.~~
5. ~~Planned processors/views: add stubs to registry and placeholder views; implement incrementally.~~
6. ~~Logging integration: ensure GUI components use injected loggers; replace any `print` in GUI modules with logger calls.~~
7. Video export utility and UI hook.
8. Threading/performance enhancements (planned after initial pass).
9. Polish: presets for colormaps/thresholds, layout persistence, logging pane (future if desired), dB/magnitude toggle wiring.


## Current Bugs/requested changes