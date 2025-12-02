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
| `altimeter` | `Altimeter` | `AltitudeView` | Planned | |
| `velocity_estimator` | `VelocityEstimator` | `VelocityView` | Planned | |
| `strip_map_SAR` | `StripMapSARProcessor` | `SarView` | Planned | |

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
3.  **Register**: Add a new entry to the `registry` dictionary in `mmwave_radar_processing/visualization/backends/processor_registry.py` with your new classes and specifications.

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

### `MainWindow`
Located in `mmwave_radar_processing/visualization/gui/main_window.py`.

-   **Responsibility**: Top-level application window. Manages the layout of the Control Panel (left) and View Grid (right).
-   **Key Functions**:
    -   `_init_ui()`: Sets up the grid layout and connects signals.
    -   `_handle_view_update(key, payload)`: Routes data from the controller to the correct `BaseView` widget.
    -   `_handle_view_toggle(states)`: Shows/hides views based on checkboxes.

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
```

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