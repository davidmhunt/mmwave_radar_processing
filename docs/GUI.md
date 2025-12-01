# mmWave Radar GUI Plan

Plan for a PyQt6 + pyqtgraph GUI that fits the existing `mmwave_radar_processing` codebase and follows an MVC pattern driven by `CpslDS` (dataset) and `ConfigManager` (radar configuration).

## Scope and constraints
- Use only dependencies already in `pyproject.toml` (PyQt6, pyqtgraph, numpy, imageio, etc.).
- Support live streaming and offline playback; reuse the same controller logic for both.
- Generate videos from any view using the existing `imageio[ffmpeg]` dependency.
- Keep plotting logic in the new visualization layer; avoid mutating processor math.
- Initial pass keeps processing sequential on the UI loop where possible; threading/performance tuning is planned, not implemented in v1.
- Default dataset/config load behavior mirrors `scripts/view_radar_data.py` (e.g., dataset root `/data/RadVel`, config dir from `CONFIG_DIRECTORY`, and `6843_RadVel_ods_20Hz.cfg` with `array_geometry="ods"`, `array_direction="down"`), with CLI flags to override.
- Dataset and processor parameters load from shared YAML files (`visualization/configs/dataset_params.yaml`, `visualization/configs/processor_params.yaml`); defaults match the current `view_radar_data.py` usage where applicable.
- Logging: stdout-only via `mmwave_radar_processing/logging/logger.py`; GUI/scripts default to INFO level with CLI override. No in-GUI log panel for v1.

## MVC architecture
- **Model**
  - `CpslDS` supplies frame-wise data (ADC cubes, point clouds, ancillary sensors) and handles dataset iteration, seek, and metadata.
  - `ConfigManager` loads and exposes radar configuration (range/velocity/angle bins, antenna layout, performance metrics) for both processors and UI displays.
  - Model facade in `visualization/models/` to wrap `CpslDS` + `ConfigManager` for thread-safe access.
- **Controller**
  - `mmWaveRadarProcessorController` in `visualization/backends/mmwave_radar_processor_controller.py`.
  - Owns processor instances (one per response type) built from a registry, wires them to the models, manages playback timers, dispatches results to views, and handles commands from UI (play/pause, seek, export, config load). Processors are created and run only inside the controller; views never construct or run processors.
  - Receives an injected logger (`get_logger(__name__)`) from the launcher; uses it for lifecycle/status/error logging.
  - Provides a simple event bus (Qt signals) for view updates and status/errors.
- **Views**
  - All live in `mmwave_radar_processing/visualization/views/`.
  - Each view is a pyqtgraph widget with a `set_data(payload)` or similar API and optional controls (colormap, dB toggle, axis limits).
  - Views register themselves with the controller via the registry; controller drives updates.
  - Main window + control panel in `visualization/gui/`.

## Proposed directory layout
```
mmwave_radar_processing/
  visualization/
    backends/
      mmwave_radar_processor_controller.py
      processor_registry.py          # maps response keys -> processors/views
      data_sources.py                # dataset vs live stream adapters
    gui/
      main_window.py                 # assembles layout, menus, status bar
      control_panel.py               # dataset selection, playback controls, toggles
    models/
      dataset_model.py               # thin wrapper over CpslDS
      config_model.py                # thin wrapper over ConfigManager
    views/
      base_view.py
      range_doppler_view.py
      range_response_view.py
      range_angle_view.py
      micro_doppler_view.py
      doppler_azimuth_view.py
      (planned) altitude_view.py
      (planned) velocity_view.py
      (planned) sar_view.py
    utils/
      video_export.py                # grabs frames from a view and writes video
      threading.py                   # worker helpers (planned for later)
    configs/
        dataset_params.yaml         # dataset + config defaults
        processor_params.yaml       # processor parameters (shared across processors)
scripts/
  launch_mmwave_viewer.py            # CLI entry point (dataset-root, config-name, log-level, processor-params)
```

## Processor registry (controller-owned)
Central mapping in `processor_registry.py` that the controller uses to instantiate processors and views once, then reuse:
- Implemented in first pass:
  - `range_doppler_resp`: `RangeDopplerProcessor` + `RangeDopplerView`
  - `range_resp`: `RangeProcessor` + `RangeResponseView`
  - `range_angle_resp`: `RangeAngleProcessor` + `RangeAngleView`
  - `micro_doppler_resp`: `MicroDopplerProcessor` + `MicroDopplerView`
  - `doppler_azimuth_resp`: `DopplerAzimuthProcessor` + `DopplerAzimuthView`
- Planned (registry entries stubbed, views/classes marked TODO):
  - `altimeter`: `Altimeter` + `AltitudeView`
  - `velocity_estimator`: `VelocityEstimator` + `VelocityView`
  - `strip_map_SAR`: `StripMapSARProcessor` + `SarView`
  - `synthetic_array_beamformer`: `SyntheticArrayBeamformerProcessor` + view TBD
  - `virtual_array_reformatter`/`beamformer` variants: view TBD

Each entry includes:
- `key`, `display_name`
- `processor_cls`
- `view_cls`
- `required_inputs` (e.g., ADC cube vs point cloud)
- `output_schema` (numpy array shape, axis labels)
- `enabled` flag (false for planned items)
- Initialization + execution location: controller instantiates all enabled processors once during startup (or when registry changes) and runs them per-frame; views only render data handed off by the controller.
- `base_view.py` is the abstract parent defining the shared widget scaffolding and `set_data`/lifecycle API that all concrete views subclass.
- Parameter injection: controller applies per-processor parameters from the YAML file at construction time (with validation and fallbacks to code defaults).
- Logging hooks: controller/views/processors accept an optional logger parameter (defaulting to `get_logger(__name__)`); no module-level logger globals in classes.
- Dataset params: controller/models load dataset and default config from `dataset_params.yaml` (CpslDS init kwargs plus config name/array geometry/direction).

## Data flow
1) Source selection: dataset (`CpslDS`) or live stream adapter (same interface: `next_frame()`, `seek(frame_idx)`, `frame_rate` hint).  
2) Controller pulls raw frame, passes config + frame to the needed processors (based on which views are visible/subscribed). Processors always run inside the controller pipeline, not in views.  
3) Processors return numpy arrays; controller emits per-view signals with dict payloads that include both data and axes bins/metadata expected by each view.  
4) Views render via pyqtgraph; control panel tweaks view params (dB toggle, cmap, thresholds).  
5) Video export requests hook into the controller to capture rendered frames from a view and stream them to `imageio` writers.  
6) Errors/slowdowns surfaced through status bar/log pane.

## Main window and control panel
- Main window layout:
  - Left control panel: dataset selection and radar config selection (load/change), processor-params YAML picker (load/reload), plus toggles to enable which responses to show.
  - Right view area: stacked/tabbed views for selected responses.
  - Bottom status bar and frame slider: play/pause/step/loop controls, FPS indicator, frame index, warnings.
- Control panel actions talk to the controller; views remain passive renderers.

## Live streaming and playback
- Playback: QTimer-driven stepping over `CpslDS` frames; supports looping, jump-to-frame, and downsampling (skip stride).
- Live: minimal initial support uses the same controller loop; background workers/buffers are planned, not in v1.
- Shared interface allows switching source at runtime without recreating views.

## Video generation
- Per-view capture pipeline: controller asks a view to render to QImage/np array each frame; `video_export.py` writes via `imageio.get_writer`.
- Supports fixed-length exports from dataset playback and timed exports during live sessions.
- Minimal UI: choose view, path, fps, duration/frame range.

## Configuration + dataset handling
- Config load via `ConfigManager` (file picker or CLI arg); controller distributes config to processors and surfaces computed performance metrics in UI.
- Dataset chooser built atop `CpslDS.print_available_folders` + path selection; controller reinitializes dataset model and resets playback state.
- Model layer ensures thread-safe reads and exposes frame count, timestamps, and sensor availability so the UI can gray out unsupported views.
- Processor parameter YAML: single shared file keyed by processor registry keys. Example:
  ```yaml
  processors:
    range_angle_resp:
      num_angle_bins: 64
      rx_antennas: []          # process() kwarg
      chirp_idx: 0             # process() kwarg
    range_doppler_resp:
      rx_idx: 0                # process() kwarg
    micro_doppler_resp:
      target_ranges: [3.0, 3.7]
      num_frames_history: 20   # __init__
      rx_idx: 0                # process() kwarg
    doppler_azimuth_resp:
      num_angle_bins: 64       # __init__
      valid_angle_range: [-1.04719755, 1.04719755]  # __init__
      min_zoom_fft_vel_span: 0.1                    # __init__
      rx_antennas: []          # process() kwarg
      range_window: []         # process() kwarg
      shift_angle: true        # process() kwarg
      use_precise_fft: false   # process() kwarg
      precise_vel_range: [-0.25, 0.25]              # process() kwarg
    range_resp:
      chirp_idx: 0             # process() kwarg
  ```
  Controller loads this once, validates keys against the registry, and applies only parameters that exist in each processor's `__init__` or `process()` signature; logs warnings for unknown keys. Missing parameters fall back to processor defaults. Control panel allows swapping/reloading the YAML at runtime.
- Dataset parameter YAML: includes CpslDS kwargs and config metadata:
  ```yaml
  dataset:
    dataset_path: /data/RadVel/CPSL_RadVel_ods_20Hz_1
    radar_adc_folder: radar_0_adc
    camera_folder: camera
    ...
  config:
    name: 6843_RadVel_ods_20Hz.cfg
    array_geometry: ods
    array_direction: down
  ```
  Launcher loads this to set defaults for dataset and config if CLI flags are not provided.
- Launcher (`scripts/launch_mmwave_viewer.py`) CLI params: `--dataset-params` (path to YAML, default `visualization/configs/dataset_params.yaml`), `--dataset-path` (overrides dataset path from YAML), `--config-name` (overrides config name from YAML), `--log-level` (e.g., INFO/DEBUG), `--processor-params` (path to YAML in `configs/`, default `processor_params.yaml`); defaults pulled from the YAML. Launcher calls `setup_logger(level=parsed_level)` once, then passes loggers into controller/views/processors.

## Threading and performance (planned)
- Future: offload heavy processors to worker threads/pools and batch emits to avoid signal storms.
- Future: optional decimation for heavy processors, configurable in the control panel.
- v1: sequential processing on the controller loop with a QTimer driving playback; target correctness first.

## View layout selection
- Initial approach: right view area uses a fixed 2x3 grid (QGridLayout) to match the five initial responses; each slot has a dropdown to choose which response to render (or hide). This keeps simultaneous comparison easy and avoids complex docking.  
- Responsive grid: dynamically resizes to show only active views, up to three columns per row; adds rows as needed. Future option: dockable/tabbed panels if more flexibility is needed.

## Implementation phases
1) ~~Scaffolding: directory creation, entry script, base controller, base view, registry with implemented items.~~  
2) ~~Models: dataset/config wrappers, source adapters, basic error handling.~~  
3) ~~Views (initial five responses) with pyqtgraph widgets and simple controls.~~  
4) ~~Controller wiring: playback/live loops, processor dispatch, signal plumbing, status updates.~~  
5) ~~Planned processors/views: add stubs to registry and placeholder views; implement incrementally.~~  
6) ~~Logging integration: ensure GUI components use injected loggers; replace any `print` in GUI modules with logger calls.~~  
7) Video export utility and UI hook.
8) Threading/performance enhancements (planned after initial pass).  
9) Polish: presets for colormaps/thresholds, layout persistence, logging pane (future if desired), dB/magnitude toggle wiring.


## Current Bugs/requested changes
1. On initialization, as a backup, use the main_window.py's populate_placeholder_data function if a dataset fails to load. 
2. For views that require multiple frames, if the next requested frame is not 1 greater than the previous frame, reset the processor and reset the buffer storing a history of adc data.
- The view should still display data (even on the 0th frame), but it will just not be complete
- Additionally, the buffer storing a history of data should be cleared/reset.
3. When we hit the max frame, do the following:
- reset the slider to the min frame
- reset all processors
- reset the buffer storing a history of adc data
4. Some processors in the mmwave_radar_processing/processors/ directory take in optional parameters in their process() function (e.g.; the range_angle_resp.py). Currently, the processor_params.yaml file contains optional parameters for both the __init__() and process() functions. All of these parameters are loaded in via the mmwave_Radar_processor_controller.py file's _load_defaults() and _init_processors() functions where the flexible parameter loading has already been enabled for each processors __init__() function. However, the optional parameters for the process() function are not loaded in. To enable this in the process() function, perform the following:
- for each processor in the mmwave_radar_processing/processors/ directory, add a **kwargs parameter to the process() function allowing it to accept optional parameters
-then in the mmwave_radar_processor_controller.py file's process_next_frame() function, prior to calling the process() function, load that class's parameters (if available) and additionally pass them to the process() function (see the #TODO: in the mmwave_radar_processor_controller.py file's process_next_frame() function)