"""Controller for coordinating dataset, processors, and views."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from mmwave_radar_processing.logging.logger import get_logger
from mmwave_radar_processing.visualization.backends.processor_registry import (
    ProcessorSpec,
)
from mmwave_radar_processing.visualization.models.config_model import ConfigModel
from mmwave_radar_processing.visualization.models.dataset_model import DatasetModel


class mmWaveRadarProcessorController(QObject):
    """Controller connecting models, processors, and views."""

    view_update = pyqtSignal(str, object)
    dataset_loaded = pyqtSignal(int)
    frame_processed = pyqtSignal(int)

    def __init__(
        self,
        parent: Optional[QObject] = None,
        registry: Optional[Dict[str, ProcessorSpec]] = None,
        logger=None,
        dataset_params_path: Optional[Path] = None,
        processor_params_path: Optional[Path] = None,
        dataset_override: Optional[Path] = None,
        config_override: Optional[str] = None,
    ) -> None:
        """Initialize the controller.

        Args:
            parent: Optional Qt parent.
            registry: Optional processor registry mapping keys to specs.
            logger: Optional logger instance; defaults to namespaced logger.
            dataset_params_path: Path to dataset params YAML.
            processor_params_path: Path to processor params YAML.
            dataset_override: Optional dataset path override.
            config_override: Optional config filename override.
        """
        super().__init__(parent)
        self.logger = logger or get_logger(__name__)
        self.registry = registry or {}
        self.dataset_params_path = dataset_params_path
        self.processor_params_path = processor_params_path
        self.dataset_override = dataset_override
        self.config_override = config_override
        self.dataset_model: Optional[DatasetModel] = None
        self.config_model: Optional[ConfigModel] = None
        self.processor_params: Dict[str, Any] = {}
        self.virtual_array_reformatter: Optional[Any] = None
        self.processors: Dict[str, Any] = {}
        self.adc_buffer: Optional[Any] = None
        self.last_processed_frame: int = -1
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer_timeout)

        self.logger.debug(
            "Controller initialized with registry keys: %s", list(self.registry.keys())
        )

        # Define REPO_ROOT relative to this file
        # This file is in mmwave_radar_processing/visualization/backends/
        # So root is 3 levels up
        self.repo_root = Path(__file__).resolve().parents[3]

        if self.dataset_params_path:
            # If path is relative, try to find it in gui_configs if not found as is
            if not Path(self.dataset_params_path).exists():
                potential_path = self.repo_root / "gui_configs" / self.dataset_params_path
                if potential_path.exists():
                    self.dataset_params_path = potential_path

        if self.dataset_params_path and Path(self.dataset_params_path).exists():
            self._load_defaults()

    def _load_defaults(self) -> None:
        """Load default dataset/config and processor params."""
        self.logger.info("Loading default parameters")
        
        if self.processor_params_path:
             if not Path(self.processor_params_path).exists():
                potential_path = self.repo_root / "gui_configs" / self.processor_params_path
                if potential_path.exists():
                    self.processor_params_path = potential_path
        
                    

        if self.processor_params_path and Path(self.processor_params_path).exists():
            with Path(self.processor_params_path).open("r") as handle:
                self.processor_params = yaml.safe_load(handle) or {}
                

        # Dataset/config params
        with Path(self.dataset_params_path).open("r") as handle:
            dataset_cfg = yaml.safe_load(handle) or {}
        dataset_path = self.dataset_override or Path(
            dataset_cfg.get("dataset", {}).get("dataset_path", "")
        )
        config_name = self.config_override or dataset_cfg.get("config", {}).get(
            "name", ""
        )
        array_geometry = dataset_cfg.get("config", {}).get("array_geometry", "ods")
        array_direction = dataset_cfg.get("config", {}).get("array_direction", "down")

        self.load_dataset(str(dataset_path), dataset_cfg)
        
        # Config path handling
        config_path = Path("configs") / config_name
        if not config_path.is_absolute():
             config_path = self.repo_root / config_path
             
        self.load_config(str(config_path), array_geometry, array_direction)

    def load_dataset(self, dataset_path: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Load a dataset source.

        Args:
            dataset_path: Root path to the dataset.
            params: Optional dataset parameter mapping.
        """
        self.logger.info("Requested dataset load: %s", dataset_path)
        
        # Handle relative paths
        dpath = Path(dataset_path)
        if not dpath.is_absolute():
            # If it starts with ~, expand user
            if str(dataset_path).startswith("~"):
                dpath = dpath.expanduser()
            else:
                # Otherwise assume relative to repo root
                dpath = self.repo_root / dpath
        
        try:
            if params is None and self.dataset_params_path:
                with Path(self.dataset_params_path).open("r") as handle:
                    params = yaml.safe_load(handle) or {}
            self.dataset_model = DatasetModel(
                params=params or {},
                logger=self.logger,
                dataset_path_override=dpath,
            )
            frame_count = self.dataset_model.frame_count()
            self.logger.info("Dataset loaded: %s", dataset_path)
            self.dataset_loaded.emit(frame_count)
            # Reset processing state
            self.last_processed_frame = -1
            if self.adc_buffer:
                self.adc_buffer.clear()
        except Exception as exc:
            self.logger.error("Failed to load dataset: %s", exc)

    def start(self) -> None:
        """Start playback or live streaming."""
        self.logger.info("Controller start requested")
        if self.dataset_model and self.dataset_model.frame_count() > 0:
            self.timer.start(50)  # 20 FPS

    def stop(self) -> None:
        """Stop playback or live streaming."""
        self.logger.info("Controller stop requested")
        self.timer.stop()

    def _on_timer_timeout(self) -> None:
        """Handle playback timer timeout."""
        if not self.dataset_model:
            self.stop()
            return

        next_frame = self.last_processed_frame + 1
        if next_frame >= self.dataset_model.frame_count():
            self.logger.info("End of dataset reached")
            self.stop()
            # Reset to start
            self.last_processed_frame = -1
            if self.adc_buffer:
                self.adc_buffer.clear()
            for processor in self.processors.values():
                if hasattr(processor, "reset"):
                    processor.reset()
            # Emit frame 0 to reset slider and view
            self.process_next_frame(0)
            return

        self.process_next_frame(next_frame)

    def process_next_frame(self, frame_idx: int) -> None:
        """Process the next frame and update views.

        Args:
            frame_idx: Index of the frame to process.
        """
        if self.dataset_model is None or self.config_model is None:
            self.logger.warning("Dataset or config not loaded, cannot process frame")
            return

        try:
            # Check for seek/discontinuity
            if frame_idx != self.last_processed_frame + 1:
                self.logger.debug("Seek detected: %d -> %d. Resetting buffer.", self.last_processed_frame, frame_idx)
                if self.adc_buffer:
                    self.adc_buffer.clear()
                # Reset processors if they have state
                for processor in self.processors.values():
                    if hasattr(processor, "reset"):
                        processor.reset()
            
            self.last_processed_frame = frame_idx

            # 1. Fetch raw ADC data
            adc_cube = self.dataset_model.get_adc_data(frame_idx)
            if adc_cube is None:
                self.logger.warning("No ADC data for frame %d", frame_idx)
                return

            # 2. Reformat using VirtualArrayReformatter
            if self.virtual_array_reformatter:
                adc_cube = self.virtual_array_reformatter.process(adc_cube)

            # 3. Update buffer
            if self.adc_buffer is not None:
                self.adc_buffer.append(adc_cube)

            # 4. Run active processors
            for key, spec in self.registry.items():
                if not spec.enabled or key not in self.processors:
                    continue

                processor = self.processors[key]
                
                # Load params for this processor
                params = self.processor_params.get("processors", {}).get(key, {})
                self.logger.debug("Processor %s params: %s", key, params)
                try:
                    # Determine input data based on history requirement
                    if spec.num_frames_history > 1:
                        if len(self.adc_buffer) < spec.num_frames_history:
                            # Not enough history yet
                            continue
                        result = processor.process(adc_cube=adc_cube, **params)
                    else:
                        result = processor.process(adc_cube=adc_cube, **params)

                    # 5. Emit update
                    # Construct payload matching view's set_data expectation
                    payload = {"data": result}
                    
                    # Add metadata if available
                    if hasattr(processor, "range_bins") and processor.range_bins is not None:
                        payload["range_bins"] = processor.range_bins
                    
                    if hasattr(processor, "vel_bins") and processor.vel_bins is not None:
                        # Special handling for DopplerAzimuthProcessor which might use zoomed bins
                        if key == "doppler_azimuth_resp" and hasattr(processor, "zoomed_vel_bins") \
                           and processor.zoomed_vel_bins is not None and processor.zoomed_vel_bins.size > 0:
                             # Check if precise FFT was used. 
                             # We can check if the output shape matches zoomed_vel_bins
                             # result shape is [vel, angle]
                             if result.shape[0] == processor.zoomed_vel_bins.size:
                                 payload["vel_bins"] = processor.zoomed_vel_bins
                             else:
                                 payload["vel_bins"] = processor.vel_bins
                        else:
                            payload["vel_bins"] = processor.vel_bins

                    if hasattr(processor, "angle_bins") and processor.angle_bins is not None:
                        # Special handling for DopplerAzimuthProcessor which filters angles
                        if key == "doppler_azimuth_resp" and hasattr(processor, "valid_angle_bins"):
                             payload["angle_bins"] = processor.valid_angle_bins
                        else:
                            payload["angle_bins"] = processor.angle_bins
                            
                    if hasattr(processor, "time_bins") and processor.time_bins is not None:
                        payload["time_bins"] = processor.time_bins
                    
                    self.view_update.emit(key, payload)

                except Exception as exc:
                    self.logger.error("Error processing %s: %s", key, exc)
            
            self.frame_processed.emit(frame_idx)

        except Exception as exc:
            self.logger.error("Error in process_next_frame: %s", exc)

    def _init_processors(self) -> None:
        """Initialize processors based on registry and config."""
        if not self.config_model or not self.config_model.config_manager:
            self.logger.warning("Config not loaded, cannot init processors")
            return

        from collections import deque
        from mmwave_radar_processing.processors.virtual_array_reformater import (
            VirtualArrayReformatter,
        )

        self.virtual_array_reformatter = VirtualArrayReformatter(
            self.config_model.config_manager
        )
        self.virtual_array_reformatter.configure()

        self.processors = {}
        max_history = 1

        for key, spec in self.registry.items():
            if not spec.enabled:
                continue
            
            try:
                # Instantiate processor
                # We assume all processors take config_manager as first arg
                # and accept kwargs from processor_params
                
                params = self.processor_params.get("processors", {}).get(key, {})
                
                processor = spec.processor_cls(
                    self.config_model.config_manager, **params
                )
                self.processors[key] = processor
                
                if spec.num_frames_history > max_history:
                    max_history = spec.num_frames_history
                    
            except Exception as exc:
                self.logger.error("Failed to init processor %s: %s", key, exc)

        self.max_history = max_history
        self.adc_buffer = deque(maxlen=max_history)
        self.logger.info("Processors initialized. Max history: %d", max_history)
        
        # Show initial frame
        self.process_next_frame(0)

    def load_config(
        self, config_path: str, array_geometry: str = "ods", array_direction: str = "down"
    ) -> None:
        """Load a radar configuration file.

        Args:
            config_path: Path to the radar configuration file.
            array_geometry: Array geometry setting.
            array_direction: Array direction setting.
        """
        self.logger.info("Requested config load: %s", config_path)
        try:
            self.config_model = ConfigModel(logger=self.logger)
            self.config_model.load(
                config_path, array_geometry=array_geometry, array_direction=array_direction
            )
            self.logger.info("Config loaded: %s", config_path)
            
            # Initialize processors after config load
            self._init_processors()
            
        except Exception as exc:
            self.logger.error("Failed to load config: %s", exc)
