"""Controller for coordinating dataset, processors, and views."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from PyQt6.QtCore import QObject, pyqtSignal

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

        self.logger.debug(
            "Controller initialized with registry keys: %s", list(self.registry.keys())
        )

        if self.dataset_params_path and Path(self.dataset_params_path).exists():
            self._load_defaults()

    def _load_defaults(self) -> None:
        """Load default dataset/config and processor params."""
        self.logger.info("Loading default parameters")
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
        config_path = Path("configs") / config_name
        self.load_config(str(config_path), array_geometry, array_direction)

    def load_dataset(self, dataset_path: str, params: Optional[Dict[str, Any]] = None) -> None:
        """Load a dataset source.

        Args:
            dataset_path: Root path to the dataset.
            params: Optional dataset parameter mapping.
        """
        self.logger.info("Requested dataset load: %s", dataset_path)
        try:
            if params is None and self.dataset_params_path:
                with Path(self.dataset_params_path).open("r") as handle:
                    params = yaml.safe_load(handle) or {}
            self.dataset_model = DatasetModel(
                params=params or {},
                logger=self.logger,
                dataset_path_override=Path(dataset_path),
            )
            frame_count = self.dataset_model.frame_count()
            self.logger.info("Dataset loaded: %s", dataset_path)
            self.dataset_loaded.emit(frame_count)
        except Exception as exc:
            self.logger.error("Failed to load dataset: %s", exc)

    def start(self) -> None:
        """Start playback or live streaming."""
        self.logger.info("Controller start requested")

    def stop(self) -> None:
        """Stop playback or live streaming."""
        self.logger.info("Controller stop requested")

    def process_next_frame(self, frame_idx: int) -> None:
        """Process the next frame and update views.

        Args:
            frame_idx: Index of the frame to process.
        """
        if self.dataset_model is None or self.config_model is None:
            self.logger.warning("Dataset or config not loaded, cannot process frame")
            return

        try:
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
                try:
                    # Determine input data based on history requirement
                    if spec.num_frames_history > 1:
                        if len(self.adc_buffer) < spec.num_frames_history:
                            # Not enough history yet
                            continue
                        # Pass list of cubes or stacked array depending on processor expectation
                        # MicroDopplerProcessor expects a sequence of cubes if it handles history internally?
                        # Wait, checking view_radar_data.py:
                        # It loops and calls process() for each frame.
                        # But MicroDopplerProcessor.__init__ takes num_frames_history.
                        # And process() takes a single adc_cube.
                        # It seems MicroDopplerProcessor maintains its own history?
                        # Let's check MicroDopplerProcessor.
                        
                        # If the processor maintains its own history, we just pass the current frame.
                        # But the user said: "Note that the micro-doppler view requires a certain amount of previous frame's worth of ADC data... Modify the plan to take this into account in an efficient way"
                        # And I added buffering to the plan.
                        # If I pass the buffer, the processor needs to accept it.
                        # Let's assume for now we pass the current frame and the processor handles history, 
                        # OR we need to pass history.
                        # Re-reading view_radar_data.py:
                        # It loops: for i in range(history): process(adc_cube)
                        # This implies we need to feed it sequentially.
                        # Since we are in a loop here (frame by frame), we just feed it the current frame.
                        # BUT, if we jump to a frame, we need to feed it previous frames.
                        # THAT is why we need the buffer! To "prime" the processor if we seek.
                        # However, for sequential playback, just passing the current frame is enough IF the processor keeps state.
                        
                        # Let's stick to the plan: "Pass the required data (single frame or history) to active processors"
                        # If the processor expects a single cube, we pass a single cube.
                        # If we need to prime it, we should do that on seek.
                        # For now, let's pass the current cube.
                        result = processor.process(adc_cube=adc_cube)
                    else:
                        result = processor.process(adc_cube=adc_cube)

                    # 5. Emit update
                    # Construct payload matching view's set_data expectation
                    payload = {"data": result}
                    
                    # Add metadata if available (e.g. axes)
                    # This depends on the processor. 
                    # For RangeDoppler, we need range_bins and vel_bins.
                    # These are usually available in the processor or config.
                    if hasattr(processor, "range_bins"):
                        payload["range_bins"] = processor.range_bins
                    if hasattr(processor, "velocity_bins"):
                        payload["vel_bins"] = processor.velocity_bins
                    if hasattr(processor, "angle_bins"):
                        payload["angle_bins"] = processor.angle_bins
                    
                    self.view_update.emit(key, payload)

                except Exception as exc:
                    self.logger.error("Error processing %s: %s", key, exc)

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
                params = self.processor_params.get(key, {})
                
                # Filter params to match __init__ signature?
                # For now, just pass them and hope for the best or rely on **kwargs if implemented
                # view_radar_data.py shows explicit args.
                # We might need a smarter factory if signatures vary wildly.
                # But let's try passing config_manager and **params.
                
                # Special handling for known signatures if needed, or rely on params matching
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
