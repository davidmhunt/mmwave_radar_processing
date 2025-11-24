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
        except Exception as exc:
            self.logger.error("Failed to load config: %s", exc)

    def load_processor_params(self, params: Dict[str, Any]) -> None:
        """Apply processor parameters.

        Args:
            params: Processor parameter mapping loaded from YAML.
        """
        self.logger.info("Applying processor params for keys: %s", list(params.keys()))
        self.processor_params = params

    def start(self) -> None:
        """Start playback or live streaming."""
        self.logger.info("Controller start requested")

    def stop(self) -> None:
        """Stop playback or live streaming."""
        self.logger.info("Controller stop requested")
