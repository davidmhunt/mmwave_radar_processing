"""Controller for coordinating dataset, processors, and views."""

from __future__ import annotations

from typing import Any, Dict, Optional

from PyQt6.QtCore import QObject, pyqtSignal

from mmwave_radar_processing.logging.logger import get_logger
from mmwave_radar_processing.visualization.backends.processor_registry import (
    ProcessorSpec,
)


class mmWaveRadarProcessorController(QObject):
    """Controller connecting models, processors, and views."""

    view_update = pyqtSignal(str, object)

    def __init__(
        self,
        parent: Optional[QObject] = None,
        registry: Optional[Dict[str, ProcessorSpec]] = None,
        logger=None,
    ) -> None:
        """Initialize the controller.

        Args:
            parent: Optional Qt parent.
            registry: Optional processor registry mapping keys to specs.
            logger: Optional logger instance; defaults to namespaced logger.
        """
        super().__init__(parent)
        self.logger = logger or get_logger(__name__)
        self.registry = registry or {}
        self.logger.debug(
            "Controller initialized with registry keys: %s", list(self.registry.keys())
        )

    def load_dataset(self, dataset_path: str) -> None:
        """Load a dataset source.

        Args:
            dataset_path: Root path to the dataset.
        """
        self.logger.info("Requested dataset load: %s", dataset_path)

    def load_config(self, config_path: str) -> None:
        """Load a radar configuration file.

        Args:
            config_path: Path to the radar configuration file.
        """
        self.logger.info("Requested config load: %s", config_path)

    def load_processor_params(self, params: Dict[str, Any]) -> None:
        """Apply processor parameters.

        Args:
            params: Processor parameter mapping loaded from YAML.
        """
        self.logger.info("Applying processor params for keys: %s", list(params.keys()))

    def start(self) -> None:
        """Start playback or live streaming."""
        self.logger.info("Controller start requested")

    def stop(self) -> None:
        """Stop playback or live streaming."""
        self.logger.info("Controller stop requested")

