"""Config model wrapper for GUI access."""

from __future__ import annotations

from typing import Optional

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.logging.logger import get_logger


class ConfigModel:
    """Thin wrapper around ConfigManager for GUI use."""

    def __init__(self, logger=None) -> None:
        """Initialize the config model.

        Args:
            logger: Optional logger instance.
        """
        self.logger = logger or get_logger(__name__)
        self.config_manager = ConfigManager()

    def load(
        self,
        config_path: str,
        array_geometry: str = "ods",
        array_direction: str = "down",
    ) -> None:
        """Load a radar configuration.

        Args:
            config_path: Path to the configuration file.
            array_geometry: Array geometry setting.
            array_direction: Array direction setting.
        """
        self.logger.info("Loading config from %s", config_path)
        self.config_manager.load_cfg(
            config_path,
            array_geometry=array_geometry,
            array_direction=array_direction,
        )
        self.config_manager.compute_radar_perforance(profile_idx=0)
        self.config_manager.print_cfg_overview()

