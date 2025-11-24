"""
Entry point for the mmWave radar GUI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from PyQt6.QtWidgets import QApplication

from mmwave_radar_processing.logging.logger import get_logger, setup_logger
from mmwave_radar_processing.visualization.backends.mmwave_radar_processor_controller import (
    mmWaveRadarProcessorController,
)
from mmwave_radar_processing.visualization.backends.processor_registry import (
    get_default_registry,
)
from mmwave_radar_processing.visualization.gui.main_window import MainWindow


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Launch the mmWave radar GUI.")
    parser.add_argument(
        "--dataset-params",
        type=Path,
        default=Path("mmwave_radar_processing/visualization/configs/dataset_params.yaml"),
        help="Path to dataset parameters YAML.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Path to the CPSL dataset root (overrides dataset_params).",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Radar config filename in the configs/ directory (overrides dataset_params).",
    )
    parser.add_argument(
        "--processor-params",
        type=Path,
        default=Path("mmwave_radar_processing/visualization/configs/processor_params.yaml"),
        help="Path to processor parameter YAML.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level: DEBUG, INFO, WARNING, ERROR.",
    )
    return parser.parse_args()


def main() -> None:
    """Configure logging and launch the GUI."""
    args = parse_args()
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    setup_logger(level=level)
    logger = get_logger(__name__)

    with args.dataset_params.open("r") as handle:
        dataset_params = yaml.safe_load(handle) or {}
    dataset_defaults = dataset_params.get("dataset", {})
    config_defaults = dataset_params.get("config", {})

    dataset_path = args.dataset_path or Path(dataset_defaults.get("dataset_path", "/data/RadVel"))
    config_name = args.config_name or config_defaults.get("name", "config.cfg")

    logger.info("Launching GUI with dataset: %s", dataset_path)
    logger.info("Using config: %s", config_name)
    logger.info("Processor params: %s", args.processor_params)
    app = QApplication(sys.argv)
    registry = get_default_registry(logger=logger)
    controller = mmWaveRadarProcessorController(
        registry=registry,
        logger=logger,
        dataset_params_path=args.dataset_params,
        processor_params_path=args.processor_params,
        dataset_override=dataset_path,
        config_override=config_name,
    )
    window = MainWindow(
        controller=controller,
        registry=registry,
        logger=logger,
        dataset_path=str(dataset_path),
        config_path=str(Path("configs") / config_name),
        params_path=str(args.processor_params),
    )
    controller.dataset_loaded.connect(lambda count: window.frame_slider.setMaximum(max(count - 1, 0)))
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
