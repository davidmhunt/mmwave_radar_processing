"""Script for generating static compilation figures from mmWave radar data.

This script operates headlessly, utilizing the same processing pipeline as the
GUI but outputting Matplotlib-based PNG figures instead of interactive views.
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List, Set
from tqdm import tqdm

from PyQt6.QtWidgets import QApplication

from mmwave_radar_processing.logging.logger import setup_logger, get_logger
from mmwave_radar_processing.visualization.backends.mmwave_radar_processor_controller import (
    mmWaveRadarProcessorController,
)
from mmwave_radar_processing.visualization.backends.processor_registry import (
    get_default_registry,
)
from mmwave_radar_processing.plotting.plotter_gui_views import PlotterGUIViews


def parse_args() -> argparse.Namespace:
    """Parses command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Generate static radar figures.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the figure generation YAML configuration file.",
    )
    return parser.parse_args()


def load_config(config_name: str) -> Dict[str, Any]:
    """Loads the YAML configuration file from the gui_configs directory.

    Args:
        config_name: Name of the configuration file.

    Returns:
        Dict[str, Any]: The loaded configuration.
    """
    script_dir = Path(__file__).parent.resolve()
    config_path = script_dir.parent / "gui_configs" / config_name
    
    if not config_path.exists():
        # Try as a direct path if not found in gui_configs
        config_path = Path(config_name)
        
    if not config_path.exists():
        print(f"Error: Config file {config_name} not found in gui_configs or as direct path.")
        sys.exit(1)
        
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Main execution function for generating the compilation figure."""
    args = parse_args()
    config = load_config(args.config)

    setup_logger(level="INFO")
    logger = get_logger(__name__)

    # Extract parameters from config
    dataset_params_path = config.get("dataset_params")
    processor_params_path = config.get("processor_params")
    dataset_path_override = config.get("dataset_path")
    config_name_override = config.get("config_name")
    frame_to_run = config.get("frame_to_run", 0)
    display_in_dB = config.get("display_in_dB", False)
    processors_to_display = config.get("processors_to_display", [])
    output_filename = config.get("output_filename", "compilation.png")

    # Initialize Qt Application (headless)
    app = QApplication(sys.argv)

    # Initialize Controller
    registry = get_default_registry(logger=logger)
    controller = mmWaveRadarProcessorController(
        registry=registry,
        logger=logger,
        dataset_params_path=Path(dataset_params_path) if dataset_params_path else None,
        processor_params_path=Path(processor_params_path) if processor_params_path else None,
        dataset_override=Path(dataset_path_override) if dataset_path_override else None,
        config_override=config_name_override,
    )

    # Dictionary to store collected payloads
    collected_payloads = {}

    def on_view_update(key: str, payload: Any) -> None:
        """Callback for view updates from the controller.

        Args:
            key: The processor key.
            payload: The data payload.
        """
        collected_payloads[key] = payload

    # Resolve view names and keys to collect
    view_types = []
    view_name_map = {}
    for proc_key in processors_to_display:
        spec = registry.get(proc_key)
        if spec and spec.view_cls:
            view_types.append(proc_key)
            view_name_map[proc_key] = spec.view_cls.__name__
        else:
            logger.warning("Processor '%s' not found in registry or has no associated view.", proc_key)

    # Run to the specific frame sequentially to build history
    logger.info("Processing frames up to %d...", frame_to_run)
    # Process all frames up to the frame BEFORE the desired one
    for i in tqdm(range(frame_to_run), desc="Building History"):
        controller.process_next_frame(i)

    # Connect to the view_update signal only for the final frame to collect data
    controller.view_update.connect(on_view_update)
    logger.info("Processing target frame %d...", frame_to_run)
    controller.process_next_frame(frame_to_run)

    # Generate the figure
    logger.info("Generating figure with resolved views: %s", view_types)
    plotter = PlotterGUIViews()
    
    # Determine output path (relative to the script's directory)
    script_dir = Path(__file__).parent.resolve()
    results_dir = script_dir / "results"
    
    if not results_dir.exists():
        logger.info("Creating directory %s", results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = results_dir / output_filename

    plotter.plot_compilation(
        payloads=collected_payloads,
        view_types=view_types,
        convert_to_dB=display_in_dB,
        output_path=str(output_path),
        view_name_map=view_name_map
    )

    logger.info("Figure saved to %s", output_path)


if __name__ == "__main__":
    main()
