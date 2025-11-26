"""Verification script for GUI logic."""

import sys
import logging
from pathlib import Path
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QObject

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from mmwave_radar_processing.visualization.backends.mmwave_radar_processor_controller import (
    mmWaveRadarProcessorController,
)
from mmwave_radar_processing.visualization.backends.processor_registry import (
    get_default_registry,
)
from mmwave_radar_processing.logging.logger import setup_logger, get_logger

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

def verify_logic():
    setup_logger(level=logging.DEBUG)
    logger = get_logger(__name__)
    
    app = QApplication(sys.argv)
    
    registry = get_default_registry(logger=logger)
    
    # Use dataset_params.yaml
    dataset_params_path = Path("mmwave_radar_processing/visualization/configs/dataset_params.yaml")
    processor_params_path = Path("mmwave_radar_processing/visualization/configs/processor_params.yaml")
    
    controller = mmWaveRadarProcessorController(
        registry=registry,
        logger=logger,
        dataset_params_path=dataset_params_path,
        processor_params_path=processor_params_path,
    )
    
    # Mock view update slot
    received_updates = {}
    def on_view_update(key, payload):
        logger.info("Received update for %s: %s", key, payload.keys())
        received_updates[key] = payload
        
    controller.view_update.connect(on_view_update)
    
    # Load defaults (which loads dataset and config)
    # The controller loads defaults in __init__ if paths are provided.
    # But we need to make sure it finished loading.
    # Actually, _load_defaults is called in __init__.
    
    # Wait for dataset to be loaded? It's synchronous in _load_defaults.
    
    # Process a few frames
    logger.info("Processing frames...")
    for i in range(25):
        logger.info("Processing frame %d", i)
        controller.process_next_frame(i)
        
    # Check if we received updates
    expected_processors = ["range_doppler_resp", "range_resp", "range_angle_resp", "micro_doppler_resp", "doppler_azimuth_resp"]
    
    for key in expected_processors:
        if key in received_updates:
            logger.info("PASS: Received update for %s", key)
            payload = received_updates[key]
            data = payload["data"]
            logger.info("  Data shape: %s", data.shape)
            
            # Check metadata
            if key == "range_doppler_resp":
                if "range_bins" in payload and "vel_bins" in payload:
                    logger.info("  PASS: Metadata present (range_bins, vel_bins)")
                else:
                    logger.error("  FAIL: Missing metadata for %s", key)
            elif key == "range_resp":
                if "range_bins" in payload:
                    logger.info("  PASS: Metadata present (range_bins)")
                else:
                    logger.error("  FAIL: Missing metadata for %s", key)
            elif key == "range_angle_resp":
                if "range_bins" in payload and "angle_bins" in payload:
                    logger.info("  PASS: Metadata present (range_bins, angle_bins)")
                else:
                    logger.error("  FAIL: Missing metadata for %s", key)
            elif key == "micro_doppler_resp":
                if "time_bins" in payload and "vel_bins" in payload:
                    logger.info("  PASS: Metadata present (time_bins, vel_bins)")
                else:
                    logger.error("  FAIL: Missing metadata for %s. Keys: %s", key, payload.keys())
            elif key == "doppler_azimuth_resp":
                if "angle_bins" in payload and "vel_bins" in payload:
                    logger.info("  PASS: Metadata present (angle_bins, vel_bins)")
                else:
                    logger.error("  FAIL: Missing metadata for %s", key)
        else:
            logger.error("FAIL: No update for %s", key)

if __name__ == "__main__":
    verify_logic()
