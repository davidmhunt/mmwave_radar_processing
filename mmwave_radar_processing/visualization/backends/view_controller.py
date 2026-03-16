"""Controller for managing processors and views."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from mmwave_radar_processing.logging.logger import get_logger
from mmwave_radar_processing.visualization.backends.processor_registry import (
    ProcessorSpec,
)


class ViewController(QObject):
    """Controller for managing processors and views."""

    view_update = pyqtSignal(str, object)

    def __init__(
        self,
        registry: Dict[str, ProcessorSpec],
        parent: Optional[QObject] = None,
        logger=None,
    ) -> None:
        """Initialize the view controller.

        Args:
            registry: Mapping of processor keys to specs.
            parent: Optional Qt parent.
            logger: Optional logger instance.
        """
        super().__init__(parent)
        self.logger = logger or get_logger(__name__)
        self.registry = registry
        self.processors: Dict[str, Any] = {}

    def initialize_processors(
        self, config_manager: Any, processor_params: Dict[str, Any]
    ) -> None:
        """Initialize processors based on registry and config.

        Args:
            config_manager: Configuration manager instance.
            processor_params: Dictionary of processor parameters.
        """
        self.processors = {}
        
        for key, spec in self.registry.items():
            if not spec.enabled:
                continue
            
            try:
                # Instantiate processor
                params = processor_params.get("processors", {}).get(key, {})
                
                processor = spec.processor_cls(
                    config_manager, **params
                )
                self.processors[key] = processor
                
            except Exception as exc:
                self.logger.error("Failed to init processor %s: %s", key, exc)

        self.logger.info("Processors initialized: %s", list(self.processors.keys()))

    def process_frame(
        self, adc_cube: Any, history_buffer: Any, processor_params: Dict[str, Any], velocity_ned: Optional[np.ndarray] = None
    ) -> None:
        """Process a frame using all active processors.

        Args:
            adc_cube: Raw ADC data for the current frame.
            history_buffer: Buffer containing previous frames.
            processor_params: Dictionary of processor parameters.
        """
        for key, spec in self.registry.items():
            if not spec.enabled or key not in self.processors:
                continue

            processor = self.processors[key]
            
            # Load params for this processor
            params = processor_params.get("processors", {}).get(key, {})
            
            try:
                # Determine input data based on history requirement
                if spec.num_frames_history > 1:
                    if len(history_buffer) < spec.num_frames_history:
                        # Not enough history yet
                        continue
                    if spec.requires_velocity and velocity_ned is not None:
                        result = processor.process(adc_cube=adc_cube, velocity_ned=velocity_ned, **params)
                    else:
                        result = processor.process(adc_cube=adc_cube, **params)
                else:
                    if spec.requires_velocity and velocity_ned is not None:
                        result = processor.process(adc_cube=adc_cube, velocity_ned=velocity_ned, **params)
                    else:
                        result = processor.process(adc_cube=adc_cube, **params)

                # Construct payload dynamically based on view_keys
                payload = {"data": result}
                
                if spec.view_keys:
                    for attr_name in spec.view_keys:
                        if hasattr(processor, attr_name):
                             val = getattr(processor, attr_name)
                             if val is not None:
                                 payload[attr_name] = val
                        # Special handling for DopplerAzimuthProcessor which might use zoomed bins
                        # This logic is preserved from the original controller but adapted
                        if key == "doppler_azimuth_resp" and attr_name == "vel_bins":
                             if hasattr(processor, "zoomed_vel_bins") \
                                and processor.zoomed_vel_bins is not None and processor.zoomed_vel_bins.size > 0:
                                  # Check if precise FFT was used by checking output shape
                                  if result.shape[0] == processor.zoomed_vel_bins.size:
                                      payload["vel_bins"] = processor.zoomed_vel_bins
                        
                        # Special handling for DopplerAzimuthProcessor which filters angles
                        if key == "doppler_azimuth_resp" and attr_name == "angle_bins":
                            if hasattr(processor, "valid_angle_bins"):
                                payload["angle_bins"] = processor.valid_angle_bins

                self.view_update.emit(key, payload)

            except Exception as exc:
                self.logger.error("Error processing %s: %s", key, exc)
