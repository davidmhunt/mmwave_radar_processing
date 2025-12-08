"""Range-Doppler Detector processor."""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .range_doppler_detector import RangeDopplerDetector
from mmwave_radar_processing.detectors.detector_registry import get_detector_registry
from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.logging.logger import get_logger


class RangeDopplerDetector2D(RangeDopplerDetector):
    """
    Processor that performs Range-Doppler processing followed by 2D CFAR detection.
    
    This processor computes the Range-Doppler response and then applies a 2D CFAR
    detector to directly detect objects in the range-doppler response. It stores the 
    raw and magnitude Range-Doppler responses as class attributes for later access.
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        cfar_type: str = "ca_cfar_2d",
        cfar_params: Dict = {},
        **kwargs
    ):
        """
        Initialize the RangeDopplerDetector.

        Args:
            config_manager (ConfigManager): Radar configuration manager.
            cfar_type (str): Key for the CFAR detector in the registry.
            cfar_params (Dict): Parameters to initialize the CFAR detector.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(config_manager, **kwargs)
        
        # Initialize CFAR Detector
        detector_registry = get_detector_registry()
        if cfar_type not in detector_registry:
            raise ValueError(f"Unknown CFAR type: {cfar_type}. Available: {list(detector_registry.keys())}")
            
        detector_cls = detector_registry[cfar_type]
        self.detector = detector_cls(**cfar_params)
        
        self.logger.info(f"RangeDopplerDetector initialized with {cfar_type} and params {cfar_params}")

    def _detect(self, adc_cube: np.ndarray, rng_dop_resp: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform detection on the Range-Doppler response.

        Args:
            adc_cube (np.ndarray): Input ADC cube.
            rng_dop_resp (np.ndarray): Magnitude Range-Doppler response.
            **kwargs: Additional arguments.

        Returns:
            np.ndarray: Array of detections.
        """
        dets = self.detector.detect(rng_dop_resp)
        
        if not dets:
            return np.empty((0, 2), dtype=int)
        else:
            return np.array(dets, dtype=int)