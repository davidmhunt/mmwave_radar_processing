"""Range-Doppler Detector processor."""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .range_doppler_detector import RangeDopplerDetector
from mmwave_radar_processing.detectors.detector_registry import get_detector_registry
from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.altimeter import Altimeter
from mmwave_radar_processing.processors.range_doppler_resp import RangeDopplerProcessor


class RangeDopplerGroundDetector(RangeDopplerDetector):
    """
    Processor that performs Range-Doppler processing followed by 2D CFAR detection.
    
    This processor computes the Range-Doppler response and then applies a sequential CFAR
    detector (range followed by velocity) to detect objects. 
    #TODO: flush this out with a better summary of what is being done in the class
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        vel_cfar_type: str = "os_cfar_1d",
        vel_cfar_params: Dict = {},
        altimeter_params: Dict = {},
        **kwargs
    ):
        """
        Initialize the RangeDopplerDetector.

        Args:
            config_manager (ConfigManager): Radar configuration manager.
            vel_cfar_type (str): Key for the velocity CFAR detector in the registry.
            vel_cfar_params (Dict): Parameters to initialize the velocity CFAR detector.
            altimeter_params (Dict): Parameters to initialize the altimeter.
            **kwargs: Additional keyword arguments.
        """        
        super().__init__(config_manager, **kwargs)

        # Initialize Velocity CFAR Detector
        detector_registry = get_detector_registry()
        if vel_cfar_type not in detector_registry:
            raise ValueError(f"Unknown CFAR type: {vel_cfar_type}. Available: {list(detector_registry.keys())}")
        detector_cls = detector_registry[vel_cfar_type]
        self.vel_detector = detector_cls(**vel_cfar_params)
        self.logger.info(f"RangeDopplerGroundDetector initialized params: {vel_cfar_params}")
        
        #initialize the range response processor (for range detections)
        self.altimeter_params = altimeter_params
        self.altimeter = Altimeter(config_manager, **altimeter_params)

        self.logger.info(f"RangeDopplerGroundDetector initialized with Velocity CFAR: {vel_cfar_type}")

    def reset(self):

        self.altimeter.reset()

        return super().reset()
    
    def _detect(self, adc_cube: np.ndarray, rng_dop_resp: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform detection on the Range-Doppler response.
        
        Note: This method is not used directly in process() as the ground detector
        requires intermediate altitude processing, but we implement it to satisfy
        the abstract base class.
        """
        raise NotImplementedError("RangeDopplerGroundDetector uses custom process logic.")

    def process(self, adc_cube: np.ndarray, **kwargs) -> np.ndarray:
        """
        Process the ADC cube to generate detections.

        Args:
            adc_cube (np.ndarray): Input ADC cube with shape (n_rx, n_range_bins, n_chirps).
            **kwargs: Additional arguments.

        Returns:
            np.ndarray: Array of detections. Each row is a detection (range_idx, doppler_idx).
        """

        altitude_m = self.altimeter.process(
            adc_cube=adc_cube,
            **self.altimeter_params
        )

        min_rng_idx = np.argmin(np.abs(self.range_bins - altitude_m))

        max_rng = min(
            np.max(self.range_bins), 
            altitude_m / np.cos(np.deg2rad(60))) 
        
        max_rng_idx = np.argmin(np.abs(self.range_bins - max_rng))

        if max_rng_idx == min_rng_idx:
            det_range_idxs = np.array([min_rng_idx])
        else:
            det_range_idxs = np.arange(
                start=min_rng_idx,
                stop=max_rng_idx + 1
            )

        self._compute_range_doppler_response(adc_cube)

        if len(det_range_idxs) > 0:
            rng_dop_dets = [
                (rng_idx, dop_idx)
                for rng_idx in det_range_idxs
                for dop_idx in self.vel_detector.detect(x=self.rng_dop_resp[rng_idx, :])
            ]

            if len(rng_dop_dets) > 0:
                self.dets = np.array(rng_dop_dets, dtype=int)
            else:
                self.dets = np.empty((0, 2), dtype=int)
        else:
            self.dets = np.empty((0, 2), dtype=int)
            
        return self.dets