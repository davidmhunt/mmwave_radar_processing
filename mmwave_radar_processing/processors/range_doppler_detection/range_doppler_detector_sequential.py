"""Range-Doppler Detector processor."""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .range_doppler_detector import RangeDopplerDetector
from mmwave_radar_processing.detectors.detector_registry import get_detector_registry
from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.range_resp import RangeProcessor


class RangeDopplerDetectorSequential(RangeDopplerDetector):
    """
    Processor that performs Range-Doppler processing followed by 2D CFAR detection.
    
    This processor computes the Range-Doppler response and then applies a sequential CFAR
    detector (range followed by velocity) to detect objects. 
    #TODO: flush this out with a better summary of what is being done in the class
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        rng_cfar_type: str = "os_cfar_1d",
        rng_cfar_params: Dict = {},
        vel_cfar_type: str = "os_cfar_1d",
        vel_cfar_params: Dict = {},
        **kwargs
    ):
        """
        Initialize the RangeDopplerDetector.

        Args:
            config_manager (ConfigManager): Radar configuration manager.
            rng_cfar_type (str): Key for the CFAR range detector in the registry.
            rng_cfar_params (Dict): Parameters to initialize the CFAR range detector.
            vel_cfar_type (str): Key for the velocity CFAR detector in the registry.
            vel_cfar_params (Dict): Parameters to initialize the velocity CFAR detector.
            **kwargs: Additional keyword arguments.
        """        
        super().__init__(config_manager, **kwargs)

        # Initialize Range CFAR Detector
        detector_registry = get_detector_registry()
        if rng_cfar_type not in detector_registry:
            raise ValueError(f"Unknown CFAR type: {rng_cfar_type}. Available: {list(detector_registry.keys())}")
        detector_cls = detector_registry[rng_cfar_type]
        self.rng_detector = detector_cls(**rng_cfar_params)
        

        # Initialize Velocity CFAR Detector
        if vel_cfar_type not in detector_registry:
            raise ValueError(f"Unknown CFAR type: {vel_cfar_type}. Available: {list(detector_registry.keys())}")
        detector_cls = detector_registry[vel_cfar_type]
        self.vel_detector = detector_cls(**vel_cfar_params)
        
        #initialize the range response processor (for range detections)
        self.range_processor = RangeProcessor(config_manager)

        self.logger.info(f"RangeDopplerDetectorSequential initialized with Range CFAR: {rng_cfar_type}, Velocity CFAR: {vel_cfar_type}")

    def _detect(self, adc_cube: np.ndarray, rng_dop_resp: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform detection on the Range-Doppler response.
        
        Note: This method is not used directly in process() as the sequential detector
        requires intermediate range response processing, but we implement it to satisfy
        the abstract base class.
        """
        raise NotImplementedError("RangeDopplerDetectorSequential uses custom process logic.")

    def process(self, adc_cube: np.ndarray, **kwargs) -> np.ndarray:
        """
        Process the ADC cube to generate detections.

        Args:
            adc_cube (np.ndarray): Input ADC cube with shape (n_rx, n_range_bins, n_chirps).
            **kwargs: Additional arguments.

        Returns:
            np.ndarray: Array of detections. Each row is a detection (range_idx, doppler_idx).
        """
        range_resp = self.range_processor.process(
            adc_cube=adc_cube,
            chirp_idx=0
        )

        det_range_idxs = self.rng_detector.detect(
            x=range_resp
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