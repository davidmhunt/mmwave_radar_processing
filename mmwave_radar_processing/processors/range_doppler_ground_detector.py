"""Range-Doppler Detector processor."""

import numpy as np
from typing import Dict, List, Optional, Tuple

from mmwave_radar_processing.processors._processor import _Processor
from mmwave_radar_processing.processors.range_doppler_resp import RangeDopplerProcessor
from mmwave_radar_processing.detectors.detector_registry import get_detector_registry
from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.altimeter import Altimeter
from mmwave_radar_processing.processors.range_resp import RangeProcessor


class RangeDopplerGroundDetector(RangeDopplerProcessor):
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

        # Initialize Velocity CFAR Detector
        detector_registry = get_detector_registry()
        if vel_cfar_type not in detector_registry:
            raise ValueError(f"Unknown CFAR type: {vel_cfar_type}. Available: {list(detector_registry.keys())}")
        detector_cls = detector_registry[vel_cfar_type]
        self.vel_detector = detector_cls(**vel_cfar_params)
        
        # Storage for latest response
        self.rng_dop_resp_raw: Optional[np.ndarray] = None
        self.rng_dop_resp: Optional[np.ndarray] = None

        #storage of most recent detections
        self.dets: Optional[np.ndarray] = None

        #initialize the range response processor (for range detections)
        self.altimeter_params = altimeter_params
        self.altimeter = Altimeter(config_manager, **altimeter_params)

        super().__init__(config_manager)

        self.logger.info(f"RangeDopplerDetectorSequential initialized with Velocity CFAR: {vel_cfar_type}")

    def configure(self):
        """
        Configure the processor and also call the parent class processor
        """
        super().configure()

        return
    
    def reset(self):
        """
        Reset the processor state.
        """
        super().reset()

    def process(self, adc_cube: np.ndarray, **kwargs) -> np.ndarray:
        """
        Process the ADC cube to generate detections.

        Args:
            adc_cube (np.ndarray): Input ADC cube with shape (n_rx, n_range_bins, n_chirps).
            **kwargs: Additional arguments.

        Returns:
            np.ndarray: Array of detections. Each row is a detection (range_idx, doppler_idx).
        """

        #1. compute the altitude estimate
        altitude_m = self.altimeter.process(
            adc_cube=adc_cube,
            **self.altimeter_params
        )

        #2. determine the range bins to search for ground detections
        # Find the range bin index corresponding to the estimated altitude
        min_rng_idx = np.argmin(np.abs(self.range_bins - altitude_m))

        #define the range window around the estimated altitude to search for ground detections
        max_rng = min(
            np.max(self.range_bins), #cap the max range to the max range bin
            altitude_m / np.cos(np.deg2rad(60))) #based on the range of a ground target at 60 degrees angle
        
        #Find the range bin index corresponding to the max range for ground detection
        max_rng_idx = np.argmin(np.abs(self.range_bins - max_rng))

        #define the desired range bins to search for ground detections
        if max_rng_idx == min_rng_idx:
            det_range_idxs = np.array([min_rng_idx])
        else:
            det_range_idxs = np.arange(
                start=min_rng_idx,
                stop=max_rng_idx + 1
            )

        # 3. Compute the range doppler response
        self.rng_dop_resp_raw = super().process(
            adc_cube=adc_cube,
            rx_idx=-1,
            return_magnitude=False
        )

        #4. Get the magnitude of the first antenna's response for detection
        self.rng_dop_resp = np.abs(self.rng_dop_resp_raw[0, :, :])

        #5. If range detections were found, proceed to velocity CFAR
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
    
    def _map_detections_to_bins(self, dets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Map detection indices to physical range and velocity values.

        Args:
            dets (List): List of detection indices.

        Returns:
            tuple: (det_ranges, det_velocities, det_range_idxs, det_velocity_idxs)
        """
        det_range_idxs = dets[:, 0].astype(int)
        det_velocity_idxs = dets[:, 1].astype(int)

        det_ranges = self.range_bins[det_range_idxs]
        det_velocities = self.vel_bins[det_velocity_idxs]

        return det_ranges, det_velocities, det_range_idxs, det_velocity_idxs