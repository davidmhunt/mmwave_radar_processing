"""Range-Doppler Detector processor."""

import numpy as np
from typing import Dict, List, Optional, Tuple

from mmwave_radar_processing.processors._processor import _Processor
from mmwave_radar_processing.processors.range_doppler_resp import RangeDopplerProcessor
from mmwave_radar_processing.detectors.detector_registry import get_detector_registry
from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.logging.logger import get_logger


class RangeDopplerDetector2D(RangeDopplerProcessor):
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
        self.logger = get_logger(__name__)
        
        # Initialize CFAR Detector
        detector_registry = get_detector_registry()
        if cfar_type not in detector_registry:
            raise ValueError(f"Unknown CFAR type: {cfar_type}. Available: {list(detector_registry.keys())}")
            
        detector_cls = detector_registry[cfar_type]
        self.detector = detector_cls(**cfar_params)
        
        # Storage for latest response
        self.rng_dop_resp_raw: Optional[np.ndarray] = None
        self.rng_dop_resp: Optional[np.ndarray] = None

        #storage of most recent detections
        self.dets: Optional[np.ndarray] = None

        super().__init__(config_manager)

        self.logger.info(f"RangeDopplerDetector initialized with {cfar_type} and params {cfar_params}")

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
        # 1. Compute the range doppler response
        self.rng_dop_resp_raw = super().process(
            adc_cube=adc_cube,
            rx_idx=-1,
            return_magnitude=False
        )

        #2. Get the magnitude of the first antenna's response for detection
        self.rng_dop_resp = np.abs(self.rng_dop_resp_raw[0, :, :])
        
        # 3. Perform CFAR Detection
        dets = self.detector.detect(self.rng_dop_resp)
        
        # 4.Convert to numpy array
        if not dets:
            self.dets = np.empty((0, 2), dtype=int)
        else:
            self.dets = np.array(dets, dtype=int)
            
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