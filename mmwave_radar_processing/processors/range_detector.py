import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.signal import ZoomFFT
from scipy.signal import find_peaks

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.range_resp import RangeProcessor
from mmwave_radar_processing.detectors.detector_registry import get_detector_registry


class RangeDetector(RangeProcessor):
    """
    Processor that performs range FFT processing followed by 1D CFAR Detection
    """
    def __init__(
            self,
            config_manager: ConfigManager,
            cfar_type: str = "os_cfar_1d",
            cfar_params: dict = {},
            **kwargs):
        """
        Initialize the RangeDetector.

        Args:
            config_manager (ConfigManager): Radar configuration manager.
            cfar_type (str): Key for the CFAR range detector in the registry.
            cfar_params (Dict): Parameters to initialize the CFAR range detector.
            **kwargs: Additional keyword arguments.
        """
        
        # Initialize Range CFAR Detector
        detector_registry = get_detector_registry()
        if cfar_type not in detector_registry:
            raise ValueError(f"Unknown CFAR type: {cfar_type}. Available: {list(detector_registry.keys())}")
        detector_cls = detector_registry[cfar_type]
        self.cfar_detector = detector_cls(**cfar_params)

        #storage of most recent detections
        self.dets: Optional[np.ndarray] = None
        self.thresholds: Optional[np.ndarray] = None

        #storage of the most recent range response
        self.range_resp: Optional[np.ndarray] = None

        #initialize parent class
        super().__init__(config_manager)

        self.logger.info(f"RangeDetector initialized with CFAR type: {cfar_type}")


    def configure(self):
        """Configure the processor and also call the parent class processor
        """
        
        super().configure()
        return


    def process(self, adc_cube: np.ndarray, **kwargs) -> np.ndarray:
        """Process the ADC cube to obtain a coarse range response.

        Args:
            adc_cube (np.ndarray): (num rx antennas) x (num adc samples) x (num chirps).
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: Magnitude of range FFT (range profile) for the selected RX channel.
        """

        self.range_resp = super().process(adc_cube,chirp_idx=0)

        det_range_idxs = self.cfar_detector.detect(self.range_resp)

        #save the thresholds
        self.thresholds = self.cfar_detector.thresholds

        #save the detections
        if len(det_range_idxs) > 0:
            self.dets = np.array(det_range_idxs)
        else:
            self.dets = np.array([])

        return self.dets
    
    def _map_detections_to_bins(self, dets: np.ndarray) -> np.ndarray:
        """
        Map detection indices to range bins.

        Args:
            dets (np.ndarray): Array of detection indices.

        Returns:
            np.ndarray: Array of range bin values corresponding to detections.
        """
        if self.range_bins is None:
            raise ValueError("Range bins are not configured.")
        
        return self.range_bins[dets]