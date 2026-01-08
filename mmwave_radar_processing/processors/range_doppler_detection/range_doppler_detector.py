"""Base class for Range-Doppler Detectors."""

import numpy as np
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod

from mmwave_radar_processing.processors.range_doppler_resp import RangeDopplerProcessor
from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.logging.logger import get_logger

class RangeDopplerDetector(RangeDopplerProcessor, ABC):
    """
    Base class for processors that perform detection on Range-Doppler maps.
    """

    def __init__(self, config_manager: ConfigManager, **kwargs):
        """
        Initialize the RangeDopplerDetector.

        Args:
            config_manager (ConfigManager): Radar configuration manager.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(config_manager)
        self.logger = get_logger(__name__)

        # Storage for latest response
        self.rng_dop_resp_raw: Optional[np.ndarray] = None
        self.rng_dop_resp: Optional[np.ndarray] = None

        # Storage of most recent detections
        self.dets: Optional[np.ndarray] = None

    def configure(self):
        """Configure the processor."""
        super().configure()

    def reset(self):
        """Reset the processor state."""
        super().reset()
        self.rng_dop_resp_raw = None
        self.rng_dop_resp = None
        self.dets = None

    def process(self, adc_cube: np.ndarray, **kwargs) -> np.ndarray:
        """
        Process the ADC cube to generate detections.

        Args:
            adc_cube (np.ndarray): Input ADC cube.
            **kwargs: Additional arguments.

        Returns:
            np.ndarray: Array of detections.
        """
        self._compute_range_doppler_response(adc_cube)

        self.dets = self._detect(adc_cube, self.rng_dop_resp, **kwargs)

        return self.dets

    def _compute_range_doppler_response(self, adc_cube: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Range-Doppler response.

        Args:
            adc_cube (np.ndarray): Input ADC cube.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (raw_response, magnitude_response)
        """
        self.rng_dop_resp_raw = super().process(
            adc_cube=adc_cube,
            rx_idx=-1,
            return_magnitude=False
        )

        self.rng_dop_resp = np.abs(self.rng_dop_resp_raw[0, :, :])
        
        return self.rng_dop_resp_raw, self.rng_dop_resp

    @abstractmethod
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
        pass

    def _map_detections_to_bins(self, dets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Map detection indices to physical range and velocity values.

        Args:
            dets (np.ndarray): Array of detection indices.

        Returns:
            tuple: (det_ranges, det_velocities, det_range_idxs, det_velocity_idxs)
        """
        if dets is None or dets.size == 0:
             return np.array([]), np.array([]), np.array([]), np.array([])

        det_range_idxs = dets[:, 0].astype(int)
        det_velocity_idxs = dets[:, 1].astype(int)

        det_ranges = self.range_bins[det_range_idxs]
        det_velocities = self.vel_bins[det_velocity_idxs]

        return det_ranges, det_velocities, det_range_idxs, det_velocity_idxs
