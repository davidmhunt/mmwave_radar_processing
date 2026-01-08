
import numpy as np
from typing import Dict, List, Union, Optional

from mmwave_radar_processing.processors._processor import _Processor
from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.logging.logger import get_logger
from mmwave_radar_processing.processors.range_doppler_detection.registry import get_range_doppler_detector_registry

class PointCloudGenerator(_Processor):
    """
    Generates a 3D point cloud from raw ADC data using a standard radar signal processing pipeline.
    
    Pipeline steps:
    1. Range-Doppler Processing (delegated to detector)
    2. CFAR Detection (delegated to detector)
    3. Angle Estimation (Azimuth & Elevation)
    4. Coordinate Transformation (Spherical -> Cartesian)
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        az_antenna_idxs: Union[List[int], np.ndarray],
        el_antenna_idxs: Union[List[int], np.ndarray],
        detector_type: str = "range_doppler_detector_2d",
        detector_params: Dict = {},
        shift_az_resp: bool = True,
        shift_el_resp: bool = False,
        num_angle_bins: int = 64,
        **kwargs
    ):
        """
        Initialize the PointCloudGenerator.

        Args:
            config_manager (ConfigManager): Radar configuration manager.
            detector_type (str): Key for the Range-Doppler detector in the registry.
            detector_params (Dict): Parameters to initialize the detector.
            num_angle_bins (int): Number of angle bins for angular response.
            az_antenna_idxs (Union[List[int], np.ndarray]): Indices of azimuth antennas.
            el_antenna_idxs (Union[List[int], np.ndarray]): Indices of elevation antennas.
            shift_az_resp (bool): Whether to fftshift the azimuth response.
            shift_el_resp (bool): Whether to fftshift the elevation response.
            **kwargs: Additional keyword arguments.
        """

        self.logger = get_logger(__name__)
              
        self.shift_az_resp = shift_az_resp
        self.shift_el_resp = shift_el_resp

        # Handle antenna indices
        if az_antenna_idxs is None:
            self.az_antenna_idxs = np.array([], dtype=int)
        elif isinstance(az_antenna_idxs, list):
            self.az_antenna_idxs = np.array(az_antenna_idxs, dtype=int)
        elif isinstance(az_antenna_idxs, np.ndarray):
            self.az_antenna_idxs = az_antenna_idxs.astype(int)
        else:
            raise ValueError("az_antenna_idxs must be a list or numpy array")
            
        if el_antenna_idxs is None:
            self.el_antenna_idxs = np.array([], dtype=int)
        elif isinstance(el_antenna_idxs, list):
            self.el_antenna_idxs = np.array(el_antenna_idxs, dtype=int)
        elif isinstance(el_antenna_idxs, np.ndarray):
            self.el_antenna_idxs = el_antenna_idxs.astype(int)
        else:
            raise ValueError("el_antenna_idxs must be a list or numpy array")

        #setting up angular processing
        self.num_angle_bins = num_angle_bins
        self.phase_shifts:np.ndarray = None
        self.angle_bins:np.ndarray = None
        
        # Initialize Range-Doppler Detector
        registry = get_range_doppler_detector_registry()
        if detector_type not in registry:
            raise ValueError(f"Unknown detector type: {detector_type}. Available: {list(registry.keys())}")
            
        detector_cls = registry[detector_type]
        self.detector = detector_cls(config_manager, **detector_params)
        
        self.logger.info(f"PointCloudGenerator initialized with detector: {detector_type}")
        
        super().__init__(config_manager)

    def configure(self):
        """
        Configure the processor.
        """
        self.detector.configure()

        self.phase_shifts = np.arange(
            start=np.pi,
            stop= -np.pi - 2 * np.pi/(self.num_angle_bins - 1),
            step=-2 * np.pi / (self.num_angle_bins - 1)
        )

        #round the last entry to be exactly pi
        self.phase_shifts[-1] = -1 * np.pi

        #compute the angle bins
        self.angle_bins = np.arcsin(self.phase_shifts / np.pi)
    

    def process(self, adc_cube: np.ndarray, **kwargs) -> np.ndarray:
        """
        Process the ADC cube to generate a point cloud.

        Args:
            adc_cube (np.ndarray): Input ADC cube.
            **kwargs: Additional arguments.

        Returns:
            np.ndarray: Generated point cloud in cartesian coordinates (x,y,z,vel).
        """

        # 1. Compute Range-Doppler Response and Detections using the composed detector
        dets = self.detector.process(adc_cube, **kwargs)
        
        if dets.shape[0] == 0:
            return np.empty((0, 4))

        # 2. Map detections to range/velocity bins (using detector's helper)
        det_ranges, det_velocities, det_range_idxs, det_velocity_idxs = self.detector._map_detections_to_bins(dets)

        # 3. Angle Estimation (Vectorized)
        # We access the raw response from the detector
        az_angles, el_angles = self._compute_angle_estimation(
            self.detector.rng_dop_resp_raw, det_range_idxs, det_velocity_idxs
        )

        # 4. Coordinate Transformation (Spherical -> Cartesian)
        dets_cartesian = self._convert_to_cartesian(
            det_ranges, az_angles, el_angles, det_velocities
        )

        return dets_cartesian


    def _compute_angle_estimation(
        self, 
        rng_dop_resp_raw: np.ndarray, 
        det_range_idxs: np.ndarray, 
        det_velocity_idxs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute azimuth and elevation angles for detections using FFT.

        Args:
            rng_dop_resp_raw (np.ndarray): Raw complex Range-Doppler response.
            det_range_idxs (np.ndarray): Range indices of detections.
            det_velocity_idxs (np.ndarray): Velocity indices of detections.

        Returns:
            tuple: (az_angles, el_angles)
        """
        num_dets = len(det_range_idxs)
        
        # 1. Extract raw data for all detections at once
        # Shape: (N_dets, N_antennas)
        # Transpose to get (N_dets, N_antennas) from (N_antennas, N_dets)
        
        # Handle Azimuth
        if self.az_antenna_idxs.size == 0:
            az_angles = np.zeros(num_dets)
            az_raw_batch = np.zeros((num_dets, 0), dtype=complex) # Placeholder for shape consistency if needed later
        else:
            az_raw_batch = rng_dop_resp_raw[self.az_antenna_idxs][:, det_range_idxs, det_velocity_idxs].T

        # Handle Elevation
        if self.el_antenna_idxs.size == 0:
            el_angles = np.zeros(num_dets)
            el_raw_batch = np.zeros((num_dets, 0), dtype=complex)
        else:
            el_raw_batch = rng_dop_resp_raw[self.el_antenna_idxs][:, det_range_idxs, det_velocity_idxs].T

        # 2. Process Azimuth
        if self.az_antenna_idxs.size > 0:
            # Prepare FFT inputs (Zero padding)
            az_fft_input = np.zeros((num_dets, self.num_angle_bins), dtype=complex)
            az_fft_input[:, :len(self.az_antenna_idxs)] = az_raw_batch

            # Compute FFT along axis 1
            az_fft = np.fft.fft(az_fft_input, axis=1)
            if self.shift_az_resp:
                az_resp_batch = np.abs(np.fft.fftshift(az_fft, axes=1))
            else:
                az_resp_batch = np.abs(az_fft)

            # Find peak angles
            az_angle_idxs = np.argmax(az_resp_batch, axis=1)
            az_angles = self.angle_bins[az_angle_idxs]
            
        # 3. Process Elevation
        if self.el_antenna_idxs.size > 0:
            # Prepare FFT inputs (Zero padding)
            el_fft_input = np.zeros((num_dets, self.num_angle_bins), dtype=complex)
            el_fft_input[:, :len(self.el_antenna_idxs)] = el_raw_batch

            # Compute FFT along axis 1
            el_fft = np.fft.fft(el_fft_input, axis=1)
            if self.shift_el_resp:
                el_resp_batch = np.abs(np.fft.fftshift(el_fft, axes=1))
            else:
                el_resp_batch = np.abs(el_fft)

            # Find peak angles
            el_angle_idxs = np.argmax(el_resp_batch, axis=1)
            el_angles = self.angle_bins[el_angle_idxs]

        return az_angles, el_angles

    def _convert_to_cartesian(
        self, 
        ranges: np.ndarray, 
        az_angles: np.ndarray, 
        el_angles: np.ndarray, 
        velocities: np.ndarray
    ) -> np.ndarray:
        """
        Convert spherical coordinates to Cartesian coordinates (FLU frame).
        
        FLU Frame:
        +x: Forward
        +y: Left
        +z: Up

        Args:
            ranges (np.ndarray): Ranges.
            az_angles (np.ndarray): Azimuth angles.
            el_angles (np.ndarray): Elevation angles.
            velocities (np.ndarray): Velocities.

        Returns:
            np.ndarray: Array of shape (N, 4) containing (x, y, z, velocity).
        """
        # x = r * cos(el) * cos(az)
        # y = r * cos(el) * sin(az)
        # z = r * sin(el)
        
        x = ranges * np.cos(el_angles) * np.cos(az_angles)
        y = ranges * np.cos(el_angles) * np.sin(az_angles)
        z = ranges * np.sin(el_angles)
        
        return np.column_stack((x, y, z, velocities))

    def reset(self):
        """
        Reset the processor state.
        """
        self.detector.reset()

        return super().reset()
