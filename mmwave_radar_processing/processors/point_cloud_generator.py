
import numpy as np
from typing import Dict, List, Union, Optional

from mmwave_radar_processing.processors._processor import _Processor
from mmwave_radar_processing.processors.range_doppler_resp import RangeDopplerProcessor
from mmwave_radar_processing.detectors.detector_registry import get_detector_registry
from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.logging.logger import get_logger
class PointCloudGenerator(_Processor):
    """
    Generates a 3D point cloud from raw ADC data using a standard radar signal processing pipeline.
    
    Pipeline steps:
    1. Range-Doppler Processing
    2. CFAR Detection
    3. Angle Estimation (Azimuth & Elevation)
    4. Coordinate Transformation (Spherical -> Cartesian)
    """

    def __init__(
        self,
        config_manager: ConfigManager,
        cfar_type: str = "ca_cfar_2d",
        cfar_params: Dict = {},
        num_angle_bins: int = 64,
        az_antenna_idxs: Union[List[int], np.ndarray] = None,
        el_antenna_idxs: Union[List[int], np.ndarray] = None,
        shift_az_resp: bool = True,
        shift_el_resp: bool = False,
        **kwargs
    ):
        """
        Initialize the PointCloudGenerator.

        Args:
            config_manager (ConfigManager): Radar configuration manager.
            cfar_type (str): Key for the CFAR detector in the registry.
            cfar_params (Dict): Parameters to initialize the CFAR detector.
            num_angle_bins (int): Number of angle bins for angular response.
            az_antenna_idxs (Union[List[int], np.ndarray]): Indices of azimuth antennas.
            el_antenna_idxs (Union[List[int], np.ndarray]): Indices of elevation antennas.
            shift_az_resp (bool): Whether to fftshift the azimuth response.
            shift_el_resp (bool): Whether to fftshift the elevation response.
            **kwargs: Additional keyword arguments.
        """
              
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

        # Initialize RangeDopplerProcessor
        self.range_doppler_processor = RangeDopplerProcessor(config_manager)
        self.range_bins = self.range_doppler_processor.range_bins
        self.vel_bins = self.range_doppler_processor.vel_bins
            
        detector_registry = get_detector_registry()
        if cfar_type not in detector_registry:
            raise ValueError(f"Unknown CFAR type: {cfar_type}. Available: {list(detector_registry.keys())}")
            
        detector_cls = detector_registry[cfar_type]
        self.detector = detector_cls(**cfar_params)

        #setting up angular processing
        self.num_angle_bins = num_angle_bins
        self.phase_shifts:np.ndarray = None
        self.angle_bins:np.ndarray = None
        
        super().__init__(config_manager)
        
        self.logger.info(f"PointCloudGenerator initialized with {cfar_type} and params {cfar_params}")

    def configure(self):
        """
        Configure the processor.
        """
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
        # 1. Range-Doppler Processing
        rng_dop_resp_raw, rng_dop_resp = self._compute_range_doppler_response(adc_cube)

        # 2. CFAR Detection
        dets = self._perform_cfar_detection(rng_dop_resp)

        if not dets:
            return np.empty((0, 4))

        # 3. Map detections to range/velocity bins
        det_ranges, det_velocities, det_range_idxs, det_velocity_idxs = self._map_detections_to_bins(dets)

        # 4. Angle Estimation (Vectorized)
        az_angles, el_angles = self._compute_angle_estimation(
            rng_dop_resp_raw, det_range_idxs, det_velocity_idxs
        )

        # 5. Coordinate Transformation (Spherical -> Cartesian)
        dets_cartesian = self._convert_to_cartesian(
            det_ranges, az_angles, el_angles, det_velocities
        )

        return dets_cartesian

    def _compute_range_doppler_response(self, adc_cube: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the Range-Doppler response.

        Args:
            adc_cube (np.ndarray): Input ADC cube.

        Returns:
            tuple[np.ndarray, np.ndarray]: (Raw complex response, Magnitude response)
        """
        # Assume virtual array processing is already done
        # Use rx_idx=-1 to get response for all antennas
        rng_dop_resp_raw = self.range_doppler_processor.process(
            adc_cube=adc_cube,
            rx_idx=-1,
            return_magnitude=False
        )
        
        # Use the first antenna (or average/sum) for detection map
        # Typically, detection is done on non-coherent integration or a specific antenna
        # Here we use the magnitude of the first antenna's response as a simple baseline
        # Or we could sum magnitudes across antennas for better SNR
        rng_dop_resp = np.abs(rng_dop_resp_raw[0, :, :])
        
        return rng_dop_resp_raw, rng_dop_resp

    def _perform_cfar_detection(self, rng_dop_resp: np.ndarray) -> List:
        """
        Perform CFAR detection on the Range-Doppler map.

        Args:
            rng_dop_resp (np.ndarray): Range-Doppler magnitude map.

        Returns:
            List: List of detections (tuples of indices).
        """
        dets = self.detector.detect(rng_dop_resp)
        return dets

    def _map_detections_to_bins(self, dets: List) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Map detection indices to physical range and velocity values.

        Args:
            dets (List): List of detection indices.

        Returns:
            tuple: (det_ranges, det_velocities, det_range_idxs, det_velocity_idxs)
        """
        dets_array = np.array(dets)
        det_range_idxs = dets_array[:, 0].astype(int)
        det_velocity_idxs = dets_array[:, 1].astype(int)

        det_ranges = self.range_bins[det_range_idxs]
        det_velocities = self.vel_bins[det_velocity_idxs]

        return det_ranges, det_velocities, det_range_idxs, det_velocity_idxs

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
        az_raw_batch = rng_dop_resp_raw[self.az_antenna_idxs][:, det_range_idxs, det_velocity_idxs].T
        el_raw_batch = rng_dop_resp_raw[self.el_antenna_idxs][:, det_range_idxs, det_velocity_idxs].T

        # 2. Prepare FFT inputs (Zero padding)
        # Shape: (N_dets, num_angle_bins)
        az_fft_input = np.zeros((num_dets, self.num_angle_bins), dtype=complex)
        el_fft_input = np.zeros((num_dets, self.num_angle_bins), dtype=complex)

        az_fft_input[:, :len(self.az_antenna_idxs)] = az_raw_batch
        el_fft_input[:, :len(self.el_antenna_idxs)] = el_raw_batch

        # 3. Compute FFTs along axis 1
        # Azimuth
        az_fft = np.fft.fft(az_fft_input, axis=1)
        if self.shift_az_resp:
            az_resp_batch = np.abs(np.fft.fftshift(az_fft, axes=1))
        else:
            az_resp_batch = np.abs(az_fft)
            
        # Elevation
        el_fft = np.fft.fft(el_fft_input, axis=1)
        if self.shift_el_resp:
            el_resp_batch = np.abs(np.fft.fftshift(el_fft, axes=1))
        else:
            el_resp_batch = np.abs(el_fft)

        # 4. Find peak angles
        az_angle_idxs = np.argmax(az_resp_batch, axis=1)
        el_angle_idxs = np.argmax(el_resp_batch, axis=1)

        az_angles = self.angle_bins[az_angle_idxs]
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
        pass
