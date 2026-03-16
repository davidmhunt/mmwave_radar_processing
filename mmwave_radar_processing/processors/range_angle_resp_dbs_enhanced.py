import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors._processor import _Processor
from mmwave_radar_processing.processors.range_angle_resp import RangeAngleProcessor

class RangeAngleProcessorDBSEnhanced(RangeAngleProcessor):
    """Range Angle Processor, but enhanced with doppler beam sharpening (DBS)."""
    def __init__(
            self,
            config_manager: ConfigManager,
            num_angle_bins_range_angle_response:int = 64,
            num_angle_bins_dbs_enhanced_response:int = 64,
            min_x_y_vel_dbs:float = 0.25,
            **kwargs) -> None:
        """Initializes the DBS-enhanced Range-Angle processor.

        Args:
            config_manager (ConfigManager): Radar configuration manager.
            num_angle_bins_range_angle_response (int, optional): Number of angle bins for the standard Range-Angle response. Defaults to 64.
            num_angle_bins_dbs_enhanced_response (int, optional): Number of angle bins for the DBS-enhanced response. Defaults to 64.
            min_x_y_vel_dbs (float, optional): Minimum required velocity to apply DBS. Defaults to 0.25.
            **kwargs: Additional keyword arguments passed to the parent processor.
        """

        #angle bins without dbs
        self.angle_bins_no_dbs_enhancement:np.ndarray = None
        
        #angle bins for DBS
        self.num_angle_bins_dbs_enhanced_response = num_angle_bins_dbs_enhanced_response
        self.angle_bins_dbs_enhanced:np.ndarray = None
        self.phase_shifts_dbs_enhanced:np.ndarray = None

        self.min_vel_dbs = min_x_y_vel_dbs

        #velocity bins for DBS
        self.vel_bins:np.ndarray = None

        #load the configuration and configure the response 
        super().__init__(
            config_manager=config_manager,
            num_angle_bins=num_angle_bins_range_angle_response,
            **kwargs
        )

    
    def configure(self) -> None:
        """Configures the processor's internal bins and mesh grid.

        Overrides the parent's `configure` method to include velocity bins and 
        calculate enhanced angle bins for Doppler Beam Sharpening.
        """
        #set the range bins
        self.num_range_bins = self.config_manager.get_num_adc_samples(profile_idx=0)
        self.range_bins = np.arange(
            start=0,
            step=self.config_manager.range_res_m,
            stop=self.config_manager.range_max_m - self.config_manager.range_res_m/2) + 1e-3

        #set the velocity bins
        self.vel_bins = np.arange(
            start=-1 * self.config_manager.vel_max_m_s,
            stop = self.config_manager.vel_max_m_s - self.config_manager.vel_res_m_s + 1e-3,
            step= self.config_manager.vel_res_m_s
        )

        #compute the phase shifts (for doppler azimuth response processing)
        self.num_rx_antennas = self.config_manager.num_rx_antennas
        self.phase_shifts = np.arange(
            start=np.pi,
            stop= -np.pi - 2 * np.pi/(self.num_angle_bins - 1),
            step=-2 * np.pi / (self.num_angle_bins - 1)
        )

        #round the last entry to be exactly pi
        self.phase_shifts[-1] = -1 * np.pi

        #compute the angle bins
        self.angle_bins_no_dbs_enhancement = np.arcsin(self.phase_shifts / np.pi)

        #compute the angle_bins_dbs_enhanced
        self.angle_bins_dbs_enhanced = np.linspace(
            start=self.angle_bins_no_dbs_enhancement[0], #should be pi/2
            stop=self.angle_bins_no_dbs_enhancement[-1], #should be pi/2
            num=self.num_angle_bins_dbs_enhanced_response
        )

        #compute the mesh grid (no dbs for now)
        self.compute_mesh_grid()
    
    def compute_mesh_grid_dbs_enhanced(self) -> None:
        """Computes the Cartesian mesh grid specifically for the DBS-enhanced angle bins."""

        self.angle_bins = self.angle_bins_dbs_enhanced

        #compute the mesh grid (for dbs enhanced)
        self.thetas,self.rhos = np.meshgrid(self.angle_bins_dbs_enhanced,self.range_bins)
        self.x_s = np.multiply(self.rhos,np.cos(self.thetas)) #pointing out
        self.y_s = np.multiply(self.rhos,np.sin(self.thetas)) #pointing left/right

    def compute_mesh_grid(self) -> None:
        """Computes the standard Cartesian mesh grid without DBS enhancement."""
        
        self.angle_bins = self.angle_bins_no_dbs_enhancement

        #compute the mesh grid (for dbs enhanced)
        self.thetas,self.rhos = np.meshgrid(self.angle_bins,self.range_bins)
        self.x_s = np.multiply(self.rhos,np.cos(self.thetas)) #pointing out
        self.y_s = np.multiply(self.rhos,np.sin(self.thetas)) #pointing left/right

    def process_no_dbs(self,
                adc_cube: np.ndarray,
                chirp_idx = 0,
                rx_antennas:np.ndarray = np.array([]),
                **kwargs) -> np.ndarray:
        """Computes the Range-Angle response without Doppler beam sharpening.

        Args:
            adc_cube (np.ndarray): The ADC data cube of complex data, shape (rx antennas, adc samples, num chirps).
            chirp_idx (int, optional): The chirp index to compute the response for. Defaults to 0.
            rx_antennas (np.ndarray | list, optional): Array or list of specific RX antenna indices to use. Defaults to empty array.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The computed Range-Azimuth response, shape (range bins, angle bins).
        """
        #compute the standard mesh grid
        self.compute_mesh_grid()
    
        return super().process(
            adc_cube=adc_cube,
            chirp_idx=chirp_idx,
            rx_antennas=rx_antennas,
            **kwargs
        )
    
    def compute_3d_windowed_fft(
        self,
        adc_cube:np.ndarray
    ) -> np.ndarray:
        """Computes the 3D windowed FFT of the ADC cube.

        Applies Hanning windows across range, velocity, and angle dimensions,
        and computes the FFT for each dimension. Designed for Doppler beam sharpening.

        Args:
            adc_cube (np.ndarray): The ADC data cube of complex data, shape (rx antennas, adc samples, num chirps).
                Assumes any virtual array processing has already been performed.

        Returns:
            np.ndarray: The 3D windowed FFT of the ADC cube, shape (num_angle_bins, adc samples, num_chirps).
        """

        #compute range hanning window
        hanning_window_range = np.hanning(adc_cube.shape[1])
        adc_cube_rng_windowed = adc_cube * hanning_window_range[np.newaxis, :, np.newaxis]

        rng_resp = np.fft.fft(
            a=adc_cube_rng_windowed,
            axis=-2
        )

        #velocity FFT - apply hanning window
        hanning_window_vel = np.hanning(adc_cube.shape[2])
        rng_resp_dop_windowed = rng_resp * hanning_window_vel[np.newaxis, np.newaxis, :]

        rng_dop_resp = np.fft.fftshift(
            np.fft.fft(
                a=rng_resp_dop_windowed,
                axis=-1
            ),
            axes=-1
        )

        #angle FFT - apply hanning window
        hanning_window_angle = np.hanning(adc_cube.shape[0])
        rng_dop_resp_angle_windowed = rng_dop_resp * hanning_window_angle[:, np.newaxis, np.newaxis]

        #perform zero padding
        zero_padded_rng_dop_resp_angle_windowed = np.zeros(
            shape=(
                self.num_angle_bins,
                adc_cube.shape[1],
                adc_cube.shape[2]
            ),
            dtype=complex
        )
        zero_padded_rng_dop_resp_angle_windowed[0:adc_cube.shape[0],:,:] = rng_dop_resp_angle_windowed

        angle_rng_dop_resp = np.fft.fftshift(
            np.fft.fft(
                a=zero_padded_rng_dop_resp_angle_windowed,
                axis=0
            ),
            axes=0
        )

        return angle_rng_dop_resp
    
    def get_dop_vel(self, angle: float, ego_vel: np.ndarray) -> float:
        """Computes the Doppler velocity for a given angle and platform velocity.

        Args:
            angle (float): The angle in radians.
            ego_vel (np.ndarray): The (x, y, z) velocity of the platform in the NED coordinate frame.

        Returns:
            float: The calculated Doppler velocity.
        """

        r = np.array([np.cos(angle),np.sin(angle),0])
        dop_vel = -1 * np.dot((r / np.linalg.norm(r)),ego_vel)

        return dop_vel

    def perform_dbs_sharpen(
        self,
        velocity_ned:np.ndarray,
        angle_rng_dop_resp_mag:np.ndarray
    ) -> np.ndarray:
        """Performs Doppler beam sharpening on the angle-range-Doppler response.

        Args:
            velocity_ned (np.ndarray): The (x, y, z) velocity of the platform in the NED coordinate frame.
            angle_rng_dop_resp_mag (np.ndarray): The magnitude of the angle-range-Doppler response, shape (angle bins, range bins, chirp bins).

        Returns:
            np.ndarray: The sharpened Range-Angle response, shape (range bins, angle bins).
        """
        
        az_rng_response_dbs = np.zeros_like(
            a=angle_rng_dop_resp_mag[:,:,0],
            shape=(
                self.angle_bins_dbs_enhanced.shape[0],
                self.range_bins.shape[0]
            )
        )

        for angle_idx in range(self.angle_bins_dbs_enhanced.shape[0]):
            dop_vel = self.get_dop_vel(
                angle=self.angle_bins_dbs_enhanced[angle_idx],
                ego_vel=velocity_ned
            )

            #get the doppler velocity bin index
            vel_bin = np.argmin(
                np.abs(
                    self.vel_bins - dop_vel
                )
            )
            angle_bin_no_dbs = np.argmin(
                np.abs(
                    self.angle_bins_no_dbs_enhancement - self.angle_bins_dbs_enhanced[angle_idx]
                )
            )

            az_rng_response_dbs[angle_idx,:] =\
                angle_rng_dop_resp_mag[angle_bin_no_dbs,:,vel_bin]
        
        #transpose to get range azimuth response
        rng_az_response_dbs = np.transpose(az_rng_response_dbs)

        return rng_az_response_dbs
    
    
    def process_dbs_enhanced(self,
                adc_cube: np.ndarray,
                velocity_ned:np.ndarray,
                rx_antennas:np.ndarray = np.array([]),
                **kwargs) -> np.ndarray:
        """Computes the DBS-enhanced Range-Angle response.

        Args:
            adc_cube (np.ndarray): The ADC data cube of complex data, shape (rx antennas, adc samples, num chirps).
            velocity_ned (np.ndarray): The (x, y, z) velocity of the platform in the NED coordinate frame.
            rx_antennas (np.ndarray | list, optional): Array or list of specific RX antenna indices to use. Defaults to empty array.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The DBS-enhanced Range-Azimuth response, shape (range bins, angle bins).
        """

        #change the mesh grid to the dbs enhanced mesh grid
        self.compute_mesh_grid_dbs_enhanced()

        # Convert list to numpy array if necessary
        if isinstance(rx_antennas, list):
            rx_antennas = np.array(rx_antennas)

        #screen for specific rx antennas if desired
        if rx_antennas.size > 0:
            adc_cube = adc_cube[rx_antennas, :, :]

        #compute complex windowed 3D FFT response
        angle_rng_dop_resp = self.compute_3d_windowed_fft(adc_cube)

        #convert to magnitude
        angle_rng_dop_resp_mag = np.abs(angle_rng_dop_resp)

        #perform doppler beam sharpening
        return self.perform_dbs_sharpen(
            velocity_ned=velocity_ned,
            angle_rng_dop_resp_mag=angle_rng_dop_resp_mag
        )



    def process(self,
                adc_cube: np.ndarray,
                velocity_ned:np.ndarray,
                rx_antennas:np.ndarray = np.array([]),
                chirp_idx:int = 0,
                **kwargs) -> np.ndarray:
        """Computes the Range-Angle response, using DBS if the platform velocity is high enough.

        If the platform's horizontal velocity is less than `self.min_vel_dbs`, it falls back
        to the standard Range-Angle response computation.

        Args:
            adc_cube (np.ndarray): The ADC data cube of complex data, shape (rx antennas, adc samples, num chirps).
            velocity_ned (np.ndarray): The (x, y, z) velocity of the platform in the NED coordinate frame.
            rx_antennas (np.ndarray | list, optional): Array or list of specific RX antenna indices to use. Defaults to empty array.
            **kwargs: Additional keyword arguments.

        Returns:
            np.ndarray: The computed Range-Azimuth response, shape (range bins, angle bins).
        """

        if np.linalg.norm(velocity_ned[0:2]) < self.min_vel_dbs:
            return self.process_no_dbs(
                adc_cube=adc_cube,
                chirp_idx=chirp_idx,
                rx_antennas=rx_antennas,
                **kwargs
            )
        else:
            return self.process_dbs_enhanced(
                adc_cube=adc_cube,
                velocity_ned=velocity_ned,
                rx_antennas=rx_antennas,
                **kwargs
            )
    