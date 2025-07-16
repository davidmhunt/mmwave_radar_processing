import numpy as np
from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.doppler_azimuth_resp import DopplerAzimuthProcessor

class VelocityEstimator(DopplerAzimuthProcessor):
    """Estimate ego velocity using Doppler-azimuth processing.
    Note the coordinate system of the Doppler-azimuth processing x-forward, y-left

    Args:
        DopplerAzimuthProcessor (_type_): _description_
    """
    def __init__(
            self,
            config_manager: ConfigManager,
            lower_range_bound: float,
            upper_range_bound: float,
            precise_vel_bound: float = 0.25,
            valid_angle_range: np.ndarray = np.array(
                [np.deg2rad(-60), np.deg2rad(60)]),
            peak_threshold_dB: float = 30.0,
            min_velocity_mag:float = 0.1,
            max_residual:float = 0.1
        ) -> None:
        """
        Args:
            config_manager (ConfigManager): Radar configuration manager.
            lower_range_bound (float): Range window size below a given altitude.
            upper_range_bound (float): Range window size above a given altitude.
            valid_angle_range (np.ndarray, optional): Valid angle range in radians. Defaults to np.array([np.deg2rad(-60), np.deg2rad(60)]).
            peak_threshold_dB (float, optional): Peak detection threshold in dB. Defaults to 30.0.
            min_velocity_mag (float, optional): Minimum velocity magnitude to consider for estimation. Defaults to 0.1 m/s.
        """
        super().__init__(
            config_manager=config_manager,
            num_angle_bins=64,
            valid_angle_range=valid_angle_range)

        #limits for computing the doppler-azimuth response at specific ranges
        self.upper_range_bound = upper_range_bound
        self.lower_range_bound = lower_range_bound

        #limits for precise FFT computation
        self.precise_vel_bound = precise_vel_bound

        #most recent doppler-azimuth response
        self.azimuth_response_mag = None
        self.elevation_response_mag = None

        self.precise_azimuth_response_mag = None
        self.precise_elevation_response_mag = None

        #peaks
        self.peak_threshold_dB = peak_threshold_dB
        self.azimuth_peaks:np.ndarray = None
        self.azimuth_peak_zero_az:np.ndarray = None
        self.elevation_peaks:np.ndarray = None
        self.elevation_peak_zero_az:np.ndarray = None

        #velocity estimates and residuals
        self.min_vel_mag = min_velocity_mag
        self.max_residual = max_residual
        self.vx_estimate:float = 0.0 #estimated velocity in the x direction of the array (0 azimuth/elevation)
        self.azimuth_velocity_estimate:np.ndarray = np.empty(shape=0)
        self.azimuth_velocity_residual:float = np.empty(shape=0)
        self.elevation_velocity_estimate:np.ndarray = np.empty(shape=0)
        self.elevation_velocity_residual:float = np.empty(shape=0)
        self.estimated_velocity:np.ndarray = np.empty(shape=0)

        #extra histories for debugging
        self.history_residuals = []
    
    def reset(self):

        self.history_residuals = []
        return super().reset()
    
    
    def update_history(
            self,
            estimated:np.ndarray=np.empty(0),
            ground_truth:np.ndarray=np.empty(0)) -> None:
        """Update the internal history of estimated and ground truth values

        Args:
            estimated (np.ndarray): 1D array of estimated values.
            ground_truth (np.ndarray): 1D array of ground truth values.
        """

        if self.azimuth_velocity_residual.size > 0 and self.elevation_velocity_residual.size > 0:   
            self.history_residuals.append(
                np.array([self.azimuth_velocity_residual[0], self.elevation_velocity_residual[0]])
            )
        else:
            self.history_residuals.append(np.array([0.0,0.0]))

        return super().update_history(
            estimated=estimated,
            ground_truth=ground_truth
        )


    def get_range_window(self,altitude: float = 0.0) -> np.ndarray:
        """
        Get the range window for computing the Doppler-azimuth response.

        Args:
            altitude (float): Altitude around which the range window is centered.

        Returns:
            np.ndarray: Range window limits as [min_range, max_range].
        """
        return np.array([
            max(0, altitude - self.lower_range_bound),
            min(self.config_manager.range_max_m, altitude + self.upper_range_bound)
        ])

    def compute_azimuth_response(
            self,
            adc_cube:np.ndarray,
            altitude:float = 0.0,
            use_precise_fft: bool = False,
            precise_fft_center_vel: float = 0.0):

        range_window = self.get_range_window(altitude=altitude)

        precise_vel_range = np.array([
            precise_fft_center_vel - self.precise_vel_bound,
            precise_fft_center_vel + self.precise_vel_bound
        ])


        if self.config_manager.array_geometry == "standard":
            if self.config_manager.virtual_antennas_enabled:
                rx_antennas = np.arange(8)
            else:
                rx_antennas = np.arange(4)

            az_resp_mag = super().process(
                adc_cube=adc_cube,
                rx_antennas=rx_antennas,
                range_window=range_window,
                use_precise_fft=use_precise_fft,
                precise_vel_range=precise_vel_range
            )
        elif self.config_manager.array_geometry == "ods":
            if self.config_manager.virtual_antennas_enabled:
                rx_antennas_1 = np.array([0,3,4,7])
                rx_antennas_2 = np.array([1,2,5,6])
            else:
                rx_antennas_1 = np.array([0, 3])
                rx_antennas_2 = np.array([1, 2])
            
            resp_1 = super().process(
                adc_cube=adc_cube,
                rx_antennas=rx_antennas_1,
                range_window=range_window,
                shift_angle=True,
                use_precise_fft=use_precise_fft,
                precise_vel_range=precise_vel_range
            )

            resp_2 = super().process(
                adc_cube=adc_cube,
                rx_antennas=rx_antennas_2,
                range_window=range_window,
                shift_angle=True,
                use_precise_fft=use_precise_fft,
                precise_vel_range=precise_vel_range
            )

            az_resp_mag = (resp_1 + resp_2) / 2

        if use_precise_fft:
            self.precise_azimuth_response_mag = az_resp_mag
        else:
            self.azimuth_response_mag = az_resp_mag
    
    def compute_elevation_response(
            self,
            adc_cube:np.ndarray,
            altitude:float = 0.0,
            use_precise_fft: bool = False,
            precise_fft_center_vel: float = 0.0):

        range_window = self.get_range_window(altitude=altitude)

        precise_vel_range = np.array([
            precise_fft_center_vel - self.precise_vel_bound,
            precise_fft_center_vel + self.precise_vel_bound
        ])

        if self.config_manager.array_geometry == "standard":
            raise NotImplementedError(
                "Elevation response computation is not implemented for standard array geometry."
            )
        elif self.config_manager.array_geometry == "ods":
            if self.config_manager.virtual_antennas_enabled:
                rx_antennas_1 = np.array([10,11,6,7])
                rx_antennas_2 = np.array([9,8,5,4])
            else:
                rx_antennas_1 = np.array([1, 0])
                rx_antennas_2 = np.array([3, 4])
            
            resp_1 = super().process(
                adc_cube=adc_cube,
                rx_antennas=rx_antennas_1,
                range_window=range_window,
                shift_angle=False,
                use_precise_fft=use_precise_fft,
                precise_vel_range=precise_vel_range
            )

            resp_2 = super().process(
                adc_cube=adc_cube,
                rx_antennas=rx_antennas_2,
                range_window=range_window,
                shift_angle=False,
                use_precise_fft=use_precise_fft,
                precise_vel_range=precise_vel_range
            )

            elevation_response_mag = (resp_1 + resp_2) / 2
        
        if use_precise_fft:
            self.precise_elevation_response_mag = elevation_response_mag
        else:
            self.elevation_response_mag = elevation_response_mag
    
    def detect_vel_row_peaks(
            self,
            use_precise_response:bool = False
    ):
        """Detect peaks in the azimuth and elevation responses.

        Args:
            use_precise_response (bool, optional): If True, use the precise (zoom FFT) responses for peak detection. Defaults to False.
        """

        if use_precise_response:
            if self.precise_azimuth_response_mag is not None:
                self.azimuth_peaks = self.detect_peaks_rows(
                    doppler_azimuth_resp_mag=self.precise_azimuth_response_mag,
                    vel_bins=self.zoomed_vel_bins,
                    min_threshold_dB=self.peak_threshold_dB
                )
            if self.precise_elevation_response_mag is not None:
                self.elevation_peaks = self.detect_peaks_rows(
                    doppler_azimuth_resp_mag=self.precise_elevation_response_mag,
                    vel_bins=self.zoomed_vel_bins,
                    min_threshold_dB=self.peak_threshold_dB
                )
        else:
            if self.azimuth_response_mag is not None:
                self.azimuth_peaks = self.detect_peaks_rows(
                    doppler_azimuth_resp_mag=self.azimuth_response_mag,
                    vel_bins=self.vel_bins,
                    min_threshold_dB=self.peak_threshold_dB
                )
            if self.elevation_response_mag is not None:
                self.elevation_peaks = self.detect_peaks_rows(
                    doppler_azimuth_resp_mag=self.elevation_response_mag,
                    vel_bins=self.vel_bins,
                    min_threshold_dB=self.peak_threshold_dB
                )
        
        return
    
    def detect_vel_zero_az_peaks(
            self,
            use_precise_response:bool = False
    ):
        """Detect the peak closest to zero azimuth in the azimuth and elevation responses.

        Args:
            use_precise_response (bool, optional): If True, use the precise (zoom FFT) responses for peak detection. Defaults to False.
        """

        if use_precise_response:
            if self.precise_azimuth_response_mag is not None:
                self.azimuth_peak_zero_az = self.detect_peak_zero_az(
                    doppler_azimuth_resp_mag=self.precise_azimuth_response_mag,
                    vel_bins=self.zoomed_vel_bins,
                    min_threshold_dB=self.peak_threshold_dB
                )
            if self.precise_elevation_response_mag is not None:
                self.elevation_peak_zero_az = self.detect_peak_zero_az(
                    doppler_azimuth_resp_mag=self.precise_elevation_response_mag,
                    vel_bins=self.zoomed_vel_bins,
                    min_threshold_dB=self.peak_threshold_dB
                )
        else:
            if self.azimuth_response_mag is not None:
                self.azimuth_peak_zero_az = self.detect_peak_zero_az(
                    doppler_azimuth_resp_mag=self.azimuth_response_mag,
                    vel_bins=self.vel_bins,
                    min_threshold_dB=self.peak_threshold_dB
                )
            if self.elevation_response_mag is not None:
                self.elevation_peak_zero_az = self.detect_peak_zero_az(
                    doppler_azimuth_resp_mag=self.elevation_response_mag,
                    vel_bins=self.vel_bins,
                    min_threshold_dB=self.peak_threshold_dB
                )
        
        return
    

    def lsq_fit_ego_velocity(
            self,
            peaks:np.ndarray
    ):
        """Estimate ego velocity (x,y -> Forward,Left) using least squares fitting.

        Args:
            peaks (np.ndarray): Nx2 Detected (angle,velocity) peaks in the velocity response.

        Returns:
            tuple: Estimated ego velocity (vx, vy) and residual.
        """

        # Check if there are any peaks
        if peaks is None or len(peaks) == 0:
            v_ego = np.array([0.0, 0.0])
            residual = np.array([0.0])
            return v_ego, residual

        # Check if there is at least one peak with a minimum velocity magnitude
        if np.all(np.abs(peaks[:, 1]) < self.min_vel_mag):
            v_ego = np.array([0.0, 0.0])
            residual = np.array([0.0])
            return v_ego, residual
        
        y = -1 * peaks[:,1]
        H = np.stack([np.cos(peaks[:,0]), np.sin(peaks[:,0])], axis=1)

        v_ego,residual,_,_ = np.linalg.lstsq(H, y, rcond=None)

        if residual.size == 0:
            residual = np.array([0.0])
            v_ego = np.array([0.0, 0.0])
        elif residual > self.max_residual:
            v_ego = np.array([0.0, 0.0])
            residual = np.array([residual[0]])

        return v_ego, residual
    
    def lsq_predict_velocity_measurement(
            self,
            v:np.ndarray
    ):
        """Predict velocity measurements given an ego velocity.

        Args:
            v (np.ndarray): Ego velocity [vx,vy]

        Returns:
            np.ndarray: Predicted velocity measurements for each angle bin.
        """

        angles = np.stack([np.cos(self.valid_angle_bins), np.sin(self.valid_angle_bins)], axis=-1)
        return -1 * np.dot(angles, v)

    def estimate_ego_vx_velocity(self):
        """Estimate ego velocity in the x direction of the radar(i.e. at zero doppler/elevation).

        Returns:
            float: Estimated ego velocity in the x direction.
        """
        
        az_peak = self.azimuth_peak_zero_az
        el_peak = self.elevation_peak_zero_az
        self.vx_estimate = (az_peak[1] + el_peak[1]) / 2.0

        return self.vx_estimate
    
    def estimate_ego_velocity(
            self):
        """Estimate ego velocity from the detected peaks.
        """

        self.azimuth_velocity_estimate, self.azimuth_velocity_residual = \
            self.lsq_fit_ego_velocity(
                peaks=self.azimuth_peaks
            )

        self.elevation_velocity_estimate, self.elevation_velocity_residual = \
            self.lsq_fit_ego_velocity(
                peaks=self.elevation_peaks
            )
        
        #TODO: combine the z estimates
        self.estimated_velocity = np.array([
            self.azimuth_velocity_estimate[1],
            self.elevation_velocity_estimate[1],
            (self.azimuth_velocity_estimate[0] + self.elevation_velocity_estimate[0]) / 2
        ])

    def get_gt_velocity_measurement_predictions(
            self,
            direction: str = "azimuth"):
        
        if len(self.history_gt) == 0:
            return np.empty(shape=(0))
        else:
            latest_gt_vel = self.history_gt[-1]
        
        if direction == "azimuth":
            return self.lsq_predict_velocity_measurement(
                v=np.array([-1 * latest_gt_vel[2],latest_gt_vel[1]])
            )
        elif direction == "elevation":
            return self.lsq_predict_velocity_measurement(
                v=np.array([-1 * latest_gt_vel[2],latest_gt_vel[0]])
            )
        else:
            raise ValueError("Direction must be either 'azimuth' or 'elevation'")
    
    def get_estimated_velocity_measurement_predictions(
            self,
            direction: str = "azimuth"):
        
        if len(self.estimated_velocity) == 0:
            return np.empty(shape=(0))
        else:
            latest_est_vel = self.estimated_velocity
        
        if direction == "azimuth":
            return self.lsq_predict_velocity_measurement(
                v=np.array([latest_est_vel[2],latest_est_vel[0]])
            )
        elif direction == "elevation":
            return self.lsq_predict_velocity_measurement(
                v=np.array([latest_est_vel[2],latest_est_vel[1]])
            )
        else:
            raise ValueError("Direction must be either 'azimuth' or 'elevation'")
    


    def process(
            self,
            adc_cube: np.ndarray,
            altitude: float = 0.0,
            enable_precise_responses: bool = False) -> np.ndarray:
        """
        Compute the velocity response over a range window centered around a given altitude.

        Args:
            adc_cube (np.ndarray): ADC cube indexed by [rx, samp, chirp].
            altitude (float): Altitude around which the range window is centered.
            enable_precise_responses (bool): If True, additionally compute precise responses for azimuth and elevation.
        Returns:
            np.ndarray: velocity estimate [vx,vy,vz]
        """
        
        #compute the responses and identify peaks in them
        self.compute_azimuth_response(
            adc_cube=adc_cube,
            altitude=altitude,
            use_precise_fft=False)
        self.compute_elevation_response(
            adc_cube=adc_cube,
            altitude=altitude,
            use_precise_fft=False)
        
        #estimate vx (corresponding to zero doppler) using the coarse doppler_azimuth_responses
        self.detect_vel_zero_az_peaks(use_precise_response=False)
        self.estimate_ego_vx_velocity()

        if enable_precise_responses:
            self.compute_azimuth_response(
                adc_cube=adc_cube,
                altitude=altitude,
                use_precise_fft=True,
                precise_fft_center_vel=self.vx_estimate)
            self.compute_elevation_response(
                adc_cube=adc_cube,
                altitude=altitude,
                use_precise_fft=True,
                precise_fft_center_vel=self.vx_estimate)

            #re-estimate ego vx using the precise doppler_azimuth_responses
            self.detect_vel_zero_az_peaks(use_precise_response=True)
            self.estimate_ego_vx_velocity()

        #estimate the velocity
        self.detect_vel_row_peaks(use_precise_response=enable_precise_responses)
            
        #estimate the velocity from the detected peaks in the respective responses
        self.estimate_ego_velocity()

        return self.estimated_velocity