import numpy as np
from sklearn.linear_model import LinearRegression,RANSACRegressor
from sklearn.metrics import r2_score
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
                [np.deg2rad(-70), np.deg2rad(70)]),
            peak_threshold_dB: float = 30.0,
            x_measurement_only: bool = False,
            min_R2_threshold: float = 0.6,
            min_inlier_percent: float = 0.75
        ) -> None:
        """
        Initialize the VelocityEstimator class.

        Args:
            config_manager (ConfigManager): Radar configuration manager for accessing radar parameters.
            lower_range_bound (float): Lower bound of the range window below a given altitude (in meters).
            upper_range_bound (float): Upper bound of the range window above a given altitude (in meters).
            precise_vel_bound (float, optional): Velocity range for precise FFT computation (in m/s). Defaults to 0.25.
            valid_angle_range (np.ndarray, optional): Valid angle range for processing in radians. Defaults to np.array([np.deg2rad(-70), np.deg2rad(70)]).
            peak_threshold_dB (float, optional): Threshold for peak detection in Doppler-azimuth responses (in dB). Defaults to 30.0.
            x_measurement_only (bool, optional): If True, only estimate velocity in the x-direction. Defaults to False.
            min_R2_threshold (float, optional): Minimum R-squared value required for valid velocity estimation. Defaults to 0.6.
            min_inlier_percent (float, optional): Minimum percentage of inliers required for robust velocity estimation. Defaults to 0.75.
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
        self.azimuth_response_mag = np.empty(shape=0)
        self.az_resps_mag = []
        self.elevation_response_mag = np.empty(shape=0)
        self.el_resps_mag = []

        self.precise_azimuth_response_mag = np.empty(shape=0)
        self.precise_elevation_response_mag = np.empty(shape=0)

        #peaks
        self.peak_threshold_dB = peak_threshold_dB
        self.azimuth_peaks:np.ndarray = np.empty(shape=0)
        self.azimuth_peak_zero_az:np.ndarray = np.empty(shape=0)
        self.elevation_peaks:np.ndarray = np.empty(shape=0)
        self.elevation_peak_zero_az:np.ndarray = np.empty(shape=0)

        #velocity measurements (if y-measurement is desired)
        self.x_measurement_only = x_measurement_only

        #velocity estimates and residuals
        self.min_R2_threshold = min_R2_threshold
        self.min_inlier_percent = min_inlier_percent
        self.ego_vx_estimate:float = -1.0 #estimated velocity in the x direction of the array (0 azimuth/elevation)
        self.azimuth_ego_vy_estimate:float = 0.0
        self.azimuth_estimate_R2:float = 0.0
        self.azimuth_inlier_percent:float = 0.0
        self.elevation_ego_vy_estimate:float = 0.0
        self.elevation_estimate_R2:float = 0.0
        self.elevation_inlier_percent:float = 0.0
        self.proposed_velocity_estimate:np.ndarray = np.empty(shape=0)
        self.current_velocity_estimate:np.ndarray = np.array([0.0,0.0,0.0])

        #extra histories for debugging
        self.history_R2_statistics = []
        self.history_inlier_statistics = []
    
    def reset(self):

        self.history_R2_statistics = []
        self.history_inlier_statistics = []
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

        self.history_R2_statistics.append(
            np.array([self.azimuth_estimate_R2, self.elevation_estimate_R2])
        )

        self.history_inlier_statistics.append(
            np.array([self.azimuth_inlier_percent,
            self.elevation_inlier_percent])
        )

        return super().update_history(
            estimated=estimated,
            ground_truth=ground_truth
        )


    def get_range_window(
            self,
            altitude: float = 0.0,
            sensing_direction:str = "down") -> np.ndarray:
        """
        Get the range window for computing the Doppler-azimuth response.

        Args:
            altitude (float): Altitude around which the range window is centered.
            sensing_direction (str): Direction of sensing, either "down" or "out". This is used to determine the range window.
    
        Returns:
            np.ndarray: Range window limits as [min_range, max_range].
        """

        if sensing_direction == "down":
            return np.array([
                max(0, altitude - self.lower_range_bound),
                min(self.config_manager.range_max_m, altitude + self.upper_range_bound)
            ])
        elif sensing_direction == "out":
            return np.array([ #TODO:update this to make it more precise
                1.0,
                self.config_manager.range_max_m
            ])

    def compute_azimuth_response(
            self,
            adc_cube:np.ndarray,
            range_window:np.ndarray,
            use_precise_fft: bool = False,
            precise_fft_center_vel: float = 0.0):


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
            self.az_resps_mag = [resp_1, resp_2]

            az_resp_mag = (resp_1 + resp_2) / 2

        if use_precise_fft:
            self.precise_azimuth_response_mag = az_resp_mag
        else:
            self.azimuth_response_mag = az_resp_mag
    
    def compute_elevation_response(
            self,
            adc_cube:np.ndarray,
            range_window:np.ndarray,
            use_precise_fft: bool = False,
            precise_fft_center_vel: float = 0.0):

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
            self.el_resps_mag = [resp_1, resp_2]

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
            if self.precise_azimuth_response_mag.shape[0] > 0:
                self.azimuth_peaks = self.detect_peaks_rows(
                    doppler_azimuth_resp_mag=self.precise_azimuth_response_mag,
                    vel_bins=self.zoomed_vel_bins,
                    min_threshold_dB=self.peak_threshold_dB
                )
            if self.precise_elevation_response_mag.shape[0] > 0:
                self.elevation_peaks = self.detect_peaks_rows(
                    doppler_azimuth_resp_mag=self.precise_elevation_response_mag,
                    vel_bins=self.zoomed_vel_bins,
                    min_threshold_dB=self.peak_threshold_dB
                )
        else:
            if self.azimuth_response_mag.shape[0] > 0:
                self.azimuth_peaks = self.detect_peaks_rows(
                    doppler_azimuth_resp_mag=self.azimuth_response_mag,
                    vel_bins=self.vel_bins,
                    min_threshold_dB=self.peak_threshold_dB
                )
            if self.elevation_response_mag.shape[0] > 0:
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
            if self.precise_azimuth_response_mag.shape[0] > 0:
                self.azimuth_peak_zero_az = self.detect_peak_zero_az(
                    doppler_azimuth_resp_mag=self.precise_azimuth_response_mag,
                    vel_bins=self.zoomed_vel_bins,
                    min_threshold_dB=self.peak_threshold_dB
                )

            if self.precise_elevation_response_mag.shape[0] > 0:
                self.elevation_peak_zero_az = self.detect_peak_zero_az(
                    doppler_azimuth_resp_mag=self.precise_elevation_response_mag,
                    vel_bins=self.zoomed_vel_bins,
                    min_threshold_dB=self.peak_threshold_dB
                )
        else:
            if self.azimuth_response_mag.shape[0] > 0:
                self.azimuth_peak_zero_az = self.detect_peak_zero_az(
                    doppler_azimuth_resp_mag=self.azimuth_response_mag,
                    vel_bins=self.vel_bins,
                    min_threshold_dB=self.peak_threshold_dB
                )
            if self.elevation_response_mag.shape[0] > 0:
                self.elevation_peak_zero_az = self.detect_peak_zero_az(
                    doppler_azimuth_resp_mag=self.elevation_response_mag,
                    vel_bins=self.vel_bins,
                    min_threshold_dB=self.peak_threshold_dB
                )
        
        return
    

    def lsq_fit_ego_vy_ransac(self, peaks: np.ndarray):
        """Estimate vy component of velocity using RANSAC-based robust linear regression.
        
        Args:
            peaks (np.ndarray): Nx2 array of (angle, velocity) peaks.

        Returns:
            float: Estimated ego vy velocity.
        """

        if self.ego_vx_estimate >= 0.1:
            return self.lsq_fit_ego_vy_ransac_standard(peaks=peaks)
        else:
            return self.lsq_fit_ego_vy_ransac_small_vx(peaks=peaks)
    
    def lsq_fit_ego_vel_ransac_points(self, points: np.ndarray=np.empty(shape=0)):
        """Estimate vx and vy component of velocity using RANSAC-based robust linear regression.
        
        Args:
            peaks (np.ndarray): Nx4 array of (x,y,z, velocity) peaks.

        Returns:
            np.ndarray: Estimated vx,vy of the UAV.
        """

        if points.shape[0] == 0:
            return np.array([0.0, 0.0])

        # Compute target values
        y = -1 * points[:,3]
        H = points[:, 0:2] / np.linalg.norm(points[:, 0:2], axis=1, keepdims=True)

        # Use RANSAC to fit
        model = RANSACRegressor(
            estimator=LinearRegression(
                fit_intercept=False
            ),
            residual_threshold=0.15, #originally 0.1
            random_state=42,
            max_trials=20,
            min_samples=10)
        try:
            model.fit(H, y)
            ego_vel = model.estimator_.coef_

            #get the score if possible
            inliers = model.inlier_mask_
            if inliers.sum() > 3:
                r2_inliers = model.score(H[inliers], y[inliers])
            else:
                r2_inliers = 0.0

            # Compute the ratio of inliers to outliers
            inlier_count = inliers.sum()
            inlier_outlier_ratio = (inlier_count / len(inliers)) if len(inliers) > 0 else 0.0
        except ValueError:
            # If model fails (e.g., too few inliers), fall back to 0
            ego_vel = np.array([0.0, 0.0])
            r2_inliers = 0.0
            inlier_outlier_ratio = 0.0

        return ego_vel,r2_inliers, inlier_outlier_ratio

    def lsq_fit_ego_vy_ransac_standard(self, peaks: np.ndarray):
        """Estimate vy component of velocity using RANSAC-based robust linear regression.
        
        Args:
            peaks (np.ndarray): Nx2 array of (angle, velocity) peaks.

        Returns:
            float: Estimated ego vy velocity.
        """

        if peaks is None or len(peaks) == 0:
            return 0.0

        # Compute target values
        y = -1 * peaks[:, 1] - self.ego_vx_estimate * np.cos(peaks[:, 0])  # shape (N,)
        H = np.sin(peaks[:, 0]).reshape(-1, 1)  # shape (N, 1)

        # Use RANSAC to fit
        model = RANSACRegressor(
            estimator=LinearRegression(
                fit_intercept=False
            ),
            residual_threshold=0.15, #originally 0.1
            random_state=42,
            max_trials=20,
            min_samples=10)
        try:
            model.fit(H, y)
            ego_vy = model.estimator_.coef_[0]

            #get the score if possible
            inliers = model.inlier_mask_
            if inliers.sum() > 3:
                r2_inliers = model.score(H[inliers], y[inliers])
            else:
                r2_inliers = 0.0

            # Compute the ratio of inliers to outliers
            inlier_count = inliers.sum()
            inlier_outlier_ratio = (inlier_count / len(inliers)) if len(inliers) > 0 else 0.0
        except ValueError:
            # If model fails (e.g., too few inliers), fall back to 0
            ego_vy = 0.0
            r2_inliers = 0.0
            inlier_outlier_ratio = 0.0

        return ego_vy,r2_inliers, inlier_outlier_ratio
    
    def lsq_fit_ego_vy_ransac_small_vx(self,peaks:np.ndarray):
        """Estimate vy component of velocity using RANSAC-based robust linear regression.
        This is a special case for when the ego vx is small, and we can use a simpler model.

        Args:
            peaks (np.ndarray): Nx2 Detected (angle,velocity) peaks in the velocity response.

        Returns:
            tuple: Estimated ego vy velocity and rss.
        """

        if peaks is None or len(peaks) == 0:
            ego_vy = 0.0
            return ego_vy, 0.0
        
        #linearly solving theta = -(1/vy)vd - (vx/vy)
        # = -(-1/vy) (vd - vx)
        y = peaks[:,0]
        H = (peaks[:,1] - self.ego_vx_estimate).reshape(-1,1) # shape (N,1) -- if vx known
        # H = (peaks[:,1]).reshape(-1,1) # shape (N,1) -- if vx unknown

        model = RANSACRegressor(
            estimator=LinearRegression(
                fit_intercept=False
            ),
            residual_threshold=0.20,
            random_state=42,
            max_trials=20,
            min_samples=10)
        
        try:
            model.fit(H, y)
            a = model.estimator_.coef_[0]
            # b = model.estimator_.intercept_

            ego_vy = -1 / a if a != 0 else 0.0
            # ego_vx = -1 * b * ego_vy if b != 0 else 0.0 # if vx unknown
            r2_inliers = model.score(H[model.inlier_mask_], y[model.inlier_mask_])

        # Compute the ratio of inliers to outliers
            inliers = model.inlier_mask_
            inlier_count = inliers.sum()
            inlier_outlier_ratio = (inlier_count / len(inliers)) if len(inliers) > 0 else 0.0
        except ValueError:
            # If model fails (e.g., too few inliers), fall back to 0
            ego_vy = 0.0
            r2_inliers = 0.0
            inlier_outlier_ratio = 0.0

        return ego_vy,r2_inliers, inlier_outlier_ratio
    
    def lsq_fit_ego_vy(
            self,
            peaks:np.ndarray
    ):
        """Estimate vy component of velocity (x,y -> Forward,Left) using least squares fitting.
        Utilizes the relationship vd = -vx*cos(theta) - vy*sin(theta) and the already estimated
        vx component from the zero azimuth peak.

        Args:
            peaks (np.ndarray): Nx2 Detected (angle,velocity) peaks in the velocity response.

        Returns:
            tuple: Estimated ego vy velocity and rss.
        """

        # Check if there are any peaks
        if peaks is None or len(peaks) == 0:
            ego_vy = 0.0
            return ego_vy
        
        y = -1 * peaks[:,1] - (self.ego_vx_estimate * np.cos(peaks[:,0]))
        H = np.sin(peaks[:,0])[:,np.newaxis]

        ego_vy,_,_,_ = np.linalg.lstsq(H, y, rcond=None)
        ego_vy = ego_vy[0]

        return ego_vy
    
    def lsq_predict_velocity_measurement(
            self,
            v:np.ndarray,
            angles_to_predict:np.ndarray = np.empty(shape=0)
    ):
        """Predict velocity measurements given an ego velocity.

        Args:
            v (np.ndarray): Ego velocity [vx,vy]

        Returns:
            np.ndarray: Predicted velocity measurements for each angle bin.
        """

        if angles_to_predict.size == 0:
            angles_to_predict = self.valid_angle_bins

        angles = np.stack([np.cos(angles_to_predict), np.sin(angles_to_predict)], axis=-1)
        return -1 * np.dot(angles, v)
    
    def compute_R2_statistics(self):

        az_peaks = self.azimuth_peaks
        el_peaks = self.elevation_peaks

        latest_vel = self.proposed_velocity_estimate

        if az_peaks.shape[0] > 0:
            if self.config_manager.array_geometry == "ods":
                az_predicted_vd = self.lsq_predict_velocity_measurement(
                    v=np.array([latest_vel[2],latest_vel[0]]),
                    angles_to_predict=az_peaks[:,0]
                )
            elif self.config_manager.array_geometry == "standard":
                az_predicted_vd = self.lsq_predict_velocity_measurement(
                    v=np.array([latest_vel[0],latest_vel[1]]),
                    angles_to_predict=az_peaks[:,0]
                )
            #compute RSS
            az_rss = np.sum((az_peaks[:,1] - az_predicted_vd)**2)
            #compute TSS
            az_tss = np.sum((az_peaks[:,1] - np.mean(az_peaks[:,1]))**2)
            #compute R2
            self.azimuth_estimate_R2 = 1 - az_rss/az_tss if az_tss > 0 else 0


        if el_peaks.shape[0] > 0:
            if self.config_manager.array_geometry == "ods":
                el_predicted_vd = self.lsq_predict_velocity_measurement(
                    v=np.array([latest_vel[2],latest_vel[1]]),
                    angles_to_predict=el_peaks[:,0]
                )
            else:
                raise NotImplementedError("Elevation R2 computation is only implemented for ODS array geometry.")
            #compute the RSS
            el_rss = np.sum((el_peaks[:,1] - el_predicted_vd)**2)
            #compute the TSS
            el_tss = np.sum((el_peaks[:,1] - np.mean(el_peaks[:,1]))**2)
            #print R2
            self.elevation_estimate_R2 = 1 - el_rss/el_tss if el_tss > 0 else 0
    
    def update_and_check_current_vel_measurements(self):
        """Update the current velocity estimates if the proposed estimates are valid.
        TODO: Find a better filtering method to improve robustness
        """

        if self.x_measurement_only:
            self.current_velocity_estimate[0] = self.proposed_velocity_estimate[0]
        else:
            if self.config_manager.array_geometry == "ods":
                #check the x estimate
                if self.azimuth_estimate_R2 >= self.min_R2_threshold and\
                    self.azimuth_inlier_percent >= self.min_inlier_percent:
                    self.current_velocity_estimate[0] = self.proposed_velocity_estimate[0]
                else:
                    self.current_velocity_estimate[0] = 0.0
                
                #check the y estimate
                if self.elevation_estimate_R2 >= self.min_R2_threshold and \
                    self.elevation_inlier_percent >= self.min_inlier_percent:
                    self.current_velocity_estimate[1] = self.proposed_velocity_estimate[1]
                else:
                    self.current_velocity_estimate[1] = 0.0
            
                #update using the z estimate
                self.current_velocity_estimate[2] = self.proposed_velocity_estimate[2]
            elif self.config_manager.array_geometry == "standard":
                if self.ego_vx_estimate < 0.0: #used points to estimate vel
                    if self.azimuth_estimate_R2 >= self.min_R2_threshold:
                        self.current_velocity_estimate = self.proposed_velocity_estimate
                    else:
                        self.current_velocity_estimate = np.array([0.0, 0.0, 0.0])
                else: #used adc cube processing
                    #check the x estimate
                    if self.azimuth_estimate_R2 >= self.min_R2_threshold:
                        self.current_velocity_estimate[1] = self.proposed_velocity_estimate[1]
                    else:
                        self.current_velocity_estimate[1] = 0.0
                    
                    #update using the y estimate
                    self.current_velocity_estimate[0] = self.proposed_velocity_estimate[0]
                    #set z to zero
                    self.current_velocity_estimate[2] = 0.0

    def estimate_ego_vx_velocity(self):
        """Estimate ego velocity in the x direction of the radar(i.e. at zero doppler/elevation).

        Returns:
            float: Estimated ego velocity in the x direction.
        """
        
        if self.azimuth_peak_zero_az.shape[0] > 0 and\
              self.elevation_peak_zero_az.shape[0] > 0:
                az_peak = self.azimuth_peak_zero_az
                el_peak = self.elevation_peak_zero_az
                self.ego_vx_estimate = -1 * (az_peak[1] + el_peak[1]) / 2.0
        elif self.azimuth_peak_zero_az.shape[0] > 0:
            az_peak = self.azimuth_peak_zero_az
            self.ego_vx_estimate = -1 * az_peak[1]
        elif self.elevation_peak_zero_az.shape[0] > 0:
            el_peak = self.elevation_peak_zero_az
            self.ego_vx_estimate = -1 * el_peak[1]
        else:
            self.ego_vx_estimate = 0.0

        return self.ego_vx_estimate
    
    def estimate_ego_velocity_adc_data(
            self):
        """Estimate ego velocity from the detected peaks obtained from processing doppler-azimuth responses
        """

        if not self.x_measurement_only:
            if self.azimuth_peaks.shape[0] > 0:
                self.azimuth_ego_vy_estimate,self.azimuth_estimate_R2,self.azimuth_inlier_percent = \
                    self.lsq_fit_ego_vy_ransac(
                        peaks=self.azimuth_peaks
                    )
            if self.elevation_peaks.shape[0] > 0:
                self.elevation_ego_vy_estimate,self.elevation_estimate_R2,self.elevation_inlier_percent = \
                    self.lsq_fit_ego_vy_ransac(
                        peaks=self.elevation_peaks
                    )
            if self.config_manager.array_geometry == "ods":
                self.proposed_velocity_estimate = np.array([
                    self.azimuth_ego_vy_estimate,
                    self.elevation_ego_vy_estimate,
                    self.ego_vx_estimate
                ])
            elif self.config_manager.array_geometry == "standard":
                self.proposed_velocity_estimate = np.array([
                    self.ego_vx_estimate,
                    self.azimuth_ego_vy_estimate,
                    0.0
                ])
        else:
            self.proposed_velocity_estimate = np.array([self.ego_vx_estimate])

    def estimate_ego_velocity_points(
            self,
            points:np.ndarray = np.empty(shape=0)):
        """Estimate ego velocity from the detected peaks obtained from processing doppler-azimuth responses
        """

        if points.shape[0] >= 0:
            if self.config_manager.array_geometry == "standard":
                vel_est, self.azimuth_estimate_R2, self.azimuth_inlier_percent = \
                    self.lsq_fit_ego_vel_ransac_points(
                        points=points
                    )
                
                if self.x_measurement_only:
                    self.proposed_velocity_estimate = np.array([vel_est[0]])
                else:
                    try:
                        self.proposed_velocity_estimate = np.array([
                            vel_est[0], #vx
                            vel_est[1], #vy
                            0.0 #vz
                        ])
                    except IndexError:
                        print("caught issue")
            elif self.config_manager.array_geometry == "ods":
                raise NotImplementedError("Velocity estimation from points is not implemented for ODS array geometry currently.")
        
        return

    def get_gt_velocity_measurement_predictions(
            self,
            direction: str = "azimuth"):
        
        if len(self.history_gt) == 0:
            return np.empty(shape=(0))
        else:
            latest_gt_vel = self.history_gt[-1]
        
        if self.config_manager.array_geometry == "ods":
            if direction == "azimuth":
                return self.lsq_predict_velocity_measurement(
                    v=np.array([-1*latest_gt_vel[2],latest_gt_vel[1]])
                )
            elif direction == "elevation":
                return self.lsq_predict_velocity_measurement(
                    v=np.array([-1 * latest_gt_vel[2],latest_gt_vel[0]])
                )
            else:
                raise ValueError("Direction must be either 'azimuth' or 'elevation'")
        elif self.config_manager.array_geometry == "standard":
            if direction == "azimuth":
                return self.lsq_predict_velocity_measurement(
                    v=np.array([latest_gt_vel[0],latest_gt_vel[1]])
                )
            elif direction == "elevation":
                raise NotImplementedError("Elevation GT prediction is not implemented for standard array geometry.")
            else:
                raise ValueError("Direction must be either 'azimuth' or 'elevation'")
    
    def get_estimated_velocity_measurement_predictions(
            self,
            direction: str = "azimuth"):
        
        if len(self.proposed_velocity_estimate) == 0:
            return np.empty(shape=(0))
        else:
            latest_est_vel = self.current_velocity_estimate
        
        if self.config_manager.array_geometry == "ods":
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
        elif self.config_manager.array_geometry == "standard":
            if direction == "azimuth":
                return self.lsq_predict_velocity_measurement(
                    v=np.array([latest_est_vel[0],latest_est_vel[1]])
                )
            elif direction == "elevation":
                raise NotImplementedError("Elevation estimation is not implemented for standard array geometry.")
            else:
                raise ValueError("Direction must be either 'azimuth' or 'elevation'")
    


    def process(
            self,
            adc_cube: np.ndarray = np.empty(shape=0),
            points: np.ndarray = np.empty(shape=0),
            altitude: float = 0.0,
            enable_precise_responses: bool = False) -> np.ndarray:
        """
        Compute the velocity response over a range window centered around a given altitude.

        Args:
            adc_cube (np.ndarray): ADC cube indexed by [rx, samp, chirp].
            points (np.ndarray): Point cloud data.
            altitude (float): Altitude around which the range window is centered.
            enable_precise_responses (bool): If True, additionally compute precise responses for azimuth and elevation.
        Returns:
            np.ndarray: velocity estimate [vx,vy,vz]
        """
        if adc_cube.shape[0] > 0:
            #compute the range window based on the altitude:
            range_window = self.get_range_window(
                altitude=altitude,
                sensing_direction=self.config_manager.array_direction)
            
            #compute the responses and identify peaks in them
            self.compute_azimuth_response(
                adc_cube=adc_cube,
                range_window=range_window,
                use_precise_fft=False)
            
            if self.config_manager.array_geometry == "ods":
                self.compute_elevation_response(
                    adc_cube=adc_cube,
                    range_window=range_window,
                    use_precise_fft=False)
            
            #estimate vx (corresponding to zero doppler) using the coarse doppler_azimuth_responses
            self.detect_vel_zero_az_peaks(use_precise_response=False)
            self.estimate_ego_vx_velocity()

            if enable_precise_responses:
                self.compute_azimuth_response(
                    adc_cube=adc_cube,
                    range_window=range_window,
                    use_precise_fft=True,
                    precise_fft_center_vel=-1 * self.ego_vx_estimate)
                
                if self.config_manager.array_geometry == "ods":
                    self.compute_elevation_response(
                        adc_cube=adc_cube,
                        range_window=range_window,
                        use_precise_fft=True,
                        precise_fft_center_vel=-1 * self.ego_vx_estimate)

                #re-estimate ego vx using the precise doppler_azimuth_responses
                self.detect_vel_zero_az_peaks(use_precise_response=True)
                self.estimate_ego_vx_velocity()

            #estimate the velocity
            if not self.x_measurement_only:
                self.detect_vel_row_peaks(use_precise_response=enable_precise_responses)
                
            #estimate the velocity from the detected peaks in the respective responses
            self.estimate_ego_velocity_adc_data()
        
        elif points.shape[0] > 0:

            #estimate the velocity from the detected points
            self.estimate_ego_velocity_points(points=points)

        #check the measurements
        # self.compute_R2_statistics()
        self.update_and_check_current_vel_measurements()
        

        return self.current_velocity_estimate