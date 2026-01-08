import numpy as np
from sklearn.linear_model import LinearRegression,RANSACRegressor
from sklearn.metrics import r2_score
from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors._processor import _Processor

class VelocityEstimator(_Processor):
    """Estimate ego velocity using LSQ + Ransac processing.
    Note the coordinate system of the Doppler-azimuth processing x-forward, y-left

    Args:
        DopplerAzimuthProcessor (_type_): _description_
    """
    def __init__(
            self,
            config_manager: ConfigManager,
            min_R2_threshold: float = 0.6,
            min_inlier_percent: float = 0.75,
            **kwargs) -> None:
        """
        Initialize the VelocityEstimator class.

        Args:
            config_manager (ConfigManager): Radar configuration manager for accessing radar parameters.
            min_R2_threshold (float, optional): Minimum R-squared value required for valid velocity estimation. Defaults to 0.6.
            min_inlier_percent (float, optional): Minimum percentage of inliers required for robust velocity estimation. Defaults to 0.75.
            **kwargs: Additional keyword arguments.
        """
        
        super().__init__(
            config_manager=config_manager)

        #velocity estimates and residuals
        self.min_R2_threshold = min_R2_threshold
        self.min_inlier_percent = min_inlier_percent
        self.estimated_R2:float = 0.0
        self.inlier_percent:float = 0.0
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
            self.estimated_R2
        )

        self.history_inlier_statistics.append(
            self.inlier_percent
        )   

        return super().update_history(
            estimated=estimated,
            ground_truth=ground_truth
        )
    
    def lsq_fit_ego_vel_ransac_points_2D(self, points: np.ndarray=np.empty(shape=0)):
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
    
    def lsq_fit_ego_vel_ransac_points_3D(self, points: np.ndarray=np.empty(shape=0)):
        """Estimate vx and vy component of velocity using RANSAC-based robust linear regression.
        
        Args:
            peaks (np.ndarray): Nx4 array of (x,y,z, velocity) peaks.

        Returns:
            np.ndarray: Estimated vx,vy, and vz of the UAV.
        """

        if points.shape[0] == 0:
            return np.array([0.0, 0.0, 0.0]),0.0,0.0
        
        # Compute target values
        y = -1 * points[:,3]
        H = points[:, 0:3] / np.linalg.norm(points[:, 0:3], axis=1, keepdims=True)

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
    
    def update_and_check_current_vel_measurements(self):
        """Update the current velocity estimates if the proposed estimates are valid.
        TODO: Find a better filtering method to improve robustness
        """

        if self.estimated_R2 >= self.min_R2_threshold and\
            self.inlier_percent >= self.min_inlier_percent:
                
                self.current_velocity_estimate = self.proposed_velocity_estimate
                return

    def estimate_ego_velocity_points(
            self,
            points:np.ndarray = np.empty(shape=0)):
        """Estimate ego velocity from the detected peaks obtained from processing doppler-azimuth responses
        """

        if points.shape[0] >= 0:
            if self.config_manager.array_geometry == "standard":
                vel_est, self.estimated_R2, self.inlier_percent = \
                    self.lsq_fit_ego_vel_ransac_points_2D(
                        points=points
                    )
                try:
                    self.proposed_velocity_estimate = np.array([
                        vel_est[0], #vx
                        vel_est[1], #vy
                        0.0 #vz
                    ])
                except IndexError:
                    print("caught issue")
            elif self.config_manager.array_geometry == "ods":

                vel_est, self.estimated_R2, self.inlier_percent = \
                    self.lsq_fit_ego_vel_ransac_points_3D(
                        points=points
                    )
                
                self.proposed_velocity_estimate = vel_est
        
        return


    def process(
            self,
            points: np.ndarray = np.empty(shape=0),
            **kwargs) -> np.ndarray:
        """
        Compute the velocity response over a range window centered around a given altitude.

        Args:
        Args:
            points (np.ndarray | list): Point cloud data.
            **kwargs: Additional keyword arguments.
        Returns:
            np.ndarray: velocity estimate [vx,vy,vz]
        """
        
        # Convert list to numpy array if necessary
        if isinstance(points, list):
            points = np.array(points)
        
        
        if points.shape[0] > 0:

            #estimate the velocity from the detected points
            self.estimate_ego_velocity_points(points=points)

        #check the measurements
        self.update_and_check_current_vel_measurements()
        

        return self.current_velocity_estimate