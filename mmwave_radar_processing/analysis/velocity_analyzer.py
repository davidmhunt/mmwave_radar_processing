import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple, Optional
from mmwave_radar_processing.analysis.base_analyzer import BaseAnalyzer

class VelocityAnalyzer(BaseAnalyzer):
    """
    Analyzer class specifically for velocity estimation analysis.
    Inherits from BaseAnalyzer.
    """

    def __init__(self) -> None:
        """
        Initialize the VelocityAnalyzer class.
        """
        super().__init__()
        self.x_errors: Optional[np.ndarray] = None
        self.y_errors: Optional[np.ndarray] = None
        self.z_errors: Optional[np.ndarray] = None
        self.norm_errors: Optional[np.ndarray] = None

    def analyze(
        self,
        history_estimated: np.ndarray,
        history_gt: np.ndarray
    ) -> None:
        """
        Analyze velocity estimation performance by computing errors for X, Y, Z components and the norm.
        Stores the computed errors in instance attributes.

        Args:
            history_estimated (np.ndarray): Array of estimated velocities (N x 3).
            history_gt (np.ndarray): Array of ground truth velocities (N x 3).
        
        Raises:
            ValueError: If input arrays have incompatible shapes.
        """
        if history_estimated.shape != history_gt.shape:
            raise ValueError(f"Shape mismatch: estimated {history_estimated.shape} vs ground truth {history_gt.shape}")
        
        if history_estimated.shape[1] != 3:
             raise ValueError(f"Expected 3D velocity vectors, got shape {history_estimated.shape}")

        # Compute signed errors (Estimated - GT)
        # We can use 'signed' to see bias, but for general error magnitude analysis often absolute is used.
        # However, for storage we might want the signed error to allow distribution analysis (centered at 0?).
        # original script used:
        # x_errors = vel_est[:,0] - vel_gt[:,0] (Signed)
        # But then also calculated absolute error for other things.
        # The user request for BaseAnalyzer asked for "absolute and magnitude of errors".
        # Let's store signed errors for histograms (to see bias) and compute absolute/norm for summary stats.
        
        # Actually, let's store the raw signed errors, so we can do whatever we want with them (abs or not).
        self.x_errors = self.compute_error(history_estimated[:, 0], history_gt[:, 0], method="signed")
        self.y_errors = self.compute_error(history_estimated[:, 1], history_gt[:, 1], method="signed")
        self.z_errors = self.compute_error(history_estimated[:, 2], history_gt[:, 2], method="signed")
        
        self.norm_errors = self.compute_norm_error(history_estimated, history_gt)


    def get_x_errors(self) -> np.ndarray:
        """
        Get the stored X velocity errors.

        Returns:
            np.ndarray: Array of X errors.
        
        Raises:
            ValueError: If analyze() has not been called yet.
        """
        if self.x_errors is None:
            raise ValueError("Analysis not performed. Call analyze() first.")
        return self.x_errors

    def get_y_errors(self) -> np.ndarray:
        """
        Get the stored Y velocity errors.

        Returns:
            np.ndarray: Array of Y errors.
        
        Raises:
            ValueError: If analyze() has not been called yet.
        """
        if self.y_errors is None:
            raise ValueError("Analysis not performed. Call analyze() first.")
        return self.y_errors
    
    def get_z_errors(self) -> np.ndarray:
        """
        Get the stored Z velocity errors.

        Returns:
            np.ndarray: Array of Z errors.
        
        Raises:
            ValueError: If analyze() has not been called yet.
        """
        if self.z_errors is None:
            raise ValueError("Analysis not performed. Call analyze() first.")
        return self.z_errors

    def get_norm_errors(self) -> np.ndarray:
        """
        Get the stored Norm velocity errors.

        Returns:
            np.ndarray: Array of Norm errors.
        
        Raises:
            ValueError: If analyze() has not been called yet.
        """
        if self.norm_errors is None:
            raise ValueError("Analysis not performed. Call analyze() first.")
        return self.norm_errors

    def generate_report(self) -> pd.DataFrame:
        """
        Generate a summary report of the analysis.

        Returns:
            pd.DataFrame: DataFrame containing Mean, Median, and RMSE for X, Y, Z, and Norm errors.
        """
        if self.x_errors is None:
             raise ValueError("Analysis not performed. Call analyze() first.")

        # For summary statistics like Mean Error, we generally look at the absolute error to see magnitude of deviation,
        # OR we look at signed error to see bias.
        # The original script calculated Mean, Median, RMSE of the ERRORs (signed or absolute? depends on the lines).
        # Original script:
        # x_errors = vel_est - vel_gt (Signed)
        # summary_stats: Mean(x_errors), Median(x_errors), RMSE(x_errors).
        # RMSE is sqrt(mean(error^2)), so sign doesn't matter.
        # Mean of signed error = Bias. Mean of absolute error = MAE.
        # The prompt asked for "summary statistics of the mean, median, and RMSE".
        # Usually standard is Bias (Mean Signed), but let's stick to what the previous script did roughly or standard practice.
        # Let's provide statistics on the MEASURED errors (which we stored as signed).
        # But for Norm, it's always positive.
        
        stats = {}
        for name, data in [
            ("X", self.x_errors),
            ("Y", self.y_errors),
            ("Z", self.z_errors),
            ("Norm", self.norm_errors)
        ]:
            s = self.compute_summary_statistics(data)
            stats[name] = s
            
        return pd.DataFrame(stats).T
