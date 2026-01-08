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
        history_gt: np.ndarray,
        error_method: str = "signed"
    ) -> None:
        """
        Perform analysis on velocity data.

        Args:
            history_estimated (np.ndarray): Array of estimated velocities (N, 3).
            history_gt (np.ndarray): Array of ground truth velocities (N, 3).
            error_method (str, optional): Method for error calculation "signed" or "absolute". Defaults to "signed".

        Raises:
            ValueError: If input arrays have mismatched shapes or incorrect dimensions.
        """
        if history_estimated.shape != history_gt.shape:
            raise ValueError(
                f"Shape mismatch: Estimated {history_estimated.shape}, GT {history_gt.shape}"
            )
        
        if history_estimated.shape[1] != 3:
             raise ValueError(f"Expected 3D velocity vectors, got shape {history_estimated.shape}")

        # Store errors based on the requested method
        self.x_errors = self.compute_error(history_estimated[:, 0], history_gt[:, 0], method=error_method)
        self.y_errors = self.compute_error(history_estimated[:, 1], history_gt[:, 1], method=error_method)
        self.z_errors = self.compute_error(history_estimated[:, 2], history_gt[:, 2], method=error_method)
        
        # Norm error is always non-negative magnitude difference or distance
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
