import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple

class BaseAnalyzer:
    """
    Base class for performing analysis on estimated vs ground truth data.
    """

    def __init__(self) -> None:
        """
        Initialize the BaseAnalyzer class.
        """
        pass

    def compute_error(
        self,
        estimated: np.ndarray,
        ground_truth: np.ndarray,
        method: str = "absolute"
    ) -> np.ndarray:
        """
        Compute the error between estimated and ground truth values.

        Args:
            estimated (np.ndarray): Array of estimated values.
            ground_truth (np.ndarray): Array of ground truth values.
            method (str, optional): Method to compute error. 
                "absolute" for absolute difference |x_est - x_gt|, 
                "signed" for signed difference (x_est - x_gt). 
                Defaults to "absolute".

        Returns:
            np.ndarray: Array of computed errors.
        
        Raises:
            ValueError: If input arrays calculate have different shapes.
        """
        if estimated.shape != ground_truth.shape:
             raise ValueError(f"Shape mismatch: estimated {estimated.shape} vs ground truth {ground_truth.shape}")

        if method == "absolute":
            return np.abs(estimated - ground_truth)
        elif method == "signed":
            return estimated - ground_truth
        else:
            raise ValueError(f"Unknown error computation method: {method}")

    def compute_norm_error(
        self,
        estimated_vectors: np.ndarray,
        ground_truth_vectors: np.ndarray
    ) -> np.ndarray:
        """
        Compute the Euclidean norm of the error vector between estimated and ground truth vectors.

        Args:
            estimated_vectors (np.ndarray): Array of estimated vectors (N x D), where D is dimension.
            ground_truth_vectors (np.ndarray): Array of ground truth vectors (N x D).

        Returns:
            np.ndarray: 1D array of norm errors of length N.
        """
        if estimated_vectors.shape != ground_truth_vectors.shape:
            raise ValueError(f"Shape mismatch: estimated {estimated_vectors.shape} vs ground truth {ground_truth_vectors.shape}")
            
        difference = estimated_vectors - ground_truth_vectors
        return np.linalg.norm(difference, axis=1)

    def compute_summary_statistics(
        self,
        data: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute summary statistics (Mean, Median, RMSE, 90% Tail) for a given dataset.

        Args:
            data (np.ndarray): Input data array.

        Returns:
            Dict[str, float]: Dictionary containing "Mean", "Median", "RMSE", and "90% Tail".
        """
        if data.size == 0:
            return {"Mean": 0.0, "Median": 0.0, "RMSE": 0.0, "90% Tail": 0.0}

        return {
            "Mean": float(np.mean(data)),
            "Median": float(np.median(data)),
            "RMSE": float(np.sqrt(np.mean(data**2))),
            "90% Tail": float(np.percentile(np.abs(data), 90))
        }
