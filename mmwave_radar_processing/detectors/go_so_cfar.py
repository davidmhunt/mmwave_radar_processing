
import numpy as np
from typing import Tuple
from .base import BaseCFAR1D

class GoCFAR1D(BaseCFAR1D):
    """
    1D Greatest-Of CFAR (GO-CFAR) detector.
    """

    def _compute_thresholds(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute thresholds using GO-CFAR.
        
        Args:
            x (np.ndarray): 1D input signal.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (thresholds, noise_estimates)
        """
        L = len(x)
        thresholds = np.full(L, np.inf)
        noise_estimates = np.zeros(L)
        
        # Get sliding windows
        # Shape: (num_windows, window_size)
        try:
            windows = self._get_window_view(x)
        except ValueError:
            return thresholds, noise_estimates

        # Split window into Left and Right training regions
        # Window structure: [Left Training | Guard Left | CUT | Guard Right | Right Training]
        # We slice the window view to get the Left and Right parts separately.
        
        left_end = self.num_train
        # Start of Right Training = (Left Training) + (Guard Left) + (CUT) + (Guard Right)
        # = N_T + N_G + 1 + N_G = N_T + 2*N_G + 1
        right_start = self.num_train + 2 * self.num_guard + 1
        
        # Slicing the windows array
        # left_region shape: (num_windows, num_train)
        # right_region shape: (num_windows, num_train)
        left_region = windows[:, :left_end]
        right_region = windows[:, right_start:]
        
        # Compute means of each side for all windows
        mean_left = np.mean(left_region, axis=1)
        mean_right = np.mean(right_region, axis=1)
        
        # GO-CFAR: Take the MAXIMUM of the two means
        # This helps maintain the threshold at clutter edges (transitions from low to high power).
        noise_est = np.maximum(mean_left, mean_right)
        
        # Alpha calculation
        # Note: We use N_side (num_train) for alpha calculation in GO/SO approximations
        # because the effective noise estimate is dominated by one side.
        alpha = self.compute_alpha_ca(self.num_train, self.pfa)
        
        computed_thresholds = alpha * noise_est
        
        # Map back to full array indices
        cut_idx = self.num_train + self.num_guard
        valid_start = cut_idx
        valid_end = valid_start + len(noise_est)
        
        thresholds[valid_start:valid_end] = computed_thresholds
        noise_estimates[valid_start:valid_end] = noise_est
        
        return thresholds, noise_estimates


class SoCFAR1D(BaseCFAR1D):
    """
    1D Smallest-Of CFAR (SO-CFAR) detector.
    """

    def _compute_thresholds(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute thresholds using SO-CFAR.
        
        Args:
            x (np.ndarray): 1D input signal.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (thresholds, noise_estimates)
        """
        L = len(x)
        thresholds = np.full(L, np.inf)
        noise_estimates = np.zeros(L)
        
        try:
            windows = self._get_window_view(x)
        except ValueError:
            return thresholds, noise_estimates

        # Split window into Left and Right training regions
        left_end = self.num_train
        right_start = self.num_train + 2 * self.num_guard + 1
        
        left_region = windows[:, :left_end]
        right_region = windows[:, right_start:]
        
        mean_left = np.mean(left_region, axis=1)
        mean_right = np.mean(right_region, axis=1)
        
        # SO-CFAR: Take the MINIMUM of the two means
        # This is useful for resolving closely spaced targets (interfering target in one side doesn't corrupt estimate).
        noise_est = np.minimum(mean_left, mean_right)
        
        # Alpha calculation
        alpha = self.compute_alpha_ca(self.num_train, self.pfa)
        
        computed_thresholds = alpha * noise_est
        
        cut_idx = self.num_train + self.num_guard
        valid_start = cut_idx
        valid_end = valid_start + len(noise_est)
        
        thresholds[valid_start:valid_end] = computed_thresholds
        noise_estimates[valid_start:valid_end] = noise_est
        
        return thresholds, noise_estimates
