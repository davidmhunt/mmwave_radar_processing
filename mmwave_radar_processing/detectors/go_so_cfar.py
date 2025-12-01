
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
        
        try:
            windows = self._get_window_view(x)
        except ValueError:
            return thresholds, noise_estimates

        # Split window into Left and Right training regions
        # Window size = 2*N_T + 2*N_G + 1
        # Left: [0 : N_T]
        # Right: [N_T + 2*N_G + 1 : ]
        
        left_end = self.num_train
        right_start = self.num_train + 2 * self.num_guard + 1
        
        left_region = windows[:, :left_end]
        right_region = windows[:, right_start:]
        
        # Compute means of each side
        mean_left = np.mean(left_region, axis=1)
        mean_right = np.mean(right_region, axis=1)
        
        # GO-CFAR: Take max of the means
        noise_est = np.maximum(mean_left, mean_right)
        
        # Alpha calculation
        # Use N_side (num_train)
        alpha = self.compute_alpha_ca(self.num_train, self.pfa)
        
        computed_thresholds = alpha * noise_est
        
        # Map back
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

        left_end = self.num_train
        right_start = self.num_train + 2 * self.num_guard + 1
        
        left_region = windows[:, :left_end]
        right_region = windows[:, right_start:]
        
        mean_left = np.mean(left_region, axis=1)
        mean_right = np.mean(right_region, axis=1)
        
        # SO-CFAR: Take min of the means
        noise_est = np.minimum(mean_left, mean_right)
        
        # Alpha calculation
        # Use N_side (num_train)
        alpha = self.compute_alpha_ca(self.num_train, self.pfa)
        
        computed_thresholds = alpha * noise_est
        
        cut_idx = self.num_train + self.num_guard
        valid_start = cut_idx
        valid_end = valid_start + len(noise_est)
        
        thresholds[valid_start:valid_end] = computed_thresholds
        noise_estimates[valid_start:valid_end] = noise_est
        
        return thresholds, noise_estimates
