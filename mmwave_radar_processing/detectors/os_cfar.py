
import numpy as np
from typing import Tuple, Optional
from .base import BaseCFAR1D, BaseCFAR2D

class OsCFAR1D(BaseCFAR1D):
    """
    1D Ordered Statistic CFAR (OS-CFAR) detector.
    """

    def __init__(self, num_train: int, num_guard: int, k_rank: int, alpha: float):
        """
        Initialize the 1D OS-CFAR detector.

        Args:
            num_train (int): Number of training cells on each side.
            num_guard (int): Number of guard cells on each side.
            k_rank (int): 1-based rank index of the training cell to use as noise estimate.
            alpha (float): Scaling factor.
        """
        # Pass dummy pfa since we use explicit alpha
        super().__init__(num_train, num_guard, pfa=0.0)
        self.k_rank = k_rank
        self.alpha = alpha

    def _compute_thresholds(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute thresholds using OS-CFAR.
        
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

        # Window size = 2*N_T + 2*N_G + 1
        # Training cells: Left + Right
        left_end = self.num_train
        right_start = self.num_train + 2 * self.num_guard + 1
        
        # Concatenate left and right regions for each window
        # windows is (N_win, Win_Size)
        left_region = windows[:, :left_end]
        right_region = windows[:, right_start:]
        
        training_cells = np.concatenate((left_region, right_region), axis=1)
        
        # Validate k_rank
        N_train = training_cells.shape[1]
        if not (1 <= self.k_rank <= N_train):
            raise ValueError(f"k_rank must be between 1 and {N_train}, got {self.k_rank}")
            
        # Get k-th smallest value (k_rank is 1-based)
        k_idx = self.k_rank - 1
        
        # Use partition for efficiency (O(N)) instead of sort (O(N log N))
        partitioned = np.partition(training_cells, k_idx, axis=1)
        noise_est = partitioned[:, k_idx]
        
        computed_thresholds = self.alpha * noise_est
        
        cut_idx = self.num_train + self.num_guard
        valid_start = cut_idx
        valid_end = valid_start + len(noise_est)
        
        thresholds[valid_start:valid_end] = computed_thresholds
        noise_estimates[valid_start:valid_end] = noise_est
        
        return thresholds, noise_estimates


class OsCFAR2D(BaseCFAR2D):
    """
    2D Ordered Statistic CFAR (OS-CFAR) detector.
    """

    def __init__(self, num_train: Tuple[int, int], num_guard: Tuple[int, int], k_rank: int, alpha: float):
        """
        Initialize the 2D OS-CFAR detector.

        Args:
            num_train (Tuple[int, int]): (train_range, train_doppler) half-widths.
            num_guard (Tuple[int, int]): (guard_range, guard_doppler) half-widths.
            k_rank (int): 1-based rank index.
            alpha (float): Scaling factor.
        """
        super().__init__(num_train, num_guard, pfa=0.0)
        self.k_rank = k_rank
        self.alpha = alpha

    def _compute_thresholds(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute thresholds using OS-CFAR.
        
        Args:
            X (np.ndarray): 2D input signal.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (thresholds, noise_estimates)
        """
        R, D = X.shape
        thresholds = np.full((R, D), np.inf)
        noise_estimates = np.zeros((R, D))
        
        try:
            windows = self._get_window_view(X)
        except ValueError:
            return thresholds, noise_estimates
            
        # windows: (R', D', Wr, Wd)
        
        # Create mask for training cells
        win_r = windows.shape[2]
        win_d = windows.shape[3]
        mask = np.ones((win_r, win_d), dtype=bool)
        
        gr_start = self.num_train[0]
        gr_end = self.num_train[0] + 2 * self.num_guard[0]
        gd_start = self.num_train[1]
        gd_end = self.num_train[1] + 2 * self.num_guard[1]
        
        mask[gr_start : gr_end + 1, gd_start : gd_end + 1] = False
        
        # Extract training cells
        # windows[..., mask] returns (R', D', N_train)
        training_cells = windows[..., mask]
        
        N_train = training_cells.shape[-1]
        if not (1 <= self.k_rank <= N_train):
            raise ValueError(f"k_rank must be between 1 and {N_train}, got {self.k_rank}")
            
        k_idx = self.k_rank - 1
        
        partitioned = np.partition(training_cells, k_idx, axis=-1)
        noise_est = partitioned[..., k_idx]
        
        computed_thresholds = self.alpha * noise_est
        
        cut_r = self.num_train[0] + self.num_guard[0]
        cut_d = self.num_train[1] + self.num_guard[1]
        
        valid_r_start = cut_r
        valid_r_end = valid_r_start + noise_est.shape[0]
        valid_d_start = cut_d
        valid_d_end = valid_d_start + noise_est.shape[1]
        
        thresholds[valid_r_start:valid_r_end, valid_d_start:valid_d_end] = computed_thresholds
        noise_estimates[valid_r_start:valid_r_end, valid_d_start:valid_d_end] = noise_est
        
        return thresholds, noise_estimates
