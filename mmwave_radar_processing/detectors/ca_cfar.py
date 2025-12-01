
import numpy as np
from typing import Tuple
from .base import BaseCFAR1D, BaseCFAR2D

class CaCFAR1D(BaseCFAR1D):
    """
    1D Cell Averaging CFAR (CA-CFAR) detector.
    """

    def _compute_thresholds(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute thresholds using CA-CFAR.
        
        Args:
            x (np.ndarray): 1D input signal.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (thresholds, noise_estimates)
        """
        L = len(x)
        # Initialize thresholds with infinity so that no detection occurs by default
        # (e.g., at edges where window doesn't fit)
        thresholds = np.full(L, np.inf)
        noise_estimates = np.zeros(L)
        
        # Get sliding windows
        # Shape: (num_windows, window_size)
        # Each row represents the window surrounding a specific CUT.
        try:
            windows = self._get_window_view(x)
        except ValueError:
            # Signal too short for window
            return thresholds, noise_estimates

        # Create mask for training cells
        # The window contains: [Training Left | Guard Left | CUT | Guard Right | Training Right]
        # We want to select only the Training cells for averaging.
        # Window size = 2*N_T + 2*N_G + 1
        win_size = windows.shape[1]
        mask = np.ones(win_size, dtype=bool)
        
        # Indices relative to the window start
        guard_start = self.num_train
        guard_end = self.num_train + 2 * self.num_guard
        
        # Set Guard + CUT region to False in the mask
        mask[guard_start : guard_end + 1] = False
        
        # Number of training cells (N)
        num_train_cells = np.sum(mask)
        
        # Compute noise estimate (mean of training cells)
        # We select only the columns corresponding to training cells using the mask.
        # windows[:, mask] returns a new array of shape (num_windows, num_train_cells)
        training_cells = windows[:, mask]
        
        # Compute the mean across the training cells (axis 1) for each window
        means = np.mean(training_cells, axis=1)
        
        # Compute alpha scaling factor based on Pfa and N
        alpha = self.compute_alpha_ca(num_train_cells, self.pfa)
        
        # Compute thresholds: T = alpha * Noise_Estimate
        computed_thresholds = alpha * means
        
        # Map back to full array indices
        # The first window corresponds to the CUT at index (num_train + num_guard).
        cut_idx = self.num_train + self.num_guard
        valid_start = cut_idx
        valid_end = valid_start + len(means)
        
        # Assign computed values to the valid central region of the output arrays
        thresholds[valid_start:valid_end] = computed_thresholds
        noise_estimates[valid_start:valid_end] = means
        
        return thresholds, noise_estimates


class CaCFAR2D(BaseCFAR2D):
    """
    2D Cell Averaging CFAR (CA-CFAR) detector.
    """

    def _compute_thresholds(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute thresholds using CA-CFAR for 2D.
        
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
            
        # windows shape: (R_out, D_out, win_r, win_d)
        # R_out, D_out are the number of valid window positions.
        # win_r, win_d are the dimensions of the window itself.
        
        # Create mask for the 2D window
        win_r = windows.shape[2]
        win_d = windows.shape[3]
        mask = np.ones((win_r, win_d), dtype=bool)
        
        # Guard region bounds within the window
        # Range: [train_r, train_r + 2*guard_r]
        # Doppler: [train_d, train_d + 2*guard_d]
        gr_start = self.num_train[0]
        gr_end = self.num_train[0] + 2 * self.num_guard[0]
        gd_start = self.num_train[1]
        gd_end = self.num_train[1] + 2 * self.num_guard[1]
        
        # Mask out the central block (Guard + CUT)
        mask[gr_start : gr_end + 1, gd_start : gd_end + 1] = False
        
        num_train_cells = np.sum(mask)
        
        # Compute mean of training cells
        # We multiply the windows by the mask (broadcasting).
        # Masked entries (Guard/CUT) become 0 (since False maps to 0).
        # We then sum over the window dimensions (axis 2 and 3).
        # Note: This works because we divide by the count of *valid* training cells.
        
        # windows is (N_win_r, N_win_d, Wr, Wd)
        # mask is (Wr, Wd)
        masked_windows = windows * mask 
        sums = np.sum(masked_windows, axis=(2, 3))
        means = sums / num_train_cells
        
        alpha = self.compute_alpha_ca(num_train_cells, self.pfa)
        
        computed_thresholds = alpha * means
        
        # Place in result arrays
        # Calculate where the valid detections start (center of the first window)
        cut_r = self.num_train[0] + self.num_guard[0]
        cut_d = self.num_train[1] + self.num_guard[1]
        
        valid_r_start = cut_r
        valid_r_end = valid_r_start + means.shape[0]
        valid_d_start = cut_d
        valid_d_end = valid_d_start + means.shape[1]
        
        thresholds[valid_r_start:valid_r_end, valid_d_start:valid_d_end] = computed_thresholds
        noise_estimates[valid_r_start:valid_r_end, valid_d_start:valid_d_end] = means
        
        return thresholds, noise_estimates
