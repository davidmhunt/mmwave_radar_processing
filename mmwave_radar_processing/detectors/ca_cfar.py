
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
        thresholds = np.full(L, np.inf)
        noise_estimates = np.zeros(L)
        
        # Get sliding windows
        # Shape: (num_windows, window_size)
        try:
            windows = self._get_window_view(x)
        except ValueError:
            # Signal too short for window
            return thresholds, noise_estimates

        # Create mask for training cells
        # Window size = 2*N_T + 2*N_G + 1
        # CUT index = N_T + N_G
        # Guard indices = [N_T, N_T + 2*N_G]
        win_size = windows.shape[1]
        mask = np.ones(win_size, dtype=bool)
        
        cut_idx = self.num_train + self.num_guard
        guard_start = self.num_train
        guard_end = self.num_train + 2 * self.num_guard
        
        mask[guard_start : guard_end + 1] = False
        
        # Number of training cells
        num_train_cells = np.sum(mask)
        
        # Compute noise estimate (mean of training cells)
        # Sum across the window dimension (axis 1)
        # We use the mask to select only training cells
        # Since mask is boolean, we can multiply.
        # However, for efficiency with sliding_window_view (which is a view), 
        # direct multiplication might create a large copy if not careful.
        # But for 1D it's fine.
        
        # windows[:, mask] selects the columns corresponding to training cells
        # Shape becomes (num_windows, num_train_cells)
        training_cells = windows[:, mask]
        
        # Mean
        means = np.mean(training_cells, axis=1)
        
        # Compute alpha
        alpha = self.compute_alpha_ca(num_train_cells, self.pfa)
        
        # Compute thresholds
        computed_thresholds = alpha * means
        
        # Map back to full array
        # The windows start at index 0 of the output, which corresponds to index 'cut_idx' in the original array
        # because the CUT is at the center of the window.
        valid_start = cut_idx
        valid_end = valid_start + len(means)
        
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
        
        # Create mask
        win_r = windows.shape[2]
        win_d = windows.shape[3]
        mask = np.ones((win_r, win_d), dtype=bool)
        
        # Center (CUT) coordinates in window
        cut_r = self.num_train[0] + self.num_guard[0]
        cut_d = self.num_train[1] + self.num_guard[1]
        
        # Guard region bounds
        # Range: [train_r, train_r + 2*guard_r]
        # Doppler: [train_d, train_d + 2*guard_d]
        gr_start = self.num_train[0]
        gr_end = self.num_train[0] + 2 * self.num_guard[0]
        gd_start = self.num_train[1]
        gd_end = self.num_train[1] + 2 * self.num_guard[1]
        
        mask[gr_start : gr_end + 1, gd_start : gd_end + 1] = False
        
        num_train_cells = np.sum(mask)
        
        # Compute mean
        # We can sum over the last two axes with the mask
        # windows * mask will zero out guard cells (if mask is 0/1)
        # But mask is boolean.
        
        # Optimization: Sum all, subtract guard region sum?
        # Or just use boolean indexing?
        # Boolean indexing on the last two dimensions of a 4D array is tricky to keep shape.
        # Easier: multiply by mask (broadcast) and sum.
        
        # windows is (N_win_r, N_win_d, Wr, Wd)
        # mask is (Wr, Wd)
        
        masked_windows = windows * mask # Broadcasts
        sums = np.sum(masked_windows, axis=(2, 3))
        means = sums / num_train_cells
        
        alpha = self.compute_alpha_ca(num_train_cells, self.pfa)
        
        computed_thresholds = alpha * means
        
        # Place in result
        valid_r_start = cut_r
        valid_r_end = valid_r_start + means.shape[0]
        valid_d_start = cut_d
        valid_d_end = valid_d_start + means.shape[1]
        
        thresholds[valid_r_start:valid_r_end, valid_d_start:valid_d_end] = computed_thresholds
        noise_estimates[valid_r_start:valid_r_end, valid_d_start:valid_d_end] = means
        
        return thresholds, noise_estimates
