
import numpy as np
from abc import ABC, abstractmethod
from numpy.lib.stride_tricks import sliding_window_view
from typing import Tuple, List, Optional, Union, Any

class BaseCFAR1D(ABC):
    """
    Abstract base class for 1D CFAR detectors.

    Attributes:
        num_train (int): Number of training cells on each side of the CUT.
        num_guard (int): Number of guard cells on each side of the CUT.
        pfa (float): Probability of false alarm.
        thresholds (Optional[np.ndarray]): Cached thresholds from the last detection.
        detections (Optional[np.ndarray]): Cached detection boolean mask from the last detection.
        noise_estimates (Optional[np.ndarray]): Cached noise estimates from the last detection.
    """

    def __init__(self, num_train: int, num_guard: int, pfa: float):
        """
        Initialize the 1D CFAR detector.

        Args:
            num_train (int): Number of training cells on each side.
            num_guard (int): Number of guard cells on each side.
            pfa (float): Desired probability of false alarm.
        """
        self.num_train = num_train
        self.num_guard = num_guard
        self.pfa = pfa
        
        self.thresholds: Optional[np.ndarray] = None
        self.detections: Optional[np.ndarray] = None
        self.noise_estimates: Optional[np.ndarray] = None

    def detect(self, x: np.ndarray) -> List[int]:
        """
        Perform detection on the input 1D signal.

        Args:
            x (np.ndarray): 1D input signal (magnitude or power).

        Returns:
            List[int]: List of indices where detections occurred.
        """
        # Ensure input is a numpy array for efficient operations
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("Input x must be a 1D array.")

        # Compute thresholds (abstract method implemented by subclasses like CaCFAR1D)
        # This returns two arrays: the computed thresholds and the noise estimates for each cell.
        self.thresholds, self.noise_estimates = self._compute_thresholds(x)
        
        # Apply decision rule: Signal > Threshold
        # This creates a boolean array where True indicates a detection.
        # Note: We only detect where we have valid thresholds.
        # The _compute_thresholds method handles edges (e.g., sets them to infinity),
        # so no detections will occur there.
        self.detections = x > self.thresholds
        
        # Return indices of True values in the boolean array
        return np.where(self.detections)[0].tolist()

    def plot_detections(self, x: np.ndarray, title: str = "CFAR Detection", convert_to_dB: bool = False) -> None:
        """
        Plot the signal, threshold, and detections.
        
        Requires matplotlib.
        
        Args:
            x (np.ndarray): The input signal that was processed.
            title (str): Title for the plot.
            convert_to_dB (bool): If True, plot signal and threshold values in dB (20*log10).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting.")
            return

        if self.thresholds is None:
            print("No thresholds computed. Run detect() first.")
            return

        plot_x = x
        plot_thresholds = self.thresholds
        ylabel = 'Magnitude'

        if convert_to_dB:
            # Avoid log of zero or negative values
            plot_x = 20 * np.log10(np.maximum(x, 1e-10))
            plot_thresholds = 20 * np.log10(np.maximum(self.thresholds, 1e-10))
            ylabel = 'Magnitude (dB)'

        plt.figure(figsize=(10, 6))
        plt.plot(plot_x, label='Signal', color='blue', alpha=0.7)
        plt.plot(plot_thresholds, label='Threshold', color='orange', linestyle='--')
        
        # Plot detections
        det_indices = np.where(self.detections)[0]
        if len(det_indices) > 0:
            # Use the already converted plot_x for detection points
            plt.scatter(det_indices, plot_x[det_indices], color='red', marker='x', label='Detections', zorder=5)
            
        plt.title(title)
        plt.xlabel('Index')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    @abstractmethod
    def _compute_thresholds(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute thresholds for the input signal.
        
        Args:
            x (np.ndarray): 1D input signal.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (thresholds, noise_estimates)
            Arrays should be of the same shape as x.
        """
        pass

    def _get_window_view(self, x: np.ndarray) -> np.ndarray:
        """
        Get a sliding window view of the input x.
        
        This method uses numpy's stride_tricks to create a 'view' of the array
        where each row represents a window. This allows us to vectorize operations
        over all windows without manually looping or copying data.
        
        The window size is 2 * (self.num_train + self.num_guard) + 1.
        
        Args:
            x (np.ndarray): 1D input signal.
            
        Returns:
            np.ndarray: Sliding window view of shape (L - window_size + 1, window_size).
        """
        window_size = 2 * (self.num_train + self.num_guard) + 1
        if len(x) < window_size:
             raise ValueError(f"Input length {len(x)} is smaller than window size {window_size}.")
        
        # sliding_window_view creates a new dimension.
        # For input [0, 1, 2, 3, 4] and window_size 3:
        # Output: [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
        return sliding_window_view(x, window_shape=window_size)

    @staticmethod
    def compute_alpha_ca(num_train_cells: int, pfa: float) -> float:
        """
        Compute the scaling factor alpha for CA-CFAR.
        
        The formula is derived based on the assumption of exponential noise distribution.
        alpha = N * (P_fa^(-1/N) - 1)
        
        Args:
            num_train_cells (int): Total number of training cells (N).
            pfa (float): Probability of false alarm.
            
        Returns:
            float: Scaling factor alpha.
        """
        return num_train_cells * (pfa ** (-1.0 / num_train_cells) - 1.0)


class BaseCFAR2D(ABC):
    """
    Abstract base class for 2D CFAR detectors.

    Attributes:
        num_train (Tuple[int, int]): Number of training cells (range, doppler) on each side.
        num_guard (Tuple[int, int]): Number of guard cells (range, doppler) on each side.
        pfa (float): Probability of false alarm.
        thresholds (Optional[np.ndarray]): Cached thresholds from the last detection.
        detections (Optional[np.ndarray]): Cached detection boolean mask from the last detection.
        noise_estimates (Optional[np.ndarray]): Cached noise estimates from the last detection.
    """

    def __init__(self, num_train: Tuple[int, int], num_guard: Tuple[int, int], pfa: float):
        """
        Initialize the 2D CFAR detector.

        Args:
            num_train (Tuple[int, int]): (train_range, train_doppler) half-widths.
            num_guard (Tuple[int, int]): (guard_range, guard_doppler) half-widths.
            pfa (float): Desired probability of false alarm.
        """
        self.num_train = num_train
        self.num_guard = num_guard
        self.pfa = pfa
        
        self.thresholds: Optional[np.ndarray] = None
        self.detections: Optional[np.ndarray] = None
        self.noise_estimates: Optional[np.ndarray] = None

    def detect(self, X: np.ndarray) -> List[Tuple[int, int]]:
        """
        Perform detection on the input 2D signal (Range-Doppler map).

        Args:
            X (np.ndarray): 2D input signal (magnitude or power).

        Returns:
            List[Tuple[int, int]]: List of (row, col) indices where detections occurred.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array.")

        # Compute thresholds (implemented by subclasses)
        self.thresholds, self.noise_estimates = self._compute_thresholds(X)
        
        # Apply decision rule
        self.detections = X > self.thresholds
        
        # Return indices as list of tuples (row, col)
        rows, cols = np.where(self.detections)
        return list(zip(rows, cols))

    def plot_detections(self, X: np.ndarray, title: str = "2D CFAR Detection") -> None:
        """
        Plot the 2D signal and detections.
        
        Requires matplotlib.
        
        Args:
            X (np.ndarray): The input signal that was processed.
            title (str): Title for the plot.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("matplotlib and numpy are required for plotting.")
            return

        if self.detections is None:
            print("No detections computed. Run detect() first.")
            return

        # The original plot_detections had convert_to_dB logic.
        # The instruction removes this parameter and its logic.
        # So, we'll plot the raw magnitude/power.
        plot_X = X
        xlabel = 'Doppler Index'
        ylabel = 'Range Index'
        colorbar_label = 'Magnitude' # Assuming magnitude/power for 2D plots

        plt.figure(figsize=(12, 5))
        
        # Plot original map
        plt.subplot(1, 2, 1)
        plt.imshow(plot_X, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label=colorbar_label)
        plt.title(f"{title} - Signal")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        # Plot detections
        plt.subplot(1, 2, 2)
        plt.imshow(self.detections, aspect='auto', origin='lower', cmap='gray')
        plt.title(f"{title} - Detections")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def compute_alpha_ca(num_train_cells: int, pfa: float) -> float:
        """
        Compute the scaling factor alpha for CA-CFAR.
        
        Args:
            num_train_cells (int): Total number of training cells (N).
            pfa (float): Probability of false alarm.
            
        Returns:
            float: Scaling factor alpha.
        """
        return num_train_cells * (pfa ** (-1.0 / num_train_cells) - 1.0)

    @abstractmethod
    def _compute_thresholds(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute thresholds for the input 2D signal.
        
        Args:
            X (np.ndarray): 2D input signal.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (thresholds, noise_estimates)
            Arrays should be of the same shape as X.
        """
        pass
    
    def _get_window_view(self, X: np.ndarray) -> np.ndarray:
        """
        Get a sliding window view of the input X.
        
        Window shape is (2*train_r + 2*guard_r + 1, 2*train_d + 2*guard_d + 1).
        
        Args:
            X (np.ndarray): 2D input signal.
            
        Returns:
            np.ndarray: Sliding window view.
        """
        win_r = 2 * (self.num_train[0] + self.num_guard[0]) + 1
        win_d = 2 * (self.num_train[1] + self.num_guard[1]) + 1
        
        if X.shape[0] < win_r or X.shape[1] < win_d:
             raise ValueError(f"Input shape {X.shape} is smaller than window size {(win_r, win_d)}.")
             
        return sliding_window_view(X, window_shape=(win_r, win_d))
