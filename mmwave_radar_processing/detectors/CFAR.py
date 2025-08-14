import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view


class CaCFAR_1D:

    def __init__(
            self,
            num_guard_cells=4,
            num_training_cells=10,
            false_alarm_rate=0.1,
            ) -> None:
        """
        Initialize the 1D CA-CFAR detector.

        Args:
            num_guard_cells (int, optional): Number of guard cells on each side of the CUT (Cell Under Test). Defaults to 4.
            num_training_cells (int, optional): Number of training cells on each side of the CUT. Defaults to 10.
            false_alarm_rate (float, optional): Desired probability of false alarm. Defaults to 0.1.
            resp_border_cells (int, optional): Number of border cells to exclude from processing. Defaults to 3.
        """
        
        # Compute the window size
        self.window_size = 2 * (num_training_cells + num_guard_cells) + 1
        self.num_guard_cells = num_guard_cells
        self.num_training_cells = num_training_cells

        # Create the window to convolve over
        self.window = np.zeros(
            shape=(1,self.window_size),
            dtype=float
        )
        self.window[0,:num_training_cells] = 1
        self.window[0,-num_training_cells:] = 1


        # Compute the alpha for the threshold
        self.N = 2 * num_training_cells
        self.alpha = self.N * (false_alarm_rate ** (-1 / self.N) - 1)
        print("alpha: {} (mag), {} dB".format(
            self.alpha,
            20 * np.log10(self.alpha)))
        
        return

    def compute(self,
                signal: np.ndarray,
                row_bins: np.ndarray = np.empty(0),
                col_bins: np.ndarray = np.empty(0)) -> tuple:
        """
        Apply the CA-CFAR algorithm to the input signal.

        Args:
            signal (np.ndarray): Input signal, either 1D or 2D. For 2D signals, CFAR is applied row-wise.
            row_bins (np.ndarray, optional): Array of bin values corresponding to the rows. Defaults to an empty array.
            col_bins (np.ndarray, optional): Array of bin values corresponding to the columns. Defaults to an empty array.

        Returns:
            tuple: A tuple containing:
                - det_bins (list of tuples): List of (row, col) indices or bins of detected targets.
                - T (np.ndarray): CFAR threshold values for each row.
        """
        signal = np.abs(signal)

        if signal.ndim == 1:
            signal = np.expand_dims(signal, axis=0)

        # Check row bins
        if row_bins.shape[0] > 0:
            assert row_bins.shape[0] == signal.shape[0], \
                "row_bins must have the same shape as the number of rows in the signal"
        else:
            row_bins = np.arange(signal.shape[0])

        # Check col bins
        if col_bins.shape[0] > 0:
            assert col_bins.shape[0] == signal.shape[1], \
                "col_bins must have the same shape as the number of columns in the signal"
        else:
            col_bins = np.arange(signal.shape[1])

        # Perform 1D convolution across each row using scipy.signal.convolve2d
        P_n = (1 / self.N) * convolve2d(signal, self.window, mode='valid')
        T = self.alpha * P_n

        # Determine valid indices for each row
        num_cols = signal.shape[1]
        invalid_region_size = self.num_training_cells + self.num_guard_cells
        valid_idxs = np.ones(num_cols, dtype=bool)
        valid_idxs[:invalid_region_size] = False
        valid_idxs[-invalid_region_size:] = False

        # Detect targets
        det_idxs = signal[:, invalid_region_size:-invalid_region_size] > T

        # Map detections to (row, col) indices or bins
        row_bins_expanded = np.repeat(row_bins[:, np.newaxis], det_idxs.shape[1], axis=1)
        col_bins_valid = col_bins[invalid_region_size:-invalid_region_size]
        det_bins = np.column_stack((row_bins_expanded[det_idxs], col_bins_valid[np.where(det_idxs)[1]]))

        return det_bins, T


    def plot_cfar(self, signal, bins: np.ndarray = np.empty(0), ax: plt.Axes = None, show=False, row: int = 0):
        """
        Plot the CFAR threshold and detections.

        Args:
            signal (np.ndarray): Input signal, either 1D or 2D. For 2D signals, specify the row to plot.
            bins (np.ndarray, optional): Array of bin values corresponding to the signal indices. Defaults to an empty array.
            ax (plt.Axes, optional): Matplotlib Axes object for plotting. If None, a new figure and axes are created.
            show (bool, optional): Whether to display the plot immediately. Defaults to False.
            row (int, optional): Row index to plot if the signal is 2D. Defaults to 0.

        Raises:
            ValueError: If the specified row index is out of bounds for a 2D signal.
        """
        if signal.ndim == 2:
            if row >= signal.shape[0]:
                raise ValueError(f"Row index {row} is out of bounds for signal with {signal.shape[0]} rows.")
            signal = signal[row]

        det_bins, T = self.compute(signal, bins=bins)

        signal_dB = 20 * np.log10(np.abs(signal))
        T_dB = 20 * np.log10(T)

        if not ax:
            fig, ax = plt.subplots()

        # Adapt plotting based on whether bins are provided or not
        x_axis = bins if bins.size > 0 else range(len(signal))

        ax.plot(x_axis, signal_dB, label='Signal')
        ax.plot(x_axis, T_dB, label='CFAR Threshold', linestyle='--')
        ax.plot(det_bins, signal_dB[det_bins], 'ro', label='Detections')
        ax.set_xlabel('Bins' if bins.size > 0 else 'Cell Index')
        ax.set_ylabel('Amplitude (dB)')
        ax.set_title('1D CA-CFAR Detector')
        plt.legend()

        if show:
            plt.show()