import numpy as np
from scipy.signal import ZoomFFT
from scipy.signal import find_peaks

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors._processor import _Processor

class RangeProcessor(_Processor):
    """
    Altimeter processor for estimating altitude (range to ground) from radar ADC data.
    Inherits from _Processor. Uses a simple range FFT on a selected RX channel.
    Supports virtual antennas if enabled in the config.
    """
    def __init__(self, config_manager: ConfigManager):
        """
        Args:
            config_manager (ConfigManager): Radar configuration manager.
            rx_idx (int, optional): RX channel index to use for altitude estimation. Defaults to 0.
        """
        self.num_range_bins = None
        self.range_bins = None
        self.virtual_array_reformatter = None
        super().__init__(config_manager)

    def configure(self):
        # Set up range bins
        self.num_range_bins = self.config_manager.get_num_adc_samples(profile_idx=0)
        self.range_bins = np.arange(
            start=0,
            step=self.config_manager.range_res_m,
            stop=self.config_manager.range_max_m - self.config_manager.range_res_m/2
        )

    def coarse_fft(self, adc_cube: np.ndarray, chirp_idx:int = 0) -> np.ndarray:
        """
        Compute a coarse FFT for initial range estimation.
        This provides a broad view of the range profile with standard resolution.

        Args:
            adc_cube (np.ndarray): Complex ADC data as (num rx antennas) x (num adc samples) x (num chirps)
            chirp_idx (int): Index of the chirp to use for the FFT

        Returns:
            np.ndarray: Magnitude of coarse range FFT
        """
        
        adc_cube = adc_cube[:, :, chirp_idx]

        # Apply Hanning window to reduce spectral leakage
        hanning_window = np.hanning(adc_cube.shape[1])
        rx_data_windowed = adc_cube * hanning_window

        # Compute FFT along the sample axis
        fft_result = np.fft.fft(rx_data_windowed,axis=1)

        # Return magnitude
        fft_magnitude = np.abs(fft_result)
        fft_avg = np.mean(fft_magnitude, axis=0)
        return fft_avg

    def zoom_fft(self, adc_cube: np.ndarray, range_start_m: float, range_stop_m: float, chirp_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute a zoom FFT around a specific range of interest for higher resolution.
        This provides finer detail in a specific range window centered around the coarse estimate.

        Args:
            adc_cube (np.ndarray): Complex ADC data from all RX channels and chirps
                Shape: (num_rx, num_samples, num_chirps)
            range_start_m (float): Start range in meters for zoom window
            range_stop_m (float): Stop range in meters for zoom window
            chirp_idx (int, optional): Index of chirp to process. Defaults to 0.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing:
                - zoom_fft_magnitude: Magnitude of zoom FFT around the range window
                - zoom_range_bins: Corresponding range bins in meters
        """
        # Select chirp across all receivers
        adc_cube = adc_cube[:, :, chirp_idx]  # shape: (num_rx, num_samples)

        # Apply Hanning window
        num_samples = adc_cube.shape[1]
        hanning_window = np.hanning(num_samples)
        rx_data_windowed = adc_cube * hanning_window

        # Calculate effective sampling rate (1/spacing in meters)
        fs = 1 / self.config_manager.range_res_m # Convert to integer Hz

        # Convert range window to frequency domain
        freq_start = range_start_m * fs / self.config_manager.range_max_m
        freq_stop = range_stop_m * fs / self.config_manager.range_max_m

        # Create and apply ZoomFFT
        zoom = ZoomFFT(num_samples, [freq_start, freq_stop], fs=fs)
        zoom_result = zoom(rx_data_windowed, axis=1)  # shape: (num_rx, num_zoom_bins)

        # Average across receivers and compute magnitude
        zoom_avg = np.mean(np.abs(zoom_result), axis=0)
        
        # Create range axis for zoomed view
        num_zoom_bins = len(zoom_avg)
        zoom_range_bins = np.linspace(range_start_m, range_stop_m, num_zoom_bins)

        return zoom_avg, zoom_range_bins
    
    def find_peaks(
        self, 
        rng_resp_db: np.ndarray, 
        rng_bins: np.ndarray, 
        max_peaks: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """
        Find and filter the most significant peaks in the range response.

        Args:
            rng_resp_db (np.ndarray): Range response magnitude in dB
            rng_bins (np.ndarray): Array of range bin values in meters
            max_peaks (int, optional): Maximum number of peaks to return. Defaults to 3.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple containing:
                - Range values (in meters) for the top peaks
                - Magnitude values (in dB) for the top peaks
        """
        # Detect peaks in the range response using prominence-based detection
        # Prominence of 6 dB ensures we only detect significant peaks
        peaks, _ = find_peaks(rng_resp_db, prominence=6)

        if len(peaks) > 0:
            # Filter peaks to only keep those within 30 dB of the maximum peak
            # This helps eliminate noise and weak reflections
            peak_vals = rng_resp_db[peaks]
            max_peak = np.max(peak_vals)
            within_20dB = peak_vals >= (max_peak - 20)
            filtered_peaks = peaks[within_20dB]
            filtered_vals = peak_vals[within_20dB]

            # Sort peaks by magnitude in descending order
            # This ensures we keep the strongest reflections
            sorted_indices = np.argsort(filtered_vals)[::-1]
            sorted_peaks = filtered_peaks[sorted_indices]

            # Keep only the top N strongest peaks
            # These likely correspond to the most significant reflectors
            top_peaks = sorted_peaks[:max_peaks]
            peak_rng_bins = rng_bins[top_peaks]  # Convert peak indices to range values
            peak_vals = rng_resp_db[top_peaks]   # Get corresponding magnitude values

            return peak_rng_bins, peak_vals
        else:
            return np.array([]), np.array([])


    def process(self, adc_cube: np.ndarray, chirp_idx: int = 0) -> np.ndarray:
        """
        Process the ADC cube to estimate the altitude (range to ground).

        Args:
            adc_cube (np.ndarray): (num rx antennas) x (num adc samples) x (num chirps)

        Returns:
            np.ndarray: Magnitude of range FFT (range profile) for the selected RX channel.
        """
        # Select the RX channel
        rx_data = adc_cube[:, :, chirp_idx]
        # Average across chirps for improved SNR
        rx_data_mean = np.mean(rx_data, axis=1)
        # Compute range FFT
        range_fft = np.fft.fft(rx_data_mean, n=self.num_range_bins)
        range_profile = np.abs(range_fft)
        return range_profile 