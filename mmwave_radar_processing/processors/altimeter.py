import numpy as np
from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.range_resp import RangeProcessor

class Altimeter(RangeProcessor):
    def __init__(
            self,
            config_manager: ConfigManager,
            min_altitude_m: float,
            zoom_search_region_m: float,
            altitude_search_limit_m: float,
            range_bias: float = 0.0) -> None:
        """
        Args:
            config_manager (ConfigManager): Radar configuration manager.
            min_altitude_m (float): Minimum altitude threshold.
            zoom_search_region_m (float): Width of the zoom FFT search region beyond the coarse measurement.
            altitude_search_limit_m (float): maximum range in meters from current altitude to search for new altitude
            range_bias (float, optional): Bias to apply to the altitude estimate. Defaults to 0.0.
        """
        super().__init__(config_manager)
        self.min_altitude_m = min_altitude_m
        self.zoom_search_region_m = zoom_search_region_m
        self.altitude_search_limit_m = altitude_search_limit_m
        self.range_bias = range_bias
        
        
        #currently estimated altitude
        self.current_altitude_measured_m = min_altitude_m #measured using radar
        self.current_altitude_corrected_m = min_altitude_m #corrected for bias

    def reset(self):

        self.current_altitude_measured_m = self.min_altitude_m
        return super().reset()

    def find_ground_peak(self,detected_peaks_m:np.ndarray):
        """
        Update the current altitude based on detected peaks from the range FFT response.

        Args:
            detected_peaks_m (np.ndarray): Detected peak ranges in meters.
        """
        if detected_peaks_m.size > 0:
            # Update current altitude to the highest detected peak within search limits
            valid_peaks = detected_peaks_m[
                (detected_peaks_m >= self.min_altitude_m) &\
                (np.abs(detected_peaks_m - self.current_altitude_measured_m) <=\
                  self.altitude_search_limit_m)
                ]
            if valid_peaks.size > 0:
                return np.min(valid_peaks)
            else:
                # If no valid peaks found, keep current altitude
                return -1.0
        else:
            # No peaks detected, keep current altitude
            return -1.0
        
    def _perform_coarse_fft(self, adc_cube: np.ndarray) -> np.ndarray:
        """Perform coarse FFT to get initial range estimate."""
        return self.coarse_fft(adc_cube=adc_cube, chirp_idx=0)

    def _get_coarse_peaks(self, coarse_fft: np.ndarray) -> np.ndarray:
        """Get the coarse peaks from the FFT result."""
        peak_rng_bins, _ = self.find_peaks(
            rng_resp_db=20 * np.log10(coarse_fft),
            rng_bins=self.range_bins,
            max_peaks=3
        )
        return peak_rng_bins

    def _refine_altitude_estimate(self, adc_cube: np.ndarray, ground_peak: float) -> float:
        """Refine the altitude estimate using zoom FFT."""
        min_range = 1e-6  # strictly greater than 0
        max_range = np.max(self.range_bins) - 1e-6  # strictly less than max

        range_start_m = max(min_range, ground_peak - self.zoom_search_region_m)
        range_end_m = min(max_range, ground_peak + self.zoom_search_region_m)

        zoom_avg, zoom_range_bins = self.zoom_fft(
            adc_cube=adc_cube,
            range_start_m=range_start_m,
            range_stop_m=range_end_m,
            chirp_idx=0
        )

        peak_rng_bins, _ = self.find_peaks(
            rng_resp_db=20 * np.log10(zoom_avg),
            rng_bins=zoom_range_bins,
            max_peaks=2
        )

        if peak_rng_bins.size > 0:
            return self.find_ground_peak(detected_peaks_m=peak_rng_bins)
        else:
            return -1.0

    def process(self, adc_cube: np.ndarray, precise_est_enabled:bool = True) -> np.ndarray:
        """
        Process the ADC cube to estimate the altitude (range to ground).

        Args:
            adc_cube (np.ndarray): (num rx antennas) x (num adc samples) x (num chirps)

        Returns:
            np.ndarray: Processed response for altitude estimation.
        """

        #compute coarse FFT to get initial range estimate
        coarse_fft = self._perform_coarse_fft(adc_cube)
        peak_rng_bins = self._get_coarse_peaks(coarse_fft)

        if peak_rng_bins.size == 0:
            return self.current_altitude_corrected_m

        #identify ground peak from detected peaks
        ground_peak = self.find_ground_peak(detected_peaks_m=peak_rng_bins)


        if ground_peak < 0:
            return self.current_altitude_corrected_m
        elif not precise_est_enabled:
            self.current_altitude_measured_m = ground_peak
            self.current_altitude_corrected_m = ground_peak + self.range_bias
            return self.current_altitude_corrected_m

        # Refine altitude estimate using zoom FFT around the ground peak
        refined_estimated_altitude_m = self._refine_altitude_estimate(adc_cube, ground_peak)
        if refined_estimated_altitude_m > 0:
            self.current_altitude_measured_m = refined_estimated_altitude_m
            self.current_altitude_corrected_m = refined_estimated_altitude_m + self.range_bias

        return self.current_altitude_corrected_m