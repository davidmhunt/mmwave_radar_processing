"""Range-Doppler FFT processing utilities."""

import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors._processor import _Processor

class RangeDopplerProcessor(_Processor):

    """Process ADC cubes into Range-Doppler responses."""

    def __init__(
            self,
            config_manager: ConfigManager,
            **kwargs) -> None:

        """Initialize the processor and set up bin containers.

        Args:
            config_manager: Loaded configuration manager with radar parameters.
        """

        #phase shifts
        self.vel_bins:np.ndarray = None

        #range bins
        self.range_bins:np.ndarray = None

        #load the configuration and configure the response 
        super().__init__(config_manager)

    
    def configure(self):

        """Compute range and velocity bin centers from configuration."""
        
        self.vel_bins = np.arange(
            start=-1 * self.config_manager.vel_max_m_s,
            stop = self.config_manager.vel_max_m_s - self.config_manager.vel_res_m_s + 1e-3,
            step= self.config_manager.vel_res_m_s
        )

        #set the range bins
        self.range_bins = np.arange(
            start=0,
            step=self.config_manager.range_res_m,
            stop=self.config_manager.range_max_m - self.config_manager.range_res_m/2 + 1e-3)
    
    def apply_range_vel_hanning_window(self,
            adc_cube: np.ndarray):

        """Apply Hanning windows along range and velocity dimensions.

        Args:
            adc_cube: Raw ADC data cube with shape (n_rx, n_range_bins, n_chirps).

        Returns:
            Windowed ADC cube ready for Range-Doppler FFT.
        """
        
        #rangeFFT - apply hanning window
        hanning_window_range = np.hanning(adc_cube.shape[1])
        adc_cube_windowed = adc_cube * hanning_window_range[np.newaxis, :, np.newaxis]

        #velocity FFT - apply hanning window
        hanning_window_vel = np.hanning(adc_cube.shape[2])
        adc_cube_windowed = adc_cube_windowed * hanning_window_vel[np.newaxis, np.newaxis, :]

        return adc_cube_windowed

    def process(
            self,
            adc_cube: np.ndarray,
            rx_idx: int = 0,
            return_magnitude: bool = True,
            **kwargs) -> np.ndarray:

        """Generate the Range-Doppler response.

        Args:
            adc_cube: Raw ADC data cube with shape (n_rx, n_range_bins, n_chirps).
            rx_idx: Antenna index to return. Use -1 to return responses for all
                antennas instead of a specific one.
            return_magnitude: If True, return the magnitude response via ``np.abs``;
                otherwise return the complex FFT output.

        Returns:
            Range-Doppler response. If ``rx_idx`` is non-negative, the output
            shape is (n_range_bins, n_chirps). If ``rx_idx`` is -1, the output
            shape is (n_rx, n_range_bins, n_chirps). Complex or magnitude is
            determined by ``return_magnitude``.
        """

        adc_cube = self.apply_range_vel_hanning_window(adc_cube)

        #compute the Range-Doppler FFT
        response = np.fft.fftshift(
            x=np.fft.fft2(
                adc_cube,axes=(-2,-1)
            ),
            axes=-1
        )

        if return_magnitude:
            response = np.abs(response)

        if rx_idx >= 0:
            response = response[rx_idx,:,:]
        return response
    
