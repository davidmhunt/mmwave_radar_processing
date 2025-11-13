import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors._processor import _Processor

class RangeDopplerProcessor(_Processor):

    def __init__(
            self,
            config_manager: ConfigManager) -> None:

        #phase shifts
        self.vel_bins:np.ndarray = None

        #range bins
        self.range_bins:np.ndarray = None

        #load the configuration and configure the response 
        super().__init__(config_manager)

    
    def configure(self):
        
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
        
        #rangeFFT - apply hanning window
        hanning_window_range = np.hanning(adc_cube.shape[1])
        adc_cube_windowed = adc_cube * hanning_window_range[np.newaxis, :, np.newaxis]

        #velocity FFT - apply hanning window
        hanning_window_vel = np.hanning(adc_cube.shape[2])
        adc_cube_windowed = adc_cube_windowed * hanning_window_vel[np.newaxis, np.newaxis, :]

        return adc_cube_windowed

    def process(self, adc_cube: np.ndarray, rx_idx = 0) -> np.ndarray:

        adc_cube = self.apply_range_vel_hanning_window(adc_cube)

        data = adc_cube[rx_idx,:,:]

        #compute the Range-Doppler FFT
        response = np.abs(np.fft.fftshift(
            x=np.fft.fft2(
                data,axes=(-2,-1)
            ),
            axes=1
        ))
        return response
    