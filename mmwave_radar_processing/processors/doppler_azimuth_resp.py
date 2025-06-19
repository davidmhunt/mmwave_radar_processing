import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors._processor import _Processor

class DopplerAzimuthProcessor(_Processor):

    def __init__(
            self,
            config_manager: ConfigManager,
            num_angle_bins:int = 64) -> None:

        #velocity bins
        self.vel_bins:np.ndarray = None

        #angle bins
        self.num_angle_bins = num_angle_bins
        self.phase_shifts:np.ndarray = None
        self.angle_bins:np.ndarray = None

        #load the configuration and configure the response 
        super().__init__(config_manager)

    
    def configure(self):
        
        self.vel_bins = np.arange(
            start=-1 * self.config_manager.vel_max_m_s,
            stop = self.config_manager.vel_max_m_s - self.config_manager.vel_res_m_s + 1e-3,
            step= self.config_manager.vel_res_m_s
        )

        #compute the phase shifts
        self.num_rx_antennas = self.config_manager.num_rx_antennas
        self.phase_shifts = np.arange(
            start=np.pi,
            stop= -np.pi - 2 * np.pi/(self.num_angle_bins - 1),
            step=-2 * np.pi / (self.num_angle_bins - 1)
        )

        #round the last entry to be exactly pi
        self.phase_shifts[-1] = -1 * np.pi

        #compute the angle bins
        self.angle_bins = np.arcsin(self.phase_shifts / np.pi)


    def process(self, adc_cube: np.ndarray) -> np.ndarray:
        """Compute a doppler-azimuth response for the radar

        Args:
            adc_cube (np.ndarray): adc cube indexed by [rx,samp,chirp]

        Returns:
            np.ndarray: doppler-azimuth response indexed by [vel,angle]
                that is the average across all samples
        """

        num_rx, num_samples, num_chirps = adc_cube.shape

        data = np.zeros((num_samples,num_chirps,self.num_angle_bins),dtype=complex)

        #re-arrange data for doppler-azimuth processing (now indexed by [samp,chirp,rx])
        data[:,:,0:num_rx] = np.transpose(adc_cube, (1,2,0))

        #compute 2D fft across 
        resp = np.abs(
            np.fft.fftshift(
                np.fft.fft2(data, axes=(1, 2)), 
                axes=(1, 2)
            )
        )

        #compute the mean across the sample dimension
        avg_resp = np.mean(resp, axis=0)

        return avg_resp
    