import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors._processor import _Processor

class MicroDopplerProcessor(_Processor):

    def __init__(
            self,
            config_manager: ConfigManager,
            target_ranges:np.ndarray = [0,1.0],
            num_frames_history:int = 20) -> None:

        #velocity bins
        self.vel_bins:np.ndarray = None

        #range bins
        self.range_bins:np.ndarray = None

        #micro-doppler parameters
        self.target_ranges = target_ranges
        self.range_bin_idxs_to_keep:np.ndarray = None
        self.num_frames_history = num_frames_history
        self.time_bins:np.ndarray = None
        self.micro_doppler_resp:np.ndarray = None

        #load the configuration and configure the response 
        super().__init__(config_manager)

    
    def configure(self):
                
        self.vel_bins = np.arange(
            start=-1 * self.config_manager.vel_max_m_s,
            stop = self.config_manager.vel_max_m_s - self.config_manager.vel_res_m_s,
            step= self.config_manager.vel_res_m_s
        )

        #set the range bins
        self.range_bins = np.arange(
            start=0,
            step=self.config_manager.range_res_m,
            stop=self.config_manager.range_max_m - self.config_manager.range_res_m/2)
        
        #configure micro-doppler parameters
        self.range_bin_idxs_to_keep = np.logical_and(
                self.range_bins >= self.target_ranges[0],
                self.range_bins <= self.target_ranges[1]
            ).astype(np.bool_)
        
        self.micro_doppler_resp = np.zeros(
            shape=(self.vel_bins.shape[0],self.num_frames_history)
        )

        #frame period 
        frame_period = self.config_manager.frameCfg_periodicity_ms * 1e-3
        self.time_bins = np.linspace(
            start=0,
            stop=self.num_frames_history * frame_period,
            num=self.num_frames_history
        )
        


    def process(self, adc_cube: np.ndarray, rx_idx = 0) -> np.ndarray:


        new_data = adc_cube[rx_idx,:,:]

        #compute the Range-Doppler FFT
        response = np.abs(np.fft.fftshift(
            x=np.fft.fft2(
                new_data,axes=(-2,-1)
            ),
            axes=1
        ))

        slice_to_keep = response[self.range_bin_idxs_to_keep,:]
        #take the average or max of the values across each range slice
        # slice_to_keep = np.mean(slice_to_keep,axis=0)
        slice_to_keep = np.max(slice_to_keep,axis=0)

        #increment current micro-doppler response
        self.micro_doppler_resp[:,1:] = self.micro_doppler_resp[:,0:-1]

        #save the micro-doppler response
        self.micro_doppler_resp[:,0] = slice_to_keep
        return self.micro_doppler_resp
    