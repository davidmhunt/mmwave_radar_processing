import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors._processor import _Processor

class VirtualArrayReformatter(_Processor):

    def __init__(self, config_manager: ConfigManager) -> None:

        #array to keep track of the chirp_cfg index for each chirp in the ADC config
        self.chirp_cfg_idxs:np.ndarray = None
        self.chirp_cfg_idxs_for_frame:np.ndarray = None
        self.chirp_cfgs_per_loop:int = 0
        self.adc_samples_per_chirp:int = 0

        super().__init__(config_manager)

    def configure(self):
        
        self.chirp_cfg_idxs = np.arange(
            start=self.config_manager.frameCfg_start_index,
            stop=self.config_manager.frameCfg_end_index + 1
        )

        self.chirp_cfg_idxs_for_frame = np.tile(
            A=self.chirp_cfg_idxs,
            reps=self.config_manager.frameCfg_loops
        )

        #compute other required variables
        self.chirp_cfgs_per_loop = self.config_manager.frameCfg_end_index \
            - self.config_manager.frameCfg_start_index + 1
        self.adc_samples_per_chirp = \
            self.config_manager.get_num_adc_samples(profile_idx=0)

        return
    
    def process(self, adc_cube: np.ndarray) -> np.ndarray:

        #initialize a new array that will align all chirps in the same frame
        out_array = np.zeros(
            shape=(self.config_manager.num_rx_antennas * self.chirp_cfgs_per_loop,
                   self.adc_samples_per_chirp,
                   self.config_manager.frameCfg_loops),
            dtype=complex
        )

        for i in range(len(self.chirp_cfg_idxs)):

            out_array_start = i * self.config_manager.num_rx_antennas
            out_array_stop = out_array_start + self.config_manager.num_rx_antennas

            config_idx = self.chirp_cfg_idxs[i]
            
            chirps_with_cfg = self.chirp_cfg_idxs_for_frame == config_idx
            
            out_array[out_array_start:out_array_stop,:,:] = \
                adc_cube[0:self.config_manager.num_rx_antennas,:,chirps_with_cfg]

        return out_array