import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager

class _Processor:
    
    def __init__(
            self,
            config_manager:ConfigManager) -> None:
        
        self.config_manager = config_manager

        self.configure()
        
        return
    
    def configure(self):
        """Function to be implemented by child class to initialize all necessary components
        """

        pass

    def process(self,adc_cube:np.ndarray) -> np.ndarray:
        """Function implemented by child class to process an ADC cube to obtain a desired response

        Args:
            adc_cube (np.ndarray): (num rx antennas) x (num adc samples) x (num chirps) ADC cube corresponding to
                a specific frame a radar data

        Returns:
            np.ndarray: np array corresponding to the computed response
        """
        pass
