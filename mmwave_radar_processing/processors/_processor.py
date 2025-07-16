import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager

class _Processor:
    
    def __init__(
            self,
            config_manager:ConfigManager) -> None:
        
        self.config_manager = config_manager
        
        #variables to track history
        self.history_estimated = []
        self.history_gt = []

        self.configure()
        
        return
    
    def configure(self):
        """Function to be implemented by child class to initialize all necessary components
        """

        pass

    def reset(self):
        """Function to be implemented by child class to reset any internal state if needed
        """

        self.history_estimated = []
        self.history_gt = []
        pass

    def update_history(
            self,
            estimated:np.ndarray=np.empty(0),
            ground_truth:np.ndarray=np.empty(0)) -> None:
        """Update the internal history of estimated and ground truth values

        Args:
            estimated (np.ndarray): 1D array of estimated values.
            ground_truth (np.ndarray): 1D array of ground truth values.
        """
        if estimated.size > 0:
            self.history_estimated.append(estimated)
        if ground_truth.size > 0:
            self.history_gt.append(ground_truth)

    def process(self,adc_cube:np.ndarray) -> np.ndarray:
        """Function implemented by child class to process an ADC cube to obtain a desired response

        Args:
            adc_cube (np.ndarray): (num rx antennas) x (num adc samples) x (num chirps) ADC cube corresponding to
                a specific frame a radar data

        Returns:
            np.ndarray: np array corresponding to the computed response
        """
        pass
