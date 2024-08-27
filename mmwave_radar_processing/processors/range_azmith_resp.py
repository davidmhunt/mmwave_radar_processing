import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors._processor import _Processor

class RangeAzimuthProcessor(_Processor):

    def __init__(
            self,
            config_manager: ConfigManager,
            num_angle_bins:int = 64) -> None:

        #phase shifts
        self.num_angle_bins = num_angle_bins
        self.phase_shifts:np.ndarray = None

        #range bins
        self.num_range_bins:int = None
        self.range_bins:np.ndarray = None

        #mesh grids for polar and cartesian plotting
        self.thetas:np.ndarray = None
        self.rhos:np.ndarray = None
        self.x_s:np.ndarray = None
        self.y_s:np.ndarray = None

        #load the configuration and configure the response 
        super().__init__(config_manager)

    
    def configure(self):
        
        #set the range bins
        self.num_range_bins = self.config_manager.get_num_adc_samples(profile_idx=0)
        self.range_bins = np.arange(
            start=0,
            step=self.config_manager.range_res_m,
            stop=self.config_manager.range_max_m)

        #compute the phase shifts
        self.num_rx_antennas = self.config_manager.num_rx_antennas
        self.phase_shifts = np.arange(
            start=np.pi,
            stop= -np.pi - 2 * np.pi/(self.num_angle_bins - 1),
            step=-2 * np.pi / (self.num_angle_bins - 1)
        )

        #round the last entry to be exactly pi
        self.phase_shifts[-1] = -1 * np.pi

        #fix the phase shifts to be correct
        self.phase_shifts = self.phase_shifts * -1 #TODO: check this value

        #compute the angle bins
        self.angle_bins = np.arcsin(self.phase_shifts / np.pi)

        #compute the mesh grid
        self.thetas,self.rhos = np.meshgrid(self.angle_bins,self.range_bins)
        self.x_s = np.multiply(self.rhos,np.sin(self.thetas))
        self.y_s = np.multiply(self.rhos,np.cos(self.thetas))

    def process(self, adc_cube: np.ndarray, chirp_idx = 0) -> np.ndarray:

        #initialize the data by zero padding the angle bins
        data = np.zeros(
            shape=(
                self.num_range_bins,
                self.num_angle_bins
            ),
            dtype=complex
        )
        data[:,0:adc_cube.shape[0]] = np.transpose(adc_cube[:,:,chirp_idx])

        #compute the range azimuth response
        response = np.abs(
            np.fft.fftshift(
                x=np.fft.fft2(data,axes=(0,1)),
                axes=1
            )
        )

        return response
    