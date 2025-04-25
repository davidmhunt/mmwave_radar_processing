import numpy as np
import scipy.constants as constants

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors._processor import _Processor
from mmwave_radar_processing.processors.virtual_array_reformater import VirtualArrayReformatter

class StripMapSARProcessor(_Processor):

    def __init__(self,
                 config_manager,
                 az_angle_range_rad:np.ndarray = \
                    np.deg2rad(np.array([-30,30]))):
        
        #virtual array re-formatter (for handling virtual arrays)
        if config_manager.virtual_antennas_enabled:
            self.virtual_array_reformatter = \
                VirtualArrayReformatter(config_manager)
        

        #derrived configuration parameters
        self.chirps_per_frame:int = None
        self.chirp_period_us:float = None
        self.chirp_tx_masks:np.ndarray = None
        self.chirp_rx_positions_m:np.ndarray = None
        self.lambda_m:float = None
        
        #phase shifts
        self.phase_shifts:np.ndarray = None
        self.angle_bins_rad:np.ndarray = None
        self.az_angle_range_rad:np.ndarray = az_angle_range_rad

        #range bins
        self.num_range_bins:int = None
        self.radar_range_bins:np.ndarray = None

        #initializing the SAR range/az bins and coming
        #up with an array for which bins can produce valid
        # sar Results for the given sensor height
        self.valid_ranges_slice:slice = None
        self.valid_angles_slice:slice = None
        self.ground_range_bins:np.ndarray = None
        self.ground_az_bins_rad:np.ndarray = None

        #mesh grids for polar and cartesian plotting
        self.thetas:np.ndarray = None
        self.rhos:np.ndarray = None
        self.x_s:np.ndarray = None
        self.y_s:np.ndarray = None

        super().__init__(config_manager)
    
    def configure(self):

        self._compute_key_radar_parameters()

    def _compute_key_radar_parameters(self):

        #configure the range bins
        self.num_range_bins = self.config_manager.get_num_adc_samples(
            profile_idx=0
        )
        
        self.range_bins = np.linspace(
            start=0,
            stop=self.config_manager.range_max_m,
            num=self.num_range_bins
        )

        #compute the radar wavelength
        start_freq_GHz = \
            float(self.config_manager.profile_cfgs[0]["startFreq_GHz"])
        self.lambda_m = constants.c / (start_freq_GHz * 1e9)
        
        #determine the number of chirps per frame
        chirp_cfgs_per_loop = self.config_manager.frameCfg_end_index \
            - self.config_manager.frameCfg_start_index + 1
        self.chirps_per_frame = self.config_manager.frameCfg_loops * \
            chirp_cfgs_per_loop
        
        #compute the phase shifts
        self.phase_shifts = np.linspace(
            start=np.pi,
            stop=-np.pi,
            num=self.config_manager.frameCfg_loops
        )

        #compute the chirp period
        self.chirp_period_us = \
            self.config_manager.profile_cfgs[0]["idleTime_us"] + \
            self.config_manager.profile_cfgs[0]["rampEndTime_us"]

        return

    def configure_array_geometry(
            self,
            vel_m_per_s:float,
            sensor_height_m:float,
            max_SAR_distance:float):
        #NOTE: For now, assuming propagation in x

        #multiply by 2 because tx and rx are moving
        d_rx = 2* self.chirp_period_us * 1e-6 * vel_m_per_s

        #compute the angle bins
        self.angle_bins_rad = np.arcsin(
            self.phase_shifts * self.lambda_m) / \
                (2 * np.pi * d_rx)
        
        #compute the valid angle bins
        min_angle_idx = np.argmin(np.abs(
            self.angle_bins_rad - \
            np.min(self.az_angle_range_rad)
        ))

        max_angle_idx = np.argmin(np.abs(
            self.angle_bins_rad - \
            np.max(self.az_angle_range_rad)
        ))
        self.valid_angles_slice = slice(
            np.min([min_angle_idx,max_angle_idx]),
            np.max([min_angle_idx,max_angle_idx]))
        self.ground_az_bins_rad = \
            self.angle_bins_rad[self.valid_angles_slice]
        #determine the valid range bins
        min_rng_idx = np.nonzero(
            self.range_bins > sensor_height_m)[0][0]
        max_rng_idx = np.nonzero(
            self.range_bins < max_SAR_distance)[0][-1]
        self.valid_ranges_slice = slice(min_rng_idx,max_rng_idx)

        #compute the SAR ground range bins from the valid range bins
        self.ground_range_bins = np.sqrt(
            np.power(self.range_bins[self.valid_ranges_slice],2) - \
            np.power(d_rx,2)
        )
        
        #recompute the mesh grid
        self.thetas,self.rhos = np.meshgrid(
            self.ground_az_bins_rad,self.ground_range_bins,
            indexing='xy')
        self.x_s = np.multiply(self.rhos,np.cos(self.thetas))
        self.y_s = np.multiply(self.rhos,np.sin(self.thetas))
        
        return
    
    def process(self,
                adc_cube,
                vel_m_per_s:float,
                sensor_height_m:float = 0.24,
                rx_index = 0,
                max_SAR_distance:float = 1.5):
        
        if self.config_manager.virtual_antennas_enabled:
            adc_cube = self.virtual_array_reformatter.process(
                adc_cube=adc_cube
            )
        
        #comput the array geometry
        self.configure_array_geometry(
            vel_m_per_s=vel_m_per_s,
            sensor_height_m=sensor_height_m,
            max_SAR_distance=max_SAR_distance
        )
        
        #get the data from the specific rx receiver
        rx_data = adc_cube[rx_index,:,:]

        #compute the enire 2D fft
        response = np.fft.fftshift(
            x=np.fft.fft2(
                a=rx_data,axes=(-2,-1)
            ),
            axes=1
        )

        response = response[self.valid_ranges_slice,
                            self.valid_angles_slice]
        
        return response
    

