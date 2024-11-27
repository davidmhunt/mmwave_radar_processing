import numpy as np
import scipy.constants as constants

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors._processor import _Processor

class SyntheticArrayProcessor(_Processor):

    def __init__(
            self,
            config_manager: ConfigManager,
            az_angle_bins_rad=\
                np.deg2rad(np.linspace(
                    start=-30,stop=30,num=60
                 )),
            el_angle_bins_rad=\
                np.deg2rad(np.linspace(
                    start=-30,
                    stop=30,
                    num=30
                ))) -> None:
        """_summary_

        Args:
            config_manager (ConfigManager): _description_
            az_angle_bins_rad (_type_, optional): _description_. Defaults to \np.deg2rad(np.linspace( start=-30,stop=30,num=60 )).
            el_angle_bins_rad (_type_, optional): _description_. Defaults to \np.deg2rad(np.linspace( start=-30, stop=30, num=30 )).
        Notes:
            NOTE: coordinate frame (x - right, y - out, z - up)
        """

        #range bins
        self.num_range_bins:int = None
        self.range_bins:np.ndarray = None

        #derrived configuration parameters
        self.chirps_per_frame:int = None
        self.chirp_period_us:float = None
        self.chirp_tx_masks:np.ndarray = None
        self.chirp_rx_positions_m:np.ndarray = None
        self.lambda_m:float = None

        #timing intervals
        self.chirp_start_times_us:np.ndarray = None

        #computing the array geometry
        self.p_x_m:np.ndarray = None #indexed by [receiver, chirp]
        self.p_y_m:np.ndarray = None #indexed by [receiver, chirp]
        self.p_z_m:np.ndarray = None #indexed by [receiver, chirp]

        self.array_geometry:np.ndarray = None #indexed by []

        #mesh grids for spherical plotting
        self.phis:np.ndarray = None #rotation az w.r.t y-axis
        self.thetas:np.ndarray = None #
        self.rhos:np.ndarray = None

        #mesh grid for cartesian plotting
        self.x_s:np.ndarray = None
        self.y_s:np.ndarray = None
        self.z_s:np.ndarray = None

        #load the configuration and configure the response
        super().__init__(config_manager)

    def configure(self):
        
        #configure the range bins
        self.num_range_bins = self.config_manager.get_num_adc_samples(
            profile_idx=0
        )
        self.range_bins = np.arange(
            start=0,
            stop=self.config_manager.range_max_m,
            step=self.config_manager.range_res_m
        )

        self._compute_key_radar_parameters()

        return
     
    def _compute_key_radar_parameters(self):

        #compute the radar wavelength
        start_freq_GHz = \
            float(self.config_manager.profile_cfgs[0]["startFreq_GHz"])
        self.lambda_m = constants.c / (start_freq_GHz * 1e9)
        
        #determine the number of chirps per frame
        chirp_cfgs_per_loop = self.config_manager.frameCfg_end_index \
            - self.config_manager.frameCfg_start_index + 1
        self.chirps_per_frame = self.config_manager.frameCfg_loops * \
            chirp_cfgs_per_loop

        #compute the chirp period
        self.chirp_period_us = \
            self.config_manager.profile_cfgs[0]["idleTime_us"] + \
            self.config_manager.profile_cfgs[0]["rampEndTime_us"]

        #compute the start time for each chirp
        self.chirp_start_times_us = np.arange(self.chirps_per_frame) * \
            self.chirp_period_us

        #determine which Tx is active for each chirp
        chirp_cfg_idxs = np.arange(
            start=self.config_manager.frameCfg_start_index,
            stop=self.config_manager.frameCfg_end_index + 1
        )
        chirp_cfg_tx_masks = np.array(
            [self.config_manager.chirp_cfgs[idx]["txMask"] \
             for idx in chirp_cfg_idxs]
        )

        self.chirp_tx_masks = np.tile(
            A=chirp_cfg_tx_masks,
            reps=self.config_manager.frameCfg_loops
        )

        #compute the position of each element 
        rx_z_coordinates = np.array([
            np.arange(self.config_manager.num_rx_antennas) * \
                  self.lambda_m/2
        ]).T
        self.chirp_rx_positions_m = np.tile(
            A=rx_z_coordinates,
            reps=(1,self.chirps_per_frame)
        )
        self.chirp_rx_positions_m[:,self.chirp_tx_masks == 2] += \
              self.lambda_m
        self.chirp_rx_positions_m[:,self.chirp_tx_masks == 4] +=\
              self.lambda_m * 2

        
        return

    def _generate_array_geometries(
        self,
        vels:np.ndarray=np.array([0,0,0])
    ):
        """Generate the array geometry based on the velocity of the vehicle

        Args:
            vels (np.ndarray, optional): The velocity of the radar platform
            in the radar coordinate frame. Expressed in [x,y,z] where
            +x is right, +y is out, and +z is up. 
            Defaults to np.array([0,0,0]).
        """        
        
        #compute the x coordinates (based on platform motion and array goemetry)
        """
            NOTE: As the Tx and Rx are both moving in the x direction
            the round trip distance is multiplied by a factor of two
            NOTE: If TX2 is active, an additional phase shift is applied
            due to the array geometry
        """
        chirp_coords = 2 * self.chirp_start_times_us * 1e-6 * vels[0]
        chirp_coords[self.chirp_tx_masks == 2] += self.lambda_m / 2
        self.p_x_m = np.tile(
            chirp_coords,
            reps=(self.config_manager.num_rx_antennas,1))
        #compute the y coordinates (purely based on motion of platform)
        """
            NOTE: As the Tx and Rx are both moving in the x direction
            the round trip distance is multiplied by a factor of two
        """
        chirp_coords = self.chirp_start_times_us * 1e-6 * vels[1]
        self.p_y_m = np.tile(
            chirp_coords,
            reps=self.config_manager.num_rx_antennas)
        
        #compute the z coordinates (based on platform motion and array geometry)
        """
            NOTE: As the Tx and Rx are both moving in the x direction
            the round trip distance is multiplied by a factor of two
            NOTE: If TX2 is active, an additional phase shift is applied
            due to the array geometry
        """
        #compute the array motion due to velocity updates
        chirp_coords = 2 * self.chirp_start_times_us * 1e-6 * vels[2]
        positions_vel = np.tile(
            chirp_coords,
            reps=(self.config_manager.num_rx_antennas,1))
        self.p_z_m = self.chirp_rx_positions_m + positions_vel
        



    def _compute_mesh_grids(self):

        pass

    def process(self,adc_cube:np.ndarray) -> np.ndarray:

        pass