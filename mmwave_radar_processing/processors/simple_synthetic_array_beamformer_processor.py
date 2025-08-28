import numpy as np
import scipy.constants as constants

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors._processor import _Processor
from mmwave_radar_processing.detectors.CFAR import CaCFAR_1D

class SyntheticArrayBeamformerProcessor(_Processor):

    def __init__(
            self,
            config_manager: ConfigManager,
            receiver_idx:int = 0,
            chirp_cfg_idx:int = 0,
            num_frames:int = 2,
            stride:int = 1,
            az_angle_bins_rad=\
                np.deg2rad(np.linspace(
                    start=-30,stop=30,num=60
                 )),
            el_angle_bins_rad=np.array([0]),
            min_vel=0.2,
            max_vel_change=0.1,) -> None:
        """_summary_

        Args:
            config_manager (ConfigManager): _description_
            receiver_idx (int, optional): The receiver index to use for
                the synthetic array. Defaults to 0.
            num_frames (int, optional): The number of frames to use for
                the synthetic array. Defaults to 2.
            stride (int, optional): The frame stride to use when constructing
                the synthetic array. Defaults to 1.
            az_angle_bins_rad (_type_, optional): _description_. Defaults to \np.deg2rad(np.linspace( start=-30,stop=30,num=60 )).
            el_angle_bins_rad (_type_, optional): _description_. Defaults to \np.deg2rad(np.linspace( start=-30, stop=30, num=30 )).
            min_vel (float, optional): Minimum velocity to perform SAR at. Defaults to 0.2.
            max_vel_change (float, optional): Maximum velocity change to allow for valid processing. Defaults to 0.1.
        Notes:
            NOTE: coordinate frames (x - forward, y - left, z - up),
        """

        #sampling parameters
        self.receiver_idx = receiver_idx
        self.chirp_cfg_idx = chirp_cfg_idx
        self.num_frames = num_frames
        self.stride = stride

        #range bins
        self.num_range_bins:int = None
        self.range_bins:np.ndarray = None

        #derrived configuration parameters
        self.chirps_per_frame:int = None
        self.chirp_period_us:float = None
        self.chirp_cfg_idxs:np.ndarray = None
        self.valid_chirps_mask:np.ndarray = None #mask for obtaining the valid chirps in an array
        # self.chirp_tx_masks:np.ndarray = None
        # self.chirp_rx_positions_m:np.ndarray = None
        self.lambda_m:float = None

        #timing intervals
        self.chirp_start_times_us:np.ndarray = None

        #computing the array geometry
        self.p_x_m:np.ndarray = None #indexed by [chirp]
        self.p_y_m:np.ndarray = None #indexed by [chirp]
        self.p_z_m:np.ndarray = None #indexed by [chirp]

        self.p:np.ndarray = None #indexed by [(x,y,z),chirp]
        # self.p_reshaped:np.ndarray = None #reshaped for efficient computation

        #compute the beam pointing vectors
        self.az_angle_bins_rad = az_angle_bins_rad
        self.el_angle_bins_rad = el_angle_bins_rad
        self.d:np.ndarray = None #indexed by [(x,y,z), theta, phis]

        #define the output grid
        self.beamformed_resp:np.ndarray = None #indexed by [rho, theta, phi]

        #mesh grids for spherical plotting - indexed by (rho, theta, phi)
        self.rhos:np.ndarray = None
        self.thetas:np.ndarray = None #az w.r.t y-axis w.r.t x-axis
        self.phis:np.ndarray = None #el w.r.t y-axis w.r.t x-axis

        #mesh grid for cartesian plotting - indexed by (rho, theta, phi) idx
        self.x_s:np.ndarray = None
        self.y_s:np.ndarray = None
        self.z_s:np.ndarray = None

        #parameters for when to generate the Synthetic response
        self.min_vel = min_vel
        self.max_vel_change = max_vel_change
        self.last_vel:np.ndarray = np.array([0,0,0])

        #flag for tracking if the response is valid or not
        self.array_geometry_valid = False

        #load the configuration and configure the response
        super().__init__(config_manager)

    def configure(self):

        #compute key radar parameters
        self._compute_key_radar_parameters()

        #compute output mesh grids
        self._compute_mesh_grids()
        
        #compute the beam stearing vectors
        self._compute_beam_stearing_vectors()

        #define the output grid
        self._init_out_resp()

        return
     
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

        #compute the chirp period
        self.chirp_period_us = \
            self.config_manager.profile_cfgs[0]["idleTime_us"] + \
            self.config_manager.profile_cfgs[0]["rampEndTime_us"]

        #determine which Tx is active for each chirp
        chirp_cfg_idxs = np.arange(
            start=self.config_manager.frameCfg_start_index,
            stop=self.config_manager.frameCfg_end_index + 1
        )
        self.chirp_cfg_idxs = np.tile(
            A=chirp_cfg_idxs,
            reps=self.config_manager.frameCfg_loops
        ).flatten()

        # Generate a mask of valid chirps based on self.chirp_cfg_idx
        self.valid_chirps_mask = (self.chirp_cfg_idxs == self.chirp_cfg_idx)

        # Refine the valid chirps mask using the stride variable
        valid_chirp_indices = np.where(self.valid_chirps_mask)[0]
        refined_indices = valid_chirp_indices[::self.stride]
        self.valid_chirps_mask = np.zeros_like(self.valid_chirps_mask, dtype=bool)
        self.valid_chirps_mask[refined_indices] = True

        #compute the start time for each chirp
        all_chirp_start_times_us = np.arange(self.chirps_per_frame) * self.chirp_period_us
        self.chirp_start_times_us = all_chirp_start_times_us[self.valid_chirps_mask]

        # chirp_cfg_tx_masks = np.array(
        #     [self.config_manager.chirp_cfgs[idx]["txMask"] \
        #      for idx in chirp_cfg_idxs]
        # )

        # self.chirp_tx_masks = np.tile(
        #     A=chirp_cfg_tx_masks,
        #     reps=self.config_manager.frameCfg_loops
        # )

        #compute the position of each element (y axis)
        # rx_coordinates = np.array([
        #     np.arange(self.config_manager.num_rx_antennas) * \
        #           self.lambda_m/2
        # ]).T
        # self.chirp_rx_positions_m = np.tile(
        #     A=rx_coordinates,
        #     reps=(1,self.chirps_per_frame)
        # )
        # self.chirp_rx_positions_m[:,self.chirp_tx_masks == 2] += \
        #       self.lambda_m
        # self.chirp_rx_positions_m[:,self.chirp_tx_masks == 4] +=\
        #       self.lambda_m * 2

        return
    
    def _compute_mesh_grids(self):

        #generate the mesh grid for the beamformed response
        self.rhos, self.thetas, self.phis = np.meshgrid(
            self.range_bins, self.az_angle_bins_rad, self.el_angle_bins_rad,
            indexing="ij"
        )
        self.x_s = np.multiply(
            np.multiply(self.rhos,np.cos(self.thetas)),
            np.cos(self.phis)
        )
        self.y_s = np.multiply(
            np.multiply(self.rhos,np.sin(self.thetas)),
            np.cos(self.phis)
        )
        self.z_s = np.multiply(self.rhos,np.sin(self.phis))

        return

    def _check_array_geometry_valid(
            self,
            vels:np.ndarray=np.array([0,0,0])
    ):  
        vel_dif = np.abs(self.last_vel - vels)

        #check to ensure min vel (in x,y) and that vel hasn't changed too much
        if np.linalg.norm(vels[0:2]) > self.min_vel and \
            np.linalg.norm(vel_dif) < self.max_vel_change:

            self.array_geometry_valid = True
        else:
            self.array_geometry_valid = False
        
        #update the latset vel
        self.last_vel = vels

        return self.array_geometry_valid

    def generate_array_geometries(
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

        #check for valid array geometry
        if self._check_array_geometry_valid(vels):
            
            #compute the x coordinates (based on platform motion and array goemetry)
            """
                NOTE: As the Tx and Rx are both moving in the x direction
                the round trip distance is multiplied by a factor of two
                NOTE: If TX2 is active, an additional phase shift is applied
                due to the array geometry
            """
            self.p_x_m = 2 * self.chirp_start_times_us * 1e-6 * vels[0]
            
            #compute the y coordinates (purely based on motion of platform)
            """
                NOTE: As the Tx and Rx are both moving in the x direction
                the round trip distance is multiplied by a factor of two
            """
            self.p_y_m = 2 * self.chirp_start_times_us * 1e-6 * vels[1]
            
            #compute the z coordinates (based on platform motion and array geometry)
            """
                NOTE: As the Tx and Rx are both moving in the x direction
                the round trip distance is multiplied by a factor of two
                NOTE: If TX2 is active, an additional phase shift is applied
                due to the array geometry
            """
            #compute the array motion due to velocity updates
            self.p_z_m = 2 * self.chirp_start_times_us * 1e-6 * vels[2]
            
            #save newly generated array geometry
            self.p = np.array([self.p_x_m,self.p_y_m,self.p_z_m])
        
        return
        
        

    def _compute_beam_stearing_vectors(self):

        #generate a mesh grid
        thetas, phis = np.meshgrid(
            self.az_angle_bins_rad,
            self.el_angle_bins_rad,
            indexing="ij")

        #compute the x,y,z coordinates of the beamforming
        x = np.multiply(np.cos(thetas),np.cos(phis))
        y = np.multiply(np.sin(thetas),np.cos(phis))
        z = np.sin(phis)

        #save the pointing vectors
        self.d = np.array([x,y,z])

    def _init_out_resp(self):

        self.beamformed_resp = np.zeros(
            shape=(self.range_bins.shape[0],
                   self.az_angle_bins_rad.shape[0],
                   self.el_angle_bins_rad.shape[0]),
            dtype=complex
        )

    def compute_response_at_stearing_angle(
            self,
            adc_cube_reshaped:np.ndarray,
            steering_vector:np.ndarray
    )->np.ndarray:
        """Compute the beamformed range response for a given steering
            vector

        Args:
            adc_cube_reshaped (np.ndarray): The input adc cube indexed by
                [adc_sample, chirp]. NOTE: this must be the 
                same order as the array geometry's p vector which is indexed
                by [(x,y,z), chirp] at a specific receiver IDX
            stearing_vector (np.ndarray): the stearing vector to compute the 
                response along expressed as an [x,y,z] unit vector

        Returns:
            np.ndarray: the beamformed range response as a complex signal
        """

        #reshape the adc cube further
        adc_cube_reshaped = np.reshape(
                            adc_cube_reshaped,
                            (adc_cube_reshaped.shape[0],-1),
                            order="F"
                        )

        #compute the phase shift to apply to each received signal
        shifts = np.exp(
                    1j * 2 * np.pi * \
                    (steering_vector @ self.p) / self.lambda_m)

        #apply the phase shifts
        #reshape the shifts to be correct
        shifts = np.reshape(shifts,(1,shifts.shape[0]))
        beamformed_resps = np.multiply(adc_cube_reshaped,shifts)
        beamformed_resp = np.sum(beamformed_resps,axis=1)

        #generate a hanning window
        window = np.hanning(self.num_range_bins)

        #compute the response
        return np.fft.fft(beamformed_resp * window)

    def process(self,adc_cube:np.ndarray) -> np.ndarray:
        """Compute the beamformed synthetic response

        Args:
            adc_cube (np.ndarray): the adc cube for the synthetic array
                indexed by [receiver, sample, and chirp]
        Returns:
            np.ndarray: _description_
        """

        #copy the adc cube to not modify the original
        adc_cube = adc_cube.copy()

        #filter the adc cube for only the relevant chirps
        adc_cube = adc_cube[self.receiver_idx,:,:] #filter for specific receiver
        adc_cube = adc_cube[:,self.valid_chirps_mask] #filter for valid chirps

        # Apply a Hanning window across each column of the adc cube
        hamming_window = np.hamming(adc_cube.shape[1]).reshape(1, -1)
        adc_cube = adc_cube * hamming_window
        

        if self.array_geometry_valid:
            for az_angle_idx in range(self.az_angle_bins_rad.shape[0]):
                for el_angle_idx in range(self.el_angle_bins_rad.shape[0]):

                    steering_vector = self.d[:,az_angle_idx,el_angle_idx]

                    # Get beamformed response at the steering angle
                    resp = self.compute_response_at_stearing_angle(
                        adc_cube_reshaped=adc_cube,
                        steering_vector=steering_vector
                    )

                    # Save the response
                    self.beamformed_resp[:,az_angle_idx,el_angle_idx] = resp

            # Commented out CFAR computation on the beamformed response
            # self.compute_2D_cfar_on_beamformed_resp()

            return self.beamformed_resp

        else:
            return None