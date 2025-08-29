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
            min_vel:np.ndarray=np.array([0.17,0.0,0.0]),
            max_vel:np.ndarray=np.array([0.25,0.05,0.05]),
            max_vel_stdev:np.ndarray=np.array([0.1,0.1,0.1]),) -> None:
        """Initialize the SyntheticArrayBeamformerProcessor.

        Args:
            config_manager (ConfigManager): The configuration manager for radar settings.
            receiver_idx (int, optional): The receiver index to use for
                the synthetic array. Defaults to 0.
            chirp_cfg_idx (int, optional): The chirp configuration index to use. Defaults to 0.
            num_frames (int, optional): The number of frames to use for
                the synthetic array. Defaults to 2.
            stride (int, optional): The frame stride to use when constructing
                the synthetic array. Defaults to 1.
            az_angle_bins_rad (np.ndarray, optional): Azimuth angle bins in radians. Defaults to 
                np.deg2rad(np.linspace(start=-30, stop=30, num=60)).
            el_angle_bins_rad (np.ndarray, optional): Elevation angle bins in radians. Defaults to 
                np.array([0]).
            min_vel (np.ndarray, optional): Minimum velocity in the radar coordinate frame 
                [x, y, z] to perform SAR at. Defaults to np.array([0.17, 0.0, 0.0]).
            max_vel (np.ndarray, optional): Maximum velocity in the radar coordinate frame 
                [x, y, z] to allow for valid processing. Defaults to np.array([0.25, 0.05, 0.05]).
            max_vel_stdev (np.ndarray, optional): Maximum velocity standard deviation in the radar 
                coordinate frame [x, y, z] to allow for valid processing. Defaults to np.array([0.1, 0.1, 0.1]).

        Notes:
            NOTE: Coordinate frames are defined as (x - forward, y - left, z - up).
            NOTE: Frames are indexed such that idx=-1 is the most recent frame.
            NOTE: Chirps are indexed such that idx=-1 is the most recent chirp.
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
        self.frame_period_ms:float = None
        self.chirp_period_us:float = None
        self.chirp_cfg_idxs:np.ndarray = None
        self.valid_chirps_mask:np.ndarray = None #mask for obtaining the valid chirps in an array
        self.lambda_m:float = None

        #timing intervals
        self.chirp_start_times_us:np.ndarray = None

        #history tracking
        self.history_avg_vel:np.ndarray = np.zeros(
            shape=(
                num_frames,
                3 #x,y,z position
            ),dtype=float) #indexed by [frame,(x,y,z)]
        
        self.history_acd_cube_valid_chirps:np.ndarray = None #indexed by [frame, sample, chirp]
        

        #computing the array geometry
        # self.p_x_m:np.ndarray = None #indexed by [chirp]
        # self.p_y_m:np.ndarray = None #indexed by [chirp]
        # self.p_z_m:np.ndarray = None #indexed by [chirp]

        self.p:np.ndarray = np.empty(shape=0) #indexed by [frame,(x,y,z),chirp]
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
        self.min_vel:np.array = min_vel
        self.max_vel:np.array = max_vel
        self.max_vel_stdev:np.array = max_vel_stdev

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
        
        #save the frame period
        self.frame_period_ms = self.config_manager.frameCfg_periodicity_ms

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

        all_chirp_start_times_us = np.arange(
            start=self.chirps_per_frame - 1,
            stop=-1,
            step=-1
        ) * -self.chirp_period_us
        # all_chirp_start_times_us = np.arange(self.chirps_per_frame) * self.chirp_period_us

        #indexed by chirp
        self.chirp_start_times_us = all_chirp_start_times_us[self.valid_chirps_mask]

        #initialize valid chirp history indexed by [frame, sample, chirp]
        self.history_acd_cube_valid_chirps = np.zeros(
            shape=(
                self.num_frames,
                self.num_range_bins,
                np.sum(self.valid_chirps_mask)
            ),
            dtype=complex
        )

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

    def _update_vel_history(
            self,
            current_vel:np.ndarray=np.array([0,0,0])
    ):  
        """
        Updates the velocity history and checks if the array geometry is valid based on 
        the velocity statistics.

        Args:
            current_vel (np.ndarray, optional): The current velocity vector as a NumPy array 
                with shape (3,). Defaults to np.array([0, 0, 0]).

        Returns:
            bool: True if the array geometry is valid based on the velocity mean and standard 
                deviation constraints, False otherwise.

        Notes:
            - The velocity history is updated by shifting the previous entries and inserting 
                the current velocity at the end.
            - The function computes the mean and standard deviation of the velocity history.
            - The array geometry is considered valid if:
                1. The mean velocity is within the specified bounds (`self.min_vel` and `self.max_vel`).
                2. The standard deviation of the velocity is below the specified threshold 
                    (`self.max_vel_stdev`).
        """

        self.history_avg_vel[0:-1,:] = self.history_avg_vel[1:,:]
        self.history_avg_vel[-1,:] = current_vel

        #compute summary statistics
        if self.history_avg_vel.shape[0] > 1:
            vel_stdev = np.std(self.history_avg_vel, axis=0)
            vel_mean = np.mean(self.history_avg_vel, axis=0)
        else:
            vel_stdev = np.array([0,0,0])
            vel_mean = current_vel

        # Check if vel_mean is within bounds and vel_stdev is below the threshold
        if np.all((self.min_vel <= np.abs(vel_mean)) & (np.abs(vel_mean) <= self.max_vel)) and \
           np.all(vel_stdev <= self.max_vel_stdev):
            self.array_geometry_valid = True
        else:
            self.array_geometry_valid = False

        return self.array_geometry_valid

    def _generate_frame_array_geometries(
        self,
        frame_vel:np.ndarray=np.array([0,0,0]),
        frame_start_pose:np.ndarray=np.array([0,0,0])
    ):
        """
        Generate the array geometry for a given frame based on the velocity 
        and starting position of the radar platform at the start of the frame.

        Args:
            frame_vel (np.ndarray, optional): The velocity of the radar platform 
                in the radar coordinate frame, expressed as [x, y, z] where 
                x is forward, y is left, and z is up. Defaults to np.array([0, 0, 0]).
            frame_start_pose (np.ndarray, optional): The starting position of the 
                radar platform in the radar coordinate frame, expressed as [x, y, z]. 
                Defaults to np.array([0, 0, 0]).

        Returns:
            tuple: A tuple containing:
            - frame_p (np.ndarray): The generated array geometry indexed by 
              [(x, y, z), chirp].
            - frame_end_pose (np.ndarray): The ending position of the radar 
              platform after the frame.

        Notes:
            - The round-trip distance is multiplied by a factor of two to account 
              for both the Tx and Rx motion.
            - The array geometry is computed based on the chirp start times, 
              velocity, and starting position.
        """

        #compute the x coordinates (based on platform motion and array goemetry)
        p_x_m = 2 * self.chirp_start_times_us * 1e-6 * frame_vel[0] \
            + frame_start_pose[0]
        p_y_m = 2 * self.chirp_start_times_us * 1e-6 * frame_vel[1] \
            + frame_start_pose[1]
        p_z_m = 2 * self.chirp_start_times_us * 1e-6 * frame_vel[2] \
            + frame_start_pose[2]

        #save newly generated array geometry
        frame_p = np.array([p_x_m,p_y_m,p_z_m])

        #compute the frame end pose
        #NOTE: using -1 as the last chirp is at 0.0 and all other chirps occur before that
        frame_end_pose = frame_start_pose + \
            2 * frame_vel * (-1 * self.frame_period_ms * 1e-3)
        
        return frame_p,frame_end_pose

    def _update_array_geometries(
            self,
            current_vel:np.ndarray=np.array([0,0,0])
    ):
        """Updates the array geometries based on the current velocity and historical velocity data.
        This function recalculates the positions of the array elements for multiple frames
        based on the provided current velocity and the historical average velocities. It ensures
        that the array geometry is updated to reflect the motion of the system.

        Args:
            current_vel (np.ndarray, optional): The current velocity of the system as a 3D vector 
                [vx, vy, vz]. Defaults to np.array([0, 0, 0]).
        Notes:
            - The function initializes the array geometry storage `self.p` if it is empty.
            - The geometry for each frame is calculated in reverse order, starting from the most
                recent frame and propagating backward.
            - The `_generate_frame_array_geometries` method is used to compute the geometry for
                each frame based on the velocity and starting pose.
        """
        
        #update the velocity history
        self._update_vel_history(current_vel=current_vel)

        #TODO: Make this condutional on if the array geometry is valid or not
        if self.p.shape[0] == 0:
            self.p = np.zeros(
                shape=(
                    self.num_frames,
                    3, #x,y,z
                    np.sum(self.valid_chirps_mask) #valid chirps
                )
            )

        current_frame_start_pose = np.array([0,0,0])

        for frame_idx in range(self.num_frames-1,-1,-1):

            frame_p,frame_end_pose = self._generate_frame_array_geometries(
                frame_vel=self.history_avg_vel[frame_idx,:],
                frame_start_pose=current_frame_start_pose
            )


            self.p[frame_idx, :, :] = frame_p

            current_frame_start_pose = frame_end_pose
        

        
        

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

    def compute_response_at_steering_angle(
            self,
            adc_cube:np.ndarray,
            array_geometry:np.ndarray,
            steering_vector:np.ndarray
    )->np.ndarray:
        """Compute the beamformed range response for a given steering
            vector

        Args:
            adc_cube (np.ndarray): The input adc cube indexed by
                [adc_sample, chirp]. 
            array_geometry (np.ndarray): the array geometry indexed by
                [(x,y,z), chirp] at a specific receiver IDX
                (obtained from p vector)
            stearing_vector (np.ndarray): the stearing vector to compute the 
                response along expressed as an [x,y,z] unit vector

        Returns:
            np.ndarray: the beamformed range response as a complex signal

        NOTE: The adc_cube must be generated using the  
            same order as the array geometry's p vector which is indexed
            by [(x,y,z), chirp] at a specific receiver IDX
        """

        #reshape the adc cube further
        # adc_cube = np.reshape(
        #                     adc_cube,
        #                     (adc_cube.shape[0],-1),
        #                     order="F"
        #                 )


        #compute the phase shift to apply to each received signal
        shifts = np.exp(
                    1j * 2 * np.pi * \
                    (steering_vector @ array_geometry) / self.lambda_m)

        #apply the phase shifts
        #reshape the shifts to be correct
        shifts = np.reshape(shifts,(1,shifts.shape[0]))
        beamformed_resps = np.multiply(adc_cube,shifts)
        beamformed_resp = np.sum(beamformed_resps,axis=1)

        #generate a hanning window
        window = np.hanning(self.num_range_bins)

        #compute the response
        return np.fft.fft(beamformed_resp * window)

    def process(self,adc_cube:np.ndarray,current_vel:np.ndarray) -> np.ndarray:
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
        #now is indexed by [sample, chirp]

        #append the adc cube to now be the history
        self.history_acd_cube_valid_chirps[0:-1,:,:] = \
            self.history_acd_cube_valid_chirps[1:,:,:]
        self.history_acd_cube_valid_chirps[-1,:,:] = adc_cube

        #update the array geometries
        self._update_array_geometries(current_vel=current_vel)

        #reshape the adc_cube
        adc_cube_reshaped = \
            self.history_acd_cube_valid_chirps.transpose((1,0,2)).reshape(
                self.history_acd_cube_valid_chirps.shape[1], #number of samples per chirp
                -1
            )
        
        #reshpae the array geometry
        array_geometry_reshaped = \
            self.p.transpose((1,0,2)).reshape(
                3,
                -1
            )

        # Apply a Hanning window across each column of the adc cube
        hamming_window = np.hamming(adc_cube_reshaped.shape[1]).reshape(1, -1)
        adc_cube_reshaped = adc_cube_reshaped * hamming_window


        if self.array_geometry_valid:
            for az_angle_idx in range(self.az_angle_bins_rad.shape[0]):
                for el_angle_idx in range(self.el_angle_bins_rad.shape[0]):

                    steering_vector = self.d[:,az_angle_idx,el_angle_idx]

                    # Get beamformed response at the steering angle
                    resp = self.compute_response_at_steering_angle(
                        adc_cube=adc_cube_reshaped,
                        array_geometry=array_geometry_reshaped,
                        steering_vector=steering_vector
                    )

                    # Save the response
                    self.beamformed_resp[:,az_angle_idx,el_angle_idx] = resp

            # Commented out CFAR computation on the beamformed response
            # self.compute_2D_cfar_on_beamformed_resp()

            return self.beamformed_resp

        else:
            return None