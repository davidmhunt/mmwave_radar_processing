import numpy as np
from scipy.signal import ZoomFFT,find_peaks

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors._processor import _Processor

class DopplerAzimuthProcessor(_Processor):
    """Process Doppler-azimuth data for radar applications.
    Note the coordinate system of the Doppler-azimuth processing x-forward, y-left

    Args:
        _Processor (_type_): _description_
    """

    def __init__(
            self,
            config_manager: ConfigManager,
            num_angle_bins:int = 64,
            valid_angle_range: np.ndarray = np.array(
                [np.deg2rad(-60), np.deg2rad(60)]),
            min_zoom_fft_vel_span = 0.1) -> None:

        #range bins
        self.num_range_bins = None
        self.range_bins = None
        #velocity bins
        self.vel_bins:np.ndarray = None
        self.zoomed_vel_bins:np.ndarray = None

        #parameters zoom FFT
        self.min_zoom_fft_vel_span = min_zoom_fft_vel_span #m/s

        #angle bins
        self.num_angle_bins = num_angle_bins
        self.phase_shifts:np.ndarray = None
        self.angle_bins:np.ndarray = None
        self.valid_angle_range = valid_angle_range
        self.valid_angle_mask = None
        self.valid_angle_bins = None

        #load the configuration and configure the response 
        super().__init__(config_manager)

    
    def configure(self):
        
        #compute the range bins
        self.num_range_bins = self.config_manager.get_num_adc_samples(profile_idx=0)
        self.range_bins = np.arange(
            start=0,
            step=self.config_manager.range_res_m,
            stop=self.config_manager.range_max_m - self.config_manager.range_res_m/2 + 1e-3
        )

        #compute the velocity bins
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

        #determine the valid angle bins
        self.valid_angle_mask = (self.angle_bins >= self.valid_angle_range[0]) &\
             (self.angle_bins <= self.valid_angle_range[1])
        self.valid_angle_bins = self.angle_bins[self.valid_angle_mask]

    
    def apply_range_vel_hanning_window(self,
            adc_cube: np.ndarray):
        
        #rangeFFT - apply hanning window
        hanning_window_range = np.hanning(adc_cube.shape[1])
        adc_cube_windowed = adc_cube * hanning_window_range[np.newaxis, :, np.newaxis]

        #apply hanning window across chirps
        hanning_window_chirp = np.hanning(adc_cube.shape[2])
        adc_cube_windowed = adc_cube_windowed * hanning_window_chirp[np.newaxis, np.newaxis, :]

        return adc_cube_windowed
    
    def range_fft_and_filter(
            self,
            adc_cube: np.ndarray,
            range_window: np.ndarray = np.array([])):
        """Compute the range FFT and filter to a specified range window.

        Args:
            adc_cube (np.ndarray): adc cube indexed by [rx,samp,chirp]
            range_window (np.ndarray): array of min and max range to keep indexed by [min_range,max_range]

        Returns:
            np.ndarray: adc cube after applying the range FFT and filtering to the specified range window
        """

        #determine range bins to use
        if range_window.size == 0:
            range_window = np.array([0, self.config_manager.range_max_m])

        #rangeFFT - compute along the sample dimension
        #now indexed by [rx,range,chirp]
        adc_cube_range_fft = np.fft.fft(adc_cube, axis=1)

        #filter the response to the specified range window
        mask = (self.range_bins >= range_window[0]) & (self.range_bins <= range_window[1])
        adc_cube_range_fft = adc_cube_range_fft[:, mask, :]

        return adc_cube_range_fft

    def zoom_fft(self,
            
            adc_cube_range_fft_rearranged: np.ndarray,
            vel_range:np.ndarray,
            num_samples:int):
        """Computes the zoom FFT on the input ADC cube.
            Args:
                adc_cube_range_fft_rearranged (np.ndarray): ADC cube after range FFT and rearrangement.
                    Note, this is expected to be indexed as [range,chirp,rx]
                vel_range (np.ndarray): Velocity range for the zoom FFT.
                num_samples (int): Number of samples to use for the zoom FFT.
            Returns:
                np.ndarray: Zoom FFT response.
            """
        
        
        #compute the sampling frequency
        fs = 1 / (self.config_manager.vel_res_m_s)

        #compute the start and stop frequencies
        freq_start = vel_range[0] * fs / self.config_manager.vel_max_m_s
        freq_stop = vel_range[1] * fs / self.config_manager.vel_max_m_s
        
        #TODO: Figure out why the factor of 2 is needed here
        zoom = ZoomFFT(num_samples, [freq_start, freq_stop], fs=fs * 2)

        #modify size to adjust for number of samples
        adc_cube_range_fft_rearranged =\
              adc_cube_range_fft_rearranged[:, :num_samples, :]
        zoom_resp = zoom(adc_cube_range_fft_rearranged,axis=1)
        zoom_resp = np.abs(np.fft.fft(zoom_resp, axis=2))

        return zoom_resp

    def set_zoomed_fft_vel_bins(self, vel_range):
        """Set the velocity bins for the zoomed FFT.

        Args:
            vel_range (np.ndarray): The velocity range for the zoomed FFT.

        Returns:
            np.ndarray: The velocity bins for the zoomed FFT.
        """

        # Generate 100 velocity bins less than or equal to zero
        neg_vel_bins = np.linspace(
            start=vel_range[0],
            stop=min(-1e-4, vel_range[1]),
            num=100 if vel_range[0] <= 0 else 0,
            endpoint=False
        )

        # Generate 100 velocity bins greater than zero
        pos_vel_bins = np.linspace(
            start=max(1e-4, vel_range[0]),
            stop=vel_range[1],
            num=100 if vel_range[1] > 0 else 0,
            endpoint=False
        )

        # Concatenate the velocity bins
        if neg_vel_bins.size > 0 and pos_vel_bins.size > 0:
            self.zoomed_vel_bins = np.concatenate((neg_vel_bins, pos_vel_bins))
        elif neg_vel_bins.size > 0:
            self.zoomed_vel_bins = neg_vel_bins
        elif pos_vel_bins.size > 0:
            self.zoomed_vel_bins = pos_vel_bins
        else:
            self.zoomed_vel_bins = np.array([])
        
        return self.zoomed_vel_bins
    
    def precise_doppler_azimuth_fft(
            self,
            adc_cube_range_fft:np.ndarray,
            shift_angle:bool = True,
            vel_range:np.ndarray = np.array([-0.25,0.25]),
    ):
        """Computes the precise (zoomFFT) Doppler-azimuth response from the range-FFT ADC cube.
            The input ADC cube is expected to be indexed as [rx, range, chirp], while the
            output response is indexed as [range, doppler, azimuth]. This function performs
            a 2D FFT across chirps and receivers for each range bin to obtain the
            Doppler and azimuth information.
            Args:
                adc_cube_range_fft (np.ndarray): The range-FFT ADC cube with shape [num_rx, num_range_bins, num_chirps].
                    Note that this is the adc_cube after the range FFT has been applied
                shift_angle (bool, optional): A flag to indicate whether to shift the angle axis. Defaults to True.
                vel_range (np.ndarray, optional): Velocity range for the zoom FFT. Defaults to np.array([-0.25,0.25]).
            Returns:
                np.ndarray: The Doppler-azimuth response with shape [num_range_bins, num_chirps, num_angle_bins].
            """

        #re-arrange the data to be indexed by [range,chirp,rx]
        num_rx, num_range_bins, num_chirps = adc_cube_range_fft.shape
        data = np.zeros((num_range_bins,num_chirps,self.num_angle_bins),dtype=complex)
        data[:,:,0:num_rx] = np.transpose(adc_cube_range_fft, (1,2,0))

        

        # Ensure vel_range is within the valid velocity range
        vel_range[0] = max(vel_range[0], -1 * self.config_manager.vel_max_m_s)
        vel_range[1] = min(vel_range[1], self.config_manager.vel_max_m_s)

        # Ensure that the bounds have a minimum spread
        vel_spread = 2 * self.min_zoom_fft_vel_span
        if (vel_range[1] - vel_range[0]) < vel_spread:
            dist_to_max_bound = abs(vel_range[1] - self.config_manager.vel_max_m_s)
            dist_to_min_bound = abs(vel_range[0] - (-1 * self.config_manager.vel_max_m_s))
            if dist_to_max_bound > dist_to_min_bound:
                vel_range[1] = vel_range[0] + vel_spread
            elif dist_to_min_bound > dist_to_max_bound:
                vel_range[0] = vel_range[1] - vel_spread

        #configure the zoomed FFT velocity bins
        self.set_zoomed_fft_vel_bins(vel_range)

        #compute negative velocity zoom FFT
        neg_vel_vals = self.zoomed_vel_bins[self.zoomed_vel_bins <= 0]
        num_samples = len(neg_vel_vals)
        if num_samples > 0 and \
            np.abs(max(neg_vel_vals) - min(neg_vel_vals)) > self.min_zoom_fft_vel_span:
                vel_range = np.array([
                    np.min(neg_vel_vals),
                    np.max(neg_vel_vals)
                ]) + 2 * self.config_manager.vel_max_m_s

                neg_vel_zoom_result = self.zoom_fft(
                    adc_cube_range_fft_rearranged=data,
                    vel_range=vel_range,
                    num_samples=num_samples
                )
        else:
            neg_vel_zoom_result = np.zeros((num_range_bins,num_samples,self.num_angle_bins),dtype=float)

        #compute positive velocity zoom FFT
        pos_vel_vals = self.zoomed_vel_bins[self.zoomed_vel_bins > 0]
        num_samples = len(pos_vel_vals)
        if num_samples > 0 and\
              np.abs(max(pos_vel_vals) - min(pos_vel_vals)) > self.min_zoom_fft_vel_span:
                vel_range = np.array([
                    np.min(pos_vel_vals),
                    np.max(pos_vel_vals)
                ])

                pos_vel_zoom_result = self.zoom_fft(
                    adc_cube_range_fft_rearranged=data,
                    vel_range=vel_range,
                    num_samples=num_samples
                )
        else:
            pos_vel_zoom_result = np.zeros((num_range_bins,num_samples,self.num_angle_bins),dtype=float)

        #combine the negative and positive velocity zoom FFT results
        zoom_result = np.concatenate((neg_vel_zoom_result, pos_vel_zoom_result), axis=1)

        if shift_angle:
            zoom_result = np.fft.fftshift(zoom_result, axes=(2))

        return zoom_result


    def coarse_doppler_azimuth_fft(
            self,
            adc_cube_range_fft: np.ndarray,
            shift_angle: bool = True
    ):
        """Computes the coarse Doppler-azimuth response from the range-FFT ADC cube.
            The input ADC cube is expected to be indexed as [rx, range, chirp], while the
            output response is indexed as [range, doppler, azimuth]. This function performs
            a 2D FFT across chirps and receivers for each range bin to obtain the
            Doppler and azimuth information.
            Args:
                adc_cube_range_fft (np.ndarray): The range-FFT ADC cube with shape [num_rx, num_range_bins, num_chirps].
                    Note that this is the adc_cube ater the range FFT has been applied
                shift_angle (bool, optional): A flag to indicate whether to shift the angle axis. Defaults to True.
            Returns:
                np.ndarray: The Doppler-azimuth response with shape [num_range_bins, num_chirps, num_angle_bins].
            """


        #re-arrange the data to be indexed by [range,chirp,rx]
        num_rx, num_range_bins, num_chirps = adc_cube_range_fft.shape
        data = np.zeros((num_range_bins,num_chirps,self.num_angle_bins),dtype=complex)
        data[:,:,0:num_rx] = np.transpose(adc_cube_range_fft, (1,2,0))

        #compute 2D fft across chirps and receivers for each range bin
        if shift_angle:
            shift_axes = (1,2)
        else:
            shift_axes = (1)
        

        resp = np.abs(
            np.fft.fftshift(
                np.fft.fft2(data, axes=(1, 2)), 
                axes=shift_axes #if error, use axes=(1,2) - observed weird behavior in angular
            )
        )

        return resp
    
    def detect_peaks_rows(
            self,
            doppler_azimuth_resp_mag: np.ndarray,
            vel_bins:np.ndarray,
            min_threshold_dB: float = 30.0,
            ):
        """Detect peaks in each row of Doppler-azimuth response (peaks in velocity bins).
        Args:
            doppler_azimuth_resp_mag (np.ndarray): Doppler-azimuth response magnitude.
            vel_bins (np.ndarray): Velocity bins corresponding to the Doppler dimension.
            min_threshold_dB (float, optional): Minimum threshold in dB. Defaults to 30.0.
        Returns:
            Tuple[np.ndarray]: An Nx2 array where each row contains the (angle (radians), velocity) of a detected peak.
        """
        
        doppler_azimuth_resp_dB = 20 * np.log10(np.abs(doppler_azimuth_resp_mag))
        thresholded_val = np.max(doppler_azimuth_resp_dB) - min_threshold_dB
        idxs = doppler_azimuth_resp_dB <= thresholded_val
        doppler_azimuth_resp_dB[idxs] = thresholded_val

        # find the peak in each row of the elevation response
        peak_angles = []
        peak_vels = []

        for ridx, row in enumerate(doppler_azimuth_resp_dB):
            peaks, _ = find_peaks(row)
            if peaks.size > 0:
                # choose the peak with the highest amplitude
                best = peaks[np.argmax(row[peaks])]
                peak_angles.append(self.valid_angle_bins[best])
                peak_vels.append(vel_bins[ridx])
        
        return np.stack(
            [np.array(peak_angles),np.array(peak_vels)],axis=1
        )
    
    def detect_peak_zero_az(
            self,
            doppler_azimuth_resp_mag: np.ndarray,
            vel_bins:np.ndarray,
            min_threshold_dB: float = 30.0,
            ):
        """Detect peaks at the zero azimuth bin (corresponding to vx).
        Args:
            doppler_azimuth_resp_mag (np.ndarray): Doppler-azimuth response magnitude.
            vel_bins (np.ndarray): Velocity bins corresponding to the Doppler dimension.
            min_threshold_dB (float, optional): Minimum threshold in dB. Defaults to 30.0.
        Returns:
            Tuple[np.ndarray]: An Nx2 array where each row contains the (angle (radians), velocity) of a detected peak.
        """
        
        doppler_azimuth_resp_dB = 20 * np.log10(np.abs(doppler_azimuth_resp_mag))
        thresholded_val = np.max(doppler_azimuth_resp_dB) - min_threshold_dB
        idxs = doppler_azimuth_resp_dB <= thresholded_val
        doppler_azimuth_resp_dB[idxs] = thresholded_val

        #find the column of the zero azimuth bin
        zero_az_col = np.argmin(np.abs(self.valid_angle_bins))
        peaks,_ = find_peaks(doppler_azimuth_resp_dB[:,zero_az_col])
        if peaks.size > 0:
            # choose the peak with the highest amplitude
            best = peaks[np.argmax(doppler_azimuth_resp_dB[peaks,zero_az_col])]
            return np.array([0.0, vel_bins[best]])
        else:
            return np.empty(shape=(0,2))
        
        

    def process(
            self,
            adc_cube: np.ndarray,
            rx_antennas: np.ndarray = np.array([]),
            range_window: np.ndarray = np.array([]),
            shift_angle:bool = True,
            use_precise_fft: bool = False,
            precise_vel_range:np.ndarray = np.array([-0.25,0.25])
            ) -> np.ndarray:
        """Compute a doppler-azimuth response for the radar

        Args:
            adc_cube (np.ndarray): adc cube indexed by [rx,samp,chirp]
            rx_antennas (np.ndarray): array of rx antenna indices to use
            range_window (np.ndarray): array of min and max range to keep indexed by [min_range,max_range]
            shift_angle (bool): shift the angle dimension to be centered at 0 when performing the 2D FFT
            use_precise_fft (bool): flag to indicate whether to use the precise (zoomFFT) Doppler-azimuth response
            precise_vel_range (np.ndarray): velocity range for the zoom FFT when using precise FFT indexed by [min_vel,max_vel]
        Returns:
            np.ndarray: doppler-azimuth response indexed by [vel,angle]
            that is the average across all samples
        """

        #specify the antennas to use for computing the response
        if rx_antennas.size > 0:
            adc_cube = adc_cube[rx_antennas, :, :]

        #Apply windows to avoid spectral leakage
        adc_cube_windowed = self.apply_range_vel_hanning_window(adc_cube)

        #apply range FFT and filter to the specified range window
        adc_cube_range_fft = self.range_fft_and_filter(
            adc_cube=adc_cube_windowed,
            range_window=range_window
        )

        if not use_precise_fft:
            resp = self.coarse_doppler_azimuth_fft(
                adc_cube_range_fft, 
                shift_angle=shift_angle)
        else:
            resp = self.precise_doppler_azimuth_fft(
                adc_cube_range_fft=adc_cube_range_fft,
                shift_angle=shift_angle,
                vel_range=precise_vel_range
            )

        
        #filter the angle bins to a given range
        resp = resp[:, :, self.valid_angle_mask]


        avg_resp = np.mean(resp, axis=0)

        return avg_resp
        