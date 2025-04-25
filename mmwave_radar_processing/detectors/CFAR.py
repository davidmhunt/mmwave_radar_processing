import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from mmwave_radar_processing.config_managers.cfgManager import ConfigManager


class CaCFAR_1D:

    def __init__(
            self,
            config_manager: ConfigManager,
            num_guard_cells = 4,
            num_training_cells = 10,
            false_alarm_rate = .1,
            resp_border_cells = 3,
            mode="valid"
            ) -> None:
        
        #compute the window size
        self.window_size = 2 * (num_training_cells + num_guard_cells) + 1
        self.num_guard_cells = num_guard_cells
        self.num_training_cells = num_training_cells

        #create the window to convolve over
        self.window = np.zeros(self.window_size,dtype=float)
        self.window[:num_training_cells] = 1
        self.window[-num_training_cells:] = 1

        #save the amount to clip the response by at the borders
        self.resp_border_cells = resp_border_cells
        

        #define the available ranges
        self.config_manager:ConfigManager = config_manager        
        self.ranges = np.linspace(
            start=0,
            stop=self.config_manager.range_max_m,
            num=self.config_manager.get_num_adc_samples(profile_idx=0)
        )

        #set the mode of CFAR computation
        self.mode = mode
        if mode=="valid":
            #compute the set of indicies that can produce CFAR detections
            self.valid_idxs = np.ones(shape=(self.ranges.shape[0]),dtype=bool)
            invalid_region_size = resp_border_cells + num_training_cells + num_guard_cells
            self.valid_idxs[:invalid_region_size] = False
            self.valid_idxs[-invalid_region_size:]= False
        elif mode=="full":
            self.valid_idxs = np.ones(shape=(self.ranges.shape[0]),dtype=bool)
            invalid_region_size = resp_border_cells
            self.valid_idxs[:invalid_region_size] = False
            self.valid_idxs[-invalid_region_size:]= False

        #compute the alpha for the threshold
        self.N = 2 * num_training_cells
        self.alpha = self.N*(false_alarm_rate**(-1/self.N) - 1)
        print("alpha: {} (mag), {} dB".format(
            self.alpha,
            20*np.log10(self.alpha)))
        
        return

    def compute(self,signal:np.ndarray):

        signal = np.abs(signal)

        if self.mode=="valid":
            return self._compute_valid_cfar(signal)
        elif self.mode=="full":
            return self._compute_full_cfar(signal)
        else:
            raise NotImplementedError
    
    def _compute_valid_cfar(self,signal:np.ndarray):

        #clip the signal
        signal_clipped = signal[
            self.resp_border_cells:-self.resp_border_cells]
        
        #compute the noise for each cell
        P_n = (1/self.N) * np.convolve(
            signal_clipped,
            self.window,mode='valid')
        
        #compute the threshold
        T = self.alpha * P_n

        #determine the detections (in the valid region)
        det_idxs = np.zeros(
            shape=(self.ranges.shape[0]),
            dtype=bool)
        
        det_idxs[self.valid_idxs] = signal[self.valid_idxs] > T

        return det_idxs,T
    
    def _compute_full_cfar(self,signal:np.ndarray):
        
        #clip the signal
        signal_clipped = signal[
            self.resp_border_cells:-self.resp_border_cells]
        
        #pad the signal
        padded_signal = np.pad(
            array = signal_clipped, 
            pad_width= (
                self.num_training_cells + self.num_guard_cells,
                self.num_training_cells + self.num_guard_cells),
            mode='constant',
            constant_values=(0, 0))

        P_n = (1/self.N) * np.convolve(
            padded_signal,self.window,mode='valid')
        T = self.alpha * P_n

        #determine the detections (in the valid region)
        det_idxs = np.zeros(
            shape=(self.ranges.shape[0]),
            dtype=bool)
        det_idxs[self.valid_idxs] = signal_clipped > T

        #determine the detections (in the valid region)
        det_idxs = np.zeros(
            shape=(self.ranges.shape[0]),
            dtype=bool)
        det_idxs[self.valid_idxs] = signal_clipped > T

        return det_idxs,T
    
    def plot_cfar(self,signal,ax:plt.Axes=None,show=False):

        det_idxs,T = self.compute(signal)

        signal_dB = 20 * np.log10(np.abs(signal))
        T_dB = 20 * np.log10(T)

        det_ranges = self.ranges[det_idxs]
        det_mags = signal_dB[det_idxs]

        if not ax:
            fig,ax = plt.subplots()
        
        ax.plot(self.ranges,signal_dB, label='Signal')
        ax.plot(self.ranges[self.valid_idxs],T_dB, label='CFAR Threshold', linestyle='--')
        ax.plot(det_ranges, det_mags, 'ro', label='Detections')
        ax.set_xlabel('Cell Index')
        ax.set_ylabel('Amplitude')
        ax.set_title('1D CA-CFAR Detector')
        plt.legend()

        if show:
            plt.show()


class CaCFAR_2D:
    """2D CFAR class intended to operate on Range-Azimuth plots
    """
    def __init__(
            self,
            num_guard_cells:np.ndarray= np.array([5,5]),
            num_training_cells:np.ndarray = np.array([10,10]),
            false_alarm_rate = .1,
            resp_border_cells:np.ndarray = np.array([5,5]),
            mode="valid"
            ) -> None:
        
        #compute the window size
        self.window_size = np.array([
            2 * (num_training_cells[0] + num_guard_cells[0]) + 1,
            2 * (num_training_cells[1] + num_guard_cells[1]) + 1
        ])
        self.num_guard_cells:np.array = num_guard_cells
        self.num_training_cells:np.array = num_training_cells

        #create the window to convolve over
        self.window = np.zeros(
            shape=(self.window_size[0],self.window_size[1]),
            dtype=float)
        self.window[:num_training_cells[0],:] = 1
        self.window[-num_training_cells[0]:,:] = 1
        self.window[:,:num_training_cells[1]] = 1
        self.window[:,-num_training_cells[1]:] = 1        

        #save the amount to clip the response by at the borders
        self.resp_border_cells = resp_border_cells
        

        #define the available ranges and angles
        #NOTE: set on first call for computing the response
        self.range_idxs = np.zeros(shape=0,dtype=float)
        self.az_angle_idxs = np.zeros(shape=0,dtype=float)

        #set the mode of CFAR computation
        self.mode = mode
        self.valid_idxs = np.zeros(shape=0,dtype=bool)

        #compute the alpha for the threshold
        self.N = np.sum(self.window)
        self.alpha = self.N*(false_alarm_rate**(-1/self.N) - 1)
        print("alpha: {} (mag), {} dB".format(
            self.alpha,
            20*np.log10(self.alpha)))
        
        return
        
    def configure_valid_regions(
            self,
            num_range_bins:int,
            num_az_bins:int):
        """_summary_

        Args:
            num_range_bins (int): _description_
            num_az_bins (int): _description_
        """

        #define the available ranges and angles
        #NOTE: set on first call for computing the response
        self.range_idxs = np.arange(num_range_bins)
        self.az_angle_idxs = np.arange(num_az_bins)


        #determine the size of the invalid region size based on 
        #the convolution mode
        if self.mode=="valid":
            invalid_region_size = np.array([
                self.resp_border_cells[0] + \
                  self.num_training_cells[0] + self.num_guard_cells[0],
                self.resp_border_cells[1] + \
                  self.num_training_cells[1] + self.num_guard_cells[1]])
        elif self.mode=="full":
            invalid_region_size = self.resp_border_cells
        
        self.row_valid_slice = slice(invalid_region_size[0],-invalid_region_size[0])
        self.col_valid_slice = slice(invalid_region_size[1],-invalid_region_size[1])



    def compute(self, signal: np.ndarray):

        if self.range_idxs.shape[0] == 0:
            self.configure_valid_regions(
                num_range_bins=signal.shape[0],
                num_az_bins=signal.shape[1]
            )

        signal = np.abs(signal)

        if self.mode == "valid":
            return self._compute_valid_cfar(signal)
        elif self.mode == "full":
            return self._compute_full_cfar(signal)
        else:
            raise NotImplementedError
    
    def _compute_valid_cfar(self,signal:np.ndarray):

        #clip the signal
        signal_clipped = signal[
            self.resp_border_cells[0]:-self.resp_border_cells[0],
            self.resp_border_cells[1]:-self.resp_border_cells[1]]
        
        #compute the noise for each cell
        P_n = (1/self.N) * convolve2d(
            in1=signal_clipped,
            in2=self.window,
            mode='valid'
        )
        
        #compute the threshold
        T = self.alpha * P_n

        #determine the detections (in the valid region)
        det_idxs = np.zeros(
            shape=(self.range_idxs.shape[0],
                   self.az_angle_idxs.shape[0]),
            dtype=bool)
        
        det_idxs[self.row_valid_slice,
                 self.col_valid_slice] = \
                signal[self.row_valid_slice,
                       self.col_valid_slice] > T

        return det_idxs,T

    def _compute_full_cfar(self,signal:np.ndarray):
        
        signal_clipped = signal[
            self.resp_border_cells[0]:-self.resp_border_cells[0],
            self.resp_border_cells[1]:-self.resp_border_cells[1]]
        
        # Pad the signal 
        padded_signal = np.pad(
            signal_clipped,
            pad_width=(
                (self.num_training_cells[0] + self.num_guard_cells[0],
                    self.num_training_cells[0] + self.num_guard_cells[0]),
                (self.num_training_cells[1] + self.num_guard_cells[1],
                    self.num_training_cells[1] + self.num_guard_cells[1])),
            mode='constant',
            constant_values=((0, 0),(0,0))
        )

        #compute the noise for each cell
        P_n = (1/self.N) * convolve2d(
            in1=padded_signal,
            in2=self.window,
            mode='valid'
        )
        
        #compute the threshold
        T = self.alpha * P_n

        #determine the detections (in the valid region)
        det_idxs = np.zeros(
            shape=(self.range_idxs.shape[0],
                   self.az_angle_idxs.shape[0]),
            dtype=bool)
        
        det_idxs[self.row_valid_slice,
                 self.col_valid_slice] = \
                signal[self.row_valid_slice,
                       self.col_valid_slice] > T

        return det_idxs,T