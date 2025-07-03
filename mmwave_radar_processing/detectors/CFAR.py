import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from numpy.lib.stride_tricks import sliding_window_view


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

class OsCFAR_1D_Vectorized:
    def __init__(
            self,
            config_manager,
            num_guard_cells=4,
            num_training_cells=10,
            k_rank=None,
            false_alarm_rate=0.1,
            resp_border_cells=3,
            mode="valid"
    ):
        self.num_guard_cells = num_guard_cells
        self.num_training_cells = num_training_cells
        self.resp_border_cells = resp_border_cells
        self.config_manager = config_manager
        self.mode = mode

        self.N = 2 * num_training_cells
        self.k_rank = k_rank if k_rank is not None else self.N // 2
        self.alpha = self.N * (false_alarm_rate ** (-1 / self.N) - 1)

        self.ranges = np.linspace(
            0,
            self.config_manager.range_max_m,
            self.config_manager.get_num_adc_samples(profile_idx=0)
        )

        # Determine valid indices for plotting
        total_guard_train = num_guard_cells + num_training_cells
        if self.mode == "valid":
            self.valid_idxs = np.ones_like(self.ranges, dtype=bool)
            self.valid_idxs[:self.resp_border_cells + total_guard_train] = False
            self.valid_idxs[-(self.resp_border_cells + total_guard_train):] = False
        elif self.mode == "full":
            self.valid_idxs = np.ones_like(self.ranges, dtype=bool)
            self.valid_idxs[:self.resp_border_cells] = False
            self.valid_idxs[-self.resp_border_cells:] = False
        else:
            raise ValueError("mode must be 'valid' or 'full'")

        print(f"Vectorized OS-CFAR using k={self.k_rank}, alpha={self.alpha:.2f}, mode={self.mode}")

    def compute(self, signal: np.ndarray):
        signal = np.abs(signal)
        num_cells = len(signal)
        det_idxs = np.zeros_like(signal, dtype=bool)
        T_full = np.zeros_like(signal)

        window_size = 2 * (self.num_training_cells + self.num_guard_cells) + 1
        guard_start = self.num_training_cells
        guard_end = guard_start + 2 * self.num_guard_cells + 1
        training_mask = np.ones(window_size, dtype=bool)
        training_mask[guard_start:guard_end] = False

        if self.mode == "valid":
            # Use only valid region (cutting off edges)
            start = self.resp_border_cells + self.num_training_cells + self.num_guard_cells
            stop = num_cells - self.resp_border_cells - self.num_training_cells - self.num_guard_cells
            signal_cut = signal[start:stop]

            windows = sliding_window_view(signal, window_shape=window_size)
            windows = windows[start:stop]

            training_cells = windows[:, training_mask]
            sorted_training = np.sort(training_cells, axis=1)
            P_n = sorted_training[:, self.k_rank]
            T = self.alpha * P_n

            dets = signal_cut > T
            det_idxs[start:stop] = dets
            T_full[start:stop] = T

        elif self.mode == "full":
            # Pad signal to compute CFAR across full width
            pad_width = self.num_guard_cells + self.num_training_cells
            signal_padded = np.pad(signal, pad_width=(pad_width,), mode='constant', constant_values=0)

            windows = sliding_window_view(signal_padded, window_shape=window_size)  # shape: (N, window_size)
            training_cells = windows[:, training_mask]
            sorted_training = np.sort(training_cells, axis=1)
            P_n = sorted_training[:, self.k_rank]
            T = self.alpha * P_n

            cut = signal
            dets = cut > T
            det_idxs = dets
            T_full = T

        return det_idxs, T_full

    def plot_cfar(self, signal, ax=None, show=False):
        det_idxs, T = self.compute(signal)

        signal_dB = 20 * np.log10(np.abs(signal) + 1e-12)
        T_dB = 20 * np.log10(T + 1e-12)

        det_ranges = self.ranges[det_idxs]
        det_mags = signal_dB[det_idxs]

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.ranges, signal_dB, label='Signal')
        ax.plot(self.ranges[self.valid_idxs], T_dB[self.valid_idxs], label='OS-CFAR Threshold', linestyle='--')
        ax.plot(det_ranges, det_mags, 'ro', label='Detections')
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Amplitude (dB)')
        ax.set_title(f'1D OS-CFAR Detector (mode="{self.mode}")')
        ax.legend()

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