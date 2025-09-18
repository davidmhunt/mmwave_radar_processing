import matplotlib.pyplot as plt
import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.range_angle_resp import RangeAngleProcessor
from mmwave_radar_processing.processors.range_doppler_resp import RangeDopplerProcessor
from mmwave_radar_processing.processors.doppler_azimuth_resp import DopplerAzimuthProcessor
from mmwave_radar_processing.processors.micro_doppler_resp import MicroDopplerProcessor
from mmwave_radar_processing.processors.range_resp import RangeProcessor
from mmwave_radar_processing.processors.altimeter import Altimeter
from mmwave_radar_processing.processors.velocity_estimator import VelocityEstimator


class PlotterMmWaveData:

    def __init__(self,config_manager:ConfigManager) -> None:
        
        #define default plot parameters:
        self.font_size_axis_labels = 12
        self.font_size_title = 15
        self.font_size_ticks = 12
        self.font_size_legend = 12
        self.plot_x_max = 10
        self.plot_y_max = 20
        self.marker_size = 10
        self.min_threshold_dB = 40

        #configuration manager
        self.config_manager:ConfigManager = config_manager
    
    ####################################################################
    #Range Azimuth Response
    #################################################################### 

    def plot_range_az_resp_cart(
        self,
        resp:np.ndarray,
        range_azimuth_processor:RangeAngleProcessor,
        convert_to_dB=False,
        cmap="viridis",
        ax:plt.Axes=None,
        show=False
        ):
        """Plot the range azimuth response in cartesian

        Args:
            resp (np.ndarray): range_bins x angle_bins np.ndarray of the already computed
                range azimuth response
            range_azimuth_processor (RangeAzimuthProcessor): RangeAzimuthProcessor object
                used to generate the response
            convert_to_dB (bool, optional): on True, converts the response to a 
                log scale. Defaults to False.
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()
        
        if convert_to_dB:
            resp = 20 * np.log10(resp)
            #remove anything below the min_threshold dB down
            thresholded_val = np.max(resp) - self.min_threshold_dB
            idxs = resp <= thresholded_val
            resp[idxs] = thresholded_val
        
        #rotate the x and y grids so that the x axis is on the y axis
        x_grid = -1 * range_azimuth_processor.y_s
        y_grid = range_azimuth_processor.x_s

        ax.pcolormesh(
            x_grid,
            y_grid,
            resp,
            cmap=cmap,
            shading='gouraud'
        )

        ax.set_xlabel("Y (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("X (m)",fontsize=self.font_size_axis_labels)
        if convert_to_dB:
            ax.set_title("Range-Azimuth\nHeatmap (dB Cart.)",fontsize=self.font_size_title)
        else:
            ax.set_title("Range-Azimuth\nHeatmap (mag Cart.)",fontsize=self.font_size_title)
        
        ax.tick_params(labelsize=self.font_size_ticks)

        if show:
            plt.show()

    
    def plot_range_az_resp_polar(
        self,
        resp:np.ndarray,
        range_azimuth_processor:RangeAngleProcessor,
        convert_to_dB=False,
        cmap="viridis",
        ax:plt.Axes=None,
        show=False
        ):
        """Plot the range azimuth response in polar

        Args:
            resp (np.ndarray): range_bins x angle_bins np.ndarray of the already computed
                range azimuth response
            range_azimuth_processor (RangeAzimuthProcessor): RangeAzimuthProcessor object
                used to generate the response
            convert_to_dB (bool, optional): on True, converts the response to a 
                log scale. Defaults to False.
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()
        
        if convert_to_dB:
            resp = 20 * np.log10(resp)
            #remove anything below the min_threshold dB down
            thresholded_val = np.max(resp) - self.min_threshold_dB
            idxs = resp <= thresholded_val
            resp[idxs] = thresholded_val
        
        ax.imshow(
            np.flip(resp,axis=0),
            extent=[
                range_azimuth_processor.angle_bins[0],
                range_azimuth_processor.angle_bins[-1],
                range_azimuth_processor.range_bins[0],
                range_azimuth_processor.range_bins[-1]
            ],
            cmap=cmap,
            aspect='auto'
            )
        
        ax.set_xlabel("Angle (radians)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("Range (m)",fontsize=self.font_size_axis_labels)
        ax.set_title("Range-Azimuth\nHeatmap (Polar.)",fontsize=self.font_size_title)
        if convert_to_dB:
            ax.set_title("Range-Azimuth\nHeatmap (dB Polar.)",fontsize=self.font_size_title)
        else:
            ax.set_title("Range-Azimuth\nHeatmap (mag Polar.)",fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)

        if show:
            plt.show()
    
    ####################################################################
    #Range Doppler Response
    ####################################################################
    def plot_range_doppler_resp(
        self,
        resp:np.ndarray,
        range_doppler_processor:RangeDopplerProcessor,
        convert_to_dB=False,
        cmap="viridis",
        ax:plt.Axes=None,
        show=False
        ):
        """Plot the range doppler response

        Args:
            resp (np.ndarray): range_bins x velocity bins np.ndarray of the already computed
                range doppler response
            range_doppler_processor (RangeDopplerProcessor): RangeDopplerProcessor object
                used to generate the response
            convert_to_dB (bool, optional): on True, converts the response to a 
                log scale. Defaults to False.
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()
        
        if convert_to_dB:
            resp = 20 * np.log10(resp)
            #remove anything below the min_threshold dB down
            thresholded_val = np.max(resp) - self.min_threshold_dB
            idxs = resp <= thresholded_val
            resp[idxs] = thresholded_val
        
        im = ax.imshow(
            np.flip(resp,axis=0),
            extent=[
                range_doppler_processor.vel_bins[0],
                range_doppler_processor.vel_bins[-1],
                range_doppler_processor.range_bins[0],
                range_doppler_processor.range_bins[-1]
            ],
            cmap=cmap,
            aspect='auto',
            interpolation='bilinear'
            )
        ax.set_xlabel("Velocity (m/s)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("Range (m)",fontsize=self.font_size_axis_labels)
        if convert_to_dB:
            ax.set_title("Range-Doppler\nHeatmap (dB)",fontsize=self.font_size_title)
        else:
            ax.set_title("Range-Doppler\nHeatmap (mag)",fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)

        #add the colorbar
        # cbar = plt.colorbar(im, ax=ax)
        # cbar.set_label('Intensity', fontsize=12)

        if show:
            plt.show()

    
    ####################################################################
    #MicroDoppler Response
    ####################################################################
    def plot_micro_doppler_resp(
        self,
        resp:np.ndarray,
        micro_doppler_processor:MicroDopplerProcessor,
        convert_to_dB=False,
        cmap="viridis",
        ax:plt.Axes=None,
        show=False
        ):
        """Plot the range doppler response

        Args:
            resp (np.ndarray): range_bins x velocity bins np.ndarray of the already computed
                range doppler response
            range_doppler_processor (RangeDopplerProcessor): RangeDopplerProcessor object
                used to generate the response
            convert_to_dB (bool, optional): on True, converts the response to a 
                log scale. Defaults to False.
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()
        
        if convert_to_dB:
            resp = 20 * np.log10(resp)
            #remove anything below the min_threshold dB down
            thresholded_val = np.max(resp) - self.min_threshold_dB
            idxs = resp <= thresholded_val
            resp[idxs] = thresholded_val
        
        im = ax.imshow(
            resp,#np.flip(resp,axis=0),
            extent=[
                micro_doppler_processor.time_bins[0],
                micro_doppler_processor.time_bins[-1],
                micro_doppler_processor.vel_bins[0],
                micro_doppler_processor.vel_bins[-1],
            ],
            cmap=cmap,
            aspect='auto',
            interpolation='bilinear'
            )
        ax.set_xlabel("Time History (s)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("Velocity (m/s)",fontsize=self.font_size_axis_labels)
        if convert_to_dB:
            ax.set_title("Micro-Doppler\nHeatmap (dB)",fontsize=self.font_size_title)
        else:
            ax.set_title("Micro-Doppler\nHeatmap (mag)",fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)

        #add the colorbar
        # cbar = plt.colorbar(im, ax=ax)
        # cbar.set_label('Intensity', fontsize=12)

        if show:
            plt.show()
    
    ####################################################################
    #Doppler Azimuth Response
    ####################################################################
    def plot_doppler_az_resp(
        self,
        resp: np.ndarray,
        doppler_azimuth_processor: DopplerAzimuthProcessor,
        peaks:np.ndarray = np.empty(shape=(0,2)),
        vd_ground_truth:np.ndarray = np.empty(shape=(0)),
        vd_estimated:np.ndarray = np.empty(shape=(0)),
        convert_to_dB=False,
        cmap="viridis",
        ax: plt.Axes = None,
        show=False
    ):
        """Plot the Doppler-Azimuth response.

        Args:
            resp (np.ndarray): Doppler-azimuth response magnitude.
            doppler_azimuth_processor (DopplerAzimuthProcessor): Processor used to generate the response.
            peaks (np.ndarray, optional): Detected peaks in the response. Defaults to np.empty(shape=(0,2)).
            vd_ground_truth (np.ndarray, optional): Ground truth velocity measurements at each valid angle based on the given ground truth velocity. Defaults to np.empty(shape=(0)).
            vd_estimated (np.ndarray, optional): Estimated velocity measurements at each valid angle based on a LSQ-estimated ego velocity. Defaults to np.empty(shape=(0)).
            convert_to_dB (bool, optional): Whether to convert the response to dB. Defaults to False.
            cmap (str, optional): Colormap for the plot. Defaults to "viridis".
            ax (plt.Axes, optional): Axes to plot on. Defaults to None.
            show (bool, optional): Whether to display the plot. Defaults to False.
        """

        if ax is None:
            fig, ax = plt.subplots()

        if convert_to_dB:
            resp = 20 * np.log10(resp)
            # Thresholding to remove very low values
            thresholded_val = np.max(resp) - self.min_threshold_dB
            resp = np.maximum(resp, thresholded_val)

        # Get velocity and angle bins
        vel_bins = doppler_azimuth_processor.vel_bins
        angle_bins = doppler_azimuth_processor.valid_angle_bins

        # Make 2D meshgrid
        angle_grid, vel_grid = np.meshgrid(angle_bins, vel_bins)

        # Flip response vertically to match orientation
        # resp_flipped = np.flip(resp, axis=0)

        # Plot using pcolormesh
        mesh = ax.pcolormesh(
            angle_grid,
            vel_grid,
            resp,
            shading='nearest',
            cmap=cmap
        )

        if peaks.shape[0] > 0:
            ax.scatter(
                peaks[:, 0],
                peaks[:, 1],
                marker='x',
                color='red',
                label='Detected Peaks'
            )
            ax.legend()

        if vd_ground_truth.shape[0] > 0:
            ax.plot(
                doppler_azimuth_processor.valid_angle_bins,
                vd_ground_truth,
                color='orange',
                linewidth=2,
                label='Ground Truth Velocity'
            )
            ax.legend()
        
        if vd_estimated.shape[0] > 0:
            ax.plot(
                doppler_azimuth_processor.valid_angle_bins,
                vd_estimated,
                color='white',
                linewidth=2,
                label='Estimated Velocity'
            )
            ax.legend()

        ax.set_ylabel("Velocity (m/s)", fontsize=self.font_size_axis_labels)
        ax.set_xlabel("Angle (radians)", fontsize=self.font_size_axis_labels)
        ax.set_title(
            "Doppler-Azimuth\nHeatmap (dB)" if convert_to_dB else "Doppler-Azimuth\nHeatmap (mag)",
            fontsize=self.font_size_title
        )
        ax.tick_params(labelsize=self.font_size_ticks)
        ax.set_ylim(
            bottom=np.min(doppler_azimuth_processor.vel_bins),
            top=np.max(doppler_azimuth_processor.vel_bins)
        )

        # Optional colorbar
        # cbar = plt.colorbar(mesh, ax=ax)
        # cbar.set_label('Intensity', fontsize=12)

        if show:
            plt.show()

    def plot_zoomed_doppler_az_resp(
        self,
        resp: np.ndarray,
        doppler_azimuth_processor: DopplerAzimuthProcessor,
        peaks:np.ndarray = np.empty(shape=(0,2)),
        vd_ground_truth:np.ndarray = np.empty(shape=(0)),
        vd_estimated:np.ndarray = np.empty(shape=(0)),
        convert_to_dB=False,
        cmap="viridis",
        ax: plt.Axes = None,
        show=False
    ):
        """Plot the zoomed Doppler-Azimuth response.

        Args:
            resp (np.ndarray): Zoomed Doppler-azimuth response magnitude.
            doppler_azimuth_processor (DopplerAzimuthProcessor): Processor used to generate the response.
            peaks (np.ndarray, optional): Detected peaks in the response. Defaults to np.empty(shape=(0,2)).
            vd_ground_truth (np.ndarray, optional): Ground truth velocity measurements at each valid angle based on the given ground truth velocity. Defaults to np.empty(shape=(0)).
            vd_estimated (np.ndarray, optional): Estimated velocity measurements at each valid angle based on a LSQ-estimated ego velocity. Defaults to np.empty(shape
            convert_to_dB (bool, optional): Whether to convert the response to dB. Defaults to False.
            cmap (str, optional): Colormap for the plot. Defaults to "viridis".
            ax (plt.Axes, optional): Axes to plot on. Defaults to None.
            show (bool, optional): Whether to display the plot. Defaults to False.
        """

        if ax is None:
            fig, ax = plt.subplots()

        if convert_to_dB:
            resp = 20 * np.log10(resp)
            # Thresholding to remove very low values
            thresholded_val = np.max(resp) - self.min_threshold_dB
            resp = np.maximum(resp, thresholded_val)

        # Get velocity and angle bins
        angle_bins = doppler_azimuth_processor.valid_angle_bins

        # Make 2D meshgrid
        angle_grid, vel_grid = np.meshgrid(angle_bins, doppler_azimuth_processor.zoomed_vel_bins)

        # Flip response vertically to match orientation
        # resp_flipped = np.flip(resp, axis=0)

        # Plot using pcolormesh
        mesh = ax.pcolormesh(
            angle_grid,
            vel_grid,
            resp,
            shading='nearest',
            cmap=cmap
        )

        if peaks.shape[0] > 0:
            ax.scatter(
                peaks[:, 0],
                peaks[:, 1],
                marker='x',
                color='red',
                label='Detected Peaks'
            )
            ax.legend()
        
        if vd_ground_truth.shape[0] > 0:
            ax.plot(
                doppler_azimuth_processor.valid_angle_bins,
                vd_ground_truth,
                color='orange',
                linewidth=2,
                label='Ground Truth Velocity'
            )
            ax.legend()
        
        if vd_estimated.shape[0] > 0:
            ax.plot(
                doppler_azimuth_processor.valid_angle_bins,
                vd_estimated,
                color='white',
                linewidth=2,
                label='Estimated Velocity'
            )
            ax.legend()

        ax.set_ylabel("Velocity (m/s)", fontsize=self.font_size_axis_labels)
        ax.set_xlabel("Angle (radians)", fontsize=self.font_size_axis_labels)
        ax.set_title(
            "Zoomed Doppler-Azimuth\nHeatmap (dB)" if convert_to_dB \
                else "Zoomed Doppler-Azimuth\nHeatmap (mag)",
            fontsize=self.font_size_title
        )
        ax.tick_params(labelsize=self.font_size_ticks)
        ax.set_ylim(
            bottom=np.min(doppler_azimuth_processor.zoomed_vel_bins),
            top=np.max(doppler_azimuth_processor.zoomed_vel_bins)
        )

        # Optional colorbar
        # cbar = plt.colorbar(mesh, ax=ax)
        # cbar.set_label('Intensity', fontsize=12)

        if show:
            plt.show()

    ####################################################################
    #Altitude / Range FFT Response
    ####################################################################
    def plot_range_resp(
        self,
        resp:np.ndarray,
        rng_bins:np.ndarray,
        peak_rng_bins:np.ndarray = np.empty(shape=0),
        peak_vals:np.ndarray = np.empty(shape=0),
        convert_to_dB=False,
        ax:plt.Axes=None,
        show=False
        ):
        """Plot the altitude/range FFT response.

        Args:
            resp (np.ndarray): 1D array of the computed range/altitude FFT response.
            range_processor (RangeProcessor): RangeProcessor object used to generate the response.
            convert_to_dB (bool, optional): If True, converts the response to a log scale. Defaults to False.
            ax (plt.Axes, optional): The axes on which to display the plot. If None, a figure is automatically generated. Defaults to None.
            show (bool, optional): If True, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()
        
        if convert_to_dB:
            resp = 20 * np.log10(resp)
        
        ax.plot(rng_bins, resp, color='blue', linewidth=2, label='Range FFT Resp')

        if peak_rng_bins.shape[0] > 0:
            ax.plot(peak_rng_bins, peak_vals, 'ro', markersize=self.marker_size, label='Detected Peaks')
        ax.set_xlabel("Range (m)",fontsize=self.font_size_axis_labels)

        if convert_to_dB:
            ax.set_ylabel("Amplitude (dB)",fontsize=self.font_size_axis_labels)
            ax.set_title("Range FFT (dB)",fontsize=self.font_size_title)
        else:
            ax.set_ylabel("Amplitude (mag)",fontsize=self.font_size_axis_labels)
            ax.set_title("Range FFT (mag)",fontsize=self.font_size_title)
        
        ax.tick_params(labelsize=self.font_size_ticks)
        ax.grid(True)
        ax.legend(fontsize=self.font_size_legend, loc='lower right')

        if show:
            plt.show()

    ####################################################################
    #Errors and Ground Truth Comparison
    ####################################################################
    def plot_estimated_vs_ground_truth(
        self,
        estimated: np.ndarray,
        ground_truth: np.ndarray,
        ax: plt.Axes = None,
        value_label="Altitude (m)",
        frame_rate=-1,
        show=False
    ):
        """Plot the estimated value against the ground truth value.

        Args:
            estimated (np.ndarray): 1D array of estimated values.
            ground_truth (np.ndarray): 1D array of ground truth values.
            ax (plt.Axes, optional): The axes on which to display the plot. If None, a figure is automatically generated. Defaults to None.
            value_label (str, optional): Label for the y-axis. Defaults to "Altitude (m)".
            frame_rate (float, optional): Frame rate in Hz. If provided, x-axis will be in time instead of frame index. Defaults to -1.
            show (bool, optional): If True, shows the plot. Defaults to False.
        """

        if not ax:
            fig, ax = plt.subplots()

        # Check to make sure that both arrays have data
        if estimated.size == 0 and ground_truth.size == 0:
            return

        # Determine x-axis values
        if frame_rate > 0:
            x_values = np.arange(len(estimated)) / frame_rate if estimated.size > 0 else np.arange(len(ground_truth)) / frame_rate
            x_label = "Time (s)"
        else:
            x_values = np.arange(len(estimated)) if estimated.size > 0 else np.arange(len(ground_truth))
            x_label = "Frame index"

        # Plot estimated and ground truth values
        if estimated.size > 0:
            ax.plot(x_values[:len(estimated)], estimated, color='blue', linewidth=2, label='Estimated Value')
        if ground_truth.size > 0:
            ax.plot(x_values[:len(ground_truth)], ground_truth, color='orange', linewidth=2, label='Ground Truth Value')

        ax.set_xlabel(x_label, fontsize=self.font_size_axis_labels)
        ax.set_ylabel(value_label, fontsize=self.font_size_axis_labels)
        ax.set_title(f"Estimated {value_label}\nvs Ground Truth", fontsize=self.font_size_title)

        ax.tick_params(labelsize=self.font_size_ticks)
        ax.grid(True)
        ax.legend(fontsize=self.font_size_legend, loc='lower left')

        if show:
            plt.show()
        
    def plot_estimated_vs_ground_truth_error(
        self,
        estimated: np.ndarray,
        ground_truth: np.ndarray,
        ax: plt.Axes = None,
        value_label="Altitude (m)",
        frame_rate=-1,
        show=False
    ):
        """Plot the error between the estimated value and the ground truth value.

        Args:
            estimated (np.ndarray): 1D array of estimated values.
            ground_truth (np.ndarray): 1D array of ground truth values.
            ax (plt.Axes, optional): The axes on which to display the plot. If None, a figure is automatically generated. Defaults to None.
            value_label (str, optional): Label for the y-axis. Defaults to "Altitude (m)".
            frame_rate (float, optional): Frame rate in Hz. If provided, x-axis will be in time instead of frame index. Defaults to -1.
            show (bool, optional): If True, shows the plot. Defaults to False.
        """

        if not ax:
            fig, ax = plt.subplots()
    
        # Check to make sure that both arrays have data
        if estimated.size == 0 or ground_truth.size == 0:
            return
        # Check to make sure arrays are the same length
        if estimated.shape[0] != ground_truth.shape[0]:
            raise ValueError("Estimated and ground truth arrays must have the same length.")

        # Calculate error
        error = estimated - ground_truth

        # Determine x-axis values
        if frame_rate > 0:
            x_values = np.arange(len(error)) / frame_rate
            x_label = "Time (s)"
        else:
            x_values = np.arange(len(error))
            x_label = "Frame index"

        # Plot the error
        ax.plot(x_values, error, color='blue', linewidth=2, label='Errors')

        ax.set_xlabel(x_label, fontsize=self.font_size_axis_labels)
        ax.set_ylabel(f"{value_label} error", fontsize=self.font_size_axis_labels)
        ax.set_title(f"Estimated {value_label} Error\nvs Ground Truth", fontsize=self.font_size_title)
        
        ax.tick_params(labelsize=self.font_size_ticks)
        ax.grid(True)
        ax.legend(fontsize=self.font_size_legend, loc='upper right')

        if show:
            plt.show()

    ####################################################################
    #ADC Data
    ####################################################################
    def plot_adc_data(
        self,
        adc_cube:np.ndarray,
        rx_antenna_idx:int=0,
        chirp_idx:int=0,
        ax:plt.Axes=None,
        show=False
        ):
        """Plot the range doppler response

        Args:
            adc_cube (np.ndarray): rx antennas x adc samples x chirps np.ndarray of the already computed
                range doppler response
            rx_antenna_idx(int, optional): the index of the rx antenna to display data from.
                Defaults fo 0
            chirp_idx(int, optional): the index of the chirp to display data from.
                Defaults fo 0
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()
        
        
        ax.plot(adc_cube[rx_antenna_idx,:,chirp_idx].real,
                label="real")
        ax.plot(adc_cube[rx_antenna_idx,:,chirp_idx].imag,
                label="imag")
        
        ax.set_xlabel("Value",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("Sample",fontsize=self.font_size_axis_labels)
        ax.set_title("ADC Samples (chirp: {}, rx: {})".format(
            chirp_idx,rx_antenna_idx
            ),fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)
        handles,labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="lower right",fontsize=self.font_size_legend)
        if show:
            plt.show()
    
    ####################################################################
    #Plotting Compilation
    ####################################################################
    def plot_compilation(
            self,
            adc_cube:np.ndarray,
            range_doppler_processor:RangeDopplerProcessor=None,
            range_azimuth_processor:RangeAngleProcessor=None,
            doppler_azimuth_processor:DopplerAzimuthProcessor=None,
            micro_doppler_processor:MicroDopplerProcessor=None,
            camera_view:np.ndarray=np.empty(shape=(0)),
            convert_to_dB=False,
            cmap="viridis",
            chirp_idx:int=0,
            rx_antenna_idx:int=0,
            axs:plt.Axes=[],
            show=False
    ):
        if len(axs) == 0:
            fig,axs=plt.subplots(2,3, figsize=(15,10))
            fig.subplots_adjust(wspace=0.3,hspace=0.30)
        
        #plot adc samples
        self.plot_adc_data(
            adc_cube=adc_cube,
            rx_antenna_idx=rx_antenna_idx,
            chirp_idx=chirp_idx,
            ax=axs[0,0],
            show=False
        )

        #plot range doppler plot
        if range_doppler_processor:
            resp = range_doppler_processor.process(
                adc_cube=adc_cube,
                rx_idx=rx_antenna_idx)
            self.plot_range_doppler_resp(
                resp=resp,
                range_doppler_processor=range_doppler_processor,
                convert_to_dB=convert_to_dB,
                cmap=cmap,
                ax=axs[0,1],
                show=False
            )


        #determine the array goemetry
        if self.config_manager.array_geometry == "standard":
            if self.config_manager.virtual_antennas_enabled:
                rx_antennas = np.array(range(8))
            else:
                rx_antennas = np.array(range(4))
        elif self.config_manager.array_geometry == "ods":
            if self.config_manager.virtual_antennas_enabled:
                rx_antennas = np.array([10,11,6,7]) #only do the elevation
            else:
                rx_antennas = np.array([0,1])
        else:
            rx_antennas = np.array([]) #no rx antennas specified

        #plot the range azimuth plots
        if range_azimuth_processor:
            resp = range_azimuth_processor.process(
                adc_cube=adc_cube,
                chirp_idx=chirp_idx,
                rx_antennas=rx_antennas
            )
            self.plot_range_az_resp_cart(
                resp=resp,
                range_azimuth_processor=range_azimuth_processor,
                convert_to_dB=convert_to_dB,
                cmap=cmap,
                ax=axs[1,0],
                show=False
            )
        # self.plot_range_az_resp_polar(
        #     resp=resp,
        #     range_azimuth_processor=range_azimuth_processor,
        #     convert_to_dB=convert_to_dB,
        #     cmap=cmap,
        #     ax=axs[1,1],
        #     show=False
        # )

        #plot micro-doppler signature
        if micro_doppler_processor:
            resp = micro_doppler_processor.process(
                adc_cube=adc_cube,
                rx_idx=rx_antenna_idx
            )
            self.plot_micro_doppler_resp(
                resp,
                micro_doppler_processor=micro_doppler_processor,
                convert_to_dB=convert_to_dB,
                cmap=cmap,
                ax=axs[1,1],
                show=False
            )

        #plot the doppler-azimuth response
        if doppler_azimuth_processor:
            resp = doppler_azimuth_processor.process(
                adc_cube=adc_cube,
                rx_antennas=rx_antennas
            )
            self.plot_doppler_az_resp(
                resp=resp,
                doppler_azimuth_processor=doppler_azimuth_processor,
                convert_to_dB=convert_to_dB,
                cmap=cmap,
                ax=axs[0,2],
                show=False
            )
        #camera view
        if camera_view.shape[0] > 0:
            axs[1,2].imshow(camera_view)
            axs[1,2].set_title("Frontal Camera View",fontsize=self.font_size_title)
        

        if show:
            plt.show()
        


