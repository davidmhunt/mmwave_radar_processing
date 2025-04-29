import matplotlib.pyplot as plt
import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.range_azmith_resp import RangeAzimuthProcessor
from mmwave_radar_processing.processors.range_doppler_resp import RangeDopplerProcessor
from mmwave_radar_processing.processors.doppler_azimuth_resp import DopplerAzimuthProcessor


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
        range_azimuth_processor:RangeAzimuthProcessor,
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
        range_azimuth_processor:RangeAzimuthProcessor,
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
    #Doppler Azimuth Response
    ####################################################################
    def plot_doppler_az_resp(
        self,
        resp:np.ndarray,
        doppler_azimuth_processor:DopplerAzimuthProcessor,
        convert_to_dB=False,
        cmap="viridis",
        ax:plt.Axes=None,
        show=False
        ):
        """Plot the range doppler response

        Args:
            resp (np.ndarray): range_bins x velocity bins np.ndarray of the already computed
                range doppler response
            doppler_azimuth_processor (DopplerAzimuthProcessor): RangeDopplerProcessor object
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
                doppler_azimuth_processor.angle_bins[0],
                doppler_azimuth_processor.angle_bins[-1],
                doppler_azimuth_processor.vel_bins[0],
                doppler_azimuth_processor.vel_bins[-1],
            ],
            cmap=cmap,
            aspect='auto',
            interpolation='bilinear'
            )
        ax.set_ylabel("Velocity (m/s)",fontsize=self.font_size_axis_labels)
        ax.set_xlabel("Angle (radians)",fontsize=self.font_size_axis_labels)
        if convert_to_dB:
            ax.set_title("Doppler-Aziuth\nHeatmap (dB)",fontsize=self.font_size_title)
        else:
            ax.set_title("Doppler-Azimuth\nHeatmap (mag)",fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)

        #add the colorbar
        # cbar = plt.colorbar(im, ax=ax)
        # cbar.set_label('Intensity', fontsize=12)

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
            range_doppler_processor:RangeDopplerProcessor,
            range_azimuth_processor:RangeAzimuthProcessor,
            doppler_azimuth_processor:DopplerAzimuthProcessor,
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

        #plot the range azimuth plots
        resp = range_azimuth_processor.process(
            adc_cube=adc_cube,
            chirp_idx=chirp_idx
        )
        self.plot_range_az_resp_cart(
            resp=resp,
            range_azimuth_processor=range_azimuth_processor,
            convert_to_dB=convert_to_dB,
            cmap=cmap,
            ax=axs[1,0],
            show=False
        )
        self.plot_range_az_resp_polar(
            resp=resp,
            range_azimuth_processor=range_azimuth_processor,
            convert_to_dB=convert_to_dB,
            cmap=cmap,
            ax=axs[1,1],
            show=False
        )

        #plot the doppler-azimuth response
        resp = doppler_azimuth_processor.process(
            adc_cube=adc_cube
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
        


