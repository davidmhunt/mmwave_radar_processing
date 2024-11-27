import matplotlib.pyplot as plt
import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.synthetic_array_processor import SyntheticArrayProcessor

class PlotterSyntheticArrayData:

    def __init__(self,
                 config_manager:ConfigManager,
                 synthetic_array_processor:SyntheticArrayProcessor) -> None:
        
        #define default plot parameters:
        self.font_size_axis_labels = 12
        self.font_size_title = 15
        self.font_size_ticks = 12
        self.font_size_legend = 12
        self.plot_x_max = 10
        self.plot_y_max = 20
        self.marker_size = 10

        #configuration manager
        self.config_manager:ConfigManager = config_manager

        #synthetic array processor
        self.synthetic_array_processor:SyntheticArrayProcessor =\
            synthetic_array_processor
    
    def plot_synthetic_array_geometry(
        self,
        vels,
        ax:plt.Axes=None,
        show=False
    ):

        #compute the geometries
        self.synthetic_array_processor._generate_array_geometries(vels)

        if not ax:
            fig,ax = plt.subplots()

        x_coords = np.reshape(
            self.synthetic_array_processor.p_x_m,
            newshape=(-1,1),
            order='F'
        )
        z_coords = np.reshape(
            self.synthetic_array_processor.p_z_m,
            newshape=(-1,1),
            order='F'
        )

        ax.scatter(x_coords,z_coords)
        ax.set_xlabel("X (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("Z (m)",fontsize=self.font_size_axis_labels)

        title_str = "Synth Array: x vel ({:.2} m/s), {} chirps".format(
            vels[0],
            self.synthetic_array_processor.chirps_per_frame
        )
        ax.set_title(title_str,
                    fontsize=self.font_size_title)
        if vels[0] > 0:
            ax.set_xlim(
                left=-1 * self.synthetic_array_processor.lambda_m/8,
                right=self.synthetic_array_processor.p_x_m[0,5] + \
                    self.synthetic_array_processor.lambda_m/8
            )
        else:
            ax.set_xlim(
                right=1 * self.synthetic_array_processor.lambda_m/8,
                left=self.synthetic_array_processor.p_x_m[0,5] - \
                    self.synthetic_array_processor.lambda_m/8
            )
        ax.set_ylim(
            bottom=-1 * self.synthetic_array_processor.lambda_m/8,
            top = self.synthetic_array_processor.p_z_m[-1,1] + \
                self.synthetic_array_processor.lambda_m/8
        )

        if show:
            plt.show()