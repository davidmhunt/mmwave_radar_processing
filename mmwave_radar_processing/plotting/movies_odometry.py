from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from cpsl_datasets.cpsl_ds import CpslDS
from mmwave_radar_processing.processors.altimeter import Altimeter
from mmwave_radar_processing.processors.velocity_estimator import VelocityEstimator
from mmwave_radar_processing.processors.virtual_array_reformater import VirtualArrayReformatter
from mmwave_radar_processing.plotting.plotter_odometry_data import PlotterOdometryData
from mmwave_radar_processing.plotting.movie_generator import MovieGenerator
import numpy as np

class MovieGeneratorOdometry(MovieGenerator):

    def __init__(self,
                 cpsl_dataset:CpslDS,
                 plotter:PlotterOdometryData,
                 altimeter:Altimeter,
                 velocity_estimator:VelocityEstimator,
                 virtual_array_reformatter:VirtualArrayReformatter,
                 temp_dir_path="~/Downloads/odometry_temp",
                 ) -> None:
        
        self.plotter:PlotterOdometryData = plotter
        self.altimeter:Altimeter = altimeter
        self.velocity_estimator:VelocityEstimator = velocity_estimator
        self.virtual_array_reformatter = virtual_array_reformatter
        
        super().__init__(
            cpsl_dataset=cpsl_dataset,
            temp_dir_path=temp_dir_path
        )

    
    def generate_movie_frame(
            self,
            idx,
            cmap="viridis",
            convert_to_dB=True):
        """Custom function for generating a movie frame given for 
        mmwave radar data processing

        Args:
            idx (_type_): _description_
            cmap (str, optional): _description_. Defaults to "viridis".
            convert_to_dB (bool, optional): _description_. Defaults to False.
        """

        #get the adc cube
        adc_cube = self.dataset.get_radar_data(idx)

        ##reformat it with virtual arrays
        adc_cube = self.virtual_array_reformatter.process(adc_cube)

        #estimate the altitude
        est_altitude = self.altimeter.process(adc_cube=adc_cube, precise_est_enabled=True)
        
        #save the altitude history
        vehicle_odom = self.dataset.get_vehicle_odom_data(idx=idx)
        gt_altitude = np.average(vehicle_odom[:, 3])
        self.altimeter.update_history(
            estimated=np.array([est_altitude]),
            ground_truth=np.array([gt_altitude])
        )


        #estimate the velocity
        vel_est = self.velocity_estimator.process(
            adc_cube=adc_cube,
            altitude=est_altitude,
            enable_precise_responses=True
        )

            #save the gt velocity history
        vehicle_vel_x = np.average(vehicle_odom[:,8])
        vehicle_vel_y = np.average(vehicle_odom[:,9])
        vehicle_vel_z = np.average(vehicle_odom[:,10])
        self.velocity_estimator.update_history(
            ground_truth=np.array([vehicle_vel_x,vehicle_vel_y,vehicle_vel_z]),
            estimated=vel_est
        )
        
        try:
            camera_view = self.dataset.get_camera_frame(idx)
        except AssertionError:
            camera_view = np.empty(shape=(0))

        #generate the figure
        self.plotter.plot_compilation(
            adc_cube=adc_cube,
            altimeter=self.altimeter,
            velocity_estimator=self.velocity_estimator,
            camera_view=camera_view,
            convert_to_dB=convert_to_dB,
            cmap=cmap,
            axs=self.axs,
            show=False
        )