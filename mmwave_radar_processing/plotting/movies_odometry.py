from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from cpsl_datasets.cpsl_ds import CpslDS
from mmwave_radar_processing.processors import altimeter
from mmwave_radar_processing.processors.altimeter import Altimeter
from mmwave_radar_processing.processors.velocity_estimator import VelocityEstimator
from mmwave_radar_processing.processors.virtual_array_reformater import VirtualArrayReformatter
from mmwave_radar_processing.plotting.plotter_odometry_data import PlotterOdometryData
from mmwave_radar_processing.plotting.movie_generator import MovieGenerator
import numpy as np
import tqdm
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
            idx = 0,
            cmap="viridis",
            convert_to_dB=True):
        """Custom function for generating a movie frame given for 
        mmwave radar data processing

        Args:
            idx (_type_): _description_
            cmap (str, optional): _description_. Defaults to "viridis".
            convert_to_dB (bool, optional): _description_. Defaults to False.
        """

        try:
            camera_view = self.dataset.get_camera_frame(idx)
        except AssertionError:
            camera_view = np.empty(shape=(0))

        if self.altimeter and self.virtual_array_reformatter:
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

            #generate the figure
            self.plotter.plot_compilation(
                adc_cube=adc_cube,
                altimeter=self.altimeter,
                velocity_estimator=None,
                camera_view=camera_view,
                convert_to_dB=convert_to_dB,
                cmap=cmap,
                axs=self.axs,
                show=False
            )
        elif self.velocity_estimator:

            #get the radar points
            radar_pts = self.dataset.get_radar_data(idx)

            #get the ground truth odometry
            gt_vel = np.mean(self.dataset.get_vehicle_odom_data(idx)[:,8:11],axis=0)

            #get the velocity estimate:
            vel_est = self.velocity_estimator.process(
                points=radar_pts
            )

            #update the velocity estimator history
            self.velocity_estimator.update_history(
                estimated=vel_est,
                ground_truth=gt_vel
            )

            self.plotter.plot_compilation(
                adc_cube=np.empty(shape=(0)),
                altimeter=None,
                velocity_estimator=self.velocity_estimator,
                camera_view=camera_view,
                convert_to_dB=convert_to_dB,
                cmap=cmap,
                axs=self.axs,
                show=False
            )

        

    def generate_movie_frames(
            self,
            start_idx = 0,
            **kwargs):
        """Generates the movie frames. May need to be modified by child class
        """

        for i in range(start_idx):

            if self.altimeter and self.virtual_array_reformatter:
                adc_cube = self.dataset.get_radar_data(i)
                adc_cube = self.virtual_array_reformatter.process(adc_cube)

                #estimate the altitude
                est_altitude = self.altimeter.process(adc_cube=adc_cube, precise_est_enabled=True)

                #save the altitude history
                vehicle_odom = self.dataset.get_vehicle_odom_data(idx=i)
                gt_altitude = np.average(vehicle_odom[:, 3])

            elif self.velocity_estimator:

                #get the radar points
                radar_pts = self.dataset.get_radar_data(i)

                #get the ground truth odometry
                gt_vel = np.mean(self.dataset.get_vehicle_odom_data(idx=i)[:,8:11],axis=0)

                #get the velocity estimate:
                vel_est = self.velocity_estimator.process(
                    points=radar_pts
                )

        for i in tqdm.tqdm(range(self.dataset.num_frames - start_idx)):

            #update the current frame
            self.generate_movie_frame(idx=start_idx + i,**kwargs)

            #save the frame
            self.save_frame(clear_axs=True)