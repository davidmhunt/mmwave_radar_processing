from cpsl_datasets.cpsl_ds import CpslDS
from mmwave_radar_processing.plotting.movie_generator import MovieGenerator
from mmwave_radar_processing.plotting.plotter_synthetic_array_data_simple_multiframe import PlotterSyntheticArrayData
from mmwave_radar_processing.processors.simple_synthetic_array_beamformer_processor_multiFrame import SyntheticArrayBeamformerProcessor
import numpy as np
import tqdm


class MovieGeneratorSyntheticArrayData(MovieGenerator):

    def __init__(
            self,
            cpsl_dataset: CpslDS,
            plotter:PlotterSyntheticArrayData,
            processor:SyntheticArrayBeamformerProcessor,
            temp_dir_path="~/Downloads/odometry_temp") -> None:

        self.plotter:PlotterSyntheticArrayData = plotter
        self.processor = processor

        super().__init__(cpsl_dataset, temp_dir_path)
    
    def generate_movie_frame(self, idx, **kwargs):

        #get the velocity of the vehicle
        try: #try accessing the full odometry data
            vel_data = np.mean(self.dataset.get_vehicle_odom_data(idx)[:,8:11],axis=0)
        except AssertionError: #if not just get the x velocity (forward)
            vel = np.mean(self.dataset.get_vehicle_vel_data(idx)[:,1])
            vel_data = np.array([vel,0,0])

        #get the adc cube
        adc_cube = self.dataset.get_radar_data(idx)

        #get the lidar pc
        try:
            lidar_pc_raw = self.dataset.get_lidar_point_cloud_raw(idx)
        except AssertionError:
            lidar_pc_raw = np.empty(shape=(0))

        #get the camera data
        try:
            camera_view = self.dataset.get_camera_frame(idx)
        except AssertionError:
            camera_view = np.empty(shape=(0))

        self.plotter.plot_compilation(
            adc_cube=adc_cube,
            vels=vel_data,
            camera_view=camera_view,
            lidar_pc_raw=lidar_pc_raw,
            lidar_radar_offset_rad=np.deg2rad(90),
            convert_to_dB=True,
            cmap="viridis",
            axs=self.axs,
            show=False
        )

        return

    def generate_movie_frames(
            self,
            **kwargs):
        """Generates the movie frames. May need to be modified by child class
        """

        for i in tqdm.tqdm(range(self.dataset.num_frames)):

            #update the current frame
            self.generate_movie_frame(idx=i,**kwargs)

            #save the frame
            if self.processor.array_geometry_valid:
                self.save_frame(clear_axs=True)