from cpsl_datasets.cpsl_ds import CpslDS
from mmwave_radar_processing.plotting.movie_generator import MovieGenerator
from mmwave_radar_processing.plotting.plotter_synthetic_array_data import PlotterSyntheticArrayData
from mmwave_radar_processing.processors.synthetic_array_beamformer_processor_revA import SyntheticArrayBeamformerProcessor
import numpy as np

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
        vel = np.mean(self.dataset.get_vehicle_vel_data(idx)[:,1])

        #generate the array geometry
        self.processor.generate_array_geometries(
            vels=np.array([-vel,0,0])
        )
        #get the adc cube
        adc_cube = self.dataset.get_radar_data(idx)

        #get the lidar pc
        lidar_pc_raw = self.dataset.get_lidar_point_cloud_raw(idx)

        #get the camera data
        try:
            camera_view = self.dataset.get_camera_frame(idx)
        except AssertionError:
            camera_view = np.empty(shape=(0))

        if self.processor.mode == SyntheticArrayBeamformerProcessor.BROADSIDE_MODE:
            vel = -vel

        self.plotter.plot_compilation(
            adc_cube=adc_cube,
            vels=np.array([vel,0,0]),
            camera_view=camera_view,
            lidar_pc_raw=lidar_pc_raw,
            show=False,
            axs=self.axs,
            **kwargs
        )

        return