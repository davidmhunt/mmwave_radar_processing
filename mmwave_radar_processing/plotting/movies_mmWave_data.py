from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from cpsl_datasets.cpsl_ds import CpslDS
from mmwave_radar_processing.processors.range_azmith_resp import RangeAzimuthProcessor
from mmwave_radar_processing.processors.range_doppler_resp import RangeDopplerProcessor
from mmwave_radar_processing.processors.virtual_array_reformater import VirtualArrayReformatter
from mmwave_radar_processing.plotting.plotter_mmWave_data import PlotterMmWaveData
from mmwave_radar_processing.plotting.movie_generator import MovieGenerator

class MovieGeneratorMmWaveData(MovieGenerator):

    def __init__(self,
                 cpsl_dataset:CpslDS,
                 plotter:PlotterMmWaveData,
                 range_azimuth_processor:RangeAzimuthProcessor,
                 range_doppler_processor:RangeDopplerProcessor,
                 virtual_array_reformatter:VirtualArrayReformatter,
                 temp_dir_path="~/Downloads/odometry_temp",
                 ) -> None:
        
        self.plotter:PlotterMmWaveData = plotter
        self.range_azimuth_processor:RangeAzimuthProcessor = range_azimuth_processor
        self.range_doppler_processor:RangeDopplerProcessor = range_doppler_processor
        self.virtual_array_reformatter = virtual_array_reformatter
        
        super().__init__(
            cpsl_dataset=cpsl_dataset,
            temp_dir_path=temp_dir_path
        )

    
    def generate_movie_frame(
            self,
            idx,
            chirp_idx=0,
            rx_antenna_idx=0,
            cmap="viridis",
            convert_to_dB=False):
        """Custom function for generating a movie frame given for 
        mmwave radar data processing

        Args:
            idx (_type_): _description_
            chirp_idx (int, optional): _description_. Defaults to 0.
            rx_antenna_idx (int, optional): _description_. Defaults to 0.
            cmap (str, optional): _description_. Defaults to "viridis".
            convert_to_dB (bool, optional): _description_. Defaults to False.
        """

        #get the adc cube
        adc_cube = self.dataset.get_radar_data(idx)

        ##reformat it with virtual arrays
        adc_cube = self.virtual_array_reformatter.process(adc_cube)
        
        #generate the figure
        self.plotter.plot_compilation(
            adc_cube=adc_cube,
            range_doppler_processor=self.range_doppler_processor,
            range_azimuth_processor=self.range_azimuth_processor,
            convert_to_dB=convert_to_dB,
            cmap=cmap,
            chirp_idx=chirp_idx,
            rx_antenna_idx=rx_antenna_idx,
            axs=self.axs,
            show=False
        )