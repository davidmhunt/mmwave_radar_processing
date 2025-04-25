import sys
import os
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
load_dotenv()
DATASET_PATH=os.getenv("DATASET_DIRECTORY")
CONFIG_DIRECTORY = os.getenv("CONFIG_DIRECTORY")

sys.path.append("../")
from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from cpsl_datasets.cpsl_ds import CpslDS
from mmwave_radar_processing.processors.strip_map_SAR_processor import StripMapSARProcessor
from mmwave_radar_processing.processors.synthetic_array_broadside_beamformer_processor import SyntheticArrayBeamformerProcessor
from mmwave_radar_processing.plotting.plotter_synthetic_array_data import PlotterSyntheticArrayData
from mmwave_radar_processing.plotting.movies_synthetic_array_data import MovieGeneratorSyntheticArrayData
from mmwave_radar_processing.detectors.CFAR import CaCFAR_1D,CaCFAR_2D

cfg_manager = ConfigManager()

cfg_path = os.path.join(CONFIG_DIRECTORY,"RadSAR_2_sr.cfg")
cfg_manager.load_cfg(cfg_path)
cfg_manager.compute_radar_perforance(profile_idx=0)

#load the dataset
dataset_name = "RadSAR_2_WILK"
dataset_path = os.path.join(DATASET_PATH,os.pardir,"RadSAR",dataset_name)
dataset = CpslDS(
    dataset_path=dataset_path,
    radar_folder="radar_0",
    lidar_folder="lidar",
    camera_folder="camera",
    imu_orientation_folder="imu_data",
    imu_full_folder="imu_data_full"
)

#initialize the key processing components
cfar_2d = CaCFAR_2D(
    num_guard_cells=np.array([2,5]),#(2,5)*,(3,5)
    num_training_cells=np.array([3,15]), #(3,15)*,(5,15)
    false_alarm_rate=1e-3, #1e-2,1e-3*,1e-4
    resp_border_cells=np.array([5,5]),
    mode="full"
)

processor_sabf = SyntheticArrayBeamformerProcessor(
    config_manager=cfg_manager,
    cfar=cfar_2d,
    az_angle_bins_rad=\
                np.deg2rad(np.linspace(
                    start=-35,stop=35,num=80 #(-35,35)*,(-25,25)
                 )),
    el_angle_bins_rad=\
                np.deg2rad(np.linspace(
                    start=-60,
                    stop=60,
                    num=16
                ))
)

processor_stripMapSAR = StripMapSARProcessor(
    config_manager=cfg_manager
)

synthetic_array_plotter = PlotterSyntheticArrayData(
    config_manager=cfg_manager,
    processor_SABF=processor_sabf,
    processor_stripMapSAR=processor_stripMapSAR,
    min_vel=0.4
)

movie_generator = MovieGeneratorSyntheticArrayData(
    cpsl_dataset=dataset,
    plotter=synthetic_array_plotter,
    processor=processor_sabf,
    temp_dir_path="~/Downloads/syntheticArrayScripts/{}_sar_dB".format(dataset_name)
)

#generate the movie#generate the movie
movie_generator.initialize_figure(nrows=3,ncols=3,figsize=(15,15))

movie_generator.generate_movie_frames(
    cmap="viridis",
    convert_to_dB=True,
    lidar_radar_offset_rad=np.deg2rad(180)
)

fps = 1 / (1e-3 * cfg_manager.frameCfg_periodicity_ms)
movie_generator.save_movie(video_file_name="{}_sar_dB.mp4".format(dataset_name),fps=fps)