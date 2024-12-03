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
from mmwave_radar_processing.processors.synthetic_array_processor import SyntheticArrayProcessor
from mmwave_radar_processing.plotting.plotter_synthetic_array_data import PlotterSyntheticArrayData
from mmwave_radar_processing.plotting.movies_synthetic_array_data import MovieGeneratorSyntheticArrayData

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

#initialize the array processor
processor = SyntheticArrayProcessor(
    config_manager=cfg_manager,
    az_angle_bins_rad=\
                np.deg2rad(np.linspace(
                    start=-30,stop=30,num=60
                 )),
    el_angle_bins_rad=\
                np.deg2rad(np.linspace(
                    start=-90,
                    stop=90,
                    num=50
                ))
)

#initialize the synthetic array plotter
synthetic_array_plotter = PlotterSyntheticArrayData(
    config_manager=cfg_manager,
    synthetic_array_processor=processor,
    min_vel=0.4
)

#initialize the movie generator
movie_generator = MovieGeneratorSyntheticArrayData(
    cpsl_dataset=dataset,
    plotter=synthetic_array_plotter,
    processor=processor,
    temp_dir_path="~/Downloads/syntheticArrayScripts/{}".format(dataset_name)
)

#generate the movie
movie_generator.initialize_figure(nrows=2,ncols=3,figsize=(15,10))

movie_generator.generate_movie_frames(
    cmap="viridis",
    convert_to_dB=False,
    lidar_radar_offset_rad=np.deg2rad(180)
)

fps = 1 / (1e-3 * cfg_manager.frameCfg_periodicity_ms)
movie_generator.save_movie(video_file_name="{}_mag.mp4".format(dataset_name),fps=fps)