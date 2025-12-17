import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import yaml

from dotenv import load_dotenv
load_dotenv()

sys.path.append("../")
from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from cpsl_datasets.cpsl_ds import CpslDS
from mmwave_radar_processing.processors.point_cloud_generator import PointCloudGenerator
from mmwave_radar_processing.processors.virtual_array_reformater import VirtualArrayReformatter
from mmwave_radar_processing.point_cloud_processing.vel_estimator import VelocityEstimator
from mmwave_radar_processing.plotting.plotter_odometry_data import PlotterOdometryData
from mmwave_radar_processing.analysis.velocity_analyzer import VelocityAnalyzer
from mmwave_radar_processing.plotting.analysis_plotter import AnalysisPlotter

# Load configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(script_dir, "../analyzer_configs/velocity_analysis_config.yaml")
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

DATASET_PATH = config['dataset']['path']
CONFIG_DIRECTORY = os.getenv("CONFIG_DIRECTORY")

cfg_manager = ConfigManager()
cfg_path = os.path.join(CONFIG_DIRECTORY, config['radar']['config_file'])
cfg_manager.load_cfg(cfg_path,
                     array_geometry=config['radar']['array_geometry'],
                     array_direction=config['radar']['array_direction'])
cfg_manager.compute_radar_perforance(profile_idx=0)
cfg_manager.print_cfg_overview()


dataset_name = config['dataset']['name']
dataset_path = os.path.join(DATASET_PATH, dataset_name)
print(f"Loading dataset from: {dataset_path}")

dataset = CpslDS(
    dataset_path=dataset_path,
    radar_adc_folder="down_radar_adc",
    lidar_folder="lidar",
    camera_folder="camera",
    hand_tracking_folder="hand_tracking",
    imu_orientation_folder="imu_orientation",
    imu_full_folder="imu_data",
    vehicle_vel_folder="vehicle_vel",
    vehicle_odom_folder="vehicle_odom"
)

plotter = PlotterOdometryData(config_manager=cfg_manager)

velocity_estimator = VelocityEstimator(
    config_manager=cfg_manager,
    min_R2_threshold=config['velocity_estimator']['min_r2_threshold'],
    min_inlier_percent=config['velocity_estimator']['min_inlier_percent']
)

virtual_array_reformatter = VirtualArrayReformatter(
    config_manager=cfg_manager
)

point_cloud_generator = PointCloudGenerator(
    config_manager=cfg_manager,
    detector_type="range_doppler_ground_detector",
    detector_params={
        "vel_cfar_type": "os_cfar_1d",
        "vel_cfar_params": {
            "num_train": 5,
            "num_guard": 2,
            "rho": 0.6,
            "alpha": 4
        },
        "altimeter_params": {
            "min_altitude_m": 25.0e-2,
            "zoom_search_region_m": 20.0e-2,
            "altitude_search_limit_m": 40.0e-2,
            "range_bias": 0.0,
            "precise_est_enabled": True
        }
    },
    az_antenna_idxs=[0, 3, 4, 7],
    el_antenna_idxs=[9, 8, 5, 4],
    shift_az_resp=True,
    shift_el_resp=False
)

velocity_estimator.reset()

for i in tqdm(range(dataset.num_frames)):

    # get the radar adc data
    adc_cube = dataset.get_radar_adc_data(i)

    adc_cube = virtual_array_reformatter.process(adc_cube)

    # compute the point cloud
    radar_pts = point_cloud_generator.process(adc_cube)

    # save the altitude history
    vehicle_odom = dataset.get_vehicle_odom_data(idx=i)

    # estimate the velocity
    vel_est = velocity_estimator.process(
        points=radar_pts)

    # obtain UAV vel est by transforming velocities
    # Using matrix from config if available, else default (hardcoded)
    if 'transformation' in config and 'uav_vel_matrix' in config['transformation']:
         matrix = np.array(config['transformation']['uav_vel_matrix'])
         vel_est_uav = matrix @ vel_est
    else:
        # Fallback to hardcoded for Hermes datasets
        vel_est_uav = np.array([
            vel_est[1],
            -vel_est[2],
            vel_est[0]
        ])

    # save the gt velocity history
    vehicle_vel_x = np.average(vehicle_odom[:, 8])
    vehicle_vel_y = np.average(vehicle_odom[:, 9])
    vehicle_vel_z = -1 * np.average(vehicle_odom[:, 10])
    
    velocity_estimator.update_history(
        ground_truth=np.array([vehicle_vel_x, vehicle_vel_y, vehicle_vel_z]),
        estimated=vel_est_uav
    )


# --- Analysis ---

vel_est = np.stack(velocity_estimator.history_estimated, axis=0)
vel_gt = np.stack(velocity_estimator.history_gt, axis=0)

start_idx = config['analysis'].get('start_idx', 0)
end_idx = config['analysis'].get('end_idx', len(vel_est))

# Ensure indices are within bounds
end_idx = min(end_idx, len(vel_est))

print(f"Running analysis from frame {start_idx} to {end_idx}")

# Initialize Analyzer and Plotter
analyzer = VelocityAnalyzer()
analysis_plotter = AnalysisPlotter()

# Perform Analysis
analyzer.analyze(
    history_estimated=vel_est[start_idx:end_idx],
    history_gt=vel_gt[start_idx:end_idx]
)

# Generate Report
summary_df = analyzer.generate_report()
print("\nSummary Statistics of Velocity Estimation Errors:")
print(summary_df.to_string())
print("\n")

# Generate Plots
analysis_plotter.plot_velocity_analysis_summary(
    x_errors=analyzer.get_x_errors(),
    y_errors=analyzer.get_y_errors(),
    z_errors=analyzer.get_z_errors(),
    norm_errors=analyzer.get_norm_errors()
)

# R2 and Inlier Plots
r2_stats = np.stack(velocity_estimator.history_R2_statistics, axis=0)
inlier_stats = np.stack(velocity_estimator.history_inlier_statistics, axis=0)

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

analysis_plotter.plot_time_series(
    data=r2_stats[20:], 
    ax=axs[0], 
    title="R2 Statistic Over Time", 
    ylabel="R2 Statistic",
    xlabel="Frame Index"
)
axs[0].set_ylim([0.0, 1])

analysis_plotter.plot_time_series(
    data=inlier_stats[20:], 
    ax=axs[1], 
    title="Inlier Percentage Over Time", 
    ylabel="Inlier Percentage",
    xlabel="Frame Index"
)
axs[1].set_ylim([0.0, 1])

plt.tight_layout()
plt.show()