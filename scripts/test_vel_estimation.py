import sys
import os
import argparse
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

def parse_args():
    parser = argparse.ArgumentParser(description="Run velocity estimation analysis.")
    parser.add_argument(
        "--config-name",
        type=str,
        default="velocity_analysis_config.yaml",
        help="Name of the configuration file in analyzer_configs/"
    )
    parser.add_argument(
        "--plot-time-series-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot error time series."
    )
    parser.add_argument(
        "--plot-distributions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot error distributions (CDF/Hist)."
    )
    parser.add_argument(
        "--plot-comparison",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot estimated vs ground truth comparison."
    )
    parser.add_argument(
        "--plot-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot R2 and Inlier statistics."
    )
    parser.add_argument(
        "--plot-histograms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot explicit error histograms for X, Y, Z."
    )
    return parser.parse_args()

def main():

    args = parse_args()

    
    # Load configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "../analyzer_configs", args.config_name)

    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)


    DATASET_PATH = config['dataset']['path']
    CONFIG_DIRECTORY = os.getenv("CONFIG_DIRECTORY")


    # Radar Configuration
    cfg_manager = ConfigManager()
    cfg_path = os.path.join(CONFIG_DIRECTORY, config['radar']['config_file'])

    cfg_manager.load_cfg(cfg_path,
                        array_geometry=config['radar']['array_geometry'],
                        array_direction=config['radar']['array_direction'])

    cfg_manager.compute_radar_perforance(profile_idx=0)
    cfg_manager.print_cfg_overview()

    # Dataset Loading
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

    # Processors Initialization
    processors_cfg = config.get('processors', {})

    # Velocity Estimator
    vel_est_cfg = processors_cfg.get('velocity_estimator', {})
    velocity_estimator = VelocityEstimator(
        config_manager=cfg_manager,
        min_R2_threshold=vel_est_cfg.get('min_r2_threshold', 0.6),
        min_inlier_percent=vel_est_cfg.get('min_inlier_percent', 0.75)
    )

    # Virtual Array Reformatter
    virtual_array_reformatter = VirtualArrayReformatter(
        config_manager=cfg_manager
    )

    # Point Cloud Generator
    pc_gen_cfg = processors_cfg.get('point_cloud_generator', {})
    point_cloud_generator = PointCloudGenerator(
        config_manager=cfg_manager,
        detector_type=pc_gen_cfg.get('detector_type', "range_doppler_ground_detector"),
        detector_params=pc_gen_cfg.get('detector_params', {}),
        az_antenna_idxs=pc_gen_cfg.get('az_antenna_idxs', [0, 3, 4, 7]),
        el_antenna_idxs=pc_gen_cfg.get('el_antenna_idxs', [9, 8, 5, 4]),
        shift_az_resp=pc_gen_cfg.get('shift_az_resp', True),
        shift_el_resp=pc_gen_cfg.get('shift_el_resp', False)
    )

    velocity_estimator.reset()

    # Processing Loop
    for i in tqdm(range(dataset.num_frames)):
        # Get Data
        adc_cube = dataset.get_radar_adc_data(i)
        
        # Process
        adc_cube = virtual_array_reformatter.process(adc_cube)
        radar_pts = point_cloud_generator.process(adc_cube)
        vehicle_odom = dataset.get_vehicle_odom_data(idx=i)
        
        vel_est = velocity_estimator.process(points=radar_pts)

        # Transformation matrices
        uav_vel_matrix = np.array(config['transformation'].get('uav_vel_matrix', np.eye(3)))
        gt_vel_matrix = np.array(config['transformation'].get('gt_vel_matrix', np.eye(3)))

        # Transform Estimated Velocity
        vel_est_uav = uav_vel_matrix @ vel_est

        # Transform Ground Truth Velocity
        
        raw_gt_x = np.average(vehicle_odom[:, 8])
        raw_gt_y = np.average(vehicle_odom[:, 9])
        raw_gt_z = np.average(vehicle_odom[:, 10]) 
        
        raw_gt_vel = np.array([raw_gt_x, raw_gt_y, raw_gt_z])
        gt_vel_uav = gt_vel_matrix @ raw_gt_vel
        
        # Update History
        velocity_estimator.update_history(
            ground_truth=gt_vel_uav,
            estimated=vel_est_uav
        )

    # --- Analysis ---
    vel_est = np.stack(velocity_estimator.history_estimated, axis=0)
    vel_gt = np.stack(velocity_estimator.history_gt, axis=0)

    start_idx = config['analysis'].get('start_idx', 0)
    end_idx = config['analysis'].get('end_idx', len(vel_est))
    end_idx = min(end_idx, len(vel_est))
    
    error_method = config['analysis'].get('error_method', "signed")

    print(f"Running analysis from frame {start_idx} to {end_idx} using '{error_method}' error method")

    analyzer = VelocityAnalyzer()
    analysis_plotter = AnalysisPlotter()

    analyzer.analyze(
        history_estimated=vel_est[start_idx:end_idx],
        history_gt=vel_gt[start_idx:end_idx],
        error_method=error_method
    )

    summary_df = analyzer.generate_report()
    print("\nSummary Statistics of Velocity Estimation Errors:")
    print(summary_df.to_string())
    print("\n")

    # Plotting
    
    # 1. Comparison Plot (New)
    if args.plot_comparison:
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        
        analysis_plotter.plot_comparison_time_series(
            estimated=vel_est[start_idx:end_idx, 0],
            ground_truth=vel_gt[start_idx:end_idx, 0],
            ax=axs[0],
            title="X Velocity Comparison",
            ylabel="Velocity (m/s)",
            est_color="red", gt_color="black"
        )
        
        analysis_plotter.plot_comparison_time_series(
            estimated=vel_est[start_idx:end_idx, 1],
            ground_truth=vel_gt[start_idx:end_idx, 1],
            ax=axs[1],
            title="Y Velocity Comparison",
            ylabel="Velocity (m/s)",
            est_color="green", gt_color="black"
        )
        
        analysis_plotter.plot_comparison_time_series(
            estimated=vel_est[start_idx:end_idx, 2],
            ground_truth=vel_gt[start_idx:end_idx, 2],
            ax=axs[2],
            title="Z Velocity Comparison",
            ylabel="Velocity (m/s)",
            est_color="blue", gt_color="black"
        )
        plt.tight_layout()
        plt.show()

    # 2. Time Series Errors and Distributions
    if args.plot_time_series_errors or args.plot_distributions:
        
        if args.plot_time_series_errors and args.plot_distributions:
            analysis_plotter.plot_velocity_analysis_summary(
                x_errors=analyzer.get_x_errors(),
                y_errors=analyzer.get_y_errors(),
                z_errors=analyzer.get_z_errors(),
                norm_errors=analyzer.get_norm_errors()
            )
        else:
             # Manual split if necessary (simpler to just show the summary one if either is requested for now, 
             # as they are on the same figure in recent implementation)
             # But let's respect the flag if possible.
             pass
    
    # 3. Explicit Histograms (New)
    if args.plot_histograms:
        analysis_plotter.plot_error_histograms(
            x_errors=analyzer.get_x_errors(),
            y_errors=analyzer.get_y_errors(),
            z_errors=analyzer.get_z_errors()
        )

    # 4. Stats
    if args.plot_stats:
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

if __name__ == "__main__":
    main()