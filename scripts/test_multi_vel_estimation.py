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
    parser = argparse.ArgumentParser(description="Run velocity estimation analysis over multiple datasets.")
    parser.add_argument(
        "--config-name",
        type=str,
        default="multi_dataset_velocity_analysis_config.yaml",
        help="Name of the configuration file in analyzer_configs/"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="ICaRAus_vel_vs_flow_comparison",#"results",
        help="Directory to save the results to (relative to where the script is run)."
    )
    # Changed defaults for multi-dataset per user request
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


    # Multi-dataset change
    datasets_config = config.get('datasets', [])
    if not datasets_config:
        print("Error: No 'datasets' entry found in the configuration.")
        sys.exit(1)
        
    CONFIG_DIRECTORY = os.getenv("CONFIG_DIRECTORY")


    # Radar Configuration
    cfg_manager = ConfigManager()
    cfg_path = os.path.join(CONFIG_DIRECTORY, config['radar']['config_file'])

    cfg_manager.load_cfg(cfg_path,
                        array_geometry=config['radar']['array_geometry'],
                        array_direction=config['radar']['array_direction'])

    cfg_manager.compute_radar_perforance(profile_idx=0)
    cfg_manager.print_cfg_overview()

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


    all_vel_est = []
    all_vel_gt = []

    for dataset_idx, ds_cfg in enumerate(datasets_config):
        dataset_path_dir = ds_cfg.get('path')
        dataset_name = ds_cfg.get('name')
        dataset_path = os.path.join(dataset_path_dir, dataset_name)
        
        print(f"[{dataset_idx+1}/{len(datasets_config)}] Loading dataset from: {dataset_path}")
        
        try:
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
        except Exception as e:
            print(f"Failed to load dataset {dataset_name}: {e}")
            continue

        velocity_estimator.reset()

        # Processing Loop across ALL frames of the dataset
        for i in tqdm(range(dataset.num_frames)):
            # Get Data
            # Note: CpslDS.get_radar_adc_data might fail if frame doesn't exist etc. Handling usually internal.
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

        # Collect data for this dataset
        if len(velocity_estimator.history_estimated) > 0:
            dataset_vel_est = np.stack(velocity_estimator.history_estimated, axis=0)
            dataset_vel_gt = np.stack(velocity_estimator.history_gt, axis=0)
            
            all_vel_est.append(dataset_vel_est)
            all_vel_gt.append(dataset_vel_gt)

    if not all_vel_est:
        print("Error: No data successfully processed across any dataset.")
        sys.exit(1)

    # --- Analysis ---
    # Concatenate all datasets' estimations and ground truths
    vel_est = np.concatenate(all_vel_est, axis=0)
    vel_gt = np.concatenate(all_vel_gt, axis=0)

    # Omit start/end index trimming to use all frames seamlessly
    error_method = config.get('analysis', {}).get('error_method', "signed")

    print(f"Running analysis on {len(vel_est)} total frames using '{error_method}' error method")

    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)

    analyzer = VelocityAnalyzer()
    analysis_plotter = AnalysisPlotter()

    analyzer.analyze(
        history_estimated=vel_est,
        history_gt=vel_gt,
        error_method=error_method
    )

    summary_df = analyzer.generate_report()
    print("\nSummary Statistics of Velocity Estimation Errors (Across All Datasets):")
    print(summary_df.to_string())
    print("\n")

    csv_path = os.path.join(args.results_dir, "summary_statistics.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary statistics saved to {csv_path}\n")

    # Plotting
    
    # 1. Comparison Plot
    if args.plot_comparison:
        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        
        analysis_plotter.plot_comparison_time_series(
            estimated=vel_est[:, 0],
            ground_truth=vel_gt[:, 0],
            ax=axs[0],
            title="X Velocity Comparison",
            ylabel="Velocity (m/s)",
            est_color="red", gt_color="black",
            show=False
        )
        
        analysis_plotter.plot_comparison_time_series(
            estimated=vel_est[:, 1],
            ground_truth=vel_gt[:, 1],
            ax=axs[1],
            title="Y Velocity Comparison",
            ylabel="Velocity (m/s)",
            est_color="green", gt_color="black",
            show=False
        )
        
        analysis_plotter.plot_comparison_time_series(
            estimated=vel_est[:, 2],
            ground_truth=vel_gt[:, 2],
            ax=axs[2],
            title="Z Velocity Comparison",
            ylabel="Velocity (m/s)",
            est_color="blue", gt_color="black",
            show=False
        )
        plt.tight_layout()
        comp_path = os.path.join(args.results_dir, "comparison_time_series.png")
        fig.savefig(comp_path)
        plt.close(fig)
        print(f"Saved comparison plot to {comp_path}")

    # 2. Time Series Errors and Distributions
    if args.plot_time_series_errors or args.plot_distributions:
        
        if args.plot_time_series_errors and args.plot_distributions:
            fig, axs = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1]})
            analysis_plotter.plot_velocity_analysis_summary(
                x_errors=analyzer.get_x_errors(),
                y_errors=analyzer.get_y_errors(),
                z_errors=analyzer.get_z_errors(),
                norm_errors=analyzer.get_norm_errors(),
                axs=axs,
                show=False
            )
            summary_path = os.path.join(args.results_dir, "velocity_analysis_summary.png")
            fig.savefig(summary_path)
            plt.close(fig)
            print(f"Saved velocity analysis summary plot to {summary_path}")
        elif args.plot_distributions:
            #TODO: Implement this functionality
            pass
    
    # 3. Explicit Histograms
    if args.plot_histograms:
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        analysis_plotter.plot_error_histograms(
            x_errors=analyzer.get_x_errors(),
            y_errors=analyzer.get_y_errors(),
            z_errors=analyzer.get_z_errors(),
            axs=axs,
            show=False
        )
        hist_path = os.path.join(args.results_dir, "error_histograms.png")
        fig.savefig(hist_path)
        plt.close(fig)
        print(f"Saved error histograms plot to {hist_path}")

if __name__ == "__main__":
    main()
