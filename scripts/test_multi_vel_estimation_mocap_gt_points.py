import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

from dotenv import load_dotenv
load_dotenv()

# Add parent directory and submodules to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "../"))
sys.path.append(os.path.join(script_dir, "../../../"))

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from cpsl_datasets.cpsl_ds import CpslDS
from mmwave_radar_processing.point_cloud_processing.vel_estimator import VelocityEstimator
from mmwave_radar_processing.analysis.velocity_analyzer import VelocityAnalyzer
from mmwave_radar_processing.plotting.analysis_plotter import AnalysisPlotter

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate aggregated Radar (from pre-computed points) and Flow velocity estimates against MOCAP.")
    parser.add_argument(
        "--config-name",
        type=str,
        default="IcaRAus_multi_dataset_velocity_analysis_points_config.yaml",
        help="Name of the multi-dataset configuration file in analyzer_configs/"
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=10,
        help="Window size for MOCAP velocity smoothing."
    )
    parser.add_argument(
        "--align-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to align MOCAP to the initial Odometry pose for each dataset."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="IcaRAus_multi_vel_mocap_points_results",
        help="Directory to save the aggregated results to."
    )
    parser.add_argument(
        "--takeoff-altitude",
        type=float,
        default=0.25,
        help="Altitude threshold for data recording. frames with abs(altitude) < threshold are ignored."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Create results directory
    repo_root = os.path.join(script_dir, "../../../")
    results_path = os.path.join(repo_root, args.results_dir)
    os.makedirs(results_path, exist_ok=True)

    # Load configuration
    config_path = os.path.join(script_dir, "../analyzer_configs", args.config_name)
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    CONFIG_DIRECTORY = os.getenv("CONFIG_DIRECTORY")
    # Radar Configuration
    cfg_manager = ConfigManager()
    cfg_path = os.path.join(CONFIG_DIRECTORY, config['radar']['config_file'])
    cfg_manager.load_cfg(cfg_path,
                        array_geometry=config['radar']['array_geometry'],
                        array_direction=config['radar']['array_direction'])
    cfg_manager.compute_radar_perforance(profile_idx=0)

    # Load Transformation Matrices
    trans_cfg = config.get('transformation', {})
    uav_vel_radar_msmt_matrix = np.array(trans_cfg.get('uav_vel_radar_msmt', np.eye(3)))
    odom_vel_matrix = np.array(trans_cfg.get('odom_vel_matrix', np.eye(3)))
    vicon_vel_matrix = np.array(trans_cfg.get('vicon_vel_matrix', np.eye(3)))

    # Processor
    vel_est_cfg = config.get('processors', {}).get('velocity_estimator', {})
    velocity_estimator = VelocityEstimator(
        config_manager=cfg_manager,
        min_R2_threshold=vel_est_cfg.get('min_r2_threshold', 0.6),
        min_inlier_percent=vel_est_cfg.get('min_inlier_percent', 0.75),
        moving_window_size=vel_est_cfg.get('moving_window_size', 10),
        z_score_threshold=vel_est_cfg.get('z_score_threshold', 3.0),
        min_std_dev=vel_est_cfg.get('min_std_dev', 0.2),
        outlier_rejection_limit=vel_est_cfg.get('outlier_rejection_limit', 5)
    )

    # Global Accumulators
    all_radar_vel = []
    all_flow_vel = []
    all_mocap_vel = []
    all_timestamps = []

    radar_pc_folder = config['radar'].get('radar_pc_folder', "radar_combined_pc")

    datasets_config = config.get('datasets', [])
    for ds_idx, ds_cfg in enumerate(datasets_config):
        ds_path = ds_cfg.get('path')
        ds_name = ds_cfg.get('name')
        dataset_full_path = os.path.join(ds_path, ds_name)
        
        print(f"[{ds_idx+1}/{len(datasets_config)}] Processing dataset: {ds_name}")
        
        try:
            dataset = CpslDS(
                dataset_path=dataset_full_path,
                radar_pc_folder=radar_pc_folder,
                vehicle_odom_folder="vehicle_odom",
                vicon_folder="vicon_x500_8"
            )
        except Exception as e:
            print(f"Failed to load dataset {ds_name}: {e}")
            continue

        velocity_estimator.reset()
        
        # Local dataset buffers
        ds_odom_raw = []
        ds_vicon_raw = []
        ds_radar_vel = []
        ds_timestamps = []

        for i in tqdm(range(dataset.num_frames)):
            # Odom check for altitude
            vehicle_odom = dataset.get_vehicle_odom_data(idx=i)
            avg_odom = np.mean(vehicle_odom, axis=0)
            
            # Filter by takeoff altitude (abs(z))
            current_alt = abs(avg_odom[3])

            # Radar (from points)
            radar_pts = dataset.get_radar_point_cloud(i)
            vel_est = velocity_estimator.process(points=radar_pts)

            if current_alt <= args.takeoff_altitude:
                continue
            
            ds_radar_vel.append(uav_vel_radar_msmt_matrix @ vel_est)

            # Record odom and timestamps
            ds_odom_raw.append(avg_odom)
            ds_timestamps.append(avg_odom[0])

            # Vicon
            vicon_data = dataset.get_vicon_data(idx=i)
            ds_vicon_raw.append(vicon_data)

        if len(ds_timestamps) == 0:
            print(f"No frames passed filter for {ds_name}. Skipping.")
            continue

        ds_odom_raw = np.array(ds_odom_raw)
        ds_vicon_raw = np.array(ds_vicon_raw)
        ds_radar_vel = np.array(ds_radar_vel)
        ds_timestamps = np.array(ds_timestamps)

        # Coordinate Processes
        odom_rot = Rotation.from_quat(ds_odom_raw[:, [5, 6, 7, 4]])
        vicon_rot = Rotation.from_quat(ds_vicon_raw[:, [4, 5, 6, 3]])
        
        ds_flow_vel = (odom_vel_matrix @ ds_odom_raw[:, 8:11].T).T

        if args.align_frames:
            rot_align = odom_rot[0] * vicon_rot[0].inv()
            vicon_rot_aligned = rot_align * vicon_rot
            vicon_pos_aligned = rot_align.apply(ds_vicon_raw[:, 0:3])
        else:
            vicon_rot_aligned = vicon_rot
            vicon_pos_aligned = ds_vicon_raw[:, 0:3]

        # MOCAP Velocity
        ds_mocap_vel_global = np.zeros_like(vicon_pos_aligned)
        for d in range(3):
            ds_mocap_vel_global[:, d] = np.gradient(vicon_pos_aligned[:, d], ds_timestamps)
        
        if args.smoothing_window > 1:
            for d in range(3):
                ds_mocap_vel_global[:, d] = pd.Series(ds_mocap_vel_global[:, d]).rolling(window=args.smoothing_window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values

        ds_mocap_vel_body = np.zeros_like(ds_mocap_vel_global)
        for i in range(len(ds_mocap_vel_global)):
            ds_mocap_vel_body[i] = vicon_vel_matrix @ vicon_rot_aligned[i].inv().apply(ds_mocap_vel_global[i])

        # Accumulate
        all_radar_vel.append(ds_radar_vel)
        all_flow_vel.append(ds_flow_vel)
        all_mocap_vel.append(ds_mocap_vel_body)
        all_timestamps.append(ds_timestamps)

    if not all_timestamps:
        print("No data collected across all datasets. Exiting.")
        return

    # Aggregator Finalization
    radar_vel = np.concatenate(all_radar_vel, axis=0)
    flow_vel = np.concatenate(all_flow_vel, axis=0)
    mocap_vel = np.concatenate(all_mocap_vel, axis=0)
    timestamps = np.concatenate(all_timestamps, axis=0)

    # --- Analysis ---
    error_method = config.get('analysis', {}).get('error_method', "signed")

    analyzer_radar = VelocityAnalyzer()
    analyzer_flow = VelocityAnalyzer()

    analyzer_radar.analyze(radar_vel, mocap_vel, error_method)
    analyzer_flow.analyze(flow_vel, mocap_vel, error_method)

    df_radar = analyzer_radar.generate_report()
    df_flow = analyzer_flow.generate_report()
    combined_report = pd.concat([df_flow, df_radar], keys=['Flow vs MOCAP', 'Radar vs MOCAP'])
    print("\nAggregated Summary Statistics:")
    print(combined_report)
    combined_report.to_csv(os.path.join(results_path, "summary_statistics.csv"))

    # Save Raw Aggregated Data
    raw_df = pd.DataFrame({
        'timestamp': timestamps,
        'mocap_vx': mocap_vel[:, 0], 'mocap_vy': mocap_vel[:, 1], 'mocap_vz': mocap_vel[:, 2],
        'flow_vx': flow_vel[:, 0], 'flow_vy': flow_vel[:, 1], 'flow_vz': flow_vel[:, 2],
        'radar_vx': radar_vel[:, 0], 'radar_vy': radar_vel[:, 1], 'radar_vz': radar_vel[:, 2]
    })
    raw_df.to_csv(os.path.join(results_path, "raw_velocity_data.csv"), index=False)

    # --- Plotting ---
    plotter = AnalysisPlotter()
    
    # 1. 3x2 Comparison Plot
    fig_comp, axs_comp = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    components = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']
    for i, component in enumerate(components):
        plotter.plot_comparison_time_series(flow_vel[:, i], mocap_vel[:, i], ax=axs_comp[i, 0], title=f"Flow vs MOCAP: {component} Velocity", ylabel="m/s", est_label="Flow", gt_label="MOCAP", est_color=colors[i], show=False)
        plotter.plot_comparison_time_series(radar_vel[:, i], mocap_vel[:, i], ax=axs_comp[i, 1], title=f"Radar (Points) vs MOCAP: {component} Velocity", ylabel="m/s", est_label="Radar", gt_label="MOCAP", est_color=colors[i], show=False)
    plt.tight_layout()
    fig_comp.savefig(os.path.join(results_path, "velocity_comparison_mocap.png"))
    plt.close(fig_comp)

    # 2. Combined Error Summary Plot (2x2)
    fig_sum, axs_sum = plt.subplots(2, 2, figsize=(18, 12), gridspec_kw={'height_ratios': [2, 1]})
    plotter.plot_velocity_analysis_summary(analyzer_flow.get_x_errors(), analyzer_flow.get_y_errors(), analyzer_flow.get_z_errors(), analyzer_flow.get_norm_errors(), axs=axs_sum[:, 0], show=False)
    axs_sum[0, 0].set_title("Flow: Velocity Estimation Errors Over Time", fontsize=15)
    plotter.plot_velocity_analysis_summary(analyzer_radar.get_x_errors(), analyzer_radar.get_y_errors(), analyzer_radar.get_z_errors(), analyzer_radar.get_norm_errors(), axs=axs_sum[:, 1], show=False)
    axs_sum[0, 1].set_title("Radar (Points): Velocity Estimation Errors Over Time", fontsize=15)
    plt.tight_layout()
    fig_sum.savefig(os.path.join(results_path, "velocity_analysis_summary_combined.png"))
    plt.close(fig_sum)

    # 3. Combined Error Histograms (3x2)
    fig_hist, axs_hist = plt.subplots(3, 2, figsize=(15, 12))
    plotter.plot_error_histograms(analyzer_flow.get_x_errors(), analyzer_flow.get_y_errors(), analyzer_flow.get_z_errors(), axs=axs_hist[:, 0], show=False)
    axs_hist[0, 0].set_title(f"Flow: X Velocity Error Distribution\nMean: {np.mean(analyzer_flow.get_x_errors()):.4f}", fontsize=12)
    plotter.plot_error_histograms(analyzer_radar.get_x_errors(), analyzer_radar.get_y_errors(), analyzer_radar.get_z_errors(), axs=axs_hist[:, 1], show=False)
    axs_hist[0, 1].set_title(f"Radar (Points): X Velocity Error Distribution\nMean: {np.mean(analyzer_radar.get_x_errors()):.4f}", fontsize=12)
    plt.tight_layout()
    fig_hist.savefig(os.path.join(results_path, "error_histograms_combined.png"))
    plt.close(fig_hist)

    print(f"Aggregated results saved to {results_path}")

if __name__ == "__main__":
    main()
