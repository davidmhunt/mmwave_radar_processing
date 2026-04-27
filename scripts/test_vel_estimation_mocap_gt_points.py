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
    parser = argparse.ArgumentParser(description="Evaluate Radar (from pre-computed points) and Flow velocity estimates against MOCAP.")
    parser.add_argument(
        "--config-name",
        type=str,
        default="IcaRAus_velocity_analysis_points_config.yaml",
        help="Name of the configuration file in analyzer_configs/"
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
        help="Whether to align MOCAP to the initial Odometry pose."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="IcaRAus_vel_mocap_points_results",
        help="Directory to save the results to."
    )
    parser.add_argument(
        "--takeoff-altitude",
        type=float,
        default=0.0,
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

    # Dataset Setup
    ds_cfg = config.get('dataset', {})
    dataset_path = os.path.join(ds_cfg.get('path'), ds_cfg.get('name'))
    
    dataset = CpslDS(
        dataset_path=dataset_path,
        radar_pc_folder=ds_cfg.get('radar_pc_folder', "front_radar_pc"),
        vehicle_odom_folder=ds_cfg.get('vehicle_odom_folder', "vehicle_odom"),
        vicon_folder=ds_cfg.get('vicon_folder', "vicon_x500_8")
    )

    CONFIG_DIRECTORY = os.getenv("CONFIG_DIRECTORY")
    # Radar Configuration for VelocityEstimator parameters
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

    # Processors
    vel_est_cfg = config.get('processors', {}).get('velocity_estimator', {})
    velocity_estimator = VelocityEstimator(
        config_manager=cfg_manager,
        min_R2_threshold=vel_est_cfg.get('min_r2_threshold', 0.6),
        min_inlier_percent=vel_est_cfg.get('min_inlier_percent', 0.75)
    )

    # History Accumulators
    radar_vel_history = []
    odom_raw_history = []
    vicon_raw_history = []
    timestamps = []

    # Processing Loop
    print(f"Processing {dataset.num_frames} frames from {ds_cfg.get('name')}...")
    for i in tqdm(range(dataset.num_frames)):
        # Odom check for altitude
        vehicle_odom = dataset.get_vehicle_odom_data(idx=i)
        avg_odom = np.mean(vehicle_odom, axis=0)
        
        # Filter by takeoff altitude (abs(z))
        current_alt = abs(avg_odom[3])
        if current_alt <= args.takeoff_altitude:
            continue

        # Radar processing (from pre-computed points)
        radar_pts = dataset.get_radar_point_cloud(i)
        vel_est = velocity_estimator.process(points=radar_pts)
        
        # Transform Estimated Radar Velocity (Body Frame)
        vel_est_uav = uav_vel_radar_msmt_matrix @ vel_est
        radar_vel_history.append(vel_est_uav)

        # Record odom and timestamps
        odom_raw_history.append(avg_odom)
        timestamps.append(avg_odom[0])

        # Vicon processing
        vicon_data = dataset.get_vicon_data(idx=i)
        vicon_raw_history.append(vicon_data)

    radar_vel = np.array(radar_vel_history)
    odom_raw = np.array(odom_raw_history)
    vicon_raw = np.array(vicon_raw_history)
    timestamps = np.array(timestamps)

    if len(timestamps) == 0:
        print("No frames passed the takeoff altitude filter. Exiting.")
        return

    # --- Coordinate Processes ---
    odom_rot = Rotation.from_quat(odom_raw[:, [5, 6, 7, 4]])
    vicon_rot = Rotation.from_quat(vicon_raw[:, [4, 5, 6, 3]])
    
    # Calculate Flow Velocity (NED to Body)
    flow_vel = (odom_vel_matrix @ odom_raw[:, 8:11].T).T

    # Align frames if requested
    if args.align_frames:
        rot_align = odom_rot[0] * vicon_rot[0].inv()
        vicon_rot_aligned = rot_align * vicon_rot
        vicon_pos_aligned = rot_align.apply(vicon_raw[:, 0:3])
    else:
        vicon_rot_aligned = vicon_rot
        vicon_pos_aligned = vicon_raw[:, 0:3]

    # Calculate MOCAP Velocity (Derivative of Position)
    mocap_vel_global = np.zeros_like(vicon_pos_aligned)
    for d in range(3):
        mocap_vel_global[:, d] = np.gradient(vicon_pos_aligned[:, d], timestamps)
    
    # Smoothing
    if args.smoothing_window > 1:
        for d in range(3):
            mocap_vel_global[:, d] = pd.Series(mocap_vel_global[:, d]).rolling(window=args.smoothing_window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values

    # Transform MOCAP Velocity to Body Frame
    mocap_vel_body = np.zeros_like(mocap_vel_global)
    for i in range(len(mocap_vel_global)):
        mocap_vel_body[i] = vicon_vel_matrix @ vicon_rot_aligned[i].inv().apply(mocap_vel_global[i])

    # --- Analysis ---
    start_idx = config.get('analysis', {}).get('start_idx', 0)
    end_idx = config.get('analysis', {}).get('end_idx', len(radar_vel))
    if end_idx == -1: end_idx = len(radar_vel)
    error_method = config.get('analysis', {}).get('error_method', "signed")

    analyzer_radar = VelocityAnalyzer()
    analyzer_flow = VelocityAnalyzer()

    analyzer_radar.analyze(radar_vel[start_idx:end_idx], mocap_vel_body[start_idx:end_idx], error_method)
    analyzer_flow.analyze(flow_vel[start_idx:end_idx], mocap_vel_body[start_idx:end_idx], error_method)

    df_radar = analyzer_radar.generate_report()
    df_flow = analyzer_flow.generate_report()
    combined_report = pd.concat([df_flow, df_radar], keys=['Flow vs MOCAP', 'Radar vs MOCAP'])
    print("\nSummary Statistics:")
    print(combined_report)
    combined_report.to_csv(os.path.join(results_path, "summary_statistics.csv"))

    # Save Raw Data
    raw_df = pd.DataFrame({
        'timestamp': timestamps[start_idx:end_idx],
        'mocap_vx': mocap_vel_body[start_idx:end_idx, 0], 'mocap_vy': mocap_vel_body[start_idx:end_idx, 1], 'mocap_vz': mocap_vel_body[start_idx:end_idx, 2],
        'flow_vx': flow_vel[start_idx:end_idx, 0], 'flow_vy': flow_vel[start_idx:end_idx, 1], 'flow_vz': flow_vel[start_idx:end_idx, 2],
        'radar_vx': radar_vel[start_idx:end_idx, 0], 'radar_vy': radar_vel[start_idx:end_idx, 1], 'radar_vz': radar_vel[start_idx:end_idx, 2]
    })
    raw_df.to_csv(os.path.join(results_path, "raw_velocity_data.csv"), index=False)

    # --- Plotting ---
    plotter = AnalysisPlotter()
    
    # 1. 3x2 Comparison Plot
    fig_comp, axs_comp = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    components = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']
    for i, component in enumerate(components):
        plotter.plot_comparison_time_series(flow_vel[start_idx:end_idx, i], mocap_vel_body[start_idx:end_idx, i], ax=axs_comp[i, 0], title=f"Flow vs MOCAP: {component} Velocity", ylabel="m/s", est_label="Flow", gt_label="MOCAP", est_color=colors[i], show=False)
        plotter.plot_comparison_time_series(radar_vel[start_idx:end_idx, i], mocap_vel_body[start_idx:end_idx, i], ax=axs_comp[i, 1], title=f"Radar (Points) vs MOCAP: {component} Velocity", ylabel="m/s", est_label="Radar", gt_label="MOCAP", est_color=colors[i], show=False)
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

    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
