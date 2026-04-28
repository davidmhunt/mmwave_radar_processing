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
from mmwave_radar_processing.processors.point_cloud_generator import PointCloudGenerator
from mmwave_radar_processing.processors.virtual_array_reformater import VirtualArrayReformatter
from mmwave_radar_processing.point_cloud_processing.vel_estimator import VelocityEstimator
from mmwave_radar_processing.analysis.velocity_analyzer import VelocityAnalyzer
from mmwave_radar_processing.plotting.analysis_plotter import AnalysisPlotter

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate radar and flow velocity estimates against MOCAP ground truth.")
    parser.add_argument(
        "--config-name",
        type=str,
        default="IcaRAus_velocity_analysis_config.yaml",
        help="Name of the configuration file in analyzer_configs/"
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=10,
        help="Window size for MOCAP velocity smoothing."
    )
    parser.add_argument(
        "--takeoff-altitude",
        type=float,
        default=0.0,
        help="Altitude threshold for data recording. frames with abs(altitude) < threshold are ignored."
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
        default="IcaRAus_vel_mocap_comparison_results",
        help="Directory to save the results to (relative to the repository root)."
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

    DATASET_PATH = config['dataset']['path']
    CONFIG_DIRECTORY = os.getenv("CONFIG_DIRECTORY")

    # Radar Configuration
    cfg_manager = ConfigManager()
    cfg_path = os.path.join(CONFIG_DIRECTORY, config['radar']['config_file'])
    cfg_manager.load_cfg(cfg_path,
                        array_geometry=config['radar']['array_geometry'],
                        array_direction=config['radar']['array_direction'])
    cfg_manager.compute_radar_perforance(profile_idx=0)

    # Dataset Loading
    dataset_name = config['dataset']['name']
    dataset_path = os.path.join(DATASET_PATH, dataset_name)
    print(f"Loading dataset from: {dataset_path}")

    dataset = CpslDS(
        dataset_path=dataset_path,
        radar_adc_folder=config['dataset'].get('radar_adc_folder', "down_radar_adc"),
        vehicle_odom_folder=config['dataset'].get('vehicle_odom_folder', "vehicle_odom"),
        vicon_folder=config['dataset'].get('vicon_folder', "vicon_x500_8")
    )

    # Processors Initialization
    processors_cfg = config.get('processors', {})
    vel_est_cfg = processors_cfg.get('velocity_estimator', {})
    velocity_estimator = VelocityEstimator(
        config_manager=cfg_manager,
        min_R2_threshold=vel_est_cfg.get('min_r2_threshold', 0.6),
        min_inlier_percent=vel_est_cfg.get('min_inlier_percent', 0.75)
    )
    virtual_array_reformatter = VirtualArrayReformatter(config_manager=cfg_manager)
    
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

    # --- Load Transformation Matrices from Config ---
    trans_cfg = config.get('transformation', {})
    # Matrix for Radar Measurements
    uav_vel_radar_msmt_matrix = np.array(trans_cfg.get('uav_vel_radar_msmt', trans_cfg.get('uav_vel_matrix', np.eye(3))))
    # Matrix for Flow/Odom Measurements
    odom_vel_matrix = np.array(trans_cfg.get('odom_vel_matrix', trans_cfg.get('gt_vel_matrix', trans_cfg.get('uav_vel_est', np.eye(3)))))
    # Matrix for Vicon GT
    vicon_vel_matrix = np.array(trans_cfg.get('vicon_vel_matrix', np.eye(3)))

    # Processing Initialization
    velocity_estimator.reset()
    odom_raw_history = []
    vicon_raw_history = []
    radar_vel_history = []
    timestamps = []

    # Processing Loop
    for i in tqdm(range(dataset.num_frames)):
        # Odom check for altitude
        vehicle_odom = dataset.get_vehicle_odom_data(idx=i)
        avg_odom = np.mean(vehicle_odom, axis=0)
        
        # Filter by takeoff altitude (abs(z))
        current_alt = abs(avg_odom[3])
        if current_alt <= args.takeoff_altitude:
            continue

        # Radar processing
        adc_cube = dataset.get_radar_adc_data(i)
        adc_cube = virtual_array_reformatter.process(adc_cube)
        radar_pts = point_cloud_generator.process(adc_cube)
        vel_est = velocity_estimator.process(points=radar_pts)
        
        # Transform Estimated Radar Velocity (Body Frame)
        vel_est_uav = uav_vel_radar_msmt_matrix @ vel_est
        radar_vel_history.append(vel_est_uav)

        # Record odom and timestamps
        odom_raw_history.append(avg_odom)
        timestamps.append(avg_odom[0])

        # Vicon processing
        vicon_data = dataset.get_vicon_data(idx=i) # [t_x, t_y, t_z, r_w, r_x, r_y, r_z]
        vicon_raw_history.append(vicon_data)

    odom_raw_history = np.array(odom_raw_history)
    vicon_raw_history = np.array(vicon_raw_history)
    radar_vel_history = np.array(radar_vel_history)
    timestamps = np.array(timestamps)
    
    if len(timestamps) == 0:
        print("No frames passed the takeoff altitude filter. Exiting.")
        return

    # Orientation Handling (NED-base)
    odom_rot = Rotation.from_quat(odom_raw_history[:, [5, 6, 7, 4]])
    vicon_rot = Rotation.from_quat(vicon_raw_history[:, [4, 5, 6, 3]])

    # Transform Ground Truth Velocity (Flow)
    flow_vel_processed = (odom_vel_matrix @ odom_raw_history[:, 8:11].T).T
    
    # Optional alignment
    if args.align_frames:
        rot_align = odom_rot[0] * vicon_rot[0].inv()
        vicon_rot_aligned = rot_align * vicon_rot
        vicon_pos_aligned = rot_align.apply(vicon_raw_history[:, 0:3])
    else:
        vicon_rot_aligned = vicon_rot
        vicon_pos_aligned = vicon_raw_history[:, 0:3]

    # Compute MOCAP velocity (World aligned frame)
    mocap_vel_global = np.zeros_like(vicon_pos_aligned)
    for d in range(3):
        mocap_vel_global[:, d] = np.gradient(vicon_pos_aligned[:, d], timestamps)

    # Smoothing
    if args.smoothing_window > 1:
        for d in range(3):
            mocap_vel_global[:, d] = pd.Series(mocap_vel_global[:, d]).rolling(window=args.smoothing_window, center=True).mean().fillna(method='bfill').fillna(method='ffill').values

    # Transform MOCAP velocity to Body Frame and apply MOCAP config matrix
    mocap_vel_body = np.zeros_like(mocap_vel_global)
    for i in range(len(mocap_vel_global)):
        # Correctly rotate to body and then apply any vicon-specific Z-flips or axes swaps from config
        mocap_vel_body[i] = vicon_vel_matrix @ vicon_rot_aligned[i].inv().apply(mocap_vel_global[i])

    # --- Analysis ---
    analyzer_radar = VelocityAnalyzer()
    analyzer_flow = VelocityAnalyzer()
    
    start_idx = config['analysis'].get('start_idx', 0)
    end_idx = config['analysis'].get('end_idx', len(radar_vel_history))
    if end_idx == -1: end_idx = len(radar_vel_history)
    error_method = config['analysis'].get('error_method', "signed")

    analyzer_radar.analyze(
        history_estimated=radar_vel_history[start_idx:end_idx],
        history_gt=mocap_vel_body[start_idx:end_idx],
        error_method=error_method
    )
    
    analyzer_flow.analyze(
        history_estimated=flow_vel_processed[start_idx:end_idx],
        history_gt=mocap_vel_body[start_idx:end_idx],
        error_method=error_method
    )

    df_radar = analyzer_radar.generate_report()
    df_flow = analyzer_flow.generate_report()
    
    # Combined Report
    combined_report = pd.concat([df_flow, df_radar], keys=['Flow vs MOCAP', 'Radar vs MOCAP'])
    print("\nSummary Statistics:")
    print(combined_report)
    combined_report.to_csv(os.path.join(results_path, "summary_statistics.csv"))

    # --- Save Raw Estimates to CSV ---
    raw_data = {
        'timestamp': timestamps[start_idx:end_idx],
        'mocap_vx': mocap_vel_body[start_idx:end_idx, 0],
        'mocap_vy': mocap_vel_body[start_idx:end_idx, 1],
        'mocap_vz': mocap_vel_body[start_idx:end_idx, 2],
        'odom_vx': flow_vel_processed[start_idx:end_idx, 0],
        'odom_vy': flow_vel_processed[start_idx:end_idx, 1],
        'odom_vz': flow_vel_processed[start_idx:end_idx, 2],
        'radar_vx': radar_vel_history[start_idx:end_idx, 0],
        'radar_vy': radar_vel_history[start_idx:end_idx, 1],
        'radar_vz': radar_vel_history[start_idx:end_idx, 2]
    }
    raw_df = pd.DataFrame(raw_data)
    raw_csv_path = os.path.join(results_path, "raw_velocity_data.csv")
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"Raw velocity data saved to {raw_csv_path}")

    # --- Plotting (3x2 Layout) ---
    plotter = AnalysisPlotter()
    fig, axs = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    
    components = ['X', 'Y', 'Z']
    colors = ['red', 'green', 'blue']
    
    for i, component in enumerate(components):
        # Column 0: Flow vs MOCAP
        plotter.plot_comparison_time_series(
            estimated=flow_vel_processed[start_idx:end_idx, i],
            ground_truth=mocap_vel_body[start_idx:end_idx, i],
            ax=axs[i, 0],
            title=f"Flow vs MOCAP: {component} Velocity",
            ylabel="Velocity (m/s)",
            est_label="Flow (NED)",
            gt_label="MOCAP (Body)",
            est_color=colors[i],
            show=False
        )
        
        # Column 1: Radar vs MOCAP
        plotter.plot_comparison_time_series(
            estimated=radar_vel_history[start_idx:end_idx, i],
            ground_truth=mocap_vel_body[start_idx:end_idx, i],
            ax=axs[i, 1],
            title=f"Radar vs MOCAP: {component} Velocity",
            ylabel="Velocity (m/s)",
            est_label="Radar (Body)",
            gt_label="MOCAP (Body)",
            est_color=colors[i],
            show=False
        )

    plt.tight_layout()
    fig.savefig(os.path.join(results_path, "velocity_comparison_mocap.png"))
    plt.close(fig)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
