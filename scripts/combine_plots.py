import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add parent directory and submodules to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "../"))
sys.path.append(os.path.join(script_dir, "../../../"))

from mmwave_radar_processing.analysis.velocity_analyzer import VelocityAnalyzer
from mmwave_radar_processing.plotting.analysis_plotter import AnalysisPlotter

# Hardcoded default values
DEFAULT_CSV_FILES = [
    "IcaRAus_vel_mocap_comparison_results/raw_velocity_data.csv",
    "IcaRAus_vel_mocap_points_results/raw_velocity_data.csv"
]
DEFAULT_LABELS = ["Radar (IcaRAus)", "Radar (XRIO)"]

def parse_args():
    parser = argparse.ArgumentParser(description="Combine velocity estimation results from multiple CSV files.")
    parser.add_argument(
        "--csv-files",
        nargs="+",
        default=DEFAULT_CSV_FILES,
        help=f"List of CSV files to combine. Default: {DEFAULT_CSV_FILES}"
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=DEFAULT_LABELS,
        help=f"Labels for each CSV file. Default: {DEFAULT_LABELS}"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="IcaRAus_combined_plots",
        help="Directory to save combined plots."
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start index for plotting."
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=-1,
        help="End index for plotting."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if len(args.csv_files) != len(args.labels):
        print("Error: Number of CSV files must match number of labels.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and process each file independently
    analyzers = {}
    mocap_vels = {}
    flow_vels = {}
    radar_vels = {}

    for f, label in zip(args.csv_files, args.labels):
        df = pd.read_csv(f)
        
        # Extract MOCAP
        mocap_vels[label] = df[['mocap_vx', 'mocap_vy', 'mocap_vz']].values
        
        # Extract Flow/Odom
        flow_cols = ['flow_vx', 'flow_vy', 'flow_vz']
        if not all(col in df.columns for col in flow_cols):
            flow_cols = ['odom_vx', 'odom_vy', 'odom_vz']
        flow_vels[label] = df[flow_cols].values if all(col in df.columns for col in flow_cols) else None
        
        # Extract Radar
        # Try exact 'radar_vx' or try to find columns starting with 'radar_vx'
        radar_cols = ['radar_vx', 'radar_vy', 'radar_vz']
        if all(col in df.columns for col in radar_cols):
            radar_vels[label] = df[radar_cols].values
        else:
            radar_vels[label] = np.zeros_like(mocap_vels[label])

    # Setup Plotter with enlarged text
    plotter = AnalysisPlotter()
    plotter.font_size_axis_labels = 16
    plotter.font_size_title = 20
    plotter.font_size_ticks = 14
    plotter.font_size_legend = 14

    # --- Analysis ---
    
    # 1. Analyze Flow (using the first file's flow as reference if available)
    first_label = args.labels[0]
    if flow_vels[first_label] is not None:
        analyzer_flow = VelocityAnalyzer()
        # Handle slicing based on arguments
        total_frames = len(mocap_vels[first_label])
        start_idx = args.start_idx
        end_idx = args.end_idx if args.end_idx != -1 else total_frames
        f_vel = flow_vels[first_label][start_idx:end_idx]
        m_vel = mocap_vels[first_label][start_idx:end_idx]
        analyzer_flow.analyze(f_vel, m_vel, error_method="signed")
        analyzers['Flow'] = analyzer_flow

    # 2. Analyze Radar from each CSV
    for label in args.labels:
        total_frames = len(mocap_vels[label])
        start_idx = args.start_idx
        end_idx = args.end_idx if args.end_idx != -1 else total_frames
        
        r_vel = radar_vels[label][start_idx:end_idx]
        m_vel = mocap_vels[label][start_idx:end_idx]
        
        analyzer = VelocityAnalyzer()
        analyzer.analyze(r_vel, m_vel, error_method="signed")
        analyzers[label] = analyzer

    # --- Summary Statistics ---
    combined_report = pd.concat(
        [analyzers[name].generate_report() for name in analyzers],
        keys=list(analyzers.keys())
    )
    print("\nCombined Summary Statistics:")
    print(combined_report)
    combined_report.to_csv(os.path.join(args.output_dir, "summary_statistics_combined.csv"))

    # --- Plot 1: Comparison Time Series (3x1) ---
    fig_comp, axs_comp = plt.subplots(3, 1, figsize=(7, 7), sharex=False)
    components = ['X', 'Y', 'Z']
    colors = ['green', 'blue', 'red', 'purple', 'orange', 'cyan']
    
    for i, component in enumerate(components):
        ax = axs_comp[i]
        
        for j, label in enumerate(args.labels):
            total_frames = len(mocap_vels[label])
            start_idx = args.start_idx
            end_idx = args.end_idx if args.end_idx != -1 else total_frames
            frame_indices = np.arange(start_idx, end_idx)
            
            m_slice = mocap_vels[label][start_idx:end_idx, i]
            r_slice = radar_vels[label][start_idx:end_idx, i]
            
            if j == 0:
                ax.plot(frame_indices, m_slice, label="MOCAP (GT)", color='black', linestyle='--', alpha=0.8, linewidth=3)
                if flow_vels[label] is not None:
                    f_slice = flow_vels[label][start_idx:end_idx, i]
                    ax.plot(frame_indices, f_slice, label="Flow (Odom)", color=colors[0], alpha=0.7, linewidth=2)
            
            ax.plot(frame_indices, r_slice, label=label, color=colors[j+1], alpha=0.7, linewidth=2)
        
        ax.set_title(f"{component} Velocity Comparison", fontsize=plotter.font_size_title)
        ax.set_ylabel("Velocity (m/s)", fontsize=plotter.font_size_axis_labels)
        ax.tick_params(axis='both', labelsize=plotter.font_size_ticks)
        ax.grid(True, alpha=0.3)
        
        if i == 2:
            ax.set_xlabel("Frame Index", fontsize=plotter.font_size_axis_labels)

    # Add single legend at the bottom
    handles, labels = axs_comp[0].get_legend_handles_labels()
    fig_comp.legend(handles, labels, loc='lower center', ncol=2, fontsize=plotter.font_size_legend)
    
    plt.tight_layout(rect=[0, 0.1, 1, 1]) # Adjust bottom to make room for legend
    fig_comp.savefig(os.path.join(args.output_dir, "velocity_comparison_mocap.png"))
    plt.close(fig_comp)

    # --- Plot 2: Error Summary (Nx2) ---
    num_methods = len(analyzers)
    fig_sum, axs_sum = plt.subplots(2, num_methods, figsize=(9 * num_methods, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Handle single method case for subplot indexing
    if num_methods == 1:
        axs_sum = axs_sum.reshape(2, 1)

    for j, (name, analyzer) in enumerate(analyzers.items()):
        plotter.plot_velocity_analysis_summary(
            analyzer.get_x_errors(), analyzer.get_y_errors(), analyzer.get_z_errors(), analyzer.get_norm_errors(), 
            axs=axs_sum[:, j], show=False
        )
        axs_sum[0, j].set_title(f"{name}: Estimation Errors", fontsize=plotter.font_size_title)

    plt.tight_layout()
    fig_sum.savefig(os.path.join(args.output_dir, "velocity_analysis_summary_combined.png"))
    plt.close(fig_sum)

    # --- Plot 3: Error Histograms (Nx3) ---
    # AnalysisPlotter.plot_error_histograms generates a 3x1 plot for ONE method.
    # We want a 3xN plot for N methods.
    fig_hist, axs_hist = plt.subplots(3, num_methods, figsize=(7 * num_methods, 12))
    
    if num_methods == 1:
        axs_hist = axs_hist.reshape(3, 1)

    for j, (name, analyzer) in enumerate(analyzers.items()):
        plotter.plot_error_histograms(
            analyzer.get_x_errors(), analyzer.get_y_errors(), analyzer.get_z_errors(), 
            axs=axs_hist[:, j], show=False
        )
        axs_hist[0, j].set_title(f"{name}: X Error Distribution\nMean: {np.mean(analyzer.get_x_errors()):.4f}", fontsize=12)

    plt.tight_layout()
    fig_hist.savefig(os.path.join(args.output_dir, "error_histograms_combined.png"))
    plt.close(fig_hist)

    print(f"\nCombined results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
