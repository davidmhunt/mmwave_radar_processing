import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, List, Optional

class PlotterGUIViews:
    """A class for generating Matplotlib-based plots of mmWave radar data.
    
    This class provides modular functions for plotting various radar data views
    using Matplotlib, mimicking the visualizations found in the GUI.
    """

    def __init__(self, font_size_title: int = 16, font_size_label: int = 14, font_size_ticks: int = 12) -> None:
        """Initializes the plotter with specific font sizes.

        Args:
            font_size_title: Font size for plot titles.
            font_size_label: Font size for axis labels.
            font_size_ticks: Font size for axis ticks.
        """
        self.font_size_title = font_size_title
        self.font_size_label = font_size_label
        self.font_size_ticks = font_size_ticks
        self.cmap = "viridis"

    def _prepare_data(self, payload: Dict[str, Any], convert_to_dB: bool = False) -> np.ndarray:
        """Extracts and optionally converts data to dB from the payload.

        Args:
            payload: Dictionary containing 'data'.
            convert_to_dB: Whether to convert the data to decibels.

        Returns:
            The prepared data as a NumPy array.
        """
        data = np.array(payload.get("data", []))
        if data.size == 0:
            return data
            
        if convert_to_dB:
            return 20 * np.log10(np.maximum(np.abs(data), 1e-12))
        return np.abs(data)

    def plot_range_angle_view(self, ax: plt.Axes, payload: Dict[str, Any], convert_to_dB: bool = False) -> None:
        """Plots the range-angle response.

        Args:
            ax: Matplotlib Axes object to plot on.
            payload: Payload containing 'data', 'angle_bins', and 'range_bins'.
            convert_to_dB: Whether to display the heatmap in dB.
        """
        data = self._prepare_data(payload, convert_to_dB)
        if data.size == 0:
            return

        angle_bins = payload.get("angle_bins")
        range_bins = payload.get("range_bins")

        if angle_bins is not None and range_bins is not None:
            extent = [angle_bins[0], angle_bins[-1], range_bins[0], range_bins[-1]]
            im = ax.imshow(data, extent=extent, origin='lower', aspect='auto', cmap=self.cmap)
        else:
            im = ax.imshow(data, origin='lower', aspect='auto', cmap=self.cmap)

        ax.set_xlabel("Angle (rad)", fontsize=self.font_size_label)
        ax.set_ylabel("Range (m)", fontsize=self.font_size_label)
        title = "Range-Angle Heatmap (dB)" if convert_to_dB else "Range-Angle Heatmap (mag)"
        ax.set_title(title, fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)

    def plot_doppler_azimuth_view(self, ax: plt.Axes, payload: Dict[str, Any], convert_to_dB: bool = False) -> None:
        """Plots the Doppler-azimuth response.

        Args:
            ax: Matplotlib Axes object to plot on.
            payload: Payload containing 'data', 'vel_bins', and 'angle_bins'.
            convert_to_dB: Whether to display the heatmap in dB.
        """
        data = self._prepare_data(payload, convert_to_dB)
        if data.size == 0:
            return

        vel_bins = payload.get("vel_bins")
        angle_bins = payload.get("angle_bins")

        if vel_bins is not None and angle_bins is not None:
            extent = [angle_bins[0], angle_bins[-1], vel_bins[0], vel_bins[-1]]
            # View expected [angle, velocity] but data is [velocity, angle] usually?
            # RangeAngleView transposes in GUI. Let's check payload structure.
            # In GUI: display = data.T then setImage. Transpose makes it [angle, range].
            # Here imshow extent [left, right, bottom, top] = [angle_0, angle_n, vel_0, vel_n]
            # So data should be [velocity, angle].
            im = ax.imshow(data, extent=extent, origin='lower', aspect='auto', cmap=self.cmap)
        else:
            im = ax.imshow(data, origin='lower', aspect='auto', cmap=self.cmap)

        ax.set_xlabel("Angle (rad)", fontsize=self.font_size_label)
        ax.set_ylabel("Velocity (m/s)", fontsize=self.font_size_label)
        title = "Doppler-Azimuth Heatmap (dB)" if convert_to_dB else "Doppler-Azimuth Heatmap (mag)"
        ax.set_title(title, fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)

    def plot_range_doppler_view(self, ax: plt.Axes, payload: Dict[str, Any], convert_to_dB: bool = False) -> None:
        """Plots the range-Doppler response.

        Args:
            ax: Matplotlib Axes object to plot on.
            payload: Payload containing 'data', 'range_bins', and 'vel_bins'.
            convert_to_dB: Whether to display the heatmap in dB.
        """
        data = self._prepare_data(payload, convert_to_dB)
        if data.size == 0:
            return

        range_bins = payload.get("range_bins")
        vel_bins = payload.get("vel_bins")

        if range_bins is not None and vel_bins is not None:
            extent = [vel_bins[0], vel_bins[-1], range_bins[0], range_bins[-1]]
            im = ax.imshow(data, extent=extent, origin='lower', aspect='auto', cmap=self.cmap)
        else:
            im = ax.imshow(data, origin='lower', aspect='auto', cmap=self.cmap)

        ax.set_xlabel("Velocity (m/s)", fontsize=self.font_size_label)
        ax.set_ylabel("Range (m)", fontsize=self.font_size_label)
        title = "Range-Doppler Heatmap (dB)" if convert_to_dB else "Range-Doppler Heatmap (mag)"
        ax.set_title(title, fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)

    def plot_micro_doppler_view(self, ax: plt.Axes, payload: Dict[str, Any], convert_to_dB: bool = False) -> None:
        """Plots the micro-Doppler spectrogram.

        Args:
            ax: Matplotlib Axes object to plot on.
            payload: Payload containing 'data', 'time_bins', and 'vel_bins'.
            convert_to_dB: Whether to display the heatmap in dB.
        """
        data = self._prepare_data(payload, convert_to_dB)
        if data.size == 0:
            return

        time_bins = payload.get("time_bins")
        vel_bins = payload.get("vel_bins")

        if time_bins is not None and vel_bins is not None:
            extent = [time_bins[0], time_bins[-1], vel_bins[0], vel_bins[-1]]
            im = ax.imshow(data, extent=extent, origin='lower', aspect='auto', cmap=self.cmap)
        else:
            im = ax.imshow(data, origin='lower', aspect='auto', cmap=self.cmap)

        ax.set_xlabel("Time (s)", fontsize=self.font_size_label)
        ax.set_ylabel("Velocity (m/s)", fontsize=self.font_size_label)
        title = "Micro-Doppler Spectrogram (dB)" if convert_to_dB else "Micro-Doppler Spectrogram (mag)"
        ax.set_title(title, fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)

    def plot_range_response_view(self, ax: plt.Axes, payload: Dict[str, Any], convert_to_dB: bool = False) -> None:
        """Plots the 1D range response.

        Args:
            ax: Matplotlib Axes object to plot on.
            payload: Payload containing 'data' and 'range_bins'.
            convert_to_dB: Whether to display the plot in dB.
        """
        data = self._prepare_data(payload, convert_to_dB)
        if data.size == 0:
            return

        range_bins = payload.get("range_bins")

        if range_bins is not None:
            ax.plot(range_bins, data, linewidth=2.5)
        else:
            ax.plot(data, linewidth=2.5)

        ax.set_xlabel("Range (m)", fontsize=self.font_size_label)
        ax.set_ylabel("Amplitude (dB)" if convert_to_dB else "Amplitude (mag)", fontsize=self.font_size_label)
        ax.set_title("Range Profile", fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)
        ax.grid(True)

    def plot_altitude_view(self, ax: plt.Axes, payload: Dict[str, Any], convert_to_dB: bool = False) -> None:
        """Plots the altitude view with a vertical line at the estimated altitude.

        Args:
            ax: Matplotlib Axes object to plot on.
            payload: Payload containing 'coarse_fft_data', 'range_bins', and 'current_altitude_corrected_m'.
            convert_to_dB: Whether to display the plot in dB.
        """
        # Map coarse_fft_data to data for plot_range_response_view
        if "coarse_fft_data" in payload:
            payload["data"] = payload["coarse_fft_data"]
            
        self.plot_range_response_view(ax, payload, convert_to_dB)
        
        altitude = payload.get("current_altitude_corrected_m")
        if altitude is not None and altitude > 0:
            ax.axvline(x=altitude, color='g', linestyle='--', linewidth=3, label=f'Alt: {altitude:.2f}m')
            ax.legend(fontsize=self.font_size_ticks)

    def plot_point_cloud_view(self, ax: plt.Axes, payload: Dict[str, Any], convert_to_dB: bool = False) -> None:
        """Plots a 3D scatter plot of the radar point cloud.

        Args:
            ax: Matplotlib Axes object (must have projection='3d') to plot on.
            payload: Payload containing 'data' (N x 4 array of x, y, z, velocity).
            convert_to_dB: Ignored for point cloud.
        """
        data = payload.get("data") if isinstance(payload, dict) else payload
        if data is None or len(data) == 0:
            return
            
        points = np.array(data)
        pos = points[:, :3]
        velocity = points[:, 3]
        
        # Color mapping logic matching GUI
        min_vel = -0.25
        max_vel = 0.25
        # norm_vel = (velocity - min_vel) / (max_vel - min_vel)
        norm_vel = np.clip(velocity, min_vel, max_vel)
        
        # Use scatter3D for points
        sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=norm_vel, cmap="coolwarm", s=10)
        
        # Add colorbar for velocity
        cbar = plt.colorbar(sc, ax=ax, pad=0.1)
        cbar.set_label("Velocity (m/s)", fontsize=self.font_size_label)
        cbar.ax.tick_params(labelsize=self.font_size_ticks)
        
        ax.set_xlabel("X (m)", fontsize=self.font_size_label)
        ax.set_ylabel("Y (m)", fontsize=self.font_size_label)
        ax.set_zlabel("Z (m)", fontsize=self.font_size_label)
        ax.set_title("Point Cloud", fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)

    def plot_point_cloud_2d_view(self, ax: plt.Axes, payload: Dict[str, Any], convert_to_dB: bool = False) -> None:
        """Plots a 2D scatter plot of the radar point cloud.

        Args:
            ax: Matplotlib Axes object to plot on.
            payload: Payload containing 'data' (N x 3 array) and 'axis_labels'.
            convert_to_dB: Ignored for point cloud.
        """
        if not isinstance(payload, dict):
            return
            
        data = payload.get("data")
        labels = payload.get("axis_labels", ["X (m)", "Y (m)"])
        
        if data is None or len(data) == 0:
            return
            
        points = np.array(data)
        pos = points[:, :2]
        velocity = points[:, 2]
        
        # Color mapping logic matching 3D
        min_vel = -0.25
        max_vel = 0.25
        # norm_vel = (velocity - min_vel) / (max_vel - min_vel)
        norm_vel = np.clip(velocity, min_vel, max_vel)
        
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=norm_vel, cmap="coolwarm", s=40)
        
        # Add colorbar for velocity
        cbar = plt.colorbar(sc, ax=ax, pad=0.1)
        cbar.set_label("Velocity (m/s)", fontsize=self.font_size_label)
        cbar.ax.tick_params(labelsize=self.font_size_ticks)
        
        ax.set_xlabel(labels[0], fontsize=self.font_size_label)
        ax.set_ylabel(labels[1], fontsize=self.font_size_label)
        ax.set_title("Point Cloud (2D)", fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)

    def plot_range_detector_view(self, ax: plt.Axes, payload: Dict[str, Any], convert_to_dB: bool = False) -> None:
        """Plots the range response with overlaid thresholds and detections.

        Args:
            ax: Matplotlib Axes object to plot on.
            payload: Payload containing 'range_resp', 'thresholds', 'dets', and 'range_bins'.
            convert_to_dB: Whether to display the plot in dB.
        """
        if "range_resp" in payload:
            payload["data"] = payload["range_resp"]
            
        self.plot_range_response_view(ax, payload, convert_to_dB)
        
        thresholds = payload.get("thresholds")
        dets = payload.get("dets")
        range_bins = payload.get("range_bins")
        
        # We need to re-extract data to get the correctly converted signal for scatter plot
        data = self._prepare_data(payload, convert_to_dB)
        
        if thresholds is not None:
            thresholds = np.array(thresholds).flatten()
            x_vals = range_bins if range_bins is not None else np.arange(len(thresholds))
            display_thresh = 20 * np.log10(np.maximum(thresholds, 1e-12)) if convert_to_dB else thresholds
            ax.plot(x_vals, display_thresh, 'y--', label='Threshold', linewidth=2.5)
            
        if dets is not None and len(dets) > 0 and range_bins is not None:
            dets = np.array(dets).astype(int).flatten()
            det_ranges = range_bins[dets]
            # Plot detections at the height of the signal
            det_mags = data[dets]
            ax.scatter(det_ranges, det_mags, color='r', marker='x', s=80, label='Detections', zorder=5)
        
        ax.legend(fontsize=self.font_size_ticks)

    def plot_range_doppler_detector_view(self, ax: plt.Axes, payload: Dict[str, Any], convert_to_dB: bool = False) -> None:
        """Plots the range-Doppler response with overlaid detections.

        Args:
            ax: Matplotlib Axes object to plot on.
            payload: Payload containing 'rng_dop_resp', 'dets', 'vel_bins', and 'range_bins'.
            convert_to_dB: Whether to display the heatmap in dB.
        """
        if "rng_dop_resp" in payload:
            payload["data"] = payload["rng_dop_resp"]
            
        self.plot_range_doppler_view(ax, payload, convert_to_dB)
        
        dets = payload.get("dets")
        vel_bins = payload.get("vel_bins")
        range_bins = payload.get("range_bins")
        
        if dets is not None and len(dets) > 0 and vel_bins is not None and range_bins is not None:
            dets = np.array(dets)
            det_range_idxs = dets[:, 0].astype(int)
            det_vel_idxs = dets[:, 1].astype(int)
            
            # Ensure indices are within bounds
            valid = (det_range_idxs < len(range_bins)) & (det_vel_idxs < len(vel_bins))
            if np.any(valid):
                det_ranges = range_bins[det_range_idxs[valid]]
                det_vels = vel_bins[det_vel_idxs[valid]]
                ax.scatter(det_vels, det_ranges, color='r', marker='o', s=40, label='Detections', edgecolors='white', linewidth=0.5)
                ax.legend(fontsize=self.font_size_ticks)

    def plot_compilation(self, 
                         payloads: Dict[str, Any], 
                         view_types: List[str], 
                         convert_to_dB: bool = False,
                         output_path: Optional[str] = None, view_name_map: Optional[Dict[str, str]] = None) -> plt.Figure:
        """Generates a compilation figure with up to 4 subplots.

        Args:
            payloads: A dictionary mapping identifiers (view names or keys) to their data payloads.
            view_types: A list of identifiers (view names or keys) to include (max 4).
            convert_to_dB: Whether to display heatmaps in dB.
            output_path: Path to save the figure. If None, the figure is not saved.
            view_name_map: Optional mapping from identifier (in view_types) to view class name.
                          If provided, payloads is expected to be indexed by these identifiers.

        Returns:
            The generated Matplotlib Figure.
        """
        num_views = min(len(view_types), 4)
        if num_views == 0:
            raise ValueError("No views provided for compilation.")

        rows = 2 if num_views > 2 else 1
        cols = 2 if num_views > 1 else 1

        # Use Figure and add_subplot for mixed 2D/3D support
        fig = plt.figure(figsize=(12, 10))
        
        view_map = {
            "RangeAngleView": self.plot_range_angle_view,
            "DopplerAzimuthView": self.plot_doppler_azimuth_view,
            "RangeDopplerView": self.plot_range_doppler_view,
            "MicroDopplerView": self.plot_micro_doppler_view,
            "RangeResponseView": self.plot_range_response_view,
            "AltitudeView": self.plot_altitude_view,
            "PointCloudView": self.plot_point_cloud_view,
            "PointCloudView2D": self.plot_point_cloud_2d_view,
            "RangeDetectorView": self.plot_range_detector_view,
            "RangeDopplerDetectorView": self.plot_range_doppler_detector_view,
        }

        for i in range(num_views):
            view_id = view_types[i]
            
            # Resolve view name (backward compatibility if map is missing)
            view_name = view_name_map[view_id] if view_name_map else view_id
            
            # Check if this view needs 3D projection
            projection = '3d' if view_name == "PointCloudView" else None
            ax = fig.add_subplot(rows, cols, i + 1, projection=projection)
            
            payload = payloads.get(view_id)
            if payload and view_name in view_map:
                view_map[view_name](ax, payload, convert_to_dB)
            else:
                ax.text(0.5, 0.5, f"View {view_name} (ID: {view_id}) not found or unsupported", 
                            ha='center', va='center', transform=ax.transAxes)

        fig.tight_layout()

        if output_path:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=150)
            
        return fig
