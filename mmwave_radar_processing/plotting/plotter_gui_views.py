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
            ax.plot(range_bins, data)
        else:
            ax.plot(data)

        ax.set_xlabel("Range (m)", fontsize=self.font_size_label)
        ax.set_ylabel("Amplitude (dB)" if convert_to_dB else "Amplitude (mag)", fontsize=self.font_size_label)
        ax.set_title("Range Profile", fontsize=self.font_size_title)
        ax.tick_params(labelsize=self.font_size_ticks)
        ax.grid(True)

    def plot_compilation(self, 
                         payloads: Dict[str, Any], 
                         view_types: List[str], 
                         convert_to_dB: bool = False,
                         output_path: Optional[str] = None) -> plt.Figure:
        """Generates a compilation figure with up to 4 subplots.

        Args:
            payloads: A dictionary mapping view class names to their data payloads.
            view_types: A list of view class names to include (max 4).
            convert_to_dB: Whether to display heatmaps in dB.
            output_path: Path to save the figure. If None, the figure is not saved.

        Returns:
            The generated Matplotlib Figure.
        """
        num_views = min(len(view_types), 4)
        if num_views == 0:
            raise ValueError("No views provided for compilation.")

        rows = 2 if num_views > 2 else 1
        cols = 2 if num_views > 1 else 1

        fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
        if num_views == 1:
            axs = np.array([axs])
        axs = axs.flatten()

        view_map = {
            "RangeAngleView": self.plot_range_angle_view,
            "DopplerAzimuthView": self.plot_doppler_azimuth_view,
            "RangeDopplerView": self.plot_range_doppler_view,
            "MicroDopplerView": self.plot_micro_doppler_view,
            "RangeResponseView": self.plot_range_response_view,
            # Add more views as needed
        }

        for i in range(num_views):
            view_name = view_types[i]
            payload = payloads.get(view_name)
            if payload and view_name in view_map:
                view_map[view_name](axs[i], payload, convert_to_dB)
            else:
                axs[i].text(0.5, 0.5, f"View {view_name} not found or unsupported", 
                            ha='center', va='center', transform=axs[i].transAxes)

        # Hide unused axes
        for i in range(num_views, len(axs)):
            axs[i].axis('off')

        fig.tight_layout()

        if output_path:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path)
            
        return fig
