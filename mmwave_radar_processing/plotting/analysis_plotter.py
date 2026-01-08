import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List
import pandas as pd

class AnalysisPlotter:
    """
    Plotter class for visualizing analysis results.
    """

    def __init__(self) -> None:
        """
        Initialize the AnalysisPlotter with default styling attributes.
        """
        self.font_size_axis_labels = 12
        self.font_size_title = 15
        self.font_size_ticks = 12
        self.font_size_legend = 12
        self.plot_x_max = 10
        self.plot_y_max = 20
        self.marker_size = 10

    def plot_error_distribution(
        self,
        errors: np.ndarray,
        ax: plt.Axes,
        type: str = "histogram",
        label: str = "Error",
        color: str = "blue",
        bins: int = 30
    ) -> None:
        """
        Plot the distribution of errors (Histogram or CDF).

        Args:
            errors (np.ndarray): Array of error values.
            ax (plt.Axes): Matplotlib axes object to plot on.
            type (str, optional): Type of plot "histogram" or "cdf". Defaults to "histogram".
            label (str, optional): Label for the data. Defaults to "Error".
            color (str, optional): Color of the plot. Defaults to "blue".
            bins (int, optional): Number of bins for histogram. Defaults to 30.
        """
        if type == "histogram":
            mean = np.mean(errors)
            std = np.std(errors)
            ax.hist(errors, bins=bins, color=color, alpha=0.7, label=label)
            ax.set_title(f"{label} Distribution\nMean: {mean:.3f}, Std: {std:.3f}", fontsize=self.font_size_title)
            ax.set_ylabel("Frequency", fontsize=self.font_size_axis_labels)
            ax.set_xlabel("Error Magnitude", fontsize=self.font_size_axis_labels)
        
        elif type == "cdf":
            sorted_errors = np.sort(errors)
            cdf = np.linspace(0, 1, len(sorted_errors))
            ax.plot(sorted_errors, cdf, label=label, color=color, linewidth=2)
            ax.set_title(f"{label} CDF", fontsize=self.font_size_title)
            ax.set_ylabel("CDF", fontsize=self.font_size_axis_labels)
            ax.set_xlabel("Error Magnitude", fontsize=self.font_size_axis_labels)
        
        ax.tick_params(axis='both', which='major', labelsize=self.font_size_ticks)
        ax.grid(True, alpha=0.3)
        if type == "cdf":
            ax.legend(fontsize=self.font_size_legend)

    def plot_time_series(
        self,
        data: np.ndarray,
        ax: plt.Axes,
        title: str,
        ylabel: str,
        xlabel: str = "Frame Index",
        label: Optional[str] = None,
        color: str = "blue",
        linestyle: str = "-"
    ) -> None:
        """
        Plot data as a time series.

        Args:
            data (np.ndarray): 1D array of data points.
            ax (plt.Axes): Matplotlib axes.
            title (str): Plot title.
            ylabel (str): Y-axis label.
            xlabel (str, optional): X-axis label. Defaults to "Frame Index".
            label (str, optional): Legend label. Defaults to None.
            color (str, optional): Line color. Defaults to "blue".
            linestyle (str, optional): Line style. Defaults to "-".
        """
        ax.plot(data, label=label, color=color, linestyle=linestyle)
        ax.set_title(title, fontsize=self.font_size_title)
        ax.set_xlabel(xlabel, fontsize=self.font_size_axis_labels)
        ax.set_ylabel(ylabel, fontsize=self.font_size_axis_labels)
        ax.tick_params(axis='both', which='major', labelsize=self.font_size_ticks)
        ax.grid(True, alpha=0.3)
        if label:
            ax.legend(fontsize=self.font_size_legend)

    def plot_velocity_analysis_summary(
        self,
        x_errors: np.ndarray,
        y_errors: np.ndarray,
        z_errors: np.ndarray,
        norm_errors: np.ndarray,
        show: bool = True
    ) -> None:
        """
        Generate a summary plot with time series and CDFs for all error components.

        Args:
            x_errors (np.ndarray): X velocity errors.
            y_errors (np.ndarray): Y velocity errors.
            z_errors (np.ndarray): Z velocity errors.
            norm_errors (np.ndarray): Norm velocity errors.
            show (bool, optional): Whether to display the plot. Defaults to True.
        """
        fig, axs = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1]})
        
        # Time Series
        self.plot_time_series(x_errors, axs[0], "", "Error (m/s)", color="red", label="X Error")
        self.plot_time_series(y_errors, axs[0], "", "Error (m/s)", color="green", label="Y Error")
        self.plot_time_series(z_errors, axs[0], "", "Error (m/s)", color="blue", label="Z Error")
        self.plot_time_series(norm_errors, axs[0], "Velocity Estimation Errors Over Time", "Error (m/s)", color="purple", linestyle="--", label="Norm Error")
        
        # CDFs
        # Need to re-plot on same axis, so we call plot_error_distribution multiple times
        for errors, label, color in zip(
            [x_errors, y_errors, z_errors, norm_errors],
            ["X Error", "Y Error", "Z Error", "Norm Error"],
            ["red", "green", "blue", "purple"]
        ):
             # We can't reuse the title setting logic inside plot_error_distribution perfectly if we want a combined title
             # So let's manually set title after
             self.plot_error_distribution(errors, axs[1], type="cdf", label=label, color=color)
        
        axs[1].set_title("Cumulative Distribution Function (CDF) of Errors", fontsize=self.font_size_title)
        
        plt.tight_layout()
        if show:
            plt.show()

    def plot_comparison_time_series(
        self,
        estimated: np.ndarray,
        ground_truth: np.ndarray,
        ax: plt.Axes,
        title: str,
        ylabel: str,
        xlabel: str = "Frame Index",
        est_label: str = "Estimated",
        gt_label: str = "Ground Truth",
        est_color: str = "blue",
        gt_color: str = "black",
        linestyle: str = "-"
    ) -> None:
        """
        Plot both estimated and ground truth data on the same axes for comparison.

        Args:
            estimated (np.ndarray): 1D array of estimated values.
            ground_truth (np.ndarray): 1D array of ground truth values.
            ax (plt.Axes): Matplotlib axes.
            title (str): Plot title.
            ylabel (str): Y-axis label.
            xlabel (str, optional): X-axis label. Defaults to "Frame Index".
            est_label (str, optional): Legend label for estimated. Defaults to "Estimated".
            gt_label (str, optional): Legend label for ground truth. Defaults to "Ground Truth".
            est_color (str, optional): Color for estimated line. Defaults to "blue".
            gt_color (str, optional): Color for ground truth line. Defaults to "black".
            linestyle (str, optional): Line style. Defaults to "-".
        """
        ax.plot(estimated, label=est_label, color=est_color, linestyle=linestyle)
        ax.plot(ground_truth, label=gt_label, color=gt_color, linestyle=linestyle, alpha=0.6)
        
        ax.set_title(title, fontsize=self.font_size_title)
        ax.set_xlabel(xlabel, fontsize=self.font_size_axis_labels)
        ax.set_ylabel(ylabel, fontsize=self.font_size_axis_labels)
        ax.tick_params(axis='both', which='major', labelsize=self.font_size_ticks)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=self.font_size_legend)

    def plot_error_histograms(
        self,
        x_errors: np.ndarray,
        y_errors: np.ndarray,
        z_errors: np.ndarray,
        bins: int = 30,
        show: bool = True
    ) -> None:
        """
        Plot histograms for X, Y, Z velocity errors.

        Args:
            x_errors (np.ndarray): X error array.
            y_errors (np.ndarray): Y error array.
            z_errors (np.ndarray): Z error array.
            bins (int, optional): Number of bins. Defaults to 30.
            show (bool, optional): Whether to display the plot. Defaults to True.
        """
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        
        for i, (errors, label, color) in enumerate(zip(
            [x_errors, y_errors, z_errors],
            ["X Velocity Error", "Y Velocity Error", "Z Velocity Error"],
            ["red", "green", "blue"]
        )):
            self.plot_error_distribution(
                errors, 
                axs[i], 
                type="histogram", 
                label=label, 
                color=color, 
                bins=bins
            )
            axs[i].set_title(f"{label} Distribution\nMean: {np.mean(errors):.4f}, Std: {np.std(errors):.4f}", fontsize=self.font_size_title)

        plt.tight_layout()
        if show:
            plt.show()

