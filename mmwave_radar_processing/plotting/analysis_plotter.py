import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, List, Tuple
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
        ax: Optional[plt.Axes] = None,
        type: str = "histogram",
        label: str = "Error",
        color: str = "blue",
        bins: int = 30,
        show: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the distribution of errors (Histogram or CDF).

        Args:
            errors (np.ndarray): Array of error values.
            ax (plt.Axes, optional): Matplotlib axes object to plot on. If None, a new figure is created. Defaults to None.
            type (str, optional): Type of plot "histogram" or "cdf". Defaults to "histogram".
            label (str, optional): Label for the data. Defaults to "Error".
            color (str, optional): Color of the plot. Defaults to "blue".
            bins (int, optional): Number of bins for histogram. Defaults to 30.
            show (bool, optional): Whether to display the plot. Defaults to True.
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: The figure and axes used for plotting.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.get_figure()

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

        if show:
            plt.show()
            
        return fig, ax

    def plot_time_series(
        self,
        data: np.ndarray,
        ax: Optional[plt.Axes] = None,
        title: str = "",
        ylabel: str = "",
        xlabel: str = "Frame Index",
        label: Optional[str] = None,
        color: str = "blue",
        linestyle: str = "-",
        show: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot data as a time series.

        Args:
            data (np.ndarray): 1D array of data points.
            ax (plt.Axes, optional): Matplotlib axes. If None, a new figure is created. Defaults to None.
            title (str, optional): Plot title. Defaults to "".
            ylabel (str, optional): Y-axis label. Defaults to "".
            xlabel (str, optional): X-axis label. Defaults to "Frame Index".
            label (str, optional): Legend label. Defaults to None.
            color (str, optional): Line color. Defaults to "blue".
            linestyle (str, optional): Line style. Defaults to "-".
            show (bool, optional): Whether to display the plot. Defaults to True.
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: The figure and axes used for plotting.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.get_figure()

        ax.plot(data, label=label, color=color, linestyle=linestyle)
        if title:
            ax.set_title(title, fontsize=self.font_size_title)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.font_size_axis_labels)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.font_size_axis_labels)
        ax.tick_params(axis='both', which='major', labelsize=self.font_size_ticks)
        ax.grid(True, alpha=0.3)
        if label:
            ax.legend(fontsize=self.font_size_legend)
            
        if show:
            plt.show()

        return fig, ax

    def plot_velocity_analysis_summary(
        self,
        x_errors: np.ndarray,
        y_errors: np.ndarray,
        z_errors: np.ndarray,
        norm_errors: np.ndarray,
        axs: Optional[np.ndarray] = None,
        show: bool = True
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Generate a summary plot with time series and CDFs for all error components.

        Args:
            x_errors (np.ndarray): X velocity errors.
            y_errors (np.ndarray): Y velocity errors.
            z_errors (np.ndarray): Z velocity errors.
            norm_errors (np.ndarray): Norm velocity errors.
            axs (np.ndarray, optional): Array of Matplotlib axes to plot on. Expected shape is (2,). 
                                        If None, a new figure is created. Defaults to None.
            show (bool, optional): Whether to display the plot. Defaults to True.
            
        Returns:
            Tuple[plt.Figure, np.ndarray]: The figure and axes array used for plotting.
        """
        if axs is None:
            fig, axs = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1]})
        else:
            fig = axs[0].get_figure()
        
        # Time Series
        self.plot_time_series(x_errors, ax=axs[0], ylabel="Error (m/s)", color="red", label="X Error", show=False)
        self.plot_time_series(y_errors, ax=axs[0], ylabel="Error (m/s)", color="green", label="Y Error", show=False)
        self.plot_time_series(z_errors, ax=axs[0], ylabel="Error (m/s)", color="blue", label="Z Error", show=False)
        self.plot_time_series(norm_errors, ax=axs[0], title="Velocity Estimation Errors Over Time", ylabel="Error (m/s)", color="purple", linestyle="--", label="Norm Error", show=False)
        
        # CDFs
        for errors, label, color in zip(
            [x_errors, y_errors, z_errors, norm_errors],
            ["X Error", "Y Error", "Z Error", "Norm Error"],
            ["red", "green", "blue", "purple"]
        ):
             self.plot_error_distribution(errors, ax=axs[1], type="cdf", label=label, color=color, show=False)
        
        axs[1].set_title("Cumulative Distribution Function (CDF) of Errors", fontsize=self.font_size_title)
        
        plt.tight_layout()
        if show:
            plt.show()
            
        return fig, axs

    def plot_comparison_time_series(
        self,
        estimated: np.ndarray,
        ground_truth: np.ndarray,
        ax: Optional[plt.Axes] = None,
        title: str = "",
        ylabel: str = "",
        xlabel: str = "Frame Index",
        est_label: str = "Estimated",
        gt_label: str = "Ground Truth",
        est_color: str = "blue",
        gt_color: str = "black",
        linestyle: str = "-",
        show: bool = True
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot both estimated and ground truth data on the same axes for comparison.

        Args:
            estimated (np.ndarray): 1D array of estimated values.
            ground_truth (np.ndarray): 1D array of ground truth values.
            ax (plt.Axes, optional): Matplotlib axes. If None, a new figure is created. Defaults to None.
            title (str, optional): Plot title. Defaults to "".
            ylabel (str, optional): Y-axis label. Defaults to "".
            xlabel (str, optional): X-axis label. Defaults to "Frame Index".
            est_label (str, optional): Legend label for estimated. Defaults to "Estimated".
            gt_label (str, optional): Legend label for ground truth. Defaults to "Ground Truth".
            est_color (str, optional): Color for estimated line. Defaults to "blue".
            gt_color (str, optional): Color for ground truth line. Defaults to "black".
            linestyle (str, optional): Line style. Defaults to "-".
            show (bool, optional): Whether to display the plot. Defaults to True.
            
        Returns:
            Tuple[plt.Figure, plt.Axes]: The figure and axes used for plotting.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.get_figure()

        ax.plot(estimated, label=est_label, color=est_color, linestyle=linestyle)
        ax.plot(ground_truth, label=gt_label, color=gt_color, linestyle=linestyle, alpha=0.6)
        
        if title:
            ax.set_title(title, fontsize=self.font_size_title)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.font_size_axis_labels)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.font_size_axis_labels)
        ax.tick_params(axis='both', which='major', labelsize=self.font_size_ticks)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=self.font_size_legend)

        if show:
            plt.show()
            
        return fig, ax

    def plot_error_histograms(
        self,
        x_errors: np.ndarray,
        y_errors: np.ndarray,
        z_errors: np.ndarray,
        axs: Optional[np.ndarray] = None,
        bins: int = 30,
        show: bool = True
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Plot histograms for X, Y, Z velocity errors.

        Args:
            x_errors (np.ndarray): X error array.
            y_errors (np.ndarray): Y error array.
            z_errors (np.ndarray): Z error array.
            axs (np.ndarray, optional): Array of Matplotlib axes to plot on. Expected shape is (3,).
                                        If None, a new figure is created. Defaults to None.
            bins (int, optional): Number of bins. Defaults to 30.
            show (bool, optional): Whether to display the plot. Defaults to True.
            
        Returns:
            Tuple[plt.Figure, np.ndarray]: The figure and axes array used for plotting.
        """
        if axs is None:
            fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        else:
            fig = axs[0].get_figure()
        
        for i, (errors, label, color) in enumerate(zip(
            [x_errors, y_errors, z_errors],
            ["X Velocity Error", "Y Velocity Error", "Z Velocity Error"],
            ["red", "green", "blue"]
        )):
            self.plot_error_distribution(
                errors, 
                ax=axs[i], 
                type="histogram", 
                label=label, 
                color=color, 
                bins=bins,
                show=False
            )
            axs[i].set_title(f"{label} Distribution\nMean: {np.mean(errors):.4f}, Std: {np.std(errors):.4f}", fontsize=self.font_size_title)

        plt.tight_layout()
        if show:
            plt.show()

        return fig, axs

