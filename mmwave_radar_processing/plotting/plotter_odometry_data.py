import matplotlib.pyplot as plt
import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors import altimeter
from mmwave_radar_processing.processors.range_resp import RangeProcessor
from mmwave_radar_processing.processors.altimeter import Altimeter
from mmwave_radar_processing.processors.velocity_estimator import VelocityEstimator
from mmwave_radar_processing.plotting.plotter_mmWave_data import PlotterMmWaveData

class PlotterOdometryData(PlotterMmWaveData):
    def __init__(self, config_manager: ConfigManager) -> None:
        """
        Args:
            config_manager (ConfigManager): Radar configuration manager.
        """
        super().__init__(config_manager)

    def plot_compilation(
            self,
            adc_cube:np.ndarray,
            altimeter:Altimeter,
            velocity_estimator:VelocityEstimator,
            camera_view:np.ndarray=np.empty(shape=(0)),
            convert_to_dB=False,
            cmap="viridis",
            axs:plt.Axes=[],
            show=False
    ):
        """
        Stub for odometry data compilation plot.
        adc_cube is assumed to have been processed by a virtual array processor if needed

        Args:
            adc_cube (np.ndarray): Input ADC cube.
            altimeter (Altimeter): Altimeter processor.
            velocity_estimator (VelocityEstimator): Velocity estimator processor.
            camera_view (np.ndarray, optional): Camera view data to plot. Defaults to empty array.
            convert_to_dB (bool, optional): Whether to convert to dB. Defaults to False.
            cmap (str, optional): Colormap for plotting. Defaults to "viridis".
            chirp_idx (int, optional): Chirp index. Defaults to 0.
            rx_antenna_idx (int, optional): Rx antenna index. Defaults to 0.
            axs (plt.Axes, optional): Axes for plotting.
            show (bool, optional): Whether to display the plot.
        """
        
        if len(axs) == 0:
            fig,axs=plt.subplots(2,3, figsize=(15,10))
            fig.subplots_adjust(wspace=0.3,hspace=0.30)


        #plot altitude measurements
        if altimeter:

            #coarse FFT
            coarse_fft = altimeter.coarse_fft(
                adc_cube,
                chirp_idx=0)
            peak_rng_bins, peak_vals = altimeter.find_peaks(
                rng_resp_db=20*np.log10(coarse_fft),
                rng_bins=altimeter.range_bins,
                max_peaks=3
            )
            self.plot_range_resp(
                resp=coarse_fft,
                rng_bins=altimeter.range_bins,
                peak_rng_bins=peak_rng_bins,
                peak_vals=peak_vals,
                ax=axs[0,0],
                convert_to_dB=convert_to_dB,
                show=False
            )

            #zoom FFT
            altimeter.process(adc_cube, precise_est_enabled=True)
            est_altitude = altimeter.current_altitude_measured_m

            range_start_m = max(0, est_altitude - altimeter.zoom_search_region_m)
            range_end_m = min(self.config_manager.range_max_m, est_altitude + altimeter.zoom_search_region_m)  
            zoom_avg, zoom_range_bins = altimeter.zoom_fft(
                adc_cube=adc_cube,
                range_start_m=range_start_m,
                range_stop_m=range_end_m,
                chirp_idx=0
            )

            peak_rng_bins, peak_vals = altimeter.find_peaks(
                rng_resp_db=20*np.log10(zoom_avg),
                rng_bins=zoom_range_bins,
                max_peaks=1
            )
            self.plot_range_resp(
                resp=zoom_avg,
                rng_bins=zoom_range_bins,
                peak_rng_bins=peak_rng_bins,
                peak_vals=peak_vals,
                ax=axs[0,1],
                convert_to_dB=convert_to_dB,
                show=False
            )
            if len(altimeter.history_estimated) > 0:
                altitude_est_history = np.concatenate(altimeter.history_estimated, axis=0)
                if len(altimeter.history_gt) > 0:
                    altitude_gt_history = np.concatenate(altimeter.history_gt, axis=0)
                else:
                    altitude_gt_history = np.array([])
                
                #plot estimated altitude vs ground truth
                self.plot_estimated_vs_ground_truth(
                    estimated=altitude_est_history,
                    ground_truth=altitude_gt_history,
                    ax=axs[0,2],
                    value_label="Altitude (m)",
                    show=False
                )

                #plot estimated altitude error
                self.plot_estimated_vs_ground_truth_error(
                    estimated=altitude_est_history,
                    ground_truth=altitude_gt_history,
                    ax=axs[1,2],
                    value_label="Altitude Error (m)",
                    show=False
                )
        else:
            est_altitude = 0.0

        #TODO: Add code to plot the error

        #plot velocity measurements
        if velocity_estimator:

            #compute responses
            velocity_estimator.process(
                adc_cube=adc_cube,
                altitude=est_altitude,
                enable_precise_responses=True
            )

            self.plot_doppler_az_resp(
                resp=velocity_estimator.azimuth_response_mag,
                doppler_azimuth_processor=velocity_estimator,
                convert_to_dB=convert_to_dB,
                cmap=cmap,
                ax=axs[1,0],
                show=False
            )
            axs[1,0].set_title("Doppler-Azimuth\nHeatmap", fontsize=self.font_size_title)

            self.plot_doppler_az_resp(
                resp=velocity_estimator.elevation_response_mag,
                doppler_azimuth_processor=velocity_estimator,
                convert_to_dB=convert_to_dB,
                cmap=cmap,
                ax=axs[1,1],
                show=False
            )
            axs[1,1].set_title("Doppler-Elevation\nHeatmap", fontsize=self.font_size_title)

            self.plot_zoomed_doppler_az_resp(
                resp=velocity_estimator.precise_azimuth_response_mag,
                doppler_azimuth_processor=velocity_estimator,
                peaks=velocity_estimator.azimuth_peaks,
                vd_ground_truth=velocity_estimator.get_gt_velocity_measurement_predictions(direction="azimuth"),
                vd_estimated=velocity_estimator.get_estimated_velocity_measurement_predictions(direction="azimuth"),
                convert_to_dB=convert_to_dB,
                cmap=cmap,
                ax=axs[2,0],
                show=False
            )
            axs[2,0].set_title("Zoomed Doppler-Azimuth\nHeatmap", fontsize=self.font_size_title)

            self.plot_zoomed_doppler_az_resp(
                resp=velocity_estimator.precise_elevation_response_mag,
                doppler_azimuth_processor=velocity_estimator,
                peaks=velocity_estimator.elevation_peaks,
                vd_ground_truth=velocity_estimator.get_gt_velocity_measurement_predictions(direction="elevation"),
                vd_estimated=velocity_estimator.get_estimated_velocity_measurement_predictions(direction="elevation"),
                convert_to_dB=convert_to_dB,
                cmap=cmap,
                ax=axs[2,1],
                show=False
            )
            axs[2,1].set_title("Zoomed Doppler-Elevation\nHeatmap", fontsize=self.font_size_title)

            if len(velocity_estimator.history_gt) > 0:
                latest_gt_vel = velocity_estimator.history_gt[-1]
                latest_est_vel = velocity_estimator.estimated_velocity
                #TODO: These need to be flipped, but I don't know why
                axs[1,0].set_title(f"Doppler-Azimuth (x)\nHeatmap (GT:{latest_gt_vel[1]:.2f})", fontsize=self.font_size_title)
                axs[1,1].set_title(f"Doppler-Elevation (y)\nHeatmap (GT:{latest_gt_vel[0]:.2f})", fontsize=self.font_size_title)

                axs[2,0].set_title(f"Zoomed Doppler-Azimuth (x)\nHeatmap (GT:{latest_gt_vel[1]:.2f}, EST:{latest_est_vel[0]:.2f})", fontsize=self.font_size_title)
                axs[2,1].set_title(f"Zoomed Doppler-Elevation (y)\nHeatmap (GT:{latest_gt_vel[0]:.2f}, EST:{latest_est_vel[1]:.2f})", fontsize=self.font_size_title)

        if show:
            plt.show()