import matplotlib.pyplot as plt
import numpy as np

from mmwave_radar_processing.config_managers.cfgManager import ConfigManager
from mmwave_radar_processing.processors.synthetic_array_beamformer_processor_revA import SyntheticArrayBeamformerProcessor
from mmwave_radar_processing.supportFns.rotation_functions import apply_rot_trans
from mmwave_radar_processing.processors.strip_map_SAR_processor import StripMapSARProcessor

class PlotterSyntheticArrayData:

    def __init__(self,
                 config_manager:ConfigManager,
                 processor_SABF:SyntheticArrayBeamformerProcessor,
                 processor_stripMapSAR:StripMapSARProcessor=None,
                 min_vel = 0.4) -> None:
        
        #define default plot parameters:
        self.font_size_axis_labels = 12
        self.font_size_title = 15
        self.font_size_ticks = 12
        self.font_size_legend = 12
        self.plot_x_max = 10
        self.plot_y_max = 20
        self.marker_size = 10
        self.min_threshold_dB = 40

        #configuration manager
        self.config_manager:ConfigManager = config_manager

        self.min_vel = min_vel

        #synthetic array processor
        self.processor_SABF:SyntheticArrayBeamformerProcessor =\
            processor_SABF
        #[optional]
        self.processor_stripMapSAR:StripMapSARProcessor = processor_stripMapSAR
    
    ####################################################################
    #Plotting the array geometry
    #################################################################### 

    def plot_synthetic_array_geometry(
        self,
        vels,
        ax:plt.Axes=None,
        show=False
    ):
        if self.processor_SABF.mode == SyntheticArrayBeamformerProcessor.BROADSIDE_MODE:
            self.plot_synthetic_array_geometry_broadside(
                vels=vels,
                ax=ax,
                show=show
            )
        elif self.processor_SABF.mode == SyntheticArrayBeamformerProcessor.ENDFIRE_MODE:
            self.plot_synthetic_array_geometry_endfire(
                vels=vels,
                ax=ax,
                show=show
            )

    def plot_synthetic_array_geometry_broadside(
        self,
        vels,
        ax:plt.Axes=None,
        show=False
    ):

        #compute the geometries
        self.processor_SABF.generate_array_geometries(vels)

        if not ax:
            fig,ax = plt.subplots()

        #only plot if array geometry is valid
        if self.processor_SABF.array_geometry_valid:

            x_coords = np.reshape(
                self.processor_SABF.p_x_m,
                newshape=(-1,1),
                order='F'
            )
            z_coords = np.reshape(
                self.processor_SABF.p_z_m,
                newshape=(-1,1),
                order='F'
            )

            ax.scatter(x_coords,z_coords)
            
            if vels[0] > 0:
                ax.set_xlim(
                    left=-1 * self.processor_SABF.lambda_m/8,
                    right=self.processor_SABF.p_x_m[0,5] + \
                        self.processor_SABF.lambda_m/8
                )
                ax.set_xticks(
                    ticks=np.linspace(
                        start=-1 * self.processor_SABF.lambda_m/8,
                        stop=self.processor_SABF.p_x_m[0,5] + \
                        self.processor_SABF.lambda_m/8,
                        num=3
                    )
                )
            else:
                ax.set_xlim(
                    right=1 * self.processor_SABF.lambda_m/8,
                    left=self.processor_SABF.p_x_m[0,5] - \
                        self.processor_SABF.lambda_m/8
                )
                ax.set_xticks(
                    ticks=np.linspace(
                        start=1 * self.processor_SABF.lambda_m/8,
                        stop=self.processor_SABF.p_x_m[0,5] - \
                        self.processor_SABF.lambda_m/8,
                        num=4
                    )
                )
            ax.set_ylim(
                bottom=-1 * self.processor_SABF.lambda_m/8,
                top = self.processor_SABF.p_z_m[-1,1] + \
                    self.processor_SABF.lambda_m/8
            )
        
        ax.set_xlabel("X (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("Z (m)",fontsize=self.font_size_axis_labels)

        formatted_vels = np.array2string(
            vels,
            formatter={'float_kind': lambda x: f"{x:.2f}"},
            separator=', '
        )
        title_str = "Synth Array: vel ({} m/s)\n, {} chirps".format(
            formatted_vels,
            self.processor_SABF.chirps_per_frame
        )
        ax.set_title(title_str,
                    fontsize=self.font_size_title)

        if show:
            plt.show()

    def plot_synthetic_array_geometry_endfire(
        self,
        vels,
        ax:plt.Axes=None,
        show=False
    ):  
        if not ax:
                fig,ax = plt.subplots()

        #compute the geometries
        self.processor_SABF.generate_array_geometries(vels)
        if self.processor_SABF.array_geometry_valid:
            
            x_coords = np.reshape(
                self.processor_SABF.p_x_m,
                newshape=(-1,1),
                order='F'
            )
            y_coords = np.reshape(
                self.processor_SABF.p_y_m,
                newshape=(-1,1),
                order='F'
            )

            ax.scatter(y_coords,x_coords)

            if vels[0] > 0:
                ax.set_ylim(
                    bottom=-1 * self.processor_SABF.lambda_m/8,
                    top=self.processor_SABF.p_x_m[0,5] + \
                        self.processor_SABF.lambda_m/8
                )
                ax.set_yticks(
                    ticks=np.linspace(
                        start=-1 * self.processor_SABF.lambda_m/8,
                        stop=self.processor_SABF.p_x_m[0,5] + \
                        self.processor_SABF.lambda_m/8,
                        num=3
                    )
                )
            else:
                ax.set_ylim(
                    bottom=1 * self.processor_SABF.lambda_m/8,
                    top=self.processor_SABF.p_x_m[0,5] - \
                        self.processor_SABF.lambda_m/8
                )
                ax.set_yticks(
                    ticks=np.linspace(
                        start=1 * self.processor_SABF.lambda_m/8,
                        stop=self.processor_SABF.p_x_m[0,5] - \
                        self.processor_SABF.lambda_m/8,
                        num=4
                    )
                )
            ax.set_xlim(
                right=-1 * self.processor_SABF.lambda_m/8,
                left = self.processor_SABF.p_y_m[-1,1] + \
                    self.processor_SABF.lambda_m/8
            )

        ax.set_xlabel("Y (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("X (m)",fontsize=self.font_size_axis_labels)

        formatted_vels = np.array2string(
            vels[0:2],
            formatter={'float_kind': lambda x: f"{x:.2f}"},
            separator=', '
        )
        title_str = "Dynamically computed array:\nvelocity: {} m/s".format(
            formatted_vels
        )
        ax.set_title(title_str,
                    fontsize=self.font_size_title)

        if show:
            plt.show()

    ####################################################################
    #Plotting beamformed response
    #################################################################### 

    def plot_2D_az_beamformed_resp_slice(
            self,
            resp:np.ndarray,
            convert_to_dB=False,
            cmap="viridis",
            ax:plt.Axes=None,
            show=False
    ):
        """Plot a 2D slice of the azimuth beamformed response at 0-deg elevation

        Args:
            resp (np.ndarray): a 2D beamformed response indexed by 
                [range bin, az angle (theta), el angle (phi)]
            convert_to_dB (bool, optional): on True, converts the response to a 
                log scale. Defaults to False.
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if self.processor_SABF.mode == SyntheticArrayBeamformerProcessor.BROADSIDE_MODE:
            self.plot_2D_az_beamformed_resp_slice_broadside(
                resp=resp,
                convert_to_dB=convert_to_dB,
                cmap=cmap,
                ax=ax,
                show=show
            )
        elif self.processor_SABF.mode == SyntheticArrayBeamformerProcessor.ENDFIRE_MODE:
            self.plot_2D_az_beamformed_resp_slice_endfire(
                resp=resp,
                convert_to_dB=convert_to_dB,
                cmap=cmap,
                ax=ax,
                show=show
            )
    
    def plot_2D_az_beamformed_resp_slice_broadside(
            self,
            resp:np.ndarray,
            convert_to_dB=False,
            cmap="viridis",
            ax:plt.Axes=None,
            show=False
    ):
        """Plot a 2D slice of the azimuth beamformed response at 0-deg elevation

        Args:
            resp (np.ndarray): a 2D beamformed response indexed by 
                [range bin, az angle (theta), el angle (phi)]
            convert_to_dB (bool, optional): on True, converts the response to a 
                log scale. Defaults to False.
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()
        
        if convert_to_dB:
            resp = 20 * np.log10(np.abs(resp))
            #remove anything below the min_threshold dB down
            thresholded_val = np.max(resp) - self.min_threshold_dB
            idxs = resp <= thresholded_val
            resp[idxs] = thresholded_val
        else:
            resp = np.abs(resp)
        
        #identify the elevation bin closest to 0
        diffs = np.abs(self.processor_SABF.el_angle_bins_rad)
        idx = np.argmin(diffs)
        
        x_s = self.processor_SABF.x_s[:,:,idx]
        y_s = self.processor_SABF.y_s[:,:,idx]

        resp = resp[:,:,idx]
        
        ax.pcolormesh(
            x_s,
            y_s,
            resp,
            shading='gouraud',
            cmap=cmap
        )
        
        ax.set_xlabel("X (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("Y (m)",fontsize=self.font_size_axis_labels)
        if convert_to_dB:
            ax.set_title("Az Beamformed response (dB)",fontsize=self.font_size_title)
        else:
            ax.set_title("Az Beamformed response (mag)",fontsize=self.font_size_title)

        if show:
            plt.show()
    
    def plot_2D_az_beamformed_resp_slice_endfire(
            self,
            resp:np.ndarray,
            convert_to_dB=False,
            cmap="viridis",
            ax:plt.Axes=None,
            show=False
    ):
        """Plot a 2D slice of the azimuth beamformed response at 0-deg elevation

        Args:
            resp (np.ndarray): a 2D beamformed response indexed by 
                [range bin, az angle (theta), el angle (phi)]
            convert_to_dB (bool, optional): on True, converts the response to a 
                log scale. Defaults to False.
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()
        
        if convert_to_dB:
            resp = 20 * np.log10(np.abs(resp))
            #remove anything below the min_threshold dB down
            thresholded_val = np.max(resp) - self.min_threshold_dB
            idxs = resp <= thresholded_val
            resp[idxs] = thresholded_val
        else:
            resp = np.abs(resp)
        
        #identify the elevation bin closest to 0
        diffs = np.abs(self.processor_SABF.el_angle_bins_rad)
        idx = np.argmin(diffs)
        
        x_s = self.processor_SABF.x_s[:,:,idx]
        y_s = self.processor_SABF.y_s[:,:,idx]

        resp = resp[:,:,idx]
        
        ax.pcolormesh(
            y_s,
            x_s,
            resp,
            shading='gouraud',
            cmap=cmap
        )
        
        ax.set_xlabel("Y (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("X (m)",fontsize=self.font_size_axis_labels)
        ax.invert_xaxis()
        if convert_to_dB:
            ax.set_title("Az Beamformed response (dB)",fontsize=self.font_size_title)
        else:
            ax.set_title("Az Beamformed response (mag)",fontsize=self.font_size_title)

        if show:
            plt.show()
    
    def plot_2D_el_beamformed_resp_slice(
            self,
            resp:np.ndarray,
            convert_to_dB=False,
            cmap="viridis",
            ax:plt.Axes=None,
            show=False
    ):
        """Plot a 2D slice of the elevation beamformed response at 0-deg az

        Args:
            resp (np.ndarray): a 2D beamformed response indexed by 
                [range bin, az angle (theta), el angle (phi)]
            convert_to_dB (bool, optional): on True, converts the response to a 
                log scale. Defaults to False.
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if self.processor_SABF.mode == SyntheticArrayBeamformerProcessor.BROADSIDE_MODE:
            self.plot_2D_el_beamformed_resp_slice_broadside(
                resp=resp,
                convert_to_dB=convert_to_dB,
                cmap=cmap,
                ax=ax,
                show=show
            )
        if self.processor_SABF.mode == SyntheticArrayBeamformerProcessor.ENDFIRE_MODE:
            self.plot_2D_el_beamformed_resp_slice_endfire(
                resp=resp,
                convert_to_dB=convert_to_dB,
                cmap=cmap,
                ax=ax,
                show=show
            )

    def plot_2D_el_beamformed_resp_slice_broadside(
            self,
            resp:np.ndarray,
            convert_to_dB=False,
            cmap="viridis",
            ax:plt.Axes=None,
            show=False
    ):
        """Plot a 2D slice of the elevation beamformed response at 0-deg az

        Args:
            resp (np.ndarray): a 2D beamformed response indexed by 
                [range bin, az angle (theta), el angle (phi)]
            convert_to_dB (bool, optional): on True, converts the response to a 
                log scale. Defaults to False.
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()
        
        if convert_to_dB:
            resp = 20 * np.log10(np.abs(resp))
            #remove anything below the min_threshold dB down
            thresholded_val = np.max(resp) - self.min_threshold_dB
            idxs = resp <= thresholded_val
            resp[idxs] = thresholded_val
        else:
            resp = np.abs(resp)
        
        #identify the elevation bin closest to 0
        diffs = np.abs(self.processor_SABF.az_angle_bins_rad)
        idx = np.argmin(diffs)
        
        y_s = self.processor_SABF.y_s[:,idx,:]
        z_s = self.processor_SABF.z_s[:,idx,:]

        resp = resp[:,idx,:]
        
        img = ax.pcolormesh(
            y_s,
            z_s,
            resp,
            shading='gouraud',
            cmap=cmap
        )
        
        ax.set_xlabel("Y (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("Z (m)",fontsize=self.font_size_axis_labels)
        if convert_to_dB:
            ax.set_title("El Beamformed response (dB)",fontsize=self.font_size_title)
        else:
            ax.set_title("El Beamformed response (mag)",fontsize=self.font_size_title)

        if show:
            plt.show()
    
    def plot_2D_el_beamformed_resp_slice_endfire(
            self,
            resp:np.ndarray,
            convert_to_dB=False,
            cmap="viridis",
            ax:plt.Axes=None,
            show=False
    ):
        """Plot a 2D slice of the elevation beamformed response at 0-deg az

        Args:
            resp (np.ndarray): a 2D beamformed response indexed by 
                [range bin, az angle (theta), el angle (phi)]
            convert_to_dB (bool, optional): on True, converts the response to a 
                log scale. Defaults to False.
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()
        
        if convert_to_dB:
            resp = 20 * np.log10(np.abs(resp))
            #remove anything below the min_threshold dB down
            thresholded_val = np.max(resp) - self.min_threshold_dB
            idxs = resp <= thresholded_val
            resp[idxs] = thresholded_val
        else:
            resp = np.abs(resp)
        
        #identify the elevation bin closest to 0
        diffs = np.abs(self.processor_SABF.az_angle_bins_rad)
        idx = np.argmin(diffs)
        
        x_s = self.processor_SABF.x_s[:,idx,:]
        z_s = self.processor_SABF.z_s[:,idx,:]

        resp = resp[:,idx,:]
        
        img = ax.pcolormesh(
            x_s,
            z_s,
            resp,
            shading='gouraud',
            cmap=cmap
        )
        
        ax.set_xlabel("X (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("Z (m)",fontsize=self.font_size_axis_labels)
        if convert_to_dB:
            ax.set_title("El Beamformed response (dB)",fontsize=self.font_size_title)
        else:
            ax.set_title("El Beamformed response (mag)",fontsize=self.font_size_title)

        if show:
            plt.show()

    ####################################################################
    #Plotting stripmap sar responses
    #################################################################### 
    def plot_stripmap_SAR_resp(
            self,
            resp:np.ndarray,
            convert_to_dB=False,
            cmap="viridis",
            ax:plt.Axes=None,
            show=False
    ):
        """Plot a stripMap SAR response (preliminary)

        Args:
            resp (np.ndarray): a 2D beamformed response indexed by 
                [range bin, az angle (theta), el angle (phi)]
            convert_to_dB (bool, optional): on True, converts the response to a 
                log scale. Defaults to False.
            cmap (str, optional): the color map used for the generated plot
                (gray is another option). Defaults to "viridis".
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()
        
        if convert_to_dB:
            resp = 20 * np.log10(np.abs(resp))
            #remove anything below the min_threshold dB down
            thresholded_val = np.max(resp) - self.min_threshold_dB
            idxs = resp <= thresholded_val
            resp[idxs] = thresholded_val
        else:
            resp = np.abs(resp)
        
        x_s = self.processor_stripMapSAR.x_s
        y_s = self.processor_stripMapSAR.y_s
        
        ax.pcolormesh(
            x_s,
            y_s,
            resp,
            shading='gouraud',
            cmap=cmap
        )
        
        ax.set_xlabel("X (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("Y (m)",fontsize=self.font_size_axis_labels)
        if convert_to_dB:
            ax.set_title("Strip-map SAR response (dB)",fontsize=self.font_size_title)
        else:
            ax.set_title("Strip-map SAR response (mag)",fontsize=self.font_size_title)

        if show:
            plt.show()
    ####################################################################
    #Plotting beamformed cfar detections
    ####################################################################
    def plot_2D_az_beamformed_dets_slice(
            self,
            cfar_det_idxs:np.ndarray,
            ax:plt.Axes=None,
            color="red",
            show=False
    ):
        """Plot a 2D slice of the azimuth beamformed response at 0-deg elevation

        Args:
            cfar_det_idxs (np.ndarray): a 2D beamformed detection array indexed by 
                [range bin, az angle (theta), el angle (phi)]
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if self.processor_SABF.mode == SyntheticArrayBeamformerProcessor.BROADSIDE_MODE:
            self.plot_2D_az_beamformed_dets_slice_broadside(
                cfar_det_idxs=cfar_det_idxs,
                ax=ax,
                color = color,
                show=show
            )
        elif self.processor_SABF.mode == SyntheticArrayBeamformerProcessor.ENDFIRE_MODE:
            self.plot_2D_az_beamformed_dets_slice_endfire(
                cfar_det_idxs=cfar_det_idxs,
                ax=ax,
                color = color,
                show=show
            )

    def plot_2D_az_beamformed_dets_slice_broadside(
            self,
            cfar_det_idxs:np.ndarray,
            ax:plt.Axes=None,
            color="red",
            show=False
    ):
        """Plot a 2D slice of the azimuth beamformed response at 0-deg elevation

        Args:
            cfar_det_idxs (np.ndarray): a 2D beamformed detection array indexed by 
                [range bin, az angle (theta), el angle (phi)]
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()
        
        #identify the elevation bin closest to 0
        diffs = np.abs(self.processor_SABF.el_angle_bins_rad)
        idx = np.argmin(diffs)

        dets = cfar_det_idxs[:,:,idx]
        x_s = self.processor_SABF.x_s[:,:,idx]
        y_s = self.processor_SABF.y_s[:,:,idx]

        x_coords = x_s[dets]
        y_coords = y_s[dets]

        ax.scatter(x_coords,y_coords,
                    c=color,
                    s=5.0,
                    alpha=0.75)
        ax.set_xlim(
            left=np.min(self.processor_SABF.x_s),
            right=np.max(self.processor_SABF.x_s)
        )
        ax.set_ylim(
            bottom=0,
            top=np.max(self.processor_SABF.y_s)
        )
        ax.set_xlabel("X (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("Y (m)",fontsize=self.font_size_axis_labels)
        ax.set_title("Az CFAR detections",fontsize=self.font_size_title)

        if show:
            plt.show()

    def plot_2D_az_beamformed_dets_slice_endfire(
            self,
            cfar_det_idxs:np.ndarray,
            ax:plt.Axes=None,
            color="red",
            show=False
    ):
        """Plot a 2D slice of the azimuth beamformed response at 0-deg elevation

        Args:
            cfar_det_idxs (np.ndarray): a 2D beamformed detection array indexed by 
                [range bin, az angle (theta), el angle (phi)]
            ax (plt.Axes, optional): the axes on which to display the plot.
                If none provided, a figure is automatically generated.
                Defaults to None.
            show (bool, optional): On true, shows the plot. Defaults to False.
        """

        if not ax:
            fig,ax = plt.subplots()
        
        #identify the elevation bin closest to 0
        diffs = np.abs(self.processor_SABF.el_angle_bins_rad)
        idx = np.argmin(diffs)

        dets = cfar_det_idxs[:,:,idx]
        x_s = self.processor_SABF.x_s[:,:,idx]
        y_s = self.processor_SABF.y_s[:,:,idx]

        x_coords = x_s[dets]
        y_coords = y_s[dets]

        ax.scatter(y_coords,x_coords,
                    c=color,
                    s=5.0,
                    alpha=0.75)
        ax.set_ylim(
            bottom=0,
            top=np.max(self.processor_SABF.x_s)
        )
        ax.set_xlim(
            right=np.min(self.processor_SABF.y_s),
            left=np.max(self.processor_SABF.y_s)
        )
        ax.set_xlabel("Y (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("X (m)",fontsize=self.font_size_axis_labels)
        ax.set_title("Az CFAR detections",fontsize=self.font_size_title)

        if show:
            plt.show()
    ####################################################################
    #Plotting depth map
    #################################################################### 
    def plot_3D_radar_depth_map_spherical(
        self,
        resp:np.ndarray,
        cmap="hot_r",
        ax:plt.Axes=None,
        show=False
    ):

        #convert the response to dB
        resp = 20 * np.log10(np.abs(resp))
        dets = self.processor_SABF.beamformed_dets
        
        #filter out non-CFAR detections
        resp[~dets] = 0.0

        #get the max values and their associated indicies
        max_vals = np.max(resp,axis=0)
        max_idxs = np.argmax(resp,axis=0)

        img = self.processor_SABF.range_bins[max_idxs]
        max_val = np.max(max_vals)
        img[max_vals < (max_val - 40)] = \
            self.processor_SABF.range_bins[-1]

        #transpose the image so that az and el are on the correct axes
        img = np.flip(img.transpose(),axis=0)

        if not ax:
            fig,ax = plt.subplots()

        #plot the image
        img = ax.imshow(
            img,
            cmap=cmap,
            interpolation='gaussian',
            extent=[
                np.rad2deg(self.processor_SABF.az_angle_bins_rad[0]),
                np.rad2deg(self.processor_SABF.az_angle_bins_rad[-1]),
                np.rad2deg(self.processor_SABF.el_angle_bins_rad[0]),
                np.rad2deg(self.processor_SABF.el_angle_bins_rad[-1])
            ],
            aspect='auto'
        )

        # plt.colorbar(img,ax=ax)
        ax.set_xlabel("Az (deg)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("El (deg)",fontsize=self.font_size_axis_labels)
        ax.set_title("Depth Map",fontsize=self.font_size_title)

        if show:
            plt.show()
        
        return

    ####################################################################
    #Plotting lidar data
    ####################################################################
    def plot_visible_lidar_pc(
        self,
        lidar_pc_raw:np.ndarray,
        lidar_radar_offset_rad:float=np.deg2rad(180),
        color="red",
        ax:plt.Axes=None,
        show=False
    ):
        if self.processor_SABF.mode == SyntheticArrayBeamformerProcessor.BROADSIDE_MODE:
            self.plot_visible_lidar_pc_broadside(
                lidar_pc_raw=lidar_pc_raw,
                lidar_radar_offset_rad=lidar_radar_offset_rad,
                color=color,
                ax=ax,
                show=show
            )
        elif self.processor_SABF.mode == SyntheticArrayBeamformerProcessor.ENDFIRE_MODE:
            self.plot_visible_lidar_pc_endfire(
                lidar_pc_raw=lidar_pc_raw,
                lidar_radar_offset_rad=lidar_radar_offset_rad,
                color=color,
                ax=ax,
                show=show
            )
    
    def plot_visible_lidar_pc_broadside(
        self,
        lidar_pc_raw:np.ndarray,
        lidar_radar_offset_rad:float=np.deg2rad(180),
        color="red",
        ax:plt.Axes=None,
        show=False
    ):

        if not ax:
            fig,ax = plt.subplots()
        
        #filter out the ground detections
        valid_idxs = (lidar_pc_raw[:,2] > 0.0) & \
                    (lidar_pc_raw[:,2] < 2.0)
        lidar_pc_raw = lidar_pc_raw[valid_idxs,0:2]

        #rotate the lidar pc into the radar's frame of view
        lidar_pc_raw = apply_rot_trans(
            lidar_pc_raw,
            rot_angle_rad=lidar_radar_offset_rad,
            trans=np.array([0,0])
        )

        # filter only the points that should be able to be seen
        valid_idxs = (lidar_pc_raw[:,0] > np.min(self.processor_SABF.x_s)) & \
                    (lidar_pc_raw[:,0] < np.max(self.processor_SABF.x_s)) & \
                    (lidar_pc_raw[:,1] > np.min(self.processor_SABF.y_s)) & \
                    (lidar_pc_raw[:,1] < np.max(self.processor_SABF.y_s)) 
        lidar_pc_raw = lidar_pc_raw[valid_idxs,:]


        ax.scatter(lidar_pc_raw[:,0],lidar_pc_raw[:,1],
                   c=color,alpha=0.75,
                   s=0.25)
        ax.set_xlim(
            left=np.min(self.processor_SABF.x_s),
            right=np.max(self.processor_SABF.x_s)
        )
        ax.set_ylim(
            bottom=0,
            top=np.max(self.processor_SABF.y_s)
        )
        ax.set_xlabel("Y (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("X (m)",fontsize=self.font_size_axis_labels)
        ax.set_title("Lidar GT",fontsize=self.font_size_title)

        if show:
            plt.show()

    def plot_visible_lidar_pc_endfire(
        self,
        lidar_pc_raw:np.ndarray,
        lidar_radar_offset_rad:float=np.deg2rad(180),
        color="red",
        ax:plt.Axes=None,
        show=False
    ):

        if not ax:
            fig,ax = plt.subplots()
        
        #filter out the ground detections
        valid_idxs = (lidar_pc_raw[:,2] > 0.2) & \
                    (lidar_pc_raw[:,2] < 1.0)
        lidar_pc_raw = lidar_pc_raw[valid_idxs,0:2]

        #rotate the lidar pc into the radar's frame of view
        lidar_pc_raw = apply_rot_trans(
            lidar_pc_raw,
            rot_angle_rad=lidar_radar_offset_rad,
            trans=np.array([0,0])
        )

        # filter only the points that should be able to be seen
        valid_idxs = (lidar_pc_raw[:,1] > np.min(self.processor_SABF.x_s)) & \
                    (lidar_pc_raw[:,1] < np.max(self.processor_SABF.x_s)) & \
                    (lidar_pc_raw[:,0] > np.min(self.processor_SABF.y_s)) & \
                    (lidar_pc_raw[:,0] < np.max(self.processor_SABF.y_s)) 
        lidar_pc_raw = lidar_pc_raw[valid_idxs,:]


        ax.scatter(-1 * lidar_pc_raw[:,0],lidar_pc_raw[:,1],
                   c=color,alpha=0.75,
                   s=0.25)
        ax.set_xlim(
            right=np.min(self.processor_SABF.y_s),
            left=np.max(self.processor_SABF.y_s)
        )
        ax.set_ylim(
            bottom=0,
            top=np.max(self.processor_SABF.x_s)
        )
        ax.set_xlabel("Y (m)",fontsize=self.font_size_axis_labels)
        ax.set_ylabel("X (m)",fontsize=self.font_size_axis_labels)
        ax.set_title("Lidar GT",fontsize=self.font_size_title)

        if show:
            plt.show()

    ####################################################################
    #Generating Compilations
    #################################################################### 

    def plot_compilation(
            self,
            adc_cube:np.ndarray,
            vels:np.ndarray,
            camera_view:np.ndarray=np.empty(shape=(0)),
            lidar_pc_raw:np.ndarray=np.empty(shape=(0)),
            lidar_radar_offset_rad:float=np.deg2rad(180),
            convert_to_dB=False,
            cmap="viridis",
            axs:plt.Axes=[],
            show=False
    ):
        if len(axs) == 0:
            fig,axs=plt.subplots(3,3, figsize=(15,15))
            fig.subplots_adjust(wspace=0.3,hspace=0.30)
        
        #plot the array geometry
        self.plot_synthetic_array_geometry(
            vels=vels,
            ax=axs[0,0],
            show=False
        )

        #plot the camera view
        if camera_view.shape[0] > 0:
            axs[1,1].imshow(camera_view)
            axs[1,1].set_title("Frontal Camera View",fontsize=self.font_size_title)
        
        if self.processor_SABF.array_geometry_valid:
            #compute the response
            resp = self.processor_SABF.process(
                adc_cube
            )

            dets = self.processor_SABF.beamformed_dets

            #plot the az beamformed response
            self.plot_2D_az_beamformed_resp_slice(
                resp=resp,
                convert_to_dB=convert_to_dB,
                cmap=cmap,
                ax=axs[0,1],
                show=False
            )

            #plot the el beamformed response
            self.plot_2D_el_beamformed_resp_slice(
                resp=resp,
                convert_to_dB=convert_to_dB,
                cmap=cmap,
                ax=axs[0,2],
                show=False
            )

            # #plot the depth map
            # self.plot_3D_radar_depth_map_spherical(
            #     resp=resp,
            #     cmap="hot_r",
            #     ax=axs[0,2],
            #     show=False
            # )

            # plot the StripMap SAR response if available
            if self.processor_stripMapSAR:
                sar_resp = self.processor_stripMapSAR.process(
                    adc_cube=adc_cube,
                    vel_m_per_s=np.abs(vels)[0],
                    sensor_height_m=0.24,
                    rx_index=0,
                    max_SAR_distance=1.5
                )
                self.plot_stripmap_SAR_resp(
                    resp=sar_resp,
                    convert_to_dB=convert_to_dB,
                    cmap=cmap,
                    ax=axs[1,2],
                    show=False
                )

            #plot the CFAR detections
            # plot the az beamformed response
            # self.plot_2D_az_beamformed_resp_slice(
            #     resp=resp,
            #     convert_to_dB=convert_to_dB,
            #     cmap=cmap,
            #     ax=axs[2,0],
            #     show=False
            # )

            # self.plot_2D_az_beamformed_dets_slice(
            #     cfar_det_idxs=dets,
            #     ax=axs[2,0],
            #     show=False
            # )

            # #plot the lidar data
            if lidar_pc_raw.shape[0] > 0:

                self.plot_2D_az_beamformed_resp_slice(
                    resp=resp,
                    convert_to_dB=convert_to_dB,
                    cmap=cmap,
                    ax=axs[1,0],
                    show=False
                )
                
                self.plot_visible_lidar_pc(
                    lidar_pc_raw=lidar_pc_raw,
                    lidar_radar_offset_rad=lidar_radar_offset_rad,
                    ax=axs[1,0],
                    color="red",
                    show=False
                )
                axs[1,0].set_title("Radar vs Lidar GT")

            # #plot CFAR detections vs lidar point cloud
            #     self.plot_2D_az_beamformed_dets_slice(
            #         cfar_det_idxs=dets,
            #         ax=axs[2,2],
            #         color="blue",
            #         show=False
            #     )
            #     self.plot_visible_lidar_pc(
            #         lidar_pc_raw=lidar_pc_raw,
            #         lidar_radar_offset_rad=lidar_radar_offset_rad,
            #         ax=axs[2,2],
            #         color="red",
            #         show=False
            #     )
            #     axs[2,2].set_title("CFAR vs Lidar GT")

        if show:
            plt.show()