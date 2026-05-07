"""Point Cloud view implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QHBoxLayout

from mmwave_radar_processing.visualization.views.base_view import BaseView


import OpenGL.GL as GL

class ThickerAxisItem(gl.GLAxisItem):
    """Axis item with thicker lines."""
    def paint(self):
        self.setupGLState()
        GL.glLineWidth(5)
        super().paint()

class PointCloudView(BaseView):
    """Displays 3D point cloud."""

    def __init__(self, parent=None, logger=None) -> None:
        """Initialize the point cloud view."""
        super().__init__(parent=parent, logger=logger)
        
        layout = QHBoxLayout(self)
        
        # 3D Plot
        self.plot = gl.GLViewWidget()
        layout.addWidget(self.plot, stretch=1)
        
        # Color bar widget
        self.cb_widget = pg.GraphicsLayoutWidget()
        self.cb_widget.setFixedWidth(80)
        layout.addWidget(self.cb_widget)
        
        # Add grid
        grid = gl.GLGridItem()
        self.plot.addItem(grid)
        
        # Add axes
        axis = ThickerAxisItem()
        axis.setSize(1, 1, 1)
        self.plot.addItem(axis)
        
        # Scatter plot item
        self.scatter = gl.GLScatterPlotItem(
            pos=np.zeros((0, 3)),
            color=(1, 1, 1, 1),
            size=5,
            pxMode=True
        )
        self.plot.addItem(self.scatter)
        
        # Set camera position
        self.plot.setCameraPosition(distance=20, elevation=30, azimuth=45)
        
        # Configurable velocity range
        self.min_vel = -0.25
        self.max_vel = 0.25
        self.cmap_name = "viridis"
        
        # Setup color bar
        self._init_colorbar()

    def _init_colorbar(self) -> None:
        """Initialize the color bar."""
        self.cb_plot = self.cb_widget.addPlot()
        self.cb_plot.hideButtons()
        self.cb_plot.setMouseEnabled(x=False, y=False)
        self.cb_plot.setMenuEnabled(False)
        
        # Hide axes except right
        self.cb_plot.hideAxis('bottom')
        self.cb_plot.hideAxis('top')
        self.cb_plot.hideAxis('left')
        self.cb_plot.showAxis('right')
        self.cb_plot.getAxis('right').setLabel('Velocity (m/s)')
        
        self.cb_img = pg.ImageItem()
        self.cb_plot.addItem(self.cb_img)
        
        self._update_colorbar()

    def _update_colorbar(self) -> None:
        """Update the color bar gradient and range."""
        try:
            cmap = plt.get_cmap(self.cmap_name)
        except Exception:
            cmap = plt.get_cmap("viridis")
            
        # Generate gradient (256 steps)
        lut = cmap(np.linspace(0, 1, 256)) # (256, 4)
        lut = (lut * 255).astype(np.uint8)
        
        # Create 1x256 image for vertical bar
        # ImageItem expects (w, h, 4)
        grad = np.expand_dims(lut, axis=0) 
        
        self.cb_img.setImage(grad)
        
        # Set rect to match velocity range
        # Image is 0..1 in x, min_vel..max_vel in y
        self.cb_img.setRect(QRectF(0, self.min_vel, 1, self.max_vel - self.min_vel))
        
        # Fix ranges
        self.cb_plot.setXRange(0, 1, padding=0)
        self.cb_plot.setYRange(self.min_vel, self.max_vel, padding=0)

    def update_view(self, payload: Any) -> None:
        """Update the view with new data.

        Args:
            payload: Numpy array of shape (N, 4) containing (x, y, z, velocity).
                     Or a dictionary containing 'data' key with the array.
        """
        # Handle both direct array and dict payload
        if isinstance(payload, dict):
            data = payload.get("data")
        else:
            data = payload
            
        if data is None:
            return
            
        points = np.array(data)
        
        if points.size == 0:
            self.scatter.setData(pos=np.zeros((0, 3)), color=np.zeros((0, 4)))
            return
            
        # Extract coordinates (x, y, z)
        pos = points[:, :3]
        
        # Extract velocity for coloring
        velocity = points[:, 3]
        
        # Map velocity to color
        norm_vel = (velocity - self.min_vel) / (self.max_vel - self.min_vel)
        norm_vel = np.clip(norm_vel, 0, 1)
        
        try:
            cmap = plt.get_cmap(self.cmap_name)
        except Exception:
            cmap = plt.get_cmap("viridis")
            
        colors = cmap(norm_vel) # Returns (N, 4) float array 0..1
        
        self.scatter.setData(pos=pos, color=colors)


class PointCloudView2D(BaseView):
    """Displays 2D projected point cloud."""

    def __init__(self, parent=None, logger=None) -> None:
        """Initialize the 2D point cloud view."""
        super().__init__(parent=parent, logger=logger)
        
        layout = QHBoxLayout(self)
        
        # 2D Plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.showGrid(x=True, y=True)
        layout.addWidget(self.plot_widget, stretch=1)
        
        # Color bar widget
        self.cb_widget = pg.GraphicsLayoutWidget()
        self.cb_widget.setFixedWidth(80)
        layout.addWidget(self.cb_widget)
        
        # Scatter plot item
        self.scatter = pg.ScatterPlotItem(
            size=5,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 255, 255, 255)
        )
        self.plot_widget.addItem(self.scatter)
        
        # Default labels
        self.plot_widget.setLabel('bottom', "X (m)")
        self.plot_widget.setLabel('left', "Y (m)")
        
        # Configurable velocity range
        self.min_vel = -0.25
        self.max_vel = 0.25
        self.cmap_name = "viridis"
        
        # Setup color bar
        self._init_colorbar()

    def _init_colorbar(self) -> None:
        """Initialize the color bar (copied from PointCloudView)."""
        self.cb_plot = self.cb_widget.addPlot()
        self.cb_plot.hideButtons()
        self.cb_plot.setMouseEnabled(x=False, y=False)
        self.cb_plot.setMenuEnabled(False)
        
        self.cb_plot.hideAxis('bottom')
        self.cb_plot.hideAxis('top')
        self.cb_plot.hideAxis('left')
        self.cb_plot.showAxis('right')
        self.cb_plot.getAxis('right').setLabel('Velocity (m/s)')
        
        self.cb_img = pg.ImageItem()
        self.cb_plot.addItem(self.cb_img)
        
        self._update_colorbar()

    def _update_colorbar(self) -> None:
        """Update the color bar gradient (copied from PointCloudView)."""
        try:
            cmap = plt.get_cmap(self.cmap_name)
        except Exception:
            cmap = plt.get_cmap("viridis")
            
        lut = cmap(np.linspace(0, 1, 256))
        lut = (lut * 255).astype(np.uint8)
        grad = np.expand_dims(lut, axis=0) 
        
        self.cb_img.setImage(grad)
        self.cb_img.setRect(QRectF(0, self.min_vel, 1, self.max_vel - self.min_vel))
        self.cb_plot.setXRange(0, 1, padding=0)
        self.cb_plot.setYRange(self.min_vel, self.max_vel, padding=0)

    def update_view(self, payload: Any) -> None:
        """Update the 2D view.
        
        Expected payload:
        {
            "data": N x 3 ndarray (axis1, axis2, velocity),
            "axis_labels": [label1, label2]
        }
        """
        if not isinstance(payload, dict):
            return
            
        data = payload.get("data")
        labels = payload.get("axis_labels")
        
        if data is None or data.size == 0:
            self.scatter.setData(pos=np.zeros((0, 2)), brush=np.zeros((0, 4)))
            return
            
        # Update axis labels if provided
        if labels and len(labels) == 2:
            self.plot_widget.setLabel('bottom', labels[0])
            self.plot_widget.setLabel('left', labels[1])
            
        points = np.array(data)
        pos = points[:, :2]
        velocity = points[:, 2]
        
        # Map velocity to color
        norm_vel = (velocity - self.min_vel) / (self.max_vel - self.min_vel)
        norm_vel = np.clip(norm_vel, 0, 1)
        
        try:
            cmap = plt.get_cmap(self.cmap_name)
        except Exception:
            cmap = plt.get_cmap("viridis")
            
        colors = cmap(norm_vel) # (N, 4) float 0..1
        # pyqtgraph expects colors in 0..255 for QBrush if using array
        colors_uint8 = (colors * 255).astype(np.uint8)
        
        self.scatter.setData(pos=pos, brush=colors_uint8)
