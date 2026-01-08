"""Doppler-Azimuth view implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QVBoxLayout

from mmwave_radar_processing.visualization.views.base_view import BaseView


class DopplerAzimuthView(BaseView):
    """Displays Doppler-azimuth response."""

    def __init__(self, parent=None, logger=None) -> None:
        """Initialize the Doppler-azimuth view."""
        super().__init__(parent=parent, logger=logger)
        layout = QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        self.plot.setLabel("bottom", "Velocity (m/s)")
        self.plot.setLabel("left", "Angle (rad)")
        self.plot.setTitle("Doppler-Azimuth Heatmap")
        layout.addWidget(self.plot)
        self.set_colormap("viridis")

    def set_colormap(self, name: str = "viridis") -> None:
        """Set the colormap for the image item."""
        try:
            cmap = pg.colormap.get(name)
            self.image.setLookupTable(cmap.getLookupTable())
        except Exception as exc:
            self.logger.warning("Failed to set colormap %s: %s", name, exc)

    def update_view(self, payload: Dict[str, Any]) -> None:
        """Update the view with new data.

        Args:
            payload: Dictionary containing Doppler-Azimuth data and metadata.
        """
        if not isinstance(payload, dict):
            self.logger.warning("DopplerAzimuthView expected dict payload, got %s", type(payload))
            return
        data = np.array(payload.get("data"))
        vel_bins = payload.get("vel_bins")
        angle_bins = payload.get("angle_bins")

        if data.size == 0:
            return

        # Processor usually returns [velocity, angle].
        # pyqtgraph expects [x, y] -> [angle, velocity].
        # So we transpose.
        display = data.T
        
        if self.convert_to_db:
            display = 20 * np.log10(np.maximum(np.abs(display), 1e-12))
        else:
            display = np.abs(display)
            
        self.image.setImage(display, autoLevels=True)

        if vel_bins is not None and angle_bins is not None:
            self.image.setRect(
                QRectF(
                    angle_bins[0],
                    vel_bins[0],
                    angle_bins[-1] - angle_bins[0],
                    vel_bins[-1] - vel_bins[0],
                )
            )

        title = "Doppler-Azimuth Heatmap (dB)" if self.convert_to_db else "Doppler-Azimuth Heatmap (mag)"
        self.plot.setTitle(title)
