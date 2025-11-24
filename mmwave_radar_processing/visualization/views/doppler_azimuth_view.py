"""Doppler-Azimuth view implementation."""

from __future__ import annotations

from typing import Any, Optional

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

    def set_data(self, payload: Any) -> None:
        """Update the view with new data."""
        if payload is None:
            return
        if isinstance(payload, dict):
            data = np.array(payload.get("data"))
            vel_bins = payload.get("vel_bins")
            angle_bins = payload.get("angle_bins")
        else:
            data = np.array(payload)
            vel_bins = None
            angle_bins = None
        if data.size == 0:
            return
        display = np.copy(data)
        if self.convert_to_db:
            display = 20 * np.log10(np.maximum(display, 1e-12))
            title = "Doppler-Azimuth Heatmap (dB)"
        else:
            title = "Doppler-Azimuth Heatmap (mag)"
        self.image.setImage(display, autoLevels=True)
        if vel_bins is not None and angle_bins is not None:
            self.image.setRect(
                QRectF(
                    vel_bins[0],
                    angle_bins[0],
                    vel_bins[-1] - vel_bins[0],
                    angle_bins[-1] - angle_bins[0],
                )
            )
        self.plot.setTitle(title)
