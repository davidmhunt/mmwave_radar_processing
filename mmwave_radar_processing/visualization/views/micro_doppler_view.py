"""Micro-Doppler view implementation."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QVBoxLayout

from mmwave_radar_processing.visualization.views.base_view import BaseView


class MicroDopplerView(BaseView):
    """Displays micro-Doppler response."""

    def __init__(self, parent=None, logger=None) -> None:
        """Initialize the micro-Doppler view."""
        super().__init__(parent=parent, logger=logger)
        layout = QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        self.plot.setLabel("bottom", "Time (s)")
        self.plot.setLabel("left", "Velocity (m/s)")
        self.plot.setTitle("Micro-Doppler Heatmap")
        layout.addWidget(self.plot)

    def set_data(self, payload: Any) -> None:
        """Update the view with new data."""
        if payload is None:
            return
        if isinstance(payload, dict):
            data = np.array(payload.get("data"))
            time_bins = payload.get("time_bins")
            vel_bins = payload.get("vel_bins")
        else:
            data = np.array(payload)
            time_bins = None
            vel_bins = None
        if data.size == 0:
            return
        display = np.copy(data)
        if self.convert_to_db:
            display = 20 * np.log10(np.maximum(display, 1e-12))
            title = "Micro-Doppler Heatmap (dB)"
        else:
            title = "Micro-Doppler Heatmap (mag)"
        self.image.setImage(display, autoLevels=True)
        if time_bins is not None and vel_bins is not None:
            self.image.setRect(
                QRectF(
                    time_bins[0],
                    vel_bins[0],
                    time_bins[-1] - time_bins[0],
                    vel_bins[-1] - vel_bins[0],
                )
            )
        self.plot.setTitle(title)
