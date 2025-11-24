"""Range-Doppler view implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QVBoxLayout

from mmwave_radar_processing.visualization.views.base_view import BaseView


class RangeDopplerView(BaseView):
    """Displays range-Doppler response."""

    def __init__(self, parent=None, logger=None) -> None:
        """Initialize the range-Doppler view."""
        super().__init__(parent=parent, logger=logger)
        layout = QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        self.plot.setLabel("bottom", "Velocity (m/s)")
        self.plot.setLabel("left", "Range (m)")
        self.plot.setTitle("Range-Doppler Heatmap")
        layout.addWidget(self.plot)

    def set_data(self, payload: Dict[str, Any]) -> None:
        """Update the view with new data.

        Args:
            payload: Dictionary containing range-Doppler data and metadata.
        """
        if not isinstance(payload, dict):
            self.logger.warning("RangeDopplerView expected dict payload, got %s", type(payload))
            return
        data = np.array(payload.get("data"))
        vel_bins = payload.get("vel_bins")
        range_bins = payload.get("range_bins")

        if data.size == 0:
            return

        display = np.flipud(np.copy(data))
        if self.convert_to_db:
            display = 20 * np.log10(np.maximum(display, 1e-12))
        self.image.setImage(display, autoLevels=True)

        if vel_bins is not None and range_bins is not None:
            self.image.setRect(
                QRectF(
                    vel_bins[0],
                    range_bins[0],
                    vel_bins[-1] - vel_bins[0],
                    range_bins[-1] - range_bins[0],
                )
            )

        title = "Range-Doppler Heatmap (dB)" if self.convert_to_db else "Range-Doppler Heatmap (mag)"
        self.plot.setTitle(title)
