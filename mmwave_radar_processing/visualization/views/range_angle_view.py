"""Range-Angle view implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QVBoxLayout

from mmwave_radar_processing.visualization.views.base_view import BaseView


class RangeAngleView(BaseView):
    """Displays range-angle response."""

    def __init__(self, parent=None, logger=None) -> None:
        """Initialize the range-angle view."""
        super().__init__(parent=parent, logger=logger)
        layout = QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        self.image = pg.ImageItem()
        self.plot.addItem(self.image)
        self.plot.setLabel("bottom", "Angle (rad)")
        self.plot.setLabel("left", "Range (m)")
        self.plot.setTitle("Range-Azimuth Heatmap (Polar.)")
        layout.addWidget(self.plot)
        self.set_colormap("viridis")

    def set_data(self, payload: Dict[str, Any]) -> None:
        """Update the view with new data.

        Args:
            payload: Dictionary containing range-angle data and metadata.
        """
        if not isinstance(payload, dict):
            self.logger.warning("RangeAngleView expected dict payload, got %s", type(payload))
            return
        data = np.array(payload.get("data"))
        angle_bins = payload.get("angle_bins")
        range_bins = payload.get("range_bins")
        if data.size == 0:
            return

        display = np.flipud(np.copy(data))
        if self.convert_to_db:
            display = 20 * np.log10(np.maximum(display, 1e-12))
            title = "Range-Azimuth Heatmap (dB Polar.)"
        else:
            title = "Range-Azimuth Heatmap (mag Polar.)"
        self.image.setImage(display, autoLevels=True)
        if angle_bins is not None and range_bins is not None:
            self.image.setRect(
                QRectF(
                    angle_bins[0],
                    range_bins[0],
                    angle_bins[-1] - angle_bins[0],
                    range_bins[-1] - range_bins[0],
                )
            )
        self.plot.setTitle(title)
