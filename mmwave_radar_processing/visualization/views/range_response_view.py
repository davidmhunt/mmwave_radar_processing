"""Range response view implementation."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QVBoxLayout

from mmwave_radar_processing.visualization.views.base_view import BaseView


class RangeResponseView(BaseView):
    """Displays range response."""

    def __init__(self, parent=None, logger=None) -> None:
        """Initialize the range response view."""
        super().__init__(parent=parent, logger=logger)
        layout = QVBoxLayout(self)
        self.plot = pg.PlotWidget()
        self.curve = self.plot.plot([], [])
        self.plot.setLabel("bottom", "Range")
        self.plot.setLabel("left", "Magnitude")
        self.plot.setTitle("Range FFT")
        layout.addWidget(self.plot)

    def set_data(self, payload: Any) -> None:
        """Update the view with new data."""
        if payload is None:
            return
        if isinstance(payload, dict):
            data = np.array(payload.get("data")).flatten()
            rng_bins = payload.get("range_bins")
        else:
            data = np.array(payload).flatten()
            rng_bins = None
        if data.size == 0:
            return
        display = data.copy()
        if self.convert_to_db:
            display = 20 * np.log10(np.maximum(display, 1e-12))
            self.plot.setLabel("left", "Amplitude (dB)")
            self.plot.setTitle("Range FFT (dB)")
        else:
            self.plot.setLabel("left", "Amplitude (mag)")
            self.plot.setTitle("Range FFT (mag)")
        if rng_bins is None:
            x_vals = np.arange(display.shape[0])
        else:
            x_vals = np.array(rng_bins).flatten()
        self.curve.setData(x_vals, display)
