"""Range response view implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional

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

    def update_view(self, payload: Dict[str, Any]) -> None:
        """Update the view with new data.

        Args:
            payload: Dictionary containing range response data and metadata.
        """
        if not isinstance(payload, dict):
            self.logger.warning("RangeResponseView expected dict payload, got %s", type(payload))
            return
        data = np.array(payload.get("data")).flatten()
        rng_bins = payload.get("range_bins")
        if data.size == 0:
            return
        
        if self.convert_to_db:
            display = 20 * np.log10(np.maximum(np.abs(data), 1e-12))
            self.plot.setLabel("left", "Amplitude (dB)")
            self.plot.setTitle("Range FFT (dB)")
        else:
            display = np.abs(data)
            self.plot.setLabel("left", "Amplitude (mag)")
            self.plot.setTitle("Range FFT (mag)")
            
        if rng_bins is None:
            x_vals = np.arange(display.shape[0])
        else:
            x_vals = np.array(rng_bins).flatten()
        self.curve.setData(x_vals, display)
