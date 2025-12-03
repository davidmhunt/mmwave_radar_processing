"""Altitude view implementation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pyqtgraph as pg

from mmwave_radar_processing.visualization.views.range_response_view import RangeResponseView


class AltitudeView(RangeResponseView):
    """Displays coarse FFT with overlaid estimated altitude."""

    def __init__(self, parent=None, logger=None) -> None:
        """Initialize the altitude view."""
        super().__init__(parent=parent, logger=logger)
        
        # Add infinite vertical line for altitude
        self.altitude_line = pg.InfiniteLine(
            pos=0,
            angle=90,
            pen=pg.mkPen(color='g', width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
            label='Alt: {value:.2f}m',
            labelOpts={'position': 0.8, 'color': (200, 200, 200), 'movable': True}
        )
        self.plot.addItem(self.altitude_line)

    def update_view(self, payload: Dict[str, Any]) -> None:
        """Update the view with new data.

        Args:
            payload: Dictionary containing coarse FFT, range bins, and estimated altitude.
        """
        # Map coarse_fft_data to data for parent class
        if "coarse_fft_data" in payload:
            payload["data"] = payload["coarse_fft_data"]
            
        super().update_view(payload)
        
        # Update altitude line
        altitude = payload.get("current_altitude_corrected_m")
        if altitude is not None and altitude > 0:
            self.altitude_line.setValue(altitude)
            self.altitude_line.show()
        else:
            self.altitude_line.hide()
