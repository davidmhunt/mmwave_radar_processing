"""Range-Doppler Detector view implementation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pyqtgraph as pg

from mmwave_radar_processing.visualization.views.range_doppler_view import RangeDopplerView


class RangeDopplerDetectorView(RangeDopplerView):
    """Displays range-Doppler response with overlaid detections."""

    def __init__(self, parent=None, logger=None) -> None:
        """Initialize the range-Doppler detector view."""
        super().__init__(parent=parent, logger=logger)
        
        # Add scatter plot item for detections
        self.scatter = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 0, 0, 255), # Red
            symbol='o',
            name='Detections'
        )
        self.plot.addItem(self.scatter)
        self.plot.addLegend()

    def update_view(self, payload: Dict[str, Any]) -> None:
        """Update the view with new data.

        Args:
            payload: Dictionary containing range-Doppler data, metadata, and detections.
        """
        # If rng_dop_resp is available, use it as the heatmap data
        # The 'data' key might contain detections if coming from the processor directly
        if "rng_dop_resp" in payload:
            payload["data"] = payload["rng_dop_resp"]

        # Update the heatmap using the parent class
        super().update_view(payload)
        
        dets = payload.get("dets")
        vel_bins = payload.get("vel_bins")
        range_bins = payload.get("range_bins")
        
        if dets is None or vel_bins is None or range_bins is None:
            self.scatter.clear()
            return
            
        dets = np.array(dets)
        if dets.size == 0:
            self.scatter.clear()
            return

        # dets contains indices [range_idx, doppler_idx]
        # We need to map these to physical coordinates [velocity, range]
        # Note: pyqtgraph uses x=velocity, y=range
        
        det_range_idxs = dets[:, 0].astype(int)
        det_vel_idxs = dets[:, 1].astype(int)
        
        # Ensure indices are within bounds
        if np.any(det_range_idxs >= len(range_bins)) or np.any(det_vel_idxs >= len(vel_bins)):
            self.logger.warning("Detection indices out of bounds")
            self.scatter.clear()
            return

        det_ranges = range_bins[det_range_idxs]
        det_vels = vel_bins[det_vel_idxs]
        
        # Update scatter plot
        # setData expects x, y
        self.scatter.setData(x=det_vels, y=det_ranges)
