"""Range detector view implementation."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pyqtgraph as pg

from mmwave_radar_processing.visualization.views.range_response_view import RangeResponseView


class RangeDetectorView(RangeResponseView):
    """Displays range response with overlaid thresholds and detections."""

    def __init__(self, parent=None, logger=None) -> None:
        """Initialize the range detector view."""
        super().__init__(parent=parent, logger=logger)
        
        # Add curve for thresholds
        self.threshold_curve = self.plot.plot(
            [], [], 
            pen=pg.mkPen(color='y', style=pg.QtCore.Qt.PenStyle.DashLine),
            name='Threshold'
        )
        
        # Add scatter plot item for detections
        self.scatter = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(255, 0, 0, 255), # Red
            symbol='x',
            name='Detections'
        )
        self.plot.addItem(self.scatter)
        self.plot.addLegend()

    def update_view(self, payload: Dict[str, Any]) -> None:
        """Update the view with new data.

        Args:
            payload: Dictionary containing range response data, metadata, thresholds, and detections.
        """

        # make "data" be the range response so that the RangeResponseView can handle it        
        if "range_resp" in payload:
            payload["data"] = payload["range_resp"]
            
        super().update_view(payload)
        
        # Get data for thresholds and detections
        thresholds = payload.get("thresholds")
        dets = payload.get("dets")
        range_bins = payload.get("range_bins")
        
        # Update thresholds
        if thresholds is not None:
            thresholds = np.array(thresholds).flatten()
            if range_bins is None:
                x_vals = np.arange(thresholds.shape[0])
            else:
                x_vals = np.array(range_bins).flatten()
            
            if self.convert_to_db:
                 # Avoid log of zero
                display_thresh = 20 * np.log10(np.maximum(thresholds, 1e-12))
            else:
                display_thresh = thresholds
                
            self.threshold_curve.setData(x_vals, display_thresh)
        else:
            self.threshold_curve.clear()
            
        # Update detections
        if dets is None or range_bins is None:
            self.scatter.clear()
            return
            
        dets = np.array(dets).astype(int)
        if dets.size == 0:
            self.scatter.clear()
            return

        # dets contains indices of range bins
        # We need to map these to physical range values (x-axis) and signal magnitude (y-axis)
        
        det_ranges = range_bins[dets]
        
        # Get the signal data to plot detections at the correct height
        data = np.array(payload.get("data")).flatten()
        if self.convert_to_db:
            det_mags = 20 * np.log10(np.maximum(np.abs(data[dets]), 1e-12))
        else:
            det_mags = np.abs(data[dets])
        
        self.scatter.setData(x=det_ranges, y=det_mags)
