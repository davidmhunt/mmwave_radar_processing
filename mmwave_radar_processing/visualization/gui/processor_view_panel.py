"""Panel containing the grid of processor views."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import numpy as np
from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QVBoxLayout,
    QWidget,
)

from mmwave_radar_processing.logging.logger import get_logger
from mmwave_radar_processing.visualization.backends.processor_registry import (
    ProcessorSpec,
)
from mmwave_radar_processing.visualization.views.base_view import BaseView


class ProcessorViewPanel(QWidget):
    """Panel displaying a 2x2 grid of selectable processor views."""

    def __init__(
        self,
        registry: Dict[str, ProcessorSpec],
        parent: Optional[QWidget] = None,
        logger=None,
    ) -> None:
        """Initialize the processor view panel.

        Args:
            registry: Mapping of processor keys to specs.
            parent: Optional parent widget.
            logger: Optional logger instance.
        """
        super().__init__(parent)
        self.logger = logger or get_logger(__name__)
        self.registry = registry
        
        # Store instantiated view widgets: key -> widget
        self.view_widgets: Dict[str, BaseView] = {}
        
        # Store active view keys for each grid cell: (row, col) -> key
        self.active_views: Dict[tuple[int, int], Optional[str]] = {}
        
        # Store layout containers for each cell to easily swap widgets
        self.cell_layouts: Dict[tuple[int, int], QVBoxLayout] = {}
        
        # Store latest payloads for each processor key
        self.latest_payloads: Dict[str, Any] = {}
        
        self._init_views()
        self._init_ui()
        self.populate_placeholder_data()

    def _init_views(self) -> None:
        """Instantiate all available views from the registry."""
        for key, spec in self.registry.items():
            if spec.view_cls:
                try:
                    view_widget = spec.view_cls(parent=self, logger=self.logger)
                    # Initially hide all views
                    view_widget.setVisible(False)
                    self.view_widgets[key] = view_widget
                except Exception as exc:
                    self.logger.error("Failed to instantiate view for %s: %s", key, exc)

    def _init_ui(self) -> None:
        """Initialize the 2x2 grid layout."""
        grid_layout = QGridLayout(self)
        grid_layout.setContentsMargins(4, 4, 4, 4)
        grid_layout.setSpacing(4)

        # Create 2x2 grid
        for row in range(2):
            for col in range(2):
                cell_widget = QWidget()
                cell_layout = QVBoxLayout(cell_widget)
                cell_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.setSpacing(2)
                
                # Dropdown for view selection
                combo = QComboBox()
                combo.addItem("None", None)
                for key, spec in self.registry.items():
                    if key in self.view_widgets:
                        combo.addItem(spec.display_name, key)
                
                # Connect signal
                # Use closure to capture row/col
                combo.currentIndexChanged.connect(
                    lambda index, r=row, c=col, cb=combo: self._on_dropdown_changed(r, c, cb)
                )
                
                cell_layout.addWidget(combo)
                
                # Container for the view
                view_container = QWidget()
                view_layout = QVBoxLayout(view_container)
                view_layout.setContentsMargins(0, 0, 0, 0)
                cell_layout.addWidget(view_container, 1) # Give it stretch
                
                self.cell_layouts[(row, col)] = view_layout
                self.active_views[(row, col)] = None
                
                grid_layout.addWidget(cell_widget, row, col)
                grid_layout.setRowStretch(row, 1)
                grid_layout.setColumnStretch(col, 1)

        # Set default selections for a nice initial state
        # Top-left: Range-Doppler
        # Top-right: Range-Angle
        # Bottom-left: Range Response
        # Bottom-right: Doppler-Azimuth
        
        defaults = [
            (0, 0, "range_doppler_resp"),
            (0, 1, "range_angle_resp"),
            (1, 0, "range_resp"),
            (1, 1, "doppler_azimuth_resp"),
        ]
        
        # We need to find the combo boxes to set their values
        # This is a bit hacky but works since we know the layout structure
        for row, col, key in defaults:
            if key in self.view_widgets:
                # Find the combo box in the layout
                # grid -> cell_widget -> cell_layout -> item 0 (combo)
                item = grid_layout.itemAtPosition(row, col)
                if item:
                    cell_widget = item.widget()
                    if cell_widget:
                        # The combo is the first child of the cell widget's layout? 
                        # Or we can just findChild QComboBox
                        combo = cell_widget.findChild(QComboBox)
                        if combo:
                            index = combo.findData(key)
                            if index >= 0:
                                combo.setCurrentIndex(index)

    def _on_dropdown_changed(self, row: int, col: int, combo: QComboBox) -> None:
        """Handle view selection change."""
        new_key = combo.currentData()
        old_key = self.active_views.get((row, col))
        
        layout = self.cell_layouts.get((row, col))
        if layout is None:
            return

        # Check if the new view is already active in another cell
        if new_key is not None:
            for (r, c), active_key in self.active_views.items():
                if active_key == new_key and (r, c) != (row, col):
                    self.logger.info("View %s moved from (%d, %d) to (%d, %d)", new_key, r, c, row, col)
                    # Find the combo for the old cell and set it to None (index 0)
                    # We need to find the combo widget in the grid layout
                    # grid -> cell_widget -> cell_layout -> combo (item 0)
                    grid = self.layout()
                    if grid:
                        item = grid.itemAtPosition(r, c)
                        if item and item.widget():
                            cell_widget = item.widget()
                            old_combo = cell_widget.findChild(QComboBox)
                            if old_combo:
                                # Block signals to prevent recursion if needed, 
                                # but we actually WANT the signal to fire to update the old cell UI
                                # However, setting it to 0 will trigger _on_dropdown_changed for (r,c)
                                # which will hide the view there. This is exactly what we want.
                                old_combo.setCurrentIndex(0) 
                    break

        # Hide old view
        if old_key and old_key in self.view_widgets:
            old_widget = self.view_widgets[old_key]
            layout.removeWidget(old_widget)
            old_widget.setVisible(False)
            old_widget.setParent(self) # Re-parent to self so it doesn't get deleted

        # Show new view
        if new_key and new_key in self.view_widgets:
            new_widget = self.view_widgets[new_key]
            layout.addWidget(new_widget)
            new_widget.setVisible(True)
            # Force layout update
            layout.update()
            
            # Update with latest data if available
            if new_key in self.latest_payloads:
                new_widget.set_data(self.latest_payloads[new_key])
        
        self.active_views[(row, col)] = new_key
        self.logger.debug("Cell (%d, %d) changed to %s", row, col, new_key)

    def handle_view_update(self, key: str, payload: Dict[str, Any]) -> None:
        """Update the view if it is currently active in any cell.

        Args:
            key: Registry key of the view.
            payload: Data payload.
        """
        # Cache the payload
        self.latest_payloads[key] = payload

        # Check if this view is active in any cell
        is_active = False
        for active_key in self.active_views.values():
            if active_key == key:
                is_active = True
                break
        
        if is_active:
            widget = self.view_widgets.get(key)
            if widget:
                widget.set_data(payload)

    def set_db_mode(self, enabled: bool) -> None:
        """Set dB mode for all views."""
        for widget in self.view_widgets.values():
            widget.set_db_mode(enabled)

    def populate_placeholder_data(self) -> None:
        """Populate placeholder data so views are immediately visible."""
        
        # Range-Doppler placeholder
        if "range_doppler_resp" in self.view_widgets:
            rd_vel = np.linspace(-5, 5, 64)
            rd_rng = np.linspace(0, 20, 64)
            xv, yv = np.meshgrid(rd_vel, rd_rng)
            rd_data = np.exp(-((xv / 3) ** 2 + ((yv - 10) / 5) ** 2))
            self.view_widgets["range_doppler_resp"].set_data(
                {"data": rd_data, "vel_bins": rd_vel, "range_bins": rd_rng}
            )

        # Range response placeholder
        if "range_resp" in self.view_widgets:
            rng_bins = np.linspace(0, 20, 128)
            resp = np.sinc((rng_bins - 8) * 0.5) + 0.2 * np.random.rand(rng_bins.size)
            self.view_widgets["range_resp"].set_data({"data": resp, "range_bins": rng_bins})

        # Range-Angle placeholder
        if "range_angle_resp" in self.view_widgets:
            ang_bins = np.linspace(-np.pi / 2, np.pi / 2, 64)
            rng_bins = np.linspace(0, 20, 64)
            aa, rr = np.meshgrid(ang_bins, rng_bins)
            ra_data = np.exp(-((aa / 0.6) ** 2 + ((rr - 12) / 4) ** 2))
            self.view_widgets["range_angle_resp"].set_data(
                {"data": ra_data, "angle_bins": ang_bins, "range_bins": rng_bins}
            )

        # Micro-Doppler placeholder
        if "micro_doppler_resp" in self.view_widgets:
            time_bins = np.linspace(0, 2, 64)
            vel_bins = np.linspace(-4, 4, 64)
            tt, vv = np.meshgrid(time_bins, vel_bins)
            md_data = np.exp(-((vv - 1.5 * np.sin(2 * np.pi * tt)) ** 2))
            self.view_widgets["micro_doppler_resp"].set_data(
                {"data": md_data, "time_bins": time_bins, "vel_bins": vel_bins}
            )

        # Doppler-Azimuth placeholder
        if "doppler_azimuth_resp" in self.view_widgets:
            vel_bins = np.linspace(-5, 5, 64)
            ang_bins = np.linspace(-np.pi / 2, np.pi / 2, 64)
            vv, aa = np.meshgrid(vel_bins, ang_bins)
            da_data = np.exp(-((vv - 2) ** 2 + (aa / 0.5) ** 2))
            self.view_widgets["doppler_azimuth_resp"].set_data(
                {"data": da_data, "vel_bins": vel_bins, "angle_bins": ang_bins}
            )
