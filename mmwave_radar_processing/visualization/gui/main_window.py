"""Main window for the mmWave radar GUI."""

from __future__ import annotations

from typing import Dict, Optional, Type

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from mmwave_radar_processing.logging.logger import get_logger
from mmwave_radar_processing.visualization.backends.processor_registry import (
    ProcessorSpec,
)
from mmwave_radar_processing.visualization.gui.control_panel import ControlPanel
from mmwave_radar_processing.visualization.views.base_view import BaseView


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(
        self,
        controller,
        registry: Dict[str, ProcessorSpec],
        parent: Optional[QWidget] = None,
        logger=None,
    ) -> None:
        """Initialize the main window.

        Args:
            controller: Controller instance coordinating data/processing.
            registry: Mapping of processor keys to processor specifications.
            parent: Optional parent widget.
            logger: Optional logger instance.
        """
        super().__init__(parent)
        self.logger = logger or get_logger(__name__)
        self.controller = controller
        self.registry = registry
        self.view_widgets: Dict[str, BaseView] = {}
        self.active_states: Dict[str, bool] = {key: True for key in registry.keys()}
        self.grid_layout: Optional[QGridLayout] = None
        self.views_container: Optional[QWidget] = None
        self._init_ui()
        self._populate_placeholder_data()

    def _init_ui(self) -> None:
        """Initialize the UI layout."""
        self.setWindowTitle("mmWave Radar Viewer")
        central = QWidget()
        root_layout = QHBoxLayout(central)

        control_panel = ControlPanel(
            available_views=list(self.registry.keys()),
            parent=self,
            logger=self.logger,
        )
        control_panel.views_toggled.connect(self._handle_view_toggle)
        control_panel.dataset_selected.connect(self.controller.load_dataset)
        control_panel.config_selected.connect(self.controller.load_config)
        control_panel.params_selected.connect(lambda path: self.logger.info("Params file selected: %s", path))
        control_panel.db_mode_changed.connect(self._set_db_mode)
        root_layout.addWidget(control_panel, 1)

        self.views_container = QWidget()
        self.grid_layout = QGridLayout(self.views_container)
        self.grid_layout.setContentsMargins(4, 4, 4, 4)
        self.grid_layout.setSpacing(4)

        for key, spec in self.registry.items():
            view_cls: Type[BaseView] = spec.view_cls  # type: ignore
            view_widget = view_cls(parent=self)
            self.view_widgets[key] = view_widget

        self._rebuild_view_grid()
        root_layout.addWidget(self.views_container, 3)

        status = QStatusBar()
        self.setStatusBar(status)
        slider_container = QWidget()
        slider_layout = QHBoxLayout(slider_container)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.valueChanged.connect(self._update_frame_label)
        self.frame_label = QLabel("Frame: 0")
        slider_layout.addWidget(self.play_button)
        slider_layout.addWidget(self.pause_button)
        slider_layout.addWidget(QLabel("Frame"))
        slider_layout.addWidget(self.frame_slider)
        slider_layout.addWidget(self.frame_label)
        status.addPermanentWidget(slider_container, 1)

        self.setCentralWidget(central)
        self.resize(1200, 800)

    def _handle_view_toggle(self, states: Dict[str, bool]) -> None:
        """Show or hide views based on toggle states."""
        self.active_states.update(states)
        self._rebuild_view_grid()

    def _rebuild_view_grid(self) -> None:
        """Rebuild the grid layout based on active views."""
        if not self.grid_layout:
            return
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                self.grid_layout.removeWidget(widget)
                widget.setParent(None)

        active_keys = [k for k, enabled in self.active_states.items() if enabled]
        for idx, key in enumerate(active_keys):
            spec = self.registry[key]
            widget = self.view_widgets[key]
            row = idx // 3
            col = idx % 3
            self.grid_layout.addWidget(widget, row, col)
            widget.show()
        for key in self.registry.keys():
            if key not in active_keys:
                widget = self.view_widgets[key]
                widget.hide()

    def _set_db_mode(self, enabled: bool) -> None:
        """Toggle dB mode across all views."""
        for widget in self.view_widgets.values():
            widget.set_db_mode(enabled)

    def _update_frame_label(self, value: int) -> None:
        """Update the current frame label."""
        self.frame_label.setText(f"Frame: {value}")

    def _populate_placeholder_data(self) -> None:
        """Populate placeholder data so views are immediately visible."""
        import numpy as np

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
