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
from mmwave_radar_processing.visualization.gui.processor_view_panel import ProcessorViewPanel

class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(
        self,
        controller,
        registry: Dict[str, ProcessorSpec],
        dataset_path: Optional[str] = None,
        config_path: Optional[str] = None,
        params_path: Optional[str] = None,
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
        self.dataset_path = dataset_path
        self.config_path = config_path
        self.params_path = params_path
        self.processor_view_panel: Optional[ProcessorViewPanel] = None
        self._init_ui()
        self.controller.process_next_frame(0)

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
        
        if self.dataset_path:
            control_panel.set_dataset_path(self.dataset_path)
        if self.config_path:
            control_panel.set_config_path(self.config_path)
        if self.params_path:
            control_panel.set_params_path(self.params_path)
        root_layout.addWidget(control_panel, 1)

        # Initialize Processor View Panel
        self.processor_view_panel = ProcessorViewPanel(
            registry=self.registry,
            parent=self,
            logger=self.logger
        )
        root_layout.addWidget(self.processor_view_panel, 3)
        
        # Connect signals to the panel
        control_panel.db_mode_changed.connect(self.processor_view_panel.set_db_mode)

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
        try:
            self.controller.dataset_loaded.connect(self._set_frame_count)
            # Route view updates to the panel
            self.controller.view_update.connect(self.processor_view_panel.handle_view_update)
            self.controller.frame_processed.connect(self._update_slider_from_controller)
            # Connect slider to controller processing
            self.frame_slider.valueChanged.connect(self.controller.process_next_frame)
            
            # Connect playback controls
            self.play_button.clicked.connect(self.controller.start)
            self.pause_button.clicked.connect(self.controller.stop)
            
            # Check if dataset is already loaded
            if self.controller.dataset_model:
                count = self.controller.dataset_model.frame_count()
                if count > 0:
                    self._set_frame_count(count)
                else:
                    self.processor_view_panel.populate_placeholder_data()
            else:
                self.processor_view_panel.populate_placeholder_data()
                    
        except Exception as exc:
            self.logger.warning("Could not connect signals: %s", exc)

    def _handle_view_toggle(self, states: Dict[str, bool]) -> None:
        """Show or hide views based on toggle states.
        
        Note: With the new dropdown system, this might be redundant for visibility,
        but we can keep it if we want to disable processing for unchecked views later.
        For now, we just log it.
        """
        self.logger.debug("View toggles changed: %s", states)

    def _update_frame_label(self, value: int) -> None:
        """Update the current frame label."""
        self.frame_label.setText(f"Frame: {value}")

    def _set_frame_count(self, count: int) -> None:
        """Update slider maximum and label based on dataset frame count.

        Args:
            count: Number of frames in the loaded dataset.
        """
        maximum = max(count - 1, 0)
        self.frame_slider.setMaximum(maximum)
        if self.frame_slider.value() > maximum:
            self.frame_slider.setValue(maximum)
        self._update_frame_label(self.frame_slider.value())

    def _update_slider_from_controller(self, frame_idx: int) -> None:
        """Update slider position from controller (e.g. during playback)."""
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_idx)
        self.frame_slider.blockSignals(False)
        self._update_frame_label(frame_idx)
