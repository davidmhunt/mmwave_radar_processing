"""Control panel for dataset, config, and view selection."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from mmwave_radar_processing.logging.logger import get_logger


class ControlPanel(QWidget):
    """Left-side control panel for the GUI."""

    dataset_selected = pyqtSignal(str)
    config_selected = pyqtSignal(str)
    params_selected = pyqtSignal(str)
    db_mode_changed = pyqtSignal(bool)

    def __init__(
        self,
        available_views: List[str],
        parent: Optional[QWidget] = None,
        logger=None,
    ) -> None:
        """Initialize the control panel.

        Args:
            available_views: List of view keys available for display.
            parent: Optional parent widget.
            logger: Optional logger instance.
        """
        super().__init__(parent)
        self.logger = logger or get_logger(__name__)
        self.available_views = available_views
        self._build_ui()

    def _build_ui(self) -> None:
        """Build the UI layout."""
        main_layout = QVBoxLayout(self)

        dataset_group = QGroupBox("Dataset")
        dataset_layout = QFormLayout()
        self.dataset_path_edit = QLineEdit()
        dataset_browse = QPushButton("Browse")
        dataset_browse.clicked.connect(self._browse_dataset)
        dataset_layout.addRow("Root:", self._row_with_button(self.dataset_path_edit, dataset_browse))
        dataset_group.setLayout(dataset_layout)
        main_layout.addWidget(dataset_group)

        config_group = QGroupBox("Radar Config")
        config_layout = QFormLayout()
        self.config_edit = QLineEdit()
        config_browse = QPushButton("Browse")
        config_browse.clicked.connect(self._browse_config)
        config_layout.addRow("Config:", self._row_with_button(self.config_edit, config_browse))
        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)

        params_group = QGroupBox("Processor Params")
        params_layout = QFormLayout()
        self.params_edit = QLineEdit()
        params_browse = QPushButton("Browse")
        params_browse.clicked.connect(self._browse_params)
        params_layout.addRow("YAML:", self._row_with_button(self.params_edit, params_browse))
        params_group.setLayout(params_layout)
        main_layout.addWidget(params_group)

        render_group = QGroupBox("Render Options")
        render_layout = QVBoxLayout()
        self.db_checkbox = QCheckBox("Display in dB (20*log10)")
        self.db_checkbox.setChecked(False)
        self.db_checkbox.stateChanged.connect(self._emit_db_mode)
        render_layout.addWidget(self.db_checkbox)
        render_group.setLayout(render_layout)
        main_layout.addWidget(render_group)

        main_layout.addStretch()

    def _row_with_button(self, line_edit: QLineEdit, button: QPushButton) -> QWidget:
        """Build a horizontal row with a line edit and button."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)
        layout.addWidget(button)
        return widget

    def _browse_dataset(self) -> None:
        """Open a file dialog to select a dataset directory."""
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Root")
        if path:
            self.dataset_path_edit.setText(path)
            self.dataset_selected.emit(path)

    def _browse_config(self) -> None:
        """Open a file dialog to select a config file."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Config File", filter="Config (*.cfg);;All Files (*)")
        if path:
            self.config_edit.setText(path)
            self.config_selected.emit(path)

    def _browse_params(self) -> None:
        """Open a file dialog to select a YAML params file."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Params File", filter="YAML (*.yaml *.yml);;All Files (*)")
        if path:
            self.params_edit.setText(path)
            self.params_selected.emit(path)


    def _emit_db_mode(self) -> None:
        """Emit the dB mode state."""
        self.db_mode_changed.emit(self.db_checkbox.isChecked())

    def set_dataset_path(self, path: str) -> None:
        """Set the dataset path text."""
        self.dataset_path_edit.setText(path)

    def set_config_path(self, path: str) -> None:
        """Set the config path text."""
        self.config_edit.setText(path)

    def set_params_path(self, path: str) -> None:
        """Set the params path text."""
        self.params_edit.setText(path)
