"""Base view class for GUI visualization components."""

from typing import Any, Dict, Optional

from PyQt6.QtWidgets import QSizePolicy, QWidget

from mmwave_radar_processing.logging.logger import get_logger


class BaseView(QWidget):
    """Abstract base class for all visualization views."""

    def __init__(self, parent: Optional[QWidget] = None, logger=None) -> None:
        """Initialize the base view.

        Args:
            parent: Optional parent widget.
            logger: Optional logger instance; defaults to a namespaced logger.
        """
        super().__init__(parent)
        self.logger = logger or get_logger(__name__)
        self.convert_to_db = False
        self.setMinimumSize(200, 150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(self, payload: Any) -> None:
        """Update the view with new data.

        Args:
            payload: Dictionary emitted by the controller for this view.
        """
        raise NotImplementedError("Subclasses must implement set_data.")

    def update_params(self, **kwargs: Any) -> None:
        """Update runtime parameters for the view.

        Args:
            **kwargs: Arbitrary keyword arguments for the view configuration.
        """
        self.logger.debug("update_params called with %s", kwargs)

    def set_db_mode(self, enabled: bool) -> None:
        """Set whether the view should render in dB.

        Args:
            enabled: True to convert data to dB before rendering.
        """
        self.convert_to_db = enabled
