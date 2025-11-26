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

        self.last_payload: Optional[Dict[str, Any]] = None

    def set_data(self, payload: Any) -> None:
        """Update the view with new data.

        Args:
            payload: Dictionary emitted by the controller for this view.
        """
        self.last_payload = payload
        self.update_view(payload)

    def update_view(self, payload: Any) -> None:
        """Update the actual view widgets. Must be implemented by subclasses.

        Args:
            payload: Dictionary containing data and metadata.
        """
        raise NotImplementedError("Subclasses must implement update_view.")

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
        if self.convert_to_db != enabled:
            self.convert_to_db = enabled
            if self.last_payload is not None:
                self.update_view(self.last_payload)

    def set_colormap(self, name: str = "viridis") -> None:
        """Set the colormap for the view.

        Args:
            name: Name of the colormap (e.g., 'viridis', 'magma', 'jet').
        """
        pass  # Subclasses should implement if applicable
