
import sys
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QPixmap, QImage
from mmwave_radar_processing.visualization.backends.video_exporter import VideoExporter

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app

def test_video_exporter_flow(qapp):
    # Mock Controller
    mock_controller = MagicMock()
    mock_controller.dataset_model.frame_count.return_value = 5
    mock_controller.last_processed_frame = 0

    # Mock Widget
    mock_widget = MagicMock(spec=QWidget)
    
    # Create valid dummy QImage (RGBA8888)
    width, height = 100, 100
    dummy_img = QImage(width, height, QImage.Format.Format_RGBA8888)
    dummy_img.fill(0) # Black
    
    mock_pixmap = MagicMock(spec=QPixmap)
    mock_pixmap.toImage.return_value = dummy_img
    mock_widget.grab.return_value = mock_pixmap

    # Mock imageio
    with patch("imageio.get_writer") as mock_get_writer:
        mock_writer = MagicMock()
        mock_get_writer.return_value = mock_writer

        # Instantiate Exporter
        exporter = VideoExporter(mock_controller, mock_widget, fps=10)
        
        # Run Export
        # We need a dummy path that passes path validation, but we can mock Path.parent.exists or just use a real tmp path
        # Using a simplistic approach: if exception raised for dir not found, we fix it.
        # But VideoExporter checks parent existence.
        
        with patch("pathlib.Path.exists", return_value=True):
             exporter.export("/tmp/test_video.mp4")

        # Verifications
        assert mock_controller.process_next_frame.call_count >= 5 # 5 frames + restore
        assert mock_widget.grab.call_count == 5
        assert mock_writer.append_data.call_count == 5
        mock_writer.close.assert_called_once()
