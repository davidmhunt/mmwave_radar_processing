"""Video exporter for the mmWave radar dataset viewer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import imageio
import numpy as np
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication, QWidget

# Try to import QImage.Format for type hinting or usage if needed,
# but usually QImage.Format.Format_RGBA8888 is accessed directly on the class or instance.
# We'll use the string 'libx264' for the codec.

class VideoExporter:
    """Orchestrates video export from a PyQt6 widget."""

    def __init__(
        self,
        controller: Any,
        target_widget: QWidget,
        fps: int = 20,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialize the video exporter.

        Args:
            controller: The processor controller instance.
            target_widget: The widget to capture.
            fps: Frames per second for the output video.
            logger: Optional logger instance.
        """
        self.controller = controller
        self.target_widget = target_widget
        self.fps = fps
        self.logger = logger or logging.getLogger(__name__)

    def export(self, output_path: str) -> None:
        """Run the export process.

        Args:
            output_path: Path to save the video file (e.g. .mp4).
        """
        output_file = Path(output_path)
        if not output_file.parent.exists():
            raise FileNotFoundError(f"Output directory does not exist: {output_file.parent}")

        if not self.controller.dataset_model:
            self.logger.error("No dataset loaded to export.")
            return

        total_frames = self.controller.dataset_model.frame_count()
        if total_frames == 0:
            self.logger.warning("Dataset has no frames.")
            return

        original_frame_idx = self.controller.last_processed_frame
        self.logger.info(
            "Starting export to %s (%d frames, %d FPS)", output_file, total_frames, self.fps
        )

        try:
            writer = imageio.get_writer(output_file, fps=self.fps, codec="libx264", quality=8)
            
            for i in range(total_frames):
                # Update logic
                self.controller.process_next_frame(i)
                
                # Force UI update
                QApplication.processEvents()
                
                # Capture frame
                frame_img = self._grab_frame()
                writer.append_data(frame_img)
                
                if i % 10 == 0:
                   self.logger.debug("Exported frame %d/%d", i, total_frames)

            writer.close()
            self.logger.info("Export completed successfully.")

        except Exception as exc:
            self.logger.error("Export failed: %s", exc)
            raise exc

        finally:
            # Restore state
            if original_frame_idx >= 0:
                self.controller.process_next_frame(original_frame_idx)

    def _grab_frame(self) -> np.ndarray:
        """Capture the current state of the target widget as an RGB array."""
        # Grab the pixmap
        pixmap = self.target_widget.grab()
        qimage = pixmap.toImage()
        
        # Ensure format is RGBA8888
        if qimage.format() != QImage.Format.Format_RGBA8888:
            qimage = qimage.convertToFormat(QImage.Format.Format_RGBA8888)

        # Get pointer to bits
        width = qimage.width()
        height = qimage.height()
        
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        
        # Create numpy array from buffer
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        
        # Discard Alpha channel, keep RGB
        return arr[..., :3]
