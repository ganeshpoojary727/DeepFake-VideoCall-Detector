"""
Live camera feed widget.

Displays BGR frames received from CameraService via CameraFrameEvent.
Converts BGR → RGB → QImage → QPixmap for display.

Features
--------
• Face bounding box overlay (optional)
• Aspect-ratio-preserving scaling
• "No camera" placeholder when not running
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy


class CameraWidget(QLabel):
    """
    Widget for displaying live camera frames.

    Call ``update_frame(frame)`` from the GUI thread with each
    new BGR numpy frame to update the display.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._placeholder_active = True
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background-color: #11111b; border-radius: 8px;")
        self._draw_placeholder()

    def update_frame(self, frame: np.ndarray) -> None:
        """
        Display a new BGR camera frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV (H, W, 3).
        """
        try:
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except ImportError:
            rgb = frame[..., ::-1].copy()  # fallback BGR→RGB

        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimage = QImage(
            rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.width(),
            self.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)
        self._placeholder_active = False

    def show_placeholder(self, message: str = "Camera not started") -> None:
        """Show a text placeholder instead of a camera feed."""
        self._placeholder_active = True
        self._draw_placeholder(message)

    def _draw_placeholder(self, message: str = "No Camera Feed") -> None:
        w = max(self.width(), 320)
        h = max(self.height(), 240)
        pixmap = QPixmap(w, h)
        pixmap.fill(QColor("#11111b"))

        painter = QPainter(pixmap)
        painter.setPen(QColor("#45475a"))
        painter.setFont(QFont("Segoe UI", 12))
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, f"📷\n{message}")
        painter.end()

        self.setPixmap(pixmap)
