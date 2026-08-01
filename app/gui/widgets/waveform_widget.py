"""
Real-time audio waveform display widget.

Renders a scrolling waveform from RMS level samples published by
the MicrophoneService via AudioLevelEvent.  Updated via Qt signals
from the GUI's QTimer dispatch loop.
"""

from __future__ import annotations

from collections import deque
from typing import Deque

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget


class WaveformWidget(QWidget):
    """
    Scrolling real-time audio waveform visualization.

    Call ``push_level(rms)`` from the GUI thread with each new
    audio level sample to update the display.

    Parameters
    ----------
    history_size : int
        Number of amplitude samples to display.
    color : str
        Hex color for the waveform line.
    """

    def __init__(
        self,
        parent=None,
        history_size: int = 100,
        color: str = "#89b4fa",
    ) -> None:
        super().__init__(parent)
        self._history: Deque[float] = deque([0.0] * history_size, maxlen=history_size)
        self._color = QColor(color)
        self._bg = QColor("#181825")
        self._grid = QColor("#313244")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(80)

    def push_level(self, level: float) -> None:
        """
        Add a new amplitude sample and redraw.

        Parameters
        ----------
        level : float
            RMS amplitude (0.0 – 1.0).
        """
        self._history.append(max(0.0, min(1.0, level)))
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        mid = h / 2

        # Background
        painter.fillRect(0, 0, w, h, self._bg)

        # Center line
        pen = QPen(self._grid)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(0, int(mid), w, int(mid))

    
        if not self._history:
            painter.end()
            return

        samples = list(self._history)
        n = len(samples)
        step = w / max(n - 1, 1)

        # Upper path
        pen = QPen(self._color)
        pen.setWidth(2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        path = QPainterPath()
        for i, amp in enumerate(samples):
            x = i * step
            y_top = mid - amp * (h * 0.45)
            if i == 0:
                path.moveTo(x, y_top)
            else:
                path.lineTo(x, y_top)
        painter.drawPath(path)

        # Mirror (bottom)
        path2 = QPainterPath()
        for i, amp in enumerate(samples):
            x = i * step
            y_bot = mid + amp * (h * 0.45)
            if i == 0:
                path2.moveTo(x, y_bot)
            else:
                path2.lineTo(x, y_bot)
        painter.drawPath(path2)

        painter.end()
