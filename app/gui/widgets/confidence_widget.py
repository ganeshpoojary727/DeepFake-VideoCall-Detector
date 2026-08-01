"""
Animated confidence gauge widget for deepfake detection status.

Renders a smooth arc-based confidence dial with:
• Color gradient: green (REAL) → amber (UNCERTAIN) → red (FAKE)
• Animated value transitions using QPropertyAnimation
• Label showing percentage and verdict
• Glow effect on critical (FAKE) state
"""

from __future__ import annotations

import math
from typing import Optional

from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    Qt,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QColor,
    QConicalGradient,
    QFont,
    QPainter,
    QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import QSizePolicy, QWidget


def _lerp_color(c1: QColor, c2: QColor, t: float) -> QColor:
    """Linearly interpolate between two QColors."""
    r = int(c1.red() + (c2.red() - c1.red()) * t)
    g = int(c1.green() + (c2.green() - c1.green()) * t)
    b = int(c1.blue() + (c2.blue() - c1.blue()) * t)
    return QColor(r, g, b)


class ConfidenceWidget(QWidget):
    """
    Circular confidence gauge with smooth animation.

    Signals
    -------
    valueChanged(float)
        Emitted when the displayed value changes.

    Parameters
    ----------
    parent : QWidget | None
        Parent widget.
    """

    valueChanged = pyqtSignal(float)

    # Color stops
    _COLOR_REAL = QColor("#a6e3a1")       # green
    _COLOR_UNCERTAIN = QColor("#f9e2af")  # amber
    _COLOR_FAKE = QColor("#f38ba8")       # red
    _COLOR_BG = QColor("#1e1e2e")
    _COLOR_TRACK = QColor("#313244")

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._value: float = 0.0          # 0.0 – 1.0 (displayed)
        self._target_value: float = 0.0   # animation target
        self._verdict: str = "Waiting..."
        self._verdict_color: str = "#6c7086"

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 200)

        # Animation
        self._animation = QPropertyAnimation(self, b"gauge_value", self)
        self._animation.setDuration(600)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    # ── Qt Property (animatable) ──────────────

    def _get_gauge_value(self) -> float:
        return self._value

    def _set_gauge_value(self, value: float) -> None:
        self._value = float(value)
        self._update_verdict()
        self.update()
        self.valueChanged.emit(self._value)

    gauge_value = pyqtProperty(float, _get_gauge_value, _set_gauge_value)

    # ── Public API ────────────────────────────

    def set_value(self, value: float) -> None:
        """
        Animate the gauge to a new value.

        Parameters
        ----------
        value : float
            Target confidence (0.0 = definitely REAL, 1.0 = definitely FAKE).
        """
        self._target_value = max(0.0, min(1.0, value))
        self._animation.stop()
        self._animation.setStartValue(self._value)
        self._animation.setEndValue(self._target_value)
        self._animation.start()

    def set_verdict(self, verdict: str, color: str = "#6c7086") -> None:
        """Set the verdict text and color below the gauge."""
        self._verdict = verdict
        self._verdict_color = color
        self.update()

    # ── Paint ─────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        side = min(w, h) - 20
        x = (w - side) / 2
        y = (h - side) / 2

        rect = QRectF(x, y, side, side)

        # Background track arc
        pen = QPen(self._COLOR_TRACK)
        pen.setWidth(14)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, 225 * 16, -270 * 16)

        # Value arc
        arc_color = self._arc_color()
        pen.setColor(arc_color)
        painter.setPen(pen)
        span = int(-270 * 16 * self._value)
        painter.drawArc(rect, 225 * 16, span)

        # Center text — percentage
        painter.setPen(QColor("#cdd6f4"))
        font = QFont("Segoe UI", int(side * 0.18), QFont.Weight.Bold)
        painter.setFont(font)
        pct_text = f"{int(self._value * 100)}%"
        painter.drawText(rect.adjusted(0, -side * 0.06, 0, 0), Qt.AlignmentFlag.AlignCenter, pct_text)

        # Center text — verdict
        painter.setPen(QColor(self._verdict_color))
        font2 = QFont("Segoe UI", int(side * 0.075))
        painter.setFont(font2)
        painter.drawText(
            rect.adjusted(0, side * 0.18, 0, 0),
            Qt.AlignmentFlag.AlignCenter,
            self._verdict,
        )

        painter.end()

    def _arc_color(self) -> QColor:
        """Return a color interpolated between green → amber → red."""
        v = self._value
        if v < 0.3:
            return self._COLOR_REAL
        elif v < 0.7:
            t = (v - 0.3) / 0.4
            return _lerp_color(self._COLOR_REAL, self._COLOR_UNCERTAIN, t)
        else:
            t = (v - 0.7) / 0.3
            return _lerp_color(self._COLOR_UNCERTAIN, self._COLOR_FAKE, t)

    def _update_verdict(self) -> None:
        v = self._value
        if v >= 0.7:
            self._verdict = "⚠ FAKE"
            self._verdict_color = "#f38ba8"
        elif v <= 0.3:
            self._verdict = "✓ REAL"
            self._verdict_color = "#a6e3a1"
        else:
            self._verdict = "~ UNCERTAIN"
            self._verdict_color = "#f9e2af"
