"""
Main GUI Window — Clean, ultra-responsive PyQt6 interface.

Architecture
────────────
• Two-tab layout: Live Call Monitor + Manual Media Tester
• Catppuccin Mocha dark theme for premium aesthetics
• All updates via PyQt6 signal/slot — zero polling overhead
• 60 FPS rendering with smooth animations
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QColor, QDragEnterEvent, QDropEvent, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
# Catppuccin Mocha Palette
# ──────────────────────────────────────────────

P = {
    "base":     "#1e1e2e",
    "mantle":   "#181825",
    "crust":    "#11111b",
    "surface0": "#313244",
    "surface1": "#45475a",
    "surface2": "#585b70",
    "overlay0": "#6c7086",
    "overlay1": "#7f849c",
    "text":     "#cdd6f4",
    "subtext0": "#a6adc8",
    "subtext1": "#bac2de",
    "blue":     "#89b4fa",
    "green":    "#a6e3a1",
    "yellow":   "#f9e2af",
    "red":      "#f38ba8",
    "peach":    "#fab387",
    "mauve":    "#cba6f7",
    "lavender": "#b4befe",
    "teal":     "#94e2d5",
    "sky":      "#89dceb",
    "pink":     "#f5c2e7",
    "rosewater":"#f5e0dc",
}


def _build_stylesheet() -> str:
    """Generate the full application stylesheet."""
    return f"""
    QMainWindow {{
        background-color: {P["base"]};
    }}
    QWidget {{
        background-color: {P["base"]};
        color: {P["text"]};
        font-family: "Segoe UI", "Inter", Arial, sans-serif;
        font-size: 13px;
    }}
    QTabWidget::pane {{
        border: 1px solid {P["surface0"]};
        border-radius: 8px;
        background-color: {P["base"]};
        padding: 4px;
    }}
    QTabBar::tab {{
        background-color: {P["mantle"]};
        color: {P["subtext0"]};
        border: 1px solid {P["surface0"]};
        border-bottom: none;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        padding: 10px 28px;
        margin-right: 2px;
        font-size: 13px;
        font-weight: bold;
    }}
    QTabBar::tab:selected {{
        background-color: {P["surface0"]};
        color: {P["blue"]};
        border-bottom: 2px solid {P["blue"]};
    }}
    QTabBar::tab:hover {{
        background-color: {P["surface1"]};
        color: {P["text"]};
    }}
    QPushButton {{
        background-color: {P["surface0"]};
        color: {P["text"]};
        border: 1px solid {P["surface1"]};
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 13px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: {P["surface1"]};
        border-color: {P["blue"]};
    }}
    QPushButton:pressed {{
        background-color: {P["surface2"]};
    }}
    QPushButton:disabled {{
        color: {P["overlay0"]};
        border-color: {P["surface0"]};
        background-color: {P["mantle"]};
    }}
    QPushButton#btnStart {{
        background-color: {P["green"]};
        color: {P["crust"]};
        border: none;
        font-size: 14px;
    }}
    QPushButton#btnStart:hover {{
        background-color: #c3f0b8;
    }}
    QPushButton#btnStop {{
        background-color: {P["red"]};
        color: {P["crust"]};
        border: none;
        font-size: 14px;
    }}
    QPushButton#btnStop:hover {{
        background-color: #f7a8c4;
    }}
    QProgressBar {{
        background-color: {P["mantle"]};
        border: 1px solid {P["surface0"]};
        border-radius: 6px;
        text-align: center;
        color: {P["text"]};
        font-size: 11px;
        font-weight: bold;
        height: 22px;
    }}
    QProgressBar::chunk {{
        border-radius: 5px;
    }}
    QTextEdit {{
        background-color: {P["mantle"]};
        color: {P["subtext0"]};
        border: 1px solid {P["surface0"]};
        border-radius: 8px;
        padding: 8px;
        font-family: "Cascadia Code", "Consolas", monospace;
        font-size: 11px;
    }}
    QStatusBar {{
        background-color: {P["crust"]};
        color: {P["overlay0"]};
        font-size: 11px;
        padding: 3px 8px;
    }}
    """


# ──────────────────────────────────────────────
# Score Badge Widget
# ──────────────────────────────────────────────


class ScoreBadge(QFrame):
    """Large visual badge showing REAL / DEEPFAKE with color."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setStyleSheet(
            f"background-color: {P['mantle']}; border: 1px solid {P['surface0']}; border-radius: 10px; padding: 12px;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._label = QLabel("WAITING FOR DETECTION")
        self._label.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(f"color: {P['overlay0']}; background: transparent; border: none;")
        layout.addWidget(self._label)

        self._score_label = QLabel("Score: —")
        self._score_label.setFont(QFont("Segoe UI", 12))
        self._score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._score_label.setStyleSheet(f"color: {P['subtext0']}; background: transparent; border: none;")
        layout.addWidget(self._score_label)

    def set_result(self, prediction: str, score: float) -> None:
        """Update the badge with a detection result."""
        self._label.setText(prediction)
        self._score_label.setText(f"Confidence Score: {score:.1%}")

        if prediction == "REAL":
            self._label.setStyleSheet(f"color: {P['green']}; background: transparent; border: none; font-size: 24px; font-weight: bold;")
            self.setStyleSheet(
                f"background-color: rgba(166, 227, 161, 0.1); "
                f"border: 2px solid {P['green']}; border-radius: 10px;"
            )
        elif prediction == "DEEPFAKE":
            self._label.setStyleSheet(f"color: {P['red']}; background: transparent; border: none; font-size: 24px; font-weight: bold;")
            self.setStyleSheet(
                f"background-color: rgba(243, 139, 168, 0.1); "
                f"border: 2px solid {P['red']}; border-radius: 10px;"
            )
        else:
            self._label.setStyleSheet(f"color: {P['overlay0']}; background: transparent; border: none; font-size: 20px; font-weight: bold;")
            self.setStyleSheet(
                f"background-color: {P['mantle']}; border: 1px solid {P['surface0']}; border-radius: 10px;"
            )

    def reset(self) -> None:
        """Reset badge to waiting state."""
        self._label.setText("WAITING FOR DETECTION")
        self._label.setStyleSheet(f"color: {P['overlay0']}; background: transparent; border: none; font-size: 20px; font-weight: bold;")
        self._score_label.setText("Score: —")
        self.setStyleSheet(
            f"background-color: {P['mantle']}; border: 1px solid {P['surface0']}; border-radius: 10px;"
        )


# ──────────────────────────────────────────────
# Main Window
# ──────────────────────────────────────────────


class MainWindow(QMainWindow):
    """
    Main application window for DeepFake Video Call Detector.

    Two-tab layout:
      Tab 1 — Live Call Monitor
      Tab 2 — Manual Media Tester
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DeepFake Video Call Detector v4.0")
        self.setMinimumSize(1000, 680)
        self.resize(1100, 720)
        self.setStyleSheet(_build_stylesheet())

        # ── Detection Service (lazy init) ─────
        self._detection_service = None
        self._manual_tester = None

        # ── Build UI ──────────────────────────
        self._build_ui()

        logger.info("MainWindow initialized")

    def _build_ui(self) -> None:
        """Build the complete UI layout."""
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        # ── Header ────────────────────────────
        header = self._build_header()
        root_layout.addWidget(header)

        # ── Tab Widget ────────────────────────
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_live_monitor_tab(), "🔴  Live Call Monitor")
        self._tabs.addTab(self._build_manual_tester_tab(), "📁  Manual Media Tester")
        root_layout.addWidget(self._tabs, stretch=1)

        # ── Status Bar ────────────────────────
        self.statusBar().showMessage(
            f"Device: {settings.DEVICE}  │  "
            f"Audio: WASAPI Loopback  │  "
            f"Fusion: {settings.AUDIO_WEIGHT:.0%} Audio + {settings.VIDEO_WEIGHT:.0%} Video  │  "
            f"Ready"
        )

    # ══════════════════════════════════════════
    # HEADER
    # ══════════════════════════════════════════

    def _build_header(self) -> QFrame:
        """Build the application header bar."""
        frame = QFrame()
        frame.setStyleSheet(
            f"background-color: {P['mantle']}; "
            f"border: 1px solid {P['surface0']}; "
            f"border-radius: 10px; padding: 12px;"
        )

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(16, 8, 16, 8)

        # App title
        title = QLabel("🛡️  DeepFake Video Call Detector")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {P['blue']}; border: none;")
        layout.addWidget(title)

        layout.addStretch()

        # Version badge
        version = QLabel("v4.0")
        version.setStyleSheet(
            f"color: {P['mauve']}; background-color: {P['surface0']}; "
            f"border-radius: 10px; padding: 4px 10px; font-size: 11px; font-weight: bold;"
        )
        layout.addWidget(version)

        return frame

    # ══════════════════════════════════════════
    # TAB 1: LIVE CALL MONITOR
    # ══════════════════════════════════════════

    def _build_live_monitor_tab(self) -> QWidget:
        """Build the live call monitoring tab."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(12)

        # ── App Status Banner ─────────────────
        self._app_status = QLabel("● Idle — No video call detected")
        self._app_status.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self._app_status.setStyleSheet(
            f"color: {P['overlay0']}; background-color: {P['mantle']}; "
            f"border: 1px solid {P['surface0']}; border-radius: 8px; padding: 12px;"
        )
        self._app_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._app_status)

        # ── Controls Row ──────────────────────
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(12)

        self._btn_start = QPushButton("▶  Start Monitoring")
        self._btn_start.setObjectName("btnStart")
        self._btn_start.setMinimumHeight(44)
        self._btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_start.clicked.connect(self._on_start)
        ctrl_layout.addWidget(self._btn_start)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setObjectName("btnStop")
        self._btn_stop.setMinimumHeight(44)
        self._btn_stop.setEnabled(False)
        self._btn_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_stop.clicked.connect(self._on_stop)
        ctrl_layout.addWidget(self._btn_stop)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # ── Buffer Progress ───────────────────
        buffer_frame = self._make_card("20-Second Buffer Progress")
        buffer_layout = buffer_frame.layout()

        self._buffer_bar = QProgressBar()
        self._buffer_bar.setRange(0, 100)
        self._buffer_bar.setValue(0)
        self._buffer_bar.setFormat("%v%  (%p% complete)")
        self._buffer_bar.setStyleSheet(
            self._buffer_bar.styleSheet()
            + f" QProgressBar::chunk {{ background-color: {P['blue']}; }}"
        )
        buffer_layout.addWidget(self._buffer_bar)
        layout.addWidget(buffer_frame)

        # ── Scores Row ────────────────────────
        scores_row = QHBoxLayout()
        scores_row.setSpacing(12)

        # Audio Score Card
        audio_card = self._make_card("🎤  Audio Confidence")
        self._audio_bar = QProgressBar()
        self._audio_bar.setRange(0, 100)
        self._audio_bar.setValue(0)
        self._audio_bar.setFormat("Audio: %v%")
        self._audio_bar.setStyleSheet(
            self._audio_bar.styleSheet()
            + f" QProgressBar::chunk {{ background-color: {P['teal']}; }}"
        )
        audio_card.layout().addWidget(self._audio_bar)
        scores_row.addWidget(audio_card)

        # Video Score Card
        video_card = self._make_card("🎬  Video Confidence")
        self._video_bar = QProgressBar()
        self._video_bar.setRange(0, 100)
        self._video_bar.setValue(0)
        self._video_bar.setFormat("Video: %v%")
        self._video_bar.setStyleSheet(
            self._video_bar.styleSheet()
            + f" QProgressBar::chunk {{ background-color: {P['mauve']}; }}"
        )
        video_card.layout().addWidget(self._video_bar)
        scores_row.addWidget(video_card)

        layout.addLayout(scores_row)

        # ── Combined Score Badge ──────────────
        self._score_badge = ScoreBadge()
        layout.addWidget(self._score_badge)

        # ── Activity Log ──────────────────────
        log_frame = self._make_card("📋  Activity Log")
        self._log_box = QTextEdit()
        self._log_box.setReadOnly(True)
        self._log_box.setMaximumHeight(140)
        self._log_box.setPlaceholderText("Waiting for activity...")
        log_frame.layout().addWidget(self._log_box)
        layout.addWidget(log_frame)

        return page

    # ══════════════════════════════════════════
    # TAB 2: MANUAL MEDIA TESTER
    # ══════════════════════════════════════════

    def _build_manual_tester_tab(self) -> QWidget:
        """Build the manual file testing tab."""
        page = QWidget()
        page.setAcceptDrops(True)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 12, 8, 8)
        layout.setSpacing(16)

        # ── Drag & Drop Zone ──────────────────
        self._drop_zone = QLabel(
            "📂  Drag & Drop Audio/Video File Here\n\n"
            "Supported: .mp4, .avi, .wav, .mp3, .mkv, .mov, .flac"
        )
        self._drop_zone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drop_zone.setMinimumHeight(120)
        self._drop_zone.setStyleSheet(
            f"background-color: {P['mantle']}; "
            f"border: 2px dashed {P['surface1']}; "
            f"border-radius: 12px; "
            f"color: {P['overlay1']}; "
            f"font-size: 14px; "
            f"padding: 20px;"
        )
        layout.addWidget(self._drop_zone)

        # ── File Selection Row ────────────────
        file_row = QHBoxLayout()
        file_row.setSpacing(12)

        self._btn_browse = QPushButton("📁  Browse Audio/Video File...")
        self._btn_browse.setMinimumHeight(44)
        self._btn_browse.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_browse.clicked.connect(self._on_browse_file)
        file_row.addWidget(self._btn_browse)

        self._file_path_label = QLabel("No file selected")
        self._file_path_label.setStyleSheet(
            f"color: {P['subtext0']}; font-size: 12px; padding: 8px;"
        )
        file_row.addWidget(self._file_path_label, stretch=1)

        layout.addLayout(file_row)

        # ── Analyze Button ────────────────────
        self._btn_analyze = QPushButton("🔍  Analyze Media File")
        self._btn_analyze.setMinimumHeight(48)
        self._btn_analyze.setEnabled(False)
        self._btn_analyze.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_analyze.setStyleSheet(
            f"background-color: {P['blue']}; color: {P['crust']}; "
            f"border: none; border-radius: 8px; font-size: 14px; font-weight: bold;"
        )
        self._btn_analyze.clicked.connect(self._on_analyze_file)
        layout.addWidget(self._btn_analyze)

        # ── Results Card ──────────────────────
        results_frame = self._make_card("📊  Analysis Results")
        results_layout = results_frame.layout()

        # Audio result
        self._mt_audio_label = QLabel("Audio Confidence: —")
        self._mt_audio_label.setFont(QFont("Segoe UI", 13))
        self._mt_audio_label.setStyleSheet(f"color: {P['teal']}; border: none;")
        results_layout.addWidget(self._mt_audio_label)

        # Video result
        self._mt_video_label = QLabel("Video Confidence: —")
        self._mt_video_label.setFont(QFont("Segoe UI", 13))
        self._mt_video_label.setStyleSheet(f"color: {P['mauve']}; border: none;")
        results_layout.addWidget(self._mt_video_label)

        # Combined result
        self._mt_combined_label = QLabel("Combined Score: —")
        self._mt_combined_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self._mt_combined_label.setStyleSheet(f"color: {P['text']}; border: none;")
        results_layout.addWidget(self._mt_combined_label)

        # Prediction badge
        self._mt_badge = ScoreBadge()
        results_layout.addWidget(self._mt_badge)

        # Analysis time
        self._mt_time_label = QLabel("")
        self._mt_time_label.setStyleSheet(f"color: {P['overlay0']}; font-size: 11px; border: none;")
        results_layout.addWidget(self._mt_time_label)

        layout.addWidget(results_frame, stretch=1)

        # ── Internal state ────────────────────
        self._selected_file: Optional[Path] = None

        return page

    # ══════════════════════════════════════════
    # DRAG & DROP SUPPORT
    # ══════════════════════════════════════════

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept drag events with file URLs."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._drop_zone.setStyleSheet(
                f"background-color: {P['mantle']}; "
                f"border: 2px dashed {P['blue']}; "
                f"border-radius: 12px; "
                f"color: {P['blue']}; "
                f"font-size: 14px; "
                f"padding: 20px;"
            )

    def dropEvent(self, event: QDropEvent) -> None:
        """Handle dropped files."""
        self._drop_zone.setStyleSheet(
            f"background-color: {P['mantle']}; "
            f"border: 2px dashed {P['surface1']}; "
            f"border-radius: 12px; "
            f"color: {P['overlay1']}; "
            f"font-size: 14px; "
            f"padding: 20px;"
        )
        urls = event.mimeData().urls()
        if urls:
            file_path = Path(urls[0].toLocalFile())
            self._set_selected_file(file_path)

    # ══════════════════════════════════════════
    # LIVE MONITOR SLOTS
    # ══════════════════════════════════════════

    def _on_start(self) -> None:
        """Start monitoring button handler."""
        if self._detection_service is None:
            from app.services.detection_service import DetectionService
            self._detection_service = DetectionService(parent=self)
            # Connect signals
            self._detection_service.call_detected.connect(self._on_call_detected)
            self._detection_service.buffer_progress.connect(self._on_buffer_progress)
            self._detection_service.analysis_result.connect(self._on_analysis_result)
            self._detection_service.status_message.connect(self._on_status_message)

        self._detection_service.start_monitoring()
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._log("Monitoring started — scanning for video calls...")

    def _on_stop(self) -> None:
        """Stop monitoring button handler."""
        if self._detection_service:
            self._detection_service.stop_monitoring()

        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._score_badge.reset()
        self._buffer_bar.setValue(0)
        self._audio_bar.setValue(0)
        self._video_bar.setValue(0)
        self._app_status.setText("● Idle — Monitoring stopped")
        self._app_status.setStyleSheet(
            f"color: {P['overlay0']}; background-color: {P['mantle']}; "
            f"border: 1px solid {P['surface0']}; border-radius: 8px; padding: 12px;"
        )
        self._log("Monitoring stopped")

    def _on_call_detected(self, app_name: str) -> None:
        """Handle call detected/lost signal."""
        if app_name:
            self._app_status.setText(f"🟢  Call Detected: {app_name}")
            self._app_status.setStyleSheet(
                f"color: {P['green']}; background-color: rgba(166, 227, 161, 0.08); "
                f"border: 1px solid {P['green']}; border-radius: 8px; padding: 12px;"
                f" font-size: 13px; font-weight: bold;"
            )
        else:
            self._app_status.setText("● Monitoring Active — No video call detected")
            self._app_status.setStyleSheet(
                f"color: {P['overlay1']}; background-color: {P['mantle']}; "
                f"border: 1px solid {P['surface0']}; border-radius: 8px; padding: 12px;"
            )

    def _on_buffer_progress(self, percent: int) -> None:
        """Handle buffer progress signal."""
        self._buffer_bar.setValue(percent)

    def _on_analysis_result(self, result: dict) -> None:
        """Handle analysis result signal."""
        audio_pct = int(result.get("audio_score", 0) * 100)
        video_pct = int(result.get("video_score", 0) * 100)
        combined = result.get("combined_score", 0)
        prediction = result.get("prediction", "UNKNOWN")

        self._audio_bar.setValue(audio_pct)
        self._video_bar.setValue(video_pct)

        # Color-code audio bar
        audio_color = P["red"] if audio_pct >= 50 else P["green"]
        self._audio_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {audio_color}; border-radius: 5px; }}"
        )

        # Color-code video bar
        video_color = P["red"] if video_pct >= 50 else P["green"]
        self._video_bar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {video_color}; border-radius: 5px; }}"
        )

        # Update score badge
        self._score_badge.set_result(prediction, combined)

        # Log
        latency = result.get("latency_ms", 0)
        state = result.get("state", "")
        self._log(
            f"[{state}] {prediction} — Combined: {combined:.1%} "
            f"(Audio: {audio_pct}%, Video: {video_pct}%) [{latency:.0f}ms]"
        )

        # Status bar
        self.statusBar().showMessage(
            f"Detection: {prediction} ({combined:.1%})  │  "
            f"Audio: {audio_pct}%  │  Video: {video_pct}%  │  "
            f"Latency: {latency:.0f}ms"
        )

    def _on_status_message(self, msg: str) -> None:
        """Handle status message signal."""
        self._log(msg)

    # ══════════════════════════════════════════
    # MANUAL TESTER SLOTS
    # ══════════════════════════════════════════

    def _on_browse_file(self) -> None:
        """Open file dialog to select a media file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio or Video File",
            "",
            "Media Files (*.mp4 *.avi *.mkv *.mov *.webm *.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*)",
        )
        if file_path:
            self._set_selected_file(Path(file_path))

    def _set_selected_file(self, file_path: Path) -> None:
        """Set the selected file for analysis."""
        self._selected_file = file_path
        self._file_path_label.setText(f"📄 {file_path.name}")
        self._file_path_label.setToolTip(str(file_path))
        self._btn_analyze.setEnabled(True)
        self._drop_zone.setText(f"✅  Selected: {file_path.name}")
        self._mt_badge.reset()

    def _on_analyze_file(self) -> None:
        """Analyze the selected file."""
        if self._selected_file is None:
            return

        self._btn_analyze.setEnabled(False)
        self._btn_analyze.setText("⏳  Analyzing...")
        self._mt_badge.reset()
        self._mt_audio_label.setText("Audio Confidence: analyzing...")
        self._mt_video_label.setText("Video Confidence: analyzing...")
        self._mt_combined_label.setText("Combined Score: analyzing...")

        # Force UI update
        QApplication.processEvents()

        # Run analysis
        if self._manual_tester is None:
            from app.services.manual_tester import ManualMediaTester
            self._manual_tester = ManualMediaTester()

        try:
            result = self._manual_tester.analyze_file(self._selected_file)

            if "error" in result:
                self._mt_combined_label.setText(f"Error: {result['error']}")
                self._btn_analyze.setEnabled(True)
                self._btn_analyze.setText("🔍  Analyze Media File")
                return

            # Update results
            audio_score = result.get("audio_score", 0)
            video_score = result.get("video_score", 0)
            combined = result.get("combined_score", 0)
            prediction = result.get("prediction", "UNKNOWN")
            elapsed = result.get("analysis_time_ms", 0)

            self._mt_audio_label.setText(f"🎤  Audio Confidence: {audio_score:.1%}")
            self._mt_video_label.setText(f"🎬  Video Confidence: {video_score:.1%}")
            self._mt_combined_label.setText(
                f"Combined Score: {combined:.1%}  "
                f"({settings.AUDIO_WEIGHT:.0%} Audio + {settings.VIDEO_WEIGHT:.0%} Video)"
            )
            self._mt_badge.set_result(prediction, combined)
            self._mt_time_label.setText(f"Analysis completed in {elapsed:.0f}ms")

            # Color-code labels
            audio_color = P["red"] if audio_score >= 0.5 else P["green"]
            video_color = P["red"] if video_score >= 0.5 else P["green"]
            self._mt_audio_label.setStyleSheet(f"color: {audio_color}; border: none; font-size: 13px;")
            self._mt_video_label.setStyleSheet(f"color: {video_color}; border: none; font-size: 13px;")

        except Exception as exc:
            self._mt_combined_label.setText(f"Analysis failed: {exc}")
            logger.error("Manual analysis failed: %s", exc)

        self._btn_analyze.setEnabled(True)
        self._btn_analyze.setText("🔍  Analyze Media File")

    # ══════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════

    def _make_card(self, title: str) -> QFrame:
        """Create a styled card frame with a title."""
        frame = QFrame()
        frame.setStyleSheet(
            f"background-color: {P['mantle']}; "
            f"border: 1px solid {P['surface0']}; "
            f"border-radius: 10px;"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        label = QLabel(title)
        label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        label.setStyleSheet(f"color: {P['overlay1']}; border: none; background: transparent;")
        layout.addWidget(label)

        return frame

    def _log(self, message: str) -> None:
        """Append a timestamped message to the activity log."""
        if hasattr(self, "_log_box") and self._log_box:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            self._log_box.append(f"[{ts}] {message}")
            # Auto-scroll to bottom
            scrollbar = self._log_box.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event) -> None:
        """Clean up on window close."""
        if self._detection_service:
            self._detection_service.cleanup()
        event.accept()

    def launch(self) -> None:
        """Legacy launcher for CLI 'gui' subcommand."""
        self.show()
