"""
Interactive Real-Time Webcam & Video Call Deepfake Monitor.

Launches a live camera feed (webcam or virtual camera stream from OBS/Zoom)
with a cyber-forensic HUD overlay displaying real-time face tracking,
biometric authenticity gauge, FPS counter, and threat alerts.

Usage:
    python scripts/run_live_webcam.py [--camera 0] [--width 1280] [--height 720]

Keys:
    Q / ESC : Exit
    R       : Reset temporal detector state
    S       : Save current forensic snapshot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from app.config.settings import settings
from app.realtime import RealtimeLiveDetector
from app.utils.logger import get_logger

logger = get_logger(__name__)


def draw_cyber_hud(
    frame: np.ndarray,
    telemetry: dict,
    width: int,
    height: int,
) -> np.ndarray:
    """Draw a forensic cyber-HUD overlay on the live camera frame."""
    canvas = frame.copy()
    verdict = telemetry.get("verdict", "UNCERTAIN")
    fake_conf = telemetry.get("fake_confidence", 0.5)
    real_conf = telemetry.get("real_confidence", 0.5)
    threat = telemetry.get("threat_level", "NOMINAL")
    fps = telemetry.get("fps", 0.0)
    latency = telemetry.get("latency_ms", 0.0)
    bbox = telemetry.get("bbox")
    status = telemetry.get("status", "active")
    history = telemetry.get("history", [])

    # Color palette (BGR)
    if verdict == "FAKE":
        primary_color = (40, 40, 240)    # Crimson red
        accent_color = (80, 80, 255)
        badge_text = "CRITICAL THREAT: SYNTHETIC DEEPFAKE DETECTED"
    elif verdict == "REAL":
        primary_color = (80, 220, 80)    # Emerald green
        accent_color = (130, 255, 130)
        badge_text = "AUTHENTIC HUMAN BIOMETRICS VERIFIED"
    elif verdict == "NOT_APPLICABLE":
        primary_color = (220, 160, 40)   # Cyan / amber
        accent_color = (255, 200, 80)
        badge_text = "STAGE-0 GUARD: NON-BIOMETRIC CONTENT"
    else:
        primary_color = (40, 200, 240)   # Electric amber / yellow
        accent_color = (100, 230, 255)
        badge_text = "ACQUIRING BIOMETRIC TELEMETRY..."

    # 1. Top Banner
    cv2.rectangle(canvas, (0, 0), (width, 50), (10, 12, 20), -1)
    cv2.line(canvas, (0, 50), (width, 50), primary_color, 2)

    # Status Pill
    cv2.rectangle(canvas, (15, 10), (320, 40), primary_color, -1)
    cv2.putText(
        canvas,
        f"DEEPGUARD LIVE: {verdict}",
        (25, 32),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (10, 10, 10),
        2,
        cv2.LINE_AA,
    )

    # FPS & Latency
    stat_str = f"FPS: {fps:4.1f} | Latency: {latency:4.1f}ms | Threat: {threat}"
    cv2.putText(
        canvas,
        stat_str,
        (340, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 210, 220),
        1,
        cv2.LINE_AA,
    )

    # 2. Draw Face Bounding Box with Cyber Corner Brackets
    if bbox is not None:
        bx, by, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        # Outer box with transparency
        overlay = canvas.copy()
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), primary_color, 2)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

        # Corner brackets
        corner_len = min(25, bw // 4, bh // 4)
        c_thick = 3
        # Top-left
        cv2.line(canvas, (bx, by), (bx + corner_len, by), accent_color, c_thick)
        cv2.line(canvas, (bx, by), (bx, by + corner_len), accent_color, c_thick)
        # Top-right
        cv2.line(canvas, (bx + bw, by), (bx + bw - corner_len, by), accent_color, c_thick)
        cv2.line(canvas, (bx + bw, by), (bx + bw, by + corner_len), accent_color, c_thick)
        # Bottom-left
        cv2.line(canvas, (bx, by + bh), (bx + corner_len, by + bh), accent_color, c_thick)
        cv2.line(canvas, (bx, by + bh), (bx, by + bh - corner_len), accent_color, c_thick)
        # Bottom-right
        cv2.line(canvas, (bx + bw, by + bh), (bx + bw - corner_len, by + bh), accent_color, c_thick)
        cv2.line(canvas, (bx + bw, by + bh), (bx + bw, by + bh - corner_len), accent_color, c_thick)

        # Face Label
        tag_text = f"TARGET FACE [{bw}x{bh}]"
        cv2.putText(
            canvas,
            tag_text,
            (bx, max(20, by - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            accent_color,
            1,
            cv2.LINE_AA,
        )

    # 3. Bottom HUD Deck (Telemetry & Confidence Meter)
    deck_y = height - 90
    cv2.rectangle(canvas, (0, deck_y), (width, height), (8, 10, 16), -1)
    cv2.line(canvas, (0, deck_y), (width, deck_y), (40, 50, 70), 1)

    # Confidence Bar
    bar_x = 20
    bar_y = deck_y + 35
    bar_w = 340
    bar_h = 16

    cv2.putText(
        canvas,
        f"AUTHENTICITY SCORE: {real_conf * 100:5.1f}%  |  DEEPFAKE PROBABILITY: {fake_conf * 100:5.1f}%",
        (bar_x, bar_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 190, 205),
        1,
        cv2.LINE_AA,
    )

    # Background track
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 35, 45), -1)
    # Fill proportional to fake_conf
    fill_w = int(bar_w * fake_conf)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), primary_color, -1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 90, 110), 1)

    # Threshold markers at 30% and 70%
    t30_x = bar_x + int(bar_w * 0.30)
    t70_x = bar_x + int(bar_w * 0.70)
    cv2.line(canvas, (t30_x, bar_y - 2), (t30_x, bar_y + bar_h + 2), (100, 200, 100), 1)
    cv2.line(canvas, (t70_x, bar_y - 2), (t70_x, bar_y + bar_h + 2), (80, 80, 240), 1)

    # 4. Live Mini Sparkline Waveform (Right side of bottom deck)
    if len(history) >= 2:
        graph_x = bar_x + bar_w + 40
        graph_w = width - graph_x - 30
        graph_y = deck_y + 15
        graph_h = 60

        cv2.rectangle(canvas, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (18, 22, 30), -1)
        cv2.rectangle(canvas, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (45, 55, 75), 1)

        # Center line (50%)
        cv2.line(canvas, (graph_x, graph_y + graph_h // 2), (graph_x + graph_w, graph_y + graph_h // 2), (40, 50, 65), 1)

        pts = []
        n_pts = len(history)
        step_x = graph_w / max(1, n_pts - 1)
        for i, pt in enumerate(history):
            px = int(graph_x + i * step_x)
            py = int(graph_y + graph_h - (pt.get("score", 0.5) * graph_h))
            pts.append((px, py))

        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], primary_color, 2, cv2.LINE_AA)

        cv2.putText(
            canvas,
            "TEMPORAL CONFIDENCE CURVE",
            (graph_x + 5, graph_y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (120, 140, 165),
            1,
            cv2.LINE_AA,
        )

    # Sub-caption
    cv2.putText(
        canvas,
        badge_text,
        (bar_x, deck_y + 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        primary_color,
        1,
        cv2.LINE_AA,
    )

    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepGuard Live Video Call & Webcam Detector")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Camera capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera capture height (default: 720)")
    args = parser.parse_args()

    print("=" * 65)
    print("  🛡️ DEEPGUARD LIVE WEBCAM & VIDEO CALL DETECTION MONITOR")
    print(f"  • Camera Index: {args.camera}")
    print(f"  • Resolution:   {args.width}x{args.height}")
    print("  • Controls:     [Q] Quit | [R] Reset State | [S] Save Snapshot")
    print("=" * 65)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"❌ Could not open camera {args.camera}. Please check device permissions.")
        sys.exit(1)

    detector = RealtimeLiveDetector()
    window_name = "DeepGuard — Real-Time Video Call Biometric Integrity Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    snapshot_dir = settings.project_root / "logs" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Process frame with temporal detector
            telemetry = detector.process_frame(frame)

            # Draw HUD
            h, w = frame.shape[:2]
            hud_frame = draw_cyber_hud(frame, telemetry, width=w, height=h)

            cv2.imshow(window_name, hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                break
            elif key in (ord("r"), ord("R")):
                detector.reset()
                print("🔄 Temporal tracking state reset.")
            elif key in (ord("s"), ord("S")):
                snap_path = snapshot_dir / f"live_scan_{int(time.time())}.jpg"
                cv2.imwrite(str(snap_path), hud_frame)
                print(f"📸 Forensic snapshot saved to: {snap_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Live webcam detection terminated.")


if __name__ == "__main__":
    main()
