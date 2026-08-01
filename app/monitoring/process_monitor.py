"""
Smart Process Monitor for Video Call Applications.

Detects running video conferencing applications (Zoom, Microsoft Teams, Google Meet,
Discord, Skype, Webex, Slack, WhatsApp Desktop, UWP apps) using:
1. Native process name matching (including WhatsApp Desktop / UWP ApplicationFrameHost).
2. Chromium / Edge media subprocess command-line inspection (--utility-sub-type=video_capture).
3. Window title scanning for video call keywords ("Google Meet", "Zoom Meeting", "WhatsApp", etc.).
4. Audio/Video activity heuristics to eliminate false positives from idle browser tabs.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Tuple

import psutil

from app.utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# Video Call Process Name Mappings
# ──────────────────────────────────────────────

NATIVE_VIDEO_CALL_APPS: Dict[str, str] = {
    # Zoom
    "zoom.exe": "Zoom",
    "zoom": "Zoom",
    "zoom.us": "Zoom",
    # Microsoft Teams
    "teams.exe": "Microsoft Teams",
    "teams": "Microsoft Teams",
    "msteams.exe": "Microsoft Teams",
    "msteams": "Microsoft Teams",
    # Cisco Webex
    "webex.exe": "Cisco Webex",
    "webex": "Cisco Webex",
    "atmgr.exe": "Cisco Webex",
    "ciscowebexstart.exe": "Cisco Webex",
    # Skype
    "skype.exe": "Skype",
    "skype": "Skype",
    "skypehost.exe": "Skype",
    # Discord
    "discord.exe": "Discord",
    "discord": "Discord",
    # Slack
    "slack.exe": "Slack",
    "slack": "Slack",
    # WhatsApp Desktop (Win32 & Windows Store UWP)
    "whatsapp.exe": "WhatsApp",
    "whatsapp": "WhatsApp",
    "whatsapp.root.exe": "WhatsApp",
    "whatsapp.desktop.exe": "WhatsApp",
    # Telegram
    "telegram.exe": "Telegram",
    "telegram": "Telegram",
}

BROWSER_PROCESSES: Set[str] = {
    "chrome.exe", "chrome",
    "msedge.exe", "msedge",
    "firefox.exe", "firefox",
    "brave.exe", "brave",
    "opera.exe", "opera",
}

VIDEO_CALL_TITLE_KEYWORDS: Set[str] = {
    "meet.google.com", "google meet",
    "zoom meeting", "zoom call", "zoom.us",
    "microsoft teams", "teams call",
    "webex meeting", "webex call",
    "whatsapp call", "whatsapp video call", "whatsapp voice call",
    "skype call", "discord call",
}


@dataclass
class ProcessInfo:
    """Detailed information about a detected video call process."""

    pid: int
    name: str
    display_name: str
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    status: str = "running"
    exe_path: Optional[str] = None
    is_media_active: bool = False


class ProcessMonitor:
    """
    Monitors active operating system processes to detect video call apps.

    Uses smart process command-line analysis, window title inspection,
    and media activity heuristics to avoid false positives.
    """

    def __init__(self, target_apps: Optional[Dict[str, str]] = None) -> None:
        self.process_map = target_apps or NATIVE_VIDEO_CALL_APPS
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_active_state = False

    def scan_processes(self) -> List[ProcessInfo]:
        """
        Scan processes and return detected active video call applications.

        Filters out idle browser windows unless an active video call tab or
        media capture subprocess is detected.

        Returns
        -------
        List[ProcessInfo]
            List of detected video conferencing processes.
        """
        detected: List[ProcessInfo] = []
        window_titles = self._get_window_titles()

        for proc in psutil.process_iter(["pid", "name", "status", "memory_info", "cpu_percent", "cmdline"]):
            try:
                proc_name = (proc.info.get("name") or "").lower()
                cmdline_list = proc.info.get("cmdline") or []
                cmdline_str = " ".join(cmdline_list).lower()
                pid = proc.info["pid"]

                # ── 1. Native Video Call Apps (Zoom, Teams, WhatsApp, etc.) ────
                if proc_name in self.process_map:
                    display_name = self.process_map[proc_name]
                    info = self._build_process_info(proc, display_name, is_media=True)
                    detected.append(info)
                    continue

                # ── 2. UWP App Wrapper (ApplicationFrameHost.exe) ─────────────
                if proc_name == "applicationframehost.exe":
                    # Check window titles associated with ApplicationFrameHost
                    title = window_titles.get(pid, "").lower()
                    if any(k in title for k in ["whatsapp", "teams", "skype", "zoom", "meet"]):
                        info = self._build_process_info(proc, f"WhatsApp (UWP: {title})", is_media=True)
                        detected.append(info)
                        continue

                # ── 3. Smart Browser Video Call Detection (Chrome / Edge / Firefox) ──
                if proc_name in BROWSER_PROCESSES:
                    is_video_call = False
                    detected_title = "Google Meet / Browser Video Call"

                    # Check A: Chromium Media Subprocess flags (--utility-sub-type=video_capture or audio.mojom)
                    if any("video_capture" in arg or "audio.mojom" in arg for arg in cmdline_list):
                        is_video_call = True
                        detected_title = f"Browser Video Call ({proc_name.split('.')[0].capitalize()})"

                    # Check B: Browser Window Title matching keywords ("Google Meet", "Zoom", etc.)
                    title = window_titles.get(pid, "").lower()
                    if any(kw in title for kw in VIDEO_CALL_TITLE_KEYWORDS):
                        is_video_call = True
                        detected_title = f"Video Call: {window_titles.get(pid)}"

                    if is_video_call:
                        info = self._build_process_info(proc, detected_title, is_media=True)
                        detected.append(info)

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            except Exception as exc:
                logger.debug("Error querying process: %s", exc)

        return detected

    def _build_process_info(
        self, proc: psutil.Process, display_name: str, is_media: bool = False
    ) -> ProcessInfo:
        """Helper to construct ProcessInfo safely."""
        pid = proc.info["pid"]
        proc_name = (proc.info.get("name") or "").lower()
        status = proc.info.get("status") or "running"

        mem_info = proc.info.get("memory_info")
        memory_mb = (mem_info.rss / (1024 * 1024)) if mem_info else 0.0
        cpu_percent = proc.info.get("cpu_percent") or 0.0

        exe_path = None
        try:
            exe_path = proc.exe()
        except Exception:
            pass

        return ProcessInfo(
            pid=pid,
            name=proc_name,
            display_name=display_name,
            cpu_percent=round(cpu_percent, 1),
            memory_mb=round(memory_mb, 1),
            status=status,
            exe_path=exe_path,
            is_media_active=is_media,
        )

    def _get_window_titles(self) -> Dict[int, str]:
        """
        Enumerate top-level window titles mapped by Process ID (PID).
        """
        titles: Dict[int, str] = {}
        if sys.platform != "win32":
            return titles

        try:
            import win32gui
            import win32process

            def enum_cb(hwnd, param):
                if win32gui.IsWindow(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if title and title.strip():
                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                        titles[pid] = title.strip()
                return True

            win32gui.EnumWindows(enum_cb, None)
        except Exception as exc:
            logger.debug("Window title enumeration unavailable: %s", exc)

        return titles

    def get_window_geometry(self, process_name: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Locate the main visible window for a given process name and return its geometry.

        Uses ``win32gui.EnumWindows`` and ``win32process.GetWindowThreadProcessId``
        to find the target window handle (HWND).

        Parameters
        ----------
        process_name : str
            The process executable name (e.g. ``"zoom.exe"``).

        Returns
        -------
        tuple[int, int, int, int] | None
            ``(left, top, width, height)`` bounding box, or ``None`` if not found.
        """
        if sys.platform != "win32":
            return None

        target_pids: Set[int] = set()
        target_lower = process_name.lower()

        # Collect all PIDs for the target process name
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                name = (proc.info.get("name") or "").lower()
                if name == target_lower:
                    target_pids.add(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if not target_pids:
            return None

        try:
            import win32gui
            import win32process

            best_hwnd = None
            best_area = 0

            def _enum_callback(hwnd, _param):
                nonlocal best_hwnd, best_area
                if not win32gui.IsWindowVisible(hwnd):
                    return True
                title = win32gui.GetWindowText(hwnd)
                if not title or not title.strip():
                    return True
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                if pid in target_pids:
                    rect = win32gui.GetWindowRect(hwnd)
                    left, top, right, bottom = rect
                    w = right - left
                    h = bottom - top
                    area = w * h
                    if area > best_area and w > 50 and h > 50:
                        best_area = area
                        best_hwnd = hwnd
                return True

            win32gui.EnumWindows(_enum_callback, None)

            if best_hwnd is not None:
                rect = win32gui.GetWindowRect(best_hwnd)
                left, top, right, bottom = rect
                width = right - left
                height = bottom - top
                logger.debug(
                    "Window geometry for %s: left=%d, top=%d, w=%d, h=%d",
                    process_name, left, top, width, height,
                )
                return (left, top, width, height)

        except ImportError:
            logger.debug("win32gui not available for window geometry lookup")
        except Exception as exc:
            logger.debug("Window geometry lookup failed: %s", exc)

        return None

    def is_video_call_active(self) -> bool:
        """Check if any active video call process is currently running."""
        return len(self.scan_processes()) > 0

    def get_active_app_names(self) -> List[str]:
        """Return unique display names of active video call applications."""
        processes = self.scan_processes()
        unique_names: Set[str] = {p.display_name for p in processes}
        return sorted(list(unique_names))

    def start_background_monitoring(
        self,
        on_status_change: Callable[[bool, List[ProcessInfo]], None],
        interval_seconds: float = 3.0,
    ) -> None:
        """
        Start an asynchronous background thread to monitor process activity.
        """
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Background process monitoring is already running.")
            return

        self._stop_event.clear()

        def _monitor_loop() -> None:
            logger.info("Started smart background process monitor (interval=%.1fs)", interval_seconds)
            last_pids: Set[int] = set()

            while not self._stop_event.is_set():
                active_procs = self.scan_processes()
                current_pids = {p.pid for p in active_procs}
                is_active = len(active_procs) > 0

                if is_active != self._is_active_state or current_pids != last_pids:
                    self._is_active_state = is_active
                    last_pids = current_pids
                    try:
                        on_status_change(is_active, active_procs)
                    except Exception as exc:
                        logger.error("Error in process monitor callback: %s", exc)

                self._stop_event.wait(interval_seconds)

            logger.info("Stopped background process monitor.")

        self._monitor_thread = threading.Thread(
            target=_monitor_loop, name="ProcessMonitorThread", daemon=True
        )
        self._monitor_thread.start()

    def stop_background_monitoring(self) -> None:
        """Stop the asynchronous background monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._stop_event.set()
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
