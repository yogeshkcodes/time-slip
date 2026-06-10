"""
Time Slip - real-time attention tracker (Windows, zero dependencies).

Logs YOUR actual computer behaviour - the foreground app, its window title and
your idle time - to a local CSV, sampling every few seconds. 100% on-device:
nothing leaves your machine. This is the real-data counterpart to the simulator:
real focus spells, real task switches, real digital rabbit holes.

    python track_me.py                     # track until Ctrl+C (samples every 5s)
    python track_me.py --minutes 90        # track for 90 minutes then stop
    python track_me.py --interval 3        # sample every 3 seconds

Logs land in outputs/tracker/track_YYYY-MM-DD.csv (appended across sessions).
Analyse them with:  python analyze_tracker.py

Privacy: window titles can contain sensitive text (document names, chat
subjects). The log stays local; review/delete it freely. Use --no-titles to
record only the app name.
"""

from __future__ import annotations
import argparse
import csv
import ctypes
import ctypes.wintypes as wt
import os
import sys
import time
from datetime import datetime

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "outputs", "tracker")


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wt.UINT), ("dwTime", wt.DWORD)]


def idle_seconds(user32, kernel32) -> float:
    lii = LASTINPUTINFO()
    lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
    if user32.GetLastInputInfo(ctypes.byref(lii)):
        return (kernel32.GetTickCount() - lii.dwTime) / 1000.0
    return 0.0


def foreground(user32, kernel32):
    """Return (exe_name, window_title) of the foreground window."""
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return "unknown", ""
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    title = buf.value
    pid = wt.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    exe = "unknown"
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if h:
        size = wt.DWORD(1024)
        pbuf = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(h, 0, pbuf, ctypes.byref(size)):
            exe = os.path.basename(pbuf.value).lower()
        kernel32.CloseHandle(h)
    return exe, title


def main():
    ap = argparse.ArgumentParser(description="Local attention tracker")
    ap.add_argument("--interval", type=float, default=5.0,
                    help="seconds between samples (default 5)")
    ap.add_argument("--minutes", type=float, default=0,
                    help="stop after N minutes (default: run until Ctrl+C)")
    ap.add_argument("--no-titles", action="store_true",
                    help="do not record window titles (extra privacy)")
    args = ap.parse_args()

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"track_{datetime.now():%Y-%m-%d}.csv")
    new = not os.path.exists(path)
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if new:
        w.writerow(["timestamp", "exe", "title", "idle_s"])

    stop_at = time.time() + args.minutes * 60 if args.minutes > 0 else None
    print(f"Tracking -> {path}  (Ctrl+C to stop"
          + (f", auto-stop in {args.minutes:g} min" if stop_at else "") + ")")
    n = 0
    try:
        while True:
            exe, title = foreground(user32, kernel32)
            if args.no_titles:
                title = ""
            w.writerow([datetime.now().isoformat(timespec="seconds"),
                        exe, title[:160], round(idle_seconds(user32, kernel32), 1)])
            n += 1
            if n % 60 == 0:
                f.flush()
            if stop_at and time.time() >= stop_at:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
    print(f"Logged {n} samples. Analyse with: python analyze_tracker.py")


if __name__ == "__main__":
    if sys.platform != "win32":
        sys.exit("track_me.py is Windows-only (uses the Win32 API).")
    main()
