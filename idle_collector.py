"""Run a collector only after 15 minutes without Windows keyboard/mouse input."""
import ctypes
from ctypes import wintypes
import subprocess
import sys
import time

IDLE_SECONDS = 15 * 60


def idle_seconds():
    class LastInput(ctypes.Structure):
        _fields_ = [('cbSize', wintypes.UINT), ('dwTime', wintypes.DWORD)]

    info = LastInput()
    info.cbSize = ctypes.sizeof(info)
    user = ctypes.WinDLL('user32', use_last_error=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetTickCount.restype = wintypes.DWORD
    if not user.GetLastInputInfo(ctypes.byref(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    return ((kernel.GetTickCount() - info.dwTime) & 0xffffffff) / 1000


def allowed():
    try:
        return idle_seconds() > IDLE_SECONDS
    except (OSError, AttributeError):
        return False  # Never collect if activity cannot be measured.


def stop_tree(child):
    subprocess.run(['taskkill.exe', '/PID', str(child.pid), '/T', '/F'],
                   check=False, capture_output=True)
    child.wait(timeout=15)


def run(command):
    # Task Scheduler supplies the recurring trigger and native idle restart.
    # Independently check actual input: Windows' idle heuristic is not precise.
    if not allowed():
        return 0
    child = subprocess.Popen(command)
    try:
        while child.poll() is None:
            if not allowed():
                stop_tree(child)
                return 0
            time.sleep(1)
        return child.returncode
    finally:
        if child.poll() is None:
            stop_tree(child)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise SystemExit('Usage: idle_collector.py executable [arguments...]')
    raise SystemExit(run(sys.argv[1:]))
