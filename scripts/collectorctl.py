#!/usr/bin/env python3
"""Start/stop/status control for the background collector.

Uses a pidfile rather than pattern-matching on process names: a ``pkill -f``
pattern also matches the shell command that contains it, which makes it
dangerously easy to kill your own session.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIDFILE = ROOT / "logs" / "collector.pid"
LOGFILE = ROOT / "logs" / "collector.log"


def read_pid() -> int | None:
    if not PIDFILE.exists():
        return None
    try:
        pid = int(PIDFILE.read_text().strip())
    except ValueError:
        return None
    return pid if alive(pid) else None


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # Confirm it is actually our collector, not a recycled PID.
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except OSError:
        return False
    return "run_collector" in cmd and "python" in cmd


def stop() -> None:
    pid = read_pid()
    if pid is None:
        print("not running")
        PIDFILE.unlink(missing_ok=True)
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        time.sleep(1)
        if not alive(pid):
            print(f"stopped {pid}")
            PIDFILE.unlink(missing_ok=True)
            return
    os.kill(pid, signal.SIGKILL)
    PIDFILE.unlink(missing_ok=True)
    print(f"force-killed {pid}")


def start(extra: list[str]) -> None:
    if (pid := read_pid()) is not None:
        print(f"already running as {pid}")
        return
    LOGFILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOGFILE, "a") as log:
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / "run_collector.py"), *extra],
            stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), start_new_session=True,
        )
    PIDFILE.write_text(str(proc.pid))
    print(f"started {proc.pid}: {' '.join(extra)}")


def status() -> None:
    pid = read_pid()
    print(f"collector: {'running pid=' + str(pid) if pid else 'stopped'}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action == "stop":
        stop()
    elif action == "start":
        start(sys.argv[2:])
    elif action == "restart":
        stop()
        start(sys.argv[2:])
    else:
        status()
