#!/usr/bin/env python3
"""Generic pidfile-based control for the background daemons.

Pidfiles rather than name matching: a ``pkill -f <pattern>`` also matches the
shell command containing that pattern, which makes it very easy to kill your own
session instead of the daemon.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAEMONS = {
    "collector": "run_collector.py",
    "backfill": "backfill_candles.py",
    "launchstream": "run_launchstream.py",
}


def pidfile(name: str) -> Path:
    return ROOT / "logs" / f"{name}.pid"


def logfile(name: str) -> Path:
    return ROOT / "logs" / f"{name}.log"


def alive(pid: int, script: str) -> bool:
    try:
        os.kill(pid, 0)
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except OSError:
        return False
    # Confirm it is the daemon and not a recycled pid or a shell mentioning it.
    return script in cmd and "python" in cmd and " -c" not in cmd


def read_pid(name: str) -> int | None:
    pf = pidfile(name)
    if not pf.exists():
        return None
    try:
        pid = int(pf.read_text().strip())
    except ValueError:
        return None
    return pid if alive(pid, DAEMONS[name]) else None


def stop(name: str) -> None:
    pid = read_pid(name)
    if pid is None:
        print(f"{name}: not running")
        pidfile(name).unlink(missing_ok=True)
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(20):
        time.sleep(1)
        if not alive(pid, DAEMONS[name]):
            print(f"{name}: stopped {pid}")
            pidfile(name).unlink(missing_ok=True)
            return
    os.kill(pid, signal.SIGKILL)
    pidfile(name).unlink(missing_ok=True)
    print(f"{name}: force-killed {pid}")


def start(name: str, extra: list[str]) -> None:
    if (pid := read_pid(name)) is not None:
        print(f"{name}: already running as {pid}")
        return
    logfile(name).parent.mkdir(parents=True, exist_ok=True)
    with open(logfile(name), "a") as log:
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "scripts" / DAEMONS[name]), *extra],
            stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), start_new_session=True,
        )
    pidfile(name).write_text(str(proc.pid))
    print(f"{name}: started {proc.pid} {' '.join(extra)}")


def status() -> None:
    for name in DAEMONS:
        pid = read_pid(name)
        print(f"  {name:10} {'running pid=' + str(pid) if pid else 'stopped'}")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action == "status":
        status()
    else:
        target = sys.argv[2] if len(sys.argv) > 2 else None
        names = [target] if target in DAEMONS else list(DAEMONS)
        rest = sys.argv[3:] if target in DAEMONS else []
        for n in names:
            if action in ("stop", "restart"):
                stop(n)
            if action in ("start", "restart"):
                start(n, rest)
