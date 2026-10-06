"""Ask Windows not to idle-sleep while the open-weight study runs.

SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED) is the per-process
request media players use; it changes no setting and lapses when this process
exits. It does not stop sleep on a closed lid. Exits when the run log says the
study is complete, or after --hours.
"""
import argparse, ctypes, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--hours", type=float, default=6)
a = ap.parse_args()
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
log = Path(__file__).resolve().parent.parent / "logs" / "openweight_run.log"
start = log.stat().st_size if log.exists() else 0
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
t_end = time.time() + a.hours * 3600
try:
    while time.time() < t_end:
        if log.exists() and log.stat().st_size > start:
            with log.open("rb") as fh:
                fh.seek(start)
                if b"=== study complete" in fh.read():
                    break
        time.sleep(30)
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
