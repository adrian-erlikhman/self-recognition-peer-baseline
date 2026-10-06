"""Run a command, killing it if free system RAM drops below a floor.

The open-weight runs share a 32 GB laptop with everything else; a model that
spills out of VRAM into shared memory can exhaust RAM and freeze the machine.

    python tools/guard.py --min-free-gb 4 -- python src/revision/run_openweight.py
"""
import argparse, subprocess, sys, time
import psutil

ap = argparse.ArgumentParser()
ap.add_argument("--min-free-gb", type=float, default=4.0)
ap.add_argument("cmd", nargs=argparse.REMAINDER)
a = ap.parse_args()
cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
if cmd and cmd[0] in ("python", "python.exe"):
    cmd[0] = sys.executable
p = subprocess.Popen(cmd)
low = 1e9
below = 0
while p.poll() is None:
    free = psutil.virtual_memory().available / 1e9
    low = min(low, free)
    below = below + 1 if free < a.min_free_gb else 0
    # Sustained for ~2 s: a model load maps its checkpoint and briefly
    # dips "available" memory with clean, reclaimable file pages.
    if below >= 4:
        print(f"[guard] free RAM {free:.1f} GB < {a.min_free_gb} GB: killing", flush=True)
        for c in psutil.Process(p.pid).children(recursive=True):
            c.kill()
        p.kill()
        p.wait()
        sys.exit(99)
    time.sleep(0.5)
print(f"[guard] exit {p.returncode}; lowest free RAM {low:.1f} GB", flush=True)
sys.exit(p.returncode)
