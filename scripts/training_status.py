#!/usr/bin/env python3
"""Report FPNet training progress, throughput and ETA.

Usage: training_status.py [LOG] [TOTAL_STEPS]
"""

from __future__ import annotations

import datetime
import re
import subprocess
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "/mnt/casmi/train.log"
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 30_000
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

text = open(LOG).read()
lines = text.splitlines()

steps = []
for line in lines:
    m = re.match(r"step\s+(\d+)\s+loss\s+([\d.]+)\s+lr\s+\S+\s+(\d+)s", line)
    if m:
        steps.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))

running = (
    subprocess.run(
        ["pgrep", "-f", "train_fpnet.py"], capture_output=True, text=True
    ).returncode
    == 0
)

if not steps:
    print("no step lines yet (still in setup)")
    print("RUNNING" if running else "NOT RUNNING")
    raise SystemExit(0)

# Recent window for rate, so a slow start does not skew the estimate.
window = steps[-40:] if len(steps) > 40 else steps
s0, _, t0 = window[0]
s1, loss1, t1 = steps[-1]
rate = (s1 - s0) / (t1 - t0) if t1 > t0 else 0.0

vals = [float(v) for v in re.findall(r"cosine similarity ([\d.]+)", text)]
now = datetime.datetime.now(datetime.UTC)

print(f"state       {'RUNNING' if running else 'FINISHED / STOPPED'}")
print(f"progress    {s1:,}/{TOTAL:,}  ({s1 / TOTAL:.1%})")
print(f"loss        {loss1:.4f}  (start {steps[0][1]:.4f})")
if vals:
    print(f"val cosine  {vals[0]:.4f} -> {vals[-1]:.4f}  over {len(vals)} evals")
print(f"rate        {rate:.2f} steps/sec")
print(f"elapsed     {t1 / 60:.0f} min")

if running and rate > 0 and s1 < TOTAL:
    remaining = (TOTAL - s1) / rate
    eta = now + datetime.timedelta(seconds=remaining)
    print(f"remaining   {remaining / 60:.0f} min")
    print(f"now         IST {now.astimezone(IST):%H:%M}")
    print(f"ETA         IST {eta.astimezone(IST):%H:%M  %a %d %b}")
else:
    print(f"now         IST {now.astimezone(IST):%H:%M}")
    if "[OK] checkpoints in" in text:
        print("final       training completed normally")
