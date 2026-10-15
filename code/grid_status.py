"""Progress of the v2 grid: completed ledger units per study against the expected totals.
Writes logs_v2/GRID_DONE when every study is complete (the launcher then stops relaunching).

    python grid_status.py
"""
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE / "fogfield_out_v2" / "results"
EXPECTED = {"main": 2 * 15 * 9 * 2 * 5, "scalability": 5 * 6 * 3, "milp": 25,
            "equilibrium": 2 * 4 * 5, "lll_schedule": 2 * 5 * 2 * 5, "viewbound": 4}
total_done = total_exp = 0
failed = 0
for study, exp in EXPECTED.items():
    d = RES / study
    files = list(d.glob("*.json")) if d.exists() else []
    for f in files:
        try:
            if json.loads(f.read_text(encoding="utf-8")).get("failed"):
                failed += 1
        except Exception:
            pass
    total_done += len(files)
    total_exp += exp
    print("{:13s} {:5d} / {:5d}".format(study, len(files), exp))
ck = HERE / "fogfield_out_v2" / "ckpt"
n_ck = len(list(ck.glob("*.pkl"))) if ck.exists() else 0
print("{:13s} {:5d} / {:5d}   ({} partial checkpoints, {} failed units)  {}".format(
    "TOTAL", total_done, total_exp, n_ck, failed, time.strftime("%H:%M:%S")))
if total_done >= total_exp:
    (HERE / "logs_v2").mkdir(exist_ok=True)
    (HERE / "logs_v2" / "GRID_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("GRID COMPLETE")
