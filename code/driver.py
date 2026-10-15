"""Canonical driver for fogfield_src_fullrun.py - always sets the headless plotting
backend before anything else, then runs in one of two modes:

  --mode grind      run/continue the experiment grid from wherever the ledger left off
                     (checkpointed per unit and every CKPT_EVERY rounds inside a unit).
                     Environment switches, all optional:
                       FOGFIELD_NWORKERS=W FOGFIELD_WORKER=k   this process takes the units whose
                                                               key hashes to k mod W (see run_study)
                       FOGFIELD_BUDGET_MIN=600                 minutes before the process stops
                                                               cleanly (default 50); just run again
                       FOGFIELD_STOP_AFTER_EXPERIMENTS=1       exit before the analysis section
                     `run_grid.ps1` starts W such workers; the scheduled task `fogfield_v2`
                     re-launches it every 10 minutes until the grid is complete.

  --mode analysis    neuters TIME_BUDGET_MIN so the auto-run_study(...) calls each do at most
                      ~1 bonus unit and bail, landing quickly on the analysis cells, which read
                      the ledger as-is and produce real figures/tables from whatever has
                      completed so far.

Every earlier crash in this project came from skipping this Agg-backend step or from
truncating the source before the functions a follow-up call needed - this file exists so
that class of mistake stops happening.
"""
import argparse
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["grind", "analysis"], required=True)
ap.add_argument("--src", default="fogfield_src_fullrun.py")
args = ap.parse_args()

SRC = Path(__file__).parent / args.src
src = SRC.read_text(encoding="utf-8")

BUDGET_LINE = r'^TIME_BUDGET_MIN = float\(os\.environ\.get\("FOGFIELD_BUDGET_MIN", "50"\)\)$'
if args.mode == "analysis":
    src, n = re.subn(BUDGET_LINE, "TIME_BUDGET_MIN = 0.03", src, flags=re.MULTILINE)
    assert n == 1, "TIME_BUDGET_MIN line not found - source shape changed?"
    print("[driver] mode=analysis: TIME_BUDGET_MIN neutered to ~2s so experiment cells "
         "fall through fast into the analysis section", flush=True)
else:
    print("[driver] mode=grind: continuing the grid from the ledger", flush=True)

g = sys.modules["__main__"].__dict__
g["__file__"] = str(SRC)
t0 = time.time()
exec(compile(src, str(SRC), "exec"), g)
print("\n[driver] exec finished in {:.1f} min".format((time.time() - t0) / 60), flush=True)
