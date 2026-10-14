"""Canonical driver for fogfield_src_fullrun.py - always sets the headless plotting
backend before anything else, then runs in one of two modes:

  --mode grind      real TIME_BUDGET_MIN (as already patched into the file, 50 min),
                     for continuing main/scalability from wherever the ledger left off.

  --mode analysis    temporarily neuters TIME_BUDGET_MIN so the four auto-run_study(...)
                      calls each do at most ~1 bonus unit and bail, landing quickly on the
                      analysis cells (section 17+), which read the ledger as-is and produce
                      real figures/tables from whatever has completed so far.

Every earlier crash in this session came from skipping this Agg-backend step or from
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

if args.mode == "analysis":
    src, n = re.subn(r"^TIME_BUDGET_MIN = 50 if QUICK else 50$",
                     "TIME_BUDGET_MIN = 0.03 if QUICK else 0.03", src, flags=re.MULTILINE)
    assert n == 1, "TIME_BUDGET_MIN line not found - source shape changed?"
    print("[driver] mode=analysis: TIME_BUDGET_MIN neutered to ~2s so experiment cells "
         "fall through fast into the analysis section", flush=True)
else:
    print("[driver] mode=grind: real TIME_BUDGET_MIN in effect, continuing main/"
         "scalability from the ledger", flush=True)

g = sys.modules["__main__"].__dict__
g["__file__"] = str(SRC)
t0 = time.time()
exec(compile(src, str(SRC), "exec"), g)
print("\n[driver] exec finished in {:.1f} min".format((time.time() - t0) / 60), flush=True)
