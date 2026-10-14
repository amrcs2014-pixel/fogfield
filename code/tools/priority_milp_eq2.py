"""Corrected priority pass.

Previous attempt truncated the source before section 16, which is exactly where
run_study/grid_milp/grid_equilibrium are DEFINED - so it failed immediately with
KeyError before doing anything (verified: crashed in 6s, ledger unchanged, no corruption).

This version keeps the full source (so the functions exist) but temporarily shrinks
TIME_BUDGET_MIN to a few seconds before exec, so the four automatic run_study(...) calls
already in the module body each do at most one extra unit and bail immediately, instead of
running for real. Control then returns here, where run_study is called AGAIN, by hand, for
milp and equilibrium specifically, with a real budget - which is the whole point of this
script: get to those two cheap, high-value studies without waiting for main/scalability to
fully drain first.
"""
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(__file__).parent / "fogfield_src_fullrun.py"
src = SRC.read_text(encoding="utf-8")
src2, n = re.subn(r"^TIME_BUDGET_MIN = 50 if QUICK else 50$",
                  "TIME_BUDGET_MIN = 0.03 if QUICK else 0.03", src, flags=re.MULTILINE)
assert n == 1, "TIME_BUDGET_MIN line not found/patched as expected"
plt.show = lambda *a, **k: None

g = sys.modules["__main__"].__dict__
g["__file__"] = str(SRC)
print("executing FULL source with neutered auto-budget (each auto-call bails after "
     "~1 extra unit)...", flush=True)
t_exec0 = time.time()
exec(compile(src2, str(SRC), "exec"), g)
print("exec done in {:.1f} min (this includes ~1 bonus unit each for main/scalability/"
     "milp/equilibrium from the neutered auto-calls)".format((time.time() - t_exec0) / 60),
     flush=True)

run_study = g["run_study"]
grid_milp = g["grid_milp"]
run_milp_unit = g["run_milp_unit"]
grid_equilibrium = g["grid_equilibrium"]
run_equilibrium_unit = g["run_equilibrium_unit"]

print("\n=== now running milp with a REAL budget ===", flush=True)
t0 = time.time()
run_study("milp", grid_milp(), run_milp_unit, budget_min=32)
print("milp phase: {:.1f} min".format((time.time() - t0) / 60), flush=True)

print("\n=== now running equilibrium with a REAL budget ===", flush=True)
t0b = time.time()
run_study("equilibrium", grid_equilibrium(), run_equilibrium_unit, budget_min=8)
print("equilibrium phase: {:.1f} min".format((time.time() - t0b) / 60), flush=True)

print("\nPRIORITY PASS DONE, total {:.1f} min".format((time.time() - t_exec0) / 60),
     flush=True)
