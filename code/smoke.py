"""Execute the notebook source locally with a shrunken grid, to catch runtime errors.

Not a scientific run: grids are cut to the bone and GeoLife is skipped. The point is to make
every code path execute at least once before the notebook ships.
"""
import re
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(__file__).parent / "fogfield_src.py"
src = SRC.read_text(encoding="utf-8")

subs = [
    # never block on a figure window
    (r"^plt\.show\(\)$", "pass"),
    # skip the 300 MB GeoLife download in the smoke test
    (r'TESTBEDS\["BEJ"\] = load_geolife\(',
     'raise RuntimeError("skipped in smoke test"); TESTBEDS["BEJ"] = load_geolife('),
    # shrink every grid
    (r"^    GRID_NODES = \[20\]$", "    GRID_NODES = [8]"),
    (r"^    GRID_SEEDS = \[0, 1\]$", "    GRID_SEEDS = [0]"),
    (r"^    GRID_ROUNDS = 50$", "    GRID_ROUNDS = 8"),
    (r"^    SCALE_NODES = \[25, 50, 100\]$", "    SCALE_NODES = [8, 16]"),
    (r"^    MILP_NODES = \[5, 6\]$", "    MILP_NODES = [4]"),
    (r"^    MILP_SEEDS = \[0, 1\]$", "    MILP_SEEDS = [0]"),
    (r"^TIME_BUDGET_MIN = 45 if QUICK else 180$", "TIME_BUDGET_MIN = 12 if QUICK else 180"),
    (r'sub = simulate\(env, algo, "static", 5000 \+ u\["seed"\], rounds=40,',
     'sub = simulate(env, algo, "static", 5000 + u["seed"], rounds=6,'),
    (r'sub = simulate\(env, u\["algo"\], "static", 7000 \+ u\["seed"\], rounds=80,',
     'sub = simulate(env, u["algo"], "static", 7000 + u["seed"], rounds=6,'),
]
for pat, rep in subs:
    src, n = re.subn(pat, rep, src, flags=re.MULTILINE)
    if n == 0:
        print("[smoke] WARNING: substitution did not match: " + pat[:70])

# a smaller scenario set keeps the smoke run bounded while still touching every branch
src = src.replace(
    'for scen in SCENARIOS:',
    'for scen in ["static", "catastrophic", "flash_crowd", "seed_loss", "byzantine", '
    '"expansion", "random_failures", "mobility", "traffic_shift"]:')

plt.show = lambda *a, **k: None

# Exec into the real __main__ namespace. Pickling instances of cell-defined classes only
# works when the class is reachable as __main__.<Name>, which is exactly the situation in a
# Jupyter kernel; a throwaway globals dict would fail here for a reason the notebook never
# hits.
g = sys.modules["__main__"].__dict__
g["__file__"] = str(SRC)
try:
    exec(compile(src, str(SRC), "exec"), g)
except Exception:
    traceback.print_exc()
    sys.exit(1)
print("\n[smoke] SUCCESS - every cell executed without error")
