# %% [markdown]
# # FogField — Decentralized Fog-Node Repositioning as a Potential Game with Gossip
#
# Reference implementation for the methodology in **"FogField: A Q1-Ready Research Plan for
# Fully Decentralized Dynamic Fog-Node Repositioning via Potential Games and Gossip."**
#
# This notebook is built to run end-to-end on a **free Colab session** and to **survive
# disconnection**: every unit of work is checkpointed to Google Drive, so re-running the
# notebook resumes where it stopped instead of restarting from scratch.
#
# ---
#
# ## What is implemented
#
# | Plan item | Implemented here |
# |---|---|
# | **R1** Exact/weighted potential game, marginal-contribution (wonderful-life) utility | `Objective.phi`, `Game.utility`, plus a **numerical exactness test** (`verify_exact_potential`) |
# | **R1** Best-response with inertia and log-linear learning, annealed beta | `alg_best_response`, `alg_log_linear` |
# | **R2** Rotary-wing UAV propulsion power (Zeng-Xu-Zhang, TWC 2019) plus compute energy | `UAVPower`, validated against the published closed form |
# | **R3** **Measured** gossip overhead (real serialized bytes, not a model) | `GossipLayer` with struct-packed messages and byte/message counters |
# | **R4** Decentralized and centralized baselines plus a MILP optimum | the `alg_*` family, `solve_milp` (PuLP/CBC) |
# | **R5** Weighted scalarization, stated as an explicit limitation | `Weights`, limitations section |
# | **R6** Seven dynamic scenarios, seed-loss recovery, Byzantine nodes | `SCENARIOS` |
# | **T1-T3** FIP/convergence, stochastic stability, empirical PoA vs MILP | `verify_exact_potential`, convergence traces, `analyze_optimality_gap` |
#
# ## What is deliberately out of scope here, and why
#
# * **R3 tier (b), the physical Raspberry-Pi/Jetson testbed**, and **Mininet-WiFi/ns-3
#   emulation** cannot run inside Colab. What this notebook measures is **application-layer
#   protocol overhead**: real serialized payload bytes and message counts per round. MAC/PHY
#   overhead, contention, and wireless loss are **not** captured and must come from the
#   emulator and testbed tier before submission. The notebook emits the byte and message
#   traces that tier needs as its input.
# * **MAPPO/QMIX with neural critics.** A faithful CTDE implementation is not affordable on a
#   free-tier session alongside everything else. The MARL baseline here is **independent
#   Q-learning with parameter sharing**, labelled as such; it is *not* presented as MAPPO.
# * **Published baselines are reimplementations.** `alg_dpoa_style` is a D-POA-*style*
#   centralized swarm reference written from the method description. It is not the authors'
#   code and it is not a claim to reproduce their reported numbers.
#
# ## Datasets (both free, open, and downloaded automatically, no account required)
#
# 1. **EUA dataset** (Melbourne CBD) - real Optus base-station sites and user locations.
#    <https://github.com/swinedge/eua-dataset> - used for real deployment geometry.
# 2. **GeoLife GPS Trajectories v1.3** (Microsoft Research, Beijing) - real device mobility
#    with timestamps. <https://www.microsoft.com/en-us/download/details.aspx?id=52367>
#    - used for the real time-varying demand field.
#
# > The Telecom Italia *Milano* grid (DOI `10.7910/DVN/EGZHFV`), which the plan's scenarios
# > evoke, is **gated behind a Dataverse guestbook response** and cannot be fetched
# > unattended, so it is not used here. See the dataset notes at the end of the notebook.

# %% [markdown]
# ## 1 - Environment setup
#
# Installs only what Colab lacks. Safe to re-run.

# %%
import importlib
import subprocess
import sys


def _ensure(pip_name, import_name=None):
    import_name = import_name or pip_name
    try:
        importlib.import_module(import_name)
        return False
    except ImportError:
        print("installing " + pip_name + " ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=True)
        return True


for _pkg, _mod in [("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"),
                   ("networkx", "networkx"), ("matplotlib", "matplotlib"),
                   ("tqdm", "tqdm"), ("pulp", "pulp")]:
    _ensure(_pkg, _mod)

import hashlib
import json
import math
import os
import pickle
import struct
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm.auto import tqdm

matplotlib.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "figure.autolayout": True,
})

print("numpy", np.__version__, "| pandas", pd.__version__, "| networkx", nx.__version__)

# %% [markdown]
# ## 2 - Google Drive, run configuration, and paths
#
# `QUICK` controls the size of the experiment grid:
#
# * `QUICK = True` (default) - a reduced but complete grid that finishes in roughly
#   20-40 minutes on a free CPU session. Use it to verify that everything runs.
# * `QUICK = False` - the full grid intended for the paper. This takes many hours and is
#   **designed to be interrupted**: re-run the runner cell after a disconnect and it resumes.

# %%
QUICK = True                 # <<< set False for the full paper-scale grid
DRIVE_SUBDIR = "FogField"    # folder created under MyDrive


def mount_drive() -> Path:
    """Mount Drive on Colab; fall back to a local directory elsewhere."""
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        root = Path("./fogfield_out").resolve()
        root.mkdir(parents=True, exist_ok=True)
        print("[storage] not on Colab -> using local path " + str(root))
        return root

    mnt = Path("/content/drive")
    if not (mnt / "MyDrive").exists():
        drive.mount(str(mnt))
    root = mnt / "MyDrive" / DRIVE_SUBDIR
    root.mkdir(parents=True, exist_ok=True)
    print("[storage] Google Drive mounted -> " + str(root))
    return root


ROOT = mount_drive()
DIRS = {k: ROOT / k for k in ("cache", "ckpt", "results", "figs", "tables", "logs")}
for _d in DIRS.values():
    _d.mkdir(parents=True, exist_ok=True)

print("\n".join("  {:8s} {}".format(k, v) for k, v in DIRS.items()))

# %% [markdown]
# ## 3 - Checkpointing core
#
# Two independent layers, both writing to Drive.
#
# **(a) `Ledger`, work-unit level.** Every `(testbed, scenario, algorithm, n_nodes, seed)`
# combination is one *unit*, identified by a deterministic hash of its parameters. Each
# finished unit is written as its **own small JSON file**. A killed session loses at most one
# unit, and there is no large shared file that can be left half-written.
#
# **(b) `RunCheckpointer`, inside a single long run.** Saves node positions, the metric
# trace, and, importantly, **the RNG state** every `ckpt_every` rounds. Restoring the RNG
# state is what makes a resumed run bit-identical to an uninterrupted one, which matters
# because the learning dynamics are stochastic.
#
# All writes go through `atomic_write`: write a temporary file, `fsync`, then `os.replace`.
# `os.replace` is atomic, so a crash mid-write can never leave a truncated result behind,
# which is the relevant failure mode on Drive where a session can vanish at any instant.

# %%
def atomic_write(path: Path, data: bytes, _retries: int = 6) -> None:
    """Write-then-rename, so a crash mid-write can never leave a truncated file.

    The final `os.replace` is retried on `PermissionError`/`OSError`: on Windows this rename
    can transiently fail with "Access is denied" (WinError 5) if antivirus real-time scanning
    or the search indexer briefly opens the temp file the instant it's created - observed in
    practice during a real multi-hour local run, not a hypothetical. A Drive-mounted path on
    Colab can similarly return a transient I/O error. Neither reflects real contention over
    the destination path, so a short retry-with-backoff is the correct fix rather than
    treating it as a fatal error and losing an otherwise-complete checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp{}".format(os.getpid()))
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    for attempt in range(_retries):
        try:
            os.replace(tmp, path)
            return
        except (PermissionError, OSError):
            if attempt == _retries - 1:
                raise
            time.sleep(0.2 * (2 ** attempt))


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError("not JSON serialisable: {}".format(type(o)))


def write_json(path: Path, obj) -> None:
    atomic_write(path, json.dumps(obj, indent=1, default=_json_default).encode("utf-8"))


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def unit_key(**params) -> str:
    """Deterministic short id for a work unit."""
    blob = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


class Ledger:
    """One-file-per-unit result store with resume support."""

    def __init__(self, root: Path, name: str = "main"):
        self.dir = Path(root) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._done = set(p.stem for p in self.dir.glob("*.json"))

    def is_done(self, key: str) -> bool:
        return key in self._done

    def save(self, key: str, record: dict) -> None:
        # keys starting with "_" carry live objects for in-session use and are not persisted
        clean = {k: v for k, v in record.items() if not k.startswith("_")}
        write_json(self.dir / (key + ".json"), clean)
        self._done.add(key)

    def load_all(self) -> pd.DataFrame:
        rows = []
        for p in sorted(self.dir.glob("*.json")):
            rec = read_json(p)
            if rec is not None:
                rows.append(rec)
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self._done)


class RunCheckpointer:
    """Mid-run checkpointing, including RNG state, for long single runs."""

    VERSION = 2

    def __init__(self, key: str, ckpt_every: int = 50, enabled: bool = True):
        self.path = DIRS["ckpt"] / (key + ".pkl")
        self.ckpt_every = max(1, int(ckpt_every))
        self.enabled = enabled

    def load(self) -> Optional[dict]:
        if not (self.enabled and self.path.exists()):
            return None
        try:
            with open(self.path, "rb") as fh:
                state = pickle.load(fh)
            if state.get("_v") != self.VERSION:
                return None
            return state
        except Exception as exc:                 # corrupt or partial -> start clean
            print("[ckpt] ignoring unreadable checkpoint {}: {}".format(self.path.name, exc))
            return None

    def maybe_save(self, rnd: int, state: dict, force: bool = False) -> None:
        if not self.enabled:
            return
        if not force and rnd % self.ckpt_every:
            return
        payload = dict(state)
        payload["_v"] = self.VERSION
        atomic_write(self.path, pickle.dumps(payload, protocol=4))

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def rng_state_of(rng: np.random.Generator) -> dict:
    return rng.bit_generator.state


def rng_from_state(state: dict) -> np.random.Generator:
    bg = np.random.PCG64()
    bg.state = state
    return np.random.Generator(bg)


LEDGER = Ledger(DIRS["results"], "main")
print("[ledger] {} completed work unit(s) already on Drive".format(len(LEDGER)))
