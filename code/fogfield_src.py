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
from functools import cached_property
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


def _stream_seed(*parts) -> int:
    """64-bit seed derived from the parameters themselves (SHA-256), never from hash()."""
    blob = json.dumps([str(p) for p in parts]).encode("utf-8")
    return int(hashlib.sha256(blob).hexdigest()[:16], 16)


def seed_streams(testbed: str, n_nodes: int, seed: int, algo: str) -> Dict[str, np.random.Generator]:
    """Common random numbers across algorithms and scenarios.

    Four independent PCG64 streams. `deploy` (initial deployment), `scenario` (which nodes
    fail and when, flash-crowd centre) and `adversary` (which nodes lie, what they report)
    depend on (testbed, n, seed) only, so every algorithm and every scenario of a given
    (testbed, n, seed) starts from the same deployment, suffers the same events and faces the
    same attacker. `algo` additionally depends on the algorithm name, so the byzantine and
    static runs of one algorithm also share its own decision randomness and differ only in the
    falsified messages. v1.0 seeded one stream from Python's per-process hash() of the unit
    key, which made units irreproducible across interpreters and unpaired across algorithms.
    """
    base = (str(testbed), int(n_nodes), int(seed))
    return {
        "deploy": np.random.default_rng(_stream_seed("deploy", *base)),
        "scenario": np.random.default_rng(_stream_seed("scenario", *base)),
        "adversary": np.random.default_rng(_stream_seed("adversary", *base)),
        "algo": np.random.default_rng(_stream_seed("algo", *base, str(algo))),
    }


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

    VERSION = 3

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
# %% [markdown]
# ## 4 - Datasets
#
# Two real, open, no-login datasets. Both are downloaded once and then **cached to Drive as
# compact `.npz` artifacts**, so a re-run after a disconnect skips the download entirely -
# the first and most expensive thing free-tier sessions otherwise repeat.
#
# ### EUA (Melbourne CBD)
# Real Optus base-station sites and real user locations in the Melbourne CBD. This is the
# standard dataset of the edge-server-placement literature. Two CSVs, a few hundred kB.
# * **Base-station sites** provide the *initial deployment* of the fog nodes - a real-world
#   starting configuration rather than a random one.
# * **User locations** provide the IoT device field.
#
# ### GeoLife v1.3 (Beijing)
# 182 users, 17,621 trajectories, 1.2 million km of real GPS traces with timestamps.
# Used to build a **real time-varying demand field**: points are binned onto a spatial grid
# and onto 10-minute time-of-day slots.
#
# > **Stated assumption.** GeoLife logging is intermittent, so few users are recording at any
# > single instant. Following common practice, trajectories are **pooled across days onto a
# > common time-of-day clock**, giving a dense diurnal demand surface. This measures *where
# > and when demand concentrates*, not a single day's simultaneous population. The notebook
# > records this in the dataset metadata so the limitation travels with the results.

# %%
EUA_BASE = "https://raw.githubusercontent.com/swinedge/eua-dataset/master"
EUA_SITES_URL = EUA_BASE + "/edge-servers/site-optus-melbCBD.csv"
EUA_USERS_URL = EUA_BASE + "/users/users-melbcbd-generated.csv"
GEOLIFE_URL = ("https://download.microsoft.com/download/F/4/8/"
               "F4894AA5-FDBC-481E-9285-D5F8C4C4F039/Geolife%20Trajectories%201.3.zip")

EARTH_M_PER_DEG_LAT = 110540.0
EARTH_M_PER_DEG_LON = 111320.0


def download(url: str, dest: Path, desc: str = "") -> Path:
    """Download to Drive cache once; reuse thereafter. Partial files never persist."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        print("[cache] hit  {} ({:.1f} MB)".format(dest.name, dest.stat().st_size / 1e6))
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print("[cache] miss {} -> downloading {}".format(dest.name, desc or url))
    req = urllib.request.Request(url, headers={"User-Agent": "FogField/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        bar = tqdm(total=total or None, unit="B", unit_scale=True, desc=dest.name)
        with open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                bar.update(len(chunk))
        bar.close()
    os.replace(tmp, dest)
    return dest


def to_local_metres(lat: np.ndarray, lon: np.ndarray,
                    lat0: float, lon0: float) -> np.ndarray:
    """Equirectangular projection to a local east-north plane in metres."""
    x = (lon - lon0) * EARTH_M_PER_DEG_LON * math.cos(math.radians(lat0))
    y = (lat - lat0) * EARTH_M_PER_DEG_LAT
    return np.stack([x, y], axis=1).astype(np.float64)


@dataclass
class Testbed:
    """A physical setting: where devices are, how much they demand, and when."""
    name: str
    device_xy: np.ndarray        # (M, 2) device positions, metres
    device_w: np.ndarray         # (T, M) demand weight per slot; T == 1 if static
    init_sites_xy: np.ndarray    # (S, 2) real infrastructure sites -> initial node positions
    bounds: Tuple[float, float, float, float]
    meta: dict = field(default_factory=dict)

    @property
    def n_devices(self) -> int:
        return int(self.device_xy.shape[0])

    @property
    def n_slots(self) -> int:
        return int(self.device_w.shape[0])

    def weights(self, t: int) -> np.ndarray:
        return self.device_w[t % self.n_slots]

    def extent(self) -> Tuple[float, float]:
        x0, x1, y0, y1 = self.bounds
        return x1 - x0, y1 - y0

    def describe(self) -> str:
        w, h = self.extent()
        return ("{}: {} devices, {} sites, {} demand slot(s), area {:.2f} x {:.2f} km"
                .format(self.name, self.n_devices, len(self.init_sites_xy),
                        self.n_slots, w / 1000, h / 1000))


# %%
def load_eua_melbcbd(max_devices: int = 600, seed: int = 0) -> Testbed:
    """EUA Melbourne CBD: real Optus sites + real user locations."""
    cache = DIRS["cache"] / "eua_melbcbd_{}.npz".format(max_devices)
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return Testbed(name="MEL", device_xy=z["device_xy"], device_w=z["device_w"],
                       init_sites_xy=z["sites_xy"], bounds=tuple(z["bounds"]),
                       meta=json.loads(str(z["meta"])))

    sites_csv = download(EUA_SITES_URL, DIRS["cache"] / "eua_sites_melbcbd.csv", "EUA sites")
    users_csv = download(EUA_USERS_URL, DIRS["cache"] / "eua_users_melbcbd.csv", "EUA users")

    sites = pd.read_csv(sites_csv)
    users = pd.read_csv(users_csv)
    sites = sites.dropna(subset=["LATITUDE", "LONGITUDE"]).drop_duplicates(
        subset=["LATITUDE", "LONGITUDE"])
    users = users.dropna(subset=["Latitude", "Longitude"])

    lat0 = float(users["Latitude"].mean())
    lon0 = float(users["Longitude"].mean())

    dev = to_local_metres(users["Latitude"].to_numpy(), users["Longitude"].to_numpy(),
                          lat0, lon0)
    sit = to_local_metres(sites["LATITUDE"].to_numpy(), sites["LONGITUDE"].to_numpy(),
                          lat0, lon0)

    # keep sites inside the user bounding box (the CBD), with a small margin
    pad = 300.0
    keep = ((sit[:, 0] > dev[:, 0].min() - pad) & (sit[:, 0] < dev[:, 0].max() + pad) &
            (sit[:, 1] > dev[:, 1].min() - pad) & (sit[:, 1] < dev[:, 1].max() + pad))
    sit = sit[keep]

    if len(dev) > max_devices:
        idx = np.random.default_rng(seed).choice(len(dev), max_devices, replace=False)
        dev = dev[np.sort(idx)]

    device_w = np.ones((1, len(dev)), dtype=np.float64)
    bounds = (float(dev[:, 0].min()), float(dev[:, 0].max()),
              float(dev[:, 1].min()), float(dev[:, 1].max()))
    meta = {
        "source": "EUA dataset (swinedge/eua-dataset), Melbourne CBD",
        "urls": [EUA_SITES_URL, EUA_USERS_URL],
        "n_sites_raw": int(len(sites)), "n_users_raw": int(len(users)),
        "origin_latlon": [lat0, lon0],
        "temporal": "static (single demand slot); dynamics come from the scenario engine",
    }
    np.savez_compressed(cache, device_xy=dev, device_w=device_w, sites_xy=sit,
                        bounds=np.array(bounds), meta=json.dumps(meta))
    return Testbed("MEL", dev, device_w, sit, bounds, meta)


# %%
def load_geolife(max_users: Optional[int] = None, grid_m: float = 400.0, slot_min: int = 60,
                 top_cells: int = 400, roi_km: float = 12.0) -> Testbed:
    """GeoLife v1.3 -> real time-varying demand field over central Beijing.

    Trajectory points are binned onto a `grid_m` spatial grid and onto time-of-day slots of
    `slot_min` minutes, pooled across all days. The busiest `top_cells` cells become the
    device field; each cell's per-slot point count is its demand weight.
    """
    tag = "geolife_u{}_g{:.0f}_s{}_c{}".format("all" if max_users is None else max_users,
                                                 grid_m, slot_min, top_cells)
    cache = DIRS["cache"] / (tag + ".npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return Testbed(name="BEJ", device_xy=z["device_xy"], device_w=z["device_w"],
                       init_sites_xy=z["sites_xy"], bounds=tuple(z["bounds"]),
                       meta=json.loads(str(z["meta"])))

    zpath = download(GEOLIFE_URL, DIRS["cache"] / "geolife_1.3.zip", "GeoLife v1.3 (~300 MB)")

    lat0, lon0 = 39.9042, 116.4074                    # Tiananmen, ROI centre
    half = roi_km * 1000.0 / 2.0
    n_slots = max(1, int(round(24 * 60 / slot_min)))

    counts: Dict[Tuple[int, int, int], float] = {}
    n_pts = 0
    with zipfile.ZipFile(zpath) as zf:
        plts = [n for n in zf.namelist() if n.lower().endswith(".plt")]
        # group by user directory so `max_users` selects whole users, not random files
        by_user: Dict[str, List[str]] = {}
        for name in plts:
            parts = name.split("/")
            # paths look like "Geolife Trajectories 1.3/Data/<user>/Trajectory/<file>.plt";
            # v1.0 took parts[1], which is the literal "Data" for every file, so the user cap
            # never applied (all 182 users were read) and the manifest reported 1 user
            uid = parts[parts.index("Data") + 1] if "Data" in parts[:-1] else "000"
            by_user.setdefault(uid, []).append(name)
        users = sorted(by_user) if max_users is None else sorted(by_user)[:max_users]
        files = [f for u in users for f in by_user[u]]

        for name in tqdm(files, desc="GeoLife .plt", unit="file"):
            try:
                raw = zf.read(name).decode("ascii", "ignore").splitlines()
            except Exception:
                continue
            for line in raw[6:]:                       # first 6 lines are a header
                f = line.split(",")
                if len(f) < 7:
                    continue
                try:
                    lat = float(f[0]); lon = float(f[1]); hhmmss = f[6]
                except ValueError:
                    continue
                x = (lon - lon0) * EARTH_M_PER_DEG_LON * math.cos(math.radians(lat0))
                y = (lat - lat0) * EARTH_M_PER_DEG_LAT
                if abs(x) > half or abs(y) > half:
                    continue
                try:
                    hh, mm = int(hhmmss[0:2]), int(hhmmss[3:5])
                except (ValueError, IndexError):
                    continue
                slot = ((hh * 60 + mm) // slot_min) % n_slots
                gx = int(math.floor(x / grid_m))
                gy = int(math.floor(y / grid_m))
                key = (gx, gy, slot)
                counts[key] = counts.get(key, 0.0) + 1.0
                n_pts += 1

    if not counts:
        raise RuntimeError("GeoLife parse produced no points inside the ROI - "
                           "check roi_km / the cached zip.")

    cell_tot: Dict[Tuple[int, int], float] = {}
    for (gx, gy, _s), c in counts.items():
        cell_tot[(gx, gy)] = cell_tot.get((gx, gy), 0.0) + c
    cells = sorted(cell_tot, key=lambda k: -cell_tot[k])[:top_cells]
    cell_ix = {c: i for i, c in enumerate(cells)}

    device_xy = np.array([[(gx + 0.5) * grid_m, (gy + 0.5) * grid_m] for gx, gy in cells],
                         dtype=np.float64)
    device_w = np.zeros((n_slots, len(cells)), dtype=np.float64)
    for (gx, gy, s), c in counts.items():
        j = cell_ix.get((gx, gy))
        if j is not None:
            device_w[s, j] = c

    # normalise so the mean slot carries unit average weight (keeps objective scales comparable)
    scale = device_w.sum(axis=1).mean()
    if scale > 0:
        device_w *= len(cells) / scale

    # initial sites: the busiest cells by total demand, a realistic greedy deployment
    order = np.argsort(-device_w.sum(axis=0))
    sites_xy = device_xy[order[: max(20, len(cells) // 8)]]

    bounds = (float(device_xy[:, 0].min()), float(device_xy[:, 0].max()),
              float(device_xy[:, 1].min()), float(device_xy[:, 1].max()))
    meta = {
        "source": "GeoLife GPS Trajectories v1.3 (Microsoft Research Asia)",
        "url": GEOLIFE_URL,
        "n_users_used": len(users), "n_points_in_roi": int(n_pts),
        "grid_m": grid_m, "slot_min": slot_min, "n_slots": n_slots,
        "roi_km": roi_km, "roi_centre_latlon": [lat0, lon0],
        "pooling": ("trajectories pooled across days onto a common time-of-day clock; "
                    "this is a diurnal demand surface, not one day's simultaneous population"),
    }
    np.savez_compressed(cache, device_xy=device_xy, device_w=device_w, sites_xy=sites_xy,
                        bounds=np.array(bounds), meta=json.dumps(meta))
    return Testbed("BEJ", device_xy, device_w, sites_xy, bounds, meta)


# %% [markdown]
# ### Load the testbeds
#
# MEL loads in seconds. BEJ downloads ~300 MB on the **first run only** and then reads from
# the Drive cache. If the GeoLife download fails (it is a large file on a Microsoft CDN), the
# notebook continues with MEL alone and says so rather than dying.

# %%
TESTBEDS: Dict[str, Testbed] = {}

TESTBEDS["MEL"] = load_eua_melbcbd(max_devices=600 if QUICK else 816)
print(TESTBEDS["MEL"].describe())

try:
    TESTBEDS["BEJ"] = load_geolife(
        max_users=None,                 # all 182 users (see the parser note above)
        grid_m=400.0,
        slot_min=60 if QUICK else 30,
        top_cells=250 if QUICK else 500,
    )
    print(TESTBEDS["BEJ"].describe())
except Exception as exc:
    print("[warn] GeoLife unavailable ({}: {}). Continuing with MEL only.".format(
        type(exc).__name__, exc))

print("\nActive testbeds:", list(TESTBEDS))


# %%
def plot_testbeds():
    tbs = list(TESTBEDS.values())
    fig, axes = plt.subplots(1, len(tbs), figsize=(5.4 * len(tbs), 4.6), squeeze=False)
    for ax, tb in zip(axes[0], tbs):
        w = tb.device_w.mean(axis=0)
        sc = ax.scatter(tb.device_xy[:, 0] / 1000, tb.device_xy[:, 1] / 1000,
                        c=w, s=14, cmap="viridis", alpha=0.85, linewidths=0)
        ax.scatter(tb.init_sites_xy[:, 0] / 1000, tb.init_sites_xy[:, 1] / 1000,
                   marker="^", s=45, facecolor="none", edgecolor="crimson", linewidths=1.0,
                   label="infrastructure sites")
        plt.colorbar(sc, ax=ax, label="mean demand weight")
        ax.set_title("{}  -  {} devices".format(tb.name, tb.n_devices))
        ax.set_xlabel("east (km)"); ax.set_ylabel("north (km)")
        ax.set_aspect("equal"); ax.legend(loc="best", fontsize=7)
    fig.savefig(DIRS["figs"] / "fig0_testbeds.png", bbox_inches="tight")
    plt.show()


def plot_demand_profile():
    tb = TESTBEDS.get("BEJ")
    if tb is None or tb.n_slots < 2:
        print("[skip] no time-varying testbed loaded")
        return
    tot = tb.device_w.sum(axis=1)
    hrs = np.arange(tb.n_slots) * (24.0 / tb.n_slots)
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.plot(hrs, tot, marker="o", ms=3)
    ax.set_xlabel("time of day (h)"); ax.set_ylabel("total demand weight")
    ax.set_title("BEJ - real diurnal demand (GeoLife, pooled across days)")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3))
    fig.savefig(DIRS["figs"] / "fig0b_demand_profile.png", bbox_inches="tight")
    plt.show()


plot_testbeds()
plot_demand_profile()
# %% [markdown]
# ## 5 - Physical model (R2): rotary-wing UAV propulsion plus compute energy
#
# The closed-form propulsion power of a rotary-wing UAV in level flight at speed `V`
# (Zeng, Xu and Zhang, *IEEE Trans. Wireless Communications* 18(4):2329-2345, 2019,
# DOI 10.1109/TWC.2019.2902559):
#
# $$P(V)=\underbrace{P_0\left(1+\frac{3V^2}{U_{\mathrm{tip}}^2}\right)}_{\text{blade profile}}
# +\underbrace{P_i\left(\sqrt{1+\frac{V^4}{4v_0^4}}-\frac{V^2}{2v_0^2}\right)^{1/2}}_{\text{induced}}
# +\underbrace{\tfrac12 d_0\rho s A V^3}_{\text{parasite}}$$
#
# with the paper's parameters. Hovering costs `P(0) = P_0 + P_i = 168.49 W`. A compute-power
# term (idle plus load-proportional) and a battery budget are layered on top.

# %%
@dataclass(frozen=True)
class UAVPower:
    """Zeng-Xu-Zhang (TWC 2019) rotary-wing propulsion power, plus onboard compute power."""
    P0: float = 79.86        # blade profile power in hover (W)
    Pi: float = 88.63        # induced power in hover (W)
    U_tip: float = 120.0     # rotor blade tip speed (m/s)
    v0: float = 4.03         # mean rotor induced velocity in hover (m/s)
    d0: float = 0.6          # fuselage drag ratio
    rho: float = 1.225       # air density (kg/m^3)
    s: float = 0.05          # rotor solidity
    A: float = 0.503         # rotor disc area (m^2)
    V_max: float = 20.0      # max horizontal speed (m/s)
    P_cpu_idle: float = 4.0  # idle compute power (W)
    P_cpu_busy: float = 12.0 # additional compute power at full load (W)
    battery_Wh: float = 275.0
    loiter: bool = True      # station-keeping by loitering at V_me rather than hovering (v2)

    def propulsion(self, V):
        V = np.asarray(V, dtype=np.float64)
        blade = self.P0 * (1.0 + 3.0 * V ** 2 / self.U_tip ** 2)
        inner = np.sqrt(1.0 + V ** 4 / (4.0 * self.v0 ** 4)) - V ** 2 / (2.0 * self.v0 ** 2)
        induced = self.Pi * np.sqrt(np.maximum(inner, 0.0))
        parasite = 0.5 * self.d0 * self.rho * self.s * self.A * V ** 3
        return blade + induced + parasite

    @property
    def P_hover(self) -> float:
        return float(self.P0 + self.Pi)

    @cached_property
    def V_me(self) -> float:
        """Maximum-endurance speed: the speed that minimises propulsion power."""
        V = np.linspace(0.0, self.V_max, 20001)
        return float(V[int(np.argmin(self.propulsion(V)))])

    @property
    def P_me(self) -> float:
        return float(self.propulsion(self.V_me))

    def flight_energy(self, dist_m, dt_s: float):
        """Energy (J) to travel `dist_m` within a step of `dt_s`.

        v2 (loiter=True): the UAV flies the whole step at speed max(d/dt, V_me), capped at
        V_max; a node that needs to cover less than V_me*dt (staying put included) loiters at
        the maximum-endurance speed V_me instead of hovering, because hovering costs more
        (P(0) = 168.5 W against P(V_me) = 126.0 W with the paper's parameters). The slot energy
        is therefore non-decreasing in the distance and minimal for staying, which is what
        makes a stationary configuration a possible rest point of the dynamics (T1).

        loiter=False reproduces v1.0: fly at the slowest sufficient speed, hover for the rest.
        """
        dist = np.asarray(dist_m, dtype=np.float64)
        if self.loiter:
            V = np.clip(np.maximum(dist / max(dt_s, 1e-9), self.V_me), 0.0, self.V_max)
            return self.propulsion(V) * dt_s
        v_need = dist / max(dt_s, 1e-9)
        V = np.clip(v_need, 0.0, self.V_max)
        t_fly = np.where(V > 1e-9, dist / np.maximum(V, 1e-9), 0.0)
        t_fly = np.minimum(t_fly, dt_s)
        t_hov = np.maximum(dt_s - t_fly, 0.0)
        return self.propulsion(V) * t_fly + self.P_hover * t_hov

    def compute_energy(self, load_frac, dt_s: float):
        load = np.clip(np.asarray(load_frac, dtype=np.float64), 0.0, 1.0)
        return (self.P_cpu_idle + self.P_cpu_busy * load) * dt_s

    def max_step_m(self, dt_s: float) -> float:
        return float(self.V_max * dt_s)


UAV = UAVPower()

# validation against the published closed form
_hover = float(UAV.propulsion(0.0))
assert abs(_hover - UAV.P_hover) < 1e-9, "hover power must equal P0 + Pi"
_V = np.linspace(0, 40, 400)
_P = UAV.propulsion(_V)
_v_opt = float(_V[int(np.argmin(_P))])
print("P(0)          = {:.2f} W   (P0 + Pi = {:.2f} W)".format(_hover, UAV.P_hover))
print("min power at V= {:.1f} m/s -> {:.2f} W".format(_v_opt, float(_P.min())))
print("battery       = {:.0f} Wh = {:.0f} kJ -> {:.1f} min of hover".format(
    UAV.battery_Wh, UAV.battery_Wh * 3600 / 1000, UAV.battery_Wh * 3600 / UAV.P_hover / 60))
print("loiter (v2)   : V_me = {:.2f} m/s, P(V_me) = {:.2f} W; a station-keeping node loiters "
      "at V_me, so staying is the cheapest move".format(UAV.V_me, UAV.P_me))
_e = UAV.flight_energy(np.array([0.0, 120.0, 306.0, 400.0, 600.0]), 30.0)
assert np.all(np.diff(_e) >= -1e-9), "slot energy must be non-decreasing in distance"

fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.plot(_V, _P)
ax.axvline(_v_opt, ls="--", lw=0.8, color="crimson")
ax.annotate("min at {:.1f} m/s".format(_v_opt), (_v_opt, float(_P.min())),
            textcoords="offset points", xytext=(8, 18), fontsize=8, color="crimson")
ax.set_xlabel("horizontal speed V (m/s)"); ax.set_ylabel("propulsion power (W)")
ax.set_title("Rotary-wing UAV power (Zeng-Xu-Zhang, TWC 2019)")
fig.savefig(DIRS["figs"] / "fig1_uav_power.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6 - System model and the global objective $\Phi$
#
# $$\Phi(a)=w_c\,\mathrm{Coverage}(a)+w_k\,\mathrm{Connectivity}(a)-w_e\,\mathrm{Energy}(a)$$
#
# Each term is normalised by a **constant that does not depend on which nodes are active**
# (total device weight, total node count, a reference per-node energy). That detail is what
# makes $\Phi$ well defined on sub-populations, which the marginal-contribution utility
# requires - normalising by the *active* count instead would silently break the exact
# potential property.
#
# * **Coverage** - a device is served if some active node is within `R_cov`. With
#   `capacity=None` (default) coverage is the uncapacitated weighted union, which is monotone
#   submodular and puts the game in Vetta's valid-utility class, so `PoA <= 2` applies.
#   Setting a finite `capacity` switches to greedy capacitated assignment; that variant is
#   **outside** the valid-utility structure and the 1/2 bound is not claimed for it.
# * **Connectivity** - fraction of all nodes lying in the largest connected component of the
#   communication graph (edge when `||a_i - a_j|| <= R_comm`).
# * **Energy** - flight energy for this step plus compute energy, from `UAVPower`.
#
# > **R5, stated plainly.** This is a *weighted scalarisation*. It returns one point on the
# > trade-off surface per weight vector, not a Pareto front. No hypervolume is reported.

# %%
@dataclass(frozen=True)
class Weights:
    coverage: float = 1.0
    connectivity: float = 0.35
    energy: float = 0.15


@dataclass
class Environment:
    """Everything fixed about a problem instance, plus the evaluation of Phi."""
    testbed: Testbed
    n_nodes: int
    R_cov: float = 500.0
    R_comm: float = 900.0
    dt_s: float = 30.0
    weights: Weights = field(default_factory=Weights)
    capacity: Optional[float] = None       # None -> uncapacitated (valid-utility structure)
    uav: UAVPower = field(default_factory=UAVPower)
    lattice_m: float = 120.0               # action-lattice spacing

    def __post_init__(self):
        self.W_total = float(self.testbed.device_w.sum(axis=1).mean())
        self.E_ref = float(self.uav.P_hover * self.dt_s)
        x0, x1, y0, y1 = self.testbed.bounds
        pad = self.R_cov * 0.5
        self.lat_x = np.arange(x0 - pad, x1 + pad + 1e-9, self.lattice_m)
        self.lat_y = np.arange(y0 - pad, y1 + pad + 1e-9, self.lattice_m)
        self.max_step = self.uav.max_step_m(self.dt_s)

    # ---- geometry helpers -------------------------------------------------
    def snap(self, xy: np.ndarray) -> np.ndarray:
        """Snap positions onto the action lattice (the discretised strategy set)."""
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        ix = np.clip(np.round((xy[:, 0] - self.lat_x[0]) / self.lattice_m), 0,
                     len(self.lat_x) - 1).astype(int)
        iy = np.clip(np.round((xy[:, 1] - self.lat_y[0]) / self.lattice_m), 0,
                     len(self.lat_y) - 1).astype(int)
        return np.stack([self.lat_x[ix], self.lat_y[iy]], axis=1)

    def action_set(self, xy_i: np.ndarray, max_actions: int = 25) -> np.ndarray:
        """Lattice cells reachable by one node in one step (its strategy set A_i)."""
        r = self.max_step
        xs = self.lat_x[(self.lat_x >= xy_i[0] - r) & (self.lat_x <= xy_i[0] + r)]
        ys = self.lat_y[(self.lat_y >= xy_i[1] - r) & (self.lat_y <= xy_i[1] + r)]
        if len(xs) == 0:
            xs = np.array([self.snap(xy_i)[0, 0]])
        if len(ys) == 0:
            ys = np.array([self.snap(xy_i)[0, 1]])
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        cand = np.stack([gx.ravel(), gy.ravel()], axis=1)
        d = np.linalg.norm(cand - xy_i[None, :], axis=1)
        cand = cand[d <= r + 1e-6]
        if len(cand) == 0:
            return self.snap(xy_i)
        if len(cand) > max_actions:                  # keep the nearest ring plus a sample
            d = np.linalg.norm(cand - xy_i[None, :], axis=1)
            keep = np.argsort(d)[:max_actions]
            cand = cand[keep]
        return cand

    # ---- objective terms --------------------------------------------------
    def coverage(self, pos: np.ndarray, active: np.ndarray, w: np.ndarray) -> float:
        if not active.any():
            return 0.0
        P = pos[active]
        d2 = ((P[:, None, :] - self.testbed.device_xy[None, :, :]) ** 2).sum(-1)
        inr = d2 <= self.R_cov ** 2
        if self.capacity is None:
            return float(w[inr.any(axis=0)].sum())
        # greedy capacitated assignment: heaviest device first, nearest feasible node
        cap = np.full(P.shape[0], float(self.capacity))
        served = 0.0
        order = np.argsort(-w)
        for m in order:
            if w[m] <= 0:
                break
            cand = np.flatnonzero(inr[:, m] & (cap >= w[m]))
            if cand.size:
                j = cand[np.argmin(d2[cand, m])]
                cap[j] -= w[m]
                served += w[m]
        return float(served)

    def load_fraction(self, pos: np.ndarray, active: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Per-node normalised served load, used for compute energy."""
        load = np.zeros(pos.shape[0])
        if not active.any():
            return load
        idx = np.flatnonzero(active)
        P = pos[idx]
        d2 = ((P[:, None, :] - self.testbed.device_xy[None, :, :]) ** 2).sum(-1)
        inr = d2 <= self.R_cov ** 2
        nearest = np.where(inr.any(axis=0), np.argmin(np.where(inr, d2, np.inf), axis=0), -1)
        for k, j in enumerate(idx):
            sel = nearest == k
            load[j] = w[sel].sum()
        denom = self.capacity if self.capacity else max(self.W_total / max(len(idx), 1), 1e-9)
        return np.clip(load / denom, 0.0, 1.0)

    def connectivity(self, pos: np.ndarray, active: np.ndarray) -> float:
        """Size of the largest connected component of the communication graph.

        Uses scipy's connected_components rather than networkx: this runs once per round
        per unit across thousands of units, and graph-object construction dominated the
        profile at the larger node counts.
        """
        idx = np.flatnonzero(active)
        if idx.size == 0:
            return 0.0
        if idx.size == 1:
            return 1.0
        P = pos[idx]
        d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        adj = d2 <= self.R_comm ** 2
        np.fill_diagonal(adj, False)
        n_comp, labels = connected_components(csr_matrix(adj), directed=False)
        if n_comp <= 1:
            return float(idx.size)
        return float(np.bincount(labels).max())

    def algebraic_connectivity(self, pos: np.ndarray, active: np.ndarray) -> float:
        """Fiedler value of the communication graph - reported, not optimised."""
        idx = np.flatnonzero(active)
        if idx.size < 2:
            return 0.0
        P = pos[idx]
        d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        A = (d2 <= self.R_comm ** 2).astype(float)
        np.fill_diagonal(A, 0.0)
        L = np.diag(A.sum(1)) - A
        ev = np.linalg.eigvalsh(L)
        return float(ev[1]) if len(ev) > 1 else 0.0

    def energy(self, pos: np.ndarray, prev: np.ndarray, active: np.ndarray,
               w: np.ndarray) -> float:
        idx = np.flatnonzero(active)
        if idx.size == 0:
            return 0.0
        dist = np.linalg.norm(pos[idx] - prev[idx], axis=1)
        e_fly = self.uav.flight_energy(dist, self.dt_s)
        e_cpu = self.uav.compute_energy(self.load_fraction(pos, active, w)[idx], self.dt_s)
        return float(np.sum(e_fly + e_cpu))

    # ---- the potential function -------------------------------------------
    def phi(self, pos: np.ndarray, prev: np.ndarray, active: np.ndarray,
            w: np.ndarray, parts: bool = False):
        """Global objective. Denominators are constants, never functions of `active`."""
        cov = self.coverage(pos, active, w)
        con = self.connectivity(pos, active)
        ene = self.energy(pos, prev, active, w)
        N = float(self.n_nodes)
        val = (self.weights.coverage * cov / max(self.W_total, 1e-9)
               + self.weights.connectivity * con / N
               - self.weights.energy * ene / (N * self.E_ref))
        if parts:
            # Phi uses the fixed denominator W_total, which the potential property requires.
            # The *reported* coverage fraction is against the demand actually present this
            # round, otherwise a flash crowd (which inflates weights) reads as >100% served.
            w_now = float(np.sum(w))
            return val, {"coverage_w": cov,
                         "coverage_frac": cov / max(w_now, 1e-9),
                         "coverage_norm": cov / max(self.W_total, 1e-9),
                         "giant_frac": con / N, "energy_J": ene,
                         "energy_per_node_J": ene / max(int(active.sum()), 1)}
        return float(val)


# %% [markdown]
# ## 7 - The potential game (R1) and its numerical verification
#
# Players are the fog nodes; the strategy set $A_i$ is the set of lattice cells reachable in
# one step; the utility is the **marginal contribution / wonderful-life utility**
#
# $$u_i(a_i,a_{-i})=\Phi(a_i,a_{-i})-\Phi(a_{-i},\varnothing_i)$$
#
# where $\varnothing_i$ means node $i$ is absent. Because the subtracted term does not depend
# on $a_i$, $\Phi$ is an **exact potential**:
#
# $$u_i(a_i,a_{-i})-u_i(a_i',a_{-i})=\Phi(a_i,a_{-i})-\Phi(a_i',a_{-i}).$$
#
# The algebra is one line, so the test below is not really testing the theorem. It is testing
# **this implementation**: if any term of `phi` leaked a dependence on $a_i$ into the
# baseline - the easy bug here is an energy or normalisation term that quietly sees node
# $i$ - the identity would fail numerically. Reviewers ask for exactly this kind of check.

# %%
class Game:
    """The strategic game: marginal-contribution utility over the discretised action sets."""

    def __init__(self, env: Environment):
        self.env = env

    def phi_without(self, pos, prev, active, w, i: int) -> float:
        a2 = active.copy()
        a2[i] = False
        return self.env.phi(pos, prev, a2, w)

    def utility(self, pos, prev, active, w, i: int) -> float:
        return self.env.phi(pos, prev, active, w) - self.phi_without(pos, prev, active, w, i)


def verify_exact_potential(env: Environment, n_trials: int = 40, seed: int = 7) -> dict:
    """Empirically check u_i(a_i,.) - u_i(a_i',.) == Phi(a_i,.) - Phi(a_i',.)."""
    rng = np.random.default_rng(seed)
    game = Game(env)
    tb = env.testbed
    x0, x1, y0, y1 = tb.bounds
    viol = []
    for _ in range(n_trials):
        pos = env.snap(np.stack([rng.uniform(x0, x1, env.n_nodes),
                                 rng.uniform(y0, y1, env.n_nodes)], axis=1))
        prev = env.snap(pos + rng.normal(0, 60, pos.shape))
        active = np.ones(env.n_nodes, dtype=bool)
        w = tb.weights(int(rng.integers(tb.n_slots)))
        i = int(rng.integers(env.n_nodes))

        alt = env.action_set(pos[i])
        a_new = alt[int(rng.integers(len(alt)))]

        u_before = game.utility(pos, prev, active, w, i)
        phi_before = env.phi(pos, prev, active, w)

        pos2 = pos.copy()
        pos2[i] = a_new
        u_after = game.utility(pos2, prev, active, w, i)
        phi_after = env.phi(pos2, prev, active, w)

        viol.append(abs((u_after - u_before) - (phi_after - phi_before)))

    viol = np.asarray(viol)
    out = {"n_trials": n_trials, "max_violation": float(viol.max()),
           "mean_violation": float(viol.mean()),
           "exact": bool(viol.max() < 1e-9)}
    return out


_env_check = Environment(TESTBEDS["MEL"], n_nodes=12)
_res = verify_exact_potential(_env_check, n_trials=40)
print("exact-potential check (uncapacitated):", _res)
assert _res["exact"], "phi/utility pair is NOT an exact potential - implementation bug"

_env_cap = Environment(TESTBEDS["MEL"], n_nodes=12, capacity=80.0)
_res_cap = verify_exact_potential(_env_cap, n_trials=25)
print("exact-potential check (capacitated)  :", _res_cap)
print("\nNote: exactness holds for both variants because it is structural in the "
      "marginal-contribution utility.\nThe submodular / valid-utility structure that the "
      "PoA <= 2 bound needs is a separate property,\nand it holds only for the "
      "uncapacitated variant.")
write_json(DIRS["tables"] / "potential_check.json",
           {"uncapacitated": _res, "capacitated": _res_cap})
# %% [markdown]
# ## 8 - Gossip layer with *measured* overhead (R3)
#
# The plan is explicit that communication cost must be **measured, not modelled**. Inside
# Colab the honest version of that is to run the real protocol and count the real bytes: each
# gossip message is genuinely serialised with `struct`, and the layer reports
# `len(payload)` - not a formula.
#
# **Message format** (little-endian, packed):
#
# | field | type | bytes |
# |---|---|---|
# | sender id | `uint16` | 2 |
# | sequence | `uint16` | 2 |
# | position x, y | `float32` x2 | 8 |
# | residual battery | `float32` | 4 |
# | local demand hint | `float32` | 4 |
# | neighbour-list length | `uint8` | 1 |
# | neighbour ids | `uint16` x k | 2k |
#
# Push-pull: an exchange with a neighbour costs one push plus one pull message. The
# neighbour-id list is what gives every node a **2-hop view**, which is what the local
# connectivity term needs in order to tell "I am a bridge" from "I am redundant".
#
# > **Scope, stated so it is not mistaken for more than it is.** These are application-layer
# > payload bytes. 802.11 MAC framing, ACKs, retransmission, contention and loss are **not**
# > here; they belong to the Mininet-WiFi/ns-3 tier and the physical testbed. The per-round
# > message and byte traces exported below are the inputs that tier needs.

# %%
_HDR = struct.Struct("<HHffffB")          # id, seq, x, y, battery, demand hint, nbr count


@dataclass
class LocalView:
    """What one node knows: neighbour states and their neighbour lists (2-hop)."""
    pos: Dict[int, np.ndarray] = field(default_factory=dict)
    battery: Dict[int, float] = field(default_factory=dict)
    demand_hint: Dict[int, float] = field(default_factory=dict)
    nbrs_of: Dict[int, List[int]] = field(default_factory=dict)
    last_seen: Dict[int, int] = field(default_factory=dict)

    def forget_older_than(self, now: int, ttl: int) -> None:
        stale = [j for j, t in self.last_seen.items() if now - t > ttl]
        for j in stale:
            self.pos.pop(j, None)
            self.battery.pop(j, None)
            self.demand_hint.pop(j, None)
            self.nbrs_of.pop(j, None)
            self.last_seen.pop(j, None)


class GossipLayer:
    """Push-pull gossip for neighbour discovery and demand hints, with real byte accounting."""

    def __init__(self, n_nodes: int, R_comm: float, fanout: int = 3, ttl: int = 4,
                 max_nbr_ids: int = 12, max_view: Optional[int] = 12):
        self.n = n_nodes
        self.R_comm = R_comm
        self.fanout = fanout
        self.ttl = ttl
        self.max_nbr_ids = max_nbr_ids      # neighbour ids carried on the wire (uint8 count)
        self.max_view = max_view            # beacon-side bound on the local view; None = unbounded
        self.views = [LocalView() for _ in range(n_nodes)]
        self.seq = np.zeros(n_nodes, dtype=np.int64)
        self.bytes_total = 0
        self.msgs_total = 0
        self.per_round: List[Tuple[int, int]] = []      # (messages, bytes) per round

    # ---- wire format ------------------------------------------------------
    def encode(self, i: int, pos_i, battery_i: float, hint_i: float,
               nbr_ids: List[int]) -> bytes:
        ids = list(nbr_ids)[: self.max_nbr_ids]
        head = _HDR.pack(int(i) & 0xFFFF, int(self.seq[i]) & 0xFFFF,
                         float(pos_i[0]), float(pos_i[1]),
                         float(battery_i), float(hint_i), len(ids))
        return head + b"".join(struct.pack("<H", int(j) & 0xFFFF) for j in ids)

    @staticmethod
    def decode(buf: bytes):
        sid, seq, x, y, bat, hint, k = _HDR.unpack_from(buf, 0)
        off = _HDR.size
        ids = [struct.unpack_from("<H", buf, off + 2 * t)[0] for t in range(k)]
        return sid, seq, np.array([x, y]), bat, hint, ids

    # ---- protocol ---------------------------------------------------------
    def neighbours(self, pos: np.ndarray, alive: np.ndarray) -> List[np.ndarray]:
        idx = np.flatnonzero(alive)
        out = [np.empty(0, dtype=int) for _ in range(self.n)]
        if idx.size == 0:
            return out
        P = pos[idx]
        d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        within = d2 <= self.R_comm ** 2
        np.fill_diagonal(within, False)
        for k, i in enumerate(idx):
            out[i] = idx[within[k]]
        return out

    def round(self, rnd: int, pos: np.ndarray, alive: np.ndarray, battery: np.ndarray,
              demand_hint: np.ndarray, rng: np.random.Generator,
              pos_report: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """One push-pull gossip round. Returns each node's current neighbour array.

        `pos` is physical truth and decides who can hear whom. `pos_report` is what each node
        *claims* on the wire, which a Byzantine node falsifies. Keeping the two separate is
        what makes the Byzantine scenario meaningful: a liar is still where it really is.
        """
        if pos_report is None:
            pos_report = pos
        nbrs = self.neighbours(pos, alive)
        msgs = 0
        nbytes = 0
        self.seq[alive] += 1

        for i in np.flatnonzero(alive):
            cand = nbrs[i]
            if cand.size == 0:
                continue
            k = min(self.fanout, cand.size)
            peers = rng.choice(cand, size=k, replace=False)
            push = self.encode(i, pos_report[i], battery[i], demand_hint[i],
                               [int(j) for j in cand])
            for j in peers:
                j = int(j)
                # push i -> j
                msgs += 1
                nbytes += len(push)
                self._absorb(j, push, rnd)
                # pull j -> i
                pull = self.encode(j, pos_report[j], battery[j], demand_hint[j],
                                   [int(q) for q in nbrs[j]])
                msgs += 1
                nbytes += len(pull)
                self._absorb(i, pull, rnd)

        # Direct neighbour beacons keep the 1-hop view fresh even at fanout 0.
        #
        # Capped to the `max_nbr_ids` NEAREST physical neighbours, matching the cap already
        # applied to what goes out on the wire. Without this cap, a node's `view.pos` grows
        # to its full physical degree - fine at the densities the main grid tests, but at
        # n=800 in the same small MEL footprint mean degree reaches ~445 nodes, and the
        # local connectivity marginal in `local_utility` does an O(d^2) pure-Python
        # neighbour-pair pass per candidate move. That measured 0.3s/node at n=800
        # (~85s/round with ~280 active nodes), making the scalability tier's 800-node point
        # impractical - not because the underlying idea is expensive, but because an
        # uncapped local view let "local" silently mean "most of the network." A real
        # bounded-memory device could not keep an unlimited neighbour table either, so this
        # is a realism fix as much as a performance one.
        for i in np.flatnonzero(alive):
            cand = nbrs[i]
            if self.max_view is not None and cand.size > self.max_view:
                d2 = ((pos[cand] - pos[i][None, :]) ** 2).sum(1)
                cand = cand[np.argsort(d2)[: self.max_view]]
            v = self.views[i]
            for j in cand:
                j = int(j)
                v.pos[j] = pos_report[j].copy()
                v.last_seen[j] = rnd

        for i in range(self.n):
            self.views[i].forget_older_than(rnd, self.ttl)

        self.msgs_total += msgs
        self.bytes_total += nbytes
        self.per_round.append((msgs, nbytes))
        return nbrs

    def _absorb(self, receiver: int, buf: bytes, rnd: int) -> None:
        sid, _seq, p, bat, hint, ids = self.decode(buf)
        v = self.views[receiver]
        if sid == receiver:
            return
        v.pos[sid] = p
        v.battery[sid] = bat
        v.demand_hint[sid] = hint
        v.nbrs_of[sid] = ids
        v.last_seen[sid] = rnd

    def view_sizes(self) -> np.ndarray:
        """Number of other nodes each node currently knows about (its local view)."""
        return np.array([len(v.pos) for v in self.views], dtype=np.int64)

    def stats(self) -> dict:
        rounds = max(len(self.per_round), 1)
        m = np.array([a for a, _ in self.per_round], dtype=float)
        b = np.array([b for _, b in self.per_round], dtype=float)
        return {
            "gossip_msgs_total": int(self.msgs_total),
            "gossip_bytes_total": int(self.bytes_total),
            "gossip_msgs_per_round": float(m.mean()) if m.size else 0.0,
            "gossip_bytes_per_round": float(b.mean()) if b.size else 0.0,
            "gossip_bytes_per_node_round": float(b.mean() / self.n) if b.size else 0.0,
            "gossip_rounds": rounds,
        }


# %% [markdown]
# ## 9 - Local (neighbour-only) utility
#
# The theory in section 7 uses the **global** marginal contribution. A decentralized node
# cannot evaluate that - it only has what gossip gave it. `local_utility` is the estimator a
# real node computes:
#
# * **Coverage** - only devices the node can sense (within `R_cov + max_step`), and only the
#   weight it would add *on top of what its known neighbours already cover*. That subtraction
#   is the local image of the wonderful-life utility.
# * **Connectivity** - the marginal change in the largest component of the **local induced
#   graph**, built from 1-hop positions and 2-hop neighbour lists carried by gossip. This is
#   what lets a node notice it is an articulation point.
# * **Energy** - exact, since a node knows its own displacement and load.
#
# The difference between this estimator and the global utility is the **locality gap**, and
# it is measured rather than assumed - `analyze_locality_gap` reports it.

# %%
def _largest_component_size(adj: Dict[int, set], nodes: List[int]) -> int:
    seen, best = set(), 0
    for s in nodes:
        if s in seen:
            continue
        stack, comp = [s], 0
        seen.add(s)
        while stack:
            u = stack.pop()
            comp += 1
            for v in adj.get(u, ()):  # noqa: E1133
                if v not in seen and v in adj:
                    seen.add(v)
                    stack.append(v)
        best = max(best, comp)
    return best


def local_utility(env: Environment, i: int, cands: np.ndarray, pos: np.ndarray,
                  prev: np.ndarray, view: LocalView, w: np.ndarray,
                  dev_idx: Optional[np.ndarray] = None) -> np.ndarray:
    """Utility of each candidate position for node `i`, from local information only."""
    tb = env.testbed
    R2 = env.R_cov ** 2

    # --- devices this node can sense -------------------------------------
    if dev_idx is None:
        d2c = ((tb.device_xy - pos[i][None, :]) ** 2).sum(1)
        dev_idx = np.flatnonzero(d2c <= (env.R_cov + env.max_step) ** 2)
    if dev_idx.size == 0:
        cov_gain = np.zeros(len(cands))
    else:
        D = tb.device_xy[dev_idx]
        wl = w[dev_idx]
        nbr_ids = [j for j in view.pos if j != i]
        if nbr_ids:
            NP = np.stack([view.pos[j] for j in nbr_ids], axis=0)
            covered_by_nbr = (((NP[:, None, :] - D[None, :, :]) ** 2).sum(-1) <= R2).any(0)
        else:
            covered_by_nbr = np.zeros(len(D), dtype=bool)
        free_w = np.where(covered_by_nbr, 0.0, wl)
        inr = ((cands[:, None, :] - D[None, :, :]) ** 2).sum(-1) <= R2
        cov_gain = inr @ free_w

    # --- local connectivity marginal --------------------------------------
    nbr_ids = sorted(j for j in view.pos if j != i)
    if not nbr_ids:
        conn_gain = np.zeros(len(cands))
        conn_gain[:] = 1.0                    # a lone node is its own component
    else:
        base_adj: Dict[int, set] = {j: set() for j in nbr_ids}
        NP = np.stack([view.pos[j] for j in nbr_ids], axis=0)
        dj2 = ((NP[:, None, :] - NP[None, :, :]) ** 2).sum(-1)
        Rc2 = env.R_comm ** 2
        for a, ja in enumerate(nbr_ids):
            for b, jb in enumerate(nbr_ids):
                if a != b and dj2[a, b] <= Rc2:
                    base_adj[ja].add(jb)
        for j in nbr_ids:                      # fold in 2-hop lists from gossip
            for q in view.nbrs_of.get(j, ()):  # noqa: E1133
                if q in base_adj:
                    base_adj[j].add(q)
                    base_adj[q].add(j)
        without_i = _largest_component_size(base_adj, nbr_ids)
        dist2 = ((cands[:, None, :] - NP[None, :, :]) ** 2).sum(-1)
        conn_gain = np.empty(len(cands))
        for c in range(len(cands)):
            adj = {j: set(base_adj[j]) for j in nbr_ids}
            adj[i] = set()
            for b, jb in enumerate(nbr_ids):
                if dist2[c, b] <= Rc2:
                    adj[i].add(jb)
                    adj[jb].add(i)
            conn_gain[c] = _largest_component_size(adj, nbr_ids + [i]) - without_i

    # --- own energy --------------------------------------------------------
    # flight energy of the move this node is deciding now: from its CURRENT position
    # (v1.0 used prev[i], the position one slot earlier)
    dist = np.linalg.norm(cands - pos[i][None, :], axis=1)
    e_fly = env.uav.flight_energy(dist, env.dt_s)
    e_cpu = env.uav.compute_energy(
        np.clip(cov_gain / max(env.W_total / max(env.n_nodes, 1), 1e-9), 0, 1), env.dt_s)
    energy = e_fly + e_cpu

    N = float(env.n_nodes)
    return (env.weights.coverage * cov_gain / max(env.W_total, 1e-9)
            + env.weights.connectivity * conn_gain / N
            - env.weights.energy * energy / (N * env.E_ref))
# %% [markdown]
# ## 10 - Algorithms (R1 and R4)
#
# Every controller exposes the same interface, so the runner, the scenarios and the
# measurement harness are shared and the comparison is apples to apples.
#
# **FogField (the contribution)**
# * `FogFieldBR` - best response with inertia. Inertia is what makes asynchronous
#   best-response converge rather than cycle; with an exact potential the finite improvement
#   property then gives convergence to a pure Nash equilibrium (T1).
# * `FogFieldLLL` - log-linear learning with annealed `beta`. The induced Markov chain is
#   reversible with stationary distribution proportional to `exp(beta * Phi)`, so as `beta`
#   grows the process concentrates on **potential maximisers**, i.e. efficient equilibria
#   (T2). This is the variant that escapes the poor local optima best response can lock into.
#
# **Decentralized baselines**
# * `GossipGreedy` - move toward the heaviest locally-known uncovered demand, with a
#   separation term. No game structure, no equilibrium argument.
# * `GuerreroStyle` - a node-repositioning adaptation of Guerrero et al.'s decentralized
#   popularity-pull placement policy, with a migration-cost threshold that damps oscillation.
#   Adapted, not reproduced: the original places services on fixed nodes.
# * `ConsensusADMM` - inexact consensus ADMM (Chang, Hong and Wang, IEEE TSP 2015) on a
#   smoothed continuous relaxation, projected back onto the lattice.
# * `MARLIndependentQ` - independent Q-learning with parameter sharing over a coarse local
#   state. Labelled as an adapted MARL baseline; it is **not** MAPPO.
#
# **Centralized references**
# * `DPOAStyleSwarm` - a D-POA-*style* swarm optimiser over the joint position vector with
#   dynamic re-initialisation on events. A reimplementation from the method description.
# * `StaticDeployment` / `RandomWalk` - lower references.

# %%
@dataclass
class StepContext:
    rnd: int
    pos: np.ndarray
    prev: np.ndarray
    alive: np.ndarray
    w: np.ndarray
    nbrs: List[np.ndarray]
    gossip: GossipLayer
    rng: np.random.Generator
    byzantine: np.ndarray


class Controller:
    name = "base"
    decentralized = True
    uses_gossip = True

    def reset(self, env: Environment, pos: np.ndarray, rng: np.random.Generator) -> None:
        """Controllers deliberately keep no reference to `env`.

        Checkpoints pickle the controller, and `env` holds the whole device field; storing it
        would multiply every checkpoint by the dataset size. `env` is passed to `step`.
        """
        return None

    def step(self, env: Environment, ctx: StepContext) -> np.ndarray:
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, d: dict) -> None:
        pass


def _apply_move(env: Environment, pos: np.ndarray, target: np.ndarray,
                alive: np.ndarray) -> np.ndarray:
    """Clip displacement to the one-step reach, snap to the lattice, freeze dead nodes."""
    out = pos.copy()
    idx = np.flatnonzero(alive)
    if idx.size == 0:
        return out
    delta = target[idx] - pos[idx]
    dist = np.linalg.norm(delta, axis=1, keepdims=True)
    scale = np.minimum(1.0, env.max_step / np.maximum(dist, 1e-9))
    out[idx] = env.snap(pos[idx] + delta * scale)
    return out


def _sensed_devices(env: Environment, xy: np.ndarray) -> np.ndarray:
    d2 = ((env.testbed.device_xy - xy[None, :]) ** 2).sum(1)
    return np.flatnonzero(d2 <= (env.R_cov + env.max_step) ** 2)


# %%
class FogFieldBR(Controller):
    """Asynchronous best response with inertia over the discretised action sets."""
    name = "FogField-BR"

    def __init__(self, p_activate: float = 0.35, inertia: float = 0.15,
                 max_actions: int = 21):
        self.p_activate = p_activate
        self.inertia = inertia
        self.max_actions = max_actions
        self.t_decide = 0.0          # seconds spent evaluating local utilities
        self.n_decide = 0            # node-decisions made (for the view-bound measurement)

    def step(self, env, ctx):
        target = ctx.pos.copy()
        for i in np.flatnonzero(ctx.alive):
            if ctx.rng.random() > self.p_activate:
                continue
            if ctx.rng.random() < self.inertia:
                continue
            cands = env.action_set(ctx.pos[i], self.max_actions)
            t0 = time.perf_counter()
            u = local_utility(env, i, cands, ctx.pos, ctx.prev,
                              ctx.gossip.views[i], ctx.w,
                              _sensed_devices(env, ctx.pos[i]))
            self.t_decide += time.perf_counter() - t0
            self.n_decide += 1
            best = int(np.argmax(u))
            here = np.flatnonzero(np.all(np.abs(cands - ctx.pos[i][None, :]) < 1e-6, axis=1))
            if here.size and u[here[0]] >= u[best] - 1e-12:
                best = int(here[0])          # ties go to staying, so rest points are absorbing
            target[i] = cands[best]
        return _apply_move(env, ctx.pos, target, ctx.alive)


class FogFieldLLL(Controller):
    """Log-linear learning; beta is annealed upward so the chain concentrates on maximisers."""
    name = "FogField-LLL"

    def __init__(self, p_activate: float = 0.35, beta0: float = 8.0, beta_growth: float = 0.45,
                 beta_max: float = 400.0, max_actions: int = 21):
        self.p_activate = p_activate
        self.beta0 = beta0
        self.beta_growth = beta_growth
        self.beta_max = beta_max
        self.max_actions = max_actions

    def beta(self, rnd: int) -> float:
        """Annealing schedule.

        Utility differences between neighbouring lattice cells are order 0.01-0.1, so beta
        has to reach the low hundreds before the softmax actually exploits. Starting at 8
        keeps the early rounds close to uniform exploration (beta * delta_u well under 1),
        and the growth rate brings beta into the exploiting regime within a few dozen
        rounds, which is the length of a run here.
        """
        return float(min(self.beta_max, self.beta0 * (1.0 + self.beta_growth * rnd)))

    def step(self, env, ctx):
        b = self.beta(ctx.rnd)
        target = ctx.pos.copy()
        for i in np.flatnonzero(ctx.alive):
            if ctx.rng.random() > self.p_activate:
                continue
            cands = env.action_set(ctx.pos[i], self.max_actions)
            u = local_utility(env, i, cands, ctx.pos, ctx.prev,
                              ctx.gossip.views[i], ctx.w,
                              _sensed_devices(env, ctx.pos[i]))
            z = b * (u - u.max())
            p = np.exp(z)
            s = p.sum()
            if not np.isfinite(s) or s <= 0:
                target[i] = cands[int(np.argmax(u))]
            else:
                target[i] = cands[int(ctx.rng.choice(len(cands), p=p / s))]
        return _apply_move(env, ctx.pos, target, ctx.alive)


# %%
class GossipGreedy(Controller):
    """Move toward the heaviest locally-known uncovered demand, repelled by close neighbours."""
    name = "GossipGreedy"

    def __init__(self, pull: float = 0.6, repel: float = 0.35):
        self.pull = pull
        self.repel = repel

    def step(self, env, ctx):
        target = ctx.pos.copy()
        R2 = env.R_cov ** 2
        for i in np.flatnonzero(ctx.alive):
            dev = _sensed_devices(env, ctx.pos[i])
            v = ctx.gossip.views[i]
            nbr_ids = [j for j in v.pos if j != i]
            vec = np.zeros(2)
            if dev.size:
                D = env.testbed.device_xy[dev]
                wl = ctx.w[dev].copy()
                if nbr_ids:
                    NP = np.stack([v.pos[j] for j in nbr_ids], axis=0)
                    taken = (((NP[:, None, :] - D[None, :, :]) ** 2).sum(-1) <= R2).any(0)
                    wl = np.where(taken, 0.0, wl)
                tot = wl.sum()
                if tot > 0:
                    centroid = (D * wl[:, None]).sum(0) / tot
                    vec += self.pull * (centroid - ctx.pos[i])
            if nbr_ids:
                NP = np.stack([v.pos[j] for j in nbr_ids], axis=0)
                d = NP - ctx.pos[i][None, :]
                r = np.linalg.norm(d, axis=1, keepdims=True)
                close = (r[:, 0] < env.R_cov) & (r[:, 0] > 1e-6)
                if close.any():
                    vec -= self.repel * (d[close] / r[close] ** 2 * env.R_cov).mean(0)
            target[i] = ctx.pos[i] + vec
        return _apply_move(env, ctx.pos, target, ctx.alive)


class GuerreroStyle(Controller):
    """Guerrero-style decentralized popularity pull, adapted from services to node movement."""
    name = "Guerrero-style"

    def __init__(self, alpha: float = 0.45, migrate_threshold: float = 0.02):
        self.alpha = alpha
        self.migrate_threshold = migrate_threshold

    def step(self, env, ctx):
        target = ctx.pos.copy()
        for i in np.flatnonzero(ctx.alive):
            dev = _sensed_devices(env, ctx.pos[i])
            if dev.size == 0:
                continue
            D = env.testbed.device_xy[dev]
            wl = ctx.w[dev]
            inr = ((D - ctx.pos[i][None, :]) ** 2).sum(1) <= env.R_cov ** 2
            if not inr.any() or wl[inr].sum() <= 0:
                served_centroid = D[np.argmax(wl)] if wl.size else ctx.pos[i]
            else:
                served_centroid = (D[inr] * wl[inr][:, None]).sum(0) / wl[inr].sum()
            move = self.alpha * (served_centroid - ctx.pos[i])
            # migration threshold: only move if the expected gain clears the cost
            cands = np.stack([ctx.pos[i], ctx.pos[i] + move], axis=0)
            u = local_utility(env, i, cands, ctx.pos, ctx.prev, ctx.gossip.views[i],
                              ctx.w, dev)
            if (u[1] - u[0]) > self.migrate_threshold:
                target[i] = ctx.pos[i] + move
        return _apply_move(env, ctx.pos, target, ctx.alive)


# %%
class ConsensusADMM(Controller):
    """Inexact consensus ADMM (Chang-Hong-Wang, IEEE TSP 2015) on a smoothed relaxation."""
    name = "Consensus-ADMM"

    def __init__(self, inner: int = 6, rho: float = 0.15, alpha: float = 0.30,
                 smooth: float = 90.0, lam_max: float = 500.0):
        self.inner = inner
        self.rho = rho              # consensus / dual weight (dimensionless)
        self.alpha = alpha          # primal step as a fraction of one-step reach
        self.smooth = smooth        # sigmoid width of the coverage relaxation (m)
        self.lam_max = lam_max      # dual-variable clip (metres); see step() for why
        self.lam: Dict[Tuple[int, int], np.ndarray] = {}

    def reset(self, env, pos, rng):
        super().reset(env, pos, rng)
        self.lam = {}

    def _cov_direction(self, env, i, x, ctx, dev) -> np.ndarray:
        """Ascent direction of the smoothed local coverage, as a unit vector.

        Coverage is relaxed to sum_m w_m * sigmoid((R_cov - ||x - d_m||) / smooth), whose
        gradient is -(1/smooth) * sum_m w_m sigma'(z_m) * (x - d_m)/||x - d_m||. The result
        is normalised so that the primal step is expressed in metres, matching the units of
        the consensus term; mixing a raw gradient with a metre-scale dual makes the step size
        meaningless.
        """
        g = np.zeros(2)
        if dev.size:
            D = env.testbed.device_xy[dev]
            wl = ctx.w[dev]
            d = x[None, :] - D
            r = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            z = (env.R_cov - r) / self.smooth
            sig = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
            coef = -(wl * sig * (1.0 - sig)) / self.smooth
            g = (coef[:, None] * (d / r[:, None])).sum(axis=0)
        # movement penalty pulls back toward where the node already is
        g -= (env.weights.energy / max(env.weights.coverage, 1e-9)) * \
            (x - ctx.pos[i]) / max(env.max_step, 1e-9)
        n = float(np.linalg.norm(g))
        return g / n if n > 1e-12 else np.zeros(2)

    def _consensus_peers(self, i: int, ctx) -> List[int]:
        """Bounded neighbour set for the consensus term.

        `ctx.nbrs[i]` is the raw PHYSICAL neighbour list - at high density (n=800 in the
        same small MEL footprint used for n=50, mean degree reaches several hundred) that
        makes the per-node dual-variable sum grow just as unboundedly as the pre-fix gossip
        view did. Using the same capped set every other controller reasons over
        (`ctx.gossip.views[i].pos`) keeps ADMM's consensus problem the size an ADMM paper
        actually analyses - a bounded neighbourhood - rather than "most of the network."
        """
        return [j for j in ctx.gossip.views[i].pos if j != i]

    def step(self, env, ctx):
        x = ctx.pos.astype(float).copy()
        alive = [int(i) for i in np.flatnonzero(ctx.alive)]
        dev_cache = {i: _sensed_devices(env, ctx.pos[i]) for i in alive}
        peers = {i: self._consensus_peers(i, ctx) for i in alive}
        step_m = self.alpha * env.max_step / max(self.inner, 1)

        for _ in range(self.inner):
            newx = x.copy()
            for i in alive:
                direction = self._cov_direction(env, i, x[i], ctx, dev_cache[i])
                pull = np.zeros(2)
                for j in peers[i]:
                    key = (min(i, j), max(i, j))
                    lam = self.lam.setdefault(key, np.zeros(2))
                    sgn = 1.0 if i < j else -1.0
                    mid = 0.5 * (x[i] + x[j])
                    pull += sgn * lam + (x[i] - mid)
                newx[i] = x[i] + step_m * direction - self.rho * pull
            x = newx
            for i in alive:                      # dual ascent on the consensus residuals
                for j in peers[i]:
                    if i < j:
                        mid = 0.5 * (x[i] + x[j])
                        key = (i, j)
                        # Clip the dual variable to a bounded ball. Plain dual ascent
                        # integrates the primal residual (x[i]-mid) every inner iteration
                        # with no leak; because positions are snapped to a lattice, that
                        # residual never fully vanishes, so unclipped lam undergoes
                        # integrator windup - observed in practice as an overflow to inf/nan
                        # after enough rounds at large, dense node counts. Clipping is the
                        # standard fix (a bounded-dual / proximal variant of consensus ADMM)
                        # and changes nothing while the residual is small.
                        upd = self.lam.get(key, np.zeros(2)) + self.rho * (x[i] - mid)
                        norm = float(np.linalg.norm(upd))
                        if norm > self.lam_max:
                            upd = upd * (self.lam_max / norm)
                        self.lam[key] = upd
            if not np.all(np.isfinite(x)):
                # a numerical excursion slipped through anyway - discard this inner
                # iteration's update rather than let non-finite values reach _apply_move,
                # which would corrupt lattice-index casts downstream.
                x = np.where(np.isfinite(x), x, ctx.pos.astype(float))
                break
        return _apply_move(env, ctx.pos, x, ctx.alive)

    def state_dict(self):
        return {"lam": {"{}_{}".format(*k): v.tolist() for k, v in self.lam.items()}}

    def load_state_dict(self, d):
        self.lam = {tuple(int(t) for t in k.split("_")): np.asarray(v)
                    for k, v in d.get("lam", {}).items()}


# %%
class MARLIndependentQ(Controller):
    """Independent Q-learning with parameter sharing. An adapted MARL baseline, not MAPPO."""
    name = "MARL-IQL"

    N_ACT = 9

    def __init__(self, lr: float = 0.25, gamma: float = 0.85, eps0: float = 0.5,
                 eps_decay: float = 0.985):
        self.lr = lr
        self.gamma = gamma
        self.eps0 = eps0
        self.eps_decay = eps_decay
        self.Q = np.zeros((3 * 3 * 2, self.N_ACT))
        self.eps = eps0
        self._last: Dict[int, Tuple[int, int]] = {}

    def reset(self, env, pos, rng):
        super().reset(env, pos, rng)
        self.eps = self.eps0
        self._last = {}

    def _dirs(self, env):
        s = env.max_step
        base = [(0, 0)]
        for k in range(8):
            a = k * math.pi / 4
            base.append((s * math.cos(a), s * math.sin(a)))
        return np.array(base)

    def _state(self, env, i, ctx, dev):
        v = ctx.gossip.views[i]
        deg = len([j for j in v.pos if j != i])
        dbin = 0 if deg == 0 else (1 if deg <= 3 else 2)
        if dev.size:
            wl = ctx.w[dev].sum()
            ref = max(env.W_total / max(env.n_nodes, 1), 1e-9)
            ubin = 0 if wl < 0.5 * ref else (1 if wl < 1.5 * ref else 2)
        else:
            ubin = 0
        bbin = 0
        return (dbin * 3 + ubin) * 2 + bbin

    def step(self, env, ctx):
        dirs = self._dirs(env)
        target = ctx.pos.copy()
        for i in np.flatnonzero(ctx.alive):
            i = int(i)
            dev = _sensed_devices(env, ctx.pos[i])
            s = self._state(env, i, ctx, dev)
            if ctx.rng.random() < self.eps:
                a = int(ctx.rng.integers(self.N_ACT))
            else:
                a = int(np.argmax(self.Q[s]))
            cand = ctx.pos[i] + dirs[a]
            r = float(local_utility(env, i, cand[None, :], ctx.pos, ctx.prev,
                                    ctx.gossip.views[i], ctx.w, dev)[0])
            if i in self._last:
                ps, pa = self._last[i]
                self.Q[ps, pa] += self.lr * (r + self.gamma * self.Q[s].max() - self.Q[ps, pa])
            self._last[i] = (s, a)
            target[i] = cand
        self.eps = max(0.03, self.eps * self.eps_decay)
        return _apply_move(env, ctx.pos, target, ctx.alive)

    def state_dict(self):
        return {"Q": self.Q.copy(), "eps": self.eps, "last": dict(self._last)}

    def load_state_dict(self, d):
        self.Q = np.asarray(d["Q"])
        self.eps = float(d["eps"])
        self._last = dict(d.get("last", {}))


# %%
class DPOAStyleSwarm(Controller):
    """Centralized D-POA-style swarm over the joint position vector. A reimplementation."""
    name = "D-POA-style (central)"
    decentralized = False
    uses_gossip = False

    def __init__(self, n_particles: int = 12, inner: int = 4, w_inertia: float = 0.65,
                 c1: float = 1.2, c2: float = 1.4):
        self.n_particles = n_particles
        self.inner = inner
        self.w_inertia = w_inertia
        self.c1 = c1
        self.c2 = c2
        self.swarm = None
        self.vel = None
        self.pbest = None
        self.pbest_f = None

    def reset(self, env, pos, rng):
        super().reset(env, pos, rng)
        self.swarm = None

    def _init_swarm(self, env, pos, rng):
        P = self.n_particles
        jitter = rng.normal(0, env.max_step * 0.5, size=(P,) + pos.shape)
        self.swarm = np.clip(pos[None, :, :] + jitter, -1e9, 1e9)
        self.swarm[0] = pos
        self.vel = np.zeros_like(self.swarm)
        self.pbest = self.swarm.copy()
        self.pbest_f = np.full(P, -np.inf)

    def step(self, env, ctx):
        # environmental change (v2): a new demand field re-initialises the swarm around the
        # current configuration, as D-POA does on detected change
        w_last = getattr(self, "_w_last", None)
        if w_last is not None and (w_last.shape != ctx.w.shape or not np.array_equal(w_last, ctx.w)):
            self.swarm = None
        self._w_last = np.array(ctx.w, copy=True)
        if self.swarm is None or self.swarm.shape[1] != ctx.pos.shape[0]:
            self._init_swarm(env, ctx.pos, ctx.rng)
        elif self.pbest_f is not None and np.isfinite(self.pbest_f).any():
            # re-evaluate the memory under this slot's demand and origin (v2): without it the
            # swarm chases optima remembered from earlier demand slots
            for p in range(self.n_particles):
                cand = _apply_move(env, ctx.pos, self.pbest[p], ctx.alive)
                self.pbest_f[p] = env.phi(cand, ctx.pos, ctx.alive, ctx.w)
        rng = ctx.rng
        for _ in range(self.inner):
            fits = np.empty(self.n_particles)
            for p in range(self.n_particles):
                cand = _apply_move(env, ctx.pos, self.swarm[p], ctx.alive)
                fits[p] = env.phi(cand, ctx.pos, ctx.alive, ctx.w)
            better = fits > self.pbest_f
            self.pbest[better] = self.swarm[better]
            self.pbest_f[better] = fits[better]
            gbest = self.pbest[int(np.argmax(self.pbest_f))]
            r1 = rng.random(self.swarm.shape)
            r2 = rng.random(self.swarm.shape)
            self.vel = (self.w_inertia * self.vel
                        + self.c1 * r1 * (self.pbest - self.swarm)
                        + self.c2 * r2 * (gbest[None] - self.swarm))
            self.vel = np.clip(self.vel, -env.max_step, env.max_step)
            self.swarm = self.swarm + self.vel
        gbest = self.pbest[int(np.argmax(self.pbest_f))]
        return _apply_move(env, ctx.pos, gbest, ctx.alive)

    def on_event(self):
        """Dynamic re-initialisation, as D-POA does on topology change (and, in v2, on a
        change of the demand field, see step)."""
        self.swarm = None
        self.pbest_f = None


class StaticDeployment(Controller):
    name = "Static"
    uses_gossip = False

    def step(self, env, ctx):
        return ctx.pos.copy()


class RandomWalk(Controller):
    name = "RandomWalk"
    uses_gossip = False

    def step(self, env, ctx):
        ang = ctx.rng.uniform(0, 2 * math.pi, ctx.pos.shape[0])
        r = ctx.rng.uniform(0, env.max_step, ctx.pos.shape[0])
        target = ctx.pos + np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
        return _apply_move(env, ctx.pos, target, ctx.alive)


ALGORITHMS = {
    "FogField-BR": FogFieldBR,
    "FogField-LLL": FogFieldLLL,
    "GossipGreedy": GossipGreedy,
    "Guerrero-style": GuerreroStyle,
    "Consensus-ADMM": ConsensusADMM,
    "MARL-IQL": MARLIndependentQ,
    "D-POA-style": DPOAStyleSwarm,
    "Static": StaticDeployment,
    "RandomWalk": RandomWalk,
}
DECENTRALIZED = [k for k, v in ALGORITHMS.items() if v.decentralized]
print("algorithms:", ", ".join(ALGORITHMS))
print("decentralized:", ", ".join(DECENTRALIZED))
# %% [markdown]
# ## 11 - Dynamic scenarios (R6)
#
# The seven D-POA-style dynamic scenarios, plus the two the plan adds.
#
# | key | what happens |
# |---|---|
# | `static` | nothing changes - the clean convergence setting |
# | `mobility` | the demand field advances through its real time-of-day slots |
# | `random_failures` | each live node fails with small probability each round |
# | `traffic_shift` | the demand field jumps to a distant slot at fixed intervals |
# | `expansion` | new nodes join the network twice during the run |
# | `catastrophic` | 50% of nodes fail simultaneously at the midpoint |
# | `flash_crowd` | demand in a disc is multiplied for a window |
# | `seed_loss` | one node (the bootstrap/seed node) is removed at the midpoint |
# | `byzantine` | a fraction of nodes gossip falsified positions and demand hints |
# | `controller_loss` | the centralised controller dies at the midpoint; a demand surge follows |
# | `late_surge` | the same demand surge with nobody dying: the control twin of `controller_loss` |
#
# **On `controller_loss` (added in v2).** The single point of failure that motivates a
# decentralised scheme is the controller, so this scenario removes it: from the midpoint on,
# the D-POA-style controller issues no commands and its nodes hold their last commanded
# position. Decentralised methods have no controller and are untouched by the event itself.
# A localised demand surge starts ten rounds after the loss so that the post-loss period
# contains a demand change on both testbeds (MEL demand is otherwise static, and a frozen but
# converged deployment would lose nothing). While the controller is alive its uplink is
# accounted as a *modelled* cost under the same codec: one 21-byte state report per live
# node per slot in, one 10-byte position command per node out.
#
# **On `seed_loss`.** v1.0 presented this as the controller-loss proxy. It is not: it removes
# one node at the midpoint, nobody needs a bootstrap node after round 0, and the ledger shows
# it has no measurable effect on any method. It is kept, as a single-node-loss null scenario.
#
# **On `mobility` for MEL.** The EUA testbed is a single static snapshot, so on MEL mobility
# is emulated by a travelling demand hotspot and is **flagged `synthetic_dynamics=True`** in
# every result record. Real temporal dynamics come from BEJ, where the slots are GeoLife's
# own time-of-day bins. Do not report MEL mobility numbers as data-driven.

# %%
@dataclass
class Scenario:
    key: str
    description: str
    synthetic_dynamics: bool = False


SCENARIOS = {
    "static": Scenario("static", "no exogenous change"),
    "mobility": Scenario("mobility", "demand advances through real time-of-day slots"),
    "random_failures": Scenario("random_failures", "independent per-round node failures"),
    "traffic_shift": Scenario("traffic_shift", "abrupt jumps in the demand field"),
    "expansion": Scenario("expansion", "new nodes join twice"),
    "catastrophic": Scenario("catastrophic", "50% simultaneous node loss at the midpoint"),
    "flash_crowd": Scenario("flash_crowd", "localised demand surge for a window"),
    "seed_loss": Scenario("seed_loss", "one node (the bootstrap/seed node) removed at the midpoint"),
    "byzantine": Scenario("byzantine", "a fraction of nodes gossip falsified state"),
    "controller_loss": Scenario("controller_loss",
                                "centralised controller dies at the midpoint; demand surge follows"),
    "late_surge": Scenario("late_surge",
                           "demand surge ten rounds after the midpoint (control twin of controller_loss)"),
    "ctrl_fail": Scenario("ctrl_fail", "controller dies at the midpoint; node failures (1%/slot) follow"),
    "late_fail": Scenario("late_fail", "node failures (1%/slot) from the midpoint (twin of ctrl_fail)"),
    "ctrl_drift": Scenario("ctrl_drift", "controller dies at the midpoint; a demand hotspot sweeps the region"),
    "late_drift": Scenario("late_drift", "demand hotspot sweeps the region from the midpoint (twin of ctrl_drift)"),
}

P_FAIL = 0.010
FLASH_FACTOR = 6.0
BYZ_FRACTION = 0.15


class ScenarioEngine:
    """Turns a scenario key into per-round mutations of the simulation state.

    Two random streams, both seeded by (testbed, n, seed) only (common random numbers, v2):
    `rng` drives the scenario itself (flash-crowd centre, which nodes fail and when) and
    `rng_adv` drives the Byzantine adversary (which nodes lie and what they report). Every
    algorithm therefore faces the same events and the same attacker, and the byzantine and
    static runs of one algorithm differ only in the falsified messages.
    """

    def __init__(self, scenario: str, env: Environment, rounds: int,
                 rng: np.random.Generator, rng_adv: Optional[np.random.Generator] = None):
        self.key = scenario
        self.env = env
        self.rounds = rounds
        self.rng = rng
        self.rng_adv = rng_adv if rng_adv is not None else rng
        self.events: List[Tuple[int, str]] = []
        tb = env.testbed
        x0, x1, y0, y1 = tb.bounds
        self.flash_centre = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
        self.flash_centre += rng.normal(0, 0.18 * (x1 - x0), 2)
        if scenario in ("controller_loss", "late_surge"):
            # the surge starts ten rounds after the (possible) controller loss
            lo = rounds // 2 + 10
        else:
            lo = rounds // 3
        self.flash_window = (lo, lo + max(6, rounds // 6))
        self.hotspot_path = np.stack([
            np.linspace(x0, x1, max(rounds, 2)),
            y0 + (y1 - y0) * (0.5 + 0.35 * np.sin(np.linspace(0, 2 * np.pi, max(rounds, 2)))),
        ], axis=1)
        self.byz_mask = np.zeros(env.n_nodes, dtype=bool)
        if scenario == "byzantine":
            k = max(1, int(round(BYZ_FRACTION * env.n_nodes)))
            self.byz_mask[self.rng_adv.choice(env.n_nodes, k, replace=False)] = True

    def controller_dead(self, rnd: int) -> bool:
        return self.key in ("controller_loss", "ctrl_fail", "ctrl_drift") and rnd >= self.rounds // 2

    def drift_centre(self, rnd: int) -> np.ndarray:
        """Hotspot centre for the drift scenarios: two sweeps over the second half of the run."""
        x0, x1, y0, y1 = self.env.testbed.bounds
        u = 2.0 * (rnd - self.rounds // 2) / max(self.rounds - self.rounds // 2, 1)
        return np.array([x0 + (x1 - x0) * (u % 1.0),
                         y0 + (y1 - y0) * (0.5 + 0.35 * math.sin(2.0 * math.pi * u))])

    # ---- demand ------------------------------------------------------------
    def weights(self, rnd: int) -> np.ndarray:
        tb = self.env.testbed
        if self.key == "mobility":
            if tb.n_slots > 1:
                w = tb.weights(rnd)
            else:                      # synthetic travelling hotspot (MEL)
                c = self.hotspot_path[min(rnd, len(self.hotspot_path) - 1)]
                d2 = ((tb.device_xy - c[None, :]) ** 2).sum(1)
                w = tb.weights(0) * (1.0 + 3.0 * np.exp(-d2 / (2 * (0.25 * self.env.R_cov * 3) ** 2)))
        elif self.key == "traffic_shift":
            stride = max(1, tb.n_slots // 3)
            w = tb.weights((rnd // max(1, self.rounds // 4)) * stride)
        else:
            w = tb.weights(0 if tb.n_slots == 1 else rnd)

        if self.key in ("ctrl_drift", "late_drift") and rnd >= self.rounds // 2:
            c = self.drift_centre(rnd)
            d2 = ((tb.device_xy - c[None, :]) ** 2).sum(1)
            w = w * (1.0 + 3.0 * np.exp(-d2 / (2 * (0.75 * self.env.R_cov) ** 2)))

        if self.key in ("flash_crowd", "controller_loss", "late_surge"):
            lo, hi = self.flash_window
            if lo <= rnd < hi:
                d2 = ((tb.device_xy - self.flash_centre[None, :]) ** 2).sum(1)
                boost = 1.0 + (FLASH_FACTOR - 1.0) * np.exp(-d2 / (2 * self.env.R_cov ** 2))
                w = w * boost
        return np.ascontiguousarray(w, dtype=np.float64)

    # ---- topology ----------------------------------------------------------
    def apply_topology(self, rnd: int, alive: np.ndarray, joined: np.ndarray) -> str:
        rng = self.rng
        ev = ""
        live = np.flatnonzero(alive)
        if self.key == "random_failures" and live.size > 2:
            die = live[rng.random(live.size) < P_FAIL]
            if die.size:
                alive[die] = False
                ev = "fail:{}".format(die.size)
        elif self.key == "catastrophic" and rnd == self.rounds // 2 and live.size > 2:
            k = live.size // 2
            die = rng.choice(live, k, replace=False)
            alive[die] = False
            ev = "catastrophic:{}".format(k)
        elif self.key == "seed_loss" and rnd == self.rounds // 2:
            seed_node = int(live[0]) if live.size else -1
            if seed_node >= 0:
                alive[seed_node] = False
                ev = "seed_loss:{}".format(seed_node)
        elif self.key == "controller_loss" and rnd == self.rounds // 2:
            ev = "controller_loss"
        elif self.key in ("ctrl_fail", "late_fail") and rnd >= self.rounds // 2:
            draw = rng.random(live.size)              # drawn in both twins, so the streams match
            if live.size > 2:
                die = live[draw < P_FAIL]
                if die.size:
                    alive[die] = False
                    ev = "fail:{}".format(die.size)
            if self.key == "ctrl_fail" and rnd == self.rounds // 2:
                ev = "controller_loss" + ("+" + ev if ev else "")
        elif self.key in ("ctrl_drift",) and rnd == self.rounds // 2:
            ev = "controller_loss"
        elif self.key == "expansion":
            for frac in (0.33, 0.66):
                if rnd == int(self.rounds * frac):
                    pend = np.flatnonzero(~joined)
                    if pend.size:
                        k = min(pend.size, max(1, self.env.n_nodes // 5))
                        add = pend[:k]
                        joined[add] = True
                        alive[add] = True
                        ev = "join:{}".format(k)
        if ev:
            self.events.append((rnd, ev))
        return ev


# %% [markdown]
# ## 12 - The simulation loop, with resume
#
# One call to `simulate` is one **work unit**. It checkpoints every `ckpt_every` rounds,
# saving positions, batteries, the gossip layer with its byte counters, the controller's
# internal state, the metric trace and **the RNG states** (the algorithm stream explicitly,
# the scenario and adversary streams inside the pickled engine). If the session dies, the
# next call with the same parameters picks up from the last checkpoint rather than round zero.
#
# **Common random numbers (v2).** See `seed_streams`: deployment, events and adversary are
# functions of (testbed, n, seed) alone, so comparisons across algorithms and across
# scenarios are paired. The record also carries explicit recovery censoring, the service
# level before and after the midpoint event, the modelled controller uplink, and the mean
# local-view size.
#
# `SELFTEST_RESUME` below actually kills a run mid-way and restarts it, then checks that the
# resumed trace matches the uninterrupted one. Checkpointing that has never been tested
# against an actual interruption is not checkpointing.

# %%
def initial_positions(env: Environment, rng: np.random.Generator) -> np.ndarray:
    """Start from the real infrastructure sites, topped up with jittered copies if needed."""
    sites = env.testbed.init_sites_xy
    n = env.n_nodes
    if len(sites) >= n:
        idx = rng.choice(len(sites), n, replace=False)
        pos = sites[idx]
    else:
        reps = int(np.ceil(n / max(len(sites), 1)))
        pos = np.repeat(sites, reps, axis=0)[:n]
        pos = pos + rng.normal(0, env.R_cov * 0.35, pos.shape)
    return env.snap(pos)


CTRL_CMD_BYTES = struct.calcsize("<Hff")      # position command: node id + x + y


def simulate(env: Environment, algo: str, scenario: str, seed: int, rounds: int,
             ckpt_every: int = 25, resume: bool = True, stop_after: Optional[int] = None,
             progress: bool = False, return_state: bool = False,
             init_pos: Optional[np.ndarray] = None, ctrl_kwargs: Optional[dict] = None,
             view_cap: Optional[int] = 12) -> dict:
    """Run one (env, algorithm, scenario, seed) unit. Resumable.

    `init_pos` overrides the drawn deployment (the MILP study starts every heuristic from
    the configuration the MILP starts from). `ctrl_kwargs` go to the controller constructor
    (the log-linear schedule sweep). `view_cap` is the beacon-side bound on the local view
    (None = unbounded; used only by the view-bound measurement).
    """
    ctrl_kwargs = dict(ctrl_kwargs or {})
    ident = dict(testbed=env.testbed.name, algo=algo, scenario=scenario, seed=seed,
                 n_nodes=env.n_nodes, rounds=rounds, R_cov=env.R_cov, R_comm=env.R_comm,
                 cap=env.capacity, lattice=env.lattice_m, quick=QUICK, seeding="crn-v2",
                 init="common" if init_pos is not None else "sites",
                 ctrl=json.dumps(ctrl_kwargs, sort_keys=True) if ctrl_kwargs else "",
                 view_cap=view_cap)
    key = unit_key(**ident)
    ck = RunCheckpointer(key, ckpt_every, enabled=resume)
    centralized = not ALGORITHMS[algo].decentralized

    saved = ck.load()
    if saved is not None:
        rnd0 = saved["rnd"]
        pos = saved["pos"]
        prev = saved["prev"]
        alive = saved["alive"]
        joined = saved["joined"]
        battery = saved["battery"]
        gossip = saved["gossip"]
        ctrl = saved["ctrl"]
        engine = saved["engine"]
        trace = saved["trace"]
        rng = rng_from_state(saved["rng"])
        resumed_from = rnd0
        diverged_rounds = saved.get("diverged_rounds", 0)
        ctrl_bytes = saved.get("ctrl_bytes", 0)
        view_sum = saved.get("view_sum", 0.0)
        view_cnt = saved.get("view_cnt", 0)
    else:
        streams = seed_streams(env.testbed.name, env.n_nodes, seed, algo)
        rng = streams["algo"]
        ctrl = ALGORITHMS[algo](**ctrl_kwargs)
        engine = ScenarioEngine(scenario, env, rounds, streams["scenario"], streams["adversary"])
        if init_pos is not None:
            pos = env.snap(np.asarray(init_pos, dtype=np.float64))
        else:
            pos = initial_positions(env, streams["deploy"])
        prev = pos.copy()
        alive = np.ones(env.n_nodes, dtype=bool)
        joined = np.ones(env.n_nodes, dtype=bool)
        if scenario == "expansion":                 # hold a reserve back for the joins
            hold = max(1, int(0.3 * env.n_nodes))
            joined[-hold:] = False
            alive[-hold:] = False
        battery = np.full(env.n_nodes, env.uav.battery_Wh * 3600.0)
        gossip = GossipLayer(env.n_nodes, env.R_comm, max_view=view_cap)
        ctrl.reset(env, pos, rng)
        trace = []
        rnd0 = 0
        resumed_from = None
        diverged_rounds = 0
        ctrl_bytes = 0
        view_sum = 0.0
        view_cnt = 0

    t_start = time.time()
    last = min(rounds, rnd0 + stop_after) if stop_after else rounds
    it = range(rnd0, last)
    if progress:
        it = tqdm(it, desc="{}/{}/{}".format(env.testbed.name, algo, scenario), leave=False)

    def _state(rnd_next):
        return {"rnd": rnd_next, "pos": pos, "prev": prev, "alive": alive, "joined": joined,
                "battery": battery, "gossip": gossip, "ctrl": ctrl, "engine": engine,
                "trace": trace, "rng": rng_state_of(rng), "diverged_rounds": diverged_rounds,
                "ctrl_bytes": ctrl_bytes, "view_sum": view_sum, "view_cnt": view_cnt}

    for rnd in it:
        w = engine.weights(rnd)
        ev = engine.apply_topology(rnd, alive, joined)
        ctrl_alive = not (centralized and engine.controller_dead(rnd))
        if ev and ctrl_alive and hasattr(ctrl, "on_event"):
            ctrl.on_event()

        # gossip: byzantine nodes report falsified state on the wire (adversary stream)
        pos_report = pos.copy()
        hint = np.zeros(env.n_nodes)
        for i in np.flatnonzero(alive):
            dev = _sensed_devices(env, pos[i])
            hint[i] = float(w[dev].sum()) if dev.size else 0.0
        if engine.byz_mask.any():
            b = engine.byz_mask & alive
            x0, x1, y0, y1 = env.testbed.bounds
            adv = engine.rng_adv
            pos_report[b] = np.stack([adv.uniform(x0, x1, b.sum()),
                                      adv.uniform(y0, y1, b.sum())], axis=1)
            hint[b] = hint.max() * 3.0 if hint.max() > 0 else 1.0

        nbrs = gossip.round(rnd, pos, alive, battery, hint, rng, pos_report=pos_report)
        view_sum += float(gossip.view_sizes()[alive].sum())
        view_cnt += int(alive.sum())

        ctx = StepContext(rnd=rnd, pos=pos, prev=prev, alive=alive, w=w, nbrs=nbrs,
                          gossip=gossip, rng=rng, byzantine=engine.byz_mask)
        if not ctrl_alive:
            # the controller is gone: nodes hold their last commanded positions
            new_pos = pos.copy()
        else:
            # A controller's own numerical failure (e.g. an iterative baseline diverging at
            # a node count/density it wasn't tuned for) must degrade to "no move this round"
            # for the affected nodes, not crash a multi-hour queue over one flaky baseline
            # unit. Observed in practice: Consensus-ADMM overflowed to NaN at n=800 before
            # its dual variables were clipped (see ConsensusADMM.step) - this is the second
            # line of defence, for whatever the next such case turns out to be.
            try:
                new_pos = ctrl.step(env, ctx)
                if not np.all(np.isfinite(new_pos)):
                    raise FloatingPointError("controller produced non-finite positions")
            except Exception as exc:
                diverged_rounds += 1
                if diverged_rounds <= 3 or diverged_rounds % 50 == 0:
                    print("[diverge] {}/{}/{} round {}: {} ({}) - holding position this round"
                         .format(env.testbed.name, algo, scenario, rnd, type(exc).__name__, exc))
                new_pos = pos.copy()
            if centralized:
                # modelled controller uplink: one state report in, one command out, per node
                ctrl_bytes += int(alive.sum()) * (_HDR.size + CTRL_CMD_BYTES)

        # energy accounting and battery depletion
        dist = np.linalg.norm(new_pos - pos, axis=1)
        e_fly = env.uav.flight_energy(dist, env.dt_s)
        load = env.load_fraction(new_pos, alive, w)
        e_cpu = env.uav.compute_energy(load, env.dt_s)
        spent = np.where(alive, e_fly + e_cpu, 0.0)
        battery = np.maximum(battery - spent, 0.0)
        depleted = alive & (battery <= 0.0)
        if depleted.any():
            alive = alive & ~depleted

        prev = pos
        pos = new_pos

        val, parts = env.phi(pos, prev, alive, w, parts=True)
        trace.append({
            "round": rnd, "phi": float(val), "alive": int(alive.sum()),
            "event": ev, "energy_step_J": float(spent.sum()), **parts,
        })

        ck.maybe_save(rnd + 1, _state(rnd + 1))

    if stop_after and last < rounds:
        ck.maybe_save(last, _state(last), force=True)
        return {"_partial": True, "rounds_done": last, "key": key}

    df = pd.DataFrame(trace)
    tail = max(1, len(df) // 5)
    final = df.tail(tail)
    phi_arr = df["phi"].to_numpy()
    cov_arr = df["coverage_frac"].to_numpy()
    mid = rounds // 2
    tail_mid = max(1, (len(phi_arr) - mid) // 3)
    rec_ev = recovery_events(df, engine.events)
    n_dec = int(getattr(ctrl, "n_decide", 0))
    rec = dict(ident)
    rec.update({
        "key": key,
        "phi_final": float(final["phi"].mean()),
        "phi_best": float(df["phi"].max()),
        "coverage_frac": float(final["coverage_frac"].mean()),
        "giant_frac": float(final["giant_frac"].mean()),
        "energy_total_J": float(df["energy_step_J"].sum()),
        "energy_per_node_J": float(df["energy_step_J"].sum() / max(env.n_nodes, 1)),
        "alive_final": int(df["alive"].iloc[-1]),
        "convergence_round": convergence_round(df["phi"].to_numpy()),
        "recovery_rounds": recovery_time(df, engine.events),
        "recovery_censored": (float(np.mean([not e["recovered"] for e in rec_ev]))
                              if rec_ev else None),
        "recovery_events": rec_ev,
        "phi_pre_mid": float(np.mean(phi_arr[max(0, mid - 5):mid])) if mid > 0 else None,
        "phi_post_mid": float(np.mean(phi_arr[mid:])),
        "phi_post_tail": float(np.mean(phi_arr[-tail_mid:])),
        "cov_post_tail": float(np.mean(cov_arr[-tail_mid:])),
        "ctrl_bytes_total": int(ctrl_bytes),
        "ctrl_bytes_per_round": float(ctrl_bytes / max(len(df), 1)),
        "ctrl_bytes_per_node_round": float(ctrl_bytes / max(len(df), 1) / max(env.n_nodes, 1)),
        "view_size_mean": float(view_sum / max(view_cnt, 1)),
        "decide_s_per_node": (float(ctrl.t_decide / n_dec) if n_dec else None),
        "ctrl_kwargs": ctrl_kwargs,
        "wallclock_s": float(time.time() - t_start),
        "diverged_rounds": int(diverged_rounds),
        "decentralized": bool(ALGORITHMS[algo].decentralized),
        "synthetic_dynamics": bool(scenario == "mobility" and env.testbed.n_slots == 1),
        "events": ["{}@{}".format(e, r) for r, e in engine.events],
        "trace_phi": [round(float(v), 6) for v in df["phi"].tolist()],
        "trace_cov": [round(float(v), 5) for v in df["coverage_frac"].tolist()],
        "energy_node_max_J": float(np.max(env.uav.battery_Wh * 3600.0 - battery)),
        "final_pos": pos.tolist(),
        "final_prev": prev.tolist(),
        "final_alive": alive.tolist(),
    })
    rec.update(gossip.stats())
    if return_state:
        # live objects for downstream diagnostics; keys starting with "_" are never
        # written to the ledger, so they cannot break JSON serialisation
        rec["_gossip"] = gossip
        rec["_engine"] = engine
        rec["_controller"] = ctrl
    ck.clear()
    return rec


def convergence_round(phi: np.ndarray, tol: float = 0.01) -> int:
    """First round after which phi stays within `tol` of its final-window mean."""
    if len(phi) < 5:
        return int(len(phi))
    target = float(np.mean(phi[-max(1, len(phi) // 5):]))
    band = tol * max(abs(target), 1e-9)
    for t in range(len(phi)):
        if np.all(np.abs(phi[t:] - target) <= band):
            return int(t)
    return int(len(phi))


def recovery_events(df: pd.DataFrame, events: List[Tuple[int, str]],
                    frac: float = 0.95) -> List[dict]:
    """Per event: rounds until phi regains `frac` of its pre-event level, with censoring.

    A unit that never regains the level before the horizon is *censored*: `recovered` is
    False and `rounds` equals the remaining horizon. v1.0 silently reported the horizon as if
    it were a recovery time; the analysis must treat censored units as such.
    """
    phi = df["phi"].to_numpy()
    out = []
    for r, ev in events:
        if r <= 0 or r >= len(phi) - 2:
            continue
        pre = float(np.mean(phi[max(0, r - 5):r]))
        thresh = frac * pre if pre > 0 else pre * (2.0 - frac)
        hit = None
        for t in range(r, len(phi)):
            if phi[t] >= thresh:
                hit = t - r
                break
        horizon = len(phi) - r
        out.append({"round": int(r), "event": str(ev), "pre_phi": pre,
                    "recovered": bool(hit is not None),
                    "rounds": int(hit if hit is not None else horizon),
                    "horizon": int(horizon),
                    "post_phi_mean": float(np.mean(phi[r:]))})
    return out


def recovery_time(df: pd.DataFrame, events: List[Tuple[int, str]]) -> Optional[float]:
    """Mean rounds to return to 95% of the pre-event phi; censored events count the horizon.

    Kept for continuity with v1.0; read it together with `recovery_censored`.
    """
    ev = recovery_events(df, events)
    return float(np.mean([e["rounds"] for e in ev])) if ev else None
# %% [markdown]
# ## 13 - Self-test: does resume actually work?
#
# This runs a short unit to completion, then runs the *same* unit again but kills it
# mid-way and restarts it from the checkpoint, and compares the two traces. Checkpointing
# that has never been interrupted on purpose is an untested claim, and this is the one
# mechanism the whole free-tier story rests on.
#
# Bit-identical traces are the pass condition, which is only achievable because the RNG
# state is part of the checkpoint.

# %%
def selftest_resume(verbose: bool = True) -> dict:
    env = Environment(TESTBEDS["MEL"], n_nodes=8, lattice_m=150.0)
    common = dict(env=env, algo="FogField-LLL", scenario="static", rounds=24, ckpt_every=4)

    # A private checkpoint folder: parallel grid workers each run this self-test at start-up,
    # and must neither race on the same checkpoint file nor delete each other's checkpoints.
    import shutil
    shared_ckpt = DIRS["ckpt"]
    DIRS["ckpt"] = shared_ckpt / "selftest_{}".format(os.getpid())
    DIRS["ckpt"].mkdir(parents=True, exist_ok=True)
    try:
        # (1) clean, uninterrupted reference run
        ref = simulate(seed=901, resume=False, **common)

        # (2) same unit, killed after 9 rounds, then resumed
        part = simulate(seed=901, resume=True, stop_after=9, **common)
        assert part.get("_partial"), "stop_after should have produced a partial result"
        resumed = simulate(seed=901, resume=True, **common)
    finally:
        shutil.rmtree(DIRS["ckpt"], ignore_errors=True)
        DIRS["ckpt"] = shared_ckpt

    a = np.asarray(ref["trace_phi"])
    b = np.asarray(resumed["trace_phi"])
    same_len = len(a) == len(b)
    max_dev = float(np.max(np.abs(a - b))) if same_len else float("inf")
    ok = same_len and max_dev < 1e-9

    out = {"identical": bool(ok), "max_deviation": max_dev,
           "ref_rounds": len(a), "resumed_rounds": len(b),
           "killed_at_round": part["rounds_done"]}
    if verbose:
        print("resume self-test:", out)
        print("  -> " + ("PASS: the resumed run reproduces the uninterrupted run exactly."
                         if ok else
                         "FAIL: resumed trace diverges - do not trust long-run results."))
    return out


RESUME_TEST = selftest_resume()
write_json(DIRS["tables"] / "resume_selftest.json", RESUME_TEST)
assert RESUME_TEST["identical"], "checkpoint/resume is broken"

# %% [markdown]
# ## 14 - Exact reference: MILP upper bound (R4, T3)
#
# The plan calls the optimality-gap plot the single most persuasive element of the paper, so
# the reference has to be defensible.
#
# **What is solved exactly.** One-shot placement maximising
# $w_c\,\mathrm{Coverage}/W - w_e\,\mathrm{Energy}/(N E_{\mathrm{ref}})$, with each node
# assigned to exactly one reachable lattice cell, solved with CBC through PuLP.
#
# **What is left out, and why that is still rigorous.** Connectivity is **not** a constraint
# in the MILP. An exact connectivity-constrained joint placement needs coupling variables over
# *pairs* of cells, which is quadratic in the lattice and does not solve at any interesting
# size inside a free session. Leaving connectivity out makes the MILP a **relaxation**, so its
# objective is a valid **upper bound** on the coverage-minus-energy component that any
# connected solution could reach. The gap reported against it is therefore **conservative** -
# the true gap to a connectivity-feasible optimum is no larger. Heuristics are scored on the
# same reduced objective, so the comparison is like for like, and their connectivity is
# reported separately rather than being quietly discarded.

# %%
import pulp


def phi_reduced(env: Environment, pos, prev, active, w) -> float:
    """Coverage minus energy (flight plus load-dependent compute), no connectivity."""
    cov = env.coverage(pos, active, w)
    ene = env.energy(pos, prev, active, w)
    N = float(env.n_nodes)
    return float(env.weights.coverage * cov / max(env.W_total, 1e-9)
                 - env.weights.energy * ene / (N * env.E_ref))


def phi_subgame(env: Environment, pos, origin, active, w) -> float:
    """The objective of the coverage-energy sub-game (Theorem T3) and of the MILP (v2):
    coverage minus flight energy from `origin`, with compute energy held at its idle level."""
    cov = env.coverage(pos, active, w)
    idx = np.flatnonzero(active)
    e_fly = float(np.sum(env.uav.flight_energy(np.linalg.norm(pos[idx] - origin[idx], axis=1),
                                               env.dt_s)))
    e_idle = float(idx.size * env.uav.P_cpu_idle * env.dt_s)
    N = float(env.n_nodes)
    return float(env.weights.coverage * cov / max(env.W_total, 1e-9)
                 - env.weights.energy * (e_fly + e_idle) / (N * env.E_ref))


def solve_milp(env: Environment, prev: np.ndarray, w: np.ndarray,
               time_limit: int = 120, lattice_cap: int = 90) -> dict:
    """Exact (CBC) upper bound on the reduced objective for a small instance."""
    tb = env.testbed
    N = env.n_nodes

    # candidate cells (v2): EVERY lattice cell, for every node. Heuristic configurations live
    # on the same lattice, so each of them is MILP-feasible and the optimum is an exact upper
    # bound on the sub-game objective. v1.0 subsampled at most `lattice_cap` cells from the
    # nodes' nearest reachable sets, and heuristics occasionally beat that restricted optimum.
    gx, gy = np.meshgrid(env.lat_x, env.lat_y, indexing="ij")
    cells = np.stack([gx.ravel(), gy.ravel()], axis=1)
    C = len(cells)
    # flight energy is charged with the same capped function the scoring uses, so cells
    # beyond one slot's reach are allowed at the capped price, exactly as in phi_subgame
    reach = np.ones((N, C), dtype=bool)

    covers = ((cells[:, None, :] - tb.device_xy[None, :, :]) ** 2).sum(-1) <= env.R_cov ** 2
    e_cost = np.stack([env.uav.flight_energy(
        np.linalg.norm(cells - prev[i][None, :], axis=1), env.dt_s) for i in range(N)])

    prob = pulp.LpProblem("fogfield_placement", pulp.LpMaximize)
    y = {(i, c): pulp.LpVariable("y_{}_{}".format(i, c), cat="Binary")
         for i in range(N) for c in range(C) if reach[i, c]}
    z = {m: pulp.LpVariable("z_{}".format(m), lowBound=0, upBound=1)
         for m in range(tb.n_devices) if covers[:, m].any() and w[m] > 0}

    for i in range(N):
        prob += pulp.lpSum(y[(i, c)] for c in range(C) if (i, c) in y) == 1

    for m in z:
        srcs = [y[(i, c)] for i in range(N) for c in np.flatnonzero(covers[:, m])
                if (i, c) in y]
        if srcs:
            prob += z[m] <= pulp.lpSum(srcs)
        else:
            prob += z[m] == 0

    Wt = max(env.W_total, 1e-9)
    denom = float(N) * env.E_ref
    cover_term = pulp.lpSum(env.weights.coverage * float(w[m]) / Wt * z[m] for m in z)
    # sub-game objective: compute energy at its idle level is a constant (every node active)
    energy_term = pulp.lpSum(env.weights.energy * float(e_cost[i, c]) / denom * y[(i, c)]
                             for (i, c) in y)
    prob += cover_term - energy_term

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    status = prob.solve(solver)
    status_name = pulp.LpStatus[status]

    pos = prev.copy()
    for i in range(N):
        for c in range(C):
            if (i, c) in y and y[(i, c)].value() is not None and y[(i, c)].value() > 0.5:
                pos[i] = cells[c]
                break

    active = np.ones(N, dtype=bool)
    obj = pulp.value(prob.objective)
    # subtract the constant idle compute energy so it is on the scale of phi_subgame
    e_idle = float(N * env.uav.P_cpu_idle * env.dt_s)
    obj_comparable = (float(obj) if obj is not None else float("nan")) \
        - env.weights.energy * e_idle / denom

    return {
        "status": status_name,
        "objective_bound": obj_comparable,
        "objective_achieved": phi_subgame(env, pos, prev, active, w),
        "objective_achieved_reduced": phi_reduced(env, pos, prev, active, w),
        "positions": pos,
        "n_cells": C,
        "n_binaries": len(y),
        "connectivity_of_milp_solution": env.connectivity(pos, active) / N,
        "note": ("connectivity is not constrained, so this is an upper bound on the "
                 "reduced objective and the reported gap is conservative"),
    }


# %% [markdown]
# ## 15 - Equilibrium diagnostics
#
# `nash_gap` asks the question the theory cares about directly: at the configuration the
# dynamics reached, **what is the largest unilateral improvement any single node could still
# make**, measured in the *global* utility? If it is zero, the configuration is a pure Nash
# equilibrium; if it is small, it is an $\varepsilon$-equilibrium and $\varepsilon$ is
# reported rather than asserted.
#
# `locality_gap` measures what the decentralization actually costs at decision time: for
# sampled nodes it compares the ranking the **local** estimator produces against the ranking
# the **global** marginal contribution would produce. This is the quantity that a reviewer
# will ask about the moment they notice the algorithm cannot see the whole network.

# %%
def nash_gap(env: Environment, pos, prev, active, w, max_actions: int = 21) -> dict:
    """Largest unilateral improvement in the global potential; 0 means pure NE."""
    game = Game(env)
    base = env.phi(pos, prev, active, w)
    best = 0.0
    per_node = []
    for i in np.flatnonzero(active):
        cands = env.action_set(pos[i], max_actions)
        p2 = pos.copy()
        gains = []
        for c in cands:
            p2[i] = c
            gains.append(env.phi(p2, prev, active, w) - base)
        g = float(max(gains)) if gains else 0.0
        per_node.append(g)
        best = max(best, g)
    # An exact pure NE (epsilon == 0) is almost never reached in a finite run, so the
    # informative quantity is epsilon relative to the potential actually achieved: it says
    # how far from equilibrium the configuration is in the units of the objective itself.
    rel = float(best / abs(base)) if abs(base) > 1e-12 else float("nan")
    return {"epsilon": float(best), "epsilon_rel": rel,
            "mean_improvement": float(np.mean(per_node)),
            "is_eps_nash_1pct": bool(rel <= 0.01),
            "is_nash": bool(best <= 1e-9), "phi": float(base)}


def locality_gap(env: Environment, pos, prev, active, w, gossip: GossipLayer,
                 n_sample: int = 12, seed: int = 3) -> dict:
    """Agreement between the local estimator and the true global marginal contribution.

    Reports `info_fraction` (mean known-neighbour coverage of the sampled nodes) alongside
    agreement and regret. This is not decorative: at small node counts, or once a
    consensus-style controller clusters its nodes, gossip reaches a full mesh and every node
    ends up with essentially total information. `rank_agreement -> 1.0` and
    `regret -> 0.0` then follow automatically from full information, not from any property
    of the algorithm's local decisions - the local/global distinction this metric is meant to
    probe has collapsed. Read agreement/regret together with `info_fraction`: a claim of
    "near-zero locality cost" is only informative when `info_fraction` is well below 1.
    """
    rng = np.random.default_rng(seed)
    live = np.flatnonzero(active)
    if live.size == 0:
        return {"rank_agreement": float("nan"), "regret_mean": float("nan")}
    sample = rng.choice(live, size=min(n_sample, live.size), replace=False)
    agree, regret, info_frac, top3 = [], [], [], []
    n_others = max(len(live) - 1, 1)
    for i in sample:
        i = int(i)
        cands = env.action_set(pos[i], 15)
        u_loc = local_utility(env, i, cands, pos, prev, gossip.views[i], w,
                              _sensed_devices(env, pos[i]))
        base = env.phi(pos, prev, active, w)
        u_glob = []
        p2 = pos.copy()
        for c in cands:
            p2[i] = c
            u_glob.append(env.phi(p2, prev, active, w) - base)
        u_glob = np.asarray(u_glob)
        pick_loc = int(np.argmax(u_loc))
        pick_glob = int(np.argmax(u_glob))
        agree.append(pick_loc == pick_glob)
        top3.append(pick_loc in set(np.argsort(-u_glob)[:3].tolist()))
        regret.append(float(u_glob[pick_glob] - u_glob[pick_loc]))
        known = len([j for j in gossip.views[i].pos if j != i])
        info_frac.append(known / n_others)
    return {"rank_agreement": float(np.mean(agree)),
            "top3_agreement": float(np.mean(top3)),
            "regret_mean": float(np.mean(regret)),
            "regret_max": float(np.max(regret)), "n_sample": int(len(sample)),
            "info_fraction": float(np.mean(info_frac))}
# %% [markdown]
# ## 16 - Experiment grid
#
# Four studies, each with its own ledger namespace:
#
# 1. **`main`** - every algorithm on every scenario, on every loaded testbed.
# 2. **`scalability`** - 50 to 1000 nodes, the decentralized methods against the
#    centralized reference, tracking both solution quality and *measured* gossip bytes.
# 3. **`milp`** - small instances against the exact CBC bound, producing the optimality-gap
#    figure.
# 4. **`equilibrium`** - the Nash gap and the locality gap at the configurations reached.
#
# **The resume contract.** `run_study` consults the ledger before every unit. A completed
# unit is skipped in microseconds, so re-running this cell after a Colab disconnect costs
# almost nothing for work already done. `TIME_BUDGET_MIN` stops the cell cleanly before a
# free session is likely to be reclaimed; just run it again to continue. Nothing is lost
# either way, because the mid-run checkpointer also holds partial units.

# %%
TIME_BUDGET_MIN = float(os.environ.get("FOGFIELD_BUDGET_MIN", "50"))

if QUICK:
    GRID_NODES = [20]
    GRID_SEEDS = [0, 1]
    GRID_ROUNDS = 50
    SCALE_NODES = [25, 50, 100]
    SCALE_SEEDS = [0]
    MILP_NODES = [5, 6]
    MILP_SEEDS = [0, 1]
else:
    GRID_NODES = [20, 50]
    GRID_SEEDS = [0, 1, 2, 3, 4]
    GRID_ROUNDS = 150
    SCALE_NODES = [50, 100, 200, 400, 800]
    SCALE_SEEDS = [0, 1, 2]
    MILP_NODES = [4, 5, 6, 7, 8]
    MILP_SEEDS = [0, 1, 2, 3, 4]

CKPT_EVERY = 10


def make_env(tb: Testbed, n_nodes: int, **kw) -> Environment:
    defaults = dict(R_cov=420.0, R_comm=820.0, lattice_m=120.0, dt_s=30.0)
    if tb.name == "BEJ":
        defaults.update(R_cov=700.0, R_comm=1400.0, lattice_m=200.0)
    defaults.update(kw)
    return Environment(tb, n_nodes=n_nodes, **defaults)


class BudgetExceeded(Exception):
    pass


def run_study(name: str, units: List[dict], fn, budget_min: float = TIME_BUDGET_MIN) -> None:
    """Execute `units` through `fn`, skipping anything the ledger already holds."""
    only = os.environ.get("FOGFIELD_STUDIES")
    if only and name not in only.split(","):
        print("[{}] skipped (FOGFIELD_STUDIES={})".format(name, only))
        return
    ledger = Ledger(DIRS["results"], name)
    todo = [u for u in units if not ledger.is_done(u["_key"])]
    # Headless parallel runs: worker k of W takes the units whose key hashes to k. The split
    # is a function of the key alone, so workers never overlap and never leave gaps however
    # often they are restarted; every unit is deterministic (seed_streams), so a duplicate
    # would in any case be harmless.
    n_workers = int(os.environ.get("FOGFIELD_NWORKERS", "1"))
    worker = int(os.environ.get("FOGFIELD_WORKER", "0"))
    if n_workers > 1:
        todo = [u for u in todo if int(u["_key"], 16) % n_workers == worker]
    print("[{}] {} unit(s) total, {} already done, {} to run{}"
          .format(name, len(units), len(units) - len(todo), len(todo),
                  " (worker {}/{})".format(worker, n_workers) if n_workers > 1 else ""))
    if not todo:
        return
    t0 = time.time()
    bar = tqdm(todo, desc=name, unit="unit")
    n_failed = 0
    try:
        for u in bar:
            if (time.time() - t0) / 60.0 > budget_min:
                raise BudgetExceeded()
            # A single unit's unhandled exception must not take the rest of a multi-hour
            # queue down with it - this has happened twice in practice during this
            # project's own full run (a numerical divergence in one baseline, a transient
            # Windows file-lock), and `simulate` now guards its own known failure mode, but
            # this is the backstop for whatever the next one turns out to be. The unit is
            # recorded with its error rather than silently dropped, so a real bug doesn't
            # masquerade as "still to do" forever.
            try:
                rec = fn(u)
            except BudgetExceeded:
                raise
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                n_failed += 1
                print("\n[{}] unit {} FAILED: {}: {} - recorded and continuing"
                     .format(name, u["_key"], type(exc).__name__, exc))
                rec = dict(u)
                rec.pop("_key", None)
                rec.update({"key": u["_key"], "failed": True,
                           "error": "{}: {}".format(type(exc).__name__, exc)})
            if rec is not None:
                ledger.save(u["_key"], rec)
            bar.set_postfix_str("done {} (failed {})".format(len(ledger), n_failed))
    except BudgetExceeded:
        bar.close()
        print("[{}] time budget of {:.0f} min reached - {} unit(s) remain. "
              "Re-run this cell to continue; nothing is lost."
              .format(name, budget_min, len(units) - len(ledger)))
        return
    except KeyboardInterrupt:
        bar.close()
        print("[{}] interrupted - {} unit(s) complete and safely on Drive."
              .format(name, len(ledger)))
        return
    bar.close()
    print("[{}] complete: {} unit(s)".format(name, len(ledger)))


# %% [markdown]
# ### Study 1 - main grid

# %%
def grid_main() -> List[dict]:
    units = []
    for tb_name in TESTBEDS:
        for scen in SCENARIOS:
            for algo in ALGORITHMS:
                for n in GRID_NODES:
                    for seed in GRID_SEEDS:
                        u = dict(study="main", testbed=tb_name, scenario=scen, algo=algo,
                                 n_nodes=n, seed=seed, rounds=GRID_ROUNDS)
                        u["_key"] = unit_key(**u)
                        units.append(u)
    return units


def run_main_unit(u: dict) -> dict:
    env = make_env(TESTBEDS[u["testbed"]], u["n_nodes"])
    rec = simulate(env, u["algo"], u["scenario"], u["seed"], u["rounds"],
                   ckpt_every=CKPT_EVERY)
    rec["study"] = "main"
    rec["scenario_desc"] = SCENARIOS[u["scenario"]].description
    return rec


run_study("main", grid_main(), run_main_unit)

# %% [markdown]
# ### Study 2 - scalability and measured communication cost
#
# This is where the decentralization claim has to earn its keep: quality should hold while
# **per-node** gossip cost stays bounded, because a node only ever talks to neighbours.
# Note that the per-node payload is bounded *by construction* (fanout 3, push plus pull, at
# most 12 neighbour ids per message gives a 270-byte ceiling for every method, Static
# included); what distinguishes the methods is quality and wall-clock, and the message count
# per round, which grows linearly with n. The centralized reference is run at every size.

# %%
def grid_scalability() -> List[dict]:
    units = []
    algos = ["FogField-LLL", "FogField-BR", "GossipGreedy", "Consensus-ADMM", "Static"]
    tb_name = "BEJ" if "BEJ" in TESTBEDS else "MEL"
    for n in SCALE_NODES:
        for algo in algos + ["D-POA-style"]:
            for seed in SCALE_SEEDS:
                u = dict(study="scalability", testbed=tb_name, scenario="static",
                         algo=algo, n_nodes=n, seed=seed,
                         rounds=max(30, GRID_ROUNDS // 2))
                u["_key"] = unit_key(**u)
                units.append(u)
    return units


def run_scale_unit(u: dict) -> dict:
    env = make_env(TESTBEDS[u["testbed"]], u["n_nodes"])
    rec = simulate(env, u["algo"], u["scenario"], u["seed"], u["rounds"],
                   ckpt_every=CKPT_EVERY)
    rec["study"] = "scalability"
    return rec


run_study("scalability", grid_scalability(), run_scale_unit)

# %% [markdown]
# ### Study 3 - optimality gap against the exact MILP bound
#
# Both sides answer the same question - *given where the nodes are now, where should they
# end up, counting the cost of getting there* - so both are scored with `phi_reduced`
# evaluated against the **common starting configuration**, from which every heuristic also
# *starts* (v2; in v1.0 the heuristics drew their own initial deployment and only the scoring
# used the common start). The heuristic's cumulative flight path is not charged against it,
# because the MILP pays for exactly one move; charging the path would make the gap look
# better than it is. The MILP instance uses a longer slot (dt = 150 s) and a coarser lattice
# (220 m) than the main grid so that one decision epoch reaches across the region.

# %%
def grid_milp() -> List[dict]:
    units = []
    for n in MILP_NODES:
        for seed in MILP_SEEDS:
            u = dict(study="milp", testbed="MEL", n_nodes=n, seed=seed)
            u["_key"] = unit_key(**u)
            units.append(u)
    return units


def run_milp_unit(u: dict) -> dict:
    # one decision epoch with a long reach: A_i is effectively the whole region
    env = make_env(TESTBEDS["MEL"], u["n_nodes"], dt_s=150.0, lattice_m=220.0)
    rng = np.random.default_rng(1000 + u["seed"])
    start = initial_positions(env, rng)
    w = env.testbed.weights(0)

    milp = solve_milp(env, start, w, time_limit=90 if QUICK else 300)
    rec = {"study": "milp", "n_nodes": u["n_nodes"], "seed": u["seed"],
           "milp_status": milp["status"], "milp_bound": milp["objective_bound"],
           "milp_achieved": milp["objective_achieved"],
           "milp_achieved_reduced": milp["objective_achieved_reduced"],
           "milp_cells": milp["n_cells"], "objective": "subgame-v2",
           "milp_binaries": milp["n_binaries"],
           "milp_connectivity": milp["connectivity_of_milp_solution"]}

    active = np.ones(env.n_nodes, dtype=bool)
    bound = milp["objective_bound"]
    for algo in ["FogField-LLL", "FogField-BR", "GossipGreedy", "Guerrero-style",
                 "Consensus-ADMM", "MARL-IQL", "D-POA-style", "Static", "RandomWalk"]:
        sub = simulate(env, algo, "static", u["seed"], rounds=40,
                       ckpt_every=20, resume=False, init_pos=start)
        # score the final configuration against the COMMON start, so the heuristic is
        # charged for one deployment move exactly as the MILP is
        final = np.asarray(sub["final_pos"])
        rec["phi_sub_" + algo] = phi_subgame(env, final, start, active, w)
        rec["phi_red_" + algo] = phi_reduced(env, final, start, active, w)
        rec["conn_" + algo] = env.connectivity(final, active) / env.n_nodes
        rec["cov_" + algo] = env.coverage(final, active, w) / max(float(np.sum(w)), 1e-9)
        if bound is not None and np.isfinite(bound) and abs(bound) > 1e-12:
            rec["gap_" + algo] = float((bound - rec["phi_sub_" + algo]) / abs(bound))
    return rec


run_study("milp", grid_milp(), run_milp_unit, budget_min=TIME_BUDGET_MIN)

# %% [markdown]
# ### Study 4 - equilibrium and locality diagnostics
#
# **A pitfall found while validating this study, kept here rather than silently patched.**
# An earlier version ran this at `n_nodes=14` with the default `R_comm=820`m in the
# ~2 x 1.4 km MEL area. Direct inspection showed `local_utility(c) - u_glob(c)` was the
# *same additive constant across every candidate, for every sampled node*: gossip had
# reached a full mesh (every node knew every other node's exact position), so the local
# estimator and the true global marginal contribution differ only by a fixed offset and
# rank candidates identically **by construction**. That produces
# `rank_agreement = 1.0, regret = 0.0` even though it demonstrates nothing about
# decentralized decision quality - the information constraint the metric is supposed to
# probe had simply stopped binding. Consensus-ADMM triggered it hardest because its
# consensus term pulls nodes into a tight, fully-connected cluster.
#
# The fix is `n_nodes = 40` with a tighter `R_comm = 500`m (still `>= R_cov`, so it stays a
# physically sensible radio range) against the same ~2.5 km deployment diagonal, so that a
# configuration achieving real coverage - which requires spreading nodes across the device
# field - cannot also be a single communication hop wide. `locality_gap` now also reports
# `info_fraction`: the mean share of the *other* active nodes each sampled node actually
# knows about. Read `rank_agreement`/`regret` together with `info_fraction` - a low regret
# is only informative about decentralization when `info_fraction` is well below 1.

# %%
def grid_equilibrium() -> List[dict]:
    units = []
    for tb_name in TESTBEDS:
        for algo in ["FogField-LLL", "FogField-BR", "GossipGreedy", "Consensus-ADMM"]:
            for seed in GRID_SEEDS:
                u = dict(study="equilibrium", testbed=tb_name, algo=algo, n_nodes=40, seed=seed)
                u["_key"] = unit_key(**u)
                units.append(u)
    return units


EQ_ROUNDS = 80


def run_equilibrium_unit(u: dict) -> dict:
    # MEL: tighter R_comm so that gossip cannot reach a full mesh (see the note above);
    # BEJ: the main-grid radii, since 40 nodes over 11.6 km are far from a full mesh anyway.
    kw = {"R_comm": 500.0} if u["testbed"] == "MEL" else {}
    env = make_env(TESTBEDS[u["testbed"]], u["n_nodes"], **kw)
    sub = simulate(env, u["algo"], "static", u["seed"], rounds=EQ_ROUNDS,
                   ckpt_every=20, resume=False, return_state=True)
    pos = np.asarray(sub["final_pos"])
    prev = np.asarray(sub["final_prev"])
    active = np.asarray(sub["final_alive"], dtype=bool)
    # evaluate at the demand slot the final configuration was reached under
    tb = env.testbed
    w = tb.weights(0 if tb.n_slots == 1 else EQ_ROUNDS - 1)
    gz = sub["_gossip"]

    # rest-point semantics: the slot game at the reached configuration, moves charged from it
    ng = nash_gap(env, pos, pos, active, w)
    lg = locality_gap(env, pos, pos, active, w, gz, n_sample=16)
    rec = {"study": "equilibrium", "testbed": u["testbed"], "algo": u["algo"],
           "seed": u["seed"], "n_nodes": u["n_nodes"], "phi_final": sub["phi_final"],
           "R_comm": env.R_comm}
    rec.update({"nash_" + k: v for k, v in ng.items()})
    rec.update({"loc_" + k: v for k, v in lg.items()})
    return rec


run_study("equilibrium", grid_equilibrium(), run_equilibrium_unit)

# %% [markdown]
# ### Study 5 - log-linear annealing schedule (v2)
#
# T2's selection property is a statement about the limit `beta -> infinity` of a fixed-`beta`
# chain; the implemented FogField-LLL anneals `beta` from 8 toward a cap of 400 over roughly
# 110 rounds, so most of a 150-round run is spent exploring. This sweep asks how much of
# LLL's deficit against FogField-BR is the schedule: the same (testbed, n, seed) units as the
# main grid, `static` scenario, under five schedules. The `base` variant repeats the
# main-grid configuration and must reproduce it exactly (common random numbers), which doubles
# as a cross-process reproducibility check.

# %%
LLL_SCHEDULES = {
    "base":     dict(beta0=8.0, beta_growth=0.45, beta_max=400.0),
    "fast":     dict(beta0=8.0, beta_growth=4.0, beta_max=400.0),
    "cap4000":  dict(beta0=8.0, beta_growth=4.0, beta_max=4000.0),
    "fixed400": dict(beta0=400.0, beta_growth=0.0, beta_max=400.0),
    "fixed50":  dict(beta0=50.0, beta_growth=0.0, beta_max=50.0),
}


def grid_lll_schedule() -> List[dict]:
    units = []
    for tb_name in TESTBEDS:
        for variant in LLL_SCHEDULES:
            for n in GRID_NODES:
                for seed in GRID_SEEDS:
                    u = dict(study="lll_schedule", testbed=tb_name, scenario="static",
                             algo="FogField-LLL", variant=variant, n_nodes=n, seed=seed,
                             rounds=GRID_ROUNDS)
                    u["_key"] = unit_key(**u)
                    units.append(u)
    return units


def run_lll_unit(u: dict) -> dict:
    env = make_env(TESTBEDS[u["testbed"]], u["n_nodes"])
    rec = simulate(env, u["algo"], u["scenario"], u["seed"], u["rounds"],
                   ckpt_every=CKPT_EVERY, ctrl_kwargs=LLL_SCHEDULES[u["variant"]])
    rec["study"] = "lll_schedule"
    rec["variant"] = u["variant"]
    return rec


run_study("lll_schedule", grid_lll_schedule(), run_lll_unit)

# %% [markdown]
# ### Study 6 - the bounded local view, as a logged measurement (v2)
#
# Section 8 bounds each node's view to its `K = 12` nearest known neighbours. v1.0 quoted the
# cost of *not* bounding it (mean view 445 nodes, 0.3 s per node-decision at n = 800 on MEL)
# from an unlogged development run. This study makes it a ledger unit: FogField-BR at
# n = 800 on each testbed, bounded and unbounded, four rounds each, recording the mean view
# size and the measured seconds per node-decision.

# %%
def grid_viewbound() -> List[dict]:
    units = []
    for tb_name in TESTBEDS:
        for cap in (12, None):
            u = dict(study="viewbound", testbed=tb_name, scenario="static", algo="FogField-BR",
                     n_nodes=800, seed=0, rounds=4, view_cap=cap)
            u["_key"] = unit_key(**u)
            units.append(u)
    return units


def run_viewbound_unit(u: dict) -> dict:
    env = make_env(TESTBEDS[u["testbed"]], u["n_nodes"])
    rec = simulate(env, u["algo"], u["scenario"], u["seed"], u["rounds"], ckpt_every=2,
                   resume=False, view_cap=u["view_cap"])
    rec["study"] = "viewbound"
    rec["sec_per_round"] = rec["wallclock_s"] / max(u["rounds"], 1)
    for k in ("trace_phi", "final_pos", "final_prev", "final_alive"):
        rec.pop(k, None)
    return rec


run_study("viewbound", grid_viewbound(), run_viewbound_unit)

if os.environ.get("FOGFIELD_STOP_AFTER_EXPERIMENTS"):
    print("[grid] FOGFIELD_STOP_AFTER_EXPERIMENTS set - stopping before the analysis section")
    raise SystemExit(0)
# %% [markdown]
# ## 17 - Results
#
# Everything below reads from the Drive ledgers, so it works in a **fresh session** with no
# re-computation: run the setup cells, skip the experiment cells, and the analysis renders
# from what is already stored. Figures are written to `figs/` and tables to `tables/` in
# both CSV and LaTeX.

# %%
def load_study(name: str) -> pd.DataFrame:
    df = Ledger(DIRS["results"], name).load_all()
    if df.empty:
        print("[{}] no results yet".format(name))
    return df


MAIN = load_study("main")
SCALE = load_study("scalability")
MILPDF = load_study("milp")
EQ = load_study("equilibrium")

for _n, _d in [("main", MAIN), ("scalability", SCALE), ("milp", MILPDF), ("equilibrium", EQ)]:
    print("{:12s} {:4d} rows".format(_n, len(_d)))


def save_table(df: pd.DataFrame, stem: str, caption: str = "", float_fmt: str = "%.3f"):
    df.to_csv(DIRS["tables"] / (stem + ".csv"), index=True)
    try:
        tex = df.to_latex(float_format=float_fmt, escape=True, caption=caption,
                          label="tab:" + stem)
    except TypeError:
        tex = df.to_latex(float_format=float_fmt)
    atomic_write(DIRS["tables"] / (stem + ".tex"), tex.encode("utf-8"))
    return df


# %% [markdown]
# ### Table 1 - main grid

# %%
if not MAIN.empty:
    cols = ["phi_final", "coverage_frac", "giant_frac", "energy_per_node_J",
            "convergence_round", "gossip_bytes_per_node_round"]
    t1 = (MAIN.groupby(["testbed", "scenario", "algo"])[cols]
          .mean().round(4).sort_index())
    save_table(t1, "table1_main_grid",
               "Mean outcome per testbed, scenario and algorithm.")
    display_cols = ["phi_final", "coverage_frac", "giant_frac", "gossip_bytes_per_node_round"]
    print(t1[display_cols].head(40).to_string())
else:
    t1 = pd.DataFrame()

# %% [markdown]
# ### Figure - convergence and the effect of annealed log-linear learning
#
# The `static` traces show the finite-improvement behaviour the potential structure predicts:
# monotone-ish climb to a plateau. `FogField-LLL` should plateau **higher** than
# `FogField-BR` despite being noisier early, because the annealed chain concentrates on
# potential maximisers instead of stopping at the first equilibrium it reaches.

# %%
def plot_convergence(testbed: str = "MEL", scenario: str = "static", seed: int = 0):
    if MAIN.empty:
        return
    sub = MAIN[(MAIN.testbed == testbed) & (MAIN.scenario == scenario) &
               (MAIN.seed == seed)]
    if sub.empty:
        print("[skip] no convergence data for", testbed, scenario)
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for _, row in sub.sort_values("algo").iterrows():
        tr = row.get("trace_phi")
        if not isinstance(tr, list) or not tr:
            continue
        style = "-" if row["decentralized"] else "--"
        lw = 2.0 if str(row["algo"]).startswith("FogField") else 1.0
        ax.plot(tr, style, lw=lw, label=row["algo"], alpha=0.9)
    ax.set_xlabel("round"); ax.set_ylabel(r"global potential $\Phi$")
    ax.set_title("Convergence - {} / {} (solid = decentralized)".format(testbed, scenario))
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(DIRS["figs"] / "fig2_convergence_{}_{}.png".format(testbed, scenario),
                bbox_inches="tight")
    plt.show()


plot_convergence("MEL", "static")
plot_convergence("MEL", "catastrophic")

# %% [markdown]
# ### Figure - outcome by scenario

# %%
def plot_scenarios(testbed: str = "MEL", metric: str = "phi_final"):
    if MAIN.empty:
        return
    sub = MAIN[MAIN.testbed == testbed]
    if sub.empty:
        return
    piv = sub.pivot_table(index="scenario", columns="algo", values=metric, aggfunc="mean")
    order = [a for a in ALGORITHMS if a in piv.columns]
    piv = piv[order]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    piv.plot(kind="bar", ax=ax, width=0.82)
    ax.set_ylabel(metric); ax.set_xlabel("")
    ax.set_title("{} by scenario - {}".format(metric, testbed))
    ax.legend(fontsize=7, ncol=3, loc="best")
    plt.xticks(rotation=25, ha="right")
    fig.savefig(DIRS["figs"] / "fig3_scenarios_{}_{}.png".format(testbed, metric),
                bbox_inches="tight")
    plt.show()
    return piv


_p = plot_scenarios("MEL", "phi_final")
_ = plot_scenarios("MEL", "coverage_frac")

# %% [markdown]
# ### Figure - optimality gap against the exact bound (the key Q1 plot)
#
# Read this against the plan's threshold: if the gap for FogField is **above about 25%**, the
# theory-first framing is the wrong one and the paper should be pitched as a practical
# decentralized scheme with measured overhead. (v2: the MILP is an exact bound on the
# coverage-energy sub-game objective over the full lattice; no PoA line is drawn, see T3.)

# %%
def analyze_optimality_gap():
    if MILPDF.empty:
        print("[skip] no MILP results")
        return None
    gapcols = [c for c in MILPDF.columns if c.startswith("gap_")]
    if not gapcols:
        print("[skip] MILP produced no comparable bound")
        return None
    g = MILPDF[gapcols].rename(columns=lambda c: c[4:])
    summary = pd.DataFrame({
        "mean_gap_%": (g.mean() * 100).round(2),
        "median_gap_%": (g.median() * 100).round(2),
        "max_gap_%": (g.max() * 100).round(2),
    }).sort_values("mean_gap_%")
    save_table(summary, "table2_optimality_gap",
               "Optimality gap versus the exact CBC upper bound on the reduced objective.")
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    data = [g[c].dropna().to_numpy() * 100 for c in summary.index]
    # `labels=` was renamed to `tick_labels` in matplotlib 3.9; setting the ticks
    # afterwards works on every version Colab might ship
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels(list(summary.index))
    # v2: no PoA line - the PoA <= 2 bound holds only for the coverage-energy sub-game (T3),
    # and the configurations scored here are not certified equilibria of it
    ax.set_ylabel("gap to exact bound (%)")
    ax.set_title("Empirical optimality gap - small instances, MEL")
    plt.xticks(rotation=20, ha="right")
    fig.savefig(DIRS["figs"] / "fig4_optimality_gap.png", bbox_inches="tight")
    plt.show()

    statuses = MILPDF["milp_status"].value_counts().to_dict()
    print("\nCBC solver status counts:", statuses)
    if statuses.get("Optimal", 0) < len(MILPDF):
        print("NOTE: not every instance solved to proven optimality; rows that hit the time "
              "limit give a weaker (still valid) bound, so their gap is an over-estimate.")
    return summary


GAP = analyze_optimality_gap()

# %% [markdown]
# ### Figure - scalability and measured communication cost
#
# The decentralization argument lives or dies here. Per-node gossip bytes should stay
# roughly flat in `N` (a node only ever talks to neighbours), while a centralized scheme must
# pull global state every round.

# %%
def plot_scalability():
    if SCALE.empty:
        print("[skip] no scalability results")
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for algo, sub in SCALE.groupby("algo"):
        agg = sub.groupby("n_nodes").mean(numeric_only=True).sort_index()
        mk = "o-" if str(algo).startswith("FogField") else ".--"
        axes[0].plot(agg.index, agg["phi_final"], mk, label=algo)
        axes[1].plot(agg.index, agg["gossip_bytes_per_node_round"], mk, label=algo)
        axes[2].plot(agg.index, agg["wallclock_s"], mk, label=algo)
    axes[0].set_ylabel(r"$\Phi$ (final)"); axes[0].set_title("solution quality")
    axes[1].set_ylabel("bytes / node / round")
    axes[1].set_title("measured gossip cost per node")
    axes[2].set_ylabel("seconds"); axes[2].set_title("wall-clock per run")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("number of fog nodes")
        ax.legend(fontsize=7)
    fig.savefig(DIRS["figs"] / "fig5_scalability.png", bbox_inches="tight")
    plt.show()

    agg = (SCALE.groupby(["algo", "n_nodes"])
           [["phi_final", "gossip_bytes_per_node_round", "gossip_msgs_per_round",
             "wallclock_s"]].mean().round(3))
    save_table(agg, "table3_scalability",
               "Scalability: quality, measured per-node gossip cost, and wall-clock time.")
    return agg


SCALE_TBL = plot_scalability()

# %% [markdown]
# ### Figure - recovery after disruption
#
# `catastrophic` (50% simultaneous loss) and `seed_loss` are where a decentralized scheme
# should separate from a coordinator-based one, because there is no coordinator to lose.

# %%
def plot_recovery():
    """Recovery per testbed with explicit censoring (v2).

    `recovery_rounds` counts the remaining horizon for a unit that never regains 95% of its
    pre-event potential, so it is only interpretable next to `recovery_censored`, the share
    of such units. Static is the null row: anything Static "recovers" is the environment.
    """
    if MAIN.empty:
        return
    scen = ["catastrophic", "controller_loss", "seed_loss", "random_failures"]
    sub = MAIN[MAIN.scenario.isin(scen)].dropna(subset=["recovery_rounds"])
    if sub.empty:
        print("[skip] no recovery data")
        return
    piv = sub.pivot_table(index=["testbed", "algo"], columns="scenario",
                          values="recovery_rounds", aggfunc="mean")
    cens = sub.pivot_table(index=["testbed", "algo"], columns="scenario",
                           values="recovery_censored", aggfunc="mean")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, tb in zip(axes, sorted(sub.testbed.unique())):
        p = piv.loc[tb] if tb in piv.index.get_level_values(0) else None
        if p is None:
            continue
        p.plot(kind="bar", ax=ax, width=0.8)
        ax.set_title("{} - mean rounds (censored at horizon)".format(tb))
        ax.set_xlabel(""); ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("rounds to regain 95% of pre-event $\\Phi$")
    fig.savefig(DIRS["figs"] / "fig6_recovery.png", bbox_inches="tight")
    plt.show()
    save_table(piv.round(2), "table4_recovery",
               "Mean rounds to recover 95% of pre-event potential, per testbed "
               "(censored units count the remaining horizon; see table4b).")
    save_table(cens.round(3), "table4b_recovery_censored",
               "Share of units that never regained 95% of pre-event potential.")
    print(piv.round(1).to_string())
    print("\ncensored share:\n" + cens.round(2).to_string())
    return piv


REC = plot_recovery()

# %% [markdown]
# ### Equilibrium and locality diagnostics
#
# * **`nash_epsilon`** and **`nash_epsilon_rel`** - the largest unilateral improvement still
#   available in the *global* potential at the configuration reached, absolute and relative
#   to the potential achieved. An exact pure equilibrium is essentially never hit in a finite
#   run, so `epsilon_rel` is the number to read: it says how far from equilibrium the
#   configuration sits in the units of the objective itself. This is the measurable content
#   of T1.
# * **`loc_rank_agreement`** and **`loc_regret_mean`** - how often the local, gossip-only
#   estimator picks the same move the global marginal contribution would, and what it costs
#   when it does not. **Read these together with `loc_info_fraction`.** If a controller's
#   converged nodes end up within gossip range of nearly everyone else (`info_fraction` near
#   1), agreement near 1.0 and regret near 0.0 follow mechanically from full information, not
#   from any property of the algorithm's local decision rule - see the note in section 16
#   on the n=14/full-mesh artifact this study was rescoped to avoid. Low agreement paired
#   with near-zero regret is not a failure either: agreement is a strict `argmax` match, so
#   near-ties between lattice cells of nearly equal value count as disagreements.

# %%
def report_equilibrium():
    if EQ.empty:
        print("[skip] no equilibrium results")
        return None
    cols = {"nash_epsilon": "mean", "nash_epsilon_rel": "mean",
            "nash_mean_improvement": "mean", "nash_is_eps_nash_1pct": "mean",
            "loc_rank_agreement": "mean", "loc_top3_agreement": "mean",
            "loc_info_fraction": "mean",
            "loc_regret_mean": "mean", "loc_regret_max": "max"}
    have = {k: v for k, v in cols.items() if k in EQ.columns}
    keys = ["testbed", "algo"] if "testbed" in EQ.columns else ["algo"]
    tbl = EQ.groupby(keys).agg(have).round(5)
    save_table(tbl, "table5_equilibrium",
               "Equilibrium quality and the cost of neighbour-only information.")
    print(tbl.to_string())
    return tbl


EQ_TBL = report_equilibrium()

# %% [markdown]
# ### Communication overhead summary (R3)
#
# These are **measured** application-layer payload bytes from the running protocol.
# The caveat from section 8 applies and is repeated in the exported table so it travels with
# the numbers: MAC/PHY framing, retransmissions and contention are not included here.

# %%
def report_overhead():
    if MAIN.empty:
        return None
    sub = MAIN[MAIN.algo.isin([a for a in ALGORITHMS if ALGORITHMS[a].uses_gossip])]
    tbl = (sub.groupby(["testbed", "algo"])
           [["gossip_msgs_per_round", "gossip_bytes_per_round",
             "gossip_bytes_per_node_round", "gossip_rounds"]]
           .mean().round(2))
    save_table(tbl, "table6_overhead",
               "Measured gossip overhead (application-layer payload only; no MAC/PHY).")
    print(tbl.to_string())
    if not sub.empty:
        b = sub["gossip_bytes_per_node_round"].mean()
        print("\nAcross the grid, a node emits about {:.0f} payload bytes per round. "
              "At one round per {:.0f} s that is {:.2f} kbit/s per node,\nwhich is the "
              "number the emulator and testbed tier should be asked to confirm under a real "
              "802.11 stack.".format(b, 30.0, b * 8 / 30.0 / 1000))
    return tbl


OVH = report_overhead()

# %% [markdown]
# ### v2 tables: paired Byzantine effect, controller loss, LLL schedule, view bound
#
# With common random numbers the byzantine and static units of one algorithm share the
# deployment, the demand process and the algorithm's own randomness, so the difference
# `phi(byzantine) - phi(static)` is a paired quantity per (testbed, n, seed). Static and
# RandomWalk ignore gossip and must come out at exactly zero; they are the null rows.

# %%
from scipy import stats as sstats


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    out = {"n": int(d.size), "mean_diff": float(d.mean()) if d.size else float("nan")}
    if d.size >= 2 and np.std(d) > 0:
        ci = sstats.t.interval(0.95, d.size - 1, loc=d.mean(), scale=sstats.sem(d))
        out["ci_lo"], out["ci_hi"] = float(ci[0]), float(ci[1])
        try:
            out["wilcoxon_p"] = float(sstats.wilcoxon(d).pvalue)
        except ValueError:
            out["wilcoxon_p"] = float("nan")
    else:
        out["ci_lo"] = out["ci_hi"] = out["mean_diff"]
        out["wilcoxon_p"] = 1.0 if d.size else float("nan")
    return out


def report_paired_scenario(scenario_a: str, scenario_b: str, stem: str, caption: str):
    if MAIN.empty:
        return None
    rows = []
    idx = ["testbed", "algo", "n_nodes", "seed"]
    a = MAIN[MAIN.scenario == scenario_a].set_index(idx)["phi_final"]
    b = MAIN[MAIN.scenario == scenario_b].set_index(idx)["phi_final"]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return None
    a, b = a.loc[common], b.loc[common]
    for (tb, algo), grp in a.groupby(level=["testbed", "algo"]):
        st = _paired(grp.to_numpy(), b.loc[grp.index].to_numpy())
        st["rel_pct"] = 100.0 * st["mean_diff"] / max(abs(b.loc[grp.index].mean()), 1e-12)
        rows.append({"testbed": tb, "algo": algo, "phi_" + scenario_b: float(b.loc[grp.index].mean()),
                     "phi_" + scenario_a: float(grp.mean()), **st})
    tbl = pd.DataFrame(rows).set_index(["testbed", "algo"]).sort_index()
    save_table(tbl.round(4), stem, caption, float_fmt="%.4f")
    print(tbl.round(4).to_string())
    return tbl


BYZ = report_paired_scenario("byzantine", "static", "table7_byzantine_paired",
                             "Paired effect of 15% Byzantine gossip on the final potential.")


def report_controller_loss():
    """Service level after the controller is lost (D-POA-style) against the others."""
    if MAIN.empty or "controller_loss" not in set(MAIN.scenario):
        return None
    sub = MAIN[MAIN.scenario == "controller_loss"]
    tbl = (sub.groupby(["testbed", "algo"])
           [["phi_pre_mid", "phi_post_mid", "phi_post_tail", "cov_post_tail",
             "recovery_rounds", "recovery_censored", "ctrl_bytes_per_round"]]
           .mean())
    tbl["retained_frac"] = tbl["phi_post_tail"] / tbl["phi_pre_mid"]
    save_table(tbl.round(4), "table8_controller_loss",
               "Controller-loss scenario: potential before and after the midpoint, and the "
               "modelled controller uplink while it is alive.", float_fmt="%.4f")
    print(tbl.round(3).to_string())
    return tbl


CTRL_LOSS = report_controller_loss()

LLL_SWEEP = load_study("lll_schedule")


def report_lll_schedule():
    if LLL_SWEEP.empty:
        return None
    tbl = (LLL_SWEEP.groupby(["testbed", "n_nodes", "variant"])["phi_final"]
           .agg(["mean", "std", "count"]))
    # reference rows from the main grid (same units, common random numbers)
    ref = (MAIN[(MAIN.scenario == "static") & MAIN.algo.isin(["FogField-BR", "FogField-LLL"])]
           .groupby(["testbed", "n_nodes", "algo"])["phi_final"].agg(["mean", "std", "count"]))
    ref.index = ref.index.set_names(["testbed", "n_nodes", "variant"])
    ref = ref.rename(index={"FogField-BR": "ref:FogField-BR", "FogField-LLL": "ref:FogField-LLL"})
    tbl = pd.concat([tbl, ref]).sort_index()
    save_table(tbl.round(4), "table9_lll_schedule",
               "FogField-LLL under five annealing schedules (static scenario), with the "
               "main-grid BR and LLL rows for the same units.", float_fmt="%.4f")
    print(tbl.round(3).to_string())
    return tbl


LLL_TBL = report_lll_schedule()

VIEWB = load_study("viewbound")
if not VIEWB.empty:
    cols = [c for c in ["testbed", "view_cap", "view_size_mean", "decide_s_per_node",
                        "sec_per_round", "wallclock_s", "phi_final"] if c in VIEWB.columns]
    vb = VIEWB[cols].sort_values(["testbed", "view_cap"], na_position="last").set_index(["testbed", "view_cap"])
    save_table(vb.round(4), "table10_viewbound",
               "Bounded (K = 12) versus unbounded local view at n = 800: mean view size and "
               "measured seconds per node-decision.", float_fmt="%.4f")
    print(vb.round(4).to_string())

# %% [markdown]
# ### Deployment snapshot
#
# Where the nodes actually end up, against the real device field.

# %%
def plot_final_deployment(testbed: str = "MEL", algo: str = "FogField-LLL",
                          scenario: str = "static"):
    if MAIN.empty:
        return
    sub = MAIN[(MAIN.testbed == testbed) & (MAIN.algo == algo) &
               (MAIN.scenario == scenario)]
    if sub.empty:
        print("[skip] no deployment snapshot for", testbed, algo)
        return
    row = sub.iloc[0]
    pos = np.asarray(row["final_pos"])
    alive = np.asarray(row["final_alive"], dtype=bool)
    tb = TESTBEDS[testbed]
    env = make_env(tb, int(row["n_nodes"]))
    w = tb.weights(0)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.scatter(tb.device_xy[:, 0] / 1000, tb.device_xy[:, 1] / 1000, s=10, c=w,
               cmap="viridis", alpha=0.7, linewidths=0, label="devices")
    P = pos[alive]
    for p in P:
        ax.add_patch(plt.Circle((p[0] / 1000, p[1] / 1000), env.R_cov / 1000,
                                fill=False, ls=":", lw=0.7, color="crimson", alpha=0.6))
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    for a in range(len(P)):
        for b in range(a + 1, len(P)):
            if d2[a, b] <= env.R_comm ** 2:
                ax.plot([P[a, 0] / 1000, P[b, 0] / 1000],
                        [P[a, 1] / 1000, P[b, 1] / 1000],
                        "-", lw=0.6, color="steelblue", alpha=0.6)
    ax.scatter(P[:, 0] / 1000, P[:, 1] / 1000, marker="^", s=60, color="crimson",
               edgecolor="k", linewidths=0.4, zorder=5, label="fog nodes")
    ax.set_aspect("equal")
    ax.set_xlabel("east (km)"); ax.set_ylabel("north (km)")
    ax.set_title("{} - {} / {}\ncoverage {:.1%}, giant component {:.1%}".format(
        testbed, algo, scenario, row["coverage_frac"], row["giant_frac"]))
    ax.legend(fontsize=7, loc="best")
    fig.savefig(DIRS["figs"] / "fig7_deployment_{}_{}.png".format(testbed, algo),
                bbox_inches="tight")
    plt.show()


plot_final_deployment("MEL", "FogField-LLL", "static")
# %% [markdown]
# ## 18 - Run manifest
#
# A single JSON capturing what produced these numbers: dataset provenance, parameters, and
# the verification results. Keep it beside the figures - it is what makes the run
# reconstructible months later, and reviewers increasingly ask for exactly this.

# %%
MANIFEST = {
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "quick_mode": QUICK,
    "datasets": {name: tb.meta for name, tb in TESTBEDS.items()},
    "testbed_summary": {name: tb.describe() for name, tb in TESTBEDS.items()},
    "uav_model": {
        "reference": "Zeng, Xu & Zhang, IEEE TWC 18(4):2329-2345, 2019, "
                     "DOI 10.1109/TWC.2019.2902559",
        "P0_W": UAV.P0, "Pi_W": UAV.Pi, "U_tip_mps": UAV.U_tip, "v0_mps": UAV.v0,
        "d0": UAV.d0, "rho": UAV.rho, "s": UAV.s, "A_m2": UAV.A,
        "V_max_mps": UAV.V_max, "battery_Wh": UAV.battery_Wh,
        "P_hover_W": UAV.P_hover,
    },
    "weights": {"coverage": Weights().coverage, "connectivity": Weights().connectivity,
                "energy": Weights().energy,
                "note": "weighted scalarisation; one point on the trade-off surface, "
                        "not a Pareto front"},
    "grid": {"nodes": GRID_NODES, "seeds": GRID_SEEDS, "rounds": GRID_ROUNDS,
             "scale_nodes": SCALE_NODES, "milp_nodes": MILP_NODES},
    "verification": {
        "exact_potential_uncapacitated": _res,
        "exact_potential_capacitated": _res_cap,
        "checkpoint_resume_selftest": RESUME_TEST,
    },
    "counts": {n: len(Ledger(DIRS["results"], n))
               for n in ("main", "scalability", "milp", "equilibrium", "lll_schedule", "viewbound")},
    "seeding": ("v2: common random numbers; deployment/scenario/adversary streams seeded by "
                "(testbed, n, seed), algorithm stream by (testbed, n, seed, algorithm); "
                "SHA-256-derived seeds, independent of PYTHONHASHSEED"),
    "lll_schedules": LLL_SCHEDULES,
    "scenarios": {k: v.description for k, v in SCENARIOS.items()},
    "algorithms": {k: {"decentralized": v.decentralized, "uses_gossip": v.uses_gossip}
                   for k, v in ALGORITHMS.items()},
}
write_json(ROOT / "run_manifest.json", MANIFEST)
print(json.dumps({k: MANIFEST[k] for k in
                  ("generated_utc", "quick_mode", "counts", "verification")},
                 indent=1, default=_json_default))
print("\nmanifest -> " + str(ROOT / "run_manifest.json"))

# %% [markdown]
# ## 19 - How to resume after a disconnect
#
# Free Colab sessions are reclaimed without warning. The recovery procedure is:
#
# 1. Reopen the notebook and **Run all**.
# 2. Setup, Drive mount and dataset loading complete in seconds - the datasets are cached on
#    Drive as `.npz`, so nothing is re-downloaded.
# 3. Each `run_study` cell consults the ledger, skips every finished unit, and continues with
#    the rest. A unit that was interrupted mid-run restarts from its last round-level
#    checkpoint, with the RNG state restored, not from round zero.
# 4. The analysis cells read the ledgers, so partial results plot fine; they simply get
#    better as more units land.
#
# To **start over deliberately**, delete the relevant folder under the Drive root:
# `results/` to rerun experiments, `ckpt/` to drop partial runs, `cache/` to force a fresh
# download. The cell below does this on request - it is guarded, because on Drive this is
# not undoable.

# %%
def reset_state(what: str = "none", confirm: bool = False) -> None:
    """Delete stored state. `what` is one of: none, ckpt, results, cache, all."""
    import shutil
    targets = {"ckpt": [DIRS["ckpt"]], "results": [DIRS["results"]],
               "cache": [DIRS["cache"]],
               "all": [DIRS["ckpt"], DIRS["results"], DIRS["cache"],
                       DIRS["figs"], DIRS["tables"]]}
    if what == "none":
        print("nothing to do. Call reset_state('ckpt'|'results'|'cache'|'all', confirm=True)")
        return
    if what not in targets:
        raise ValueError("unknown target: " + what)
    if not confirm:
        print("This would permanently delete, from Google Drive:")
        for p in targets[what]:
            n = len(list(p.rglob("*"))) if p.exists() else 0
            print("  {}  ({} file(s))".format(p, n))
        print("Re-run with confirm=True if that is what you want.")
        return
    for p in targets[what]:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
    print("reset '{}' done.".format(what))


reset_state("none")

# %% [markdown]
# ## 20 - Limitations, stated plainly
#
# The plan's own risk register is that overclaiming is what gets this paper rejected. These
# are the limits of what this notebook establishes, and none of them should be softened in
# the write-up.
#
# 1. **No measured wireless cost.** Byte and message counts here are application-layer
#    payload from the real protocol. 802.11 MAC framing, ACKs, retransmission, contention
#    and loss are absent. The plan's R3 requires a Mininet-WiFi/ns-3 tier and a 10-20 node
#    Raspberry-Pi/Jetson testbed; **neither runs in Colab**, and the sim-to-real gap stays
#    unquantified until they do. Report these numbers as *protocol payload*, never as
#    "measured communication overhead" without that qualifier.
# 2. **No claim of a provable global-optimum approximation.** What is established is the
#    exact-potential property (verified numerically), convergence of idealised best response
#    with inertia to a constrained equilibrium (T1, which needs the loiter energy model), and an
#    empirical gap to an exact bound. The only worst-case guarantee (T3) is for the
#    coverage-energy sub-game: gamma(NE) >= (gamma(OPT) - E(NE)) / 2. The full objective is NOT a
#    valid utility system (its giant-component term is not submodular), so no PoA bound is
#    claimed for it.
# 3. **The MILP is a relaxation.** Connectivity is not constrained in it, so it is an upper
#    bound and the reported gap is conservative rather than exact. Instances are small
#    (4-8 nodes) because the exact problem does not scale; do not extrapolate the gap to
#    1000-node deployments.
# 4. **Baselines are reimplementations.** D-POA-style, Guerrero-style and the ADMM baseline
#    are written from published method descriptions, not the authors' code, and are not
#    calibrated to reproduce their published numbers. The MARL baseline is independent
#    Q-learning, **not** MAPPO; the plan already notes that no verified 2024-2026
#    peer-reviewed MARL-for-node-*placement* exemplar exists, and this notebook does not
#    close that gap.
# 5. **Weighted scalarisation only.** One weight vector gives one point on the trade-off
#    surface. No Pareto front is computed and **no hypervolume is reported** - the plan is
#    explicit that promising hypervolume without delivering it is a rejection trigger.
# 6. **Synthetic dynamics on MEL.** The EUA snapshot is static, so `mobility` on MEL is a
#    travelling hotspot. Every affected record carries `synthetic_dynamics=True`. Real
#    temporal structure comes only from BEJ.
# 7. **GeoLife pooling.** BEJ demand is pooled across days onto a time-of-day clock, so it
#    is a diurnal demand surface, not one day's simultaneous population. It also covers
#    182 volunteers in Beijing, which is not an IoT deployment; it is a defensible *proxy*
#    for spatio-temporal demand and should be described as one.
# 8. **Discretised action sets.** The potential-game formulation needs a finite strategy set,
#    so nodes move on a lattice. Optimality in continuous space is not addressed.
# 9. **Dynamic regret is not proven.** Re-triggered dynamics on topology events are
#    implemented and recovery is measured, but the equilibrium-tracking argument stays a
#    discussion point, exactly as the plan's T4 advises.
#
# ### Dataset note - why Milano is absent
#
# The Telecom Italia *Milano* grid (Harvard Dataverse, DOI `10.7910/DVN/EGZHFV`, ODbL) is the
# obvious spatio-temporal demand dataset for this problem, and the plan's scenario design
# echoes it. It is **not** used here because its files require a **Dataverse guestbook
# response** before download; the access API returns
# `"You may not download this file without the required Guestbook response"` to an
# unauthenticated request, so it cannot be fetched by a notebook that is meant to run
# unattended. To use it anyway: accept the guestbook in a browser, place a day file such as
# `sms-call-internet-mi-2013-11-01.txt` into the Drive `cache/` folder, and add a loader that
# bins `(square_id, time_interval, internet_activity)` the way `load_geolife` bins GeoLife.
# The square grid is 100x100 cells of 235 m, so cell `id` maps to metres as
# `x = ((id - 1) % 100) * 235`, `y = ((id - 1) // 100) * 235`.
#
# ### Next steps toward submission
#
# 1. Run with `QUICK = False` to fill the full grid, including 800-node scalability.
# 2. Stand up the Mininet-WiFi and Pi/Jetson tiers; feed them the byte and message traces
#    from `table6_overhead` and report the sim-to-real deviation.
# 3. If the FogField gap in `table2_optimality_gap` exceeds ~25%, switch the framing to a
#    practical decentralized scheme with measured overhead and target IoT-J or FGCS rather
#    than leading with theory at TNSM.

# %%
print("FogField notebook complete.\n")
print("Artifacts on Drive:")
for _k in ("figs", "tables", "results", "ckpt", "cache"):
    _p = DIRS[_k]
    _n = len(list(_p.rglob("*"))) if _p.exists() else 0
    print("  {:9s} {:4d} file(s)   {}".format(_k, _n, _p))
print("\n  manifest  " + str(ROOT / "run_manifest.json"))
