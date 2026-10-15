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
