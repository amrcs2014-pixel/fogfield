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
# | `seed_loss` | the bootstrap/seed node is removed at the midpoint |
# | `byzantine` | a fraction of nodes gossip falsified positions and demand hints |
#
# **On `seed_loss`.** FogField has no controller, so "controller-loss recovery" has no
# referent. The plan's reinterpretation is used instead: kill the **bootstrap/seed node**
# that new joiners rendezvous with, and measure re-convergence. Centralized baselines are
# not exempt from this scenario - losing the coordinator is exactly where they should suffer,
# and that contrast is the point of the experiment.
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
    "seed_loss": Scenario("seed_loss", "bootstrap/seed node removed at the midpoint"),
    "byzantine": Scenario("byzantine", "a fraction of nodes gossip falsified state"),
}

P_FAIL = 0.010
FLASH_FACTOR = 6.0
BYZ_FRACTION = 0.15


class ScenarioEngine:
    """Turns a scenario key into per-round mutations of the simulation state."""

    def __init__(self, scenario: str, env: Environment, rounds: int,
                 rng: np.random.Generator):
        self.key = scenario
        self.env = env
        self.rounds = rounds
        self.events: List[Tuple[int, str]] = []
        tb = env.testbed
        x0, x1, y0, y1 = tb.bounds
        self.flash_centre = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1)])
        self.flash_centre += rng.normal(0, 0.18 * (x1 - x0), 2)
        self.flash_window = (rounds // 3, rounds // 3 + max(6, rounds // 6))
        self.hotspot_path = np.stack([
            np.linspace(x0, x1, max(rounds, 2)),
            y0 + (y1 - y0) * (0.5 + 0.35 * np.sin(np.linspace(0, 2 * np.pi, max(rounds, 2)))),
        ], axis=1)
        self.byz_mask = np.zeros(env.n_nodes, dtype=bool)
        if scenario == "byzantine":
            k = max(1, int(round(BYZ_FRACTION * env.n_nodes)))
            self.byz_mask[rng.choice(env.n_nodes, k, replace=False)] = True

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

        if self.key == "flash_crowd":
            lo, hi = self.flash_window
            if lo <= rnd < hi:
                d2 = ((tb.device_xy - self.flash_centre[None, :]) ** 2).sum(1)
                boost = 1.0 + (FLASH_FACTOR - 1.0) * np.exp(-d2 / (2 * self.env.R_cov ** 2))
                w = w * boost
        return np.ascontiguousarray(w, dtype=np.float64)

    # ---- topology ----------------------------------------------------------
    def apply_topology(self, rnd: int, alive: np.ndarray, joined: np.ndarray,
                       rng: np.random.Generator) -> str:
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
# internal state, the metric trace and **the RNG state**. If the session dies, the next call
# with the same parameters picks up from the last checkpoint rather than round zero.
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


def simulate(env: Environment, algo: str, scenario: str, seed: int, rounds: int,
             ckpt_every: int = 25, resume: bool = True, stop_after: Optional[int] = None,
             progress: bool = False, return_state: bool = False) -> dict:
    """Run one (env, algorithm, scenario, seed) unit. Resumable."""
    ident = dict(testbed=env.testbed.name, algo=algo, scenario=scenario, seed=seed,
                 n_nodes=env.n_nodes, rounds=rounds, R_cov=env.R_cov, R_comm=env.R_comm,
                 cap=env.capacity, lattice=env.lattice_m, quick=QUICK)
    key = unit_key(**ident)
    ck = RunCheckpointer(key, ckpt_every, enabled=resume)

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
    else:
        rng = np.random.default_rng(abs(hash(key)) % (2 ** 32))
        ctrl = ALGORITHMS[algo]()
        engine = ScenarioEngine(scenario, env, rounds, rng)
        pos = initial_positions(env, rng)
        prev = pos.copy()
        alive = np.ones(env.n_nodes, dtype=bool)
        joined = np.ones(env.n_nodes, dtype=bool)
        if scenario == "expansion":                 # hold a reserve back for the joins
            hold = max(1, int(0.3 * env.n_nodes))
            joined[-hold:] = False
            alive[-hold:] = False
        battery = np.full(env.n_nodes, env.uav.battery_Wh * 3600.0)
        gossip = GossipLayer(env.n_nodes, env.R_comm)
        ctrl.reset(env, pos, rng)
        trace = []
        rnd0 = 0
        resumed_from = None
        diverged_rounds = 0

    t_start = time.time()
    last = min(rounds, rnd0 + stop_after) if stop_after else rounds
    it = range(rnd0, last)
    if progress:
        it = tqdm(it, desc="{}/{}/{}".format(env.testbed.name, algo, scenario), leave=False)

    for rnd in it:
        w = engine.weights(rnd)
        ev = engine.apply_topology(rnd, alive, joined, rng)
        if ev and hasattr(ctrl, "on_event"):
            ctrl.on_event()

        # gossip: byzantine nodes report falsified state on the wire
        pos_report = pos.copy()
        hint = np.zeros(env.n_nodes)
        for i in np.flatnonzero(alive):
            dev = _sensed_devices(env, pos[i])
            hint[i] = float(w[dev].sum()) if dev.size else 0.0
        if engine.byz_mask.any():
            b = engine.byz_mask & alive
            x0, x1, y0, y1 = env.testbed.bounds
            pos_report[b] = np.stack([rng.uniform(x0, x1, b.sum()),
                                      rng.uniform(y0, y1, b.sum())], axis=1)
            hint[b] = hint.max() * 3.0 if hint.max() > 0 else 1.0

        nbrs = gossip.round(rnd, pos, alive, battery, hint, rng, pos_report=pos_report)

        ctx = StepContext(rnd=rnd, pos=pos, prev=prev, alive=alive, w=w, nbrs=nbrs,
                          gossip=gossip, rng=rng, byzantine=engine.byz_mask)
        # A controller's own numerical failure (e.g. an iterative baseline diverging at a
        # node count/density it wasn't tuned for) must degrade to "no move this round" for
        # the affected nodes, not crash a multi-hour queue over one flaky baseline unit.
        # Observed in practice: Consensus-ADMM overflowed to NaN at n=800 before its dual
        # variables were clipped (see ConsensusADMM.step) - this is the second line of
        # defence, for whatever the next such case turns out to be.
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

        ck.maybe_save(rnd + 1, {
            "rnd": rnd + 1, "pos": pos, "prev": prev, "alive": alive, "joined": joined,
            "battery": battery, "gossip": gossip, "ctrl": ctrl, "engine": engine,
            "trace": trace, "rng": rng_state_of(rng), "diverged_rounds": diverged_rounds,
        })

    if stop_after and last < rounds:
        ck.maybe_save(last, {
            "rnd": last, "pos": pos, "prev": prev, "alive": alive, "joined": joined,
            "battery": battery, "gossip": gossip, "ctrl": ctrl, "engine": engine,
            "trace": trace, "rng": rng_state_of(rng), "diverged_rounds": diverged_rounds,
        }, force=True)
        return {"_partial": True, "rounds_done": last, "key": key}

    df = pd.DataFrame(trace)
    tail = max(1, len(df) // 5)
    final = df.tail(tail)
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
        "wallclock_s": float(time.time() - t_start),
        "diverged_rounds": int(diverged_rounds),
        "decentralized": bool(ALGORITHMS[algo].decentralized),
        "synthetic_dynamics": bool(scenario == "mobility" and env.testbed.n_slots == 1),
        "events": ["{}@{}".format(e, r) for r, e in engine.events],
        "trace_phi": [round(float(v), 6) for v in df["phi"].tolist()],
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


def recovery_time(df: pd.DataFrame, events: List[Tuple[int, str]]) -> Optional[float]:
    """Mean rounds to return to 95% of the pre-event phi after a disruptive event."""
    if not events:
        return None
    phi = df["phi"].to_numpy()
    outs = []
    for r, _ev in events:
        if r <= 0 or r >= len(phi) - 2:
            continue
        pre = float(np.mean(phi[max(0, r - 5):r]))
        thresh = 0.95 * pre if pre > 0 else pre * 1.05
        rec = None
        for t in range(r, len(phi)):
            if phi[t] >= thresh:
                rec = t - r
                break
        outs.append(rec if rec is not None else len(phi) - r)
    return float(np.mean(outs)) if outs else None
