"""One-off patch (15 Oct 2026) that turned fogfield_src_fullrun.py (v1.0, 23 Sep 2026) into the
v2 source used for the TNSM major revision. Kept for the record; do NOT re-run it on the
patched file (every replacement asserts that its target text occurs exactly once).

What it changes (referee report items in brackets):
  * common random numbers: deployment / scenario / adversary streams seeded by
    (testbed, n, seed), algorithm stream by (testbed, n, seed, algorithm); no hash()  [M1, C4]
  * `controller_loss` scenario: the centralised controller dies at the midpoint, its nodes
    hold position; a demand surge follows; modelled controller uplink bytes logged      [C5]
  * bounded-view measurement as a logged study (`viewbound`)                            [C6]
  * recovery with explicit censoring, pre/post-event service levels                      [C3]
  * D-POA-style scalability at every n; equilibrium study on 5 seeds x 2 testbeds with the
    rank-agreement column; heuristics in the MILP study start from the MILP start       [M4, M5, M9]
  * log-linear annealing-schedule sweep (`lll_schedule`)                                 [M6]
  * GeoLife parser user-id bug (manifest said n_users_used = 1)                          [m8]
  * worker partitioning + stop-after-experiments switches for parallel headless runs
"""
import re
import sys
from pathlib import Path

SRC = Path(__file__).parent / "fogfield_src_fullrun.py"
src = SRC.read_text(encoding="utf-8")
orig = src
n_applied = 0


def rep(old, new, count=1):
    global src, n_applied
    k = src.count(old)
    assert k == count, "expected {} occurrence(s), found {} for:\n{}".format(count, k, old[:200])
    src = src.replace(old, new)
    n_applied += 1


def splice(start_marker, end_marker, new_block):
    """Replace everything from start_marker up to and including end_marker."""
    global src, n_applied
    assert src.count(start_marker) == 1, start_marker
    assert src.count(end_marker) == 1, end_marker
    a = src.index(start_marker)
    b = src.index(end_marker) + len(end_marker)
    assert b > a
    src = src[:a] + new_block + src[b:]
    n_applied += 1


# ------------------------------------------------------------------ 2 - output root
rep('        root = Path("./fogfield_out_full").resolve()',
    '        root = Path("./fogfield_out_v2").resolve()')

# ------------------------------------------------------------------ 3 - seeding + ckpt version
rep('''def unit_key(**params) -> str:
    """Deterministic short id for a work unit."""
    blob = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]
''', '''def unit_key(**params) -> str:
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
''')
rep('    VERSION = 2\n', '    VERSION = 3\n')

# ------------------------------------------------------------------ 4 - GeoLife user ids
rep('''            parts = name.split("/")
            uid = parts[1] if len(parts) > 2 else "000"
            by_user.setdefault(uid, []).append(name)
        users = sorted(by_user)[:max_users]''',
    '''            parts = name.split("/")
            # paths look like "Geolife Trajectories 1.3/Data/<user>/Trajectory/<file>.plt";
            # v1.0 took parts[1], which is the literal "Data" for every file, so the user cap
            # never applied (all 182 users were read) and the manifest reported 1 user
            uid = parts[parts.index("Data") + 1] if "Data" in parts[:-1] else "000"
            by_user.setdefault(uid, []).append(name)
        users = sorted(by_user) if max_users is None else sorted(by_user)[:max_users]''')
rep('''    tag = "geolife_u{}_g{:.0f}_s{}_c{}".format(max_users, grid_m, slot_min, top_cells)''',
    '''    tag = "geolife_u{}_g{:.0f}_s{}_c{}".format("all" if max_users is None else max_users,
                                                 grid_m, slot_min, top_cells)''')
rep('''def load_geolife(max_users: int = 50, grid_m: float = 400.0, slot_min: int = 60,''',
    '''def load_geolife(max_users: Optional[int] = None, grid_m: float = 400.0, slot_min: int = 60,''')
rep('''        max_users=25 if QUICK else 60,''', '''        max_users=None,                 # all 182 users (see the parser note above)''')

# ------------------------------------------------------------------ 8 - gossip: beacon-side view bound
rep('''    def __init__(self, n_nodes: int, R_comm: float, fanout: int = 3, ttl: int = 4,
                 max_nbr_ids: int = 12):
        self.n = n_nodes
        self.R_comm = R_comm
        self.fanout = fanout
        self.ttl = ttl
        self.max_nbr_ids = max_nbr_ids
''', '''    def __init__(self, n_nodes: int, R_comm: float, fanout: int = 3, ttl: int = 4,
                 max_nbr_ids: int = 12, max_view: Optional[int] = 12):
        self.n = n_nodes
        self.R_comm = R_comm
        self.fanout = fanout
        self.ttl = ttl
        self.max_nbr_ids = max_nbr_ids      # neighbour ids carried on the wire (uint8 count)
        self.max_view = max_view            # beacon-side bound on the local view; None = unbounded
''')
rep('''        for i in np.flatnonzero(alive):
            cand = nbrs[i]
            if cand.size > self.max_nbr_ids:
                d2 = ((pos[cand] - pos[i][None, :]) ** 2).sum(1)
                cand = cand[np.argsort(d2)[: self.max_nbr_ids]]
            v = self.views[i]''',
    '''        for i in np.flatnonzero(alive):
            cand = nbrs[i]
            if self.max_view is not None and cand.size > self.max_view:
                d2 = ((pos[cand] - pos[i][None, :]) ** 2).sum(1)
                cand = cand[np.argsort(d2)[: self.max_view]]
            v = self.views[i]''')
rep('''    def stats(self) -> dict:
        rounds = max(len(self.per_round), 1)''',
    '''    def view_sizes(self) -> np.ndarray:
        """Number of other nodes each node currently knows about (its local view)."""
        return np.array([len(v.pos) for v in self.views], dtype=np.int64)

    def stats(self) -> dict:
        rounds = max(len(self.per_round), 1)''')

# ------------------------------------------------------------------ 10 - FogField-BR decision timer
rep('''    def __init__(self, p_activate: float = 0.35, inertia: float = 0.15,
                 max_actions: int = 21):
        self.p_activate = p_activate
        self.inertia = inertia
        self.max_actions = max_actions

    def step(self, env, ctx):
        target = ctx.pos.copy()
        for i in np.flatnonzero(ctx.alive):
            if ctx.rng.random() > self.p_activate:
                continue
            if ctx.rng.random() < self.inertia:
                continue
            cands = env.action_set(ctx.pos[i], self.max_actions)
            u = local_utility(env, i, cands, ctx.pos, ctx.prev,
                              ctx.gossip.views[i], ctx.w,
                              _sensed_devices(env, ctx.pos[i]))
            target[i] = cands[int(np.argmax(u))]
        return _apply_move(env, ctx.pos, target, ctx.alive)
''', '''    def __init__(self, p_activate: float = 0.35, inertia: float = 0.15,
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
            target[i] = cands[int(np.argmax(u))]
        return _apply_move(env, ctx.pos, target, ctx.alive)
''')

# ------------------------------------------------------------------ 11 - scenarios
rep('''# | `seed_loss` | the bootstrap/seed node is removed at the midpoint |
# | `byzantine` | a fraction of nodes gossip falsified positions and demand hints |
#
# **On `seed_loss`.** FogField has no controller, so "controller-loss recovery" has no
# referent. The plan's reinterpretation is used instead: kill the **bootstrap/seed node**
# that new joiners rendezvous with, and measure re-convergence. Centralized baselines are
# not exempt from this scenario - losing the coordinator is exactly where they should suffer,
# and that contrast is the point of the experiment.
''', '''# | `seed_loss` | one node (the bootstrap/seed node) is removed at the midpoint |
# | `byzantine` | a fraction of nodes gossip falsified positions and demand hints |
# | `controller_loss` | the centralised controller dies at the midpoint; a demand surge follows |
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
''')
rep('''    "seed_loss": Scenario("seed_loss", "bootstrap/seed node removed at the midpoint"),
    "byzantine": Scenario("byzantine", "a fraction of nodes gossip falsified state"),
}
''', '''    "seed_loss": Scenario("seed_loss", "one node (the bootstrap/seed node) removed at the midpoint"),
    "byzantine": Scenario("byzantine", "a fraction of nodes gossip falsified state"),
    "controller_loss": Scenario("controller_loss",
                                "centralised controller dies at the midpoint; demand surge follows"),
}
''')

splice('''class ScenarioEngine:
    """Turns a scenario key into per-round mutations of the simulation state."""''',
       '''        if ev:
            self.events.append((rnd, ev))
        return ev
''', '''class ScenarioEngine:
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
        if scenario == "controller_loss":
            # the surge starts ten rounds after the controller is lost
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
        return self.key == "controller_loss" and rnd >= self.rounds // 2

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

        if self.key in ("flash_crowd", "controller_loss"):
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
''')

# ------------------------------------------------------------------ 12 - simulate
rep('''# One call to `simulate` is one **work unit**. It checkpoints every `ckpt_every` rounds,
# saving positions, batteries, the gossip layer with its byte counters, the controller's
# internal state, the metric trace and **the RNG state**. If the session dies, the next call
# with the same parameters picks up from the last checkpoint rather than round zero.
''', '''# One call to `simulate` is one **work unit**. It checkpoints every `ckpt_every` rounds,
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
''')

splice('''def simulate(env: Environment, algo: str, scenario: str, seed: int, rounds: int,''',
       '''    return float(np.mean(outs)) if outs else None
''', '''CTRL_CMD_BYTES = struct.calcsize("<Hff")      # position command: node id + x + y


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
''')

# ------------------------------------------------------------------ 16 - grid
rep('''TIME_BUDGET_MIN = 50 if QUICK else 50''',
    '''TIME_BUDGET_MIN = float(os.environ.get("FOGFIELD_BUDGET_MIN", "50"))''')
rep('''    ledger = Ledger(DIRS["results"], name)
    todo = [u for u in units if not ledger.is_done(u["_key"])]
    print("[{}] {} unit(s) total, {} already done, {} to run"
          .format(name, len(units), len(units) - len(todo), len(todo)))
''', '''    ledger = Ledger(DIRS["results"], name)
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
''')
rep('''# This is where the decentralization claim has to earn its keep: quality should hold while
# **per-node** gossip cost stays roughly flat, because a node only ever talks to neighbours.
# The centralized reference is included at the sizes where it is still affordable.
''', '''# This is where the decentralization claim has to earn its keep: quality should hold while
# **per-node** gossip cost stays bounded, because a node only ever talks to neighbours.
# Note that the per-node payload is bounded *by construction* (fanout 3, push plus pull, at
# most 12 neighbour ids per message gives a 270-byte ceiling for every method, Static
# included); what distinguishes the methods is quality and wall-clock, and the message count
# per round, which grows linearly with n. The centralized reference is run at every size.
''')
rep('''        for algo in algos + (["D-POA-style"] if n <= 100 else []):''',
    '''        for algo in algos + ["D-POA-style"]:''')
rep('''# Both sides answer the same question - *given where the nodes are now, where should they
# end up, counting the cost of getting there* - so both are scored with `phi_reduced`
# evaluated against the **common starting configuration**. The heuristic's cumulative flight
# path is not charged against it, because the MILP pays for exactly one move; charging the
# path would make the gap look better than it is.
''', '''# Both sides answer the same question - *given where the nodes are now, where should they
# end up, counting the cost of getting there* - so both are scored with `phi_reduced`
# evaluated against the **common starting configuration**, from which every heuristic also
# *starts* (v2; in v1.0 the heuristics drew their own initial deployment and only the scoring
# used the common start). The heuristic's cumulative flight path is not charged against it,
# because the MILP pays for exactly one move; charging the path would make the gap look
# better than it is. The MILP instance uses a longer slot (dt = 150 s) and a coarser lattice
# (220 m) than the main grid so that one decision epoch reaches across the region.
''')
rep('''        sub = simulate(env, algo, "static", 5000 + u["seed"], rounds=40,
                       ckpt_every=20, resume=False)''',
    '''        sub = simulate(env, algo, "static", u["seed"], rounds=40,
                       ckpt_every=20, resume=False, init_pos=start)''')
rep('''def grid_equilibrium() -> List[dict]:
    units = []
    for algo in ["FogField-LLL", "FogField-BR", "GossipGreedy", "Consensus-ADMM"]:
        for seed in GRID_SEEDS[:3]:
            u = dict(study="equilibrium", testbed="MEL", algo=algo, n_nodes=40, seed=seed)
            u["_key"] = unit_key(**u)
            units.append(u)
    return units


def run_equilibrium_unit(u: dict) -> dict:
    env = make_env(TESTBEDS["MEL"], u["n_nodes"], R_comm=500.0)
    sub = simulate(env, u["algo"], "static", 7000 + u["seed"], rounds=80,
                   ckpt_every=20, resume=False, return_state=True)
    pos = np.asarray(sub["final_pos"])
    prev = np.asarray(sub["final_prev"])
    active = np.asarray(sub["final_alive"], dtype=bool)
    w = env.testbed.weights(0)
    gz = sub["_gossip"]

    ng = nash_gap(env, pos, prev, active, w)
    lg = locality_gap(env, pos, prev, active, w, gz)
    rec = {"study": "equilibrium", "algo": u["algo"], "seed": u["seed"],
           "n_nodes": u["n_nodes"]}''',
    '''def grid_equilibrium() -> List[dict]:
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

    ng = nash_gap(env, pos, prev, active, w)
    lg = locality_gap(env, pos, prev, active, w, gz, n_sample=16)
    rec = {"study": "equilibrium", "testbed": u["testbed"], "algo": u["algo"],
           "seed": u["seed"], "n_nodes": u["n_nodes"], "phi_final": sub["phi_final"],
           "R_comm": env.R_comm}''')

rep('''run_study("equilibrium", grid_equilibrium(), run_equilibrium_unit)
''', '''run_study("equilibrium", grid_equilibrium(), run_equilibrium_unit)

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
''')

# ------------------------------------------------------------------ 17 - analysis additions
rep('''def plot_recovery():
    if MAIN.empty:
        return
    sub = MAIN[MAIN.scenario.isin(["catastrophic", "seed_loss", "random_failures"])]
    sub = sub.dropna(subset=["recovery_rounds"])
    if sub.empty:
        print("[skip] no recovery data")
        return
    piv = sub.pivot_table(index="algo", columns="scenario", values="recovery_rounds",
                          aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    piv.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("rounds to regain 95% of pre-event $\\\\Phi$")
    ax.set_xlabel(""); ax.set_title("Recovery after disruption")
    plt.xticks(rotation=25, ha="right")
    ax.legend(fontsize=7)
    fig.savefig(DIRS["figs"] / "fig6_recovery.png", bbox_inches="tight")
    plt.show()
    save_table(piv.round(2), "table4_recovery",
               "Mean rounds to recover 95% of pre-event potential.")
    return piv
''', '''def plot_recovery():
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
    axes[0].set_ylabel("rounds to regain 95% of pre-event $\\\\Phi$")
    fig.savefig(DIRS["figs"] / "fig6_recovery.png", bbox_inches="tight")
    plt.show()
    save_table(piv.round(2), "table4_recovery",
               "Mean rounds to recover 95% of pre-event potential, per testbed "
               "(censored units count the remaining horizon; see table4b).")
    save_table(cens.round(3), "table4b_recovery_censored",
               "Share of units that never regained 95% of pre-event potential.")
    print(piv.round(1).to_string())
    print("\\ncensored share:\\n" + cens.round(2).to_string())
    return piv
''')

rep('''OVH = report_overhead()
''', '''OVH = report_overhead()

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
''')

# ------------------------------------------------------------------ 18 - manifest
rep('''    "counts": {n: len(Ledger(DIRS["results"], n))
               for n in ("main", "scalability", "milp", "equilibrium")},''',
    '''    "counts": {n: len(Ledger(DIRS["results"], n))
               for n in ("main", "scalability", "milp", "equilibrium", "lll_schedule", "viewbound")},
    "seeding": ("v2: common random numbers; deployment/scenario/adversary streams seeded by "
                "(testbed, n, seed), algorithm stream by (testbed, n, seed, algorithm); "
                "SHA-256-derived seeds, independent of PYTHONHASHSEED"),
    "lll_schedules": LLL_SCHEDULES,''')

# ------------------------------------------------------------------ write
assert src != orig
SRC.write_text(src, encoding="utf-8")
print("applied {} replacement(s) -> {}".format(n_applied, SRC))
