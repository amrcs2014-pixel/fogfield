"""Close the loop: re-run FogField units with the message losses measured in ns-3.

Gossip methods (FogField-BR, FogField-LLL, GossipGreedy): each push is delivered with
probability q_push, its pull (only if the push arrived) with q_pull, and each beacon update of a
neighbour within R_comm with q_beacon. Loss draws come from their own random stream
(_stream_seed("loss", testbed, n, seed)), so the algorithm's own randomness is untouched and every
run stays paired with the lossless run of the same unit in the v2 ledger. With all q = 1 the run
reproduces the ledger exactly (checked by the `check` units).

Centralized reference over the mesh (D-POA-style, "mesh" variant): each slot the controller sits
in the largest connected component of the communication graph. Nodes outside it cannot receive a
command and hold position; nodes inside receive their command with probability q_cmd (end to end,
measured), otherwise they hold position.

    python loss_study.py                 # all units in loss_rates.json + sweeps
    python loss_study.py --check         # q = 1 reproduction check only
"""
import json
import multiprocessing as mp
import os
import sys
import time
import types
from pathlib import Path

EMU = Path(__file__).resolve().parent
# the simulator's folder: code/ in the public repository, build/ in the project; FOGFIELD_CODE overrides
BUILD = Path(os.environ.get("FOGFIELD_CODE") or next(
    (p for p in (EMU.parent / "code", EMU.parent / "build") if (p / "fogfield_src_fullrun.py").exists()),
    EMU.parent / "code"))
OUT = EMU / "loss_results"
RATES = EMU / "loss_rates.json"

GOSSIP_ALGOS = ["FogField-BR", "FogField-LLL", "GossipGreedy"]
_G = None
LOSS = {"q_push": 1.0, "q_pull": 1.0, "q_beacon": 1.0, "q_cmd": 1.0, "mesh": False, "rng": None}


def _beacon_ok(lr, i, j, rnd):
    """Is j's beacon heard by i in slot rnd? i.i.d. with probability q_beacon, or, if ge_L is set,
    a two-state Gilbert-Elliott chain per directed link j -> i that advances once per slot, with
    stationary loss 1 - q_beacon and mean bad-state duration ge_L slots. With ge_link the same chain
    also gates the push and the pull on that directed link (a link outage that blocks unicast frames
    and their retries as well)."""
    L = LOSS
    if not L.get("ge_L"):
        return lr.random() < L["q_beacon"]
    p = 1.0 - L["q_beacon"]
    b = 1.0 / L["ge_L"]                       # P(bad -> good) per slot
    a = p * b / max(1e-12, 1.0 - p)            # P(good -> bad), so that the stationary loss is p
    lam = 1.0 - a - b
    st = L.setdefault("ge_state", {})
    prev = st.get((j, i))
    if prev is None:
        bad = lr.random() < p                  # first use of the link: stationary draw
    else:
        was_bad, t_prev = prev
        pb = p + ((1.0 if was_bad else 0.0) - p) * lam ** max(rnd - t_prev, 0)
        bad = lr.random() < pb
    st[(j, i)] = (bad, rnd)
    return not bad


def _round_lossy_factory(np):
    def round_lossy(self, rnd, pos, alive, battery, demand_hint, rng, pos_report=None):
        # GossipLayer.round (v2) with losses drawn from LOSS["rng"]; the algorithm stream `rng`
        # is consumed exactly as in the lossless protocol
        L = LOSS
        lr = L["rng"]
        if pos_report is None:
            pos_report = pos
        if L.get("log") is not None:              # positions for a collection-mode ns-3 replay
            live = np.flatnonzero(alive)
            L["log"].append({"rnd": int(rnd), "ids": [int(i) for i in live],
                             "pos": [(float(pos[i][0]), float(pos[i][1])) for i in live], "ex": []})
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
                msgs += 1
                nbytes += len(push)
                link_ok = _beacon_ok(lr, j, int(i), int(rnd)) if L.get("ge_link") else True
                if lr.random() >= L["q_push"] or not link_ok:
                    continue                      # push lost: j learns nothing, sends no pull
                self._absorb(j, push, rnd)
                pull = self.encode(j, pos_report[j], battery[j], demand_hint[j],
                                   [int(q) for q in nbrs[j]])
                msgs += 1
                nbytes += len(pull)
                link_ok = _beacon_ok(lr, int(i), j, int(rnd)) if L.get("ge_link") else True
                if lr.random() < L["q_pull"] and link_ok:
                    self._absorb(i, pull, rnd)
        for i in np.flatnonzero(alive):
            cand = nbrs[i]
            if self.max_view is not None and cand.size > self.max_view:
                d2 = ((pos[cand] - pos[i][None, :]) ** 2).sum(1)
                cand = cand[np.argsort(d2)[: self.max_view]]
            v = self.views[i]
            for j in cand:
                j = int(j)
                if _beacon_ok(lr, int(i), j, int(rnd)):
                    v.pos[j] = pos_report[j].copy()
                    v.last_seen[j] = rnd
        for i in range(self.n):
            self.views[i].forget_older_than(rnd, self.ttl)
        self.msgs_total += msgs
        self.bytes_total += nbytes
        self.per_round.append((msgs, nbytes))
        return nbrs
    return round_lossy


def _aware_step_factory(g):
    """DPOAStyleSwarm.step with reachability-aware planning: nodes the controller cannot reach
    keep their current cell in every particle, so the swarm plans only the nodes it can command
    (the objective still counts every node). Otherwise identical to the original step."""
    np = g["np"]
    _apply_move = g["_apply_move"]

    def aware_step(self, env, ctx, frozen):
        def clamp(arr):
            arr[:, frozen, :] = ctx.pos[frozen][None, :, :]
        w_last = getattr(self, "_w_last", None)
        if w_last is not None and (w_last.shape != ctx.w.shape or not np.array_equal(w_last, ctx.w)):
            self.swarm = None
        self._w_last = np.array(ctx.w, copy=True)
        if self.swarm is None or self.swarm.shape[1] != ctx.pos.shape[0]:
            self._init_swarm(env, ctx.pos, ctx.rng)
            clamp(self.swarm)
            clamp(self.pbest)
        elif self.pbest_f is not None and np.isfinite(self.pbest_f).any():
            clamp(self.pbest)
            for p in range(self.n_particles):
                cand = _apply_move(env, ctx.pos, self.pbest[p], ctx.alive)
                self.pbest_f[p] = env.phi(cand, ctx.pos, ctx.alive, ctx.w)
        rng = ctx.rng
        for _ in range(self.inner):
            clamp(self.swarm)
            self.vel[:, frozen, :] = 0.0
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
            self.vel[:, frozen, :] = 0.0
            self.swarm = self.swarm + self.vel
        gbest = self.pbest[int(np.argmax(self.pbest_f))]
        return _apply_move(env, ctx.pos, gbest, ctx.alive)
    return aware_step


def _mesh_step_factory(g, orig_step):
    np = g["np"]
    _apply_move = g["_apply_move"]
    aware_step = _aware_step_factory(g)

    def step(self, env, ctx):
        if not LOSS["mesh"]:
            return orig_step(self, env, ctx)
        alive = np.flatnonzero(ctx.alive)
        if alive.size == 0:
            return orig_step(self, env, ctx)
        P = ctx.pos[alive]
        adj = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1) <= env.R_comm ** 2
        comp = -np.ones(alive.size, dtype=int)
        best, best_size, c = -1, 0, 0
        for s in range(alive.size):
            if comp[s] >= 0:
                continue
            stack = [s]
            comp[s] = c
            size = 0
            while stack:
                u = stack.pop()
                size += 1
                for w in np.flatnonzero(adj[u] & (comp < 0)):
                    comp[w] = c
                    stack.append(w)
            if size > best_size:
                best, best_size = c, size
            c += 1
        reach = comp == best
        if LOSS.get("aware"):
            frozen = np.zeros(len(ctx.pos), dtype=bool)
            frozen[alive[~reach]] = True
            new_pos = aware_step(self, env, ctx, frozen)
        else:
            new_pos = orig_step(self, env, ctx)
        got = reach & (LOSS["rng"].random(alive.size) < LOSS["q_cmd"])
        hold = alive[~got]
        new_pos = new_pos.copy()
        new_pos[hold] = ctx.pos[hold]
        if LOSS.get("fallback") and reach.any() and (~reach).any():
            # a stranded node runs a local fallback rule: fly toward the nearest node of the
            # controller's component, one slot's reach at a time, until it rejoins
            stranded = alive[~reach]
            C = ctx.pos[alive[reach]]
            target = new_pos.copy()
            for i in stranded:
                target[i] = C[int(np.argmin(((C - ctx.pos[i]) ** 2).sum(1)))]
            moved = _apply_move(env, ctx.pos, target, ctx.alive)
            new_pos[stranded] = moved[stranded]
        LOSS["reach_sum"] = LOSS.get("reach_sum", 0) + int(reach.sum())
        LOSS["alive_sum"] = LOSS.get("alive_sum", 0) + int(alive.size)
        if "reach_first" not in LOSS:
            LOSS["reach_first"] = float(reach.mean())
        return new_pos
    return step


def _gg_act_factory(np, orig_step, p_act=0.35, inertia=0.15):
    """GossipGreedy with FogField-BR's update schedule: each node moves only if activated
    (probability p_act) and not held by inertia; draws come from LOSS["act_rng"]."""
    def step(self, env, ctx):
        new_pos = orig_step(self, env, ctx)
        if not LOSS.get("gg_act"):
            return new_pos
        ar = LOSS["act_rng"]
        new_pos = new_pos.copy()
        for i in np.flatnonzero(ctx.alive):
            if ar.random() > p_act or ar.random() < inertia:
                new_pos[i] = ctx.pos[i]
        return new_pos
    return step


def load_sim():
    global _G
    if _G is not None:
        return _G
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: None
    os.environ["FOGFIELD_BUDGET_MIN"] = "0"
    os.environ["FOGFIELD_STOP_AFTER_EXPERIMENTS"] = "1"
    os.environ["FOGFIELD_STUDIES"] = "none"
    os.chdir(BUILD)
    src = BUILD / "fogfield_src_fullrun.py"
    mod = types.ModuleType("fogfield_sim")
    mod.__file__ = str(src)
    sys.modules["fogfield_sim"] = mod
    g = mod.__dict__
    try:
        exec(compile(src.read_text(encoding="utf-8"), str(src), "exec"), g)
    except SystemExit:
        pass
    g["GossipLayer"].round = _round_lossy_factory(g["np"])
    cls = g["DPOAStyleSwarm"]
    cls.step = _mesh_step_factory(g, cls.step)
    gg = g["GossipGreedy"]
    gg.step = _gg_act_factory(g["np"], gg.step)
    _G = g
    return g


def unit_name(u):
    return "{tag}_{testbed}_{algo}_n{n}_s{seed}.json".format(**u)


def units(check_only=False, measured=True):
    out = []
    if check_only:
        for algo in ("FogField-BR", "D-POA-style"):
            out.append(dict(tag="check", testbed="MEL", algo=algo, n=20, seed=0, q_push=1.0,
                            q_pull=1.0, q_beacon=1.0, q_cmd=1.0, mesh=False))
        return out
    rates = json.loads(RATES.read_text()) if measured else {}
    for tb in ("MEL", "BEJ"):
        for n in (20, 50):
            for seed in range(5):
                if measured:
                    for algo in GOSSIP_ALGOS:
                        r = rates["{}|{}|{}".format(tb, n, algo)]
                        out.append(dict(tag="measured", testbed=tb, algo=algo, n=n, seed=seed,
                                        q_push=r["q_push"], q_pull=r["q_pull"],
                                        q_beacon=r["q_beacon"], q_cmd=1.0, mesh=False))
                    r = rates["{}|{}|D-POA-style".format(tb, n)]
                    out.append(dict(tag="measured", testbed=tb, algo="D-POA-style", n=n, seed=seed,
                                    q_push=1.0, q_pull=1.0, q_beacon=1.0, q_cmd=r["q_cmd"],
                                    mesh=True))
                out.append(dict(tag="meshonly", testbed=tb, algo="D-POA-style", n=n, seed=seed,
                                q_push=1.0, q_pull=1.0, q_beacon=1.0, q_cmd=1.0, mesh=True))
    for tb in ("MEL", "BEJ"):
        for seed in range(5):
            for algo in ("FogField-BR", "GossipGreedy"):
                for loss in (0.1, 0.2, 0.4):
                    q = 1.0 - loss
                    out.append(dict(tag="sweep{:02d}".format(int(round(100 * loss))), testbed=tb,
                                    algo=algo, n=50, seed=seed, q_push=q, q_pull=q, q_beacon=q,
                                    q_cmd=1.0, mesh=False))
    return out


def units_v2():
    """Second set of units (--v2): controls for the reference over the mesh, correlated
    beacon loss for FogField-BR, GossipGreedy with FogField-BR's update schedule, and logged
    re-runs of the reference's mesh-only units (for a collection-mode ns-3 replay)."""
    rates = json.loads(RATES.read_text())
    base = dict(q_push=1.0, q_pull=1.0, q_beacon=1.0, q_cmd=1.0)
    out = []
    for n in (20, 50):
        for seed in range(5):
            for tag, extra in (("fallback", dict(fallback=True)), ("aware", dict(aware=True)),
                               ("awarefb", dict(aware=True, fallback=True)), ("meshlog", dict(log=True))):
                out.append(dict(tag=tag, testbed="BEJ", algo="D-POA-style", n=n, seed=seed,
                                mesh=True, **base, **extra))
    for tb in ("MEL", "BEJ"):
        for n in (20, 50):
            for seed in range(5):
                r = rates["{}|{}|FogField-BR".format(tb, n)]
                out.append(dict(tag="ge5", testbed=tb, algo="FogField-BR", n=n, seed=seed, mesh=False,
                                q_push=r["q_push"], q_pull=r["q_pull"], q_beacon=r["q_beacon"],
                                q_cmd=1.0, ge_L=5))
                # link outages: the same chain also blocks the push and the pull on the link
                out.append(dict(tag="ge5link", testbed=tb, algo="FogField-BR", n=n, seed=seed, mesh=False,
                                q_push=r["q_push"], q_pull=r["q_pull"], q_beacon=r["q_beacon"],
                                q_cmd=1.0, ge_L=5, ge_link=True))
                out.append(dict(tag="ggact", testbed=tb, algo="GossipGreedy", n=n, seed=seed,
                                mesh=False, gg_act=True, **base))
    return out


def write_trace(u, env, rec, slots):
    """Same text format as export_traffic.py; collection mode reads only the positions."""
    tdir = EMU / "traces"
    tdir.mkdir(exist_ok=True)
    path = tdir / "meshonly_{}_static_{}_n{}_s{}.trace".format(u["testbed"], u["algo"], u["n"], u["seed"])
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="ascii", newline="\n") as fh:
        fh.write("FOGFIELD_TRACE 1\n")
        meta = dict(study="meshonly", testbed=u["testbed"], scenario="static", algo=u["algo"], n=u["n"],
                    seed=u["seed"], rounds=150, R_comm=env.R_comm, R_cov=env.R_cov, dt_s=env.dt_s,
                    phi_final=rec["phi_final"], payload_B_per_node_slot=rec["gossip_bytes_per_node_round"])
        fh.write("META " + " ".join("{} {}".format(k, v) for k, v in meta.items()) + "\n")
        for s in slots:
            fh.write("SLOT {} {} 0\n".format(s["rnd"], len(s["ids"])))
            for i, (x, y) in zip(s["ids"], s["pos"]):
                fh.write("{} {:.3f} {:.3f}\n".format(i, x, y))
        fh.write("END\n")
    os.replace(tmp, path)


def run_one(u):
    path = OUT / unit_name(u)
    if path.exists():
        return {"unit": path.name, "status": "exists"}
    g = load_sim()
    np = g["np"]
    LOSS.clear()
    LOSS.update(q_push=u["q_push"], q_pull=u["q_pull"], q_beacon=u["q_beacon"], q_cmd=u["q_cmd"],
                mesh=u["mesh"], fallback=u.get("fallback", False), aware=u.get("aware", False),
                ge_L=u.get("ge_L"), ge_link=u.get("ge_link", False), gg_act=u.get("gg_act", False),
                log=[] if u.get("log") else None,
                rng=np.random.default_rng(g["_stream_seed"]("loss", u["testbed"], u["n"], u["seed"])),
                act_rng=np.random.default_rng(g["_stream_seed"]("activation", u["testbed"], u["n"], u["seed"])))
    env = g["make_env"](g["TESTBEDS"][u["testbed"]], u["n"])
    t0 = time.time()
    rec = g["simulate"](env, u["algo"], "static", u["seed"], 150, ckpt_every=10 ** 9, resume=False)
    out = dict(u)
    out.update({k: rec[k] for k in ("phi_final", "coverage_frac", "giant_frac", "energy_per_node_J",
                                    "gossip_bytes_per_node_round", "gossip_msgs_per_round",
                                    "view_size_mean")})
    if u["mesh"]:
        out["reach_frac"] = LOSS.get("reach_sum", 0) / max(LOSS.get("alive_sum", 1), 1)
        out["reach_first"] = LOSS.get("reach_first")
    if u.get("log"):
        write_trace(u, env, rec, LOSS["log"])
    out["sec"] = round(time.time() - t0, 1)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, path)
    return {"unit": path.name, "status": "ok", "phi": out["phi_final"], "sec": out["sec"]}


def main():
    check = "--check" in sys.argv
    measured = "--no-measured" not in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    allu = units_v2() if "--v2" in sys.argv else units(check, measured)
    todo = [u for u in allu if not (OUT / unit_name(u)).exists()]
    todo.sort(key=lambda u: -(u["n"] * (4 if u["algo"] == "D-POA-style" else 1)))
    print("units:", len(todo), flush=True)
    workers = 1 if check else int(os.environ.get("LOSS_WORKERS", "4"))
    if workers == 1:
        results = map(run_one, todo)
    else:
        results = mp.get_context("spawn").Pool(workers).imap_unordered(run_one, todo)
    for r in results:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    main()
