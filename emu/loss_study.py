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


def _round_lossy_factory(np):
    def round_lossy(self, rnd, pos, alive, battery, demand_hint, rng, pos_report=None):
        # GossipLayer.round (v2) with losses drawn from LOSS["rng"]; the algorithm stream `rng`
        # is consumed exactly as in the lossless protocol
        L = LOSS
        lr = L["rng"]
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
                msgs += 1
                nbytes += len(push)
                if lr.random() >= L["q_push"]:
                    continue                      # push lost: j learns nothing, sends no pull
                self._absorb(j, push, rnd)
                pull = self.encode(j, pos_report[j], battery[j], demand_hint[j],
                                   [int(q) for q in nbrs[j]])
                msgs += 1
                nbytes += len(pull)
                if lr.random() < L["q_pull"]:
                    self._absorb(i, pull, rnd)
        for i in np.flatnonzero(alive):
            cand = nbrs[i]
            if self.max_view is not None and cand.size > self.max_view:
                d2 = ((pos[cand] - pos[i][None, :]) ** 2).sum(1)
                cand = cand[np.argsort(d2)[: self.max_view]]
            v = self.views[i]
            for j in cand:
                j = int(j)
                if lr.random() < L["q_beacon"]:
                    v.pos[j] = pos_report[j].copy()
                    v.last_seen[j] = rnd
        for i in range(self.n):
            self.views[i].forget_older_than(rnd, self.ttl)
        self.msgs_total += msgs
        self.bytes_total += nbytes
        self.per_round.append((msgs, nbytes))
        return nbrs
    return round_lossy


def _mesh_step_factory(np, orig_step):
    def step(self, env, ctx):
        new_pos = orig_step(self, env, ctx)
        if not LOSS["mesh"]:
            return new_pos
        alive = np.flatnonzero(ctx.alive)
        if alive.size == 0:
            return new_pos
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
        got = reach & (LOSS["rng"].random(alive.size) < LOSS["q_cmd"])
        hold = alive[~got]
        new_pos = new_pos.copy()
        new_pos[hold] = ctx.pos[hold]
        LOSS["reach_sum"] = LOSS.get("reach_sum", 0) + int(reach.sum())
        LOSS["alive_sum"] = LOSS.get("alive_sum", 0) + int(alive.size)
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
    cls.step = _mesh_step_factory(g["np"], cls.step)
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


def run_one(u):
    path = OUT / unit_name(u)
    if path.exists():
        return {"unit": path.name, "status": "exists"}
    g = load_sim()
    np = g["np"]
    LOSS.clear()
    LOSS.update(q_push=u["q_push"], q_pull=u["q_pull"], q_beacon=u["q_beacon"], q_cmd=u["q_cmd"],
                mesh=u["mesh"],
                rng=np.random.default_rng(g["_stream_seed"]("loss", u["testbed"], u["n"], u["seed"])))
    env = g["make_env"](g["TESTBEDS"][u["testbed"]], u["n"])
    t0 = time.time()
    rec = g["simulate"](env, u["algo"], "static", u["seed"], 150, ckpt_every=10 ** 9, resume=False)
    out = dict(u)
    out.update({k: rec[k] for k in ("phi_final", "coverage_frac", "giant_frac", "energy_per_node_J",
                                    "gossip_bytes_per_node_round", "gossip_msgs_per_round",
                                    "view_size_mean")})
    if u["mesh"]:
        out["reach_frac"] = LOSS.get("reach_sum", 0) / max(LOSS.get("alive_sum", 1), 1)
    out["sec"] = round(time.time() - t0, 1)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    os.replace(tmp, path)
    return {"unit": path.name, "status": "ok", "phi": out["phi_final"], "sec": out["sec"]}


def main():
    check = "--check" in sys.argv
    measured = "--no-measured" not in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [u for u in units(check, measured) if not (OUT / unit_name(u)).exists()]
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
