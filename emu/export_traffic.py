"""Export per-slot node positions and gossip exchanges from FogField runs for the ns-3 tier.

Each selected unit of the v2 grid is re-simulated with a logging copy of GossipLayer.round.
The copy performs exactly the same operations, in the same order, on the same random
generator, and only records what it does. Every re-run is checked against the ledger: its
phi_final must equal the ledger value bit for bit, or the trace is rejected.

    python export_traffic.py            # all units, 4 worker processes
    python export_traffic.py --quick    # one small unit, for a smoke test

Trace format (text, one file per unit, read by ns3/fogfield-emu.cc):
    FOGFIELD_TRACE 1
    META <key> <value> ...
    SLOT <rnd> <n_alive> <n_exchanges>
    <node id> <x m> <y m>                      n_alive lines
    <i> <j> <push bytes> <pull bytes>          n_exchanges lines, in protocol order
    END
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
OUT = EMU / "traces"

MAIN_ALGOS = ["FogField-BR", "FogField-LLL", "GossipGreedy", "D-POA-style"]
SCALE_ALGOS = ["FogField-BR", "D-POA-style"]


def units(quick: bool = False):
    if quick:
        return [dict(study="main", testbed="MEL", scenario="static", algo="FogField-BR",
                     n=20, seed=0, rounds=150)]
    out = []
    for tb in ("MEL", "BEJ"):
        for n in (20, 50):
            for seed in range(5):
                for algo in MAIN_ALGOS:
                    out.append(dict(study="main", testbed=tb, scenario="static", algo=algo,
                                    n=n, seed=seed, rounds=150))
    for n in (50, 100, 200, 400, 800):
        for seed in range(3):
            for algo in SCALE_ALGOS:
                out.append(dict(study="scalability", testbed="BEJ", scenario="static",
                                algo=algo, n=n, seed=seed, rounds=75))
    return out


def trace_name(u):
    return "{study}_{testbed}_{scenario}_{algo}_n{n}_s{seed}.trace".format(**u)


# ---------------------------------------------------------------------------- simulator
_G = None
_LOG = {"slots": None}


def _round_logged_factory(np):
    def round_logged(self, rnd, pos, alive, battery, demand_hint, rng, pos_report=None):
        # identical to GossipLayer.round in fogfield_src_fullrun.py (v2), plus recording
        if pos_report is None:
            pos_report = pos
        nbrs = self.neighbours(pos, alive)
        msgs = 0
        nbytes = 0
        self.seq[alive] += 1
        live = np.flatnonzero(alive)
        rec = {"rnd": int(rnd), "ids": [int(i) for i in live],
               "pos": [(float(pos[i][0]), float(pos[i][1])) for i in live], "ex": []}

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
                self._absorb(j, push, rnd)
                pull = self.encode(j, pos_report[j], battery[j], demand_hint[j],
                                   [int(q) for q in nbrs[j]])
                msgs += 1
                nbytes += len(pull)
                self._absorb(i, pull, rnd)
                rec["ex"].append((int(i), j, len(push), len(pull)))

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
        if _LOG["slots"] is not None:
            _LOG["slots"].append(rec)
        return nbrs
    return round_logged


def load_sim():
    """Execute the canonical simulator source up to (not including) the analysis section."""
    global _G
    if _G is not None:
        return _G
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: None
    os.environ["FOGFIELD_BUDGET_MIN"] = "0"
    os.environ["FOGFIELD_STOP_AFTER_EXPERIMENTS"] = "1"
    os.environ["FOGFIELD_STUDIES"] = "none"          # run_study skips every study
    os.chdir(BUILD)                                   # the source resolves ./fogfield_out_v2
    src = BUILD / "fogfield_src_fullrun.py"
    mod = types.ModuleType("fogfield_sim")
    mod.__file__ = str(src)
    sys.modules["fogfield_sim"] = mod
    g = mod.__dict__
    try:
        exec(compile(src.read_text(encoding="utf-8"), str(src), "exec"), g)
    except SystemExit:
        pass
    g["GossipLayer"].round = _round_logged_factory(g["np"])
    _G = g
    return g


def ledger_value(u):
    """phi_final of the same unit in the v2 ledger."""
    g = load_sim()
    results = BUILD / "fogfield_out_v2" / "results" / u["study"]
    for f in results.glob("*.json"):
        r = json.loads(f.read_text())
        if (r.get("testbed") == u["testbed"] and r.get("scenario") == u["scenario"]
                and r.get("algo") == u["algo"] and r.get("n_nodes") == u["n"]
                and r.get("seed") == u["seed"]):
            return r
    return None


_INDEX = {}


def ledger_index():
    if not _INDEX:
        for study in ("main", "scalability"):
            for f in (BUILD / "fogfield_out_v2" / "results" / study).glob("*.json"):
                r = json.loads(f.read_text())
                _INDEX[(study, r["testbed"], r["scenario"], r["algo"], r["n_nodes"], r["seed"])] = r
    return _INDEX


def export_one(u):
    path = OUT / trace_name(u)
    if path.exists():
        return {"unit": trace_name(u), "status": "exists"}
    g = load_sim()
    idx = ledger_index()
    ref = idx.get((u["study"], u["testbed"], u["scenario"], u["algo"], u["n"], u["seed"]))
    env = g["make_env"](g["TESTBEDS"][u["testbed"]], u["n"])
    _LOG["slots"] = []
    t0 = time.time()
    rec = g["simulate"](env, u["algo"], u["scenario"], u["seed"], u["rounds"],
                        ckpt_every=10 ** 9, resume=False)
    slots, _LOG["slots"] = _LOG["slots"], None
    same = ref is not None and rec["phi_final"] == ref["phi_final"]
    if not same:
        return {"unit": trace_name(u), "status": "MISMATCH",
                "phi": rec["phi_final"], "ledger": None if ref is None else ref["phi_final"]}
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="ascii", newline="\n") as fh:
        fh.write("FOGFIELD_TRACE 1\n")
        meta = dict(study=u["study"], testbed=u["testbed"], scenario=u["scenario"],
                    algo=u["algo"], n=u["n"], seed=u["seed"], rounds=u["rounds"],
                    R_comm=env.R_comm, R_cov=env.R_cov, dt_s=env.dt_s,
                    phi_final=rec["phi_final"],
                    payload_B_per_node_slot=rec["gossip_bytes_per_node_round"])
        fh.write("META " + " ".join("{} {}".format(k, v) for k, v in meta.items()) + "\n")
        for s in slots:
            fh.write("SLOT {} {} {}\n".format(s["rnd"], len(s["ids"]), len(s["ex"])))
            for i, (x, y) in zip(s["ids"], s["pos"]):
                fh.write("{} {:.3f} {:.3f}\n".format(i, x, y))
            for i, j, a, b in s["ex"]:
                fh.write("{} {} {} {}\n".format(i, j, a, b))
        fh.write("END\n")
    os.replace(tmp, path)
    return {"unit": trace_name(u), "status": "ok", "slots": len(slots),
            "exchanges": sum(len(s["ex"]) for s in slots), "sec": round(time.time() - t0, 1)}


def main():
    quick = "--quick" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [u for u in units(quick) if not (OUT / trace_name(u)).exists()]
    # biggest units first so the pool does not end on one long n = 800 run
    todo.sort(key=lambda u: -(u["n"] * u["rounds"] * (4 if u["algo"] == "D-POA-style" else 1)))
    print("units to export:", len(todo), flush=True)
    workers = 1 if quick else int(os.environ.get("EXPORT_WORKERS", "4"))
    log = EMU / "export_traffic.log"
    if workers == 1:
        results = map(export_one, todo)
    else:
        pool = mp.get_context("spawn").Pool(workers)
        results = pool.imap_unordered(export_one, todo)
    bad = 0
    with open(log, "a", encoding="utf-8") as fh:
        for r in results:
            line = json.dumps(r)
            print(line, flush=True)
            fh.write(line + "\n")
            bad += r["status"] == "MISMATCH"
    print("mismatches:", bad)


if __name__ == "__main__":
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    main()
