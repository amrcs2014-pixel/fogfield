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
