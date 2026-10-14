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
TIME_BUDGET_MIN = 45 if QUICK else 180

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
    ledger = Ledger(DIRS["results"], name)
    todo = [u for u in units if not ledger.is_done(u["_key"])]
    print("[{}] {} unit(s) total, {} already done, {} to run"
          .format(name, len(units), len(units) - len(todo), len(todo)))
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
# **per-node** gossip cost stays roughly flat, because a node only ever talks to neighbours.
# The centralized reference is included at the sizes where it is still affordable.

# %%
def grid_scalability() -> List[dict]:
    units = []
    algos = ["FogField-LLL", "FogField-BR", "GossipGreedy", "Consensus-ADMM", "Static"]
    tb_name = "BEJ" if "BEJ" in TESTBEDS else "MEL"
    for n in SCALE_NODES:
        for algo in algos + (["D-POA-style"] if n <= 100 else []):
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
# evaluated against the **common starting configuration**. The heuristic's cumulative flight
# path is not charged against it, because the MILP pays for exactly one move; charging the
# path would make the gap look better than it is.

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
           "milp_achieved": milp["objective_achieved"], "milp_cells": milp["n_cells"],
           "milp_binaries": milp["n_binaries"],
           "milp_connectivity": milp["connectivity_of_milp_solution"]}

    active = np.ones(env.n_nodes, dtype=bool)
    bound = milp["objective_bound"]
    for algo in ["FogField-LLL", "FogField-BR", "GossipGreedy", "Guerrero-style",
                 "Consensus-ADMM", "MARL-IQL", "D-POA-style"]:
        sub = simulate(env, algo, "static", 5000 + u["seed"], rounds=40,
                       ckpt_every=20, resume=False)
        # score the final configuration against the COMMON start, so the heuristic is
        # charged for one deployment move exactly as the MILP is
        final = np.asarray(sub["final_pos"])
        rec["phi_red_" + algo] = phi_reduced(env, final, start, active, w)
        rec["conn_" + algo] = env.connectivity(final, active) / env.n_nodes
        if bound is not None and np.isfinite(bound) and abs(bound) > 1e-12:
            rec["gap_" + algo] = float((bound - rec["phi_red_" + algo]) / abs(bound))
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
           "n_nodes": u["n_nodes"]}
    rec.update({"nash_" + k: v for k, v in ng.items()})
    rec.update({"loc_" + k: v for k, v in lg.items()})
    return rec


run_study("equilibrium", grid_equilibrium(), run_equilibrium_unit)
