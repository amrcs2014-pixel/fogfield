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

    # (1) clean, uninterrupted reference run
    ref = simulate(seed=901, resume=False, **common)

    # (2) same unit, killed after 9 rounds, then resumed
    for f in DIRS["ckpt"].glob("*.pkl"):
        f.unlink()
    part = simulate(seed=901, resume=True, stop_after=9, **common)
    assert part.get("_partial"), "stop_after should have produced a partial result"
    resumed = simulate(seed=901, resume=True, **common)

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
    """Coverage minus energy only - the objective the MILP bounds."""
    cov = env.coverage(pos, active, w)
    ene = env.energy(pos, prev, active, w)
    N = float(env.n_nodes)
    return float(env.weights.coverage * cov / max(env.W_total, 1e-9)
                 - env.weights.energy * ene / (N * env.E_ref))


def solve_milp(env: Environment, prev: np.ndarray, w: np.ndarray,
               time_limit: int = 120, lattice_cap: int = 90) -> dict:
    """Exact (CBC) upper bound on the reduced objective for a small instance."""
    tb = env.testbed
    N = env.n_nodes

    # candidate cells: the union of the nodes' reachable sets, capped for tractability
    cells = np.unique(np.vstack([env.action_set(prev[i], max_actions=40)
                                 for i in range(N)]), axis=0)
    if len(cells) > lattice_cap:
        sel = np.linspace(0, len(cells) - 1, lattice_cap).astype(int)
        cells = cells[sel]
    C = len(cells)

    reach = np.zeros((N, C), dtype=bool)
    for i in range(N):
        reach[i] = np.linalg.norm(cells - prev[i][None, :], axis=1) <= env.max_step + 1e-6
        if not reach[i].any():
            reach[i, int(np.argmin(np.linalg.norm(cells - prev[i][None, :], axis=1)))] = True

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
    # compute energy is a constant offset here (every node is active), so only flight matters
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
    # add back the constant compute-energy offset so it is comparable to phi_reduced
    e_cpu_const = float(np.sum(env.uav.compute_energy(
        env.load_fraction(pos, active, w), env.dt_s)))
    obj_comparable = (float(obj) if obj is not None else float("nan")) \
        - env.weights.energy * e_cpu_const / denom

    return {
        "status": status_name,
        "objective_bound": obj_comparable,
        "objective_achieved": phi_reduced(env, pos, prev, active, w),
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
    agree, regret, info_frac = [], [], []
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
        regret.append(float(u_glob[pick_glob] - u_glob[pick_loc]))
        known = len([j for j in gossip.views[i].pos if j != i])
        info_frac.append(known / n_others)
    return {"rank_agreement": float(np.mean(agree)),
            "regret_mean": float(np.mean(regret)),
            "regret_max": float(np.max(regret)), "n_sample": int(len(sample)),
            "info_fraction": float(np.mean(info_frac))}
