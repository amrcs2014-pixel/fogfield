"""Fourth one-off patch (15 Oct 2026): make the MILP an exact upper bound. Do NOT re-run it.

v1.0's MILP chose each node's cell from at most 90 cells subsampled from the union of the nodes'
40 nearest reachable cells, while the heuristics moved on the full lattice; in v1.0 FogField-BR,
FogField-LLL and D-POA-style each exceeded the "bound" on one or two of the 25 instances (gap down
to -2.2 %). The load-dependent compute energy was also only approximated (evaluated at the MILP's
own solution). v2:

  * every lattice cell is a candidate for every node (108 cells on MEL at 220 m), with the same
    capped flight-energy function the scoring uses, so every configuration a heuristic can reach is
    MILP-feasible;
  * the efficiency study is scored on gamma = coverage - flight energy - IDLE compute energy, the
    objective of the coverage-energy sub-game in Theorem T3; the load-proportional compute term
    (at most 12 W of a ~140 W budget) is excluded from it, so the MILP optimum is an exact upper
    bound on gamma over the whole lattice;
  * the full reduced objective (with load-dependent compute energy) is still recorded.
"""
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


rep('''def phi_reduced(env: Environment, pos, prev, active, w) -> float:
    """Coverage minus energy only - the objective the MILP bounds."""
    cov = env.coverage(pos, active, w)
    ene = env.energy(pos, prev, active, w)
    N = float(env.n_nodes)
    return float(env.weights.coverage * cov / max(env.W_total, 1e-9)
                 - env.weights.energy * ene / (N * env.E_ref))
''', '''def phi_reduced(env: Environment, pos, prev, active, w) -> float:
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
''')

rep('''    # candidate cells: the union of the nodes' reachable sets, capped for tractability
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
''', '''    # candidate cells (v2): EVERY lattice cell, for every node. Heuristic configurations live
    # on the same lattice, so each of them is MILP-feasible and the optimum is an exact upper
    # bound on the sub-game objective. v1.0 subsampled at most `lattice_cap` cells from the
    # nodes' nearest reachable sets, and heuristics occasionally beat that restricted optimum.
    gx, gy = np.meshgrid(env.lat_x, env.lat_y, indexing="ij")
    cells = np.stack([gx.ravel(), gy.ravel()], axis=1)
    C = len(cells)
    # flight energy is charged with the same capped function the scoring uses, so cells
    # beyond one slot's reach are allowed at the capped price, exactly as in phi_subgame
    reach = np.ones((N, C), dtype=bool)
''')

rep('''    cover_term = pulp.lpSum(env.weights.coverage * float(w[m]) / Wt * z[m] for m in z)
    # compute energy is a constant offset here (every node is active), so only flight matters
    energy_term = pulp.lpSum(env.weights.energy * float(e_cost[i, c]) / denom * y[(i, c)]
                             for (i, c) in y)
    prob += cover_term - energy_term''', '''    cover_term = pulp.lpSum(env.weights.coverage * float(w[m]) / Wt * z[m] for m in z)
    # sub-game objective: compute energy at its idle level is a constant (every node active)
    energy_term = pulp.lpSum(env.weights.energy * float(e_cost[i, c]) / denom * y[(i, c)]
                             for (i, c) in y)
    prob += cover_term - energy_term''')

rep('''    active = np.ones(N, dtype=bool)
    obj = pulp.value(prob.objective)
    # add back the constant compute-energy offset so it is comparable to phi_reduced
    e_cpu_const = float(np.sum(env.uav.compute_energy(
        env.load_fraction(pos, active, w), env.dt_s)))
    obj_comparable = (float(obj) if obj is not None else float("nan")) \\
        - env.weights.energy * e_cpu_const / denom

    return {
        "status": status_name,
        "objective_bound": obj_comparable,
        "objective_achieved": phi_reduced(env, pos, prev, active, w),''', '''    active = np.ones(N, dtype=bool)
    obj = pulp.value(prob.objective)
    # subtract the constant idle compute energy so it is on the scale of phi_subgame
    e_idle = float(N * env.uav.P_cpu_idle * env.dt_s)
    obj_comparable = (float(obj) if obj is not None else float("nan")) \\
        - env.weights.energy * e_idle / denom

    return {
        "status": status_name,
        "objective_bound": obj_comparable,
        "objective_achieved": phi_subgame(env, pos, prev, active, w),
        "objective_achieved_reduced": phi_reduced(env, pos, prev, active, w),''')

rep('''           "milp_achieved": milp["objective_achieved"], "milp_cells": milp["n_cells"],''',
    '''           "milp_achieved": milp["objective_achieved"],
           "milp_achieved_reduced": milp["objective_achieved_reduced"],
           "milp_cells": milp["n_cells"], "objective": "subgame-v2",''')

rep('''        final = np.asarray(sub["final_pos"])
        rec["phi_red_" + algo] = phi_reduced(env, final, start, active, w)
        rec["conn_" + algo] = env.connectivity(final, active) / env.n_nodes
        if bound is not None and np.isfinite(bound) and abs(bound) > 1e-12:
            rec["gap_" + algo] = float((bound - rec["phi_red_" + algo]) / abs(bound))''',
    '''        final = np.asarray(sub["final_pos"])
        rec["phi_sub_" + algo] = phi_subgame(env, final, start, active, w)
        rec["phi_red_" + algo] = phi_reduced(env, final, start, active, w)
        rec["conn_" + algo] = env.connectivity(final, active) / env.n_nodes
        rec["cov_" + algo] = env.coverage(final, active, w) / max(float(np.sum(w)), 1e-9)
        if bound is not None and np.isfinite(bound) and abs(bound) > 1e-12:
            rec["gap_" + algo] = float((bound - rec["phi_sub_" + algo]) / abs(bound))''')

assert src != orig
SRC.write_text(src, encoding="utf-8")
print("applied {} replacement(s) -> {}".format(n_applied, SRC))
