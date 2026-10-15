"""Second one-off patch (15 Oct 2026) on top of revise_v2_patch.py, applied before the v2 grid ran.
Do NOT re-run it on the patched file.

  * `late_surge` scenario: the control twin of `controller_loss` (same surge, controller alive);
    with common random numbers every decentralised method must produce identical traces in the
    two scenarios, and for D-POA-style the paired difference is the cost of losing the controller
  * per-round coverage trace, per-node peak energy (referee m6)
  * locality gap: top-3 agreement next to the strict argmax agreement (author question 2)
  * MILP study: Static (= the common start) and RandomWalk rows (m10)
  * self-test uses a private checkpoint folder, so parallel workers never delete each other's
    checkpoints; FOGFIELD_STUDIES restricts a process to some studies
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


# ---- late_surge scenario
rep('''# | `controller_loss` | the centralised controller dies at the midpoint; a demand surge follows |
''', '''# | `controller_loss` | the centralised controller dies at the midpoint; a demand surge follows |
# | `late_surge` | the same demand surge with nobody dying: the control twin of `controller_loss` |
''')
rep('''    "controller_loss": Scenario("controller_loss",
                                "centralised controller dies at the midpoint; demand surge follows"),
}
''', '''    "controller_loss": Scenario("controller_loss",
                                "centralised controller dies at the midpoint; demand surge follows"),
    "late_surge": Scenario("late_surge",
                           "demand surge ten rounds after the midpoint (control twin of controller_loss)"),
}
''')
rep('''        if scenario == "controller_loss":
            # the surge starts ten rounds after the controller is lost
            lo = rounds // 2 + 10''',
    '''        if scenario in ("controller_loss", "late_surge"):
            # the surge starts ten rounds after the (possible) controller loss
            lo = rounds // 2 + 10''')
rep('''        if self.key in ("flash_crowd", "controller_loss"):''',
    '''        if self.key in ("flash_crowd", "controller_loss", "late_surge"):''')

# ---- per-round coverage trace and per-node peak energy
rep('''        "trace_phi": [round(float(v), 6) for v in df["phi"].tolist()],
        "final_pos": pos.tolist(),''',
    '''        "trace_phi": [round(float(v), 6) for v in df["phi"].tolist()],
        "trace_cov": [round(float(v), 5) for v in df["coverage_frac"].tolist()],
        "energy_node_max_J": float(np.max(env.uav.battery_Wh * 3600.0 - battery)),
        "final_pos": pos.tolist(),''')

# ---- locality gap: top-3 agreement
rep('''        agree.append(pick_loc == pick_glob)
        regret.append(float(u_glob[pick_glob] - u_glob[pick_loc]))''',
    '''        agree.append(pick_loc == pick_glob)
        top3.append(pick_loc in set(np.argsort(-u_glob)[:3].tolist()))
        regret.append(float(u_glob[pick_glob] - u_glob[pick_loc]))''')
rep('''    agree, regret, info_frac = [], [], []''', '''    agree, regret, info_frac, top3 = [], [], [], []''')
rep('''    return {"rank_agreement": float(np.mean(agree)),
            "regret_mean": float(np.mean(regret)),''',
    '''    return {"rank_agreement": float(np.mean(agree)),
            "top3_agreement": float(np.mean(top3)),
            "regret_mean": float(np.mean(regret)),''')
rep('''            "loc_rank_agreement": "mean", "loc_info_fraction": "mean",''',
    '''            "loc_rank_agreement": "mean", "loc_top3_agreement": "mean",
            "loc_info_fraction": "mean",''')

# ---- MILP null rows
rep('''    for algo in ["FogField-LLL", "FogField-BR", "GossipGreedy", "Guerrero-style",
                 "Consensus-ADMM", "MARL-IQL", "D-POA-style"]:''',
    '''    for algo in ["FogField-LLL", "FogField-BR", "GossipGreedy", "Guerrero-style",
                 "Consensus-ADMM", "MARL-IQL", "D-POA-style", "Static", "RandomWalk"]:''')

# ---- self-test in a private checkpoint folder
rep('''    env = Environment(TESTBEDS["MEL"], n_nodes=8, lattice_m=150.0)
    common = dict(env=env, algo="FogField-LLL", scenario="static", rounds=24, ckpt_every=4)

    # (1) clean, uninterrupted reference run
    ref = simulate(seed=901, resume=False, **common)

    # (2) same unit, killed after 9 rounds, then resumed
    for f in DIRS["ckpt"].glob("*.pkl"):
        f.unlink()
    part = simulate(seed=901, resume=True, stop_after=9, **common)
    assert part.get("_partial"), "stop_after should have produced a partial result"
    resumed = simulate(seed=901, resume=True, **common)
''', '''    env = Environment(TESTBEDS["MEL"], n_nodes=8, lattice_m=150.0)
    common = dict(env=env, algo="FogField-LLL", scenario="static", rounds=24, ckpt_every=4)

    # A private checkpoint folder: parallel grid workers each run this self-test at start-up,
    # and must neither race on the same checkpoint file nor delete each other's checkpoints.
    import shutil
    shared_ckpt = DIRS["ckpt"]
    DIRS["ckpt"] = shared_ckpt / "selftest_{}".format(os.getpid())
    DIRS["ckpt"].mkdir(parents=True, exist_ok=True)
    try:
        # (1) clean, uninterrupted reference run
        ref = simulate(seed=901, resume=False, **common)

        # (2) same unit, killed after 9 rounds, then resumed
        part = simulate(seed=901, resume=True, stop_after=9, **common)
        assert part.get("_partial"), "stop_after should have produced a partial result"
        resumed = simulate(seed=901, resume=True, **common)
    finally:
        shutil.rmtree(DIRS["ckpt"], ignore_errors=True)
        DIRS["ckpt"] = shared_ckpt
''')

# ---- FOGFIELD_STUDIES
rep('''    ledger = Ledger(DIRS["results"], name)
    todo = [u for u in units if not ledger.is_done(u["_key"])]
    # Headless parallel runs:''',
    '''    only = os.environ.get("FOGFIELD_STUDIES")
    if only and name not in only.split(","):
        print("[{}] skipped (FOGFIELD_STUDIES={})".format(name, only))
        return
    ledger = Ledger(DIRS["results"], name)
    todo = [u for u in units if not ledger.is_done(u["_key"])]
    # Headless parallel runs:''')

assert src != orig
SRC.write_text(src, encoding="utf-8")
print("applied {} replacement(s) -> {}".format(n_applied, SRC))
