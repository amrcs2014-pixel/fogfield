# %% [markdown]
# ## 18 - Run manifest
#
# A single JSON capturing what produced these numbers: dataset provenance, parameters, and
# the verification results. Keep it beside the figures - it is what makes the run
# reconstructible months later, and reviewers increasingly ask for exactly this.

# %%
MANIFEST = {
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "quick_mode": QUICK,
    "datasets": {name: tb.meta for name, tb in TESTBEDS.items()},
    "testbed_summary": {name: tb.describe() for name, tb in TESTBEDS.items()},
    "uav_model": {
        "reference": "Zeng, Xu & Zhang, IEEE TWC 18(4):2329-2345, 2019, "
                     "DOI 10.1109/TWC.2019.2902559",
        "P0_W": UAV.P0, "Pi_W": UAV.Pi, "U_tip_mps": UAV.U_tip, "v0_mps": UAV.v0,
        "d0": UAV.d0, "rho": UAV.rho, "s": UAV.s, "A_m2": UAV.A,
        "V_max_mps": UAV.V_max, "battery_Wh": UAV.battery_Wh,
        "P_hover_W": UAV.P_hover,
    },
    "weights": {"coverage": Weights().coverage, "connectivity": Weights().connectivity,
                "energy": Weights().energy,
                "note": "weighted scalarisation; one point on the trade-off surface, "
                        "not a Pareto front"},
    "grid": {"nodes": GRID_NODES, "seeds": GRID_SEEDS, "rounds": GRID_ROUNDS,
             "scale_nodes": SCALE_NODES, "milp_nodes": MILP_NODES},
    "verification": {
        "exact_potential_uncapacitated": _res,
        "exact_potential_capacitated": _res_cap,
        "checkpoint_resume_selftest": RESUME_TEST,
    },
    "counts": {n: len(Ledger(DIRS["results"], n))
               for n in ("main", "scalability", "milp", "equilibrium")},
    "scenarios": {k: v.description for k, v in SCENARIOS.items()},
    "algorithms": {k: {"decentralized": v.decentralized, "uses_gossip": v.uses_gossip}
                   for k, v in ALGORITHMS.items()},
}
write_json(ROOT / "run_manifest.json", MANIFEST)
print(json.dumps({k: MANIFEST[k] for k in
                  ("generated_utc", "quick_mode", "counts", "verification")},
                 indent=1, default=_json_default))
print("\nmanifest -> " + str(ROOT / "run_manifest.json"))

# %% [markdown]
# ## 19 - How to resume after a disconnect
#
# Free Colab sessions are reclaimed without warning. The recovery procedure is:
#
# 1. Reopen the notebook and **Run all**.
# 2. Setup, Drive mount and dataset loading complete in seconds - the datasets are cached on
#    Drive as `.npz`, so nothing is re-downloaded.
# 3. Each `run_study` cell consults the ledger, skips every finished unit, and continues with
#    the rest. A unit that was interrupted mid-run restarts from its last round-level
#    checkpoint, with the RNG state restored, not from round zero.
# 4. The analysis cells read the ledgers, so partial results plot fine; they simply get
#    better as more units land.
#
# To **start over deliberately**, delete the relevant folder under the Drive root:
# `results/` to rerun experiments, `ckpt/` to drop partial runs, `cache/` to force a fresh
# download. The cell below does this on request - it is guarded, because on Drive this is
# not undoable.

# %%
def reset_state(what: str = "none", confirm: bool = False) -> None:
    """Delete stored state. `what` is one of: none, ckpt, results, cache, all."""
    import shutil
    targets = {"ckpt": [DIRS["ckpt"]], "results": [DIRS["results"]],
               "cache": [DIRS["cache"]],
               "all": [DIRS["ckpt"], DIRS["results"], DIRS["cache"],
                       DIRS["figs"], DIRS["tables"]]}
    if what == "none":
        print("nothing to do. Call reset_state('ckpt'|'results'|'cache'|'all', confirm=True)")
        return
    if what not in targets:
        raise ValueError("unknown target: " + what)
    if not confirm:
        print("This would permanently delete, from Google Drive:")
        for p in targets[what]:
            n = len(list(p.rglob("*"))) if p.exists() else 0
            print("  {}  ({} file(s))".format(p, n))
        print("Re-run with confirm=True if that is what you want.")
        return
    for p in targets[what]:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
    print("reset '{}' done.".format(what))


reset_state("none")

# %% [markdown]
# ## 20 - Limitations, stated plainly
#
# The plan's own risk register is that overclaiming is what gets this paper rejected. These
# are the limits of what this notebook establishes, and none of them should be softened in
# the write-up.
#
# 1. **No measured wireless cost.** Byte and message counts here are application-layer
#    payload from the real protocol. 802.11 MAC framing, ACKs, retransmission, contention
#    and loss are absent. The plan's R3 requires a Mininet-WiFi/ns-3 tier and a 10-20 node
#    Raspberry-Pi/Jetson testbed; **neither runs in Colab**, and the sim-to-real gap stays
#    unquantified until they do. Report these numbers as *protocol payload*, never as
#    "measured communication overhead" without that qualifier.
# 2. **No claim of a provable global-optimum approximation.** What is established is the
#    exact-potential property (verified numerically), convergence to a pure equilibrium
#    (measured as the Nash gap), and an empirical gap to an exact bound. The only worst-case
#    guarantee is `NE >= 1/2 optimum` for the uncapacitated, valid-utility variant
#    (Vetta 2002). The capacitated variant is outside that structure and the bound is **not**
#    claimed for it.
# 3. **The MILP is a relaxation.** Connectivity is not constrained in it, so it is an upper
#    bound and the reported gap is conservative rather than exact. Instances are small
#    (4-8 nodes) because the exact problem does not scale; do not extrapolate the gap to
#    1000-node deployments.
# 4. **Baselines are reimplementations.** D-POA-style, Guerrero-style and the ADMM baseline
#    are written from published method descriptions, not the authors' code, and are not
#    calibrated to reproduce their published numbers. The MARL baseline is independent
#    Q-learning, **not** MAPPO; the plan already notes that no verified 2024-2026
#    peer-reviewed MARL-for-node-*placement* exemplar exists, and this notebook does not
#    close that gap.
# 5. **Weighted scalarisation only.** One weight vector gives one point on the trade-off
#    surface. No Pareto front is computed and **no hypervolume is reported** - the plan is
#    explicit that promising hypervolume without delivering it is a rejection trigger.
# 6. **Synthetic dynamics on MEL.** The EUA snapshot is static, so `mobility` on MEL is a
#    travelling hotspot. Every affected record carries `synthetic_dynamics=True`. Real
#    temporal structure comes only from BEJ.
# 7. **GeoLife pooling.** BEJ demand is pooled across days onto a time-of-day clock, so it
#    is a diurnal demand surface, not one day's simultaneous population. It also covers
#    182 volunteers in Beijing, which is not an IoT deployment; it is a defensible *proxy*
#    for spatio-temporal demand and should be described as one.
# 8. **Discretised action sets.** The potential-game formulation needs a finite strategy set,
#    so nodes move on a lattice. Optimality in continuous space is not addressed.
# 9. **Dynamic regret is not proven.** Re-triggered dynamics on topology events are
#    implemented and recovery is measured, but the equilibrium-tracking argument stays a
#    discussion point, exactly as the plan's T4 advises.
#
# ### Dataset note - why Milano is absent
#
# The Telecom Italia *Milano* grid (Harvard Dataverse, DOI `10.7910/DVN/EGZHFV`, ODbL) is the
# obvious spatio-temporal demand dataset for this problem, and the plan's scenario design
# echoes it. It is **not** used here because its files require a **Dataverse guestbook
# response** before download; the access API returns
# `"You may not download this file without the required Guestbook response"` to an
# unauthenticated request, so it cannot be fetched by a notebook that is meant to run
# unattended. To use it anyway: accept the guestbook in a browser, place a day file such as
# `sms-call-internet-mi-2013-11-01.txt` into the Drive `cache/` folder, and add a loader that
# bins `(square_id, time_interval, internet_activity)` the way `load_geolife` bins GeoLife.
# The square grid is 100x100 cells of 235 m, so cell `id` maps to metres as
# `x = ((id - 1) % 100) * 235`, `y = ((id - 1) // 100) * 235`.
#
# ### Next steps toward submission
#
# 1. Run with `QUICK = False` to fill the full grid, including 800-node scalability.
# 2. Stand up the Mininet-WiFi and Pi/Jetson tiers; feed them the byte and message traces
#    from `table6_overhead` and report the sim-to-real deviation.
# 3. If the FogField gap in `table2_optimality_gap` exceeds ~25%, switch the framing to a
#    practical decentralized scheme with measured overhead and target IoT-J or FGCS rather
#    than leading with theory at TNSM.

# %%
print("FogField notebook complete.\n")
print("Artifacts on Drive:")
for _k in ("figs", "tables", "results", "ckpt", "cache"):
    _p = DIRS[_k]
    _n = len(list(_p.rglob("*"))) if _p.exists() else 0
    print("  {:9s} {:4d} file(s)   {}".format(_k, _n, _p))
print("\n  manifest  " + str(ROOT / "run_manifest.json"))
