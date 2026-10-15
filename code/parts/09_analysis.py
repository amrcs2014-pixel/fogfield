# %% [markdown]
# ## 17 - Results
#
# Everything below reads from the Drive ledgers, so it works in a **fresh session** with no
# re-computation: run the setup cells, skip the experiment cells, and the analysis renders
# from what is already stored. Figures are written to `figs/` and tables to `tables/` in
# both CSV and LaTeX.

# %%
def load_study(name: str) -> pd.DataFrame:
    df = Ledger(DIRS["results"], name).load_all()
    if df.empty:
        print("[{}] no results yet".format(name))
    return df


MAIN = load_study("main")
SCALE = load_study("scalability")
MILPDF = load_study("milp")
EQ = load_study("equilibrium")

for _n, _d in [("main", MAIN), ("scalability", SCALE), ("milp", MILPDF), ("equilibrium", EQ)]:
    print("{:12s} {:4d} rows".format(_n, len(_d)))


def save_table(df: pd.DataFrame, stem: str, caption: str = "", float_fmt: str = "%.3f"):
    df.to_csv(DIRS["tables"] / (stem + ".csv"), index=True)
    try:
        tex = df.to_latex(float_format=float_fmt, escape=True, caption=caption,
                          label="tab:" + stem)
    except TypeError:
        tex = df.to_latex(float_format=float_fmt)
    atomic_write(DIRS["tables"] / (stem + ".tex"), tex.encode("utf-8"))
    return df


# %% [markdown]
# ### Table 1 - main grid

# %%
if not MAIN.empty:
    cols = ["phi_final", "coverage_frac", "giant_frac", "energy_per_node_J",
            "convergence_round", "gossip_bytes_per_node_round"]
    t1 = (MAIN.groupby(["testbed", "scenario", "algo"])[cols]
          .mean().round(4).sort_index())
    save_table(t1, "table1_main_grid",
               "Mean outcome per testbed, scenario and algorithm.")
    display_cols = ["phi_final", "coverage_frac", "giant_frac", "gossip_bytes_per_node_round"]
    print(t1[display_cols].head(40).to_string())
else:
    t1 = pd.DataFrame()

# %% [markdown]
# ### Figure - convergence and the effect of annealed log-linear learning
#
# The `static` traces show the finite-improvement behaviour the potential structure predicts:
# monotone-ish climb to a plateau. `FogField-LLL` should plateau **higher** than
# `FogField-BR` despite being noisier early, because the annealed chain concentrates on
# potential maximisers instead of stopping at the first equilibrium it reaches.

# %%
def plot_convergence(testbed: str = "MEL", scenario: str = "static", seed: int = 0):
    if MAIN.empty:
        return
    sub = MAIN[(MAIN.testbed == testbed) & (MAIN.scenario == scenario) &
               (MAIN.seed == seed)]
    if sub.empty:
        print("[skip] no convergence data for", testbed, scenario)
        return
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for _, row in sub.sort_values("algo").iterrows():
        tr = row.get("trace_phi")
        if not isinstance(tr, list) or not tr:
            continue
        style = "-" if row["decentralized"] else "--"
        lw = 2.0 if str(row["algo"]).startswith("FogField") else 1.0
        ax.plot(tr, style, lw=lw, label=row["algo"], alpha=0.9)
    ax.set_xlabel("round"); ax.set_ylabel(r"global potential $\Phi$")
    ax.set_title("Convergence - {} / {} (solid = decentralized)".format(testbed, scenario))
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(DIRS["figs"] / "fig2_convergence_{}_{}.png".format(testbed, scenario),
                bbox_inches="tight")
    plt.show()


plot_convergence("MEL", "static")
plot_convergence("MEL", "catastrophic")

# %% [markdown]
# ### Figure - outcome by scenario

# %%
def plot_scenarios(testbed: str = "MEL", metric: str = "phi_final"):
    if MAIN.empty:
        return
    sub = MAIN[MAIN.testbed == testbed]
    if sub.empty:
        return
    piv = sub.pivot_table(index="scenario", columns="algo", values=metric, aggfunc="mean")
    order = [a for a in ALGORITHMS if a in piv.columns]
    piv = piv[order]
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    piv.plot(kind="bar", ax=ax, width=0.82)
    ax.set_ylabel(metric); ax.set_xlabel("")
    ax.set_title("{} by scenario - {}".format(metric, testbed))
    ax.legend(fontsize=7, ncol=3, loc="best")
    plt.xticks(rotation=25, ha="right")
    fig.savefig(DIRS["figs"] / "fig3_scenarios_{}_{}.png".format(testbed, metric),
                bbox_inches="tight")
    plt.show()
    return piv


_p = plot_scenarios("MEL", "phi_final")
_ = plot_scenarios("MEL", "coverage_frac")

# %% [markdown]
# ### Figure - optimality gap against the exact bound (the key Q1 plot)
#
# Read this against the plan's threshold: if the gap for FogField is **above about 25%**, the
# theory-first framing is the wrong one and the paper should be pitched as a practical
# decentralized scheme with measured overhead. (v2: the MILP is an exact bound on the
# coverage-energy sub-game objective over the full lattice; no PoA line is drawn, see T3.)

# %%
def analyze_optimality_gap():
    if MILPDF.empty:
        print("[skip] no MILP results")
        return None
    gapcols = [c for c in MILPDF.columns if c.startswith("gap_")]
    if not gapcols:
        print("[skip] MILP produced no comparable bound")
        return None
    g = MILPDF[gapcols].rename(columns=lambda c: c[4:])
    summary = pd.DataFrame({
        "mean_gap_%": (g.mean() * 100).round(2),
        "median_gap_%": (g.median() * 100).round(2),
        "max_gap_%": (g.max() * 100).round(2),
    }).sort_values("mean_gap_%")
    save_table(summary, "table2_optimality_gap",
               "Optimality gap versus the exact CBC upper bound on the reduced objective.")
    print(summary.to_string())

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    data = [g[c].dropna().to_numpy() * 100 for c in summary.index]
    # `labels=` was renamed to `tick_labels` in matplotlib 3.9; setting the ticks
    # afterwards works on every version Colab might ship
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels(list(summary.index))
    # v2: no PoA line - the PoA <= 2 bound holds only for the coverage-energy sub-game (T3),
    # and the configurations scored here are not certified equilibria of it
    ax.set_ylabel("gap to exact bound (%)")
    ax.set_title("Empirical optimality gap - small instances, MEL")
    plt.xticks(rotation=20, ha="right")
    fig.savefig(DIRS["figs"] / "fig4_optimality_gap.png", bbox_inches="tight")
    plt.show()

    statuses = MILPDF["milp_status"].value_counts().to_dict()
    print("\nCBC solver status counts:", statuses)
    if statuses.get("Optimal", 0) < len(MILPDF):
        print("NOTE: not every instance solved to proven optimality; rows that hit the time "
              "limit give a weaker (still valid) bound, so their gap is an over-estimate.")
    return summary


GAP = analyze_optimality_gap()

# %% [markdown]
# ### Figure - scalability and measured communication cost
#
# The decentralization argument lives or dies here. Per-node gossip bytes should stay
# roughly flat in `N` (a node only ever talks to neighbours), while a centralized scheme must
# pull global state every round.

# %%
def plot_scalability():
    if SCALE.empty:
        print("[skip] no scalability results")
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for algo, sub in SCALE.groupby("algo"):
        agg = sub.groupby("n_nodes").mean(numeric_only=True).sort_index()
        mk = "o-" if str(algo).startswith("FogField") else ".--"
        axes[0].plot(agg.index, agg["phi_final"], mk, label=algo)
        axes[1].plot(agg.index, agg["gossip_bytes_per_node_round"], mk, label=algo)
        axes[2].plot(agg.index, agg["wallclock_s"], mk, label=algo)
    axes[0].set_ylabel(r"$\Phi$ (final)"); axes[0].set_title("solution quality")
    axes[1].set_ylabel("bytes / node / round")
    axes[1].set_title("measured gossip cost per node")
    axes[2].set_ylabel("seconds"); axes[2].set_title("wall-clock per run")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("number of fog nodes")
        ax.legend(fontsize=7)
    fig.savefig(DIRS["figs"] / "fig5_scalability.png", bbox_inches="tight")
    plt.show()

    agg = (SCALE.groupby(["algo", "n_nodes"])
           [["phi_final", "gossip_bytes_per_node_round", "gossip_msgs_per_round",
             "wallclock_s"]].mean().round(3))
    save_table(agg, "table3_scalability",
               "Scalability: quality, measured per-node gossip cost, and wall-clock time.")
    return agg


SCALE_TBL = plot_scalability()

# %% [markdown]
# ### Figure - recovery after disruption
#
# `catastrophic` (50% simultaneous loss) and `seed_loss` are where a decentralized scheme
# should separate from a coordinator-based one, because there is no coordinator to lose.

# %%
def plot_recovery():
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
    axes[0].set_ylabel("rounds to regain 95% of pre-event $\\Phi$")
    fig.savefig(DIRS["figs"] / "fig6_recovery.png", bbox_inches="tight")
    plt.show()
    save_table(piv.round(2), "table4_recovery",
               "Mean rounds to recover 95% of pre-event potential, per testbed "
               "(censored units count the remaining horizon; see table4b).")
    save_table(cens.round(3), "table4b_recovery_censored",
               "Share of units that never regained 95% of pre-event potential.")
    print(piv.round(1).to_string())
    print("\ncensored share:\n" + cens.round(2).to_string())
    return piv


REC = plot_recovery()

# %% [markdown]
# ### Equilibrium and locality diagnostics
#
# * **`nash_epsilon`** and **`nash_epsilon_rel`** - the largest unilateral improvement still
#   available in the *global* potential at the configuration reached, absolute and relative
#   to the potential achieved. An exact pure equilibrium is essentially never hit in a finite
#   run, so `epsilon_rel` is the number to read: it says how far from equilibrium the
#   configuration sits in the units of the objective itself. This is the measurable content
#   of T1.
# * **`loc_rank_agreement`** and **`loc_regret_mean`** - how often the local, gossip-only
#   estimator picks the same move the global marginal contribution would, and what it costs
#   when it does not. **Read these together with `loc_info_fraction`.** If a controller's
#   converged nodes end up within gossip range of nearly everyone else (`info_fraction` near
#   1), agreement near 1.0 and regret near 0.0 follow mechanically from full information, not
#   from any property of the algorithm's local decision rule - see the note in section 16
#   on the n=14/full-mesh artifact this study was rescoped to avoid. Low agreement paired
#   with near-zero regret is not a failure either: agreement is a strict `argmax` match, so
#   near-ties between lattice cells of nearly equal value count as disagreements.

# %%
def report_equilibrium():
    if EQ.empty:
        print("[skip] no equilibrium results")
        return None
    cols = {"nash_epsilon": "mean", "nash_epsilon_rel": "mean",
            "nash_mean_improvement": "mean", "nash_is_eps_nash_1pct": "mean",
            "loc_rank_agreement": "mean", "loc_top3_agreement": "mean",
            "loc_info_fraction": "mean",
            "loc_regret_mean": "mean", "loc_regret_max": "max"}
    have = {k: v for k, v in cols.items() if k in EQ.columns}
    keys = ["testbed", "algo"] if "testbed" in EQ.columns else ["algo"]
    tbl = EQ.groupby(keys).agg(have).round(5)
    save_table(tbl, "table5_equilibrium",
               "Equilibrium quality and the cost of neighbour-only information.")
    print(tbl.to_string())
    return tbl


EQ_TBL = report_equilibrium()

# %% [markdown]
# ### Communication overhead summary (R3)
#
# These are **measured** application-layer payload bytes from the running protocol.
# The caveat from section 8 applies and is repeated in the exported table so it travels with
# the numbers: MAC/PHY framing, retransmissions and contention are not included here.

# %%
def report_overhead():
    if MAIN.empty:
        return None
    sub = MAIN[MAIN.algo.isin([a for a in ALGORITHMS if ALGORITHMS[a].uses_gossip])]
    tbl = (sub.groupby(["testbed", "algo"])
           [["gossip_msgs_per_round", "gossip_bytes_per_round",
             "gossip_bytes_per_node_round", "gossip_rounds"]]
           .mean().round(2))
    save_table(tbl, "table6_overhead",
               "Measured gossip overhead (application-layer payload only; no MAC/PHY).")
    print(tbl.to_string())
    if not sub.empty:
        b = sub["gossip_bytes_per_node_round"].mean()
        print("\nAcross the grid, a node emits about {:.0f} payload bytes per round. "
              "At one round per {:.0f} s that is {:.2f} kbit/s per node,\nwhich is the "
              "number the emulator and testbed tier should be asked to confirm under a real "
              "802.11 stack.".format(b, 30.0, b * 8 / 30.0 / 1000))
    return tbl


OVH = report_overhead()

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

# %% [markdown]
# ### Deployment snapshot
#
# Where the nodes actually end up, against the real device field.

# %%
def plot_final_deployment(testbed: str = "MEL", algo: str = "FogField-LLL",
                          scenario: str = "static"):
    if MAIN.empty:
        return
    sub = MAIN[(MAIN.testbed == testbed) & (MAIN.algo == algo) &
               (MAIN.scenario == scenario)]
    if sub.empty:
        print("[skip] no deployment snapshot for", testbed, algo)
        return
    row = sub.iloc[0]
    pos = np.asarray(row["final_pos"])
    alive = np.asarray(row["final_alive"], dtype=bool)
    tb = TESTBEDS[testbed]
    env = make_env(tb, int(row["n_nodes"]))
    w = tb.weights(0)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.scatter(tb.device_xy[:, 0] / 1000, tb.device_xy[:, 1] / 1000, s=10, c=w,
               cmap="viridis", alpha=0.7, linewidths=0, label="devices")
    P = pos[alive]
    for p in P:
        ax.add_patch(plt.Circle((p[0] / 1000, p[1] / 1000), env.R_cov / 1000,
                                fill=False, ls=":", lw=0.7, color="crimson", alpha=0.6))
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    for a in range(len(P)):
        for b in range(a + 1, len(P)):
            if d2[a, b] <= env.R_comm ** 2:
                ax.plot([P[a, 0] / 1000, P[b, 0] / 1000],
                        [P[a, 1] / 1000, P[b, 1] / 1000],
                        "-", lw=0.6, color="steelblue", alpha=0.6)
    ax.scatter(P[:, 0] / 1000, P[:, 1] / 1000, marker="^", s=60, color="crimson",
               edgecolor="k", linewidths=0.4, zorder=5, label="fog nodes")
    ax.set_aspect("equal")
    ax.set_xlabel("east (km)"); ax.set_ylabel("north (km)")
    ax.set_title("{} - {} / {}\ncoverage {:.1%}, giant component {:.1%}".format(
        testbed, algo, scenario, row["coverage_frac"], row["giant_frac"]))
    ax.legend(fontsize=7, loc="best")
    fig.savefig(DIRS["figs"] / "fig7_deployment_{}_{}.png".format(testbed, algo),
                bbox_inches="tight")
    plt.show()


plot_final_deployment("MEL", "FogField-LLL", "static")
