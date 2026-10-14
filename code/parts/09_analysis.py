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
# decentralized scheme with measured overhead. The worst-case guarantee for this structure is
# only `NE >= 1/2 optimum` (Vetta 2002, PoA <= 2), i.e. a 50% gap, and the point of this
# figure is to show that the *realised* gap is far smaller than the bound admits.

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
    ax.axhline(50, ls="--", color="crimson", lw=1.0)
    ax.annotate("worst-case bound admitted by PoA <= 2 (Vetta 2002)", (0.55, 51),
                fontsize=7.5, color="crimson")
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
    if MAIN.empty:
        return
    sub = MAIN[MAIN.scenario.isin(["catastrophic", "seed_loss", "random_failures"])]
    sub = sub.dropna(subset=["recovery_rounds"])
    if sub.empty:
        print("[skip] no recovery data")
        return
    piv = sub.pivot_table(index="algo", columns="scenario", values="recovery_rounds",
                          aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    piv.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("rounds to regain 95% of pre-event $\\Phi$")
    ax.set_xlabel(""); ax.set_title("Recovery after disruption")
    plt.xticks(rotation=25, ha="right")
    ax.legend(fontsize=7)
    fig.savefig(DIRS["figs"] / "fig6_recovery.png", bbox_inches="tight")
    plt.show()
    save_table(piv.round(2), "table4_recovery",
               "Mean rounds to recover 95% of pre-event potential.")
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
            "loc_rank_agreement": "mean", "loc_info_fraction": "mean",
            "loc_regret_mean": "mean", "loc_regret_max": "max"}
    have = {k: v for k, v in cols.items() if k in EQ.columns}
    tbl = EQ.groupby("algo").agg(have).round(5)
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
