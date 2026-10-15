"""Every table, figure and number of the revised FogField manuscript, computed from the v2 ledger
(build/fogfield_out_v2/results/*/*.json). Nothing here re-runs an experiment.

    python paper_numbers.py

Writes
  ../paper/latex_source/tab_{gap,bej,scale,eq,rec,ctrl,byz,lll,viewbound}.tex   typeset tables
  ../paper/table*.csv                                                          the same as CSV
  paper_figs_v2/*.pdf|png  (and copies the ones the paper uses into ../paper/latex_source/figs/)
  paper_numbers.json, paper_numbers.md                                         every scalar quoted
"""
import glob
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = HERE / "fogfield_out_v2" / "results"
LATEX = ROOT / "paper" / "latex_source"
PAPER = ROOT / "paper"
if not LATEX.exists():            # running inside the public repository (code/): write next to it
    LATEX = PAPER = HERE / "paper_tables"
    (LATEX / "figs").mkdir(parents=True, exist_ok=True)
FIGS = HERE / "paper_figs_v2"
FIGS.mkdir(exist_ok=True)

ALGOS = ["FogField-BR", "FogField-LLL", "D-POA-style", "GossipGreedy", "Guerrero-style",
         "Consensus-ADMM", "MARL-IQL", "Static", "RandomWalk"]
SHORT = {"FogField-BR": "FogField-BR", "FogField-LLL": "FogField-LLL", "D-POA-style": "D-POA-style",
         "GossipGreedy": "GossipGreedy", "Guerrero-style": "Guerrero", "Consensus-ADMM": "Cons.-ADMM",
         "MARL-IQL": "MARL-IQL", "Static": "Static", "RandomWalk": "RandomWalk"}
TINY = {"FogField-BR": "BR", "FogField-LLL": "LLL", "D-POA-style": "D-POA", "GossipGreedy": "GG",
        "Guerrero-style": "Guer.", "Consensus-ADMM": "ADMM", "MARL-IQL": "MARL", "Static": "Static",
        "RandomWalk": "RW"}
COL = {"FogField-BR": "#1b6ca8", "FogField-LLL": "#2ca02c", "D-POA-style": "#d62728",
       "GossipGreedy": "#ff7f0e", "Guerrero-style": "#9467bd", "Consensus-ADMM": "#8c564b",
       "MARL-IQL": "#7f7f7f", "Static": "#bcbd22", "RandomWalk": "#17becf"}
FOG = ("FogField-BR", "FogField-LLL")
SCEN_BEJ = ["static", "traffic_shift", "flash_crowd", "late_surge", "expansion", "random_failures",
            "catastrophic", "seed_loss", "controller_loss", "byzantine"]
SCEN_LABEL = {"static": "static", "mobility": "mobility", "traffic_shift": "traffic\\_shift",
              "flash_crowd": "flash\\_crowd", "late_surge": "late\\_surge", "expansion": "expansion",
              "random_failures": "random\\_failures", "catastrophic": "catastrophic",
              "seed_loss": "single-node loss", "controller_loss": "controller\\_loss",
              "byzantine": "byzantine"}
NUM = {}          # every scalar the text quotes


def load(study):
    rows = []
    for f in sorted(glob.glob(str(RES / study / "*.json"))):
        with open(f, encoding="utf-8") as fh:
            rows.append(json.load(fh))
    return pd.DataFrame(rows)


def paired(d):
    d = np.asarray(d, dtype=float)
    out = {"n": int(d.size), "mean": float(d.mean()) if d.size else float("nan")}
    if d.size >= 2 and np.std(d) > 0:
        lo, hi = stats.t.interval(0.95, d.size - 1, loc=d.mean(), scale=stats.sem(d))
        try:
            p = float(stats.wilcoxon(d).pvalue)
        except ValueError:
            p = 1.0
    else:
        lo = hi = out["mean"]
        p = 1.0
    out.update(lo=float(lo), hi=float(hi), p=p, wins=int(np.sum(d > 0)), losses=int(np.sum(d < 0)))
    return out


def km_median(times, observed):
    """Kaplan-Meier median of right-censored times; inf when the curve never reaches 0.5."""
    t = np.asarray(times, dtype=float)
    e = np.asarray(observed, dtype=bool)
    S = 1.0
    for x in np.unique(t[e]):
        d = np.sum((t == x) & e)
        r = np.sum(t >= x)
        S *= 1.0 - d / r
        if S <= 0.5:
            return float(x)
    return float("inf")


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return "{:.{}f}".format(x, nd)


def fmt_p(p):
    if p < 0.001:
        return "$<$0.001"
    return "{:.3f}".format(p) if p < 0.1 else "{:.2f}".format(p)


def km_str(med, cens):
    m = "NR" if not np.isfinite(med) else "{:.0f}".format(med)
    return "{} ({:.0f}\\%)".format(m, 100 * cens)


def write(name, text):
    (LATEX / name).write_text(text, encoding="utf-8")
    print("  wrote", name)


def savefig(fig, name):
    fig.savefig(FIGS / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(FIGS / (name + ".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)
    shutil.copy(FIGS / (name + ".pdf"), LATEX / "figs" / (name + ".pdf"))
    if (ROOT / "paper" / "figs").exists():
        shutil.copy(FIGS / (name + ".png"), ROOT / "paper" / "figs" / (name + ".png"))
    print("  figure", name)


# =========================================================================== load
MAIN = load("main")
SCALE = load("scalability")
MILP = load("milp")
EQ = load("equilibrium")
LLL = load("lll_schedule")
VB = load("viewbound")
for nm, d in [("main", MAIN), ("scalability", SCALE), ("milp", MILP), ("equilibrium", EQ),
              ("lll_schedule", LLL), ("viewbound", VB)]:
    NUM["n_" + nm] = int(len(d))
    print("{:13s} {:5d} units".format(nm, len(d)))
NUM["n_total"] = int(sum(NUM["n_" + k] for k in ["main", "scalability", "milp", "equilibrium",
                                                  "lll_schedule", "viewbound"]))
NUM["n_failed"] = int(sum(int(d.get("failed", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
                          for d in [MAIN, SCALE, MILP, EQ, LLL, VB] if not d.empty))
KEY = ["testbed", "n_nodes", "seed"]


def cell(tb, scen, algo):
    return MAIN[(MAIN.testbed == tb) & (MAIN.scenario == scen) & (MAIN.algo == algo)]


def series(tb, scen, algo, col="phi_final"):
    return cell(tb, scen, algo).set_index(["n_nodes", "seed"])[col].sort_index()


# =========================================================================== null checks (CRN)
def same_traces(a, b):
    if len(a) == 0 or len(a) != len(b):
        return None
    a = a.set_index(["n_nodes", "seed"]).sort_index()
    b = b.set_index(["n_nodes", "seed"]).sort_index()
    if not a.index.equals(b.index):
        return None
    return bool(all(x == y for x, y in zip(a["trace_phi"], b["trace_phi"])))


checks = {}
for tb in ["MEL", "BEJ"]:
    for algo in ["Static", "RandomWalk"]:
        checks["byz_eq_static_{}_{}".format(tb, algo)] = same_traces(cell(tb, "byzantine", algo),
                                                                    cell(tb, "static", algo))
    for algo in [a for a in ALGOS if a != "D-POA-style"]:
        checks["ctrl_eq_surge_{}_{}".format(tb, algo)] = same_traces(cell(tb, "controller_loss", algo),
                                                                    cell(tb, "late_surge", algo))
for tb in ["MEL", "BEJ"]:
    for pair in (("ctrl_fail", "late_fail"), ("ctrl_drift", "late_drift")):
        for algo in [a for a in ALGOS if a != "D-POA-style"]:
            checks["{}_eq_{}_{}_{}".format(pair[0], pair[1], tb, algo)] = same_traces(
                cell(tb, pair[0], algo), cell(tb, pair[1], algo))
for algo in ALGOS:
    checks["bej_static_eq_mobility_" + algo] = same_traces(cell("BEJ", "static", algo),
                                                         cell("BEJ", "mobility", algo))
    checks["mel_static_eq_traffic_" + algo] = same_traces(cell("MEL", "static", algo),
                                                        cell("MEL", "traffic_shift", algo))
NUM["null_checks"] = checks
vals = [v for v in checks.values() if v is not None]
NUM["null_checks_all_true"] = bool(vals and all(vals))
NUM["null_checks_evaluated"] = len(vals)
print("null checks: {} evaluated, all identical: {}".format(len(vals), NUM["null_checks_all_true"]))

# LLL 'base' schedule == main-grid FogField-LLL static (cross-process reproducibility)
if not LLL.empty and not MAIN.empty:
    base = LLL[LLL.variant == "base"]
    ref = MAIN[(MAIN.scenario == "static") & (MAIN.algo == "FogField-LLL")]
    m = base.merge(ref, on=KEY, suffixes=("_s", "_m"))
    NUM["lll_base_reproduces_main"] = {"pairs": int(len(m)),
                                       "identical": int(sum(a == b for a, b in zip(m.trace_phi_s, m.trace_phi_m)))}
    print("LLL base == main LLL static:", NUM["lll_base_reproduces_main"])


# =========================================================================== optimality gap (MILP)
def do_gap():
    if MILP.empty:
        return
    gcols = [c for c in MILP.columns if c.startswith("gap_")]
    g = MILP[gcols].rename(columns=lambda c: c[4:]) * 100
    conn = MILP[[c for c in MILP.columns if c.startswith("conn_")]].rename(columns=lambda c: c[5:])
    summ = pd.DataFrame({"mean": g.mean(), "median": g.median(), "max": g.max(), "min": g.min(),
                         "conn": conn.mean()}).sort_values("mean")
    summ.to_csv(PAPER / "table2_optimality_gap.csv")
    NUM["gap"] = {a: {k: float(v) for k, v in summ.loc[a].items()} for a in summ.index}
    NUM["milp_status"] = MILP.milp_status.value_counts().to_dict()
    NUM["milp_cells"] = [int(MILP.milp_cells.min()), int(MILP.milp_cells.max())]
    NUM["milp_min_gap_any"] = float(g.min().min())
    if "FogField-BR" in g and "D-POA-style" in g:
        d = (g["FogField-BR"] - g["D-POA-style"]).to_numpy()
        t = stats.ttest_rel(g["FogField-BR"], g["D-POA-style"])
        pr = paired(d)
        NUM["gap_br_vs_dpoa"] = {"mean_diff_pp": pr["mean"], "ci": [pr["lo"], pr["hi"]], "t": float(t.statistic),
                                 "t_p": float(t.pvalue), "wilcoxon_p": pr["p"],
                                 "br_better": int(np.sum(d < 0)), "dpoa_better": int(np.sum(d > 0)),
                                 "ties": int(np.sum(np.abs(d) < 1e-9))}
        for a in ["FogField-BR", "D-POA-style", "FogField-LLL"]:
            x = g[a].to_numpy()
            NUM["gap_disp_" + a] = {"sd": float(np.std(x, ddof=1)), "iqr": float(np.subtract(*np.percentile(x, [75, 25]))),
                                    "p90": float(np.percentile(x, 90)), "n_optimal": int(np.sum(x < 1e-6))}
        NUM["gap_levene_br_dpoa"] = float(stats.levene(g["FogField-BR"], g["D-POA-style"]).pvalue)
        dl = (g["FogField-LLL"] - g["FogField-BR"]).to_numpy()
        NUM["gap_lll_vs_br"] = paired(dl)
    lines = [r"\begin{table}[t]",
             r"\caption{Optimality gap of the sub-game objective $\gamma$~\eqref{eq:gamma} to the exact",
             r"MILP optimum over the full lattice (25 MEL instances, $n = 4$--$8$). Lower is better.",
             r"Every method starts from the MILP's starting configuration. Conn.: share of nodes in the",
             r"largest component, which $\gamma$ does not reward.}",
             r"\label{tab:gap}", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Method & Mean (\%) & Median (\%) & Max (\%) & Conn. \\", r"\midrule"]
    best = summ["mean"].idxmin()
    for a in summ.index:
        nm = SHORT.get(a, a) + (r" \emph{(centralized)}" if a == "D-POA-style" else "")
        r = summ.loc[a]
        cells = [fmt(r["mean"], 2), fmt(r["median"], 2), fmt(r["max"], 2), fmt(r["conn"], 2)]
        if a == best:
            cells[0] = r"\textbf{" + cells[0] + "}"
        lines.append("{} & {} \\\\".format(nm, " & ".join(cells)))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("tab_gap.tex", "\n".join(lines))

    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    order = list(summ.index)
    data = [g[a].dropna().to_numpy() for a in order]
    bp = ax.boxplot(data, showmeans=True, widths=0.6, patch_artist=True)
    for patch, name in zip(bp["boxes"], order):
        patch.set_facecolor("#1b6ca8" if name.startswith("FogField") else "#dddddd")
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([SHORT.get(a, a) for a in order], rotation=35, ha="right", fontsize=6.8)
    ax.set_ylabel("gap to exact optimum (%)")
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    savefig(fig, "fig_optimality_gap")


# =========================================================================== BEJ scenario table
def do_bej():
    sub = MAIN[MAIN.testbed == "BEJ"]
    if sub.empty:
        return
    piv = sub.pivot_table(index=["scenario", "n_nodes"], columns="algo", values="phi_final", aggfunc="mean")
    piv.to_csv(PAPER / "table1_bej_by_scenario.csv")
    full = MAIN.pivot_table(index=["testbed", "scenario", "n_nodes", "algo"],
                            values=["phi_final", "coverage_frac", "giant_frac", "energy_per_node_J",
                                    "convergence_round", "gossip_bytes_per_node_round"], aggfunc="mean")
    full.to_csv(PAPER / "table1_main_grid.csv")
    rows, res = [], {}
    for scen in SCEN_BEJ:
        if scen not in sub.scenario.unique():
            continue
        means = sub[sub.scenario == scen].groupby("algo").phi_final.mean()
        fog = max([a for a in FOG if a in means], key=lambda a: means[a])
        base = max([a for a in means.index if a not in FOG], key=lambda a: means[a])
        f = series("BEJ", scen, fog)
        b = series("BEJ", scen, base)
        idx = f.index.intersection(b.index)
        rel = ((f[idx] - b[idx]) / b[idx].abs()).to_numpy() * 100
        pr = paired((f[idx] - b[idx]).to_numpy())
        marg = {}
        for n in (20, 50):
            mf = f[idx][f[idx].index.get_level_values(0) == n].mean()
            mb = b[idx][b[idx].index.get_level_values(0) == n].mean()
            marg[n] = 100 * (mf - mb) / abs(mb) if mb else float("nan")
        res[scen] = {"fog": fog, "base": base, "margin20": marg[20], "margin50": marg[50],
                     "margin_pooled": 100 * (f[idx].mean() - b[idx].mean()) / abs(b[idx].mean()),
                     "wins": pr["wins"], "n": pr["n"], "p": pr["p"],
                     "mean_rel_unit": float(rel.mean())}
        m50 = sub[(sub.scenario == scen) & (sub.n_nodes == 50)].groupby("algo").phi_final.mean()
        rows.append((scen, m50, res[scen]))
    NUM["bej"] = res
    sig = [s for s, r in res.items() if r["p"] < 0.05 and r["margin_pooled"] > 0]
    NUM["bej_scenarios_significant"] = sig
    NUM["bej_scenarios_not_significant"] = [s for s in res if s not in sig]
    show = [a for a in ALGOS if a != "RandomWalk"]
    lines = [r"\begin{table*}[t]",
             r"\caption{Final objective $\Phi$ by scenario on BEJ (GeoLife, real diurnal demand) at $n = 50$,",
             r"mean over 5 seeds; best in bold. On BEJ \emph{mobility} is the same run as \emph{static} and is not",
             r"listed. Right: margin of the better FogField variant over the best other method at $n = 20$ and",
             r"$n = 50$ (from unrounded means), and the paired comparison over the ten $(n, \text{seed})$ units",
             r"(wins out of 10, Wilcoxon signed-rank $p$). All $n = 20$ values are in the replication package.}",
             r"\label{tab:bej}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{l" + "r" * len(show) + "rrrr}", r"\toprule",
             "Scenario & " + " & ".join(TINY[a] for a in show) +
             r" & $\Delta_{20}$ & $\Delta_{50}$ & Wins & $p$ \\", r"\midrule"]
    for scen, m50, r in rows:
        best = max([a for a in show if a in m50], key=lambda a: m50[a])
        cells = []
        for a in show:
            v = fmt(m50.get(a, float("nan")), 3)
            cells.append(r"\textbf{" + v + "}" if a == best else v)
        lines.append("{} & {} & {:+.1f}\\% & {:+.1f}\\% & {}/{} & {} \\\\".format(
            SCEN_LABEL[scen], " & ".join(cells), r["margin20"], r["margin50"], r["wins"], r["n"], fmt_p(r["p"])))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    write("tab_bej.tex", "\n".join(lines))

    # figure: two panels, n = 20 and n = 50
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 5.6), sharex=True)
    for ax, n in zip(axes, (20, 50)):
        p = sub[sub.n_nodes == n].pivot_table(index="scenario", columns="algo", values="phi_final", aggfunc="mean")
        p = p.reindex([s for s in SCEN_BEJ if s in p.index])[[a for a in show if a in p.columns]]
        p.index = [SCEN_LABEL[s].replace("\\_", "_") for s in p.index]
        p.plot(kind="bar", ax=ax, width=0.84, color=[COL[a] for a in p.columns], edgecolor="none", legend=(n == 20))
        ax.set_ylabel(r"final $\Phi$, $n$ = {}".format(n))
        ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
    axes[0].legend(fontsize=7.2, ncol=4, loc="upper right", framealpha=0.92)
    plt.setp(axes[1].get_xticklabels(), rotation=18, ha="right")
    savefig(fig, "fig_scenarios_bej")


# =========================================================================== MEL: saturation, paired, convergence
def do_mel():
    sub = MAIN[MAIN.testbed == "MEL"]
    if sub.empty:
        return
    out = {}
    means = sub.groupby("algo").phi_final.mean()
    out["phi_mean_all_scen"] = {a: float(means[a]) for a in means.index}
    st = sub[sub.scenario == "static"]
    out["static_phi"] = st.groupby("algo").phi_final.mean().to_dict()
    out["static_cov"] = st.groupby("algo").coverage_frac.mean().to_dict()
    out["static_conv_round"] = st.groupby("algo").convergence_round.mean().to_dict()
    for a in ["FogField-BR", "FogField-LLL"]:
        x = sub[sub.algo == a].set_index(["scenario", "n_nodes", "seed"]).phi_final
        y = sub[sub.algo == "D-POA-style"].set_index(["scenario", "n_nodes", "seed"]).phi_final
        idx = x.index.intersection(y.index)
        pr = paired((x[idx] - y[idx]).to_numpy())
        pr["rel_pct"] = 100 * pr["mean"] / float(y[idx].mean())
        out["paired_vs_dpoa_" + a] = pr
    # per scenario: does D-POA beat FogField anywhere on MEL?
    sc = sub.pivot_table(index="scenario", columns="algo", values="phi_final", aggfunc="mean")
    out["dpoa_beats_both_fog_in"] = [s for s in sc.index if all(sc.loc[s, "D-POA-style"] > sc.loc[s, f] for f in FOG
                                                               if f in sc.columns)] if "D-POA-style" in sc else []
    NUM["mel"] = out
    sc.to_csv(PAPER / "table1b_mel_by_scenario.csv")


# =========================================================================== byzantine (paired)
def do_byz():
    res = {}
    lines = [r"\begin{table}[t]",
             r"\caption{Paired effect of $15\%$ Byzantine gossip: $\Phi(\text{byzantine}) - \Phi(\text{static})$",
             r"over the ten $(n, \text{seed})$ units of each testbed, as a percentage of the static mean, with a",
             r"$95\%$ interval and the Wilcoxon $p$. Static and RandomWalk ignore gossip and are exactly zero.}",
             r"\label{tab:byz}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{MEL} & \multicolumn{2}{c}{BEJ} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Method & $\Delta$ (\%) [95\% CI] & $p$ & $\Delta$ (\%) [95\% CI] & $p$ \\", r"\midrule"]
    for a in ALGOS:
        cells = []
        for tb in ["MEL", "BEJ"]:
            b = series(tb, "byzantine", a)
            s = series(tb, "static", a)
            idx = b.index.intersection(s.index)
            if len(idx) == 0:
                cells += ["--", "--"]
                continue
            pr = paired((b[idx] - s[idx]).to_numpy())
            sm = float(s[idx].mean())
            pr.update(rel=100 * pr["mean"] / sm, rel_lo=100 * pr["lo"] / sm, rel_hi=100 * pr["hi"] / sm)
            res["{}_{}".format(tb, a)] = pr
            cells += ["{:+.1f} [{:+.1f}, {:+.1f}]".format(pr["rel"], pr["rel_lo"], pr["rel_hi"]),
                      fmt_p(pr["p"]) if pr["n"] > 1 and (pr["hi"] != pr["lo"]) else "--"]
        lines.append("{} & {} \\\\".format(SHORT[a], " & ".join(cells)))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    NUM["byz"] = res
    pd.DataFrame(res).T.to_csv(PAPER / "table7_byzantine_paired.csv")
    write("tab_byz.tex", "\n".join(lines))


# =========================================================================== controller loss
SURGE = (85, 110)


def window_mean(tr, lo, hi):
    tr = np.asarray(tr, dtype=float)
    return float(tr[lo:hi].mean())


def do_ctrl():
    res = {}
    rows = [("D-POA-style", "controller_loss", "D-POA-style, controller lost"),
            ("D-POA-style", "late_surge", "D-POA-style, controller alive"),
            ("FogField-BR", "controller_loss", "FogField-BR"),
            ("FogField-LLL", "controller_loss", "FogField-LLL"),
            ("GossipGreedy", "controller_loss", "GossipGreedy"),
            ("Static", "controller_loss", "Static")]
    table = {}
    for algo, scen, label in rows:
        for tb in ["MEL", "BEJ"]:
            c = cell(tb, scen, algo)
            if c.empty:
                continue
            ph = np.mean([window_mean(t, *SURGE) for t in c.trace_phi])
            cv = np.mean([window_mean(t, *SURGE) for t in c.trace_cov]) if "trace_cov" in c else float("nan")
            pre = float(c.phi_pre_mid.mean())
            table[(label, tb)] = (ph, cv, pre)
    for tb in ["MEL", "BEJ"]:
        dead = cell(tb, "controller_loss", "D-POA-style").set_index(["n_nodes", "seed"]).sort_index()
        alive = cell(tb, "late_surge", "D-POA-style").set_index(["n_nodes", "seed"]).sort_index()
        br = cell(tb, "controller_loss", "FogField-BR").set_index(["n_nodes", "seed"]).sort_index()
        idx = dead.index.intersection(alive.index).intersection(br.index)
        if len(idx) == 0:
            continue
        wd = np.array([window_mean(t, *SURGE) for t in dead.loc[idx, "trace_phi"]])
        wa = np.array([window_mean(t, *SURGE) for t in alive.loc[idx, "trace_phi"]])
        wb = np.array([window_mean(t, *SURGE) for t in br.loc[idx, "trace_phi"]])
        cd = np.array([window_mean(t, *SURGE) for t in dead.loc[idx, "trace_cov"]])
        ca = np.array([window_mean(t, *SURGE) for t in alive.loc[idx, "trace_cov"]])
        pr = paired(wd - wa)
        pr["rel"] = 100 * pr["mean"] / float(wa.mean())
        prc = paired(cd - ca)
        prb = paired(wb - wd)
        prb["rel"] = 100 * prb["mean"] / float(wd.mean())
        prba = paired(wb - wa)
        prba["rel"] = 100 * prba["mean"] / float(wa.mean())
        res[tb] = {"dead_minus_alive": pr, "cov_dead_minus_alive_pp": {k: (v * 100 if k in ("mean", "lo", "hi") else v)
                                                                          for k, v in prc.items()},
                   "br_minus_dead": prb, "br_minus_alive": prba,
                   "uplink_B_per_node_slot": float(alive.loc[idx, "ctrl_bytes_per_node_round"].mean()),
                   "dpoa_gossip_B_per_node_slot": float(alive.loc[idx, "gossip_bytes_per_node_round"].mean()),
                   "br_gossip_B_per_node_slot": float(br.loc[idx, "gossip_bytes_per_node_round"].mean())}
        # per n
        for n in (20, 50):
            sel = [i for i in range(len(idx)) if idx[i][0] == n]
            res[tb]["dead_minus_alive_rel_n{}".format(n)] = float(100 * (wd[sel] - wa[sel]).mean() / wa[sel].mean())
    NUM["ctrl"] = res
    lines = [r"\begin{table}[t]",
             r"\caption{Controller loss. Mean $\Phi$ and coverage during the demand surge (slots 85--109) that",
             r"follows the loss of the centralised controller at slot 75, against the same surge with the",
             r"controller alive (\emph{late\_surge}); 10 units per testbed. For decentralised methods the two",
             r"scenarios are identical runs. Pre: mean $\Phi$ in slots 70--74.}",
             r"\label{tab:ctrl}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{MEL} & \multicolumn{3}{c}{BEJ} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Method & Pre & $\Phi$ & Cov. & Pre & $\Phi$ & Cov. \\", r"\midrule"]
    for algo, scen, label in rows:
        cells = []
        for tb in ["MEL", "BEJ"]:
            v = table.get((label, tb))
            cells += [fmt(v[2]), fmt(v[0]), fmt(v[1], 3)] if v else ["--"] * 3
        lines.append("{} & {} \\\\".format(label, " & ".join(cells)))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("tab_ctrl.tex", "\n".join(lines))
    pd.DataFrame({"{}|{}".format(*k): {"surge_phi": v[0], "surge_cov": v[1], "pre_phi": v[2]}
                  for k, v in table.items()}).T.to_csv(PAPER / "table8_controller_loss.csv")

    # figure: mean traces
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, tb in zip(axes, ["MEL", "BEJ"]):
        for algo, scen, label, ls in [("D-POA-style", "late_surge", "D-POA-style, alive", "--"),
                                      ("D-POA-style", "controller_loss", "D-POA-style, lost", "-"),
                                      ("FogField-BR", "controller_loss", "FogField-BR", "-"),
                                      ("GossipGreedy", "controller_loss", "GossipGreedy", "-"),
                                      ("Static", "controller_loss", "Static", ":")]:
            c = cell(tb, scen, algo)
            if c.empty:
                continue
            tr = np.mean(np.stack([np.asarray(t, dtype=float) for t in c.trace_phi]), axis=0)
            ax.plot(np.arange(len(tr)), tr, ls, lw=1.6 if "FogField" in algo else 1.1,
                    color=COL[algo] if "alive" not in label else "#f4a6a6", label=label)
        ax.axvline(75, color="k", lw=0.7, ls=":")
        ax.axvspan(SURGE[0], SURGE[1], color="#f2e6c9", alpha=0.6, lw=0)
        ax.set_xlim(55, 149); ax.set_xlabel("slot"); ax.set_title(tb, fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"mean $\Phi$ (10 units)")
    axes[1].legend(fontsize=6.3, loc="upper right", framealpha=0.9)
    savefig(fig, "fig_controller_loss")


# =========================================================================== harder controller-loss pairs
PAIRS = [("ctrl_fail", "late_fail", "node failures"),
         ("ctrl_drift", "late_drift", "sweeping hotspot")]
POST = (80, 150)


def do_ctrl2():
    res = {}
    lines = [r"\begin{table}[t]",
             r"\caption{Harder controller loss. Mean $\Phi$ over slots 80--149 after the controller is lost at slot 75,",
             r"with node failures ($1\%$ per slot) or a sweeping demand hotspot continuing after the loss. Lost / Alive:",
             r"the D-POA-style reference without / with its controller (same events). $\Delta_{\mathrm{loss}}$: paired",
             r"Lost$-$Alive over the 10 units, as a percentage of Alive, with Wilcoxon $p$. BR: FogField-BR;",
             r"$\Delta_{\mathrm{BR}}$: paired FogField-BR$-$Lost. Decentralised methods are identical runs in the two",
             r"scenarios.}",
             r"\label{tab:ctrl2}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Testbed & After loss & Lost & Alive & $\Delta_{\mathrm{loss}}$ (\%) & $p$ & BR & $\Delta_{\mathrm{BR}}$ (\%) \\",
             r"\midrule"]
    for tb in ["MEL", "BEJ"]:
        first = True
        for dead_s, alive_s, label in PAIRS:
            dead = cell(tb, dead_s, "D-POA-style").set_index(["n_nodes", "seed"]).sort_index()
            alive = cell(tb, alive_s, "D-POA-style").set_index(["n_nodes", "seed"]).sort_index()
            br = cell(tb, dead_s, "FogField-BR").set_index(["n_nodes", "seed"]).sort_index()
            idx = dead.index.intersection(alive.index).intersection(br.index)
            if len(idx) == 0:
                continue

            def wm(df):
                return np.array([window_mean(t, *POST) for t in df.loc[idx, "trace_phi"]])
            wd, wa, wb = wm(dead), wm(alive), wm(br)
            pl = paired(wd - wa)
            pl["rel"] = 100 * pl["mean"] / float(wa.mean())
            pb = paired(wb - wd)
            pb["rel"] = 100 * pb["mean"] / float(wd.mean())
            cd = np.array([window_mean(t, *POST) for t in dead.loc[idx, "trace_cov"]])
            ca = np.array([window_mean(t, *POST) for t in alive.loc[idx, "trace_cov"]])
            key = "{}_{}".format(tb, dead_s)
            res[key] = {"lost": float(wd.mean()), "alive": float(wa.mean()), "br": float(wb.mean()),
                        "loss": pl, "br_minus_lost": pb, "cov_lost": float(cd.mean()),
                        "cov_alive": float(ca.mean()),
                        "alive_final_dead": float(dead.loc[idx, "alive_final"].mean())}
            for n_ in (20, 50):
                sel = [i for i in range(len(idx)) if idx[i][0] == n_]
                res[key]["loss_rel_n{}".format(n_)] = float(100 * (wd[sel] - wa[sel]).mean() / wa[sel].mean())
            lines.append("{} & {} & {} & {} & {:+.1f} & {} & {} & {:+.1f} \\\\".format(
                tb if first else "", label, fmt(wd.mean()), fmt(wa.mean()), pl["rel"], fmt_p(pl["p"]),
                fmt(wb.mean()), pb["rel"]))
            first = False
        lines.append(r"\addlinespace")
    if lines[-1] == r"\addlinespace":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    NUM["ctrl2"] = res
    write("tab_ctrl2.tex", "\n".join(lines))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
    for r_, (dead_s, alive_s, label) in enumerate(PAIRS):
        for c_, tb in enumerate(["MEL", "BEJ"]):
            ax = axes[r_, c_]
            for algo, scen, lab, ls, col in [
                    ("D-POA-style", alive_s, "D-POA-style, alive", "--", "#f4a6a6"),
                    ("D-POA-style", dead_s, "D-POA-style, lost", "-", COL["D-POA-style"]),
                    ("FogField-BR", dead_s, "FogField-BR", "-", COL["FogField-BR"]),
                    ("GossipGreedy", dead_s, "GossipGreedy", "-", COL["GossipGreedy"])]:
                c = cell(tb, scen, algo)
                if c.empty:
                    continue
                tr = np.mean(np.stack([np.asarray(t, dtype=float) for t in c.trace_phi]), axis=0)
                ax.plot(np.arange(len(tr)), tr, ls, lw=1.6 if algo == "FogField-BR" else 1.1,
                        color=col, label=lab)
            ax.axvline(75, color="k", lw=0.7, ls=":")
            ax.set_xlim(60, 149)
            ax.grid(alpha=0.3)
            ax.set_title("{} - {}".format(tb, label), fontsize=7.5)
            if r_ == 1:
                ax.set_xlabel("slot")
        axes[r_, 0].set_ylabel(r"mean $\Phi$ (10 units)")
    axes[0, 1].legend(fontsize=6.0, loc="upper right", framealpha=0.9)
    savefig(fig, "fig_controller_loss2")


# =========================================================================== recovery (catastrophic), censoring handled
def do_rec():
    res = {}
    lines = [r"\begin{table*}[t]",
             r"\caption{After $50\%$ simultaneous node loss at slot 75 (\emph{catastrophic}), per testbed, 10 units per",
             r"method. Pre / Post: mean $\Phi$ in slots 70--74 / 75--149. Rel.: Kaplan--Meier median slots until $\Phi$",
             r"regains $95\%$ of the method's own pre-event value. Abs.: the same for a common threshold, $95\%$ of",
             r"the median pre-event $\Phi$ over all nine methods in the same $(n, \text{seed})$ unit. In brackets the",
             r"share of units that never reach the threshold before the horizon (censored); NR: median not reached.",
             r"Static never moves, so what it ``recovers'' is the environment.}",
             r"\label{tab:rec}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r" & \multicolumn{4}{c}{MEL} & \multicolumn{4}{c}{BEJ} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
             r"Method & Pre & Post & Rel. & Abs. & Pre & Post & Rel. & Abs. \\", r"\midrule"]
    tab = {}
    for tb in ["MEL", "BEJ"]:
        sub = MAIN[(MAIN.testbed == tb) & (MAIN.scenario == "catastrophic")]
        if sub.empty:
            continue
        # common absolute threshold per (n, seed)
        thr = {}
        for (n, seed), grp in sub.groupby(["n_nodes", "seed"]):
            pre = [np.mean(np.asarray(t, float)[70:75]) for t in grp.trace_phi]
            thr[(n, seed)] = 0.95 * float(np.median(pre))
        for a in ALGOS:
            c = sub[sub.algo == a]
            if c.empty:
                continue
            rt, ro, at, ao = [], [], [], []
            for _, r in c.iterrows():
                ev = [e for e in r.recovery_events if e["event"].startswith("catastrophic")]
                if ev:
                    rt.append(ev[0]["rounds"]); ro.append(ev[0]["recovered"])
                tr = np.asarray(r.trace_phi, float)
                hit = np.flatnonzero(tr[75:] >= thr[(r.n_nodes, r.seed)])
                at.append(int(hit[0]) if hit.size else len(tr) - 75)
                ao.append(bool(hit.size))
            d = {"pre": float(c.phi_pre_mid.mean()), "post": float(c.phi_post_mid.mean()),
                 "rel_km": km_median(rt, ro), "rel_cens": 1 - float(np.mean(ro)),
                 "abs_km": km_median(at, ao), "abs_cens": 1 - float(np.mean(ao))}
            tab[(a, tb)] = d
            res["{}_{}".format(tb, a)] = d
        # correlation across methods: pre-event phi vs share never recovering (own threshold)
        pre = [tab[(a, tb)]["pre"] for a in ALGOS if (a, tb) in tab]
        cen = [tab[(a, tb)]["rel_cens"] for a in ALGOS if (a, tb) in tab]
        if len(pre) > 2 and np.std(cen) > 0:
            rho = stats.spearmanr(pre, cen)
            res[tb + "_spearman_pre_vs_relcens"] = {"rho": float(rho.correlation), "p": float(rho.pvalue)}
    for a in ALGOS:
        cells = []
        for tb in ["MEL", "BEJ"]:
            d = tab.get((a, tb))
            cells += ([fmt(d["pre"]), fmt(d["post"]), km_str(d["rel_km"], d["rel_cens"]),
                       km_str(d["abs_km"], d["abs_cens"])] if d else ["--"] * 4)
        lines.append("{} & {} \\\\".format(SHORT[a], " & ".join(cells)))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    NUM["rec"] = res
    pd.DataFrame({"{}|{}".format(*k): v for k, v in tab.items()}).T.to_csv(PAPER / "table4_recovery.csv")
    write("tab_rec.tex", "\n".join(lines))
    # single-node loss and random failures: paired effect vs static
    out = {}
    for tb in ["MEL", "BEJ"]:
        for scen in ["seed_loss", "random_failures", "catastrophic"]:
            diffs = []
            for a in ALGOS:
                x = series(tb, scen, a)
                s = series(tb, "static", a)
                idx = x.index.intersection(s.index)
                if len(idx):
                    diffs.append(100 * float((x[idx] - s[idx]).mean() / s[idx].mean()))
            if diffs:
                out["{}_{}".format(tb, scen)] = {"min_rel": min(diffs), "max_rel": max(diffs)}
    NUM["events_vs_static"] = out


# =========================================================================== scalability
def do_scale():
    if SCALE.empty:
        return
    agg = SCALE.groupby(["algo", "n_nodes"])[["phi_final", "coverage_frac", "gossip_bytes_per_node_round",
                                               "gossip_msgs_per_round", "wallclock_s"]].mean()
    agg.to_csv(PAPER / "table3_scalability.csv")
    NUM["scale"] = {"{}_{}".format(a, n): {k: float(v) for k, v in agg.loc[(a, n)].items()} for a, n in agg.index}
    NUM["scale_payload_max"] = float(SCALE.gossip_bytes_per_node_round.max())
    NUM["scale_rounds"] = sorted(SCALE.rounds.unique().tolist())
    NUM["scale_seeds"] = int(SCALE.seed.nunique())
    NUM["scale_testbed"] = sorted(SCALE.testbed.unique().tolist())
    for n in (400, 800):
        try:
            x = SCALE[(SCALE.algo == "GossipGreedy") & (SCALE.n_nodes == n)].set_index("seed").phi_final
            y = SCALE[(SCALE.algo == "FogField-BR") & (SCALE.n_nodes == n)].set_index("seed").phi_final
            NUM["scale_gg_minus_br_{}".format(n)] = {"mean": float((x - y).mean()), "gg_wins": int(((x - y) > 0).sum())}
        except Exception:
            pass
    lines = [r"\begin{table}[t]",
             r"\caption{Scalability on BEJ, static scenario, 75 slots, 3 seeds. Payload is \emph{measured}",
             r"application-layer gossip bytes; its ceiling for any method is 270 B per node per slot.}",
             r"\label{tab:scale}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Method & $n$ & $\Phi$ & Cov. & B/node/slot & Msg/slot & Time (s) \\", r"\midrule"]

    def row(a, n, first):
        if (a, n) not in agg.index:
            return None
        r = agg.loc[(a, n)]
        return "{} & {} & {} & {} & {:.1f} & {:.0f} & {:.1f} \\\\".format(
            SHORT[a] if first else "", n, fmt(r.phi_final), fmt(r.coverage_frac, 3),
            r.gossip_bytes_per_node_round, r.gossip_msgs_per_round, r.wallclock_s)
    for i, n in enumerate([50, 100, 200, 400, 800]):
        x = row("FogField-BR", n, i == 0)
        if x:
            lines.append(x)
    lines.append(r"\addlinespace")
    for i, n in enumerate([50, 100, 200, 400, 800]):
        x = row("D-POA-style", n, i == 0)
        if x:
            lines.append(x)
    lines.append(r"\addlinespace")
    for a in ["FogField-LLL", "GossipGreedy", "Consensus-ADMM", "Static"]:
        x = row(a, 800, True)
        if x:
            lines.append(x)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write("tab_scale.tex", "\n".join(lines))

    fig, axes = plt.subplots(1, 4, figsize=(13.6, 2.7))
    for a in ["FogField-BR", "FogField-LLL", "D-POA-style", "GossipGreedy", "Consensus-ADMM", "Static"]:
        if a not in agg.index.get_level_values(0):
            continue
        s = agg.loc[a].sort_index()
        mk = "o-" if a.startswith("FogField") else ".--"
        lw = 1.8 if a.startswith("FogField") else 1.0
        for ax, col in zip(axes, ["phi_final", "gossip_bytes_per_node_round", "gossip_msgs_per_round", "wallclock_s"]):
            ax.plot(s.index, s[col], mk, lw=lw, color=COL[a], label=a, ms=4)
    axes[1].axhline(270, color="k", lw=0.8, ls=":")
    axes[1].annotate("codec ceiling 270 B", (60, 262), fontsize=6.5)
    axes[0].set_ylabel(r"$\Phi$ at the end of the run")
    axes[1].set_ylabel("payload bytes / node / slot")
    axes[2].set_ylabel("messages / slot"); axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[3].set_ylabel("wall-clock per run (s)"); axes[3].set_yscale("log"); axes[3].set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("number of fog nodes $n$"); ax.grid(alpha=0.3); ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(FixedLocator([50, 100, 200, 400, 800]))
        ax.xaxis.set_major_formatter(FixedFormatter(["50", "100", "200", "400", "800"]))
        ax.xaxis.set_minor_locator(NullLocator())
    axes[0].legend(fontsize=6.6, loc="lower right")
    fig.subplots_adjust(wspace=0.42)
    savefig(fig, "fig_scalability")


# =========================================================================== equilibrium / locality
def do_eq():
    if EQ.empty:
        return
    res = {}
    lines = [r"\begin{table}[t]",
             r"\caption{Distance to a rest point and the cost of neighbour-only information, $n = 40$,",
             r"80 slots, 5 seeds per row. $\varepsilon_{\mathrm{rel}}$: largest one-move improvement of the",
             r"global $\Phi$ still available at the reached configuration, relative to $\Phi$ (mean [max]);",
             r"$\varepsilon$@1\%: runs with $\varepsilon_{\mathrm{rel}} \le 1\%$. Info: share of the other",
             r"nodes a sampled node knows. Agree / Top-3: share of 16 sampled decisions whose local choice",
             r"is the global best / among the global three best. Regret: global utility lost per decision.",
             r"MEL uses $R_{\mathrm{comm}} = 500$\,m so that gossip cannot reach a full mesh.}",
             r"\label{tab:eq}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r" & Method & $\varepsilon_{\mathrm{rel}}$ (\%) & $\varepsilon$@1\% & Info & Agree & Top-3 & Regret \\",
             r"\midrule"]
    for tb in ["MEL", "BEJ"]:
        first = True
        for a in ["FogField-LLL", "FogField-BR", "GossipGreedy", "Consensus-ADMM"]:
            c = EQ[(EQ.testbed == tb) & (EQ.algo == a)]
            if c.empty:
                continue
            d = {"eps_rel_mean": float(c.nash_epsilon_rel.mean() * 100), "eps_rel_max": float(c.nash_epsilon_rel.max() * 100),
                 "eps_ne_1pct": int(c.nash_is_eps_nash_1pct.sum()), "runs": int(len(c)),
                 "info": float(c.loc_info_fraction.mean()), "agree": float(c.loc_rank_agreement.mean()),
                 "top3": float(c.loc_top3_agreement.mean()), "regret": float(c.loc_regret_mean.mean()),
                 "regret_max": float(c.loc_regret_max.max()), "phi": float(c.nash_phi.mean())}
            res["{}_{}".format(tb, a)] = d
            lines.append("{} & {} & {:.2f} [{:.2f}] & {}/{} & {:.2f} & {:.2f} & {:.2f} & {:.4f} \\\\".format(
                tb if first else "", SHORT[a], d["eps_rel_mean"], d["eps_rel_max"], d["eps_ne_1pct"], d["runs"],
                d["info"], d["agree"], d["top3"], d["regret"]))
            first = False
        lines.append(r"\addlinespace")
    if lines[-1] == r"\addlinespace":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    NUM["eq"] = res
    EQ.drop(columns=[c for c in EQ.columns if c.startswith("_")], errors="ignore").to_csv(PAPER / "table5_equilibrium.csv", index=False)
    write("tab_eq.tex", "\n".join(lines))


# =========================================================================== LLL schedule
def do_lll():
    if LLL.empty:
        return
    res = {}
    variants = ["base", "fast", "cap4000", "fixed400", "fixed50"]
    vlabel = {"base": r"default, $\min\{400, 8(1+0.45t)\}$", "fast": r"fast, $\min\{400, 8(1+4t)\}$",
              "cap4000": r"fast, cap $4000$", "fixed400": r"fixed $\beta = 400$", "fixed50": r"fixed $\beta = 50$"}
    lines = [r"\begin{table}[t]",
             r"\caption{FogField-LLL under five annealing schedules: final $\Phi$ in the static scenario,",
             r"mean over 5 seeds (the same units as the main grid), with FogField-BR for reference.}",
             r"\label{tab:lll}", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{MEL} & \multicolumn{2}{c}{BEJ} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Schedule & $n{=}20$ & $n{=}50$ & $n{=}20$ & $n{=}50$ \\", r"\midrule"]
    for v in variants:
        cells = []
        for tb in ["MEL", "BEJ"]:
            for n in (20, 50):
                c = LLL[(LLL.testbed == tb) & (LLL.variant == v) & (LLL.n_nodes == n)]
                m = float(c.phi_final.mean()) if len(c) else float("nan")
                res["{}_{}_{}".format(tb, n, v)] = m
                cells.append(fmt(m))
        lines.append("{} & {} \\\\".format(vlabel[v], " & ".join(cells)))
    cells = []
    for tb in ["MEL", "BEJ"]:
        for n in (20, 50):
            c = MAIN[(MAIN.testbed == tb) & (MAIN.scenario == "static") & (MAIN.algo == "FogField-BR") & (MAIN.n_nodes == n)]
            m = float(c.phi_final.mean()) if len(c) else float("nan")
            res["{}_{}_BR".format(tb, n)] = m
            cells.append(fmt(m))
    lines += [r"\addlinespace", "FogField-BR (reference) & {} \\\\".format(" & ".join(cells)),
              r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    # paired: best LLL variant vs BR per testbed (10 units)
    for tb in ["MEL", "BEJ"]:
        br = series(tb, "static", "FogField-BR")
        best = None
        for v in variants:
            c = LLL[(LLL.testbed == tb) & (LLL.variant == v)].set_index(["n_nodes", "seed"]).phi_final.sort_index()
            idx = c.index.intersection(br.index)
            if len(idx) == 0:
                continue
            pr = paired((c[idx] - br[idx]).to_numpy())
            pr["variant"] = v
            pr["rel"] = 100 * pr["mean"] / float(br[idx].mean())
            res["{}_paired_vs_BR_{}".format(tb, v)] = pr
            if best is None or pr["mean"] > best["mean"]:
                best = pr
        res[tb + "_best_vs_BR"] = best
    NUM["lll"] = res
    pd.Series(res).to_json(PAPER / "table9_lll_schedule.json")
    write("tab_lll.tex", "\n".join(lines))


# =========================================================================== view bound
def do_vb():
    if VB.empty:
        return
    res = {}
    lines = [r"\begin{table}[t]",
             r"\caption{The bounded local view, measured: FogField-BR at $n = 800$, static, 4 slots.",
             r"View: mean number of other nodes a node knows. Decision: seconds of local-utility",
             r"evaluation per node-decision (one CPU core).}",
             r"\label{tab:viewbound}", r"\centering", r"\scriptsize",
             r"\begin{tabular}{llrrr}", r"\toprule",
             r"Testbed & View & Mean view & Decision (s) & Slot (s) \\", r"\midrule"]
    for tb in ["MEL", "BEJ"]:
        for cap in [12, None]:
            c = VB[(VB.testbed == tb) & (VB.view_cap.isna() if cap is None else (VB.view_cap == cap))]
            if c.empty:
                continue
            r = c.iloc[0]
            k = "{}_{}".format(tb, "unbounded" if cap is None else "bounded")
            res[k] = {"view": float(r.view_size_mean), "decide_s": float(r.decide_s_per_node),
                      "slot_s": float(r.sec_per_round)}
            lines.append("{} & {} & {:.1f} & {:.4f} & {:.1f} \\\\".format(
                tb, "unbounded" if cap is None else "$K = 12$", r.view_size_mean, r.decide_s_per_node, r.sec_per_round))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    for tb in ["MEL", "BEJ"]:
        if tb + "_bounded" in res and tb + "_unbounded" in res:
            res[tb + "_ratio"] = res[tb + "_unbounded"]["decide_s"] / res[tb + "_bounded"]["decide_s"]
    NUM["vb"] = res
    write("tab_viewbound.tex", "\n".join(lines))


# =========================================================================== misc numbers
def do_misc():
    if MAIN.empty:
        return
    benign = MAIN[~MAIN.scenario.isin(["random_failures", "catastrophic", "seed_loss"])]
    NUM["energy_peak_node_kJ"] = float(MAIN.energy_node_max_J.max() / 1e3)
    NUM["battery_kJ"] = 275 * 3.6
    NUM["benign_units"] = int(len(benign))
    NUM["benign_units_with_depletion"] = int((benign.alive_final < benign.n_nodes).sum() -
                                             ((benign.scenario == "expansion") & (benign.alive_final < benign.n_nodes)).sum())
    pay = MAIN[MAIN.algo.isin(FOG)].groupby(["testbed", "algo"]).gossip_bytes_per_node_round.mean()
    NUM["payload_main"] = {"{}_{}".format(*k): float(v) for k, v in pay.items()}
    # trace statistics for the convergence figure (BEJ, n=50, seed 0, static, FogField-BR)
    c = MAIN[(MAIN.testbed == "BEJ") & (MAIN.scenario == "static") & (MAIN.algo == "FogField-BR") &
             (MAIN.n_nodes == 50) & (MAIN.seed == 0)]
    if len(c):
        tr = np.asarray(c.iloc[0].trace_phi, float)[-30:]
        NUM["fig_conv_br_last30"] = {"sd": float(tr.std(ddof=1)), "min": float(tr.min()), "max": float(tr.max())}
        NUM["fig_deploy_cov"] = float(c.iloc[0].coverage_frac)
    # BEJ static: margins by n for the abstract
    NUM["mel_static_conv"] = (MAIN[(MAIN.testbed == "MEL") & (MAIN.scenario == "static")]
                              .groupby("algo").convergence_round.mean().to_dict())
    NUM["bej_static_cov_n50"] = (MAIN[(MAIN.testbed == "BEJ") & (MAIN.scenario == "static") & (MAIN.n_nodes == 50)]
                                 .groupby("algo").coverage_frac.mean().to_dict())


def fig_convergence():
    c = MAIN[(MAIN.testbed == "BEJ") & (MAIN.scenario == "static") & (MAIN.n_nodes == 50) & (MAIN.seed == 0)]
    if c.empty:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for a in [x for x in ALGOS if x != "RandomWalk"]:
        r = c[c.algo == a]
        if r.empty:
            continue
        r = r.iloc[0]
        ax.plot(r.trace_phi, "-" if r.decentralized else "--", lw=2.0 if a.startswith("FogField") else 1.0,
                color=COL[a], label=SHORT[a], alpha=0.92)
    ax.set_xlabel("slot"); ax.set_ylabel(r"global potential $\Phi$")
    ax.legend(fontsize=6.8, ncol=2, loc="lower right", framealpha=0.92); ax.grid(alpha=0.3)
    savefig(fig, "fig_convergence_bej_static")


def fig_deployment():
    import sys
    c = MAIN[(MAIN.testbed == "BEJ") & (MAIN.scenario == "static") & (MAIN.algo == "FogField-BR") &
             (MAIN.n_nodes == 50) & (MAIN.seed == 0)]
    if c.empty:
        return
    z = np.load(HERE / "fogfield_out_v2" / "cache" / "geolife_uall_g400_s30_c500.npz", allow_pickle=True)
    dev, w = z["device_xy"], z["device_w"].mean(axis=0)
    r = c.iloc[0]
    pos = np.asarray(r.final_pos); alive = np.asarray(r.final_alive, bool)
    P = pos[alive]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    sc = ax.scatter(dev[:, 0] / 1000, dev[:, 1] / 1000, s=12, c=w, cmap="viridis", alpha=0.75, linewidths=0)
    plt.colorbar(sc, ax=ax, label="mean demand weight", shrink=0.85)
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if d2[i, j] <= 1400.0 ** 2:
                ax.plot([P[i, 0] / 1000, P[j, 0] / 1000], [P[i, 1] / 1000, P[j, 1] / 1000], "-", lw=0.5,
                        color="steelblue", alpha=0.55)
    ax.scatter(P[:, 0] / 1000, P[:, 1] / 1000, marker="^", s=46, color="crimson", edgecolor="k",
               linewidths=0.4, zorder=5, label="fog nodes")
    ax.set_aspect("equal"); ax.set_xlabel("east (km)"); ax.set_ylabel("north (km)")
    ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=0.25)
    savefig(fig, "fig_deployment_bej")


def fig_power():
    P0, Pi, U, v0, d0, rho, s, A = 79.86, 88.63, 120.0, 4.03, 0.6, 1.225, 0.05, 0.503
    V = np.linspace(0, 30, 600)
    P = P0 * (1 + 3 * V ** 2 / U ** 2) + Pi * np.sqrt(np.maximum(np.sqrt(1 + V ** 4 / (4 * v0 ** 4)) - V ** 2 / (2 * v0 ** 2), 0)) \
        + 0.5 * d0 * rho * s * A * V ** 3
    vme = float(V[int(np.argmin(P))])
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.plot(V, P, lw=1.6, color="#1b6ca8")
    ax.axvline(vme, ls="--", lw=0.9, color="crimson")
    ax.axvline(20, ls=":", lw=0.9, color="gray")
    ax.axhline(P0 + Pi, ls=":", lw=0.9, color="gray")
    ax.annotate("hover {:.0f} W".format(P0 + Pi), (0.6, P0 + Pi + 6), fontsize=7, color="gray")
    ax.annotate("$V_{{me}}$ = {:.1f} m/s\n{:.0f} W (loiter)".format(vme, float(P.min())), (vme + 0.8, float(P.min()) + 6),
                fontsize=7, color="crimson")
    ax.annotate("$V_{max}$", (20.3, 230), fontsize=7, color="gray")
    ax.set_xlabel("horizontal speed $V$ (m/s)"); ax.set_ylabel("propulsion power (W)"); ax.grid(alpha=0.3)
    savefig(fig, "fig_uav_power")
    NUM["V_me"] = vme; NUM["P_me"] = float(P.min()); NUM["P_hover"] = P0 + Pi


def fig_testbeds():
    """Both device fields with their initial sites (legend below the panels, referee m18)."""
    cache = HERE / "fogfield_out_v2" / "cache"
    tbs = [("MEL", cache / "eua_melbcbd_816.npz"), ("BEJ", cache / "geolife_uall_g400_s30_c500.npz")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), layout="constrained",
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    for ax, (name, f) in zip(axes, tbs):
        z = np.load(f, allow_pickle=True)
        xy, w, sites = z["device_xy"], z["device_w"].mean(axis=0), z["sites_xy"]
        if np.ptp(w) > 0:          # BEJ: mean diurnal weight per cell
            sc = ax.scatter(xy[:, 0] / 1000, xy[:, 1] / 1000, c=w, s=9, cmap="viridis", alpha=0.85,
                            linewidths=0, label="demand cells")
            plt.colorbar(sc, ax=ax, label="mean demand weight", shrink=0.8)
        else:                      # MEL: every user has unit weight
            ax.scatter(xy[:, 0] / 1000, xy[:, 1] / 1000, color="#2a788e", s=7, alpha=0.7,
                       linewidths=0, label="users (unit weight)")
        ax.scatter(sites[:, 0] / 1000, sites[:, 1] / 1000, marker="^", s=26, facecolor="none",
                   edgecolor="crimson", linewidths=0.8, label="initial sites")
        ax.legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
        ax.set_title("{} ({} devices)".format(name, len(xy)), fontsize=8)
        ax.set_xlabel("east (km)"); ax.set_ylabel("north (km)")
        ax.set_aspect("equal"); ax.grid(alpha=0.25)
    savefig(fig, "fig_testbeds")
    z = np.load(tbs[1][1], allow_pickle=True)
    tot = z["device_w"].sum(axis=1)
    hrs = np.arange(len(tot)) * (24.0 / len(tot))
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    ax.plot(hrs, tot, lw=1.5, color="#2ca02c")
    ax.fill_between(hrs, tot, alpha=0.18, color="#2ca02c")
    ax.set_xlabel("time of day (h)"); ax.set_ylabel("total demand")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4)); ax.grid(alpha=0.3)
    savefig(fig, "fig_demand_profile")


if __name__ == "__main__":
    for f in [do_gap, do_bej, do_mel, do_byz, do_ctrl, do_ctrl2, do_rec, do_scale, do_eq, do_lll, do_vb, do_misc,
              fig_convergence, fig_deployment, fig_power, fig_testbeds]:
        try:
            f()
        except Exception as exc:          # partial ledgers during development
            import traceback
            print("[{}] skipped: {}: {}".format(f.__name__, type(exc).__name__, exc))
            traceback.print_exc(limit=2)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o
    (HERE / "paper_numbers.json").write_text(json.dumps(clean(NUM), indent=1), encoding="utf-8")
    print("numbers -> paper_numbers.json")
