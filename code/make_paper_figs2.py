"""Re-render the shared figures paper-style: no in-image titles (those belong in the
LaTeX caption for an IEEE submission), tighter sizing, vector PDF output."""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None

SRC = Path(__file__).parent / "fogfield_src_fullrun.py"
src = SRC.read_text(encoding="utf-8")
src, n = re.subn(r"^TIME_BUDGET_MIN = 50 if QUICK else 50$",
                 "TIME_BUDGET_MIN = 0.03 if QUICK else 0.03", src, flags=re.MULTILINE)
assert n == 1
g = sys.modules["__main__"].__dict__
g["__file__"] = str(SRC)
exec(compile(src, str(SRC), "exec"), g)

np = g["np"]; pd = g["pd"]
MILPDF = g["MILPDF"]; SCALE = g["SCALE"]; TESTBEDS = g["TESTBEDS"]; UAV = g["UAV"]
OUT = Path("paper_figs"); OUT.mkdir(exist_ok=True)
print("\n[figs2] rendering paper-style versions", flush=True)


def save(fig, name):
    fig.savefig(OUT / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(OUT / (name + ".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("  " + name)


# ---- UAV power curve ---------------------------------------------------------
V = np.linspace(0, 40, 400)
P = UAV.propulsion(V)
vopt = float(V[int(np.argmin(P))])
fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.plot(V, P, lw=1.6, color="#1b6ca8")
ax.axvline(vopt, ls="--", lw=0.9, color="crimson")
ax.axhline(UAV.P_hover, ls=":", lw=0.9, color="gray")
ax.annotate("hover {:.0f} W".format(UAV.P_hover), (0.6, UAV.P_hover + 6), fontsize=7,
            color="gray")
ax.annotate("min {:.0f} W\nat {:.1f} m/s".format(float(P.min()), vopt),
            (vopt + 1.5, float(P.min()) + 4), fontsize=7, color="crimson")
ax.set_xlabel("horizontal speed $V$ (m/s)"); ax.set_ylabel("propulsion power (W)")
ax.grid(alpha=0.3)
save(fig, "fig_uav_power")

# ---- optimality gap boxplot --------------------------------------------------
gapcols = [c for c in MILPDF.columns if c.startswith("gap_")]
gdf = MILPDF[gapcols].rename(columns=lambda c: c[4:])
order = (gdf.mean() * 100).sort_values().index.tolist()
data = [gdf[c].dropna().to_numpy() * 100 for c in order]
fig, ax = plt.subplots(figsize=(3.5, 2.9))
bp = ax.boxplot(data, showmeans=True, widths=0.6, patch_artist=True)
for patch, name in zip(bp["boxes"], order):
    patch.set_facecolor("#1b6ca8" if name.startswith("FogField") else "#dddddd")
    patch.set_alpha(0.75)
ax.axhline(50, ls="--", color="crimson", lw=1.0)
ax.annotate("worst case admitted by PoA $\\leq$ 2", (0.6, 53), fontsize=6.5,
            color="crimson")
ax.set_xticks(range(1, len(data) + 1))
ax.set_xticklabels(order, rotation=32, ha="right", fontsize=7)
ax.set_ylabel("gap to exact bound (\\%)")
ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)
save(fig, "fig_optimality_gap")

# ---- scalability: quality + per-node bytes -----------------------------------
COL = {"FogField-BR": "#1b6ca8", "FogField-LLL": "#2ca02c", "D-POA-style": "#d62728",
       "GossipGreedy": "#ff7f0e", "Consensus-ADMM": "#8c564b", "Static": "#bcbd22"}
fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.7))
for algo, sub in SCALE.groupby("algo"):
    agg = sub.groupby("n_nodes").mean(numeric_only=True).sort_index()
    mk = "o-" if str(algo).startswith("FogField") else ".--"
    lw = 1.8 if str(algo).startswith("FogField") else 1.0
    c = COL.get(algo, "#777")
    axes[0].plot(agg.index, agg["phi_final"], mk, lw=lw, color=c, label=algo, ms=4)
    axes[1].plot(agg.index, agg["gossip_bytes_per_node_round"], mk, lw=lw, color=c,
                 label=algo, ms=4)
    axes[2].plot(agg.index, agg["wallclock_s"], mk, lw=lw, color=c, label=algo, ms=4)
axes[0].set_ylabel(r"$\Phi$ at convergence")
axes[1].set_ylabel("payload bytes / node / round")
axes[2].set_ylabel("wall-clock per run (s)"); axes[2].set_yscale("log")
for ax in axes:
    ax.set_xlabel("number of fog nodes $n$")
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
axes[0].legend(fontsize=6.8, loc="lower right")
save(fig, "fig_scalability")

# ---- testbeds ----------------------------------------------------------------
tbs = [TESTBEDS[k] for k in ("MEL", "BEJ") if k in TESTBEDS]
fig, axes = plt.subplots(1, len(tbs), figsize=(7.2, 3.0))
for ax, tb in zip(np.atleast_1d(axes), tbs):
    w = tb.device_w.mean(axis=0)
    sc = ax.scatter(tb.device_xy[:, 0] / 1000, tb.device_xy[:, 1] / 1000, c=w, s=9,
                    cmap="viridis", alpha=0.85, linewidths=0)
    ax.scatter(tb.init_sites_xy[:, 0] / 1000, tb.init_sites_xy[:, 1] / 1000, marker="^",
               s=26, facecolor="none", edgecolor="crimson", linewidths=0.8,
               label="initial sites")
    plt.colorbar(sc, ax=ax, label="demand weight", shrink=0.85)
    ax.set_title("{} ({} devices)".format(tb.name, tb.n_devices), fontsize=8)
    ax.set_xlabel("east (km)"); ax.set_ylabel("north (km)")
    ax.set_aspect("equal"); ax.legend(fontsize=6.5, loc="best")
    ax.grid(alpha=0.25)
save(fig, "fig_testbeds")

# ---- BEJ diurnal demand ------------------------------------------------------
tb = TESTBEDS.get("BEJ")
if tb is not None and tb.n_slots > 1:
    tot = tb.device_w.sum(axis=1)
    hrs = np.arange(tb.n_slots) * (24.0 / tb.n_slots)
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    ax.plot(hrs, tot, lw=1.5, color="#2ca02c")
    ax.fill_between(hrs, tot, alpha=0.18, color="#2ca02c")
    ax.set_xlabel("time of day (h)"); ax.set_ylabel("total demand")
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4)); ax.grid(alpha=0.3)
    save(fig, "fig_demand_profile")

print("\n[figs2] done ->", OUT.resolve())
