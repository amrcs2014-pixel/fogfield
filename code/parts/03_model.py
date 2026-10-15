# %% [markdown]
# ## 5 - Physical model (R2): rotary-wing UAV propulsion plus compute energy
#
# The closed-form propulsion power of a rotary-wing UAV in level flight at speed `V`
# (Zeng, Xu and Zhang, *IEEE Trans. Wireless Communications* 18(4):2329-2345, 2019,
# DOI 10.1109/TWC.2019.2902559):
#
# $$P(V)=\underbrace{P_0\left(1+\frac{3V^2}{U_{\mathrm{tip}}^2}\right)}_{\text{blade profile}}
# +\underbrace{P_i\left(\sqrt{1+\frac{V^4}{4v_0^4}}-\frac{V^2}{2v_0^2}\right)^{1/2}}_{\text{induced}}
# +\underbrace{\tfrac12 d_0\rho s A V^3}_{\text{parasite}}$$
#
# with the paper's parameters. Hovering costs `P(0) = P_0 + P_i = 168.49 W`. A compute-power
# term (idle plus load-proportional) and a battery budget are layered on top.

# %%
@dataclass(frozen=True)
class UAVPower:
    """Zeng-Xu-Zhang (TWC 2019) rotary-wing propulsion power, plus onboard compute power."""
    P0: float = 79.86        # blade profile power in hover (W)
    Pi: float = 88.63        # induced power in hover (W)
    U_tip: float = 120.0     # rotor blade tip speed (m/s)
    v0: float = 4.03         # mean rotor induced velocity in hover (m/s)
    d0: float = 0.6          # fuselage drag ratio
    rho: float = 1.225       # air density (kg/m^3)
    s: float = 0.05          # rotor solidity
    A: float = 0.503         # rotor disc area (m^2)
    V_max: float = 20.0      # max horizontal speed (m/s)
    P_cpu_idle: float = 4.0  # idle compute power (W)
    P_cpu_busy: float = 12.0 # additional compute power at full load (W)
    battery_Wh: float = 275.0
    loiter: bool = True      # station-keeping by loitering at V_me rather than hovering (v2)

    def propulsion(self, V):
        V = np.asarray(V, dtype=np.float64)
        blade = self.P0 * (1.0 + 3.0 * V ** 2 / self.U_tip ** 2)
        inner = np.sqrt(1.0 + V ** 4 / (4.0 * self.v0 ** 4)) - V ** 2 / (2.0 * self.v0 ** 2)
        induced = self.Pi * np.sqrt(np.maximum(inner, 0.0))
        parasite = 0.5 * self.d0 * self.rho * self.s * self.A * V ** 3
        return blade + induced + parasite

    @property
    def P_hover(self) -> float:
        return float(self.P0 + self.Pi)

    @cached_property
    def V_me(self) -> float:
        """Maximum-endurance speed: the speed that minimises propulsion power."""
        V = np.linspace(0.0, self.V_max, 20001)
        return float(V[int(np.argmin(self.propulsion(V)))])

    @property
    def P_me(self) -> float:
        return float(self.propulsion(self.V_me))

    def flight_energy(self, dist_m, dt_s: float):
        """Energy (J) to travel `dist_m` within a step of `dt_s`.

        v2 (loiter=True): the UAV flies the whole step at speed max(d/dt, V_me), capped at
        V_max; a node that needs to cover less than V_me*dt (staying put included) loiters at
        the maximum-endurance speed V_me instead of hovering, because hovering costs more
        (P(0) = 168.5 W against P(V_me) = 126.0 W with the paper's parameters). The slot energy
        is therefore non-decreasing in the distance and minimal for staying, which is what
        makes a stationary configuration a possible rest point of the dynamics (T1).

        loiter=False reproduces v1.0: fly at the slowest sufficient speed, hover for the rest.
        """
        dist = np.asarray(dist_m, dtype=np.float64)
        if self.loiter:
            V = np.clip(np.maximum(dist / max(dt_s, 1e-9), self.V_me), 0.0, self.V_max)
            return self.propulsion(V) * dt_s
        v_need = dist / max(dt_s, 1e-9)
        V = np.clip(v_need, 0.0, self.V_max)
        t_fly = np.where(V > 1e-9, dist / np.maximum(V, 1e-9), 0.0)
        t_fly = np.minimum(t_fly, dt_s)
        t_hov = np.maximum(dt_s - t_fly, 0.0)
        return self.propulsion(V) * t_fly + self.P_hover * t_hov

    def compute_energy(self, load_frac, dt_s: float):
        load = np.clip(np.asarray(load_frac, dtype=np.float64), 0.0, 1.0)
        return (self.P_cpu_idle + self.P_cpu_busy * load) * dt_s

    def max_step_m(self, dt_s: float) -> float:
        return float(self.V_max * dt_s)


UAV = UAVPower()

# validation against the published closed form
_hover = float(UAV.propulsion(0.0))
assert abs(_hover - UAV.P_hover) < 1e-9, "hover power must equal P0 + Pi"
_V = np.linspace(0, 40, 400)
_P = UAV.propulsion(_V)
_v_opt = float(_V[int(np.argmin(_P))])
print("P(0)          = {:.2f} W   (P0 + Pi = {:.2f} W)".format(_hover, UAV.P_hover))
print("min power at V= {:.1f} m/s -> {:.2f} W".format(_v_opt, float(_P.min())))
print("battery       = {:.0f} Wh = {:.0f} kJ -> {:.1f} min of hover".format(
    UAV.battery_Wh, UAV.battery_Wh * 3600 / 1000, UAV.battery_Wh * 3600 / UAV.P_hover / 60))
print("loiter (v2)   : V_me = {:.2f} m/s, P(V_me) = {:.2f} W; a station-keeping node loiters "
      "at V_me, so staying is the cheapest move".format(UAV.V_me, UAV.P_me))
_e = UAV.flight_energy(np.array([0.0, 120.0, 306.0, 400.0, 600.0]), 30.0)
assert np.all(np.diff(_e) >= -1e-9), "slot energy must be non-decreasing in distance"

fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.plot(_V, _P)
ax.axvline(_v_opt, ls="--", lw=0.8, color="crimson")
ax.annotate("min at {:.1f} m/s".format(_v_opt), (_v_opt, float(_P.min())),
            textcoords="offset points", xytext=(8, 18), fontsize=8, color="crimson")
ax.set_xlabel("horizontal speed V (m/s)"); ax.set_ylabel("propulsion power (W)")
ax.set_title("Rotary-wing UAV power (Zeng-Xu-Zhang, TWC 2019)")
fig.savefig(DIRS["figs"] / "fig1_uav_power.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6 - System model and the global objective $\Phi$
#
# $$\Phi(a)=w_c\,\mathrm{Coverage}(a)+w_k\,\mathrm{Connectivity}(a)-w_e\,\mathrm{Energy}(a)$$
#
# Each term is normalised by a **constant that does not depend on which nodes are active**
# (total device weight, total node count, a reference per-node energy). That detail is what
# makes $\Phi$ well defined on sub-populations, which the marginal-contribution utility
# requires - normalising by the *active* count instead would silently break the exact
# potential property.
#
# * **Coverage** - a device is served if some active node is within `R_cov`. With
#   `capacity=None` (default) coverage is the uncapacitated weighted union, which is monotone
#   submodular and puts the game in Vetta's valid-utility class, so `PoA <= 2` applies.
#   Setting a finite `capacity` switches to greedy capacitated assignment; that variant is
#   **outside** the valid-utility structure and the 1/2 bound is not claimed for it.
# * **Connectivity** - fraction of all nodes lying in the largest connected component of the
#   communication graph (edge when `||a_i - a_j|| <= R_comm`).
# * **Energy** - flight energy for this step plus compute energy, from `UAVPower`.
#
# > **R5, stated plainly.** This is a *weighted scalarisation*. It returns one point on the
# > trade-off surface per weight vector, not a Pareto front. No hypervolume is reported.

# %%
@dataclass(frozen=True)
class Weights:
    coverage: float = 1.0
    connectivity: float = 0.35
    energy: float = 0.15


@dataclass
class Environment:
    """Everything fixed about a problem instance, plus the evaluation of Phi."""
    testbed: Testbed
    n_nodes: int
    R_cov: float = 500.0
    R_comm: float = 900.0
    dt_s: float = 30.0
    weights: Weights = field(default_factory=Weights)
    capacity: Optional[float] = None       # None -> uncapacitated (valid-utility structure)
    uav: UAVPower = field(default_factory=UAVPower)
    lattice_m: float = 120.0               # action-lattice spacing

    def __post_init__(self):
        self.W_total = float(self.testbed.device_w.sum(axis=1).mean())
        self.E_ref = float(self.uav.P_hover * self.dt_s)
        x0, x1, y0, y1 = self.testbed.bounds
        pad = self.R_cov * 0.5
        self.lat_x = np.arange(x0 - pad, x1 + pad + 1e-9, self.lattice_m)
        self.lat_y = np.arange(y0 - pad, y1 + pad + 1e-9, self.lattice_m)
        self.max_step = self.uav.max_step_m(self.dt_s)

    # ---- geometry helpers -------------------------------------------------
    def snap(self, xy: np.ndarray) -> np.ndarray:
        """Snap positions onto the action lattice (the discretised strategy set)."""
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        ix = np.clip(np.round((xy[:, 0] - self.lat_x[0]) / self.lattice_m), 0,
                     len(self.lat_x) - 1).astype(int)
        iy = np.clip(np.round((xy[:, 1] - self.lat_y[0]) / self.lattice_m), 0,
                     len(self.lat_y) - 1).astype(int)
        return np.stack([self.lat_x[ix], self.lat_y[iy]], axis=1)

    def action_set(self, xy_i: np.ndarray, max_actions: int = 25) -> np.ndarray:
        """Lattice cells reachable by one node in one step (its strategy set A_i)."""
        r = self.max_step
        xs = self.lat_x[(self.lat_x >= xy_i[0] - r) & (self.lat_x <= xy_i[0] + r)]
        ys = self.lat_y[(self.lat_y >= xy_i[1] - r) & (self.lat_y <= xy_i[1] + r)]
        if len(xs) == 0:
            xs = np.array([self.snap(xy_i)[0, 0]])
        if len(ys) == 0:
            ys = np.array([self.snap(xy_i)[0, 1]])
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        cand = np.stack([gx.ravel(), gy.ravel()], axis=1)
        d = np.linalg.norm(cand - xy_i[None, :], axis=1)
        cand = cand[d <= r + 1e-6]
        if len(cand) == 0:
            return self.snap(xy_i)
        if len(cand) > max_actions:                  # keep the nearest ring plus a sample
            d = np.linalg.norm(cand - xy_i[None, :], axis=1)
            keep = np.argsort(d)[:max_actions]
            cand = cand[keep]
        return cand

    # ---- objective terms --------------------------------------------------
    def coverage(self, pos: np.ndarray, active: np.ndarray, w: np.ndarray) -> float:
        if not active.any():
            return 0.0
        P = pos[active]
        d2 = ((P[:, None, :] - self.testbed.device_xy[None, :, :]) ** 2).sum(-1)
        inr = d2 <= self.R_cov ** 2
        if self.capacity is None:
            return float(w[inr.any(axis=0)].sum())
        # greedy capacitated assignment: heaviest device first, nearest feasible node
        cap = np.full(P.shape[0], float(self.capacity))
        served = 0.0
        order = np.argsort(-w)
        for m in order:
            if w[m] <= 0:
                break
            cand = np.flatnonzero(inr[:, m] & (cap >= w[m]))
            if cand.size:
                j = cand[np.argmin(d2[cand, m])]
                cap[j] -= w[m]
                served += w[m]
        return float(served)

    def load_fraction(self, pos: np.ndarray, active: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Per-node normalised served load, used for compute energy."""
        load = np.zeros(pos.shape[0])
        if not active.any():
            return load
        idx = np.flatnonzero(active)
        P = pos[idx]
        d2 = ((P[:, None, :] - self.testbed.device_xy[None, :, :]) ** 2).sum(-1)
        inr = d2 <= self.R_cov ** 2
        nearest = np.where(inr.any(axis=0), np.argmin(np.where(inr, d2, np.inf), axis=0), -1)
        for k, j in enumerate(idx):
            sel = nearest == k
            load[j] = w[sel].sum()
        denom = self.capacity if self.capacity else max(self.W_total / max(len(idx), 1), 1e-9)
        return np.clip(load / denom, 0.0, 1.0)

    def connectivity(self, pos: np.ndarray, active: np.ndarray) -> float:
        """Size of the largest connected component of the communication graph.

        Uses scipy's connected_components rather than networkx: this runs once per round
        per unit across thousands of units, and graph-object construction dominated the
        profile at the larger node counts.
        """
        idx = np.flatnonzero(active)
        if idx.size == 0:
            return 0.0
        if idx.size == 1:
            return 1.0
        P = pos[idx]
        d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        adj = d2 <= self.R_comm ** 2
        np.fill_diagonal(adj, False)
        n_comp, labels = connected_components(csr_matrix(adj), directed=False)
        if n_comp <= 1:
            return float(idx.size)
        return float(np.bincount(labels).max())

    def algebraic_connectivity(self, pos: np.ndarray, active: np.ndarray) -> float:
        """Fiedler value of the communication graph - reported, not optimised."""
        idx = np.flatnonzero(active)
        if idx.size < 2:
            return 0.0
        P = pos[idx]
        d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        A = (d2 <= self.R_comm ** 2).astype(float)
        np.fill_diagonal(A, 0.0)
        L = np.diag(A.sum(1)) - A
        ev = np.linalg.eigvalsh(L)
        return float(ev[1]) if len(ev) > 1 else 0.0

    def energy(self, pos: np.ndarray, prev: np.ndarray, active: np.ndarray,
               w: np.ndarray) -> float:
        idx = np.flatnonzero(active)
        if idx.size == 0:
            return 0.0
        dist = np.linalg.norm(pos[idx] - prev[idx], axis=1)
        e_fly = self.uav.flight_energy(dist, self.dt_s)
        e_cpu = self.uav.compute_energy(self.load_fraction(pos, active, w)[idx], self.dt_s)
        return float(np.sum(e_fly + e_cpu))

    # ---- the potential function -------------------------------------------
    def phi(self, pos: np.ndarray, prev: np.ndarray, active: np.ndarray,
            w: np.ndarray, parts: bool = False):
        """Global objective. Denominators are constants, never functions of `active`."""
        cov = self.coverage(pos, active, w)
        con = self.connectivity(pos, active)
        ene = self.energy(pos, prev, active, w)
        N = float(self.n_nodes)
        val = (self.weights.coverage * cov / max(self.W_total, 1e-9)
               + self.weights.connectivity * con / N
               - self.weights.energy * ene / (N * self.E_ref))
        if parts:
            # Phi uses the fixed denominator W_total, which the potential property requires.
            # The *reported* coverage fraction is against the demand actually present this
            # round, otherwise a flash crowd (which inflates weights) reads as >100% served.
            w_now = float(np.sum(w))
            return val, {"coverage_w": cov,
                         "coverage_frac": cov / max(w_now, 1e-9),
                         "coverage_norm": cov / max(self.W_total, 1e-9),
                         "giant_frac": con / N, "energy_J": ene,
                         "energy_per_node_J": ene / max(int(active.sum()), 1)}
        return float(val)


# %% [markdown]
# ## 7 - The potential game (R1) and its numerical verification
#
# Players are the fog nodes; the strategy set $A_i$ is the set of lattice cells reachable in
# one step; the utility is the **marginal contribution / wonderful-life utility**
#
# $$u_i(a_i,a_{-i})=\Phi(a_i,a_{-i})-\Phi(a_{-i},\varnothing_i)$$
#
# where $\varnothing_i$ means node $i$ is absent. Because the subtracted term does not depend
# on $a_i$, $\Phi$ is an **exact potential**:
#
# $$u_i(a_i,a_{-i})-u_i(a_i',a_{-i})=\Phi(a_i,a_{-i})-\Phi(a_i',a_{-i}).$$
#
# The algebra is one line, so the test below is not really testing the theorem. It is testing
# **this implementation**: if any term of `phi` leaked a dependence on $a_i$ into the
# baseline - the easy bug here is an energy or normalisation term that quietly sees node
# $i$ - the identity would fail numerically. Reviewers ask for exactly this kind of check.

# %%
class Game:
    """The strategic game: marginal-contribution utility over the discretised action sets."""

    def __init__(self, env: Environment):
        self.env = env

    def phi_without(self, pos, prev, active, w, i: int) -> float:
        a2 = active.copy()
        a2[i] = False
        return self.env.phi(pos, prev, a2, w)

    def utility(self, pos, prev, active, w, i: int) -> float:
        return self.env.phi(pos, prev, active, w) - self.phi_without(pos, prev, active, w, i)


def verify_exact_potential(env: Environment, n_trials: int = 40, seed: int = 7) -> dict:
    """Empirically check u_i(a_i,.) - u_i(a_i',.) == Phi(a_i,.) - Phi(a_i',.)."""
    rng = np.random.default_rng(seed)
    game = Game(env)
    tb = env.testbed
    x0, x1, y0, y1 = tb.bounds
    viol = []
    for _ in range(n_trials):
        pos = env.snap(np.stack([rng.uniform(x0, x1, env.n_nodes),
                                 rng.uniform(y0, y1, env.n_nodes)], axis=1))
        prev = env.snap(pos + rng.normal(0, 60, pos.shape))
        active = np.ones(env.n_nodes, dtype=bool)
        w = tb.weights(int(rng.integers(tb.n_slots)))
        i = int(rng.integers(env.n_nodes))

        alt = env.action_set(pos[i])
        a_new = alt[int(rng.integers(len(alt)))]

        u_before = game.utility(pos, prev, active, w, i)
        phi_before = env.phi(pos, prev, active, w)

        pos2 = pos.copy()
        pos2[i] = a_new
        u_after = game.utility(pos2, prev, active, w, i)
        phi_after = env.phi(pos2, prev, active, w)

        viol.append(abs((u_after - u_before) - (phi_after - phi_before)))

    viol = np.asarray(viol)
    out = {"n_trials": n_trials, "max_violation": float(viol.max()),
           "mean_violation": float(viol.mean()),
           "exact": bool(viol.max() < 1e-9)}
    return out


_env_check = Environment(TESTBEDS["MEL"], n_nodes=12)
_res = verify_exact_potential(_env_check, n_trials=40)
print("exact-potential check (uncapacitated):", _res)
assert _res["exact"], "phi/utility pair is NOT an exact potential - implementation bug"

_env_cap = Environment(TESTBEDS["MEL"], n_nodes=12, capacity=80.0)
_res_cap = verify_exact_potential(_env_cap, n_trials=25)
print("exact-potential check (capacitated)  :", _res_cap)
print("\nNote: exactness holds for both variants because it is structural in the "
      "marginal-contribution utility.\nThe submodular / valid-utility structure that the "
      "PoA <= 2 bound needs is a separate property,\nand it holds only for the "
      "uncapacitated variant.")
write_json(DIRS["tables"] / "potential_check.json",
           {"uncapacitated": _res, "capacitated": _res_cap})
