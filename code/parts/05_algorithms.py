# %% [markdown]
# ## 10 - Algorithms (R1 and R4)
#
# Every controller exposes the same interface, so the runner, the scenarios and the
# measurement harness are shared and the comparison is apples to apples.
#
# **FogField (the contribution)**
# * `FogFieldBR` - best response with inertia. Inertia is what makes asynchronous
#   best-response converge rather than cycle; with an exact potential the finite improvement
#   property then gives convergence to a pure Nash equilibrium (T1).
# * `FogFieldLLL` - log-linear learning with annealed `beta`. The induced Markov chain is
#   reversible with stationary distribution proportional to `exp(beta * Phi)`, so as `beta`
#   grows the process concentrates on **potential maximisers**, i.e. efficient equilibria
#   (T2). This is the variant that escapes the poor local optima best response can lock into.
#
# **Decentralized baselines**
# * `GossipGreedy` - move toward the heaviest locally-known uncovered demand, with a
#   separation term. No game structure, no equilibrium argument.
# * `GuerreroStyle` - a node-repositioning adaptation of Guerrero et al.'s decentralized
#   popularity-pull placement policy, with a migration-cost threshold that damps oscillation.
#   Adapted, not reproduced: the original places services on fixed nodes.
# * `ConsensusADMM` - inexact consensus ADMM (Chang, Hong and Wang, IEEE TSP 2015) on a
#   smoothed continuous relaxation, projected back onto the lattice.
# * `MARLIndependentQ` - independent Q-learning with parameter sharing over a coarse local
#   state. Labelled as an adapted MARL baseline; it is **not** MAPPO.
#
# **Centralized references**
# * `DPOAStyleSwarm` - a D-POA-*style* swarm optimiser over the joint position vector with
#   dynamic re-initialisation on events. A reimplementation from the method description.
# * `StaticDeployment` / `RandomWalk` - lower references.

# %%
@dataclass
class StepContext:
    rnd: int
    pos: np.ndarray
    prev: np.ndarray
    alive: np.ndarray
    w: np.ndarray
    nbrs: List[np.ndarray]
    gossip: GossipLayer
    rng: np.random.Generator
    byzantine: np.ndarray


class Controller:
    name = "base"
    decentralized = True
    uses_gossip = True

    def reset(self, env: Environment, pos: np.ndarray, rng: np.random.Generator) -> None:
        """Controllers deliberately keep no reference to `env`.

        Checkpoints pickle the controller, and `env` holds the whole device field; storing it
        would multiply every checkpoint by the dataset size. `env` is passed to `step`.
        """
        return None

    def step(self, env: Environment, ctx: StepContext) -> np.ndarray:
        raise NotImplementedError

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, d: dict) -> None:
        pass


def _apply_move(env: Environment, pos: np.ndarray, target: np.ndarray,
                alive: np.ndarray) -> np.ndarray:
    """Clip displacement to the one-step reach, snap to the lattice, freeze dead nodes."""
    out = pos.copy()
    idx = np.flatnonzero(alive)
    if idx.size == 0:
        return out
    delta = target[idx] - pos[idx]
    dist = np.linalg.norm(delta, axis=1, keepdims=True)
    scale = np.minimum(1.0, env.max_step / np.maximum(dist, 1e-9))
    out[idx] = env.snap(pos[idx] + delta * scale)
    return out


def _sensed_devices(env: Environment, xy: np.ndarray) -> np.ndarray:
    d2 = ((env.testbed.device_xy - xy[None, :]) ** 2).sum(1)
    return np.flatnonzero(d2 <= (env.R_cov + env.max_step) ** 2)


# %%
class FogFieldBR(Controller):
    """Asynchronous best response with inertia over the discretised action sets."""
    name = "FogField-BR"

    def __init__(self, p_activate: float = 0.35, inertia: float = 0.15,
                 max_actions: int = 21):
        self.p_activate = p_activate
        self.inertia = inertia
        self.max_actions = max_actions

    def step(self, env, ctx):
        target = ctx.pos.copy()
        for i in np.flatnonzero(ctx.alive):
            if ctx.rng.random() > self.p_activate:
                continue
            if ctx.rng.random() < self.inertia:
                continue
            cands = env.action_set(ctx.pos[i], self.max_actions)
            u = local_utility(env, i, cands, ctx.pos, ctx.prev,
                              ctx.gossip.views[i], ctx.w,
                              _sensed_devices(env, ctx.pos[i]))
            target[i] = cands[int(np.argmax(u))]
        return _apply_move(env, ctx.pos, target, ctx.alive)


class FogFieldLLL(Controller):
    """Log-linear learning; beta is annealed upward so the chain concentrates on maximisers."""
    name = "FogField-LLL"

    def __init__(self, p_activate: float = 0.35, beta0: float = 8.0, beta_growth: float = 0.45,
                 beta_max: float = 400.0, max_actions: int = 21):
        self.p_activate = p_activate
        self.beta0 = beta0
        self.beta_growth = beta_growth
        self.beta_max = beta_max
        self.max_actions = max_actions

    def beta(self, rnd: int) -> float:
        """Annealing schedule.

        Utility differences between neighbouring lattice cells are order 0.01-0.1, so beta
        has to reach the low hundreds before the softmax actually exploits. Starting at 8
        keeps the early rounds close to uniform exploration (beta * delta_u well under 1),
        and the growth rate brings beta into the exploiting regime within a few dozen
        rounds, which is the length of a run here.
        """
        return float(min(self.beta_max, self.beta0 * (1.0 + self.beta_growth * rnd)))

    def step(self, env, ctx):
        b = self.beta(ctx.rnd)
        target = ctx.pos.copy()
        for i in np.flatnonzero(ctx.alive):
            if ctx.rng.random() > self.p_activate:
                continue
            cands = env.action_set(ctx.pos[i], self.max_actions)
            u = local_utility(env, i, cands, ctx.pos, ctx.prev,
                              ctx.gossip.views[i], ctx.w,
                              _sensed_devices(env, ctx.pos[i]))
            z = b * (u - u.max())
            p = np.exp(z)
            s = p.sum()
            if not np.isfinite(s) or s <= 0:
                target[i] = cands[int(np.argmax(u))]
            else:
                target[i] = cands[int(ctx.rng.choice(len(cands), p=p / s))]
        return _apply_move(env, ctx.pos, target, ctx.alive)


# %%
class GossipGreedy(Controller):
    """Move toward the heaviest locally-known uncovered demand, repelled by close neighbours."""
    name = "GossipGreedy"

    def __init__(self, pull: float = 0.6, repel: float = 0.35):
        self.pull = pull
        self.repel = repel

    def step(self, env, ctx):
        target = ctx.pos.copy()
        R2 = env.R_cov ** 2
        for i in np.flatnonzero(ctx.alive):
            dev = _sensed_devices(env, ctx.pos[i])
            v = ctx.gossip.views[i]
            nbr_ids = [j for j in v.pos if j != i]
            vec = np.zeros(2)
            if dev.size:
                D = env.testbed.device_xy[dev]
                wl = ctx.w[dev].copy()
                if nbr_ids:
                    NP = np.stack([v.pos[j] for j in nbr_ids], axis=0)
                    taken = (((NP[:, None, :] - D[None, :, :]) ** 2).sum(-1) <= R2).any(0)
                    wl = np.where(taken, 0.0, wl)
                tot = wl.sum()
                if tot > 0:
                    centroid = (D * wl[:, None]).sum(0) / tot
                    vec += self.pull * (centroid - ctx.pos[i])
            if nbr_ids:
                NP = np.stack([v.pos[j] for j in nbr_ids], axis=0)
                d = NP - ctx.pos[i][None, :]
                r = np.linalg.norm(d, axis=1, keepdims=True)
                close = (r[:, 0] < env.R_cov) & (r[:, 0] > 1e-6)
                if close.any():
                    vec -= self.repel * (d[close] / r[close] ** 2 * env.R_cov).mean(0)
            target[i] = ctx.pos[i] + vec
        return _apply_move(env, ctx.pos, target, ctx.alive)


class GuerreroStyle(Controller):
    """Guerrero-style decentralized popularity pull, adapted from services to node movement."""
    name = "Guerrero-style"

    def __init__(self, alpha: float = 0.45, migrate_threshold: float = 0.02):
        self.alpha = alpha
        self.migrate_threshold = migrate_threshold

    def step(self, env, ctx):
        target = ctx.pos.copy()
        for i in np.flatnonzero(ctx.alive):
            dev = _sensed_devices(env, ctx.pos[i])
            if dev.size == 0:
                continue
            D = env.testbed.device_xy[dev]
            wl = ctx.w[dev]
            inr = ((D - ctx.pos[i][None, :]) ** 2).sum(1) <= env.R_cov ** 2
            if not inr.any() or wl[inr].sum() <= 0:
                served_centroid = D[np.argmax(wl)] if wl.size else ctx.pos[i]
            else:
                served_centroid = (D[inr] * wl[inr][:, None]).sum(0) / wl[inr].sum()
            move = self.alpha * (served_centroid - ctx.pos[i])
            # migration threshold: only move if the expected gain clears the cost
            cands = np.stack([ctx.pos[i], ctx.pos[i] + move], axis=0)
            u = local_utility(env, i, cands, ctx.pos, ctx.prev, ctx.gossip.views[i],
                              ctx.w, dev)
            if (u[1] - u[0]) > self.migrate_threshold:
                target[i] = ctx.pos[i] + move
        return _apply_move(env, ctx.pos, target, ctx.alive)


# %%
class ConsensusADMM(Controller):
    """Inexact consensus ADMM (Chang-Hong-Wang, IEEE TSP 2015) on a smoothed relaxation."""
    name = "Consensus-ADMM"

    def __init__(self, inner: int = 6, rho: float = 0.15, alpha: float = 0.30,
                 smooth: float = 90.0, lam_max: float = 500.0):
        self.inner = inner
        self.rho = rho              # consensus / dual weight (dimensionless)
        self.alpha = alpha          # primal step as a fraction of one-step reach
        self.smooth = smooth        # sigmoid width of the coverage relaxation (m)
        self.lam_max = lam_max      # dual-variable clip (metres); see step() for why
        self.lam: Dict[Tuple[int, int], np.ndarray] = {}

    def reset(self, env, pos, rng):
        super().reset(env, pos, rng)
        self.lam = {}

    def _cov_direction(self, env, i, x, ctx, dev) -> np.ndarray:
        """Ascent direction of the smoothed local coverage, as a unit vector.

        Coverage is relaxed to sum_m w_m * sigmoid((R_cov - ||x - d_m||) / smooth), whose
        gradient is -(1/smooth) * sum_m w_m sigma'(z_m) * (x - d_m)/||x - d_m||. The result
        is normalised so that the primal step is expressed in metres, matching the units of
        the consensus term; mixing a raw gradient with a metre-scale dual makes the step size
        meaningless.
        """
        g = np.zeros(2)
        if dev.size:
            D = env.testbed.device_xy[dev]
            wl = ctx.w[dev]
            d = x[None, :] - D
            r = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
            z = (env.R_cov - r) / self.smooth
            sig = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
            coef = -(wl * sig * (1.0 - sig)) / self.smooth
            g = (coef[:, None] * (d / r[:, None])).sum(axis=0)
        # movement penalty pulls back toward where the node already is
        g -= (env.weights.energy / max(env.weights.coverage, 1e-9)) * \
            (x - ctx.prev[i]) / max(env.max_step, 1e-9)
        n = float(np.linalg.norm(g))
        return g / n if n > 1e-12 else np.zeros(2)

    def _consensus_peers(self, i: int, ctx) -> List[int]:
        """Bounded neighbour set for the consensus term.

        `ctx.nbrs[i]` is the raw PHYSICAL neighbour list - at high density (n=800 in the
        same small MEL footprint used for n=50, mean degree reaches several hundred) that
        makes the per-node dual-variable sum grow just as unboundedly as the pre-fix gossip
        view did. Using the same capped set every other controller reasons over
        (`ctx.gossip.views[i].pos`) keeps ADMM's consensus problem the size an ADMM paper
        actually analyses - a bounded neighbourhood - rather than "most of the network."
        """
        return [j for j in ctx.gossip.views[i].pos if j != i]

    def step(self, env, ctx):
        x = ctx.pos.astype(float).copy()
        alive = [int(i) for i in np.flatnonzero(ctx.alive)]
        dev_cache = {i: _sensed_devices(env, ctx.pos[i]) for i in alive}
        peers = {i: self._consensus_peers(i, ctx) for i in alive}
        step_m = self.alpha * env.max_step / max(self.inner, 1)

        for _ in range(self.inner):
            newx = x.copy()
            for i in alive:
                direction = self._cov_direction(env, i, x[i], ctx, dev_cache[i])
                pull = np.zeros(2)
                for j in peers[i]:
                    key = (min(i, j), max(i, j))
                    lam = self.lam.setdefault(key, np.zeros(2))
                    sgn = 1.0 if i < j else -1.0
                    mid = 0.5 * (x[i] + x[j])
                    pull += sgn * lam + (x[i] - mid)
                newx[i] = x[i] + step_m * direction - self.rho * pull
            x = newx
            for i in alive:                      # dual ascent on the consensus residuals
                for j in peers[i]:
                    if i < j:
                        mid = 0.5 * (x[i] + x[j])
                        key = (i, j)
                        # Clip the dual variable to a bounded ball. Plain dual ascent
                        # integrates the primal residual (x[i]-mid) every inner iteration
                        # with no leak; because positions are snapped to a lattice, that
                        # residual never fully vanishes, so unclipped lam undergoes
                        # integrator windup - observed in practice as an overflow to inf/nan
                        # after enough rounds at large, dense node counts. Clipping is the
                        # standard fix (a bounded-dual / proximal variant of consensus ADMM)
                        # and changes nothing while the residual is small.
                        upd = self.lam.get(key, np.zeros(2)) + self.rho * (x[i] - mid)
                        norm = float(np.linalg.norm(upd))
                        if norm > self.lam_max:
                            upd = upd * (self.lam_max / norm)
                        self.lam[key] = upd
            if not np.all(np.isfinite(x)):
                # a numerical excursion slipped through anyway - discard this inner
                # iteration's update rather than let non-finite values reach _apply_move,
                # which would corrupt lattice-index casts downstream.
                x = np.where(np.isfinite(x), x, ctx.pos.astype(float))
                break
        return _apply_move(env, ctx.pos, x, ctx.alive)

    def state_dict(self):
        return {"lam": {"{}_{}".format(*k): v.tolist() for k, v in self.lam.items()}}

    def load_state_dict(self, d):
        self.lam = {tuple(int(t) for t in k.split("_")): np.asarray(v)
                    for k, v in d.get("lam", {}).items()}


# %%
class MARLIndependentQ(Controller):
    """Independent Q-learning with parameter sharing. An adapted MARL baseline, not MAPPO."""
    name = "MARL-IQL"

    N_ACT = 9

    def __init__(self, lr: float = 0.25, gamma: float = 0.85, eps0: float = 0.5,
                 eps_decay: float = 0.985):
        self.lr = lr
        self.gamma = gamma
        self.eps0 = eps0
        self.eps_decay = eps_decay
        self.Q = np.zeros((3 * 3 * 2, self.N_ACT))
        self.eps = eps0
        self._last: Dict[int, Tuple[int, int]] = {}

    def reset(self, env, pos, rng):
        super().reset(env, pos, rng)
        self.eps = self.eps0
        self._last = {}

    def _dirs(self, env):
        s = env.max_step
        base = [(0, 0)]
        for k in range(8):
            a = k * math.pi / 4
            base.append((s * math.cos(a), s * math.sin(a)))
        return np.array(base)

    def _state(self, env, i, ctx, dev):
        v = ctx.gossip.views[i]
        deg = len([j for j in v.pos if j != i])
        dbin = 0 if deg == 0 else (1 if deg <= 3 else 2)
        if dev.size:
            wl = ctx.w[dev].sum()
            ref = max(env.W_total / max(env.n_nodes, 1), 1e-9)
            ubin = 0 if wl < 0.5 * ref else (1 if wl < 1.5 * ref else 2)
        else:
            ubin = 0
        bbin = 0
        return (dbin * 3 + ubin) * 2 + bbin

    def step(self, env, ctx):
        dirs = self._dirs(env)
        target = ctx.pos.copy()
        for i in np.flatnonzero(ctx.alive):
            i = int(i)
            dev = _sensed_devices(env, ctx.pos[i])
            s = self._state(env, i, ctx, dev)
            if ctx.rng.random() < self.eps:
                a = int(ctx.rng.integers(self.N_ACT))
            else:
                a = int(np.argmax(self.Q[s]))
            cand = ctx.pos[i] + dirs[a]
            r = float(local_utility(env, i, cand[None, :], ctx.pos, ctx.prev,
                                    ctx.gossip.views[i], ctx.w, dev)[0])
            if i in self._last:
                ps, pa = self._last[i]
                self.Q[ps, pa] += self.lr * (r + self.gamma * self.Q[s].max() - self.Q[ps, pa])
            self._last[i] = (s, a)
            target[i] = cand
        self.eps = max(0.03, self.eps * self.eps_decay)
        return _apply_move(env, ctx.pos, target, ctx.alive)

    def state_dict(self):
        return {"Q": self.Q.copy(), "eps": self.eps, "last": dict(self._last)}

    def load_state_dict(self, d):
        self.Q = np.asarray(d["Q"])
        self.eps = float(d["eps"])
        self._last = dict(d.get("last", {}))


# %%
class DPOAStyleSwarm(Controller):
    """Centralized D-POA-style swarm over the joint position vector. A reimplementation."""
    name = "D-POA-style (central)"
    decentralized = False
    uses_gossip = False

    def __init__(self, n_particles: int = 12, inner: int = 4, w_inertia: float = 0.65,
                 c1: float = 1.2, c2: float = 1.4):
        self.n_particles = n_particles
        self.inner = inner
        self.w_inertia = w_inertia
        self.c1 = c1
        self.c2 = c2
        self.swarm = None
        self.vel = None
        self.pbest = None
        self.pbest_f = None

    def reset(self, env, pos, rng):
        super().reset(env, pos, rng)
        self.swarm = None

    def _init_swarm(self, env, pos, rng):
        P = self.n_particles
        jitter = rng.normal(0, env.max_step * 0.5, size=(P,) + pos.shape)
        self.swarm = np.clip(pos[None, :, :] + jitter, -1e9, 1e9)
        self.swarm[0] = pos
        self.vel = np.zeros_like(self.swarm)
        self.pbest = self.swarm.copy()
        self.pbest_f = np.full(P, -np.inf)

    def step(self, env, ctx):
        if self.swarm is None or self.swarm.shape[1] != ctx.pos.shape[0]:
            self._init_swarm(env, ctx.pos, ctx.rng)
        rng = ctx.rng
        for _ in range(self.inner):
            fits = np.empty(self.n_particles)
            for p in range(self.n_particles):
                cand = _apply_move(env, ctx.pos, self.swarm[p], ctx.alive)
                fits[p] = env.phi(cand, ctx.prev, ctx.alive, ctx.w)
            better = fits > self.pbest_f
            self.pbest[better] = self.swarm[better]
            self.pbest_f[better] = fits[better]
            gbest = self.pbest[int(np.argmax(self.pbest_f))]
            r1 = rng.random(self.swarm.shape)
            r2 = rng.random(self.swarm.shape)
            self.vel = (self.w_inertia * self.vel
                        + self.c1 * r1 * (self.pbest - self.swarm)
                        + self.c2 * r2 * (gbest[None] - self.swarm))
            self.vel = np.clip(self.vel, -env.max_step, env.max_step)
            self.swarm = self.swarm + self.vel
        gbest = self.pbest[int(np.argmax(self.pbest_f))]
        return _apply_move(env, ctx.pos, gbest, ctx.alive)

    def on_event(self):
        """Dynamic re-initialisation, as D-POA does on topology change."""
        self.swarm = None
        self.pbest_f = None


class StaticDeployment(Controller):
    name = "Static"
    uses_gossip = False

    def step(self, env, ctx):
        return ctx.pos.copy()


class RandomWalk(Controller):
    name = "RandomWalk"
    uses_gossip = False

    def step(self, env, ctx):
        ang = ctx.rng.uniform(0, 2 * math.pi, ctx.pos.shape[0])
        r = ctx.rng.uniform(0, env.max_step, ctx.pos.shape[0])
        target = ctx.pos + np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
        return _apply_move(env, ctx.pos, target, ctx.alive)


ALGORITHMS = {
    "FogField-BR": FogFieldBR,
    "FogField-LLL": FogFieldLLL,
    "GossipGreedy": GossipGreedy,
    "Guerrero-style": GuerreroStyle,
    "Consensus-ADMM": ConsensusADMM,
    "MARL-IQL": MARLIndependentQ,
    "D-POA-style": DPOAStyleSwarm,
    "Static": StaticDeployment,
    "RandomWalk": RandomWalk,
}
DECENTRALIZED = [k for k, v in ALGORITHMS.items() if v.decentralized]
print("algorithms:", ", ".join(ALGORITHMS))
print("decentralized:", ", ".join(DECENTRALIZED))
