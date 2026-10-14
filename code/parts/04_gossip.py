# %% [markdown]
# ## 8 - Gossip layer with *measured* overhead (R3)
#
# The plan is explicit that communication cost must be **measured, not modelled**. Inside
# Colab the honest version of that is to run the real protocol and count the real bytes: each
# gossip message is genuinely serialised with `struct`, and the layer reports
# `len(payload)` - not a formula.
#
# **Message format** (little-endian, packed):
#
# | field | type | bytes |
# |---|---|---|
# | sender id | `uint16` | 2 |
# | sequence | `uint16` | 2 |
# | position x, y | `float32` x2 | 8 |
# | residual battery | `float32` | 4 |
# | local demand hint | `float32` | 4 |
# | neighbour-list length | `uint8` | 1 |
# | neighbour ids | `uint16` x k | 2k |
#
# Push-pull: an exchange with a neighbour costs one push plus one pull message. The
# neighbour-id list is what gives every node a **2-hop view**, which is what the local
# connectivity term needs in order to tell "I am a bridge" from "I am redundant".
#
# > **Scope, stated so it is not mistaken for more than it is.** These are application-layer
# > payload bytes. 802.11 MAC framing, ACKs, retransmission, contention and loss are **not**
# > here; they belong to the Mininet-WiFi/ns-3 tier and the physical testbed. The per-round
# > message and byte traces exported below are the inputs that tier needs.

# %%
_HDR = struct.Struct("<HHffffB")          # id, seq, x, y, battery, demand hint, nbr count


@dataclass
class LocalView:
    """What one node knows: neighbour states and their neighbour lists (2-hop)."""
    pos: Dict[int, np.ndarray] = field(default_factory=dict)
    battery: Dict[int, float] = field(default_factory=dict)
    demand_hint: Dict[int, float] = field(default_factory=dict)
    nbrs_of: Dict[int, List[int]] = field(default_factory=dict)
    last_seen: Dict[int, int] = field(default_factory=dict)

    def forget_older_than(self, now: int, ttl: int) -> None:
        stale = [j for j, t in self.last_seen.items() if now - t > ttl]
        for j in stale:
            self.pos.pop(j, None)
            self.battery.pop(j, None)
            self.demand_hint.pop(j, None)
            self.nbrs_of.pop(j, None)
            self.last_seen.pop(j, None)


class GossipLayer:
    """Push-pull gossip for neighbour discovery and demand hints, with real byte accounting."""

    def __init__(self, n_nodes: int, R_comm: float, fanout: int = 3, ttl: int = 4,
                 max_nbr_ids: int = 12):
        self.n = n_nodes
        self.R_comm = R_comm
        self.fanout = fanout
        self.ttl = ttl
        self.max_nbr_ids = max_nbr_ids
        self.views = [LocalView() for _ in range(n_nodes)]
        self.seq = np.zeros(n_nodes, dtype=np.int64)
        self.bytes_total = 0
        self.msgs_total = 0
        self.per_round: List[Tuple[int, int]] = []      # (messages, bytes) per round

    # ---- wire format ------------------------------------------------------
    def encode(self, i: int, pos_i, battery_i: float, hint_i: float,
               nbr_ids: List[int]) -> bytes:
        ids = list(nbr_ids)[: self.max_nbr_ids]
        head = _HDR.pack(int(i) & 0xFFFF, int(self.seq[i]) & 0xFFFF,
                         float(pos_i[0]), float(pos_i[1]),
                         float(battery_i), float(hint_i), len(ids))
        return head + b"".join(struct.pack("<H", int(j) & 0xFFFF) for j in ids)

    @staticmethod
    def decode(buf: bytes):
        sid, seq, x, y, bat, hint, k = _HDR.unpack_from(buf, 0)
        off = _HDR.size
        ids = [struct.unpack_from("<H", buf, off + 2 * t)[0] for t in range(k)]
        return sid, seq, np.array([x, y]), bat, hint, ids

    # ---- protocol ---------------------------------------------------------
    def neighbours(self, pos: np.ndarray, alive: np.ndarray) -> List[np.ndarray]:
        idx = np.flatnonzero(alive)
        out = [np.empty(0, dtype=int) for _ in range(self.n)]
        if idx.size == 0:
            return out
        P = pos[idx]
        d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        within = d2 <= self.R_comm ** 2
        np.fill_diagonal(within, False)
        for k, i in enumerate(idx):
            out[i] = idx[within[k]]
        return out

    def round(self, rnd: int, pos: np.ndarray, alive: np.ndarray, battery: np.ndarray,
              demand_hint: np.ndarray, rng: np.random.Generator,
              pos_report: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """One push-pull gossip round. Returns each node's current neighbour array.

        `pos` is physical truth and decides who can hear whom. `pos_report` is what each node
        *claims* on the wire, which a Byzantine node falsifies. Keeping the two separate is
        what makes the Byzantine scenario meaningful: a liar is still where it really is.
        """
        if pos_report is None:
            pos_report = pos
        nbrs = self.neighbours(pos, alive)
        msgs = 0
        nbytes = 0
        self.seq[alive] += 1

        for i in np.flatnonzero(alive):
            cand = nbrs[i]
            if cand.size == 0:
                continue
            k = min(self.fanout, cand.size)
            peers = rng.choice(cand, size=k, replace=False)
            push = self.encode(i, pos_report[i], battery[i], demand_hint[i],
                               [int(j) for j in cand])
            for j in peers:
                j = int(j)
                # push i -> j
                msgs += 1
                nbytes += len(push)
                self._absorb(j, push, rnd)
                # pull j -> i
                pull = self.encode(j, pos_report[j], battery[j], demand_hint[j],
                                   [int(q) for q in nbrs[j]])
                msgs += 1
                nbytes += len(pull)
                self._absorb(i, pull, rnd)

        # Direct neighbour beacons keep the 1-hop view fresh even at fanout 0.
        #
        # Capped to the `max_nbr_ids` NEAREST physical neighbours, matching the cap already
        # applied to what goes out on the wire. Without this cap, a node's `view.pos` grows
        # to its full physical degree - fine at the densities the main grid tests, but at
        # n=800 in the same small MEL footprint mean degree reaches ~445 nodes, and the
        # local connectivity marginal in `local_utility` does an O(d^2) pure-Python
        # neighbour-pair pass per candidate move. That measured 0.3s/node at n=800
        # (~85s/round with ~280 active nodes), making the scalability tier's 800-node point
        # impractical - not because the underlying idea is expensive, but because an
        # uncapped local view let "local" silently mean "most of the network." A real
        # bounded-memory device could not keep an unlimited neighbour table either, so this
        # is a realism fix as much as a performance one.
        for i in np.flatnonzero(alive):
            cand = nbrs[i]
            if cand.size > self.max_nbr_ids:
                d2 = ((pos[cand] - pos[i][None, :]) ** 2).sum(1)
                cand = cand[np.argsort(d2)[: self.max_nbr_ids]]
            v = self.views[i]
            for j in cand:
                j = int(j)
                v.pos[j] = pos_report[j].copy()
                v.last_seen[j] = rnd

        for i in range(self.n):
            self.views[i].forget_older_than(rnd, self.ttl)

        self.msgs_total += msgs
        self.bytes_total += nbytes
        self.per_round.append((msgs, nbytes))
        return nbrs

    def _absorb(self, receiver: int, buf: bytes, rnd: int) -> None:
        sid, _seq, p, bat, hint, ids = self.decode(buf)
        v = self.views[receiver]
        if sid == receiver:
            return
        v.pos[sid] = p
        v.battery[sid] = bat
        v.demand_hint[sid] = hint
        v.nbrs_of[sid] = ids
        v.last_seen[sid] = rnd

    def stats(self) -> dict:
        rounds = max(len(self.per_round), 1)
        m = np.array([a for a, _ in self.per_round], dtype=float)
        b = np.array([b for _, b in self.per_round], dtype=float)
        return {
            "gossip_msgs_total": int(self.msgs_total),
            "gossip_bytes_total": int(self.bytes_total),
            "gossip_msgs_per_round": float(m.mean()) if m.size else 0.0,
            "gossip_bytes_per_round": float(b.mean()) if b.size else 0.0,
            "gossip_bytes_per_node_round": float(b.mean() / self.n) if b.size else 0.0,
            "gossip_rounds": rounds,
        }


# %% [markdown]
# ## 9 - Local (neighbour-only) utility
#
# The theory in section 7 uses the **global** marginal contribution. A decentralized node
# cannot evaluate that - it only has what gossip gave it. `local_utility` is the estimator a
# real node computes:
#
# * **Coverage** - only devices the node can sense (within `R_cov + max_step`), and only the
#   weight it would add *on top of what its known neighbours already cover*. That subtraction
#   is the local image of the wonderful-life utility.
# * **Connectivity** - the marginal change in the largest component of the **local induced
#   graph**, built from 1-hop positions and 2-hop neighbour lists carried by gossip. This is
#   what lets a node notice it is an articulation point.
# * **Energy** - exact, since a node knows its own displacement and load.
#
# The difference between this estimator and the global utility is the **locality gap**, and
# it is measured rather than assumed - `analyze_locality_gap` reports it.

# %%
def _largest_component_size(adj: Dict[int, set], nodes: List[int]) -> int:
    seen, best = set(), 0
    for s in nodes:
        if s in seen:
            continue
        stack, comp = [s], 0
        seen.add(s)
        while stack:
            u = stack.pop()
            comp += 1
            for v in adj.get(u, ()):  # noqa: E1133
                if v not in seen and v in adj:
                    seen.add(v)
                    stack.append(v)
        best = max(best, comp)
    return best


def local_utility(env: Environment, i: int, cands: np.ndarray, pos: np.ndarray,
                  prev: np.ndarray, view: LocalView, w: np.ndarray,
                  dev_idx: Optional[np.ndarray] = None) -> np.ndarray:
    """Utility of each candidate position for node `i`, from local information only."""
    tb = env.testbed
    R2 = env.R_cov ** 2

    # --- devices this node can sense -------------------------------------
    if dev_idx is None:
        d2c = ((tb.device_xy - pos[i][None, :]) ** 2).sum(1)
        dev_idx = np.flatnonzero(d2c <= (env.R_cov + env.max_step) ** 2)
    if dev_idx.size == 0:
        cov_gain = np.zeros(len(cands))
    else:
        D = tb.device_xy[dev_idx]
        wl = w[dev_idx]
        nbr_ids = [j for j in view.pos if j != i]
        if nbr_ids:
            NP = np.stack([view.pos[j] for j in nbr_ids], axis=0)
            covered_by_nbr = (((NP[:, None, :] - D[None, :, :]) ** 2).sum(-1) <= R2).any(0)
        else:
            covered_by_nbr = np.zeros(len(D), dtype=bool)
        free_w = np.where(covered_by_nbr, 0.0, wl)
        inr = ((cands[:, None, :] - D[None, :, :]) ** 2).sum(-1) <= R2
        cov_gain = inr @ free_w

    # --- local connectivity marginal --------------------------------------
    nbr_ids = sorted(j for j in view.pos if j != i)
    if not nbr_ids:
        conn_gain = np.zeros(len(cands))
        conn_gain[:] = 1.0                    # a lone node is its own component
    else:
        base_adj: Dict[int, set] = {j: set() for j in nbr_ids}
        NP = np.stack([view.pos[j] for j in nbr_ids], axis=0)
        dj2 = ((NP[:, None, :] - NP[None, :, :]) ** 2).sum(-1)
        Rc2 = env.R_comm ** 2
        for a, ja in enumerate(nbr_ids):
            for b, jb in enumerate(nbr_ids):
                if a != b and dj2[a, b] <= Rc2:
                    base_adj[ja].add(jb)
        for j in nbr_ids:                      # fold in 2-hop lists from gossip
            for q in view.nbrs_of.get(j, ()):  # noqa: E1133
                if q in base_adj:
                    base_adj[j].add(q)
                    base_adj[q].add(j)
        without_i = _largest_component_size(base_adj, nbr_ids)
        dist2 = ((cands[:, None, :] - NP[None, :, :]) ** 2).sum(-1)
        conn_gain = np.empty(len(cands))
        for c in range(len(cands)):
            adj = {j: set(base_adj[j]) for j in nbr_ids}
            adj[i] = set()
            for b, jb in enumerate(nbr_ids):
                if dist2[c, b] <= Rc2:
                    adj[i].add(jb)
                    adj[jb].add(i)
            conn_gain[c] = _largest_component_size(adj, nbr_ids + [i]) - without_i

    # --- own energy --------------------------------------------------------
    dist = np.linalg.norm(cands - prev[i][None, :], axis=1)
    e_fly = env.uav.flight_energy(dist, env.dt_s)
    e_cpu = env.uav.compute_energy(
        np.clip(cov_gain / max(env.W_total / max(env.n_nodes, 1), 1e-9), 0, 1), env.dt_s)
    energy = e_fly + e_cpu

    N = float(env.n_nodes)
    return (env.weights.coverage * cov_gain / max(env.W_total, 1e-9)
            + env.weights.connectivity * conn_gain / N
            - env.weights.energy * energy / (N * env.E_ref))
