"""Fifth one-off patch (15 Oct 2026): the D-POA-style swarm re-evaluates its memory every slot.
Do NOT re-run it.

Found in the v2 controller-loss analysis: on BEJ, losing the D-POA-style controller changed nothing
(identical traces in 9 of 10 units). The cause is a stale particle memory. The swarm kept each
particle's best fitness (and the global best) from the slot in which it was found and never
re-scored it, but on BEJ the demand field changes every slot without any topology event, so the
remembered optimum (fitness ~1.05, found under a busy demand slot) could never be beaten by the
current configuration (~0.27) and the swarm kept steering toward an outdated target. v1.0 had the
same flaw. Dynamic swarm optimisers re-evaluate their memory when the objective changes; v2 does
so at the start of every slot (12 extra objective evaluations per slot). Re-initialisation on
topology events is unchanged.
"""
from pathlib import Path

SRC = Path(__file__).parent / "fogfield_src_fullrun.py"
src = SRC.read_text(encoding="utf-8")
old = '''        if self.swarm is None or self.swarm.shape[1] != ctx.pos.shape[0]:
            self._init_swarm(env, ctx.pos, ctx.rng)
        rng = ctx.rng
        for _ in range(self.inner):'''
new = '''        if self.swarm is None or self.swarm.shape[1] != ctx.pos.shape[0]:
            self._init_swarm(env, ctx.pos, ctx.rng)
        elif self.pbest_f is not None and np.isfinite(self.pbest_f).any():
            # re-evaluate the memory under this slot's demand and origin (v2): without it the
            # swarm chases optima remembered from earlier demand slots
            for p in range(self.n_particles):
                cand = _apply_move(env, ctx.pos, self.pbest[p], ctx.alive)
                self.pbest_f[p] = env.phi(cand, ctx.pos, ctx.alive, ctx.w)
        rng = ctx.rng
        for _ in range(self.inner):'''
assert src.count(old) == 1
src = src.replace(old, new)
SRC.write_text(src, encoding="utf-8")
print("patched", SRC)
