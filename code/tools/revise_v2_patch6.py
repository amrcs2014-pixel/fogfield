"""Sixth one-off patch (15 Oct 2026), on top of patch 5. Do NOT re-run it.

D-POA re-initialises its population when it detects an environmental change. v1.0 (and patch 5)
did this only on topology events. v2 also treats a change of the demand field as an environmental
change (every slot on BEJ, the flash-crowd and surge windows on MEL) and re-initialises the swarm
around the current configuration; between changes the memory is re-evaluated (patch 5). In a
pilot on BEJ static (n = 20 and 50, seeds 0-1) this was the stronger of the two variants in three
of four units (n = 50: about 0.60 against 0.51), so the centralised reference uses it.
"""
from pathlib import Path

SRC = Path(__file__).parent / "fogfield_src_fullrun.py"
src = SRC.read_text(encoding="utf-8")
old = '''        if self.swarm is None or self.swarm.shape[1] != ctx.pos.shape[0]:
            self._init_swarm(env, ctx.pos, ctx.rng)
        elif self.pbest_f is not None and np.isfinite(self.pbest_f).any():'''
new = '''        # environmental change (v2): a new demand field re-initialises the swarm around the
        # current configuration, as D-POA does on detected change
        w_last = getattr(self, "_w_last", None)
        if w_last is not None and (w_last.shape != ctx.w.shape or not np.array_equal(w_last, ctx.w)):
            self.swarm = None
        self._w_last = np.array(ctx.w, copy=True)
        if self.swarm is None or self.swarm.shape[1] != ctx.pos.shape[0]:
            self._init_swarm(env, ctx.pos, ctx.rng)
        elif self.pbest_f is not None and np.isfinite(self.pbest_f).any():'''
assert src.count(old) == 1
src = src.replace(old, new)
old2 = '''    def on_event(self):
        """Dynamic re-initialisation, as D-POA does on topology change."""
        self.swarm = None
        self.pbest_f = None'''
new2 = '''    def on_event(self):
        """Dynamic re-initialisation, as D-POA does on topology change (and, in v2, on a
        change of the demand field, see step)."""
        self.swarm = None
        self.pbest_f = None'''
assert src.count(old2) == 1
src = src.replace(old2, new2)
SRC.write_text(src, encoding="utf-8")
print("patched", SRC)
