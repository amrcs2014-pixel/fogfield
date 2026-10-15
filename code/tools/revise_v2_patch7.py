"""Seventh one-off patch (15 Oct 2026): harder controller-loss scenarios. Do NOT re-run it.

The first controller-loss experiment (`controller_loss` / `late_surge`) showed no measurable cost
of losing the centralised controller, because after the loss nothing forced the deployment to keep
adapting. Two harder pairs, each with a controller-alive twin that is an identical run for every
decentralised method:

  ctrl_fail  / late_fail   the controller dies at slot 75 (twin: stays alive); from slot 75 on, each
                           live node fails with probability 1 % per slot (same events in both)
  ctrl_drift / late_drift  the controller dies at slot 75 (twin: stays alive); from slot 75 on, a
                           demand hotspot (peak 4x, width 0.75 R_cov) sweeps the region twice, so
                           the optimal deployment keeps moving on both testbeds
"""
from pathlib import Path

SRC = Path(__file__).parent / "fogfield_src_fullrun.py"
src = SRC.read_text(encoding="utf-8")
n = 0


def rep(old, new):
    global src, n
    assert src.count(old) == 1, old[:80]
    src = src.replace(old, new)
    n += 1


rep('''    "late_surge": Scenario("late_surge",
                           "demand surge ten rounds after the midpoint (control twin of controller_loss)"),
}
''', '''    "late_surge": Scenario("late_surge",
                           "demand surge ten rounds after the midpoint (control twin of controller_loss)"),
    "ctrl_fail": Scenario("ctrl_fail", "controller dies at the midpoint; node failures (1%/slot) follow"),
    "late_fail": Scenario("late_fail", "node failures (1%/slot) from the midpoint (twin of ctrl_fail)"),
    "ctrl_drift": Scenario("ctrl_drift", "controller dies at the midpoint; a demand hotspot sweeps the region"),
    "late_drift": Scenario("late_drift", "demand hotspot sweeps the region from the midpoint (twin of ctrl_drift)"),
}
''')

rep('''    def controller_dead(self, rnd: int) -> bool:
        return self.key == "controller_loss" and rnd >= self.rounds // 2
''', '''    def controller_dead(self, rnd: int) -> bool:
        return self.key in ("controller_loss", "ctrl_fail", "ctrl_drift") and rnd >= self.rounds // 2

    def drift_centre(self, rnd: int) -> np.ndarray:
        """Hotspot centre for the drift scenarios: two sweeps over the second half of the run."""
        x0, x1, y0, y1 = self.env.testbed.bounds
        u = 2.0 * (rnd - self.rounds // 2) / max(self.rounds - self.rounds // 2, 1)
        return np.array([x0 + (x1 - x0) * (u % 1.0),
                         y0 + (y1 - y0) * (0.5 + 0.35 * math.sin(2.0 * math.pi * u))])
''')

rep('''        if self.key in ("flash_crowd", "controller_loss", "late_surge"):
            lo, hi = self.flash_window''', '''        if self.key in ("ctrl_drift", "late_drift") and rnd >= self.rounds // 2:
            c = self.drift_centre(rnd)
            d2 = ((tb.device_xy - c[None, :]) ** 2).sum(1)
            w = w * (1.0 + 3.0 * np.exp(-d2 / (2 * (0.75 * self.env.R_cov) ** 2)))

        if self.key in ("flash_crowd", "controller_loss", "late_surge"):
            lo, hi = self.flash_window''')

rep('''        elif self.key == "controller_loss" and rnd == self.rounds // 2:
            ev = "controller_loss"''', '''        elif self.key == "controller_loss" and rnd == self.rounds // 2:
            ev = "controller_loss"
        elif self.key in ("ctrl_fail", "late_fail") and rnd >= self.rounds // 2:
            draw = rng.random(live.size)              # drawn in both twins, so the streams match
            if live.size > 2:
                die = live[draw < P_FAIL]
                if die.size:
                    alive[die] = False
                    ev = "fail:{}".format(die.size)
            if self.key == "ctrl_fail" and rnd == self.rounds // 2:
                ev = "controller_loss" + ("+" + ev if ev else "")
        elif self.key in ("ctrl_drift",) and rnd == self.rounds // 2:
            ev = "controller_loss"''')

SRC.write_text(src, encoding="utf-8")
print("applied", n, "replacements")
