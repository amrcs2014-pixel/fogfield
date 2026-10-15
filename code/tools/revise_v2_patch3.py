"""Third one-off patch (15 Oct 2026), applied before the v2 grid ran. Do NOT re-run it.

Found while restating Theorem T1 for the implemented dynamics (referee C2):

  1. Station-keeping by loitering. In the Zeng-Xu-Zhang model hovering (168.5 W) costs more
     than slow forward flight (126.0 W at the maximum-endurance speed V_me = 10.2 m/s). v1.0
     charged a node that stays put the hover power, so a node could lower its energy term by
     moving back and forth: no profile was a rest point, and the finite-improvement argument
     behind T1 failed across slots. v2 lets a node that needs to cover less than V_me * dt
     loiter at V_me (Zeng et al. define V_me for exactly this purpose), so the slot energy is
     P(max(d/dt, V_me)) * dt: non-decreasing in the distance d and minimal for staying.
  2. Energy origin. The decision rules evaluated a candidate move's flight energy from the
     node's position one slot earlier (`prev`) although the move starts from the current
     position; the realised energy in `simulate` was (correctly) charged from the current
     position. Fixed in `local_utility`, the D-POA-style fitness and the ADMM movement penalty.
  3. Best response breaks ties in favour of staying, so a rest point is absorbing (T1).
  4. The equilibrium diagnostics evaluate the slot game at the reached configuration with
     origin equal to that configuration (rest-point semantics, the object T1 is about).
"""
from pathlib import Path

SRC = Path(__file__).parent / "fogfield_src_fullrun.py"
src = SRC.read_text(encoding="utf-8")
orig = src
n_applied = 0


def rep(old, new, count=1):
    global src, n_applied
    k = src.count(old)
    assert k == count, "expected {} occurrence(s), found {} for:\n{}".format(count, k, old[:200])
    src = src.replace(old, new)
    n_applied += 1


rep('''from dataclasses import dataclass, field
''', '''from dataclasses import dataclass, field
from functools import cached_property
''')

rep('''    battery_Wh: float = 275.0

    def propulsion(self, V):''', '''    battery_Wh: float = 275.0
    loiter: bool = True      # station-keeping by loitering at V_me rather than hovering (v2)

    def propulsion(self, V):''')

rep('''    def flight_energy(self, dist_m, dt_s: float):
        """Energy (J) to travel `dist_m` within a step of `dt_s`, hovering for the remainder.

        The UAV flies at the slowest speed that covers the distance in time, capped at
        `V_max`; slower flight is cheaper per second but the step budget is fixed.
        """
        dist = np.asarray(dist_m, dtype=np.float64)
        v_need = dist / max(dt_s, 1e-9)
        V = np.clip(v_need, 0.0, self.V_max)''', '''    @cached_property
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
        V = np.clip(v_need, 0.0, self.V_max)''')

rep('''print("battery       = {:.0f} Wh = {:.0f} kJ -> {:.1f} min of hover".format(
    UAV.battery_Wh, UAV.battery_Wh * 3600 / 1000, UAV.battery_Wh * 3600 / UAV.P_hover / 60))''',
    '''print("battery       = {:.0f} Wh = {:.0f} kJ -> {:.1f} min of hover".format(
    UAV.battery_Wh, UAV.battery_Wh * 3600 / 1000, UAV.battery_Wh * 3600 / UAV.P_hover / 60))
print("loiter (v2)   : V_me = {:.2f} m/s, P(V_me) = {:.2f} W; a station-keeping node loiters "
      "at V_me, so staying is the cheapest move".format(UAV.V_me, UAV.P_me))
_e = UAV.flight_energy(np.array([0.0, 120.0, 306.0, 400.0, 600.0]), 30.0)
assert np.all(np.diff(_e) >= -1e-9), "slot energy must be non-decreasing in distance"''')

# 2. energy origin = current position
rep('''    dist = np.linalg.norm(cands - prev[i][None, :], axis=1)''',
    '''    # flight energy of the move this node is deciding now: from its CURRENT position
    # (v1.0 used prev[i], the position one slot earlier)
    dist = np.linalg.norm(cands - pos[i][None, :], axis=1)''')
rep('''                fits[p] = env.phi(cand, ctx.prev, ctx.alive, ctx.w)''',
    '''                fits[p] = env.phi(cand, ctx.pos, ctx.alive, ctx.w)''')
rep('''            (x - ctx.prev[i]) / max(env.max_step, 1e-9)''',
    '''            (x - ctx.pos[i]) / max(env.max_step, 1e-9)''')

# 3. best response: ties go to staying
rep('''            self.t_decide += time.perf_counter() - t0
            self.n_decide += 1
            target[i] = cands[int(np.argmax(u))]''',
    '''            self.t_decide += time.perf_counter() - t0
            self.n_decide += 1
            best = int(np.argmax(u))
            here = np.flatnonzero(np.all(np.abs(cands - ctx.pos[i][None, :]) < 1e-6, axis=1))
            if here.size and u[here[0]] >= u[best] - 1e-12:
                best = int(here[0])          # ties go to staying, so rest points are absorbing
            target[i] = cands[best]''')

# 4. rest-point semantics for the equilibrium diagnostics
rep('''    ng = nash_gap(env, pos, prev, active, w)
    lg = locality_gap(env, pos, prev, active, w, gz, n_sample=16)''',
    '''    # rest-point semantics: the slot game at the reached configuration, moves charged from it
    ng = nash_gap(env, pos, pos, active, w)
    lg = locality_gap(env, pos, pos, active, w, gz, n_sample=16)''')

assert src != orig
SRC.write_text(src, encoding="utf-8")
print("applied {} replacement(s) -> {}".format(n_applied, SRC))
