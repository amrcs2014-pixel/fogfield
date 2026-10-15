"""Checks of the v2 seeding (common random numbers) on tiny units. Not part of the grid.

    python test_crn.py

1. The same unit run twice in this process gives bit-identical traces.
2. Every algorithm draws the same initial deployment for a given (testbed, n, seed).
3. Static under `byzantine` equals Static under `static` exactly (null row), and so does
   RandomWalk (it ignores gossip): the adversary stream does not leak into the algorithm stream.
4. FogField-BR under `byzantine` and `static` share the deployment and diverge only later.
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None

os.environ["FOGFIELD_BUDGET_MIN"] = "0"
os.environ["FOGFIELD_STOP_AFTER_EXPERIMENTS"] = "1"
SRC = Path(__file__).parent / "fogfield_src_fullrun.py"
g = sys.modules["__main__"].__dict__
g["__file__"] = str(SRC)
try:
    exec(compile(SRC.read_text(encoding="utf-8"), str(SRC), "exec"), g)
except SystemExit:
    pass

np = g["np"]
simulate, make_env, TESTBEDS = g["simulate"], g["make_env"], g["TESTBEDS"]
seed_streams, initial_positions, ALGORITHMS = g["seed_streams"], g["initial_positions"], g["ALGORITHMS"]

ok = True
for tb_name in TESTBEDS:
    env = make_env(TESTBEDS[tb_name], 20)
    # 1. determinism within a process
    r1 = simulate(env, "FogField-BR", "static", 3, 20, ckpt_every=50, resume=False)
    r2 = simulate(env, "FogField-BR", "static", 3, 20, ckpt_every=50, resume=False)
    same = r1["trace_phi"] == r2["trace_phi"] and r1["final_pos"] == r2["final_pos"]
    print(tb_name, "1. repeat identical:", same); ok &= same
    # 2. common deployment across algorithms
    P = [initial_positions(env, seed_streams(tb_name, 20, 3, a)["deploy"]) for a in ALGORITHMS]
    same = all(np.array_equal(P[0], p) for p in P)
    print(tb_name, "2. same deployment for all algorithms:", same); ok &= same
    # 3. null rows under byzantine
    for algo in ("Static", "RandomWalk"):
        s = simulate(env, algo, "static", 3, 20, ckpt_every=50, resume=False)
        b = simulate(env, algo, "byzantine", 3, 20, ckpt_every=50, resume=False)
        same = s["trace_phi"] == b["trace_phi"] and s["final_pos"] == b["final_pos"]
        print(tb_name, "3.", algo, "byzantine == static:", same); ok &= same
    # 4. FogField-BR shares the deployment, then diverges
    s = simulate(env, "FogField-BR", "static", 3, 20, ckpt_every=50, resume=False)
    b = simulate(env, "FogField-BR", "byzantine", 3, 20, ckpt_every=50, resume=False)
    print(tb_name, "4. FogField-BR phi static vs byzantine (first 3 rounds):",
          s["trace_phi"][:3], b["trace_phi"][:3], "final", round(s["phi_final"], 4), round(b["phi_final"], 4))
    # 5. controller loss: D-POA-style holds position after the midpoint
    c = simulate(env, "D-POA-style", "controller_loss", 3, 20, ckpt_every=50, resume=False)
    print(tb_name, "5. controller_loss events:", c["events"], "ctrl B/round", round(c["ctrl_bytes_per_round"], 1),
          "phi pre/post", round(c["phi_pre_mid"], 3), round(c["phi_post_mid"], 3), "view", round(c["view_size_mean"], 2))
    # 6. late_surge is the controller-alive twin: decentralised traces must be identical
    for algo in ("FogField-BR", "GossipGreedy"):
        x = simulate(env, algo, "controller_loss", 3, 30, ckpt_every=50, resume=False)
        y = simulate(env, algo, "late_surge", 3, 30, ckpt_every=50, resume=False)
        same = x["trace_phi"] == y["trace_phi"]
        print(tb_name, "6.", algo, "controller_loss == late_surge:", same); ok &= same
    x = simulate(env, "D-POA-style", "controller_loss", 3, 30, ckpt_every=50, resume=False)
    y = simulate(env, "D-POA-style", "late_surge", 3, 30, ckpt_every=50, resume=False)
    print(tb_name, "   D-POA-style phi post-mid: dead", round(x["phi_post_mid"], 4), "alive", round(y["phi_post_mid"], 4),
          "| peak node energy kJ", round(y["energy_node_max_J"] / 1e3, 1))
    d = simulate(env, "FogField-BR", "controller_loss", 3, 20, ckpt_every=50, resume=False)
    print(tb_name, "   FogField-BR under controller_loss: ctrl bytes", d["ctrl_bytes_total"],
          "recovery", d["recovery_rounds"], d["recovery_censored"])
    # 7. harder controller-loss pairs: decentralised traces identical in the twins; D-POA differs
    for a_, b_ in (("ctrl_fail", "late_fail"), ("ctrl_drift", "late_drift")):
        for algo in ("FogField-BR", "GossipGreedy", "Static"):
            x = simulate(env, algo, a_, 3, 30, ckpt_every=50, resume=False)
            y = simulate(env, algo, b_, 3, 30, ckpt_every=50, resume=False)
            same = x["trace_phi"] == y["trace_phi"]
            print(tb_name, "7.", algo, a_, "==", b_, ":", same); ok &= same
        x = simulate(env, "D-POA-style", a_, 3, 30, ckpt_every=50, resume=False)
        y = simulate(env, "D-POA-style", b_, 3, 30, ckpt_every=50, resume=False)
        print(tb_name, "   D-POA-style", a_, "phi_post", round(x["phi_post_mid"], 4), "| twin", round(y["phi_post_mid"], 4),
              "| alive end", x["alive_final"], y["alive_final"])
print("ALL OK" if ok else "FAILURES")
