#!/usr/bin/env python3
"""Run the ns-3 emulation grid inside WSL. Resumable: a run whose JSON exists is skipped.

    python3 run_grid.py [workers]

Traces are copied from the Windows project folder into ~/fogfield-emu-out/traces (paths with
spaces do not survive the wsl.exe hand-off, and local ext4 is faster); results are written to
~/fogfield-emu-out/runs and mirrored to <project>/emu/runs.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

EMU_WIN = os.environ.get("FOGFIELD_EMU") or os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
TR = os.path.join(HOME, "fogfield-emu-out", "traces")
OUT = os.path.join(HOME, "fogfield-emu-out", "runs")
MIRROR = os.path.join(EMU_WIN, "runs")
BIN = os.path.join(HOME, "ns-3-fogfield", "build", "scratch", "ns3.48-fogfield-emu-optimized")
LIB = os.path.join(HOME, "ns-3-fogfield", "build", "lib")
TXP = {"MEL": 17.1, "BEJ": 21.8}          # fading channel: single-attempt PDR 0.5 at R_comm
STRIDE = {50: 1, 100: 1, 200: 5, 400: 15, 800: 25}   # scalability study (75 slots)
TMP_SUFFIX = "r" if "--reverse" in sys.argv else ""


def runs():
    out = []
    # 1. main grid, fading first (it feeds the loss study), then disk (FogField-BR and the
    #    centralized reference only: the disk channel isolates MAC effects, one gossip method suffices)
    for ch in ("fading", "disk"):
        algos = (("FogField-BR", "FogField-LLL", "GossipGreedy", "D-POA-style") if ch == "fading"
                 else ("FogField-BR", "D-POA-style"))
        for tb in ("MEL", "BEJ"):
            for n in (20, 50):
                for seed in range(5):
                    for algo in algos:
                        mode = "controller" if algo == "D-POA-style" else "gossip"
                        out.append(dict(trace="main_{}_static_{}_n{}_s{}.trace".format(tb, algo, n, seed),
                                        mode=mode, channel=ch, tb=tb, window=1.0, stride=1))
    # 2. scalability on BEJ
    for n in (50, 100, 200, 400, 800):
        for ch in ("disk", "fading"):
            for seed in (range(3) if n <= 200 else range(1)):   # n >= 400: one seed (cost ~ n^2)
                for algo in ("FogField-BR", "D-POA-style"):
                    mode = "controller" if algo == "D-POA-style" else "gossip"
                    out.append(dict(trace="scalability_BEJ_static_{}_n{}_s{}.trace".format(algo, n, seed),
                                    mode=mode, channel=ch, tb="BEJ", window=1.0, stride=STRIDE[n]))
    # 3. sensitivity: a 5 s traffic window instead of 1 s
    for tb in ("MEL", "BEJ"):
        for seed in range(2):
            for algo in ("FogField-BR", "D-POA-style"):
                mode = "controller" if algo == "D-POA-style" else "gossip"
                out.append(dict(trace="main_{}_static_{}_n50_s{}.trace".format(tb, algo, seed),
                                mode=mode, channel="fading", tb=tb, window=5.0, stride=1))
    return out


def runs_v2():
    """Second set of runs (--v2)."""
    out = []
    # large n, re-sampled: skip the initial deployment (offset 5) and replicate. n = 200: three
    # trace seeds; n = 400 and 800: one trace seed, three ns-3 runs. Both methods, both channels.
    for n in (200, 400, 800):
        reps = [(s, 1) for s in range(3)] if n == 200 else [(0, k) for k in (1, 2, 3)]
        for ch in ("fading", "disk"):
            for seed, run in reps:
                for algo in ("D-POA-style", "FogField-BR"):
                    mode = "controller" if algo == "D-POA-style" else "gossip"
                    out.append(dict(trace="scalability_BEJ_static_{}_n{}_s{}.trace".format(algo, n, seed),
                                    mode=mode, channel=ch, tb="BEJ", window=1.0, stride=STRIDE[n],
                                    offset=5, run=run))
        # the controller's traffic spread over 10 s of the slot instead of 1 s (fading)
        for seed, run in reps:
            out.append(dict(trace="scalability_BEJ_static_D-POA-style_n{}_s{}.trace".format(n, seed),
                            mode="controller", channel="fading", tb="BEJ", window=10.0, stride=STRIDE[n],
                            offset=5, run=run))
    # n = 50 and 100 on the same footing as n >= 200: from slot 5 on, with the controller log
    if "--v2b" in sys.argv:
        out = []
        for n in (50, 100):
            for ch in ("fading", "disk"):
                for seed in range(3):
                    for algo in ("D-POA-style", "FogField-BR"):
                        mode = "controller" if algo == "D-POA-style" else "gossip"
                        out.append(dict(trace="scalability_BEJ_static_{}_n{}_s{}.trace".format(algo, n, seed),
                                        mode=mode, channel=ch, tb="BEJ", window=1.0, stride=1,
                                        offset=5, run=1))
        return out
    # the reference's cost on its own mesh-only trajectories (same runs as its Phi in the loss study)
    for ch in ("fading", "disk"):
        for n in (20, 50):
            for seed in range(5):
                out.append(dict(trace="meshonly_BEJ_static_D-POA-style_n{}_s{}.trace".format(n, seed),
                                mode="controller", channel=ch, tb="BEJ", window=1.0, stride=1,
                                offset=0, run=1))
    return out


def name(r):
    base = r["trace"].replace(".trace", "")
    tail = ""
    if r.get("offset", 0) or r.get("run", 1) != 1:
        tail = "_o{}_r{}".format(r.get("offset", 0), r.get("run", 1))
    return "{}__{}_{}_w{:g}{}.json".format(base, r["mode"], r["channel"], r["window"], tail)


def run_one(r):
    dst = os.path.join(OUT, name(r))
    if os.path.exists(dst):
        return name(r), "exists", 0.0
    src = os.path.join(EMU_WIN, "traces", r["trace"])
    loc = os.path.join(TR, r["trace"])
    if not os.path.exists(loc):
        if not os.path.exists(src):
            return name(r), "no-trace", 0.0
        shutil.copy(src, loc + ".tmp")
        os.replace(loc + ".tmp", loc)
    tmp = dst + ".tmp" + TMP_SUFFIX     # distinct per pass: two passes may meet on one run
    args = [BIN, "--trace=" + loc, "--mode=" + r["mode"], "--channel=" + r["channel"],
            "--txPowerDbm={}".format(TXP[r["tb"]]), "--window={}".format(r["window"]),
            "--slotStride={}".format(r["stride"]), "--slotOffset={}".format(r.get("offset", 0)),
            "--run={}".format(r.get("run", 1)), "--out=" + tmp]
    env = dict(os.environ, LD_LIBRARY_PATH=LIB)
    t0 = time.time()
    p = subprocess.run(args, env=env, capture_output=True, text=True)
    sec = time.time() - t0
    if p.returncode != 0 or not os.path.exists(tmp):
        return name(r), "FAILED rc={} {}".format(p.returncode, p.stderr[-300:]), sec
    os.replace(tmp, dst)
    os.makedirs(MIRROR, exist_ok=True)
    shutil.copy(dst, os.path.join(MIRROR, name(r)))
    return name(r), "ok", sec


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    os.makedirs(TR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    v2 = "--v2" in sys.argv or "--v2b" in sys.argv
    todo = [r for r in (runs_v2() if v2 else runs())
            if not os.path.exists(os.path.join(OUT, name(r)))]
    if v2:
        # the costliest first, so that the pool does not end on one long n = 800 run
        todo.sort(key=lambda r: -(int(r["trace"].split("_n")[1].split("_")[0]) ** 2
                                  // (STRIDE.get(int(r["trace"].split("_n")[1].split("_")[0]), 1)
                                      if r["trace"].startswith("scal") else 1)
                                  * (3 if r["channel"] == "fading" else 1)))
    if "--missing-only" in sys.argv:
        # second pass while a first pass is still running: take only the runs the log reports
        # as skipped for want of a trace, which the first pass will never revisit
        skipped = set()
        with open(os.path.join(EMU_WIN, "run_grid.log")) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 4 and parts[1] == "no-trace":
                    skipped.add(parts[-1])
        todo = [r for r in todo if name(r) in skipped]
    if "--reverse" in sys.argv:
        todo = todo[::-1]       # a concurrent pass from the other end of the list
    print("runs to do:", len(todo), flush=True)
    log = open(os.path.join(EMU_WIN, "run_grid.log"), "a")
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run_one, r) for r in todo]
        for f in as_completed(futs):
            nm, st, sec = f.result()
            line = "{} {} {:.0f}s {}".format(time.strftime("%H:%M:%S"), st, sec, nm)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()


if __name__ == "__main__":
    main()
