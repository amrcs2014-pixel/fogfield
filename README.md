# FogField

Simulator and experiment code for **FogField**, fully decentralized repositioning of UAV-mounted fog nodes via potential games and gossip. Author: Amr H. Abdelhaliem, Irbid National University.

FogField repositions UAV-mounted fog nodes with no central controller.

- **Game.** Repositioning is a sequence of exact potential games, one per slot. Players are nodes and strategies are the lattice cells reachable in one slot. Each node's marginal-contribution (wonderful-life) utility has the global coverage, connectivity and energy objective as its potential.
- **Dynamics.** Nodes play best response with inertia (FogField-BR) or annealed log-linear learning (FogField-LLL). They use only a bounded neighbour view maintained by push-pull gossip.
- **Energy and payload.** Node energy follows the Zeng–Xu–Zhang rotary-wing model, and station-keeping is modelled as loitering at the maximum-endurance speed. Gossip payload is *measured* by running the real serialisation codec.

## Version 1.3 (27 Nov 2026): emulation tier, second set of runs

v1.3 extends `emu/`. The ns-3 program logs per slot the controller's transmit time, the commands due, reachable and delivered, and every slot's peak transmit time over all nodes; it can skip the first slots of a trace (`--slotOffset`) and replicate a run with another ns-3 run number (`--run`). `run_grid.py --v2` and `--v2b` run a second set of ns-3 runs. `loss_study.py --v2` adds link outages (a Gilbert–Elliott chain per directed link that blocks beacons, pushes and pulls), a fallback rule and reachability-aware planning for the centralized reference over the mesh, and GossipGreedy with FogField-BR's activation schedule. The simulator and the ledger are unchanged.

## Version 1.2 (27 Nov 2026): packet-level emulation tier

v1.2 adds `emu/`: code that replays the gossip protocol, and the centralized reference's state collection, packet by packet in ns-3.48 over IEEE 802.11g (a disk channel and a Nakagami-fading channel), and re-runs the simulator with the delivery probabilities measured there. The simulator and the ledger of v1.1 are unchanged. See `emu/README.md`.

## Version 1.1 (15 Oct 2026): common random numbers and a full re-run

v1.1 re-runs the whole study. The tag `v1.0` keeps the original code and ledger for comparison. The changes:

- **Common random numbers.** Each work unit draws its deployment, scenario events and adversary from (testbed, n, seed), and its algorithm stream adds the algorithm. Every comparison is paired, and every unit regenerates bit-identically in a fresh process. v1.0 seeded from Python's per-process `hash()`.
- **Loitering energy model.** Hovering (168.5 W) costs more than slow flight (126.0 W), so a station-keeping node now loiters at the maximum-endurance speed. Under v1.0's model a node could save energy by moving back and forth.
- **Energy charged from the current position.** v1.0 charged a candidate move from the node's previous position.
- **Exact MILP bound.** The MILP now covers every lattice cell and scores the coverage–energy sub-game objective. Heuristics start from the MILP's start.
- **Repaired centralized reference.** D-POA-style re-evaluates its swarm memory every slot and re-initialises when demand changes. v1.0 chased optima remembered from earlier demand slots.
- **New studies.** Three controller-loss experiments, each with a controller-alive twin; logged bounded-versus-unbounded view at 800 nodes; an annealing-schedule sweep; and a five-seed equilibrium study on both testbeds.

## What the code runs

- **Testbeds.**
  - MEL: the EUA dataset, Melbourne CBD. It has 816 real user locations, and runs start from 125 real base-station sites.
  - BEJ: GeoLife, Beijing, all 182 users. It has 500 demand cells over 48 real time-of-day slots, pooled across days.
- **Scenarios.** There are fifteen: static, mobility, random failures, traffic shift, expansion, catastrophic (50 % loss), flash crowd, single-node loss, byzantine (15 % falsified gossip), controller loss, late surge (the control twin of controller loss), and two harder controller-loss pairs: node failures after the loss (`ctrl_fail`/`late_fail`) and a sweeping demand hotspot (`ctrl_drift`/`late_drift`).
- **Baselines.** All share the same runner and measurement harness:
  - Decentralised: GossipGreedy, Guerrero-style, Consensus-ADMM and MARL-IQL.
  - Centralised: a D-POA-style swarm.
  - References: Static, RandomWalk, and an exact CBC MILP.
- **Scale.** 2,959 checkpointed work units: 2,700 main grid, 90 scalability (50–800 nodes), 25 MILP, 40 equilibrium, 100 annealing schedule and 4 view bound.

Every result is a record in the ledger `results/ledger.zip`; `python driver.py --mode analysis` computes the notebook's tables and figures from it without re-running anything.

## Repository layout

```
FogField_Q1.ipynb          the whole study as one notebook (compiled from code/parts/)
code/
  fogfield_src_fullrun.py  the canonical script, full grid (QUICK = False)
  fogfield_src.py          the same source with the quick configuration (QUICK = True)
  parts/01..10_*.py        the ten sections the source is assembled from (sync_sources.py)
  driver.py                --mode grind (resumable; FOGFIELD_NWORKERS / FOGFIELD_WORKER for
                           parallel workers) or --mode analysis (tables and figures from the ledger)
  run_grid.ps1             self-restarting Windows launcher for the grid
  grid_status.py           progress of the grid
  test_crn.py              checks of the common random numbers
  restore_ledger.py        unpack results/ledger.zip and data/cache/ into code/fogfield_out_v2/
  smoke.py                 execute every code path on a shrunken grid
  build_notebook.py        compile the # %% source into the .ipynb
results/
  run_manifest.json        datasets, UAV parameters, weights, grid, seeding, verification
  potential_check.json     exact-potential identity check (uncapacitated and capacitated)
  resume_selftest.json     checkpoint-resume self-test
  ledger.zip               2,959 work-unit records (one JSON each, with per-slot traces)
data/cache/                derived dataset caches (EUA Melbourne; GeoLife Beijing, all users)
emu/                       packet-level emulation tier (v1.2, v1.3): trace exporter, ns-3 program,
                           grid runner, loss-injection study and analysis (emu/README.md)
```

## Data

Both testbeds are built from public datasets that the script downloads itself when the cache is absent. The derived caches in `data/cache/` are small and let the analysis run without any download.

| Testbed | Source | Licence / terms |
|---|---|---|
| MEL | EUA dataset, `swinedge/eua-dataset` (Lai et al., ICSOC 2018) | as published by the EUA authors |
| BEJ | GeoLife GPS Trajectories v1.3 (Microsoft Research Asia; Zheng et al.): 18,670 trajectory files, binned to a 400 m grid and 48 time-of-day slots | Microsoft Research licence; only the aggregated demand surface is redistributed here |

## Reproduce

Tested with Python 3.11 on Windows 11, CPU only.

```bash
pip install -r requirements.txt
cd code
python restore_ledger.py          # ledger + caches -> code/fogfield_out_v2/
python driver.py --mode analysis  # tables and figures -> code/fogfield_out_v2/tables/, figs/
```

To re-run the experiments, delete `code/fogfield_out_v2/results/` and call `python driver.py --mode grind` until `python grid_status.py` reports completion. Because the seeds are derived from each unit's parameters, a re-run reproduces the released records exactly, apart from the wall-clock fields. The v1.1 grid took about three hours with four CPU workers.

The packet-level tier needs Linux or WSL 2 and ns-3.48; `emu/README.md` gives the commands.

## License

MIT; see `LICENSE`.
