# FogField

Code and results for **"FogField: Fully Decentralized Repositioning of UAV-Mounted Fog Nodes via
Potential Games and Gossip"**. The author is Amr H. Abdelhaliem, Irbid National University. The manuscript is prepared for
*IEEE Transactions on Network and Service Management*.

FogField repositions UAV-mounted fog nodes with no central controller.

- **Game.** Repositioning is a sequence of exact potential games, one per slot. Players are nodes and strategies are the lattice cells reachable in one slot. Each node's marginal-contribution (wonderful-life) utility has the global coverage, connectivity and energy objective as its potential.
- **Dynamics.** Nodes play best response with inertia (FogField-BR) or annealed log-linear learning (FogField-LLL). They use only a bounded neighbour view maintained by push-pull gossip.
- **Energy and payload.** Node energy follows the Zeng–Xu–Zhang rotary-wing model, and station-keeping is modelled as loitering at the maximum-endurance speed. Gossip payload is *measured* by running the real serialisation codec.

![Deployment](results/figures/fig_deployment_bej.png)

## Version 1.1 (15 Oct 2026): revision after internal review

v1.1 re-runs the whole study. The tag `v1.0` keeps the original code and ledger for comparison. The changes:

- **Common random numbers.** Each work unit draws its deployment, scenario events and adversary from (testbed, n, seed), and its algorithm stream adds the algorithm. Every comparison is paired, and every unit regenerates bit-identically in a fresh process. v1.0 seeded from Python's per-process `hash()`.
- **Loitering energy model.** Hovering (168.5 W) costs more than slow flight (126.0 W), so a station-keeping node now loiters at the maximum-endurance speed. Under v1.0's model a node could save energy by moving back and forth.
- **Energy charged from the current position.** v1.0 charged a candidate move from the node's previous position.
- **Exact MILP bound.** The MILP now covers every lattice cell and scores the coverage–energy sub-game objective. Heuristics start from the MILP's start.
- **Repaired centralized reference.** D-POA-style re-evaluates its swarm memory every slot and re-initialises when demand changes. v1.0 chased optima remembered from earlier demand slots.
- **New studies.** Three controller-loss experiments, each with a controller-alive twin; logged bounded-versus-unbounded view at 800 nodes; an annealing-schedule sweep; and a five-seed equilibrium study on both testbeds.

## What the evaluation covers

- **Testbeds.**
  - MEL: the EUA dataset, Melbourne CBD. It has 816 real user locations, and runs start from 125 real base-station sites.
  - BEJ: GeoLife, Beijing, all 182 users. It has 500 demand cells over 48 real time-of-day slots, pooled across days.
- **Scenarios.** There are fifteen: static, mobility, random failures, traffic shift, expansion, catastrophic (50 % loss), flash crowd, single-node loss, byzantine (15 % falsified gossip), controller loss, late surge (the control twin of controller loss), and two harder controller-loss pairs: node failures after the loss (`ctrl_fail`/`late_fail`) and a sweeping demand hotspot (`ctrl_drift`/`late_drift`).
- **Baselines.** All share the same runner and measurement harness:
  - Decentralised: GossipGreedy, Guerrero-style, Consensus-ADMM and MARL-IQL.
  - Centralised: a D-POA-style swarm.
  - References: Static, RandomWalk, and an exact CBC MILP.
- **Scale.** 2,959 checkpointed work units with zero failures: 2,700 main grid, 90 scalability (50–800 nodes), 25 MILP, 40 equilibrium, 100 annealing schedule and 4 view bound.

### Headline results (v1.1)

- **Optimality gap.** FogField-BR's mean gap to the exact MILP optimum is 1.24 %, against 2.59 % for the centralised reference (paired p = 0.010). It is exactly optimal on 11 of 25 instances.
- **BEJ.** FogField-BR has the highest objective in all 10 distinct scenarios. Its lead is significant in 8, by +9 % to +27 %. It is not significant under byzantine or random failures.
- **MEL.** MEL is saturated, and the centralised reference is ahead by 0.3 %.
- **Scalability.** FogField-BR is best at n ≤ 100 and GossipGreedy from n = 200 on. Per-node payload is bounded by the codec at 270 B per slot for every method.
- **Limits reported in the paper:**
  - Falsified gossip costs FogField-BR 7.9 %.
  - Losing the centralised controller costs the reference 13.6 % (BEJ) and 4.9 % (MEL) only when node failures continue afterwards; under a surge or a sweeping hotspot it costs nothing measurable.
  - Guerrero-style never clears its migration threshold, so its results equal Static.
  - Payload is application-layer only.

All numbers come from the ledger in `results/ledger.zip`. `code/paper_numbers.py` regenerates every table, figure and quoted number of the paper from it (`results/paper_numbers.json`).

## Repository layout

```
FogField_Q1.ipynb          the whole study as one notebook (compiled from code/parts/)
code/
  fogfield_src_fullrun.py  the canonical script, paper-scale grid (QUICK = False)
  fogfield_src.py          the same source with the quick configuration (QUICK = True)
  parts/01..10_*.py        the ten sections the source is assembled from (sync_sources.py)
  driver.py                --mode grind (resumable; FOGFIELD_NWORKERS / FOGFIELD_WORKER for
                           parallel workers) or --mode analysis (notebook tables and figures)
  run_grid.ps1             self-restarting Windows launcher used for the v1.1 run
  grid_status.py           progress of the grid
  test_crn.py              checks of the common random numbers
  paper_numbers.py         the paper's tables, figures and numbers from the ledger
  restore_ledger.py        unpack results/ledger.zip and data/cache/ into code/fogfield_out_v2/
  smoke.py                 execute every code path on a shrunken grid
  tools/                   packaging, Word conversion, repository assembly, and the one-off
                           v1.0 -> v1.1 source patches (revise_v2_patch*.py)
results/
  run_manifest.json        datasets, UAV parameters, weights, grid, seeding, verification
  ledger.zip               2,959 work-unit records (one JSON each, with per-slot traces)
  paper_numbers.json       every scalar the paper quotes
  tables/                  notebook tables and the paper's tables (paper_*.csv)
  figures/                 paper figures (PDF + PNG) and notebook diagnostics
data/cache/                derived dataset caches (EUA Melbourne; GeoLife Beijing, all users)
```

## Data

| Testbed | Source | Licence / terms |
|---|---|---|
| MEL | EUA dataset, `swinedge/eua-dataset` (Lai et al., ICSOC 2018) | as published by the EUA authors |
| BEJ | GeoLife GPS Trajectories v1.3 (Microsoft Research Asia; Zheng et al.): 18,670 trajectory files, binned to a 400 m grid and 48 time-of-day slots | Microsoft Research licence; only the aggregated demand surface is redistributed here |

## Reproduce

Tested with Python 3.11 on Windows 11, CPU only.

```bash
pip install -r requirements.txt
cd code
python restore_ledger.py        # ledger + caches -> code/fogfield_out_v2/
python paper_numbers.py         # the paper's tables, figures and numbers -> code/paper_tables/
python driver.py --mode analysis
```

To re-run the experiments, delete `code/fogfield_out_v2/results/` and call `python driver.py --mode grind` until `python grid_status.py` reports completion. Because the seeds are derived from each unit's parameters, a re-run reproduces the released records exactly, apart from the wall-clock fields.

The v1.1 grid took about three hours with four CPU workers. The run logs of both versions (`fogfield_run_logs_v1.1.zip`) are meant as a release asset rather than a committed file.

## License

MIT; see `LICENSE`.
