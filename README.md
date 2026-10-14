# FogField

Simulator and experiment code for **FogField**, fully decentralized repositioning of UAV-mounted fog nodes via potential games and gossip. Author: Amr H. Abdelhaliem, Irbid National University.

FogField repositions UAV-mounted fog nodes with no central controller. Repositioning is formulated as an
exact potential game over the communication graph: players are nodes, strategies are the lattice cells
reachable in one slot, and each node's marginal-contribution (wonderful-life) utility has the global
coverage, connectivity, energy objective as its potential. Nodes play the game by best response with inertia
(FogField-BR) or by log-linear learning with an annealed inverse temperature (FogField-LLL), using only a
bounded neighbour view maintained by push-pull gossip. The gossip cost is *measured* by running the real
serialisation codec and counting payload bytes, and the node energy follows the Zeng, Xu and Zhang
rotary-wing propulsion model.

## What the code runs

- **Two open testbeds**: MEL (EUA dataset, Melbourne CBD: 816 real user locations, runs seeded from 125 real
  base-station sites) and BEJ (GeoLife, Beijing: 500 demand cells over 48 real time-of-day slots, pooled
  across days into a diurnal demand surface).
- **Nine dynamic scenarios**: static, mobility, random failures, traffic shift, expansion, catastrophic
  (50 % simultaneous loss), flash crowd, seed loss, and byzantine (15 % of nodes gossip falsified state).
- **Baselines sharing the same runner and measurement harness**: GossipGreedy, Guerrero-style, Consensus-ADMM
  and MARL-IQL (decentralised); a D-POA-style centralised swarm; Static and RandomWalk; and an exact CBC MILP
  bound on small instances.
- **Diagnostics**: a numerical check of the exact-potential identity, an epsilon-Nash test on the global
  potential, a locality gap between the neighbour-only utility estimator and the true marginal contribution,
  and a checkpoint-resume self-test.
- **Scale**: 1,738 checkpointed work units: 1,620 main grid (2 testbeds × 9 scenarios × 9 algorithms ×
  n ∈ {20, 50} × 5 seeds), 81 scalability (50 to 800 nodes), 25 MILP, 12 equilibrium.

Every result is a record in the ledger `results/ledger.zip`; `python driver.py --mode analysis` computes the
notebook's tables and figures from it without re-running anything.

## Repository layout

```
FogField_Q1.ipynb          the whole study as one Colab-style notebook (compiled from code/parts/)
code/
  fogfield_src_fullrun.py  the notebook as a script, configured for the full grid (QUICK = False)
  fogfield_src.py          the same source with the quick Colab configuration (QUICK = True)
  parts/01..10_*.py        the ten sections the source is assembled from
  driver.py                run the script headless: --mode grind (resumable, 50-min budget per call)
                           or --mode analysis (tables and figures from the ledger, no re-computation)
  restore_ledger.py        unpack results/ledger.zip and data/cache/ into code/fogfield_out_full/
  build_notebook.py        compile the # %% source into the .ipynb
  smoke.py                 execute every code path on a shrunken grid
  tools/priority_milp_eq2.py  run the MILP and equilibrium studies before the main grid has drained
results/
  run_manifest.json        datasets, UAV parameters, weights, grid, verification results
  potential_check.json     exact-potential identity check (uncapacitated and capacitated)
  resume_selftest.json     checkpoint-resume self-test
  ledger.zip               1,738 work-unit records (one JSON each, with per-round traces)
data/cache/
  eua_melbcbd_816.npz      EUA Melbourne CBD devices and sites, projected to metres
  geolife_u60_g400_s30_c500.npz   GeoLife Beijing demand surface (400 m grid, 30-min slots, 500 cells)
```

## Data

Both testbeds are built from public datasets that the script downloads itself when the cache is absent.
The derived caches in `data/cache/` are small and let the analysis run without any download.

| Testbed | Source | Licence / terms |
|---|---|---|
| MEL | EUA dataset, `swinedge/eua-dataset` on GitHub (Lai et al., ICSOC 2018): Optus base-station sites and user locations in Melbourne CBD | as published by the EUA authors |
| BEJ | GeoLife GPS Trajectories v1.3 (Microsoft Research Asia; Zheng et al.): 18,670 trajectory files, binned to a 400 m grid and 48 time-of-day slots, pooled across days | Microsoft Research licence; only the aggregated demand surface is redistributed here |

The Telecom Italia Milano grid is not used because it requires a Dataverse guestbook response.

## Reproduce

Tested with Python 3.11 on Windows 11, CPU only.

```bash
pip install -r requirements.txt
cd code
python restore_ledger.py                 # ledger + caches -> code/fogfield_out_full/
python driver.py --mode analysis         # tables and figures from the ledger
```

To re-run the experiments, delete `code/fogfield_out_full/results/` and call
`python driver.py --mode grind` repeatedly; every work unit is checkpointed with its RNG state, so a
resumed run is bit-identical to an uninterrupted one. The full grid took about four and a half hours of
wall-clock time on a desktop CPU (23 Sep 2026). The notebook does the same on a free Colab session with
checkpoints on Google Drive. v1.0 seeds each unit from Python's per-process `hash()`, so a run from scratch
in a new process does not reproduce these records; v1.1 fixes this.

## License

MIT; see `LICENSE`.
