# Packet-level emulation tier (ns-3)

Code that replays FogField's gossip protocol, and the centralized reference's state collection, packet by packet in ns-3.48 over IEEE 802.11g, and feeds the measured delivery probabilities back into the FogField simulator. Added in release v1.2 and extended in v1.3.

## Requirements

- Linux or WSL 2 (Ubuntu 22.04 was used), g++ ≥ 11, CMake ≥ 3.25, git.
- ns-3.48 (tag `ns-3.48`), built by `build_ns3.sh` with the wifi, internet, applications, mobility and flow-monitor modules.
- Python 3 with numpy, scipy and matplotlib (the simulator's own requirements in `../requirements.txt`).
- The v2 ledger restored into `../code/fogfield_out_v2/` (`cd ../code && python restore_ledger.py`).

## Files

| File | Role |
|---|---|
| `export_traffic.py` | Re-runs selected simulator units with a logging copy of `GossipLayer.round` (same random draws, same order) and writes one text trace per unit to `traces/`: per slot, live node positions and every push–pull exchange. A trace is written only if the re-run's `phi_final` equals the ledger value bit for bit. |
| `build_ns3.sh` | Clones ns-3 (`NS3_SRC`, default the official repository), checks out `ns-3.48` and builds it optimized in `~/ns-3-fogfield`. |
| `ns3/fogfield-emu.cc` | The ns-3 program. Modes: `gossip` (one 21-byte broadcast beacon per node per slot plus the traced push–pull exchanges over UDP unicast), `controller` (state reports to, and commands from, a controller hosted by the minimum-eccentricity node of the largest connected component, re-elected every slot, over a min-hop tree with oracle routes; commands are sent whether or not the report arrived), `calib` (broadcast delivery ratio versus distance). Channels: `disk` (delivery iff within R_comm) and `fading` (log-distance path loss with exponent 2 plus Nakagami-m fading). Writes one JSON summary per run: application delivery and latency, bytes and frames on air, retransmissions, MAC drops, and per-node transmit and busy time; per slot, the peak transmit time over all nodes (`slot_max_tx_ms`) and, in controller mode, the controller's own transmit time and the commands due, reachable and delivered (`ctrl_slots`). Options added in v1.3: `--slotOffset` (first replayed slot), `--run` (ns-3 run number, for replications), `--calibBytes` (calibration payload). |
| `compile_emu.sh`, `run_emu.sh` | Copy the program into `~/ns-3-fogfield/scratch/` and build it; run the built binary. |
| `calib.sh` | Broadcast delivery ratio versus distance on the fading channel, used to choose the transmit power that gives a delivery ratio of 0.5 at R_comm (17.1 dBm for R_comm = 820 m, 21.8 dBm for 1400 m). |
| `calib_at_rcomm.sh` | Calibration exactly at R_comm, for 21-byte and 45-byte payloads. |
| `run_grid.py` | Runs the ns-3 grid inside Linux/WSL with a worker pool; resumable (a run whose JSON exists is skipped). Results go to `~/fogfield-emu-out/runs/` and are mirrored to `runs/`. `--v2` and `--v2b` run the second set (below); `--reverse` runs a list from its other end, as a second concurrent pass. |
| `rerun_mel_ctrl.sh`, `rerun_br_slotmax.sh` | Re-run, with the v1.3 binary, the collection-mode runs on MEL at n = 50 and the FogField-BR runs that the controller comparison uses, so that their records carry the per-slot logs (every earlier field is unchanged). |
| `controller_hosts.py` | Which node hosts the controller in each slot, from the traces (`controller_hosts.json`). |
| `analyze_emu.py` | Per-run and per-group metrics (`emu_summary.csv`, `emu_numbers.json`) and the delivery probabilities used by the loss study (`loss_rates.json`). |
| `loss_study.py` | Re-runs the simulator's static units with the measured losses: each push, pull and beacon update is dropped with the measured probability, drawn from a separate random stream so that every run stays paired with the lossless ledger (at zero loss the runs reproduce the ledger exactly; `--check`). Also runs the centralized reference with collection over the mesh (nodes outside the controller's component hold position) and a 10/20/40 % uniform-loss sweep. `--v2` adds: Gilbert–Elliott beacon outages (`ge5`) and link outages that block beacons, pushes and pulls (`ge5link`), both with the measured mean loss and a mean outage of 5 slots; for the reference over the mesh, a fallback rule (a stranded node flies toward the controller's component), reachability-aware planning, and both; GossipGreedy with FogField-BR's activation probability and inertia (`ggact`); and logged copies of the reference's mesh-only units, which write `traces/meshonly_*.trace` for a collection-mode replay. |

## Grid

- Main grid: static scenario, 150 slots, n ∈ {20, 50}, seeds 0–4, both testbeds. FogField-BR, FogField-LLL and GossipGreedy in gossip mode and D-POA-style in collection mode on the fading channel; FogField-BR and D-POA-style on the disk channel.
- Scalability: BEJ, 75 slots, FogField-BR and D-POA-style, both channels, n = 50, 100, 200 (seeds 0–2) and 400, 800 (seed 0), replaying every 1st, 1st, 5th, 15th and 25th slot.
- Sensitivity: 5 s instead of 1 s traffic window, n = 50, seeds 0–1, fading channel.

Second set (v1.3):
- `run_grid.py --v2`: scalability from slot 5 on at n = 200 (seeds 0–2) and n = 400, 800 (seed 0, ns-3 runs 1–3), both methods and channels; the reference on the fading channel also with a 10 s window; the reference replayed on its mesh-only trajectories (BEJ, n = 20 and 50, seeds 0–4, both channels).
- `run_grid.py --v2b`: n = 50 and 100 from slot 5 on, seeds 0–2, both methods and channels.
- `loss_study.py --v2`: the variants listed above.

In total 120 traces, 261 ns-3 runs and 262 loss-study units. On a 12-thread laptop with 5 workers the ns-3 grid took about 2.5 hours; packet-level cost grows roughly with n² × slots.

## Reproduce

```
cd code && python restore_ledger.py && cd ..
python emu/export_traffic.py                  # traces/ (about 30 min with 3 workers)
bash emu/build_ns3.sh                         # in Linux/WSL
bash emu/compile_emu.sh                       # in Linux/WSL
python3 emu/run_grid.py 5                     # in Linux/WSL; runs/
python emu/analyze_emu.py                     # emu_summary.csv, loss_rates.json
python emu/loss_study.py                      # loss_results/
python emu/loss_study.py --v2                 # second set, incl. traces/meshonly_*
python3 emu/run_grid.py 4 --v2                # in Linux/WSL
python3 emu/run_grid.py 2 --v2b               # in Linux/WSL
python emu/controller_hosts.py
python emu/analyze_emu.py
```

Environment variables: `FOGFIELD_CODE` (the simulator folder, default `../code`), `FOGFIELD_EMU` (this folder, for `run_grid.py`), `NS3_SRC` (where to clone ns-3 from), `EXPORT_WORKERS` and `LOSS_WORKERS` (process counts).

Under WSL, keep paths that contain spaces inside scripts rather than on the `wsl.exe` command line, where they are split.
