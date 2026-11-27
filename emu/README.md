# Packet-level emulation tier (ns-3)

Code that replays FogField's gossip protocol, and the centralized reference's state collection, packet by packet in ns-3.48 over IEEE 802.11g, and feeds the measured delivery probabilities back into the FogField simulator. Added in release v1.2.

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
| `ns3/fogfield-emu.cc` | The ns-3 program. Modes: `gossip` (one 21-byte broadcast beacon per node per slot plus the traced push–pull exchanges over UDP unicast), `controller` (state reports to, and commands from, a controller hosted by the most central node of the largest connected component, over a shortest-path tree with oracle routes), `calib` (broadcast delivery ratio versus distance). Channels: `disk` (delivery iff within R_comm) and `fading` (log-distance path loss with exponent 2 plus Nakagami-m fading). Writes one JSON summary per run: application delivery and latency, bytes and frames on air, retransmissions, MAC drops, and per-node transmit and busy time. |
| `compile_emu.sh`, `run_emu.sh` | Copy the program into `~/ns-3-fogfield/scratch/` and build it; run the built binary. |
| `calib.sh` | Broadcast delivery ratio versus distance on the fading channel, used to choose the transmit power that gives a delivery ratio of 0.5 at R_comm (17.1 dBm for R_comm = 820 m, 21.8 dBm for 1400 m). |
| `run_grid.py` | Runs the ns-3 grid inside Linux/WSL with a worker pool; resumable (a run whose JSON exists is skipped). Results go to `~/fogfield-emu-out/runs/` and are mirrored to `runs/`. |
| `analyze_emu.py` | Per-run and per-group metrics (`emu_summary.csv`, `emu_numbers.json`) and the delivery probabilities used by the loss study (`loss_rates.json`). |
| `loss_study.py` | Re-runs the simulator's static units with the measured losses: each push, pull and beacon update is dropped with the measured probability, drawn from a separate random stream so that every run stays paired with the lossless ledger (at zero loss the runs reproduce the ledger exactly; `--check`). Also runs the centralized reference with collection over the mesh (nodes outside the controller's component hold position) and a 10/20/40 % uniform-loss sweep. |

## Grid

- Main grid: static scenario, 150 slots, n ∈ {20, 50}, seeds 0–4, both testbeds. FogField-BR, FogField-LLL and GossipGreedy in gossip mode and D-POA-style in collection mode on the fading channel; FogField-BR and D-POA-style on the disk channel.
- Scalability: BEJ, 75 slots, FogField-BR and D-POA-style, both channels, n = 50, 100, 200 (seeds 0–2) and 400, 800 (seed 0), replaying every 1st, 1st, 5th, 15th and 25th slot.
- Sensitivity: 5 s instead of 1 s traffic window, n = 50, seeds 0–1, fading channel.

That is 110 traces and 172 ns-3 runs, plus 160 loss-study units. On a 12-thread laptop with 5 workers the ns-3 grid took about 2.5 hours; packet-level cost grows roughly with n² × slots.

## Reproduce

```
cd code && python restore_ledger.py && cd ..
python emu/export_traffic.py                  # traces/ (about 30 min with 3 workers)
bash emu/build_ns3.sh                         # in Linux/WSL
bash emu/compile_emu.sh                       # in Linux/WSL
python3 emu/run_grid.py 5                     # in Linux/WSL; runs/
python emu/analyze_emu.py                     # emu_summary.csv, loss_rates.json
python emu/loss_study.py                      # loss_results/
```

Environment variables: `FOGFIELD_CODE` (the simulator folder, default `../code`), `FOGFIELD_EMU` (this folder, for `run_grid.py`), `NS3_SRC` (where to clone ns-3 from), `EXPORT_WORKERS` and `LOSS_WORKERS` (process counts).

Under WSL, keep paths that contain spaces inside scripts rather than on the `wsl.exe` command line, where they are split.
