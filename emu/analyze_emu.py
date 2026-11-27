"""Summarise the ns-3 emulation runs (emu/runs/*.json).

Writes
  emu/emu_summary.csv        one row per (study, testbed, n, method, mode, channel, window)
  emu/loss_rates.json        delivery probabilities for loss_study.py (fading channel, main grid)
  emu/emu_numbers.json       summary scalars

    python analyze_emu.py
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

EMU = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(EMU, "runs")
PAT = re.compile(r"^(main|scalability|meshonly)_(MEL|BEJ)_static_(.+)_n(\d+)_s(\d+)__(gossip|controller)_(disk|fading)"
                 r"_w([\d.]+)(?:_o(\d+)_r(\d+))?\.json$")

RADIO_TX_W, RADIO_RX_W = 2.0, 1.0      # electrical power of an 802.11 radio, TX and RX/busy (assumed)
RADIO_IDLE_W = 1.0                     # idle listening, radio always on (no power save)


def load():
    rows = []
    for f in glob.glob(os.path.join(RUNS, "*.json")):
        m = PAT.match(os.path.basename(f))
        if not m:
            continue
        r = json.load(open(f))
        study, tb, algo, n, seed, mode, ch, w, off, run = m.groups()
        if off is not None and int(off) > 0:
            study += "_o" + off             # re-sampled from slot `off` on: a separate group
        r.update(study=study, tb=tb, algo=algo, nn=int(n), seed=int(seed), w=float(w),
                 rep=int(run) if run else 1)
        rows.append(r)
    return rows


def per_run(r):
    a, p, mac = r["app"], r["phy"], r["mac"]
    ns = r["node_slots"]
    d = dict(study=r["study"], tb=r["tb"], algo=r["algo"], n=r["nn"], seed=r["seed"], mode=r["mode"],
             channel=r["channel"], window=r["w"], slots=r["slots"], stride=r.get("slot_stride", 1),
             rep=r.get("rep", 1))
    d["drop_lifetime"] = mac.get("drop_lifetime", 0)
    d["drop_retry"] = mac.get("drop_retry_limit", 0)
    d["air_B_node_slot"] = p["bytes_on_air"] / ns
    d["app_B_node_slot"] = a["app_payload_bytes"] / ns
    d["overhead_x"] = p["bytes_on_air"] / max(a["app_payload_bytes"], 1)
    d["airtime_ms_node_slot"] = 1e3 * p["tx_time_total_s"] / ns
    tx = np.array(r["node_tx_time_s"], dtype=float)
    per_slot = r["slots"]
    d["airtime_ms_maxnode_slot"] = 1e3 * tx.max() / per_slot      # node with the largest run total
    if r.get("slot_max_tx_ms"):          # per-slot peak over all nodes, averaged over slots
        d["slot_max_ms"] = float(np.mean(r["slot_max_tx_ms"]))
    d["busy_ms_node_slot"] = 1e3 * p["busy_time_total_s"] / ns
    d["retry_ratio"] = p["f_data_retry"] / max(p["f_data_unicast"], 1)
    d["final_fail"] = mac["tx_final_failed"]
    d["radio_J_node_slot"] = (RADIO_TX_W * p["tx_time_total_s"]
                              + RADIO_RX_W * (p["busy_time_total_s"] - p["tx_time_total_s"])) / ns
    # radio always on: transmit time at 2 W, every other second of the slot (receive, sense, idle) at 1 W
    d["radio_J_alwayson"] = (RADIO_TX_W * p["tx_time_total_s"]
                             + RADIO_IDLE_W * (ns * r["dt"] - p["tx_time_total_s"])) / ns
    if r["mode"] == "gossip":
        d["push_dr"] = a["push_rx"] / max(a["push_sent"], 1)
        d["pull_dr"] = a["pull_rx"] / max(a["pull_sent"], 1)
        d["exch_ok"] = a["pull_rx"] / max(a["exchanges"], 1)
        d["beacon_dr"] = a["beacon_rx_inrange"] / max(a["beacon_pairs_inrange"], 1)
        d["beacon_extra"] = a["beacon_rx_outrange"] / max(a["beacon_sent"], 1)
        d["rtt_ms"] = 1e3 * a["rtt_mean_s"]
        d["rtt_p95_ms"] = 1e3 * a["rtt_p95_s"]
        d["_sums"] = (a["push_rx"], a["push_sent"], a["pull_rx"], a["pull_sent"],
                      a["beacon_rx_inrange"], a["beacon_pairs_inrange"])
    else:
        due = max(a["rep_due"], 1)
        d["unreach"] = a["unreachable"] / due
        d["rep_dr"] = a["rep_rx"] / due
        d["cmd_dr"] = a["cmd_rx"] / due
        reach = max(a["cmd_due"] - a["unreachable"], 1)
        d["cmd_dr_reach"] = a["cmd_rx"] / reach
        d["rep_dr_reach"] = a["rep_rx"] / reach
        d["hops"] = a["hops_mean"]
        d["hops_max"] = a["hops_max"]
        d["rep_lat_ms"] = 1e3 * a["rep_lat_mean_s"]
        d["rep_lat_p95_ms"] = 1e3 * a["rep_lat_p95_s"]
        d["cmd_lat_ms"] = 1e3 * a["cmd_lat_mean_s"]
        d["cmd_lat_p95_ms"] = 1e3 * a["cmd_lat_p95_s"]
        cs = r.get("ctrl_slots")
        if cs:                               # per-slot log of the controller itself
            d["ctrl_tx_ms"] = float(np.mean(cs["ctrl_tx_ms"]))
            d["ctrl_hosts"] = len(set(int(c) for c in cs["ctrl"] if c >= 0))
            due = np.array(cs["due"], float)
            d["cmd_dr_slots"] = [float(x) for x in np.array(cs["cmd_rx"]) / np.maximum(due, 1)]
            d["reach_slots"] = [float(x) for x in np.array(cs["reach"]) / np.maximum(due, 1)]
        d["_sums"] = (a["cmd_rx"], reach, a["rep_rx"], reach)
    return d


def main():
    rows = [per_run(r) for r in load()]
    print("runs:", len(rows))
    groups = defaultdict(list)
    for d in rows:
        groups[(d["study"], d["tb"], d["n"], d["algo"], d["mode"], d["channel"], d["window"])].append(d)
    keys = sorted({k for d in rows for k in d if not k.startswith("_")})
    num_cols = [k for k in keys if k not in ("study", "tb", "algo", "mode", "channel", "seed")]
    lines = []
    summary = {}
    for g, ds in sorted(groups.items()):
        agg = {"study": g[0], "tb": g[1], "n": g[2], "algo": g[3], "mode": g[4], "channel": g[5],
               "window": g[6], "runs": len(ds)}
        for k in num_cols:
            vals = [d[k] for d in ds if k in d and d[k] is not None]
            if vals and k not in ("n", "window"):
                agg[k] = float(np.mean(vals))
                agg[k + "_min"] = float(np.min(vals))
                agg[k + "_max"] = float(np.max(vals))
        summary["|".join(map(str, g))] = agg
        lines.append(agg)
    cols = sorted({k for a in lines for k in a}, key=lambda c: (c not in ("study", "tb", "n", "algo", "mode", "channel", "window", "runs"), c))
    with open(os.path.join(EMU, "emu_summary.csv"), "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for a in lines:
            fh.write(",".join("" if a.get(c) is None else (("%.6g" % a[c]) if isinstance(a[c], float) else str(a[c])) for c in cols) + "\n")

    # loss rates from the main-grid fading runs, pooled over seeds (ratio of sums)
    rates = {}
    for d in rows:
        if d["study"] != "main" or d["channel"] != "fading" or d["window"] != 1.0:
            continue
        key = "{}|{}|{}".format(d["tb"], d["n"], d["algo"])
        acc = rates.setdefault(key, np.zeros(len(d["_sums"])))
        acc += np.array(d["_sums"], dtype=float)
    out = {}
    for key, s in rates.items():
        if key.endswith("D-POA-style"):
            out[key] = {"q_cmd": float(s[0] / s[1]), "q_rep": float(s[2] / s[3])}
        else:
            out[key] = {"q_push": float(s[0] / s[1]), "q_pull": float(s[2] / s[3]),
                        "q_beacon": float(s[4] / s[5])}
    with open(os.path.join(EMU, "loss_rates.json"), "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    with open(os.path.join(EMU, "emu_numbers.json"), "w") as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    print("groups:", len(summary), "| loss-rate keys:", len(out))
    for k in sorted(out):
        print(" ", k, {kk: round(float(v), 4) for kk, v in out[k].items()})


if __name__ == "__main__":
    main()
