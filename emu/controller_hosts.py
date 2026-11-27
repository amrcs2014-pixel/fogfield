"""How often does the controller's host change? Re-implements BuildControllerTree's election
(minimum-eccentricity node of the largest connected component of the disk graph, ties to the
lowest id) on the reference's traces, and reports per run the share of slots held by the most
frequent host and the number of distinct hosts.

    python controller_hosts.py
"""
import glob
import json
import os
from collections import Counter, deque

import numpy as np

EMU = os.path.dirname(os.path.abspath(__file__))


def read_trace(path):
    meta, slots = {}, []
    with open(path) as fh:
        fh.readline()
        parts = fh.readline().split()[1:]
        meta = dict(zip(parts[::2], parts[1::2]))
        while True:
            head = fh.readline().split()
            if not head or head[0] == "END":
                break
            _, rnd, na, ne = head
            ids, xy = [], []
            for _ in range(int(na)):
                i, x, y = fh.readline().split()
                ids.append(int(i))
                xy.append((float(x), float(y)))
            for _ in range(int(ne)):
                fh.readline()
            slots.append((int(rnd), np.array(ids), np.array(xy)))
    return meta, slots


def host(ids, xy, rcomm):
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    adj = [np.flatnonzero((d2[k] <= rcomm ** 2) & (np.arange(len(ids)) != k)) for k in range(len(ids))]
    comp = -np.ones(len(ids), int)
    best, best_size, c = -1, 0, 0
    for s in range(len(ids)):          # components in id order (ids are ascending in traces)
        if comp[s] >= 0:
            continue
        q, comp[s], size = deque([s]), c, 0
        while q:
            u = q.popleft()
            size += 1
            for w in adj[u]:
                if comp[w] < 0:
                    comp[w] = c
                    q.append(w)
        if size > best_size:
            best, best_size = c, size
        c += 1
    members = np.flatnonzero(comp == best)
    best_ecc, h = 1 << 30, -1
    for s in members:
        dist = {s: 0}
        q = deque([s])
        ecc = 0
        while q:
            u = q.popleft()
            ecc = max(ecc, dist[u])
            for w in adj[u]:
                if w not in dist:
                    dist[w] = dist[u] + 1
                    q.append(w)
        if ecc < best_ecc:
            best_ecc, h = ecc, s
    return int(ids[h]), best_size / len(ids)


def main():
    out = {}
    pats = ["main_MEL_static_D-POA-style_n50_s*.trace", "main_BEJ_static_D-POA-style_n50_s*.trace",
            "meshonly_BEJ_static_D-POA-style_n50_s*.trace"]
    for pat in pats:
        for f in sorted(glob.glob(os.path.join(EMU, "traces", pat))):
            meta, slots = read_trace(f)
            rc = float(meta["R_comm"])
            hosts = [host(ids, xy, rc)[0] for _, ids, xy in slots]
            cnt = Counter(hosts)
            changes = sum(1 for a, b in zip(hosts, hosts[1:]) if a != b)
            out[os.path.basename(f)] = {"top_host_share": cnt.most_common(1)[0][1] / len(hosts),
                                        "hosts": len(cnt), "changes": changes}
            print(os.path.basename(f), out[os.path.basename(f)])
    json.dump(out, open(os.path.join(EMU, "controller_hosts.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
