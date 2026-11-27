/*
 * FogField emulation tier: replay a FogField trace over an IEEE 802.11 ad hoc network in ns-3.
 *
 * Input: a trace written by export_traffic.py (per slot: alive node positions and the exact
 * push/pull gossip exchanges of the FogField simulator, in protocol order).
 *
 * Modes
 *   gossip      every alive node broadcasts one beacon per slot (header only, 21 B) and runs the
 *               traced push-pull exchanges as UDP unicast (push i->j, pull j->i on receipt).
 *   controller  centralized state collection over the same mesh: each slot the most central node
 *               of the largest connected component hosts the controller; every other node sends a
 *               21 B state report to it over a shortest-path tree, then the controller sends each
 *               node a 10 B position command. Routes are installed by an oracle every slot (no
 *               routing control traffic), so this is a lower bound on the controller's cost.
 *   calib       two nodes, broadcast PDR versus distance (for choosing the fading-channel power).
 *
 * Channels
 *   disk        RangePropagationLossModel with MaxRange = R_comm: reception iff within R_comm and
 *               no collision. Matches the FogField simulator's neighbourhood definition, so it
 *               isolates MAC effects (contention, collisions, hidden terminals, ACK, retries).
 *   fading      log-distance path loss (exponent 2, free space at 2.4 GHz) with Nakagami-m fading
 *               (m = 2); transmit power calibrated so that the single-attempt broadcast PDR at
 *               R_comm is 0.5.
 *
 * PHY/MAC: IEEE 802.11g ERP-OFDM at 6 Mb/s for data, control and broadcast frames, ad hoc MAC,
 * UDP/IPv4 with pre-populated neighbour (ARP) caches.
 */
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/propagation-module.h"
#include "ns3/wifi-module.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("FogFieldEmu");

namespace
{
const uint16_t PORT = 9;
const double ALT = 100.0;      // all UAVs fly at the same altitude
const uint32_t BEACON_B = 21;  // codec header: id, seq, x, y, battery, hint, count
const uint32_t REPORT_B = 21;  // state report to the controller (same header)
const uint32_t CMD_B = 10;     // position command: node id + x + y

struct Slot
{
    int rnd;
    std::vector<int> ids;
    std::vector<double> x, y;
    std::vector<std::array<int, 4>> ex; // i, j, push bytes, pull bytes
};

enum PktType : uint8_t
{
    BEACON,
    PUSH,
    PULL,
    REPORT,
    CMD
};

struct PktInfo
{
    PktType type;
    int src;
    int dst;
    int64_t ex;      // exchange index (PUSH/PULL), or slot-local id
    double tSend;    // seconds
    int slot;
};

// ---------------------------------------------------------------------------- global state
std::vector<Slot> g_slots;
std::map<std::string, std::string> g_meta;
int g_n = 0;
double g_rcomm = 0, g_dt = 30, g_window = 1.0;
std::string g_mode = "gossip";
NodeContainer g_nodes;
std::vector<Ipv4Address> g_ip;
std::vector<Ptr<Socket>> g_sock;
std::vector<Vector> g_pos;
std::vector<char> g_alive;
int g_curSlot = -1;
Ptr<UniformRandomVariable> g_u;

std::unordered_map<uint64_t, PktInfo> g_pkt;   // packet uid -> what it is
std::vector<std::array<int, 4>> g_exList;      // flattened exchanges (i, j, push, pull)
std::vector<double> g_exPushT;                 // send time of each push
std::vector<char> g_exPushRx, g_exPullRx;
std::vector<double> g_rtt;

// counters
uint64_t c_beaconSent = 0, c_beaconPairs = 0, c_beaconRxIn = 0, c_beaconRxOut = 0;
uint64_t c_pushSent = 0, c_pushRx = 0, c_pullSent = 0, c_pullRx = 0, c_appBytes = 0;
uint64_t c_repDue = 0, c_repSent = 0, c_repRx = 0, c_cmdDue = 0, c_cmdSent = 0, c_cmdRx = 0;
uint64_t c_unreach = 0;
std::vector<double> g_repLat, g_cmdLat, g_hops;
std::set<std::pair<int, int>> g_beaconSeen; // (slot*N + receiver, sender) to count once

// PHY/MAC per node
std::vector<double> n_txTime, n_rxTime, n_ccaTime, n_bytesAir;
uint64_t f_dataUni = 0, f_dataRetry = 0, f_dataBcast = 0, f_ack = 0, f_other = 0;
uint64_t m_txFailed = 0, m_txFinalFailed = 0, m_dropRetry = 0, m_dropLifetime = 0, m_dropEnqueue = 0;
uint64_t b_air = 0;

// controller: per slot
int g_ctrl = -1;
std::vector<int> g_parent, g_depth;
// per-slot controller log (collection mode): who hosted the controller, its own transmit time,
// and how many reports and commands were due, reachable and delivered in that slot
std::vector<int> g_slotCtrl, g_slotDue, g_slotReach, g_slotRepRx, g_slotCmdRx, g_slotRnd;
std::vector<double> g_slotCtrlTx;
std::vector<double> g_txAtSlotStart;
// per-slot peak (all modes): the largest transmit time of any node within the slot
std::vector<double> g_slotMaxTx;

void
CloseSlot()
{
    // attribute the transmit time accumulated since the slot started to that slot's controller
    if (g_mode == "controller" && g_curSlot >= 0 && g_ctrl >= 0)
    {
        g_slotCtrlTx[g_curSlot] = n_txTime[g_ctrl] - g_txAtSlotStart[g_ctrl];
    }
    if (g_curSlot >= 0 && g_txAtSlotStart.size() == n_txTime.size())
    {
        double mx = 0.0;
        for (size_t i = 0; i < n_txTime.size(); ++i)
        {
            mx = std::max(mx, n_txTime[i] - g_txAtSlotStart[i]);
        }
        g_slotMaxTx[g_curSlot] = mx;
    }
}

int
NodeFromContext(const std::string& ctx)
{
    auto a = ctx.find("/NodeList/") + 10;
    auto b = ctx.find('/', a);
    return std::stoi(ctx.substr(a, b - a));
}

double
Dist(int a, int b)
{
    return CalculateDistance(g_pos[a], g_pos[b]);
}

// ---------------------------------------------------------------------------- trace input
void
ReadTrace(const std::string& path)
{
    std::ifstream in(path);
    NS_ABORT_MSG_IF(!in, "cannot open trace " << path);
    std::string tok;
    in >> tok;
    NS_ABORT_MSG_IF(tok != "FOGFIELD_TRACE", "not a FogField trace");
    int ver;
    in >> ver;
    std::string line;
    std::getline(in, line);
    std::getline(in, line);
    std::istringstream ms(line);
    ms >> tok; // META
    std::string k, v;
    while (ms >> k >> v)
    {
        g_meta[k] = v;
    }
    g_n = std::stoi(g_meta["n"]);
    g_rcomm = std::stod(g_meta["R_comm"]);
    g_dt = std::stod(g_meta["dt_s"]);
    while (in >> tok)
    {
        if (tok == "END")
        {
            break;
        }
        NS_ABORT_MSG_IF(tok != "SLOT", "bad trace token " << tok);
        Slot s;
        int na, ne;
        in >> s.rnd >> na >> ne;
        s.ids.resize(na);
        s.x.resize(na);
        s.y.resize(na);
        for (int q = 0; q < na; ++q)
        {
            in >> s.ids[q] >> s.x[q] >> s.y[q];
        }
        s.ex.resize(ne);
        for (int q = 0; q < ne; ++q)
        {
            in >> s.ex[q][0] >> s.ex[q][1] >> s.ex[q][2] >> s.ex[q][3];
        }
        g_slots.push_back(std::move(s));
    }
}

// ---------------------------------------------------------------------------- PHY/MAC traces
void
OnPsdu(std::string ctx, WifiConstPsduMap psduMap, WifiTxVector txVector, double txPowerW)
{
    int node = NodeFromContext(ctx);
    for (const auto& kv : psduMap)
    {
        Ptr<const WifiPsdu> psdu = kv.second;
        uint32_t sz = psdu->GetSize();
        n_bytesAir[node] += sz;
        b_air += sz;
        const WifiMacHeader& h = psdu->GetHeader(0);
        if (h.IsAck())
        {
            ++f_ack;
        }
        else if (h.IsData() || h.IsQosData())
        {
            if (psdu->GetAddr1().IsGroup())
            {
                ++f_dataBcast;
            }
            else if (h.IsRetry())
            {
                ++f_dataRetry;
            }
            else
            {
                ++f_dataUni;
            }
        }
        else
        {
            ++f_other;
        }
    }
}

void
OnState(std::string ctx, Time start, Time duration, WifiPhyState state)
{
    int node = NodeFromContext(ctx);
    double d = duration.GetSeconds();
    if (state == WifiPhyState::TX)
    {
        n_txTime[node] += d;
    }
    else if (state == WifiPhyState::RX)
    {
        n_rxTime[node] += d;
    }
    else if (state == WifiPhyState::CCA_BUSY)
    {
        n_ccaTime[node] += d;
    }
}

void
OnTxFailed(std::string, Mac48Address)
{
    ++m_txFailed;
}

void
OnTxFinalFailed(std::string, Mac48Address)
{
    ++m_txFinalFailed;
}

void
OnDropped(std::string, WifiMacDropReason reason, Ptr<const WifiMpdu>)
{
    if (reason == WIFI_MAC_DROP_REACHED_RETRY_LIMIT)
    {
        ++m_dropRetry;
    }
    else if (reason == WIFI_MAC_DROP_EXPIRED_LIFETIME)
    {
        ++m_dropLifetime;
    }
    else
    {
        ++m_dropEnqueue;
    }
}

// ---------------------------------------------------------------------------- sending
void
SendPkt(int src, Ipv4Address dst, uint32_t bytes, PktInfo info)
{
    Ptr<Packet> p = Create<Packet>(bytes);
    info.tSend = Simulator::Now().GetSeconds();
    info.slot = g_curSlot;
    g_pkt[p->GetUid()] = info;
    g_sock[src]->SendTo(p, 0, InetSocketAddress(dst, PORT));
}

void
SendBeacon(int i)
{
    if (!g_alive[i])
    {
        return;
    }
    ++c_beaconSent;
    c_appBytes += BEACON_B;
    SendPkt(i, Ipv4Address("10.0.255.255"), BEACON_B, PktInfo{BEACON, i, -1, 0, 0, 0});
}

void
SendPush(int64_t e)
{
    int i = g_exList[e][0], j = g_exList[e][1];
    ++c_pushSent;
    c_appBytes += g_exList[e][2];
    g_exPushT[e] = Simulator::Now().GetSeconds();
    SendPkt(i, g_ip[j], g_exList[e][2], PktInfo{PUSH, i, j, e, 0, 0});
}

void
SendPull(int64_t e)
{
    int i = g_exList[e][0], j = g_exList[e][1];
    ++c_pullSent;
    c_appBytes += g_exList[e][3];
    SendPkt(j, g_ip[i], g_exList[e][3], PktInfo{PULL, j, i, e, 0, 0});
}

void
SendReport(int v)
{
    ++c_repSent;
    c_appBytes += REPORT_B;
    SendPkt(v, g_ip[g_ctrl], REPORT_B, PktInfo{REPORT, v, g_ctrl, 0, 0, 0});
}

void
SendCmd(int v)
{
    ++c_cmdSent;
    c_appBytes += CMD_B;
    SendPkt(g_ctrl, g_ip[v], CMD_B, PktInfo{CMD, g_ctrl, v, 0, 0, 0});
}

// ---------------------------------------------------------------------------- receiving
void
OnRecv(Ptr<Socket> s)
{
    Address from;
    Ptr<Packet> p;
    int me = s->GetNode()->GetId();
    while ((p = s->RecvFrom(from)))
    {
        auto it = g_pkt.find(p->GetUid());
        if (it == g_pkt.end())
        {
            continue;
        }
        const PktInfo& info = it->second;
        double now = Simulator::Now().GetSeconds();
        switch (info.type)
        {
        case BEACON: {
            if (info.src == me)
            {
                break;
            }
            auto key = std::make_pair(info.slot * (g_n + 1) + me, info.src);
            if (g_beaconSeen.insert(key).second)
            {
                if (Dist(me, info.src) <= g_rcomm)
                {
                    ++c_beaconRxIn;
                }
                else
                {
                    ++c_beaconRxOut;
                }
            }
            break;
        }
        case PUSH:
            if (me == info.dst && !g_exPushRx[info.ex])
            {
                g_exPushRx[info.ex] = 1;
                ++c_pushRx;
                int64_t e = info.ex;
                Simulator::Schedule(MicroSeconds(50), &SendPull, e);
            }
            break;
        case PULL:
            if (me == info.dst && !g_exPullRx[info.ex])
            {
                g_exPullRx[info.ex] = 1;
                ++c_pullRx;
                g_rtt.push_back(now - g_exPushT[info.ex]);
            }
            break;
        case REPORT:
            if (me == info.dst)
            {
                ++c_repRx;
                ++g_slotRepRx[info.slot];
                g_repLat.push_back(now - info.tSend);
            }
            break;
        case CMD:
            if (me == info.dst)
            {
                ++c_cmdRx;
                ++g_slotCmdRx[info.slot];
                g_cmdLat.push_back(now - info.tSend);
            }
            break;
        }
    }
}

// ---------------------------------------------------------------------------- controller routing
std::vector<Ptr<Ipv4StaticRouting>> g_srt;

void
ClearHostRoutes()
{
    for (int v = 0; v < g_n; ++v)
    {
        Ptr<Ipv4StaticRouting> r = g_srt[v];
        for (int k = (int)r->GetNRoutes() - 1; k >= 0; --k)
        {
            Ipv4RoutingTableEntry e = r->GetRoute(k);
            if (e.IsHost())
            {
                r->RemoveRoute(k);
            }
        }
    }
}

// Most central node of the largest component of the disk graph over alive nodes; BFS tree.
void
BuildControllerTree()
{
    std::vector<std::vector<int>> adj(g_n);
    for (int a = 0; a < g_n; ++a)
    {
        if (!g_alive[a])
        {
            continue;
        }
        for (int b = a + 1; b < g_n; ++b)
        {
            if (g_alive[b] && Dist(a, b) <= g_rcomm)
            {
                adj[a].push_back(b);
                adj[b].push_back(a);
            }
        }
    }
    // components
    std::vector<int> comp(g_n, -1);
    int best = -1, bestSize = 0, cid = 0;
    for (int a = 0; a < g_n; ++a)
    {
        if (!g_alive[a] || comp[a] >= 0)
        {
            continue;
        }
        std::queue<int> q;
        q.push(a);
        comp[a] = cid;
        int size = 0;
        while (!q.empty())
        {
            int u = q.front();
            q.pop();
            ++size;
            for (int w : adj[u])
            {
                if (comp[w] < 0)
                {
                    comp[w] = cid;
                    q.push(w);
                }
            }
        }
        if (size > bestSize)
        {
            bestSize = size;
            best = cid;
        }
        ++cid;
    }
    // minimum-eccentricity node of the largest component
    auto bfs = [&](int src, std::vector<int>& dist, std::vector<int>& par) {
        dist.assign(g_n, -1);
        par.assign(g_n, -1);
        std::queue<int> q;
        q.push(src);
        dist[src] = 0;
        int ecc = 0;
        while (!q.empty())
        {
            int u = q.front();
            q.pop();
            ecc = std::max(ecc, dist[u]);
            for (int w : adj[u])
            {
                if (dist[w] < 0)
                {
                    dist[w] = dist[u] + 1;
                    par[w] = u;
                    q.push(w);
                }
            }
        }
        return ecc;
    };
    std::vector<int> dist, par;
    int bestEcc = 1 << 30;
    g_ctrl = -1;
    for (int a = 0; a < g_n; ++a)
    {
        if (g_alive[a] && comp[a] == best)
        {
            int e = bfs(a, dist, par);
            if (e < bestEcc)
            {
                bestEcc = e;
                g_ctrl = a;
            }
        }
    }
    bfs(g_ctrl, g_depth, g_parent);
    // install routes: up via parent to the controller; down via the child on the path
    ClearHostRoutes();
    for (int v = 0; v < g_n; ++v)
    {
        if (v == g_ctrl || g_depth[v] < 0)
        {
            continue;
        }
        g_srt[v]->AddHostRouteTo(g_ip[g_ctrl], g_ip[g_parent[v]], 1);
        // walk up from v: every ancestor a forwards traffic for v to its child on the path
        int child = v;
        int a = g_parent[v];
        while (a >= 0)
        {
            g_srt[a]->AddHostRouteTo(g_ip[v], g_ip[child], 1);
            child = a;
            a = g_parent[a];
        }
    }
}

// ---------------------------------------------------------------------------- slot driver
void
StartSlot(int s)
{
    CloseSlot();
    g_txAtSlotStart = n_txTime;
    g_curSlot = s;
    g_beaconSeen.clear(); // receptions of the previous slot ended long before (window << slot)
    const Slot& sl = g_slots[s];
    double t0 = Simulator::Now().GetSeconds();
    std::fill(g_alive.begin(), g_alive.end(), 0);
    for (size_t q = 0; q < sl.ids.size(); ++q)
    {
        int i = sl.ids[q];
        g_alive[i] = 1;
        g_pos[i] = Vector(sl.x[q], sl.y[q], ALT);
    }
    for (int i = 0; i < g_n; ++i)
    {
        if (!g_alive[i])
        {
            g_pos[i] = Vector(1.0e7 + 1.0e5 * i, 0.0, ALT); // parked far away, silent
        }
        g_nodes.Get(i)->GetObject<MobilityModel>()->SetPosition(g_pos[i]);
    }
    if (g_mode == "gossip")
    {
        for (int i = 0; i < g_n; ++i)
        {
            if (!g_alive[i])
            {
                continue;
            }
            for (int j = 0; j < g_n; ++j)
            {
                if (j != i && g_alive[j] && Dist(i, j) <= g_rcomm)
                {
                    ++c_beaconPairs;
                }
            }
            Simulator::Schedule(Seconds(g_u->GetValue(0, g_window)), &SendBeacon, i);
        }
        for (const auto& e : sl.ex)
        {
            int64_t idx = (int64_t)g_exList.size();
            g_exList.push_back(e);
            g_exPushT.push_back(0);
            g_exPushRx.push_back(0);
            g_exPullRx.push_back(0);
            Simulator::Schedule(Seconds(g_u->GetValue(0, g_window)), &SendPush, idx);
        }
    }
    else // controller
    {
        BuildControllerTree();
        g_slotCtrl[s] = g_ctrl;
        for (int v = 0; v < g_n; ++v)
        {
            if (!g_alive[v] || v == g_ctrl)
            {
                continue;
            }
            ++c_repDue;
            ++c_cmdDue;
            ++g_slotDue[s];
            if (g_depth[v] < 0)
            {
                c_unreach += 1;
                continue;
            }
            ++g_slotReach[s];
            g_hops.push_back(g_depth[v]);
            Simulator::Schedule(Seconds(g_u->GetValue(0, g_window)), &SendReport, v);
            Simulator::Schedule(Seconds(g_window + g_u->GetValue(0, g_window)), &SendCmd, v);
        }
    }
    (void)t0;
}

// ---------------------------------------------------------------------------- output helpers
double
Mean(const std::vector<double>& v)
{
    if (v.empty())
    {
        return 0;
    }
    double s = 0;
    for (double x : v)
    {
        s += x;
    }
    return s / v.size();
}

double
Pct(std::vector<double> v, double p)
{
    if (v.empty())
    {
        return 0;
    }
    std::sort(v.begin(), v.end());
    size_t k = std::min(v.size() - 1, (size_t)(p * (v.size() - 1) + 0.5));
    return v[k];
}

std::string
JsonArr(const std::vector<double>& v, int prec = 6)
{
    std::ostringstream o;
    o << std::setprecision(prec) << "[";
    for (size_t i = 0; i < v.size(); ++i)
    {
        o << (i ? "," : "") << v[i];
    }
    o << "]";
    return o.str();
}

// ---------------------------------------------------------------------------- calibration
uint32_t g_calibRx = 0;
uint32_t g_calibBytes = 45;

void
CalibRecv(Ptr<Socket> s)
{
    Address f;
    while (s->RecvFrom(f))
    {
        ++g_calibRx;
    }
}

void
CalibSend(Ptr<Socket> s)
{
    s->Send(Create<Packet>(g_calibBytes));
}

int
RunCalib(double txp, double m, double dmin, double dmax, double dstep, uint32_t npk, std::string out)
{
    std::ofstream os(out);
    os << "{\"mode\":\"calib\",\"txPowerDbm\":" << txp << ",\"m\":" << m << ",\"bytes\":" << g_calibBytes
       << ",\"points\":[";
    bool first = true;
    for (double d = dmin; d <= dmax + 1e-9; d += dstep)
    {
        NodeContainer nodes;
        nodes.Create(2);
        MobilityHelper mob;
        Ptr<ListPositionAllocator> pa = CreateObject<ListPositionAllocator>();
        pa->Add(Vector(0, 0, ALT));
        pa->Add(Vector(d, 0, ALT));
        mob.SetPositionAllocator(pa);
        mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
        mob.Install(nodes);
        WifiHelper wifi;
        wifi.SetStandard(WIFI_STANDARD_80211g);
        wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager", "DataMode",
                                     StringValue("ErpOfdmRate6Mbps"), "ControlMode",
                                     StringValue("ErpOfdmRate6Mbps"), "NonUnicastMode",
                                     StringValue("ErpOfdmRate6Mbps"));
        YansWifiChannelHelper ch;
        ch.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
        ch.AddPropagationLoss("ns3::LogDistancePropagationLossModel", "Exponent", DoubleValue(2.0),
                              "ReferenceDistance", DoubleValue(1.0), "ReferenceLoss",
                              DoubleValue(40.05));
        ch.AddPropagationLoss("ns3::NakagamiPropagationLossModel", "m0", DoubleValue(m), "m1",
                              DoubleValue(m), "m2", DoubleValue(m));
        YansWifiPhyHelper phy;
        phy.SetChannel(ch.Create());
        phy.Set("TxPowerStart", DoubleValue(txp));
        phy.Set("TxPowerEnd", DoubleValue(txp));
        WifiMacHelper mac;
        mac.SetType("ns3::AdhocWifiMac");
        NetDeviceContainer devs = wifi.Install(phy, mac, nodes);
        InternetStackHelper inet;
        inet.Install(nodes);
        Ipv4AddressGenerator::Reset();
        Ipv4AddressHelper ip;
        ip.SetBase("10.1.0.0", "255.255.0.0");
        ip.Assign(devs);
        g_calibRx = 0;
        Ptr<Socket> rs = Socket::CreateSocket(nodes.Get(1), UdpSocketFactory::GetTypeId());
        rs->Bind(InetSocketAddress(Ipv4Address::GetAny(), PORT));
        rs->SetRecvCallback(MakeCallback(&CalibRecv));
        Ptr<Socket> ts = Socket::CreateSocket(nodes.Get(0), UdpSocketFactory::GetTypeId());
        ts->SetAllowBroadcast(true);
        ts->Connect(InetSocketAddress(Ipv4Address("10.1.255.255"), PORT));
        for (uint32_t k = 0; k < npk; ++k)
        {
            Simulator::Schedule(MilliSeconds(10 * (k + 1)), &CalibSend, ts);
        }
        Simulator::Stop(MilliSeconds(10 * (npk + 5)));
        Simulator::Run();
        Simulator::Destroy();
        os << (first ? "" : ",") << "[" << d << "," << (double)g_calibRx / npk << "]";
        first = false;
    }
    os << "]}\n";
    return 0;
}

} // namespace

int
main(int argc, char* argv[])
{
    std::string trace, out = "emu.json", channel = "disk";
    double txp = 16.0206, nakM = 2.0;
    uint32_t run = 1;
    double cdMin = 100, cdMax = 3000, cdStep = 100;
    uint32_t cN = 500;
    int maxSlots = -1;
    int stride = 1;
    int offset = 0;
    bool hookTraces = true;
    CommandLine cmd(__FILE__);
    cmd.AddValue("maxSlots", "replay only the first k slots (profiling)", maxSlots);
    cmd.AddValue("slotStride", "replay every k-th slot of the trace (large n)", stride);
    cmd.AddValue("slotOffset", "first trace slot to replay (e.g. 5 to skip the initial deployment)", offset);
    cmd.AddValue("calibBytes", "calib: broadcast payload size (bytes)", g_calibBytes);
    cmd.AddValue("hookTraces", "connect the PHY/MAC trace sources", hookTraces);
    cmd.AddValue("trace", "trace file from export_traffic.py", trace);
    cmd.AddValue("mode", "gossip | controller | calib", g_mode);
    cmd.AddValue("channel", "disk | fading", channel);
    cmd.AddValue("txPowerDbm", "transmit power (dBm)", txp);
    cmd.AddValue("nakagamiM", "Nakagami m (fading channel)", nakM);
    cmd.AddValue("window", "traffic window at the start of each slot (s)", g_window);
    cmd.AddValue("run", "RNG run number", run);
    cmd.AddValue("out", "output JSON", out);
    cmd.AddValue("calibMin", "calib: first distance (m)", cdMin);
    cmd.AddValue("calibMax", "calib: last distance (m)", cdMax);
    cmd.AddValue("calibStep", "calib: distance step (m)", cdStep);
    cmd.AddValue("calibN", "calib: packets per distance", cN);
    cmd.Parse(argc, argv);
    RngSeedManager::SetSeed(20261126);
    RngSeedManager::SetRun(run);

    if (g_mode == "calib")
    {
        return RunCalib(txp, nakM, cdMin, cdMax, cdStep, cN, out);
    }

    auto wall0 = std::chrono::steady_clock::now();
    ReadTrace(trace);
    if (stride > 1 || offset > 0)
    {
        std::vector<Slot> kept;
        for (size_t s = offset; s < g_slots.size(); s += std::max(stride, 1))
        {
            kept.push_back(g_slots[s]);
        }
        g_slots.swap(kept);
    }
    {
        size_t S = g_slots.size();
        g_slotCtrl.assign(S, -1);
        g_slotDue.assign(S, 0);
        g_slotReach.assign(S, 0);
        g_slotRepRx.assign(S, 0);
        g_slotCmdRx.assign(S, 0);
        g_slotCtrlTx.assign(S, 0.0);
        g_slotMaxTx.assign(S, 0.0);
        for (const auto& sl : g_slots)
        {
            g_slotRnd.push_back(sl.rnd);
        }
    }
    if (maxSlots > 0 && (int)g_slots.size() > maxSlots)
    {
        g_slots.resize(maxSlots);
    }
    g_nodes.Create(g_n);
    g_pos.assign(g_n, Vector(0, 0, ALT));
    g_alive.assign(g_n, 0);
    n_txTime.assign(g_n, 0);
    n_rxTime.assign(g_n, 0);
    n_ccaTime.assign(g_n, 0);
    n_bytesAir.assign(g_n, 0);

    MobilityHelper mob;
    mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mob.Install(g_nodes);
    for (int i = 0; i < g_n; ++i)
    {
        g_nodes.Get(i)->GetObject<MobilityModel>()->SetPosition(
            Vector(1.0e7 + 1.0e5 * i, 0.0, ALT));
    }

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211g);
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager", "DataMode",
                                 StringValue("ErpOfdmRate6Mbps"), "ControlMode",
                                 StringValue("ErpOfdmRate6Mbps"), "NonUnicastMode",
                                 StringValue("ErpOfdmRate6Mbps"));
    YansWifiChannelHelper ch;
    ch.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    if (channel == "disk")
    {
        ch.AddPropagationLoss("ns3::RangePropagationLossModel", "MaxRange", DoubleValue(g_rcomm));
    }
    else
    {
        ch.AddPropagationLoss("ns3::LogDistancePropagationLossModel", "Exponent", DoubleValue(2.0),
                              "ReferenceDistance", DoubleValue(1.0), "ReferenceLoss",
                              DoubleValue(40.05));
        ch.AddPropagationLoss("ns3::NakagamiPropagationLossModel", "m0", DoubleValue(nakM), "m1",
                              DoubleValue(nakM), "m2", DoubleValue(nakM));
    }
    YansWifiPhyHelper phy;
    phy.SetChannel(ch.Create());
    phy.Set("TxPowerStart", DoubleValue(txp));
    phy.Set("TxPowerEnd", DoubleValue(txp));
    WifiMacHelper mac;
    mac.SetType("ns3::AdhocWifiMac");
    NetDeviceContainer devs = wifi.Install(phy, mac, g_nodes);
    wifi.AssignStreams(devs, 100);

    InternetStackHelper inet;
    Ipv4StaticRoutingHelper srh;
    inet.SetRoutingHelper(srh);
    inet.Install(g_nodes);
    Ipv4AddressHelper ip;
    ip.SetBase("10.0.0.0", "255.255.0.0");
    Ipv4InterfaceContainer ifs = ip.Assign(devs);
    NeighborCacheHelper nch;
    nch.PopulateNeighborCache();
    for (int i = 0; i < g_n; ++i)
    {
        g_ip.push_back(ifs.GetAddress(i));
        g_srt.push_back(srh.GetStaticRouting(g_nodes.Get(i)->GetObject<Ipv4>()));
        Ptr<Socket> s = Socket::CreateSocket(g_nodes.Get(i), UdpSocketFactory::GetTypeId());
        s->Bind(InetSocketAddress(Ipv4Address::GetAny(), PORT));
        s->SetAllowBroadcast(true);
        s->SetRecvCallback(MakeCallback(&OnRecv));
        g_sock.push_back(s);
    }
    g_u = CreateObject<UniformRandomVariable>();
    g_u->SetStream(7);

    bool ok = true;
    if (hookTraces)
    {
    ok &= Config::ConnectFailSafe("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/PhyTxPsduBegin",
                                  MakeCallback(&OnPsdu));
    ok &= Config::ConnectFailSafe("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Phy/State/State",
                                  MakeCallback(&OnState));
    ok &= Config::ConnectFailSafe(
        "/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/MacTxDataFailed",
        MakeCallback(&OnTxFailed));
    ok &= Config::ConnectFailSafe(
        "/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/RemoteStationManager/MacTxFinalDataFailed",
        MakeCallback(&OnTxFinalFailed));
    ok &= Config::ConnectFailSafe("/NodeList/*/DeviceList/*/$ns3::WifiNetDevice/Mac/DroppedMpdu",
                                  MakeCallback(&OnDropped));
    }
    NS_ABORT_MSG_IF(!ok, "a trace source failed to connect");

    for (size_t s = 0; s < g_slots.size(); ++s)
    {
        Simulator::Schedule(Seconds(s * g_dt), &StartSlot, (int)s);
    }
    double tEnd = g_slots.size() * g_dt;
    Simulator::Stop(Seconds(tEnd));
    auto wall1 = std::chrono::steady_clock::now();
    Simulator::Run();
    CloseSlot();
    auto wall2 = std::chrono::steady_clock::now();
    std::cerr << "setup " << std::chrono::duration<double>(wall1 - wall0).count() << " s, run "
              << std::chrono::duration<double>(wall2 - wall1).count() << " s, events "
              << Simulator::GetEventCount() << std::endl;

    // per-node figures over alive node-slots
    double nodeSlots = 0;
    for (const auto& s : g_slots)
    {
        nodeSlots += s.ids.size();
    }
    double maxTx = *std::max_element(n_txTime.begin(), n_txTime.end());
    double sumTx = 0, sumBusy = 0;
    for (int i = 0; i < g_n; ++i)
    {
        sumTx += n_txTime[i];
        sumBusy += n_txTime[i] + n_rxTime[i] + n_ccaTime[i];
    }

    std::ofstream os(out);
    os << std::setprecision(10);
    os << "{\"trace\":\"" << trace << "\",\"mode\":\"" << g_mode << "\",\"channel\":\"" << channel
       << "\",\"txPowerDbm\":" << txp << ",\"nakagamiM\":" << nakM << ",\"window\":" << g_window
       << ",\"run\":" << run << ",\"n\":" << g_n << ",\"slots\":" << g_slots.size()
       << ",\"slot_stride\":" << stride << ",\"slot_offset\":" << offset
       << ",\"R_comm\":" << g_rcomm << ",\"dt\":" << g_dt << ",\"node_slots\":" << nodeSlots;
    os << ",\"meta\":{";
    bool first = true;
    for (const auto& kv : g_meta)
    {
        os << (first ? "" : ",") << "\"" << kv.first << "\":\"" << kv.second << "\"";
        first = false;
    }
    os << "}";
    os << ",\"app\":{\"beacon_sent\":" << c_beaconSent << ",\"beacon_pairs_inrange\":" << c_beaconPairs
       << ",\"beacon_rx_inrange\":" << c_beaconRxIn << ",\"beacon_rx_outrange\":" << c_beaconRxOut
       << ",\"push_sent\":" << c_pushSent << ",\"push_rx\":" << c_pushRx << ",\"pull_sent\":" << c_pullSent
       << ",\"pull_rx\":" << c_pullRx << ",\"exchanges\":" << g_exList.size()
       << ",\"rtt_mean_s\":" << Mean(g_rtt) << ",\"rtt_p95_s\":" << Pct(g_rtt, 0.95)
       << ",\"rtt_max_s\":" << Pct(g_rtt, 1.0) << ",\"app_payload_bytes\":" << c_appBytes
       << ",\"rep_due\":" << c_repDue << ",\"rep_sent\":" << c_repSent << ",\"rep_rx\":" << c_repRx
       << ",\"cmd_due\":" << c_cmdDue << ",\"cmd_sent\":" << c_cmdSent << ",\"cmd_rx\":" << c_cmdRx
       << ",\"unreachable\":" << c_unreach << ",\"hops_mean\":" << Mean(g_hops)
       << ",\"hops_max\":" << Pct(g_hops, 1.0) << ",\"rep_lat_mean_s\":" << Mean(g_repLat)
       << ",\"rep_lat_p95_s\":" << Pct(g_repLat, 0.95) << ",\"cmd_lat_mean_s\":" << Mean(g_cmdLat)
       << ",\"cmd_lat_p95_s\":" << Pct(g_cmdLat, 0.95) << "}";
    os << ",\"phy\":{\"bytes_on_air\":" << b_air << ",\"f_data_unicast\":" << f_dataUni
       << ",\"f_data_retry\":" << f_dataRetry << ",\"f_data_bcast\":" << f_dataBcast
       << ",\"f_ack\":" << f_ack << ",\"f_other\":" << f_other << ",\"tx_time_total_s\":" << sumTx
       << ",\"tx_time_max_node_s\":" << maxTx << ",\"busy_time_total_s\":" << sumBusy
       << ",\"sim_time_s\":" << tEnd << "}";
    os << ",\"mac\":{\"tx_data_failed\":" << m_txFailed << ",\"tx_final_failed\":" << m_txFinalFailed
       << ",\"drop_retry_limit\":" << m_dropRetry << ",\"drop_lifetime\":" << m_dropLifetime
       << ",\"drop_other\":" << m_dropEnqueue << "}";
    os << ",\"node_tx_time_s\":" << JsonArr(n_txTime) << ",\"node_bytes_air\":" << JsonArr(n_bytesAir, 10)
       << ",\"node_busy_s\":";
    std::vector<double> busy(g_n);
    for (int i = 0; i < g_n; ++i)
    {
        busy[i] = n_txTime[i] + n_rxTime[i] + n_ccaTime[i];
    }
    os << JsonArr(busy);
    {
        std::vector<double> mxMs(g_slotMaxTx.size());
        for (size_t k = 0; k < g_slotMaxTx.size(); ++k)
        {
            mxMs[k] = 1e3 * g_slotMaxTx[k];
        }
        os << ",\"slot_max_tx_ms\":" << JsonArr(mxMs);
    }
    if (g_mode == "controller")
    {
        auto ints = [](const std::vector<int>& v) {
            std::vector<double> d(v.begin(), v.end());
            return JsonArr(d, 10);
        };
        std::vector<double> ctrlTxMs(g_slotCtrlTx.size());
        for (size_t k = 0; k < g_slotCtrlTx.size(); ++k)
        {
            ctrlTxMs[k] = 1e3 * g_slotCtrlTx[k];
        }
        os << ",\"ctrl_slots\":{\"rnd\":" << ints(g_slotRnd) << ",\"ctrl\":" << ints(g_slotCtrl)
           << ",\"ctrl_tx_ms\":" << JsonArr(ctrlTxMs) << ",\"due\":" << ints(g_slotDue)
           << ",\"reach\":" << ints(g_slotReach) << ",\"rep_rx\":" << ints(g_slotRepRx)
           << ",\"cmd_rx\":" << ints(g_slotCmdRx) << "}";
    }
    os << "}\n";
    Simulator::Destroy();
    return 0;
}
