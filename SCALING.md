# NEURON — Scaling Architecture (prototype → worldwide)

**The question this answers:** the current setup (one cloud coordinator + one relay VM,
~100 relay ports) is fine for the first strangers — does it scale to a worldwide network
of millions of nodes? **No — and it isn't meant to.** You never build planet-scale infra
before the first 10 users. This doc is the honest plan for how NEURON gets from a 3-node
prototype to a global network, and where today's work fits.

Companion docs: `ROADMAP.md` (session plan), `PROBLEMS.md` (risks/decisions), `sessions.md`
(build log). This file = the long-horizon scaling design.

---

## Where we are today (the prototype ceiling)

Everything routes through **one free micro-VM**: the coordinator (registry/router/ledger)
and the reverse-tunnel relay (`relay.py`, ~100 public ports = ~100 relayed nodes). Nodes
that are on Tailscale talk peer-to-peer; NAT'd/stranger nodes reach each other via the relay.

- ✅ Correct for **Phase 1** (first strangers, dozens of nodes).
- ❌ Single point of failure + bottleneck; hard cap on relayed nodes; all relayed traffic
  funnels through one box in one region.

**This is expected.** The prototype exists to prove the mechanism and get the first stranger
earning. The scaling layers below are grown into as real demand appears.

---

## The three things that must change for scale

### 1. Connectivity — stop routing everything through the relay
Relaying-through-one-VM is a **fallback**, not the main path. At scale:
- Most node↔node traffic goes **direct, peer-to-peer**, via NAT hole-punching
  (STUN / ICE / WebRTC / `libp2p`). Direct connections succeed for the majority of home
  NATs.
- Relays become the **rare exception** (symmetric/hard NAT), and there are **many** relays
  worldwide (a relay fabric), not one. Connectivity load spreads across the network itself.
- Interim step before full P2P: single-port relay multiplexing (one port for all nodes,
  routed by node_id) or fold the relay into the coordinator's existing port 8001 over
  WebSocket — removes the per-port firewall limit entirely (never open another port).

### 2. Coordination — one coordinator can't be the brain of the planet
- **Near term:** a few **regional coordinator instances** behind a load balancer; stateless
  API + a replicated/sharded DB. Removes the single point of failure.
- **Real scale:** **decentralized peer discovery via a DHT** (distributed hash table, à la
  BitTorrent/Kademlia) — nodes find each other and assemble pipelines with **no central
  server**. This is where the blockchain/decentralization roadmap (S17) belongs: the ledger
  and discovery go on-chain / into the DHT rather than a single SQLite file.

### 3. Topology — many small pipelines, not one giant one
The key insight: **a million nodes ≠ a million-stage pipeline.** Deep pipelines are slower
(more hops) and fragile. Instead:
- The network is **hundreds of thousands of independent ~3–8-node pipelines running in
  parallel** (replicas of the model split across small groups).
- **Aggregate throughput scales ~linearly** with node count — that's the whole thesis
  (capacity, not single-user speed; see PROBLEMS.md [P1], [P8]).
- The coordinator's job becomes: **match each request to a nearby, healthy pipeline**
  (load balancing across the swarm) and **continuously re-form pipelines** as nodes
  join / sleep / leave.

---

## Hard problems that come with scale

- **Churn:** millions of volunteer machines constantly join/leave/sleep. Pipelines must
  re-form dynamically; requests reroute mid-flight; replication provides redundancy.
- **Latency-aware assembly:** pipelines should be built from network-close nodes (same
  region) — network is already a dominant per-token cost (PROBLEMS.md [P3]).
- **Trust / correctness at scale:** strangers' nodes could return garbage. Needs
  proof-of-compute + reputation (ROADMAP S16) so bad nodes don't poison results or steal NRN.
- **Model quality vs. size vs. speed:** bigger models need more nodes per pipeline and are
  slower (PROBLEMS.md [P6]) — a permanent three-way trade to manage per model.

---

## This is a solved model (not speculative)

**Petals** (BigScience) already runs 100B+ parameter models across volunteer machines
worldwide using exactly this shape: a **DHT for discovery + direct P2P connections + relay
fallback**, with hundreds of real nodes. So the networking at scale is proven engineering.
NEURON's differentiator is the **packaging** — 1-click, no-GPU, no crypto-staking, earn-while-
idle — **not** inventing new distributed-systems primitives. When we build the scale layer,
we lean on proven libraries (`libp2p`, `hivemind`) rather than from scratch.

---

## Phased plan

| Phase | Nodes | Connectivity | Coordination | Status |
|-------|-------|--------------|--------------|--------|
| **1 — Prototype (now, S12)** | 1–50 | Tailscale + single relay VM | one cloud coordinator | ✅ coordinator live; relay built |
| **2 — Growth** | 50–5,000 | single-port / port-8001 relay mux | a few regional coordinators, latency-aware assembly | planned |
| **3 — Scale** | 5k–millions | P2P hole-punching, relay fabric | DHT peer discovery (no central coordinator), on-chain ledger | future (leans on libp2p/hivemind) |

Rough mapping to `ROADMAP.md`: S12 = Phase 1 (first stranger); S16 (security/proof-of-compute)
+ S17 (on-chain NRN) + S18/S19 (launch + scale testing) = the Phase 2→3 groundwork.

---

## The one rule of sequencing

**Don't build the worldwide layer now.** It is real, substantial engineering (weeks–months)
and is the *wrong* investment before there are real users. Get the **first stranger earning**
(proves people will run it at all), watch how real nodes actually behave, and design the
scaling architecture from that evidence. Prototype → learn → scale, in that order.
