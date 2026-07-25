# NEURON

**N**etwork of **E**xisting **U**tilised **R**esources — **O**pen **N**odes

NEURON is a distributed LLM inference network built from ordinary computers. Instead of
running a whole model on one machine, it splits a transformer's layers into contiguous
slices and spreads them across nodes on a network: each node loads *only its own layers*,
runs them on the incoming activations, and passes the result to the next node over plain
TCP. A tiny (~20 KB) agent turns spare CPU on any laptop into one stage of a collective
inference pipeline. A **coordinator** tracks which nodes are online, routes each request to
a complete layer chain, and pays node operators in **NRN**, the network coin.

The current build runs **Qwen2.5-1.5B-Instruct** (28 layers) split across **three real
machines** with a per-node KV cache — numerically **bit-exact** versus running the whole
model on one machine — and proves the core thesis: adding nodes scales *throughput* (more
simultaneous users), which is where distribution actually pays off.

## Architecture

```
   User
    │  1. POST /infer {prompt}
    ▼
 ┌─────────────┐   returns the chain (which nodes / which layers / where) + request_id
 │ Coordinator │   registry · health-checks · routing · NRN ledger · dashboard
 └─────────────┘
    │  2. the client runs the returned pipeline, activations over TCP:
    ▼
  node_a ─────────► node_c ─────────► node_b
  layers 0–9        layers 10–18      layers 19–27
  embed + lm_head   (middle relay)    + final norm
    ▲                                     │
    └────────── final hidden state ◄──────┘   node_a applies lm_head, picks the
    │                                          next token, and loops until done
    │  3. POST /infer/{id}/complete  → coordinator credits NRN to each node
    ▼
   User  (receives the generated text)
```

One line: **User → Coordinator → [ node_a → node_c → node_b ] → Coordinator → User.**

## Scaling proof

Aggregate throughput on the same 4-prompt workload, as machines are added to the pipeline:

| Nodes | Pipeline | Throughput |
|------:|----------|-----------:|
| 1 | single machine | ~3.2 tok/s |
| 2 | `node_a → node_b` | 4.61 tok/s |
| 3 | `node_a → node_c → node_b` | **6.16 tok/s** |

All nodes run busy simultaneously (up to 3.8× pipeline overlap); parallel throughput beats
serial by ~3.6×. Scaling is sub-linear because the nodes are *heterogeneous* (the added
machines are slower) and the driver carries the output head — see `sessions.md` for the full
per-node utilisation breakdown and analysis.

## What hardware do I need?

Any machine with a normal CPU and a few GB of free RAM can be a node — no GPU required. A
node only holds **its slice** of the model (roughly `layers × ~110 MB` in fp32, plus the
~0.9 GB embedding on the first node). The reference network:

| Node | Role | Machine | CPU | RAM | OS |
|------|------|---------|-----|-----|-----|
| `node_a` | driver: embed + first layers + `lm_head` | Windows 11 PC | 16 cores | 63 GB | Windows 11 |
| `node_c` | middle layers (relay) | HP Pavilion (Ryzen) | 4 cores | 11 GB | Ubuntu 24.04 |
| `node_b` | last layers + norm | Dell OptiPlex | 6 cores | 15 GB | Ubuntu |
| coordinator | registry / routing / ledger | Dell OptiPlex (always-on) | — | — | Ubuntu |

Stack on every node: CPU-only `torch==2.4.1`, `transformers==4.44.2`, `accelerate`. The
coordinator needs only `fastapi` + `uvicorn`. Nodes reach each other over
[Tailscale](https://tailscale.com/); they need not share a Python minor version — only the
torch/transformers versions must match (that's what keeps the TCP tensor pickles compatible).

## How to run a node

A "node" is just one of the pipeline scripts, each serving a contiguous layer range:

```bash
# middle stage (node_c) — layers 10–18
python node_c.py --port 50999
# last stage (node_b) — layers 19–27 + norm
python node_b.py --port 50999
```
The first node (`node_a`) is the driver; it embeds the prompt, runs its layers, drives
generation, and applies `lm_head`. Each node loads only its own shard on first connect
(see `common.load_model_shard`).

**Direct mode** (no coordinator — hardcode the chain):
```bash
python node_a.py --host-c <node-c-ip> --host-b <node-b-ip> --s1 9 --s2 18 --max-new-tokens 80
```
Flags: `--serial` (baseline), `--copies 2` (N=8 concurrent), `--prompt "..."` (single
request), `--s1/--s2` (layer boundaries). Verify a split is bit-exact vs. one machine with
`python selftest_shard.py`.

## How to run the coordinator

The coordinator (FastAPI + SQLite, in `coordinator/`) is the network brain. From the repo
root:

```bash
pip install fastapi uvicorn
python -m uvicorn coordinator.main:app --host 0.0.0.0 --port 8001
```
Register the nodes and keep them alive (registers + heartbeats):
```bash
python coordinator/register_nodes.py
```
Then let `node_a` discover the chain from the coordinator instead of hardcoding it:
```bash
python node_a.py --coordinator http://<coordinator-host>:8001 --prompt "Why is the sky blue"
```

### Dashboard

`http://<coordinator>:8001/dashboard` is a plain, auto-refreshing (5 s) HTML page: a green
**HEALTHY** / red **DEGRADED** banner, summary cards (nodes online, layers covered, requests
served, NRN distributed), and a table of every node — layer range, online/offline badge,
address, cores, RAM, and NRN balance. Interactive API docs live at `/docs`.

### Coordinator API

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| POST | `/node/register` | `X-Register-Secret` | register a node, get a `node_token` |
| GET | `/node/list` | — | all nodes + status + layers |
| DELETE | `/node/{id}` | `X-Node-Token` | unregister |
| GET | `/node/{id}/ping` | `X-Node-Token` | heartbeat (every 30 s; offline after 90 s) |
| POST | `/infer` | — | get a valid chain `{chain, request_id}` |
| POST | `/infer/{id}/complete` | — | report done → credits NRN |
| GET | `/ledger/{id}` | — | balance / total_earned / requests_served |
| GET | `/status` | — | network + stats JSON |
| GET | `/dashboard` | — | HTML dashboard |

## How do I earn NRN?

Run a node, register it with the coordinator, and stay online. Every completed inference
request mints **1.0 NRN**:

- the coordinator keeps a **10% fee** (0.10 NRN),
- the remaining **0.9 NRN is split across the chain in proportion to layers held** — a node
  holding `L` of the model's 28 layers earns `0.9 · L/28` per request.

So a 10-layer node earns `0.9 · 10/28 = 0.321` NRN per request; a 9-layer node `0.289`. Over
a full chain the nodes share 0.9 and the coordinator keeps 0.1. Check your balance any time
at `GET /ledger/<node_id>`. (Tunable in `coordinator/config.py`.)

## Status

**Session 7 complete** — the coordinator is deployed to an always-on host (the OptiPlex,
`:8001`) and the full flow works end-to-end through it: register → route → infer → credit
NRN, with a live dashboard. 3-node pipeline proven at **6.16 tok/s**, bit-exact.

**Next:** nodes that self-report health, heterogeneity-aware auto-balancing of the layer
split, dynamic membership (nodes join/leave), and models too big for any single node (the
capacity case for distribution).

## Repository layout

```
common.py            model sharding, manual layer driver, KV cache, TCP framing
node_a.py            driver: embed + first layers + lm_head, parallel request driver
node_b.py            last stage: layers + final norm
node_c.py            middle relay stage
selftest_shard.py    proves the sharded split is bit-exact vs. one machine
coordinator/         FastAPI + SQLite: registry, health, routing, NRN ledger, dashboard
sessions.md          full engineering log, session by session
```

## License

Apache License 2.0 — see [LICENSE](LICENSE). © 2026 Raman Kumar Sharma.
