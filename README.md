# NEURON

**N**etwork of **E**xisting **U**tilised **R**esources — **O**pen **N**odes

NEURON is a distributed LLM inference network built from ordinary computers. Instead of
running a whole model on one machine, it splits a transformer's layers into contiguous
slices and spreads them across nodes on a network: each node loads *only its own layers*,
runs them on the incoming activations, and passes the result to the next node over plain
TCP. A tiny (~20 KB) agent turns spare CPU on any laptop into one stage of a collective
inference pipeline. The current build runs **Qwen2.5-1.5B-Instruct** (28 layers) split
across **three real machines** with a per-node KV cache — numerically **bit-exact** versus
running the whole model on one machine — and it proves the core thesis: adding nodes scales
*throughput* (more simultaneous users), which is where distribution actually pays off.

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

## How to run

Three stages, three commands. Nodes load their shard on first connect.

**1. Last stage** (layers 19–27 + norm) on the OptiPlex:
```bash
ssh homeadmin@100.114.189.46 "cd ~/neuron && ./.venv/bin/python node_b.py --port 50999"
```

**2. Middle stage** (layers 9–17) on the Pavilion:
```bash
ssh raman@100.79.125.112 "cd ~/neuron && ./.venv/bin/python node_c.py --port 50999"
```

**3. Driver** (embed + layers 0–8 + `lm_head`) on the Windows PC — runs the 4-prompt batch:
```bash
python node_a.py --host-c 100.79.125.112 --host-b 100.114.189.46 --s1 9 --s2 18 --max-new-tokens 80
```

Flags: `--serial` (one-at-a-time baseline), `--copies 2` (N=8 concurrent), `--prompt "..."`
(single request). `--s1/--s2` set the 3-way layer boundaries. Verify correctness (bit-exact
3-stage chain) with `python selftest_shard.py`.

*(Addresses above are the author's Tailscale IPs — substitute your own nodes.)*

## Hardware used

| Node | Role | Machine | CPU | RAM | OS |
|------|------|---------|-----|-----|-----|
| `node_a` | driver: embed + first layers + `lm_head` | Windows 11 PC | 16 cores | 63 GB | Windows 11 |
| `node_c` | middle layers (relay) | HP Pavilion (Ryzen) | 4 cores | 11 GB | Ubuntu 24.04 |
| `node_b` | last layers + norm | Dell OptiPlex | 6 cores | 15 GB | Ubuntu |

Stack on all three: CPU-only `torch==2.4.1`, `transformers==4.44.2`, `accelerate`. Nodes
reach each other over [Tailscale](https://tailscale.com/); they need not share a Python
minor version — only the torch/transformers versions must match.

## Status

**Session 5 complete** — 3-node pipeline proven at 6.16 tok/s, bit-exact.
**Next:** a coordinator to manage node membership and layer assignment dynamically (so nodes
can join/leave and the split auto-balances to the hardware).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
