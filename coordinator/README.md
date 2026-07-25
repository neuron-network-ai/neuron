# NEURON Coordinator

The brain of the NEURON network: a FastAPI + SQLite service that tracks nodes,
health-checks them, routes inference requests to a valid layer chain, and pays
node operators in **NRN** (the network coin).

No Docker, no DB setup — SQLite is created on first run (`coordinator/neuron.db`).

## Install

```bash
C:\Users\user\neuron\.venv\Scripts\python.exe -m pip install fastapi "uvicorn[standard]"
```
(`requests` is already in the venv for the register/heartbeat script and node_a.)

## Run

From the project root `C:\Users\user\neuron\`:

```bash
C:\Users\user\neuron\.venv\Scripts\python.exe -m uvicorn coordinator.main:app --reload --port 8000
```

Then open **http://localhost:8000/dashboard** (auto-refreshing HTML) or
**/docs** for the interactive API.

## Register the nodes + keep them alive

```bash
C:\Users\user\neuron\.venv\Scripts\python.exe coordinator\register_nodes.py
```
This registers node_a (0–9), node_c (10–18), node_b (19–27), saves their tokens to
`coordinator/node_tokens.json`, and then heartbeats: every 30 s it checks whether
each server node's port is really listening and pings the coordinator on its
behalf. Kill a node's server and it goes offline after 90 s. Add `--register-only`
to just register and exit.

## Run inference through the coordinator

```bash
C:\Users\user\neuron\.venv\Scripts\python.exe node_a.py --coordinator http://localhost:8000 --prompt "Why is the sky blue"
```
node_a asks the coordinator for the chain, runs the 3-stage pipeline, and reports
completion so NRN is credited. (`node_b.py`/`node_c.py` must be running.)

## API

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| POST | `/node/register` | `X-Register-Secret` | register a node, get a `node_token` |
| GET | `/node/list` | — | all nodes + status + layers |
| DELETE | `/node/{id}` | `X-Node-Token` | unregister |
| GET | `/node/{id}/ping` | `X-Node-Token` | heartbeat (call every 30 s) |
| POST | `/infer` | — | get a valid chain `{chain, request_id}` |
| POST | `/infer/{id}/complete` | — | report done → credits NRN |
| GET | `/ledger/{id}` | — | balance / total_earned / requests_served |
| GET | `/status` | — | network + stats JSON |
| GET | `/dashboard` | — | plain HTML dashboard |

## Security (basics)

- **Registration** requires the shared `X-Register-Secret` header (default
  `neuron-dev-secret`, set via env `NEURON_REGISTER_SECRET`) — stops random people
  registering fake nodes.
- Each node gets a **`node_token`**; its own `ping`/`delete` require the matching
  `X-Node-Token`.

## Economics (NRN)

Each completed request mints **1.0 NRN**. The coordinator always keeps a **10%
fee**; the remaining **0.9** is split across the chain **proportionally to layers
held** — a node with `L` of 28 layers earns `0.9 · L/28`. Over a full chain the
nodes share 0.9 and the coordinator keeps 0.1. Tunable in `config.py`.

## Files

- `main.py` — FastAPI app (endpoints, auth, background health sweep, dashboard)
- `models.py` — SQLite storage (nodes / ledger / requests)
- `router.py` — chain assembly + gap detection
- `ledger.py` — NRN reward split
- `config.py` — settings (env-overridable)
- `register_nodes.py` — register the 3 nodes + heartbeat prober
