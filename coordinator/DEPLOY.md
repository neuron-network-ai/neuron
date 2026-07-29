# Deploying the NEURON coordinator to a free-tier cloud VM

Goal: run the coordinator on a small, always-on **cloud** VM with a public address —
so nodes (and, later, strangers) can reach it **without exposing your personal
machines** (OptiPlex / Pavilion / your PC stay private, as nodes only).

The coordinator is tiny (FastAPI + SQLite, no torch). A free "micro" VM is plenty.

---

## 1. Create the VM  *(you do this — needs your account + card for verification)*

Pick an **always-on** free tier (avoid ones that sleep on idle — the coordinator
runs a health sweep and must always be reachable):

| Provider | Free tier | Notes |
|----------|-----------|-------|
| **Oracle Cloud — Always Free** | ARM Ampere (up to 4 vCPU / 24 GB) or 2× x86 micro | most generous; always on; ARM is fine (pure Python) |
| **Google Cloud** | `e2-micro` (us-west1/central1/east1) | always-free, x86, ~1 GB |
| AWS | `t2.micro` / `t3.micro` | free for 12 months only |

Steps (Oracle example):
1. Create the instance with **Ubuntu 22.04/24.04**. Save the SSH private key.
2. In the instance's **VCN → Security List / Network Security Group**, add an
   **ingress rule**: TCP **8001** (or 443 if you add TLS below) from `0.0.0.0/0`.
3. Note the VM's **public IP** (and optionally point a domain at it).

Then give me SSH access (host + key) and I deploy the rest in minutes. Or run the
steps below yourself.

---

## 0. Updating an already-running coordinator — use the script

Once the VM exists, don't hand-scp files: `coordinator/deploy.sh` ships the code, backs up the
database first, restarts the service, and **verifies the security gates are actually live**
before declaring success (rolling back if they aren't).

```bash
./coordinator/deploy.sh --dry-run     # show what would change
./coordinator/deploy.sh               # ship, restart, verify
```

It checks that `/wallet/faucet` and `/admin/identities` reject unauthenticated callers — those
gates are silently load-bearing, and a half-updated coordinator that still answers 200 to an
anonymous faucet call has the whole login/ban system switched off. It also sweeps any wallet
with no login behind it (only the old open faucet could have created one), returning its NRN to
`__ecosystem__` so the fixed-supply invariant still holds.

---

## 2. Deploy the coordinator  *(automatable once the VM exists)*

```bash
# on the VM (Ubuntu)
sudo apt update && sudo apt install -y python3-venv git
git clone <your-repo-or-scp-the-coordinator-dir> neuron && cd neuron
python3 -m venv .venv
./.venv/bin/pip install -r coordinator/requirements.txt

# a REAL registration secret (never the dev default on a public host)
export NEURON_REGISTER_SECRET="$(python3 -c 'import secrets;print(secrets.token_urlsafe(24))')"
echo "SAVE THIS -> $NEURON_REGISTER_SECRET"

# run it (bind all interfaces so the public IP works)
./.venv/bin/python -m uvicorn coordinator.main:app --host 0.0.0.0 --port 8001
```

Verify from your laptop: `curl http://<VM_PUBLIC_IP>:8001/status`.

### Keep it running (systemd)
```ini
# /etc/systemd/system/neuron-coordinator.service
[Unit]
Description=NEURON coordinator
After=network-online.target

[Service]
WorkingDirectory=/home/ubuntu/neuron
Environment=NEURON_REGISTER_SECRET=<your-real-secret>
ExecStart=/home/ubuntu/neuron/.venv/bin/python -m uvicorn coordinator.main:app --host 0.0.0.0 --port 8001
Restart=always
User=ubuntu

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable --now neuron-coordinator
sudo systemctl status neuron-coordinator
```

---

## 3. Point the network at the new coordinator

Everything defaults to the cloud coordinator `http://150.230.22.250:8001` (override with
`NEURON_COORDINATOR` / `--coordinator`; no code edits needed — all read env or flags):

- **Nodes / driver:** `node_a.py --coordinator http://<VM_PUBLIC_IP>:8001 ...`
- **UI + API server:** `NEURON_COORDINATOR=http://<VM_PUBLIC_IP>:8001 uvicorn ui.app:app ...`
- **Agent:** set `"coordinator"` in `agent/config.json` (or `install.py --coordinator ...`).
- **register_nodes.py:** `--coordinator http://<VM_PUBLIC_IP>:8001`.

---

## 3b. Login — configured HERE, once, for the whole network

Sign-in runs on the coordinator (`coordinator/auth.py`), **not** on each installed agent. It
has to: every agent serves its own Chat UI, so an OAuth *client secret* would otherwise sit on
every stranger's PC, where it is extractable from the binary — and this repo is public. Asking
each user to create their own Google Cloud project is not a product. Set it once here and every
existing and future install picks it up with no reinstall and no user configuration.

### GitHub — works today, no domain needed
GitHub accepts an `http://` callback on a bare IP.

1. GitHub → Settings → Developer settings → **New OAuth App**
2. Authorization callback URL: `http://150.230.22.250:8001/auth/callback/github`
3. Copy the Client ID, generate a Client Secret.

### Google — needs a DOMAIN and HTTPS first
Google **rejects** redirect URIs that are plain HTTP or a raw IP (only `localhost` is exempt),
so `http://150.230.22.250:8001/...` cannot be registered. Do the TLS step below first, then:

1. Google Cloud Console → Credentials → **Create OAuth client ID → Web application**
2. Authorized redirect URI: `https://neuron.example.com/auth/callback/google`
3. Copy the Client ID and Client Secret.

### Put them in the service
```ini
# /etc/systemd/system/neuron-coordinator.service  ->  [Service]
Environment=NEURON_GITHUB_CLIENT_ID=...
Environment=NEURON_GITHUB_CLIENT_SECRET=...
Environment=NEURON_GOOGLE_CLIENT_ID=...
Environment=NEURON_GOOGLE_CLIENT_SECRET=...
Environment=NEURON_PUBLIC_BASE_URL=https://neuron.example.com
```
```bash
sudo systemctl daemon-reload && sudo systemctl restart neuron-coordinator
curl http://150.230.22.250:8001/auth/providers      # -> {"providers":["github",...]}
```

`NEURON_PUBLIC_BASE_URL` **must exactly match** the redirect URI registered with the provider —
it is what the coordinator sends as `redirect_uri`, and providers compare it verbatim.

Only providers with BOTH an id and a secret are offered; the Chat UI draws buttons from
`/auth/providers`, so GitHub-only is a perfectly good launch state.

---

## 4. Before it's truly public — hardening (see PROBLEMS.md [P11], [P10])

- **TLS.** The coordinator speaks plain HTTP; node tokens would cross the internet in
  the clear. Put **Caddy** in front for automatic HTTPS (needs a domain):
  ```
  # /etc/caddy/Caddyfile
  neuron.example.com { reverse_proxy 127.0.0.1:8001 }
  ```
  then open 443 instead of 8001 and use `https://neuron.example.com`.
- **Registration secret** must be the real one (step 2), not `neuron-dev-secret`.
- `/infer` is unauthenticated by design (public inference) — add rate limiting before a
  real launch (ROADMAP S16).
- **Node ↔ node connectivity for strangers is still open** ([P10] sub-problem b): a
  public coordinator lets a stranger *register/heartbeat*, but pipeline traffic is still
  direct TCP between nodes. Until the coordinator brokers/relays that traffic (or nodes
  use Tailscale), a NAT'd stranger can't fully participate. The cloud VM is the natural
  place to host that relay later.

---

*The coordinator moving to the cloud also means your OptiPlex is no longer the network's
public front door — it can go back to being just node_b (or drop out entirely) without
taking the network down.*
