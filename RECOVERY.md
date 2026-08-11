# NEURON — Recovery

What to do when a release goes wrong, or a node ends up in a state it cannot leave on its own.

The problem this document exists for: **a volunteer's PC is behind a NAT in somebody's house.**
"Please reinstall it" is not an instruction this product can give at any scale, and the number of
machines one person can babysit is not a plan. Every procedure here is remote-first, and where a
remedy needs a human at the keyboard that is stated plainly as a limitation rather than a step.

Companion to `PROBLEMS.md` (what went wrong and why) and `RESILIENCE.md` (what survives a
failure). Update when a recovery is actually performed — a runbook nobody has run is a draft.

---

## 1. A bad agent release is live

**Symptom.** Nodes updated to a version that misbehaves: they stop serving, fail
proof-of-compute, crash on start, or serve wrong answers.

**The remedy: roll the fleet backwards.** Requires 0.20.1 or later on the node — see the
limitation at the end of this section.

Three values on the coordinator, all in the systemd drop-in:

```bash
# on the VM
sudo tee /etc/systemd/system/neuron-coordinator.service.d/zz-agent-release.conf >/dev/null <<'EOF'
[Service]
Environment=NEURON_AGENT_VERSION=<the KNOWN-GOOD version, e.g. 0.20.1>
Environment=NEURON_AGENT_SHA256=<sha256 of THAT version's installer>
Environment=NEURON_AGENT_ROLLBACK=1
EOF
sudo systemctl daemon-reload && sudo systemctl restart neuron-coordinator
```

Then confirm what the fleet will actually be told:

```bash
curl -s https://neuronnet.duckdns.org/agent/version
```

`version` must be the known-good one, `sha256` must match that installer, and `rollback` must be
`true`. Nodes pick it up within `CHECK_SECONDS` (24 h), skipping any node mid-request.

**The `zz-` prefix is not cosmetic.** systemd applies drop-ins in **lexical order** and this
service already has `google.conf`, `oauth.conf`, `override.conf` and `tiers.conf`. A file named
`agent-release.conf` sorts *before* `override.conf`, which re-assigns the same variables and wins
— the change appears to be applied and does nothing. This cost a live debugging cycle on
2026-08-11. Any new drop-in must sort after `override.conf`.

**Turn rollback off again once the fleet is back.** Leaving `NEURON_AGENT_ROLLBACK=1` set means
the next version bump can walk nodes backwards as easily as forwards.

**Why an explicit flag rather than just publishing the old version.** `is_newer` refuses to
install anything that is not newer, which is the property that stops a stale or hostile version
field walking a fleet backwards. That same line is what made a bad release unrecoverable. So the
downgrade exists but must be *asked for*: the version alone never implies it. Every other
guarantee still holds — the SHA-256 must match, a source checkout is never touched, and a node
serving a request is never interrupted.

**Limitation, stated honestly.** An agent older than 0.20.1 ignores `rollback` entirely. A node
that is dead enough not to run its updater thread cannot be reached by any of this either. For
those, section 3.

---

### Confirming a rollback actually took

From 0.20.2 a node reports the build it is running, whether auto-update is on, and the verdict of
its last check. Before that the recovery above fired blind — you flipped the switch and the only
evidence was behaviour changing.

```bash
curl -s -H "X-Register-Secret: $NEURON_REGISTER_SECRET" \
  https://neuronnet.duckdns.org/node/list \
  | python -c "import json,sys; [print(f\"{n['node_id']:32} {n.get('agent_version') or 'unreported':10} auto={n.get('auto_update')} last={n.get('update_check')}\") for n in json.load(sys.stdin)['nodes']]"
```

The public dashboard states the same thing in aggregate (*"3 of 4 online node(s) on the latest
agent"*), and each operator sees their own node's version on their private page.

**A node still on the old build is one of three things, and it takes all three fields to tell
them apart:** it has not made its daily check yet (fixes itself), `auto_update` is off (needs the
operator), or its download is failing (`update_check` says so). `agent_version: null` means an
agent older than 0.20.2 — **not** "up to date".

**Allow up to 24 h before concluding a rollback failed.** `CHECK_SECONDS` is 86400 and a node
defers while serving.

## 2. Publishing a release (and the two ways it silently does nothing)

1. Bump all three, which must agree: `agent/updater.py:LOCAL_VERSION`,
   `packaging/neuron.iss:AppVersion`, `coordinator/config.py:AGENT_VERSION`.
2. Build: PyInstaller via `packaging/neuron-agent.spec`, then
   `ISCC.exe packaging\neuron.iss` → `dist/installer/NEURON-Setup-<ver>.exe`.
3. Publish a GitHub release tagged **`v<ver>`** with the asset named exactly
   **`NEURON-Setup-<ver>.exe`**. `AGENT_DOWNLOAD_URL` is *derived from the version string*, so a
   mismatched tag or filename means every node downloads a 404.
4. Set `NEURON_AGENT_VERSION` and `NEURON_AGENT_SHA256` in the drop-in (see the `zz-` note above)
   and restart.
5. Verify `/agent/version` returns all three fields, and that `download_url` resolves with a
   `content-length` equal to the built installer.

**`AGENT_SHA256` empty means no node installs anything.** That is the intended kill switch and
the correct failure direction: an unverified binary pushed to every volunteer's machine is the
worst thing this project could do, and it would be doing it automatically. Publish the release
first, confirm the URL resolves, and set the hash last.

---

## 3. A node is broken and cannot be reached

Ordered by how much of the machine still works.

**It still runs and talks to the coordinator** → section 1 covers it.

**It runs but holds the wrong layers.** The coordinator owns placement ([P32]), but an agent
before 0.20 never re-asks where it belongs once its config has a range, so it will not download
a corrected slice. From 0.20 the node reports `holds` in its challenge ack and the verifier logs
`challenged on layers A-B but the node holds C-D` — that line is the diagnosis. Correct the
placement with `./coordinator/pin_layers.sh`, then section 4.

**It does not run at all.** There is no remote remedy, and pretending otherwise would be the
dangerous kind of documentation. `apply_update` ends in `os._exit(0)` with no rollback, no
resume and no disk-space check. Someone at that machine must install the last good release from
the [Releases page](https://github.com/neuron-network-ai/neuron/releases). An in-place install
keeps `config.json`, the payout key and the model slice — they live in `%LOCALAPPDATA%\NEURON`,
not in the program directory, so the node keeps its identity and its earnings.

**Nothing here should cost the volunteer their NRN.** Earnings live in the coordinator's ledger
against the node id, not on the machine. A reinstall that preserves `config.json` keeps the node
id, and therefore the balance.

---

## 4. A node is flagged and cannot climb back

**Read this before resetting anything.** Reputation is derived from cumulative counters that only
grow, so a node deep in the hole (`1/23` on 2026-08-11) cannot reach the 0.6 threshold on passes
alone. `POST /node/{id}/reputation-reset` exists for exactly that. It is also the wrong first
move almost every time.

**Fix the cause first, then reset.** A flag is usually not the node's fault:

- **Placement drift** — the node is being challenged on layers it does not hold, so it fails
  deterministically ([P37] flagged three honest machines this way). While `placement_drift` is
  set the verifier now records nothing at all, so the counters stop moving; they do not un-move.
- **A half-downloaded slice** — the node's own dashboard says so, and the remedy is on the
  machine: stop the agent, delete the slice directory, let it re-download.

Resetting before the cause is fixed clears the counter on a node that then simply re-earns the
flag, and the second flag looks like confirmation of the first.

```bash
# only after the node passes a challenge again
curl -X POST -H "X-Register-Secret: $NEURON_REGISTER_SECRET" \
  https://neuronnet.duckdns.org/node/<node_id>/reputation-reset
```

Then watch `verify_service.log` for `<node> passed while FLAGGED` — the flag lifting is logged at
INFO precisely because it is the network recovering a machine it had written off.

---

## 5. The coordinator itself

`./coordinator/deploy.sh` backs up the database before shipping anything, verifies `/status` and
the auth gates afterwards, and restarts the service if verification fails. The database is never
touched except to copy it aside: it holds every identity, wallet balance and ban, and
`models.init_db()` migrates the schema forward on startup by itself.

To undo a coordinator deploy, ship the previous commit's tree — the code is stateless. To undo a
drop-in change, delete the file, `daemon-reload`, restart.

**Do not restore a systemd unit from an old backup without reading it.** Unit backups on the VM
have held retired copies of `NEURON_REGISTER_SECRET`; restoring one wholesale would reinstate a
compromised credential *and* 401 every client using the current one (Session 56).
