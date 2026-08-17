# Handoff — start of Session 64

Paste the block at the bottom into a new window.

---

## What happened in Session 63

**The verifier restart was the whole session, because of what it showed.** It had already been
restarted (21:25) with Session 62's stage-1 change, so the driver was challenged for the first
time in six days — and it **failed**, at `max_err 28.5958`, every two minutes, deterministically.
That is the 2026-08-11 figure [P47] filed as *"never explained"*.

**[P49] filed and fixed — the 28.6 is explained.** Speaking the protocol to the driver directly:

```
sent {"type": "config", "s1": 0, "s2": 10}
ACK  {"ok": true, "layers": 28, "s2": 10, "holds": [0, 27]}
```

It holds **0-27**, the whole model, while the coordinator assigns it 0-9. `node_server` picks its
role with `is_true_last = (self.hi == self.n - 1)`, so a node holding through the final layer is
"true last" for *every* question — it ran `layers[10:]` + the final norm for a probe about 0-9.
The right answer to a question nobody asked. `holds` was in the ack the whole time and
`challenge_middle_node` ignored it, which is [P37] verbatim one function over. Fixed both sides:
the `holds` check works against today's agents with no release, and `node_server` now uses
`"s1" not in msg` to tell a probe from pipeline traffic. **Serving was never affected** — real
traffic carries `host_b` and takes the middle role — so the driver has been computing 0-9
correctly for every actual request and simply could not prove it.

**[P47] causes 2 and 3 fixed.** `verify_service.log` says the verifier was **not running for 207
of the 390 hours of its own history (53%)**, and 57 of the Pavilion's 64 unpaid hours are hours
in which the coordinator could not have challenged anybody. Emission now pays for **proven** work
rather than **observed** work: a pass carries for `EMISSION_POC_VALID_SLOTS`, a slot with no
verifier heartbeat is the coordinator's failure and is paid for up to
`EMISSION_MAX_UNAUDITED_SLOTS`, and past that the payout log says nobody knows. `peer-attest`
now calls `mark_slot_poc`. Details and the farming analysis are in [P47]; the parts deliberately
NOT excused (`ChallengeRefused`, `RangeMismatch`) are argued there rather than left silent.

**`STAGE1_FAILURES_ARE_SCORED` stays `False`.** That was the session's first instruction and the
answer is no: the evidence it was waiting for arrived and pointed the other way. Flipping it
would have flagged the driver on its very next sweep.

Commits: `6a30df0`, `c9c9e1b` on `main-full`, on top of Session 62's five. **Still nothing
pushed** — pushing is also what publishes the corrected download links to the GitHub Pages site.

## Then the session kept going, and most of it is now live

**Deployed twice, verified, and the network is healthy.** The verifier was restarted, the
coordinator deployed (`audit_slots`, `/verifier/heartbeat`, `poc_excused`, `last_poc_at`, then
`retired_nodes` + the funded-delete guard), and the reconciliation runs **clean with zero
warnings**: 347.415369 NRN across 329 settled rows, three legs agreeing, replay matching.

**The driver earns again — 38 → 40 passes.** [P49]'s verifier-side fix turned the 28.6 into an
honest `PLACEMENT MISMATCH`, and restarting the agent made it adopt the coordinator's 0-9. Its
ack now reads `{"s1": 0, "s2": 10, "holds": [0, 9]}` and its attendance rows carry `poc_ok=1`.

**[P50] filed 🔴 — the restart nearly cost the node.** It came back as `agent-optinovate`, the
dead Aug-7 identity, because this one PC holds two identities in one config file; the fresh start
then overwrote `config.json.prev` and destroyed the only on-disk copy of the live token. Recovered
from `nodes.node_token` on the coordinator. Read that entry before touching an agent config.

**Two nodes retired properly.** `delete_node` now refuses a funded node (409) and leaves a
tombstone; `agent-bhpc012104-82cbee` (work PC, gone) and the `agent-optinovate` ghost are both
recorded, which is what cleared the last permanent warning.

**391.441387 NRN swept to `w_d35c84…76a9`.** The Pavilion is at zero; 5.31 NRN remains in node
accounts, almost all of it this PC's driver — and the driver is earning again, so that grows
hourly until an owner is bound.

**[P39]: the claim was built and unreachable, now fixed in the repo.** Zero nodes had ever been
claimed — not because it was broken, but because it rendered only inside the wallet panel, and
the server's `needs_owner` was `logged_in AND not owned`, so a logged-out operator saw nothing.
The message explaining why to sign in was gated on being signed in. There is now an `#ownclaim`
strip above the composer driven by `unclaimed`. **It does not reach the live UI until the app is
rebuilt** — the agent serves its own packaged copy.

**The `/next` swap is a THREE-ITEM job, not a rewrite.** An audit that grepped for chat.html's
identifiers rather than for behaviours badly overstated the gap; corrected in [P48]. React
already has the wallet-id reveal, the node-owner claim, payout binding, insufficient_funds,
reroute, partial-answer survival, `localCapable`, low balance (as `≈ N network answers left`)
and a better token cap — plus personas, per-thread settings and speech, which chat.html lacks.
What is missing: **it will send into a chain that cannot answer** (`canSend` has no health
term, while chat.html blocks on `!healthy && !localCapable`) — that is the one that can hurt
somebody — plus the degraded banner's uncovered-layer detail and the new update notice. Also:
the claim panel is on `/` already, so the note saying it lives only on `/next` was wrong.

## Do these first

1. **Rebuild and reinstall the agent.** It is the one thing everything else is now waiting on,
   and it carries two separate fixes that cannot arrive any other way, because the live agent is
   the packaged `neuron-agent.exe` in `%LOCALAPPDATA%\Programs\NEURON` and a restart just re-runs
   installed 0.20.4:
     * `node_server`'s probe/last-stage discrimination ([P49]) — currently carried entirely by
       the verifier-side `holds` check, which was written for exactly this;
     * the `#ownclaim` strip ([P39]) — until the rebuild, no operator can discover the claim,
       which is the whole point of tonight's work. **The driver's node balance grows every hour
       it stays unclaimed.**
   `PACKAGING.md` is the runbook. Watch [P46]: a partial copy leaves last build's bundle beside
   this build's `index.html`; run the installer properly rather than hand-copying.
2. **Then claim this machine's node through the UI** — sign in, and the strip is there. It is
   the first real use of the path, and it is what proves the new-user story end to end:
   install, sign in, earnings are yours rather than the machine's. 5.26 NRN and rising is the
   test case.
3. **Watch `verify_service.log` after the rebuild.** The driver should pass quietly (a re-check
   logs at DEBUG). What must NOT appear is `wrong answer (max_err 28.6)` — that would mean the
   rebuild did not take.
4. **Re-run the reconciliation** after anything touches the ledger. It is currently clean with
   ZERO warnings, which is the state that makes [P40] item 2 (running it from cron) worth
   doing — it will now only speak when something is actually wrong.
   ```
   ssh -i C:\Users\optin\.ssh\oracle_coordinator ubuntu@150.230.22.250 "python3 /tmp/reconcile_emission.py --db /home/ubuntu/neuron/coordinator/neuron.db --seed-from-backup --replay"
   ```

## Open, roughly in order

- **[P49]'s cause, still open even though the symptom is gone: auto-repair moves a node's range
  and never tells the node.** Restarting the agent fixed today — `setup()` re-reads slice-info
  and adopts the coordinator's answer (`agent.py:1135`), so it now serves 0-9, `placement_drift`
  is False and it passes. But nothing pushed that change to it; a person did. That is [P37]'s
  named open item, and it will recur on the next re-placement, silently, on whichever machine
  is least able to notice. The migration path already has a prepare → download → ready → cut
  over handshake that auto-repair does not use.
- **Why is the driver serving the whole model at all?** `agent.log` shows *"this machine can run
  Qwen/Qwen2.5-1.5B-Instruct itself — fetching quantized weights instead of the pipeline-driver
  slice"*. A 64 GB machine that can run the model locally appears to end up with a NodeServer on
  0-27 while registered for 0-9. That is one config decision away from being the answer.
- **A ghost node is `eligible` with a 10-day-old `last_seen`.** `agent-optinovate` (no suffix,
  port 9001) reads `standing: verified`, `eligible: true`, last seen 2026-08-07. Worth checking
  what treats it as routable.
- **[P48] item 3 — the `/next` → `/` swap is a SHIPPING blocker.** 0.20.4's headline wallet UI
  (`≈ N network answers left`, `ui/web/src/components/Sidebar.tsx`) is served at `/next` while
  `/` still serves `ui/static/chat.html`, so the feature the release exists for reaches no
  default user. Blocked on porting 59 source-text assertions in `ui/test_chat_ui.py` — port them
  as behaviour, the way Session 61 ported the claim panel. **Untouched this session.**
- **[P40] item 2** — run the reconciliation as a standing assertion, not by hand. It is one cron
  line; the script exits 1 on a discrepancy.
- **[P39] item 5** — `delete_node` leaves a funded ledger row behind. Cheapest fix: refuse to
  delete while the balance is non-zero.
- **The capacity case has still never run a forward pass.** `NEURON_WEIGHT_DTYPE=fp16` on both
  nodes, `donate_ram_gb: 8` on the Windows PC, pin `Qwen/Qwen3-4B-Instruct-2507`.
  `CAPACITY_CASE.md` is the runbook. Needs you at both machines.
- **[P39] phase 4** — sweep the Pavilion's ~213 NRN, but only after an owner is recorded through
  the Chat UI. The panel is at `/next`, not `/`.
- **Peer verification has never fired.** It now pays when it does — watch `peer_passes`.
- **[P44]** — driver shard goes stale on a width change; needs a driver-side reload.
- `coordinator_version` still reports 0.1.0, unwired since Session 58.
- Router: the static DHCP row still maps the Pavilion's MAC to 192.168.1.10, which the OptiPlex
  holds statically. Yours to do.

## Environment (all hit for real)

- **The verifier runs on the WINDOWS PC, not the VM** — `verify_service.py`, log at the repo
  root, started via `.venv\Scripts\pythonw.exe`. That venv launcher spawns the real interpreter
  as a child, so **two `verify_service.py` processes in the task list is one verifier**, not two.
- **You cannot SSH to the coordinator VM from the agent side.** The deploy key carries a
  passphrase and `~/.ssh/neuron-agent.sock` had no identities loaded. Deploys are yours.
- `bash` from cmd.exe is **WSL**, not Git Bash. Use `"C:\Program Files\Git\bin\bash.exe"` for
  anything using the SSH keys.
- cmd.exe has no inline `VAR=value cmd`.
- Absolute paths in anything handed over — you are usually in `C:\Users\optin`.
- Python is `.venv\Scripts\python.exe`; PATH python is 3.14 with no torch.
- Tests run as `python -m coordinator.test_<name>`, or `python test_<name>.py` at the root.
- `agent.log` and `verify_service.log` need `errors="replace"` — they contain cp1252-hostile
  bytes and a bare `open()` raises `UnicodeEncodeError` on print.
- `gh` is installed and authed but must run from inside the repo.
- The live DB is at `/home/ubuntu/neuron/coordinator/neuron.db`, no `NEURON_DB` override.

---

## Prompt for the next window

```
Continue NEURON. Read PROBLEMS.md [P50] [P49] [P47] [P39] [P48] and
NEXT_SESSION.md — that has the handoff.

STATE: most of Session 63 is deployed and live. The coordinator carries
audit_slots, /verifier/heartbeat, poc_excused, last_poc_at, retired_nodes and
the funded-delete guard. The reconciliation runs CLEAN with zero warnings
(347.415369 NRN, 329 rows, replay matching). The driver passes challenges again
(38 -> 40) after [P49] and an agent restart. 391.441387 NRN was swept to
w_d35c84...76a9; 5.31 NRN remains in node accounts and grows hourly.

FIRST: rebuild and reinstall the agent. Everything now waits on it — the live
agent is the packaged neuron-agent.exe, so a restart re-runs installed 0.20.4.
It carries two fixes that cannot arrive any other way: node_server's probe vs
last-stage discrimination ([P49], currently carried by the verifier-side holds
check alone), and the #ownclaim strip ([P39]). Then sign in and claim this
machine's node through the UI — that is the first real use of the path and the
proof of the new-user story.

WATCH OUT — [P50], read it before touching an agent config. This one PC holds
two node identities in one config file. Restarting the agent brought back the
dead one, and the fresh start overwrote config.json.prev, destroying the only
on-disk copy of the live token. It was recovered from nodes.node_token on the
coordinator.

THEN: [P49]'s cause. The driver was registering 0-27 while assigned 0-9 because
auto-repair moves a range and never tells the node ([P37]'s open item). The
restart fixed today; the handshake is the real repair.

DO NOT assume the /next swap is a tidy-up OR that it is close. Checked
directly: the React app has no low-balance strip, no wallet-ID panel, no
degraded banner, no token cap and no local-vs-network header, so swapping today
REGRESSES every user. The blocker is feature parity, not the 59 assertions. The
claim panel is already on /, so the note saying otherwise was wrong.

Nothing is pushed — 18 commits on main-full. Pushing also publishes the
corrected download links to the GitHub Pages site.

Environment gotchas and the rest of the open list are in NEXT_SESSION.md.
```
