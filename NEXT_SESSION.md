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

## Do these first

1. **Restart `verify_service.py`** — it runs from the repo, so this alone picks up the
   verifier-side `holds` check and turns the driver's failure into an honest *PLACEMENT
   MISMATCH, not a bad node* line instead of a wrong answer scored against it. Do this first;
   it is the cheap half and needs nothing else.

   **The `node_server` fix does NOT arrive with a restart.** The live agent is the packaged
   `neuron-agent.exe` in `%LOCALAPPDATA%\Programs\NEURON` — restarting it re-runs the installed
   0.20.4 build, not the repo. It needs a rebuild and reinstall (or a release the auto-updater
   picks up). Same trap as the desktop UI: repo edits are invisible until the build is redone.
   Until then the verifier-side fix is carrying this on its own, which is exactly why it was
   written to work against today's agents.
2. **Then watch `verify_service.log`.** Expect `PLACEMENT MISMATCH` for the driver, not a pass:
   it really does hold 0-27 while assigned 0-9, and that is the next thing to fix (below). What
   must NOT appear is `wrong answer (max_err 28.6)`.
3. **Deploy the coordinator** — `audit_slots`, `/verifier/heartbeat`, `poc_excused`,
   `last_poc_at` are all schema/endpoint changes and `init_db()` migrates on start. Nothing pays
   an excused hour until the verifier is actually heartbeating, by construction
   (`audit_epoch()`), so the deploy order does not matter.
4. **Re-run the reconciliation** afterwards. It now understands `poc_excused`; without the
   coordinator deploy it reads an older DB exactly as before.
   ```
   ssh -i C:\Users\optin\.ssh\oracle_coordinator ubuntu@150.230.22.250 "python3 /tmp/reconcile_emission.py --db /home/ubuntu/neuron/coordinator/neuron.db --seed-from-backup --replay"
   ```

## Open, roughly in order

- **[P49]'s cause, not its symptom: the driver registers as 0-9 and serves 0-27, and the
  coordinator cannot see it.** `placement_drift` compares the assignment against
  `reported_layer_*` — what the node CLAIMED at registration — and both are 0-9, so the field
  reads false while the disagreement is real. A node's actual range is only ever visible in a
  challenge ack. [P37] deliberately did not feed an observed range back into placement ([P32]'s
  ownership inversion) and that is still right, but a coordinator that can never learn a node is
  serving a different range has no way to raise the alarm. Decide what it should do with `holds`
  besides refuse.
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
Continue NEURON. Read PROBLEMS.md [P49] [P47] [P48] [P39] and sessions.md,
plus NEXT_SESSION.md — that has the handoff.

STATE: Session 63 restarted the verifier and the driver FAILED, at max_err 28.6
— the figure [P47] called unexplained. [P49] explains it: the driver holds the
whole model (0-27) while assigned 0-9, so node_server's is_true_last sent a
stage-1 probe into the LAST-stage branch and it computed layers[10:] + norm.
Its ack said "holds": [0, 27] the whole time and challenge_middle_node ignored
it — [P37] verbatim, one function over. Fixed both sides. Serving was never
affected; only verification was, which is why it stayed invisible.

[P47] causes 2 and 3 are fixed: emission now pays for PROVEN work rather than
OBSERVED work, because verify_service.log shows the verifier was not running
for 53% of its own history and 57 of the Pavilion's 64 unpaid hours are hours
nobody could have been challenged in. STAGE1_FAILURES_ARE_SCORED stays False —
the evidence arrived and pointed the other way.

FIRST: restart verify_service.py (it runs from the repo, so it picks up the
holds check on its own — the node_server fix needs a REBUILD, since the live
agent is the packaged neuron-agent.exe), then deploy the coordinator —
audit_slots, /verifier/heartbeat, poc_excused and last_poc_at are schema and
endpoint changes, and init_db() migrates on start. Nothing pays an excused hour
until the verifier is heartbeating, by construction, so deploy order is free.
Then watch for PLACEMENT MISMATCH on the driver rather than a wrong answer.

THEN — [P49]'s cause rather than its symptom: the driver registers as 0-9 and
serves 0-27, and placement_drift cannot see it because it compares against what
the node CLAIMED at registration, not what it holds. Decide what the
coordinator should do with `holds` besides refuse — [P32]'s ownership inversion
says not "believe it", so the question is what it may legitimately raise.

ALSO OPEN: the /next -> / swap is still a SHIPPING blocker ([P48] item 3),
untouched. 0.20.4's wallet UI lives only on /next while / serves the old
chat.html, so the feature the release exists for reaches no default user.
Blocked on 59 source-text assertions in ui/test_chat_ui.py.

Nothing is pushed — nine commits on main-full. Pushing is also what publishes
the corrected download links to the GitHub Pages site.

Environment gotchas and the rest of the open list are in NEXT_SESSION.md.
```
