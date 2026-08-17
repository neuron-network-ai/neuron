# Handoff — start of Session 63

Paste the block at the bottom into a new window.

---

## What happened in Session 62

**[P40] closed.** `coordinator/reconcile_emission.py` — read-only, exits non-zero on a
discrepancy. Run on the live ledger: 333.409646 NRN recorded against a 333.4096458 NRN drop in
the pool, all 318 settled rows re-pricing correctly from their own frozen inputs, supply
invariant intact. First independent confirmation the distributed NRN is right.

**[P47] filed and cause 1 fixed.** The reconciliation showed 199 of 318 settled node-hours paid
nothing. Dating them showed the misses running to the current slot, concentrated on the driver.
`verify_service` skipped every `layer_start == 0` node believing the middle probe could not
validate one — false; `node_server`'s probe role never embeds either. Measured end to end:
max_err **0**. Since emission pays only on a passing challenge, the driver had been unable to
earn an availability hour for its entire existence (~243 NRN).

**Two latent bugs fixed on the way:** `settle_attendance(..., 0.0)` could never fire (its
`WHERE paid_at IS NULL` was already falsified by the claim four lines earlier), so a payment the
pool could not make was recorded as though made — `models.void_settlement` replaces it. And
`genesis.py` now records `emission_pool_seed`, which was previously unrecoverable after seeding.

**[P48] filed.** Three things found by looking at the live product, not the code. Fixed: a node
ahead of the network was told to downgrade (`av == latest` read as "current or behind"), and the
download links on `docs/index.html` and `README.md` sat two releases back at v0.20.2. Not fixed
and the interesting one: 0.20.4's wallet UI only exists on `/next` — see the open list.

Commits: `a8cb3ba`, `386b9db`, `7c5aa4b`, `ef9c9ae` on `main-full`. **Not pushed** — pushing is
also what publishes the corrected download links to the GitHub Pages site.

## Do these first

1. **Restart the verifier** — the stage-1 change does nothing until then. It runs as
   `verify_service.py`; find it with `neuron_doctor` if you have forgotten where.
2. **Watch the driver's next slot.** After a full slot with the new verifier, re-run the
   reconciliation. `agent-optinovate-6ff49d` should stop accruing `no-proof-of-compute` hours.
   If it does, flip `STAGE1_FAILURES_ARE_SCORED = True` in `verify_service.py` — that is the
   evidence the flag is waiting for.
3. **The reconciliation command** (script is at `/tmp/reconcile_emission.py` on the VM; re-scp
   after any edit):
   ```
   ssh -i C:\Users\optin\.ssh\oracle_coordinator ubuntu@150.230.22.250 "python3 /tmp/reconcile_emission.py --db /home/ubuntu/neuron/coordinator/neuron.db --seed-from-backup --replay"
   ```

## Open, roughly in order

- **[P47] causes 2 and 3.** Every "the node is fine, nothing recorded" branch
  (`RangeMismatch`, `ChallengeRefused`, pre-strike unreachable) leaves `poc_ok` at 0, so
  emission cannot tell "not checked" from "failed" — that is likely the Pavilion's 64 lost
  hours. And `/node/{id}/peer-attest` never calls `mark_slot_poc`, so peer verification firing
  would still not pay anyone. Cause 2 is a decision about what emission pays FOR, not just a fix.
- **[P48] item 3 — the `/next` → `/` swap is a SHIPPING blocker, not a tidy-up.** 0.20.4's
  headline wallet UI (`≈ N network answers left`, `ui/web/src/components/Sidebar.tsx`) is served
  at `/next`, while `/` still serves `ui/static/chat.html`. Every default user sees the old page,
  so the feature the release exists for reaches nobody. Blocked on 59 source-text assertions in
  `ui/test_chat_ui.py` that assert on chat.html's HTML strings and therefore cannot follow the
  behaviour across implementations — port them as behaviour, the way Session 61 ported the claim
  panel. Items 1 and 2 of [P48] (version display, download links) are already fixed in `ef9c9ae`.
- **[P40] item 2** — run the reconciliation as a standing assertion, not by hand.
- **[P39] item 5** — `delete_node` leaves a funded ledger row behind. The 8.69 NRN on
  `agent-bhpc012104-82cbee` was recovered only because that node was yours; on a volunteer's
  machine the same sequence strands real NRN. Cheapest fix: refuse to delete while the balance
  is non-zero.
- **The capacity case has still never run a forward pass.** `NEURON_WEIGHT_DTYPE=fp16` on both
  nodes, `donate_ram_gb: 8` on the Windows PC, pin `Qwen/Qwen3-4B-Instruct-2507`.
  `CAPACITY_CASE.md` is the runbook. Needs you at both machines.
- **[P39] phase 4** — sweep the Pavilion's ~213 NRN, but only after an owner is recorded through
  the Chat UI. The panel is at `/next`, not `/`.
- **Peer verification has never fired.** Watch `peer_passes` when a stranger arrives.
- **[P44]** — driver shard goes stale on a width change; needs a driver-side reload.
- `coordinator_version` still reports 0.1.0, unwired since Session 58.
- `/next` -> `/` swap: blocked on porting 59 source-text assertions in `ui/test_chat_ui.py`.
- Router: the static DHCP row still maps the Pavilion's MAC to 192.168.1.10, which the OptiPlex
  holds statically. Yours to do.

## Environment (all hit for real)

- `bash` from cmd.exe is **WSL**, not Git Bash. Use `"C:\Program Files\Git\bin\bash.exe"` for
  anything using the SSH keys.
- cmd.exe has no inline `VAR=value cmd`.
- Absolute paths in anything handed over — you are usually in `C:\Users\optin`.
- Python is `.venv\Scripts\python.exe`; PATH python is 3.14 with no torch.
- Tests run as `python -m coordinator.test_<name>`, not pytest.
- `gh` is installed and authed but must run from inside the repo.
- You cannot SSH to the coordinator VM from the agent side. Deploys are yours.
- The VM's systemd drop-in dir has `zz-agent-release.conf`, which wins on lexical order, and
  `google.conf` is root-only so `systemctl cat` cannot read it. Use
  `systemctl show <unit> -p Environment | tr ' ' '\n' | grep -i <key>`.
- The live DB is at `/home/ubuntu/neuron/coordinator/neuron.db`, no `NEURON_DB` override.

---

## Prompt for the next window

```
Continue NEURON. Read PROBLEMS.md [P47] [P40] [P39] and sessions.md Session 62,
plus NEXT_SESSION.md — that has the handoff.

STATE: [P40] is closed — emission reconciles on the live ledger (333.41 NRN,
three legs agreeing, all 318 settled rows re-pricing correctly). [P47] cause 1
is fixed but NOT YET DEPLOYED: the verifier now challenges stage-1 nodes, which
is what lets the driver earn availability emission at all, and it does nothing
until verify_service is restarted. Two commits on main-full, unpushed:
a8cb3ba, 386b9db.

FIRST: restart the verifier, let one slot close, then re-run the reconciliation
and check whether agent-optinovate-6ff49d has stopped accruing
no-proof-of-compute hours. If it has, flip STAGE1_FAILURES_ARE_SCORED to True —
that is the evidence it is waiting on.

THEN — [P47] cause 2. Emission cannot tell "we did not check" from "it failed":
RangeMismatch, ChallengeRefused and pre-strike unreachability all leave poc_ok
at 0, and each of those is the verifier explicitly concluding the node is FINE.
That is likely the Pavilion's 64 unpaid hours. Decide what emission should pay
for, then build it — this is a tokenomics decision, not just a bug fix. Cause 3
is smaller: /node/{id}/peer-attest never calls mark_slot_poc, so peer
verification firing for the first time still would not pay anyone.

ALSO OPEN, and reclassified: the /next -> / swap is a SHIPPING blocker, not a
tidy-up ([P48] item 3). 0.20.4's headline wallet UI lives only on /next while /
still serves the old chat.html, so the feature the release exists for reaches no
default user. Blocked on 59 source-text assertions in ui/test_chat_ui.py.

Nothing is pushed. Pushing is also what publishes the corrected download links
to the GitHub Pages site.

Environment gotchas and the rest of the open list are in NEXT_SESSION.md.
Budget note: weekly usage was at 97% on 2026-08-17, resetting Friday 4pm.
```
