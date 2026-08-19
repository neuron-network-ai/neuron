# Handoff — start of Session 66

Paste the block at the bottom into a new window.

---

## What Session 65 did

**The claim was the session.** The plan was install 0.20.5 then [P30] phases 3-4. Phase 3 got
measured, and then the first genuine claim on the live network **failed on the founder's own
machine** — and the reason turned out to be structural, so the rest of the session went there.

**[P53], and it is the reason zero nodes have ever been claimed.** `agent/payout_key.py` mints
a payout key and binds it on an early start. So by the time anybody signs in, an address is
ALWAYS on file — which makes the claim a *rebind*, which `require_rebind_authority` correctly
refuses without `old_signature`. Every self-hosted node reaches that state unprompted. And the
error told the operator their key might be lost while it sat in `payout_key.json` on the same
disk, for exactly the address named in the message.

Under that was the founder's real objection: **claiming demanded a browser wallet at all.**
Somebody who signs in with Google and owns no crypto wallet could not claim anything.

`POST /node/claim` re-binds the address already bound, signed locally by the key this machine
holds, carrying `owner_wallet_id` from the session. Same-address binds are exempt, so no
`old_signature` and no register secret. The address never moves; only ownership is recorded.
**Not relaxed:** a bound address this machine has no key for is refused with a 409 — that is a
real address change and must keep needing the incumbent key.

The fix nearly missed the only page that matters: it went into the React app first, but `/`
serves `ui/static/chat.html`. Both its claim buttons share `runNodeClaim()`, so one change
covered them. Verified live: a clean 0.20.6 install serves **"Claim with my account"**.

**The uninstaller stopped thanking people for money it had just destroyed.** Mid-session the
founder uninstalled and reinstalled; the machine came back as `7fc2ff` and `6ff49d` was left
holding **33.49 NRN**. `new_node_id()` mints a fresh suffix whenever config has no node_id, and
the uninstaller deletes the config — then printed *"Thank you for contributing 33.49 NRN"*. It
now reads owner and balance BEFORE the DELETE kills the token, says plainly when claimed
earnings are safe, and when they are not it names the amount, the node id, that reinstalling
will not recover it, and writes `unclaimed-earnings.json` with the id a sweep needs.

A later reinstall came back as `6ff49d`, so **the 33.49 NRN is on the live node and one click
settles it.**

**[P30] phase 3, measured on both machines** — `ggml-rpc-server` on the Pavilion (authorised),
loopback-bound, through an authenticated tunnel:

| configuration | rate |
|---|---|
| this PC alone, llama.cpp q4_k_m | **32.11 tok/s** |
| split across this PC + the Pavilion | **3.97 tok/s** |

Distribution costs **8x**. The projection was ~16.7. Splitting a model that already fits is
pure loss — decode is sequential, so the second machine adds a hop without removing work. Still
~2.8x the live chain's 1.44, so the engine is worth wiring; it is just not the win the
arithmetic promised.

**ROADMAP vs PROBLEMS, reconciled with that number.** ROADMAP governs the pitch (never sell on
latency — distribution is 8x slower than not distributing); [P1] governs a floor (1.44 tok/s is
not "slower", it is unusable). The floor is TOKENOMICS §11.6's "answers under 30 s". Optimise to
it, then stop. In the PROBLEMS decisions log; ROADMAP untouched per build rule 2 — it needs one
line of founder sign-off, since its bullet is right and now has a measurement behind it.

**Also settled there:** the driver holds the whole model on DISK. Disk is cheap and RAM is the
binding constraint, so the capacity claim survives — but Route 1-prime therefore cannot serve a
model no single machine can hold on disk. Only Route 2 ever will. Two products, not one.

## Read this before planning

**Phase 2 is NOT wired.** `agent/rpc_engine.py` is imported by nothing except its own tests.
`node_server` still computes with PyTorch. So **phase 4 (packaging the binary) is premature** —
it would ship 17 MB that no code calls. The real next step is the wiring, and it is a session's
work, because in ggml's design the *driver* holds the model and drives remote servers: it means
`neuron_driver` opening a [P52] channel per chain member, bridging each to a local port, and
running llama.cpp with `--rpc` plus the coordinator's `--tensor-split`.

**Two small traps found the hard way.** `llama-bench`'s `-ts` wants `1/7`, not `1,7` — the comma
form silently runs two single-device configs. And `rpc_engine.find_binary()` looked for
`rpc-server` while the Ubuntu archive ships **`ggml-rpc-server`**; a Linux node would have found
nothing and stayed on PyTorch forever, silently, because a missing engine is deliberately not an
error. Both fixed.

## Do these first

1. **DONE — the claim executed on the live network at 09:07:23 on 2026-08-19**, the first in
   this project's history. And 62.21 NRN was swept to the wallet: 35.82 from `6ff49d` plus
   **26.39 from `7fc2ff`, the orphan the reinstall created**, which nobody could ever have
   signed in as. Wallet 391.39 → 453.60, supply invariant intact.
   **The check still worth doing next session:** a claimed node is supposed to credit its
   owner's wallet directly from now on ([P39]), making the sweep a one-time repair. Confirm it
   by watching the node's own balance stay at 0.0 while the wallet rises. If the node balance
   climbs again, [P39]'s crediting path is not actually wired and that is a new problem.
2. **Load the VM deploy key** if anything needs the coordinator:
   `SSH_AUTH_SOCK=/c/Users/optin/.ssh/neuron-agent.sock ssh-add /c/Users/optin/.ssh/oracle_coordinator`
   It carries a passphrase, so only a human can do it, and it drops when the PC sleeps.
3. **Push, or decide not to.** 55 commits, nothing pushed. Pushing also publishes the corrected
   download links and PRIVACY.md to the Pages site. Do NOT point downloads at 0.20.6 until a
   release exists — `test_download_links.py` enforces it.

## Open, roughly in order

- **[P43] the 4B has still never run a forward pass.** Hard-blocked on step 1 of
  `CAPACITY_CASE.md`: `bash coordinator/deploy.sh`, because the tier, the per-model stage-1
  width and the `weight_dtype` sizing are not on the VM yet. Steps 2-6 are cheap once that is
  done. Note the runbook assumes a third node; with two machines the fp16 caps still cover 36
  layers, and the OptiPlex stays out.
- **[P30] wiring, then phase 4.** In that order, for the reason above.
- **The first stranger.** ROADMAP's One Rule, unmet after 65 sessions. Outreach is a human act.
- **[P52] residual** — the channel authenticates the CALLER, not the PEER.
- **`NEURON_REQUIRE_SECURE=1`** once the fleet has moved.
- **[P37]'s open item** — auto-repair moves a node's range and never tells it.

## Environment gotchas

- **The Pavilion is `raman@100.79.125.112`**, not `optin@`. Five sessions were blocked on this.
  `~/neuron-engine/llama-b10485/` now holds `ggml-rpc-server` there.
- **Bash from cmd.exe is WSL, not Git Bash.** Use `"C:\Program Files\Git\bin\bash.exe"` for
  anything touching the SSH keys.
- **Python is `.venv\Scripts\python.exe`.** There is no pytest in it — the suites are
  `python <file>` or `python -m <pkg.module>`; `ui/test_node_owner_ui.py` needs `-m`.
- **The desktop app serves its own packaged copy**, and `/` is `ui/static/chat.html`, NOT the
  React app at `/next`. A UI fix that only lands in `ui/web` is invisible to real users.
- **Installing by hand over a live agent ABORTS** with `/VERYSILENT /SUPPRESSMSGBOXES` —
  RestartManager cannot close a tray app, the suppressed box defaults to Abort, Inno rolls back.
  Stop the agent first. The auto-updater is unaffected (it exits itself one second after
  launching the installer).
- **Deleting the GGUF cache costs a 1.1 GB re-download** before the chat UI comes up at all.
- **`good-state-0.20.5`** is a tag; `git reset --hard good-state-0.20.5` returns to it.

## Prompt for the next window

```
Continue NEURON. Read PROBLEMS.md [P53] [P30] [P43] [P52], ROADMAP.md, and
NEXT_SESSION.md — that has the handoff. Read ROADMAP.md properly: build rule 1
says every session.

HARD BOUNDARY. Work with EXACTLY TWO MACHINES: this Windows PC
(agent-optinovate-6ff49d) and node-c-pavilion (raman@100.79.125.112). They are a
TESTBED, not the network. Do NOT touch optiplex-server / nuc / 192.168.1.10.
The coordinator VM (150.230.22.250) is a THIRD machine — ask before deploying to
it, and its deploy key needs a passphrase only the founder can enter.

STATE: [P53] is fixed and installed — claiming a node now uses your Google/GitHub
account and the key the machine already holds, no browser wallet, no
old_signature. Verified live on 0.20.6. The uninstaller no longer destroys
unclaimed NRN silently. [P30] phase 3 is MEASURED: 32.11 tok/s on one machine
against 3.97 split across two — distribution costs 8x and never bought speed.

FIRST: confirm that a CLAIMED node now credits the wallet directly. The claim
ran successfully on 2026-08-19 and 62.21 NRN was swept to the wallet, but the
sweep is a repair, not the mechanism. Watch the node's own balance: it should
stay at 0.0 while the wallet rises. If it climbs, [P39]'s crediting path is not
wired and every operator would need a manual sweep forever.

THEN the engine, IN THIS ORDER, and the order is the point: agent/rpc_engine.py
is imported by NOTHING but its own tests, so phase 4 (packaging the binary) would
ship 17 MB that no code calls. Wire it first. In ggml's design the DRIVER holds
the model and drives remote servers, so this means neuron_driver opening a [P52]
channel per chain member, bridging each to a local port, and running llama.cpp
with --rpc plus the coordinator's --tensor-split. The whole-model-on-disk
question is already SETTLED in the PROBLEMS decisions log — do not re-litigate
it.

ALSO NEVER DEMONSTRATED: [P43], no forward pass has ever run on the 4B. It is
hard-blocked on CAPACITY_CASE.md step 1 (coordinator/deploy.sh to the VM), which
needs the founder. Ask early so it is not discovered at the end.

WATCH OUT: `/` serves ui/static/chat.html, NOT the React app at /next — a UI fix
that only lands in ui/web is invisible to real users. Stop the agent before
running an installer by hand or it silently rolls back.

Nothing is pushed. 55 commits.
```
