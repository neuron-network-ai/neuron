# NEURON — Problems, Risks & Decisions

Living log of open problems, risks, and the decisions taken about them. Complements
`ROADMAP.md` (the plan) and `sessions.md` (what was built). Update as things change.

Status keys: 🔴 open/unaddressed · 🟡 mitigation known, not done · 🟢 resolved · 🔵 design note

---

## Decisions log

- **2026-07-24 — Speed spike before scaling the roadmap.** The founder is counting on
  per-user speed for the Green AI thesis. Rather than complete Sessions 12–20 on an
  unverified speed assumption, OR prematurely productionise quantization before there
  are real users, run a **cheap measurement spike** (int8 vs fp32 on real hardware) to
  find the true speed ceiling with data. Measurement only — not shipped. Then resume
  Session 12 (first stranger node). See [P1], [P2]. Result recorded under [P2].
- **2026-07-24 — Spike outcome & go-forward.** int8 = **3.46× faster** (speed ceiling is
  fine) but naive int8 **broke quality** [P9]. Decision: **thesis de-risked → resume the
  roadmap at Session 12** as planned; do NOT integrate quantization now. Log a dedicated
  *quality-preserving quantization* session (GPTQ/AWQ or llama.cpp GGUF+RPC) as prioritized
  future work — evaluate the llama.cpp engine pivot deliberately, with real users in view.
- **2026-08-08 — Rent a cloud GPU rather than keep deferring the GPU path.** Every remaining
  GPU item is blocked on the same thing: **no machine in this project has an NVIDIA card**, so
  not one line of the CUDA path can be executed, let alone measured. That is exactly how
  [P31] happened — a capability built, never run, then documented as though it had been — and
  it is not a gap that more code review closes. Decision: **hire GPU time online** and do the
  work on real hardware.
  What that session must cover, in order, because the order is what keeps it safe:
  1. Verify **Phase 4a** on CUDA. It is written and bit-identical on CPU (`test_batching`'s
     figure is unchanged to the digit), but "passes on CPU" is not "works on a GPU".
  2. Then **Phase 4b** — the one-line device move at `slice_downloader.py:299`.
     `test_device_path.py` fails until it is done deliberately, and says what to verify first.
  3. Reconcile **`selftest_shard.py`**, which requires `diff == 0` between a CPU `load_model()`
     and a CUDA `load_model_shard()` — as written, **build rule 6 is unsatisfiable on a GPU
     box**, so it fails on the first machine that could actually prove the feature works.
  4. Only then **`balancer.GPU_EXECUTION`** back on, with the VRAM reserve and bound that are
     already in place and dormant.
  5. **CUDA packaging last**, and it is its own problem: ~2.4 GB of torch against today's
     207 MB installer, `updater.DOWNLOAD_TIMEOUT = 600` (needing a sustained ~32 Mbit/s), no
     resume, no disk-space check, no rollback (`updater.py:173` is `os._exit(0)`), a
     `/agent/version` that cannot express a per-capability build, and NVIDIA redistributables
     that `tools/gen_notices.py` does not cover. **With one installer, every volunteer pays
     that download, including the ones with no GPU.** Fix the updater before shipping it.
  Rented time is cheap next to the lever: decode is bandwidth-bound at ~30 GB/s on DDR against
  360–1000 GB/s on a consumer card, and 200B needs ~30–80 CPU machines or ~10 GPU ones.
- **2026-08-11 — An agent can be rolled BACKWARDS, and only when asked explicitly (0.20.1).**
  The founder's condition for shipping: *"if it goes wrong we have to make setup to fix, we can
  not leave users in loss."* There was no such setup. An agent could only ever move forward —
  `is_newer` refuses anything not newer — so a bad release had no remote remedy at all:
  `apply_update` ends in `os._exit(0)`, the machine is behind a NAT in somebody's house, and
  publishing the previous installer did nothing because the agent correctly declined it. **The
  safety property and the trap were the same line.**
  So a downgrade is possible, but never inferred from the version alone: `NEURON_AGENT_ROLLBACK=1`
  on the coordinator, surfaced as `rollback` in `/agent/version`, is a separate act by an operator
  who knows what they are doing. A stale config, or anyone able to influence the version field,
  must not be able to walk a fleet backwards silently. Every other guarantee is unchanged — the
  SHA-256 must match, a source checkout is untouched, a serving node is never interrupted, and an
  empty hash still means nobody installs anything.
  **Shipped as 0.20.1 rather than added later, deliberately.** 0.20.0 was already published and
  advertised when this gap was named, and a node that lands on 0.20.0 could not be rescued from a
  future bad release. Nodes must arrive on a version that can be pulled back, so 0.20.0 was
  superseded before any node had taken it. Agents older than 0.20.1 ignore the field entirely,
  which is right for them: they stay where they are rather than acting on something they cannot
  verify. Procedures in `RECOVERY.md`; 6 cases in `agent/test_updater.py`, including that the flag
  never bypasses the hash check and that rolling back to the running version is a no-op rather
  than a reinstall loop.
- **2026-08-11 — A node's standing and proof-of-compute record leave the public dashboard.**
  The founder's objection: a row reading `flagged · 4%` beside a named node *"will make users
  worried and they may leave the network"*. Correct, and the deeper problem is that the number
  is the fact most likely to be **wrong**: [P37] flagged three honest machines for the
  coordinator's own bookkeeping, and `18f1da` reached `1/23` today entirely on a range the
  coordinator had moved under it. Publishing a verdict the network cannot always justify, about
  an identifiable volunteer's computer, readable by every other volunteer — including the owner,
  who sees it on the same page as everyone else — is a pillory, not transparency.
  The page already established the principle for balances: *"per-node balances are NOT shown
  here. The public dashboard is network health only."* Standing is the same class of fact and a
  harsher one. So per-node `standing` and `proof-of-compute` are gone from `/dashboard`, and the
  exclusion is stated **in aggregate** instead — *"N node(s) excluded from routing — the network
  is serving around them"*, which answers the only question a visitor actually has.
  **Nothing a visitor needs is hidden:** the capacity line still reads "across N eligible
  node(s)", the coverage strip still shows any unbacked layer, and `network_healthy` still means
  a request can complete. **Nothing the operator needs is hidden either** — the full standing,
  tally and the *reason* stay on their own token-gated page, which is the only place they are
  actionable. Both halves are pinned in `coordinator/test_dashboard.py`, including that the
  operator's own page must keep what the public one drops: a fix that also blinded the operator
  would be worse than the problem, since they are the only person who can repair the machine.
- **2026-07-28 — Pre-launch audit before the first real external stranger.** No real
  stranger has ever run a node — everything "live" so far is the founder's own machines.
  Before handing the installer to an actual friend, ran three parallel audits (security,
  economics, reliability) across the current coordinator/agent/relay code, independently
  verified every finding against the real code (not just the audit's claims), and fixed
  everything ranked "must-fix before a real stranger joins." See [P14]. Also added an
  honest first-run disclosure (earnings have no cash value; why Windows may flag the
  unsigned installer) to `INSTALL.md` and to the installer itself (`packaging/DISCLOSURE.txt`,
  shown as an Inno Setup `InfoBeforeFile` page) — a friend who only runs the exe and never
  reads `INSTALL.md` still sees it.

---

## Problems & risks

### [P37] 🟡 Proof-of-compute flagged three honest nodes for the coordinator's own bookkeeping, and the flag had no exit — mostly fixed (2026-08-10)

**The network was down to one point of failure and every signal said the nodes were cheating.**
`DEGRADED — every layer is covered but the chain walks to 1 stage(s)`: 28/28 covered, `routable:
false`, one eligible node holding all 28 layers, and three machines sitting at `33% (1/3)`
proof-of-compute with no route back.

**None of the three was faulty.** All three were on ranges the coordinator had moved under them:

| node | coordinator says | last registration claimed | actually held |
|---|---|---|---|
| `node-c-pavilion` | 10-27 | 0-27 | **10-18** |
| `agent-bhpc012104-82cbee` | 10-27 | 14-27 | — |
| `agent-bhpc012101-18f1da` | 19-27 | 10-13 | — |

Note the middle column is `reported_layer_*`, a record of the LAST registration, and pavilion's
own `config.json` read `10-18` when it was finally inspected — matching its disk, not the 0-27
it had once claimed. So the node and its bytes agreed all along; the coordinator and the
registration history were the two that disagreed with them. A stale diagnostic field read as a
live one is worth knowing before it is used as evidence again.

`placement_drift` was **true on exactly those three and false on the two unflagged nodes** — a
perfect predictor, sitting in `/node/list` the whole time and read by nothing.

Pavilion's real range was confirmed live by speaking the protocol to it directly. Challenged as a
last-stage node for 10-27, it answered from the **probe** branch — `{"ok": true, "s1": 10,
"s2": 19}` — which is a node saying, in the ack, *I hold 10-18*. It then computed layers 10-18
with no final norm, and the verifier compared that against layers 10-27 **+ norm**. That is the
`max_err 33.79`: deterministic across attempts and across verifier restarts, because it is not
noise — it is the right answer to a different question. **The ack that explained it was on the
wire from the beginning and `challenge_node` threw it away.** `challenge_middle_node` has
always checked exactly this, four lines further down the same file.

The other two failed the other way, as `socket closed mid-message`. The last-stage role took
`s2` **from the caller** (`agent/node_server.py`), so layers below the node's own start are
uninitialized meta tensors; the forward pass raised something that is not a `ConnectionError`,
`_handle` caught only `(ConnectionError, TimeoutError, EOFError)`, the thread died, and
`finally: conn.close()` slammed the socket. So a node asked for layers it never downloaded was
indistinguishable from one that hung up — and [P35], which had just started counting persistent
unreachability as a failure, dutifully counted it.

**Then it was permanent, and that is the part that cost the network.** `flagged` is DERIVED
every sweep from cumulative counters (`models.py`: `total >= REPUTATION_MIN_SAMPLES and
passed/total < REPUTATION_THRESHOLD`), so it is arithmetically clearable — 1/3 needs only two
passes to reach 3/5 = 0.6. It was never clearable in practice, because `verify_service` excluded
flagged nodes from **both** the probationary list and the re-check rotation. A flagged node was
never challenged again, so its counters could never move, so the flag could never lift. No
endpoint reset them either: the only exit was `DELETE /node/{id}`, which destroys the node's
identity and token and makes a volunteer reinstall to escape a verdict.

The log shows the mechanism working perfectly with nothing left to work on — after the last
flag landed at 13:21, every sweep for the rest of the day reads
`5 node(s), 0 awaiting verification`. That is [P24]'s failure class exactly, and [P35]'s
sentence one layer up: *a node is only as trustworthy as its last check*, unless the last check
is the one that made it unaskable.

**A second asymmetry, found on the way and just as slow-acting:** re-verification recorded
failures and **not passes** — the pass path `continue`d before attesting. So a node's ratio
could only ever move down. One bad afternoon was permanent no matter how well it behaved
afterwards, on a long enough timeline.

**Fixed, five parts.**

1. **The node reports what it actually holds** (`holds: [lo, hi]`) on the last-stage ack as
   well as the probe one. The last-stage ack echoed the caller's own `s2` back at it, which can
   never disagree with itself and so could never reveal anything.
2. **`challenge_node` checks it**, and a disagreement raises `RangeMismatch` — its own type, so
   it cannot be swallowed by the verifier's blanket `except Exception` back into the strike path
   it just escaped. Works against **today's** agents with no release: the probe ack has always
   carried the node's real range.
3. **The last-stage role refuses an `s2` its slice does not cover**, as a typed
   `{"ok": false, "error": "range_mismatch"}` — the same mechanism as `paused` and `reloading`,
   and the same reasoning as `reload()`'s `_layers_in_slice` guard one level up: refuse a range
   rather than run garbage for it. `_handle` now also catches everything else and logs the
   traceback locally, so no node-side fault can ever again reach a peer as a bare closed socket
   ([P28]'s complaint, self-inflicted).
4. **A range mismatch is attested NEITHER way and takes no strike.** It is a fact about
   placement, not about the machine; recording it against the node files our own bookkeeping
   error in its permanent record. It logs at ERROR with the remedy, and stamps `last_checked`
   so the rotation still advances.
5. **A flagged node is back on the re-check rotation**, and **every** pass is now recorded,
   re-checks included. Those two together are the whole exit: a node that is genuinely fine
   earns its way back with no operator involved, one challenge a minute, and the flag lifts
   itself. `POST /node/{id}/reputation-reset` (operator) exists for the case that cannot fix —
   evidence the network *manufactured*, where no number of future passes makes the old entries
   true. Deliberately not self-serve, and it does not touch peer attestations: those are other
   machines' testimony, not ours to erase.

`test_flag_recovery.py` **35/35** (new), including the live pavilion ack byte-for-byte, and a
loopback socket so the check is pinned at its **call site** — testing `_check_range` alone would
pass just as happily with the call deleted, and a missing call site is the entire bug.
**Tripwires verified to fire:** restoring the `not flagged` filter fails `a FLAGGED node is
challenged again`, `its pass is RECORDED` and `three cycles reach all three`; then restored.
Regression: `test_proof_of_compute` 14/14, `test_open_join` 26/26, `test_reload_lifetime` 14/14,
`test_pause_admission`, `test_routability` 16/16, `test_placement_ownership` 9/9.

**Still open, and it is the cause rather than the symptom: auto-repair moves a node's range and
never tells the node.** `main.py`'s repair path calls `models.update_layers` directly — a pure
DB write. A node reconciles its slice only in `setup()`, i.e. at startup. So a live re-placement
leaves the node serving its old slice for as long as it stays up, which is precisely how
pavilion came to hold 10-18 while assigned 10-27. The released agent's `ensure_slice` already
discards a slice that does not cover its range ([P36]) — it simply never runs again. The
migration path has a remote reload handshake (prepare → download → ready → cut over) that
auto-repair does not use. **Until that is wired up, the coordinator can hand a node a range it
is physically unable to serve, and now at least it finds out.**

Deliberately NOT built: having the verifier report a node's observed range back into placement.
That is [P32]'s ownership inversion re-entering through the diagnostics, and a node that can
edit its own placement by answering a challenge is worse than the bug being fixed.

**Recovered live, 2026-08-10 22:45-22:57, and the sequence is the point.** Un-flagging pavilion
first would have moved it from *excluded* to *in the chain and failing every request*, because
a flag is not what makes a node wrong — the slice is. So the slice went first:

1. **Restart pavilion's agent.** No `git pull` and no `rm -rf`: `~/neuron` there is not a git
   checkout at all (files were scp'd), its `slice_dir` is `model_slice_10_18/` rather than the
   assumed `model_slice/`, and `ensure_slice` deletes a non-covering slice itself. The remedy
   that *looked* right would have pulled nothing, deleted nothing, and been reported as done.
   Looking at the machine first is what turned a three-part command into a one-line restart:
   `cached slice holds layers 10-18 but this node serves 10-27 — discarding it and downloading
   the right one`, 1.69 GB at 3.9 MB/s, then `measured 11.032 ms/layer over layers 10-27`.
2. **Verify before recording anything.** A direct challenge returned
   `passed: True, max_err 0.000217, 358 ms` — against 33.79 an hour earlier, on the same
   machine, with no change to the machine.
3. **The flag then lifted itself**, which is the whole fix demonstrating itself: 1/3 → 2/4 →
   3/5 = 0.6. Two challenges, ~2 minutes, no operator, and `reputation-reset` never used. The
   endpoint exists for evidence that cannot be outvoted; this was not that case, and it should
   not have been treated as one.

`/status` went `stages: 1, routable: false, eligible: 1, flagged: 3, network_healthy: false`
→ **`stages: 2, chain_ranges: [[0,9],[10,27]], routable: true, eligible: 2, flagged: 0,
network_healthy: true`**.

**The nine sweeps before the restart are the better evidence.** With the new code live but the
slice still wrong, the verifier logged `PLACEMENT MISMATCH, not a bad node — challenged on
layers 10-27 but the node holds 10-18` nine times and **recorded nothing**. Under the old code
those same nine sweeps were strikes and attested failures. The bug reproduced in full and cost
the node nothing.

**Cosmetic, and a reminder of what is deployed:** those lines read `standing '?'` because the
live coordinator does not yet return `standing` from `/attest`. The verifier half of this fix
ships by restarting a local process; the coordinator half (`standing` in the attest reply,
`POST /node/{id}/reputation-reset`) and the node half (`holds`, the typed `range_mismatch`
refusal) are still undeployed.

**Still flagging honest nodes, because the typed refusal is the path drift usually does NOT
take (2026-08-11).** The founder's objection was blunt and correct: *these flagged PCs are your
work, in the name of a fix.* The live table proves it — `agent-bhpc012101-18f1da` went
**1/18 → 1/22 in twenty minutes**, four more challenges and not one pass, on a range the
coordinator had moved under it. `agent-bhpc012104-82cbee` sat unchanged at 2/4 across the same
window, so the rotation was spending its budget punishing one drifted node and never re-checking
the other.

`RangeMismatch` only fires when the node gets to **say** it holds something else. [P37]'s own
analysis says why that is the minority case: asked for layers it never downloaded, the node
raises on uninitialized meta tensors, `_handle` catches only
`(ConnectionError, TimeoutError, EOFError)`, the thread dies, and `finally: conn.close()` slams
the socket. Drift therefore arrives at the verifier as **`socket closed mid-message`** — the
generic `except Exception` branch — and [P35]'s `UNREACHABLE_STRIKES` then attests it as a
genuine failure every 5 cycles. The two fixes composed into a machine that manufactures evidence
against honest volunteers. Pavilion's `max_err 33.79` is the same story on the other branch: a
correct answer to a question we should not have asked.

**Fixed by reading the field that predicted all of this.** `placement_drift` was *"a perfect
predictor, sitting in `/node/list` the whole time and read by nothing"* — it is now read by
`verify_service`, and while it is set **neither** failure path records anything: not the
hang-up, not the wrong answer. Both log at ERROR naming placement as the thing to fix.

**And the same table showed why the recovery half never ran.** `82cbee` sat unchanged at 2/4 —
**one pass from clearing its flag** — because the re-check rotation sorts `due` by `last_checked`
with a default of `0.0` and takes ONE node per cycle, while the unreachable path `continue`d
*without stamping it*. A node that never answers therefore stays permanently at the front and
starves every other node off the rotation. [P35] put flagged nodes back on this rotation
precisely so they could recover; an unstamped failure quietly monopolised it and denied exactly
that. The attempt is now stamped whatever the outcome — `unreachable_strikes` already counts the
consecutive failures, so nothing is lost by letting the rotation move on.

Deliberately **asymmetric**: drift suppresses failures, never passes. A pass under drift is real
evidence — it proves the node does hold the assigned range, so `reported_*` is the stale field —
and since the counters only ever grow, suppressing passes would leave a wrongly flagged node no
way back. The cost is that a node could dodge strikes by registering a range it was not given;
that is visible as `placement_drift` on the dashboard, and it is the right side to err on when
the alternative has already cost three honest machines their standing and the network its
routability. `test_drift_is_not_evidence.py` (6 tests) pins both directions, including that
`/node/list` must never add `placement_drift` to its `hidden` set — doing so would blind the
verifier again with every test still green.

**The hardening then stopped verifying the driver entirely, and looked fine doing it
(2026-08-11).** `_check_range` (the last-stage path) deliberately tolerates an agent too old to
report what it holds — *"pre-0.20 agent on the last-stage path"*, return early. `challenge_middle_node`
compared the WHOLE TUPLE for equality instead. The driver `agent-optinovate-6ff49d` — 26/26,
the healthiest machine in the chain — acks `s2` correctly and simply omits `s1`, so every sweep
read `(None, 10) != (0, 10)` and raised PLACEMENT MISMATCH. It lands in the *"nothing recorded"*
branch, so the log looked benign while proof-of-compute quietly stopped checking the most
important node on the network, and would have kept doing so until 0.20 shipped.

That is [P35]'s disease returning through its own cure: a verifier running perfectly and checking
nothing. **A check that cannot pass is not a strict check, it is a dead one.**

Fixed by comparing only the fields the node actually claims: an absent `s1` is silence, not
disagreement. Equality (not coverage) stays correct on this path — the probe makes the node run
its OWN `lo`/`hi` in isolation, so a node holding more really would compute different arithmetic
— but silence is not a claim. Seven cases added to `security/test_proof_of_compute.py`, whose
header had recorded this protocol as manually-verified-only; the ack comparison is pure logic and
needs no weights, which is exactly why it went unpinned for so long.

**Fixing that comparison then aimed a never-executed check at the driver, and it nearly flagged
it (2026-08-11).** With the tuple compare corrected, the stage-1 probe ran for the first time
ever and `agent-optinovate-6ff49d` came back `max_err 28.6, strike 1 of 3`. The verifier was
stopped at strike 1. Three would have attested a failure against the only machine holding stage 1
— and a flagged driver is not a degraded network, it is no network at all.

The cause is a genuine incompatibility, not a bad node: `make_middle_challenge` computes
`layers[s1:s2]` on a raw hidden state with **no embedding**, while a first-stage node embeds
token ids first. They are not the same function, so a healthy driver cannot pass. `28.6` has
[P37]'s signature — deterministic, and the right answer to a different question — but that is a
**hypothesis**, and a guess is not grounds for scoring somebody's machine.

So stage-1 nodes are **not challenged, and are said to be unchallenged**, once per process at
WARNING: *"Proof-of-compute does not currently cover the driver."* Deliberately the inverse of
[P31], where an unexecuted capability was documented as working. The gap is real either way; the
choice is only whether it is visible. A node that is stage 1 **and** last stage (a one-node
network on 0-27) still goes down the `make_challenge` path and is fully verified.

**Open:** build a stage-1 challenge that embeds token ids, so the driver is covered. Until then
the most important node in the chain is unverified, and now says so.

**This stops the bleeding; it does not heal it.** The counters already written stay written.
`18f1da` at 1/22 cannot climb to the 0.6 threshold on passes alone, so the order is: fix its
placement, confirm it passes, **then** `POST /node/{id}/reputation-reset`. Resetting first would
clear the counter on a node that is still mis-placed and it would simply re-earn the flag.

### [P35] 🟢 Proof-of-compute was a one-time gate, so a node that went bad was never caught — fixed (2026-08-10)

A node was challenged exactly once, while probationary, and never again. Verification proved a
node was honest **on the day it joined** and nothing after. A node that later went wrong —
reloaded onto a different model, a corrupted slice, failing memory — kept a perfect reputation
and kept receiving live traffic.

Live 2026-08-10, while chat ran at 0.25 tok/s: the verifier logged
`5 node(s), 0 awaiting verification` on **every sweep for hours**. It was running perfectly and
checking nothing, because every node had already passed once.

Two distinct failures surfaced within minutes of turning re-verification on, and neither could
have been found any other way:

- **`node-c-pavilion` — deterministic wrong answers**, `max_err 33.79`, identical across
  attempts. Not noise: wrong weights for the range it was assigned.
- **`agent-bhpc012101` — `socket closed mid-message` on every challenge**, the exact error the
  driver reports when a chat dies.

**Fixed, two parts.**

1. **Re-verification.** One already-verified node per cycle, oldest-checked first, so the cost is
   one challenge a minute however large the network grows. Passes log at DEBUG — a healthy
   network stays silent, or this becomes the heartbeat mistake again (Session 55 half seven). A
   wrong answer takes the normal `FAIL_STRIKES` path, so the existing reputation machinery does
   the excluding; this only supplies the evidence it was never given.
2. **Persistent unreachability now counts.** "Could not challenge" was inconclusive *forever* —
   right for one hiccup, wrong as a permanent amnesty, because a node running the wrong weights
   fails by **hanging up** rather than by answering wrong. It accumulated nothing and kept a
   perfect reputation while breaking every request it touched. `UNREACHABLE_STRIKES = 5`,
   deliberately higher than `FAIL_STRIKES` because an unfinished answer is far more often a
   restart or a cold shard than a bad node.

**Also fixed: the verifier could be killed by its own success message.** `UnicodeEncodeError` from
the `→` in the VERIFIED line, raised by the cp1252 console handler on Windows — it took the
service down and it restarted with a traceback in its own log. The stream handler now degrades to
`?` instead of raising. A monitor that dies on the shape of its own output is worse than none.

**What this replaces:** deleting bad nodes by hand. That does not scale past machines you can
name, and the network already had the right mechanism — it was simply never asked to run.

### [P33] 🟡 The coordinator forgot which model it was serving, and nothing could tell — partly fixed (2026-08-10)

`_serving` lived **only** in a module-level dict in `main.py`, initialised to the configured
floor. So any restart after a tier migration silently reverted the coordinator's *belief* about
which model the network runs, while every node carried on serving the new one.

**Invisible because every health signal stayed green.** Qwen2.5-1.5B and Qwen2.5-7B both have 28
layers, so every layer range still validated, coverage read 28/28, and the chain reported
`routable: true`. What the user got was `socket closed mid-message`, four retries, and a
`RuntimeError` — nodes feeding each other activations from *different models*.

Live 2026-08-10: the 7B migration cut over at 07:06 UTC; the coordinator was restarted minutes
later to disable auto-promotion; it came back believing 1.5B; `agent-bhpc012104` kept serving
`Qwen2.5-7B layers 24-27` and every distributed chat failed.

**Fixed:** a `settings` table, `set_serving_model` writes through to it, and `_load_serving()`
runs at startup *before* the health loop — the sweep assigns ranges against the serving model, so
a coordinator that has not yet remembered which model it serves would hand out ranges for the
wrong one. Startup now logs the model it restored. Auto-repair is also gated on
`migration.phase == "steady"`, so a repair cannot rewrite ranges mid-cutover and strand half the
network on each partition.

**And the repair had to be remote, because "restart the agent" is not an instruction this product
can give.** A volunteer's PC is 100 km away and behind a NAT. `POST /network/model` (operator)
now pins the model AND moves every node onto it using the migration handshake that already does
remote reloads — prepare, download, report ready, cut over together. It is how the network
reached 7B this morning; nothing could aim it deliberately until now. `model_id: null` clears the
pin and restores capacity-driven tiering. Used live on 2026-08-10 to walk the network back from
7B to 1.5B with no physical access to any machine.

Note the ordering that makes it work: the coordinator must first be told the **truth** about what
it is serving. Pinning 1.5B while it wrongly believed it already served 1.5B was a no-op — the
migration only plans when target != serving.

**Nodes now report the model they serve (2026-08-10).** `nodes.reported_model_id`, sent at
registration, compared against the serving model — `/status` carries `model_mismatch` and
`network_healthy` is false while any node is on the wrong weights. Never used to place or route:
placement stays the coordinator's ([P32]). NULL means an agent too old to report, which must not
read as a mismatch. **Takes effect for a node once it runs 0.20**; the coordinator side is live.

**Memory-aware assignment (2026-08-10).** `canonical_assignment` now caps each stage at
`balancer.max_layers_for` and spills the remainder onto the next stage. It previously split
evenly, which is harmless on the 1.5B floor and on the 7B would have handed an 8 GB office PC
9 layers (~8.4 GB at fp32) — [P26]'s hazard, reproduced in new code.

### [P34] 🟢 One replica 188× slower than its peers dragged every request it won — fixed (2026-08-10)

`agent-bhpc012101` reported **4150 ms/layer** against 8–22 ms for the other three, a measurement
taken while it thrashed during the 7B migration. `fastest_pick` weights replicas by throughput,
so it still won ~1 request in 189 — and a chain runs at the speed of its slowest stage, so those
requests crawled. Observed: **0.24 tok/s**.

Weighted-random is right for "somewhat slower": a node at half speed should carry half the
traffic, which is what makes an added machine throughput rather than a deeper pipeline ([P16]).
It is wrong for orders of magnitude, where the right share is zero. `REPLICA_SLOWDOWN_LIMIT`
(8×) drops outliers before weighting, and keeps them if *every* replica of a segment is an
outlier — an empty segment breaks the chain, which is worse than serving it slowly.

Best-path estimate after the fix: **384 ms/token ≈ 2.6 tok/s** compute-only, from 0.24.

The stale figure itself is a separate gap: `ms_per_layer` is measured once and never re-measured
unless the agent restarts, so a bad reading taken under load is believed indefinitely.

### [P38] 🟢 The fix for [P34] cost an order of magnitude in production — fixed same day (2026-08-11)

**The TTL is a 0.20 feature's consumer, and it was shipped to a 0.19 network.** Deploying the
coordinator put `MS_PER_LAYER_TTL_S` in front of live routing for the first time. With
`DEFAULT_MS_PER_LAYER = 40.0`, it scored `node-c-pavilion` — 4 cores, 99% reliable, **measured at
11.0** — at the 40.0 prior because its reading was 12.9 h old, while the 8 GB `82cbee` kept a
fresher **20.4**. `fastest_pick` weights by throughput, so the preference between them inverted
and traffic moved off the reliable machine onto the box whose hardware twin self-measures 5308
ms/layer. Chat fell to **0.05 tok/s** — 20 seconds a token.

The TTL's entire justification was [P34]'s outlier trap: an outlier is excluded by
`REPLICA_SLOWDOWN_LIMIT`, so it never serves, so nothing ever revises it. **The thing that
revises it is `agent.remeasure_loop`, which ships in 0.20.** On a 0.19 fleet there is no second
measurement, so expiry does not age a bad figure back in — it discards a good one and substitutes
a guess about the machine for a number taken on it.

**Fixed:** `stage_ms` uses the measurement whatever its age; only the public display treats an old
figure as unknown, which is the half that answers the privacy problem it was written for. The
outlier case is still covered — `REPLICA_SLOWDOWN_LIMIT` drops an 8× replica before weighting, and
an agent re-measures on restart (`18f1da` did exactly that, 4146 → 5308). Measured recovery:
**0.05 → 0.9 tok/s**, against 0.64 as the last clean distributed figure.

**REVISIT WHEN 0.20 IS ON EVERY NODE.** With hourly re-measurement the TTL becomes correct again
and `stage_ms` should go back to `ms_per_layer_fresh`. Both tests that asserted the old rule were
rewritten rather than deleted, one reproducing the live inversion (`11.0` stale must beat `20.4`
fresh), so the trade is stated where the next person will meet it.

**The lesson, and it generalises past this bug:** the change was correct, tested, reviewed, and
recorded. What made it a regression was an undocumented dependency — it assumed a fleet that
re-measures. **"Tested" says nothing about which fleet the assumptions hold on**, and this repo
now has several 0.20-dependent behaviours live on 0.19 nodes ([P33]'s `model_id`, [P37]'s `holds`).
Those fail safe. This one did not, because it silently substituted a worse number instead of
declining to answer.

**Closed on both sides, but only one side is deployed (2026-08-10).** The agent re-measures
hourly (`REMEASURE_INTERVAL_S`), and the coordinator ages a figure out after
`MS_PER_LAYER_TTL_S` (6 h), scoring the node at the default prior again — the same treatment a
never-measured node gets, which is what lets an excluded outlier back in to be re-measured at
all. The agent half **needs 0.20 on the node**; the coordinator half is live.

**The display read the raw column, and that is how the stale figure kept being used as evidence
(2026-08-11).** Routing expired it; `main.py`'s node table and the node's own dashboard did not.
Live proof, from the founder's own status table: `agent-bhpc012101-18f1da … ms/layer 4146.6`
sitting beside peers at 8–20, read as the current state of the network — while the router had
already discarded it and was scoring that node at the prior. Worse on the operator's own page,
which told the volunteer *"4146.6 ms/layer measured"*: an accusation that their PC is 500×
slower than its peers, sourced from a number the coordinator itself no longer believed.

Fixed by giving freshness **one definition** — `router.ms_per_layer_fresh`, called by
`stage_ms` and by both dashboards, so routing and display cannot drift again. The table keeps
the number but marks it `· stale` (it is evidence of a bad measurement, just not a current
one); the node's own page omits the clause entirely rather than accusing the hardware. A NULL
`ms_per_layer_at` is treated as legacy, **not** expired — reading it as stale would silently
reset every pre-column node to the default prior, network-wide.
`coordinator/test_stale_speed_display.py` (13 tests) pins the rule, including that one.

This is [P37]'s lesson — *a stale diagnostic field read as a live one* — recurring in the
surface a person actually looks at, which is the one place it turns into a wrong decision.

### [P32] 🔴 A pinned layer range does not survive the node re-registering — the mechanism behind [P27] recurring (2026-08-09)

**`neuron fix` writes to the coordinator. The node's own `config.json` is what actually decides.**
So the pin holds only until that node next re-registers, and then it silently reverts.

Observed, not theorised: the exact command
`./coordinator/pin_layers.sh --driver agent-optinovate-6ff49d` was run and verified on
2026-08-09 (Session 55, half nine — *"both nodes adopted their new ranges without a restart"*).
**~4 hours later `node-c-pavilion` was back on 0-27**, the whole model, and the network was
unroutable again.

**The loop, each step verified against the code:**

1. `POST /network/layers` updates the **coordinator's `nodes` table** only. Nothing is pushed
   into any node's local config.
2. `agent.py:413` (`ensure_placement`) returns early when config already holds a range — so a
   node that has one **never re-asks** the coordinator where it belongs.
3. `agent.py:569` puts that config range straight into the registration body.
4. `models.py:361` applies it with **no COALESCE**:
   `layer_start=excluded.layer_start, layer_end=excluded.layer_end`.
   The four fields immediately below it — `ms_per_layer`, `head_ms`, `platform`,
   `hw_fingerprint` — **are** COALESCEd, and `gpu_vram_gb`/`gpu_name` were deliberately
   reasoned about in [P31]. The protective pattern was applied all around this line and not to it.

Any re-registration closes the loop: a restart, a reconnect, or the stale-relay-ticket refresh
(the path Session 55 added, which re-registers an already-credentialed node). The operator's pin
is overwritten by the node's own stale opinion, and nothing reports that it happened.

**`layers_pinned` does NOT prevent this.** It guards `reconsider_placement` only, which
`agent.py:435` states is *"Only ever called while PROBATIONARY"*. `node-c-pavilion` is
**trusted**, so on that node the flag is close to a no-op. What actually keeps this machine's
driver correct is that `agent/config.driver.json` holds `0-9` — the right value — so its
re-assertion is harmless. Pavilion re-asserts `0-27`, and that is the entire difference.

**Why nobody notices: coverage is not routability.** With pavilion on 0-27 and the driver on
0-9, both start at layer 0; `router._walk` (`router.py:104`) advances by `max(layer_end)` from
each cursor, takes 27, and the cursor jumps to 28. The chain is **one stage**, and
`node_a.coord_get_chain` accepts only two or three. Meanwhile `/status` reports
`total_layers_covered: 28/28`, `uncovered_layers: []`, **`network_healthy: true`** — because
every layer genuinely IS covered. The health check answers a different question from the one
routing asks, and only routing's answer reaches a user.

**What it costs.** Distributed chat is dead. `pin_layers.sh:11` records the sharper cost: each
request *"dies client-side after the coordinator has already taken a wallet hold"*, so every
attempt locks escrow that only returns after `HOLD_TTL_S` (600 s). And because no request ever
completes, **neither machine earns** — pavilion is not doing all the work, it is doing none.

**Relationship to [P27].** Same root, opposite halves. [P27] is the OFFLINE case: a node that
was away keeps a stale range because `POST /network/layers` only rewrites nodes that are online.
This is the ONLINE case: a node that *was* correctly rewritten reverts anyway on its next
registration. Both are the same ownership question — the coordinator is treated as authoritative
for placement by the operator tooling, and as advisory by the agent. Fixing [P27] alone leaves
this one live.

**Three ways out, deliberately not chosen here — they differ a lot in blast radius.** A and B
below; C follows the self-heal note, because the reason it exists only becomes clear there:

- **A. Correct the node's local config** (`layer_start: 10`, `layer_end: 27` on pavilion).
  **REJECTED (2026-08-09), and the reason is the point of the project.** It only works because
  pavilion is in the founder's house. At a thousand nodes you will never have access to a
  volunteer's `config.json` — so A is not merely narrow, it is *structurally unavailable* the
  moment the network is real. Any fix that requires touching a specific machine has a ceiling of
  "the number of machines one person can babysit", and `SCALING.md` is a plan for rather more
  than that. Recorded because it was very nearly done: it is cheap, it works tonight, and that
  is exactly what makes it the wrong habit to form.
- **B. The coordinator owns placement — a node PROPOSES, the coordinator DISPOSES.**
  **DONE (2026-08-09), see below.** Not the bare `COALESCE`, which is unsafe for the reason
  below; the working version turned out to be smaller *and* safer than expected.

**The automatic repair has the same blind spot, which is why nothing caught this.** The
coordinator is not passive about placement: a node joining with no configured range gets
`suggest_placement`, which IS stage-aware (`main.py:898` caps stages at `PIPELINE_STAGES`, extras
become replicas — *"Handing out a 4-stage chain is the same failure as the 1-stage one"*), and a
node leaving that opens a hole is covered by `self_heal` each health sweep. But `self_heal`
(`migration.py:297`) opens with:

```python
missing, covering_ids = router.covering_and_missing(nodes, serving["layers"])
if not missing:
    ...
    return self.heal_status()
```

**It keys entirely on `missing`.** Pavilion on 0-27 plus the driver on 0-9 leaves nothing
uncovered, so self-heal returns having done nothing — it inspected the network, found it
healthy, and moved on. `PIPELINE_STAGES` is enforced when handing out a NEW placement and never
re-checked against the roster afterwards. So the coverage/routability confusion is not just in
`/status`: it is in the repair path too, which is why a manual `neuron fix` is still the
backstop.

- **C. Make the health sweep check stage count, not just coverage** — **DONE (2026-08-09), see
  below.** Catches the SYMPTOM class however it arises (including causes neither A nor B
  anticipates). Complementary to A and B rather than an alternative: it does not address the
  ownership inversion, it ends the silence.

**C, as built.** Deliberately **detection only — it moves nothing.** Auto-repair was considered
and rejected for now: repairing means rewriting ranges that the nodes themselves re-assert, so
until A or B settles ownership, a repairing sweep and a re-registering node would take turns
overwriting each other every 60 seconds. Making a silent failure loud is the half that is
correct under every outcome of that decision.

- **`router.chain_shape(nodes, total)`** — a pure description of the roster: `stages`, `ranges`,
  `routable`. Deterministic chooser, because this describes a network rather than routing a
  request. Checks `missing` **as well as** the stage count: a chain that stops at a gap still has
  a plausible-looking stage count, and calling that routable would be the same half-answer again.
- **`/status` now reports `stages`, `chain_ranges` and `routable`**, and **`network_healthy` now
  means "a request can actually complete"** — which is what every consumer already believed it
  meant (the dashboard dot, `neuron_doctor`'s verdict, `docs/index.html`, `ui/app.py`). Coverage
  alone answered a narrower question while presenting as that one.
- **The health sweep logs the transition**, not the state — an unroutable network is not
  self-correcting, so logging it every 60s would produce 1,440 lines a day and be read as
  wallpaper (the heartbeat lesson, Session 55 half seven). The line names the cost and the
  remedy: chats fail *after* a wallet hold is taken, fix with `pin_layers.sh --driver`.
- **The dashboard and `neuron_doctor` now distinguish the two failures.** Both used to say
  "chain incomplete", which sent an operator hunting for a missing node while every layer was
  present. The covered-but-unroutable case is the one that reads as fine everywhere else, so it
  is the one that has to name itself.
- **`config.MIN_PIPELINE_STAGES = 2`**, named next to `PIPELINE_STAGES`. The ceiling was already
  enforced when handing out a new placement; the floor was enforced nowhere. They are one rule,
  and splitting them is what let half of it go unchecked.

`coordinator/test_routability.py` **14/14** (new), pinning the distinction rather than the
incident: the exact live roster is covered-and-unroutable, a gap is unroutable despite 2 stages,
offline and probationary nodes are excluded (relevant here — [P27]'s office PCs are offline
holding a stale 0-13), and the shape is stable across repeated calls. **Tripwire verified to
fire:** reverting `network_healthy` to coverage-only fails
`test_summary_is_not_healthy_when_covered_but_unroutable`, then restored.

**B, as built.** The naive `COALESCE` is unsafe: `reconsider_placement` moves a probationary node
by rewriting its config and re-registering, so ignoring the claim would leave the node serving one
range while the coordinator recorded another — a correctness hazard (wrong activations, failed
proof-of-compute, no clue why), which is worse than the bug being fixed.

What makes the safe version small is a fact already true in the code: **the node already follows
the coordinator.** `/node/{id}/slice-info` returns the coordinator's stored range (`main.py:679`)
and `agent.setup()` serves whatever that returns. So the registration echo was never how a node
learned its range — it was only ever how a *stale* range got written back. Ignoring it removes the
one path by which the two could disagree; it does not create one.

1. **`models.register_node` no longer updates `layer_start`/`layer_end` on conflict.** A first
   registration (INSERT) still establishes the range, so a brand-new node — and a wiped machine
   rejoining after `DELETE` — is unaffected. The authoritative paths are untouched and all go
   through `update_layers()`: `/network/layers`, migration cutover, self-heal.
2. **The claim is recorded, not discarded** — `reported_layer_start`/`reported_layer_end`, with
   `placement_drift` on every node dict. Silently ignoring the node would have traded one
   invisible fact for another. NULL means "no claim seen since this shipped", never a mismatch,
   so live nodes do not all light up on first deploy.
3. **The registration reply now reads `assigned_layers` back from the DB**, exactly as `standing`
   above it already did and for the same reason: it used to echo the caller, so the agent's
   startup line would have confidently printed the stale range it had just failed to impose.
4. **`reconsider_placement` is superseded rather than broken.** It existed for the 2026-08-07
   failure where several nodes joining at once were all told to take the identical gap — which
   `self_heal` now handles coordinator-side, from the roster, without needing the node to
   cooperate. Under B a probationary node simply stays where the coordinator put it.
5. **Agent side (needs 0.20 to reach volunteers):** `setup()` now persists the served range into
   config, making it a CACHE of the coordinator's answer instead of a rival opinion — this ends
   the drift at source. And if `--layers` disagrees with the assignment it **says so**: an
   override that silently does nothing is [P31] repeated with a manual trigger.

**Status: DEPLOYED 2026-08-09; agent half pending 0.20.** The coordinator change ships by
**restart alone** and fixes every already-installed 0.18/0.19 agent, because it stops *listening*
to the stale claim rather than requiring the agent to stop making it — the same lever
`GPU_EXECUTION` used in Session 55. Schema migration verified on the live DB: both columns added,
5 node rows and 39 ledger rows intact, supply still exactly 1,000,000,000.

**It caught a real recurrence within minutes of going live, which is the empirical argument for
C.** Between the afternoon pin and this deploy, `agent-optinovate-6ff49d` had drifted to **0-27**
— the driver itself this time, not pavilion — collapsing the chain to one stage again. The
coordinator said so, in the journal, unprompted:

```
19:47:29 [health] NETWORK NOT ROUTABLE: the chain walks to 1 stage(s) [[0, 27]], and a driver
         accepts 2-3. Every layer may still be covered -- coverage is not routability. Chats
         will fail AFTER a wallet hold is taken. Fix: ./coordinator/pin_layers.sh --driver ...
19:48:32 [health] chain is routable again: 2 stage(s) [[0, 9], [10, 27]]
```

Under the previous code that same state reported `network_healthy: true`. Twice in one day the
pin was lost silently; the third time the network reported it itself, in one line, with the
remedy. Re-pinned to `[[0,9],[10,27]]` — and this is the first pin that placement ownership will
actually hold.

`coordinator/test_placement_ownership.py` **9/9** (new), including the full live sequence: pin,
re-register, still 2 stages. **Tripwire verified to fire** — restoring the overwrite fails
`test_reregistration_does_not_move_an_assigned_node` ("the pin must survive") and
`test_the_live_sequence_no_longer_reverts` ("the pin reverted — [P32] is back"), then restored.

### [P31] 🟡 GPU support was shipped, documented, and had never once executed — partly fixed (2026-08-08)

**The capability announced in v0.18.0 does not exist in the binary that announced it.** Verified
2026-08-08 against the installed package, not inferred:

| claim | reality |
|---|---|
| `common.py` moves a shard onto CUDA | true, and **unreached**: `slice_downloader.py:299` returns `cast_linears(model)` with no device move, and `agent/node_server.py` uses that loader. `common.move_model_to_device` has **one caller repo-wide** (`common.py:256`), on the bench/verifier path. |
| torch can use the card | `dist/neuron-agent/_internal/torch/version.py:4` is `2.4.1+cpu`, `cuda = None`. |
| `local_gguf` offloads every layer | `llama_supports_gpu_offload()` → **False**; the package ships ggml-base/ggml-cpu/ggml/llama/mtmd and **no `ggml-cuda`**. `n_gpu_layers=-1` was accepted and silently ignored while the log said "offloading every layer". |
| `agent.log` says `device: cuda:0` | comes from `common.device_name()`, whose only caller is `common.py:257` — **a line no agent can print**. `INSTALL.md` asked first-GPU volunteers to send it. |

**The one that was a live hazard, not just a false claim.** `balancer.GPU_EXECUTION = True` sized
a GPU node by VRAM, with **no OS reserve** on that branch while the RAM branch reserved 3 GB. The
weights were in system RAM the whole time. A volunteer with a **12 GB card and 8 GB of RAM** was
eligible for **19 layers** of Qwen2.5-7B — 8.9 GB at the tier's fp16 basis, ~17.7 GB at the fp32
the runtime actually defaults to. That is the OOM `max_layers_for` exists to prevent, produced by
the function that prevents it. `coordinator/test_gpu_capability.py:108-142` **asserted the harm**
(24 GB ⇒ 18 layers), so the suite was green throughout.

**Fixed (2026-08-08):**

1. **`GPU_EXECUTION = False`.** Ships by **coordinator restart alone** — every already-installed
   0.18 agent stops being over-assigned with no installer and no agent release.
2. **`sane_vram_gb()` + `VRAM_OS_RESERVE_GB = 1.5`**, so re-enabling later is safe rather than a
   second guess. `main.py` clamps the field at the registration edge — **to `None`, never a
   rejection**, since a 422 over a cosmetic field is [P24]'s failure class.
3. **VRAM/name follow `has_gpu`** in `models.py` instead of being COALESCEd, so a node that loses
   its card stops carrying phantom VRAM forever.
4. **The offload gate**, checked before the hardware probe and applied to `NEURON_GPU_LAYERS`
   too — an override that silently does nothing is the same bug with a manual trigger.
5. **`weights on <device>`** on every slice load, read off the loaded tensors, never from
   `common.DEVICE`. The gap between configured intent and where the bytes actually are is what
   let this survive a whole release, so the log now reports the bytes.
6. **Corrections** in `RELEASE_NOTES_v0.18.0.md` (dated block, published text left intact),
   `CHANGELOG.md`, `INSTALL.md`, and the five source comments that still said "CPU-only pipeline"
   while `GPU_EXECUTION` was `True`.

**Also fixed (2026-08-08, second pass):**

7. **The device path now follows the activations**, so fixing the loader would no longer mean
   crash-on-first-token. `batching.py`'s auxiliary tensors (KV padding, mask, position_ids,
   cache_position) derive their device from `hidden`/`k`; the batched stages return CPU like
   their unbatched twins in `common.py` already did — that divergence was the bug, since a real
   chain serves through the batcher and never touches the unbatched path; `wire_codec.py`'s
   Hadamard matrix follows its operand and every `.numpy()` reaches CPU first; and every
   `send_msg` caller passes `common._to_cpu`, which fixes `common.py:474-482`'s legacy
   `torch.save` **from the callers** without editing `common.py` (build rule 7).
   **Provably inert on CPU:** `test_batching.py`'s batched-vs-sequential figure is `4.530e-06`
   before the change and `4.530e-06` after, to the digit.
8. **`torch.cuda.synchronize()` in the self-benchmark.** CUDA kernels are queued, not run, so
   the old timing measured how fast a machine can *enqueue* work. A GPU node would have reported
   an absurd `ms_per_layer`, and `balancer.solve` would have handed the fastest-looking machine
   in the network nearly every layer — the same OOM by a second route, through the speed field
   instead of the memory one.
9. **A tripwire instead of a fix for the loader.** `slice_downloader.py:299` is deliberately
   left alone and now carries a comment saying why; root `test_device_path.py` fails if it
   starts moving weights, and says what must be verified on real hardware first. It also fails
   if `GPU_EXECUTION` is switched on while the loader still leaves weights in system RAM —
   those two facts are a pair, and splitting them is what caused this entry.

10. **The fp32 sizing gap — every node on every tier was cleared for twice its real footprint.**
    Not a GPU problem at all, and the largest of the lot. `model_tiers.gb_per_layer` is computed
    at fp16 (2 bytes/param) and its comment justified that with "common.WEIGHT_DTYPE=fp16".
    **That is false**: `common.py:62` reads `NEURON_WEIGHT_DTYPE` and defaults to **fp32**,
    nothing in the agent or installer sets it, and `cast_linears` is a no-op at fp32. So an
    8 GB machine on the 7B tier was cleared for 8 layers = **7.46 GB of weights against a
    3.75 GB budget**. Latent only because the network serves the 1.5B, where 28 layers is too
    few for the cap to bind. Fixed by `balancer.effective_gb_per_layer`, which scales the
    tier's figure by `weight_bytes_for(node)` — pessimistic (fp32) by default, and honest for a
    node once it reports `weight_dtype`. Deliberately NOT fixed by doubling the table: a
    layer's size is a property of the MODEL, the dtype is a property of the NODE.
    `coordinator/test_weight_dtype_sizing.py` 28/28.
    - **What it revealed about the fleet:** the live trio (8/8/12 GB) holds **15 of the 28
      layers** a 7B needs at fp32 and therefore cannot serve it. Two test fixtures asserted
      that it could. They were resized to machines that genuinely fit rather than pinned to
      fp16, because pinning would have preserved exactly the kind of false premise that caused
      this entry. `test_migration` 49/49 and `test_model_tiers` 23/23 with no dtype pin.
    - **No live impact from deploying it:** the network is on the 1.5B floor with one online
      node, and the 7B already reads `feasible: false, placeable: false`. The fix changes what
      the ladder will clear as machines join, not what is served today.

**Still open:**
- **Phase 4b — the loader itself.** One line, guarded by the tripwire above. It needs real GPU
  hardware to verify, because "passes on CPU" is not "works on a GPU" and that distinction is
  the whole subject of this entry. **Decided 2026-08-08: rent GPU time online rather than wait
  for a card** — see the decisions log for the order that session must follow, which is not
  arbitrary: 4a verified before 4b, `selftest_shard` reconciled before `GPU_EXECUTION` goes
  back on, packaging last.
- **`selftest_shard.py`** compares a CPU `load_model()` against a CUDA `load_model_shard()` and
  requires `diff == 0`, so **on a GPU box build rule 6 is unsatisfiable** as written.
- **CUDA packaging.** With one installer, bundling a CUDA torch means every volunteer downloads
  ~2.5 GB against today's 206.6 MB, and `updater.py` has a 600 s timeout, no resume, no
  disk-space check and no rollback. `/agent/version` cannot express a per-capability build, and
  NVIDIA redistributables are not covered by `tools/gen_notices.py` — a real licensing gap.
- **The real lever is still unbuilt.** Decode is bandwidth-bound at ~30 GB/s on DDR against
  360–1000 GB/s on a consumer card; 200B needs ~30–80 CPU machines or ~10 GPU ones. See [P30].

**The process finding worth keeping.** Every individual caveat in Session 42 was honest — "the
CUDA path has never run", "no speedup is claimed" — and the conclusion drawn from them was still
wrong, because they all blamed the **absent test card** (a hardware gap, which reads as
"untested") when the cause was **packaging** (which reads as "impossible"). "Written but
unverified" and "cannot execute in this binary" are different claims, and only the second one
tells you not to size a volunteer's memory by it.

### [P30] 🔴 The fast engine cannot reach the models NEURON exists to serve

`agent/node_server.py` runs **PyTorch fp32** (`load_slice_model`). `llama_cpp` appears only on
driver/local paths — `local_gguf.py`, `local_chat.py`, `openai_compat.py`, `node_a.py`,
`neuron_driver.py`, `ui/app.py`. The local engine only triggers when a machine can hold the
**whole** model, and a 200B model never fits on anyone's machine by definition. **So the ~17×
engine is structurally unreachable for exactly the models the network exists for.**

Decode is memory-bandwidth bound; this project's own three benchmarks (1.5B, 7B, 70B — 47×
apart) all imply ~30 GB/s, which is the DDR bus. Projected 200B across 80 machines: **~32
s/token** on fp32, **~2-5 s/token** with llama.cpp-class kernels. `PIPELINE.md`'s stated
ceiling of "80 hops ≈ 2.4 s/token" counts network traversals **only** — compute is ~15× that
and dominates, so its steps 3 and 4 optimise the smaller term. The kernel decides 200B, not
the split.

**Blocked on a missing capability, verified 2026-08-07 against the installed package:**
`llama-cpp-python` **0.3.34** does **not** expose `rpc_servers` on `Llama.__init__`, and offers
no layer-range entry point — `Llama` is a whole-model abstraction. The routes are therefore:

1. a llama-cpp-python build that exposes the RPC backend, or the standalone `rpc-server` binary
   driven out-of-process;
2. C++ against `ggml` directly, embedding llama.cpp and speaking NEURON's existing wire
   protocol — which would also remove Python and PyTorch from volunteer machines entirely;
3. stay on PyTorch for the network and accept ~32 s/token at 200B.

Route 2 is the one worth doing: it owns the layer that is NEURON's (the wire protocol,
placement, economics) and borrows the layer that is not (the kernel). **Do not write another
matmul** — this repo has measured a hand-written AVX2 int8 kernel at **1.44×** against
llama.cpp's ~17×, on the same CPU, in the same language.

**GPU is the larger multiplier and is half-done.** `has_gpu`/`gpu_vram_gb`/`gpu_name` are in
the coordinator schema, `agent/gpu.py` detects, and as of 2026-08-07 `local_gguf` offloads to
VRAM. But the *network* path still cannot use a GPU, for the same reason above. With GPUs the
arithmetic changes completely — 200B on ten RTX 3060s is ~3 tok/s on **8× fewer machines** than
the CPU fleet — and network latency then becomes the bottleneck, which **flips `PIPELINE.md`'s
build order**: the direct-reply and single-codec fixes stop being premature the moment the
first GPU joins.

### [P29] 🔴 A user who runs out of NRN has no way to get more

`/infer` holds ~0.158 NRN before it dispatches, so a wallet below that is refused before a
chain is built. The faucet is **one-time per wallet** (409 on re-claim) and gated on
`is_oauth_wallet`. Node earnings land in a *separate* ledger row with no route into a user
wallet. So the sequence for any real user is: sign in, get 25 NRN, spend it over ~158
messages, and then the product stops permanently with no action available.

Hit on 2026-08-07 by the founder's own two wallets — both at `balance 0.0` with `total_earned`
25.0 and 25.962. Unblocked by hand: `models.transfer('__ecosystem__', <wallet>, 25.0)`, which
moves rather than mints (supply invariant still exactly 1,000,000,000.0). That is an operator
action over SSH, not a product.

The UI now at least *says* so before the message is sent, and invites the user to contribute a
machine — but earning by donating hardware credits the **node**, not the wallet, so the
invitation does not yet actually solve the problem it points at. **Either node earnings need a
path into the owner's wallet, or the faucet needs a recurring allowance.** Decide before
strangers arrive, not after.

**Worse than "no route" — there is no OWNER (2026-08-09).** `nodes` has no wallet or owner
column at all, so the coordinator cannot even name the wallet a node's earnings belong to. This
is a missing relationship, not a missing transfer. The pattern to copy already exists:
`/node/{id}/payout-address` proves node ownership with the node's token plus a signature, and an
owner-wallet binding can mirror it exactly.

**And availability emission (§11.4, built 2026-08-09) makes it urgent rather than theoretical.**
Nodes now earn NRN for *being available*, not only for serving — so the network is actively
accruing balances to node rows that no human can spend. Paying volunteers in something they
cannot reach is worse than not paying them: it manufactures a promise the product cannot keep.
**The node→wallet link is now the blocking item for the whole earn-and-spend loop.**

### [P28] 🟡 A node that is merely slow is reported to the user as a node that died

`common.HOT_TIMEOUT_S = 30` applies per socket read during generation. On 2026-08-07 the first
real distributed answer tripped it — driver log `TimeoutError: timed out` — and the driver
classified the timeout as a dead node, tore down the chain and re-prefilled. The reply carried
`↻ recovered from 1 node drop`. **Nothing dropped.**

Three costs. The user is told a machine failed when none did, which is the same class of
misdirection as the no-tokens bug fixed the same day. The reroute re-prefills the whole chain,
so the request gets *slower*, which makes the next timeout more likely — a node that is merely
slow can be driven into a loop of self-inflicted "deaths". And a genuinely slow-but-working
volunteer looks unreliable in the logs.

A read timeout and a closed socket are different events and should be reported differently;
the timeout also wants to scale with the stage's measured `ms_per_layer` rather than being one
constant for every node on every network.

### [P27] 🟡 An offline node's stale range outbids a correctly pinned one when it returns

`router._walk` advances by `max(layer_end)` among the nodes starting at each cursor. Two office
PCs are offline holding a stale **0-13** from before the 2026-08-07 placement fix.
`POST /network/layers` only rewrites nodes that are **online**, so those ranges were not
corrected when the split was pinned to 0-9 / 10-27.

When either powers on: at cursor 0 the candidates are the driver (0-9) and a stale node (0-13),
`max` picks 13, the **driver is dropped from the chain entirely**, the cursor jumps to 14, and
nothing starts at 14 — `missing (14, 27)`, 503, chat dead. Same mechanism as the morning's
`missing layers 21-27`, different numbers.

Preferring the widest claim is right for genuine replicas and wrong for stale ranges, and
nothing currently distinguishes them. Workaround: **run `neuron fix` whenever a machine joins
or leaves.** Unverified: whether self-heal makes the window worse — the driver, having lost the
tie, is no longer in `covering_ids` and may read as idle surplus to the reassignment sweep.

**Superseded (2026-08-10): the coordinator now repairs this itself.** `router.canonical_assignment`
is `pin_layers.sh` expressed server-side, applied from the health sweep whenever the chain is not
routable — stage 1 pinned to the driver's shard, at most `PIPELINE_STAGES` stages, everyone else
replicating. `neuron fix` being a MANUAL step was the real defect: on 2026-08-10 four machines
came online, flapped overnight, and by morning the chain was `[[0,9],[10,13],[14,20],[21,27]]`
with the driver stranded on 14-20 — unrepaired because nobody was awake to type a command.
Verified live by deliberately breaking the chain and leaving it: repaired in under 15 seconds.

**That workaround is weaker than it reads — see [P32] (2026-08-09).** `neuron fix` writes the
range to the COORDINATOR, while the node's own `config.json` is what it re-asserts on every
registration (`models.py:361` overwrites the range with no COALESCE). So the pin reverts on the
next restart, reconnect or relay-ticket refresh, and the ONLINE case fails the same way this
entry describes for the OFFLINE one.

### [P26] 🟢 The network promoted itself onto a model no single node could hold — fixed (2026-08-07)

**Observed live.** The network auto-upgraded from Qwen2.5-1.5B to **Qwen2.5-7B-Instruct** the
moment three nodes were eligible, and the even split handed **9–10 layers** of it — about
4.5 GB of weights at fp16 — to machines with **8 GB and 12 GB of TOTAL RAM**, on top of Windows,
a browser and whatever else the volunteer was doing. That is an OOM kill, not a slow stage.

**Cause — two gates that each assumed the other was asking.**

1. `model_tiers` qualified the promotion on **aggregate** capacity: `min_nodes` and
   `min_ram_gb`, checked against the sum across the network. "3 nodes · 20 GB" is satisfied by
   8+8+12, and *none of those machines can hold a third of a 7B model*. A pipeline stage lives on
   one machine; summing RAM answers a question nobody asked.
2. `migration.plan_migration` — used by both tier migrations and self-heal — partitioned layers
   **evenly**, with no memory awareness at all. It never looked at a node's RAM, so it could not
   have refused.

`balancer.py` had solved exactly this in Session 14 (`max_layers_for`, `solve(gb_per_layer=…)`,
prompted by the identical arithmetic: Llama-3.1-8B at fp32 is 0.87 GB/layer, and an equal 3-way
split assigned 9.3 GB to a machine with 5–6 GB free). Two things kept that from helping:
the migration path never called it, **and it was dead code in production anyway** — the cap reads
`ram_free_gb`, which no node has ever reported (`agent.py` sends `ram_gb`, psutil *total*) and
`_balanced_plan` never passed `gb_per_layer` either. The guard existed, passed its unit tests, and
had never once applied to a real plan.

**Fixed, four parts:**

1. **`max_layers_for` falls back to total RAM** minus `RAM_OS_RESERVE_GB` (3 GB — Windows 11 idling
   with a browser, on the machines here) when no free figure is reported, so the cap is live against
   the data real nodes actually send. VRAM still wins for a GPU node, unchanged.
2. **`plan_migration` fits the split to the machines.** It still starts even, then shifts layers
   off any node over its cap onto nodes with room (`balancer.fit_to_capacity`, now shared with
   `solve`). On the trio above: 10/9/9 becomes 8/8/12.
3. **A migration whose target no partition can hold is refused**, not attempted —
   `partition_shortfall > 0` leaves the network serving what it has, with the reason in
   `/network/migration` and on the dashboard. Deliberately *not* a new phase: `self_heal` runs only
   while `phase == "steady"`, so parking a blocked migration elsewhere would have left the network
   unable to grow **and** unable to repair itself, for as long as the unservable tier stayed
   qualified.
4. **The tier gate asks the per-node question too.** `placeable(nodes, tier)` gates promotion and
   demotion alongside the aggregate check, so the ladder stops advertising an upgrade that can
   never happen. Tiers carry a measured `gb_per_layer`; a tier without one (env-injected via
   `NEURON_MODEL_TIERS`) is unconstrained, because an unknown footprint is not grounds to refuse
   to serve.

Self-heal is deliberately **not** gated on capacity: a coverage gap means no request can complete
at all, so it re-splits on the best fit available and reports `capacity_shortfall` instead. The
honest response to "the survivors cannot hold this model" is to serve a smaller one, and that is
the tier controller's demotion — which now happens, because placement feeds it.

**Known gap, recorded not guessed:** the driver additionally holds the embedding and lm_head
(~2.2 GB for the 7B) and no cap accounts for it, so the first node in a plan is under-charged by
that much. Wants a measured `head_gb` column.

**Files:** `coordinator/balancer.py`, `coordinator/migration.py`, `coordinator/model_tiers.py`,
`coordinator/main.py`. **Tests:** `coordinator/test_migration.py`, `coordinator/test_model_tiers.py`.

### [P25] 🟢 Every stranger was sent to the same layers, so the chain could never close — fixed (2026-08-07)

**Observed live.** The dashboard read **21/28 layers, DEGRADED — chain incomplete** with four
nodes online and three of them serving the *identical* range, layers 0–13:

| node | layers | standing |
|---|---|---|
| agent-optinovate-67e4eb | 0–13 | verified |
| agent-bhpc012104-5452f9 | 0–13 | probationary |
| agent-bhpc012101-80c3fc | 0–13 | probationary |
| node-c-pavilion | 14–20 | trusted |

Layers 21–27 belonged to nobody. Four machines, three copies of one slice, and not a single
request able to complete.

**Cause — placement reasoned over the ELIGIBLE nodes only.** `suggest_placement` walked the
chain from `build_chain`, which filters to `eligible` (trusted or proof-of-compute verified,
`models.py`). That filter is correct for **routing** — an unverified node must not receive live
traffic — and wrong for **placement**. A probationary node has already been handed a range and
has already downloaded that slice; it is a real claim on that segment. Filtering it out made
every newcomer invisible to the next one:

1. Only the trusted node (14–20) was eligible, so the first gap read **0–13**.
2. Stranger #1 asked, was told 0–13, registered — probationary, therefore invisible.
3. Stranger #2 asked. The coordinator's view was *unchanged*. Told **0–13** again.
4. Stranger #3, same. Nobody was ever offered 21–27.

Then it froze: `ensure_placement` returns early whenever config already holds a range, so no
node re-asked, and the answer written on first run was permanent.

**Nothing self-corrected**, either. Self-heal only reassigns nodes that are eligible, online AND
*true idle surplus*; here every eligible node was covering something, so surplus was empty and
the sweep did nothing on every tick, forever — [R6] in `RESILIENCE.md`. The balancer meanwhile
had a perfectly good split sitting on `/network/plan` (0–22 / 23–27, 2.8× better than an equal
split) that only a human calling `POST /network/rebalance` would ever apply. Nobody was looking.

**Fixed, three parts:**

1. **Placement now reasons over every ONLINE node** (`router.placement_roster`), routing still
   over eligible ones only. An unverified node's range counts as taken, so the next joiner is
   sent somewhere useful. Regression test: three strangers joining a network whose only eligible
   node holds 14–20 must close both gaps before anyone duplicates a slice.
2. **`GET /node/placement?exclude=<node_id>`** answers "where would I go if I weren't already
   here?", and a **probationary** agent asks it right after registering — moving if the answer
   differs, then re-registering on the new range (once; `_replaced` bounds it). Free at exactly
   that moment and no other: a probationary node serves no live traffic, so nothing is lost.
   Self-stabilising, because a node whose range is genuinely needed is handed the same range
   back. `--layers` sets `layers_pinned` and is never second-guessed.
3. **[R6]** — a coverage gap with no idle surplus now triggers a full re-split instead of a
   no-op. See `RESILIENCE.md`.

Also fixed on the way: `_walk` did not clamp a gap to the model's layer count, so a node holding
a range beyond `total` (what a migration onto a *smaller* model leaves behind until each node
reloads) stretched the gap past the end of the model and self-heal planned a slice tens of layers
too long.

**Files:** `coordinator/router.py`, `coordinator/migration.py`, `coordinator/main.py`,
`agent/agent.py`. **Tests:** `coordinator/test_placement.py`, `coordinator/test_migration.py`,
`agent/test_replacement.py`.

### [P24] 🟡 Strangers register fine, then sit PROBATIONARY forever — BLOCKS S12

**ROOT CAUSE FOUND 2026-08-07, and it was never the installer.** The coordinator's journal has
the registration failure in full:

```
Aug 06 06:25:55 … "POST /node/register HTTP/1.1" 500 Internal Server Error
  File "/home/ubuntu/neuron/coordinator/main.py", line 347, in register
      fingerprint = models.register_node(
  TypeError: register_node() got an unexpected keyword argument 'has_gpu'
```

**A half-deployed coordinator.** `main.py` had been updated to pass `has_gpu=`; the `models.py`
on the VM had not. Every registration 500'd for about an hour. Nothing was wrong with the
agent, the installer, or the volunteer's machine — and the agent could only log
`500, retrying in 60s`, forever, at WARNING. This is precisely what `coordinator/deploy.sh`'s
header warns about; shipping `coordinator/` as one atomic tar is why it cannot recur. Verified
fixed: every `/node/register` in the 36h to 2026-08-07 returned 200, and the driver registered
200 OK at 21:38.

**Two theories this killed:**

- **`_save()` truncation is NOT the mechanism.** The work PC's `config.json` was finally pulled
  and is complete, valid JSON with `node_id`/`node_token` intact. Session 53 said
  truncated-or-empty would confirm it; it is neither. The atomic-save fix stays — it is correct
  work — but it does not explain 0.18.
- **`registered_at` is not a restart indicator.** `coordinator/models.py:357` is
  `ON CONFLICT(node_id) DO UPDATE SET … status='online', last_seen=excluded.last_seen`, and
  `registered_at` is **not in that list**. It records only a node's first-ever join. An hour was
  spent restarting a machine on the strength of that field reading "8 days".

**The finding that reframes the whole entry: no stranger's machine has ever reached the
coordinator.** Since 2026-07-26 exactly **two** IP addresses have hit `/node/register`, and both
are ours — the office and the home connection. (The addresses themselves are deliberately not
recorded here: this repository is public, and they identify a residence. The count is the
finding; the numbers add nothing.) The 2026-08-07 install did not fail *at* registration; it
never made a successful request at all. v0.18 cannot say which, because its logging starts
after the config read. **Do not put 0.18 in front of another stranger** — cut 0.19 first, and
if that machine is still reachable, take `%LOCALAPPDATA%\NEURON\` off it before a reinstall
destroys the evidence. The absence of a log is itself the finding.

**Fixed on 2026-08-05 (everything except the 0.18 crash itself, which needs the machine).**
The four numbered fixes below are done, plus the two the entry did not name — the silent
startup and the config-shaped crash suspect. What is still open is stated at the end.

- **Logging now exists before anything can fail.** `_setup_logging()` is the first statement of
  `main()`, ahead of the config read that used to precede it. The module-level imports, the
  config read, `Agent.run()`, the tray, and `packaging/neuron_app_entry.py` each write their own
  traceback to `agent.log` via a stdlib-only `crash_log()` that needs no logging config and no
  successful import. The frozen entry duplicates the log-path rule deliberately: the import it
  is reporting on is the one it must not depend on. Windowed tray mode also puts the log's
  location in a message box, because there is no console for a traceback to reach. The first
  line of the log now names the version, which the log from the machine could not.
  `agent/test_startup_is_never_silent.py` 10/10.
- **A 0.17 config no longer kills 0.18 in the constructor.** `Agent.__init__` read
  `cfg["coordinator"]` directly, so any key a newer build expects and an older one never wrote
  was a `KeyError` before a single log line existed — [P24]'s own second suspect. Missing keys
  now fall back to `DEFAULT_CONFIG` and are *named in the log*; they are not written into the
  user's file (injecting defaults an operator deliberately omitted has its own failure mode —
  see `install.py`'s `write_config`).
- **The agent says when it is online and useless (fix 3).** `GET /node/{id}/ping` now returns
  `standing`, so a node learns it more than once — it used to learn it exactly once, in the
  reply to its registration, which is why the fact scrolled away on day one. After 60
  heartbeats still probationary the agent WARNs, repeatedly, that this machine is healthy but
  serves nothing and earns nothing and that the operator's verifier may be down; the heartbeat
  line itself says `probationary, not yet serving or earning` instead of `active`; and
  promotion is announced. `agent/test_probation_is_visible.py` 14/14.
- **The verifier retries, escalates, and proves it is alive (fix 4).** A roster read is retried
  3× with backoff inside the cycle (502s and DNS failures are the normal weather on a home
  connection — its last three lines ever were exactly those), a sustained outage escalates
  WARNING → ERROR naming the consequence, and recovery is logged. **The important one:** a
  healthy verifier used to log *nothing* (`nothing to verify` is debug-level), so its log looked
  identical whether it was running or had been dead since Monday. It now writes an alive line
  every 30 cycles, so silence means dead. `test_verifier_survives.py` 15/15.
- **Something watches it now (fix 1 and 2).** `neuron_doctor.py` fails when `verify_service.log`
  stops growing for 90 minutes — run against the live network on 2026-08-05 it reported the
  verifier dead for **2,970 minutes**, confirming this entry's diagnosis from the outside. The
  probationary check the entry called for is also in the doctor. And `agent/verifier_keepalive.py`
  restarts the verifier if it is not running: the Windows Run key that "installed" it fires once
  at login and never again, so it survives a reboot and not a crash. Install the 5-minute check
  with `python agent/install.py --verifier-keepalive` (a scheduled task on Windows, a cron line
  elsewhere); `--with-verifier` now installs it too.
- **Supervision is now installed and proven (2026-08-05 22:55).** `NEURONVerifierKeepalive` runs
  every 5 minutes as a scheduled task, pointed at the **venv** interpreter — not PATH's, which
  has no torch and would start a verifier that dies on import every five minutes, the trap
  `install.py` already documents for the Run key. Both branches have live evidence: the *start*
  branch is what brought the verifier back at 19:00 after two days dead, and the *detect-and-
  skip* branch was exercised at 22:55 (task exit 0, no second process spawned). Remove with
  `schtasks /delete /f /tn NEURONVerifierKeepalive`.
  - **Its own liveness check had to be fixed first.** The first version counted any process
    whose command line mentioned `verify_service.py` as healthy, so a *hung* verifier would be
    reported fine forever — "online means nothing" for the third time in this file, after [P21]
    and [P22], committed by the file written to prevent [P24]. It now reads the alive line's age:
    a process silent for 90 minutes is stopped and replaced. Two guards that are load-bearing —
    **process age** (right after an outage the log is days stale and the process seconds old:
    that is the verifier just restarted, still loading torch, and without this guard the
    keepalive would kill what it had itself started, forever) and **all matching PIDs, not the
    first** (a venv's `pythonw.exe` on Windows is a redirector that spawns the base interpreter
    as a child, so one verifier is two processes — 5.8 MB shim plus the 177 MB child doing the
    work — and killing only the first leaves the working half behind).
  - **Known limit, stated rather than assumed:** the task is `Logon Mode: Interactive only`, so
    it covers a crash — the failure that actually happened — but not a reboot with nobody
    signed in. That needs `/ru SYSTEM` and admin rights.
- **Investigated further (2026-08-05, same day). One suspect eliminated, a new mechanism found.**
  - **The packaging suspect is DEAD.** The shipped `dist/neuron-agent/neuron-agent.exe` (0.18.0,
    built 2026-08-03) was run here as `--headless --help`, which executes every module-level
    import — the whole suspect surface, `agent.gpu` included — and then exits at argparse before
    touching the network or the config. It printed usage and **exited 0**. A Python module
    missing from a PyInstaller bundle fails identically on every machine, so the frozen import
    chain is not what dies.
  - **`_save()` was not atomic, and that produces this exact signature.** It was
    `json.dump(self.cfg, open(self.config_path, "w"), indent=2)`: the open **truncates
    immediately**, the handle was never explicitly closed, and it runs from eight places
    including registration and migration cutover. Anything stopping the process mid-write — an
    OS shutdown, a task kill, installing a new version over a running agent (`neuron.iss` has no
    stop-the-app step) — leaves a truncated `config.json`. The next start's `json.load` then
    raises *before* v0.18's logging existed: silent death, no log file, **on every start,
    forever**, which is the reported symptom precisely (not an intermittent crash — an install
    that simply does nothing). It also explains "0.17 worked, 0.18 does not": the config is
    corrupted at the moment of the upgrade, and only the newer binary ever reads it again.
  - **Fixed:** `_save()` writes a temp file in the same directory, `flush()` + `fsync()`, then
    `os.replace()` (atomic on Windows and POSIX), keeping the previous good copy as `.prev`.
    New `load_config()` falls back to `.prev` when the live file is unreadable, preserves the
    broken one as `.corrupt` for evidence, and **raises rather than inventing a fresh identity**
    when there is no backup — `node_id`/`node_token` are the node's claim on everything it has
    earned, so silently regenerating them would orphan the owner's balance and rejoin as a
    stranger. `agent/test_startup_is_never_silent.py` 19/19.
  - **This is a mechanism, not a proof.** Confirming it needs that machine's `config.json`; if
    it is truncated or empty, that is the answer. Worth checking before anything else.
- **Still open:** the v0.18.0 crash has not been *identified*, only made diagnosable — that
  needs the machine. Run the installed 0.18 exe from a terminal, or just run it again once a
  build with these changes exists: it writes `agent.log` no matter where it dies. The two
  suspects the entry names (the GPU path, the in-place config upgrade) are unchanged, though the
  second is no longer fatal. Note that machine cannot auto-update to the fix — auto-update only
  runs inside a working agent, and that one crashes — so it needs a manual install.

**Original entry follows.**

### [P24-orig] 🔴 Strangers register fine, then sit PROBATIONARY forever

**This entry replaces an earlier, wrong diagnosis.** The first version blamed a failed
registration producing a null node_id and a `/node/None/slice-info` 404. The agent log from the
machine disproves that: registration worked perfectly.

```
13:03:33 auto-placed on layers 10-18 (fill-gap: chain is missing layers 10-18)
13:03:33 registered as agent-bhpc012104 [probationary], assigned layers [10, 18] (8 cores, 8 GB)
13:03:35 downloading slice: layers 10-18 (~0.84 GB) ...
13:05:50 node server started on port 50999
13:05:50 relay tunnel started - reachable via 150.230.22.250:9001 (NAT-friendly)
```

Node id assigned, slice downloaded, server up, NAT relay up, then `heartbeat ok - active`
continuously across three separate days. The join path works end to end. The 404 was a
different, unlogged event; the null-node_id mechanism is real but was not what happened here,
and the earlier entry asserted it without the log.

- **What actually went wrong**, from the same log's third line:

  ```
  PROBATIONARY: serving challenges only - a verifier must confirm this node
  (proof-of-compute) before it receives live requests or earns NRN
  ```

  It was never promoted. Not once in the whole log. A probationary node serves no live
  requests and earns no NRN, so the owner's experience is: installed it, it says it is running,
  nothing ever happens, zero balance, forever.

- **Root cause: `verify_service.py` is not running.** It is the operator-side service that
  promotes probationary nodes, and `verify_service.log` stops dead at **2026-08-03 17:27**,
  two days before this was investigated. Its final entries are all failures:

  ```
  2026-08-03 00:01:02 coordinator unreachable: 502 Bad Gateway .../node/list
  2026-08-03 14:28:21 coordinator unreachable: Failed to resolve 'neuronnet.duckdns.org'
  2026-08-03 17:27:47 coordinator unreachable: Failed to resolve 'neuronnet.duckdns.org'
  ```

  Every node it ever verified is `agent-optinovate-*` — the founder's own machines. It never
  reached the outside one.

- **This is the exact failure the verifier was written to prevent**, in its own words: *"a
  stranger who joined at 3am sat at zero NRN until somebody noticed them. A network whose
  onboarding requires the operator to be awake is a demo."* The service exists, it works, and
  it died silently — so the network is back to being a demo and nothing said so. Same class as
  [R6]: the information existed, nothing was watching it.

- **Fixes, in order:**
  1. **Restart the verifier and keep it up.** It has a systemd unit on Linux and is installed by
     `agent/install.py`; on the founder's Windows box it is evidently not supervised. A service
     whose death is invisible will die again.
  2. **The doctor must check it.** `/status` already returns `probationary_nodes`. Any node
     stuck probationary across more than a couple of verifier cycles means promotion is broken.
     Added to `neuron_doctor.py` in the same change as this entry.
  3. **The agent should say so.** After N heartbeats still probationary, log a WARNING naming
     the situation ("still awaiting verification after 30 minutes — the network operator's
     verifier may be down") rather than an indefinite calm `heartbeat ok - active`.
  4. **Retry harder on transient coordinator errors.** 502 and DNS failures are expected on a
     home connection; the verifier should back off and keep trying, not stop.

- **Secondary finding from the same log:** the work PC ran at 76-100% CPU and repeatedly logged
  `paused (cpu 91% > donation ceiling 85%) - not advertising availability`. Even fully verified,
  a busy work machine would contribute rarely. Not a bug — the resource guard behaving exactly
  as designed — but it means "install it on a work PC" is not a path to useful capacity, and the
  install guide should set that expectation.

- **v0.18.0 is a REGRESSION: 0.17 worked on this machine, 0.18 does not.** The log above is
  0.17 -- it registered, downloaded, served heartbeats for days. 0.18 produced **no log at all**
  and reportedly a 404. Two separate faults, and the first is what makes the second
  undiagnosable.

  **Why there is no log: `_setup_logging()` is called at `agent/agent.py:994`, inside `main()`,
  and only after the config has been read** (`cfg.get("log_level")`). Every module-level import
  (`from agent import gpu`, line 46, among them) and the whole config load run *before* any log
  file exists. Any failure in that window is completely silent -- no file, no message, nothing
  the owner can send. **This is the highest-value fix in this entry:** move file logging to the
  first statement of `main()`, ahead of the config read, so a bad config gets logged instead of
  swallowing itself. A run that produces no log is indistinguishable from a run that never
  happened.

  **Prime suspect for the crash: the GPU path, the substantive change in 0.18.** Its own release
  notes say *"No machine in this project has an NVIDIA GPU ... it has never actually executed on
  a GPU."* A work PC is exactly the machine likely to have one. `agent/gpu.py` guards its
  `nvidia-smi` and torch calls and the import itself looks safe, so this is a suspect, not a
  conclusion -- but it is the only substantive difference between the version that worked and
  the version that does not.

  **Second suspect: the in-place upgrade.** The release notes tell users to install over the top
  and keep `%LOCALAPPDATA%\NEURON`, so 0.18 reads 0.17's config. A field 0.18 expects and 0.17
  never wrote would throw during the config load -- before logging exists.

  **The one action that identifies it:** run the installed 0.18 executable from a terminal
  rather than the tray or service, so the traceback reaches stderr. Until logging moves earlier,
  there is no other way to see this class of failure.

  Windows blocking is real and separate (code-signing, ROADMAP S16), but it does not explain a
  404 -- a blocked binary does not make HTTP requests.

### [P23] 🔴 `prune_test_accounts.py` will sweep the first stranger's wallet — BLOCKS S12

- **Symptom:** the founder's two OAuth wallets show `balance: 0.0` on the live coordinator while
  `total_earned` survives intact (25.0 and 25.962). Not a bug and not a database loss — they
  were deliberately swept by the Session 33–35 dev-account cleanup, when they genuinely were
  dev wallets. `coordinator/test_prune_test_accounts.py` even fixtures their real balances
  (24.295, 25.0). The money went to `__ecosystem__` as designed.
- **The actual problem is forward-looking.** `classify()` prunes by prefix:

  ```python
  PRUNE_PREFIXES = (
      ("node_a-cli-", "CLI test wallet from wallet-settlement development"),
      ("w_",          "faucet-funded test wallet (OAuth wallet development)"),
  )
  ```

  and `wallet_for_oauth()` mints **every** user wallet as `"w_" + secrets.token_hex(16)`. There
  is no other format. So the rule that means "dev wallet" today means "every real user who ever
  signs in" tomorrow. Run the script once after a stranger joins and it takes their 25 NRN
  welcome grant, silently, and files it as a test account in the audit log.
- **Why it was right when written and wrong now:** when the rule was added, the only `w_`
  wallets in existence *were* the founder's. Open join (S12) changes that, and nothing in the
  script notices the change.
- **Mitigating for the moment:** the script is a manual `--execute` one-time repair, not a
  routine, and it refuses to run while `__escrow__` is non-empty. So this is not firing today.
- **Fix (decide before the first stranger, not after):** drop the blanket `w_` prefix so real
  wallets fall through to `unclassified` — which the script already refuses to execute on — and
  name the handful of genuine dev wallets with `--prune-also`. Alternative: gate the prefix
  behind an explicit `--include-oauth-wallets` flag defaulting off. Either way
  `test_prune_test_accounts.py` needs its expectations updated in the same change: three `w_`
  fixtures move from `PRUNED` to `unclassified`, and the "already-empty prune target" case
  needs a non-`w_` target.
- **Restoring the founder's 49.3 NRN is a separate, optional decision.** The pre-sweep balances
  are recoverable from `backups-offbox/neuron-20260802-115534.db`
  (`w_ef7ca467…` 25.0, `w_d35c84dd…` 24.295). Not done automatically: moving balances on a live
  ledger is the founder's call, not a repair to be applied silently.

### [P1] 🔴 Single-user speed vs. the Green AI thesis — HIGHEST
- **Symptom:** the chat UI shows ~0.7–1.4 tok/s for one user; the headline 6.16 tok/s
  is *aggregate throughput* under concurrent load, not single-request speed.
- **Reality:** a serial CPU pipeline over the internet **cannot** beat a GPU datacenter
  on single-user latency. That is physics, not a bug.
- **Why it may be OK:** Green AI competes on **energy, cost, capacity, and access**, not
  latency. "Fast enough" + free + green + runs-huge-models is the real moat. Petals runs
  70B across volunteers at ~1–2 tok/s and is respected.
- **Risk if ignored:** pitching NEURON as "fast" sets up a credibility collapse; adoption
  stalls if users expect ChatGPT speed.
- **Mitigations (see P2–P3, P8):** quantization, better networking, speculative decoding,
  right-sizing splits, and a *positioning* that leads with green/cost/capacity.

### [P2] 🟡 Still running fp32 — biggest untapped speedup
- **Symptom:** all inference is fp32. bf16 was rejected (these CPUs lack a bf16 GEMM).
- **Opportunity:** int8/int4 could give 2–4×. Prior note (sessions): "true int4/int8
  speedup = the llama.cpp/GGUF path" — i.e. PyTorch int8 on these CPUs was *assumed*
  weak but never measured on the full model.
- **Action:** MEASURE int8 dynamic-quant vs fp32 on the real model/CPU. → done, below.
- **Measurement (2026-07-24, this Windows CPU, full Qwen2.5-1.5B, greedy, 40 tok):**
  - fp32: **3.18 tok/s** (matches the known ~3.2 single-machine baseline ✅)
  - int8 dynamic (all `nn.Linear` → qint8): **10.99 tok/s = 3.46× faster** ✅
  - **CONCLUSION — speed is achievable; the thesis is not dead on speed.** ~7–11 tok/s
    (comfortably readable) is within reach → see new caveat [P9].
  - qengines available on this box: `onednn, x86, fbgemm` (fbgemm used).

### [P19] 🟢 The pipeline wire ran arbitrary code from any peer — fixed (2026-07-29)
- **Symptom:** `common.recv_msg` did `torch.load(io.BytesIO(data), weights_only=False)` on
  whatever arrived on the socket. A `torch.save` payload is a **pickle**, and unpickling with
  `weights_only=False` calls whatever the sender's `__reduce__` names. Demonstrated locally:
  a crafted `act` message executes the sender's code in the receiver's process.
- **Why it mattered here specifically:** this is not a "don't accept files from strangers"
  theoretical. Every node deserialises messages from the node before it in the chain, and the
  driver deserialises the reply — so the reach was **both directions**, driver ↔ node. Since
  Session 12 each node's port is published on a **public relay**, so the sender need not even
  be in the chain. `router.build_chain` will happily put an open-join stranger's machine in
  the pipeline of a request originating on the founder's PC. It was the single most direct
  path from "a stranger installed the agent" to "a stranger runs code on your desktop", and
  no document (`SECURITY.md`, the S14 audit, [P14]) had ever named it.
- **Fixed:** `weights_only=True` on the legacy path — verified against every message shape the
  protocol actually sends (config, config-ack, act, act-reply, bye all carry only
  dict/str/int/float/bool/Tensor). The new `wire_codec` frames are JSON + raw tensor bytes and
  contain nothing executable at all. Also capped the 8-byte length prefix at 512 MB: node
  ports face the open internet, and an unchecked length let a stray scanner make a 1 GB relay
  VM allocate an arbitrary buffer (same class of bug as the `relay.recv_json` hardening).
- **Regression tests:** `test_wire_codec.py` — a hostile pickle is refused, an absurd length
  prefix is refused before allocating, and a legacy `torch.save` sender still round-trips.

### [P20] 🟡 The wire ships raw fp32 activations, once per token per hop — measured, now 4.3× smaller
- **Symptom:** `common.send_msg` `torch.save`d full-precision tensors. Measured on the real
  3-stage chain (Qwen2.5-1.5B, H=1536, 6 prompts × 48 tokens): **12,508 bytes per message**,
  of which 1,153 is pure pickle framing. Paid at every junction, every token — and since
  Session 12 a relayed hop crosses the public VM **twice**, so relay egress pays it twice.
- **What the number means at the size NEURON actually exists for.** A 70B model is H=8192
  over ~20 stages. One decode token then costs 33.6 KB per hop, **0.69 MB across the chain**,
  before TCP and relay overhead. On a 10 Mbit/s home upload that is ~27 ms of pure
  serialisation per hop — **~0.55 s/token spent on the wire**, latency the compute never
  sees. At `i8h` it is 8.2 KB/hop, 0.17 MB/token, ~0.13 s. [P3] already observed the network
  dominates per-token cost; this is one of the reasons why, and it was never measured
  until now.
- **Measured (2026-07-29, `bench_wire.py`, 6 prompts × 48 tokens, codec at all three
  junctions so error compounds exactly as on the wire):**

  | codec | B/msg | vs before | text identical to fp32 | max abs Δlogit |
  |---|---|---|---|---|
  | `torch.save` fp32 (was) | 12508 | 1.00× | 6/6 | 0.0000 |
  | `f32` (new framing, no pickle) | 11355 | 1.10× | 6/6 | 0.0000 |
  | `f16` | 5723 | 2.19× | 6/6 | 0.0069 |
  | **`i8h`** (Hadamard + blockwise int8) | **2946** | **4.25×** | **6/6** | **0.2054** |

  and the schemes measured and **rejected** (exploratory sweep, 3 prompts × 48 tokens, so
  the identity column is out of 3):

  | codec | B/msg | vs before | identical | max abs Δlogit |
  |---|---|---|---|---|
  | fp8 e4m3 | 2786 | 4.43× | 0/3 | **nan** |
  | int8 per-tensor | 2792 | 4.42× | 0/3 | 30.46 |
  | int8 blockwise-256 (*Petals' scheme*) | 2812 | 4.39× | 2/3 | 1.13 |
  | int8 blockwise-64 | 3014 | 4.09× | 1/3 | 0.59 |
  | int4 blockwise-32 | 1577 | 7.82× | 0/3 | 4.52 |

- **[P9] again, on the wire.** Real junction activations measured at **absmax 6620, std 42,
  worst channel ≈ 750× the median**. An absmax quantizer takes its scale from that one
  channel, so everything else collapses — which is why plain int8 lands at Δlogit 30. fp8
  e4m3 cannot even represent 6620 (its max is 448), overflows to inf, and the generation goes
  NaN. **Note that Petals' own scheme — blockwise int8, no rotation — diverged on 1 of 3
  prompts here.** Copying the paper's mechanism verbatim was not sufficient.
- **What fixed it:** QuaRot's insight (arXiv:2404.00456), applied at the transport layer
  instead of the model. A Hadamard rotation is orthogonal, so it spreads the outlier evenly
  across the block without changing the vector; the sender rotates before quantizing and the
  receiver rotates back. Because it is pure transport there is **no weight surgery and no
  calibration** — the model never sees it. Same bytes as unrotated int8, ~7× less error
  (rel_l2 0.0037 vs 0.0276).
- **The rotation has to be cheap or it is not worth doing.** The textbook log-n butterfly
  cost 1.26 ms per call at H=8192 — a fifth of the wire time it was saving. Done instead as
  a single matmul against a cached Hadamard matrix: **0.045 ms, 28× faster**. Whole-codec
  cost is now 0.54 ms encode+decode per hop against ~6.4 ms of transmission saved on a
  10 Mbit/s upload. **On a fast link that trade reverses** (0.54 ms to save 0.06 ms), so
  `NEURON_WIRE_CODEC` pins the codec for LAN/datacenter deployments; the default assumes
  volunteers' home connections, which is what NEURON is.
- **int4 was measured and deliberately NOT shipped:** ~0.53 B/elem at ~9% relative error per
  hop. Survivable over a 3-node chain, not over the 20-node chain a 70B model implies, and
  the wire is the one place where being wrong is silent.
- **i8h is gated on model size, because the same benchmark run against 0.5B disagrees with
  the 1.5B result:**

  | model | `i8h` identical | `i8h` max Δlogit | `f16` identical | `f16` max Δlogit |
  |---|---|---|---|---|
  | Qwen2.5-1.5B (H=1536) | 6/6 | 0.2054 | 6/6 | 0.0069 |
  | Qwen2.5-0.5B (H=896) | **3/6** | 0.5034 | 6/6 | 0.0075 |

  The diverging 0.5B answers stay correct and on-topic — they re-word, typically 100+
  characters in — so this is drift, not the [P9]-style collapse unrotated int8 causes. But
  it is drift the larger model does not show, in the expected direction: fewer parameters,
  less redundancy to absorb the noise. `wire_codec.preference()` therefore offers `i8h` only
  at H ≥ 1536 and `f16` (still 2.3×, 6/6) below it. Small models are both the fragile ones
  and the cheap ones to ship uncompressed, so nothing is given up. **Two data points, not a
  curve** — measure a third model before trusting the threshold away from 896/1536.
- **A false lead worth recording.** An end-to-end socket run appeared to show i8h giving a
  factually worse answer on 0.5B ("the sky is blue because it reflects sunlight" vs
  "because of the scattering of sunlight by tiny…"). That was a test-rig artifact: a stray
  earlier `node_b.py` was still bound to the port, and `SO_REUSEADDR` let a second one bind
  alongside it, so connections landed on either process. With one listener, all four codecs
  return the identical answer. The size gate rests on the in-process benchmark above, which
  has no sockets and no such failure mode.
- **Rollout:** codec is negotiated per hop in the config handshake, and a peer that offers
  nothing recognised keeps the legacy format — so a half-upgraded fleet keeps working.
  **Not yet deployed to the live nodes** (Pavilion/OptiPlex still run the old build; they
  will negotiate down to legacy until updated).

### [P21] 🟢 Nothing restarts a node — fixed and installed on both remote nodes (2026-08-01)
- **Fixed by:** `agent/install.py --startup` on the Pavilion and the OptiPlex. Both now run the
  agent as a `systemd --user` service with **`Restart=always` / `RestartSec=10`** (was
  `Restart=on-failure`, which leaves a node dead after any clean-looking exit).
- **The flag in this file's "what to do" did not exist.** `install.py` had `--no-startup`, not
  `--startup`, so the documented command failed with an argparse error — and running plain
  `install.py` instead would have been worse: `write_config()` wrote DEFAULT_CONFIG straight
  over the existing file, discarding `node_id`, `node_token`, the layer range the machine was
  actually serving and its `register_secret`, and re-pinning every node to layers 10-18. The
  documented repair for [P21] would have de-identified both live nodes. `write_config()` now
  merges and `--startup` exists as a non-interactive repair path for an already-working node.
- **A unit file alone was never going to be enough.** A `systemd --user` service does **not**
  start at boot unless the user has *lingering* enabled — without it the unit is bound to a
  login session, so `WantedBy=default.target` fires when somebody logs in and never after an
  unattended reboot, while looking correctly installed and `enabled` the whole time. Both
  machines had `Linger=no`. The installer now enables it (unprivileged first, then `sudo -n`)
  and, when neither is permitted — a machine with no sudo, i.e. exactly a stranger's laptop —
  falls back to **cron**, which needs no privileges: an `@reboot` line plus a two-minute
  keepalive (`agent/neuron-keepalive.sh`) that restarts the agent if it is not running. That
  fallback also bounds a crash at two minutes instead of "until somebody notices".
  `agent/uninstall.py` removes both, or an uninstalled agent would be resurrected every two
  minutes by a cron job nobody remembered.
- **The keepalive greps with the `[a]gent[.]py` bracket trick and lives in its own FILE** for
  the reason `sessions.md` already documents one level up: inline in the crontab, the guard's
  own command line contains the string it greps for, so it matches itself, concludes the agent
  is already running, and never starts anything.
- **"Online" now has to mean "serving".** `NodeServer.run()` reports a failed bind
  (`self.listening` / `self.bind_error`) instead of dying silently in its daemon thread;
  `agent.setup()` waits for the listener before advertising and retries the whole setup
  otherwise, and `heartbeat_loop()` refuses to ping while the listener is down, so the
  coordinator marks the node offline and routes around it. **This caught a live instance the
  moment it shipped:** both remote machines still had hand-started `node_ns.py` servers from
  Session 23 squatting port 50999 (17h 57m uptime), so the agent's bind failed — before this
  change it would have gone on logging `heartbeat ok — active` forever against a node that
  never bound. The stale processes were stopped; the agents own the port now.
- **VERIFIED BY A REAL REBOOT (2026-08-01).** The Pavilion was power-cycled: booted 14:03:14,
  `neuron-agent` active at **14:03:27 — 13 s after boot**, nobody logged in, no SSH, listening
  on :50999 and back in `/node/list` far inside the 2-minute bar. Linger is the part that
  earned its keep — a user-level unit starting with no login is exactly what a reboot tests,
  and it is what was missing. A `SIGKILL` of the agent on the OptiPlex separately recovered in
  **20 s**, covering the crash case. (The Pavilion cannot be rebooted remotely: no passwordless
  sudo, and polkit refuses `systemctl reboot` over SSH — it needs someone at the machine.)
- **Original entry, kept for the record:**

### [P21-orig] 🔴 Nothing restarts a node — the network dies on any reboot, and the fix already exists
- **Symptom:** the Pavilion and OptiPlex run the agent as a bare foreground process. A reboot,
  a crash, a closed SSH session or an OOM kill takes that node off the network permanently
  until somebody SSHes in and starts it by hand. Hit repeatedly on 2026-07-30: `nohup … &`
  over SSH does **not** survive the session ending (`setsid`, with a delay before the SSH
  command returns, does) — and even then nothing brings it back after a reboot.
- **Why it looked worse than it was:** a dead agent is invisible. The coordinator just shows
  the node `offline`, identical to "the owner closed their laptop", so the network silently
  runs degraded. Combined with [P4] this is the single most common way NEURON appears broken.
- **The fix is already written and was simply never used here.** `agent/install.py --startup`
  installs a real service on all three platforms — Windows registry Run key, `systemd --user`
  on Linux, and a `launchd` LaunchAgent on macOS ([P15]) — and `agent/uninstall.py` mirrors
  it. The packaged installer (`packaging/neuron.iss`) is the intended path. The two remote
  machines bypassed all of it: they were set up by hand-copying a few files in an early
  session, so they have no service, no auto-start and no restart-on-failure.
- **What to do:** (a) run the installer path (or `install.py --startup`) on the Pavilion and
  OptiPlex instead of launching the agent by hand; (b) make the systemd unit
  `Restart=always` with a `RestartSec` backoff, matching what the coordinator's own unit
  already does; (c) surface "node was auto-restarted N times" so a flapping machine is
  visible rather than silently degrading the network.
- **Ship this before handing anyone an installer.** A stranger will never SSH in to restart
  anything — if the agent does not come back by itself, that machine leaves the network on
  its first reboot and never returns, and the earn-rate they were promised quietly becomes
  zero.
- **Worse than dying: a node can report itself healthy while serving nothing.** `agent.py`
  starts `NodeServer.run()` in a daemon thread and never checks that it bound. Restart an
  agent while the previous process still holds port 50999 and the bind raises inside that
  thread, the thread dies silently, and the main loop carries on logging `heartbeat ok —
  active` forever. Observed live on the Pavilion: the coordinator showed the network
  `healthy: True, 28/28 layers` while the node refused every connection, so `/infer` handed
  drivers a chain that could not run. **The heartbeat must assert the listener is actually
  accepting**, not merely that the process is alive — otherwise "online" means nothing and
  routing sends real requests into a black hole.
- **Deploy note learned the hard way:** `nohup … &` over SSH dies with the session, and even
  `setsid … &` needs the SSH command to stay alive a few seconds before returning or the
  process never detaches. `pkill -f 'agent.agent'` also matches the `bash -c` running it and
  kills its own shell — use the bracket form (`'[a]gent.agent'`), the same trick `sessions.md`
  already documents for `[n]ode_b.py`.

### [P22] 🟢 A relay tunnel died silently while the node reported healthy — fixed (2026-08-01)
- **Symptom:** after ~2 hours the Windows node's public relay port accepted TCP and then
  answered nothing (20 s timeout), while both Linux nodes completed the same handshake in
  0.10 s. The agent logged `heartbeat ok — active` throughout, so `/node/list` showed the node
  **online and verified**, routing sent it real requests, and it served none of them. Restarting
  the agent fixed it instantly.
- **Root cause, measured not guessed:** `tunnel_client.run_tunnel` holds an outbound control
  connection and blocks in `recv_json` waiting for the relay to push `new_conn`. That socket is
  idle by design. When the relay's end goes away the socket stays ESTABLISHED until TCP
  keepalive gives up — and `SO_KEEPALIVE` alone uses the OS default idle timer, which on this
  machine is the Windows default **7,200,000 ms = 2 hours** (confirmed: no `KeepAliveTime`
  override in the registry). The observed ~2 h dead window is that timer, exactly.
- **Why it mattered more than it looks:** every stranger is relayed (`behind_nat` is the
  default since [P10]), and it reproduced on Windows, which is what most of them will run. It is
  [P21]'s "online means nothing" in a second place — Session 26 made the heartbeat assert the
  *local listener* had bound, and nothing asserted the *tunnel* still carried traffic.
- **Fixed, two layers:**
  1. `tunnel_client.set_keepalive()` — keepalive at **60 s idle / 10 s interval** via
     `SIO_KEEPALIVE_VALS` on Windows and `TCP_KEEPIDLE`/`INTVL`/`CNT` elsewhere. Detection drops
     from hours to about a minute, and the existing reconnect loop then does its job.
  2. `agent.relay_reachable()` — every 4th heartbeat the agent dials **its own public relay
     endpoint** and completes a real config handshake. A plain TCP connect proves nothing here:
     the relay accepts on the public port whether or not it can still reach the node, so a dead
     tunnel is indistinguishable from a healthy one until bytes come back. On failure the agent
     restarts the tunnel (its own stop flag, separate from the agent's) and **withholds the
     heartbeat**, so the coordinator marks it offline and routes around it instead of feeding a
     black hole.
- **Tests:** `agent/test_relay_liveness.py` — 5/5. The load-bearing case is a fake endpoint that
  *accepts and never speaks*, which a connect-only check would wrongly pass; plus a healthy
  handshake, a non-relayed node (no false alarm), and the assertion that a dead tunnel restarts
  the tunnel and sends no heartbeat.
- **Verified live:** deployed to all three nodes; all three public endpoints answer in
  0.03-0.10 s, and **zero false alarms** across 4+ probe cycles on every node.

### [P3] 🟡 Network latency dominates per-token cost
- **Symptom:** in the 8-request run, ~0.4 s **per token** was network. The Pavilion was on
  an Amsterdam **relay** (`relay "ams"`), not a direct link.
- **Mitigations:** prefer direct Tailscale links (avoid relays), co-locate nodes / LAN
  clusters, send fewer/batched hops, and use fewer stages for small models (P8).

### [P4] 🔴 Node availability / laptop suspend
- **Symptom:** the Pavilion (node_c) suspends and drops off Tailscale when idle, degrading
  the network to 2 nodes → incomplete 28-layer chain → no service. The S9 resource guard
  also pauses nodes on load/battery/user-activity.
- **Reality:** node churn is *expected* in a volunteer network. A single fixed 3-node chain
  has no redundancy.
- **Mitigation:** replication (multiple independent pipelines, P8) + coordinator routing
  around offline nodes; keep dev nodes awake for now.

### [P5] 🟡 No per-wallet NRN balance / debit (from S11)
- **Symptom:** the OpenAI API records the wallet and reports 1.0 NRN/request cost, but does
  not persist a user-side balance or debit it. Coordinator ledger only credits nodes.
- **Fix:** coordinator-side user-balance table + debit on `/complete`. Ties to S17 (on-chain).

### [P6] 🔵 Quality vs. size vs. speed tension
- Qwen2.5-1.5B is small; output is not production-grade. Bigger models (S15) improve quality
  but need more/bigger nodes and are slower. Every quality gain costs speed and vice-versa —
  the core three-way trade to manage deliberately.

### [P7] 🟢 Heterogeneous nodes + manual layer split — RESOLVED (S14)
- Pipeline runs at the speed of its slowest stage; layer split WAS hand-tuned (`--s1/--s2`).
- **Fixed (2026-07-25, Session 14):** `coordinator/balancer.py` solves for the time-equalizing
  split from each node's self-measured `ms_per_layer` + the driver's `head_ms` (`benchmark.py`).
  `GET /network/plan` computes it, `POST /network/rebalance` applies it. Reproduces the hand-tuned
  9/9/10 automatically; live on the cloud coordinator. Remaining nicety: dynamic re-balance +
  auto-reload when nodes join/leave mid-flight (today a range change needs the driver to reload
  with the new S1).

### [P9] 🔴 Naive int8 quantization destroys output quality (found in the P2 spike)
- **Symptom:** with 3.46× speed, the int8 model answered "Explain how a rainbow forms"
  with *"I'm sorry, but I can't provide an answer…"* while fp32 answered correctly.
  Naive dynamic int8 (all Linear incl. attention + the 150k-vocab head, no calibration)
  is too aggressive for a small (1.5B) model.
- **The real path (speed AND quality):** quality-preserving quantization —
  - **GPTQ / AWQ** (int4, per-channel scales + calibration; ~1% quality loss typical), or
  - **llama.cpp GGUF** (`Q4_K_M` / `Q8_0`) — proper k-quant schemes; also has an **RPC
    backend that distributes layers across machines**, i.e. quantized *and* distributed.
    This is a possible production-engine pivot from the hand-rolled fp32 PyTorch split
    (relates to [P8]); big change, big payoff — evaluate deliberately, not now.
  - cheaper interim: weight-only int8, or quantize MLP-only and keep attention/head in fp.
- **Status:** speed is proven reachable; the *method* is the open work. Schedule a
  dedicated "quality-preserving quantization" session (candidate: alongside/after S14),
  NOT a rushed integration now. No real users yet (per ROADMAP's One Rule).

### [P10] 🔴 No stranger can actually join yet — everything assumes ONE Tailscale net (BLOCKS S12)
- **Symptom:** the coordinator is Tailscale-only (`100.114.189.46:8001`, ufw scoped to
  `tailscale0`) and nodes reach each other over Tailscale IPs (`register_nodes` stores
  `tailscale_ip`; node_a dials node_c's IP; node_c dials node_b's). A stranger is **not**
  on the tailnet, so they can neither reach the coordinator nor be reached by/reach peers.
- **Two sub-problems:**
  1. **Public coordinator** — needs a public address: Cloudflare Tunnel, **Tailscale Funnel**,
     ngrok, or a cloud VPS (Oracle free tier was noted but never set up). Easy-ish.
  2. **Node ↔ node connectivity across NAT** — the hard one. Home nodes are behind NAT and
     can't accept inbound TCP. Middle/last stages currently MUST accept inbound. Options:
     (a) stranger installs Tailscale + joins via auth key — works today, but not "one click"
         and doesn't scale/secure to thousands; good enough to *prove* a first stranger;
     (b) relay all pipeline traffic through the public coordinator (re-architecture: nodes
         hold an outbound connection, coordinator brokers) — the real scalable fix;
     (c) NAT hole-punching (libp2p/WebRTC/holepunch) — most work.
- **Consequence for S12:** the "5-step, no-tech, one-click" install in the ROADMAP is not
  achievable on the current architecture. The realistic first-stranger proof is path (a);
  the true product needs (b). Decide the path before writing the install guide.
- **DECISION (2026-07-24):** move the coordinator to a **free-tier cloud VM** (Oracle Always
  Free / GCP e2-micro) so it has a public address WITHOUT exposing the founder's personal
  OptiPlex homeserver or Pavilion — those stay private, node-only. Solves sub-problem 1
  cleanly. Sub-problem 2 (stranger node↔node NAT) is still open; the cloud VM is the natural
  future home for the relay/broker. Deploy guide written: `coordinator/DEPLOY.md`; the
  coordinator is dependency-light (`coordinator/requirements.txt`, no torch). Repo stays
  PRIVATE for now. Blocked on the founder creating the VM (account/card = their action);
  deployment is turnkey once SSH exists.
- **DONE (2026-07-24) — coordinator DEPLOYED & PUBLICLY LIVE.** Oracle Always Free VM,
  Amsterdam, **x86 `VM.Standard.E2.1.Micro`** (ARM A1.Flex was out of capacity — known Oracle
  issue), Ubuntu 22.04, 1 GB RAM (ample; coordinator ~100 MB). Public IP **150.230.22.250:8001**.
  systemd service `neuron-coordinator` (auto-restart), **strong `NEURON_REGISTER_SECRET`** set
  (saved locally in gitignored `.env.coordinator`), iptables opened for 8001 (Oracle Ubuntu
  REJECTs non-22 by default) + Oracle VCN security-list ingress rule for 8001. `/status`,
  `/dashboard`, `/agent/version` all return 200 from the public internet. SSH:
  `ssh -i ~/.ssh/oracle_coordinator ubuntu@150.230.22.250`. **[P10] sub-problem 1 (public
  coordinator) = RESOLVED.** Remaining: sub-problem 2 (stranger node↔node NAT).
- **DONE (2026-07-24) — live network MIGRATED to the cloud coordinator + inference PROVEN.**
  `register_nodes.py` now reads `NEURON_REGISTER_SECRET` from env; registered node_a/node_c/node_b
  against `150.230.22.250:8001` (healthy 28/28), ran a prompt via `node_a.py --coordinator <cloud>`
  → correct answer, NRN credited on the CLOUD ledger (a 0.3214 / c 0.2893 / b 0.2893, fee 0.10).
  Nodes keep Tailscale for pipeline traffic; only the coordinator moved. Old OptiPlex coordinator
  left running as a harmless fallback (its nodes time out offline).
- **2026-07-24 — [P10] sub-problem 2 (stranger node↔NAT) — relay BUILT, PROVEN, DEPLOYED.**
  New `relay.py` (public host) + `tunnel_client.py` (node), **pure stdlib, ARM-safe,
  protocol-agnostic byte-splice → ZERO changes to node_*/common**. A node makes only OUTBOUND
  connections to the relay; the relay exposes a public port and reverse-tunnels to it (like a tiny
  self-hosted ngrok). Local selftest PASS (50 KB binary + 8 concurrent, byte-exact) against a node
  reachable ONLY via outbound. Deployed to the cloud VM as systemd `neuron-relay` (control 8010,
  data 8011, public 9000-9100 opened in iptables). **REMAINING for a live relayed/stranger node:**
  (a) open 8010/8011 + the node's public port (e.g. 9002) in the Oracle **security list** (console);
  (b) run `tunnel_client.py` on the node; (c) register it in the coordinator with the relay endpoint
  (`150.230.22.250:900X`) instead of a Tailscale IP; (d) prove an inference. The coordinator handing
  out a relay endpoint for NAT'd nodes is what finally makes a real first stranger possible.
- **2026-07-24 — LIVE PROOF PASSED (real node through the cloud relay).** Opened `8010-9100` in the
  Oracle security list; ran `tunnel_client` on node_b (OptiPlex, OUTBOUND-only to the cloud relay,
  public :9002); ran a real inference with the node_c→node_b hop FORCED through the relay
  (`node_a.py --host-b 150.230.22.250 --port-b 9002`) — over the public internet, NO Tailscale for
  that hop → correct answer (19 tok). node_b behaved exactly as a stranger reachable only via the
  relay. Test tunnel torn down after; node_b back to normal. **NAT-traversal mechanism proven
  end-to-end.**
- **2026-07-24 — relay onboarding AUTOMATED + deployed (one-click for a NAT'd node).** Coordinator
  `/node/register` now accepts `behind_nat` → assigns a free port from the pool (config `RELAY_*`),
  stores the node at the relay endpoint, returns a `relay` block. `tunnel_client.run_tunnel()` is a
  callable; `agent.py` auto-starts it from that block on registration (persists it in config for
  re-runs) → **a NAT'd node self-configures, zero manual steps**. Isolated test PASS (registered
  behind_nat → got port 9000 → reachable via the cloud relay byte-exact using only the coordinator's
  response). Deployed to the cloud coordinator (DB persisted, 3 nodes intact; `behind_nat` register
  live-verified). **Open-join DONE (2026-07-25, Session 17):** registration no longer needs the
  shared secret — a secret-less node joins *probationary* (excluded from routing/earning) and is
  promoted to *verified* by a proof-of-compute pass; the secret now just marks a node *trusted*
  (fast-path). `coordinator/test_open_join.py` 17/17; not yet deployed to the live coordinator.
  **What's genuinely left for a first stranger:** an actual outside person installs the agent
  (package + install guide + 4th-node placement), and `/complete` still needs auth ([P12]).
  Beyond ~100 relayed nodes: `SCALING.md`.
- **2026-08-01 — the relay is now the DEFAULT for every node, and the live nodes were moved
  onto it.** The machinery from 2026-07-24 was all present and none of it was in use: both
  remote nodes had `behind_nat: false` in their configs, so the coordinator stored their
  **Tailscale** addresses and `router.chain_public` handed those out to every peer that asked
  for a chain. A stranger placed anywhere except the final segment must dial the next hop, and
  `100.114.189.46` is not a routable address for anyone outside the founder's tailnet — so the
  relay existed, ran, and was bypassed by every real request. Changes:
  - `agent.use_relay()` defaults `behind_nat` to **True** (it was `.get("behind_nat", False)`,
    which contradicted `DEFAULT_CONFIG`'s `true` for any config that simply omitted the key),
    with `--relay/--no-relay` to override. `--no-relay` is now the exception a LAN cluster opts
    into, rather than the accidental default.
  - `setup()` re-registers when a node is in relay mode but holds no relay endpoint, not just
    when its ticket is stale. Without that, a node switched to relay mode after its first
    registration would keep advertising its old direct address forever, since `register()` is
    skipped once credentials exist.
  - Registering with `behind_nat` and getting **no** relay block back is now logged as a
    warning naming the address peers were given instead. It used to pass silently, which is
    the exact failure this entry exists to prevent.
  - Both live nodes re-registered onto relay endpoints (`150.230.22.250:9002` / `:9003`) and
    both were confirmed reachable on those ports from a machine with Tailscale **stopped**.
- **2026-08-01 — two bugs found by actually running the stranger path, either one fatal:**
  1. **`agent/__init__.py` did not exist**, so INSTALL.md's step 4 (`python agent/agent.py`)
     died on `ImportError: cannot import name 'local_chat' from partially initialized module
     'agent'`. Python puts `<repo>/agent` on `sys.path` ahead of anything the script adds, and
     a regular module named `agent` beats a namespace-package directory of the same name, so
     agent.py imported *itself*. Both live nodes had an `__init__.py` created by hand during
     setup, which is why this was invisible for eleven sessions — every stranger would have hit
     it on the first command in the install guide.
  2. **Proof-of-compute could not promote a node on any segment except the last.**
     `node_server.py`'s probe role set `s2 = self.hi` (the inclusive last layer) where `s2` is
     used as a Python slice bound (`layers[s1:s2]`), so it ran one layer too few and advertised
     a range `security/proof_of_compute.attest_middle` rejects outright. Auto-placement puts a
     joining stranger wherever the coverage GAP is — which was layers 0-9 here — so the
     realistic case was the broken one. Fixed to `self.hi + 1`.

### [P11] 🟡 Public-launch hygiene (before the repo goes public)
- Hardcoded private Tailscale/LAN IPs across README/ROADMAP/agent/coordinator — genericize to
  a placeholder + a real public coordinator URL (low security risk — CGNAT, unreachable — but
  leaks infra and is useless to strangers).
- `neuron-dev-secret` default registration secret — the PUBLIC coordinator must set
  `NEURON_REGISTER_SECRET` to a real value, else anyone can register nodes.
- Otherwise clean: no tokens/keys/passwords/DB in tracked files (`.gitignore` verified).

### [P8] 🔵 Scaling model: pipeline depth vs. replication (key architecture decision)
- **More nodes ≠ faster single request.** A deeper single pipeline is *slower* per request
  (more hops). The right use of a huge group is **replication** (many independent 3-node
  pipelines → ~linear aggregate throughput) and **bigger models** (split a model no single
  machine can hold). The coordinator currently assembles ONE chain; scaling throughput needs
  it to build and load-balance across **many** pipelines. This is the big future design step.
  **Partial (2026-07-25, Session 18): REPLICATION landed at the segment level** — `router.build_chain`
  now load-balances across nodes that share a segment (a 4th node registers with an existing node's
  layer range; each request picks a replica at random, so both earn under concurrent load). This is
  the throughput-via-replication shape for ONE pipeline slot; full many-pipeline / cross-driver
  load-balancing (multiple driver hosts) is still the bigger open step.
  **→ Full prototype→worldwide scaling plan now written up in `SCALING.md`** (connectivity: P2P +
  relay fabric; coordination: regional → DHT; topology: many small pipelines; phased plan; Petals
  as the proven reference model; and the rule: don't build the scale layer before the first stranger).

### [P18] 🟢 Sign-in was unshippable, and the network ran on plain HTTP — both fixed live (2026-07-29)
- **Login could not ship.** It ran on the driver, so an OAuth *client secret* had to exist in
  that process — i.e. on every stranger's PC once each installed agent began serving its own
  Chat UI. A secret inside a distributed binary is extractable, and the alternative (each user
  registering their own Google Cloud project before they can send a message) is not a product.
  Moved to the coordinator: the founder registers ONE app, every install just gets a button.
  Agent holds no secret at all; the browser hand-back is a single-use 120s code, never the
  wallet_id (which spends real NRN and must not sit in a URL or history).
- **Google forced the HTTPS work.** Google rejects redirect URIs that are plain HTTP or a raw
  IP, so `http://150.230.22.250:8001/...` could not be registered at all — GitHub accepted it,
  but GitHub is a developer platform, so GitHub-only silently capped the audience at people who
  already write code. `coordinator/setup_https.sh` (Caddy + Let's Encrypt, auto-renewing) took
  the coordinator to `https://neuronnet.duckdns.org`. Node tokens no longer cross the internet
  in cleartext either, which mattered independently ([P11]).
- **Verified live, both providers, in the ledger** — `google … email_verified: 1` and `github`,
  distinct identities and wallets. A normal person can now install, click once, and chat.
- **Gotchas worth keeping:** (1) systemd **drop-ins override the main unit**, so
  setup_https.sh's edit to `.service` was silently beaten by an older `PUBLIC_BASE_URL` in
  `.service.d/*.conf` — the coordinator advertised a redirect_uri that no longer matched, with
  no error until a login was attempted. The script now writes the drop-in and prints the
  effective value. (2) DuckDNS defaults a new subdomain to the IP of whoever created it, not
  the server. (3) Oracle has a SECOND firewall in the cloud console; without ingress rules for
  80/443 the certificate request just times out. (4) Changing PUBLIC_BASE_URL invalidates every
  registered redirect URI — each provider's callback must be updated in the same breath.
- **Open:** `RELAY_HOST` stays a bare IP on purpose (raw TCP, no TLS name). The consent screen
  must be PUBLISHED, not left in Testing, or only hand-added test users can sign in (cap 100);
  NEURON asks only for openid/email/profile, all non-sensitive, so publishing needs no review.

### [P17] 🟢 Anyone could mint a funded wallet with no login — the whole ban system was decorative — fixed (2026-07-29)
- **`POST /wallet/faucet` was completely ungated**, unlike its two sibling endpoints
  (`/wallet/oauth`, `/wallet/{id}/violation`), which both verify `X-Wallet-Link-Secret`. And
  `models.claim_faucet` opens with `INSERT OR IGNORE INTO ledger … VALUES (?, 'wallet')` — it
  *creates* the row for whatever string it is handed. Chained with `api/openai_compat.py`'s
  `_auth()`, which accepted any non-empty bearer string as a wallet with zero validation:

      POST /wallet/faucet {"wallet_id":"abuse-1"}  -> funded wallet, no account, no cost
      use "abuse-1" as the API key                 -> full anonymous model access
      banned -> mint "abuse-2"                     -> unlimited, instant ban reset

  So every ban was one HTTP call away from being reset, and there was no real identity behind
  any request to act on. `SAFETY.md`'s "repeat violations escalate against your wallet
  identity" was, in practice, unenforceable.
- **Fix (three gates):** the faucet now requires the operator secret **and** a wallet already
  linked to a real Google/GitHub login; **`/infer` refuses any wallet with no login behind it**
  — the load-bearing one, since every driver must call `/infer` to get a node chain, so it is
  the one check a user cannot patch out of their own client; and the API verifies bearer keys
  against the coordinator (60 s TTL cache, so it is not a per-request round-trip).
- **Plus the operator lever that was missing:** bans previously only fired via the automatic
  `MODERATION_BAN_THRESHOLD`, which counts violations the **driver self-reports** — and for a
  self-hosted install the driver is the user's own machine, so a stripped client never reports
  itself. Added `/admin` (identity console: provider, provider-verified-email flag, violation
  and request counts, last seen, per-identity history, ban/unban) backed by secret-gated
  endpoints. 23 regression tests in `coordinator/test_identity_gate.py`.
- **Still true:** a determined abuser can make a fresh Google account (`SAFETY.md` says so
  honestly), and **content policy remains unenforceable on a driver NEURON does not run** —
  that user holds the plaintext and the moderation code. Against them the control is not
  content-based at all: they need a real account to get a chain, and that account can be banned.
- **Also added:** `requests`-table retention (`NEURON_REQUEST_RETENTION_DAYS`, 90d default,
  pruned on the health sweep). It is the only table that grows with *traffic* rather than users
  — ~1.25 GB/day at 1M users × 5 requests, which no single-file SQLite on the 1 GB coordinator
  VM survives. Identities, ledger rows and `moderation_events` are never pruned: bans depend on
  them. **Open:** SQLite on a 1-core/1 GB VM will hit write-concurrency limits in the low
  hundreds of concurrent users, well before disk — Postgres is the migration when that nears.

### [P16] 🟢 Auto-placement piled every new machine on the tail, capping network throughput at ONE request — fixed (2026-07-29)
- **Found by simulating the founder's question "what happens when 10 users hit send at once?"**
  `router.suggest_placement` advised a joining node to *"replicate the LAST segment"* once the
  chain was complete. With layers split 0-9 / 10-18 / 19-27, seven machines joining a 3-node
  network **all** landed on 19-27. Every request still funnelled through the single node holding
  0-9 and the single node holding 10-18 — and `agent/node_server.py`'s module-level
  `compute_lock` serialises each machine's forward pass, so those two ran strictly one request
  at a time. Simulated: **node_a served 10/10 concurrent requests, node_b 10/10.** Ten machines
  delivered one machine's throughput; the seven added zero.
- **Why it hid:** the policy's own comment claimed replicating the tail "adds throughput", and
  `[P8]`'s segment-level replication (S18) genuinely does work — `build_chain` picks among
  replicas correctly at *every* cursor position. The routing engine was never the problem; it
  was being handed a lopsided topology. Nothing tested the resulting layout, only that a single
  request could be routed.
- **Fix:** placement now replicates the **least-replicated stage**, ties breaking toward the
  earliest (every request traverses the front first, so a shortfall there throttles everything
  behind it). Same 10-machine simulation now spreads 4/3/3 across stages, busiest node 5/10
  instead of 10/10. Regression tests in `coordinator/test_placement.py` assert the layout is
  balanced and that many distinct parallel chains exist.
- **Still true after the fix (physics, not bugs):** concurrent users ≈ machines ÷ machines-per-
  chain, so ~3 machines per concurrent user at 28 layers — 100k concurrent needs ~300k machines.
  And `compute_lock` means **no batching**: a datacenter GPU amortises one forward pass across
  many users, this cannot, which is a genuine structural ceiling. Related: `[P13]` — latency,
  not concurrency, is the nearer blocker (a session is ~40 min wall-clock today).
- **Also corrected the same day:** `TOKENOMICS.md` §0/§8's "green AI / no new power draw" claim,
  which §11.5 already contradicted (*"the ~0.1 W green figure describes idle, not inference"*).
  Consumer-CPU inference costs *more* energy per token than a datacenter GPU; the defensible
  claims are no-new-hardware, sovereignty, and privacy-by-architecture.

### [P12] 🟢 The ledger minted NRN and the payout path was unauthenticated — both fixed

**Re-verified 2026-08-08 against the running code.** This entry described a system that no
longer exists; it was left saying the ledger mints and `/complete` has no auth long after both
were fixed. A stale security entry is its own hazard — it either causes work that is already
done, or gets ignored on the day it matters.

**Nothing is created from nothing any more.** `coordinator/ledger.py` settles by TRANSFER out
of escrow — `models.transfer(ESCROW_LEDGER_ID, node_id, share)` for each node and
`models.transfer(ESCROW_LEDGER_ID, COORDINATOR_LEDGER_ID, fee)` — so every payout moves NRN
that already existed. `models.credit()` still exists but has **no production caller**: the only
references left are test fixtures. `coordinator/test_escrow_conservation.py` is 40/40 including
`the supply invariant holds`, and the live ledger reads exactly 1,000,000,000.0.

**A user is now debited, not just the nodes credited** ([P5]'s half). `/infer` places a HOLD
before dispatch; `settle()` charges `min(actual, hold_amount)` — never more than was held — and
refunds the remainder to the payer. The entry's "no debit function exists anywhere" is obsolete.

**`/complete` is authenticated and settles from the coordinator's own plan.** Four properties,
all at `coordinator/main.py:700-715`:
  * `/infer` issues a per-request `complete_token`; `/complete` requires it (`compare_digest`,
    401 on mismatch), so a third party cannot forge a completion;
  * settlement uses `req["plan_node_ids"]` — **the chain the coordinator recorded** — never the
    caller-supplied `node_ids`, so a completion cannot pay an arbitrary node or an unchosen
    replica;
  * a second completion is refused 409, so it cannot be replayed;
  * `tokens_generated` is clamped to `max_tokens`.
  `coordinator/test_complete_auth.py` covers these.

**The duplicated price constant is gone.** `PRICE_PER_1K_WEIGHTED` lives once in
`coordinator/config.py` and `api/openai_compat.py` reads it from there. `NRN_PER_REQUEST`
survives as a dead constant whose only remaining mention is a comment naming the bug.

**What this means for a public repo, since it is the obvious question:** reading the source
grants no ability to move NRN. There is no HTTP route to `models.transfer` at all — its callers
are settlement, the fee, the refund and tests. Balances are moved by an operator over SSH.
Security here rests on credentials and signatures, not on nobody reading the code, which is the
only posture worth having.

**Residual, and it is operational rather than protocol:** the register secret is the one
credential that grants trusted standing AND overrides the payout-rebind check
(`payout.require_rebind_authority`), so it is the single path to redirecting another node's
earnings. Treat it as the crown jewel; rotate it if it is ever exposed. The economics rewrite
itself (TOKENOMICS.md §11 genesis buckets) is **done**, not deferred.

### [P13] 🟡 Prefill path is UNMEASURED — blocks token pricing AND may be a UX killer
- Per-chat-turn compute spans **3.2×** (28 vs 91 node-seconds) depending on whether prefill
  is batched or token-sequential; if sequential, TTFT on a 200-token prompt is ~152 s (fatal
  for chat) and any discounted input-token price undercharges real compute ~4×.
- Also a farming surface: under any work-metered subsidy, cheap batched prefill + expensive
  metering = prompt-stuffing exploit (see TOKENOMICS.md §11.4/§11.8).
- **Action: measure prefill (one selftest run with a long prompt, time the prefill pass)
  before publishing any per-token price sheet; charge input at full weight until then.**

### [P14] 🟢 Pre-launch audit (2026-07-28) — 4 real bugs found and fixed before the first real stranger
Three parallel audits (security / economics / reliability) against the current code, each
finding independently spot-verified by reading the actual source before acting on it.

- **RESOLVED — Relay had zero authentication on tunnel registration.** Any stranger who could
  reach the public relay control port (`150.230.22.250:8010`) could register `{node_id,
  public_port}` with **no credential at all**, and `relay.py`'s `self.controls[pub]`
  unconditionally overwrote whatever was already registered on that port — a stranger could
  hijack (traffic interception) or blackhole any NAT'd node's public port, or squat an
  unclaimed one. This is exactly the path a real behind-NAT stranger (`behind_nat: true` is
  the agent default) depends on. **Fix:** new `relay_auth.py` — the coordinator is the only
  party that knows the `node_id -> public_port` binding, so it mints an HMAC ticket
  (`HMAC(shared secret, node_id + ":" + public_port)`) at registration time and hands it to
  the node; the relay (still a DB-less, dependency-free stdlib process) independently
  recomputes the same HMAC to verify, no callback to the coordinator needed. Binding node_id
  + port together blocks replay onto a different port; without the secret an attacker can't
  mint a ticket for ANY node/port, closing squatting too. `coordinator/config.py`
  `RELAY_SECRET` (env `NEURON_RELAY_SECRET`) shared with `relay.py --secret`. Threaded through
  `tunnel_client.run_tunnel(ticket=...)` and `agent/agent.py`. Test: `test_relay_auth.py` 9/9
  + `coordinator/test_open_join.py` (ticket issued/verified on `behind_nat` register).
- **RESOLVED — Duplicate-payout race in `/infer/{id}/complete`.** The endpoint read
  `req["status"]=="completed"` as a pre-check, then called `ledger.distribute(plan)`
  **without checking** whether its own `models.complete_request()` call actually won the
  atomic `UPDATE ... WHERE status='pending'`. N concurrent completion calls with the SAME
  valid `complete_token` (trivial with `asyncio.gather` — anyone holding a token from their
  own `/infer` call, which open join lets any stranger make, could do this to themselves)
  would all pass the pre-check before any of them committed, and every one of them minted a
  fresh payout. **Fix:** `coordinator/main.py`'s `complete()` now gates `ledger.distribute()`
  on `complete_request()`'s own return value — only the caller that actually flips the row to
  `completed` gets paid; every racer after that gets a clean 409, not a second mint. Verified
  the fix by first reproducing the multi-payout with 8 concurrent racers against the
  pre-fix code (reliably failed), then confirming the fix closes it (reliably passes, run 3×).
  Test: `coordinator/test_complete_auth.py` (`8 concurrent racers -> exactly 1 paid`).
- **RESOLVED — Verified (non-trusted) nodes could be identity-hijacked on re-registration.**
  `POST /node/register`'s hijack guard only blocked a secret-less re-registration when
  `existing["trusted"]` was true — a node that open-joined and passed proof-of-compute
  (`standing="verified"`, `trusted=False` in the DB) was **unprotected**: anyone could
  re-register that exact `node_id` with no secret, the `ON CONFLICT` update would hand them
  a **fresh `node_token`**, and they'd inherit the victim's verified standing while silently
  locking the real owner out of their own dashboard/balance. **Fix:** the guard now checks
  `existing["standing"] in ("trusted", "verified")`; the one way around the secret is
  presenting that exact node's *current* `X-Node-Token` (proves you already control it —
  legitimate self-recovery, e.g. after losing local config, stays possible). Test:
  `coordinator/test_open_join.py` (`hijack of verified id blocked (409)`, wrong-token
  rejected, correct-token re-register succeeds).
- **RESOLVED — Model migration ([P-tiering], `coordinator/migration.py`) had no
  offline-eviction; one flaky node could wedge it forever, invisibly.** A node's plan was
  only recomputed when entering "preparing" fresh or the target model changed — never on
  later ticks. A planned node going offline mid-preparing (very plausible for a real
  stranger's laptop under the `idle` donation mode, which pauses on any owner activity)
  meant cutover required `planned <= ready` forever against a node that would never report
  ready again — silently stuck, since the dashboard didn't surface migration phase at all
  (only the raw `/network/migration` JSON did). **Fix:** `update()` now replans (against
  whoever is currently online+eligible) whenever a planned node drops out, not just on a
  target change; the dashboard now shows an in-progress migration inline. Test:
  `coordinator/test_migration.py` (`test_replan_when_a_planned_node_drops_offline`,
  `test_no_replan_when_nothing_changed` — a plain tick must NOT reset ready progress).
- **Found live while deploying this fix, also shipped — the relay's public control/data
  ports were ALREADY receiving malformed data from the open internet** (background
  scanners, most likely — not necessarily a targeted attack) that crashed handler threads:
  the journal on the live VM showed repeated `MemoryError` in `relay.py`'s `_recvn`, because
  an unvalidated 4-byte length prefix let garbage bytes be read as a ~4GB length and reach
  `sock.recv(huge_number)` — a real resource-exhaustion risk on a 1GB-RAM VM, not just log
  noise. **Fix:** `recv_json` now caps the declared length at `MAX_MSG_BYTES` (64KB, generous
  for a handshake message) and catches decode/parse errors, treating any of it as "not a
  real client" (clean `None`, connection dropped) rather than an unhandled exception. Test:
  `test_relay_auth.py` (huge length / undecodable bytes / malformed JSON all handled; a real
  message still round-trips).
- **Cheap fix, also shipped — mid-request node drops failed cleanly but logged nothing
  server-side** (`ui/app.py`'s SSE error branch). Now logged via `logging.getLogger
  ("neuron.ui")` so a real stranger's dropped connection shows up in the server log instead
  of only ever being visible if they report it themselves.
- **Accepted, not fixed now — zero monitoring/alerting beyond stdout/systemd logs, and no
  database backup mechanism** beyond the one manual pre-deploy snapshot noted in
  `neuron-machines` history. `Restart=always` in the systemd unit covers process crashes;
  SQLite corruption/data loss has no recovery path. Reasonable to leave for a single-friend
  test; revisit before any wider rollout.
- **Accepted, inherent to open-join — Sybil replica-stacking.** `router.build_chain` picks
  uniformly among nodes tied for a segment; nothing stops one machine registering many
  `node_id`s to capture a disproportionate share of the random draws. Proof-of-compute
  verifies correctness, not identity uniqueness. A fundamental open-join tradeoff, not a bug
  — no hardware fingerprinting exists to fix it, and it doesn't inflate total mint (see [P12]).
- **Confirmed solid, no action needed:** `/complete` still correctly settles from the
  coordinator-recorded plan (never caller-supplied `node_ids`) — [P12]'s fix holds up under
  concurrency too; SQL is parameterized everywhere (no attacker-influenced column names);
  node tokens are never logged; no CORS/debug/reload misconfiguration; `/node/list` and
  per-node dashboards stay correctly privacy-gated ([P11] holds); `tunnel_client.py`'s
  earlier timeout-churn bug stays fixed; reputation/proof-of-compute is manual-only today so
  a flapping stranger node cannot get auto-flagged just for churning.

---

### [P15] 🟢 macOS idle-detection silently always-idle — fixed (2026-07-28)
The founder's friend (the planned first real stranger) has a Mac. `INSTALL.md` lists macOS as
supported for the source-install path, but `agent/resource_guard.py`'s idle detection only
handled Windows (`GetLastInputInfo`) and Linux (`xprintidle`) — on macOS neither branch
matched, so it silently fell through to the headless-server default (`return 1e9`, "always
idle"). Under the default `idle` donation mode (which is supposed to yield the moment the
owner touches the machine), a Mac node would never actually detect real user activity and
would keep donating compute while the owner was using it — defeating the mode's whole point.
Not a crash, a silent correctness gap, found by directly re-checking the code rather than
assuming "macOS: supported" in `INSTALL.md` was accurate. **Fix:** `_macos_idle_seconds()` —
pure `ctypes` call to `CGEventSourceSecondsSinceLastEventType` (CoreGraphics), no `pyobjc`
dependency, matching the module's existing minimal-deps style; needs no special OS permission.
Wired into `seconds_since_input()`'s dispatch, fails safe to always-idle if the framework
somehow isn't loadable. Never tested on real macOS hardware (this dev environment is
Windows) — verified by flipping the platform-detection flags and mocking the CoreGraphics
call (`agent/test_resource_guard.py`, 2 new cases, 18/18 total). **Still open:** no macOS
packaging exists at all (no `.dmg`/PyInstaller build — only Windows has a real installer);
the friend would need to run the source-install path (`INSTALL.md`).

**Update (2026-07-28, same day) — macOS `launchd` auto-start added.** `agent/install.py`'s
optional `--startup` helper only handled Windows (registry) / Linux (`systemd --user`) —
found by the same "verify, don't trust the platform list" check: on macOS it took the Linux
branch and would have crashed outright (`FileNotFoundError: systemctl`, no systemd on
macOS). Added `add_to_startup_macos()` — writes a per-user LaunchAgent plist
(`~/Library/LaunchAgents/com.neuron.agent.plist`, `RunAtLoad`+`KeepAlive`) and loads it via
`launchctl load -w`; `agent/uninstall.py` mirrors it (`launchctl unload -w` to stop, then
removes the plist file). `start_background()`'s macOS path spawns `agent.py` directly
(like Windows) rather than assuming a LaunchAgent already exists, so `--no-startup` still
works. New `agent/test_install_macos.py` (13/13, mocks `launchctl`/file paths — no real Mac
available to test against). This is still the OPTIONAL auto-start helper, not the base
`INSTALL.md` flow (`python agent/agent.py` run directly) — the base flow already worked on
macOS without this.

---

## Not-doing (deliberately, for now)
- Chasing single-user latency parity with GPU clouds — unwinnable, wrong hill.
- Productionising quantization before there are real users (measure first, integrate later).
- On-chain NRN before Session 12 (first stranger node) — per ROADMAP.
