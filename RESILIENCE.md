# NEURON — Resilience: a request must never die because a node did

**Status:** design + gap register. Layer 0 is the current work; Layers 1–4 are planned.
**Owner problem:** a user waiting on an answer must not be handed a blank screen because a
volunteer closed their laptop. On a network built from other people's spare machines, node
churn is the normal case, not the exception.

This file is the single place that records what already exists, what is missing, and in what
order it gets fixed. Read it before touching `neuron_driver.py`, `junction_cache.py`, or the
agent's shutdown path.

---

## 1. What already exists — read this before designing anything

**The request already survives a node dropping, and the recovery is exact.**

`junction_cache.py` + `_reroute()` in `neuron_driver.py`:

- The driver caches every activation it sends into the chain, in fp16.
- NEURON's pipeline shape means there is exactly **one** junction to cache — the driver → chain
  boundary. Replaying that one sequence rebuilds *every* downstream node's KV cache, because
  each activation flows through all of them again.
- On a dead peer the driver closes the socket, settles billing for the dead chain, asks the
  coordinator for a fresh chain, replays the whole history as **one concatenated block**
  (identical to replaying token by token under causal attention, but one round trip instead of
  N), and resumes.
- Recovery is **bit-exact, not approximate**: fp16 replay was measured token-identical to fp32
  (max Δlogit 0.0069, `bench_wire.py`).
- Cost is about one prefill of the tokens so far.
- Ceiling: 256 MB (~87k tokens at H=1536). On overflow it *stops caching* and
  `recoverable()` goes False, so the driver fails honestly rather than replaying a truncated
  history and producing quiet nonsense.
- `MAX_REROUTES = 3` (env `NEURON_MAX_REROUTES`).

**Design consequence, and it is the important one:** any scheme that *predicts* or
*approximates* the missing activations is competing against an exact mechanism that already
ships. Replay wins on quality by construction. Approximation belongs only where replay is
impossible — never as the first response to a timeout. If a proposal starts with "predict the
missing state", check first whether replay already covers it.

---

## 2. Gap register — where a request can still die

Numbered so they can be referenced across sessions. Status updated in place.

### [R1] The recovery path had never been proved end to end — 🟢 CLOSED (Session 52)
`test_junction_cache.py` says so itself: it is model-free and tests the recovery *bookkeeping*.
Replaying into a genuinely fresh chain had never been executed anywhere in the repo.

**Closed by `test_node_death.py`.** Three real processes with real Qwen2.5-1.5B slices, the real
wire protocol over real TCP, the real driver and its junction cache, and a real
`TerminateProcess` on the middle node six tokens into a generation. Result:

```
[2] middle node SIGKILLed after 6 tokens
  PASS  node_c process is actually dead
  PASS  request survived the death
  PASS  a reroute actually happened
  PASS  recovery is EXACT, not approximate
  PASS  reroute is reported to the caller
```

The load-bearing assertion is the fourth: the killed run is **token-identical** to an
uninterrupted baseline, not merely "completed". A request that finishes with different text has
not recovered, it has degraded silently — which is the failure this whole file exists to
prevent. Recovery cost no measurable wall-clock (16.7 s killed vs 19.7 s baseline, 24 tokens).

Only the coordinator's chain handout and billing settlement are stubbed; those have their own
tests. Everything recovery depends on — cached activations, the replay block, the replacement's
rebuilt KV cache — is real.

### [R2] There may be nowhere to reroute *to* — 🔴 HIGH
`_reroute` asks the coordinator for a fresh chain. With three nodes and no replica covering the
middle segment, there is no replacement — recovery fails for lack of spare capacity, not
because the mechanism is wrong.

The router is already replica-aware (`coordinator/router.py`, `coordinator/test_replica.py`
9/9). What is missing is spare machines. **This gap closes by getting more nodes, not by
writing code** — which makes it another reason the first-stranger milestone is the binding
constraint on everything.

### [R3] The driver is a single point of failure — 🟡 MEDIUM
The junction cache lives in the driver's memory. If the driver dies, every in-flight request it
holds dies with it, and nothing can recover them.

Severity depends on who is running the driver:
- **Self-hosted driver** (user runs `neuron_driver.py` on their own machine): low. If their
  machine died, their session died anyway. This is not a network failure.
- **Server-side driver** (`ui/app.py` and the `/v1/*` API run the driver for the user): real.
  One process death drops every concurrent request.

**Fix:** not a replay problem. Either accept it for self-hosted, or give the server-side driver
a persistence/handoff story. Deliberately deferred until [R1] and [R2] are closed.

### [R4] Reroute budget is a fixed count, not a time budget — 🟡 MEDIUM
`MAX_REROUTES = 3`. A fourth failure in one request kills it regardless of how long the request
has been running or how cheap the retries were. On a churning volunteer network a long answer
can plausibly outlive three replacements.

**Fix:** make the budget time- and progress-aware (e.g. allow another reroute whenever tokens
have advanced since the last one), so a request that is making progress is not killed by an
arbitrary counter.

### [R5] Planned withdrawal is treated as a crash — 🟡 MEDIUM
When a volunteer reclaims their machine — resource guard throttling, OS shutdown, agent
update/restart, laptop lid — **the node knows in advance**. Today it simply stops, the driver
sees a dead socket, and pays the full replay cost as though it were a power cut.

This is the largest *avoidable* cost in the system, because a node that knows it is leaving can
say so, and a driver that is told can move at a token boundary for free.

**Fix:** Layer 1 below.

---

### [R6] Losing a node leaves a coverage hole nothing closes automatically — 🟢 CLOSED (2026-08-07)
Observed live, 2026-08-04: the network sat at **21/28 layers, DEGRADED — chain incomplete**,
with two healthy nodes online. No request could complete; every chat attempt failed. The cause
was not capacity. Two nodes cover a 1.5B model comfortably (~6 GB total, and one of them has
68 GB). The cause was that when the third machine left the chain **the survivors kept their old
three-way assignments** — 0–13 and 14–20 — so layers 21–27 belonged to nobody.

The existing self-heal (`main.py`, `[gap-heal]`) cannot fix this: it closes gaps by reassigning
**true-idle surplus** nodes, and a shrunken network has no surplus by definition. Exactly the
case where healing is most needed is the case it skips.

`POST /network/rebalance` (admin, `require_register_secret`) does re-split across the nodes that
are actually online, so the recovery exists — it just has to be triggered by a human who has
first noticed. Nothing noticed. The failure surfaced as a user typing "hi" eight times.

**Fix:** treat a coverage gap with zero idle surplus as a trigger for a full re-split of the
online nodes, not a no-op. Related to [R2] but distinct: [R2] is "no spare machine to take
over", this is "the machines present are not re-divided to cover what is missing".

**Done, 2026-08-07** (`coordinator/migration.py`). `self_heal` still prefers idle surplus — a
working segment is never stripped to patch a broken one — but when nothing is idle it falls
through to `plan_migration` over every online+eligible node, re-splitting the whole serving
model. Reassigning nodes that *are* covering is sound only because the gap already means no
request can complete; the guard is that this path runs only when `missing` is non-empty. It goes
through the same download-then-report-ready handshake as any other heal, so nothing cuts over
until every planned node holds its new slice, and `/network/gap-heal` now reports `mode`
(`"surplus"` vs `"resplit"`) so the log distinguishes "a newcomer landed" from "the network
shrank". The same sweep discovered the placement half of the story — see [P25] in `PROBLEMS.md`,
where three machines were handed the identical layer range and 21–27 went unclaimed.

**The delivery half.** This fix spent its first days written, tested and not running, because
reaching production needed the founder at their machine — a network that heals itself but cannot
receive the code that heals it is only half autonomous. `coordinator/selfupdate.py` closes that:
the VM installs a *published* version on an hourly timer, proves it healthy (serving, reporting
the version just installed, auth gates still returning 401) and restores the previous build if it
is not. Publishing stays an explicit act, because with one coordinator and no redundancy an
auto-pull from git would put every bad commit straight into production. See `coordinator/DEPLOY.md`
§0b.

**Cross-cutting:** this is also the strongest argument for the founder's own suggestion of a
health-check script shipped with the installer — layers covered, every node answering,
coordinator reachable, wallet loads. A network that is silently unable to serve is worse than
one that is visibly down.

## 3. The layered plan

Ordered by dependency, not ambition. Each layer is useful alone and none of them requires the
next one to exist.

### Layer 0 — Prove the net (closes [R1]) — ✅ DONE, Session 52
`test_node_death.py`, 7/7. See [R1] for the result. Nothing else was built until this passed,
and the same bar applies to Layers 1 and 3: each gets a kill/withdrawal test that asserts
token-identical output before it is considered done.

### Layer 1 — Graceful withdrawal (closes [R5])
The node announces its departure instead of vanishing.

**Announce, do not transfer.** The obvious design is for the leaving node to hand its KV state
to a replacement. Do not do this: the driver can already rebuild any replacement from its own
junction cache, so a node-to-node state transfer adds a protocol, bandwidth, and — worse — a
trust surface, because a malicious node could hand its successor poisoned state. Announcing is
strictly simpler and reuses machinery that will be proven by Layer 0.

Triggers: `resource_guard` about to throttle, OS shutdown signal, agent update, tray quit,
donation-mode change that reduces the CPU cap.

Protocol sketch: node sends `{"type": "leaving", "after_token": N}` → driver finishes the
current token, reroutes at the boundary, resumes. No mid-token surgery, no state on the wire.

### Layer 2 — Reactive replay (exists; hardening only)
Already built (§1). Work needed: the [R4] reroute budget, and [R2] spare capacity, which is a
node-count problem rather than a code problem.

### Layer 3 — Conditional pre-warm
Warm a standby chain *before* the failure, so recovery is a switch rather than a rebuild.

**Conditional, not always-on.** Continuously mirroring every request doubles wire traffic and
node compute, and on a network where nodes earn per token that is a real cost paid on every
request to insure against a rare one. Trigger pre-warm only on a degrading health signal:
missed heartbeats, battery unplugged, load spike, resource guard approaching its cap, donation
mode about to change.

Worth noting explicitly: **this is a prediction problem that will actually work.** Node-level
telemetry is genuinely predictive of node departure in a way that the contents of the tensors
are not. Predicting *which machine is about to leave* is tractable; predicting *what it would
have computed* is not.

### Layer 4 — Honest degradation, in this order
When replay is impossible (cache overflowed, or no spare node exists):

1. **Stall.** Hold the stream and keep trying. For a streaming UI a two-second pause is close to
   invisible.
2. **Degraded continuation.** Only as a last resort before failing, and only if it can be
   flagged.
3. **Fail with a clear message**, and never bill for what was not delivered.

**The governing principle: stall beats degrade.** A pause is recoverable and the user forgives
it. Wrong tokens are permanent, the user cannot tell they are wrong, and they are exactly the
"damage to NEURON" this whole file exists to prevent. Any fallback that silently lowers quality
must be visible in the response metadata, never silent.

---

## 4. How the three proposed directions map on

| proposal | verdict | change |
|---|---|---|
| **D1 — Farewell Protocol** | **Adopt**, as Layer 1 | Announce, don't transfer state (§3 Layer 1). The "covers 97% of failures" figure is an assumption, not a measurement — see below. |
| **D2 — Preemptive reassembly** | **Adopt in modified form**, as Layer 3 | Make pre-warm *conditional* on a health signal rather than continuous, or every request pays for insurance against a rare event. "Switch at a token boundary, not mid-token" is exactly right and applies to Layer 1 too. |
| **D3 — Hard-failure floor** | **Partly already built; one part rejected** | "Replay from last good token, not from zero" is precisely what `junction_cache` does today — this is done, it just needs proving ([R1]). **Rejected: the coordinator keeping the token stream.** |

**Why the coordinator must not keep the token stream.** D3 proposes the coordinator store tokens
rather than activations. Tokens are far smaller, and it would survive driver death ([R3]) — both
true. But the token stream *is the user's prompt and the model's answer in plain text*. Storing
it centrally would put the full content of every conversation on the coordinator, which
contradicts the project's stated position that no personal data is collected and that the claim
is provable by reading the code. That is a much larger cost than the availability it buys.

If [R3] is to be solved, it must be solved without giving the coordinator the plaintext — e.g.
driver-side persistence, or a client-held resume token. Recorded here so the idea is not
re-proposed without the objection attached.

**On "97% of failures are graceful":** plausible, and currently unmeasured. Laptop suspend and
user shutdown are graceful; wifi changes, ISP drops, OOM kills, antivirus process kills and
power cuts are not. The split is worth *measuring* once there are real volunteer nodes — log the
cause of every chain failure and count. Until then the number should not be used to justify
skipping Layer 2, which handles the ungraceful remainder whatever its true size.

---

## 5. What is deliberately not being done

- **No prediction of missing activations.** Replay is exact; approximation can only lose to it.
  See §1.
- **No node-to-node state transfer.** See Layer 1.
- **No always-on mirrored chains.** See Layer 3.
- **No plaintext token storage on the coordinator.** See §4.

---

*Cross-references: `[P4]` (node availability / laptop suspend) and `[P10]` in `PROBLEMS.md`;
`junction_cache.py`'s module docstring for the mechanism; `FIRST_STRANGER.md` for why [R2]
depends on node count.*
