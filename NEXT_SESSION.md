# Handoff — start of Session 72

## STATE AT CLOSE, 2026-08-21 (late) — READ THE FALLBACK LINE FIRST

**Fallback: tag `good-state-2026-08-21-final`** (pushed). Also `good-state-0.20.21`, and the
installer `dist/installer/NEURON-Setup-0.20.21.exe`.

```
this PC   agent-optinovate-6ff49d        0-9     driver + stage 1   0.20.21 installed
OptiPlex  agent-optiplex-server-ce473b   0-9     stage-1 replica    claimed today
Pavilion  agent-raman-...-e4920b         10-27   last stage
coordinator  AGENT_VERSION 0.20.17 published; ledger.py DEPLOYED with the owner-forward
wallet  847.59 NRN · 5 machines · 0.00 unswept
```

**RESOLVED 2026-08-21: 0.20.22 was built and installed, so the installer now carries all three drop-ins. The paragraph below is kept for the reasoning.** ~~THE INSTALLED APP DIFFERS FROM ITS INSTALLER.~~ Three files were dropped straight into
`%LOCALAPPDATA%\Programs\NEURON\_internal\ui\static\` rather than rebuilt, because a full
build is ~20 minutes and these were copy/asset fixes:

  * `workspace/assets/index-CeOXSjCm.js` + `workspace/index.html` (wallet copy, the signing link)
  * `sign_payout.html`
  * `chat.html`

**Reinstalling NEURON-Setup-0.20.21.exe reverts all three.** The repo carries the same files, so
the NEXT REAL BUILD makes them permanent — and should be cut before anything is published.
Rollback for the bundle alone: `scratchpad/workspace-rollback-0.20.21.tgz`.

## The money is fixed, and this is the important one

**Node earnings now go to the owner's wallet automatically. No more sweeping.**

[P39] recorded ownership and left the money on the machine. `emission.py` routed to the owner
(`get_node_owner(node_id) or node_id`); `ledger.settle` paid `ESCROW -> node` and stopped. So
availability emission was spendable and REQUEST earnings were not — they piled up on an account
nobody can sign into, drained only by running `claim_node_earnings.py` by hand, forever.

`ledger.py` now forwards each share to the owner after paying the node. Paid to the node FIRST
so `requests_served`/`total_earned` still show the machine's work; a failed forward leaves the
share on the node (today's behaviour, recoverable by the sweep) and can never fail a settlement.
**Deployed and verified with a real request:** unswept stayed 0.00, node `total_earned` rose,
node balance stayed 0.00. `coordinator/test_earnings_reach_the_owner.py`, 11 assertions.

**Swept 98.99 NRN** across five machines first (wallet 748.60 -> 847.59), invariant OK at every
step, logged to `claim_log.json` on the VM. The sweep tool is now a REPAIR for unowned nodes,
not routine maintenance.

## Claiming, and the trap that hid a machine

The OptiPlex was **never claimed** and had **28.87 NRN** belonging to nobody. "My machines" lists
only nodes whose `owner_wallet_id` is your wallet, so an unclaimed machine simply does not
appear — absence again.

Claiming was a BROWSER action and a headless node (`local_chat: false`) has no page to click.
Now: `payout_key.claim_for_owner` is the rule with no browser in it, `agent.py --claim WALLET_ID`
drives it, and an unclaimed node logs a WARNING on every registration naming the fix. It stays
silent when it merely could not ask.

## Wallets: any EIP-1193 extension works

`ui/static/sign_payout.html` uses `window.ethereum` + `eth_requestAccounts` + `personal_sign` —
the standard interface. **MetaMask, Rabby, Coinbase Wallet, Brave Wallet, Trust Wallet** all
work. It is provider-agnostic: no `isMetaMask` check anywhere. With several extensions installed
the browser decides which one owns `window.ethereum`, which is the usual multi-wallet ambiguity
and not something this page controls.

It is served at **`/static/sign_payout.html`** now. It used to be `tools/sign_payout.html` — a
source path that shipped nowhere and 404'd, while THREE UIs told people to use it.

## Still open

  * **A real build is owed.** Three drop-ins live only in the install. Build before publishing.
  * **[P56] part three** — declared precision. Needs MEASURED tolerances per dtype.
  * **[P60] residual** — a node learns a new range only when it next registers ([P37]).
  * **[P61] — the installer is unsigned and F-Secure quarantines it.** The founder REMOVED
    F-Secure to install; suggest putting it back with an exclusion. Code signing deferred by
    decision; no plumbing was wired, so nothing is half-built.
  * **Node id flip** between `-6ff49d` and `-7fc2ff`, plus stale `node-c-pavilion`. 5 registered,
    3 real.
  * **"3 nodes online for bigger models"** while one is a replica — wording, fix on next build.
  * **[P30] phase 2** — imported by nothing.

## Traps this session paid for

  * **`cd` persists between commands in one bash call.** A commit meant for NEURON landed in
    Trust chat and swept in `skills.ts`/`docx.ts`, which must stay uncommitted. Use `git -C`.
  * **Deleting a function signature can still PARSE** — the orphaned body attaches to the
    function above. Import success proves nothing; run the tests.
  * **Truncated debug output invents bugs.** `healthy` and `total_earned` both existed and were
    cut off by a key limit.
  * **`NEURON_ONLY=1` AND `NEURON_BASE=/workspace/` AND `MSYS_NO_PATHCONV=1`** — all three, every
    workspace build. `ui/test_workspace_bundle_is_neuron_only.py` checks the artefact.
  * A source-text test pins the code it was written against; four broke on refactors today.

## Facts that save time

  * Full suite **122/122**. No pytest: `.venv\Scripts\python.exe -m <module>`.
    `packaging/` has no `__init__.py` — run its test as a FILE.
  * Rebuild the workspace UI:
    `cd "C:\Users\optin\Trust chat" && MSYS_NO_PATHCONV=1 NEURON_ONLY=1 NEURON_BASE=/workspace/ npx vite build`
    then REPLACE `ui/static/workspace/assets/` (never merge).
  * Attest before trusting a chat result after any placement change or restart.
  * VM: `SSH_AUTH_SOCK=/c/Users/optin/.ssh/neuron-agent.sock ssh-add /c/Users/optin/.ssh/oracle_coordinator`
    then `bash coordinator/deploy.sh`. **Deploying code is not publishing** — the live version is
    governed by `/etc/systemd/system/neuron-coordinator.service.d/zz-agent-release.conf`.

---

# Handoff — start of Session 71

## STATE AT CLOSE, 2026-08-21 — 0.20.21 INSTALLED AND VERIFIED

**Fallback point: tag `good-state-0.20.21`** (pushed), installer at
`dist/installer/NEURON-Setup-0.20.21.exe`, bundle source tagged `neuron-good-0.20.21` in
`C:\Users\optin\Trust chat`. `git reset --hard good-state-0.20.21` returns here.

```
this PC   agent-optinovate-6ff49d        0-9     driver + stage 1   0.20.21 installed
OptiPlex  agent-optiplex-server-ce473b   0-9     REPLICA of stage 1 0.20.17
Pavilion  agent-raman-...-e4920b         10-27   last stage         0.20.17
coordinator AGENT_VERSION 0.20.17 (published, SHA verified)
```

Network routable and healthy. **A request walks 2 stages, so the footer says "2 nodes" while
the bar says "3 nodes online" — both correct.** The OptiPlex holds the same range as this PC,
so it is redundancy and throughput, not depth, and it never appears in a chain.

**Founder's decision, 2026-08-21: leave it that way.** Measured on these machines: 2 stages is
2.2-3.0 tok/s, 3 stages is 1.84-1.9. Splitting layers does not reduce compute, it adds a hop.
To put all three in the chain (slower per answer, and the only way to serve a model two
machines cannot hold):

```
bash coordinator/pin_layers.sh --driver agent-optinovate-6ff49d
```

**0.20.21 is NOT released.** No tag on GitHub, `AGENT_VERSION` still 0.20.17. The installer is
unsigned and antivirus-flagged ([P61]), which is the reason to think before publishing.

## What this session fixed, after 0.20.18

  * **[P62]** — a web-search citation blanked the whole workspace. `rag/retriever.py` has always
    sent `{title, href}`; the client typed it `string[]` and rendered each entry directly, so an
    object reached React as a child and the render unwound the ENTIRE tree. Fired for anyone who
    ticked "Web search", and persisted, because the object form is written to localStorage.
    Fixed at the render site (so threads already on disk work) plus a **per-message error
    boundary** — one bad message must never take out the app.
  * **[P63]** — **I shipped the wrong product twice.** `ui/static/workspace/` is compiled from
    another repo and needs `NEURON_ONLY=1` as well as `NEURON_BASE=/workspace/`. I passed only
    the second, so `{__NEURON_ONLY__ && <NeuronBar />}` compiled away and the wallet, balance,
    sign-in link and server-conversation import vanished — in 0.20.19 AND 0.20.20 — while the
    app still worked perfectly. `ui/test_workspace_bundle_is_neuron_only.py` now checks the
    artefact by what it contains; all 11 assertions would have failed on what I shipped.
  * **A deleted chat came back on refresh.** The client never called `DELETE /conversations/{id}`,
    so `fetchServerThreads` re-imported it on the next load. It calls it now.
  * **Signing out hid the sign-in link.** `fetchSession` collapsed "could not ask" into "not
    signed in", and the bar hid on both — removing the only way back in. [P24]'s rule applied to
    a session.

## Traps this session paid for, worth not paying again

  * **`NEURON_BASE=/workspace/` under Git Bash becomes `/Program Files/Git/workspace/`** — MSYS
    path translation. Use `MSYS_NO_PATHCONV=1`, and read the emitted `index.html` afterwards.
  * **Rendering in a clean browser profile proves nothing** about a browser holding real data.
    Two blank-page diagnoses were wrong because of this. Ask for the CONSOLE error first.
  * **Truncated debug output nearly produced two false bug reports** (`healthy`, `total_earned`
    both existed and were cut off by a key limit). Print the whole thing before concluding.
  * **A source-text test pins the code it was written against.** Three suites broke on the
    [P56] constructor refactor — `test_driver_s1`, `test_lan_direct`,
    `test_last_stage_is_not_a_probe`. All updated to pin the property, not the literal.
  * `packaging/` has no `__init__.py`, so its test runs as a FILE, not `-m`.

## Still open

  * **[P56] part three** — declared precision. Needs a MEASURED tolerance per dtype and a
    declaration that costs something. Urgent the day [P30] lands.
  * **[P60] residual** — a node still learns a new range only when it next registers ([P37]).
    Now a reroute rather than wrong answers, but the window remains.
  * **[P61] — the installer is unsigned and F-Secure quarantines it.** The founder removed
    F-Secure to install; **suggest putting it back with an exclusion.** Code signing was offered
    and deferred; the plumbing was deliberately NOT wired, so nothing is half-built.
  * **The node id keeps flipping** between `agent-optinovate-6ff49d` and `-7fc2ff`, and
    `node-c-pavilion` is a second stale identity for the Pavilion. Five registered, three real.
  * **The bar says "3 nodes online for bigger models"** while one of them is a replica, which
    does not help with bigger models. Wording, not a defect — fix on the next bundle build.
  * **[P30] phase 2** — still imported by nothing.

## Facts that save time

  * Full suite: **121 modules, all clean.** No pytest —
    `C:\Users\optin\neuron\.venv\Scripts\python.exe -m <module>`.
  * Rebuilding the workspace UI:
    `cd "C:\Users\optin\Trust chat" && MSYS_NO_PATHCONV=1 NEURON_ONLY=1 NEURON_BASE=/workspace/ npx vite build`
    then copy `dist/index.html` + `dist/assets/*` into `ui/static/workspace/`, **replacing** the
    assets directory rather than merging it.
  * Attest before trusting a chat result after any placement change or restart. Speed is not
    evidence — [P60]'s garbage ran at the fastest tok/s this network has produced.

---

# Handoff — start of Session 70

## STATE AT CLOSE, 2026-08-21 — 0.20.18 INSTALLED HERE, 0.20.17 IS WHAT THE FLEET IS TOLD

```
this PC       0.20.18   installed, running, serving the UI
Pavilion      0.20.17   node
OptiPlex      0.20.17   node
coordinator   AGENT_VERSION 0.20.17  (published, with a matching SHA)
```

That is a legal, normal state — `test_version_lockstep` allows the coordinator to lag a built
version so a build can exist before anyone is told to install it. **0.20.18 is built and
installed but NOT released**: there is no v0.20.18 tag, no release notes, and `AGENT_VERSION`
was deliberately left at 0.20.17. Publishing it is a decision, and the antivirus finding below
is the reason to think about it first.

Network: 3 nodes, `[[0,9],[10,27]]`, routable, healthy. Attested clean (`max_err` 0.000217)
and answering — *"On a clear day, the sky is typically blue."*

## [P61] — the blank workspace page, and the antivirus underneath it

**Two bugs, both fixed and both verified from the installed build.** `/workspace/index.html` was
served with no `Cache-Control` at all, and `[InstallDelete]` purged `static/app/assets` but not
`static/workspace/assets`. Together, a cached `index.html` naming an older hashed bundle found
that bundle still on disk and loaded it — a silent, working, two-builds-old app. Now: the page
returns `no-store, must-revalidate`, hashed assets stay cacheable, and a stale bundle 404s.

**The larger finding is that F-Secure quarantines the installer** —
`Drop.Win32.Startup.11003`, and it removed `neuron-agent.exe` from the install directory. **This
also corrects an entry written earlier the same day**, which blamed Inno's child-process timing
for a vanishing exe; the event log shows the antivirus firing at 01:26 and 01:28 with the same
detection. A wrong cause in a handoff is worse than no cause.

The founder removed F-Secure to complete the install. **Suggest putting it back with an
exclusion for the NEURON install directory rather than leaving the machine without it.**

**Code signing was discussed and DEFERRED (founder, 2026-08-21).** The plumbing was NOT wired
into `neuron.iss` — it was offered and declined, so there is nothing half-built to trip over.
When it is wanted: an Authenticode certificate (Azure Trusted Signing is the cheapest and needs
no hardware token; Certum has an open-source tier; EV is the only one that grants SmartScreen
reputation immediately), then sign the PyInstaller exe and let Inno sign its own output, always
with `/tr` timestamping. Signing is not an AV bypass — `Drop.Win32.Startup.*` is behavioural —
so submit the false positive to F-Secure regardless; that costs nothing.

**Why this is a roadmap item and not a chore:** ROADMAP's One Rule is the first stranger. This
fired on the founder's own machine, where he knew it was safe and had the repo to check against.
A stranger gets a scary warning, no context, and no way to tell a false positive from a real
one — and the rational move for them is to delete it. **The installer is currently a wall in
front of the one metric the project is measured on.**

---

## [P60] IS FIXED, SHIPPED AND PUBLISHED — 0.20.17 (2026-08-21)

**The fix is one missing arm.** `node_server.serve` chose its role in three branches and the
last was a catch-all. A node whose range excludes the model's final layer was assumed to be
talking to a verifier; it can equally be a node the coordinator believes is last while the node
knows it is not. Real traffic fell into the probe arm, ran the node's own layers, skipped the
final norm, and the driver ran `lm_head` on it.

There is now a fourth arm: **real last-stage traffic to a node that is not the last stage is
refused with `range_mismatch`.** Refusing costs nothing — the driver already turns a refused
config into a reroute, and proof-of-compute already reads `range_mismatch` as stale placement
rather than as a failed challenge.

**Proven live, before and after.** Before, the OptiPlex — holding 10-18 — was asked for a last
stage and replied `ok: True`. After:

```
ok=False  error='range_mismatch'  holds=[10, 18]
asked to serve the LAST stage from layer 10 to 27, but this node holds 10-18 and does not
hold the model's final layer. Placement here is stale -- re-register or re-place this node
```

`agent/test_stale_placement_is_refused_not_answered.py`, 13 assertions, including that nothing
which worked before is refused now.

**Published as 0.20.17.** `/agent/version` returns it, and the SHA the coordinator requires
(`38acc6a9…ff7d`) matches the published asset — verified by downloading it back. All three nodes
report 0.20.17.

**Current topology** (2 stages, the faster shape):

```
agent-optinovate-6ff49d          0-9     this PC, driver + stage 1
agent-optiplex-server-ce473b     0-9     OptiPlex, stage-1 REPLICA (throughput, not depth)
agent-raman-...-e4920b           10-27   Pavilion, last stage
```

Attested clean (`max_err` 0.000217) and answering correctly at ~1.9 tok/s.

**What is STILL open under [P60]:** the notification itself. A node learns its new range only
when it next registers ([P37]). This turns that window from *wrong answers* into *reroutes* —
the difference between a bug and an outage — but the window remains. Closing it means the node
re-reading `slice-info` on a heartbeat.

**The operational rule this earned, and it is not optional:** after any placement change or
restart, **attest before trusting a chat result.** Speed is not evidence — the garbage ran at
the fastest tok/s this network has ever produced.

---


## STATE, 2026-08-21 — 0.20.16 IS PUBLISHED AND TWO THINGS WAIT ON YOU

**NEURON is on GitHub.** 115 commits pushed to `main-full`, release cut at
[v0.20.16](https://github.com/neuron-network-ai/neuron/releases/tag/v0.20.16) with the
installer attached, `AGENT_VERSION` moved 0.20.3 → 0.20.16 and `AGENT_SHA256` set to
`bf8589376027503051d4aec6f7bae7dbab36b06a3d2a354603c895f4997a42e7` — verified by downloading
the published asset back and hashing it, so a node's integrity check passes rather than
refusing the file.

**Network:** 3 nodes, `[[0,9],[10,18],[19,27]]`, routable, healthy, ~1.84 tok/s, correct
answers. All three report **0.20.16**.

```
agent-optinovate-6ff49d                          0-9     this PC, driver + stage 1
agent-optiplex-server-ce473b                     10-18   OptiPlex, MIDDLE relay
agent-raman-hp-pavilion-laptop-15-eh3xxx-e4920b  19-27   Pavilion, last stage
```

## DEPLOYED — 2026-08-21. Publishing is complete and the [P59] fix is live.

The founder loaded the key; the coordinator was deployed and verified. What a node now sees
from `/agent/version`:

```
{"version": "0.20.16",
 "download_url": ".../releases/download/v0.20.16/NEURON-Setup-0.20.16.exe",
 "sha256": "bf8589376027503051d4aec6f7bae7dbab36b06a3d2a354603c895f4997a42e7"}
```

`repair_blocked` is live on `/status` (currently `None` — nothing blocking), and
`preparing_since` is in the deployed `migration.py`.

**A TRAP TO KNOW ABOUT, because it nearly ended the session looking finished when it was not.**
`deploy.sh` shipped `config.py` with `AGENT_VERSION = 0.20.16` and `/agent/version` kept
answering **0.20.3** — because the live service is pinned by a systemd drop-in,
`/etc/systemd/system/neuron-coordinator.service.d/zz-agent-release.conf`, whose
`Environment=NEURON_AGENT_VERSION` overrides the file default. `test_download_links.py` says
this in its own failure text: *"Production is only correct while an env pin overrides it."*
**Deploying the code is not publishing. Check `/agent/version` afterwards, every time.** The
previous pin is backed up beside it as `zz-agent-release.conf.bak-pre-0.20.16`, and the whole
coordinator tree as `~/coordinator-backup-pre-0.20.16.tgz` on the VM.

**Also checked before deploying, and worth repeating:** `deploy.sh` ships
`coordinator/node_tokens.json` from the local checkout. It was byte-identical to the VM's and
only `register_nodes.py` reads it, so it was harmless here — but a stale local copy going over
live remote state is exactly the shape of Session 67's unexplained token divergence. Compare
before shipping.

## [P60] — a node served garbage at 3 tok/s while every light was green

**This is the most important thing in this handoff.** Installing 0.20.16 restarted this PC's
agent; the coordinator re-placed the roster while it was away and moved the Pavilion from 10-18
to 10-27. **It was never told.** It served `layers[10:]` over a skeleton whose 19-27 were never
materialised, and answered a question about the sky with
`Sovereberg Sovere ABCDEFGHITestCategory`.

`/status` said `routable: true`, `stage1_ok: true`, `network_healthy: true`, 28/28 covered.
Decode was **3.00 tok/s, the fastest this network has ever produced**, because a node running
nine layers instead of eighteen is genuinely quicker. **Speed went up as correctness went to
zero — on this network throughput is not evidence of anything.**

One command found it, and it was [P56]'s own fix from that morning:

```
RangeMismatch: challenged on layers 10-27 but the node holds 10-18
               -- placement disagreement, not a bad answer
```

**The gap is the NOTIFICATION, not the placement** — [P37]'s open item. The fix that closes it
without trusting the coordinator or timing: `node_server` should refuse a config whose range
exceeds what `unmaterialized_layers` already told it at load time, and answer `range_mismatch`,
which the driver already handles as a reroute. Filed, not built. **Do this next.**

**And the operational rule it earns:** after any placement change or restart, attest before
trusting a chat result.

```
.venv\Scripts\python.exe -m security.proof_of_compute --host <ip> --port 50999 --s2 <lo> --n 28
```

## What this session did

  * **[P58]** — the account claim looked for the payout key in a place only Windows has, so it
    was broken for every Linux node, source run and self-hoster. Fixed; the Pavilion's node is
    claimed, the first claim of a node that was not already owned.
  * **[P56] parts one and two** — one constructor for the `config` message, shared by driver and
    verifier; the MIDDLE role challenged as a relay for the first time, graded on what it
    forwards. Proven against a real middle node (`max_err` 2.8e-05). Part three (declared
    precision) is still open and still gated on [P30].
  * **[P59]** — a middle node now prefers a same-LAN next hop. With the OptiPlex's firewall
    opened (`ufw allow from 192.168.1.0/24 to any port 50999 proto tcp`) the hop went from a
    3 s timeout to **0.2 ms**, confirmed on the wire. Also: an unreachable offer is now
    remembered, because retrying it cost +1.5 s of first-token on every request.
  * **The third machine joined** and immediately found two bugs only a third machine could: a
    partial config crash-looping the agent (`KeyError: slice_dir`, while a comment claimed
    defaults were applied), and `agent/requirements.txt` never listing `cryptography`.
  * **`/` is the workspace UI**, `/classic` keeps the old page.

## Still open

  * **[P60]** the range-notification gap — above.
  * **[P56] part three**, declared precision. Needs a MEASURED tolerance per dtype and a
    declaration that costs something. Urgent the day [P30] lands, not before.
  * **[P30] phase 2** — `engine/ggml_pipeline.py` is imported by nothing; the binaries are
    absent; `router.SECURE_HOP_SINCE=(0,99,0)` withholds every grant.
  * **The node id keeps flipping** between `agent-optinovate-6ff49d` and `-7fc2ff` across
    restarts. Both are registered for this one PC. Never mint a fresh one — it orphans earnings.
  * **The driver's own hop still uses the relay** because this PC is on a hotspot
    (`192.168.137`) while both nodes are on `192.168.1`. On the home network it would go
    direct, worth roughly another 76 ms/token.

## Facts that save time

  * **Three stages is slower than two** and always will be on a model that fits on two.
    Splitting does not reduce compute — decode is sequential — it only adds a round trip. More
    machines buy capacity and verification, never per-user speed. Two stages measured 2.16
    tok/s against three at 1.84.
  * Tests: NO pytest. `C:\Users\optin\neuron\.venv\Scripts\python.exe -m <module>`.
  * Ship source to a node as ONE tarball, then verify by **importing** every module the agent
    loads. `compileall` passes while imports fail.
  * `pkill -f <pattern>` over ssh matches its own command line and kills the shell running it.
  * An installer exit code of 0 does not prove an install; check the exe is still there a
    minute later, and check the routes.

---

# Handoff — start of Session 69

## ADDENDUM 2, 2026-08-21 07:50 — THERE ARE THREE MACHINES NOW, AND THE CHAIN HAS A MIDDLE

**The hard boundary is lifted.** The founder brought `optiplex-server` in as a third node
(2026-08-21). Earlier handoffs said *"Do NOT touch optiplex-server / nuc / 192.168.1.10"* —
that no longer applies to the OptiPlex. The NUC and 192.168.1.10 were never discussed.

```
agent-optinovate-7fc2ff                          0-9     this PC, driver + stage 1
agent-raman-hp-pavilion-laptop-15-eh3xxx-e4920b  10-18   Pavilion, MIDDLE relay
agent-optiplex-server-ce473b                     19-27   OptiPlex, last stage
```

`[[0,9],[10,18],[19,27]]`, routable, `stage1_ok`, healthy. A three-stage chat answers
correctly — all three node ids in the chain, 0 reroutes, 1.66 decode tok/s (two stages was
2.16; the extra hop costs, exactly as [P57] says distribution does).

**The OptiPlex, as configured:** `homeadmin@100.114.189.46`, Ubuntu, 6 cores, 16 GB, Python
**3.10**, `~/neuron`, systemd `--user` unit with linger enabled, source synced as one tarball
from HEAD. `donation_mode: max` and **`local_chat: false` from the first start** — set before
it ever ran, because of what starting a chat UI did to the Pavilion. Its own payout key was
minted on first start: `0x50bbBA48aF0a935e90D7914C0b121F0996F59da5`. **Its node is UNCLAIMED**
— claiming it is now a one-command check that [P58]'s fix works on a machine other than the
one it was found on.

**[P56] part two is verified against a real middle node.** It could not be until there was one:
two machines make a driver and a last stage and nothing in between. Relay challenge on layers
10-18: **passed, `max_err` 2.8e-05, `path: relay`.**

**Two bugs that only a third machine could find** — both fixed, committed, tested:

  * **The config did not fall back to defaults while a comment said it did.** `main()` logs
    "config predates this build; using built-in defaults for: ..." and carries on, but the
    agent reads its config forty times with a plain `self.cfg[...]` and `ensure_slice` reads
    `slice_dir` that way. A partial config raised `KeyError` out of `setup()` and systemd
    restarted it every ten seconds forever. `load_config` now returns a `_Config` whose
    `__missing__` falls back to `DEFAULT_CONFIG`. **Caveat worth knowing:** `.get(k, other)`
    call sites still bypass that — `donation_mode` is read as
    `self.cfg.get("donation_mode", resource_guard.DEFAULT_MODE)`, so a config without the key
    gets `idle` (ceiling 15%) even though `DEFAULT_CONFIG` says `balanced` (50%). Two defaults
    for one setting. Not fixed; it wants those call sites turned into subscripts.
  * **`agent/requirements.txt` never learned about `cryptography`.** A venv built from it gives
    `ModuleNotFoundError` for `neuron_driver`, `node_server`, `agent.agent` and
    `api.openai_compat` at once — `security/wire_crypto.py` is imported at MODULE scope, so it
    is not a degraded feature, it is an agent that never reaches its first log line. This is
    [P54]'s fourth registration site one file over: `coordinator/requirements.txt` got that
    line and a paragraph explaining why, and the file listing the NODE's dependencies did not.
    `agent/test_requirements_carry_eager_imports.py` now parses the startup path and checks it.

**Lessons that held up:** ship source as ONE tarball; verify by IMPORTING every module the
agent loads, not by compiling them (compileall passed on both machines while imports failed);
set `local_chat: false` before the first start on any node you are not sure has spare RAM.

## ADDENDUM, 2026-08-21 06:10 — PLACEMENT COLLAPSED OVERNIGHT AND WAS REPAIRED

**Everything below was written at 01:50. Between then and 06:00 the chain collapsed on its own,
with both machines up the whole time, and it is the most likely thing to be broken when you
next look.**

At 05:46 the coordinator reassigned the Pavilion `[0, 27]` while this PC also spanned the whole
model. Two nodes each holding everything → `build_chain` advances to the farthest `layer_end`
and the pinned stages collapse into one. `chain_ranges` was `[[0,27]]`, `routable false`,
`network_healthy false`. Nobody could chat over the network.

**Repaired with:**

```
bash coordinator/pin_layers.sh --driver agent-optinovate-7fc2ff
```

`--driver` is REQUIRED — the roster has `head_ms: None` on every node, so the script cannot work
out which machine you chat from and refuses rather than guessing. Verified after: `[[0,9],
[10,27]]`, routable, `stage1_ok`, and a real network answer — *"Hello! How can I assist you
today?"*, 9 tokens, 0 reroutes, 2.16 decode tok/s.

**This will recur, and it is now the top item — above [P56].** Two facts sit under it and they
are probably the same fact: this PC's node id keeps flipping between `agent-optinovate-7fc2ff`
and `-6ff49d` across restarts (both are registered for one machine), and the coordinator
gap-heals a briefly-absent node by handing the survivor the whole model without ever handing it
back. `pin_layers.sh` is a repair, not a fix — it has now been needed twice in two days, and a
network that needs a human to run a shell script after every restart cannot be given to
strangers. Find out WHY a node that is up gets reassigned `[0,27]`, and whether the id flip is
what triggers it.

## STATE AT SHUTDOWN, 2026-08-21 — READ THIS FIRST

**Network HEALTHY, re-verified 06:10 after the repair above:** 2 nodes, 2 stages,
`[[0,9],[10,27]]`, routable, `stage1_ok`. Both machines are up.

**0.20.15 is built and INSTALLED on this PC, and `/` is now the workspace UI.**

```
/            307 -> http://127.0.0.1:8080/workspace/
/classic     200   (what / used to serve, byte for byte, same no-store)
/workspace/  200
```

Done as a redirect exactly as the last handoff specified — the bundle was NOT rebuilt at base
`/`, so there is no collision with the `/assets` mount that serves the old React app's chunks.

**THE PAVILION'S NODE IS CLAIMED.** `agent-raman-hp-pavilion-laptop-15-eh3xxx-e4920b`,
`owner_wallet_id` recorded, payout address unchanged at `0xA39F…E3Ee`. First successful
account-claim of a node that was not already owned.

**`coordinator/config.AGENT_VERSION` is still 0.20.3 and NOTHING is pushed.** Publishing
remains a deliberate act the founder has not taken. `test_version_lockstep` permits the
coordinator to lag a built version precisely so this state is legal.

## [P58] — the claim was broken for every node that is not a Windows installer install

**Testing it was the session, and it failed on the first honest try.** Run against the
Pavilion — the only genuinely unclaimed node — `POST /node/claim` returned:

```
409  this node pays out to 0xA39F...E3Ee, and this machine holds no key at all —
     nothing here can sign for it. Use the browser-wallet claim...
```

**The key was on disk, for that exact address, in the same directory as the config file the
same endpoint had just read successfully.**

`agent/agent.py:1568` binds with `state_dir=os.path.dirname(self.config_path) or HERE` — the
key lives BESIDE the config. `_node_identity` knew the config could be in either of two places;
`_local_payout_key` knew about neither and hard-coded `LOCALAPPDATA/NEURON`. Those coincide on
a Windows installer install and nowhere else, so the flow worked on the one machine it had ever
been run on and failed on the entire population [P53] was written for — and it failed by
telling the operator to go find a browser wallet, which is the exact thing [P53] existed to
make unnecessary.

Fixed, committed, in 0.20.15, verified live. `ui/test_claim_finds_the_key_beside_the_config.py`.

## I took the network down for ~25 minutes, and the cause is worth carrying forward

**Never start `ui.app` on a node that is tight on RAM.** Its lifespan calls
`local_gguf.can_serve()`, and when llama.cpp cannot serve locally it loads the pipeline-driver
shard — embed + layers 0..9 + lm_head. The Pavilion is a 12 GB laptop already holding an
18-layer fp32 node slice at ~8 GB resident. Starting a uvicorn there put it into swap thrash
so deep that **sshd could not answer for 25 minutes while the machine still answered ping.**
The node dropped, the coordinator gap-healed, `chain_ranges` collapsed to `[[0,27]]`.

**It recovered on its own.** Once the Pavilion came back and re-advertised, the coordinator
restored `[[0,9],[10,27]]` — `pin_layers.sh` was NOT needed. Do not reach for it reflexively.

**The way to exercise a UI endpoint on a node is `starlette.testclient.TestClient` WITHOUT the
`with` form** — used that way it never runs lifespan, so nothing loads, and the request still
goes through the real router, the real SessionMiddleware and the real endpoint. Run it under
`systemd-run --user --scope -p MemoryMax=1600M -p MemorySwapMax=0` so a mistake kills the test
instead of the machine. `/tmp/claim_test.py` on the Pavilion is that harness.

**And `pkill -f <pattern>` over ssh matches its own command line.** Half an hour of "still
down" was `pkill -9 -f uvicorn` killing the shell that ran it, before it could print anything.
Kill by PID, or use a bracketed pattern.

## What the Pavilion looks like now

  * **Full 0.20.14 source sync**, shipped as ONE tarball rather than file-by-file — that is the
    fix for last session's partial-copy outage. Backup of the previous tree is at
    `~/neuron-src-backup-0.20.8.tgz`. Verified after the copy by importing every module the
    agent loads, not just by compiling them.
  * **Its venv now has the web stack**: fastapi 0.139.2, uvicorn 0.51.0, starlette 1.3.1,
    itsdangerous, Authlib, httpx2. torch 2.4.1+cpu, transformers 4.44.2 and pydantic 2.13.4 are
    UNCHANGED — checked before and after. The "no uvicorn" note in older handoffs is gone.
  * **`local_chat` is now `false` in its config.json, and it MUST stay false.** Installing
    uvicorn turned that old "cosmetic" note into a memory bomb: the agent starts its own Chat
    UI on boot, the UI loads the pipeline-driver shard (layers 0-9), and on a 12 GB laptop
    already holding an 18-layer fp32 slice at ~8 GB that is an OOM. It went round that loop
    twice — start, `paused (low RAM (160 MB))`, OOM, systemd restart — and each pass dropped
    the node off the chain. With `local_chat: false` it came back within one poll and stayed.
    **A node that cannot hold the driver shard as well must not serve a chat page.** If that
    machine ever needs one, the fix is to make `start_local_chat` refuse on insufficient RAM
    rather than to turn the flag back on.
  * **It also needed `coordinator/*.py`**, because `api/openai_compat.py` imports
    `coordinator.config/ledger/model_registry`. Only the .py files were sent — `neuron.db` and
    `node_tokens.json` were deliberately NOT copied.
  * **Its `ui/app.py` carries the [P58] fix but NOT the `/` redirect.** It is a node, so this
    does not matter for serving; re-sync it if anyone wants the workspace UI there.

## Do these, in this order

1. **[P56] — TWO OF THREE PARTS ARE NOW DONE (2026-08-21). Read this before the plan below.**

   **Shipped to the repo, NOT yet to a build.** `common.stage_config()` is the only place the
   `config` message is shaped, and `neuron_driver._connect`, `challenge_node` and the new
   `challenge_relay_node` all call it. The middle role is challenged as a RELAY for the first
   time — the node dials a sink the verifier opens and is graded on what it FORWARDED, which is
   its own output through the same `_batcher("middle", s1, s2)` that serves users. `sink_host`
   is required and never guessed; the probe survives only as a RECORDED fallback, and every
   attest result now carries `path`. `security/test_verifier_drives_the_user_path.py`, 26
   assertions. Verified live against the Pavilion: passed, `max_err` 0.000217, `path: last`.

   **The installed 0.20.15 predates this.** The live driver still builds its own dict; the next
   build carries the constructor. Nothing about the running network changed.

   **To run the relay challenge you must give it an address the node can dial you at:**
   `--sink-host <ip>` or `$NEURON_VERIFY_SINK_HOST`. On this testbed that is this PC's
   Tailscale address. Note the live topology has NO middle node (driver 0-9 + last 10-27), so
   the relay path cannot be exercised end to end here until there is a third machine — it is
   covered by real sockets against a scripted node in the test instead.

   **WHAT IS LEFT is part three, and it is a measurement, not a decision:** `verify()` compares
   against fp32 with `atol=0.05`, so a legitimately quantized node is indistinguishable from a
   cheating one. Load the same shard fp32 and in the declared dtype, run the same seeded
   challenge, read the drift. And the declaration must COST something — if declaring q4 only
   buys a looser tolerance, every cheat declares q4. Urgent the day [P30] lands, not before.

   **And the endgame, which none of the above reaches:** a challenge that is DISTINGUISHABLE
   from real traffic can always be special-cased. Verification by REPLICATION — the coordinator
   sending one sampled real request down two chains and comparing — never needs to know the
   right answer, only that two machines disagree, and is the only version a node cannot detect.

   The original plan, kept because parts of it are still the reference:
     * **One constructor for the `config` message.** `neuron_driver._connect` builds it at
       `neuron_driver.py:301`; `proof_of_compute.challenge_node` and `challenge_middle_node`
       each build their own. Move it to `common.stage_config()` and have all three call it —
       then the role-deciding fields cannot drift apart again, which is the whole of [P55].
       Note `verify()` deliberately sends NO `wire` field so the reply is lossless; keep that,
       it is a transport choice and does not touch the role decision.
     * **Exercise the MIDDLE role for real.** Today it is challenged with `probe: True`, a role
       no user request ever produces. Instead give the node a next hop it can actually reach —
       a sink the verifier opens — and compare what it forwards against `common.mid_stage`.
       The node's side is already read: it dials `host_b:port_b`, sends
       `{"type":"config","s2":..,"n":..,"wire":..}`, expects `{"ok":True}`, then forwards
       `{"type":"act","hidden":h2}` and expects `{"hidden":..,"b_compute_ms":..}` back. A sink
       that acks with no `wire` keeps the forwarded tensor lossless. Fall back to the probe
       when the node cannot dial back, and RECORD which path ran — a probe-only attestation
       must not be reported as equal to a real one.
     * **Declared precision.** `verify(..., atol=0.05)` compares against fp32, so a quantized
       node is indistinguishable from a cheating one. This needs a MEASURED tolerance per
       declared dtype, not a guessed table — run the same challenge against a shard loaded
       fp32 and fp16 and read the drift. `agent/test_weight_dtype_report.py` already exists,
       so the node has somewhere to declare it. **And the declaration has to COST something:**
       if declaring q4 only buys a looser tolerance, every cheat declares q4. It has to be the
       same number the node is placed and paid on.
     * **Build order: 1, then 2. Treat 3 as gated on [P30]** — it becomes urgent the day
       k-quants land, not before.
     * **What none of it solves, and the endgame.** A challenge that is DISTINGUISHABLE from
       real traffic can always be special-cased by a node that wants to; parts 1 and 2 narrow
       the gap, they do not close it in principle. The durable answer is verification by
       REPLICATION: the coordinator sends the same real request down two independent chains on
       a sampled fraction of traffic and compares. It never needs to know the right answer —
       only that two machines disagree. It is the only version a node cannot detect, because
       there is nothing to detect. Cost is one duplicated request per sample.
2. **Publishing, or a decision not to.** 70+ commits, nothing on GitHub, `AGENT_VERSION` 0.20.3.
   Until that happens the fleet stays on 0.20.3 and none of this reaches anyone.
3. **[P30] phase 2**, unchanged: `engine/ggml_pipeline.py` is imported by nothing; the
   `llama-server`/`ggml-rpc-server` binaries are absent; `router.SECURE_HOP_SINCE=(0,99,0)`
   withholds every grant. Settle the transport before building the last mile.

## Known open, smaller

  * **An installer exit code of 0 does not prove an install, and neither does the timestamp.**
    The first silent run of NEURON-Setup-0.20.15.exe returned 0, the exe appeared with a moved
    timestamp, and a minute later it was NOT on disk. Inno's setup.exe returns before its child
    finishes, so `Start-Process -Wait` is not a wait. Re-running with `/LOG=` installed
    cleanly. **Check the exe is still there a minute later, and check the routes.**
  * **The node id flipped back to `agent-optinovate-6ff49d`** after this reinstall (it was
    `-7fc2ff`). Both are registered for this one PC and both hold history. Still worth
    reconciling; a fresh id would orphan earnings ([P53]), so never mint one.
  * The Pavilion's `node_token` divergence from Session 67 is still unexplained and can recur.
  * The workspace UI's tool loop is OFF by decision: it runs shell and file actions.
  * `trust-chat` still has uncommitted `src/server/skills.ts` and `src/server/docx.ts`.

## Facts that save time

  * Speed is settled: 2.18 tok/s on the network, two thirds a MEMORY BANDWIDTH wall
    (36.9 GB/s, 187 MB per fp32 layer). Only k-quants move it, which means [P30]. Local is
    32 tok/s and free. `tools/bench_quant.py` reproduces it. Do not re-litigate.
  * Tests: NO pytest. `C:\Users\optin\neuron\.venv\Scripts\python.exe -m <module>` per file.
    `python` on PATH has no torch.
  * The Pavilion is `raman@100.79.125.112`, a PLAIN FILE COPY run by `systemd --user`, not a
    git checkout. Ship a TARBALL, not individual files.
  * `pkill` is unreliable here and matches its own command line. Use PowerShell `Stop-Process`
    by PID, and verify the port is free before starting a replacement.

---

# Handoff — start of Session 68

## STATE AT SHUTDOWN, 2026-08-20 — READ THIS FIRST

**0.20.14 is built and INSTALLED on this PC and serves two chat UIs.** `/` is the original
page, untouched. `/workspace` is the new one (from the founder's `Trust chat` repo, rebranded
and cut down to NEURON only). Both answer through NEURON. Coordinator shows
`agent-optinovate-6ff49d ver=0.20.14`.

**NOTHING HAS BEEN PUSHED.** Not NEURON, not trust-chat. `coordinator/config.AGENT_VERSION` is
still **0.20.3**, so no node in the fleet is told about any of this. Publishing is a deliberate
act the founder has not taken.

**The Pavilion is still on 0.20.8** and is the ONLY machine where the claim flow can be tested,
because this PC's node is already claimed (`unclaimed: false`).

## The outage at the end of Session 67, and what actually caused it

**Recovered. `[[0,9],[10,27]]`, routable, healthy, 2 nodes.** Recorded because none of it was
obvious from the symptom, and one part is still unexplained.

**The symptom was placement, not corruption.** The Pavilion dropped off; the coordinator
gap-healed by giving the only remaining node the whole model; `chain_ranges` collapsed to
`[[0,27]]` and `stage1_ok` went false, because `node_a.coord_get_chain` requires stage 1 to be
exactly `[0,9]`. Network chat was down, local chat was untouched throughout. This is precisely
what `pin_layers.sh`'s header warns about: *a node left spanning the whole model wins the chain
walk and collapses the pinned stages*.

**Three separate faults were stacked underneath it:**

  * **`stream_timing.py` was MISSING on the Pavilion** while `neuron_driver.py` and
    `engine/local_gguf.py` both import it — a partial file copy, new modules omitted. Its chat
    UI crashed on every start. **Lesson: copying a changed file to that machine means copying
    every NEW module it imports; there is no packaging step to catch it, because it is a plain
    file copy and not an installer.**
  * **The node_token stopped matching the coordinator's copy** — 409 on register, 401 on ping,
    so it could not rejoin at all. Fixed by putting `register_secret` in its config so it could
    re-register under its EXISTING node id (a fresh id would have orphaned its earnings,
    [P53]), then removing the secret again once the new token was persisted. **WHY the token
    diverged is still unknown, and it can therefore recur.**
  * **It self-paused**: `cpu 51% > donation ceiling 50%`. Normal behaviour, not a fault, but it
    means the node can be healthy, credentialed and still not advertising.

**Also still true there:** its venv has no `uvicorn`, so the Pavilion's own chat page cannot
start. Cosmetic — node serving is unaffected.

## What this session did

**[P55] — fixed, shipped, confirmed live.** The network returned `'  1  2   3'` and billed for
it because the last node ran the right layers with the FINAL NORM SKIPPED: `node_server` read
the presence of `s1` as "a verifier is probing me", and `neuron_driver` sends `s1` on every
config, so in a TWO-stage chain every real request took the probe branch. Fixed at both ends,
neither needing the other. `agent/test_last_stage_is_not_a_probe.py`.

**[P56] filed and STILL OPEN — the most serious thing on this list.** Proof-of-compute only
ever exercises the PROBE path, so it certified the Pavilion healthy on 5662 challenges while it
returned garbage to every user. It also compares against fp32 with `atol=0.05`, which means a
quantized node is indistinguishable from a cheating one — a wall [P30] will hit.

**[P57] filed.** The network is 2.18 tok/s and two thirds of that is a MEMORY BANDWIDTH wall,
not CPU: 36.9 GB/s measured, 187 MB per fp32 layer. `tools/bench_quant.py` shows int8 gives
2.87x from bytes alone and destroys the answer (34.78% drift), so k-quants are the only route
and that means [P30]. Threads, cores, fp16 storage, the wire codec and rebalancing are each
ruled out with a number.

**The relay detour is fixed but DORMANT.** `lan_direct.py` lets two machines on one LAN skip
the relay by having the DRIVER name its subnets and the node answer only from inside them — no
Tailscale, no address published, nothing stored. It is dormant only because this PC is
currently on a hotspot (`192.168.137`) while the Pavilion is at home (`192.168.1`).

## Do these first

1. **Test "Claim with my account" on the Pavilion.** Untested end to end. Needs 0.20.14 there,
   which means a source update (it is a plain file copy run by systemd, NOT a git checkout).
   **And note: something silently restored `agent/node_server.py` there once during a gap.**
   Cause never identified; the hourly `iris_autonomous_v7.py` cron was suspected and CLEARED by
   18 consecutive checks across its firing. Verify with
   `md5sum ~/neuron/agent/node_server.py` after any copy.
2. **Decide what `/` serves.** The workspace UI is additive today. Making it the default is a
   product decision, not a code one, and everything below is downstream of it.
3. **[P56].** The verifier has to drive the path a USER's request takes, and needs a notion of
   a node's declared precision before quantized nodes can ever be honest.
4. **[P30] phase 2.** Still not wired: `engine/ggml_pipeline.py` is imported by NOTHING. Three
   blockers in order — the `llama-server`/`ggml-rpc-server` binaries are absent; the rpc-bridge
   refuses every caller because `router.SECURE_HOP_SINCE = (0,99,0)` withholds all grants; then
   the wiring. Settle the transport before building the last mile onto a bolted door.
5. **Publishing.** Release notes exist for 0.20.9 and 0.20.13. Nothing is on GitHub, no SHA is
   set, `AGENT_VERSION` unchanged. Until that happens the fleet stays on 0.20.3.

## Known-open, smaller

  * `trust-chat` still has uncommitted `src/server/skills.ts` and `src/server/docx.ts`. The
    latter makes Tailwind emit one dead `.table` rule, which is the ONLY reason the CSS differs
    between a clean checkout and the shipped bundle. The JS is byte-identical.
  * The workspace UI's tool loop is OFF: it runs shell and file actions through a route that
    does not exist in a frozen app, and shipping it to strangers is a security decision.
  * Memory is per-machine, not per-account, and costs 174 MB while in use (released after 10
    minutes idle). Conversations DO follow the account.
  * `agent-optinovate-6ff49d` and `-7fc2ff` are both registered for this one PC; the id it
    reports has flipped between them across restarts. Worth reconciling.

## Still open from before


## STATE AT SHUTDOWN, 2026-08-20 — READ THIS FIRST

**The network returns correct answers again, and it is 2.18 tok/s.** [P55] is fixed, shipped as
**0.20.9**, built, installed on this PC and confirmed live by the founder: *"Hello! How can I
assist you today?"* over the chain. Three commits, tree clean, nothing pushed.

**What [P55] actually was.** The last node ran the right layers and skipped the FINAL NORM,
because `node_server` decided its role from whether the config carried an `s1` — which the
verifier sends and which `neuron_driver` also sent on every request. Two-stage chains (driver
0-9 + one node 10-27, i.e. the live topology) therefore served every real request as if it were
a verifier's probe. Fixed at both ends independently, so no upgrade ordering is needed. See
[P55]; the regression test is `agent/test_last_stage_is_not_a_probe.py`.

**Do not chase 51.3 tok/s. It does not exist.** It is a memory-bandwidth floor computed from
36.9 GB/s and a q4_k_m layer, quoted once and then asked for as a target. The real numbers:

| path | tok/s | status |
|---|---|---|
| local llama.cpp q4_k_m, one machine | **32.11** | works today, default, free |
| ggml split across both (hand-run) | 3.97 | code exists, NOT wired |
| PyTorch chain | **2.18** | what the network actually is |

**The speed question is settled and the answer is uncomfortable: the wall is the memory bus.**
`tools/bench_quant.py` proves it — dynamic int8 on the real slice is **2.87x** (8.43 -> 2.94
ms/layer) purely from 4x fewer weight bytes. fp32 gives 7.0 tok/s as a HARD ceiling on this
machine with a perfect implementation and no network at all. So quantization is the only lever
on the 67% of a token that is compute, and everything else that looks like a lever is not:
threads, cores, fp16 storage, the wire codec, rebalancing layers, adding machines. Each is
ruled out with a number in [P57].

**But the cheap quantization is closed too.** That same int8 run drifts 34.78% from fp32 —
garbage, not a tolerance argument. Only k-quants (per-block scales) stay correct, which means
llama.cpp, which means [P30]'s engine is not one option among several. **It is the only route.**

## Do these first

1. **[P30] phase 2 is still NOT wired, and the git log reads as if it is.** `a2d7b81` added the
   node-side door and a driver-side dialer; `engine/ggml_pipeline.py` (which carries the 3.97
   tok/s) is imported by NOTHING — full-tree grep returns only its own docstring and logger.
   `agent/rpc_bridge_client.py` is imported only by its test. `neuron_driver`, `ui/app.py`,
   `api/openai_compat.py`, `agent/local_chat.py` and `agent/agent.py` have zero references.
   There is no `engine/test_ggml_pipeline.py`.
2. **Three blockers on it, in this order.** (a) `llama-server` and `ggml-rpc-server` are NOT on
   this machine — `llama_cpp` is installed as a LIBRARY (that is the 32.11 tok/s path) and
   ships `llama_cpp/lib/`, but neither executable. (b) `_serve_rpc_bridge` refuses any caller
   without a [P52] channel, and `router.SECURE_HOP_SINCE = (0, 99, 0)` withholds every grant
   network-wide because the encrypted hop does not survive the relay. **Settle the transport
   before building the last mile onto a door that refuses everyone.** (c) then the wiring.
3. **[P56], and it is the larger finding of this session.** Proof-of-compute exercises only the
   PROBE path, so it certified the Pavilion healthy on 5662 challenges while it returned
   garbage to every user. It also compares against fp32 with `atol=0.05`, so **a quantized node
   is numerically indistinguishable from a cheating one** — which the roadmap will hit the day
   [P30] lands. The verifier needs to challenge at a node's DECLARED precision.
4. **The relay detour was measured and the fix was DECLINED** (founder, 2026-08-20). Dialling
   this PC's own node through the relay costs **88 ms** median; the Pavilion is 5.3 ms away
   directly on the LAN. `--no-relay` would recover ~38% and was refused because it publishes
   Tailscale addresses and makes nodes unreachable to anyone off the tailnet. **Do not re-apply
   it.** The durable version is the coordinator publishing BOTH addresses and the driver
   preferring the direct one with a fallback — that needs a VM deploy, which needs the SSH key
   passphrase only the founder can enter.

## Still open from before


## STATE AT SHUTDOWN, 2026-08-19 16:18 — READ THIS FIRST

**The network is UP and back to exactly where it started:** `chain=[[0,9],[10,27]]`,
`routable=True`, `healthy=True`, 2 nodes online, chat UI answering. Wallet **585.98 NRN**
(up from 550.97 — claimed nodes now credit the account directly, which confirms [P39] works).
65 commits local, nothing pushed, working tree clean.

**I broke the live network for ~50 minutes and this is why.** Pinning Qwen3-4B put both nodes
into a crash loop:

```
ValueError: The checkpoint you are trying to load has model type `qwen3`
            but Transformers does not recognize this architecture
```

**`transformers` is pinned at 4.44.2, which predates Qwen3.** That is the real reason [P43] has
never run, after many sessions of treating it as a placement/sizing problem — `CAPACITY_CASE.md`
walks through fp16, caps and pinning and never mentions it, because nobody had ever executed the
load. **The 4B cannot run on the PyTorch path until every node upgrades transformers**, which is
a fleet-wide dependency change, not a config step.

Recovery took DB surgery on the coordinator (`settings.serving_model_id` back to the 1.5B, and
`nodes.layer_start/layer_end` back to 0-9 / 10-27) because clearing the pin does NOT move the
serving model — migration needs nodes to report ready, and crash-looping nodes never do. **A
pinned model that no node can load is a trap with no in-product way out.** That is worth fixing
before anyone pins anything again.

**What was reverted:** the 4B pin, `donate_ram_gb` on this PC, and `NEURON_WEIGHT_DTYPE=fp16` on
both machines. `config.json.bak_pre_4b` holds the pre-experiment config.

**What fp16 measured before it was reverted, and it is worth keeping:** the Pavilion's agent
dropped from **7.19 GB to 4.23 GB** resident, free RAM 2.9 GB → 5.9 GB. NEURON was taking 62%
of an 11 GB laptop to serve an 18-layer slice of a 1.5B model. That is a recruitment problem in
its own right.

---


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

A later reinstall came back as `6ff49d`. **Both identities were then swept: 35.82 NRN from
`6ff49d` and 26.39 from the `7fc2ff` orphan — 62.21 NRN that a reinstall had put out of reach.**

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
