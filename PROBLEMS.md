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
- **2026-08-11 — A node reports the build it runs, so a rollout can be watched and a rollback
  confirmed (0.20.2).** The founder's words on the third release of one afternoon: *"take
  everything, plan, then make it."* Fair. 0.20.0 shipped, then 0.20.1 for rollback, then this —
  each gap found by reacting rather than by asking up front what a release must carry.
  The gap itself: nothing reported the running build. The version string existed only in the
  first line of that machine's own log, unreadable on a PC behind a NAT. So adoption was
  invisible and — the sharp part — **the rollback added hours earlier could not be confirmed**.
  A recovery path you cannot check is one you have to trust.
  Three fields, because it takes three to answer one question. A node still on an old build has
  either not made its daily check (fixes itself), has `auto_update` off (needs its operator), or
  is failing the download (`update_check` says which). NULL is "an agent too old to say", never
  "up to date" — merging those would report a stale fleet as patched, the same class of mistake
  as [P34] and [P37]. All three are COALESCEd so a node rolled back to a pre-0.20.2 build does
  not erase what we already knew, at exactly the moment someone is checking whether the rollback
  worked; and `update_checked_at` follows the verdict so it always carries its own age.
  Public page aggregates (*"3 of 4 on the latest agent"*) — same rule as standing: *is this
  network patched* is a fair question for a visitor, *which volunteer is behind* is not. The
  operator's own page keeps the detail, since they are the only one who can act on it.
  **Verified before cutting, so this is the last release in the chain:** `holds` (0.20.0),
  `model_id` (0.20.0), hourly re-measure (0.20.0), rollback (0.20.1) and slice self-heal on a
  range change are all already shipped, and `setup()` re-reads its range from
  `/node/{id}/slice-info` and re-downloads when the cached slice does not cover it — so the
  0.20.2 restart is itself the repair for a mis-placed node. 12 cases in
  `coordinator/test_agent_version_report.py`.
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

- **2026-08-18 - The engine switch: Route 2, not Route 1-prime, and the transport is the reason.**
  The [P30] spike measured ggml-rpc splitting a model across two machines at **32 tok/s against
  NEURON's live 1.44**, with `--tensor-split` leaving placement under the coordinator's control.
  That made "tunnel ggml-rpc over the relay" look like the cheap answer. **It is not, and the
  founder's question about the tunnel is what exposed why.**

  **The relay is NAT traversal, not security.** `relay_auth.py` mints an HMAC ticket so a node
  cannot register a tunnel on *another* node's public port - that is the whole of it. The relay
  then "splices raw bytes", and `node_server.py` has **no caller authentication of any kind**.
  Each node's port is published on a public host and anyone may dial it. So NEURON has no
  private channel between machines today, and Route 1-prime was written as though it did.

  That is survivable for NEURON's own wire and deliberately so: `SECURITY.md` argues the wire is
  **safe to expose** - a JSON header plus raw tensor bytes (`wire_codec.py`), nothing executable
  after [P19], size-capped so a scanner cannot make a node allocate arbitrarily. A public port
  speaking that protocol is an acceptable risk.

  **None of that is true of ggml-rpc.** It is a *memory* protocol - allocate a buffer, write
  bytes into it, execute a graph - which is why upstream says never run it on an open network.
  Putting it on the relay would hand every scanner on the internet a memory-write primitive into
  a volunteer's PC. It would undo [P19] on purpose, with a worse payload than the one [P19]
  fixed.

  **So the decision inverts.** Route 1-prime requires building an authenticated, encrypted
  transport BEFORE it can be tried - a real project on its own, and one that must hold against a
  hostile *participant*, not merely an eavesdropper, because a volunteer network contains both.
  Route 2 - embed ggml behind NEURON's existing safe wire - keeps the exposed surface exactly as
  it is today and changes only what happens *inside* the process. **The fast kernels are wanted;
  the fast transport is not.**

  What the spike still bought, and it is most of the value: ggml-rpc is proven to split layers
  correctly, `--tensor-split` is proven to keep placement with the coordinator, the prebuilt
  binaries remove the toolchain problem, and the per-microarchitecture kernels retire [P41] and
  the load-bearing `torch` pin. Route 2 inherits all four.

  **Phases, each with a gate that can stop the project:**
  1. **Measure a real hop.** `rpc-server` on the Pavilion, driven from here over the relay,
     against the 8% loopback floor. If a real round trip per token is ruinous, pipelining across
     machines is the wrong shape at any engine speed, and the answer is replication rather than
     stages. Cheap, and it is the only number that can invalidate everything after it.
  2. **One node, fast, alone.** Make `node_server` compute its layer range with ggml instead of
     PyTorch, wire protocol untouched, on ONE machine. `selftest_shard.py` already demands
     bit-exactness against a full local forward pass; that becomes the acceptance test.
  3. **Two nodes.** The existing chain, one node on each engine, then both. Proof-of-compute
     must still pass - the challenge is at the transformer-layer level, and ggml must answer it
     identically or the reputation system silently stops working.
  4. **Only then packaging.** A C++ dependency per platform, and [P41]'s CPU floor becomes a
     dispatch table rather than a refusal.

  **Not decided, deliberately:** whether the driver's embedding and `lm_head` move too, and
  whether quantized weights change what a node must download ([P36] provenance, [P43] sizing).
  Both are downstream of phase 1 and neither is worth designing before that number exists.

- **2026-08-18 (later) - Route 1-prime is back, because the objection that killed it was
  BUILT.** Earlier the same day this log rejected tunnelling ggml-rpc and chose route 2
  (embed ggml behind NEURON's wire), for one reason: llama.cpp says *"never run the RPC server
  on an open network"*, NEURON had no private channel, and creating one was "a real project on
  its own". [P52] then built that project for an unrelated reason -- the site was advertising
  privacy the wire did not provide -- and the premise expired the moment it shipped.

  **Measured, not assumed.** llama.cpp driven through NEURON's own authenticated, encrypted
  channel, with `ggml-rpc-server` bound to 127.0.0.1 and never on a public port:

  | configuration | rate |
  |---|---|
  | no RPC at all, plain local CPU (control) | 34.97 tok/s |
  | ggml-rpc, plaintext loopback | 32.11 tok/s |
  | **ggml-rpc through the [P52] channel** (grant + X25519 + AES-GCM) | **28.07 tok/s** |
  | NEURON's live chain today (PyTorch fp32, relayed) | **1.44 tok/s** |

  18,542 frames and 1.02 GB of plaintext sealed in a 6-second run. Encryption costs ~12% on
  top of RPC and ~20% against the no-RPC control -- and the result is still **~19x what the
  network does today**.

  **Why this changes the plan.** Route 2 means writing and shipping a C++ component per
  platform. Route 1-prime is now: run a prebuilt `ggml-rpc-server` on localhost, and carry its
  traffic over a channel that already exists, is already tested, and a scanner cannot open.
  Phases 2-4 shrink from "embed ggml" to "supervise a local process and point the driver at
  it", and [P41]'s CPU floor plus the `torch` pin still retire with it.

  **The residual risk, stated rather than buried.** The channel authenticates the CALLER, so a
  stranger who dialled the port cannot speak ggml-rpc. It does not make ggml-rpc safe against
  an authenticated PEER -- a chain member is coordinator-selected, not trusted, and ggml-rpc is
  a memory protocol. The surface shrinks from "anyone on the internet" to "a machine the
  coordinator put in this chain", which is a large reduction and not an elimination. Whether
  that is acceptable is a decision about who is allowed into a chain, and it belongs with the
  open-join question in `SECURITY.md`, not with this measurement.

  **This is the second reversal in one day, and the reasoning is sound both times.** Route
  1-prime was rejected on a true premise that a later commit falsified. Recording it that way
  rather than quietly switching, because the next person needs to know the decision is
  contingent on [P52] shipping -- if the channel is ever removed, the objection returns intact.

- **2026-08-18 - Place chain neighbours by MEASURED latency, not by IP geography.**
  The founder's proposal: read a node's IP, infer its region, and build chains from machines
  near each other. The instinct is right - decode is sequential, so every token pays every hop,
  and that cost is the product's latency. Two corrections to the mechanism.

  **Measure, do not infer.** IP geolocation is unreliable (VPNs, carrier-grade NAT, mobile) and
  it answers the wrong question: what matters is round-trip time between two specific machines,
  which is directly measurable and nearly free - nodes already heartbeat, and the coordinator
  already stores `ms_per_layer` measured that same way. An RTT matrix between candidate
  neighbours is the honest version of this idea, and it degrades gracefully: an unmeasured pair
  scores at a default prior, exactly as a never-measured node already does.

  **And today the relay dominates geography.** Both live machines could sit in one room and
  their traffic would still cross an Oracle VM in another country, because that is how a NAT'd
  node is reachable at all. Grouping by region cannot help while every hop is relayed. So the
  ordering is: **direct connections first** (LAN, or the Tailscale path the agent already
  reports), relay only as the fallback it was built to be - and *then* latency-aware placement,
  which is where the founder's idea pays off, and which needs the roster size they correctly
  identified.

- **2026-08-18 (Session 65) - ROADMAP and PROBLEMS disagreed about speed. They are reconciled
  here, and the reconciliation has a NUMBER attached.** `ROADMAP.md` lists *"not faster than a
  single machine for one user"* under What NEURON Is Not. `PROBLEMS.md` ranks single-user speed
  [P1] **HIGHEST**. Every session since has worked from whichever file it happened to open.

  **They are not actually in conflict, and today's measurement is what shows it.** Phase 3 ran
  the same model, same engine, on one machine and then split across two:

  | configuration | rate |
  |---|---|
  | this PC alone, llama.cpp q4_k_m, no RPC | **32.11 tok/s** |
  | the same model split across this PC + the Pavilion (ggml-rpc over an authenticated tunnel) | **3.97 tok/s** |

  **Distribution costs 8x. It does not buy speed and it was never going to.** ROADMAP is
  simply right: for a model that fits one machine, one machine wins, and no engine change
  reverses that. What the engine work buys is the 32.11 - and 32.11 is a SINGLE-MACHINE number,
  which is the local path, not the network path.

  **So the two documents are answering different questions, and the reconciliation is a
  threshold, not a ranking:**

    * **ROADMAP governs the pitch.** NEURON competes on capacity, cost and access. It must
      never be sold on latency, because splitting a model across machines is *slower* than not
      splitting it, by 8x on measured hardware. Any copy that implies otherwise is false.
    * **[P1] governs the floor.** 1.44 tok/s is not "slower than a single machine", it is
      unusable, and an unusable network cannot deliver the capacity claim to anybody. Speed
      work is justified *up to a usability floor* and is waste beyond it.
    * **The floor is already written down**: TOKENOMICS §11.6's "answers under 30 s". That is
      the number that decides when speed work stops.

  **What this tells a future session, which is the point of writing it down.** Optimise until
  the network clears §11.6, then stop and spend the effort on capacity and on the first
  stranger. Do not chase single-machine parity - it is unreachable by construction, and
  chasing it is how a distributed network ends up justifying itself on the one axis it must
  lose.

  **ROADMAP.md itself is unchanged**, because build rule 2 forbids editing it in-session. It
  needs one line from the founder: its "not faster than a single machine for one user" bullet
  is correct and should stay, and the *reason* now has a measurement behind it rather than an
  intuition.

- **2026-08-18 (Session 65) - [P30] phase 3's open design question, settled: the driver holds
  the whole model on DISK, and that is the exact boundary of what Route 1-prime can serve.**
  In ggml's RPC design the CLIENT reads the model file and uploads tensors; the server holds no
  model of its own. With mmap the driver does not need the model in RAM, but it does need the
  whole file on disk - where NEURON's driver downloads only its own slice. For the 1.5B that is
  nothing; for the models this project exists for it is ~100 GB on whoever drives.

  **The decision: accept it, and stop pretending Route 1-prime is the capacity path.**

  1. **Disk is the cheap resource; RAM is the binding one.** "Too large for any single machine"
     has always meant too large for one machine's *memory* - that is what makes inference
     impossible rather than merely slow. A 100 GB file on a 1 TB disk is ordinary; 100 GB of
     RAM is not. So the claim survives the change in its true form, and `slice_downloader.py`
     stays exactly as it is for every node that is not driving.
  2. **The cost is the download, and it is real.** One machine per chain must fetch the whole
     model once. That is a barrier to *becoming a driver*, not to joining the network, and it
     must be stated in the product rather than discovered.
  3. **It gives the coordinator a new placement input it does not have today:** who holds the
     file. Driver-capable is now a property of a node, alongside `ram_gb` and `ms_per_layer`.
  4. **And it draws the line honestly.** Route 1-prime cannot serve a model no single machine
     can hold on disk. Route 2 (embed ggml, each node loads its own slice) remains the only
     path to that, and today's 3.97 tok/s says the reward for building it is capacity, never
     speed.

  **Which makes the roadmap for the engine two products, not one:** the fast local path (32.11
  tok/s, one machine, model fits) and the capacity path (slow, many machines, model does not
  fit). Route 1-prime serves the first. Only Route 2 serves the second. Anything that reports
  one number for both is measuring the wrong thing.

---

## Problems & risks

### [P58] 🟢 "Claim with my account" looked for the payout key in a place only Windows has (2026-08-21)

**Five sessions of green checks on one machine.** [P53] fixed claiming so an operator with no
browser wallet could record ownership with a Google/GitHub account, and it was verified on a
clean 0.20.6 install — on the founder's Windows PC, whose node was already claimed. Run for
the first time against a node that was genuinely UNCLAIMED (the Pavilion, 2026-08-21), it
returned:

```
409  this node pays out to 0xA39F...E3Ee, and this machine holds no key at all —
     nothing here can sign for it. Use the browser-wallet claim, or ask the operator
     to rebind with the register secret.
```

**The key was on disk, for that exact address, in the same directory as the config file the
same endpoint had just read successfully.**

**Two rules for one directory.** `agent/agent.py:1568` binds with
`state_dir=os.path.dirname(self.config_path) or HERE` — the payout key lives BESIDE the config
that names the node it pays for. `ui/app.py::_node_identity` knew that config could be in
either of two places: the installed state directory, or the `agent/` folder a source checkout
runs from. `_local_payout_key` knew about neither and hard-coded `LOCALAPPDATA/NEURON`.

| install | config.json | payout_key.json | `_local_payout_key` looked in | claim |
|---|---|---|---|---|
| Windows installer | `LOCALAPPDATA/NEURON/` | `LOCALAPPDATA/NEURON/` | `LOCALAPPDATA/NEURON/` | works |
| anything else | `<checkout>/agent/` | `<checkout>/agent/` | `~/.local/share/NEURON/` | **409** |

So the flow worked on precisely the one machine it was ever exercised on, and failed on every
Linux node, every source run, every self-hoster — the entire population [P53] was written for.

**And it failed the way [P53] failed.** [P53]'s indictment was that the error told an operator
their key might be lost while it sat in `payout_key.json` on the same disk. This error says
the machine "holds no key at all" about a file one directory lookup away, and sends them to
find a browser wallet — the exact thing [P53] existed to make unnecessary.

**Fixed by making it one rule.** `_node_config()` now returns the config's PATH as well as its
contents, and the key is read from `path.parent`. `create=True` inherits the same directory,
so a first claim on a keyless machine mints the key where the agent will look for it rather
than in a second location nobody reads — a key nobody knows exists being [P53]'s other half.

**Verified live.** The Pavilion's node is claimed, `owner_wallet_id` recorded, payout address
unchanged at `0xA39F…E3Ee`, `unclaimed: false`. First successful account-claim of a node that
was not already owned.

**The lesson is the same one [P54] drew.** A path computed twice by two rules is a bug waiting
for the first machine where the rules disagree, and a UI resolving state by its own rule
instead of asking the component that wrote it will always find that machine eventually. What
made this survive five sessions is that the only test bed was the one layout where the two
rules happen to agree.

`ui/test_claim_finds_the_key_beside_the_config.py`. `ui/test_node_identity_is_current.py` now
pins the whole resolver group, because pinning `_node_identity` alone said nothing about where
the config was looked for — which is where this bug was.

Related: [P53] (the claim this was supposed to have fixed), [P54] (one thing registered in
several places, each failing silently), [P39] (ownership crediting, which is what a claim
turns on).

### [P63] 🟢 I shipped the standalone product wearing NEURON's URL, twice, and the wallet disappeared (2026-08-21)

**The founder reported "and in chrome wallet is gone too" after installing 0.20.19. That was
me.** `ui/static/workspace/` is a compiled artefact from a separate repo, and two environment
variables decide what comes out of that build:

```
NEURON_ONLY=1            strips Gemini/Ollama/KoboldCPP AND gates <NeuronBar />
NEURON_BASE=/workspace/  makes the page ask for /workspace/assets/... not /assets/...
```

I rebuilt the bundle to fix [P62] and passed only the second. `{__NEURON_ONLY__ && <NeuronBar />}`
therefore compiled away, and with it the wallet, the balance, the sign-in link and the
server-conversation import — while the app itself still worked perfectly. **A build that
succeeds with the wrong environment does not produce a broken product; it produces a DIFFERENT
one.** It shipped that way in 0.20.19 and again in 0.20.20 before the missing bar was traced to
its cause rather than to the user's session.

`NEURON_BASE` went wrong the same afternoon in the other direction: under Git Bash it became
`/Program Files/Git/workspace/` through MSYS path translation, and was caught only by reading
the emitted `index.html`. Use `MSYS_NO_PATHCONV=1`.

**Nothing checked either of them, because a compiled bundle cannot be reviewed in a diff.**
`ui/test_workspace_bundle_is_neuron_only.py` now checks the artefact by what it CONTAINS: every
asset reference lives under `/workspace/`, no reference carries a translated Windows path, every
referenced file is on disk, there is exactly ONE bundle of each kind ([P61]), the wallet bar and
the conversation import are compiled in, and the three [P62] fixes are present. Every assertion
is a string that only survives the correct build.

**Two real bugs were found underneath it while looking**, and both are fixed:

  * **A deleted chat came back on refresh.** `fetchServerThreads` imports the wallet's
    conversations and skips those already local — judged by the very thread the delete removed.
    So the id dropped out of `known` and the next load re-imported it. The server copy was never
    deleted; `DELETE /conversations/{id}` has existed the whole time and the client never called
    it.
  * **Signing out hid the way back in.** `fetchSession` collapsed "could not ask" and "asked,
    nobody is signed in" into one value, and the bar hid on the second as if it were the first —
    removing the sign-in link along with the wallet. That is [P24]'s rule ("we could not check"
    must never render as a fact) applied to a session.

**The lesson is mine and it is specific.** I verified 0.20.19 by checking that the page rendered
and that the citation fix was in the bundle. Both were true. I never checked that the bundle was
the same PRODUCT — and the one person who would notice was the one looking at a header that used
to have his balance in it.

Related: [P62] (the crash this rebuild was fixing), [P61] (stale bundles, the other way a
compiled artefact goes wrong), [P46] (a build that verifies itself while the install is
incoherent).

### [P62] 🟢 A web-search citation blanked the entire workspace, and the failure was total (2026-08-21)

**`/workspace/` rendered nothing while the server served it correctly.** `curl` returned the
right HTML, both assets 200'd, and the same URL rendered in a browser with no stored data. The
console had the answer the whole time:

```
Uncaught Error: Minified React error #31 ... object with keys {title, href}
```

**Both ends of one field, and each was self-consistent.** `rag/retriever.py` has always sent
objects — `sources = [{"title": r.get("title"), "href": r.get("href")} for r in results]` —
while the client declared `neuronSources?: string[]` and rendered each entry directly:
`label = src` … `<span title={src}>{label}</span>`. An object reached React as a child, React
threw, and **the render unwound the whole tree**: no sidebar, no threads, no composer.

**Not a corner case, and not transient.** It fired for anyone who ticked *"Web search — answer
with current info"*, and the object form is written into
`localStorage.localai_chat_threads_v1`, so the thread stayed unopenable afterwards. The
founder's own console located it exactly: `0.messages.3.neuronSources[0..4]`.

**Fixed in two places, and the second matters more.** The render site now accepts either shape
— normalised THERE rather than at the fetch boundary, because conversations already on disk
carry the object form and a fix that only cleaned new replies would leave every existing thread
broken forever. The object form is now strictly better: `title` becomes the tooltip, where
before the tooltip was the raw url the hostname label was already derived from. And
`NeuronSource = string | {title?, href?}` puts that fact in the type.

**Then a per-message error boundary**, because one malformed field three messages back must not
be able to take out the workspace. React unwinds the whole tree on a render error, so the
symptom was a blank screen with nothing to read — the field fix is one line, and *that* is why
it cost a day. Scoped per message, not around the list: a boundary around the list would keep
the app alive and still lose the conversation.

**What this cost, and the lesson that is actually mine.** I checked the page in a browser with
no extensions and no stored conversation, saw it render, and concluded "the bundle is fine" —
then diagnosed a caching bug ([P61], real, and not this), shipped it, and the page was still
blank. **Rendering in a clean profile is not evidence about a browser holding real data.** The
console error was available from the first screenshot and I did not ask for it. The founder
said "last time also you said that", and was right.

Related: [P61] (the caching bug found while looking for this one — genuine, and a red herring
for this), [P55]/[P60] (the same shape at the wire level: two components each self-consistent
about a field they disagreed on).

### [P61] 🟡 The workspace page could be served from cache while its bundle was two builds old — and the installer is blocked by antivirus (2026-08-21)

**Reported as "I don't see anything here" on a blank `/workspace/`, while the server was serving
a correct page.** `curl` returned the right HTML, both assets 200'd, and the same URL rendered
perfectly in another browser. Two things were true at once and each is a bug:

  * **`/workspace/index.html` was served with no `Cache-Control` at all.** The `/` route has
    carried `no-store` and a paragraph about exactly this since 2026-08-19 — *"genuinely
    installed, genuinely served, verifiable with `curl`, and invisible to the person sitting in
    front of it"*. The workspace UI is mounted through `StaticFiles`, which sends an ETag and
    nothing else, so it never inherited the rule.
  * **The installer never purged the workspace bundle.** `[InstallDelete]` clears
    `static/app/assets` — the line [P46] added — and the workspace bundle, added later, did not
    inherit it. Three builds' bundles were in that directory at once (`index-BZqz4suc.js`,
    `index-CPoEaZR7.js`, `index-BRCSWGK2.js`).

Together they mean a cached `index.html` naming an older hashed bundle **finds that bundle still
on disk and loads it**. Not a 404 — a silent, working, two-builds-old app. **A stale page that
404s is a bug report; a stale page that works is a mystery.**

Both fixed. `_EntryPointNotCached` sets `no-store` on `.html` only, leaving the content-hashed
assets cacheable, which is correct and free. `[InstallDelete]` now covers both asset
directories. Verified on the wire: the page returns `no-store, must-revalidate` and the bundle
returns no `Cache-Control`; after the install the assets directory holds exactly one JS and one
CSS, down from three each. `ui/test_entry_point_is_never_stale.py`, 12 assertions.

`ui/test_install_integrity.py` asserted `count("Type: filesandordirs") == 1` as a proxy for
"scoped to assets" — so it FAILED the fix for the bug it exists to prevent. It now checks the
property it names: every delete ends in `ssets`, and both bundle directories are covered.

---

**AND THE INSTALLER IS BEING QUARANTINED BY ANTIVIRUS, which is the larger finding.**

F-Secure blocked `NEURON-Setup-0.20.18.exe` twice — *"Application blocked"* then *"Harmful file
blocked"* — with `Reason: Drop.Win32.Startup.11003`, and removed `neuron-agent.exe` from the
install directory.

**This also explains something recorded wrongly earlier the same day.** The 0.20.15 install was
described as exit-code-0 with the exe vanishing a minute later, and attributed to Inno's setup
process returning before its child finishes. The event log shows F-Secure firing at **01:26 and
01:28** with the same detection. **The antivirus was the cause; the timing explanation was
wrong.** It is corrected here rather than left standing, because a wrong cause in a handoff is
worse than no cause.

`Drop.Win32.Startup.*` is a heuristic on an unsigned binary that writes a startup entry — which
is exactly what this installer legitimately does. **It is almost certainly a false positive, and
that does not make it less serious.** ROADMAP's One Rule is the first stranger; if F-Secure
flags it on the founder's own machine, a stranger downloading it hits the same wall with far
less patience and no way to tell a false positive from a real one.

The durable fix is **code-signing the installer** — an Authenticode certificate, which is a
purchase and a founder decision, not a code change. Until then every operator needs an exclusion
they should not have to grant. Nothing here has been worked around: disabling or excluding on
this machine is the founder's call, and the node was restored by running the agent from source
instead.

Related: [P46] (the merged-install failure this is the other half of), [P58]/[P54] (a rule
applied in one place and not in the sibling added later — the same shape, three times today).

### [P60] 🟢 The coordinator moved a node's range, the node never learned, and the network served garbage while reporting healthy (2026-08-21, fixed same day)

**[P37]'s open item, live.** Installing 0.20.16 restarted this PC's agent. The coordinator
re-placed the roster while it was away, and the Pavilion's assignment moved from 10-18 to
10-27. **The node was never told.** It kept the slice it had, and served `layers[10:]` over a
skeleton whose layers 19-27 were never materialised.

```
prompt : What colour is the sky on a clear day?
answer :   Sovereberg   Sovere  ABCDEFGHITestCategory  ABCDEFGHI#Endowments  ???e
```

**And every indicator was green.** `/status` said `routable: true`, `stage1_ok: true`,
`network_healthy: true`, 3 nodes, 28/28 covered. Decode was **3.00 tok/s — the fastest figure
this network has ever produced** — because a node running nine layers instead of eighteen is
genuinely quicker. Speed went UP as correctness went to zero, which is worth stating plainly:
on this network throughput is not evidence of anything.

**What found it in one command was [P56]'s own fix**, shipped hours earlier:

```
RangeMismatch: challenged on layers 10-27 (s2=10, n=28) but the node holds 10-18
               -- placement disagreement, not a bad answer
```

Named the fault, named the node, and distinguished placement from compute — which is precisely
the distinction that entry exists to preserve. Repaired with `pin_layers.sh` onto the shape the
nodes actually hold, then a restart so they load it, then re-attested (`passed`, `max_err`
2.8e-05 and 5.5e-05) BEFORE trusting any chat output.

**The gap is the notification, not the placement.** `models.update_layers` writes the new range
to the coordinator's DB and nothing pushes it to the node; the node discovers it only when it
next registers. A restart happens to fix it, which is why every previous occurrence looked like
a transient. Candidate fixes, in order of how little they trust:

  * the node re-reads `slice-info` on a heartbeat and reloads when its range changed ([P37]);
  * `node_server` refuses a config whose range exceeds what `unmaterialized_layers` says it can
    actually serve — it already computes this at load time and then never consults it again. A
    node asked for layers it does not hold should answer `range_mismatch`, which the driver
    already handles as a reroute, instead of computing over meta tensors;
  * the coordinator should not consider a chain routable until every node in it has
    acknowledged its current range.

The second is the one that closes it without trusting the coordinator, the network, or timing —
and it is the one that would have turned this incident into a reroute instead of an answer.

---

**FIXED, and the fix is one missing arm.** `serve()` chose its role in three branches: middle if
the config carries `host_b`, LAST if this node holds the model's final layer and the message is
not a probe, and PROBE for **everything else**. That third branch was a catch-all, and its own
comment justified it with an inference that is false:

> a config with no host_b that ... reaches a node whose own range does NOT include the model's
> final layer can only be a verifier challenging this node in isolation

It can equally be a node the COORDINATOR believes is last while the node knows it is not. So
real pipeline traffic fell into the probe arm, ran `mid_stage` over the layers the node really
had, skipped the final norm, and handed the driver something to run `lm_head` on.

There is now a fourth arm: **real last-stage traffic reaching a node that is not the last stage
is refused with `range_mismatch`**, naming what the node actually holds. Refusing costs nothing
— the driver already turns a refused config into `PeerUnavailable`, which is a `ConnectionError`
and therefore already in `DEAD_PEER`, so it reroutes; and proof-of-compute already reads
`range_mismatch` as placement rather than as a failed challenge, so an honest node with stale
bookkeeping is not flagged for it ([P28]). The sibling guard four lines up refuses
`s2 < self.lo` for exactly this reason — it simply never ran here, because it lives inside the
branch this node was not taking.

**The rule, stated positively: a node answers real pipeline traffic only for a role it can
actually serve.** Anything else is a named refusal, never a plausible-looking tensor.

Proven against the live network, before and after. Before, the OptiPlex — holding 10-18 — was
asked for a last stage and replied `ok: True` while reporting `holds: [10, 18]`. After:

```
ok=False  error='range_mismatch'  holds=[10, 18]
asked to serve the LAST stage from layer 10 to 27, but this node holds 10-18 and does not
hold the model's final layer. Placement here is stale -- re-register or re-place this node;
it cannot apply the final norm.
```

The true last stage still attests clean (`max_err` 5.5e-05) and the network answers correctly.
`agent/test_stale_placement_is_refused_not_answered.py`, 13 assertions, including that nothing
which worked before is refused now — a true last stage, a middle relay, and a verifier's probe
to both a mid-range and a full-model node all still take the arms they always did.

**What is still open under this entry** is the notification itself: the node still learns its
new range only when it next registers ([P37]). This turns that window from *wrong answers* into
*reroutes*, which is the difference between a bug and an outage — but the window remains, and
closing it means the node re-reading `slice-info` on a heartbeat.

Related: [P37] (filed this exact gap and left it open), [P56] (whose verifier diagnosed it),
[P55] (the same symptom from a different cause), [P42] (unmaterialized layers, the check that
exists and is not consulted here).

### [P59] 🟡 A third stage costs a relay round trip per token, and the two nodes paying it share a LAN (2026-08-21)

**Adding `optiplex-server` as a third node dropped the network from ~3.3 tok/s to ~2.1-2.4.**
The first guess was that the new machine was slow. It is not, and the measurement says so:
relay-challenged on nine layers each, with the node's own `c_compute_ms` read off the wire so
compute and network are separated —

| node | layers | compute | per layer |
|---|---|---|---|
| Pavilion (`192.168.1.11`) | 10-18 | 118.5 / 112.4 ms | **13.2 / 12.5 ms** |
| OptiPlex (`192.168.1.10`) | 19-27 | 102.5 / 122.1 ms | **11.4 / 13.6 ms** |

Identical. The `26.9 ms/layer` in the OptiPlex's own log was taken seconds after startup while
it was still loading its slice, which is how a healthy machine acquires a reputation for being
slow. **The hop in those same runs cost 61-98 ms against ~115 ms of compute.**

**Splitting layers cannot make decode faster, and this is arithmetic, not tuning.** Decode is
sequential: every token traverses all 28 layers in order, so two machines running nine layers
each run one AFTER the other, not in parallel.

```
2 stages:  driver 0-9 local -> [hop] -> 18 layers   ~225 ms compute + 1 hop
3 stages:  driver 0-9 local -> [hop] -> 9 -> [hop] -> 9   ~225 ms compute + 2 hops
```

Same compute, one more round trip. 3.3 -> 2.1 tok/s is 303 -> 476 ms per token, and that
+173 ms lands on the measured cost of an added relay leg each way. This is [P30] phase 3's
8x, seen from the other end: **more machines buy CAPACITY — bigger models, more concurrent
users — and never per-user speed, until the model stops fitting on fewer machines.**

**The part that is fixable.** The Pavilion is `192.168.1.11` and the OptiPlex is
`192.168.1.10`. They are in the same house, on the same switch, and the hop between them goes
through the relay in Amsterdam. [P57] measured the relay at **88 ms against 5.3 ms** for a
direct LAN neighbour.

`lan_direct` already exists for exactly this and does not cover this hop. It works
driver->node: the driver names the private /24s it sits on, the node answers with `direct`
only if it holds an address inside one of them, and nothing is published or stored. But a
MIDDLE node dials `host_b` with whatever address the driver was handed by the coordinator —
the relay — and its onward `config` carries no `lan_hint` at all, so the last node is never
asked and can never offer. The feature stops at the first hop because the first hop is the
only one anybody had two machines for.

---

**2026-08-21 — the hop is fixed in code, and blocked by a firewall rule on the last mile.**

A middle node now asks its next hop about the LAN, using the same constructor and the same
trust rule as the driver: `lan_hint` names our own /24s, the peer answers with `direct` only
from inside them, `lan_direct.usable` re-checks the answer against what we asked, and the relay
stays the address of record for every failure. A [P52]-sealed hop is deliberately never
re-dialled — a grant is single-use, and presenting it twice is indistinguishable from a replay.
`agent/test_middle_hop_takes_the_lan.py`, 15 assertions over real sockets against a real
RFC1918 address, so the trust check is doing its job rather than being stubbed.

**And then it made things worse, which is the part worth writing down.** The OptiPlex's `ufw`
scopes port 50999 to `tailscale0`, so the LAN dial to `192.168.1.10:50999` **times out** while
Tailscale connects in 18 ms. The offer arrives on every request — the peer cannot know we
failed to reach it — so the dial was retried every time at `DIRECT_TIMEOUT_S` a go:

| | decode | time to first token |
|---|---|---|
| before | 1.66 tok/s | 2493 ms |
| the "optimisation", first cut | 1.81 tok/s | **~4000 ms** |
| with a memory of failure | **1.85 tok/s** | **2034 ms** |

**A cache of successes is not enough; the failures are what cost.** `lan_direct` now remembers
an unreachable peer for `DIRECT_RETRY_S` (300 s by default) — bounded rather than permanent,
because the reason is usually transient or fixable and a node that gives up forever never
notices. The DRIVER had the same flaw and the same fix; it had simply never met a peer that
offered an address it could not reach.

**The firewall rule went in, and the hop is now on the switch.** `ufw allow from
192.168.1.0/24 to any port 50999 proto tcp`, and the LAN dial went from a 3 s timeout to
**0.2 ms**. Confirmed on the wire rather than inferred — during a live generation the last node
shows `192.168.1.10:50999 <- 192.168.1.11:36734`, the middle node's own LAN address, not the
relay.

| three-stage chain | decode | time to first token |
|---|---|---|
| before any of this | 1.66 tok/s | 2493 ms |
| middle hop on the LAN | **1.90 tok/s** | **1887 ms** |

**+14% and 606 ms off first token, which is ~76 ms per token — one relay round trip, the size
[P57] predicted.** Two stages is still 2.16 tok/s, and that is the point of the entry above:
the third machine costs a hop that splitting the layers does not earn back.

**The remaining relay leg is the DRIVER's, and it is an accident of where the PC is sitting.**
Both nodes are on `192.168.1`; this PC is on a phone hotspot at `192.168.137`, so its own
`lan_hint` matches neither and the first hop still goes to Amsterdam. On the home network that
hop would go direct too, which is worth roughly another 76 ms/token — enough to put the
three-stage chain back level with two. `lan_direct` has been dormant since it shipped for
exactly this reason, and nothing about it needs changing to find out.

Related: [P57] (the bandwidth wall this sits on top of, and the 88 ms/5.3 ms measurement),
[P30] (phase 3, where distribution first measured 8x), [P56] (the three-stage topology exists
because verifying a middle node needed one), [P52] (the grant that stops a sealed hop being
re-dialled).

### [P57] 🔴 The network path is 2.18 tok/s, and two thirds of that is memory bandwidth nobody can optimise away (2026-08-20)

**Measured on the live two-machine chain the day [P55] was fixed**, first honest end-to-end
figures with the network actually returning correct answers: **2.18 tok/s, 119 tokens,
0.1470 NRN**. That is **459 ms per token**, and it decomposes:

| term | cost | share |
|---|---|---|
| driver stage, 10 layers @ 9.67 ms/layer | 97 ms | 21% |
| Pavilion stage, 18 layers @ 11.67 ms/layer | 210 ms | 46% |
| network round trip (through the relay) | ~152 ms | 33% |

**The compute two thirds is a MEMORY BANDWIDTH wall, not a CPU one, and that is the finding.**
This PC streams **36.9 GB/s** (measured: 256 MB fp32 reduction, best of 5). A Qwen2.5-1.5B
decoder layer at fp32 is **187.2 MB of weights**, and decode at batch 1 reads every byte of
every layer for every token:

| storage | per layer | whole model per token | bandwidth floor |
|---|---|---|---|
| fp32 (today) | 187.2 MB | 5.24 GB | 142 ms -> **7.0 tok/s** |
| q4_k_m | 25.7 MB | 0.72 GB | 19.5 ms -> 51.3 tok/s |

So **7.0 tok/s is the hard ceiling of the PyTorch fp32 path on this machine even with a perfect
implementation and no network at all.** Measured local PyTorch is 3.09 tok/s = 44% of that
floor; measured local llama.cpp q4_k_m is 32.11 tok/s = 63% of ITS floor. The 8.7x between the
two engines is not implementation quality. **It is 7.3x fewer bytes crossing the memory bus.**

**Which settles what is and is not worth doing, with numbers rather than intuition.**

Worth doing:
  * **quantization is the only lever on the 67% that is compute.** Nothing else touches it. That
    reframes [P30]'s engine work: it is not an optimisation, it is the only one available.
  * **the relay detour is the only lever available for free.** See below.

NOT worth doing, each ruled out by measurement:
  * **more threads / more cores** — bandwidth-bound. 16 cores here give 9.67 ms/layer against
    the Pavilion's 4 cores at 11.67. A 4x core advantage buys 1.2x, because neither machine is
    waiting on arithmetic.
  * **`NEURON_WEIGHT_DTYPE=fp16`** — halves STORED bytes, but `CastLinear` materialises fp32 on
    every forward and its own docstring says the cast is amortised across a batch. At batch 1
    there is no batch to amortise across, and [P2] already measured half-precision compute ~8x
    slower on these CPUs. A RAM lever, not a speed lever, and it will read like one to the next
    person who tries it.
  * **the wire codec (i8h/f16)** — the 152 ms is latency, not bandwidth. A hidden state is
    `[1,1,1536]` fp32 = 6 KB. There is nothing to compress that matters.
  * **rebalancing layers between the two nodes** — 9.67 vs 11.67 ms/layer is a 20% spread, so
    moving a layer changes the SUM by ~2 ms. The coordinator's `/network/plan` reports
    `speedup_vs_equal: 1.077`, and that is a THROUGHPUT figure (bottleneck = max stage) which
    does not apply to single-answer latency (= sum of stages) at all. Reading it as a latency
    win is a trap.
  * **adding a third machine** — strictly slower. Another hop, and decode stays sequential.

**THE RELAY DETOUR, and it is free to fix.** Both machines sit in the same house and are
directly reachable from each other, yet every activation goes to a 1 GB Oracle free-tier VM in
Amsterdam — which is also the coordinator — and back. `coordinator/main.py:574` does it:
`if body.behind_nat and config.RELAY_ENABLED: tailscale_ip, port = config.RELAY_HOST, relay_port`,
and `behind_nat` **defaults to True** (`agent/agent.py:651`).

Measured with `tools/bench_hop.py` (read-only, a `config` probe answered before any model work):

| path | round trip |
|---|---|
| **this PC's OWN node, dialled through the relay** | **88 ms** median, 124 max |
| Pavilion via relay | 49.5 ms median, 75 max |
| Pavilion direct, LAN `192.168.1.11:50999` | 5.3 ms |
| Pavilion direct, Tailscale `100.79.125.112:50999` | 5.1 ms |

**The first row is the whole argument.** That is this machine talking to a process on itself,
costing 88 ms, because it published a relay address. And these are the FLOOR: a config probe is
a few bytes where a real `act` carries 6 KB each way through a 1 GB shared VM.

The relay also looks like the source of the drops — the 30 s `TimeoutError` that turned one
reply into 0.23 tok/s, and [P52]'s finding that the encrypted hop does not survive the relay at
all (`router.SECURE_HOP_SINCE = (0, 99, 0)` withholds every grant network-wide because of it).

**The trade-off, stated because it is a product decision and not an optimisation.**
`--no-relay` publishes a machine's own Tailscale/LAN address, so only peers on the same tailnet
can reach it. Correct for two machines in one house; wrong the day a stranger joins, which is
the entire point of the project. The durable fix is for the coordinator to publish BOTH and let
a driver prefer the direct address with a short fallback to the relay — that keeps strangers
working and costs LAN peers nothing. Not done here, because the coordinator is on the VM and
deploying to it needs a key passphrase only the founder can enter.

**Where this lands, honestly.** Relay fix: 459 -> ~330 ms/token, about **3 tok/s**. Plus the
[P30] engine: plausibly 5-6 tok/s. Local llama.cpp on one machine: 32.11 tok/s, today, free,
and already the default. **Distribution remains a capacity feature, never a speed one** — and
TOKENOMICS 11.6's "answers under 30 s" needs ~4.3 tok/s for a 128-token reply, which only
quantization reaches.

**THE BANDWIDTH ARGUMENT IS CONFIRMED, AND THE CHEAP WAY THROUGH IT IS CLOSED**
(`tools/bench_quant.py`, 2026-08-20, on the real 19-27 slice). If compute is bytes rather than
arithmetic, cutting bytes per weight must cut latency by about the same factor and nothing else
will. `torch.ao.quantization.quantize_dynamic` to qint8 -- 4x fewer weight bytes, no new binary,
no wire change, no relay change -- measured:

| | ms / decode token | ms / layer |
|---|---|---|
| fp32, what nodes serve today | 75.84 | 8.43 |
| dynamic int8 | **26.45** | **2.94** |

**2.87x, from bytes alone.** The wall is confirmed: it is the memory bus, not the CPU, and
quantization is the only lever on it.

**And the same run closes this route.** `max|drift|` against fp32 is **24.67, a relative 34.78%**
-- not a tolerance question, that is garbage output. Per-tensor int8 destroys the outlier
features transformer activations depend on. llama.cpp's k-quants keep the answer correct at
*four* bits because they carry per-block scales; that difference, not the bit width, is why
[P30]'s engine is the route and this is not.

**A second finding falls out of it, and it outlives this experiment.** 24.67 sits against
`proof_of_compute`'s `atol=0.05`, where an honest fp32 node drifts ~1e-5 and a cheating one ~25.
**A node serving quantized weights is therefore numerically indistinguishable from a node
faking its work.** Today that is the correct answer, because this quantization IS wrong. But the
network's own roadmap needs quantized nodes to be fast, and on the day one joins, the verifier
will flag it. **Proof-of-compute compares against fp32 and has no notion of a declared
precision** -- the challenge would have to be computed at the precision the node advertises, and
the reputation system would have to carry that claim. Nothing in the design does yet. Filed here
rather than as its own entry because it is the same mechanism as [P56]: the verifier measures a
node against a reference that is not what the node is actually being asked to be.

**FIXED WITHOUT PUBLISHING ANYTHING (2026-08-20), because the founder's two constraints were
both right.** Turning the relay off was refused for a reason worth writing down: NEURON's users
will not run Tailscale, so a mesh VPN must never be what makes the network fast -- and **the
relay is a PRIVACY feature**, not merely a NAT workaround. Peers see `150.230.22.250` and never
where a volunteer lives, which is the same decision `/node/list` already encodes when it hides
node addresses from public callers.

So the disclosure is inverted instead of widened (`lan_direct.py`). The DRIVER names the private
/24s it is already on; a node answers **only** from inside one of them:

    driver -> node   config { ..., "lan_hint": ["192.168.1"] }
    node   -> driver ack    { ..., "direct": {"ip": "192.168.1.11", "port": 50999} }

A peer therefore learns nothing it could not have found by scanning its own LAN, a node on any
other network answers with no field at all, and the coordinator neither sees nor stores a home
address -- so **no schema change and no VM deploy**, which matters because deploying to the
coordinator needs a key passphrase only the founder can enter. Rules, each pinned by
`test_lan_direct.py` (45 assertions): RFC1918 only, so a public address can never be offered or
accepted; **100.64/10 excluded on purpose**, so Tailscale is never the route; the caller
re-checks the answer, so a node cannot name some other machine on the caller's network and have
activations sent there; and every failure falls back to the relay, so this can make a request
faster and never break one.

Expected: 459 -> ~320 ms/token, about **3 tok/s**. Needs both ends -- a driver that asks and a
node that answers -- so it is live only once this PC's app is rebuilt AND the Pavilion picks up
the code.

**What is still left, on top of that** (2026-08-20, founder's call -- publishing
Tailscale addresses would make these nodes unreachable to anyone off the tailnet, which is the
opposite of the project's point): nothing free. The 33% network term stays. The 67% compute term
moves only with k-quant weights, which means [P30]'s engine, which needs `llama-server` and
`ggml-rpc-server` binaries that are **not on this machine** -- `llama_cpp` is installed as a
LIBRARY (that is the 32.11 tok/s local path) and ships `llama_cpp/lib/`, but not those two
executables. So the ordering is: get the binaries, settle [P52]'s grant hold so the rpc-bridge
will accept a caller, then wire it.

Related: [P30] (the engine, now demonstrably the ONLY compute lever), [P1] (the usability
floor), [P52] (the relay breaks the encrypted hop, and its hold currently bolts the rpc-bridge
shut), [P2] (half precision is slower on these CPUs), [P56] (the verifier measuring the wrong
reference -- same shape as the precision problem above), [P55] (fixed the correctness that made
these the first measurable numbers).

### [P56] 🟡 Proof-of-compute certifies the PROBE path, and users are served by a different one (2026-08-19, two of three parts fixed 2026-08-21)

**Filed out of [P55], and it is the larger half of it.** For a day the Pavilion returned garbage
to every real request while `challenges_passed` stood at **5662/4**. That number was not broken.
It was a correct measurement of a question nobody was asking.

`security/proof_of_compute.py` challenges a node through `challenge_middle_node` (the PROBE
role) or `challenge_node` (the LAST role on a node it believes is last). A real request arrives
through `neuron_driver` with a different message, and `node_server.serve()` **selects a
different code path from it**. In [P55] those paths ran the same layers and differed only by the
final norm — so the verifier's answer was right, the user's answer was wrong, and no signal
anywhere connected the two.

**Everything downstream of that signal inherits the gap:**

  * reputation, and the probation/promotion machinery that reads it;
  * emission — availability hours are paid on a PASSING challenge, so the network paid this
    node for hours in which it served nothing usable;
  * the operator dashboards, which showed a healthy node because it was healthy at the thing
    being measured.

**What would actually close it.** The verifier has to exercise the path a USER's request takes,
not a path adjacent to it. Concretely: challenge through the same `config` message
`neuron_driver._connect` sends, so the node makes the same role decision it makes in production,
and compare against `common.last_stage`/`mid_stage` accordingly. The challenge inputs are
already deterministic and seeded; what is missing is that the request LOOKS like a request.

**Why this is not simply "add a test".** A test pins today's code. This is a claim the network
makes to strangers — *we detect a node that computes incorrectly* — and it is the mechanism
strangers are asked to trust before lending their hardware. [P55] is the proof that the claim is
currently narrower than it sounds: what is detected is a node that computes incorrectly **for
the verifier**.

**Not fixed here.** [P55]'s fix makes the two paths agree again; it does not make the verifier
able to notice the next time they diverge.

---

**2026-08-21 — two of the three parts are fixed. The third is a measurement, not a decision.**

**One constructor for the `config` message.** `common.stage_config()` is now the only place
that message is shaped, and `neuron_driver._connect`, `challenge_node` and
`challenge_relay_node` all call it. Two callers writing their own dict is two paths waiting to
diverge, and the fix for that is not to keep them in step by hand — it is to make there be one
shape. The last-stage challenge reached the right branch before this only because `s1` happened
to be absent; it now sends `stage: "last"` outright, as a real request does. It still sends no
`wire` field, deliberately: the reply must stay lossless or a quantized codec's ~0.3% error
eats `verify()`'s `atol` budget and gives a cheating node cover. That is safe only because
`wire` is role-INERT, which is now written down where both callers can see it.

**The middle role is verified for the first time.** It was challenged with `probe: True` — a
role no user request produces. Its production role is different code: dial a next hop, forward
what it computed, relay the answer back, none of it ever exercised. `challenge_relay_node`
gives the node a next hop it can really reach, a sink the verifier opens, and grades it on what
it FORWARDED — its own output through the same `_batcher("middle", s1, s2)` that serves users.

**Reachability is not assumed, and the fallback is not silent.** `sink_host` is required and
never guessed; without one, the probe is all that is honestly available. When a relay attempt
fails, the probe runs AND the result records why, because a probe pass is a weaker claim than a
relay pass and reporting them as the same number is the whole of this problem. Every attest
result now carries `path`. A `RangeMismatch` is never retried as a probe — placement is not
compute. A node that never dials the sink is a NAMED refusal rather than a bare `TimeoutError`,
or the verifier's blanket `except Exception` scores the network between us against the node
([P28]).

`security/test_verifier_drives_the_user_path.py` — 26 assertions, the load-bearing one being
that the node's ROLE DECISION is identical for a request and a challenge across every chain
shape. Verified live against the Pavilion: passed, `max_err` 0.000217, `path: last`.

**The relay challenge is now proven against a real middle node (2026-08-21).** It could not be
until there WAS one: two machines make a driver and a last stage and nothing in between, which
is its own comment on how this went unnoticed — the network had no middle node to verify, so
the fact that middle nodes were never verified cost nothing visible. A third machine
(`optiplex-server`) joined and the chain was re-split to `[[0,9],[10,18],[19,27]]`, making the
Pavilion a genuine relay. Challenged on 10-18 through the relay path: **passed, `max_err`
2.8e-05, `path: relay`, 370 ms.** And the three-stage chain serves users correctly —
*"Hello! How can I assist you today?"*, all three nodes in the chain, 0 reroutes.

**STILL OPEN — part three, declared precision.** `verify()` compares against fp32 with
`atol=0.05`, so a legitimately quantized node is numerically indistinguishable from a cheating
one. This needs a tolerance per declared dtype that is **measured** — load the same shard fp32
and in the declared dtype, run the same seeded challenge, read the drift — because a guessed
threshold either flags honest nodes or hands cheats cover. And **the declaration has to cost
something**: if declaring q4 only buys a looser tolerance, every cheat declares q4. It has to
be the same number the node is placed and paid on. Urgent the day [P30] lands, not before.

**And what none of this closes.** A challenge that is DISTINGUISHABLE from real traffic can
always be special-cased by a node that wants to. Parts one and two narrow the gap; they do not
close it in principle. The endgame is verification by REPLICATION: the coordinator sends one
sampled real request down two independent chains and compares. It never needs to know the right
answer — only that two machines disagree. It is the only version a node cannot detect, because
there is nothing to detect. Cost is one duplicated request per sample.

Related: [P55] (the divergence that exposed this), [P47] (a node the verifier skipped
entirely, so it could not earn), [P16] (placement, which reads the same reputation signal),
[P30] (whose quantized nodes part three is a precondition for).

### [P54] 🟢 [P52] shipped, was tested, was proven against a real relay — and had never once executed (2026-08-19)

**The network path raised `NameError` on every single request, and nobody knew for a day.**

```
503  {"code":"chain_unavailable",
      "message":"NameError: name 'wire_crypto' is not defined"}
```

`neuron_driver._connect()` calls `wire_crypto.client_handshake(...)` and catches
`wire_crypto.HandshakeError`. **Nothing imported the name.** `base64` was added on the line
above it for the grant decode; the module itself never was. So the first hop the coordinator
minted a grant for died — and since the coordinator mints one for every hop, that is every
request, from the moment [P52] shipped.

**Why a day passed with the product's core path dead.** Local-first execution answers almost
everything on the user's own machine (`ui/app.py` prefers `best_local_model()`, [P29]). The
driver path therefore essentially never runs on a machine that can hold the model — so the
encryption shipped, passed its tests, was demonstrated against a byte-recording relay, and was
never executed by the product even once. **Every token the founder had ever seen in the chat
window came from their own PC.**

**Four registrations were missing for one new package, and each failed silently in its own
way.** `security/` was created by [P52]. Nothing in this repo checks that a new top-level
package is wired into all the places that must carry it:

| where | what happened | how it presented |
|---|---|---|
| `neuron_driver.py` | no import at all | `NameError` at runtime, only on the path nobody exercised |
| `coordinator/deploy.sh` `FILES` | `security/` not shipped | live coordinator in a **systemd restart loop**, `/status` 502, whole network unroutable — while `systemctl is-active` answered *"activating"* |
| `coordinator/requirements.txt` | no `cryptography` | same outage, one layer down, after the first was fixed |
| `packaging/neuron-agent.spec` `hiddenimports` | package not frozen | a build that succeeds and an app whose NETWORK path fails while local answers keep working, so every smoke test passes |

**The lesson is not "remember four places".** It is that a missing registration fails at RUNTIME
on a path nobody exercises, and that is indistinguishable from working. Three of the four were
found only by deliberately forcing a request onto the network.

**And the instrument for doing that was itself broken**, which is the sharpest part.
`NEURON_FORCE_NETWORK=1` was honoured in `ui/app.py:_drive` and **not** in
`api/openai_compat.py`, which branches straight on `local_gguf.available()`. So the first
attempt to measure end-to-end network speed silently measured the LOCAL engine and reported
**4.97 tok/s** — about 3x too good. A broken measuring instrument does not look like a failure;
it looks like good news. It was caught only because the coordinator's `requests_served` counter
did not move.

**Fixed:**
  * the import, and `test_encrypted_hop_is_reachable.py` asserts by PARSING rather than
    importing that every module using `wire_crypto` imports it, that the spec declares the
    package, that `deploy.sh` ships it and that `requirements.txt` pins `cryptography`;
  * `local_gguf.network_forced()` — the override now lives where every caller that asks "can
    this machine serve it itself" inherits it, and outranks `NEURON_LOCAL_MODEL`, which answers
    *which* model to run locally and must not smuggle a request back onto this machine. 13 cases
    in `engine/test_force_network.py`;
  * shipped as **0.20.7**. 0.20.6 and everything before it have a network path that cannot
    answer, so this is the first installer worth giving to anyone.

**Then the hop failed for a second, unrelated reason, and it is the other half of [P52]'s
rolling upgrade.** With the import fixed, every request to `node-c-pavilion` (0.20.3) died with
*"socket closed during handshake"*. [P52] handled old-coordinator/new-node: no grant arrives,
the node accepts plaintext, nothing breaks. **The reverse was never handled** — a NEW
coordinator mints a grant for an OLD node, the driver opens a handshake the node has never
heard of, the node closes the socket, and the driver correctly treats that as an identity
failure and reroutes. With no replica to reroute to, the request simply fails. A version
mismatch wearing the costume of a network fault, which the coordinator had `agent_version` on
hand to avoid. `router._speaks_secure_hop` now withholds the grant below 0.20.5, and an
UNKNOWN version counts as too old — the node that will not say what it runs is the one not to
assume about.

**Measured after both fixes, with the coordinator's counter as the witness:** `requests_served`
84 → 85, **1.36 tok/s** end to end. That is the first honest network number this project has
ever had, because it is the first one taken while the network could actually answer.

Related: [P52] (the channel), [P29] (local-first, which hid this), [P51] (a failure with no
error, same shape one layer up), [P46] (a build that verifies itself and an install that does
not).

### [P55] 🟢 The network returned GARBAGE and billed for it, while proof-of-compute reported it healthy — the last node was answering a verifier's question (2026-08-19)

**Same model, same prompt, two paths, measured minutes apart:**

| path | output | tokens | cost |
|---|---|---|---|
| local (`engine/local_gguf`) | `Hello! How can I assist you today?` | 9 | 0.0000 NRN |
| network (driver + node-c-pavilion) | `  1  2   3` then whitespace to the cap | 128 | **0.1330 NRN** |

The model and tokenizer are fine — the local path proves that in the same process. **The
distributed pipeline is producing wrong output**, running to the 128-token cap emitting
whitespace, and every request is billed for it. A user gets nonsense and pays ~0.13 NRN.

**What makes it worse than a wrong answer.** `challenges_passed` is 5662/4 on the Pavilion and
508/0 on the driver. **Proof-of-compute — the mechanism whose entire purpose is catching a node
that computes incorrectly — reports both nodes healthy while the chain they form returns
garbage.** Whatever the challenge checks, it does not check the thing that matters. That is a
bigger finding than the corruption itself: the reputation system, the emission that pays on it,
and the operator dashboards are all downstream of a signal that just failed silently.

**Ruled out, by measurement rather than reasoning:**

  * **the wire** — `common.py` and `wire_codec.py` are byte-identical on both machines
    (same md5), and neither has changed since before the running build;
  * **weight dtype** — both nodes report `fp32` to the coordinator, and the slice on disk is
    BF16, which is simply how Qwen2.5 ships;
  * **layer ranges** — assigned and reported agree on both nodes: 0-9 and 10-27, no gap;
  * **the model id** — both report `Qwen/Qwen2.5-1.5B-Instruct`;
  * **encryption** — grants are currently withheld ([P52] hold), so this hop is plaintext, the
    same wire that worked earlier today.

**It is a REGRESSION, not a standing flaw.** Earlier the same day the same two machines answered
coherently over the network twice: *"Sure, I'm ready to help with any questions you have."*
(1.36 tok/s) and *"Hello! How can I assist you today?"* (1.26 tok/s). Between those and this,
the Pavilion re-downloaded its slice during a model migration and now serves 10-27 out of a
**full-model** slice — `neuron_slice.json` records `layer_start: 0, layer_end: 27` and all 338
tensors are present. A node serving a sub-range out of a superset slice is the one condition
that changed and is not covered by any test.

**BISECTED, 2026-08-19. Every component is correct and the whole is wrong, which narrows it
to one place.** Measured, in this order, each ruling out a suspect:

  * **the sharding math is bit-exact.** `selftest_shard.py`: `max|delta| = 0.000e+00` and it
    decodes *"Hello! How can I assist you today"*. Layer splitting, norm placement, `lm_head`
    and the KV cache are all correct IN PROCESS.
  * **the chain shapes are correct.** `test_short_chain.py`, 13 pass, including that a 2-stage
    chain routes and that `s2` marks where the last stage begins.
  * **the config on the wire is correct.** Captured live: `{'s1': 10, 's2': 10}`, and the last
    stage runs `layers[s2:]` = 10-27. Not the empty range it looks like.
  * **the wire codec is irrelevant.** Forcing `f32`, `f16` and `i8h` in turn produced
    byte-identical garbage — so the corruption is deterministic and upstream of quantization.
  * **the remote weights are identical.** Layer-10 tensor sums match the reference to four
    decimals (`1416.5790`, `41.1904`, `-264.7318`).
  * **the remote COMPUTE is correct in isolation.** Running layers 10-27 + norm on a fixed
    seeded input, on the Pavilion with its own slice and its own loader: `sum=-314.9242`
    against the reference's `-314.9192`. A 1.6e-5 relative difference — bf16 rounding, nothing
    more.

**And then the measurement that matters.** Captured the hidden state the driver actually put on
the wire for a real request, captured what came back, and computed locally what that reply
should have been from that exact input:

| | sum of the returned hidden |
|---|---|
| what the node returned | **2358.6965** |
| what `layers[10:]` + norm produce from the same input | **1145.9401** |
| `max|difference|` | **247.09** |

**So the node returns the wrong thing on the LIVE path while computing correctly in process.**
The remaining difference between the two is that serving goes through the BATCHED
implementation — `batching.last_stage_batched` / `run_layers_batched` with a padded batch cache
— rather than `common.last_stage`. That is where to look, and it is the only place left.

It also explains why proof-of-compute never noticed: whatever the challenge exercises, it is not
the path that real requests take. Filed as [P56], which outlives this bug.

**The next step is a bisect, not more inspection.** Point the driver at a `node_server` running
locally on the driver's own machine for layers 10-27, and compare:
  * coherent -> the Pavilion's slice/compute is at fault, and the full-model-slice path is the
    first suspect;
  * still garbage -> the driver's own half (embed, layers 0-9, `lm_head`, or the norm placement
    when there are exactly two stages) is at fault.

**A COMPLETELY FRESH INSTALL REPRODUCES IT, which changes what this is.** The Pavilion was
wiped and rebuilt from nothing on 2026-08-19: NEURON deleted entirely, a new venv
(`torch 2.4.1+cpu`, `transformers 4.44.2`, `cryptography`), the current code, a brand-new node
identity (`agent-raman-hp-pavilion-laptop-15-eh3xxx-e4920b`) and a freshly downloaded, correctly
ranged slice. The output is **byte-identical garbage**: `'  1  2   3  '`.

So it is NOT the stale full-model slice, NOT accumulated machine state, and NOT damage from a
session's churn. **It is a code regression**, reproducible from a clean machine.

**And the version correlation names the window.** Same driver build throughout the day:

| the NODE's agent version | what the network returned |
|---|---|
| 0.20.3 | correct — *"Hello! How can I assist you today?"*, 1.26 tok/s |
| 0.20.8 | `'  1  2   3  '` |

So the regression is in `agent/node_server.py` between those two builds. The candidates, from
`git log`, are [P52]'s two commits that rewired the serve path (`d03fbc4`, `15499ab`), [P49]'s
full-model-node fix (`6a30df0`) and [P42]'s slice guards.

**The obvious bisect does not work and that is worth recording** so nobody repeats it: dropping
the pre-[P52] `node_server.py` onto a current checkout fails with *"socket closed mid-message"*
— the old serve loop is not compatible with the current driver and `common.py`. Bisecting this
needs matched pairs (driver and node from the same commit), not a single file swapped.

**Containment while it was unfixed:** local-first is the default and the "Use the network"
toggle is off unless ticked, so an ordinary user got the correct local answer. Anyone who ticked
it got nonsense and was charged.

**FOUND AND FIXED (2026-08-19, 0.20.9). The last node was running the PROBE role — the same
layers, without the final norm.**

`agent/node_server.py` picks a node's role from the `config` message, and it has to, because
`is_true_last` asks what the node HOLDS: a machine holding the model's final layer is "true
last" for every question anyone ever asks it, including a verifier's challenge about some other
range. [P49] fixed one direction of that by reading the mere PRESENCE of `s1` as "a verifier is
probing me":

```python
elif is_true_last and "s1" not in msg:      # LAST stage role
```

**`neuron_driver._connect` sends `s1` on every config it ever writes.** In a THREE-stage chain
that config goes to the middle node, which forwards `{"s2", "n", "wire"}` and no `s1`, so the
last node was reached correctly — which is why the network answered coherently earlier the
same day. In a TWO-stage chain — driver 0-9, one node 10-27, `host_b` absent, **the shape
the live network has been running since the re-split** — the driver *is* the previous stage,
its `s1` goes straight to the last node, and every real request fell into the probe branch:

| what ran | what it computes |
|---|---|
| what should have run | `last_stage_batched(model, 10)` = `layers[10:]` **+ `model.model.norm`** |
| what actually ran | `mid_stage_batched(model, 10, 28)` = `layers[10:]`, **no norm** |

Same layers, same weights, same wire, same cache — which is why every single thing measured
during the bisect was correct. The driver then applied `lm_head` to an **un-normed** hidden
state, whose values are several times larger than anything the head has ever seen, and got
`'  1  2   3  '` and whitespace to the 128-token cap.

**That is the 2358.6965.** Reproduced end to end, real weights, real socket, a node holding
19-27 driven with the exact config the driver puts on the wire (`s1 == s2 == 19`), seeded input:

| | sum of the returned hidden | max|diff| vs the node's answer |
|---|---|---|
| before the fix | **-966.1324** | `layers[19:]` **without** norm: **0.0** |
| after the fix | **-627.7026** | `layers[19:]` **with** norm: **0.000e+00** |

Bit-exact against the un-normed computation before, bit-exact against the normed one after.
The 247.09 in the bisect above is this, at the live chain's split point.

**Why proof-of-compute could not see it, and never will have been able to.** The probe path is
the ONLY path a challenge exercises. The node was answering every challenge perfectly — it
was answering *real requests* with the challenge's computation. `challenges_passed` 5662/4 was
not a broken signal; it was a correct signal about a different question. **A verifier that only
ever asks the probe question cannot certify the serve path.** That is the finding to carry
forward and it outlives this bug, so it is filed on its own as **[P56]** — this fix makes the
two paths agree again, it does not make the verifier able to notice the next divergence.

**Fixed in two places, and which one matters depends on which machine you can update.**

**1. The driver stops sending a field its recipient cannot use.** `s1` is where the DRIVER's own
layers stop, and only a middle relay has ever read it (`layers[s1:s2]`); a last stage's own start
is implied by `s2`. So `neuron_driver._connect` now sets `s1` **only inside the `host_b`
branch**. That is the half that repairs the live network **with no node update at all** — a
Pavilion still running 0.20.8 selects its last-stage branch with `is_true_last and "s1" not in
msg`, and a config with no `s1` satisfies it. One rebuild on the driver's own machine, and the
answer is correct against every node already installed.

**2. The node reads what `s1` MEANS rather than whether it is there**
(`node_server._is_range_probe`), which covers the other direction — drivers already in the field
that still send it:

  * a **chain junction** has `s1 == s2` — the driver owns `0..s1-1` and the next stage begins
    at `s2`, and a chain with no gap makes them equal at any depth. Real traffic.
  * a **challenge** has `s1 < s2` — `challenge_middle_node` asks about a non-empty range.
  * `stage: "last"` (driver) and `probe: true` (verifier) now state the intent outright, so the
    next reader does not have to re-derive it.

**Neither half needs the other, and that is deliberate.** The driver and the nodes update on
their owners' schedules, never together, so a fix that required both would have left the network
returning garbage until the slowest volunteer restarted. Every combination of old and new on
either end now answers correctly.

**What actually caught it, and what did not.** `selftest_shard.py` (bit-exact), `test_batching`
(batched == unbatched), `test_short_chain` (13 pass) and proof-of-compute were all green
throughout, because every one of them tests a COMPONENT or a MESSAGE SHAPE. Nothing anywhere
drove `node_server.serve()` with the message `neuron_driver` really sends. That is now
`agent/test_last_stage_is_not_a_probe.py` (26 assertions): each real caller's verbatim config
through a live `serve()`, asserting which role it selects, for a 10-27 node, a full-model node
and a mid-range node, with and without the new explicit fields — plus a check against the
literal 0.20.8 rule, since the machine that runs it cannot be imported, and a check that the two
roles differ by the norm, so picking wrong is a wrong answer and not a rounding error. It fails
on the pre-fix code with exactly the failures that describe [P55].

**The generalisable lesson, and it is the third time this project has paid for it.** [P54] was
a network path nothing exercised. [P47] was a driver nothing verified. This is a role nothing
asserted. **Every one of them was a seam between two components, each correct, tested only from
one side.** A test that sends a component the message its real caller sends is worth more than
a test that sends it a message the test author designed.

Related: [P56] (proof-of-compute measured the wrong path and could not have known — filed out
of this), [P49] (the other direction of exactly this discriminator),
[P54] (a path nothing exercised),
[P30] (the engine work this blocked), [P42] (a slice that serves a range it does not fully
hold — the near neighbour of this).

### [P53] 🟢 A node that bound its own payout key could never be claimed through the UI — found on the first real claim, fixed (2026-08-19)

**The first genuine execution of connect → sign → bind failed, and it fails for every node that
has been running long enough to matter.** The founder signed in on this machine (391.39 NRN,
`raman011sharma@gmail.com`), pressed **Claim these earnings**, and got:

> *"this node already pays out to 0x29772e94d9D31287C10032Fd9dF2b3C4E9af7ff2. Changing it needs
> `old_signature`: the same message signed by that address's key. If the key is lost, the
> operator must rebind with the register secret."*

**The mechanism, and why it is structural rather than a mishap.** `ui/app.py`'s
`/node/payout/bind` binds the payout ADDRESS and records the OWNER **in one call** —
deliberately, so a copied `node_token` cannot move ownership without a signature. But
`agent/payout_key.py:ensure_bound` has already generated a key for this node and bound it,
automatically, on an earlier start. So by the time anybody signs in to claim, an address is
always on file, the claim's own call is therefore a *rebind*, and
`coordinator/payout.py:require_rebind_authority` correctly refuses it without `old_signature`.

**Every self-hosted node reaches this state on its own, unprompted.** Nothing the operator did
caused it. The two behaviours — the agent binds a key by itself, and claiming re-binds — are
each correct alone and together make the feature unreachable. That is a second reason zero nodes
on the live network have an owner recorded, alongside the `needs_owner` gating in [P39].

**The advice in the error is wrong here, and that is the sharp part.** It tells the operator the
key may be *lost* and to go and find the register secret. The key is not lost: it is in
`payout_key.json` in the agent's own state directory on the very machine displaying the message,
and it is the key for exactly the address the message names. The product held the answer and
told the user to seek an administrator. Verified this session: the local key's address is
`0x29772e...af7ff2`, identical to the bound one.

**The fix is small and the coordinator already permits it.** `require_rebind_authority` exempts
a bind to the same address — `if not current_address or current_address.lower() ==
new_address.lower(): return`. So when this machine holds the key for the address already bound,
the claim can re-bind that same address, signed locally by `payout_key.sign_binding`, carrying
`owner_wallet_id` from the session. Nothing about where the money goes changes; only who is
recorded as owning it. No `old_signature`, no register secret, one click.

  * **Do NOT let the claim silently rebind to a DIFFERENT address.** The `old_signature`
    requirement is the control that stops a stolen `node_token` redirecting earnings, and it
    must stay exactly as strict for a genuine address change.
  * **The error text needs its third case.** It offers "sign with the old key" and "the operator
    has the register secret" and omits the common one: *this machine still has the key — keep
    the address and just record the owner.*

**Still open:** the fix touches the packaged UI, which the desktop app serves from its own
bundle, so it needs a rebuild before any operator sees it.

**EXECUTED ON THE LIVE NETWORK, 2026-08-19 09:07:23** — the first successful claim in this
project's history, after 65 sessions in which the count was zero:

```
neuron.ui node agent-optinovate-6ff49d claimed by the signed-in account
          (address unchanged: 0x29772e94d9D31287C10032Fd9dF2b3C4E9af7ff2)
```

One click, no wallet extension, no signature prompt, and the payout address is byte-identical
before and after — which is the whole design: the claim records an owner, it does not move
money. `owner_wallet_id` on that node is no longer null.

**And the accrued balance was swept separately, because recording an owner does not move what
was already earned.** `coordinator/claim_node_earnings.py` on the VM, dry run first:

  * `agent-optinovate-7fc2ff` -> **26.39 NRN**. This is the orphan the reinstall created. Nobody
    could ever have signed in as it; without the sweep that NRN was simply gone, and the only
    reason it was findable is that this session happened to still know the id.
  * `agent-optinovate-6ff49d` -> **35.82 NRN** (grown from 33.49 while the fix was being built).

62.21 NRN recovered, wallet 391.39 -> 453.60, supply invariant `1000000000.0000001` verified
intact after each, DB backed up before each, both appended to `claim_log.json`. `total_earned`
correctly did NOT fall on the node rows — the network really did distribute that NRN for
compute, and a later transfer does not un-earn it.

**What this changes for the next node, and it is the point of the whole entry:** a claimed node
credits its owner's wallet from here on ([P39]), so the sweep is a one-time repair rather than a
recurring chore. The check that proves it is cheap — watch the node's own balance stay at 0.0
while the wallet rises.

**FIXED (2026-08-19, 0.20.6) — and the fix is that ownership stops depending on a wallet at
all.** The founder's words, after the claim failed on their own machine: *"I asked you to
assign node with github or google ID of a user, not this."* That is the correct requirement and
the old flow never met it — `claimNodeEarnings` opens a browser extension, signs with whatever
address that extension holds, and binds THAT. A person with a Google account and no MetaMask
could not claim anything, and a person with MetaMask claimed with an address that was, by
construction, not the one already bound.

`POST /node/claim` (`ui/app.py`) does the whole thing with no wallet in the picture: read the
address already bound, confirm this machine holds its key, sign the challenge locally, and
re-bind the SAME address carrying `owner_wallet_id` from the session. The payout address never
moves. The button is now **"Claim with my account"**, and the wallet path is demoted to *"Pay
out to a different wallet instead"* for the operator who genuinely wants an external address.

**What was deliberately NOT relaxed.** A bound address this machine has no key for is refused
with a 409 that says so. That case is a real address change, and it must keep needing the
incumbent key — a claim endpoint able to override it would be exactly the bypass the
`old_signature` control exists to prevent. Asserted, not assumed:
`ui/test_node_owner_ui.py` (26 pass) pins that the owner comes from the SESSION, that no
`old_signature` is sent, that a foreign bound address is refused **and nothing is POSTed**, and
that a missing local key is an honest 409 rather than a crash. `nodeOwner.test.ts` (69 pass)
pins that the browser sends an empty body, never a wallet id, and never touches
`window.ethereum`.

**And the reason this matters more than the claim button.** The same day, the founder
uninstalled and reinstalled the app. `new_node_id()` mints `agent-{hostname}-{random6}` whenever
the config has no `node_id`, and the uninstaller deletes the config — so the machine came back
as `agent-optinovate-7fc2ff` and **`agent-optinovate-6ff49d` was left holding 33.49 NRN across
41 served requests, unreachable by its own owner.** A claimed node does not have that problem:
the earnings are recorded against an account that survives the disk. Identity churn is
survivable; unclaimed identity churn is not. That is the argument for making the claim reachable
on day one rather than treating it as a later nicety.

### [P48] 🟡 Three things the operator sees are wrong or stale, and each looked fine from inside the repo (2026-08-17)

All three found by the founder **looking at the live product** rather than at the code. That is
the pattern of [P46] and it is worth naming again: every one of these is invisible from a
passing test suite, because the repo is self-consistent and the thing that is wrong is the
relationship between the repo and what a person is actually served.

**1. A node AHEAD of the network was told to downgrade — fixed.** The founder's own dashboard
read **"v0.20.4 — v0.20.3 is available"**. `main.py` compared with `av == latest`, which answers
*same or different* and was being read as *current or behind*, so any difference rendered as an
upgrade prompt — including a node newer than the network. It also promised "nodes check once a
day", which that node will never act on because there is nothing above it to install.

Both sides are ordinary and the fix deliberately diagnoses neither: 0.20.3 is the latest
PUBLISHED release and the coordinator advertises it correctly, while the founder's machine runs
a locally-built 0.20.4 that was never released — and the mirror case, a coordinator whose
`NEURON_AGENT_VERSION` pin went stale behind a shadowed systemd drop-in, is Session 61 and just
as real. Naming either as the fault would be a guess printed as a finding.

`_version_gt` compares numerically, because swapping `==` for `>` would only have moved the
bug: `"0.20.10" > "0.20.9"` is **False** as strings, so the lexical fix breaks at the tenth
patch release. An unparseable version is never "ahead" — inventing certainty about a build we
cannot read is what `unknown` is bucketed separately to avoid. The same equality shape was in
the public rollout counter (`on_latest`) and the `stuck` list, so a fully-updated fleet read as
behind the moment a release pin went stale; both fixed.

**2. Every download link pointed two releases back — fixed.** `docs/index.html` (twice) and
`README.md` sent visitors to **v0.20.2** while **v0.20.3** was the published release. Silent by
construction: a stale link still *works*, serving an older installer perfectly happily, so
nothing fails and every new volunteer gets the old build. The opposite mistake is one keystroke
away and worse — pointing at v0.20.4, which is built but was never released, 404s for everyone.
`test_download_links.py` now asserts every link names one version and that the version has
release notes in the repo, the cheapest offline proxy for "actually released".

**3. 0.20.4's headline feature is on a route nobody visits — NOT fixed, and it is the
interesting one.** The founder expected `≈ N network answers left` and saw
`NEURON v0.20.4 — 2.99 NRN`. Nothing is broken: that line lives in
`ui/web/src/components/Sidebar.tsx`, which is the React app served at **`/next`**, while `/`
still serves `ui/static/chat.html`. So the entire wallet UI that 0.20.4 exists to ship is
invisible to anyone using the default route, which is everyone.

**And it is worse than a shipping problem — it is the money-safety blocker (2026-08-17).** The
founder put it exactly: *"the NRN distribution is a matter of ID, not the node — even if a person
removes NEURON from the PC, his balance should come back to him."* That is the right design and
emission already reaches for it: `close_slots` pays `get_node_owner(node) or node`, so a node
with an OWNER recorded pays into the wallet that person logs into, and uninstalling costs them
nothing.

**Nobody can bind an owner, because the panel that does it is on this route.** Measured on the
live ledger:

| account | balance | owner |
|---|---|---|
| `node-c-pavilion` | 114.799539 NRN | **none** |
| `agent-optinovate-6ff49d` | 5.259616 NRN | **none** |
| `agent-bhpc012101-18f1da` | 0.051484 NRN | **none** |

**120.11 NRN, and zero nodes with an owner bound.** Every one of those balances is protected by
nothing but a `node_token` in one config.json on one disk — and [P50] is that file being
*rotated*, not even deleted, and coming within one write of orphaning 5.26 NRN. The uninstaller
would do it deliberately.

So this route swap is not "the wallet UI is on the wrong URL". It is the reason a volunteer's
earnings live on a machine instead of in an account.

**What the swap actually costs, audited properly (2026-08-18).** A first pass over this claimed
the React app was missing five things and that swapping would regress every user. That was
wrong — it grepped for chat.html's *identifiers* rather than for the behaviours, so anything
React spells differently read as absent. Checked again against the source, React **has** the
wallet-id reveal, the node-owner claim, payout binding, `insufficient_funds`, reroute-is-not-an-
error, partial-answer survival, the local-vs-network header (`localCapable`), and low balance —
as `≈ N network answers left · contribute this machine to earn more`, in the sidebar rather than
as a strip. It has a token cap and a **better** one (`maxTokens` per thread, adjustable, not a
constant). It also has personas, per-thread settings and speech input, none of which chat.html
has.

**All three closed (2026-08-18), and the swap is now unblocked.** `blockReason` and
`degradedNotice` are pure functions in `services/wallet.ts`, so the rule is testable without a
DOM; `services/update.ts` renders what `/app/update` decides. Verified against the built bundle
rather than only in tests — and the first attempt at that was wrong, which is worth recording:
setting `textarea.value` directly does not reach React's state, so an early "Send is disabled"
reading proved nothing. Driven through the native value setter against a stub serving each
state: **healthy + text → Send enabled, degraded + text → Send disabled**, with the banner
reading *"19/28 model layers are online (missing 10–12, 27)"*.

**And the claim panel is reachable in React too (2026-08-18).** It had the identical gate one
implementation over — `fetchNodeOwner` set `needsOwner = is_node && needs_owner`, which is false
whenever nobody is signed in, so a logged-out operator got a blank sidebar. It now renders on
`unclaimed`, states the stake and offers the sign-in links. Verified in the built bundle against
the live shape: *"THIS COMPUTER'S EARNINGS — This computer is earning NRN. Sign in to…"*.

**So the swap is no longer blocked by anything.** `/next` is at parity on everything that
matters and ahead on personas, per-thread settings and speech. What is left is the decision to
make it `/`, and porting the 59 source-text assertions in `ui/test_chat_ui.py` — which were
never the blocker, only a bad proxy for one; the behaviours they defend are now covered by
vitest against pure functions.

The three gaps as they were:

  1. **It will send into a chain that cannot answer.** `canSend` in `ChatInput.tsx` is
     `(text || attachments) && !isGenerating` — no health term at all, while chat.html has
     `setBlocked` driven by `!n.healthy && !localCapable`. So on a degraded network React lets
     you type, send, and collect a failure, where the old page says up front that it cannot go.
     That is the one item that must land before the swap.
  2. No degraded banner naming the uncovered layers (React shows a health dot only).
  3. No update notice — added to chat.html on 2026-08-18, not yet in React.

So this is a **three-item** job, not a rewrite, and the 59 assertions were never the blocker:
they read chat.html's HTML strings, so they were a proxy for parity and a bad one. Port the
behaviours that matter, close (1) especially, and swap.

That makes the `/next` → `/` swap a **shipping** problem rather than the tidy-up it has been
filed as. It is blocked on porting 59 source-text assertions in `ui/test_chat_ui.py` — tests
that assert on chat.html's HTML strings, which is exactly why they cannot follow the behaviour
to a different implementation. Same root as Session 61's finding that the claim panel's tests
stubbed `window.ethereum` and asserted on source text, so connect → sign → bind had never
actually executed. **A release note describing a feature no default user can reach is [P31]'s
mistake wearing different clothes.**

Related: [P46] (a build that was self-consistent in the repo and broken once installed),
[P39] (the claim panel, on the same unswapped route).

### [P52] 🔴 The site advertises "privacy by architecture" and the pipeline wire is plaintext across strangers' machines (2026-08-18)

**Found by the founder asking a one-line question — "so are we making every machine encrypted or
not?" — and then checking what the product already promises.**

`docs/index.html` says **"privacy-preserving"**, and puts a **tick against "Privacy by
architecture"** in a comparison table where ChatGPT, Claude and Gemini all get a cross. The
pitch quote reads *"private by architecture, no single operator can withdraw access"*.

**What is actually true, split cleanly in two:**

  * **Local inference is genuinely private.** When the machine can hold the model it answers
    itself (`engine/local_gguf.py`) and nothing leaves it. `README.md`'s *"Nothing you type
    leaves your computer"* is scoped to that case one line later, and it is honest.
  * **Networked inference is not.** The driver tokenises and embeds the prompt, then hidden
    states cross the internet as **plaintext** — a JSON header plus raw tensor bytes
    (`wire_codec.py`) — through a **public relay** and into **strangers' machines**. There is no
    TLS on that path, and `node_server.py` does not authenticate the caller, so a node cannot
    even tell the difference between the driver and anyone who dialled its published port.

And networked inference is *the product*: running models too large for one machine is the
sentence the site leads with.

**The hazard is not "someone sees numbers".** Hidden states are derived from the prompt and must
not be treated as opaque — inverting embeddings and intermediate activations back to text is an
active and productive research area, and the honest engineering posture is that anything derived
from user text carries user text until proven otherwise. This entry does not claim a specific
inversion attack has been demonstrated against this wire; it claims the opposite of what the
site does — that **plaintext prompt-derived data crossing untrusted machines cannot be sold as
privacy by architecture.**

**Who is exposed, concretely:** every operator whose machine sits in a chain, anyone able to
observe the relay (today the founder's own VM, tomorrow whoever runs one), and any network
between them. `SECURITY.md` already argues the wire is safe to EXPOSE — it carries nothing
executable and is size-capped — and that argument is about *the node's* safety, not *the user's*
confidentiality. Those are different properties and the site claims the second.

**Why 🔴.** It is the one class of defect this project has repeatedly caught in itself
([P31]: a capability shipped, documented, never executed) and it is worse here because it is
advertised competitively, against named products, to recruit strangers. A volunteer joining on
that promise is also a person whose own prompts travel this way.

**What would close it, cheapest first:**

  1. **Say what is true, today.** The tick becomes a qualified claim: private when the model runs
     on your machine, and in transit across the network it is not yet encrypted. One edit,
     removes the false part immediately, and costs nothing but the sentence.
  2. **Encrypt the node wire.** The transport already frames messages (`wire_codec.py`), so this
     is a session key and an AEAD around the existing frame rather than a redesign. The
     coordinator already issues per-node tokens and mints relay tickets, so there is a key
     distribution point that exists.
  3. **Authenticate the caller.** Encryption without knowing who is on the other end still lets
     anyone with the published port open a session. This is the same gap that makes ggml-rpc
     unusable on the relay (see the 2026-08-18 engine decision), so both wants converge on one
     piece of work.
  4. **Then, and only then, the architectural claim is defensible** — and a stronger version
     becomes available, because no single node ever holds the whole model or the whole
     conversation, which is a real structural privacy argument that today's plaintext wire
     undercuts.

Related: [P19] (the same wire, the node's safety rather than the user's), [P31] (documented and
never executed), the 2026-08-18 engine decision (which needs items 2 and 3 for its own reasons).

### [P51] 🔴 The agent went mute: alive, listening, answering challenges — and unregistered for 81 minutes with nothing saying so (2026-08-18)

**Found by asking "is it OK?" and looking, not by any alarm.** The Windows PC slept overnight
and woke around 07:37. Everything auto-started: the growth bot (07:37), `neuron-agent.exe`
(07:38:30), `verify_service.py` (07:40). Then:

| | |
|---|---|
| process | alive, `Responding=True`, 46 threads, 278s CPU and climbing, 7.1 GB resident |
| port 50999 | listening, and answering a probe correctly — `{"ok": true, "holds": [0, 9]}` |
| Chat UI :8080 | up, serving |
| `agent.log` | **last line 07:33:09, from the PREVIOUS process. Zero lines in 81 minutes.** |
| coordinator | `status=offline`, `placement_drift=true` |

So the node was doing real work and was invisible. `auto_repair` reacted correctly to what it
could see and re-placed `node-c-pavilion` onto all 28 layers; `neuron_doctor` now reports the
whole network `NOT HEALTHY — no model tier is in a serving state`.

**What is proven, and it is a real defect on its own: the Chat UI cached a rotated token.**
`/node/owner` answered `401 Client Error: Unauthorized` for this machine's own node while
`node_server` on the same box authenticated fine. Two snapshots of a value the agent rotates —
`local_chat.start_local_chat` does `os.environ.setdefault("NEURON_NODE_TOKEN", …)` once at
startup, and `ui/app.py` read that env var once at import. `register()` issues a fresh token on
a relay-ticket refresh, on stale-token recovery, and on any re-registration (`agent.py:1103`),
and from that instant the UI carried a dead credential until the whole process restarted.
`config.json` holds a token ending `c1d311`; the one the UI was using is older.

That matters beyond a status endpoint: **`/node/owner` is what the claim panel reads.** A token
rotation silently switched off the feature [P39] exists to offer — the same blank panel, reached
by a different road. **Fixed:** identity is resolved WHEN USED, from `config.json` (which is what
`_save()` writes on rotation), with the environment kept as a fallback for a dev override and
for a driver-only machine that has neither. `ui/test_node_identity_is_current.py`: 14, including
a rotation mid-process with no restart.

**MECHANISM FOUND AND CLOSED (2026-08-18).** `Tray.run` was:

```python
threading.Thread(target=self.agent.run, daemon=True).start()
```

No wrapper. An exception in that thread goes to `threading.excepthook`, which writes to
`sys.stderr` — and tray mode is a **frozen windowed app whose console `_hide_console()` has
already hidden**, so stderr goes nowhere at all. The agent loop could therefore stop dead while
the tray icon, the poll thread, the Chat UI and `node_server`'s listener all carried on: alive,
holding its port, answering probes correctly, unregistered, and completely silent. That is the
observed state, exactly, and it explains the one detail that made no sense — that a machine
that had clearly completed `setup()` wrote no line about it.

Corroborating: **no `[CRASH]` marker has ever been written to `agent.log`**, on a machine that
has crashed before. `neuron_app_entry` records its own death, and `tray.main` catches around
`Tray().run()` — but neither can see inside a daemon thread that has already been handed off.

**Honest about what this does and does not prove.** The restart destroyed the evidence, so
whether *this* is what happened on 18 August cannot be established after the fact. What is
established is that it is a path to precisely that symptom, that the path existed, and that it
is now closed.

**Fixed, in two halves, because the loop can fail two ways:**
  1. `_supervise_agent` wraps the thread. A raise **or a plain return** — an endless loop
     reaching its end is a stop too, and the quieter one — is logged, sent through `crash_log`
     (stdlib only, because this is the failure class where the logging config is itself a
     suspect), and turns the tray red with the remedy rather than a traceback.
  2. `_watch_heartbeat` covers the case no exception handler could: a loop still running and no
     longer reaching anybody. `agent.state["last_beat_at"]` is stamped after `ping()` RETURNS,
     so it records that the coordinator answered rather than that we tried; five minutes stale
     is reported once, with recovery announced when beats resume. Once, not every 30s — an
     alarm that repeats forever is one people silence, and then it cannot report the next thing.

`agent/test_agent_death_is_loud.py`: 17, driving all three failure shapes.

**Still unexplained, and smaller:** why the process

writes nothing at all. `agent.log` is writable — verified by opening it for append while the
agent held it. Logging is configured on the `neuron` PARENT logger and `_setup_logging` is
idempotent. And the Chat UI came up with `NEURON_NODE_ID` set, which means `start_local_chat`
ran with an identity, which means `setup()` completed and *should* have logged
`registered as …`. So the evidence says the agent got further than its log admits. Until that is
understood, the mute state is a live risk and not a fixed one.

**Also unexplained and suspicious:** `config.json` now reads `layer_end: 27` while the running
`node_server` answers `holds: [0, 9]` and the coordinator assigns `0-9`. The config was rewritten
after the server was built. That is [P49]'s shape returning, exactly as the handoff predicted it
would — *"it will recur on the next re-placement, silently, on whichever machine is least able to
notice"*.

**Guarded now, because the cost here was entirely in nobody knowing.** `neuron_doctor` gained
`check_agent`: `agent.log` untouched for 30 minutes is reported as BAD with the remedy. It keys
on the log's mtime rather than on the process list on purpose — a running process proves nothing
here, and that is the whole point. It reports FAIL on this machine right now, which is what a
check firing on the incident that produced it should do.

**Still open, cheapest first:**
  1. **Explain the silence.** Until then every other guard is downstream of a mute agent.
  2. **A heartbeat watchdog inside the agent**, reporting through `crash_log` rather than
     `logging` — because in this incident `logging` is precisely what appears to have failed,
     and a watchdog that reports through the broken channel reports nothing.
  3. **The coordinator already knows.** A node that was online and stops beating is detectable
     centrally and nothing acts on it beyond dropping it from routing. On a two-node network
     that is the difference between a degraded chain and none.

Related: [P24] (a service up, silent, and dead — the same class, one process over), [P50] (the
identity in one file), [P49] and [P37] (the range that drifts back because nothing tells the
node), [P39] (the claim panel this took offline).

### [P50] 🔴 One PC held two node identities in one config file, and a restart picked the wrong one — 5.26 NRN was one overwrite from orphaned (2026-08-17)

**Lived, not theorised.** Restarting the agent to clear [P49]'s stale range brought it back as
**`agent-optinovate`** — a node last seen 2026-08-07 — instead of `agent-optinovate-6ff49d`, the
identity that had been serving all week and holds **5.259616 NRN**. The log said so plainly and
nobody had ever read it: *"this node's token has been superseded, most likely by another copy of
the agent registering the same node id"*.

`%LOCALAPPDATA%\NEURON\config.json` held the 2026-08-11 `agent-optinovate` config; the live
`-6ff49d` config was in `config.json.prev`. Something had rotated them. Then the restart made it
irreversible on disk: `_save()` copies config.json to `.prev` before writing, so the fresh start
**overwrote `.prev` with the wrong config** and the only on-disk copy of the live node's token
was gone — inside two minutes, as a side effect of a fix for something else.

**Recovered because the coordinator keeps the token too.** `nodes.node_token` is stored in
plaintext and is what `node_by_token` matches, so the credential was re-readable from the live
DB and written back into a rebuilt config. The node came up as `agent-optinovate-6ff49d
[verified], assigned layers [0, 9]`, retook relay port 9004, and **passed its next challenge**.
Had the coordinator hashed that column — which is the obvious hardening, and still right — this
recovery would have been impossible and the balance stranded.

**Why 🔴, when nothing was actually lost.** It came within one file write, and every property
that made it survivable was luck rather than design:
  * **A node's whole identity is one token in one file** ([P39], [P45]) — and this shows the file
    does not even have to be deleted. It only has to be *rotated*, by any of several paths, none
    of which announce themselves.
  * **The 401 was not fatal.** The agent logged `heartbeat failed: 401` and `could not report
    ms_per_layer: 409` and carried on serving, so a machine that has silently lost its identity
    looks exactly like a working one. It earns nothing while doing so, which is [P47] again by
    another road.
  * **One machine can hold many identities.** The ledger carries seven `agent-optinovate-*`
    accounts; six are empty registrations from earlier runs. Nothing reaps them, and nothing
    warns that this PC has registered eight times.

**What would close it, cheapest first:**
  1. **Say it loudly.** A 401 on ping means *this agent is no longer who it thinks it is*. It
     should be an ERROR naming the node id and the remedy, not a warning between two INFO lines.
  2. **Never rotate a config that holds a token without keeping a copy that is not the rotation
     target.** `.prev` being a single slot is what made this destructive; the backup and the
     thing being overwritten were the same file one generation apart.
  3. **Bind the owner, and this stops being about the machine at all** — see below. This is the
     real fix and the founder named it while it was happening.

**The founder's point, and it is the right one: pay the person, not the box.** Emission already
resolves `get_node_owner(node) or node` at settle time ([P39] phase 3), so a node with an OWNER
recorded pays into the wallet that person logs into. Then a rotated config costs a registration
and nothing else — the balance was never on the disk. Today no owner is bound to any live node,
because the claim panel that binds one is served at `/next` and every default user gets
`chat.html` — which makes **[P48] item 3 the blocker for this too**, not only for the wallet UI.
A machine-shaped identity is not merely inconvenient; it is the reason a config file rotation is
a financial event.

Related: [P45] (the token in one file, and the `git checkout` that would have wiped it), [P39]
(the credential and the owner link), [P49] (the restart this happened during), [P47] (a node that
has lost its identity earns nothing and looks fine).

### [P49] 🟢 The 28.6 is explained: a node holding the whole model answered a different question, and said so in an ack nobody read — fixed (2026-08-17)

**Found by watching the fix from Session 62 run live, which is the only way it could have been
found.** The verifier was restarted, the driver was challenged for the first time in six days,
and it **failed** — `max_err 28.5958`, every two minutes, deterministically. That is the figure
from 2026-08-11 that [P47] recorded as *"never explained, only hypothesised about"*. It was
live, it was reproducible on demand, and the hypothesis [P47] offered for it (a pre-0.20 agent
whose ack omitted `s1`) was wrong: this agent is 0.20.4.

**The mechanism, from the node's own ack.** Speaking the protocol to `agent-optinovate-6ff49d`
directly, with a probe config for layers 0-9:

```
sent {"type": "config", "s1": 0, "s2": 10}
ACK  {"ok": true, "layers": 28, "s2": 10, "holds": [0, 27]}
```

Two facts in one line. **It holds 0-27** — the whole model, on a 64 GB machine that loaded all
of it — while the coordinator assigns it 0-9. And the ack carries **no `s1`**, which identifies
the branch that answered: `node_server` selects its role with
`is_true_last = (self.hi == self.n - 1)`, and a node holding through the final layer is "true
last" for *every* question, including a verifier's probe about layers 0-9. So it ran
`last_stage(model, s2=10)` — `layers[10:]` plus the final norm — and returned it confidently.
The verifier compared that against `layers[0:10]`. **The right answer to a question nobody
asked**, which is why the error was large, stable, and identical across two agent versions and
six days apart.

**And the range check could not see it, for a reason worth keeping.** `challenge_middle_node`
compares the ack's `s1`/`s2`: `s1` is absent, which it correctly reads as *silence, not
disagreement* (that rule exists because omitting a field is what an older agent does, and
treating silence as a mismatch is what broke verification on 2026-08-11); and `s2` is the
caller's own value echoed back, which can never disagree. Both checks pass. Meanwhile `holds`
— the one field that was neither silent nor self-referential — was on the wire and thrown away.

**That is [P37] verbatim, one function over.** Its finding was *"the ack that explained it was
on the wire from the beginning and `challenge_node` threw it away"*, and the fix was to read
`holds` on the last-stage path. `challenge_middle_node`, four lines further down the same file,
never got it. A fix applied to the path where the bug was found and not to its sibling is how
the same defect gets discovered twice.

**Fixed on both sides, and they are independent on purpose.**

1. **`challenge_middle_node` checks `holds` first.** Works against **today's** agents with no
   release, which matters because the machine concerned is behind a NAT in a house. The driver
   now produces `RangeMismatch` — *"the node is fine, the placement is stale"* — so it is
   attested in neither direction and takes no strike ([P37] rule 4), instead of a wrong answer
   that reads as a bad machine.
2. **`node_server` no longer mistakes a probe for pipeline traffic** — `is_true_last and "s1"
   not in msg`. `s1` is the discriminator because the two callers genuinely differ rather than
   by convention: a middle relay hands its next hop `{"s2", "n", "wire"}` and never an `s1`
   (there is nothing for it to mean — the last stage's start is implied by `s2`), while
   `challenge_middle_node` always sends one. A full-model node now answers the probe from its
   own range, and the disagreement surfaces as a refusal rather than as arithmetic.

**Serving was never affected, which is why this stayed invisible.** Real pipeline traffic
arrives with `host_b` and takes the middle role, which uses the caller's `s1`/`s2` — so this
node has been computing layers 0-9 correctly for every actual request. Only verification was
broken. A node can therefore serve perfectly and be unable to prove it, and under [P47] that
means it serves perfectly and is never paid.

`test_stage1_challenge.py`: 10 checks. The full-model case is built on the 0-9 slice and then
moved to `hi == n-1`, because [P42]'s reload guard rightly refuses to construct a 0-27 server
over a 0-9 slice; role selection reads `self.hi` and `self.n` and nothing else, so the state is
reproduced exactly where it matters and the forward pass is never reached. **The two fixes are
pinned separately** — the `holds` check alone makes the RangeMismatch assertion pass, so it
cannot stand as evidence for the node-side fix; the ack's `s1` is what tells the branches apart.
Both tripwires verified to fire, then restored.

**Verified live after deploying, 2026-08-17 22:40.** The driver now produces
*"PLACEMENT MISMATCH, not a bad node — node holds layers 0-27 but was challenged on 0-9 …
nothing recorded against it"*, on the verifier-side fix alone; the packaged agent still runs the
old `node_server`, which is exactly the case that half was written for.

**Still open, and it is the cause rather than the symptom: the node serves 0-27 while assigned
0-9.** Corrected from this entry's first draft, which claimed the coordinator could not see it —
`reported_layer_*` read 0-9 when first checked and **0-27** a few hours later, so the node
re-registered with its real range and `placement_drift` is now **True**. The coordinator's own
drift signal names it, and has since before the deploy: the pre-restart verifier was already
logging *"placement_drift is set"* while still computing a wrong answer, which is the [P37]
shape once more — the field that explains it sitting in `/node/list`, read by nothing that could
act on it.

So the remaining question is not detection, it is **which way to resolve it**: assign the driver
the 0-27 it actually holds, or make it serve the 0-9 it was given. That is a placement decision
with a routing consequence and it is not the verifier's to take — [P37] deliberately did not
feed an observed range back into placement ([P32]'s ownership inversion), and that is still
right. **Until it is resolved the driver still earns nothing**, and correctly so: the slot IS
audited, so [P47]'s unaudited-hour excuse does not and must not apply to it.

**And the deeper question underneath: why does a 64 GB machine end up serving the whole model
while registered for a slice?** `agent.log` reads *"this machine can run
Qwen/Qwen2.5-1.5B-Instruct itself — fetching quantized weights instead of the pipeline-driver
slice"*. A node that can run the model locally appears to take a path that leaves its NodeServer
on 0-27. That is one config decision away from being the whole answer.

Related: [P37] (the same ack, the same lesson, the sibling function), [P47] (the emission this
was silently costing), [P42] (the reload guard that made the test harder and was right to).

### [P47] 🟡 The driver could not earn availability emission at all — cause 1's skip removed but the driver still fails, cause 2 fixed, cause 3 fixed (2026-08-17)

**Cause 2 fixed: emission pays for PROVEN work, not for OBSERVED work.** The rule was *"a
proof-of-compute challenge passed inside this slot"*. That is the right shape and it was
measuring the wrong thing: whether a challenge lands inside a given hour is decided by the
verifier's rotation and by whether the operator's PC is awake, and a node can influence neither.

**Measured, from `verify_service.log`, and it is not a corner case.** The verifier was **not
running for 207 of the 390 hours of its own history — 53%**. In the exact window [P47] measured,
of 174 hourly slots it was down for 46 and unable to read the roster for 11. **57 of
`node-c-pavilion`'s 64 unpaid hours are hours in which the coordinator could not have challenged
anybody.** That machine has passed 4,523 challenges. It was billed for our downtime.

So the rule is now: **a slot pays when the node's proof is current, or when the reason it is not
is provably ours.** Three mechanisms, each bounded so that no path to payment exists which a
node can create, detect, or exploit — rule 2 of `emission.py` (presence alone must never pay) is
unchanged and was the constraint the design was built around.

  1. **A pass carries** (`EMISSION_POC_VALID_SLOTS`, 2). A challenge that passed at 13:58 has
     not stopped being true at 14:00. Written onto the attendance row by `touch_node` rather
     than read at settlement, so the row still freezes its own inputs — which is what
     `--replay` rests on ([P40]). Cannot pay for presence: `last_poc_at` is only ever written by
     an affirmative pass, so every covered hour still has a real challenge behind it.
  2. **An unaudited slot is the coordinator's failure, not the node's.** `verify_service` now
     posts `/verifier/heartbeat` every cycle, register-secret authenticated, and `audit_slots`
     records it. A node cannot write one of those rows, cannot suppress one, and cannot observe
     whether one exists — which is precisely what makes an unaudited hour safe to pay for.
     **The epoch is the load-bearing part:** absence of a row is the signal, and every slot in
     history has no row, so without `audit_epoch()` the deploy itself would read as a
     network-wide blackout and pay every present node for the whole of the past. That is paying
     for presence, arriving through the door built to protect volunteers. Pinned by test, and
     the tripwire was checked twice — the first attempt did not fire, because the epoch turned
     out to be guarded in two places.
  3. **The excuse runs out** (`EMISSION_MAX_UNAUDITED_SLOTS`, 6). Bounded by COUNT, not
     discounted by RATE: a discount would be a penalty for our own downtime, which is the thing
     being fixed, while "we could not check" stops being an excuse once nobody has verified the
     network since yesterday. Past the bound the payout log says so.

Excused rows count toward replica depth, deliberately — if they did not, our outage would read
as network-wide scarcity and pay a **premium** at exactly the moment the coordinator knows
least, turning downtime into a payout event. The attendance floor and the
held-a-block-of-the-serving-model gate both still apply, so an excuse only ever upgrades a node
that was demonstrably there and correctly placed.

**Deliberately NOT excused, and this is where the farming risk actually is.** `ChallengeRefused`
(paused, mid-reload) is **self-declared** — a node says "I am paused" and would be paid for
saying it — and a paused node is not serving, so paying it is paying for presence. `RangeMismatch`
is genuinely our fault and it is tempting, but a node reports its own `holds`, so excusing it
would let a machine become permanently unauditable-and-paid by misreporting one field. Both stay
unpaid until there is coordinator-side corroboration to hang them on; the reason is now recorded
either way, which is the half that costs nothing.

`reconcile_emission.py` had to learn this or [P40]'s standing assertion would have started
alarming on [P47]'s fix — every excused hour would read as a reward paid to a row that does not
qualify. It reads `poc_excused` from the row (frozen, like every other input) for the payment
decision, and `audit_slots` for the *wording* of a zero, because reporting "no proof-of-compute"
when the truth is "nobody was watching for two days" is this entry's own conflation reappearing
in the reporting layer. The equivalence test that keeps the duplicated pricing honest now
randomises across the unaudited threshold — and immediately caught a real bug: the excused
short-circuit skipped the block-of-serving-model gate, paying a node for holding layers 27-32 of
a 28-layer model.

**Cause 3 fixed: a peer's pass now unlocks the hour.** `/node/{id}/peer-attest` recorded the
vote and never called `mark_slot_poc`, so peer verification could fire for the first time and
still earn nobody anything. One passing vote unlocks the slot, which is deliberately **not** the
promotion bar: promotion needs `PEER_VERIFY_QUORUM` distinct passes because it grants routing
and earnings in perpetuity, while this pays one hour to a node already present and correctly
placed. Requiring quorum here would make emission hostage to how many peers happen to be awake —
this entry's mistake one level up. Pinned at the call site in `test_peer_verify.py`, because the
function was always correct and a missing call site is the entire bug ([P37]); tripwire fired.

**Cause 1: the skip is gone, and the driver still cannot earn — see [P49].** Session 62 removed
`verify_service`'s stage-1 skip on the correct grounds that its stated premise was false, and
that change is now deployed. Underneath it was a second, real reason: the driver holds the whole
model, so its probe fell into the last-stage branch and answered about layers 10-27. It has been
failing every challenge since the restart at `max_err 28.6`. **So `STAGE1_FAILURES_ARE_SCORED`
must stay `False`** — the evidence it was waiting for arrived and pointed the other way. Flipping
it would have flagged the driver, and a flagged driver is not a degraded network, it is no
network at all. **The flag's asymmetry did exactly the job it was written for.**

The driver's lost hours are still lost, and [P49]'s fixes are what make the next ones earnable.

`coordinator/test_emission_unaudited.py`: 19. `test_reconcile_emission.py`: 40.
`test_peer_verify.py`: 16. `test_stage1_challenge.py`: 10.

The original filing follows.

### [P47-s62] 🟡 As it stood after Session 62 (2026-08-17)

**Fixed: the driver is challenged now, so it can earn.** The skip's premise was measured and it
does not hold. `verify_service` claimed the middle probe "computes layers without the embedding
a first-stage node applies" — but `node_server`'s probe role does not embed either. It runs
`common.mid_stage(model, self.lo, self.hi + 1, hidden)`, which is exactly what
`make_middle_challenge` computes. End to end against a real `NodeServer` on the real 0-9 slice:
**max_err 0**. Not inside tolerance — zero.

The code had labelled its own reasoning a hypothesis and refused to act on a guess, which was
right. What made the guess costly was that nobody joined it to emission: a node that is never
challenged never gets `mark_slot_poc`, so the skip silently made the driver unpayable, and it
took [P40]'s reconciliation to notice.

**Failures stay unscored, deliberately** (`STAGE1_FAILURES_ARE_SCORED = False`). A pass is real
evidence and recording it is what pays the driver; a failure would flag the one machine holding
stage 1, and a flagged driver is not a degraded network, it is no network at all. Same
asymmetry `drifted` already uses, and `unscored = drifted or (is_stage1 and not …)` widens that
existing meaning rather than running a second mechanism alongside it. Flip the flag once the
live driver has been seen passing — one deliberate line, with evidence.

**The 2026-08-11 `max_err 28.6` is still unexplained**, and the entry says so rather than
claiming the credit. What is now measured is that today's `node_server` on a correct slice
answers exactly, and that a node challenged on a range it does not hold raises `RangeMismatch`
— refused, recorded in neither direction — instead of returning a wrong-looking answer. That
second property is what makes removing the skip safe, and it is why an unexplained failure can
no longer flag anyone. The 2026-08-11 driver ran a pre-0.20 agent whose ack omitted `s1`, so a
range disagreement could pass unnoticed into `verify()`; both live nodes now send `s1` and
`holds`. A good explanation, not a proven one.

`test_stage1_challenge.py`: 6 checks, end to end, skipping cleanly where there are no weights.
`test_verifier_survives.py`: 36.

**Still open — causes 2 and 3 below are untouched**, and the driver's 81 lost hours are not
recoverable. Verify on the live network by watching `poc_ok` for the driver in the next
reconciliation run.

The original filing follows.

### [P47-orig] 🔴 As first found (2026-08-17)

**Measured, not suspected.** `reconcile_emission.py` on the live ledger: **199 of 318 settled
node-hours (63%) paid nothing**, 168 of them for want of a proof-of-compute challenge inside
the slot. Concentrated, and the dates say it is not history:

| node | no-proof-of-compute | window (UTC) |
|---|---|---|
| `agent-optinovate-6ff49d` | **81** | 2026-08-10 07:00 .. **2026-08-17 13:00** |
| `node-c-pavilion` | 64 | 2026-08-10 07:00 .. 2026-08-16 04:00 |
| `agent-bhpc012101-18f1da` | 15 | 2026-08-10 08:00 .. 2026-08-12 07:00 |
| `agent-bhpc012104-82cbee` | 8 | 2026-08-10 07:00 .. 2026-08-11 06:00 |

The first row runs to the most recent closed slot. This is happening now.

**Cause 1 — the driver is structurally unverifiable, and therefore structurally unpaid.**
`verify_service.py:305` skips stage-1 nodes outright:

```python
if int(n["layer_start"]) == 0 and int(n["layer_end"]) != total - 1:
    self.last_checked[nid] = time.time()
    ... log.warning("%s (stage 1 …) is NOT BEING VERIFIED …")
    continue
```

That skip is *correct on its own terms* and was added deliberately: `make_middle_challenge`
computes `layers[s1:s2]` on a raw hidden state, a first-stage node embeds token ids first, so
the two compute different functions and the driver answers "wrong" every time. Refusing to
score a machine on a check that cannot pass is right, and saying so out loud is the opposite of
[P31].

What nobody joined up is that **emission pays only on a passing challenge**
(`main.py:831`, `mark_slot_poc`), so a node that is never challenged can never earn an
availability hour. `agent-optinovate-6ff49d` — named in `verify_service.py`'s own comments as
the driver — has therefore been unable to earn emission for its entire existence. At the
current 3.0× scarcity multiplier those 81 hours are **~243 NRN**, against 333 NRN distributed
in total.

**And it lands on exactly the wrong machine.** [P43] established that the driver carries the
largest fixed cost on the network — the embedding and `lm_head`, 41% of an 8 GB budget on
Qwen3-4B — and [P44] that it is chosen for having the most RAM. The machine asked to give the
most is the one machine the reward for giving cannot reach.

**Cause 2 — emission cannot tell "not checked" from "failed".** `poc_ok` is 0 by default and
only ever set by an affirmative pass, so every branch where the verifier deliberately declines
to conclude anything reads to emission as a failure:

  * `RangeMismatch` — *"THE NODE IS FINE; THE PLACEMENT IS STALE … no attestation, in either
    direction"*;
  * `ChallengeRefused` — *"paused by its owner, or mid-reload. Both are a node behaving
    correctly … nothing recorded"*;
  * unreachable before `UNREACHABLE_STRIKES` — inconclusive by design.

Each of those was written to protect a node's *reputation* from the coordinator's own
bookkeeping — [P37] is what it cost when they were absent. None of them protects its *earnings*,
because nothing told emission the difference. The Pavilion is the likely case: it holds 10-27,
so it is on the rotation and IS challenged, and it has **zero** below-attendance-floor hours —
online all 64 times, unpaid all 64 times.

**Cause 3 — peer verification would not fix it either.** `/node/{id}/peer-attest` records the
vote and never calls `mark_slot_poc`. So even once a stranger arrives and peer verification
fires for the first time, the driver still earns nothing.

**Why 🔴.** It is live, it is silent, and it is the number a volunteer checks to decide whether
donating their machine was worth it — the exact reasoning that ranked [P40] first. A node
running 24/7 and correctly serving traffic sees roughly half its hours pay zero, with no
explanation anywhere in the product. And it is self-concealing: emission's own design note says
presence must never pay, so a low payout *looks* like the rule working.

**What would close it, cheapest first:**

  1. **Say it.** The node dashboard and `/node/{id}` know the slot's `poc_ok`; a node that was
     online and unpaid should be able to see why. Costs nothing and turns a silent loss into a
     visible one.
  2. **Stop conflating inconclusive with failed.** `RangeMismatch` and `ChallengeRefused` are
     the coordinator saying *this machine is fine*. Either mark the slot as attended-and-
     excused, or record the reason on the attendance row so the distinction survives to
     settlement. Note this is a decision about what emission pays FOR, not a bug fix.
  3. **Build a stage-1 challenge.** `verify_service.py` already says this is the missing piece
     and refuses to guess without it: a driver challenge must embed token ids first, then
     compare `layers[0:s1]`. That closes cause 1 properly and is the only fix that also lets
     the driver's reputation be established at all.
  4. **Have peer-attest unlock the slot** alongside `record_peer_attestation`, so the quorum
     path is worth as much as the trusted verifier's.

Related: [P40] (the reconciliation that found this — it was invisible until the zeros were
counted per node and dated), [P37] (the incident that produced the careful "nothing recorded"
branches this now argues are too careful by half), [P43] and [P44] (the driver's cost and how
it is chosen).

### [P46] 🟡 The installed React UI is a blank page, because index.html and its bundle came from different builds (2026-08-17)

**Live on the founder's machine, found by opening it.** `/next` served an `index.html` asking
for `assets/index-DBnmnt4a.js`; that file returned **404**, while `assets/index-eBBdaPpV.js`
from a build six days earlier sat beside it returning 200. The page loads, renders nothing, and
every other signal is green — `/` is fine, `/status` is fine, the node serves and earns.

**The mechanism is a property of the toolchain, not a one-off.** Vite content-hashes every
bundle, so `index.html` names a DIFFERENT file on each build. `neuron.iss` copies with
`ignoreversion`, which overwrites same-named files and **never prunes** ones that vanished. So
anything that copies a SUBSET — an interrupted install, a file locked by the running agent, a
hand-copy — leaves last build's bundle beside this build's index.html, and the two do not refer
to each other. Files whose names did not change (`react-*.js`, `markdown-*.js`) update fine and
disguise it further.

Here the install directory held `neuron-agent.exe` and `index.html` from 2026-08-17 08:02 with
`unins000.dat` from 2026-08-13, so the 0.20.3 installer had not run at all — a subset had been
copied in. The immediate remedy is to quit the app and run the installer properly.

**Guarded now:** `ui/test_app_assets_resolve.py` reads the built `index.html` and asserts every
local asset it references exists, plus a second check for orphaned bundles — which is the
fingerprint of a merged-rather-than-replaced copy and was present here. Verified against the
broken directory: it reports exactly the two missing files. It runs on the SOURCE tree, so it
fails before a bad build is ever packaged, and skips cleanly when `ui/static/app` is absent
(a checkout with no Node is a legitimate state that `ui/app.py` already handles).

**Still open, and it is the harder half:** the guard proves the BUILD is coherent; it cannot
prove the INSTALL is. Nothing checks, on the machine, that what was copied matches what was
shipped. Options, cheapest first — have `ui/app.py` verify its own index.html's assets at
startup and log loudly if any are missing (it already knows the path); add an `[InstallDelete]`
to `neuron.iss` clearing the app's `assets` directory before the copy, which removes the
stale-orphan half entirely; and give the installer a stop-the-app step, whose absence [P24]
already names.

Related: [P24] (installing over a running agent, `neuron.iss` has no stop-the-app step),
[P39] (the claim panel this was hiding — it is in the bundle that 404s).

### [P45] 🟢 The file that holds a node's token was tracked in git — fixed (2026-08-17)

**`agent/config.json` was in the repository.** `.gitignore` carried
`agent/config.*.json` — added deliberately, with a comment explaining that
`config.driver.json` holds a node_token — and that pattern never matches the BASE name. So the
one path that on every real installation holds a live `node_token` was version-controlled.

The committed contents were a null template, and `git log -S'"node_token": "'` over that file is
empty, so **nothing ever leaked**. That is why this sat unnoticed: the failure needs somebody to
commit *after* their agent has run, and so far only the founder has, from a machine whose agent
runs from `%LOCALAPPDATA%` rather than the checkout.

**Found one command from doing real damage.** The Pavilion's `~/neuron` turned out not to be a
git repository at all — it was provisioned by file copy — so updating it meant making it one.
`git checkout -f` overwrites tracked files, and `agent/config.json` was tracked, so the obvious
command would have replaced a working config with nulls and detached `node-c-pavilion` from the
account holding **~213 NRN**. That is [P39]'s thesis exactly: a node's entire credential is the
token in one file on one disk, and nothing else on the network knows the account exists.

Two ways it ends badly, only one of which needs a mistake:
  1. an operator commits after their agent has run and publishes their own token publicly;
  2. any `git checkout`/`git restore` on a node silently detaches it from its earnings.

**Fixed by untracking** (`git rm --cached`) plus an ignore rule for the base name. Checked
first, because removing a tracked file is only safe if nothing needed it: `ensure_config()`
writes `DEFAULT_CONFIG` when the file is absent, so a fresh clone still starts — asserted
against a scratch directory, which produced a config equal to `DEFAULT_CONFIG` with
`node_token: None`. Verified live: the Pavilion's checkout kept its identity, `diff` against the
pre-update backup was clean, and it came back as `node-c-pavilion` with its layers.

**Left open, and it is the more interesting half:** the Pavilion is now the FIRST node that is a
git checkout. Every other node was provisioned by copy, and nothing has audited what else that
path assumed. Two things already differ — a checkout carries the whole repo rather than the
subset a copy shipped, and it makes `git pull` a real update mechanism on a machine whose
`updater.py` deliberately refuses to touch source trees. Worth deciding whether nodes SHOULD be
checkouts (cheap updates, and `git status` shows drift) or whether that is a footgun on a
stranger's PC, before the next node is provisioned either way.

Related: [P39] (the token is the only credential), [P36] (provenance that cannot be established
is treated as a mismatch — the same reflex, applied to weights).

### [P44] 🟡 Auto-repair assigns the driver and the last node slices it never checks they can hold — two of three fixed (2026-08-16)

**Fixed: the driver is now checked.** `_can_drive` asks `max_layers_for(n, gpl, head_gb=…)`
before a node is given stage 1, and the incumbent rule yields to it — stability is worth a
great deal, and it is not worth keeping a driver that is OOM-killed on the first token, because
that is not stability, it is a chain that breaks every time it is repaired. When no eligible
node can hold stage 1 the function returns `[]`, the same answer it already gives for too few
nodes; the remedy is a smaller model, which is the TierController's demotion, not a plan that
OOM-kills whoever drew the short straw.

**Fixed: the overfilled tail is no longer silent.** `router.assignment_overflow` is a pure
follow-up query — separate because `canonical_assignment` returns a list every caller unpacks —
and `main.py` now logs *"node-b was given 26 layers but can hold 18 — 8 over"* beside the
repair line. **The tail is still assigned**, deliberately: a gap means not one request
completes, while an over-full node might swap rather than die. Covering it beats refusing to;
doing so while printing `routable=True` and nothing else was the half-answer.

**Fixed: `s1` no longer has to be identical on two machines.** It was read from `NEURON_S1`
**at import, in two processes** — `coordinator/config.py` and `neuron_driver.py`, each with a
comment saying it had to match the other — while `node_a.coord_get_chain` refuses any chain
whose stage 1 is not `[0, expected_s1 - 1]`. Changing it meant a coordinated restart of the
coordinator and every driver, one of which is a volunteer's PC nobody can reach.

The constant was deleted rather than distributed. `coord_get_chain` never checked a fixed 10 —
it checks stage 1 against `expected_s1`, supplied by the caller — so the coordinator now
publishes `driver_stage1_layers` on `/node/{id}/slice-info` (which the agent already calls),
the agent fetches a driver shard of that width, and `_Driver` reads the width back off the
shard it loaded. **The shard decides, deliberately:** a driver must assert only what it can
serve, because claiming the coordinator's newer value while holding the old weights would run
10 layers where the chain expects 18 and hand the next node an activation from the wrong depth
— a wrong answer instead of a clean refusal.

Note the width is a NETWORK fact, not a node's own range: a machine assigned 18-35 still
drives a chain whose stage 1 is 0-17, and its driver shard is a separate download.

**And that unlocked the actual fix: `s1` is now per MODEL.** `model_tiers.stage1_for()` reads
a tier's `stage1_layers`, falling back to the global default, so the 4b tier declares 18 while
the 1.5B floor keeps 10 and is placed exactly as before. A global constant was always the
wrong shape for a value that depends on the model and the roster; it simply could not vary
while two machines had to agree on it by hand.

Three failure modes guarded:
  * **Fallbacks never raise.** Coordinator unreachable, non-200, or an older build omitting
    the field all fall back to what is on disk, then to the constant. Being unreachable is
    when a personal Chat UI matters most, and a driver that will not load because a status
    call timed out has turned an outage into a local one.
  * **A cached driver shard is checked for WIDTH, not just existence.** That test was correct
    only while the width could never change. A shard of the wrong width is worse than none:
    the driver asserts a stage 1 the coordinator is not planning and every request is refused
    on a machine whose weights look perfectly fine. Model and provenance are checked too, as
    `agent.ensure_slice` already does for a compute slice ([P36]).
  * **Placement and validation must not drift.** `canonical_assignment` and `chain_shape` are
    two readers of the same per-model width; if they disagreed, auto-repair would re-place a
    chain on every 60-second sweep that validation then called unroutable — a loop that never
    converges. Pinned by test.

**Still open, and verified rather than assumed:** the migration handshake does **not** cover
the driver shard. It covers a node's COMPUTE slice (`ensure_slice` + `node_server.reload()`);
the driver shard is a separate download, loaded once per process, and `start_local_chat()` is
called once at agent startup with nothing re-invoking it. A driver whose shard predates a width
change keeps refusing chains until the agent restarts. Refusing is the safe direction, so the
remedy for now is that `coord_get_chain` says *the shard is stale, restart the agent* instead
of printing two ranges at a person. A real fix is a driver-side reload — `_Driver` would need
to drop `self.model` and rebuild its batchers under the load lock, with requests in flight —
and that is inference-path surgery worth doing deliberately.

Verified end to end, with no environment variables set anywhere: two nodes reporting fp16,
`canonical_assignment` produces `pavilion 0-17 / node-b 18-35`, `assignment_overflow` is empty,
`chain_shape` reports routable with `expected_stage1 [0, 17]`, and slice-info hands a driver
the same 18. At fp32 the same roster is refused outright.

`coordinator/test_auto_repair.py`: 30 tests. `test_driver_s1.py`: 17.
`agent/test_local_chat.py`: 25.

The original filing follows.

### [P44-orig] 🔴 The three holes as first found (2026-08-16)

**`router.canonical_assignment` caps the memory of every stage except the two that most need
it.** It is applied automatically — `main.py:302`, on any health sweep where the chain reads
unroutable — and it writes real layer ranges through `models.update_layers`. Three holes, in
one function:

1. **The driver is never capped at all.** Stage 1 is emitted directly as
   `{"layer_start": 0, "layer_end": s1 - 1}`, and the `caps` list it computes covers only
   `rest`. The driver is also the one node holding the embedding and `lm_head` ([P43]), so the
   machine carrying the largest fixed cost is the one whose capacity is never consulted.
2. **The last stage's cap is deliberately bypassed** — `cnt = min(cnt, left) if i < n_stages - 1
   else left`. The comment above it explains the middle-stage caps and says nothing about this;
   the reasoning is visible in the code, and it is defensible on its own terms: leaving the tail
   uncovered means not one request completes, so covering it beats refusing. But "assign it
   anyway" and "assign it anyway *and say nothing*" are different choices, and only the second
   one is implemented.
3. **`gb_per_layer` arrives, `head_gb` does not.** Even the middle stages are sized without the
   driver's head — harmless while the head sits on the uncapped driver, and wrong the moment
   either of the above is fixed without the other.

**Concretely, on the capacity case.** Two nodes, Qwen3-4B at 36 layers, `DRIVER_STAGE1_LAYERS`
= 10. The driver takes 0–9; `rest` is one node, which is therefore the last stage, so it takes
`left` = 26 layers. At fp16 storage that is 5.25 GB against an 8 GB machine's 3.75 GB budget —
the node is OOM-killed and the chain breaks, having just been "repaired". `balancer.solve`
proposes 18/18 for the same roster and `plan_migration` respects the caps; this function
overrides both, because it runs last and writes directly.

**Not only the experiment.** The same shape applies whenever a stage's fair share exceeds what
the last machine can hold, which is every tier above the 1.5B floor. It has stayed quiet
because the network serves a model small enough that no cap binds — the same reason [P26] was
latent, and the same reason it stopped being latent the day a promotion landed.

**Why 🔴 rather than 🟡.** Every other memory guard in the coordinator refuses a plan it cannot
place: `capacity_shortfall` reports, `MigrationController` blocks, `plan_migration` fits. This
one writes an assignment nothing downstream re-checks, on the automatic path, at the moment the
network is already degraded.

**What would close it, cheapest first:**

1. **Cap the driver like everyone else** — compute `max_layers_for(driver, gpl, head_gb=...)`
   and, if `s1` exceeds it, do not emit that node as the driver. The rule that picks the driver
   is already "most RAM" (and `balancer.head_node_index` mirrors it), so the fix is a check, not
   a new policy.
2. **Return the overflow rather than hiding it.** The tail must still be covered; what must stop
   is covering it silently. A `capacity_shortfall` key in the result lets `main.py` log *"chain
   repaired, node-b is 8 layers over its budget"* instead of `routable=True`. A check that
   reports instead of failing is not a check — but a repair that cannot report at all is worse.
3. **Let `s1` follow the model.** `DRIVER_STAGE1_LAYERS` is a fixed 10 for a 28-layer model and
   is env-overridable (`NEURON_S1`). On 36 layers across two machines the right stage 1 is 18,
   which is what the balancer independently proposes. Until this is resolved, the capacity-case
   experiment needs `NEURON_S1=18` set on the coordinator, and that is a workaround, not a fix.

Related: [P43] (the head this function does not charge), [P26] (an even split that was a fine
default and a bad promise), [P32] (why applying placement automatically is safe at all).

### [P43] 🟡 A model too big for one machine could not be sized, because no node could say what it stores — partly fixed (2026-08-16)

**The capacity case is the product.** "Run a model your machine cannot run" is a different
claim from "run a 1.5B model, slower than your laptop", and it is the one that makes a
distributed network worth joining. `Uraroga/spikingbrain-cpu-cluster` demonstrates it on
2012-era hardware via selective safetensors loading, which `slice_downloader.py` has done since
Session 8. So the download side was never the blocker. The coordinator's arithmetic was.

**Measured first, because the target as stated does not fit.** Qwen3-4B-Instruct-2507 from its
published header (`tools/measure_model.py`): 36 layers × 100,930,816 params, embedding
388,956,160 and tied, Apache-2.0, ungated. **16.09 GB at fp32.** The two machines hold 20 GB
between them *in total*; after the OS reserve and headroom the coordinator budgets 10.5 GB, and
even at zero reserve it is 16.09 GB into 20 GB with nothing left for two OSes, two Python
processes, the KV cache or the transient fp32 cast. It does not fit, and refusing it is correct.

**At fp16 STORAGE it does, with room** — 8.04 GB, caps of 29 + 18 against 36 layers needed.
That path is not new and not speculative: `common.WEIGHT_DTYPE` implements it, `cast_linears`
keeps every GEMM in fp32 because these CPUs have no half-precision GEMM ([P2]), and
`test_weight_dtype.py` has already measured it at 2 B/param resident, ~2.9× slower at batch 1
falling to ~1.6× at batch 8, checksum drift under 1e-2. **So the capacity case is real at fp16
and not at fp32, and the pitch should say fp16.**

**Two sizing holes, both closed:**

  * **The driver's head was never charged.** `model_tiers` said so in a comment and left the
    column out because there was no measurement. Qwen3-4B's embedding is 1.56 GB at fp32 —
    41% of an 8 GB machine's budget, spent invisibly. `head_gb` now exists and is charged to
    exactly one node, on the same fp16 basis as `gb_per_layer` so the two cannot drift apart.
  * **A node could not say what it stores at.** `balancer.weight_bytes_for` had read
    `weight_dtype` since the dtype correction shipped, and its docstring claimed the field was
    "accepted" — there was no `RegisterBody` field and no column, so it could never arrive and
    every node was sized at the pessimistic 4 bytes/param whichever precision it ran. Safe, and
    exactly what made this case impossible to express.

**Found while testing:** `common.py` lowercased `NEURON_WEIGHT_DTYPE` without stripping, so a
trailing space raised `KeyError: 'fp16 '` at import — before the node server's logging existed,
so the operator got a traceback naming a dict literal — while `agent.weight_dtype()` stripped
and reported `fp16`. The coordinator then sized that machine for half the footprint of a
process that was not running.

**Deliberately not done: `ram_free_gb`.** `max_layers_for` prefers it and a comment wished for
it, but that branch skips the OS reserve entirely, so a node reporting a free figure is sized
*more generously* than one reporting a total — and it is a snapshot taken during registration,
with no age, that nothing re-reads. A machine that registers at 3am idle keeps a 3am-idle
budget all day. That is [P34] a third time. It needs an age and a heartbeat re-read first.

**Still open:**
  1. **[P44] blocks the end-to-end run.** Auto-repair would hand the second node 26 of 36
     layers regardless of the caps above. `NEURON_S1=18` works around it; the experiment has
     not been run.
  2. **Nothing has executed a forward pass of this model.** Every number here is arithmetic
     over a published header plus a dtype measurement taken on the 1.5B. The download is
     ~1.6 GB and ~4.4 GB to two machines that have never held a 4B slice.
  3. **The 4b tier is `manual_only`** and reachable only by an operator pin, deliberately — at
     min_nodes 2 the live 3-node network clears the promote margin, so shipping it on the
     ladder would migrate production onto an experiment on a health sweep.
  4. **Every node is credited slightly less RAM than it has, and the roster reads oddly
     because of it.** `agent.py` sends `int(psutil.virtual_memory().total // 10**9)` — decimal
     GB, truncated — while RAM is installed in binary GiB. So **the 64 GiB Windows PC reports
     68**, which is arithmetically right (68.7 decimal GB) and looks like a typo, because
     nobody sells 68 GB of RAM. Raised by the founder reading the number back, which is exactly
     how a unit artifact gets caught.
     The sizing is not wrong — the tier table is decimal too, so the units agree — but the
     truncation always rounds DOWN: the 12 GiB Pavilion is 12.88 GB and is credited 12, the
     8 GiB node is 8.59 and is credited 8. That is ~0.9 GB and ~0.6 GB of real budget thrown
     away, which at fp16 is about 3 and 2 layers of Qwen3-4B. Conservative, so not urgent —
     under-crediting costs throughput, over-crediting costs a volunteer's machine — but it is a
     systematic error in the one number every capacity decision rests on. Reporting bytes, or
     GiB with the tier table converted to match, would remove both the lost budget and the
     confusing display. **Every figure in this entry is on the reported (truncated) basis**, so
     the real margins are slightly better than stated; the fp32 refusal is unaffected — on true
     decimal GB the two machines hold 24 of 36 layers rather than 21.

Related: [P44] (the placement path that ignores all of this), [P2] (why compute stays fp32),
[P26] and the 2026-08-07 promotion (aggregate RAM is not a per-node answer).

### [P42] 🟢 The slice range guard checked the ENDS of the range, so a hole in the middle still served — fixed (2026-08-16)

**`node_server._layers_in_slice` returns `(min, max)` of the decoder layer indices present in
the safetensors header, and the guard asserts `held[0] <= layer_start and layer_end <=
held[1]`. That is a bounds check, not a completeness check.** It catches the incident it was
written for — 2026-08-07, assigned 0–27 while the disk held 19–27, where `min` is 19 and the
comparison fails. It does **not** catch a slice holding 0–9 and 19–27 with 10–18 missing:
`min` is 0, `max` is 27, the assignment of 0–27 passes, and the node announces it serves a
range with a hole in it.

That is the same ending as the 2026-08-07 incident the guard exists to prevent — uninitialized
meta tensors in the forward pass — reached by a path the guard does not inspect. `strict=False`
in `load_slice_model` means the missing layers never raise at load, and `common.py`'s device
move deliberately leaves meta tensors alone (correct: most of the model legitimately IS meta on
a sliced node). So nothing between download and serving looks at the interior of the range.

**Why it matters beyond wrong output:** this is the mechanism behind [P37]. A node serving a
gap raises on meta tensors mid-forward, `_handle` catches only three exception types, the
thread dies and `conn.close()` slams the socket — which reaches the verifier as *"socket
closed mid-message"*, indistinguishable from a crashed host, where [P35]'s
`UNREACHABLE_STRIKES` attests it as a real failure. The node is then flagged for holding the
wrong bytes, which is exactly how three honest machines lost their standing.

**External evidence that the stricter check is cheap.**
`Uraroga/spikingbrain-cpu-cluster` (`src/spikingbrain_cpu/selective_loader.py`) does this after
materializing each layer:

```python
remaining_meta = [... if tensor.device.type == "meta"]
if remaining_meta:
    raise RuntimeError(f"unmaterialized meta tensors: {remaining_meta}")
```

He can assert it globally because his two ranks hold the whole model between them; we cannot,
because a sliced node's other layers are legitimately meta. The scoped version is the fix.

**Two levels, cheapest first:**

1. **Assert the header contains every index in `[layer_start, layer_end]`**, not that the range
   falls between the smallest and largest present. `_layers_in_slice` already builds the set
   (`idx`) and then discards everything but its extremes — return the set and check membership.
   Two lines, no new I/O, and it turns a hole into a startup error naming the missing layers.
2. **After load, assert no parameter belonging to an assigned layer is still on the meta
   device.** Catches a layer that is present in the header but incompletely assigned — a
   truncated download, a tensor absent from the header it is mapped to. This is his check,
   scoped to our range.

Keep the existing "unreadable header → do not block" behaviour: refusing to start on a corrupt
header would take working nodes down for a check meant to catch a mismatch.

**Fixed (2026-08-16), level 1.** `_layers_in_slice` already built the set of present layers
and then discarded everything but its extremes. Split into `_layer_set` (returns the set, or
None when the header is unreadable) with `_layers_in_slice` kept as a thin wrapper for the
error message. `reload()` now guards on set membership across the whole assigned range and
names the missing layers: *"this slice holds 0-27 and is missing 10-18 (9 layers)"*.

The "unreadable header → do not block" behaviour is unchanged — that is a deliberate choice
about which failure to guard, not an oversight.

`agent/test_slice_range_guard.py` gains the case the old check could not see: a slice holding
0-9 and 19-27, assigned 0-27. It asserts explicitly that `_layers_in_slice` still reports
`(0, 27)` — i.e. that the extremes alone would have waved it through — then that reload
refuses and names `10-18`. 15 pass. `agent/test_reload_lifetime.py` also patched: it disabled
the guard by monkeypatching `_layers_in_slice`, which after the split no longer patches what
reload() calls; it now patches both, so the pair cannot drift into a no-op. 14 pass.

**Level 2 remains open:** asserting after load that no parameter of an assigned layer is still
on the meta device. That catches a layer present in the header but incompletely assigned — a
truncated download, or a tensor mapped to a shard it is absent from. The header check cannot
see that.

Related: [P37] (the flagging that followed), [P36] (a slice that does not cover its range),
[P35] (`UNREACHABLE_STRIKES` attesting an ambiguous socket close as a real failure).

### [P41] 🟢 No CPU floor is checked anywhere, and torch's bundled MKL can fault on the old machines we recruit — detect-and-explain shipped (2026-08-16)

**Closed as far as it honestly can be, which is not the same as fixed.** `agent/cpu_check.py`
probes for AVX2 and refuses to start with a sentence a person can act on, so the failure below
is now a message instead of `0xC000001D`. The underlying incompatibility is untouched: this
does not make a pre-AVX2 machine work, it makes the refusal legible.

Four decisions worth keeping, all of them about not over-reaching on an unreproduced risk:
  * **It runs before the heavy imports.** `agent.py` pulls torch in at module level via
    `node_server`, and it is torch's bundled MKL that faults — a probe underneath that import
    sits downstream of the thing it guards. It reports through `crash_log`, which is the one
    path that works before `_setup_logging()` and inside the frozen tray app where stderr goes
    nowhere.
  * **It refuses only a POSITIVE determination** of x86-without-AVX2. An unreadable probe, an
    OS nobody here has tried, an unrecognised `platform.machine()`, or an exception inside the
    probe itself all resolve to UNDETERMINED, which proceeds. "We could not check" must never
    behave like "we checked and it failed" ([P24]).
  * **ARM is undetermined, not unsupported.** The agent is pure Python + psutil + requests
    precisely so a phone or a Pi can run it, and AVX is not a question there.
  * **The refusal names an override** (`NEURON_SKIP_CPU_CHECK=1`) and says plainly that this
    has not been reproduced on hardware like theirs. The machine being turned away may well be
    one that works, and its owner is better placed to find out than we are.

`STRANGER_INSTALL.md` now states the floor in a sentence a non-engineer can check against their
own machine ("any Intel Core from the 4th generation onwards, and any AMD Ryzen"), which is the
part that prevents the download rather than explaining it afterwards.
`agent/test_cpu_floor.py`: 24 tests, most of them about the refusals it must NOT make.

**Still open:** the capability is not reported at registration, so the coordinator still cannot
see the fleet's real instruction-set floor and nobody can answer "would raising it exclude
anyone?" from data. `cpu_check.summary()` produces the string; it needs a `RegisterBody` field
and a column, the same shape `weight_dtype` took in [P43]. And the real fix — a torch built
`USE_MKL=0 USE_MKLDNN=0` against OpenBLAS — remains a wire-compatibility decision rather than a
packaging one, because `requirements.txt` says the pin is load-bearing: **nodes exchange pickled
tensors over TCP.**

The original entry follows, because the risk it describes is unchanged.

### [P41-orig] 🟡 The report this was filed from (2026-08-16)

**The pitch is "ordinary computers" and "spare hardware". That is exactly the population
with the oldest CPUs, and nothing in the agent, the installer or the docs checks what the
machine supports.** `agent/requirements.txt` pins `torch==2.4.1` — the standard wheel, which
bundles MKL on x86 — and there is no capability probe, no documented CPU floor, and no
friendly failure.

**External evidence, not ours.** `Uraroga/spikingbrain-cpu-cluster` documents this happening
on hardware indistinguishable from a volunteer's: an Ivy Bridge i3-3240 (AVX/F16C, **no
AVX2**) took an invalid-opcode trap in `libtorch_cpu.so`, exit 132. Symbolisation identified
`mkl_vml_kernel_sExp_Z0HAynn+0xab`; disassembly showed an **EVEX/ZMM AVX-512 instruction** on
a CPU that has no AVX-512. Critically, the environment knobs did not save them —
`ATEN_CPU_CAPABILITY`, oneDNN/DNNL ISA settings and MKL ISA settings all failed to make that
runtime stable. Their fix was a **custom PyTorch build** with `USE_MKL=0 USE_MKLDNN=0`,
OpenBLAS, and `-march=ivybridge`.

**Why this is worse for us than for them.** They are two engineers on their own machines who
could read a kernel trap and rebuild PyTorch. Our failure lands on a stranger who
double-clicked an installer: on Windows the symptom is `0xC000001D`
(STATUS_ILLEGAL_INSTRUCTION), typically with no message at all — the app simply dies. That is
the first-run experience for the exact person we spent a release making the installer for,
and they have no way to know it is their CPU rather than our software.

**Not yet verified for us**, and that matters: the report is Linux, a specific torch build and
an embedded MKL VML path. Whether the Windows `torch==2.4.1` wheel dispatches the same way on
a pre-AVX2 CPU has not been tested here, and we own no machine old enough to test it on. So
this is a credible, documented risk rather than a reproduced defect.

**Cheapest mitigation, and it does not require touching torch:** probe the CPU at first run
(before the model loads) and fail with a sentence a person can act on — *"your processor
lacks AVX2, which this build needs"* — instead of an illegal-instruction crash. Record the
flags in the registration payload so the coordinator can see the real floor across the fleet.
A documented minimum in `STRANGER_INSTALL.md` costs nothing and prevents the download.

Note the constraint that makes the real fix expensive: `requirements.txt` says the torch pin
is load-bearing because **nodes exchange pickled tensors over TCP**, so swapping to a
differently-built torch is a wire-compatibility decision, not a packaging one. That argues for
detect-and-explain now, and a considered answer later.

### [P40] 🟢 Emission's live ledger reconciles against its own attendance rows — run 2026-08-17, all three legs agree (2026-08-16, resolved 2026-08-17)

**RUN ON THE LIVE DATABASE, 2026-08-17.** `/home/ubuntu/neuron/coordinator/neuron.db`,
318 attendance rows, 316 settled, 118 paying, 2 unsettled (the slot that had just closed).

| | |
|---|---|
| A — recorded (sum of settled `attendance.reward`) | **330.427894 NRN** |
| B — left the pool (seed − balance) | **330.4278938 NRN** |
| C — arrived (per payee, + the orphan below) | **321.741023 + 8.686871 = 330.427894 NRN** |

The A↔B gap is 1.4e-5 against a tolerance of the same order — float residue from ~118
decrements of a 6e8 balance, which is exactly why the tolerance is computed from the arithmetic
performed rather than fixed at 1e-6. **Every NRN that left the emission pool is explained by a
settled attendance row, and every settled row is explained by the pool.** Supply invariant
intact at exactly 1,000,000,000.

**`--replay` re-priced all 316 settled rows from their own frozen inputs and every one
matched.** That is the stronger result: not merely that the totals add up, but that each
individual payment — attendance fraction, proof-of-compute, block validity, and the replica
depth within its own slot — was priced correctly. There is no mispricing anywhere in the
history. The 262.89 NRN this entry was filed against is now 330.43; the difference is one day
of emission at the 74.9 NRN/day the run measured (against a 5,000 NRN daily cap, which has
therefore never bound).

**The seed is no longer an inference.** `599,999,971.999972` from the 2026-08-02 backup, minus
the 330.4278938 of settled rewards, lands on the live pool balance to five decimals. Had that
backup not predated emission, leg B would have been wrong by the difference. It wasn't — so the
provenance argument is now confirmed by production rather than by reasoning about a file date.

**Two things the run surfaced, neither of them an accounting error:**

  1. **8.686871 NRN was paid to a machine that no longer exists.**
     `agent-bhpc012104-82cbee` — one of [P37]'s three flagged nodes — was deleted, and
     `delete_node` removes only the `nodes` row, leaving its attendance and its ledger. The
     reconciler declined to attribute those rows rather than guessing, which is that branch
     firing on real data for the first time. The money turned out to be fine (balance 0,
     `total_earned` 8.787881 — it was swept), so this is a mechanism problem, not a missing
     NRN problem. Followed up as [P39] item 5.
  2. **199 of 318 settled node-hours (63%) paid nothing**, and `zeroed-though-qualified` did
     not fire, so every one of them genuinely failed a gate rather than being zeroed by the
     cap. That is emission refusing to pay for presence, which is its purpose — but "I was up
     all night and earned nothing" is the first question an operator asks, and the run could
     not answer it. `why-hours-earned-nothing` now counts the zeros per gate and per node,
     **with the window each run of them falls in**, because the count alone is not actionable:
     168 no-proof-of-compute against 31 below-attendance-floor, concentrated as
     `agent-optinovate-6ff49d` 81 and `node-c-pavilion` 64 — and the Pavilion has **zero**
     below-floor hours, so it was reliably online and reliably unpaid.
     That should not be possible in normal operation: `verify_service` re-checks one node per
     60-second cycle, oldest first, so on this roster every node is challenged 15–30 times an
     hour. The shape fits [P37] instead — flagged nodes were excluded from **both** verifier
     lists, so they could never be re-challenged and therefore could never earn — which was
     fixed 2026-08-10. The dates now printed alongside each count are what settles whether
     these are that incident or something still live. Until they are read, this is unresolved.

Remaining: item 2 below — running this as a standing assertion rather than once.

The entry as it stood before the run follows.

### [P40-built] 🟡 The check, before it had been run (2026-08-17)

**The reconciliation exists now: `coordinator/reconcile_emission.py`.** Read-only by
construction — one `mode=ro` connection, no `--execute`, and no SQL that is not a `SELECT`,
each asserted from the module's own syntax tree rather than by grepping it, plus a test that
runs the whole thing over a database copy and compares the file's SHA-256 before and after.
Exit status is the answer (0 reconciled / 1 discrepancy / 2 could not run), so item 2 below is
one cron line rather than a project.

    python coordinator/reconcile_emission.py --db /path/to/neuron.db --seed 599999971.999972 --replay

**It found a real defect before it was ever pointed at production, which is the argument for
writing it.** `close_slots` claims the attendance row *before* it moves the money — deliberate,
and correct: claim-then-pay can at worst pay nothing for a claimed slot, pay-then-claim can pay
twice. The walk-back for that "at worst" was `settle_attendance(entry, slot, 0.0)`, and **it
could never once have fired**: that UPDATE carries `WHERE paid_at IS NULL`, which the successful
claim four lines earlier has just falsified. So a payment the pool could not make left the row
saying it had been paid in full, permanently — and `emitted_since`, which is what the daily cap
is read against, counted NRN that never moved. The comment above it said *"the row stays settled
at 0"*; the row stayed settled at the full reward. Latent only because the pool holds ~600M
against 262 NRN spent, so the branch has never been reached. Fixed by `models.void_settlement`,
which can only ever move a reward DOWN on a row that is already final — deliberately its own
function rather than a `force` flag on `settle_attendance`, whose entire guard is the
`paid_at IS NULL` a flag would switch off. `close_slots` now counts these apart from rows that
legitimately earned zero, because the same zero in the ledger stands for two completely
different events. Verified against the real code with the pool drained: the row reads 0,
`emitted_since` reads 0, the hour stays claimed, and the supply invariant holds.

**Leg B needs a number the database does not contain, and that is worth knowing before the
run.** `genesis.seed_genesis` computes the pool's seed as `600,000,000 − already_minted` and
stores it nowhere, so "the drop in the pool" is not computable from the live DB alone. Three
answers, weakest last: `genesis.py` now records `settings['emission_pool_seed']` — which helps
every future database and not this one; `--seed` supplied by the operator; or an inference the
script reports *as* an inference, since without a seed leg B equals leg A by construction and
calling that agreement would be the check lying about its own strength. Inferred, it still
bounds: the implied `already_minted` must be ≥ 0 and ≤ today's node `total_earned`, because
nothing ever decrements `total_earned`.

**For the live database there is a better answer, and it is already in the repository.**
`backups-offbox/neuron-20260802-115534.db` has no `attendance` table at all, so it predates
every emission payment — which makes the pool balance it carries, **599999971.999972**, the
seed. `--seed-from-backup` passes it, and a test asserts both halves of that claim against the
file so editing either is caught rather than quietly believed. (It also says
`already_minted = 28.000028` at genesis, against 28.678027 of node earnings five days later —
consistent, and the first independent corroboration the figure has had.)

**Leg C cannot be exact, and the script says so rather than pretending.** The ledger has no
transactions table, so one `total_earned` scalar cannot be split into emission vs. per-request
earnings vs. faucet vs. an operator sweep. What is assertible is a floor — an account must have
earned at least the emission it was paid — and two attribution hazards are reported instead of
guessed: [P39] phase 3 pays `get_node_owner(node) or node` **decided at settle time**, so a node
that has an owner today may have been paid as itself yesterday and nothing records when the link
was made; and `claim_node_earnings.py` credits the destination's `total_earned` without
decrementing the source's, so summing `total_earned` across the ledger double-counts every swept
NRN — the 251 already swept included. Leg C therefore works per-payee and never in aggregate.

**Beyond the three sums**, because the sums alone would miss it: rows settled with a NULL
reward, rewards without a `paid_at`, negative rewards, rewards above the `base × scarcity_max`
ceiling no attendance fraction can produce, a payee with no ledger row, closed slots nobody
settled (the sweep has stopped — invisible in the logs, since an idle sweep returns early and
stays quiet), and the rolling 24h cap windowed on `paid_at` because that is what `emitted_since`
reads. `--replay` re-prices every settled slot from its own rows: sound because `touch_node` and
`mark_slot_poc` both carry `WHERE paid_at IS NULL`, so settlement freezes a row's inputs
alongside its output — asserted directly rather than assumed. The one input it does **not**
freeze is `total_layers`, so rows whose verdict depends on the serving model are named rather
than silently priced against today's.

The A↔B tolerance scales with the arithmetic actually done — one ULP of a 6e8 float is ~1.2e-7,
so a few hundred payments can honestly disagree by ~3e-5, and a fixed 1e-6 epsilon would have
reported a discrepancy on the very first clean run.

`coordinator/test_reconcile_emission.py`: 37 tests. Every discrepancy it claims to detect is
written into a hand-built database and fired, because a detector nobody has seen fire is not a
detector — which is [P40]'s own thesis applied to [P40]'s own fix. The pricing copy it carries
(so it can run against a snapshot without importing the coordinator's live config) is pinned
equal to `emission.plan_slot` on 200 randomised slots, and its constants to `config`'s.
`test_emission_slots.py` is 17.

~~**Still open, and it is the whole point:** *this has not been run against the live
database.*~~ **Run 2026-08-17 — see the head of this entry.** Everything below this line was
proved on a synthesised three-day, 201-row ledger built by driving the real `close_slots`, plus
the 2026-08-02 backup; the live run then agreed with all of it. Worth keeping as the record of
what was and was not established *before* production confirmed it, since that gap is the whole
subject of this entry.

The original filing follows.

### [P40-orig] 🟡 Nothing had compared the ledger to the attendance rows (2026-08-16)

**Corrected on filing day.** This was first written as "nobody has watched this code run",
which overstated it. `coordinator/test_emission_slots.py` is 15 tests and covers the
economics properly: supply is conserved and the pool pays for it, presence without
proof-of-compute earns nothing, partial attendance is prorated, the daily cap bounds a
scarcity spike, the multiplier is bounded and monotonic, and zero rows are reported rather
than dropped. The *logic* is not the gap.

**The gap is the live data.** Those tests run on a fresh temp DB with fabricated attendance.
Nothing has ever compared what the production ledger *holds* against what the production
attendance rows *say it should hold*. Emission has been settling on every health sweep and
262.89 NRN has gone out, and the only evidence is negative: no `[emission] sweep failed` in
the deploy logs. "No error appeared" is not "the distribution is correct" — the same
reasoning 0.20.2 exists to disprove.

Why this one outranks the other unverified items: it is the only unobserved path that
produces a **number a person will care about and act on**. A wrong `ms_per_layer` embarrasses
the dashboard; a wrong balance is the thing a volunteer checks to decide whether donating
their machine was worth it — and 262.89 NRN has already been distributed against it. It is
also cumulative: `total_earned` only ever grows, so an error does not show up as a spike, it
compounds quietly and every later payout inherits it.

The failure would also be silent by construction. Nothing reconciles what emission *should*
have paid against what the ledger *says* it paid, so the first symptom is an operator
disputing their balance — at which point there is no independent record to check them
against.

**What would close it** — and note none of these re-test the formula, they check the data:

1. ~~**Reconcile against the live rows.**~~ **Built (2026-08-17), not yet run on the live DB.**
   `settle_attendance` stores the reward it paid, so the check does not need to replay
   `plan_slot` or reconstruct the replica count at settle time: sum the settled
   `attendance.reward` values and compare against the drop in `GENESIS_BUCKETS_EMISSION_ID`
   and the corresponding rise in node `total_earned`. Those three numbers must agree. That is
   a read-only query and it either passes or names the slot where it stops.
   `coordinator/reconcile_emission.py` is that query; see the head of this entry for what
   writing it turned up, and for where leg B and leg C turned out to be weaker than this
   sentence assumed.
2. **Run it as a periodic assertion,** not once. The unit test proves the invariant holds
   for a fabricated slot; running the same check against production every sweep proves it
   holds for the real ones, and costs one query. Now one cron line — the script exits 1 on a
   discrepancy — but deliberately NOT wired into the health sweep yet: until the live seed is
   recorded it would warn on every pass, and a check that always warns is a check nobody
   reads.
3. ~~**Log what a sweep decided, not just that it ran.**~~ **Done (2026-08-16).**
   `close_slots` only logged when `nodes_paid` was non-zero, so a sweep that settled ten
   node-slots at **zero** and a sweep that did nothing produced identical silence — which is
   half of why this question could not be answered from the logs. It now counts zero
   settlements and logs on every sweep that had rows (*"settled 4 node-slot(s) at 0 across 1
   slot(s), paid nothing"*), while a genuinely idle sweep still returns early and stays quiet.
   `zero` added to the return dict; additive, and `main.py:265` ignores the return anyway.
   15/15 emission tests still pass.

   Item 1 is now written and tested; **running** it, and item 2, still cannot be done from
   here, because the live DB is on the Oracle VM and its deploy key carries a passphrase.

Related: the payout-binding work in flight makes this more urgent, not less — binding a real
wallet address to a balance is the point at which an unverified number stops being internal
bookkeeping. Same family as [P32] and [P34]: a value that is trusted because nothing has
contradicted it yet.

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

### [P32] 🟢 A pinned layer range does not survive the node re-registering — fixed (2026-08-09), confirmed live (2026-08-17)

**FIXED. Confirmed against the code, the tests and the live network on 2026-08-17** — the entry
below is the original diagnosis, kept because the reasoning in it is still the reasoning that
matters. It stayed marked 🔴 long after the fix shipped, which is its own small lesson: a status
field nobody re-reads is a stale diagnostic, and this log has an entry about exactly that
([P34], [P37]).

**Option B was taken: the coordinator owns placement — a node PROPOSES, the coordinator
DISPOSES.** Verified point by point against the loop described below:

  * **Step 4 is gone.** `models.py` no longer carries `layer_start=excluded.layer_start`. The
    only surviving assignment from that field is `reported_layer_start=excluded.layer_start` —
    the claim is RECORDED, not applied. That is the single line this whole entry was about.
  * **Step 2 is unchanged, and that is fine.** `ensure_placement` still returns early when the
    config holds a range, so a node still re-asserts its own opinion on every registration. It
    simply no longer wins, which is the correct place to have cut the loop: it needs no
    cooperation from a machine nobody can reach.
  * **The node now moves the other way.** `agent.py:1133` logs *"layer range updated by the
    coordinator"* and persists the assignment — and if `--layers` disagrees it says so out loud
    rather than silently ignoring the operator, which is [P31]'s lesson applied here.
  * **9/9 in `coordinator/test_placement_ownership.py`**, including
    `test_the_live_sequence_no_longer_reverts` and
    `test_reregistration_does_not_move_an_assigned_node`.

**The live proof, which is better than any of the above.** Both nodes restarted onto 0.20.3 on
2026-08-17 holding a stale `0-27` in their configs — the exact state that killed the network on
2026-08-09. `/node/list` showed `placement_drift: true` with `reported 0-27` against
`assigned [0,9]` and `[10,27]`: **the claim was recorded and the assignment held.** The Pavilion
then adopted its assigned range (`0-27 -> 10-27`), and both nodes now report ranges matching
what they are assigned, with drift back to false. The chain stayed `[[0,9],[10,27]]`, routable,
throughout.

Related and still true: `placement_drift` exists because of this, and is a diagnostic rather
than a fault — a node serving the assigned range while its local config disagrees is working
correctly, which is precisely what the field is for.

---


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

### [P30] 🔴 The fast engine cannot reach the models NEURON exists to serve — measured and routed (2026-08-18)

**The prize, measured here rather than cited.** The entry below is filed on a quoted ~17×, and
that number decides whether the network's engine is worth a C++ dependency on every volunteer's
PC. `tools/bench_engines.py`, same model, same machine, decode only:

| engine | rate |
|---|---|
| PyTorch fp32 — what the NETWORK runs (`node_server.py`) | **3.09 tok/s** |
| llama.cpp q4_k_m — what the LOCAL engine runs (`local_gguf.py`) | **26.93 tok/s** |

**8.7×**, half the quoted figure and still decisive. For scale the live two-machine chain
measured **1.44 tok/s** the same day; the same chain on these kernels is roughly 12, which
clears §11.6's "under 30s answers" gate that currently blocks any purchase path. The gap mixes
kernel quality with memory bandwidth and the script says so — the question is not which matmul
is better but how fast a volunteer's machine could answer.

**The blocker re-checked, because it dated from 2026-08-07 and [P47] is what an unverified
premise costs.** It holds, and is sharper than filed: `llama_supports_rpc()` returns **False**
and the wheel ships `ggml-base`/`ggml-cpu`/`ggml`/`llama`/`mtmd` with **no `ggml-rpc` at all**.
RPC is compiled OUT, so patching the Python wrapper cannot reach it — it needs a rebuild with
`-DGGML_RPC=ON`. `llama_model_params` does now carry a `devices` field, which is the modern
backend-device API that path goes through.

**Route 1 is disqualified in its obvious form, and this is the finding that matters.**
llama.cpp's own `tools/rpc/README.md`:

> *"the RPC backend are currently in a proof-of-concept development stage. As such, the
> functionality is fragile and insecure. **Never run the RPC server on an open network or in a
> sensitive environment!**"*

NEURON is, by definition, an open network of strangers' machines. Pointing volunteers at
`rpc-server` would re-open [P19] — the pipeline wire that ran arbitrary code from any peer —
deliberately, on upstream's explicit warning. **Route 1 as written is off the table.**

**Route 1′, which the entry did not consider and which survives that objection.** The warning is
about EXPOSURE, not about the kernels. NEURON already owns an authenticated transport: the relay
with tickets (`relay_auth.py`), built precisely because volunteer machines sit behind NAT and
cannot be trusted to face the internet. If `rpc-server` binds **localhost only** and its traffic
rides the existing authenticated tunnel, nothing is ever on an open network — and the same
README confirms the capability is real: `--rpc host:port,host:port` splits layers automatically
by memory, with `--tensor-split` to override it, which is exactly where the coordinator's
placement decision would be injected so economics stay NEURON's.

Two frictions to answer before committing to it, and neither is fatal:
  * **proof-of-compute is built on `node_server`'s own protocol.** An `rpc-server` node is a
    dumb ggml backend; challenging it means either keeping a thin NEURON listener beside it, or
    moving the challenge down to a tensor-level operation. Unsolved, and cheap to prototype.
  * **the split must stay the coordinator's**, or the network cannot know who computed what,
    and emission is priced on exactly that. `--tensor-split` is the hook.

**Route 2 (embed ggml, speak NEURON's wire protocol) remains the coherent end state** and is
unchanged by any of this. It keeps placement, economics and proof-of-compute where they are, and
removes Python and PyTorch from volunteer machines. It is also a C++ component to build and ship
per platform, which is why measuring 1′ first is worth a session and building 2 blind is not.

**SPIKE RUN, 2026-08-18, and it lands better than the entry assumed.** Isolated from this repo
entirely — downloaded to a scratch directory, `git status` clean throughout.

**The toolchain problem evaporated: RPC ships prebuilt.** The official
`llama-b10485-bin-win-cpu-x64.zip` (17 MB) contains **`ggml-rpc-server.exe` and `ggml-rpc.dll`**.
No cmake, no MSVC, no build. The wheel installed in this venv has RPC compiled out; the released
binaries do not.

Two `ggml-rpc-server.exe` on loopback, one Qwen2.5-1.5B q4_k_m split across them:

| configuration | rate |
|---|---|
| NEURON's live two-machine chain (PyTorch fp32, real relay) | **1.44 tok/s** |
| PyTorch fp32, one machine | 3.09 tok/s |
| llama.cpp, no RPC, plain local CPU — the control | **34.97 tok/s** |
| llama.cpp split across **two** RPC servers on loopback | **32.11 tok/s** |

**The RPC hop costs ~8%** against its own control. That is the protocol floor with zero network
latency, not a prediction about the real path.

**Four things the spike established, each of which was an open question:**
  1. **It binds `127.0.0.1` by default.** The posture Route 1′ needs is the default, not a
     precaution to remember.
  2. **Layers really do split** — 15 to the first device, 14 to the second, listed per layer in
     the load log.
  3. **`--tensor-split` controls placement exactly**: `25/75` → 8/21, `75/25` → 22/7. This is the
     answer to the friction above — **the coordinator keeps placement authority**, computing the
     split as it does today and passing it through. The economics do not have to move.
  4. Its default split for two devices put **layers 0-9 on the first**, which is NEURON's own
     stage-1 shard arrived at independently.

**One unknown left, and it is now the only one that matters:** what ggml-rpc costs over
NEURON's *actual* path — a relay through an Oracle VM, not loopback. Decode is sequential, so
every token pays the round trip; 8% is a floor and the real figure could be much worse. That
needs `rpc-server` on the Pavilion and a measurement through the relay, and it is the last thing
standing between Route 1′ and a decision.

**A second finding, unrelated to speed and worth its own line.** The release ships
per-microarchitecture CPU kernels — `ggml-cpu-ivybridge.dll`, `sse42`, `haswell`, `zen4`,
`sandybridge` and more — and dispatches at runtime. That is **[P41] solved as a side effect**:
the AVX2 illegal-instruction crash on an old volunteer machine is a PyTorch-wheel problem, and
llama.cpp simply picks a kernel the CPU can run. It would also remove the load-bearing
`torch==2.4.1` pin, since nodes would no longer exchange pickled tensors.

**Next step is a SPIKE, not a commitment** — the shape [P2] already used here for int8. Isolated
from this repo: build llama.cpp with `-DGGML_RPC=ON`, run two `rpc-server` instances on
loopback, split one model across them, and measure. That answers the only question that decides
between 1′ and 2 — whether ggml's RPC path is fast enough across a hop to be worth keeping —
without touching a line of NEURON. No toolchain is installed here yet: cmake and ninja are
absent, though Visual Studio is present.

Do NOT write another matmul. This repo has measured a hand-written AVX2 int8 kernel at **1.44×**
against llama.cpp's 8.7×, on the same CPU, in the same language.

The original entry follows.

### [P30-orig] 🔴 As first filed

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

### [P29] 🟢 A user who runs out of NRN has no way to get more — decided and closed (2026-08-17)

**DECISION: no recurring faucet. The one-time grant stands, and the product now says so before
you reach the end of it.** TOKENOMICS §12.6 posed this and refused to pre-empt it — *"correct
for a coin with utility and wrong for a first-stranger trial. Decide which one the next release
is for."* Decided here, on the reasoning below, because an installer is now public and the
question cannot stay open.

**Who actually pays for a free tier.** NRN is not a project credit; it is a claim on other
volunteers' electricity and CPU time. A recurring grant would commit *their* hardware to
unlimited free consumption by people who contribute nothing — and that is not the project's to
give away. The faucet is OAuth-gated, and free Google accounts are free, so a renewing grant is
a standing invitation to farm exactly the resource the network depends on donations for. Saying
no here protects volunteers, which is the only constituency that cannot protect itself.

**And the population it strands is far smaller than this entry assumed.** Three things changed:

  1. **Local-first inference is the default.** `ui/app.py` tries `best_local_model()` before the
     network, and `local_gguf.stream` reports `cost_nrn: 0.0` and makes no `/infer` call —
     *nobody else's hardware ran this.* A machine with 3.6 GB available runs the 1.5B itself,
     free and unlimited; 1.5 GB runs the 0.5B. Such a machine never reaches this error at all.
  2. **Contributing needs LESS machine than running the model does.** One layer of the 1.5B is
     ~3.3 GB total RAM, against 3.6 GB *available* for local inference. So a machine too weak to
     answer for itself can still earn by answering for others.
  3. **Earnings now reach the person.** [P39] phases 1–3 deployed today: a claimed node credits
     the owner's wallet, not the machine's ledger row.

What remains stranded is a machine under ~3.3 GB total RAM whose owner will not contribute it —
and [P41]'s instruction-set floor excludes most of that hardware before this ever applies. That
is a hardware limit, not a policy cruelty.

**What was genuinely wrong, and is now fixed: the product never said any of this.** A hard limit
is not unfair; *discovering it by hitting it* is. So:

  * the sidebar shows `≈ N network answers left` while there is still room to act, and says
    "contribute this machine to earn more" under 30 — hidden entirely when the balance could not
    be read, because "we could not ask" must never render as a confident number;
  * the out-of-NRN message states the reason rather than the refusal — *"It paid other people to
    run answers on their machines, and there is no automatic top-up"* — and names the thing
    nobody would guess, that contributing needs less memory than running the model locally.

Asserted rather than left to taste: `wallet.test.ts` pins the three-state balance and the
warning threshold; `neuron.test.ts` pins that the message says WHY, promises no top-up ("for
now", "try again later" are asserted ABSENT), and names both the contribute and claim steps.
45 vitest.

**Deliberately still not built: a purchase path.** TOKENOMICS §12.7 is explicit that a coin
people buy is a different regulatory object and *"a public sale is not something to ship on
engineering judgement."* That review has not happened, so the third option stays closed.

### [P29-detail] 🟡 How the blocking half was fixed (2026-08-17)

**The entry's own closing line was "the node→wallet link is now the blocking item for the whole
earn-and-spend loop." That link shipped and deployed today.** `nodes.owner_wallet_id` records
which wallet a node's earnings belong to, `bind_payout_address` sets it as part of a signed
payout binding, and `close_slots` transfers availability emission to
`get_node_owner(node_id) or node_id` — so a claimed node credits the person, not the machine.
[P39] phases 1–3, live on the coordinator as of 2026-08-17.

**So a contributor now has a route, and the UI's invitation stopped being a lie.** It said
"Contributing a machine earns more" while earnings landed in a row nobody could spend from —
the entry names that precisely: *the invitation does not yet actually solve the problem it
points at.* It does now. Both pages were updated to name BOTH steps, because contributing
credits the NODE and only claiming links it to the wallet, and a message naming half of a
two-step action is the same false promise with better grammar. `neuron.test.ts` asserts the copy
mentions contributing, claiming, and this wallet.

**What is genuinely left is a DECISION, not an implementation**, and TOKENOMICS §12.6 already
frames it and refuses to pre-empt it:

> a one-time 25 NRN grant … then earn-or-buy … ~100 chat turns, then the account is dry
> forever. That is correct for a coin with utility and *wrong* for a first-stranger trial.
> **Decide which one the next release is for before handing anybody an installer.**

A person who contributes hardware is now served. A person who only ever consumes still stops
permanently at ~100 turns, and whether that is a bug or the product is the question above.
Deliberately not implemented here: a recurring allowance changes circulating supply and opens
faucet farming across sybil logins, which is an economic decision with a legal surface (§10,
§12.7), not a repair to apply on engineering judgement.

Downgraded 🔴 → 🟡 rather than closed: the mechanism that made this red is gone, the remaining
half is a policy call, and an installer is now public — so the deadline §12.6 sets ("before
handing anybody an installer") has technically already passed.

The original entry follows.

### [P29-orig] 🔴 As first filed

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

### [P24] 🟢 Strangers register fine, then sit PROBATIONARY forever — no longer blocks S12 (2026-08-17)

**The condition in the title cannot persist any more, and the one thing still open is not
fixable by anyone here.** Re-checked 2026-08-17.

*Forever* was the load-bearing word: a stranger stayed probationary until the founder's laptop
was awake to run a verifier, and that verifier had been dead for two days without anything
noticing. Two independent routes to `verified` exist now — the operator's own proof-of-compute
run, and **a quorum of DISTINCT already-verified peers** — and `models.py` says why: *"a
stranger who joins at 3am is promoted by the network itself rather than waiting for the
founder's laptop to be switched on."* Every node runs `peer_verify_loop` on a 60-second poll.
`coordinator/test_peer_verify.py` 14/14.

The rest of the entry's fixes verified present and passing: `test_startup_is_never_silent.py`
19/19, `test_probation_is_visible.py` 14/14, `_save()` atomic via `os.replace()` with a `.prev`
fallback, the doctor's verifier-liveness and probationary checks, and the keepalive task.

**Why it no longer blocks S12.** The entry's own instruction was *"do not put 0.18 in front of
another stranger — cut 0.19 first."* The current release is **0.20.3**, five versions on, and
every diagnosability fix listed above is in it. A stranger installing today cannot land in the
state this entry describes, and if they hit something new they get a log — which is the whole
of what was missing.

**Genuinely still open, and unresolvable from here:** the specific v0.18.0 crash on the
2026-08-07 machine was never *identified*, only made diagnosable, and identifying it needs that
machine. Nobody has it. That is an acceptable place to stop — the version cannot be installed
from anything the project now advertises, and the failure class it belonged to is closed.

**Worth stating rather than assuming: peer verification has never actually fired.** Every live
node reads `peer_passes: 0` and was promoted by `challenges_passed` — the founder's own
verifier. So the mechanism that removes the human is built, tested and deployed, and *unproven
in production*, which is the same shape as the relay being built and bypassed for a week before
2026-08-01. It needs two verified peers and a genuine newcomer to exercise it, and the first
stranger is exactly that occasion. Watch `peer_passes` when one arrives; if it stays 0, this
entry is not as closed as it looks.

The history follows.

### [P24-hist] 🟡 The investigation that got here (2026-08-05)

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

### [P23] 🟢 `prune_test_accounts.py` will sweep the first stranger's wallet — fixed (2026-08-17)

**Fixed by deleting the `w_` prefix rule**, which is the first of the two options below. Real
wallets now fall through to `unclassified`, which `--execute` already refuses to sweep, and a
genuine dev wallet is named with `--prune-also` — so the failure direction is "a dev wallet
survives until somebody names it" rather than "a person's balance disappears".

**It had stopped being forward-looking.** The entry was written when the only `w_` wallets were
the founder's. Two external users exist now with real spend, so the next `--execute` would have
taken their balances and filed them as test accounts in the audit log.

The prefix table now carries the rule that lets it stay safe: a prefix may only remain if it
CANNOT match an account a real person could be issued. `node_a-cli-` qualifies (a development
CLI mints it, nothing user-facing does); `w_` never could, because `wallet_for_oauth()` mints
every user wallet in exactly that format and there is no other.

`test_prune_test_accounts.py` updated in the same change, as the entry required: the two `w_`
fixtures move from PRUNED to unclassified and are asserted to STILL HOLD THEIR BALANCE after
`--execute`, the already-empty prune target is now a `node_a-cli-` account, and a new check
asserts an OAuth-shaped wallet is not pruned by prefix. 42 pass.

Checked against the real ids rather than fixtures: `w_d35c84ddd33ea857d74c29db22cd76a9` and
`w_ef7ca467…` both classify `unclassified` now.

**Still optional and still the founder's call:** restoring the 49.3 NRN swept from those two
wallets in Sessions 33–35. Recoverable from `backups-offbox/neuron-20260802-115534.db`. Moving
balances on a live ledger is a decision, not a repair to apply silently.

The original entry follows.

### [P23-orig] 🔴 The rule as first written (2026-08-09)

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
- **External evidence for the method (2026-08-16).** `Uraroga/spikingbrain-cpu-cluster`
  published `GOAL13B_INT8_BENCH_REPORT.md`, a controlled benchmark on one real
  `mlp.gate_proj` matrix (67.9M codes) that isolates *where the error comes from* — which is
  the question this entry has been open on:
  - **Saturation is the corruption, and it is now quantified.** With the checkpoint's own
    stored group scales, **1.039% of rounded weight codes fell outside `[-128,127]`** (actual
    range `[-173,173]`). Naive saturation of just those (his path B1) gave worst relative L2
    **3.25e-3**; reconstructing them exactly (B2) gave **5.06e-7** — roughly 6400× better,
    and B2's residual error was accumulation order, not the weights. So "cast to int8"
    silently changes ~1 weight in 100, with no exception and no obvious symptom. Compounded
    across every Linear and 28 layers, that fits our observed failure — the model *refusing*
    to answer rather than degrading — better than precision loss does.
  - **The method: keep an INT8 base and correct the outliers exactly** — group-128 FP32
    scales, plus sparse residuals (row pointers + K indices + INT16 residuals) for the
    out-of-range codes. Cost of exactness was ~4.4 MB on a 67.9 MB base, about **6.5%
    storage overhead**. That is cheap enough to be the default rather than an optimisation.
  - **His 22.49× does NOT transfer, and his own control proves why.** His baseline rebuilds
    the whole FP32 fake-quantized matrix on every forward; his path C (cache those weights,
    change nothing else) is **bit-identical** to the baseline and already worth 2.87–3.56×.
    We run a plain fp32 Linear and have no such waste to reclaim, so **our 3.46× remains the
    honest number** for an fp32→int8 swap. Do not let the headline reset the estimate.
  - **Caveat on transfer:** his checkpoint is W8ASpike and *already carries* group scales —
    his problem is faithfully reproducing an existing quantization. Ours is *choosing* one on
    a normally-trained fp32 checkpoint. The transferable parts are the group-wise (not
    per-tensor) scale, and handling outliers explicitly instead of saturating them.
- **We are ahead of him on the half that bit us hardest, and he has flagged it as his next
  unknown.** His report notes his 32 real inputs came from one prompt and one layer, so
  *"other layers/prompts may contain activation outliers"*. We measured exactly that at the
  junction: **absmax 6620, std 42, worst channel ≈750× the median** — see [P20], where an
  absmax quantizer scaled to that one channel collapsed everything else, fp8 e4m3 could not
  represent 6620 at all and went NaN, and Petals' own blockwise-int8 scheme diverged on 1 of
  3 prompts. Our fix was QuaRot's Hadamard rotation at the **transport layer**: orthogonal,
  so it spreads the outlier without changing the vector, and needs no weight surgery and no
  calibration. Worth sending him: it is the answer to the risk he has already named.
- **Status:** speed is proven reachable; the *method* is the open work. Schedule a
  dedicated "quality-preserving quantization" session (candidate: alongside/after S14),
  NOT a rushed integration now. No real users yet (per ROADMAP's One Rule).

### [P10] 🟢 No stranger can actually join yet — everything assumes ONE Tailscale net — RESOLVED, confirmed live (2026-08-17)

**As written, this is now false: a stranger CAN join. None has.** Those are different
statements, and keeping the first one on the board because the second is true is what left this
marked 🔴 — and marked **BLOCKS S12** — long after the work landed. Same stale-diagnostic
failure as [P32], on the entry that gates the roadmap.

Every claim re-checked on 2026-08-17, against the running system rather than the history below:

  * **Sub-problem 1, public coordinator.** `https://neuronnet.duckdns.org/status` returns 200
    over the open internet, from a machine on no tailnet. No Tailscale anywhere in the path.
  * **Sub-problem 2, node↔node across NAT.** The relay is not merely built, it is the DEFAULT:
    `use_relay()` returns `bool(self.cfg.get("behind_nat", True))`. Both live nodes came up on
    relay endpoints today — `150.230.22.250:9003` (Pavilion) and `:9004` (Windows PC) — logged
    as *"relay tunnel started — reachable via … (NAT-friendly)"*. Peers are handed a publicly
    routable address, which is the whole of what this entry demanded.
  * **Open join.** Registration needs no shared secret; a secret-less node joins probationary
    and is promoted by proof-of-compute. `coordinator/test_open_join.py` 26/26.
  * **The two bugs that would have killed the first stranger on their first command.**
    `agent/__init__.py` exists; the probe role's slice bound is `self.hi + 1`.
  * **[P12], `/complete` auth**, listed here as outstanding: `complete_token` is issued by
    `/infer` and required to settle.

And the thing the entry called *"what's genuinely left"* — an outside person installing a
package — now exists: **NEURON-Setup-0.20.3.exe** is a public GitHub release, the coordinator
advertises it with a verified SHA-256, and `STRANGER_INSTALL.md` is the guide.

**So what remains is not engineering.** Nobody outside this project has run the installer. That
is an adoption fact, and it belongs to `GROWTH_PLAN.md` and the scout, not to a problems log —
holding it here made an architectural blocker out of a marketing one, and stopped S12 reading as
unblocked when it was.

**Genuinely open, and neither blocks a first stranger:** every relayed hop costs an extra
round trip through Amsterdam, and beyond roughly 100 relayed nodes the single relay is the
bottleneck (`SCALING.md`). Both are volume problems, and this network has two nodes.

The history that got here follows.

### [P10-orig] 🔴 The problem as first found (2026-07-22)
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

### [P39] 🟡 Node earnings accrue to an account nobody can sign in as — partly fixed (2026-08-16)

The founder asked where the 252 NRN that has left `__emission_pool__` actually is, and the
answer exposed a seam nobody had looked at. `node-c-pavilion` holds **213.4954 NRN** — 85% of
everything the network has ever paid out — in a ledger row created by `register_node`'s
`INSERT OR IGNORE INTO ledger (node_id)`. No email, no login, no owner. Its entire credential is
the `node_token` in one `config.json` on one laptop, and there is no recovery path: no endpoint
moves a node's balance to the wallet a person actually signs into, and `models.transfer` — the
primitive that could — is exposed nowhere.

Two consequences, both live before this was written:
1. **Lose the token, lose the money.** Nothing else on the network knows the account exists.
2. **It would not survive the chain migration.** `blockchain/migrate_ledger.py` marks an account
   with no bound payout address `unmapped` and skips it. Pavilion has never bound one.

And the obvious fix had a trap inside it: sweeping a node into a wallet made the NRN spendable
in the chat UI and simultaneously *unmappable*, because payout binding existed only at
`/node/{id}/payout-*`. Spend it now and keep it later were mutually exclusive, and nothing said
so. Both halves shipped together for that reason:
  * `coordinator/claim_node_earnings.py` — sweeps a node account into a wallet. Dry run by
    default, invariant-checked before and after, appends to `claim_log.json` because the ledger
    has no transactions table. Most of its 29 tests are refusals: `transfer()` ends in
    `INSERT OR IGNORE`, so paying a typo'd wallet id does not error, it silently creates the
    account and reports success — money moved somewhere nobody can ever authenticate as, which
    is strictly worse than leaving it where it was.
  * `GET/POST /wallet/{id}/payout-challenge|payout-address` — the wallet half of payout binding
    (21 tests). `binding_message` gained a `label`, so the prompt reads `wallet:` and a
    signature made for a node cannot bind a wallet of the same id. Node text is byte-identical.

**The owner link now exists (2026-08-16), phase 1 of 4.** `nodes.owner_wallet_id` records the
wallet a person actually signs into, and `bind_payout_address` accepts it. 17 tests.

Two decisions worth keeping:
  * **It is recorded only as part of a successful payout binding.** Rebinding already requires
    the incumbent key, so a copied `node_token` cannot move ownership — which a standalone
    "set my owner" endpoint would have handed it. Tested: a stolen token rebinding to its own
    address is refused and the owner still points at the original wallet.
  * **It never leaves `_node_dict`.** `list_nodes` feeds the public `/node/list`, the public
    dashboard and the router, and a node→person map is exactly the correlation that private
    balances and private payout addresses exist to prevent. Dropped at the source rather than
    filtered per consumer; readable deliberately via `get_node_owner()`, and over the wire
    only from `/node/{id}/payout-address`, behind the node's own token.

An invented `owner_wallet_id` is refused rather than recorded: `set_payout_address` and
`transfer` both end in `INSERT OR IGNORE`, so a typo would otherwise write a phantom owner
nobody can authenticate as — the same trap `claim_node_earnings.py` spends most of its tests
refusing, one layer earlier. An older agent that sends no owner still binds normally.

**Phase 2 done (2026-08-16): the Chat UI asks.** `local_chat.start()` passes the node identity
into the UI's environment; `ui/app.py` gains `/node/owner`, `/node/payout/challenge` and
`/node/payout/bind`, mirroring the wallet routes. The `node_token` is used server-side only and
never reaches the browser, and `owner_wallet_id` is injected from the SESSION — `NodeBindBody`
has no such field, so a page cannot nominate somebody else's wallet as the owner of this
machine's earnings. The chat page shows a panel only when this machine serves a node AND has no
owner recorded: a driver-only machine sees nothing and nobody is asked twice. A panel, not a
gate — a volunteer without a browser wallet must still be able to run a node. 16 tests.

**Phase 2 completed in the React app (2026-08-17), and clicked for the first time.** `/` serves
chat.html and `/next` serves the rewrite; the rewrite had no claim panel, so swapping the routes
would have silently dropped this — invisibly, because the panel only ever appears for a
contributor who has not claimed yet. Ported as BEHAVIOUR rather than DOM:
`ui/web/src/services/nodeOwner.ts` holds the rules, `components/NodeOwnerPanel.tsx` renders
them, wired into the Sidebar under the wallet block. Same reason `services/neuron.ts` exists —
the old page's guarantees were asserted by grepping chat.html's source text, which a compiled
bundle makes impossible.

`claimNodeEarnings` takes its EIP-1193 provider as an ARGUMENT rather than reading
`window.ethereum`, and that one choice is what made the path testable: the old tests stubbed
`window.ethereum` wholesale and then asserted on source strings, so connect → challenge → sign
→ bind had never once executed anywhere. Driven through the real component in a browser with
two wallets:
  * **declined** (`code 4001`) — *"You declined the signature. Nothing changed — you can do this
    any time."*, button present and NOT disabled, and **nothing POSTed**;
  * **signed** — signs the exact challenge text for the address the server returned, POSTs only
    `/node/payout/bind`, panel shows the bound address and the button is gone.

The POST body is exactly `{address, nonce, signature}` — asserted in vitest and re-checked live
in the browser, because "the page cannot name somebody else as owner" is the property this whole
phase exists for. 20 tests in `nodeOwner.test.ts`; `ui/test_node_owner_ui.py`'s 16 server-side
tests untouched.

`vite.config.ts` gained a DEV-ONLY mock middleware (`apply: 'serve'`, never in a build). Without
it `npm run dev` serves a signed-out machine with no node, so the wallet rows and this panel
could not be looked at without running an agent, a coordinator and an OAuth round trip — a large
part of why this had never been clicked.

**Still open on phase 2:** a real MetaMask signature (the in-app browser has no extension, so the
provider was a stub — the sequence, wording, state transitions and the no-wallet-id property are
verified, MetaMask's own popup behaviour is not), and the desktop rebuild, since an installed
0.20.2 serves its own copy of the UI.

**Phase 3 done (2026-08-16): emission pays the person, not the machine.** `close_slots` now
transfers to `get_node_owner(node_id) or node_id`, so a node with a recorded owner credits the
wallet directly and needs no sweep, while nodes that predate this keep earning exactly as
before. The attendance row is still claimed against the NODE — that is which machine was
present, and settling it against the owner would let one person's two nodes settle each
other's hours.

**Still open — phase 4, and it needs the live DB:**
  4. **Sweep the history once.** `claim_node_earnings.py` already does it. Order matters:
     deploy the coordinator, let the operator record an owner through the UI, and only then
     sweep — running it first moves the balance to an account with no proven owner, which is
     the thing this whole entry exists to stop. Pavilion's ~213 NRN is the case.
  5. **`delete_node` orphans a funded ledger row — the money was recoverable this time by
     luck, not by design.** Live 2026-08-17, found by `reconcile_emission.py`:
     `agent-bhpc012104-82cbee` — one of [P37]'s three — has **8.686871 NRN of settled
     attendance and no row in `nodes`**. `delete_node` is `DELETE FROM nodes` alone; the
     attendance rows and the ledger row both survive it. Emission resolves its payee as
     `get_node_owner(id) or id`, and a deleted node has no owner, so those hours paid into an
     account whose only credential was a `node_token` on a machine that no longer exists.

     **The NRN itself is fine, and the first draft of this item said otherwise.** The ledger
     row reads `balance 0.000000, total_earned 8.787881` — and `total_earned` is never
     decremented, so that pairing means the balance was moved out, not lost. It was sweepable
     because `claim_node_earnings.py` reads the **ledger**, not `nodes`, so a deleted node's
     row is an ordinary source; the 8.69 was almost certainly part of the 251 NRN swept on
     2026-08-16. Worth stating plainly because the reconciler reported *"payee unknowable"*,
     which is a statement about attribution and was briefly mistaken here for a statement
     about the money.

     What remains is the mechanism, and it only worked out because the deleted node belonged
     to the founder. On a volunteer's node the same sequence pays real NRN into an account
     nobody can authenticate as and nobody would think to sweep — and phase 4's ordering
     (*record an owner through the UI, then sweep*) cannot help, because there is no node to
     record an owner against. Cheapest fix: have `delete_node` refuse while the ledger row
     holds a balance, which turns a silent stranding into a visible refusal at the one moment
     somebody is looking. Recording the owner at registration ([P39]'s durable fix) closes it
     properly, since the payee would then be a wallet that outlives the machine.

Note for packaging: the desktop build serves its own copy of the UI, so the phase 2 panel does
not appear in an installed 0.20.2 until the app is rebuilt.

Also still unfixed: a leaked `wallet_id` is enough to bind a FIRST payout address (rebinding
needs the incumbent key) — the same honest limit `payout.py` documents for `node_token`.
Phase 2 is the shape that closes it.

Related display bug, not yet fixed: `main.py`'s node dashboard computes `spent = total_earned -
balance`, so a swept node reports its earnings as **spent on inference**, which is not what
happened.

---

## Not-doing (deliberately, for now)
- Chasing single-user latency parity with GPU clouds — unwinnable, wrong hill.
- Productionising quantization before there are real users (measure first, integrate later).
- On-chain NRN before Session 12 (first stranger node) — per ROADMAP.
