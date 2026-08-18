"""
agent/node_server.py — one generalized NEURON node server for ANY layer range.

Loads this node's downloaded SLICE (not the full model) and serves whichever role
the incoming config implies, staying compatible with node_a.py's existing wire
protocol so it drops straight into the chain:
  - MIDDLE relay  (config carries host_b/port_b): run my layers, forward the hidden
    to the next hop, relay its result back  (the node_c role)
  - LAST stage    (config carries s2/n only):     run my layers + final norm, return
    the normed hidden  (the node_b role)

Reuses common.py (first/mid/last_stage, KV cache, TCP framing) and the slice
loader — does NOT modify any existing node script. A per-machine compute lock
serialises this node's own math (pipelining across concurrent requests).

Usage (normally launched by agent.py):
  python node_server.py --slice-dir ./model_slice --layer-start 10 --layer-end 18 --port 50999
"""
import argparse
import base64
import gc
import os
import socket
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import batching                                  # noqa: E402
import common                                    # noqa: E402
import wire_codec                                # noqa: E402
from slice_downloader import load_slice_model    # noqa: E402
from security import wire_crypto                # noqa: E402  [P52] encrypted hop

# Accept ONLY encrypted connections. Off by default through the rolling upgrade: flipping it
# while 0.20.5 nodes are still on the network would partition the chain rather than secure it.
REQUIRE_SECURE = os.environ.get("NEURON_REQUIRE_SECURE") == "1"

# Held only for swapping the model pointer during a migration reload. It used to wrap every
# forward pass too, which meant a machine served exactly ONE request at a time no matter how
# many cores it had -- the single biggest reason a volunteer node was worth so much less than
# its hardware suggested (TOKENOMICS.md §12.5). Concurrent compute is now handled by
# batching.MicroBatcher, which serves a whole batch of requests in one forward pass instead
# of serialising them.
compute_lock = threading.Lock()


def _layer_set(slice_dir):
    """The SET of decoder layer indices present in a downloaded slice, or None.

    Read straight from the safetensors header (8-byte little-endian length, then JSON) so the
    BYTES decide what this node can serve, not a config file or a coordinator's claim. Returns
    None when it cannot be determined, which is treated as "do not block" -- a node that cannot
    read its own header has a bigger problem than a range check, and refusing to start on an
    unreadable header would take working nodes down for a check that is meant to catch a
    mismatch, not a corrupt file.

    The SET, not the extremes: a slice holding 0-9 and 19-27 has min 0 and max 27, so a
    bounds check passes an assignment of 0-27 while layers 10-18 are absent. See [P42].
    """
    import json
    try:
        with open(os.path.join(slice_dir, "model.safetensors"), "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if not 0 < n < 100 * 1024 * 1024:
                return None
            keys = json.loads(f.read(n).decode())
    except (OSError, ValueError):
        return None
    idx = set()
    for k in keys:
        parts = k.split(".")
        if k.startswith("model.layers.") and len(parts) > 2 and parts[2].isdigit():
            idx.add(int(parts[2]))
    return idx or None


def unmaterialized_layers(model, layer_start, layer_end):
    """Assigned decoder layers still holding meta tensors after a load. [] when all are real.

    **The second half of [P42], and it catches what the header check cannot.** `_layer_set`
    asks whether a layer is NAMED in the safetensors header; this asks whether it actually
    arrived. A truncated download, or a tensor mapped to a shard it is absent from, leaves a
    layer listed and unmaterialized — `load_slice_model` fills the skeleton with
    `strict=False`, so nothing raises, and the node serves uninitialized weights and returns
    fluent nonsense. Same ending as the 2026-08-07 incident, reached by the one path the range
    guard does not inspect.

    SCOPED to the assigned range, which is the whole difficulty. `Uraroga/spikingbrain-cpu-
    cluster` asserts this globally because its two ranks hold the whole model between them;
    here most of the model is legitimately meta, because `load_slice_model` builds a FULL
    skeleton and this node holds a slice of it. A global assertion would refuse every node on
    the network.

    Decoder layers only, deliberately: `embed_tokens` and `lm_head` belong to the DRIVER
    (`neuron_driver`, stage 1), and node_server only ever serves the `mid` and `last` roles.
    A node that has them is a node that was handed weights it will not run.

    Returns None when the model is not shaped the way this expects — unknown architecture is
    "do not block", the same answer an unreadable header gets.
    """
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return None
    bad = []
    for i in range(layer_start, min(layer_end + 1, len(layers))):
        for name, p in layers[i].named_parameters():
            if getattr(p, "is_meta", False):
                bad.append(f"layers.{i}.{name}")
    return bad


def _layers_in_slice(slice_dir):
    """(lo, hi) of the decoder layers present, or None. For messages, not for the check --
    the extremes cannot see a hole in the middle. `_layer_set` is what reload() guards on."""
    idx = _layer_set(slice_dir)
    return (min(idx), max(idx)) if idx else None

# Is this machine in the middle of serving somebody? Used by the auto-updater, which must never
# replace the app underneath a request in flight -- a dropped hop shows up to the driver as
# "socket closed mid-message" and the whole inference fails, for every user on that chain.
#
# compute_lock is NOT the signal: it only guards a slice reload now (the serving path moved to
# batching.MicroBatcher), so it is almost always free even mid-request. Counting live
# connections is what actually reflects work, and the timestamp covers the gaps between a
# chain's per-token round trips, when the connection is open but momentarily idle.
_serving_lock = threading.Lock()
_serving_conns = 0
_last_activity = 0.0


def _serving_enter():
    global _serving_conns, _last_activity
    with _serving_lock:
        _serving_conns += 1
        _last_activity = time.time()


def _serving_exit():
    global _serving_conns, _last_activity
    with _serving_lock:
        _serving_conns = max(0, _serving_conns - 1)
        _last_activity = time.time()


def is_busy(idle_seconds=120):
    """True if a connection is open, or one closed less than `idle_seconds` ago.

    The grace period is deliberate: a driver holds a chain across many token round trips and
    may reconnect between them, so "no open socket right now" does not mean "nobody is using
    this node". Erring towards busy only delays an update by a couple of minutes; erring the
    other way breaks somebody's answer.
    """
    with _serving_lock:
        if _serving_conns > 0:
            return True
        return (time.time() - _last_activity) < idle_seconds


# A never-set Event, used as the default for `self._reloading`. Test helpers construct a
# NodeServer without running __init__ (the point being to exercise reload/serve without a
# model), and a missing attribute there would raise AttributeError from the request path --
# turning a diagnostic into the outage it was added to describe.
_NOT_RELOADING = threading.Event()


def _empty_device_cache():
    """Return freed GPU blocks to the driver. A no-op on CPU, and never fatal.

    torch keeps a caching allocator: freeing a tensor returns its memory to torch, not to the
    GPU, so the card still reports it as in use. During a migration that is the difference
    between the new slice fitting and not — and `nvidia-smi` showing memory the owner cannot
    use is also what makes a volunteer think the agent is leaking.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                               # noqa: BLE001
        pass


def _weights_device(model):
    """Where this slice's weights ACTUALLY are, read off the loaded tensors.

    Deliberately NOT `common.DEVICE`. That is the configured *intent* -- what the process
    resolved at import -- and the gap between that intent and where the bytes really ended up
    is the whole reason `balancer.GPU_EXECUTION` had to be turned back off: the coordinator
    sized volunteers by VRAM for weights that were sitting in system RAM the entire time.
    A number nobody can check is how that survived a whole release, so this reads the tensors.

    `INSTALL.md` used to ask a first GPU volunteer to send back a `device: cuda:0` line that
    **cannot appear on any machine** -- it comes from `common.device_name()`, whose only caller
    is on the bench path an agent never reaches. This line replaces that request with one the
    node actually prints.

    Meta tensors are skipped: `load_slice_model` fills a full skeleton with `strict=False`, so
    every layer this node does NOT hold stays meta, and those are not where the weights are.
    Never raises -- a diagnostic that can stop a node from starting is worse than no
    diagnostic.
    """
    try:
        for p in model.parameters():
            if p.device.type != "meta":
                return str(p.device)
        return "meta (no materialized weights)"
    except Exception:                                               # noqa: BLE001
        return "unknown"


class NodeServer:
    def __init__(self, slice_dir, layer_start, layer_end, total_layers, paused_flag=None):
        self.lo = self.hi = self.n = None
        self.model = None
        self._batchers = {}
        self._batcher_lock = threading.Lock()
        self.paused = paused_flag if paused_flag is not None else threading.Event()  # set = paused
        # Whether this server is actually ACCEPTING connections. agent.py runs run() in a
        # daemon thread, so a failed bind used to kill that thread silently while the agent
        # kept heartbeating "active" forever -- the coordinator then advertised a healthy
        # 28/28 network and routed real requests into a node refusing every connection
        # ([P21], observed live on the Pavilion). Nothing may advertise availability without
        # checking this first.
        self.listening = threading.Event()
        self.bind_error = None
        # Set while the served slice is being swapped. reload() releases the old model before
        # loading the new one, so there is a real window with no weights; requests arriving in
        # it get a named refusal instead of an AttributeError from somewhere deep in a batcher.
        self._reloading = threading.Event()
        self.reload(slice_dir, layer_start, layer_end, total_layers)

    def reload(self, slice_dir, layer_start, layer_end, total_layers):
        """Hot-swap the served slice in place (model migration, Build 3 node-side). The new
        slice is loaded OUTSIDE compute_lock (I/O + weight materialization is the slow part);
        only the pointer swap is locked, so it can't land mid-forward-pass of an in-flight
        request. Existing connections keep using self.model/self.n by reference, so the very
        next request after the swap is served by the new slice with no reconnect needed."""
        print(f"[node] loading slice from {slice_dir} (layers {layer_start}-{layer_end}) ...")
        t0 = time.time()
        # Refuse to serve layers this slice does not contain. `load_slice_model` builds a FULL
        # model skeleton from config.json and fills in whatever the file holds (strict=False),
        # so a missing layer is not an error -- it stays an uninitialized meta tensor and the
        # forward pass runs on garbage. No exception, no wrong-looking output at this end: the
        # node answers confidently and the user gets fluent nonsense.
        #
        # Seen live 2026-08-07 on agent-optinovate-67e4eb: the coordinator had it on 0-27 after
        # a re-split, the disk held only 19-27 from an earlier assignment, and it started up,
        # announced "serving layers 0-27", passed its own ms/layer benchmark and reported
        # healthy. Two thirds of the model it claimed to serve was never downloaded.
        # Checked against the SET of layers present, not its extremes. The 2026-08-07 case
        # above is caught either way (min 19 > 0), but a slice holding 0-9 and 19-27 has
        # min 0 and max 27, so a bounds check would wave through an assignment of 0-27 with
        # layers 10-18 missing -- the same uninitialized-meta-tensor ending, reached by a
        # path the old check could not see. [P42].
        present = _layer_set(slice_dir)
        if present is not None:
            missing = sorted(set(range(layer_start, layer_end + 1)) - present)
            if missing:
                held = (min(present), max(present))
                gap = (f"{missing[0]}-{missing[-1]}" if len(missing) > 1
                       else str(missing[0]))
                raise RuntimeError(
                    f"refusing to serve layers {layer_start}-{layer_end}: this slice holds "
                    f"{held[0]}-{held[1]} and is missing {gap} "
                    f"({len(missing)} layer{'s' if len(missing) != 1 else ''}). Serving the "
                    f"gap would run uninitialized weights and return plausible nonsense. "
                    f"Delete {slice_dir} to re-download the right range.")
        # RELEASE THE OLD SLICE BEFORE LOADING THE NEW ONE.
        #
        # This used to be `model = load_slice_model(...)` with `self.model` still holding the
        # previous slice, so both existed at once and the node peaked at ~150% of one slice —
        # during a migration, on machines chosen precisely because they had room for one. The
        # peak is the number that decides whether a volunteer gets OOM-killed, not the steady
        # state.
        #
        # The cost of this ordering, stated rather than discovered later: a load that FAILS now
        # leaves the node with no model instead of still serving the old slice. That is the
        # right trade here — the node is being migrated off that slice anyway, and the
        # coordinator's re-placement is what recovers it — but it is a real change, which is
        # why the failure path below is explicit rather than incidental.
        if getattr(self, "_reloading", None) is None:
            self._reloading = threading.Event()   # subclasses in tests bypass __init__
        self._reloading.set()
        try:
            with compute_lock:
                for b in getattr(self, "_batchers", {}).values():
                    b.stop()                     # batchers close over the OLD model
                self._batchers = {}
                self.model = None
            gc.collect()                         # drop the last reference before allocating
            _empty_device_cache()
            model = load_slice_model(slice_dir)
            # [P42] level 2. The header said these layers were here; this asks whether they
            # arrived. Raised BEFORE the pointer swap, so a node that fails this keeps serving
            # nothing rather than serving garbage — `self.model` is already None by now, and
            # the coordinator's re-placement is what recovers it (see the ordering note above).
            unfilled = unmaterialized_layers(model, layer_start, layer_end)
            if unfilled:
                shown = ", ".join(unfilled[:4]) + (" …" if len(unfilled) > 4 else "")
                raise RuntimeError(
                    f"refusing to serve layers {layer_start}-{layer_end}: {len(unfilled)} "
                    f"weight tensor(s) in the assigned range are still uninitialized after "
                    f"loading {slice_dir} ({shown}). The header listed these layers but the "
                    f"file did not deliver them — a truncated or partially-written download. "
                    f"Serving them would return plausible nonsense. Delete {slice_dir} to "
                    f"re-download.")
            with compute_lock:
                self.model = model
                self.lo, self.hi, self.n = layer_start, layer_end, total_layers
        finally:
            # Cleared whether the load worked or not: a node stuck reporting "reloading"
            # forever is a node that never recovers and never says why.
            self._reloading.clear()
        print(f"[node] slice ready in {time.time()-t0:.1f}s | serving layers "
              f"{layer_start}-{layer_end} | weights on {_weights_device(model)}")

    def _batcher(self, role, lo, hi):
        """One MicroBatcher per (role, layer range). Keyed rather than global because a
        batch's slots must all run the SAME layers -- the range arrives in the caller's
        config, so it is not safe to assume every connection asked for the same one.

        Named lo/hi, not s1/s2, deliberately. They were s1/s2, and the LAST role is called as
        `_batcher("last", s2, self.n)` -- so the parameter named `s1` held the real s2 while
        the closure used `s2`, which was self.n. `layers[self.n:]` is EMPTY, so the last node
        ran zero layers and just normed whatever arrived. The chain still produced fluent-
        looking tokens, which is why it took an end-to-end read of the actual text to catch:
        "There noinspectionably..." instead of "The sky is blue because...".
        """
        key = (role, lo, hi)
        with self._batcher_lock:
            b = self._batchers.get(key)
            if b is None:
                model = self.model
                if role == "last":
                    def run(h, cache, lengths):
                        return batching.last_stage_batched(model, lo, h, cache, lengths)
                else:
                    def run(h, cache, lengths):
                        return batching.mid_stage_batched(model, lo, hi, h, cache, lengths)
                b = batching.MicroBatcher(run)
                self._batchers[key] = b
            return b

    # Set by the agent once it knows its identity. Without it this node cannot open a grant,
    # so it cannot accept a secure connection -- and it says so rather than silently falling
    # back to plaintext, because a node that quietly downgrades is the whole reason [P52]
    # could be true while the site claimed otherwise.
    node_token = None
    node_id = None
    # Grants already used, so one cannot be replayed inside its TTL. Bounded
    # by clearing wholesale: every entry older than GRANT_TTL_S is already
    # unusable, so forgetting them costs nothing and an unbounded set on a
    # node that runs for months does not.

    _grants_seen = None
    _grants_since = 0.0

    def _grants(self):
        now = time.time()
        if self._grants_seen is None or now - self._grants_since > wire_crypto.GRANT_TTL_S:
            type(self)._grants_seen = set()
            type(self)._grants_since = now
        return self._grants_seen

    def _maybe_secure(self, conn):
        """Establish the [P52] channel if the caller asked for one. Returns True if secure.

        THE ROLLING UPGRADE LIVES HERE. A legacy driver opens with `common.send_msg`'s 8-byte
        length, whose first four bytes are always zero for any real message; a secure driver
        leads with `NRNS`. Four bytes decide, and a legacy sender loses nothing because those
        four bytes are still its own length prefix, pushed back for `recv_msg` to read.

        Refusing plaintext outright would strand every 0.20.5 node on the network mid-upgrade,
        which is why this accepts both for now. `REQUIRE_SECURE` flips that once the fleet has
        moved -- one deliberate line, with evidence, the same shape as
        STAGE1_FAILURES_ARE_SCORED.
        """
        first = common._recv_all(conn, 4)
        if not wire_crypto.peek_is_secure(first):
            # Not secure. Hand those four bytes back so recv_msg sees an intact stream.
            common.pushback(conn, first)
            if REQUIRE_SECURE:
                raise wire_crypto.HandshakeError(
                    "this node accepts only encrypted connections (NEURON_REQUIRE_SECURE=1)")
            return False
        if not (self.node_token and self.node_id):
            raise wire_crypto.HandshakeError(
                "a secure connection was offered but this node has no identity to open it with")
        ch, request_id = wire_crypto.server_handshake(
            conn, self.node_token, self.node_id, seen=self._grants(), magic_consumed=True)
        del request_id
        common.attach_channel(conn, ch)
        return True

    def serve(self, conn, addr):
        cache, past, role, s1, s2, bconn = None, 0, None, None, None, None
        codec = bcodec = None
        try:
            try:
                self._maybe_secure(conn)
            except wire_crypto.HandshakeError as e:
                # A failed handshake is not a node fault and must not look like one. Log it and
                # drop the connection: there is nothing to reply WITH, since we have no key.
                # print, not logging: this module reports through print everywhere else, and
                # a refused connection must be visible in the same place as everything else.
                print(f"[node] refused a connection from {addr}: {e}")
                return
            while True:
                msg = common.recv_msg(conn)
                mtype = msg.get("type")

                if mtype == "config":
                    # PAUSE IS CHECKED HERE, not at accept(), and only on `config`.
                    #
                    # `config` is the message that starts a NEW request; `act` continues one
                    # already in flight. Refusing here therefore stops new work while letting
                    # an answer somebody is already waiting for finish — which is the whole
                    # point of a pause that does not punish the user who asked first.
                    #
                    # It is a typed reply rather than a closed socket on purpose. The driver's
                    # DEAD_PEER tuple is (ConnectionError, TimeoutError, EOFError, OSError),
                    # and ConnectionRefusedError is a subclass of ConnectionError — so
                    # refusing the TCP connection is indistinguishable from this machine
                    # dying. The driver would tear the chain down, ask for a new one, replay
                    # the whole junction cache into it, and tell the user "a machine dropped
                    # out" ([P28]'s complaint, self-inflicted). A named refusal lets the peer
                    # say what actually happened.
                    if self.paused.is_set():
                        common.send_msg(conn, {"ok": False, "error": "paused",
                                               "detail": "this node is paused by its owner; "
                                                         "it is not accepting new requests"})
                        return
                    # Same typed-refusal mechanism, different reason: mid-reload this node
                    # genuinely has no weights (reload() frees the old slice before loading
                    # the new one, to halve the migration peak). Named, so the driver reroutes
                    # instead of reading an AttributeError as a crash.
                    if (getattr(self, "_reloading", _NOT_RELOADING).is_set()
                            or self.model is None):
                        common.send_msg(conn, {"ok": False, "error": "reloading",
                                               "detail": "this node is loading a new model "
                                                         "slice; try another node"})
                        return
                    cache, past = common.new_cache(), 0
                    is_true_last = (self.hi == self.n - 1)
                    # Negotiated per hop and per connection: the caller lists what it can
                    # decode, we answer with our pick (or omit the field, which keeps an
                    # un-upgraded caller on the legacy format). See wire_codec.
                    codec = wire_codec.negotiate(msg.get("wire"))
                    ack_wire = {"wire": codec} if codec else {}
                    if "host_b" in msg:                      # MIDDLE relay role (real pipeline traffic)
                        role, s1, s2 = "middle", msg["s1"], msg["s2"]
                        bconn = socket.create_connection((msg["host_b"], msg["port_b"]),
                                                         timeout=common.COLD_CONNECT_TIMEOUT_S)
                        # [P52] encrypt the onward hop as well, with the grant the driver
                        # carried down. Without this the chain is private only as far as the
                        # first machine, which is a privacy claim that is true of one link and
                        # false of the request.
                        _gb = msg.get("grant_b")
                        if _gb:
                            try:
                                _ch = wire_crypto.client_handshake(
                                    bconn, base64.b64decode(_gb), msg.get("node_b", ""))
                                common.attach_channel(bconn, _ch)
                            except wire_crypto.HandshakeError as e:
                                bconn.close()
                                raise ConnectionError(
                                    f"next hop {msg.get('node_b')} failed the handshake: {e}")
                        common.send_msg(bconn, {"type": "config", "s2": s2, "n": msg.get("n", self.n),
                                                "wire": wire_codec.preference(self.model.config.hidden_size)})
                        back = common.recv_msg(bconn)
                        assert back.get("ok"), f"next hop refused: {back}"
                        bcodec = wire_codec.negotiate([back["wire"]] if back.get("wire") else None)
                        bconn.settimeout(common.HOT_TIMEOUT_S)
                        common.send_msg(conn, {"ok": True, "layers": self.n, "s1": s1, "s2": s2,
                                               **ack_wire})
                    elif is_true_last and "s1" not in msg:    # LAST stage role (real pipeline traffic)
                        # `and "s1" not in msg` is load-bearing, and it is what let the driver be
                        # unpayable for its entire existence ([P47]). `is_true_last` asks what
                        # THIS node holds, never what it was ASKED -- so a machine holding the
                        # whole model (0-27 of 28) is "true last" for every question, including a
                        # verifier's probe about layers 0-9. It then ran `last_stage(model, 10)`
                        # = layers[10:] + the final norm, and answered a question nobody asked:
                        # deterministic, confident, and wrong by max_err 28.6 -- the unexplained
                        # figure in [P47], live again on 2026-08-17.
                        #
                        # `s1` is the discriminator because the two callers genuinely differ, and
                        # not by convention: real pipeline traffic reaches a last stage from the
                        # middle relay 12 lines above, which sends {"s2", "n", "wire"} and never
                        # an `s1` -- there is nothing for it to mean, since the last stage's own
                        # start is implied by `s2`. `challenge_middle_node` always sends one.
                        # So the presence of `s1` on a config with no `host_b` can only be a
                        # verifier asking about a specific range, which is the probe.
                        #
                        # A full-model node now answers the probe from its own lo/hi (0-27), the
                        # ack says so, and `challenge_middle_node` raises RangeMismatch: the node
                        # is fine, the placement is stale, nothing recorded in either direction.
                        # That is the [P37] outcome and it is the correct one -- refuse a range
                        # rather than run garbage for it.
                        # `s2` arrives from the CALLER and decides which layers actually run --
                        # `last_stage(model, s2)` is `layers[s2:]`. It used to be taken on trust.
                        # A caller working from a stale placement then asks for a range this
                        # slice does not hold; those layers are uninitialized meta tensors, the
                        # forward pass raises something that is NOT a ConnectionError, the
                        # connection thread dies, and `finally: conn.close()` slams the socket.
                        # What the far end sees is a bare "socket closed mid-message" -- so a
                        # node that was asked for layers it never downloaded is indistinguishable
                        # from one that hung up, and proof-of-compute records it as a failure.
                        # Live 2026-08-10: agent-bhpc012104 held 14-27 while the coordinator had
                        # it on 10-27, and it failed every challenge this way until it was
                        # flagged. Same rule as reload()'s _layers_in_slice guard, one level up:
                        # refuse a range rather than run garbage for it.
                        s2 = msg["s2"]
                        if s2 < self.lo:
                            common.send_msg(conn, {
                                "ok": False, "error": "range_mismatch",
                                "detail": f"asked to serve layers {s2}-{self.n - 1}, but this "
                                          f"node holds {self.lo}-{self.hi}: layers "
                                          f"{s2}-{self.lo - 1} were never downloaded here",
                                "holds": [self.lo, self.hi]})
                            return
                        role = "last"
                        # `holds` is what this node ACTUALLY has, so a caller can tell a
                        # placement disagreement from a wrong answer. The last-stage ack used to
                        # echo the caller's own s2 back at it, which can never disagree and so
                        # could never reveal anything; the PROBE branch below has always reported
                        # its real range, which is why only middle-node challenges could ever
                        # say what went wrong.
                        common.send_msg(conn, {"ok": True, "layers": msg.get("n", self.n),
                                               "s2": s2, "holds": [self.lo, self.hi],
                                               **ack_wire})
                    else:
                        # PROBE role (security/proof_of_compute.py): a config with no host_b,
                        # on a node whose own range does NOT reach the model's final layer, can
                        # only mean a verifier challenging this node's layers in isolation --
                        # calling last_stage() here would be WRONG (and likely crash: this
                        # shard was downloaded without norm/later layers, which stay on the
                        # meta device, uninitialized). Uses OUR OWN self.lo/self.hi, never the
                        # caller's claimed s1/s2 -- this tests what we actually loaded, not
                        # what a challenger asserts.
                        # self.hi is the INCLUSIVE last layer this node owns, but s2 is used
                        # as a Python slice bound (`layers[s1:s2]`, see common.mid_stage), so
                        # it has to be hi+1. Passing hi ran one layer too few and advertised a
                        # range the verifier rejects -- which meant proof-of-compute could
                        # never promote a probationary node on any segment except the last,
                        # and auto-placement puts a joining stranger wherever the GAP is.
                        role, s1, s2 = "probe", self.lo, self.hi + 1
                        common.send_msg(conn, {"ok": True, "layers": self.n, "s1": s1, "s2": s2,
                                               "holds": [self.lo, self.hi], **ack_wire})

                elif mtype == "act":
                    # An in-flight request whose node started reloading underneath it. Before
                    # the release-then-load change the old model stayed alive by reference and
                    # this could not happen; now it can, and it must not surface as an
                    # AttributeError from inside a batcher — which reaches the driver as a
                    # closed socket with no explanation.
                    if self.model is None:
                        common.send_msg(conn, {"ok": False, "error": "reloading",
                                               "detail": "this node released its slice to load "
                                                         "another; this request must reroute"})
                        return
                    hidden = msg["hidden"]
                    q = hidden.shape[1]
                    # Submitting instead of locking is the whole change: concurrent requests
                    # now ride the SAME forward pass rather than queueing for the machine.
                    # The reported *_compute_ms therefore includes any time spent waiting to
                    # fill a batch (capped at NEURON_BATCH_WINDOW_MS) -- it is what the hop
                    # actually cost the caller, which is what the driver's net_ms accounting
                    # and the coordinator's balancer both want.
                    if role == "middle":
                        tc = time.time()
                        h2 = self._batcher("middle", s1, s2).submit(hidden, cache, past)
                        c_ms = (time.time() - tc) * 1000
                        past += q
                        # _to_cpu at the wire boundary, on every tensor leaving this node.
                        # The batched stages already return CPU, but `codec` is None whenever
                        # negotiation falls back — and the legacy path is a bare torch.save,
                        # which happily serialises a CUDA tensor. The peer then fails to
                        # deserialise it, so a GPU node would break its CPU neighbour rather
                        # than itself. Identity on a CPU-only machine.
                        common.send_msg(bconn, {"type": "act", "hidden": common._to_cpu(h2)},
                                        codec=bcodec)
                        resp = common.recv_msg(bconn)
                        common.send_msg(conn, {"hidden": common._to_cpu(resp["hidden"]),
                                               "c_compute_ms": c_ms,
                                               "b_compute_ms": resp["b_compute_ms"]}, codec=codec)
                    elif role == "probe":
                        tc = time.time()
                        h2 = self._batcher("probe", s1, s2).submit(hidden, cache, past)
                        c_ms = (time.time() - tc) * 1000
                        past += q
                        common.send_msg(conn, {"hidden": common._to_cpu(h2),
                                               "c_compute_ms": c_ms}, codec=codec)
                    else:  # last
                        tb = time.time()
                        out = self._batcher("last", s2, self.n).submit(hidden, cache, past)
                        b_ms = (time.time() - tb) * 1000
                        past += q
                        common.send_msg(conn, {"hidden": common._to_cpu(out),
                                               "b_compute_ms": b_ms}, codec=codec)

                elif mtype == "bye":
                    if bconn:
                        common.send_msg(bconn, {"type": "bye"})
                    return
        finally:
            if bconn:
                try:
                    bconn.close()
                except OSError:
                    pass

    def run(self, host, port):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind((host, port))
            srv.listen(16)
        except OSError as e:
            # Reported, not raised: this usually runs in a daemon thread, where an exception
            # is swallowed and the caller never learns the node is deaf. The agent polls
            # bind_error/listening and refuses to advertise this node until it clears.
            self.bind_error = f"{e.__class__.__name__}: {e}"
            print(f"[node] FAILED to bind {host}:{port} — {self.bind_error}")
            srv.close()
            return
        self.bind_error = None
        self.listening.set()
        print(f"[node] listening on {host}:{port}  (Ctrl-C to stop)")
        try:
            while True:
                conn, addr = srv.accept()
                threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()
        finally:
            self.listening.clear()
            srv.close()

    def _handle(self, conn, addr):
        _serving_enter()
        try:
            self.serve(conn, addr)
        # TimeoutError is a sibling of ConnectionError under OSError, not caught by it --
        # see node_c.py's handle() for why this matters (a slow next-hop cold-start would
        # otherwise die as an uncaught thread exception and silently slam this connection
        # shut, surfacing upstream as an unexplained "socket closed mid-message").
        except (ConnectionError, TimeoutError, EOFError) as e:
            print(f"[node] conn {addr} ended: {e}")
        # EVERYTHING else, for the same reason one line up. The three above are the EXPECTED
        # ways a connection ends; anything else -- a meta tensor in a forward pass, an OOM, a
        # KeyError on a malformed message -- used to escape this handler entirely, kill the
        # thread, and reach `finally: conn.close()` anyway. The peer's diagnosis was then the
        # closed socket and nothing else, which reads as "that machine died" no matter what
        # really happened ([P28]) and, to the verifier, as a failed proof-of-compute. Caught and
        # named here so the traceback lands in THIS node's log, where the fault actually is.
        except Exception as e:                                          # noqa: BLE001
            print(f"[node] conn {addr} FAILED: {e.__class__.__name__}: {e}")
            traceback.print_exc()
        finally:
            _serving_exit()
            conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-dir", required=True)
    ap.add_argument("--layer-start", type=int, required=True)
    ap.add_argument("--layer-end", type=int, required=True)
    ap.add_argument("--total-layers", type=int, default=28)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=50999)
    args = ap.parse_args()
    NodeServer(args.slice_dir, args.layer_start, args.layer_end, args.total_layers).run(
        args.host, args.port)


if __name__ == "__main__":
    main()
