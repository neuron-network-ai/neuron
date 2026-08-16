"""agent/local_chat.py — bundles a personal Chat UI into the NEURON agent.

WHY: every installed agent can ALSO run its own local driver + Chat UI, so the person
running it gets their OWN front door to the network instead of everyone relying on one
centralized website. This is also where content moderation belongs (safety/moderation.py):
a compute node is blind to plaintext by design (see SAFETY.md) and can never judge content,
but THIS local driver already handles its own user's plaintext to generate their replies --
it's the correct place for a personal moderation gate too, same principle as ui/app.py's
existing gate, just running on every installation instead of one shared machine.

Downloads a FIXED driver shard (embed + layers[0:neuron_driver.S1] + lm_head, matching
node_a's own shape) separately from whatever compute slice this node happens to be assigned
to serve for the network -- the two are unrelated: a compute node's assigned range can be
anywhere the coordinator places it; the driver role is always the same fixed front segment.
Serves ui.app (the same Chat UI code, unmodified) on localhost by default -- NOT exposed to
the internet. Sharing it publicly is a separate, deliberate decision (e.g. the same relay
tunnel already used for compute nodes) left to the user, not the default.
"""
import logging
import os
import shutil
import threading

import requests

log = logging.getLogger("neuron.agent.local_chat")

DEFAULT_PORT = 8080

# Why the last start() failed, for whoever has to tell the user. None until something fails.
LAST_ERROR = None

# config.json's "oauth" keys -> the env vars ui/oauth.py reads at import time. config.json is
# the only place a packaged, console-less desktop install can supply these -- there's no shell
# to set NEURON_GOOGLE_CLIENT_ID etc. before a double-clicked tray app starts.
# An installed agent no longer holds any OAuth client secret: sign-in is delegated to the
# coordinator (coordinator/auth.py, ui/oauth.py), because a secret shipped to every stranger's
# PC is not a secret. What is left is this machine's OWN session signing key -- it only protects
# this browser's local cookie, never anything network-wide.
_OAUTH_ENV_MAP = {
    "session_secret": "NEURON_SESSION_SECRET",
}


def driver_stage1_layers(coordinator, node_id, fallback=None):
    """How wide stage 1 is, asked of the coordinator that decides it. [P44]

    The width used to come from `NEURON_S1`, read at import in BOTH this process and the
    coordinator's, and `node_a.coord_get_chain` refuses any chain whose stage 1 is not the
    driver's own shard -- so changing it meant a coordinated restart across machines including
    volunteers' PCs. The coordinator owns it now and publishes it on slice-info, which the
    agent already calls, so there is no extra request and nothing to keep in step by hand.

    **A driver that cannot reach the coordinator must not become unable to infer**, so every
    failure here falls back rather than raising: the caller passes what is already on disk, and
    the last resort is `neuron_driver.S1`. Being unreachable is the moment a personal Chat UI
    matters most, and a driver that refuses to load because a status endpoint timed out has
    turned a coordinator outage into a local one.

    Note this is a NETWORK fact, not this node's own range: a machine assigned layers 18-35
    still drives a chain whose stage 1 is 0-17, and its driver shard is a separate download.
    """
    import neuron_driver

    if coordinator and node_id:
        try:
            r = requests.get(f"{coordinator.rstrip('/')}/node/{node_id}/slice-info", timeout=15)
            if r.status_code == 200:
                s1 = r.json().get("driver_stage1_layers")
                if isinstance(s1, int) and s1 > 0:
                    return s1
                log.info("coordinator did not report driver_stage1_layers (an older build) — "
                         "using %s", fallback or neuron_driver.S1)
            else:
                log.warning("slice-info returned %s asking for the stage-1 width; using %s",
                            r.status_code, fallback or neuron_driver.S1)
        except Exception as e:
            log.warning("could not ask the coordinator for the stage-1 width (%s); using %s",
                        e, fallback or neuron_driver.S1)
    return fallback or neuron_driver.S1


def _stage1(coordinator, node_id, slice_dir):
    """The width to build this driver at: what the coordinator says, else what is already on
    disk, else the compiled-in fallback.

    The middle term is what keeps an offline start working. A driver that has run before holds
    a shard of a known width, and re-using it is strictly better than downloading a fallback
    width that may be wrong -- and better than refusing to start, which would turn a
    coordinator outage into a dead local Chat UI.
    """
    import slice_downloader
    have = slice_downloader.slice_range(slice_dir)
    on_disk = have[1] + 1 if have and have[0] == 0 else None
    return driver_stage1_layers(coordinator, node_id, fallback=on_disk)


def ensure_driver_slice(model_id, target_dir, s1=None):
    """Download the driver shard if what is on disk is not the shard we need.

    Reuses the exact same byte-range mechanism as a compute-node's slice (slice_downloader) --
    just a DIFFERENT layer range (0..s1-1) with is_first_node=True, which for a tied-lm_head
    model also pulls in everything this role needs (no separate lm_head.weight to fetch).

    `s1` now comes from the coordinator rather than the environment ([P44]), so the width can
    change -- and a cached shard of the WRONG width is worse than no shard: the driver would
    assert a stage 1 the coordinator is not planning and every request would be refused by its
    own consistency check. Existence is therefore no longer sufficient; the range on disk has
    to match. `os.path.exists(weights)` was the whole test before, which was correct only
    because the width could never change.
    """
    import slice_downloader
    import neuron_driver

    s1 = int(s1 or neuron_driver.S1)
    weights = os.path.join(target_dir, "model.safetensors")
    if os.path.exists(weights):
        have = slice_downloader.slice_range(target_dir)
        on_model = slice_downloader.slice_provenance(target_dir)
        if have == (0, s1 - 1) and on_model == model_id:
            log.info("personal driver slice already present (%s, layers 0-%d) — skipping "
                     "download", target_dir, s1 - 1)
            return target_dir
        # An unrecorded range is treated as a mismatch for the reason `agent.ensure_slice`
        # treats an unrecorded MODEL as one: provenance we cannot establish is exactly the
        # case that produces a shard serving something it is not.
        log.warning("driver slice on disk is %s of %s but this driver needs layers 0-%d of "
                    "%s — discarding it",
                    f"layers {have[0]}-{have[1]}" if have else "an unrecorded range",
                    on_model or "an unrecorded model", s1 - 1, model_id)
        shutil.rmtree(target_dir, ignore_errors=True)
    log.info("downloading personal driver slice (layers 0-%d) ...", s1 - 1)
    slice_downloader.download_slice(model_id, 0, s1 - 1, target_dir,
                                    is_first_node=True, is_last_node=False)
    return target_dir


def start(coordinator, model_id, slice_dir, port=DEFAULT_PORT, host="127.0.0.1", oauth_cfg=None,
          node_id=None, node_token=None):
    """Download the driver slice (if needed) and serve the Chat UI on `host:port` in a
    background thread. Returns the running uvicorn.Server (call .should_exit = True to stop
    it) or None if startup failed -- a broken local chat UI must never take down the agent's
    actual node-serving role, so failures here are logged and swallowed, not raised."""
    try:
        # Fetch exactly ONE set of weights, whichever this machine will actually use.
        # If it can run the model itself (engine/local_gguf.py) that is the ~1.1 GB quantized
        # build, and the ~1.4 GB pipeline-driver slice is never needed -- answers come back in
        # ~10s instead of minutes, and nothing is spent on a role this machine won't play.
        # Otherwise it takes the driver role for the node pipeline and needs the slice.
        from engine import local_gguf
        if local_gguf.can_serve(model_id):
            log.info("this machine can run %s itself — fetching quantized weights instead of "
                     "the pipeline-driver slice", model_id)
            if local_gguf.ensure_weights(model_id) is None:
                log.warning("quantized weights unavailable; falling back to the driver slice")
                ensure_driver_slice(model_id, slice_dir, _stage1(coordinator, node_id,
                                                                slice_dir))
                import neuron_driver
                neuron_driver.DRIVER.load_from_slice(slice_dir)
        else:
            # Ask the coordinator how wide stage 1 is BEFORE downloading, so the shard matches
            # what the chain will be planned as ([P44]). This is the whole of "no coordinated
            # restart": the width follows placement, and a change is carried by re-downloading
            # a shard rather than by editing an environment variable on every machine.
            ensure_driver_slice(model_id, slice_dir, _stage1(coordinator, node_id, slice_dir))
            # neuron_driver.DRIVER is a process-wide singleton also used by ui.app / api.
            # openai_compat -- pre-load it from OUR slice dir before ui.app's own lifespan
            # hook calls ensure_loaded(), which is then a no-op (self.model is already set).
            import neuron_driver
            neuron_driver.DRIVER.load_from_slice(slice_dir)

        os.environ.setdefault("NEURON_COORDINATOR", coordinator)
        # ui/oauth.py tells the coordinator which loopback port to hand the login back to.
        os.environ.setdefault("NEURON_LOCAL_CHAT_PORT", str(port))
        # setdefault, not assignment: a real shell env var (dev testing) still wins over
        # whatever's saved in config.json.
        for cfg_key, env_name in _OAUTH_ENV_MAP.items():
            value = (oauth_cfg or {}).get(cfg_key)
            if value:
                os.environ.setdefault(env_name, value)
        # This machine's own node identity, so the Chat UI can offer to record the wallet that
        # owns its earnings ([P39] phase 2). The UI runs in THIS process and proxies
        # server-side, exactly as it already does for the wallet payout routes -- the token
        # never reaches the browser. Absent when this machine is a driver only, and the UI
        # then simply does not show the prompt.
        if node_id:
            os.environ.setdefault("NEURON_NODE_ID", node_id)
        if node_token:
            os.environ.setdefault("NEURON_NODE_TOKEN", node_token)
        import uvicorn
        from ui.app import app

        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)

        def _run():
            log.info("personal Chat UI ready at http://%s:%d", host, port)
            try:
                server.run()
            except Exception as e:
                log.error("personal Chat UI stopped: %s", e)

        threading.Thread(target=_run, daemon=True, name="local-chat").start()
        return server
    except Exception as e:
        # Record WHY, not just that. start() returns None on failure, so the caller had no way
        # to report the cause and guessed at one -- it told users to check whether port 8080
        # was in use, when the real answer here was a missing _sqlite3 in the packaged build.
        # A confident wrong diagnosis costs more than no diagnosis.
        global LAST_ERROR
        LAST_ERROR = f"{type(e).__name__}: {e}"
        log.warning("could not start the personal Chat UI (node-serving continues): %s", e)
        return None
