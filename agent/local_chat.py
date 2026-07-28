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
import threading

log = logging.getLogger("neuron.agent.local_chat")

DEFAULT_PORT = 8080

# config.json's "oauth" keys -> the env vars ui/oauth.py reads at import time. config.json is
# the only place a packaged, console-less desktop install can supply these -- there's no shell
# to set NEURON_GOOGLE_CLIENT_ID etc. before a double-clicked tray app starts.
_OAUTH_ENV_MAP = {
    "google_client_id": "NEURON_GOOGLE_CLIENT_ID",
    "google_client_secret": "NEURON_GOOGLE_CLIENT_SECRET",
    "github_client_id": "NEURON_GITHUB_CLIENT_ID",
    "github_client_secret": "NEURON_GITHUB_CLIENT_SECRET",
    "session_secret": "NEURON_SESSION_SECRET",
    "wallet_link_secret": "NEURON_WALLET_LINK_SECRET",
}


def ensure_driver_slice(model_id, target_dir):
    """Download the fixed driver shard if not already present. Reuses the exact same
    byte-range slice mechanism as a compute-node's slice (slice_downloader) -- just a
    DIFFERENT, fixed layer range (0..S1-1) with is_first_node=True, which for a tied-
    lm_head model (Qwen2.5) also pulls in everything this role needs (no separate
    lm_head.weight tensor exists to fetch)."""
    import slice_downloader
    from neuron_driver import S1

    weights = os.path.join(target_dir, "model.safetensors")
    if os.path.exists(weights):
        log.info("personal driver slice already present (%s) — skipping download", target_dir)
        return target_dir
    log.info("downloading personal driver slice (layers 0-%d, ~1.4GB) ...", S1 - 1)
    slice_downloader.download_slice(model_id, 0, S1 - 1, target_dir,
                                    is_first_node=True, is_last_node=False)
    return target_dir


def start(coordinator, model_id, slice_dir, port=DEFAULT_PORT, host="127.0.0.1", oauth_cfg=None):
    """Download the driver slice (if needed) and serve the Chat UI on `host:port` in a
    background thread. Returns the running uvicorn.Server (call .should_exit = True to stop
    it) or None if startup failed -- a broken local chat UI must never take down the agent's
    actual node-serving role, so failures here are logged and swallowed, not raised."""
    try:
        ensure_driver_slice(model_id, slice_dir)

        # neuron_driver.DRIVER is a process-wide singleton also used by ui.app / api.
        # openai_compat -- pre-load it from OUR slice dir before ui.app's own lifespan
        # hook calls ensure_loaded(), which is then a no-op (self.model is already set).
        import neuron_driver
        neuron_driver.DRIVER.load_from_slice(slice_dir)

        os.environ.setdefault("NEURON_COORDINATOR", coordinator)
        # setdefault, not assignment: a real shell env var (dev testing) still wins over
        # whatever's saved in config.json.
        for cfg_key, env_name in _OAUTH_ENV_MAP.items():
            value = (oauth_cfg or {}).get(cfg_key)
            if value:
                os.environ.setdefault(env_name, value)
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
        log.warning("could not start the personal Chat UI (node-serving continues): %s", e)
        return None
