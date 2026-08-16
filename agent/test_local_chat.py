"""agent/test_local_chat.py — bundled personal Chat UI wiring. Run: python -m agent.test_local_chat

Monkeypatches slice_downloader.download_slice, neuron_driver.DRIVER, and uvicorn.Server.run --
no real network, no real model weights, no real port binding. The actual end-to-end behavior
(real slice download + real HTTP requests against a live driver) was verified manually against
the real coordinator; this covers the wiring logic that's cheap and safe to run every time.
"""
import json
import os
import tempfile
import time

import agent.agent as agentmod
import agent.local_chat as local_chat
import neuron_driver
import slice_downloader

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def _agent(tmpdir, **cfg_overrides):
    cfg_path = os.path.join(tmpdir, "config.json")
    cfg = dict(agentmod.DEFAULT_CONFIG)
    cfg.update(node_id="test-node", node_token="tok-1", slice_dir="./model_slice/")
    cfg.update(cfg_overrides)
    import json
    json.dump(cfg, open(cfg_path, "w"))
    return agentmod.Agent(config_path=cfg_path)


def main():
    tmpdir = tempfile.mkdtemp(prefix="neuron_local_chat_test_")

    # ---- ensure_driver_slice: the shard must be the RIGHT one, not merely present ---- #
    #
    # These checks used to be "does model.safetensors exist". That was correct only while s1
    # could never change, which is precisely what [P44] changed: the width now comes from the
    # coordinator, so a cached shard of the wrong width is worse than none at all -- the driver
    # would assert a stage 1 the coordinator is not planning and `coord_get_chain` would refuse
    # every request, on a machine whose weights look perfectly fine.
    calls = []
    real_download = slice_downloader.download_slice
    slice_downloader.download_slice = lambda *a, **k: calls.append((a, k))

    def _seed(name, model_id, ls, le):
        """A driver slice dir as `download_slice` would leave it: weights + marker."""
        d = os.path.join(tmpdir, name)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "model.safetensors"), "w").close()
        with open(os.path.join(d, slice_downloader.SLICE_MARKER), "w") as f:
            json.dump({"model_id": model_id, "layer_start": ls, "layer_end": le}, f)
        return d

    try:
        matching = _seed("already-there", "some/model", 0, 9)
        local_chat.ensure_driver_slice("some/model", matching, s1=10)
        check("a shard of the right width and model is reused", calls == [])

        fresh_dir = os.path.join(tmpdir, "fresh")
        local_chat.ensure_driver_slice("some/model", fresh_dir, s1=10)
        check("ensure_driver_slice downloads when missing", len(calls) == 1)
        args, kwargs = calls[0]
        check("downloads layers 0..s1-1", args[1:3] == (0, 9))
        check("downloads with is_first_node=True, is_last_node=False (tied lm_head)",
              kwargs.get("is_first_node") is True and kwargs.get("is_last_node") is False)

        # The case the old contract could not see, and the reason for the change.
        calls.clear()
        narrow = _seed("narrow", "some/model", 0, 9)
        local_chat.ensure_driver_slice("some/model", narrow, s1=18)
        # A 10-wide shard kept while the coordinator plans 18 makes coord_get_chain refuse
        # every chain, on a machine whose weights look perfectly fine.
        check("a shard of the WRONG width is discarded and re-fetched",
              len(calls) == 1 and calls[0][0][1:3] == (0, 17))
        check("...and the stale directory is actually gone, not left to be half-read",
              not os.path.exists(os.path.join(narrow, slice_downloader.SLICE_MARKER)))

        calls.clear()
        wrong_model = _seed("wrongmodel", "other/model", 0, 9)
        local_chat.ensure_driver_slice("some/model", wrong_model, s1=10)
        # [P36]: layer numbers cannot tell two models apart.
        check("a shard of the right width but the WRONG MODEL is discarded too",
              len(calls) == 1)

        calls.clear()
        unrecorded = os.path.join(tmpdir, "unrecorded")
        os.makedirs(unrecorded)
        open(os.path.join(unrecorded, "model.safetensors"), "w").close()
        local_chat.ensure_driver_slice("some/model", unrecorded, s1=10)
        # Provenance we cannot establish is the case that produced [P36]; one re-download is
        # cheap against a driver that refuses every request.
        check("an UNRECORDED shard is treated as a mismatch, not trusted", len(calls) == 1)

        calls.clear()
        local_chat.ensure_driver_slice("some/model", os.path.join(tmpdir, "defaulted"))
        check("no s1 given falls back to the compiled-in default",
              calls[0][0][1:3] == (0, neuron_driver.S1 - 1))
    finally:
        slice_downloader.download_slice = real_download

    # ---- the width comes from the coordinator, and never blocks the driver ---- #
    real_get = local_chat.requests.get

    class _R:
        def __init__(self, code, payload):
            self.status_code, self._p = code, payload

        def json(self):
            return self._p

    try:
        local_chat.requests.get = lambda *a, **k: _R(200, {"driver_stage1_layers": 18})
        check("the coordinator's width is used when it answers",
              local_chat.driver_stage1_layers("http://c", "n1") == 18)

        local_chat.requests.get = lambda *a, **k: _R(200, {})
        check("an older coordinator that omits the field falls back",
              local_chat.driver_stage1_layers("http://c", "n1", fallback=14) == 14)

        local_chat.requests.get = lambda *a, **k: _R(503, {})
        check("a coordinator error falls back rather than raising",
              local_chat.driver_stage1_layers("http://c", "n1", fallback=14) == 14)

        def _boom(*a, **k):
            raise OSError("network is down")

        local_chat.requests.get = _boom
        # Being unreachable is when a personal Chat UI matters most; a driver that will not
        # load because a status call timed out has turned an outage into a local one.
        check("an UNREACHABLE coordinator falls back to what is on disk",
              local_chat.driver_stage1_layers("http://c", "n1", fallback=14) == 14)
        check("...and to the compiled-in default when there is no disk either",
              local_chat.driver_stage1_layers("http://c", "n1") == neuron_driver.S1)
        check("no coordinator configured at all is not an error",
              local_chat.driver_stage1_layers(None, None, fallback=12) == 12)
    finally:
        local_chat.requests.get = real_get

    # ---- Agent.start_local_chat(): config gates whether local_chat.start() is called ---- #
    started_with = []
    real_start = local_chat.start
    local_chat.start = lambda *a, **k: started_with.append((a, k))
    try:
        a1 = _agent(tmpdir, local_chat=False, model_id="m")
        a1.start_local_chat()
        check("local_chat: false -> local_chat.start() never called", started_with == [])

        a2 = _agent(tmpdir, local_chat=True, model_id=None)
        a2.start_local_chat()
        check("model_id not yet known -> local_chat.start() never called (setup() hasn't run)",
              started_with == [])

        a3 = _agent(tmpdir, local_chat=True, model_id="Qwen/Qwen2.5-1.5B-Instruct",
                    local_chat_port=9999)
        a3.start_local_chat()
        check("local_chat: true + known model_id -> local_chat.start() IS called",
              len(started_with) == 1)
        args, kwargs = started_with[0]
        check("passes this agent's coordinator base URL", args[0] == a3.base)
        check("passes the model this agent is actually serving", args[1] == "Qwen/Qwen2.5-1.5B-Instruct")
        check("uses the configured port", kwargs.get("port") == 9999)

        started_with.clear()
        a4 = _agent(tmpdir, model_id="m")   # local_chat defaults True (not explicitly set)
        a4.start_local_chat()
        check("local_chat defaults to on when unset", len(started_with) == 1)
    finally:
        local_chat.start = real_start

    # ---- local_chat.start(): a broken driver/slice load never propagates or crashes ---- #
    real_ensure = local_chat.ensure_driver_slice
    local_chat.ensure_driver_slice = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        result = local_chat.start("http://coord.example", "some/model",
                                  os.path.join(tmpdir, "broken"))
        check("a failed slice download returns None instead of raising", result is None)
    finally:
        local_chat.ensure_driver_slice = real_ensure

    # ---- local_chat.start(): happy path actually returns a running server ---- #
    local_chat.ensure_driver_slice = lambda *a, **k: None
    real_load = neuron_driver.DRIVER.load_from_slice
    # DRIVER is an INSTANCE -- assigning here makes this a plain instance attribute, not a
    # bound method, so Python won't auto-pass `self`; the replacement takes just slice_dir.
    neuron_driver.DRIVER.load_from_slice = lambda slice_dir: None
    import uvicorn
    real_run = uvicorn.Server.run

    def fake_run(self):
        self.started = True   # what agent code polls for; real run() sets this too

    uvicorn.Server.run = fake_run
    try:
        server = local_chat.start("http://coord.example", "some/model",
                                  os.path.join(tmpdir, "happy"), port=59999)
        check("happy path returns a server object, not None", server is not None)
        for _ in range(20):
            if getattr(server, "started", False):
                break
            time.sleep(0.05)
        check("server thread actually ran (started flag set)", getattr(server, "started", False))
    finally:
        uvicorn.Server.run = real_run
        neuron_driver.DRIVER.load_from_slice = real_load

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
