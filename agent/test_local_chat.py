"""agent/test_local_chat.py — bundled personal Chat UI wiring. Run: python -m agent.test_local_chat

Monkeypatches slice_downloader.download_slice, neuron_driver.DRIVER, and uvicorn.Server.run --
no real network, no real model weights, no real port binding. The actual end-to-end behavior
(real slice download + real HTTP requests against a live driver) was verified manually against
the real coordinator; this covers the wiring logic that's cheap and safe to run every time.
"""
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

    # ---- ensure_driver_slice: skips download if the weights already exist ---- #
    calls = []
    real_download = slice_downloader.download_slice
    slice_downloader.download_slice = lambda *a, **k: calls.append((a, k))
    try:
        present_dir = os.path.join(tmpdir, "already-there")
        os.makedirs(present_dir)
        open(os.path.join(present_dir, "model.safetensors"), "w").close()
        local_chat.ensure_driver_slice("some/model", present_dir)
        check("ensure_driver_slice skips download when the slice already exists", calls == [])

        fresh_dir = os.path.join(tmpdir, "fresh")
        local_chat.ensure_driver_slice("some/model", fresh_dir)
        check("ensure_driver_slice downloads when missing", len(calls) == 1)
        args, kwargs = calls[0]
        check("downloads the FIXED driver range (layers 0..S1-1)",
              args[1:3] == (0, neuron_driver.S1 - 1))
        check("downloads with is_first_node=True, is_last_node=False (tied lm_head)",
              kwargs.get("is_first_node") is True and kwargs.get("is_last_node") is False)
    finally:
        slice_downloader.download_slice = real_download

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
