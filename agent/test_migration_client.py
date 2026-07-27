"""Node-side model-migration tests — run: python -m agent.test_migration_client

Coordinator-side migration (coordinator/test_migration.py, coordinator/test_serving_model.py)
proves the state machine + cutover persistence. This covers the NODE side: prepare (download
in the background, stay on the old model), report ready, and only swap once the coordinator
confirms cutover actually happened (never on an abort). Monkeypatches agent.requests and
slice_downloader.download_slice — no real network, no live coordinator, no real model weights.
"""
import json
import os
import shutil
import tempfile
import types

import agent.agent as agentmod
import slice_downloader

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


class FakeResp:
    def __init__(self, payload=None, status=200):
        self._payload, self.status_code = payload or {}, status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise agentmod.requests.RequestException(f"status {self.status_code}")


class FakeServer:
    def __init__(self):
        self.calls = []

    def reload(self, slice_dir, layer_start, layer_end, total_layers):
        self.calls.append((slice_dir, layer_start, layer_end, total_layers))


def _agent(tmpdir):
    cfg_path = os.path.join(tmpdir, "config.json")
    cfg = dict(agentmod.DEFAULT_CONFIG)
    cfg.update(node_id="test-node", node_token="tok-1", model_id="old/model",
              layer_start=0, layer_end=9, slice_dir="./slice_current/")
    json.dump(cfg, open(cfg_path, "w"))
    a = agentmod.Agent(config_path=cfg_path)
    a.server = FakeServer()
    return a


def main():
    tmpdir = tempfile.mkdtemp(prefix="neuron_migclient_")
    real_download, real_get, real_post = (
        slice_downloader.download_slice, agentmod.requests.get, agentmod.requests.post)
    downloads = []

    def fake_download(model_id, layer_start, layer_end, target_dir, is_first_node, is_last_node,
                      revision="main"):
        downloads.append((model_id, layer_start, layer_end, is_first_node, is_last_node))
        os.makedirs(target_dir, exist_ok=True)
        open(os.path.join(target_dir, "model.safetensors"), "w").close()

    slice_downloader.download_slice = fake_download
    try:
        # -- prepare + report ready --------------------------------------------------- #
        a = _agent(tmpdir)
        asg = {"migrating": True, "model_id": "new/model", "total_layers": 20,
               "layer_start": 0, "layer_end": 9, "ready": False}
        agentmod.requests.post = lambda *a_, **k: FakeResp()
        prepared = a._prepare_migration_target(asg)
        check("prepare downloads the target slice (not the model we're serving now)",
              downloads and downloads[0][:3] == ("new/model", 0, 9))
        check("prepare marks is_first_node from layer_start==0", downloads[0][3] is True)
        check("prepare reports ready and marks it locally", prepared is not None and prepared["ready"])
        check("target dir is SEPARATE from the live slice dir",
              prepared["slice_dir"] != os.path.join(agentmod.HERE, "slice_current"))

        # -- report-ready failure is retried, not fatal -------------------------------- #
        a2 = _agent(tmpdir)
        downloads.clear()
        agentmod.requests.post = lambda *a_, **k: FakeResp(status=500)
        prepared2 = a2._prepare_migration_target(asg)
        check("download succeeds even if the ready-report fails",
              prepared2 is not None and prepared2["ready"] is False)

        # -- cutover: coordinator confirms it flipped to OUR target -------------------- #
        a3 = _agent(tmpdir)
        prepared3 = {"model_id": "new/model", "layer_start": 0, "layer_end": 9,
                    "total_layers": 20, "slice_dir": os.path.join(tmpdir, "prep_ok"),
                    "ready": True}
        os.makedirs(prepared3["slice_dir"], exist_ok=True)
        open(os.path.join(prepared3["slice_dir"], "model.safetensors"), "w").close()
        agentmod.requests.get = lambda url, timeout=None: FakeResp(
            {"serving": {"model_id": "new/model", "layers": 20}})
        a3._maybe_cutover(prepared3)
        check("cutover calls server.reload with the prepared target",
              a3.server.calls == [(prepared3["slice_dir"], 0, 9, 20)])
        check("cutover updates cfg to the new model/layers",
              a3.cfg["model_id"] == "new/model" and a3.cfg["layer_start"] == 0
              and a3.cfg["layer_end"] == 9)
        check("cutover moves the prepared slice into the live slice_dir",
              os.path.exists(os.path.join(agentmod.HERE, "slice_current", "model.safetensors")))
        shutil.rmtree(os.path.join(agentmod.HERE, "slice_current"), ignore_errors=True)

        # -- abort: migration ended but coordinator never actually cut over ------------ #
        a4 = _agent(tmpdir)
        prepared4 = {"model_id": "new/model", "layer_start": 0, "layer_end": 9,
                    "total_layers": 20, "slice_dir": os.path.join(tmpdir, "prep_aborted"),
                    "ready": True}
        os.makedirs(prepared4["slice_dir"], exist_ok=True)
        agentmod.requests.get = lambda url, timeout=None: FakeResp(
            {"serving": {"model_id": "old/model", "layers": 28}})   # network reverted
        a4._maybe_cutover(prepared4)
        check("abort never touches server.reload", a4.server.calls == [])
        check("abort discards the prepared (unused) slice", not os.path.exists(prepared4["slice_dir"]))

        # -- a target we never reported ready for is discarded without confirming ------ #
        a5 = _agent(tmpdir)
        prepared5 = {"model_id": "new/model", "layer_start": 0, "layer_end": 9,
                    "total_layers": 20, "slice_dir": os.path.join(tmpdir, "prep_unready"),
                    "ready": False}
        os.makedirs(prepared5["slice_dir"], exist_ok=True)
        called = []
        agentmod.requests.get = lambda url, timeout=None: (called.append(1), FakeResp())[1]
        a5._maybe_cutover(prepared5)
        check("an un-ready prepared target is discarded WITHOUT a confirm round-trip",
              not called and not os.path.exists(prepared5["slice_dir"]))
    finally:
        slice_downloader.download_slice = real_download
        agentmod.requests.get, agentmod.requests.post = real_get, real_post
        shutil.rmtree(tmpdir, ignore_errors=True)
        shutil.rmtree(os.path.join(agentmod.HERE, "slice_current"), ignore_errors=True)

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
