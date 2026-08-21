"""A re-placement reaches a RUNNING node — run: python -m agent.test_placement_reaches_the_node

[P37] filed the gap and [P60] is what it cost: `models.update_layers` moves a node's range on
the coordinator, nothing pushed it to the node, and the node went on serving the range it was
handed at start-up. On 2026-08-21 the Pavilion held 10-18 while 10-27 was routed to it, and it
answered — over layers 19-27 that were never downloaded. `/status` said healthy, decode hit the
fastest tok/s this network has produced, and the output was word salad.

`agent/test_stale_placement_is_refused_not_answered.py` covers the OTHER half: a node asked for
a role it cannot serve refuses with `range_mismatch` instead of computing over meta tensors,
which turned that failure from wrong answers into reroutes. This covers the window itself —
the node learning its new range from the heartbeat instead of from its next registration.

The load-bearing assertion is not that a reload happens. It is WHAT IS COMPARED: the range the
coordinator assigns against the range the SERVER IS LOADED WITH, never against config.json.
config is what the node believes it was told; `server.lo/hi` is what the machine actually
computes, and every version of this bug is those two disagreeing while all the reporting is
drawn from the first.

Monkeypatches agent.requests and slice_downloader — no network, no coordinator, no weights.
"""
import json
import os
import shutil
import tempfile

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
    """Stands in for NodeServer. lo/hi/n are the range actually LOADED, which is the fact
    resync_placement is required to compare against."""

    def __init__(self, lo, hi, n=28):
        self.lo, self.hi, self.n = lo, hi, n
        self.calls = []
        self.raise_on_reload = False

    def reload(self, slice_dir, lo, hi, total):
        self.calls.append((slice_dir, lo, hi, total))
        if self.raise_on_reload:
            raise RuntimeError("refusing to serve layers: this slice is missing 19-27")
        self.lo, self.hi, self.n = lo, hi, total


def write_slice(path, layers, first=False, last=False):
    """A safetensors header only — enough for slice_layers_on_disk/_slice_covers, no weights."""
    keys = {f"model.layers.{i}.self_attn.q_proj.weight": {} for i in layers}
    if first:
        keys["model.embed_tokens.weight"] = {}
    if last:
        keys["model.norm.weight"] = {}
    blob = json.dumps(keys).encode()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(len(blob).to_bytes(8, "little"))
        f.write(blob)
    if first:
        open(os.path.join(os.path.dirname(path), "tokenizer.json"), "w").close()


def _agent(tmpdir, name="cfg.json", **over):
    cfg_path = os.path.join(tmpdir, name)
    cfg = dict(agentmod.DEFAULT_CONFIG)
    cfg.update(node_id="test-node", node_token="tok-1", model_id="the/model",
               layer_start=10, layer_end=18, slice_dir="./slice_current/")
    cfg.update(over)
    json.dump(cfg, open(cfg_path, "w"))
    return agentmod.Agent(config_path=cfg_path)


LIVE = os.path.join(agentmod.HERE, "slice_current")
STAGED = os.path.join(agentmod.HERE, "model_slice_migrating")


def info_for(lo, hi, model="the/model", total=28, gb=0.4):
    return {"model_id": model, "layer_start": lo, "layer_end": hi, "total_layers": total,
            "is_first_node": lo == 0, "is_last_node": hi == total - 1,
            "estimated_download_gb": gb}


def main():
    tmpdir = tempfile.mkdtemp(prefix="neuron_placement_")
    real_get, real_download, real_prov = (agentmod.requests.get,
                                          slice_downloader.download_slice,
                                          slice_downloader.slice_provenance)
    downloads = []

    def fake_download(model_id, lo, hi, target_dir, is_first_node, is_last_node, revision="main"):
        downloads.append((model_id, lo, hi, target_dir, is_first_node, is_last_node))
        write_slice(os.path.join(target_dir, "model.safetensors"), range(lo, hi + 1),
                    first=is_first_node, last=is_last_node)

    slice_downloader.download_slice = fake_download
    slice_downloader.slice_provenance = lambda d: "the/model"
    try:
        # -- 1. the heartbeat is where the assignment arrives ------------------------- #
        a = _agent(tmpdir)
        a.note_assignment({"status": "alive", "layer_start": 10, "layer_end": 27})
        check("a ping carrying a range records it", a.assigned == (10, 27))

        a2 = _agent(tmpdir, "cfg2.json")
        a2.note_assignment({"status": "alive", "standing": "trusted"})
        check("a coordinator too old to send a range leaves it UNKNOWN, never guessed",
              a2.assigned is None)
        a2.note_assignment({"layer_start": "x", "layer_end": None})
        check("an unreadable range is ignored rather than half-applied", a2.assigned is None)

        # -- 2. already right: the common case, and it must cost nothing --------------- #
        a3 = _agent(tmpdir, "cfg3.json")
        a3.server = FakeServer(10, 18)
        a3.assigned = (10, 18)
        asked = []
        agentmod.requests.get = lambda *ar, **kw: (asked.append(ar), FakeResp())[1]
        a3.resync_placement()
        check("a node already on its assigned range asks the coordinator nothing", not asked)
        check("...and does not reload", a3.server.calls == [])

        # -- 3. the comparison is against the SERVER, not config.json ------------------ #
        # config says 10-27 (a range this node was told about and never managed to load);
        # the server is loaded with 10-18. That is exactly [P60], and a config-based check
        # would call it settled.
        a4 = _agent(tmpdir, "cfg4.json", layer_start=10, layer_end=27)
        a4.server = FakeServer(10, 18)
        a4.assigned = (10, 27)
        write_slice(os.path.join(LIVE, "model.safetensors"), range(10, 28), last=True)
        agentmod.requests.get = lambda *ar, **kw: FakeResp(info_for(10, 27))
        downloads.clear()
        a4.resync_placement()
        check("config agreeing with the coordinator does NOT stop the resync — the server's "
              "loaded range is what counts", a4.server.calls and a4.server.calls[0][1:] == (10, 27, 28))

        # -- 4. a slice already covering the new range is reloaded, never re-downloaded - #
        check("a covering slice on disk is reused, with no download", not downloads)
        check("...reloaded from the LIVE dir, so nothing is deleted under the open model",
              bool(a4.server.calls) and a4.server.calls[0][0] == LIVE)
        check("cfg is persisted, so a restart does not re-assert the old range",
              json.load(open(a4.config_path))["layer_end"] == 27)
        check("the tray's state follows", a4.state["layers"] == [10, 27])
        shutil.rmtree(LIVE, ignore_errors=True)

        # -- 5. slice-info is the authority; a stale hint moves nothing ---------------- #
        a5 = _agent(tmpdir, "cfg5.json")
        a5.server = FakeServer(10, 18)
        a5.assigned = (10, 27)                       # hint says moved...
        agentmod.requests.get = lambda *ar, **kw: FakeResp(info_for(10, 18))   # ...it did not
        a5.resync_placement()
        check("a hint slice-info does not confirm is dropped, not acted on",
              a5.server.calls == [])
        check("...and the hint is corrected so it is not re-read every poll",
              a5.assigned == (10, 18))

        # -- 6. a MODEL change belongs to the migration path, which stages it ---------- #
        a6 = _agent(tmpdir, "cfg6.json")
        a6.server = FakeServer(10, 18)
        a6.assigned = (10, 27)
        agentmod.requests.get = lambda *ar, **kw: FakeResp(info_for(10, 27, model="other/model"))
        downloads.clear()
        a6.resync_placement()
        check("a different model is left to the migration path, not swapped in here",
              a6.server.calls == [] and not downloads)

        # -- 7. a range NOT on disk downloads to a staging dir and swaps --------------- #
        a7 = _agent(tmpdir, "cfg7.json")
        a7.server = FakeServer(10, 18)
        a7.assigned = (19, 27)
        write_slice(os.path.join(LIVE, "model.safetensors"), range(10, 19))
        agentmod.requests.get = lambda *ar, **kw: FakeResp(info_for(19, 27))
        downloads.clear()
        a7.resync_placement()
        check("a range the disk does not hold is downloaded", len(downloads) == 1)
        check("...into a dir SEPARATE from the one being served, so the node keeps answering",
              bool(downloads) and downloads[0][3] == STAGED and downloads[0][3] != LIVE)
        check("...told it is the last node, which decides whether the final norm comes too",
              bool(downloads) and downloads[0][5] is True)
        check("the reload comes from the staged dir", a7.server.calls
              and a7.server.calls[0][:3] == (STAGED, 19, 27))
        check("and the staged slice becomes the live one",
              os.path.exists(os.path.join(LIVE, "model.safetensors"))
              and not os.path.exists(STAGED))
        check("cfg follows the swap", json.load(open(a7.config_path))["layer_start"] == 19)
        shutil.rmtree(LIVE, ignore_errors=True)

        # -- 8. a failed download leaves the node serving what it already had ---------- #
        a8 = _agent(tmpdir, "cfg8.json")
        a8.server = FakeServer(10, 18)
        a8.assigned = (19, 27)
        agentmod.requests.get = lambda *ar, **kw: FakeResp(info_for(19, 27))

        def boom(*ar, **kw):
            raise OSError("connection reset")

        slice_downloader.download_slice = boom
        a8.resync_placement()
        check("a failed download never reloads, so the old range keeps serving",
              a8.server.calls == [] and (a8.server.lo, a8.server.hi) == (10, 18))
        check("...and leaves no half-downloaded staging dir behind", not os.path.exists(STAGED))
        check("...and does not persist a range this node cannot serve",
              json.load(open(a8.config_path))["layer_end"] == 18)
        slice_downloader.download_slice = fake_download

        # -- 9. nothing here may kill the thread that also runs migrations ------------- #
        a9 = _agent(tmpdir, "cfg9.json")
        a9.server = FakeServer(10, 18)
        a9.server.raise_on_reload = True             # [P42]: reload refuses a gappy slice
        a9.assigned = (10, 27)
        write_slice(os.path.join(LIVE, "model.safetensors"), range(10, 28), last=True)
        agentmod.requests.get = lambda *ar, **kw: FakeResp(info_for(10, 27))
        raised = None
        try:
            a9.resync_placement()
        except Exception as e:                                              # noqa: BLE001
            raised = e
        check("a refused reload is swallowed and retried, not raised into the loop",
              raised is None)
        check("...and the config is not moved onto a range that failed to load",
              json.load(open(a9.config_path))["layer_end"] == 18)
        shutil.rmtree(LIVE, ignore_errors=True)

        a10 = _agent(tmpdir, "cfg10.json")
        a10.server = FakeServer(10, 18)
        a10.assigned = (10, 27)

        def unreachable(*ar, **kw):
            raise agentmod.requests.RequestException("coordinator unreachable")

        agentmod.requests.get = unreachable
        raised = None
        try:
            a10.resync_placement()
        except Exception as e:                                              # noqa: BLE001
            raised = e
        check("an unreachable coordinator is a retry, not an exception", raised is None)

        # -- 10. an operator's --layers does not veto the coordinator ------------------ #
        a11 = _agent(tmpdir, "cfg11.json", layers_pinned=True)
        a11.server = FakeServer(10, 18)
        a11.assigned = (10, 27)
        write_slice(os.path.join(LIVE, "model.safetensors"), range(10, 28), last=True)
        agentmod.requests.get = lambda *ar, **kw: FakeResp(info_for(10, 27))
        a11.resync_placement()
        check("a pinned node still serves the coordinator's placement (it is the "
              "coordinator's to decide) and is told so", a11.server.calls
              and a11.server.calls[0][1:3] == (10, 27))
        shutil.rmtree(LIVE, ignore_errors=True)

        # -- 11. one thread owns slice work, and a migration in flight wins ------------ #
        real_poll = agentmod.MIGRATION_POLL_SECONDS
        agentmod.MIGRATION_POLL_SECONDS = 0
        for migrating, expect in ((True, 0), (False, 1)):
            a12 = _agent(tmpdir, f"cfg12{migrating}.json")
            a12.server = FakeServer(10, 18)
            resyncs = []
            a12.resync_placement = lambda: resyncs.append(1)
            a12._prepare_migration_target = lambda asg: {
                "model_id": "the/model", "layer_start": 0, "layer_end": 9,
                "total_layers": 28, "slice_dir": STAGED, "ready": True}

            def one_pass(*ar, **kw):
                a12._stop.set()                       # exactly one iteration
                return FakeResp({"migrating": migrating, "model_id": "the/model",
                                 "total_layers": 28, "layer_start": 0, "layer_end": 9})

            agentmod.requests.get = one_pass
            a12.migration_loop()
            check(f"a migration {'in flight defers' if migrating else 'not running lets'} "
                  f"the re-placement{'' if migrating else ' run'}", len(resyncs) == expect)
        agentmod.MIGRATION_POLL_SECONDS = real_poll
    finally:
        agentmod.requests.get = real_get
        slice_downloader.download_slice = real_download
        slice_downloader.slice_provenance = real_prov
        shutil.rmtree(tmpdir, ignore_errors=True)
        shutil.rmtree(LIVE, ignore_errors=True)
        shutil.rmtree(STAGED, ignore_errors=True)

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
