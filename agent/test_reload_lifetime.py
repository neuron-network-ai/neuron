"""agent/test_reload_lifetime.py — run: python -m agent.test_reload_lifetime

Model LIFETIME during a slice migration — deliberately a separate concern from request
admission (`test_pause_admission.py`), and a separate commit.

`reload()` used to load the new slice while `self.model` still held the old one, so a migrating
node peaked at roughly **150% of one slice**: two full sets of weights alive at the same moment,
on machines chosen precisely because they had room for one. Peak is the number that gets a
volunteer OOM-killed; steady state is not.

The properties:

  1. The old slice is released BEFORE the new one is loaded. Asserted by observing the order of
     events, not by measuring memory — memory is not deterministic enough to test.
  2. The window where there are no weights is guarded. Requests arriving in it get a named
     refusal, not an `AttributeError` from inside a batcher, which would reach the driver as a
     closed socket with no explanation.
  3. Batchers, which close over the OLD model, go with it.
  4. A load that FAILS clears the reloading flag. This ordering has a real cost — a failed load
     now leaves the node with no model, where before it kept serving the old slice — and a node
     stuck reporting "reloading" forever would be a node that never recovers and never says why.
"""
import os
import socket
import sys
import tempfile
import threading
import types

import torch

import common
from agent import node_server

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


class _Batcher:
    def __init__(self, tag):
        self.tag, self.stopped = tag, False

    def stop(self):
        self.stopped = True
        events.append(f"batcher-stopped:{self.tag}")

    def submit(self, hidden, cache, past):
        return torch.zeros_like(hidden)


events = []


class _Server(node_server.NodeServer):
    """A NodeServer whose reload does everything except touch real weights."""

    def __init__(self):
        self.lo, self.hi, self.n = 0, 9, 28
        self.model = "OLD-SLICE"
        self._batchers = {"k": _Batcher("old")}
        self._batcher_lock = threading.Lock()
        self.paused = threading.Event()
        self.listening = threading.Event()
        self.bind_error = None
        self._reloading = threading.Event()

    def _batcher(self, role, lo, hi):
        return _Batcher("live")


def main():
    tmp = tempfile.mkdtemp(prefix="neuron-reload-")

    # ---------- 1 & 3: release before load ----------
    events.clear()
    srv = _Server()
    old_batcher = srv._batchers["k"]

    def fake_load(d):
        # Whatever the old model was, it must already be unreferenced by the server when the
        # new one starts allocating. That is the entire point of the change.
        events.append(f"load-begins(model={srv.model!r})")
        return "NEW-SLICE"

    real_load, node_server.load_slice_model = node_server.load_slice_model, fake_load
    # Disable the slice-range guard: these tests are about reload LIFETIME, not the guard.
    # `_layer_set` is the one reload() actually calls ([P42]); `_layers_in_slice` is kept in
    # step so the pair cannot drift into a patch that silently no longer patches anything.
    real_layers, node_server._layers_in_slice = node_server._layers_in_slice, lambda d: None
    real_set, node_server._layer_set = node_server._layer_set, lambda d: None
    real_empty = node_server._empty_device_cache
    node_server._empty_device_cache = lambda: events.append("empty_cache")
    try:
        srv.reload(tmp, 0, 9, 28)
    finally:
        node_server.load_slice_model = real_load
        node_server._layers_in_slice = real_layers
        node_server._layer_set = real_set
        node_server._empty_device_cache = real_empty

    check("the old slice is released before the new one is loaded",
          "load-begins(model=None)" in events, events)
    check("...and the old batchers are stopped with it", old_batcher.stopped)
    check("...before the load, not after",
          events.index("batcher-stopped:old") < events.index("load-begins(model=None)"), events)
    check("the device cache is emptied in the gap (a no-op on CPU, load-bearing on a GPU)",
          "empty_cache" in events
          and events.index("empty_cache") < events.index("load-begins(model=None)"), events)
    check("the new slice is installed", srv.model == "NEW-SLICE")
    check("...and the reloading flag is cleared afterwards", not srv._reloading.is_set())
    check("the served range is updated", (srv.lo, srv.hi, srv.n) == (0, 9, 28))

    # ---------- 2: the window is guarded ----------
    srv2 = _Server()
    srv2._reloading.set()                        # pretend we are mid-swap
    srv2.model = None

    a, b = socket.socketpair()

    def run():
        try:
            srv2.serve(b, ("test", 0))
        except (ConnectionError, TimeoutError, EOFError, OSError):
            pass
        finally:
            b.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    common.send_msg(a, {"type": "config", "s1": 0, "s2": 10, "wire": ["f32"]})
    ack = common.recv_msg(a)
    check("a new request during a reload is refused, not crashed on",
          ack.get("ok") is False and ack.get("error") == "reloading", ack)
    check("...with a reason a human can act on", "slice" in (ack.get("detail") or ""), ack)
    a.close()
    t.join(timeout=5)

    # An `act` arriving on an already-configured connection whose model vanished underneath it.
    srv3 = _Server()
    a3, b3 = socket.socketpair()

    def run3():
        try:
            srv3.serve(b3, ("test", 0))
        except (ConnectionError, TimeoutError, EOFError, OSError):
            pass
        finally:
            b3.close()

    t3 = threading.Thread(target=run3, daemon=True)
    t3.start()
    common.send_msg(a3, {"type": "config", "s1": 0, "s2": 10, "wire": ["f32"]})
    check("the connection configures normally first", common.recv_msg(a3).get("ok") is True)
    srv3.model = None                            # the reload lands mid-request
    common.send_msg(a3, {"type": "act", "hidden": torch.zeros(1, 1, 8)})
    resp = common.recv_msg(a3)
    check("an in-flight act whose slice was released gets a named refusal, not an "
          "AttributeError", resp.get("error") == "reloading", resp)
    a3.close()
    t3.join(timeout=5)

    # ---------- 4: a failed load does not leave the node stuck ----------
    srv4 = _Server()

    def boom(d):
        raise RuntimeError("disk went away")

    node_server.load_slice_model = boom
    node_server._layers_in_slice = lambda d: None
    node_server._layer_set = lambda d: None
    try:
        srv4.reload(tmp, 0, 9, 28)
        check("a failed load propagates rather than being swallowed", False)
    except RuntimeError:
        check("a failed load propagates rather than being swallowed", True)
    finally:
        node_server.load_slice_model = real_load
        node_server._layers_in_slice = real_layers
        node_server._layer_set = real_set

    check("...and the reloading flag is cleared, so the node can be retried",
          not srv4._reloading.is_set())
    check("...and the node reports having no slice rather than a stale one",
          srv4.model is None)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
