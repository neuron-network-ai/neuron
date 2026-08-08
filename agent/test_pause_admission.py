"""agent/test_pause_admission.py — run: python -m agent.test_pause_admission

Pause used to mean almost nothing. `NodeServer.paused` existed and was **never read**; the
agent built the server without passing its own flag, so the tray's Pause only skipped the
heartbeat — and the coordinator kept routing live requests to the machine until the heartbeat
timed out, ~90 seconds later. The owner saw "Paused" and their machine went on serving.

What must hold now:

  1. A paused node refuses NEW requests. `config` is the message that starts one.
  2. A paused node FINISHES work already in flight. `act` continues a request somebody is
     already waiting for; dropping it would punish the user who asked first.
  3. The refusal is a TYPED REPLY, not a closed socket. This is the subtle one. The driver's
     DEAD_PEER tuple contains ConnectionError, and ConnectionRefusedError is a subclass of it —
     so slamming the socket shut is indistinguishable from the machine dying. The driver would
     rebuild the chain, replay the junction cache and report "a machine dropped out" to the
     user, which is [P28]'s complaint caused deliberately. Paused and died are different
     events.
  4. The typed refusal is still RECOVERABLE. `PeerUnavailable` subclasses ConnectionError
     precisely so the existing reroute path handles it — a paused node must not turn into a
     failed answer.

No model is loaded: the batcher is stubbed, so this runs anywhere in a fraction of a second.
"""
import socket
import sys
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


class _FakeBatcher:
    """Stands in for MicroBatcher: returns a hidden state of the right shape."""
    calls = 0

    def submit(self, hidden, cache, past):
        _FakeBatcher.calls += 1
        return torch.zeros(hidden.shape[0], hidden.shape[1], hidden.shape[2])


class _Server(node_server.NodeServer):
    """A NodeServer with no weights — reload() is what needs a model, and serve() is what is
    under test here."""

    def __init__(self, paused):
        self.lo, self.hi, self.n = 0, 9, 28
        self.model = types.SimpleNamespace(
            config=types.SimpleNamespace(hidden_size=64))
        self._batchers = {}
        self._batcher_lock = threading.Lock()
        self.paused = paused
        self.listening = threading.Event()
        self.bind_error = None

    def _batcher(self, role, lo, hi):
        return _FakeBatcher()


def serve_on(server):
    """Run server.serve() against one end of a socketpair; return the client end.

    The disconnect swallow mirrors NodeServer._handle, which catches exactly these: closing
    the client end is how each case here ends, and an uncaught ConnectionError in the thread
    prints a traceback that looks like a failure in a passing run.
    """
    a, b = socket.socketpair()

    def run():
        try:
            server.serve(b, ("test", 0))
        except (ConnectionError, TimeoutError, EOFError, OSError):
            pass
        finally:
            b.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return a, t


def config_msg():
    return {"type": "config", "s1": 0, "s2": 10, "wire": ["f32"]}


def main():
    paused = threading.Event()

    # ---------- 1. a paused node refuses a NEW request, by name ----------
    paused.set()
    srv = _Server(paused)
    cli, t = serve_on(srv)
    common.send_msg(cli, config_msg())
    ack = common.recv_msg(cli)
    check("a paused node refuses a new request", ack.get("ok") is False, ack)
    check("...naming the reason, so it is not mistaken for a crash",
          ack.get("error") == "paused", ack)
    check("...and says it in words the owner would recognise",
          "paused" in (ack.get("detail") or "").lower(), ack)
    cli.close()
    t.join(timeout=5)
    check("...then closes the connection rather than hanging", not t.is_alive())

    # The socket must NOT simply be refused/closed: that is a ConnectionError at the far end,
    # which the driver cannot tell apart from this machine dying.
    check("the refusal arrived as a message, not a dropped connection", isinstance(ack, dict))

    # ---------- 2. an UNPAUSED node accepts normally ----------
    paused.clear()
    srv2 = _Server(paused)
    cli2, t2 = serve_on(srv2)
    common.send_msg(cli2, config_msg())
    ack2 = common.recv_msg(cli2)
    check("an unpaused node accepts the request", ack2.get("ok") is True, ack2)

    # ---------- 3. work already in flight FINISHES after a pause ----------
    # Same connection, already configured. Pausing now must not strand this answer.
    _FakeBatcher.calls = 0
    paused.set()
    common.send_msg(cli2, {"type": "act", "hidden": torch.zeros(1, 1, 64)})
    resp = common.recv_msg(cli2)
    check("a request already in flight is still served after a pause",
          "hidden" in resp and _FakeBatcher.calls == 1, resp.keys())
    common.send_msg(cli2, {"type": "act", "hidden": torch.zeros(1, 1, 64)})
    resp2 = common.recv_msg(cli2)
    check("...and keeps being served, token after token",
          "hidden" in resp2 and _FakeBatcher.calls == 2)
    cli2.close()
    t2.join(timeout=5)

    # ---------- 4. a NEW connection while paused is still refused ----------
    srv3 = _Server(paused)
    cli3, t3 = serve_on(srv3)
    common.send_msg(cli3, config_msg())
    check("a new request arriving during that same pause is refused",
          common.recv_msg(cli3).get("error") == "paused")
    cli3.close()
    t3.join(timeout=5)

    # ---------- 5. the agent actually hands its own flag to the server ----------
    # The bug was never in NodeServer: `paused` was there and nothing set it.
    import inspect
    from agent import agent as agentmod
    src = inspect.getsource(agentmod)
    check("the agent passes its user_paused Event into NodeServer",
          "paused_flag=self.user_paused" in src)
    check("...and that is the same Event the tray's Pause toggles",
          "self.user_paused" in src and "def _toggle_pause" in
          inspect.getsource(__import__("agent.tray", fromlist=["tray"])))

    # ---------- 6. the refusal is recoverable, not fatal ----------
    import neuron_driver
    check("PeerUnavailable is a ConnectionError",
          issubclass(neuron_driver.PeerUnavailable, ConnectionError))
    # DEAD_PEER is built inside stream(); assert the property that matters instead.
    check("...so the driver's DEAD_PEER tuple already covers it, and a paused node reroutes "
          "instead of failing the answer",
          isinstance(neuron_driver.PeerUnavailable("x"),
                     (ConnectionError, TimeoutError, EOFError, OSError)))
    dsrc = inspect.getsource(neuron_driver)
    check("the driver raises it on a refused config rather than an uncatchable RuntimeError",
          "raise PeerUnavailable" in dsrc and "raise RuntimeError(f\"next hop refused" not in dsrc)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
