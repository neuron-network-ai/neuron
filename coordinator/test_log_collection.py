"""coordinator/test_log_collection.py — run: python -m coordinator.test_log_collection

The endpoint half of log collection: a node is asked via its heartbeat, uploads with its own
token, and only an operator can read the result. Uses a throwaway DB and the real endpoint
functions -- no HTTP server.

The auth tests are the point. This feature moves a file off a volunteer's personal computer, so
"who may ask for it" and "who may read it" are the properties that decide whether it is a
diagnostic tool or a way to read strangers' machines.
"""
import os
import tempfile

os.environ["NEURON_OPEN_JOIN"] = "1"
os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron_logep_"), "l.db")

from fastapi import HTTPException  # noqa: E402

from coordinator import config, models, nodelogs  # noqa: E402
from coordinator.main import (LogRequestBody, NodeLogBody, admin_logs_get,  # noqa: E402
                              admin_logs_request, admin_logs_summary, node_logs_upload, ping,
                              register, require_node_token, require_register_secret,
                              RegisterBody)

SECRET = config.REGISTRATION_SECRET
ok = fail = 0


def check(label, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}")


def raises(fn, *a, **kw):
    try:
        fn(*a, **kw)
        return None
    except HTTPException as e:
        return e.status_code


def main():
    models.init_db()
    nodelogs.init()

    tok = {}
    for nid in ("node-1", "node-2"):
        r = register(RegisterBody(node_id=nid, tailscale_ip="127.0.0.1", port=50000 + len(tok),
                                  layer_start=0, layer_end=13, cores=4, ram_gb=8),
                     x_register_secret=SECRET)
        tok[nid] = r["node_token"]

    n1 = models.get_node("node-1")

    # Auth dependencies are passed POSITIONALLY throughout: an endpoint names that parameter
    # with a leading underscore when it ignores it and without one when it reads it, and a test
    # should not break because an endpoint started reading its own auth result.
    # ---- the heartbeat is the channel: nothing asked -> nothing carried ------- #
    check("a heartbeat does not ask for logs by default",
          ping("node-1", n1)["want_logs"] is False)

    # ---- only an operator may ask -------------------------------------------- #
    check("asking without the operator secret is refused",
          raises(require_register_secret, x_register_secret=None) == 401)
    check("...and with a wrong one",
          raises(require_register_secret, x_register_secret="nope") == 401)

    admin_logs_request(LogRequestBody(nodes=["node-1"]), True)
    check("once asked, the heartbeat carries the request",
          ping("node-1", n1)["want_logs"] is True)
    check("...and only to the node that was asked",
          ping("node-2", models.get_node("node-2"))["want_logs"] is False)

    # ---- a node may upload only its OWN log ---------------------------------- #
    check("uploading with no token is refused",
          raises(require_node_token, "node-1", x_node_token=None) == 401)
    check("uploading with ANOTHER node's token is refused",
          raises(require_node_token, "node-1", x_node_token=tok["node-2"]) == 401)
    check("its own token is accepted",
          require_node_token("node-1", x_node_token=tok["node-1"])["node_id"] == "node-1")

    # ---- upload, and the request stops ---------------------------------------- #
    leak = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0"
    body = (f"starting up\nX-Node-Token: {leak}\n"
            r"loading C:\Users\someone\NEURON\config.json" "\nheartbeat ok\n")
    node_logs_upload("node-1", NodeLogBody(body=body), n1)
    check("after uploading, the heartbeat stops asking",
          ping("node-1", n1)["want_logs"] is False)

    stored = admin_logs_get("node-1", True)
    check("the log is readable by an operator", "starting up" in stored["body"])
    check("a token the node sent is NOT in what an operator reads", leak not in stored["body"])
    check("nor is the machine owner's name", "someone" not in stored["body"])
    check("but the useful content survived", "heartbeat ok" in stored["body"])

    # ---- reading is operator-gated ------------------------------------------- #
    rows = admin_logs_summary(True)
    check("summary shows who has answered", [r["node_id"] for r in rows] == ["node-1"])
    check("summary carries no bodies", all("body" not in r for r in rows))

    # ---- a node that never answers reads as a finding, not an error ----------- #
    silent = admin_logs_get("node-2", True)
    check("a silent node returns an empty body rather than failing",
          silent["body"] == "" and silent["uploaded_at"] == 0)

    # ---- asking everyone ------------------------------------------------------ #
    out = admin_logs_request(LogRequestBody(nodes=[]), True)
    check("an empty list asks every known node",
          set(out["requested"]) == {"node-1", "node-2"})

    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
