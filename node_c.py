"""
NEURON — node_c  (Machine 3 / HP Pavilion, MIDDLE relay)  [Session 5]

Sits between node_a and node_b in the pipeline:
    node_a --h1--> node_c --h2--> node_b --h3--> node_c --h3--> node_a (lm_head)

node_c is BOTH a server (to node_a) and a client (to node_b). Threaded: one
thread per node_a connection, each opening its own connection to node_b and
holding its own KV cache. A per-machine compute lock serialises node_c's own
layer math while the node_b round-trip overlaps other threads.

Config from node_a carries s1, s2 and node_b's address. node_c loads layers
[s1:s2], then configures node_b (layers [s2:n] + norm).

Usage (on the Pavilion):  python node_c.py --port 50999
"""

import argparse
import socket
import threading
import time

import common

state = {"model": None, "n": None, "s1": None, "s2": None}
load_lock = threading.Lock()
compute_lock = threading.Lock()


def ensure_loaded(s1, s2):
    with load_lock:
        if state["model"] is None or state["s1"] != s1 or state["s2"] != s2:
            print(f"[C] loading shard C for layers [{s1}:{s2}] ...")
            t0 = time.time()
            _tok, model, n = common.load_model_shard(s1, s2)
            state.update(model=model, n=n, s1=s1, s2=s2)
            print(f"[C] shard ready in {time.time()-t0:.1f}s (my layers {s1}..{s2-1})")


def serve(conn, addr):
    cache, past = None, 0
    bconn = None
    try:
        while True:
            msg = common.recv_msg(conn)
            mtype = msg.get("type")

            if mtype == "config":
                s1, s2 = msg["s1"], msg["s2"]
                ensure_loaded(s1, s2)
                cache, past = common.new_cache(), 0
                probing = "host_b" not in msg
                if not probing:
                    # open + configure our own connection to node_b (the last stage)
                    bconn = socket.create_connection((msg["host_b"], msg["port_b"]),
                                                     timeout=common.COLD_CONNECT_TIMEOUT_S)
                    common.send_msg(bconn, {"type": "config", "s2": s2, "n": state["n"]})
                    back = common.recv_msg(bconn)
                    assert back.get("ok"), f"node_b refused: {back}"
                    bconn.settimeout(common.HOT_TIMEOUT_S)
                common.send_msg(conn, {"ok": True, "layers": state["n"], "s1": s1, "s2": s2})

            elif mtype == "act":
                hidden = msg["hidden"]
                q = hidden.shape[1]
                with compute_lock:                 # one C-compute at a time, full cores
                    tc = time.time()
                    h2 = common.mid_stage(state["model"], state["s1"], state["s2"],
                                          hidden, cache, past)
                    c_ms = (time.time() - tc) * 1000
                past += q
                if bconn is None:
                    # PROBE mode (security/proof_of_compute.py): a config with no host_b
                    # means the caller wants to verify THIS node's own layers in isolation --
                    # return the raw mid-stage output directly, no relay to a next hop.
                    common.send_msg(conn, {"hidden": h2, "c_compute_ms": c_ms})
                else:
                    common.send_msg(bconn, {"type": "act", "hidden": h2})    # -> node_b
                    resp = common.recv_msg(bconn)                            # <- node_b
                    common.send_msg(conn, {"hidden": resp["hidden"],
                                           "c_compute_ms": c_ms,
                                           "b_compute_ms": resp["b_compute_ms"]})

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


def handle(conn, addr):
    try:
        serve(conn, addr)
    # TimeoutError is a sibling of ConnectionError under OSError, not a subclass -- it
    # wasn't caught here before, so a slow-to-load node_b (cold shard load can legitimately
    # take >30s) would raise it uncaught inside serve(), print a bare traceback, and leave
    # this thread silently dead. The `finally` below still ran, actively closing our
    # connection back to node_a mid-handshake -- which is what made node_a see a plain
    # "socket closed mid-message" with no hint it was actually a cold-start timeout one hop
    # further down the chain.
    except (ConnectionError, TimeoutError, EOFError) as e:
        print(f"[C] conn {addr} ended: {e}")
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=50999)
    args = ap.parse_args()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(16)
    print(f"[C] listening on {args.host}:{args.port} | {common.MODEL_ID} | "
          f"middle relay, threaded  (Ctrl-C to stop)")

    while True:
        conn, addr = srv.accept()
        print(f"[C] connection from {addr}")
        threading.Thread(target=handle, args=(conn, addr), daemon=True).start()


if __name__ == "__main__":
    main()
