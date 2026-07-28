"""
NEURON — node_b  (Machine 2 / OptiPlex, LAST stage)  [Session 5]

Runs layers[s2:n] + final norm and returns the normed hidden state. Threaded:
one thread per connection (connections come from node_c, the middle relay), a
shared shard, and a per-machine compute lock. Config carries s2 and n.

Usage (on the OptiPlex):  python node_b.py --port 50999
"""

import argparse
import socket
import threading
import time

import common

state = {"model": None, "n": None, "s2": None}
load_lock = threading.Lock()
compute_lock = threading.Lock()


def ensure_loaded(s2, n):
    with load_lock:
        if state["model"] is None or state["s2"] != s2:
            print(f"[B] loading shard B for layers [{s2}:{n}] + norm ...")
            t0 = time.time()
            _tok, model, nn = common.load_model_shard(s2, n, norm=True)
            state.update(model=model, n=nn, s2=s2)
            print(f"[B] shard ready in {time.time()-t0:.1f}s (my layers {s2}..{n-1})")


def serve(conn, addr):
    cache, past = None, 0
    while True:
        msg = common.recv_msg(conn)
        mtype = msg.get("type")

        if mtype == "config":
            s2, n = msg["s2"], msg["n"]
            ensure_loaded(s2, n)
            cache, past = common.new_cache(), 0
            common.send_msg(conn, {"ok": True, "layers": state["n"], "s2": s2})

        elif mtype == "act":
            hidden = msg["hidden"]
            q = hidden.shape[1]
            with compute_lock:                     # one B-compute at a time, full cores
                tb = time.time()
                out = common.last_stage(state["model"], state["s2"], hidden, cache, past)
                b_ms = (time.time() - tb) * 1000
            past += q
            common.send_msg(conn, {"hidden": out, "b_compute_ms": b_ms})

        elif mtype == "bye":
            return


def handle(conn, addr):
    try:
        serve(conn, addr)
    # TimeoutError is a sibling of ConnectionError under OSError, not caught by it -- kept
    # in sync with node_c.py's handle() for the same reason (see its comment).
    except (ConnectionError, TimeoutError, EOFError) as e:
        print(f"[B] conn {addr} ended: {e}")
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
    print(f"[B] listening on {args.host}:{args.port} | {common.MODEL_ID} | "
          f"last stage, threaded  (Ctrl-C to stop)")

    while True:
        conn, addr = srv.accept()
        print(f"[B] connection from {addr}")
        threading.Thread(target=handle, args=(conn, addr), daemon=True).start()


if __name__ == "__main__":
    main()
