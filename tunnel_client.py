"""
NEURON tunnel client  [Session 12 — stranger-NAT support]

Run this next to a node server (node_b.py / node_c.py / the agent's node_server) to
make that node reachable through the relay WITHOUT accepting any inbound connection —
so it works from behind home NAT. Pure stdlib (no torch/common), ARM-safe.

It connects OUT to the relay's control port and registers the node's public port. When
the relay signals a new connection, it opens an OUTBOUND data connection to the relay
and bridges it to the local node server (127.0.0.1:NODE_PORT). Every connection here is
outbound, so no port-forwarding / no inbound firewall hole is ever needed.

Run (on the node, alongside e.g. `node_b.py --host 127.0.0.1 --port 50999`):
  python3 tunnel_client.py --relay-host <RELAY_PUBLIC_IP> --public-port 9002 \
      --node-id node_b --local-port 50999
"""
import argparse
import json
import socket
import struct
import threading
import time

BUF = 65536
CFG = {}   # filled in main(): relay_host, control_port, data_port, local_host, local_port


def _recvn(sock, n):
    buf = b""
    while len(buf) < n:
        c = sock.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return buf


def send_json(sock, obj):
    b = json.dumps(obj).encode()
    sock.sendall(struct.pack("!I", len(b)) + b)


def recv_json(sock):
    h = _recvn(sock, 4)
    if not h:
        return None
    (n,) = struct.unpack("!I", h)
    b = _recvn(sock, n)
    return json.loads(b.decode()) if b else None


def splice(a, b):
    def pump(src, dst):
        try:
            while True:
                data = src.recv(BUF)
                if not data:
                    break
                dst.sendall(data)
        except OSError:
            pass
        finally:
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass
    t = threading.Thread(target=pump, args=(a, b), daemon=True)
    t.start()
    pump(b, a)
    t.join()
    for s in (a, b):
        try:
            s.close()
        except OSError:
            pass


def handle_new_conn(conn_id):
    """Relay asked for a new connection: dial the relay's data port + the local node,
    hand off the conn_id, and bridge them."""
    try:
        data = socket.create_connection((CFG["relay_host"], CFG["data_port"]), timeout=15)
        send_json(data, {"conn_id": conn_id})
        local = socket.create_connection((CFG["local_host"], CFG["local_port"]), timeout=15)
        # 15s was only the CONNECT budget; the splice must then BLOCK on recv, not time out.
        # A multi-round-trip inference has gaps > 15s under concurrent load, and a leaked
        # timeout there drops the connection mid-stream ("socket closed mid-message").
        data.settimeout(None)
        local.settimeout(None)
    except OSError as e:
        print(f"[tunnel] new_conn {conn_id[:8]} setup failed: {e}")
        return
    splice(data, local)


def run_tunnel(node_id, public_port, relay_host, control_port=8010, data_port=8011,
               local_host="127.0.0.1", local_port=50999, stop=None):
    """Maintain the outbound control connection + bridge relayed connections to the
    local node server. Reconnects forever (or until `stop` [a threading.Event] is set).
    Callable directly by the agent, or via main()/CLI."""
    CFG.update(relay_host=relay_host, control_port=control_port, data_port=data_port,
               local_host=local_host, local_port=local_port)
    print(f"[tunnel] node '{node_id}' -> relay {relay_host}:{control_port} "
          f"(public :{public_port} -> local {local_host}:{local_port})")
    while stop is None or not stop.is_set():
        try:
            ctrl = socket.create_connection((relay_host, control_port), timeout=15)
            # The 15s was only the CONNECT budget. The control connection then stays open,
            # idle, waiting for the relay to push new_conn — so clear the timeout, otherwise
            # recv() raises timeout every 15s and the tunnel churns (reconnect loop). Enable
            # TCP keepalive so a genuinely idle NAT mapping isn't silently dropped either.
            ctrl.settimeout(None)
            try:
                ctrl.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            except OSError:
                pass
            send_json(ctrl, {"node_id": node_id, "public_port": public_port})
            print("[tunnel] registered; waiting for connections")
            while stop is None or not stop.is_set():
                msg = recv_json(ctrl)
                if msg is None:
                    break
                if "new_conn" in msg:
                    threading.Thread(target=handle_new_conn, args=(msg["new_conn"],),
                                     daemon=True).start()
        except OSError as e:
            print(f"[tunnel] control error: {e}")
        if stop is not None and stop.is_set():
            break
        print("[tunnel] reconnecting in 3s ...")
        time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--relay-host", required=True, help="relay public IP/host")
    ap.add_argument("--control-port", type=int, default=8010)
    ap.add_argument("--data-port", type=int, default=8011)
    ap.add_argument("--public-port", type=int, required=True,
                    help="the relay public port this node should be reachable on")
    ap.add_argument("--node-id", required=True)
    ap.add_argument("--local-host", default="127.0.0.1")
    ap.add_argument("--local-port", type=int, default=50999,
                    help="the local node server port to bridge to")
    args = ap.parse_args()
    run_tunnel(args.node_id, args.public_port, args.relay_host,
               control_port=args.control_port, data_port=args.data_port,
               local_host=args.local_host, local_port=args.local_port)


if __name__ == "__main__":
    main()
