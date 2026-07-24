"""
NEURON relay  [Session 12 — stranger-NAT support]

Lets a node behind home NAT serve in the pipeline WITHOUT accepting inbound
connections. Runs on a public host (the cloud coordinator VM). The node runs
`tunnel_client.py`, which makes only OUTBOUND connections here; this relay exposes a
public port for the node and reverse-tunnels traffic to it.

Pure stdlib (no torch/common) so it drops onto any tiny VM, ARM or x86. It is
PROTOCOL-AGNOSTIC — it splices raw bytes, so the existing node wire protocol (and
node_*.py / common.py) are completely unchanged.

    driver ──▶ relay:PUBLIC_PORT
                 relay ──"new_conn(id)"──▶ node's control conn (held open, outbound)
                 node dials relay:DATA_PORT with id ; relay splices [driver ⇄ data]
                 tunnel_client bridges [data ⇄ 127.0.0.1:NODE_PORT]

All node connections are OUTBOUND → works through NAT. One public port per node; the
coordinator registers that node with the relay's public IP + that port, so node_a /
node_c dial it exactly like a direct node.

Run (on the public VM):  python3 relay.py --control-port 8010 --data-port 8011
Open these + each node's public port in the VM firewall (iptables + cloud security list).
"""
import argparse
import json
import socket
import struct
import threading
import uuid

BUF = 65536


# --- tiny length-prefixed JSON framing (control/handshake only) ------------- #
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
    """Pump bytes both directions until either side closes."""
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


class Relay:
    def __init__(self, control_port, data_port):
        self.control_port = control_port
        self.data_port = data_port
        self.pending = {}          # conn_id -> waiting public client socket
        self.controls = {}         # public_port -> (control_sock, write_lock, node_id)
        self.listening = set()     # public ports that already have a listener
        self.lock = threading.Lock()

    # -- accept helpers ------------------------------------------------------ #
    def _serve(self, port, handler):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", port))
        srv.listen(128)
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=handler, args=(conn, addr), daemon=True).start()

    # -- a node's control connection (persistent, outbound from the node) ---- #
    def _handle_control(self, conn, addr):
        reg = recv_json(conn)
        if not reg or "node_id" not in reg or "public_port" not in reg:
            conn.close()
            return
        node_id, pub = reg["node_id"], int(reg["public_port"])
        wlock = threading.Lock()
        with self.lock:
            self.controls[pub] = (conn, wlock, node_id)
            start_listener = pub not in self.listening
            if start_listener:
                self.listening.add(pub)
        print(f"[relay] '{node_id}' registered from {addr[0]} -> public :{pub}")
        if start_listener:
            threading.Thread(target=self._serve, args=(pub, self._handle_public),
                             daemon=True).start()
            print(f"[relay] public listener up on :{pub}")
        # hold the control conn open; block until the node disconnects
        try:
            while _recvn(conn, 1):
                pass
        except OSError:
            pass
        with self.lock:
            if self.controls.get(pub, (None,))[0] is conn:
                del self.controls[pub]
        conn.close()
        print(f"[relay] '{node_id}' (:{pub}) disconnected")

    # -- a driver/upstream connecting to a node's public port ---------------- #
    def _handle_public(self, client, addr):
        pub = client.getsockname()[1]
        with self.lock:
            ctrl = self.controls.get(pub)
        if not ctrl:
            client.close()
            return
        control_sock, wlock, _node = ctrl
        cid = uuid.uuid4().hex
        with self.lock:
            self.pending[cid] = client
        try:
            with wlock:
                send_json(control_sock, {"new_conn": cid})
        except OSError:
            with self.lock:
                self.pending.pop(cid, None)
            client.close()

    # -- a node's outbound data connection, carrying its conn_id ------------- #
    def _handle_data(self, conn, addr):
        hs = recv_json(conn)
        if not hs or "conn_id" not in hs:
            conn.close()
            return
        with self.lock:
            client = self.pending.pop(hs["conn_id"], None)
        if not client:
            conn.close()
            return
        splice(client, conn)

    def start(self):
        threading.Thread(target=self._serve,
                         args=(self.control_port, self._handle_control), daemon=True).start()
        threading.Thread(target=self._serve,
                         args=(self.data_port, self._handle_data), daemon=True).start()
        print(f"[relay] up | control :{self.control_port} | data :{self.data_port} "
              f"| public ports assigned per node")
        threading.Event().wait()   # run forever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control-port", type=int, default=8010)
    ap.add_argument("--data-port", type=int, default=8011)
    args = ap.parse_args()
    Relay(args.control_port, args.data_port).start()


if __name__ == "__main__":
    main()
