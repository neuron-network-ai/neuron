"""relay_auth.py — HMAC ticket so the relay can verify a node's tunnel registration
without a database or a network call back to the coordinator (post-launch-audit fix).

Problem this closes: relay.py previously accepted ANY {node_id, public_port} registration
on its public control port with no credential at all, and silently overwrote whichever
control connection was already registered on that port — a stranger could hijack or
blackhole another node's traffic just by connecting and claiming its port.

Fix: the coordinator is the only party that assigns node_id -> public_port bindings
(_assign_relay_port in coordinator/main.py), so it mints a ticket = HMAC(shared secret,
node_id + ":" + public_port) and hands it to the node in the registration response. The
node presents the ticket when it opens its control connection; the relay (which knows the
same shared secret, but has no DB) recomputes the HMAC independently and compares. Binding
node_id AND port together means a ticket minted for one node can't be replayed against a
different port, and without the secret an attacker cannot mint a valid ticket for ANY
node_id/port at all — so squatting an unclaimed port is blocked too, not just hijacking an
already-claimed one. Pure stdlib (hmac/hashlib) so relay.py stays dependency-free.
"""
import hashlib
import hmac


def make_ticket(secret, node_id, public_port):
    msg = f"{node_id}:{public_port}".encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


def verify_ticket(secret, node_id, public_port, ticket):
    if not ticket:
        return False
    return hmac.compare_digest(make_ticket(secret, node_id, public_port), ticket)
