"""security/test_wire_crypto.py — the pipeline hop is private and authenticated ([P52]).

    python -m security.test_wire_crypto

Every attack this claims to stop is run against a real socket pair and required to FAIL. A
security module whose negative cases were never executed is [P40]'s thesis applied to crypto:
a check nobody has seen fire is not a check.

What is being defended, in the order it matters:

  * a stranger who dialled a node's PUBLIC relay port cannot open a session at all;
  * a chain member cannot use its grant against a node it was not sent to;
  * an eavesdropper on the relay cannot read the prompt-derived activations;
  * a recording cannot be opened later even by the COORDINATOR that introduced the two
    parties, which is what X25519 buys over simply shipping a key inside the grant;
  * a tampered byte is rejected rather than decrypted into something plausible.
"""
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from security import wire_crypto as W                        # noqa: E402

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


TOKEN_B = "b" * 48
NODE_B = "node-c-pavilion"
REQ = "req-abc123"


def _pair():
    """A real connected socket pair — not a mock. The handshake is a protocol, and a protocol
    tested against a stub is a protocol tested against my own assumptions."""
    lis = socket.socket()
    lis.bind(("127.0.0.1", 0))
    lis.listen(1)
    c = socket.create_connection(lis.getsockname())
    s, _ = lis.accept()
    lis.close()
    return c, s


def _run(client_fn, server_fn):
    """Both halves concurrently; returns (client_result, server_result), exceptions included."""
    out = {}

    def wrap(name, fn, sock):
        try:
            out[name] = fn(sock)
        except Exception as e:                               # noqa: BLE001
            out[name] = e
        finally:
            sock.close()

    c, s = _pair()
    ts = threading.Thread(target=wrap, args=("server", server_fn, s))
    tc = threading.Thread(target=wrap, args=("client", client_fn, c))
    ts.start(), tc.start()
    ts.join(20), tc.join(20)
    return out.get("client"), out.get("server")


def main():
    print("\n-- the ordinary case: the driver and the node it was sent to")
    grant = W.mint_grant(TOKEN_B, REQ, NODE_B)
    cli, srv = _run(lambda so: W.client_handshake(so, grant, NODE_B),
                    lambda so: W.server_handshake(so, TOKEN_B, NODE_B))
    check("the handshake completes", isinstance(cli, W.Channel), repr(cli))
    check("...and the node learns which request it is for",
          isinstance(srv, tuple) and srv[1] == REQ, repr(srv))

    print("\n-- a stranger who found the public port")
    # The whole reason node ports being public was survivable is that the wire carried nothing
    # executable. It still let anyone SPEAK. Now they cannot start.
    def stranger(so):
        forged = W.mint_grant("not-the-node-token", REQ, NODE_B)
        return W.client_handshake(so, forged, NODE_B)

    cli, srv = _run(stranger, lambda so: W.server_handshake(so, TOKEN_B, NODE_B))
    check("a forged grant is refused by the node",
          isinstance(srv, W.HandshakeError), repr(srv))
    check("...and the stranger gets no channel", not isinstance(cli, W.Channel), repr(cli))

    print("\n-- a grant is for ONE node, not a skeleton key for the chain")
    # A chain member legitimately holds a grant. It must not be reusable sideways against the
    # next machine, or one compromised volunteer unlocks every peer it is routed with.
    other_grant = W.mint_grant(TOKEN_B, REQ, "some-other-node")
    cli, srv = _run(lambda so: W.client_handshake(so, other_grant, NODE_B),
                    lambda so: W.server_handshake(so, TOKEN_B, NODE_B))
    check("a grant minted for another node is refused",
          isinstance(srv, W.HandshakeError), repr(srv))

    print("\n-- an expired grant")
    old = W.mint_grant(TOKEN_B, REQ, NODE_B, now=time.time() - W.GRANT_TTL_S - 60)
    try:
        W.open_grant(TOKEN_B, NODE_B, old)
        check("an expired grant is refused", False, "it was accepted")
    except W.HandshakeError:
        check("an expired grant is refused", True)

    print("\n-- a replayed grant, inside its TTL")
    seen = set()
    g = W.mint_grant(TOKEN_B, REQ, NODE_B)
    _run(lambda so: W.client_handshake(so, g, NODE_B),
         lambda so: W.server_handshake(so, TOKEN_B, NODE_B, seen=seen))
    _, srv2 = _run(lambda so: W.client_handshake(so, g, NODE_B),
                   lambda so: W.server_handshake(so, TOKEN_B, NODE_B, seen=seen))
    check("the same grant cannot be used twice", isinstance(srv2, W.HandshakeError), repr(srv2))

    print("\n-- the failure messages do not become an oracle")
    # "bad key" vs "expired" vs "wrong node" tells a prober which of its guesses was close.
    msgs = set()
    for bad in (W.mint_grant("x" * 48, REQ, NODE_B),
                W.mint_grant(TOKEN_B, REQ, "elsewhere"),
                W.mint_grant(TOKEN_B, REQ, NODE_B, now=time.time() - 9999)):
        try:
            W.open_grant(TOKEN_B, NODE_B, bad)
        except W.HandshakeError as e:
            msgs.add(str(e))
    check("every rejection reads the same", len(msgs) == 1, str(msgs))

    print("\n-- the traffic itself")
    grant = W.mint_grant(TOKEN_B, REQ, NODE_B)
    cli, srv = _run(lambda so: W.client_handshake(so, grant, NODE_B),
                    lambda so: W.server_handshake(so, TOKEN_B, NODE_B))
    ch_c, ch_s = cli, srv[0]
    secret = b"the user's prompt, and the activations made from it" * 40
    sealed = ch_c.seal(secret)
    check("the plaintext is not on the wire", secret not in sealed)
    check("...and the peer reads it back exactly", ch_s.open(sealed) == secret)

    print("\n-- a tampered frame is rejected, not decrypted into something plausible")
    ch_c2, ch_s2 = None, None
    grant = W.mint_grant(TOKEN_B, REQ, NODE_B)
    cli, srv = _run(lambda so: W.client_handshake(so, grant, NODE_B),
                    lambda so: W.server_handshake(so, TOKEN_B, NODE_B))
    ch_c2, ch_s2 = cli, srv[0]
    blob = bytearray(ch_c2.seal(b"activations"))
    blob[len(blob) // 2] ^= 0x01
    try:
        ch_s2.open(bytes(blob))
        check("a flipped bit is caught", False, "it decrypted")
    except W.HandshakeError:
        check("a flipped bit is caught", True)
    # ...and the session is now DEAD, which is the correct answer rather than the convenient
    # one. On a stream the receiver cannot tell an injected frame from a corrupted real one, so
    # carrying on either desynchronises the counters or silently drops part of the
    # conversation. TLS closes the connection on a bad MAC; so does this.
    try:
        ch_s2.open(ch_c2.seal(b"next"))
        check("...and the session is closed, not carried on", False, "it kept going")
    except W.HandshakeError:
        check("...and the session is closed, not carried on", True)

    print("\n-- nonces never repeat on a key")
    grant = W.mint_grant(TOKEN_B, REQ, NODE_B)
    cli, srv = _run(lambda so: W.client_handshake(so, grant, NODE_B),
                    lambda so: W.server_handshake(so, TOKEN_B, NODE_B))
    ch_c3, ch_s3 = cli, srv[0]
    # Same plaintext twice must not produce the same ciphertext; under AES-GCM a repeated
    # nonce is not a weakness, it is a break.
    a, b = ch_c3.seal(b"same"), ch_c3.seal(b"same")
    check("identical plaintexts encrypt differently", a != b)
    check("...and both still decrypt in order",
          ch_s3.open(a) == b"same" and ch_s3.open(b) == b"same")

    print("\n-- forward secrecy: the COORDINATOR cannot read a recording")
    # This is the reason for X25519 rather than shipping a key inside the grant. The
    # coordinator knows node_token_B and minted the grant -- it holds every long-term secret
    # in the exchange -- and still cannot derive the session key, because the ephemeral private
    # keys existed only in the two peers' memory and are gone.
    grant = W.mint_grant(TOKEN_B, REQ, NODE_B)
    cli, srv = _run(lambda so: W.client_handshake(so, grant, NODE_B),
                    lambda so: W.server_handshake(so, TOKEN_B, NODE_B))
    recording = cli.seal(b"a private conversation")
    replay_c, replay_s = _run(lambda so: W.client_handshake(so, grant, NODE_B),
                              lambda so: W.server_handshake(so, TOKEN_B, NODE_B))
    try:
        got = replay_s[0].open(recording)
        check("a recording cannot be opened by re-running the handshake", False, repr(got))
    except W.HandshakeError:
        check("a recording cannot be opened by re-running the handshake", True)
    check("...even though the coordinator holds the node token and the grant",
          W.open_grant(TOKEN_B, NODE_B, grant)[0] == REQ,
          "the grant still opens -- it is the SESSION that is unreachable")

    print("\n-- a CHAIN, not a pair: every hop is its own sealed channel")
    # The founder's question: does this only cover two machines? No -- "hop" is one LINK, and a
    # chain of N machines has N-1 of them. Each hop gets its own grant, sealed to that node's
    # own token, and its own session key. Nothing here is per-pair-of-machines or capped at two;
    # the coordinator mints one grant per hop when it builds the plan, exactly as it already
    # names one node per stage.
    chain = [("node-1", "t1" * 24), ("node-2", "t2" * 24),
             ("node-3", "t3" * 24), ("node-4", "t4" * 24)]
    keys, opened = [], []
    for node_id, token in chain:
        g = W.mint_grant(token, REQ, node_id)
        c, sv = _run(lambda so, gg=g, nn=node_id: W.client_handshake(so, gg, nn),
                     lambda so, tt=token, nn=node_id: W.server_handshake(so, tt, nn))
        opened.append(isinstance(c, W.Channel))
        # The session key itself is private to the Channel; its ciphertext of a fixed plaintext
        # is a sound proxy for "is this a different key".
        keys.append(c.seal(b"same plaintext everywhere") if isinstance(c, W.Channel) else None)
    check(f"all {len(chain)} hops establish independently", all(opened), str(opened))
    check("...and every hop has a DIFFERENT key",
          len(set(keys)) == len(keys),
          "two hops sharing a key would mean one compromised machine reads its neighbours")

    print("\n-- one compromised machine does not open the rest of the chain")
    # The property that matters as the network grows: node-2 being hostile (or seized) must not
    # give it anything it can use against node-3. It holds its own token and its own grant.
    victim_id, victim_token = chain[2]
    stolen_grant = W.mint_grant(chain[1][1], REQ, chain[1][0])     # node-2's own grant
    _, srv = _run(lambda so: W.client_handshake(so, stolen_grant, victim_id),
                  lambda so: W.server_handshake(so, victim_token, victim_id))
    check("a hostile node cannot reuse its grant against the next machine",
          isinstance(srv, W.HandshakeError), repr(srv))

    print("\n-- the same pair on a later request gets a fresh key")
    # Keys are per REQUEST, not per pair, so a machine that serves the same neighbour a
    # thousand times does not reuse one key a thousand times.
    n_id, n_tok = chain[0]
    seals = []
    for req in ("req-1", "req-2"):
        g = W.mint_grant(n_tok, req, n_id)
        c, _ = _run(lambda so, gg=g: W.client_handshake(so, gg, n_id),
                    lambda so: W.server_handshake(so, n_tok, n_id))
        seals.append(c.seal(b"identical") if isinstance(c, W.Channel) else None)
    check("two requests over the same pair do not share a key",
          seals[0] is not None and seals[0] != seals[1])

    print("\n-- junk on the port")
    for name, payload in (("random bytes", os.urandom(64)),
                          ("an HTTP request", b"GET / HTTP/1.1\r\nHost: x\r\n\r\n"),
                          ("empty", b"")):
        def junk(so, p=payload):
            if p:
                so.sendall(len(p).to_bytes(4, "big") + p)
            so.close()
            return "sent"

        _, srv = _run(junk, lambda so: W.server_handshake(so, TOKEN_B, NODE_B))
        check(f"{name} is refused cleanly", isinstance(srv, W.HandshakeError), repr(srv))

    print("\n-- a declared frame size cannot size our heap")
    def huge(so):
        so.sendall((1 << 31).to_bytes(4, "big"))
        time.sleep(0.2)
        so.close()
        return "sent"

    _, srv = _run(huge, lambda so: W.server_handshake(so, TOKEN_B, NODE_B))
    check("an absurd length is refused before allocating",
          isinstance(srv, W.HandshakeError), repr(srv))

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
