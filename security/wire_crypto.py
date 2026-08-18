"""security/wire_crypto.py — an authenticated, encrypted channel for the pipeline hop ([P52]).

**Why this exists.** `docs/index.html` sells "privacy by architecture". Until this module the
pipeline wire was plaintext: the driver embeds a user's prompt, and the resulting hidden states
crossed the internet as a JSON header plus raw tensor bytes, through a PUBLIC relay port, into
STRANGERS' machines. `node_server` authenticated nobody, so it could not tell the driver from
anyone else who had dialled that published port.

[P19] made the wire safe to PARSE — nothing on it can execute. That protects the *node*. It
does nothing for the *user*, whose prompt-derived data is what is travelling, and those are the
two different properties the site was conflating.

**The trust that already exists, and the design that follows from it.** The coordinator knows
every node's `node_token` (it issued them) and already hands the driver a per-request plan over
HTTPS. So the coordinator can act as the introducer, and no new PKI, certificate or
out-of-band exchange is needed:

    coordinator                      driver                        node B
    -----------                      ------                        ------
    knows node_token_B
    mints GRANT for this hop  ---->  opaque blob, cannot read it
                                     dials node B, sends grant  ->  decrypts with its OWN
                                                                    node_token -- only the real
                                                                    node B can
    <---------------- X25519 ephemeral keys both ways, bound to the grant ------------->
                                     shared session key, AEAD from here on

**Both properties come out of one exchange.** A grant only decrypts under `node_token_B`, so
replying at all proves the peer really is node B; and the grant only exists because the
coordinator authorised this request, so a scanner that dials the port has nothing to send.
Authentication is not a separate step bolted beside encryption -- it is the same step.

**Forward secrecy is deliberate, and it is the reason for X25519 rather than simply shipping a
key inside the grant.** The simpler design (coordinator picks the session key, wraps it to the
node) would work and is half the code -- but the coordinator would then be able to decrypt every
conversation on the network, forever, from a recording. The network already trusts the
coordinator for placement and payment; it does not have to trust it with the content of
everyone's prompts, and after this it does not. A recorded session cannot be opened later even
by the machine that introduced the two parties.

**What this does NOT claim.** It secures the hop between two nodes. The node at the other end
still *computes on* the plaintext activations it receives -- that is what a node is for -- so
this is confidentiality in transit against the network, the relay and any bystander, not
confidentiality against a chain member. Saying otherwise would be [P31] again.

Depends on `cryptography` (already present transitively via `eth-account`), which supplies
X25519, HKDF and AES-GCM from a maintained C implementation. Nothing here invents a primitive.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import struct
import time

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey, X25519PublicKey)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

MAGIC = b"NRNS"                 # NEURON Secure; distinct from wire_codec's NRNW
VERSION = 1
GRANT_TTL_S = 300.0             # a grant is for one request, not a standing key
KEY_LEN = 32
NONCE_LEN = 12
MAX_FRAME = 512 << 20           # mirrors common.MAX_MSG_BYTES: a scanner must not size our heap


class HandshakeError(Exception):
    """The peer could not prove it is who the coordinator said it is, or the grant is bad.

    Its own type so callers can tell "this is not the node I was sent to" from an ordinary
    network fault -- the same reasoning that made `RangeMismatch` a type in [P37]: a failure
    that means something specific must not be swallowed into the generic retry path.
    """


def _node_key(node_token: str) -> bytes:
    """A node's long-term symmetric key, derived from the token it already has.

    Derived rather than used raw so that a grant blob can never be replayed against any other
    context that also happens to use the token (the coordinator's HTTP headers, for one).
    """
    return HKDF(algorithm=hashes.SHA256(), length=KEY_LEN, salt=b"neuron-wire-v1",
                info=b"node-long-term").derive(node_token.encode())


def mint_grant(node_token: str, request_id: str, node_id: str, now=None) -> bytes:
    """COORDINATOR SIDE. Authorise one hop to `node_id` for `request_id`.

    The result is opaque to the driver that carries it: it is sealed to the destination node's
    own token, so the driver cannot read it, alter it, or mint one for a node it was not sent
    to. `node_id` and `request_id` are authenticated as associated data, which is what stops a
    grant for one node being presented to another.
    """
    now = time.time() if now is None else now
    key = _node_key(node_token)
    nonce = os.urandom(NONCE_LEN)
    body = struct.pack("<d", now + GRANT_TTL_S) + request_id.encode()
    sealed = AESGCM(key).encrypt(nonce, body, node_id.encode())
    return MAGIC + bytes([VERSION]) + nonce + sealed


def open_grant(node_token: str, node_id: str, grant: bytes, now=None):
    """NODE SIDE. Verify a grant really was minted by someone holding our token.

    Returns (request_id, expires_at). Raises HandshakeError on anything wrong -- a forged blob,
    a grant for a different node, or an expired one. Nothing here reports WHICH, deliberately:
    an oracle that distinguishes "bad key" from "expired" is a gift to whoever is probing.
    """
    now = time.time() if now is None else now
    if not grant.startswith(MAGIC) or len(grant) < len(MAGIC) + 1 + NONCE_LEN + 16:
        raise HandshakeError("not a NEURON grant")
    if grant[len(MAGIC)] != VERSION:
        raise HandshakeError("unsupported grant version")
    off = len(MAGIC) + 1
    nonce, sealed = grant[off:off + NONCE_LEN], grant[off + NONCE_LEN:]
    try:
        body = AESGCM(_node_key(node_token)).decrypt(nonce, sealed, node_id.encode())
    except Exception:                                        # noqa: BLE001
        raise HandshakeError("grant is not valid for this node")
    (expires,) = struct.unpack_from("<d", body, 0)
    if now > expires:
        raise HandshakeError("grant is not valid for this node")
    return body[8:].decode(), expires


def _derive(shared: bytes, transcript: bytes) -> bytes:
    """Session key from the X25519 secret, bound to everything both sides said.

    The transcript is in the KDF, not merely checked afterwards: if an attacker altered a public
    key in flight the two sides derive DIFFERENT keys and the first frame fails to authenticate,
    rather than the two of them agreeing on something an attacker chose.
    """
    return HKDF(algorithm=hashes.SHA256(), length=KEY_LEN, salt=b"neuron-wire-v1",
                info=b"session:" + transcript).derive(shared)


class Channel:
    """An established session. Encrypts each frame with its own nonce.

    Nonces are a counter, not random: at 96 bits random collision is negligible in theory and a
    counter removes the question entirely, and a nonce reuse under AES-GCM is catastrophic
    rather than merely weak. Send and receive counters are separate, and each direction has its
    own key, so the two can never collide with each other.
    """

    def __init__(self, send_key: bytes, recv_key: bytes):
        self._send, self._recv = AESGCM(send_key), AESGCM(recv_key)
        self._send_n, self._recv_n = 0, 0
        self._dead = False

    def _nonce(self, counter: int) -> bytes:
        return counter.to_bytes(NONCE_LEN, "big")

    def seal(self, plaintext: bytes) -> bytes:
        if self._dead:
            raise HandshakeError("channel is closed after an authentication failure")
        out = self._send.encrypt(self._nonce(self._send_n), plaintext, None)
        self._send_n += 1
        return out

    def open(self, ciphertext: bytes) -> bytes:
        """Decrypt one frame. A failure is FATAL to the session, deliberately.

        The first version of this tried to survive a bad frame without advancing the counter,
        so that "an attacker shouting at the socket cannot desynchronise us". That reasoning is
        wrong on a stream, and the test caught it: sender and receiver counters are a shared
        clock, so tolerating a gap on one side guarantees every later frame fails anyway.

        More importantly it is not decidable. On TCP a frame that fails authentication is either
        injected (skip it, stay in sync) or a real frame that was corrupted or truncated (skip
        it and you have silently dropped part of the conversation) -- and the receiver cannot
        tell which. TLS closes the connection on a bad MAC for exactly this reason, and this
        does the same: mark the channel dead and refuse to continue.
        """
        if self._dead:
            raise HandshakeError("channel is closed after an authentication failure")
        try:
            out = self._recv.decrypt(self._nonce(self._recv_n), ciphertext, None)
        except Exception:                                    # noqa: BLE001
            self._dead = True
            raise HandshakeError("frame failed authentication")
        self._recv_n += 1
        return out


def _send(sock, blob: bytes):
    sock.sendall(struct.pack(">I", len(blob)) + blob)


def peek_is_secure(first4: bytes) -> bool:
    """Does this connection want the secure handshake? ([P52] rolling upgrade)

    A legacy connection opens with `common.send_msg`'s 8-byte big-endian length, and any real
    message is far under 4 GiB, so its first four bytes are always `\x00\x00\x00\x00`. The
    secure client therefore leads with MAGIC in the clear, which cannot collide, and a node can
    decide from four bytes without consuming anything a legacy sender needs.

    Deliberately a pure function on bytes the caller already has: peeking on a socket differs
    across platforms, and a node must not have to guess.
    """
    return first4 == MAGIC


def _recv(sock, limit=MAX_FRAME) -> bytes:
    head = _recv_exact(sock, 4)
    (n,) = struct.unpack(">I", head)
    if n > limit:
        raise HandshakeError(f"declared handshake frame {n} exceeds {limit}")
    return _recv_exact(sock, n)


def _recv_exact(sock, n: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        b = sock.recv(min(n - got, 1 << 20))
        if not b:
            raise HandshakeError("socket closed during handshake")
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def client_handshake(sock, grant: bytes, node_id: str) -> Channel:
    """DRIVER SIDE. Prove we were authorised, and agree a key nobody else holds."""
    eph = X25519PrivateKey.generate()
    pub = eph.public_key().public_bytes(serialization.Encoding.Raw,
                                        serialization.PublicFormat.Raw)
    # MAGIC goes on the wire RAW and first, outside the length frame, so a node can tell a
    # secure connection from a legacy one by reading four bytes (see peek_is_secure).
    sock.sendall(MAGIC)
    _send(sock, bytes([VERSION]) + struct.pack(">I", len(grant)) + grant + pub)
    reply = _recv(sock)
    if len(reply) != 32 + 32:
        raise HandshakeError("malformed server hello")
    peer_pub, confirm = reply[:32], reply[32:]
    shared = eph.exchange(X25519PublicKey.from_public_bytes(peer_pub))
    transcript = hashlib.sha256(pub + peer_pub + grant + node_id.encode()).digest()
    base = _derive(shared, transcript)
    # The node's confirmation is what proves it opened the grant. Without it an attacker who
    # simply relays our bytes to a machine that cannot read the grant still completes an X25519
    # exchange with us -- encrypted, and to the wrong party.
    expect = hmac.new(base, b"server-confirm" + transcript, hashlib.sha256).digest()
    if not hmac.compare_digest(expect, confirm):
        raise HandshakeError("peer did not prove it holds this node's token")
    return Channel(send_key=_dir_key(base, b"c2s"), recv_key=_dir_key(base, b"s2c"))


def server_handshake(sock, node_token: str, node_id: str, seen=None,
                     magic_consumed: bool = False) -> tuple[Channel, str]:
    """NODE SIDE. Returns (channel, request_id).

    `seen` is an optional set of already-used grants; passing one makes a replayed grant fail
    even inside its TTL. Kept as a caller-owned set rather than module state so a long-running
    node can bound it however it likes.
    """
    # MAGIC has already been consumed by the caller's peek (node_server reads it to decide
    # which protocol this is), so `magic_consumed` says whether we still need to eat it.
    if not magic_consumed:
        if _recv_exact(sock, len(MAGIC)) != MAGIC:
            raise HandshakeError("not a NEURON secure hello")
    hello = _recv(sock)
    if len(hello) < 1 + 4 + 32:
        raise HandshakeError("not a NEURON secure hello")
    if hello[0] != VERSION:
        raise HandshakeError("unsupported wire version")
    off = 1
    (glen,) = struct.unpack_from(">I", hello, off)
    off += 4
    grant, peer_pub = hello[off:off + glen], hello[off + glen:]
    if len(peer_pub) != 32:
        raise HandshakeError("malformed client hello")
    request_id, _ = open_grant(node_token, node_id, grant)
    if seen is not None:
        fp = hashlib.sha256(grant).digest()
        if fp in seen:
            raise HandshakeError("grant is not valid for this node")   # replayed
        seen.add(fp)
    eph = X25519PrivateKey.generate()
    pub = eph.public_key().public_bytes(serialization.Encoding.Raw,
                                        serialization.PublicFormat.Raw)
    shared = eph.exchange(X25519PublicKey.from_public_bytes(peer_pub))
    transcript = hashlib.sha256(peer_pub + pub + grant + node_id.encode()).digest()
    base = _derive(shared, transcript)
    confirm = hmac.new(base, b"server-confirm" + transcript, hashlib.sha256).digest()
    _send(sock, pub + confirm)
    return Channel(send_key=_dir_key(base, b"s2c"), recv_key=_dir_key(base, b"c2s")), request_id


def _dir_key(base: bytes, label: bytes) -> bytes:
    """One key per direction. Sharing a key both ways would put two nonce counters on one key."""
    return HKDF(algorithm=hashes.SHA256(), length=KEY_LEN, salt=b"neuron-wire-v1",
                info=b"dir:" + label).derive(base)
