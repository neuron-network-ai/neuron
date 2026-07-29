"""test_wire_codec.py — the pipeline wire: what it costs and what it will accept.

Two things are being protected here:
  1. the wire no longer executes whatever the peer sends (the pickle RCE in recv_msg), and
  2. the smaller encodings stay faithful enough to survive a chain of many hops.

Run: python test_wire_codec.py
"""
import io
import socket
import struct

import torch

import common
import wire_codec

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def approx(a, b, rel=1e-5, abs_=0.0):
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


SHAPES = [(1, 1, 1536), (1, 45, 1536), (1, 1, 896), (1, 3, 100), (4, 2048)]


def outlier_tensor(shape, scale=6600.0):
    """A stand-in for a real hidden state: mostly ordinary, one channel enormous. Measured
    at NEURON's own junctions the worst channel is ~750x the median."""
    torch.manual_seed(0)
    t = torch.randn(*shape) * 40.0
    t[..., 11] = scale
    return t


def rel_err(y, x):
    return ((y - x).norm() / x.norm()).item()


# --------------------------------------------------------------------------- #
def test_hadamard():
    print("\nHadamard rotation")
    x = torch.randn(7, 256)
    check("fwht is its own inverse", torch.allclose(wire_codec.fwht(wire_codec.fwht(x)), x, atol=1e-4))
    # norm-preserving is what makes the rotation free: it moves no energy, only redistributes
    # which channel holds it, so the quantizer downstream sees a well-conditioned block.
    check("fwht is orthogonal (norm preserved)",
          approx(wire_codec.fwht(x).norm().item(), x.norm().item(), rel=1e-4))

    t = outlier_tensor((4, 256))
    before = (t.abs().amax() / t.abs().median()).item()
    r = wire_codec.fwht(t)
    after = (r.abs().amax() / r.abs().median()).item()
    check(f"rotation flattens outliers ({before:.0f}x -> {after:.0f}x)",
          before > 100 and after < before / 5)


def test_roundtrip():
    print("\nround-trip: shape, dtype and non-tensor fields survive")
    for codec in wire_codec.CODECS:
        good = True
        for shape in SHAPES:
            t = outlier_tensor(shape)
            msg = {"type": "act", "hidden": t, "c_compute_ms": 12.5, "ok": True, "who": "node_c"}
            out = wire_codec.decode(wire_codec.encode(msg, codec))
            good &= (tuple(out["hidden"].shape) == shape
                     and out["hidden"].dtype is torch.float32
                     and out["type"] == "act" and out["ok"] is True and out["who"] == "node_c"
                     and approx(out["c_compute_ms"], 12.5))
        check(f"{codec} round-trips every shape {SHAPES}", good)

    t = outlier_tensor((1, 8, 1536))
    check("f32 is bit-exact",
          torch.equal(wire_codec.decode(wire_codec.encode({"h": t}, "f32"))["h"], t))

    msg = {"type": "config", "s1": 9, "s2": 18, "wire": ["i8h", "f16"]}
    check("a message with no tensors round-trips", wire_codec.decode(wire_codec.encode(msg)) == msg)


def test_accuracy():
    print("\naccuracy on an outlier-heavy activation")
    t = outlier_tensor((1, 45, 1536))
    errs = {c: rel_err(wire_codec.decode(wire_codec.encode({"h": t}, c))["h"], t)
            for c in wire_codec.CODECS}
    for c, e in errs.items():
        print(f"      {c}: rel_l2 {e:.5f}")
    check("f16 error < 1e-3", errs["f16"] < 1e-3)
    check("i8h error < 2%", errs["i8h"] < 0.02)

    # The claim that justifies the rotation: same bytes on the wire, several times less
    # error. If this stops holding, i8h should be dropped for plain blockwise int8.
    flat = t.reshape(-1, wire_codec.BLOCK)
    scale = flat.abs().amax(dim=1).clamp_min(1e-12).to(torch.float16).to(torch.float32)
    q = torch.round(flat / scale[:, None] * 127.0).clamp(-127, 127)
    plain = (q * scale[:, None] / 127.0).reshape(t.shape)
    check(f"i8h beats unrotated int8 at the same size "
          f"({errs['i8h']:.5f} vs {rel_err(plain, t):.5f})",
          errs["i8h"] < rel_err(plain, t) / 3)


def test_huge_magnitude_does_not_silently_zero():
    """i8h's per-block scale used to travel as fp16. The rotation preserves each block's L2
    norm, so a block with norm > 65504 overflowed the scale to inf and the whole block
    decoded as ZEROS -- no exception, no warning, just a dead activation. Scales are fp32
    now; this pins that down."""
    print("\nlarge-magnitude activations")
    t = torch.full((1, 2, 1536), 5e4)
    out = wire_codec.decode(wire_codec.encode({"h": t}, "i8h"))["h"]
    check(f"a block with L2 norm >> 65504 survives (rel_l2 {rel_err(out, t):.5f})",
          torch.isfinite(out).all() and rel_err(out, t) < 0.02)


def test_sizes():
    print("\nbytes on the wire")
    t = torch.randn(1, 64, 1536)
    n = t.numel()
    for c, want in (("f32", 4.0), ("f16", 2.0), ("i8h", 1.016)):
        got = len(wire_codec.encode({"hidden": t}, c)) / n
        check(f"{c} is {want} B/elem (got {got:.3f})", approx(got, want, abs_=0.05))


def test_negotiation():
    print("\ncodec negotiation")
    check("prefers the offerer's order", wire_codec.negotiate(["f16", "i8h"]) == "f16"
          and wire_codec.negotiate(["i8h", "f16"]) == "i8h")
    # None is the signal that keeps a half-upgraded fleet working: the caller then sends the
    # legacy format, which every build can still read.
    check("unknown/absent offer -> None (stay legacy)",
          wire_codec.negotiate(None) is None and wire_codec.negotiate([]) is None
          and wire_codec.negotiate(["some-future-codec"]) is None)

    buf = io.BytesIO()
    torch.save({"type": "bye"}, buf)
    check("frame detection does not mistake a pickle for a frame",
          not wire_codec.is_frame(buf.getvalue()) and wire_codec.is_frame(wire_codec.encode({"t": 1})))

    bad = False
    try:
        wire_codec.encode({"hidden": torch.zeros(1, 4)}, "nope")
    except ValueError:
        bad = True
    check("unknown codec is refused", bad)


# --------------------------------------------------------------------------- #
class Evil:
    """What an untrusted peer can put on the wire. torch.load(weights_only=False) calls
    whatever __reduce__ names -- here a harmless marker, in the wild anything at all."""

    def __reduce__(self):
        return (dict, ({"pwned": True},))


def _pair():
    a, b = socket.socketpair()
    a.settimeout(5)
    b.settimeout(5)
    return a, b


def _raw_send(sock, obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    data = buf.getvalue()
    sock.sendall(struct.pack(">Q", len(data)))
    sock.sendall(data)


def test_socket():
    print("\ncommon.send_msg / recv_msg over a real socket")
    for codec in [None] + list(wire_codec.CODECS):
        a, b = _pair()
        try:
            t = outlier_tensor((1, 6, 1536))
            common.send_msg(a, {"type": "act", "hidden": t, "b_compute_ms": 3.5}, codec=codec)
            got = common.recv_msg(b)
            check(f"codec={codec} round-trips over a socket",
                  got["type"] == "act" and approx(got["b_compute_ms"], 3.5)
                  and tuple(got["hidden"].shape) == (1, 6, 1536))
        finally:
            a.close(), b.close()

    # A node predating wire_codec sends a plain torch.save pickle. It must still work, or a
    # rolling upgrade takes the network down.
    a, b = _pair()
    try:
        _raw_send(a, {"type": "act", "hidden": torch.ones(1, 1, 4)})
        check("a legacy (torch.save) sender is still readable",
              torch.equal(common.recv_msg(b)["hidden"], torch.ones(1, 1, 4)))
    finally:
        a.close(), b.close()


def test_security():
    print("\nsecurity")
    # The regression test for the RCE: recv_msg used to unpickle the payload and run
    # Evil.__reduce__. Every node port is reachable by the next node in the chain, and since
    # Session 12 it is published on a public relay.
    a, b = _pair()
    try:
        _raw_send(a, {"type": "act", "hidden": Evil()})
        blocked = False
        try:
            got = common.recv_msg(b)
            blocked = got.get("hidden") != {"pwned": True}
        except Exception:
            blocked = True
        check("a hostile pickle does NOT get executed", blocked)
    finally:
        a.close(), b.close()

    # Node ports see scanner garbage from the open internet. An unchecked 8-byte length let
    # a stray probe ask a 1 GB VM for an arbitrary buffer.
    a, b = _pair()
    try:
        a.sendall(struct.pack(">Q", 1 << 62))
        refused = False
        try:
            common.recv_msg(b)
        except ConnectionError as e:
            refused = "exceeds" in str(e)
        check("an absurd length prefix is refused before allocating", refused)
    finally:
        a.close(), b.close()

    refused = False
    try:
        wire_codec.decode(b"PK\x03\x04not a frame at all")
    except ValueError:
        refused = True
    check("decode refuses a non-frame", refused)


def main():
    test_hadamard()
    test_roundtrip()
    test_accuracy()
    test_huge_magnitude_does_not_silently_zero()
    test_sizes()
    test_negotiation()
    test_socket()
    test_security()
    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
