"""test_driver_s1.py — the driver's stage-1 width follows placement, not an env var. [P44]

Run: python test_driver_s1.py

`node_a.coord_get_chain` refuses any chain whose stage 1 is not `[0, expected_s1 - 1]`. That
number used to be `NEURON_S1`, read AT IMPORT in two processes on two different machines --
`neuron_driver` and `coordinator/config` -- with a comment in each saying it had to match the
other. Changing the width of stage 1 therefore meant a coordinated restart of the coordinator
and every driver, including volunteers' PCs nobody can reach. Layer ranges have the whole
prepare->ready->cutover handshake for exactly this; s1 had an environment variable.

The coordinator owns it now. It publishes the width on `/node/{id}/slice-info`, the agent
downloads a driver shard of that width, and `_Driver` reads the width back off the shard it
loaded. Three properties have to hold, and the third is the one that makes it safe:

  1. the driver asserts the width it LOADED, not the compiled-in constant;
  2. every consumer inside the driver moves together -- the shard load, the stage batcher, the
     chain request and the config sent on the wire. A driver that requests an 18-wide chain and
     then runs 10 layers is worse than one that refuses;
  3. it asserts only what it can SERVE. If the coordinator has moved on and this shard has not,
     refusing is correct: running 10 layers where the chain expects 18 hands the next node an
     activation from the wrong depth, which is a wrong answer instead of a clean error.

No torch, no network, no weights -- the loaders are stubbed. This is about which number is used
where, which is exactly the part that was wrong.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import neuron_driver
import node_a
import slice_downloader

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def seed(tmp, name, ls, le, model_id="some/model"):
    d = os.path.join(tmp, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, slice_downloader.SLICE_MARKER), "w") as f:
        json.dump({"model_id": model_id, "layer_start": ls, "layer_end": le}, f)
    return d


def driver_loading(slice_dir):
    """A fresh _Driver that 'loads' `slice_dir` with the heavy parts stubbed out."""
    d = neuron_driver._Driver()
    real_load, real_st = slice_downloader.load_slice_model, sys.modules.get("transformers")

    class _Tok:
        eos_token_id = 0

    class _Cfg:
        num_hidden_layers = 28

    fake = type(sys)("transformers")
    fake.AutoTokenizer = type("T", (), {"from_pretrained": staticmethod(lambda p: _Tok())})
    fake.AutoConfig = type("C", (), {"from_pretrained": staticmethod(lambda p: _Cfg())})
    sys.modules["transformers"] = fake
    slice_downloader.load_slice_model = lambda p: "MODEL"
    try:
        d.load_from_slice(slice_dir)
    finally:
        slice_downloader.load_slice_model = real_load
        if real_st is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = real_st
    return d


def chain(s1_of_stage1):
    """A coordinator chain whose stage 1 is that wide."""
    return [{"layers": [0, s1_of_stage1 - 1], "ip": "1.1.1.1", "port": 1, "node_id": "a"},
            {"layers": [s1_of_stage1, 27], "ip": "1.1.1.2", "port": 2, "node_id": "b"}]


def main():
    tmp = tempfile.mkdtemp(prefix="neuron_s1_")

    # ---------- 1. the width comes from the shard -------------------------------
    d = driver_loading(seed(tmp, "wide", 0, 17))
    check("a driver that loaded layers 0-17 says s1=18, not the env default",
          d.s1 == 18, f"got {d.s1}, module fallback is {neuron_driver.S1}")
    check("...and a driver that loaded 0-9 says 10",
          driver_loading(seed(tmp, "narrow", 0, 9)).s1 == 10)
    check("a fresh driver with nothing loaded falls back to the constant",
          neuron_driver._Driver().s1 == neuron_driver.S1)

    # ---------- 2. a shard that is not a driver shard is refused ----------------
    # A marker starting anywhere but 0 describes a COMPUTE slice. Its width is not stage 1's,
    # and adopting it would have the driver assert a number with no relation to the chain.
    mid = driver_loading(seed(tmp, "middle", 10, 27))
    check("a compute slice (0 != layer_start) does not redefine s1",
          mid.s1 == neuron_driver.S1, f"got {mid.s1}")
    unmarked = os.path.join(tmp, "unmarked")
    os.makedirs(unmarked, exist_ok=True)
    check("an unreadable marker leaves the fallback in place rather than crashing the load",
          driver_loading(unmarked).s1 == neuron_driver.S1)

    # ---------- 3. slice_range reads what download_slice writes -----------------
    check("slice_range returns the recorded range",
          slice_downloader.slice_range(seed(tmp, "r", 3, 11)) == (3, 11))
    check("...and None when there is no marker at all",
          slice_downloader.slice_range(unmarked) is None)
    bad = os.path.join(tmp, "corrupt")
    os.makedirs(bad, exist_ok=True)
    with open(os.path.join(bad, slice_downloader.SLICE_MARKER), "w") as f:
        f.write("{not json")
    check("...and None for a corrupt marker, rather than raising into the load path",
          slice_downloader.slice_range(bad) is None)
    with open(os.path.join(bad, slice_downloader.SLICE_MARKER), "w") as f:
        json.dump({"model_id": "m"}, f)          # a marker from before ranges were recorded
    check("...and None for a marker with no range recorded",
          slice_downloader.slice_range(bad) is None)

    # ---------- 4. every consumer inside the driver moves together --------------
    # The failure this guards: `coord_get_chain` asking for an 18-wide chain while the stage
    # batcher still runs 10 layers. Both read self.s1, so they cannot disagree -- asserted by
    # reading the source, because the alternative is loading a real model.
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "neuron_driver.py"),
               encoding="utf-8").read()
    body = src.split("class _Driver:", 1)[1]
    for consumer, snippet in (
            ("the shard load", "common.load_model_shard(0, self.s1"),
            ("the stage batcher", "model, self.s1, ids, cache, lengths"),
            ("the chain request", "self.s1, wallet_id"),
            # `cfg["s1"] = ...`, not a key in the dict literal: since [P55] the driver sends
            # `s1` ONLY to a middle relay, which is the only hop that has ever read it. A last
            # stage that receives it cannot tell real traffic from a verifier's probe. The
            # thing this loop pins is unchanged -- every consumer reads self.s1, so none of
            # them can disagree about how wide stage 1 is.
            ("the config sent on the wire", 'cfg["s1"] = self.s1')):
        check(f"{consumer} uses self.s1", snippet in body)
    check("no consumer inside _Driver still reads the module constant",
          not any(ln.strip().startswith("#") is False and "S1" in ln and "self.s1" not in ln
                  and "NEURON_S1" not in ln and "0..S1-1" not in ln
                  for ln in body.split("\n")),
          "\n".join(ln for ln in body.split("\n")
                    if "S1" in ln and "self.s1" not in ln and "NEURON_S1" not in ln))

    # ---------- 5. and it is still asserted against the coordinator's chain -----
    posted = {}

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"chain": posted["chain"], "request_id": "r1"}

    real_post = node_a.requests.post
    try:
        node_a.requests.post = lambda *a, **k: _Resp()
        posted["chain"] = chain(18)
        node_a.coord_get_chain("http://c", "hi", 4, 18, "w1")
        check("a driver holding 0-17 accepts an 18-wide stage 1", True)

        posted["chain"] = chain(10)
        try:
            node_a.coord_get_chain("http://c", "hi", 4, 18, "w1")
            check("a driver holding 0-17 REFUSES a 10-wide stage 1", False,
                  "it would run 17 layers where the chain expects 9 and hand the next node "
                  "an activation from the wrong depth")
        except RuntimeError as e:
            check("a driver holding 0-17 REFUSES a 10-wide stage 1", True)
            check("...and the refusal says the shard is stale and how to fix it",
                  "0..17" in str(e) and "Restart" in str(e), str(e))
    finally:
        node_a.requests.post = real_post

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
