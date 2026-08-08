"""agent/test_device_choice.py — run: python -m agent.test_device_choice

The post-install compute-device choice (CPU / GPU / automatic).

The property this file exists for is an ORDERING one, and it is the whole reason the setting
lives where it does. `common.DEVICE` is resolved exactly once, at import (`common.py:113`), and
`agent/agent.py` imports `agent.node_server` -> `common` at MODULE level. So a device setting
read in `main()` would be read, saved, ticked in the tray menu, and have no effect whatsoever:
the device was chosen while the module was still being imported. `_apply_device_preference()`
therefore runs at import time, ahead of that block, and these tests pin that.

The second property is honesty. This build ships a CPU-only torch, so choosing "GPU" cannot
work — and it must SAY so rather than appear to succeed. A menu that looks like it worked and
did nothing is what v0.18.0 shipped ([P31]); it is not repeated here.

Config files are written to a temp dir; nothing here touches a real install or imports torch
for its own sake.
"""
import json
import os
import sys
import tempfile

import agent.agent as A

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def write_cfg(d, **keys):
    p = os.path.join(d, "config.json")
    with open(p, "w") as f:
        json.dump(keys, f)
    return p


def apply(path, env=None, argv=None):
    """Run the real _apply_device_preference against a fake env, return (env, note)."""
    env = {} if env is None else dict(env)
    note = A._apply_device_preference(argv=argv if argv is not None else ["--config", path],
                                      environ=env)
    return env, note


def main():
    d = tempfile.mkdtemp(prefix="neuron-device-")

    # ---------- the ordering property ----------
    # Stated as an assertion rather than a comment: if someone later moves this call into
    # main(), this test is what tells them the setting stopped working.
    # Matched as STATEMENTS (line-anchored), not as substrings. A plain `src.index()` finds the
    # mention of this import inside _apply_device_preference's own docstring and reports the
    # ordering backwards -- which it did, on the first run of this test.
    import re
    def line_of(pattern):
        m = re.search(pattern, src, re.MULTILINE)
        assert m, pattern
        return src[:m.start()].count("\n") + 1

    src = open(A.__file__, encoding="utf-8").read()
    apply_at = line_of(r"^_DEVICE_NOTE = _apply_device_preference\(\)")
    import_at = line_of(r"^\s+from agent\.node_server import NodeServer")
    check("the device preference is applied BEFORE node_server (and so before common) is "
          "imported", apply_at < import_at,
          f"applied at line {apply_at}, node_server imported at line {import_at}")
    check("...and it runs at module level, not inside main()",
          apply_at < line_of(r"^def main\(\):"))

    # `common` resolves its device at import and never revisits it -- the reason for all of
    # the above. Asserted so the constraint is documented by something that fails.
    import common
    check("common.DEVICE is a module-level constant resolved once at import",
          hasattr(common, "DEVICE") and not callable(getattr(common, "DEVICE")))

    # ---------- the mapping ----------
    env, note = apply(write_cfg(d, device="cpu"))
    check("device: cpu pins NEURON_DEVICE=cpu", env.get("NEURON_DEVICE") == "cpu", env)
    check("...and says so", note and "cpu" in note)

    env, note = apply(write_cfg(d, device="auto"))
    check("device: auto sets nothing, leaving common._resolve_device() to decide",
          "NEURON_DEVICE" not in env, env)
    check("...and says nothing", note is None)

    env, note = apply(write_cfg(d))
    check("a config with no device key behaves as auto", "NEURON_DEVICE" not in env, env)

    env, note = apply(os.path.join(d, "does-not-exist.json"))
    check("no config at all (first run) behaves as auto", "NEURON_DEVICE" not in env, env)

    env, note = apply(write_cfg(d, device="nonsense"))
    check("an unrecognised device is ignored rather than pinned",
          "NEURON_DEVICE" not in env, env)
    check("...and the log says which value was ignored", note and "nonsense" in note, note)

    # ---------- an explicit environment override always wins ----------
    env, note = apply(write_cfg(d, device="cpu"), env={"NEURON_DEVICE": "cuda:1"})
    check("NEURON_DEVICE from the environment beats the config file",
          env["NEURON_DEVICE"] == "cuda:1", env)
    check("...and the log says the environment won", note and "environment" in note)

    # ---------- honesty: GPU on a build that cannot use one ----------
    # This is the shipped state. torch.cuda.is_available() is False on a `+cpu` wheel.
    env, note = apply(write_cfg(d, device="gpu"))
    import torch
    if torch.cuda.is_available():
        check("device: gpu pins cuda:0 when torch can actually use a card",
              env.get("NEURON_DEVICE") == "cuda:0", env)
    else:
        check("device: gpu does NOT pin a CUDA device this build cannot use",
              "NEURON_DEVICE" not in env, env)
        check("...and it says the build computes on CPU, rather than silently doing nothing",
              note and "CPU" in note, note)
        check("...naming the build as the reason, not the absent hardware",
              note and "build" in note.lower(), note)

    # ---------- --config is honoured at import time, before argparse exists ----------
    p = write_cfg(d, device="cpu")
    check("--config <path> is found in argv", A._config_path_from_argv(["--config", p]) == p)
    check("--config=<path> is found too", A._config_path_from_argv([f"--config={p}"]) == p)
    check("no --config falls back to the default path",
          A._config_path_from_argv([]) == A.CONFIG_PATH)
    check("a trailing --config with no value does not raise",
          A._config_path_from_argv(["--config"]) == A.CONFIG_PATH)

    # ---------- a device preference must never stop a node starting ----------
    bad = os.path.join(d, "broken.json")
    with open(bad, "w") as f:
        f.write("{not json at all")
    env, note = apply(bad)
    check("an unreadable config does not raise and does not pin a device",
          "NEURON_DEVICE" not in env, env)

    # ---------- fresh-install defaults ----------
    import agent.install as I
    check("install and agent agree on the donation default",
          I.DEFAULT_CONFIG["donation_mode"] == A.DEFAULT_CONFIG["donation_mode"] == "balanced")
    check("both carry a device key", "device" in I.DEFAULT_CONFIG and "device" in A.DEFAULT_CONFIG)
    check("detect_device answers concretely, never 'auto'",
          I.detect_device() in ("cpu", "gpu"), I.detect_device())
    check("on a build torch cannot use a GPU with, detection says cpu",
          I.detect_device() == ("gpu" if torch.cuda.is_available() else "cpu"))

    # write_config on a FRESH install stamps the detected device and the balanced default.
    real_path, I.CONFIG_PATH = I.CONFIG_PATH, os.path.join(d, "fresh.json")
    try:
        I.write_config("https://example.invalid", None, None)
        fresh = json.load(open(I.CONFIG_PATH))
        check("a fresh install gets donation_mode: balanced",
              fresh["donation_mode"] == "balanced", fresh.get("donation_mode"))
        check("a fresh install gets a concrete device, detected from the machine",
              fresh["device"] in ("cpu", "gpu"), fresh.get("device"))

        # An EXISTING config must not be re-defaulted -- write_config merges, and someone who
        # chose 'idle' or pinned a device keeps it. This is the failure write_config already
        # documents for max_cpu_pct, applied to the two keys added here.
        with open(I.CONFIG_PATH, "w") as f:
            json.dump({"donation_mode": "idle", "device": "cpu", "node_id": "n1"}, f)
        I.write_config("https://example.invalid", None, None)
        kept = json.load(open(I.CONFIG_PATH))
        check("an existing config keeps its donation_mode", kept["donation_mode"] == "idle")
        check("...and its device", kept["device"] == "cpu")
        check("...and its identity", kept["node_id"] == "n1")

        # The machine this runs on has no usable GPU, so the "gpu" branch is unreachable
        # without a stub -- and an untested branch is how the original GPU support shipped.
        os.remove(I.CONFIG_PATH)
        real_detect, I.detect_device = I.detect_device, lambda: "gpu"
        try:
            I.write_config("https://example.invalid", None, None)
            check("with a usable GPU detected, a fresh install is stamped device: gpu",
                  json.load(open(I.CONFIG_PATH))["device"] == "gpu")
        finally:
            I.detect_device = real_detect

        # detect_device's own logic, exercised both ways against torch rather than assumed.
        real_avail = torch.cuda.is_available
        try:
            torch.cuda.is_available = lambda: True
            check("detect_device says gpu when torch reports a usable card",
                  I.detect_device() == "gpu")
            torch.cuda.is_available = lambda: False
            check("...and cpu when it does not", I.detect_device() == "cpu")

            def _boom():
                raise RuntimeError("driver exploded")
            torch.cuda.is_available = _boom
            check("...and cpu, not a crash, when the probe raises",
                  I.detect_device() == "cpu")
        finally:
            torch.cuda.is_available = real_avail
    finally:
        I.CONFIG_PATH = real_path

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
