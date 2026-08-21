"""agent/test_config_falls_back_to_defaults.py — run:
       python -m agent.test_config_falls_back_to_defaults

**A comment asserted this for months and the code did the opposite.** `main()` logs

    config predates this build; using built-in defaults for: slice_dir, ...

and carries on, because a config written by an older build is missing whatever this build
added and that is explicitly not meant to be an error. But the agent reads its config forty
times with a plain `self.cfg[...]`, and `ensure_slice` reads `slice_dir` that way — so a
config predating that key did not fall back at all. It raised `KeyError` out of `setup()`,
logged `[CRASH]`, and systemd restarted it every ten seconds forever.

Live on `optiplex-server`, 2026-08-21, joining a third machine to the network with a config
naming only `coordinator` and `local_chat`. A fresh install writes the full dict, so the one
install anybody ever tested never showed it — the same shape as [P58] and [P54] before it: a
path that only breaks on the machines nobody had.

What is pinned:

  * a subscript for a key the file does not carry returns the build's default;
  * a key the file DOES carry still wins — a fallback that overrode the operator would be a
    worse bug than the one it fixed;
  * only the operator's own keys are written back to disk, because defaults belong to the
    build and freezing them into a file makes the next upgrade silently keep the old value;
  * `in` and `.keys()` still see only real keys, so the "predates this build" log keeps
    reporting exactly what it always reported;
  * a genuinely unknown key is still a KeyError, so a typo is not silently None.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from agent.agent import _Config, DEFAULT_CONFIG, load_config

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def main():
    print("-- the exact config that crash-looped the OptiPlex")
    cfg = _Config({"coordinator": "https://neuronnet.duckdns.org", "local_chat": False})
    try:
        got = cfg["slice_dir"]
        check("slice_dir falls back instead of raising KeyError",
              got == DEFAULT_CONFIG["slice_dir"], repr(got))
    except KeyError as e:
        check("slice_dir falls back instead of raising KeyError", False, f"KeyError: {e}")

    print("\n-- every key the build knows about is reachable by subscript")
    missing = []
    for k in DEFAULT_CONFIG:
        try:
            cfg[k]
        except KeyError:
            missing.append(k)
    check("no DEFAULT_CONFIG key raises on an empty-ish config", not missing, str(missing))

    print("\n-- but the operator's own values still win")
    check("a key the file carries is not overridden by the default",
          cfg["local_chat"] is False and DEFAULT_CONFIG["local_chat"] is True,
          "a fallback that beat the operator would be worse than the crash it fixed")

    print("\n-- what reaches disk, and what does not")
    written = json.loads(json.dumps(cfg))
    check("only the operator's keys are serialised",
          set(written) == {"coordinator", "local_chat"}, str(sorted(written)))
    check("...so defaults are not frozen into the file",
          "slice_dir" not in written,
          "a default written to disk stops being a default: the next build's new value is "
          "silently ignored on every machine that ever ran this one")

    print("\n-- the 'predates this build' log still sees what it always saw")
    absent = [k for k in DEFAULT_CONFIG if k not in cfg]
    check("membership reports the file's keys, not the defaults'",
          "slice_dir" in absent and "coordinator" not in absent)

    print("\n-- a typo is still an error")
    try:
        cfg["slize_dir"]
        check("an unknown key raises rather than returning None", False, "it returned")
    except KeyError:
        check("an unknown key raises rather than returning None", True)

    print("\n-- load_config returns one of these, not a bare dict")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "config.json")
        with open(p, "w") as f:
            json.dump({"coordinator": "https://x"}, f)
        loaded = load_config(p)
        check("a config read from disk falls back too", isinstance(loaded, _Config)
              and loaded["slice_dir"] == DEFAULT_CONFIG["slice_dir"])

        # The .prev recovery path returns separately, and returned a bare dict — so a config
        # recovered from corruption would have crash-looped exactly like the OptiPlex did,
        # on the path that exists to survive corruption ([P24]).
        with open(p, "w") as f:
            f.write("{ truncated")
        with open(p + ".prev", "w") as f:
            json.dump({"coordinator": "https://x"}, f)
        recovered = load_config(p)
        check("a config RECOVERED from .prev falls back too", isinstance(recovered, _Config)
              and recovered["slice_dir"] == DEFAULT_CONFIG["slice_dir"],
              "the corruption-recovery path must not hand back something that crashes")

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
