"""engine/test_force_network.py — run: python -m engine.test_force_network

NEURON_FORCE_NETWORK=1 is how an operator says "send this over the node chain even though this
machine could answer it itself". It is the only way to see what the NETWORK actually does, which
makes it the instrument every speed claim about the network is measured with.

It was honoured in exactly one of the two dispatch paths. `ui/app.py:_drive` checked it;
`api/openai_compat.py` branched straight on `local_gguf.available()` and never looked. So on
2026-08-19 an attempt to measure end-to-end network speed through the OpenAI-compatible API
measured the LOCAL engine instead and reported 4.97 tok/s — about 3x better than the real
figure. The only reason it was caught is that the coordinator's `requests_served` never moved.

A broken measuring instrument is worse than no measurement: it does not look like a failure, it
looks like good news. So the override now lives at the bottom of `local_gguf`, where every
caller that asks "can this machine serve it" inherits it, and these tests pin that it cannot be
routed around — including by the model pin, which is a DIFFERENT question (which model to run
locally) and must not smuggle a request back onto this machine.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import local_gguf                       # noqa: E402

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
    saved = {k: os.environ.get(k) for k in ("NEURON_FORCE_NETWORK", "NEURON_LOCAL_MODEL")}
    real_can_serve = local_gguf.can_serve
    MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    try:
        # Pretend this machine can always run everything, so the only thing under test is the
        # override rather than whatever RAM the test runner happens to have.
        local_gguf.can_serve = lambda mid, ram_gb=None: mid in local_gguf.GGUF_MODELS

        print("\n-- off by default: nothing changes for an ordinary user")
        os.environ.pop("NEURON_FORCE_NETWORK", None)
        os.environ.pop("NEURON_LOCAL_MODEL", None)
        check("network_forced() is False when unset", local_gguf.network_forced() is False)
        check("...and a local model is still offered",
              local_gguf.best_local_model(require_cached=False) is not None)

        print("\n-- set: this machine stops volunteering to answer")
        os.environ["NEURON_FORCE_NETWORK"] = "1"
        check("network_forced() is True", local_gguf.network_forced() is True)
        check("available() says no, which is what api/openai_compat.py branches on",
              local_gguf.available(MODEL) is False,
              "this is the exact check that made the API unable to reach the network")
        check("best_local_model() offers nothing, which is what ui/app.py branches on",
              local_gguf.best_local_model(require_cached=False) is None)

        print("\n-- the model PIN must not smuggle the request back onto this machine")
        os.environ["NEURON_LOCAL_MODEL"] = MODEL
        check("a pinned local model is still refused while the override is set",
              local_gguf.best_local_model(require_cached=False) is None,
              "NEURON_LOCAL_MODEL answers WHICH model, not WHETHER to run locally")
        os.environ.pop("NEURON_LOCAL_MODEL", None)

        print("\n-- only the exact value counts, so a typo fails safe (local, as before)")
        for val in ("0", "true", "yes", ""):
            os.environ["NEURON_FORCE_NETWORK"] = val
            check(f"{val!r} does not force the network", local_gguf.network_forced() is False)

        print("\n-- unset again: the machine goes back to answering locally")
        os.environ.pop("NEURON_FORCE_NETWORK", None)
        check("best_local_model() offers a model again",
              local_gguf.best_local_model(require_cached=False) is not None,
              "the override must be reversible without a reinstall")

        print("\n-- both dispatch paths consult it, and that is the whole point")
        api_src = open("api/openai_compat.py", encoding="utf-8").read()
        ui_src = open("ui/app.py", encoding="utf-8").read()
        check("the API path goes through local_gguf.available()",
              "local_gguf.available(" in api_src)
        check("the UI path goes through best_local_model()", "best_local_model(" in ui_src)
    finally:
        local_gguf.can_serve = real_can_serve
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
