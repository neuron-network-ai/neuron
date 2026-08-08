"""agent/test_startup_is_never_silent.py — run: python -m agent.test_startup_is_never_silent

v0.18.0 was installed on a real machine and produced **no log file at all**, and a 404 nobody
could explain. The missing log is the worse half: a run that leaves no trace is
indistinguishable from a run that never happened, and there was nothing the owner could send.
PROBLEMS.md [P24].

The cause was ordering. `_setup_logging()` ran inside `main()` *after* the config was read,
because the log level comes from the config — so anything that failed before that point (a
module-level import, an unreadable config, a config key this build expects and the previous
one never wrote) died in silence. In the packaged tray app it is worse still: the console is
hidden before the imports even begin, so stderr has nowhere to go either.

So these tests are about one property: **if the agent dies during startup, the machine's owner
has a file to send.** Not "logging is configured" — it was, just too late to matter.
"""
import json
import logging
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _reset_logging():
    parent = logging.getLogger("neuron")
    for h in list(parent.handlers):
        parent.removeHandler(h)
        h.close()


def main():
    from agent import agent as mod
    tmp = tempfile.mkdtemp(prefix="neuron-startup-")

    print("\n-- crash_log writes without any logging configuration at all")
    _reset_logging()
    path = os.path.join(tmp, "agent.log")
    mod.LOG_PATH = path
    try:
        raise RuntimeError("boom-during-import")
    except RuntimeError:
        mod.crash_log("the agent could not import its own modules")
    body = read(path)
    check("the reason is in the file", "could not import its own modules" in body, body)
    check("the traceback is in the file", "boom-during-import" in body, body)
    check("it is marked as a crash", "[CRASH]" in body, body)

    print("\n-- crash_log never raises, whatever the filesystem does")
    mod.LOG_PATH = os.path.join(tmp, "no-such-dir", "agent.log")
    try:
        mod.crash_log("unwritable")
        raised = None
    except Exception as e:                       # the whole point: it must swallow this
        raised = e
    check("an unwritable log path is not a second crash", raised is None, str(raised))
    mod.LOG_PATH = path

    print("\n-- main() configures logging BEFORE it reads the config")
    # The regression this file exists for. A config main() cannot parse must produce a log
    # entry naming it -- previously it produced nothing, because logging was set up two dozen
    # lines further down using a value from the very file that failed to load.
    _reset_logging()
    bad_cfg = os.path.join(tmp, "broken.json")
    with open(bad_cfg, "w") as f:
        f.write("{ this is not json")
    open(path, "w").close()                      # truncate: only this run's output counts
    argv = sys.argv
    try:
        sys.argv = ["agent.py", "--config", bad_cfg]
        try:
            mod.main()
            died = False
        except (ValueError, json.JSONDecodeError):
            died = True
    finally:
        sys.argv = argv
    body = read(path)
    check("an unreadable config still raises", died)
    check("...and says so in agent.log", "could not read the config" in body, body)
    check("...naming the file", bad_cfg in body, body)

    print("\n-- a config from an older build does not take the agent down")
    # [P24]'s second suspect: 0.18 installed over 0.17 reads 0.17's config. A key this build
    # expects and that one never wrote used to be a KeyError in Agent.__init__ -- before any
    # log line existed.
    _reset_logging()
    old_cfg = os.path.join(tmp, "v017.json")
    with open(old_cfg, "w") as f:                # no coordinator, no auto_update, no oauth
        json.dump({"node_id": "agent-old", "log_level": "INFO"}, f)
    a = mod.Agent(config_path=old_cfg)
    check("a config with no 'coordinator' key still constructs an Agent",
          a.base == mod.DEFAULT_CONFIG["coordinator"].rstrip("/"), a.base)

    print("\n-- a config write cannot leave a half-written file behind")
    # The most plausible mechanism found for [P24]'s "no log at all". `_save` used to be
    # json.dump(cfg, open(path, "w")) -- the open truncates immediately and the handle was
    # never explicitly closed, so anything that stopped the process mid-write (an OS shutdown,
    # a task kill, installing over a running agent) left a truncated config.json. The agent
    # then died before logging existed, on EVERY start, forever.
    _reset_logging()
    live = os.path.join(tmp, "live.json")
    with open(live, "w") as f:
        json.dump({"coordinator": "https://example.test", "node_id": "agent-keepme",
                   "node_token": "tok-keepme", "log_level": "INFO"}, f)
    a = mod.Agent(config_path=live)
    a.cfg["layer_start"] = 3
    a._save()
    check("no temp file is left behind", not os.path.exists(live + ".tmp"))
    check("the previous copy is kept", os.path.exists(live + ".prev"))
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.py"),
               encoding="utf-8").read()
    check("the real file is replaced atomically, never truncated in place",
          "os.replace(tmp, self.config_path)" in src and 'open(self.config_path, "w")' not in src)
    check("...and the bytes are on disk before the rename", "os.fsync(f.fileno())" in src)

    print("\n-- a config that is ALREADY corrupt does not cost the owner their identity")
    # node_id and node_token ARE the node's claim on everything it has earned. Regenerating
    # them silently would orphan the balance and rejoin as a brand-new node.
    with open(live, "w") as f:
        f.write('{"coordinator": "https://exa')          # truncated, as an interrupted write
    recovered = mod.load_config(live)
    check("the previous copy is used", recovered.get("node_id") == "agent-keepme",
          str(recovered))
    check("the node keeps its token", recovered.get("node_token") == "tok-keepme")
    check("the broken file is kept as evidence", os.path.exists(live + ".corrupt"))
    check("the repaired config is written back", mod.load_config(live).get("node_id")
          == "agent-keepme")
    # With no .prev there is nothing safe to do, and inventing a fresh identity is NOT safe.
    lone = os.path.join(tmp, "lone.json")
    with open(lone, "w") as f:
        f.write("{ truncated")
    try:
        mod.load_config(lone)
        raised = False
    except (ValueError, OSError):
        raised = True
    check("with no backup it raises rather than inventing a new identity", raised)

    print("\n-- the frozen entry point records a crash it cannot log any other way")
    # Tray mode hides the console before importing torch. If that import fails there is no
    # console, no logging config and no window -- so the entry point writes the file itself.
    entry = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "packaging", "neuron_app_entry.py")
    probe = (
        "import sys; sys.path.insert(0, %r)\n"
        "import importlib.util as u\n"
        "spec = u.spec_from_file_location('entry', %r); m = u.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "m._log_path = lambda: %r\n"
        "try:\n"
        "    raise RuntimeError('tray-import-exploded')\n"
        "except RuntimeError:\n"
        "    m._record_crash('the NEURON app failed to start')\n"
    ) % (os.path.dirname(os.path.dirname(os.path.abspath(__file__))), entry,
         os.path.join(tmp, "entry.log"))
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    entry_log = os.path.join(tmp, "entry.log")
    wrote = os.path.exists(entry_log) and "tray-import-exploded" in read(entry_log)
    check("the entry point's crash reaches agent.log", wrote,
          r.stdout + r.stderr + (read(entry_log) if os.path.exists(entry_log) else "<no file>"))

    print("\n-- the entry point computes the log path without importing the agent")
    # It must not import agent.agent to find out where to write: the import that failed is
    # exactly the thing being reported.
    src = read(entry)
    check("no agent.agent import at module level in neuron_app_entry",
          "from agent.agent import" not in src.split("def _main(")[0], "")

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
