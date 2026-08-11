"""coordinator/test_agent_version_report.py — why a node is not on the version we published.

Until 0.20.2 the coordinator was told nothing about the build a node runs. The version string
existed only in the first line of that machine's own log, which nobody can read on a PC behind a
NAT in somebody's house. Two consequences, and the second is the one that matters:

  - a rollout could not be watched — "3 of 4 updated" was unanswerable, by anyone;
  - a **rollback could not be confirmed**. The recovery path built on 2026-08-11 to satisfy
    "we cannot leave users in loss" fired blind: flip the switch, and the only evidence it
    worked was behaviour changing.

A node still on an old build is one of three things, and they are indistinguishable without all
three fields: it has not made its daily check yet, it has `auto_update` off, or its download is
failing. Only the first fixes itself.

Run:  python -m coordinator.test_agent_version_report     (from repo root)
"""
import os
import tempfile
import time

os.environ.setdefault("NEURON_DB", tempfile.mktemp(suffix=".db"))
from coordinator import config, main, models

models.init_db()


def _clear():
    with models._db() as c:
        c.execute("DELETE FROM nodes")


def _reg(node_id="n1", **kw):
    models.register_node(node_id, "1.1.1.1", 50999, 0, 9, 8, 16, f"tok-{node_id}",
                         trusted=True, **kw)
    return models.get_node(node_id)


# --------------------------------------------------------------------------- #
# the round trip
# --------------------------------------------------------------------------- #
def test_a_reported_version_is_stored_and_read_back():
    _clear()
    n = _reg(agent_version="0.20.2", auto_update=True, update_check="current")
    assert n["agent_version"] == "0.20.2"
    assert n["auto_update"] == 1
    assert n["update_check"] == "current"
    assert n["update_checked_at"] and time.time() - n["update_checked_at"] < 60


def test_an_agent_too_old_to_report_reads_as_unknown_not_current():
    """NULL must never be mistaken for "up to date" — that would report a stale fleet as patched,
    which is the exact class of mistake [P34] and [P37] were about."""
    _clear()
    n = _reg()
    assert n["agent_version"] is None
    assert n["auto_update"] is None
    assert n["update_check"] is None


def test_auto_update_false_is_stored_as_false_not_as_missing():
    """`auto_update=False` is the answer to "why has this node never moved". Folding it into NULL
    would erase the one field that distinguishes a stuck node from a slow one."""
    _clear()
    n = _reg(auto_update=False)
    assert n["auto_update"] == 0 and n["auto_update"] is not None


# --------------------------------------------------------------------------- #
# a later registration must not erase what an earlier one told us
# --------------------------------------------------------------------------- #
def test_a_silent_re_registration_does_not_erase_a_known_version():
    """This is the ROLLBACK case, and it is why these are COALESCEd. A node rolled back to a
    pre-0.20.2 build stops reporting entirely; the last known version is real evidence and a
    sudden NULL would destroy it at the exact moment an operator is trying to confirm the
    rollback took."""
    _clear()
    _reg(agent_version="0.20.2", auto_update=True, update_check="current")
    n = _reg()                                    # same node, older build, reports nothing
    assert n["agent_version"] == "0.20.2"
    assert n["auto_update"] == 1


def test_a_new_version_overwrites_the_old_one():
    _clear()
    _reg(agent_version="0.20.1")
    assert _reg(agent_version="0.20.2")["agent_version"] == "0.20.2"


def test_the_verdict_carries_its_own_age():
    """A verdict without an age is a stale field read as live — three times now. `download-failed`
    from last week must be distinguishable from one a minute ago."""
    _clear()
    _reg(update_check="download-failed")
    with models._db() as c:                       # backdate it
        c.execute("UPDATE nodes SET update_checked_at=? WHERE node_id='n1'",
                  (time.time() - 86400,))
    before = models.get_node("n1")["update_checked_at"]
    _reg()                                        # a registration carrying NO verdict
    after = models.get_node("n1")
    assert after["update_check"] == "download-failed", "the verdict survives"
    assert after["update_checked_at"] == before, "and does NOT get a fresh timestamp"


def test_a_new_verdict_restamps_it():
    _clear()
    _reg(update_check="download-failed")
    with models._db() as c:
        c.execute("UPDATE nodes SET update_checked_at=? WHERE node_id='n1'",
                  (time.time() - 86400,))
    n = _reg(update_check="current")
    assert n["update_check"] == "current"
    assert time.time() - n["update_checked_at"] < 60


# --------------------------------------------------------------------------- #
# what each audience sees
# --------------------------------------------------------------------------- #
def test_the_public_page_aggregates_and_names_nobody():
    """Same rule as standing: "is this network patched" is a fair question for a visitor, "which
    volunteer is behind" is not."""
    _clear()
    _reg("up-to-date", agent_version=config.AGENT_VERSION)
    _reg("behind", agent_version="0.19.0")
    html = main.dashboard()
    assert "on the latest agent" in html
    assert "1 of 2" in html
    assert "<th>agent</th>" not in html, "per-node versions do not belong in the public table"


def test_the_public_page_counts_unknown_separately_from_behind():
    """An agent too old to report is not the same as one known to be behind. Merging them would
    invent certainty the coordinator does not have."""
    _clear()
    _reg("silent")                                  # reports nothing
    html = main.dashboard()
    assert "too old to report its version" in html


def test_a_node_with_auto_update_off_is_called_out_in_aggregate():
    _clear()
    _reg("stuck", agent_version="0.19.0", auto_update=False)
    html = main.dashboard()
    assert "auto-update switched off" in html


def test_the_operator_sees_their_own_version_and_why_it_is_behind():
    """The operator is the only person who can act on it — switch auto-update back on, or run
    the installer by hand. The public page aggregates; this page must not."""
    _clear()
    models.register_node("mine", "1.1.1.1", 50999, 0, 9, 8, 16, "tok-mine", trusted=True,
                         agent_version="0.19.0", auto_update=False,
                         update_check="download-failed")
    page = main.node_dashboard("mine", token="tok-mine")
    html = page.body.decode() if hasattr(page, "body") else str(page)
    assert "v0.19.0" in html
    assert f"v{config.AGENT_VERSION} is available" in html
    assert "auto-update is OFF" in html
    assert "download-failed" in html


def test_the_operator_of_a_silent_build_is_told_it_is_unreported_not_current():
    _clear()
    models.register_node("old", "1.1.1.1", 50999, 0, 9, 8, 16, "tok-old", trusted=True)
    page = main.node_dashboard("old", token="tok-old")
    html = page.body.decode() if hasattr(page, "body") else str(page)
    assert "not reported" in html and "does not say" in html


def _run():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
    return True


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run() else 1)
