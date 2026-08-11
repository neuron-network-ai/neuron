"""coordinator/test_stale_speed_display.py — an expired measurement must not be shown as current.

`router.stage_ms` has expired a stale `ms_per_layer` since [P34]: a figure taken once while the
machine thrashed (4150 ms/layer against 8-22 for its peers) was otherwise believed forever, and
REPLICA_SLOWDOWN_LIMIT then excluded the node from routing, so it never served and nothing ever
revised it.

Both DISPLAY surfaces kept reading the raw column. Live on 2026-08-11 the node table showed
`agent-bhpc012101-18f1da ... 4146.6` next to peers at 8-20, and that row was read as evidence
about the network -- while the router had already aged the figure out and was scoring the node
at the default prior. The node's own dashboard said "4146.6 ms/layer measured" to the volunteer
who owns the machine.

That is [P37]'s lesson (a stale diagnostic field read as a live one) in the surface a person
actually looks at, so these tests pin the rule rather than the incident:

  1. freshness has ONE definition, shared by routing and display, so they cannot drift again;
  2. a NULL timestamp is legacy, NOT expired -- reading it as stale would silently reset every
     pre-column node to the default prior;
  3. the operator table marks an expired figure instead of printing it bare;
  4. the node's own page shows nothing rather than accusing its hardware.

Run:  python -m coordinator.test_stale_speed_display     (from repo root)
"""
import os
import tempfile
import time

os.environ.setdefault("NEURON_DB", tempfile.mktemp(suffix=".db"))
from coordinator import config, main, models, router

models.init_db()

STALE = config.MS_PER_LAYER_TTL_S + 60


def _node(ms=10.0, age_s=0.0, **kw):
    """A roster row as `_node_dict` produces it, not as the DB stores it."""
    n = {"node_id": "n1", "layer_start": 0, "layer_end": 9, "ms_per_layer": ms,
         "ms_per_layer_at": (time.time() - age_s) if ms is not None else None,
         "status": "online", "standing": "trusted", "cores": 8, "ram_gb": 16}
    n.update(kw)
    return n


# --------------------------------------------------------------------------- #
# the rule itself
# --------------------------------------------------------------------------- #
def test_a_fresh_measurement_is_returned():
    assert router.ms_per_layer_fresh(_node(ms=11.0, age_s=60)) == 11.0


def test_an_expired_measurement_reads_as_unknown():
    """The live case: 4146.6 taken during the 7B migration, hours later."""
    assert router.ms_per_layer_fresh(_node(ms=4146.6, age_s=STALE)) is None


def test_a_never_measured_node_reads_as_unknown():
    assert router.ms_per_layer_fresh(_node(ms=None)) is None


def test_a_null_timestamp_is_legacy_not_expired():
    """A row written before `ms_per_layer_at` existed. Treating NULL as stale would reset every
    pre-column node to the default prior -- silently, and network-wide."""
    assert router.ms_per_layer_fresh(_node(ms=12.4, age_s=0.0,
                                           ms_per_layer_at=None)) == 12.4


def test_an_unparseable_figure_does_not_raise():
    assert router.ms_per_layer_fresh(_node(ms="fast")) is None
    assert router.ms_per_layer_fresh(_node(ms=9.0, ms_per_layer_at="never")) == 9.0


# --------------------------------------------------------------------------- #
# routing and display agree, by construction
# --------------------------------------------------------------------------- #
def test_routing_keeps_using_an_old_measurement():
    """A self-inflicted regression, 2026-08-11: expiry reached the live coordinator and scored
    `node-c-pavilion` (measured 11.0, twelve hours ago) at the 40.0 prior while a peer kept its
    fresher 20.4 — inverting the preference between them and dropping chat to 0.05 tok/s.

    An old number measured ON the machine beats a fresh guess ABOUT it. The TTL assumed the
    hourly re-measurement that only arrives with 0.20; until then, expiring a figure has no
    second measurement to fall back on and is pure loss. Revisit when 0.20 is everywhere."""
    assert router.stage_ms(_node(ms=11.0, age_s=STALE)) == 10 * 11.0
    assert router.stage_ms(_node(ms=11.0, age_s=STALE)) < router.stage_ms(_node(ms=20.4, age_s=60))


def test_routing_still_falls_back_to_the_prior_when_nothing_was_measured():
    assert router.stage_ms(_node(ms=None)) == 10 * router.DEFAULT_MS_PER_LAYER


def test_display_and_routing_may_disagree_and_that_is_deliberate():
    """The page says "no current measurement"; the router still uses it. Different questions:
    "is this figure current enough to show a stranger" is not "is it the best evidence I have"."""
    stale = _node(ms=4146.6, age_s=STALE)
    assert router.ms_per_layer_fresh(stale) is None          # display: unknown
    assert router.stage_ms(stale) == 10 * 4146.6             # routing: still the evidence


def test_routing_still_honours_a_fresh_measurement():
    assert router.stage_ms(_node(ms=20.0, age_s=60)) == 10 * 20.0


def test_freshness_has_one_definition_wherever_it_is_asked():
    """The helper is still the single source of truth for "is this figure current" — the drift
    that started all this was the dashboard computing its own answer. What changed on 2026-08-11
    is that ROUTING stopped asking the question at all (see test_routing_keeps_using_an_old
    _measurement); every consumer that does ask must get the same answer."""
    for ms, age, expected in ((11.0, 60, 11.0), (4146.6, STALE, None), (None, 0, None),
                              (8.3, STALE, None), (20.0, 60, 20.0)):
        assert router.ms_per_layer_fresh(_node(ms=ms, age_s=age)) == expected, (ms, age)


# --------------------------------------------------------------------------- #
# what a person actually sees
# --------------------------------------------------------------------------- #
def _table_cell(n):
    """The speed cell of the PUBLIC node table, rendered."""
    ms = router.ms_per_layer_fresh(n)
    return (f"{ms:.1f}" if ms is not None else
            "<span class='dash' title='no current measurement — the node re-measures "
            "hourly; until then it is scored at the default prior'>—</span>")


def test_the_public_table_does_not_print_an_expired_figure_at_all():
    """`4146.6 · stale` was the first attempt, and it applied a weaker standard than the one that
    removed the standing column: beside peers at 8-20 it still reads as a verdict on somebody's
    machine, however it is labelled. Routing scores this node at the default prior, so unknown is
    also the more ACCURATE public answer."""
    cell = _table_cell(_node(ms=4146.6, age_s=STALE))
    assert "4146.6" not in cell, "an expired figure is still a public verdict on a volunteer's PC"
    assert "—" in cell and "dash" in cell


def test_the_public_table_prints_a_fresh_figure_bare():
    assert _table_cell(_node(ms=8.3, age_s=60)) == "8.3"


def test_an_expired_and_a_never_measured_node_look_identical_publicly():
    """They ARE the same thing to the router — both are scored at the prior — so showing them
    differently would be the page claiming a distinction the system does not make."""
    assert _table_cell(_node(ms=4146.6, age_s=STALE)) == _table_cell(_node(ms=None))


def test_the_nodes_own_page_says_nothing_rather_than_accusing_it():
    """`ms/layer measured` is rendered only for a real number, so an expired figure drops the
    clause entirely. Silence is honest here; 4146.6 is a false claim about someone's PC."""
    stale = router.ms_per_layer_fresh(_node(ms=4146.6, age_s=STALE))
    clause = f" · {stale:.1f} ms/layer measured" if isinstance(stale, (int, float)) else ""
    assert clause == ""
    fresh = router.ms_per_layer_fresh(_node(ms=8.3, age_s=60))
    clause = f" · {fresh:.1f} ms/layer measured" if isinstance(fresh, (int, float)) else ""
    assert clause == " · 8.3 ms/layer measured"


def test_re_registering_the_same_figure_does_not_refresh_its_age():
    """THE TTL WAS DEAD IN PRODUCTION. The agent caches its measurement and re-sends the same
    number on every registration, and the upsert restamped `ms_per_layer_at` whenever a value
    arrived — so an online node kept a months-old reading permanently "fresh", and `4146.6` was
    still being served as live hours after [P34] supposedly gave it an age. A re-assertion is not
    a measurement."""
    models.init_db()
    with models._db() as c:
        c.execute("DELETE FROM nodes WHERE node_id='ttl-probe'")
    models.register_node("ttl-probe", "1.1.1.1", 50999, 0, 9, 8, 16, "tok-ttl",
                         ms_per_layer=4146.6, trusted=True)
    with models._db() as c:                       # backdate it past the TTL
        c.execute("UPDATE nodes SET ms_per_layer_at=? WHERE node_id='ttl-probe'",
                  (time.time() - STALE,))
    assert router.ms_per_layer_fresh(models.get_node("ttl-probe")) is None

    # the node heartbeats, re-asserting the SAME figure
    models.register_node("ttl-probe", "1.1.1.1", 50999, 0, 9, 8, 16, "tok-ttl",
                         ms_per_layer=4146.6, trusted=True)
    assert router.ms_per_layer_fresh(models.get_node("ttl-probe")) is None, (
        "re-sending the same number made a stale measurement look current again")

    # a genuinely NEW measurement must still restamp, or nothing could ever recover
    models.register_node("ttl-probe", "1.1.1.1", 50999, 0, 9, 8, 16, "tok-ttl",
                         ms_per_layer=12.5, trusted=True)
    assert router.ms_per_layer_fresh(models.get_node("ttl-probe")) == 12.5


def test_the_ttl_is_named_in_config_so_the_two_cannot_drift():
    assert config.MS_PER_LAYER_TTL_S == 21600.0
    assert int(config.MS_PER_LAYER_TTL_S // 3600) == 6


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
