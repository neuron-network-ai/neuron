"""coordinator/test_dashboard.py — run: python -m coordinator.test_dashboard

The dashboards are the two pages a person actually looks at, and they are built by string
concatenation with no template engine, so nothing catches a mistake in them except this.

Two kinds of property are tested, and the second matters more than the first:

  1. **It is the same product as the landing page.** `docs/index.html` links "Live network
     dashboard →" and the destination used to be unstyled system-grey with Google's palette,
     which read as a different site. The theme now comes from `coordinator/theme.py`.

  2. **Privacy invariants.** The public page is network health only. Node addresses, per-node
     balances and GPU model names are operator- or owner-private ([P11], and the GPU rule from
     the 0.18 notes: a card model is distinctive enough to correlate a roster on). A page built
     by pasting node dicts into f-strings is exactly where one of those leaks, and adding
     columns to it — which is what this change did — is exactly when.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_db = os.path.join(tempfile.mkdtemp(prefix="neuron-dash-"), "t.db")
os.environ["NEURON_DB"] = _db
from coordinator import config                                          # noqa: E402
config.DB_PATH = _db
from coordinator import main, models, theme                             # noqa: E402

ok = fail = 0

ADDRESS = "100.64.99.7"
GPU_NAME = "NVIDIA GeForce RTX 3070"


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def seed(cover_everything):
    """Two eligible nodes, plus a probationary GPU box. When `cover_everything` the pair spans
    the whole model; otherwise they leave 19-27 with no eligible node — the live state that
    made the DEGRADED banner appear in the first place."""
    with models._db() as c:
        c.execute("DELETE FROM nodes")
    hi = 27 if cover_everything else 18
    models.register_node("agent-alpha", "100.64.0.1", 50999, 0, 9, 8, 16, "tok-a",
                         ms_per_layer=41.2, trusted=True)
    models.register_node("agent-beta", "100.64.0.2", 50999, 10, hi, 4, 8, "tok-b",
                         trusted=True)
    models.register_node("agent-newcomer", ADDRESS, 50999, 19, 27, 8, 8, "tok-c",
                         trusted=False, has_gpu=True, gpu_vram_gb=8.0, gpu_name=GPU_NAME)


def main_():
    models.init_db()

    print("\n-- the public dashboard wears the landing page's theme")
    seed(cover_everything=False)
    html = main.dashboard()
    check("page background is the landing page's", theme.BG in html)
    check("brand green is present", theme.GREEN in html)
    check("Google's leftover palette is gone",
          not any(c in html for c in ("#1a73e8", "#f9ab00", "#c5221f", "#137333")),
          "an old Google-console colour survives in the dashboard")
    check("the logo mark is inline (no image request)", "<svg" in html)
    check("no web font is fetched", "fonts.googleapis.com" not in html and
          "fonts.gstatic" not in html,
          "the dashboard refreshes every 5s — a font CDN would be a repeated third-party call")

    print("\n-- privacy: the public page is health only")
    check("no node address", ADDRESS not in html, "a node's endpoint leaked onto /dashboard")
    check("no GPU model name", GPU_NAME not in html,
          "gpu_name is operator-only — distinctive enough to correlate a roster on")
    check("no per-node balance column", "NRN balance" not in html)
    check("GPU presence IS shown, without the model", "GPU" in html)

    print("\n-- a broken chain says WHERE it is broken")
    net, _ = main._network_summary()
    check("the gap is reported as ranges", net["uncovered_layers"] == [[19, 27]],
          str(net["uncovered_layers"]))
    check("the page names the missing layers", "19–27" in html, "")
    check("the banner is degraded", "DEGRADED" in html)
    check("every layer has a cell in the strip",
          html.count("title='layer ") == net["total_layers"],
          str(html.count("title='layer ")))
    check("the missing ones are marked", html.count("class='gap'") == 9,
          str(html.count("class='gap'")))

    print("\n-- a healthy network says so instead")
    seed(cover_everything=True)
    html = main.dashboard()
    net, _ = main._network_summary()
    check("no gaps reported", net["uncovered_layers"] == [], str(net["uncovered_layers"]))
    check("banner is healthy", "HEALTHY" in html and "DEGRADED" not in html)
    check("no cell is marked as a gap", "class='gap'" not in html)

    print("\n-- a probationary node is stated, not just badged")
    # [P24]: "online" and "serving" are different things, and the difference was one word in
    # one table cell.
    check("the page says how many are awaiting verification",
          "awaiting verification" in html, "")
    check("...and what that costs them", "earn no NRN" in html, "")

    print("\n-- the private node dashboard: same theme, own numbers")
    node = models.get_node("agent-newcomer")
    priv = main.node_dashboard("agent-newcomer", token=node["node_token"])
    check("themed like the rest", theme.BG in priv)
    check("shows the owner their OWN address", ADDRESS in priv,
          "the endpoint is private to this page, not secret from its owner")
    check("shows the owner their OWN GPU model", GPU_NAME in priv)
    check("shows a balance", "NRN balance" in priv)
    check("a probationary owner is told why they earn nothing",
          "not yet serving" in priv, "")

    print("\n-- the token in this page's URL cannot leak through a link")
    # The URL *is* the credential (?token=...), so every outbound link posts it to the
    # destination's logs in the Referer header. Both defences must hold.
    check("no third-party link on the page", theme.SITE not in priv,
          "an outbound link would send ?token=... to that host as the Referer")
    check("the browser is told to send no referrer",
          'name="referrer" content="no-referrer"' in priv, "")
    check("the public dashboard, which has no token, keeps its nav",
          theme.SITE in main.dashboard())

    print("\n-- the private page still refuses the wrong token")
    try:
        main.node_dashboard("agent-newcomer", token="wrong")
        refused = False
    except Exception as e:
        refused = getattr(e, "status_code", None) in (401, 403)
    check("a wrong token is rejected", refused)

    print("\n-- a node's standing and proof-of-compute record are not published per node")
    # A flag is the fact most likely to be WRONG about a volunteer's machine: [P37] flagged three
    # honest nodes for the coordinator's own bookkeeping, and on 2026-08-11 `18f1da` reached 1/23
    # entirely on a range the coordinator had moved under it. Publishing `flagged · 4%` beside a
    # named node is a public verdict the network cannot always justify -- and the operator whose
    # machine it is reads it on the same page as everyone else.
    seed(cover_everything=True)
    with models._db() as c:
        c.execute("UPDATE nodes SET challenges_passed=1, challenges_failed=22 "
                  "WHERE node_id='agent-beta'")
    html = main.dashboard()
    beta = models.get_node("agent-beta")
    check("the roster still knows it is flagged", bool(beta.get("flagged")), beta.get("standing"))
    check("the node is still listed", "agent-beta" in html)
    check("but its reputation percentage is not on the public page", "4%" not in html)
    check("nor the raw challenge tally", "(1/23)" not in html and "1/23" not in html)
    check("nor a per-node 'flagged' verdict", ">flagged<" not in html)
    check("the table no longer offers the columns at all",
          "<th>standing</th>" not in html and "<th>proof-of-compute</th>" not in html)
    check("the exclusion is still stated, in aggregate", "excluded from routing" in html)
    check("and the privacy note says why it is missing",
          "standing" in html and "belongs to them" in html)
    # The operator's OWN page must keep every one of those facts -- the point is where they are
    # visible, not whether they exist. A "fix" that also blinded the operator would be worse than
    # the problem: they are the only person who can repair the machine.
    own = main.node_dashboard("agent-beta", token="tok-b")
    own_html = own.body.decode() if hasattr(own, "body") else str(own)
    check("the operator still sees their own standing", "flagged" in own_html)
    check("the operator still sees their own tally",
          "1" in own_html and "22" in own_html)

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main_() else 1)
