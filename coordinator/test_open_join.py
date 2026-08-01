"""Open-join tests (Session 12) — run: python -m coordinator.test_open_join

Uses a throwaway DB and calls the real endpoint/router/ledger/models functions directly
(no HTTP server needed). Proves: open registration yields a PROBATIONARY node that is
excluded from routing and earns nothing; presenting the secret yields a TRUSTED node that
routes and earns; a proof-of-compute pass promotes a probationary node to eligible; a
secret-less registration cannot hijack a trusted node id; OPEN_JOIN=0 restores the 401;
and the migration grandfathers pre-open-join nodes in as trusted.
"""
import os
import sqlite3
import tempfile

os.environ["NEURON_OPEN_JOIN"] = "1"
os.environ["NEURON_PROBATION_MIN_PASSES"] = "1"
_tmp = tempfile.mkdtemp(prefix="neuron_oj_")
os.environ["NEURON_DB"] = os.path.join(_tmp, "oj.db")

from fastapi import HTTPException  # noqa: E402

import relay_auth  # noqa: E402
from coordinator import config, ledger, models, router  # noqa: E402
from coordinator.main import RegisterBody, register  # noqa: E402

SECRET = config.REGISTRATION_SECRET
ok = 0
fail = 0


def check(name, cond):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name}")
    else:
        fail += 1
        print(f"  FAIL  {name}")


def body(node_id, ls, le, behind_nat=False):
    return RegisterBody(node_id=node_id, tailscale_ip="127.0.0.1", port=50000 + le,
                        layer_start=ls, layer_end=le, cores=4, ram_gb=8, behind_nat=behind_nat)


_req_n = [0]


def earn(node_ids):
    """ledger.distribute() no longer exists (fixed-supply settle() replaced unconditional
    minting) -- this test only cares about eligibility gating (probationary/verified earn or
    don't), not real economics, so fund __escrow__ directly as a test-only shortcut rather
    than going through a real hold. See coordinator/test_wallet_settlement.py for the
    invariant-preserving hold->settle lifecycle tests."""
    _req_n[0] += 1
    models.credit(config.ESCROW_LEDGER_ID, 1.0)
    plan_nodes = [models.get_node(nid) for nid in node_ids]
    return ledger.settle(f"oj-req-{_req_n[0]}", "oj-test-wallet", 1.0,
                         prompt_tokens=0, completion_tokens=1000, plan_nodes=plan_nodes)


def main():
    models.init_db()
    N = config.TOTAL_LAYERS  # 28

    # 1) trusted registration (with secret) + probationary registration (no secret)
    r_trust = register(body("trust-a", 0, 13), x_register_secret=SECRET)
    r_prob = register(body("stranger-b", 14, N - 1), x_register_secret=None)
    check("secret -> trusted standing", r_trust["standing"] == "trusted")
    check("no secret -> probationary standing", r_prob["standing"] == "probationary")
    check("probationary response carries a note", "note" in r_prob)

    a = models.get_node("trust-a")
    b = models.get_node("stranger-b")
    check("trusted node is eligible", a["eligible"] and a["standing"] == "trusted")
    check("probationary node not eligible", (not b["eligible"]) and b["standing"] == "probationary")

    # 2) routing excludes the probationary node -> chain is missing its layers
    chain, missing = router.build_chain()
    check("chain incomplete while stranger probationary", len(missing) > 0)
    check("probationary node absent from chain",
          "stranger-b" not in [n["node_id"] for n in chain])

    # 3) earning excludes the probationary node
    earn(["trust-a", "stranger-b"])
    check("trusted node earned", models.get_ledger("trust-a")["balance"] > 0)
    check("probationary node earned nothing", models.get_ledger("stranger-b")["balance"] == 0)

    # 4) proof-of-compute pass promotes the stranger
    models.record_attestation("stranger-b", True)
    b2 = models.get_node("stranger-b")
    check("passed PoC -> verified + eligible", b2["eligible"] and b2["standing"] == "verified")
    chain2, missing2 = router.build_chain()
    check("chain complete after verification", len(missing2) == 0)
    check("verified node now in chain", "stranger-b" in [n["node_id"] for n in chain2])
    earn(["trust-a", "stranger-b"])
    check("verified node now earns", models.get_ledger("stranger-b")["balance"] > 0)

    # 5) a secret-less registration cannot hijack a trusted node id
    try:
        register(body("trust-a", 0, 13), x_register_secret=None)
        check("hijack of trusted id blocked (409)", False)
    except HTTPException as e:
        check("hijack of trusted id blocked (409)", e.status_code == 409)

    # 6) OPEN_JOIN=0 restores the 401 for secret-less registration
    config.OPEN_JOIN = False
    try:
        register(body("stranger-c", 0, 5), x_register_secret=None)
        check("private mode rejects no-secret (401)", False)
    except HTTPException as e:
        check("private mode rejects no-secret (401)", e.status_code == 401)
    finally:
        config.OPEN_JOIN = True

    # 7) flagged node (failed PoC) is excluded even though it is 'verified' by passes
    # (regression guard on the eligibility predicate)
    for _ in range(config.REPUTATION_MIN_SAMPLES + 1):
        models.record_attestation("stranger-b", False)
    b3 = models.get_node("stranger-b")
    check("many failures -> flagged + not eligible", b3["flagged"] and not b3["eligible"])

    # 8) a secret-less registration cannot hijack a VERIFIED (not just trusted) node id — the
    # earlier guard only checked `trusted`, leaving open-joined-and-verified strangers exposed
    # to identity theft (post-launch-audit fix). The real owner CAN still recover by presenting
    # that exact node's current token, proving they already control it.
    register(body("stranger-v", 0, 5), x_register_secret=None)
    models.record_attestation("stranger-v", True)
    v = models.get_node("stranger-v")
    check("setup: stranger-v is verified", v["standing"] == "verified")
    try:
        register(body("stranger-v", 0, 5), x_register_secret=None)
        check("hijack of verified id blocked (409)", False)
    except HTTPException as e:
        check("hijack of verified id blocked (409)", e.status_code == 409)
    try:
        register(body("stranger-v", 0, 5), x_register_secret=None, x_node_token="not-the-token")
        check("hijack with a wrong token blocked (409)", False)
    except HTTPException as e:
        check("hijack with a wrong token blocked (409)", e.status_code == 409)
    r_owner = register(body("stranger-v", 0, 5), x_register_secret=None,
                       x_node_token=v["node_token"])
    check("re-register with own token succeeds", r_owner["status"] == "registered")
    # The response must report the node's REAL standing, not re-derive it from "did this call
    # carry the secret". stranger-v is verified; telling it "probationary — you will not earn
    # NRN" on every restart (which a relayed node does routinely) is a lie that would push a
    # stranger to uninstall.
    check("re-registering a VERIFIED node still reports verified",
          r_owner["standing"] == "verified")
    check("...and carries no 'you will not earn' note", "note" not in r_owner)

    # 9) a behind-NAT registration is handed a relay ticket the relay can verify offline
    r_nat = register(body("stranger-nat", 0, 5, behind_nat=True), x_register_secret=None)
    relay_block = r_nat["relay"]
    check("relay block carries a ticket", "ticket" in relay_block)
    check("relay ticket verifies for the right node/port", relay_auth.verify_ticket(
        config.RELAY_SECRET, "stranger-nat", relay_block["public_port"], relay_block["ticket"]))
    check("relay ticket rejected for a different node id", not relay_auth.verify_ticket(
        config.RELAY_SECRET, "someone-else", relay_block["public_port"], relay_block["ticket"]))

    migration_test()

    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)


def migration_test():
    """A DB created before open join (no `trusted` column) must grandfather its nodes
    in as trusted when init_db() migrates it."""
    path = os.path.join(_tmp, "legacy.db")
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE nodes (
            node_id TEXT PRIMARY KEY, tailscale_ip TEXT NOT NULL, port INTEGER NOT NULL,
            layer_start INTEGER NOT NULL, layer_end INTEGER NOT NULL, cores INTEGER, ram_gb REAL,
            ms_per_layer REAL, head_ms REAL,
            challenges_passed INTEGER NOT NULL DEFAULT 0, challenges_failed INTEGER NOT NULL DEFAULT 0,
            node_token TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'online',
            last_seen REAL NOT NULL, registered_at REAL NOT NULL);
        INSERT INTO nodes (node_id, tailscale_ip, port, layer_start, layer_end, node_token,
                           last_seen, registered_at)
        VALUES ('legacy-node', '127.0.0.1', 50999, 0, 27, 'tok', 9e18, 0);
    """)
    conn.commit()
    conn.close()

    old_db = config.DB_PATH
    config.DB_PATH = path
    try:
        models.init_db()
        n = models.get_node("legacy-node")
        check("legacy node grandfathered as trusted", n["trusted"] and n["standing"] == "trusted")
    finally:
        config.DB_PATH = old_db


if __name__ == "__main__":
    main()
