"""coordinator/test_sybil_signals.py — run: python -m coordinator.test_sybil_signals

Lightweight Sybil signals: record what looks duplicated, block nothing.

The thing most worth testing is the *not blocking*, because that is the part a future reader is
most likely to "fix". A flagged node must still register, still be routable, still earn. The
fingerprint is only CPU count / RAM / OS, so it collides between two identical laptops and lies
freely inside a VM -- enforcement on evidence that weak would lock out honest volunteers to
protect NRN that has no value yet.

The faucet rule is the one real piece of enforcement here, and it is per verified email rather
than per identity: signing in with Google and then GitHub on the same address used to mint two
wallets and two grants.
"""
import os
import sys
import tempfile

os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron-sybil-"), "t.db")

from coordinator import config, genesis, models      # noqa: E402
from coordinator import main as coord                # noqa: E402
from fastapi import HTTPException                    # noqa: E402

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def register(node_id, cores=8, ram_gb=16.0, platform="Windows-11-10.0.26200", port=51000):
    body = coord.RegisterBody(node_id=node_id, tailscale_ip="127.0.0.1", port=port,
                              layer_start=0, layer_end=9, cores=cores, ram_gb=ram_gb,
                              platform=platform)
    return coord.register(body, x_register_secret=None, x_node_token=None)


def flags(kind=None):
    return models.list_sybil_flags(kind=kind)


def main():
    models.init_db()
    genesis.seed_genesis()

    print("\n-- the fingerprint")
    fp = models.hardware_fingerprint(8, 16.0, "Windows-11")
    check("is built from cores, RAM and platform", fp == "8c/16g/Windows-11", fp)
    check("differs when any part differs",
          len({models.hardware_fingerprint(8, 16.0, "Windows-11"),
               models.hardware_fingerprint(4, 16.0, "Windows-11"),
               models.hardware_fingerprint(8, 32.0, "Windows-11"),
               models.hardware_fingerprint(8, 16.0, "Linux-6.8")}) == 4)
    check("is None when a node reported nothing -- absent data must not group nodes",
          models.hardware_fingerprint(None, None, None) is None)
    check("survives partial data", models.hardware_fingerprint(8, None, None) == "8c/?g/?")

    print("\n-- it is stored")
    register("sy-1", port=51001)
    n = models.get_node("sy-1")
    check("the platform is stored on the node", n["platform"] == "Windows-11-10.0.26200")
    check("and so is the fingerprint", n["hw_fingerprint"] == "8c/16g/Windows-11-10.0.26200")
    check("a first registration raises no flag", flags() == [])

    print("\n-- a second node id on the same machine is FLAGGED")
    register("sy-2", port=51002)
    f = flags("fingerprint_reuse")
    check("a flag is recorded", len(f) == 1, str(f))
    check("it names the fingerprint and the new node",
          f and f[0]["subject"] == "8c/16g/Windows-11-10.0.26200" and f[0]["node_id"] == "sy-2")
    check("the detail names the sibling", f and "sy-1" in f[0]["detail"], str(f))

    print("\n-- ...and NOT blocked")
    check("the flagged node registered successfully", models.get_node("sy-2") is not None)
    check("it is eligible, exactly like any other newcomer",
          models.get_node("sy-2")["standing"] == "probationary")
    check("it is not marked flagged (that is the proof-of-compute flag, a different thing)",
          not models.get_node("sy-2").get("flagged"))
    check("its ledger row exists so it can earn",
          models.get_ledger("sy-2") is not None)

    print("\n-- different hardware is not flagged")
    register("sy-3", cores=4, ram_gb=8.0, platform="Linux-6.8", port=51003)
    check("no new flag", len(flags("fingerprint_reuse")) == 1)

    print("\n-- re-registering an existing node is not a new machine")
    before = len(flags("fingerprint_reuse"))
    register("sy-1", port=51001)          # relay ticket refresh / restart
    check("a re-registration of a known node raises nothing",
          len(flags("fingerprint_reuse")) == before)

    print("\n-- a node that reports no hardware is not grouped with other silent nodes")
    register("sy-quiet-1", cores=None, ram_gb=None, platform=None, port=51004)
    register("sy-quiet-2", cores=None, ram_gb=None, platform=None, port=51005)
    check("two nodes with no hardware data raise no flag",
          len(flags("fingerprint_reuse")) == before)

    print("\n-- one faucet per verified email")
    w1, created1 = models.wallet_for_oauth("google", "ext-1", "person@example.com", True)
    check("the first identity is created", created1)
    check("and receives the faucet",
          models.get_ledger(w1)["balance"] == config.FAUCET_AMOUNT_NRN,
          str(models.get_ledger(w1)))

    w2, created2 = models.wallet_for_oauth("github", "ext-2", "person@example.com", True)
    check("the same email on another provider still gets a wallet", created2 and w2 != w1)
    check("but NOT a second faucet grant", models.get_ledger(w2)["balance"] == 0.0,
          str(models.get_ledger(w2)))
    f = flags("faucet_email_reuse")
    check("the withheld grant is flagged for review", len(f) == 1, str(f))
    check("the flag names the email and the new wallet",
          f and f[0]["subject"] == "person@example.com" and f[0]["node_id"] == w2)

    print("\n-- the rule does not punish the innocent")
    w3, _ = models.wallet_for_oauth("google", "ext-3", "other@example.com", True)
    check("a different email still gets its grant",
          models.get_ledger(w3)["balance"] == config.FAUCET_AMOUNT_NRN)
    w4, _ = models.wallet_for_oauth("google", "ext-4", "UNverified@example.com", False)
    check("an UNVERIFIED email is not treated as a claim on that address",
          models.get_ledger(w4)["balance"] == config.FAUCET_AMOUNT_NRN)
    w5, _ = models.wallet_for_oauth("github", "ext-5", "UNverified@example.com", False)
    check("two unverified identities on one address are both funded -- an unverified "
          "address is a claim, not a fact",
          models.get_ledger(w5)["balance"] == config.FAUCET_AMOUNT_NRN)
    check("case does not defeat the check",
          models.email_already_faucet_claimed("PERSON@EXAMPLE.COM"))

    print("\n-- returning users are unaffected")
    again, created = models.wallet_for_oauth("google", "ext-1", "person@example.com", True)
    check("logging back in returns the same wallet", again == w1 and created is False)
    check("and does not re-grant or withhold anything",
          models.get_ledger(w1)["balance"] == config.FAUCET_AMOUNT_NRN)

    print("\n-- the supply survives all of it")
    check("SUM(balance) is still exactly 1,000,000,000",
          models.supply_snapshot()["invariant_ok"],
          str(models.supply_snapshot()["total_supply"]))

    print("\n-- the console is operator-only")
    try:
        coord.admin_sybil_flags(x_wallet_link_secret="wrong")
        check("flags need the operator key", False)
    except HTTPException as e:
        check("flags need the operator key", e.status_code == 401)
    out = coord.admin_sybil_flags(x_wallet_link_secret=config.WALLET_LINK_SECRET)
    check("the operator sees every flag", len(out["flags"]) == len(flags()))
    pub = coord.node_list(x_register_secret=None)["nodes"]
    check("fingerprints are not in the public node list",
          all("hw_fingerprint" not in n and "platform" not in n for n in pub), str(pub[:1]))
    priv = coord.node_list(x_register_secret=config.REGISTRATION_SECRET)["nodes"]
    check("but the operator can see them", any(n.get("hw_fingerprint") for n in priv))
    check("the public dashboard never shows one",
          "8c/16g/" not in coord.dashboard())

    print(f"\n{ok} passed, {fail} failed")
    return fail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
