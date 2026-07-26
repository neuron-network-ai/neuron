"""Private per-node dashboard tests — run: python -m coordinator.test_node_dashboard

Earnings are private: the public /dashboard must show NO per-node balances; /ledger/{id} and
/node/{id}/dashboard require that node's OWN token (401 otherwise). Uses a throwaway DB and
calls the endpoint functions directly — no HTTP server.
"""
import os
import tempfile

os.environ["NEURON_OPEN_JOIN"] = "1"
os.environ["NEURON_DB"] = os.path.join(tempfile.mkdtemp(prefix="neuron_dash_"), "d.db")

from fastapi import HTTPException  # noqa: E402

from coordinator import config, models  # noqa: E402
from coordinator.main import (RegisterBody, dashboard, get_ledger, node_dashboard,  # noqa: E402
                              register)

SECRET = config.REGISTRATION_SECRET
ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def main():
    models.init_db()
    r1 = register(RegisterBody(node_id="mine", tailscale_ip="127.0.0.1", port=50001,
                               layer_start=0, layer_end=13, cores=4, ram_gb=8),
                  x_register_secret=SECRET)
    register(RegisterBody(node_id="other", tailscale_ip="127.0.0.1", port=50002,
                          layer_start=14, layer_end=27, cores=4, ram_gb=8),
             x_register_secret=SECRET)
    tok = r1["node_token"]
    models.credit("mine", 7.777)

    # ---- /ledger is private to the node ----
    led = get_ledger("mine", x_node_token=tok)
    check("own token reads own ledger", led["balance"] == 7.777)
    for name, bad in (("missing token -> 401", None), ("wrong token -> 401", "nope")):
        try:
            get_ledger("mine", x_node_token=bad)
            check(name, False)
        except HTTPException as e:
            check(name, e.status_code == 401)
    try:
        get_ledger("ghost", x_node_token="x")
        check("unknown node -> 404", False)
    except HTTPException as e:
        check("unknown node -> 404", e.status_code == 404)

    # ---- per-node dashboard: token via query (browser) or header ----
    html = node_dashboard("mine", token=tok)
    check("my dashboard shows my balance", "7.777" in html and "mine" in html)
    check("my dashboard shows served/earned cards", "total earned" in html and "requests served" in html)
    html2 = node_dashboard("mine", token=None, x_node_token=tok)
    check("header auth also accepted", "7.777" in html2)
    try:
        node_dashboard("mine", token="wrong")
        check("dashboard wrong token -> 401", False)
    except HTTPException as e:
        check("dashboard wrong token -> 401", e.status_code == 401)
    try:
        node_dashboard("mine", token=r1 and "")   # empty
        check("dashboard empty token -> 401", False)
    except HTTPException as e:
        check("dashboard empty token -> 401", e.status_code == 401)

    # ---- the PUBLIC dashboard leaks no balances ----
    pub = dashboard()
    check("public dashboard has no balance column", "NRN balance" not in pub)
    check("public dashboard has no per-node earnings value", "7.777" not in pub)
    check("public dashboard still shows nodes + standing", "mine" in pub and "standing" in pub)
    check("public dashboard aggregate cards intact", "NRN distributed" in pub)

    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
