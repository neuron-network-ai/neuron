"""
verify_service.py — promote joining nodes automatically, forever.

A stranger who installs the agent joins **probationary**: reachable and challengeable, but
excluded from routing and earning until a verifier confirms their node actually computes its
layers correctly (proof-of-compute, Session 16/17). Until this service existed, that
confirmation was a command the founder ran by hand — so a stranger who joined at 3am sat at
zero NRN until somebody noticed them. A network whose onboarding requires the operator to be
awake is a demo.

    python verify_service.py                 # run forever, 60s cycle
    python verify_service.py --once          # one sweep, for testing
    python verify_service.py --interval 30

Needs the operator's NEURON_REGISTER_SECRET (node addresses are private, and /attest is
secret-gated) and PyTorch — it recomputes each node's layers locally to compare. That is why
this is the OPERATOR's service and not something a stranger's agent runs.

Logs to verify_service.log next to this file.
"""
import argparse
import logging
import os
import sys
import time

import requests

from security import proof_of_compute

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "verify_service.log")
DEFAULT_COORDINATOR = "https://neuronnet.duckdns.org"
DEFAULT_INTERVAL = 60

# A node is only marked FAILED after this many consecutive bad answers. One wrong reply is far
# more often a dropped connection, a node mid-restart or a cold shard than a cheater, and a
# failed attestation counts against reputation permanently — at REPUTATION_THRESHOLD 0.6 it
# takes only a few to exclude an honest machine from the network for good. Passing is
# attested immediately; only condemnation waits for proof.
FAIL_STRIKES = 3

log = logging.getLogger("neuron.verifier")


def load_secret():
    """Operator secret from the environment, falling back to the gitignored .env.coordinator.

    The fallback exists because of how this actually gets run: auto-start (a Windows Run key
    or a systemd user unit) launches it with a bare environment, so an env-var-only lookup
    means the service starts at boot, finds nothing, and exits — the failure mode this whole
    service was written to eliminate.
    """
    val = os.environ.get("NEURON_REGISTER_SECRET")
    if val:
        return val
    env_file = os.path.join(HERE, ".env.coordinator")
    try:
        with open(env_file) as f:
            for line in f:
                if line.startswith("NEURON_REGISTER_SECRET="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return None


def _setup_logging(level="INFO"):
    log.setLevel(getattr(logging, level, logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)


class Verifier:
    def __init__(self, coordinator, secret, interval=DEFAULT_INTERVAL):
        self.base = coordinator.rstrip("/")
        self.secret = secret
        self.interval = interval
        self.strikes = {}          # node_id -> consecutive wrong answers
        # Building a challenge means loading that layer range with torch, which costs seconds
        # and hundreds of MB. Without this cache a 60s loop would reload the same shard every
        # minute forever; nodes cluster on a handful of ranges, so the cache is tiny.
        self._challenges = {}      # (lo, hi, is_last) -> (input, expected)

    # -- coordinator ---------------------------------------------------------- #
    def _headers(self):
        return {"X-Register-Secret": self.secret}

    def nodes(self):
        r = requests.get(f"{self.base}/node/list", headers=self._headers(), timeout=15)
        r.raise_for_status()
        return r.json()["nodes"]

    def total_layers(self):
        try:
            r = requests.get(f"{self.base}/network/model", timeout=15)
            r.raise_for_status()
            return int(r.json()["serving"]["layers"])
        except (requests.RequestException, KeyError, ValueError, TypeError):
            return 28

    def attest(self, node_id, passed, max_err):
        r = requests.post(f"{self.base}/node/{node_id}/attest",
                          json={"passed": passed, "max_err": max_err},
                          headers=self._headers(), timeout=15)
        r.raise_for_status()
        return r.json()

    # -- challenge ------------------------------------------------------------ #
    def challenge(self, node, total):
        """Run proof-of-compute against one node. Returns the result dict from
        security.proof_of_compute, whose `passed` is a real answer-correctness verdict."""
        lo, hi = int(node["layer_start"]), int(node["layer_end"])
        is_last = hi == total - 1
        key = (lo, hi, is_last)
        if key not in self._challenges:
            self._challenges[key] = (proof_of_compute.make_challenge(lo, total) if is_last
                                     else proof_of_compute.make_middle_challenge(lo, hi + 1))
        inp, expected = self._challenges[key]
        host, port = node["tailscale_ip"], node["port"]
        t0 = time.time()
        if is_last:
            out = proof_of_compute.challenge_node(host, port, lo, total, inp)
        else:
            out = proof_of_compute.challenge_middle_node(host, port, lo, hi + 1, inp)
        ok, err = proof_of_compute.verify(out, expected)
        return {"passed": ok, "max_err": round(err, 6), "ms": int((time.time() - t0) * 1000)}

    # -- one sweep ------------------------------------------------------------ #
    def sweep(self):
        try:
            nodes = self.nodes()
        except requests.RequestException as e:
            log.warning("coordinator unreachable: %s", e)
            return 0
        if nodes and "tailscale_ip" not in nodes[0]:
            log.error("coordinator did not return node addresses — the register secret is "
                      "wrong, so nothing can be verified")
            return 0

        total = self.total_layers()
        pending = [n for n in nodes
                   if n.get("standing") == "probationary" and not n.get("flagged")
                   and n.get("status") == "online"]
        if not pending:
            log.debug("nothing to verify")
            return 0

        promoted = 0
        for n in pending:
            nid = n["node_id"]
            try:
                res = self.challenge(n, total)
            except Exception as e:
                # Could not even get an answer (offline mid-sweep, relay hiccup, cold shard).
                # That is not evidence of cheating, so it must NOT be attested as a failure.
                log.warning("%s: could not challenge (%s: %s) — will retry next cycle",
                            nid, e.__class__.__name__, e)
                continue
            if res["passed"]:
                self.strikes.pop(nid, None)
                try:
                    out = self.attest(nid, True, res["max_err"])
                except requests.RequestException as e:
                    log.warning("%s: passed but could not record attestation: %s", nid, e)
                    continue
                promoted += 1
                log.info("%s VERIFIED — layers %d-%d, max_err %.2e, %dms → standing now '%s'",
                         nid, n["layer_start"], n["layer_end"], res["max_err"], res["ms"],
                         out.get("standing", "verified"))
            else:
                self.strikes[nid] = self.strikes.get(nid, 0) + 1
                s = self.strikes[nid]
                if s < FAIL_STRIKES:
                    log.warning("%s: wrong answer (max_err %.4g), strike %d of %d — not "
                                "recorded yet", nid, res["max_err"], s, FAIL_STRIKES)
                    continue
                try:
                    self.attest(nid, False, res["max_err"])
                except requests.RequestException as e:
                    log.warning("%s: failed but could not record attestation: %s", nid, e)
                    continue
                log.error("%s FAILED proof-of-compute %d times (max_err %.4g) — recorded; "
                          "repeated failures exclude it from routing", nid, s, res["max_err"])
        return promoted

    def run(self):
        log.info("verifier started | coordinator=%s | every %ds | log=%s",
                 self.base, self.interval, LOG_PATH)
        while True:
            try:
                self.sweep()
            except Exception as e:                     # never let one bad cycle kill the loop
                log.exception("sweep failed: %s", e)
            time.sleep(self.interval)


def main():
    ap = argparse.ArgumentParser(description="Continuously verify probationary NEURON nodes.")
    ap.add_argument("--coordinator", default=os.environ.get("NEURON_COORDINATOR",
                                                            DEFAULT_COORDINATOR))
    ap.add_argument("--register-secret", default=None)
    ap.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    ap.add_argument("--once", action="store_true", help="run a single sweep and exit")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    _setup_logging(args.log_level)
    secret = args.register_secret or load_secret()
    if not secret:
        log.error("no register secret — set NEURON_REGISTER_SECRET, pass --register-secret, or "
                  "put it in .env.coordinator. Node addresses are operator-private and /attest "
                  "is secret-gated, so the verifier cannot work without it.")
        return 2
    v = Verifier(args.coordinator, secret, args.interval)
    if args.once:
        n = v.sweep()
        log.info("single sweep done — %d node(s) promoted", n)
        return 0
    v.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
