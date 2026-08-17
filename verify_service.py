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

# Consecutive cycles a node may fail to COMPLETE a challenge before that counts as a failure.
# Deliberately higher than FAIL_STRIKES: a wrong answer is unambiguous, whereas an unfinished
# one is usually a restart, a cold shard or a relay hiccup, and none of those deserve a mark.
# But "inconclusive" cannot mean "forgiven forever" -- a node running the wrong weights fails
# by closing the socket, not by answering wrong, so it accumulated nothing and kept its perfect
# reputation while breaking every request it touched (live 2026-08-10, [P33]).
UNREACHABLE_STRIKES = 5

# Whether a FAILED challenge against the stage-1 node counts against its reputation.
#
# Stage-1 nodes were skipped entirely until 2026-08-17, on the stated grounds that the middle
# probe "computes layers without the embedding a first-stage node applies". That was flagged in
# the code as a hypothesis, and it is wrong: `node_server`'s probe role never embeds either --
# it runs `common.mid_stage(model, self.lo, self.hi + 1, hidden)`, which is exactly what
# `make_middle_challenge` computes. Measured end to end against a real NodeServer on the real
# 0-9 slice: max_err 0. Exactly zero, not merely inside atol.
#
# The skip's cost was not a gap in reputation. Emission pays only on a passing challenge
# (`models.mark_slot_poc`), so a node that is never challenged can never earn an availability
# hour -- and the driver is the machine [P43] shows carries the largest fixed cost on the
# network. 81 unearnable hours on `agent-optinovate-6ff49d`, ~243 NRN, found by [P40]'s
# reconciliation.
#
# So stage-1 nodes are challenged now. Their FAILURES are still not scored, and the asymmetry is
# deliberate -- the same shape `drifted` already uses, for a sharper reason. A pass is real
# evidence: it proves the node computes its own range correctly, and recording it is what makes
# the driver payable. A failure would flag the one machine holding stage 1, and a flagged driver
# is not a degraded network, it is no network at all. This path has never run against a live
# driver, and an unproven check must not be able to take the network down on its first day.
# Flip this to True once the live driver has been seen passing -- one deliberate line, with
# evidence, which is what the original skip was written to insist on.
#
# The history the old comment carried, kept because it is the reason for the caution above.
# The stage-1 path had NEVER EXECUTED: a whole-tuple ack comparison raised PLACEMENT MISMATCH
# first, every sweep, and hid it -- the driver acked `s2` and omitted `s1`, so `(None, 10) !=
# (0, 10)`. Fixing that comparison on 2026-08-11 pointed a never-run check at the driver, which
# came back `max_err 28.6, strike 1 of 3`, and the skip was added rather than risk three of
# those flagging it. 28.6 was read as "the right answer to a different question" and explicitly
# labelled a hypothesis. It is still a hypothesis: what is now measured is only that TODAY's
# node_server, on a correct slice, answers the probe exactly, and that a node challenged on a
# range it does not hold raises RangeMismatch (refused, nothing recorded) rather than returning
# a wrong-looking answer -- both checked end to end, 2026-08-17. The 2026-08-11 driver was
# running a pre-0.20 agent whose ack omitted `s1`, so a range disagreement could pass unnoticed
# into `verify()`; both live nodes now report `s1` and `holds`. That is a good explanation, not
# a proven one, which is exactly why failures stay unscored until the live driver is seen to
# pass.
STAGE1_FAILURES_ARE_SCORED = False

# Reading the roster is retried WITHIN a cycle. A home connection produces 502s and DNS
# failures routinely — the last three lines this service ever logged were one 502 and two
# "Failed to resolve neuronnet.duckdns.org" ([P24]) — and one bad read used to cost a whole
# cycle's promotions.
ROSTER_ATTEMPTS = 3
ROSTER_BACKOFF_S = 2

# After this many consecutive cycles unable to reach the coordinator, the log says ERROR
# rather than WARNING: at that point the verifier is running but blind, and no node anywhere
# can be promoted.
UNREACHABLE_ESCALATE = 5

# Sweeps between "still alive" lines. A healthy verifier used to log NOTHING at all — the
# "nothing to verify" line is debug-level — so its log looked identical whether it was
# running or had been dead for two days. That is exactly how [P24] went unnoticed. With a
# heartbeat line, silence means dead.
ALIVE_EVERY = 30

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
    # The console handler must never be able to kill this service. Windows gives it cp1252, and
    # a single non-ASCII character in a message -- the "→" in the VERIFIED line -- raised
    # UnicodeEncodeError from inside logging and took the verifier down (observed 2026-08-10,
    # it restarted at 08:50 with a traceback in its own log). A monitor that dies on the shape
    # of its own success message is worse than no monitor. The FILE handler is already utf-8;
    # this makes the stream one degrade to "?" instead of raising.
    stream = logging.StreamHandler()
    try:
        stream.stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    for h in (logging.FileHandler(LOG_PATH, encoding="utf-8"), stream):
        h.setFormatter(fmt)
        log.addHandler(h)


class Verifier:
    def __init__(self, coordinator, secret, interval=DEFAULT_INTERVAL):
        self.base = coordinator.rstrip("/")
        self.secret = secret
        self.interval = interval
        self.strikes = {}          # node_id -> consecutive wrong answers
        self.unreachable = 0       # consecutive cycles that could not read the roster
        self.last_roster = None    # (nodes, probationary) from the last successful sweep
        # node_id -> when it was last challenged, so re-verification rotates oldest-first
        # instead of hammering whichever node happens to sort first.
        self.last_checked = {}
        # node_id -> consecutive cycles where the challenge could not even be COMPLETED.
        # Distinct from `strikes` (wrong answers) on purpose: one failed connection is a hiccup,
        # a hundred is a broken node.
        self.unreachable_strikes = {}
        # node_id -> already warned that we cannot challenge it. The warning is worth exactly one
        # line per node per process: an unverifiable node is a standing condition, not an event.
        self.unchallengeable = set()
        # Building a challenge means loading that layer range with torch, which costs seconds
        # and hundreds of MB. Without this cache a 60s loop would reload the same shard every
        # minute forever; nodes cluster on a handful of ranges, so the cache is tiny.
        self._challenges = {}      # (lo, hi, is_last) -> (input, expected)

    # -- coordinator ---------------------------------------------------------- #
    def _headers(self):
        return {"X-Register-Secret": self.secret}

    def nodes(self):
        """The node roster, retried with backoff.

        A transient 502 or DNS failure is the normal weather on a home connection, not a
        reason to skip a cycle — every cycle skipped is a stranger sitting at zero NRN for
        another minute. Raises the last error only after ROSTER_ATTEMPTS have failed.
        """
        last = None
        for attempt in range(ROSTER_ATTEMPTS):
            try:
                r = requests.get(f"{self.base}/node/list", headers=self._headers(), timeout=15)
                r.raise_for_status()
                return r.json()["nodes"]
            except (requests.RequestException, KeyError, ValueError) as e:
                last = e
                if attempt + 1 < ROSTER_ATTEMPTS:
                    time.sleep(ROSTER_BACKOFF_S * 2 ** attempt)
        raise last

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
        except (requests.RequestException, KeyError, ValueError) as e:
            self.unreachable += 1
            # Escalates rather than repeating the same WARNING forever: one failed read is
            # weather, five in a row is an outage during which nothing can be promoted, and
            # the log should not make those look the same ([P24]).
            say = log.error if self.unreachable >= UNREACHABLE_ESCALATE else log.warning
            say("coordinator unreachable (%d cycle(s) in a row, %d attempts each): %s — "
                "no node can be verified or promoted while this lasts",
                self.unreachable, ROSTER_ATTEMPTS, e)
            return 0
        if self.unreachable:
            log.info("coordinator reachable again after %d failed cycle(s)", self.unreachable)
            self.unreachable = 0
        if nodes and "tailscale_ip" not in nodes[0]:
            log.error("coordinator did not return node addresses — the register secret is "
                      "wrong, so nothing can be verified")
            return 0

        total = self.total_layers()
        pending = [n for n in nodes
                   if n.get("standing") == "probationary" and not n.get("flagged")
                   and n.get("status") == "online"]

        # RE-verification. Proof-of-compute was a one-time GATE: pass once and never be asked
        # again. So a node that later goes WRONG -- reloaded onto a different model, a corrupted
        # slice, failing memory -- kept serving bad activations indefinitely, and the only
        # symptom the network showed was a dropped socket the driver blamed on a node "dying".
        # Live 2026-08-10: a node left on 7B weights inside a 1.5B chain failed every request it
        # touched for hours while this verifier logged "0 awaiting verification" on every sweep.
        #
        # A node is only as trustworthy as its last check. Re-check the already-verified on a
        # slow rotation: one per cycle, oldest-checked first, so the cost is one challenge per
        # minute across the whole network no matter how large it grows. A wrong answer takes the
        # normal FAIL_STRIKES path -- the reputation machinery already excludes a flagged node
        # from routing, so this needs no new enforcement, only the evidence it was never given.
        #
        # FLAGGED NODES ARE ON THIS ROTATION TOO, and that is the whole difference between a
        # reputation and a death sentence. `flagged` is DERIVED, every sweep, from cumulative
        # counters (models.py: total >= REPUTATION_MIN_SAMPLES and passed/total < THRESHOLD) --
        # so it is arithmetically clearable. It was never clearable in practice, because this
        # filter excluded flagged nodes from BOTH lists: a flagged node was never challenged
        # again, so its counters could never move, so the flag could never lift. No endpoint
        # reset it either. The only exit was DELETE /node/{id}, which destroys the node's
        # identity and token and makes a volunteer reinstall.
        #
        # Live 2026-08-10, and it cost the whole network: three nodes were flagged for a
        # PLACEMENT bug (challenged on layers they did not hold), the verifier then had nothing
        # left it was allowed to check, logged "5 node(s), 0 awaiting verification" for hours,
        # and the chain collapsed to one eligible machine holding all 28 layers -- covered, and
        # unroutable. A permanent terminal state reached by accident is [P24]'s failure class:
        # the mechanism was working perfectly and had been given nothing to work on.
        due = [n for n in nodes
               if n.get("status") == "online"
               and n.get("standing") in ("verified", "trusted", "flagged")]
        due.sort(key=lambda n: self.last_checked.get(n["node_id"], 0.0))
        recheck = due[:1] if due else []

        self.last_roster = (len(nodes), len(pending))
        if not pending and not recheck:
            log.debug("nothing to verify")
            return 0
        pending = pending + recheck

        promoted = 0
        for n in pending:
            nid = n["node_id"]
            # THE COORDINATOR'S OWN BOOKKEEPING IS NOT EVIDENCE ABOUT THE NODE.
            #
            # `RangeMismatch` below already refuses to punish a node that TELLS us it holds
            # something else. But [P37]'s three machines mostly did not get to say so: asked for
            # layers they never downloaded, the forward pass raised on uninitialized meta tensors,
            # `_handle` caught only (ConnectionError, TimeoutError, EOFError), the thread died and
            # `finally: conn.close()` slammed the socket. So drift arrives here as `socket closed
            # mid-message` -- the generic branch -- and after UNREACHABLE_STRIKES cycles it is
            # attested as a real failure. Live 2026-08-11: `agent-bhpc012101-18f1da` went 1/18 to
            # 1/22 in twenty minutes, on a range the coordinator had moved under it.
            #
            # `placement_drift` predicted all three on 2026-08-10 and was read by nothing ([P37]).
            # It is read here now: while it is set, a FAILURE says our range and this node's slice
            # disagree, which is a fact about the coordinator, not about the volunteer's PC.
            #
            # Deliberately ASYMMETRIC -- drift suppresses failures, never passes. A pass under
            # drift is real evidence (it proves the node does hold the range we assigned, so the
            # `reported_*` field is the stale one) and recording it is what lets a wrongly flagged
            # node climb back. The cost is that a node could dodge strikes by registering a range
            # it was not given; that is visible as `placement_drift` on the dashboard, and it is
            # the right side to err on when the alternative already cost three honest machines
            # their standing and the network its routability.
            drifted = bool(n.get("placement_drift"))

            # STAGE 1 IS CHALLENGED, AND ITS FAILURES ARE NOT SCORED. See
            # STAGE1_FAILURES_ARE_SCORED above for the measurement that retired the old skip and
            # for why the asymmetry is the right side to err on. `unscored` widens `drifted`'s
            # existing meaning -- "a bad answer from this node is not evidence about this node"
            # -- rather than adding a second parallel mechanism to keep in step with it.
            is_stage1 = int(n["layer_start"]) == 0 and int(n["layer_end"]) != total - 1
            if is_stage1 and nid not in self.unchallengeable:
                self.unchallengeable.add(nid)
                log.info("%s (stage 1, layers %d-%d) is being challenged with the middle probe. "
                         "A pass is recorded and unlocks its availability emission; a failure "
                         "is logged and NOT recorded, because flagging the driver would take "
                         "the network down and this path has not yet run against a live one.",
                         nid, int(n["layer_start"]), int(n["layer_end"]))
            # From here on the two reasons a failure means nothing are handled as one.
            unscored = drifted or (is_stage1 and not STAGE1_FAILURES_ARE_SCORED)

            try:
                res = self.challenge(n, total)
            except proof_of_compute.RangeMismatch as e:
                # THE NODE IS FINE; THE PLACEMENT IS STALE. It answered, it told the truth about
                # what it holds, and it holds something other than what the coordinator says.
                # Recording that against its reputation is recording our own bookkeeping error
                # in its file -- which is precisely what flagged three honest machines on
                # 2026-08-10 ([P32] drift: assigned 10-27 / holds 14-27, assigned 19-27 / holds
                # 10-13, assigned 10-27 / holds 0-27). So: no attestation, in either direction,
                # and no strike. `last_checked` IS stamped, or the rotation would return to this
                # same node every cycle and never reach the others.
                self.last_checked[nid] = time.time()
                self.unreachable_strikes.pop(nid, None)
                log.error("%s: PLACEMENT MISMATCH, not a bad node — %s. Nothing recorded "
                          "against it. The coordinator's range and this node's slice disagree; "
                          "fix placement (./coordinator/pin_layers.sh --driver) rather than the "
                          "node.", nid, e)
                continue
            except proof_of_compute.ChallengeRefused as e:
                # A named "no" — paused by its owner, or mid-reload. Both are a node behaving
                # correctly, and neither is evidence of anything. Not a strike: the typed
                # refusal exists exactly so this stops looking like a machine that died.
                self.last_checked[nid] = time.time()
                self.unreachable_strikes.pop(nid, None)
                log.info("%s: declined the challenge (%s) — healthy, nothing recorded", nid, e)
                continue
            except Exception as e:
                # Could not even get an answer (offline mid-sweep, relay hiccup, cold shard).
                # That is not evidence of cheating, so it must NOT be attested as a failure.
                # A single failure is not evidence of cheating and must not be attested as one.
                # But treating it as inconclusive FOREVER means a node that fails by hanging up
                # is never penalised at all -- and "socket closed mid-message" is exactly how a
                # node running the wrong weights fails. Live 2026-08-10: one node failed every
                # challenge this way for hours while its reputation stayed perfect and routing
                # kept sending it real traffic. Persistent inability to answer IS a fact about
                # the node, so after enough consecutive cycles it counts.
                # STAMP IT EVEN THOUGH IT FAILED. The re-check rotation is `due` sorted by
                # `last_checked` with a default of 0.0, one node per cycle -- so a node that is
                # never stamped stays permanently at the front and every other node is starved
                # off the rotation. Live 2026-08-11: `18f1da` was challenged four times in twenty
                # minutes (1/18 → 1/22) while `82cbee` sat at 2/4, never re-checked, one pass away
                # from clearing its flag. [P35]'s whole point was putting flagged nodes back on
                # this rotation so they could recover; an unstamped failure quietly monopolised it
                # and denied exactly that. The attempt is what the rotation measures, not the
                # outcome -- `unreachable_strikes` already counts the consecutive failures.
                self.last_checked[nid] = time.time()
                u = self.unreachable_strikes.get(nid, 0) + 1
                self.unreachable_strikes[nid] = u
                if u < UNREACHABLE_STRIKES:
                    log.warning("%s: could not challenge (%s: %s) — will retry next cycle "
                                "(%d of %d)", nid, e.__class__.__name__, e, u,
                                UNREACHABLE_STRIKES)
                    continue
                self.unreachable_strikes[nid] = 0
                if unscored:
                    # The hang-up IS the drift, arriving as an exception instead of a typed
                    # RangeMismatch. Recording it would put our stale range in the node's file.
                    self.last_checked[nid] = time.time()
                    why = ("placement_drift is set, so this is the coordinator's range "
                           "disagreeing with the node's slice, not a bad machine. Fix placement "
                           "(./coordinator/pin_layers.sh --driver)" if drifted else
                           "this is the stage-1 node and its failures are not scored yet "
                           "(STAGE1_FAILURES_ARE_SCORED) — a flagged driver is no network at all")
                    log.error("%s: could not answer in %d consecutive cycles (%s) — NOTHING "
                              "RECORDED: %s — until then no challenge against this node can "
                              "mean anything.", nid, u, e.__class__.__name__, why)
                    continue
                log.error("%s: could not complete a challenge in %d consecutive cycles (%s) — "
                          "recording a failure. A node that cannot answer cannot serve.",
                          nid, u, e.__class__.__name__)
                try:
                    self.attest(nid, False, None)
                except requests.RequestException as ae:
                    log.warning("%s: could not record the failure: %s", nid, ae)
                continue
            self.last_checked[nid] = time.time()
            self.unreachable_strikes.pop(nid, None)   # it answered; the streak is broken
            if res["passed"]:
                self.strikes.pop(nid, None)
                was = n.get("standing")
                # EVERY pass is recorded now, including a re-check's. It used to `continue`
                # before attesting, so re-verification could only ever move a node's ratio
                # DOWN: a failure was written to challenges_failed, a pass was written nowhere.
                # A machine that has one bad afternoon in a year of good ones therefore drifts
                # towards the flag and can never earn its way back. The reason passes were
                # skipped was log noise, which is a logging problem -- so the LINE stays at
                # DEBUG and the COUNTER is written.
                try:
                    out = self.attest(nid, True, res["max_err"])
                except requests.RequestException as e:
                    log.warning("%s: passed but could not record attestation: %s", nid, e)
                    continue
                if was in ("verified", "trusted"):
                    # a re-check, not a promotion. DEBUG so a healthy network stays quiet -- an
                    # INFO line per minute per node is the heartbeat mistake again (Session 55
                    # half seven).
                    log.debug("%s re-verified — max_err %.2e, %dms", nid, res["max_err"],
                              res["ms"])
                    continue
                if was == "flagged":
                    # Worth a line at INFO, unlike the others: a flag lifting is the network
                    # recovering a node it had written off, and until this release it could not
                    # happen at all.
                    log.info("%s passed while FLAGGED — reputation now %s, standing '%s' "
                             "(max_err %.2e, %dms)", nid, out.get("reputation"),
                             out.get("standing", "?"), res["max_err"], res["ms"])
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
                if unscored:
                    # Pavilion's `max_err 33.79` was deterministic across attempts and across
                    # verifier restarts, because it was the right answer to a different question:
                    # it computed 10-18 with no final norm while the verifier compared against
                    # 10-27 + norm ([P37]). A wrong answer under drift is the expected result of
                    # asking the wrong question, so it is not evidence either.
                    #
                    # A stage-1 wrong answer is unscored for a different reason: the 28.6 of
                    # 2026-08-11 was never explained, only hypothesised about, and the machine
                    # it would flag is the one holding stage 1.
                    self.strikes[nid] = 0
                    why = ("placement_drift is set, so the node is being asked about layers the "
                           "coordinator only believes it holds. Fix placement first."
                           if drifted else
                           "this is the stage-1 node. A wrong answer here is worth "
                           "INVESTIGATING (it is the 28.6 of 2026-08-11 recurring) but not "
                           "scoring — flagging the driver would take the whole network down.")
                    log.error("%s: wrong answer (max_err %.4g) — NOTHING RECORDED: %s",
                              nid, res["max_err"], why)
                    continue
                try:
                    self.attest(nid, False, res["max_err"])
                except requests.RequestException as e:
                    log.warning("%s: failed but could not record attestation: %s", nid, e)
                    continue
                log.error("%s FAILED proof-of-compute %d times (max_err %.4g) — recorded; "
                          "repeated failures exclude it from routing", nid, s, res["max_err"])
        return promoted

    def _alive_line(self):
        if self.unreachable:
            return f"coordinator unreachable for {self.unreachable} cycle(s)"
        if self.last_roster is None:
            return "no roster read yet"
        total, pending = self.last_roster
        return f"{total} node(s), {pending} awaiting verification"

    def run(self):
        log.info("verifier started | coordinator=%s | every %ds | alive line every %d cycles "
                 "| log=%s", self.base, self.interval, ALIVE_EVERY, LOG_PATH)
        cycles = 0
        while True:
            cycles += 1
            try:
                self.sweep()
            except Exception as e:                     # never let one bad cycle kill the loop
                log.exception("sweep failed: %s", e)
            # The liveness heartbeat. Without it a dead verifier and an idle one write exactly
            # the same thing — nothing — and [P24] is what that costs: the log's last line was
            # two days old and there was no way to tell which of the two it meant.
            if cycles % ALIVE_EVERY == 0:
                log.info("alive | %d sweeps | %s", cycles, self._alive_line())
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
