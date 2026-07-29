"""
NEURON — shared driver  [Session 11]

The node_a "driver" role, factored out so both the Chat UI (ui/app.py) and the
OpenAI-compatible API (api/openai_compat.py) reuse ONE implementation instead of
each carrying its own copy of the loop.

It loads the driver shard (embed + layers 0..S1-1 + lm_head) ONCE per process and,
for each prompt, asks the coordinator for a live chain, runs the autoregressive
loop across node_a -> node_c -> node_b, and yields structured events:

    {"type": "meta",  "request_id", "node_ids", "nodes", "cost_nrn"}   # cost_nrn = held quote
    {"type": "token", "text": <decoded delta>}          # zero or more
    {"type": "done",  "completion_tokens", "prompt_tokens", "finish_reason",
                      "latency_ms", "tok_per_s", "text": <full decoded output>,
                      "cost_nrn"}                        # cost_nrn = real settled cost
    {"type": "error", "detail": <str>, "code": <str, optional>}   # instead of token/done

Callers turn these into whatever wire format they need (SSE for the browser,
OpenAI chunks for the API). Reuses node_a.coord_get_chain / coord_complete and
common's stage primitives — nothing in common.py / node_*.py is modified.
"""
import logging
import os
import socket
import threading
import time

import torch

import common
import junction_cache
import wire_codec
import node_a  # coord_get_chain / coord_complete (its main() is __main__-guarded)
from safety import moderation

log = logging.getLogger("neuron.driver")

# node_a owns layers 0..S1-1; MAX cap guards against runaway generations.
S1 = int(os.environ.get("NEURON_S1", "10"))
MAX_TOKENS_CAP = int(os.environ.get("NEURON_MAX_TOKENS", "512"))

# How many times one request may survive a node dying before we give up. Each reroute costs a
# coordinator round trip plus a replay of the tokens so far, so this is not free -- but on a
# volunteer network a single retry is not enough either (the replacement can be somebody
# else's sleeping laptop). 3 is the point where "the chain is genuinely broken" is the more
# likely explanation than bad luck.
MAX_REROUTES = int(os.environ.get("NEURON_MAX_REROUTES", "3"))


def log_chain_failure(node_ids, err):
    """A node dropping mid-generation used to be invisible unless the user complained
    ([P14] noted the same gap in ui/app.py). It is the single most common failure on a
    volunteer network, so it gets a log line naming the chain that was in flight."""
    log.warning("chain failed mid-generation (nodes=%s): %s: %s",
                ",".join(node_ids or []), err.__class__.__name__, err)


class _Driver:
    def __init__(self):
        self.model = self.tok = self.n = self.eos_id = None
        self.compute_lock = threading.Lock()   # serialise node_a's own compute
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def ensure_loaded(self):
        with self._load_lock:
            if self.model is None:
                print(f"[driver] loading shard (embed + layers 0..{S1 - 1} + lm_head) ...")
                t0 = time.time()
                tok, model, n = common.load_model_shard(0, S1, embed=True, head=True)
                self._finish_load(tok, model, n, t0, common.MODEL_ID)

    def load_from_slice(self, slice_dir):
        """Alternate loader for a byte-range-downloaded SLICE directory (agent/local_chat.py)
        instead of the full local HF cache ensure_loaded() reads from -- same driver role
        (embed + layers 0..S1-1 + lm_head), same fixed S1, just a lighter on-disk footprint
        so a personal agent install doesn't need the whole model just to run its own Chat UI.
        The slice must have been downloaded with is_first_node=True (slice_downloader),
        which for a tied-lm_head model pulls in everything this role needs."""
        with self._load_lock:
            if self.model is None:
                from transformers import AutoConfig, AutoTokenizer

                import slice_downloader
                print(f"[driver] loading personal driver slice from {slice_dir} ...")
                t0 = time.time()
                model = slice_downloader.load_slice_model(slice_dir)
                tok = AutoTokenizer.from_pretrained(slice_dir)
                n = AutoConfig.from_pretrained(slice_dir).num_hidden_layers
                self._finish_load(tok, model, n, t0, f"slice:{slice_dir}")

    def _finish_load(self, tok, model, n, t0, source):
        self.tok, self.model, self.n, self.eos_id = tok, model, n, tok.eos_token_id
        print(f"[driver] ready in {time.time() - t0:.1f}s | {source} | "
              f"layers={n} | owns 0..{S1 - 1}")

    # -- input helpers (caller picks chat-template vs raw text) --------------- #
    def encode_chat(self, messages):
        return self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt")

    def encode_text(self, prompt: str):
        return self.tok(prompt, return_tensors="pt").input_ids

    # -- the generation loop -------------------------------------------------- #
    def stream(self, input_ids, max_new: int, coordinator: str, router_prompt: str,
              wallet_id: str = None):
        """Yield event dicts (see module docstring). `router_prompt` is only what
        the coordinator logs for the request; generation is driven by input_ids.
        `wallet_id` (Workstream B) is who pays -- /infer holds the worst-case cost from it
        before dispatch; a missing/underfunded wallet surfaces as an "insufficient_funds"
        error event rather than a raw coordinator 402."""
        model, tok, eos_id = self.model, self.tok, self.eos_id
        max_new = max(1, min(int(max_new), MAX_TOKENS_CAP))
        prompt_tokens = int(input_ids.shape[1])
        t_start = time.time()

        # 1) ask the coordinator for a chain matching our shard (layers 0..S1-1)
        try:
            (host_c, port_c, host_b, port_b, s2, node_ids, request_id, complete_token,
             hold_amount) = node_a.coord_get_chain(coordinator, router_prompt, max_new, S1,
                                                   wallet_id, prompt_tokens_estimate=prompt_tokens)
        except node_a.InsufficientFunds as e:
            yield {"type": "error", "detail": str(e), "code": "insufficient_funds"}
            return
        except Exception as e:
            yield {"type": "error", "detail": str(e)}
            return

        yield {"type": "meta", "request_id": request_id, "node_ids": node_ids,
               "nodes": len(node_ids), "cost_nrn": hold_amount}

        # 2) open + configure the chain (node_a -> node_c -> node_b)
        sock = None
        # State that a mid-generation reroute has to be able to replace. `chain` is rebound by
        # _reroute(), so `step` closes over the names rather than the values.
        chain = {"host_c": host_c, "port_c": port_c, "host_b": host_b, "port_b": port_b,
                 "s2": s2, "node_ids": node_ids, "request_id": request_id,
                 "complete_token": complete_token, "hold_amount": hold_amount,
                 "tokens_billed_elsewhere": 0}
        jcache = junction_cache.JunctionCache()
        reroutes = []

        def _connect():
            """Open + configure a connection to the current chain. Returns (sock, codec)."""
            s = socket.create_connection((chain["host_c"], chain["port_c"]),
                                        timeout=common.COLD_CONNECT_TIMEOUT_S)
            # The config goes out in the legacy format -- it is the one message whose reader
            # might predate wire_codec -- and offers the codecs we can decode. The ack names
            # the peer's pick, or omits it, in which case codec stays None and this
            # connection keeps using the legacy format for the whole request.
            common.send_msg(s, {"type": "config", "s1": S1, "s2": chain["s2"],
                                "host_b": chain["host_b"], "port_b": chain["port_b"],
                                "wire": wire_codec.preference(model.config.hidden_size)})
            a = common.recv_msg(s)
            if not a.get("ok"):
                s.close()
                raise RuntimeError(f"node_c refused config: {a}")
            c = wire_codec.negotiate([a["wire"]] if a.get("wire") else None)
            s.settimeout(common.HOT_TIMEOUT_S)
            return s, c

        try:
            sock, codec = _connect()
        except Exception as e:
            yield {"type": "error", "detail": f"{e.__class__.__name__}: {e}"}
            return

        cache, past = common.new_cache(), 0

        def _settle_current(tokens_now):
            """Close the books on the chain we are abandoning, so the nodes that DID serve
            the first part of this answer get paid and the unused part of the hold is
            refunded. Uses only the existing /complete endpoint, and [P12]'s rule still
            holds: the coordinator settles from the plan IT recorded, not from anything we
            claim here."""
            billable = max(tokens_now - chain["tokens_billed_elsewhere"], 0)
            node_a.coord_complete(coordinator, chain["request_id"], billable,
                                  int((time.time() - t_start) * 1000), chain["node_ids"],
                                  chain["complete_token"], prompt_tokens=prompt_tokens)

        def _reroute(tokens_now, why):
            """A node in the chain died. Get a different chain and bring it up to date from
            the junction cache, rather than losing the whole generation."""
            nonlocal sock, codec
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
                sock = None
            if not jcache.recoverable():
                # Cache overflowed (a very long answer): a replacement cannot be given the
                # real history, and replaying a truncated one would silently corrupt the
                # answer. Fail honestly instead.
                raise RuntimeError(f"{why}; cannot recover (junction cache incomplete after "
                                   f"{jcache.tokens} tokens)")
            _settle_current(tokens_now)
            (hc, pc, hb, pb, s2b, nids, rid, ctok, hold) = node_a.coord_get_chain(
                coordinator, router_prompt, max_new, S1, wallet_id,
                prompt_tokens_estimate=prompt_tokens)
            chain.update(host_c=hc, port_c=pc, host_b=hb, port_b=pb, s2=s2b, node_ids=nids,
                         request_id=rid, complete_token=ctok, hold_amount=hold,
                         tokens_billed_elsewhere=tokens_now)
            sock, codec = _connect()
            # Rebuild the fresh chain's K/V from the one junction we cache. Sent as a single
            # block, which for causal attention is identical to replaying each token in turn
            # but costs one round trip instead of N.
            replay = jcache.replay_block()
            common.send_msg(sock, {"type": "act", "hidden": replay}, codec=codec)
            resp = common.recv_msg(sock)
            reroutes.append({"reason": why, "at_token": tokens_now,
                             "replayed_tokens": int(replay.shape[1]), "node_ids": nids})
            return resp

        # Errors that mean "this peer is gone", as opposed to a bug in our own code. Kept in
        # sync with node_c/node_b's handle(): TimeoutError is a sibling of ConnectionError
        # under OSError, not caught by it.
        DEAD_PEER = (ConnectionError, TimeoutError, EOFError, OSError)

        def step(token_block, tokens_now):
            nonlocal past
            with self.compute_lock:
                h1 = common.first_stage(model, S1, token_block, cache, past)
            past += token_block.shape[1]
            # Cached BEFORE the send, so a failure on this very hop is still recoverable --
            # the replay block includes the activation the dead node never answered.
            jcache.add(h1)
            last_err = None
            for attempt in range(MAX_REROUTES + 1):
                try:
                    if attempt == 0:
                        common.send_msg(sock, {"type": "act", "hidden": h1}, codec=codec)
                        resp = common.recv_msg(sock)
                    else:
                        resp = _reroute(tokens_now, f"{last_err.__class__.__name__}: {last_err}")
                    with self.compute_lock:
                        return int(common.apply_lm_head(model, resp["hidden"]).argmax(-1))
                except DEAD_PEER as e:
                    last_err = e
                    log_chain_failure(chain["node_ids"], e)
            raise RuntimeError(f"chain failed {MAX_REROUTES + 1}x, last: {last_err}")

        try:
            # 3) autoregressive loop, emitting each new piece of decoded text
            produced, prev_text, completion, finish = [], "", 0, "length"
            tok_id = step(input_ids, 0)
            while True:
                if tok_id == eos_id:
                    finish = "stop"
                    break
                produced.append(tok_id)
                completion += 1
                full = tok.decode(produced, skip_special_tokens=True)
                delta = full[len(prev_text):]
                prev_text = full
                if delta:
                    # Output moderation gate (Workstream A) — checked against the FULL
                    # accumulated text every token, not just the new delta, so a phrase
                    # split across a token-decode boundary is still caught. This is a
                    # cheap regex scan (not a classifier call), so per-token cost is
                    # negligible; checking every token also aborts as early as possible,
                    # minimizing how much of a blocked response ever reaches the user.
                    verdict = moderation.check_text(full)
                    if verdict.blocked:
                        moderation.log_event("out", verdict.category, chain["request_id"],
                                             snippet=full)
                        moderation.report_violation(coordinator, wallet_id, "out", verdict.category)
                        common.send_msg(sock, {"type": "bye"})
                        # deliberately NOT calling coord_complete() -- an aborted, policy-
                        # blocked generation must not be reported/billed as a completion.
                        yield {"type": "error",
                              "detail": "This response was blocked by NEURON's acceptable-use "
                                        "policy (see SAFETY.md).",
                              "code": "content_policy_violation"}
                        return
                    yield {"type": "token", "text": delta}
                if completion >= max_new:
                    finish = "length"
                    break
                tok_id = step(torch.tensor([[tok_id]]), completion)

            common.send_msg(sock, {"type": "bye"})

            # 4) report completion so the coordinator settles the hold (Workstream B: real
            #    metered cost, node payouts, refund of anything unused). node_a.py counts
            #    len(generated), which includes the stopping token.
            #    After a reroute this settles only the tokens THIS chain served; the earlier
            #    chain was already settled for its own share by _settle_current(), so the
            #    nodes that served the first half of the answer still got paid.
            latency_ms = int((time.time() - t_start) * 1000)
            tokens_generated = completion + (1 if finish == "stop" else 0)
            billable = max(tokens_generated - chain["tokens_billed_elsewhere"], 0)
            result = node_a.coord_complete(coordinator, chain["request_id"], billable,
                                           latency_ms, chain["node_ids"],
                                           chain["complete_token"],
                                           prompt_tokens=prompt_tokens)
            rewards = (result or {}).get("rewards") or {}
            hold_now = chain["hold_amount"]
            actual_cost = (round(hold_now - rewards.get("__refund__", 0.0), 6)
                          if hold_now is not None else None)
            done = {"type": "done", "completion_tokens": completion,
                    "prompt_tokens": prompt_tokens, "finish_reason": finish,
                    "latency_ms": latency_ms,
                    "tok_per_s": round(completion / max(time.time() - t_start, 1e-6), 2),
                    "text": prev_text, "cost_nrn": actual_cost}
            if reroutes:
                # Surfaced rather than hidden: a recovered answer is still a degraded one, and
                # whoever is debugging a slow reply needs to know a node died mid-generation.
                done["reroutes"] = reroutes
                done["final_request_id"] = chain["request_id"]
            yield done
        except Exception as e:
            yield {"type": "error", "detail": f"{e.__class__.__name__}: {e}"}
        finally:
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass


# one shared driver per process (single model load, shared by UI + API)
DRIVER = _Driver()
