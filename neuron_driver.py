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
import os
import socket
import threading
import time

import torch

import common
import node_a  # coord_get_chain / coord_complete (its main() is __main__-guarded)
from safety import moderation

# node_a owns layers 0..S1-1; MAX cap guards against runaway generations.
S1 = int(os.environ.get("NEURON_S1", "10"))
MAX_TOKENS_CAP = int(os.environ.get("NEURON_MAX_TOKENS", "512"))


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
                self.tok, self.model, self.n, self.eos_id = tok, model, n, tok.eos_token_id
                print(f"[driver] ready in {time.time() - t0:.1f}s | {common.MODEL_ID} | "
                      f"layers={n} | A owns 0..{S1 - 1}")

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
        try:
            sock = socket.create_connection((host_c, port_c), timeout=common.COLD_CONNECT_TIMEOUT_S)
            common.send_msg(sock, {"type": "config", "s1": S1, "s2": s2,
                                   "host_b": host_b, "port_b": port_b})
            ack = common.recv_msg(sock)
            if not ack.get("ok"):
                yield {"type": "error", "detail": f"node_c refused config: {ack}"}
                return
            sock.settimeout(common.HOT_TIMEOUT_S)

            cache, past = common.new_cache(), 0

            def step(token_block):
                nonlocal past
                with self.compute_lock:
                    h1 = common.first_stage(model, S1, token_block, cache, past)
                common.send_msg(sock, {"type": "act", "hidden": h1})
                resp = common.recv_msg(sock)
                past += token_block.shape[1]
                with self.compute_lock:
                    return int(common.apply_lm_head(model, resp["hidden"]).argmax(-1))

            # 3) autoregressive loop, emitting each new piece of decoded text
            produced, prev_text, completion, finish = [], "", 0, "length"
            tok_id = step(input_ids)
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
                        moderation.log_event("out", verdict.category, request_id, snippet=full)
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
                tok_id = step(torch.tensor([[tok_id]]))

            common.send_msg(sock, {"type": "bye"})

            # 4) report completion so the coordinator settles the hold (Workstream B: real
            #    metered cost, node payouts, refund of anything unused). node_a.py counts
            #    len(generated), which includes the stopping token.
            latency_ms = int((time.time() - t_start) * 1000)
            tokens_generated = completion + (1 if finish == "stop" else 0)
            result = node_a.coord_complete(coordinator, request_id, tokens_generated,
                                           latency_ms, node_ids, complete_token,
                                           prompt_tokens=prompt_tokens)
            rewards = (result or {}).get("rewards") or {}
            actual_cost = (round(hold_amount - rewards.get("__refund__", 0.0), 6)
                          if hold_amount is not None else None)
            yield {"type": "done", "completion_tokens": completion,
                   "prompt_tokens": prompt_tokens, "finish_reason": finish,
                   "latency_ms": latency_ms,
                   "tok_per_s": round(completion / max(time.time() - t_start, 1e-6), 2),
                   "text": prev_text, "cost_nrn": actual_cost}
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
