"""
NEURON — node_a  (Machine 1 / Windows, CLIENT + driver)  [Session 5: 3-stage]

Pipeline:  node_a -> node_c -> node_b -> node_a(lm_head)
  node_a : embed + layers[0:s1] + lm_head
  node_c : layers[s1:s2]        (HP Pavilion, middle)
  node_b : layers[s2:n] + norm  (OptiPlex, last)

node_a connects only to node_c and passes node_b's address in the config, so
node_c relays. N requests run in parallel (own thread + connection + cache); a
per-machine compute lock serialises node_a's own math (stage-A + head) while the
network/remote compute overlaps. Reports throughput, per-request latency, and
per-node utilisation to show all three machines busy at once.

Session 6: with --coordinator, node_a asks the coordinator for the chain (which
nodes / which layers / where), runs inference over it, and reports completion so
the coordinator can credit NRN. --host-c/--host-b still work as a direct fallback.

Usage:
  python node_a.py --coordinator http://localhost:8000 --prompt "Why is the sky blue"
  python node_a.py --host-c <node-c-ip> --host-b <node-b-ip> --s1 10 --s2 19
  add --serial for the one-at-a-time baseline.
"""

import argparse
import socket
import threading
import time

import requests
import torch

import batching
import common
import wire_codec

PROMPTS = [
    "Why is the sky blue",
    "What is photosynthesis",
    "Explain gravity",
    "What is DNA",
]

# Kept for the model-load path. It used to serialise this process's own forward pass, which
# made the DRIVER the bottleneck once the nodes started batching (measured: 99% driver vs
# 7-78% nodes). Concurrent compute now goes through these two shared batchers -- built lazily
# because they close over the loaded model.
compute_lock = threading.Lock()
_batchers = {}
_batchers_lock = threading.Lock()


def get_batchers(model, s1):
    with _batchers_lock:
        if not _batchers:
            _batchers["stage"] = batching.MicroBatcher(
                lambda ids, cache, lengths: batching.first_stage_batched(
                    model, s1, ids, cache, lengths))
            _batchers["head"] = batching.MicroBatcher(
                lambda h, _c, _l: batching.apply_lm_head_batched(model, h))
        return _batchers["stage"], _batchers["head"]


class InsufficientFunds(Exception):
    """Raised when the coordinator's /infer returns 402 -- the wallet's balance doesn't
    cover this request's worst-case quoted cost. Distinct from a generic RuntimeError so
    callers (neuron_driver.py) can show a specific "add funds" message rather than a
    generic chain-unavailable error."""


# --------------------------------------------------------------------------- #
# Coordinator client (Session 6; wallet/hold wiring added Workstream B)
# --------------------------------------------------------------------------- #
def coord_get_chain(base, prompt, max_tokens, expected_s1, wallet_id, prompt_tokens_estimate=None):
    """POST /infer -> (host_c, port_c, host_b, port_b, s2, node_ids, request_id, complete_token,
    hold_amount). Assumes a 3-node chain matching node_a's loaded shard (layers 0..expected_s1-1).
    The complete_token authenticates the later /complete call ([P12]); wallet_id is who pays
    (Workstream B — /infer now holds the worst-case cost from this wallet before dispatch)."""
    body = {"prompt": prompt, "max_tokens": max_tokens, "wallet_id": wallet_id}
    if prompt_tokens_estimate is not None:
        body["prompt_tokens_estimate"] = prompt_tokens_estimate
    r = requests.post(f"{base}/infer", json=body, timeout=15)
    if r.status_code == 402:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise InsufficientFunds(detail)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(f"coordinator /infer {r.status_code}: {detail}")
    data = r.json()
    chain, request_id = data["chain"], data["request_id"]
    if len(chain) != 3:
        raise RuntimeError(f"expected a 3-node chain, got {len(chain)}")
    a, c, b = chain
    if a["layers"] != [0, expected_s1 - 1]:
        raise RuntimeError(f"chain assigns node_a {a['layers']} but shard is 0..{expected_s1-1}")
    return c["ip"], c["port"], b["ip"], b["port"], c["layers"][1] + 1, \
        [a["node_id"], c["node_id"], b["node_id"]], request_id, data.get("complete_token"), \
        data.get("hold_amount")


def coord_complete(base, request_id, tokens, duration_ms, node_ids, complete_token=None,
                   prompt_tokens=0):
    """Report completion so the coordinator settles the hold (Workstream B: real metered
    cost, node payouts, refund). Returns the parsed response body (rewards breakdown, etc.)
    -- previously discarded; callers can now learn the real settled cost."""
    try:
        r = requests.post(f"{base}/infer/{request_id}/complete",
                          json={"tokens_generated": tokens, "duration_ms": duration_ms,
                                "node_ids": node_ids, "complete_token": complete_token,
                                "prompt_tokens": prompt_tokens}, timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"[A] warn: completion report failed: {e}")
        return None


def run_request(idx, prompt, model, tok, cfg, eos_id, results, coordinator=None, wallet_id=None):
    s1, s2, host_c, port_c, host_b, port_b, max_new = cfg
    request_id, node_ids, complete_token = None, None, None
    t_start = time.time()
    try:
        if coordinator:
            host_c, port_c, host_b, port_b, s2, node_ids, request_id, complete_token, _hold = \
                coord_get_chain(coordinator, prompt, max_new, s1, wallet_id)
        _run(idx, prompt, model, tok, s1, s2, host_c, port_c, host_b, port_b,
             max_new, eos_id, results, coordinator, request_id, node_ids, t_start,
             complete_token)
    except Exception as e:
        results[idx] = {"prompt": prompt, "error": str(e), "text": "", "tokens": 0,
                        "latency": time.time() - t_start, "request_id": request_id,
                        "a_ms": 0.0, "c_ms": 0.0, "b_ms": 0.0, "head_ms": 0.0, "net_ms": 0.0}
        print(f"[A] request {idx} FAILED: {e}")


def _run(idx, prompt, model, tok, s1, s2, host_c, port_c, host_b, port_b,
         max_new, eos_id, results, coordinator, request_id, node_ids, t_start,
         complete_token=None):
    sock = socket.create_connection((host_c, port_c), timeout=common.COLD_CONNECT_TIMEOUT_S)
    # The config itself goes out in the legacy format -- it is the one message whose reader
    # might predate wire_codec, and it is tiny. Its "wire" field is the offer; the ack names
    # the codec the peer picked, or omits it, in which case codec stays None (legacy).
    common.send_msg(sock, {"type": "config", "s1": s1, "s2": s2,
                           "host_b": host_b, "port_b": port_b,
                           "wire": wire_codec.preference(model.config.hidden_size)})
    ack = common.recv_msg(sock)
    assert ack.get("ok"), f"node_c refused: {ack}"
    codec = wire_codec.negotiate([ack["wire"]] if ack.get("wire") else None)
    sock.settimeout(common.HOT_TIMEOUT_S)

    input_ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt",
    )
    cache, past = common.new_cache(), 0
    a_ms = c_ms = b_ms = head_ms = net_ms = 0.0

    stage_batcher, head_batcher = get_batchers(model, s1)

    def step(token_block):
        nonlocal a_ms, c_ms, b_ms, head_ms, net_ms, past
        t = time.time()
        h1 = stage_batcher.submit(token_block, cache, past)
        a_ms += (time.time() - t) * 1000
        t = time.time()
        common.send_msg(sock, {"type": "act", "hidden": h1}, codec=codec)
        resp = common.recv_msg(sock)
        rt = (time.time() - t) * 1000
        past += token_block.shape[1]
        th = time.time()
        tok_id = int(head_batcher.submit(resp["hidden"], None, 0).argmax(-1))
        head_ms += (time.time() - th) * 1000
        c_ms += resp["c_compute_ms"]
        b_ms += resp["b_compute_ms"]
        net_ms += max(rt - resp["c_compute_ms"] - resp["b_compute_ms"], 0.0)
        return tok_id

    generated = [step(input_ids)]
    while len(generated) < max_new and generated[-1] != eos_id:
        generated.append(step(torch.tensor([[generated[-1]]])))

    common.send_msg(sock, {"type": "bye"})
    sock.close()

    latency = time.time() - t_start
    if coordinator and request_id:
        coord_complete(coordinator, request_id, len(generated), int(latency * 1000),
                       node_ids, complete_token, prompt_tokens=int(input_ids.shape[1]))

    hit_eos = generated[-1] == eos_id
    out_ids = generated[:-1] if hit_eos else generated
    results[idx] = {
        "prompt": prompt,
        "text": tok.decode(out_ids, skip_special_tokens=True),
        "tokens": len(generated),
        "latency": latency,
        "a_ms": a_ms, "c_ms": c_ms, "b_ms": b_ms, "head_ms": head_ms, "net_ms": net_ms,
        "request_id": request_id,
    }


def warmup(host_c, port_c, s1, s2, host_b, port_b):
    s = socket.create_connection((host_c, port_c), timeout=common.COLD_CONNECT_TIMEOUT_S)
    common.send_msg(s, {"type": "config", "s1": s1, "s2": s2,
                        "host_b": host_b, "port_b": port_b})
    common.recv_msg(s)
    common.send_msg(s, {"type": "bye"})
    s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coordinator", default=None,
                    help="coordinator URL (e.g. http://localhost:8000); asks it for the chain")
    ap.add_argument("--host-c", default=None, help="node_c address (direct mode / fallback)")
    ap.add_argument("--host-b", default=None, help="node_b address (direct mode / fallback)")
    ap.add_argument("--port-c", type=int, default=50999)
    ap.add_argument("--port-b", type=int, default=50999)
    ap.add_argument("--s1", type=int, default=10, help="node_a owns layers 0..s1-1")
    ap.add_argument("--s2", type=int, default=19, help="node_c owns s1..s2-1, node_b s2..n-1")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--copies", type=int, default=1, help="repeat the prompt set N times")
    ap.add_argument("--engine", choices=["torch", "ns"], default="torch",
                    help="'ns' runs this driver's Linears (and the lm_head) on the AVX2 int8 "
                         "kernel via ns_engine. The head is the largest GEMM in the whole "
                         "pipeline, so the driver is where it matters most. Needs "
                         "NEURON_NS_LIB pointing at a built library.")
    ap.add_argument("--slice-dir", default=None,
                    help="load the driver shard from a byte-range slice (slice_downloader) "
                         "instead of the full HF cache. The driver is the only role that "
                         "still pulled the WHOLE checkpoint -- fine at 1.5B (3 GB), absurd "
                         "at 7B+ where snapshot_download fetches every shard (15 GB) to use "
                         "a third of it. With --first, a slice carries embed + lm_head, "
                         "which is exactly what this role needs.")
    ap.add_argument("--wallet-id", default=None,
                    help="wallet to charge (coordinator mode only). If omitted, a fresh "
                         "throwaway test wallet is auto-created and faucet-funded — "
                         "convenient for benchmarking/testing, not for a real user.")
    args = ap.parse_args()

    if not args.coordinator and not (args.host_c and args.host_b):
        ap.error("provide --coordinator URL, or both --host-c and --host-b")

    prompts = ([args.prompt] if args.prompt else PROMPTS) * args.copies
    mode = "SERIAL" if args.serial else "PARALLEL"
    coordinator = args.coordinator.rstrip("/") if args.coordinator else None
    route = f"coordinator {coordinator}" if coordinator else \
            f"direct c={args.host_c} b={args.host_b}"

    wallet_id = args.wallet_id
    if coordinator and not wallet_id:
        import uuid as _uuid
        wallet_id = f"node_a-cli-{_uuid.uuid4().hex[:12]}"
        try:
            fr = requests.post(f"{coordinator}/wallet/faucet", json={"wallet_id": wallet_id},
                               timeout=15)
            fr.raise_for_status()
            print(f"[A] auto-created test wallet '{wallet_id}' "
                  f"(faucet-funded: {fr.json().get('granted')} NRN)")
        except requests.RequestException as e:
            print(f"[A] warn: could not faucet-fund the auto wallet ({e}); "
                  f"/infer will likely 402 unless '{wallet_id}' already has a balance")

    print(f"[A] loading my shard (embed + layers 0..{args.s1-1} + lm_head) ...")
    t0 = time.time()
    if args.slice_dir:
        from transformers import AutoConfig, AutoTokenizer

        import slice_downloader
        model = slice_downloader.load_slice_model(args.slice_dir)
        tok = AutoTokenizer.from_pretrained(args.slice_dir)
        n = AutoConfig.from_pretrained(args.slice_dir).num_hidden_layers
    else:
        tok, model, n = common.load_model_shard(0, args.s1, embed=True, head=True)

    if args.engine == "ns":
        import ns_engine
        lib = ns_engine.load()
        if lib is None:
            print(f"[A] WARNING: no kernel at {ns_engine.DEFAULT_LIB} -- staying on PyTorch")
        else:
            # head=True: lm_head sits outside model.model.layers and is the biggest GEMM
            # here, so skipping it would forfeit most of the driver-side win.
            model, converted = ns_engine.convert(model, 0, args.s1 - 1, lib, head=True)
            print(f"[A] engine=ns | {converted} Linear layers on the int8 kernel "
                  f"(layers 0-{args.s1 - 1} + lm_head)")
    eos_id = tok.eos_token_id
    print(f"[A] ready in {time.time()-t0:.1f}s | {common.MODEL_ID} | layers={n} | "
          f"A owns 0..{args.s1-1} | route: {route}")

    if coordinator:
        print("[A] chain via coordinator (first request pays node_c/node_b cold-start)")
    else:
        print("[A] warming up node_c + node_b shards ...")
        warmup(args.host_c, args.port_c, args.s1, args.s2, args.host_b, args.port_b)

    print(f"[A] running {len(prompts)} requests [{mode}] ...\n")
    cfg = (args.s1, args.s2, args.host_c, args.port_c, args.host_b, args.port_b,
           args.max_new_tokens)
    results = [None] * len(prompts)
    batch_t0 = time.time()
    if args.serial:
        for i, p in enumerate(prompts):
            run_request(i, p, model, tok, cfg, eos_id, results, coordinator, wallet_id)
    else:
        threads = [threading.Thread(target=run_request,
                                    args=(i, p, model, tok, cfg, eos_id, results, coordinator,
                                          wallet_id))
                   for i, p in enumerate(prompts)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    batch_wall = time.time() - batch_t0

    # ---- report ------------------------------------------------------------- #
    print("[A] ===== PER-REQUEST =====")
    for r in results:
        if r.get("error"):
            print(f"    [FAILED]  {r['prompt']!r}: {r['error']}")
            continue
        snip = r["text"].replace("\n", " ")
        snip = snip[:60] + "..." if len(snip) > 60 else snip
        rid = f" req={r['request_id'][:8]}" if r.get("request_id") else ""
        print(f"    [{r['tokens']:>3} tok, {r['latency']:5.1f}s]{rid}  {r['prompt']!r}: {snip}")

    total_tokens = sum(r["tokens"] for r in results)
    sum_a = sum(r["a_ms"] for r in results) / 1000
    sum_c = sum(r["c_ms"] for r in results) / 1000
    sum_b = sum(r["b_ms"] for r in results) / 1000
    sum_head = sum(r["head_ms"] for r in results) / 1000
    sum_net = sum(r["net_ms"] for r in results) / 1000
    node_a_busy = sum_a + sum_head
    serial_pred = sum_a + sum_c + sum_b + sum_head + sum_net

    print(f"\n[A] ===== AGGREGATE ({mode}, {len(prompts)} requests) =====")
    print(f"    total tokens          : {total_tokens}")
    print(f"    batch wall time       : {batch_wall:.1f} s")
    print(f"    THROUGHPUT            : {total_tokens / batch_wall:.2f} tokens/sec")
    print(f"    avg latency / request : {sum(r['latency'] for r in results)/len(results):.1f} s")
    print(f"\n[A] ===== NODE UTILISATION (compute time / batch wall) =====")
    print(f"    node_a (A + head)     : {node_a_busy:5.1f}s  = {100*node_a_busy/batch_wall:.0f}%")
    print(f"    node_c (middle layers): {sum_c:5.1f}s  = {100*sum_c/batch_wall:.0f}%")
    print(f"    node_b (layers + norm): {sum_b:5.1f}s  = {100*sum_b/batch_wall:.0f}%")
    print(f"    -> all three high = 3 machines working at once (3-stage pipeline)")
    print(f"\n[A] fully-serial would take ~{serial_pred:.1f}s "
          f"(A {sum_a:.1f} + C {sum_c:.1f} + B {sum_b:.1f} + head {sum_head:.1f} "
          f"+ net {sum_net:.1f}); actual {batch_wall:.1f}s -> {serial_pred/batch_wall:.2f}x overlap")


if __name__ == "__main__":
    main()
