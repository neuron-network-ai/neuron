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

Usage:
  python node_a.py --host-c 100.79.125.112 --host-b 100.114.189.46 --s1 10 --s2 19
  add --serial for the one-at-a-time baseline.
"""

import argparse
import socket
import threading
import time

import torch

import common

PROMPTS = [
    "Why is the sky blue",
    "What is photosynthesis",
    "Explain gravity",
    "What is DNA",
]

compute_lock = threading.Lock()   # serialise node_a's own compute across request threads


def run_request(idx, prompt, model, tok, cfg, eos_id, results):
    s1, s2, host_c, port_c, host_b, port_b, max_new = cfg
    t_start = time.time()
    sock = socket.create_connection((host_c, port_c), timeout=30)
    common.send_msg(sock, {"type": "config", "s1": s1, "s2": s2,
                           "host_b": host_b, "port_b": port_b})
    ack = common.recv_msg(sock)
    assert ack.get("ok"), f"node_c refused: {ack}"

    input_ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True, return_tensors="pt",
    )
    cache, past = common.new_cache(), 0
    a_ms = c_ms = b_ms = head_ms = net_ms = 0.0

    def step(token_block):
        nonlocal a_ms, c_ms, b_ms, head_ms, net_ms, past
        with compute_lock:
            t = time.time()
            h1 = common.first_stage(model, s1, token_block, cache, past)
            a_ms += (time.time() - t) * 1000
        t = time.time()
        common.send_msg(sock, {"type": "act", "hidden": h1})
        resp = common.recv_msg(sock)
        rt = (time.time() - t) * 1000
        past += token_block.shape[1]
        with compute_lock:
            th = time.time()
            tok_id = int(common.apply_lm_head(model, resp["hidden"]).argmax(-1))
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

    hit_eos = generated[-1] == eos_id
    out_ids = generated[:-1] if hit_eos else generated
    results[idx] = {
        "prompt": prompt,
        "text": tok.decode(out_ids, skip_special_tokens=True),
        "tokens": len(generated),
        "latency": time.time() - t_start,
        "a_ms": a_ms, "c_ms": c_ms, "b_ms": b_ms, "head_ms": head_ms, "net_ms": net_ms,
    }


def warmup(host_c, port_c, s1, s2, host_b, port_b):
    s = socket.create_connection((host_c, port_c), timeout=120)
    common.send_msg(s, {"type": "config", "s1": s1, "s2": s2,
                        "host_b": host_b, "port_b": port_b})
    common.recv_msg(s)
    common.send_msg(s, {"type": "bye"})
    s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host-c", required=True, help="node_c (middle) address")
    ap.add_argument("--host-b", required=True, help="node_b (last) address")
    ap.add_argument("--port-c", type=int, default=50999)
    ap.add_argument("--port-b", type=int, default=50999)
    ap.add_argument("--s1", type=int, default=10, help="node_a owns layers 0..s1-1")
    ap.add_argument("--s2", type=int, default=19, help="node_c owns s1..s2-1, node_b s2..n-1")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--serial", action="store_true")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--copies", type=int, default=1, help="repeat the prompt set N times")
    args = ap.parse_args()

    prompts = ([args.prompt] if args.prompt else PROMPTS) * args.copies
    mode = "SERIAL" if args.serial else "PARALLEL"

    print(f"[A] loading my shard (embed + layers 0..{args.s1-1} + lm_head) ...")
    t0 = time.time()
    tok, model, n = common.load_model_shard(0, args.s1, embed=True, head=True)
    eos_id = tok.eos_token_id
    print(f"[A] ready in {time.time()-t0:.1f}s | {common.MODEL_ID} | layers={n} | "
          f"A=0..{args.s1-1}, C={args.s1}..{args.s2-1}, B={args.s2}..{n-1}")

    print("[A] warming up node_c + node_b shards ...")
    warmup(args.host_c, args.port_c, args.s1, args.s2, args.host_b, args.port_b)

    print(f"[A] running {len(prompts)} requests [{mode}] ...\n")
    cfg = (args.s1, args.s2, args.host_c, args.port_c, args.host_b, args.port_b,
           args.max_new_tokens)
    results = [None] * len(prompts)
    batch_t0 = time.time()
    if args.serial:
        for i, p in enumerate(prompts):
            run_request(i, p, model, tok, cfg, eos_id, results)
    else:
        threads = [threading.Thread(target=run_request,
                                    args=(i, p, model, tok, cfg, eos_id, results))
                   for i, p in enumerate(prompts)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    batch_wall = time.time() - batch_t0

    # ---- report ------------------------------------------------------------- #
    print("[A] ===== PER-REQUEST =====")
    for r in results:
        snip = r["text"].replace("\n", " ")
        snip = snip[:60] + "..." if len(snip) > 60 else snip
        print(f"    [{r['tokens']:>3} tok, {r['latency']:5.1f}s]  {r['prompt']!r}: {snip}")

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
