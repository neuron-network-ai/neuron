"""test_slice_layout.py — does each node get exactly the tensors its ROLE actually runs?

This is the seam where two files have to agree and nothing checked that they did:

    common.load_model_shard(head=True)   <- the FIRST node asks for lm_head (the driver runs
                                            the output head, Session 3)
    slice_downloader.get_tensors_for_layers  <- decides which tensors that node downloads

They disagreed for the whole life of the project, and it was invisible because every model
served so far had `tie_word_embeddings=True`: lm_head IS embed_tokens, so the first node got
it regardless. The first untied model (Qwen2.5-7B and everything above it) would have handed
the driver an uninitialized meta tensor.

Uses only the safetensors header (a ~40 KB range request), so it costs nothing and needs no
weights on disk.

Run: python test_slice_layout.py
"""
import slice_downloader as sd

ok = fail = 0


def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")


def roles_for(model_id, n_layers, s1, s2):
    """The three real roles, exactly as agent/node_server assigns them."""
    header, _start, _url = sd.fetch_header(model_id)
    first = sd.get_tensors_for_layers(header, 0, s1 - 1, True, False)
    middle = sd.get_tensors_for_layers(header, s1, s2 - 1, False, False)
    last = sd.get_tensors_for_layers(header, s2, n_layers - 1, False, True)
    return header, first, middle, last


def test_model(model_id, n_layers, s1, s2, tied):
    print(f"\n{model_id}  (tie_word_embeddings={tied})")
    header, first, middle, last = roles_for(model_id, n_layers, s1, s2)

    # The load-side contract: load_model_shard(0, s1, embed=True, head=True) wants BOTH of
    # these on the driver, because the driver embeds tokens AND applies the head.
    check("driver gets embed_tokens (it embeds the prompt)",
          "model.embed_tokens.weight" in first)
    if "lm_head.weight" in header:
        check("driver gets lm_head (it runs the output head, NOT the last node)",
              "lm_head.weight" in first)
        check("last node does NOT get lm_head (it only returns a normed hidden)",
              "lm_head.weight" not in last)
    else:
        check("no separate lm_head in this checkpoint (tied to embed_tokens)", tied)

    check("last node gets the final norm", "model.norm.weight" in last)
    check("middle node gets neither embed, head nor norm",
          not ({"model.embed_tokens.weight", "lm_head.weight",
                "model.norm.weight"} & set(middle)))

    # Coverage: every layer lands on exactly one node, and nothing is downloaded twice.
    def layers_of(keep):
        return {int(k.split(".")[2]) for k in keep if k.startswith("model.layers.")}

    a, m, b = layers_of(first), layers_of(middle), layers_of(last)
    check(f"layers partition cleanly 0-{n_layers-1} with no overlap",
          a | m | b == set(range(n_layers)) and not (a & m) and not (m & b) and not (a & b))

    total = sum(sd._bytes(v) for v in header.values())
    got = sum(sum(sd._bytes(v) for v in part.values()) for part in (first, middle, last))
    print(f"      whole model {total/1e9:.2f} GB, three slices sum to {got/1e9:.2f} GB")
    check("the three slices together cover the model exactly once",
          abs(got - total) < 1024)


def main():
    # tied: the case that has always worked, kept so the fix does not regress it
    test_model("Qwen/Qwen2.5-1.5B-Instruct", 28, 10, 19, tied=True)
    # untied: the case that was broken, and the first size where NEURON's premise is real
    test_model("Qwen/Qwen2.5-7B-Instruct", 28, 11, 20, tied=False)
    print(f"\n{ok} passed, {fail} failed")
    raise SystemExit(1 if fail else 0)


if __name__ == "__main__":
    main()
