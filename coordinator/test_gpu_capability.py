"""coordinator/test_gpu_capability.py — run: python -m coordinator.test_gpu_capability

Nodes report GPU capability; the coordinator stores it and the balancer may use it. The
properties worth pinning down are the ones a future reader is most likely to get wrong:

1. **VRAM is not capacity while the pipeline computes on the CPU.** `common.py` resolves an
   execution device, but the loader every volunteer node actually uses --
   `slice_downloader.load_slice_model` -- returns the model without moving it, and the shipped
   torch is a `+cpu` build regardless. A GPU node's VRAM therefore holds nothing. Counting it
   over-assigns layers and OOM-kills a volunteer's machine; it did, against a 12 GB card in an
   8 GB machine. `balancer.GPU_EXECUTION` gates that arithmetic off, and there is a test
   asserting the gate holds.
2. **A GPU is not a speed multiplier.** Speed is the measured `ms_per_layer`. The GPU only
   breaks ties.
3. **`gpu_name` is operator-only**, like `platform` and the addresses — a card model is
   distinctive enough to correlate a roster on.
4. **An older agent that reports nothing is CPU-only**, not unknown-and-special.
"""
import os
import sqlite3
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="neuron-gpu-")
os.environ["NEURON_DB"] = os.path.join(_TMP, "t.db")

from coordinator import balancer, config, models      # noqa: E402
from coordinator import main as coord                 # noqa: E402

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def register(node_id, has_gpu=False, vram=None, name=None, port=51000):
    body = coord.RegisterBody(node_id=node_id, tailscale_ip="127.0.0.1", port=port,
                              layer_start=0, layer_end=9, cores=8, ram_gb=16.0,
                              platform="Linux-6.8", has_gpu=has_gpu,
                              gpu_vram_gb=vram, gpu_name=name)
    return coord.register(body, x_register_secret=None, x_node_token=None)


def main():
    models.init_db()

    # ---------- storage ----------
    register("gpu-node", has_gpu=True, vram=12.0, name="NVIDIA GeForce RTX 4070")
    n = models.get_node("gpu-node")
    check("GPU fields stored", n["has_gpu"] is True and n["gpu_vram_gb"] == 12.0
          and n["gpu_name"] == "NVIDIA GeForce RTX 4070")

    register("cpu-node", port=51001)
    n = models.get_node("cpu-node")
    check("a node reporting no GPU is has_gpu False with null VRAM/name",
          n["has_gpu"] is False and n["gpu_vram_gb"] is None and n["gpu_name"] is None)
    check("has_gpu is a real bool, not SQLite's 0/1",
          isinstance(models.get_node("cpu-node")["has_gpu"], bool))

    # An agent build that detects the card but omits the VRAM figure must not erase a good one.
    register("gpu-node", has_gpu=True)
    n = models.get_node("gpu-node")
    check("re-registration with no VRAM figure keeps the known one",
          n["has_gpu"] and n["gpu_vram_gb"] == 12.0)

    # But a node that LOSES its card must not keep phantom VRAM. It was COALESCEd like
    # `platform`, so the figure outlived the card that justified it and the balancer could size
    # a slice from memory the machine no longer had.
    register("gpu-node", has_gpu=False)
    n = models.get_node("gpu-node")
    check("a node that loses its GPU drops the VRAM and name with it",
          n["has_gpu"] is False and n["gpu_vram_gb"] is None and n["gpu_name"] is None,
          dict(n))
    register("gpu-node", has_gpu=True, vram=12.0, name="NVIDIA GeForce RTX 4070")

    # The registration edge clamps an unbelievable VRAM claim to NULL -- and still registers
    # the node. Refusing it would be [P24] again: a healthy machine that simply never joins.
    r = register("liar-node", has_gpu=True, vram=100000.0, name="NVIDIA RTX 9090", port=51002)
    n = models.get_node("liar-node")
    check("an absurd VRAM claim is stored as NULL", n["gpu_vram_gb"] is None, dict(n))
    check("...and the node is still registered, not refused",
          n["has_gpu"] is True and r.get("status") == "registered", r)

    # ---------- privacy ----------
    pub = {x["node_id"]: x for x in coord.node_list(x_register_secret=None)["nodes"]}
    check("public /node/list hides gpu_name", "gpu_name" not in pub["gpu-node"])
    check("public /node/list still shows has_gpu (as coarse as cores/ram_gb)",
          pub["gpu-node"]["has_gpu"] is True)
    check("public /node/list still shows gpu_vram_gb", pub["gpu-node"]["gpu_vram_gb"] == 12.0)
    priv = {x["node_id"]: x
            for x in coord.node_list(x_register_secret=config.REGISTRATION_SECRET)["nodes"]}
    check("operator sees gpu_name",
          priv["gpu-node"]["gpu_name"] == "NVIDIA GeForce RTX 4070")

    # ---------- migration from a pre-GPU database ----------
    legacy = os.path.join(_TMP, "legacy.db")
    con = sqlite3.connect(legacy)
    con.executescript("""
        CREATE TABLE nodes (node_id TEXT PRIMARY KEY, tailscale_ip TEXT NOT NULL,
            port INTEGER NOT NULL, layer_start INTEGER NOT NULL, layer_end INTEGER NOT NULL,
            cores INTEGER, ram_gb REAL, node_token TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'online', last_seen REAL NOT NULL,
            registered_at REAL NOT NULL);
        INSERT INTO nodes VALUES ('old','10.0.0.1',50999,0,9,4,8.0,'tok','online',1.0,1.0);
    """)
    con.commit()
    con.close()
    real_db = config.DB_PATH
    try:
        config.DB_PATH = legacy
        models.init_db()
        cols = {r["name"] for r in sqlite3.connect(legacy).execute(
            "PRAGMA table_info(nodes)").fetchall() for r in [dict(zip(["cid", "name"], r[:2]))]}
        check("migration adds the three GPU columns",
              {"has_gpu", "gpu_vram_gb", "gpu_name"} <= cols, sorted(cols))
        old = models.get_node("old")
        check("a pre-GPU row migrates to has_gpu False, not NULL",
              old["has_gpu"] is False and old["gpu_vram_gb"] is None)
    finally:
        config.DB_PATH = real_db

    # ---------- balancer: VRAM is NOT capacity, because the pipeline never reaches it -------
    #
    # These assertions were inverted on 2026-08-08, and the inversion IS the fix, not a way of
    # greening a red suite. What they asserted before was the harm itself: that a node claiming
    # 24 GB of VRAM may be handed 18 layers to hold. The premise -- Session 42's claim that
    # `common.py` moves a shard onto CUDA -- is false on the path a volunteer's machine runs:
    # `slice_downloader.py:299` returns `cast_linears(model)` with no device move, and the
    # shipped wheel is torch 2.4.1+cpu. So the weights are in system RAM and the sizing was
    # done against VRAM. See balancer.GPU_EXECUTION.
    check("GPU_EXECUTION is off (the loader never moves a shard to the device)",
          balancer.GPU_EXECUTION is False)

    # These cases use a synthetic gb_per_layer=1.0 to test WHICH memory figure is chosen
    # (VRAM vs free RAM vs total RAM), which is a different question from how many bytes a
    # parameter takes. `effective_gb_per_layer` scales by the node's storage dtype, so with
    # the real fp32 default a "1.0 GB layer" is charged as 2.0 and every count below halves —
    # obscuring the branch each check is actually about. Neutralised here, restored after, and
    # the dtype scaling itself is asserted with real footprints in test_weight_dtype_sizing.py.
    _real_bytes = balancer.ASSUMED_WEIGHT_BYTES
    balancer.ASSUMED_WEIGHT_BYTES = balancer.TIER_BASIS_BYTES        # factor == 1.0
    check("the dtype correction is neutral for these branch tests",
          balancer.effective_gb_per_layer(1.0) == 1.0)

    gpu_node = {"node_id": "g", "ms_per_layer": 10.0, "ram_free_gb": 4.0,
                "has_gpu": True, "gpu_vram_gb": 24.0}
    check("a GPU node is sized by system RAM (4 * 0.75), not by its 24 GB of VRAM",
          balancer.max_layers_for(gpu_node, gb_per_layer=1.0) == 3)

    # The volunteer this shipped against, by name. A 12 GB card in a machine with 8 GB of RAM
    # was assigned from VRAM with no OS reserve -- ~19 layers of weights into 8 GB of RAM, on
    # a machine whose owner is using it. The cap must come from RAM: (8 - 3) * 0.75 = 3.
    volunteer = {"node_id": "v", "ms_per_layer": 10.0,
                 "has_gpu": True, "gpu_vram_gb": 12.0, "ram_gb": 8.0}
    check("12 GB card + 8 GB RAM is sized from RAM, not from VRAM",
          balancer.max_layers_for(volunteer, gb_per_layer=1.0) == 3,
          balancer.max_layers_for(volunteer, gb_per_layer=1.0))
    check("...and that is strictly fewer layers than the VRAM figure would have given",
          balancer.max_layers_for(volunteer, gb_per_layer=1.0) < int(12.0 * 0.75))

    small_vram = {"node_id": "g2", "ram_free_gb": 32.0, "has_gpu": True, "gpu_vram_gb": 4.0}
    check("a GPU node with lots of RAM is sized by that RAM",
          balancer.max_layers_for(small_vram, gb_per_layer=1.0) == 24)

    check("VRAM alone constrains nothing -- a node reporting only a card is unconstrained",
          balancer.max_layers_for({"has_gpu": True, "gpu_vram_gb": 8.0},
                                  gb_per_layer=1.0) is None)

    # ---------- the VRAM figure is bounded for when the flag goes back on ----------
    # `gpu_vram_gb` is the one registration field that becomes a memory budget, and open join
    # means it arrives with no credential behind it. Unbelievable -> None -> size from RAM.
    check("a plausible VRAM figure passes through", balancer.sane_vram_gb(12.0) == 12.0)
    for bad, label in [(None, "missing"), ("lots", "not a number"), (0.0, "zero"),
                       (-8.0, "negative"), (float("nan"), "NaN"), (float("inf"), "infinite"),
                       (1e9, "absurdly large")]:
        check(f"an unbelievable VRAM figure ({label}) is dropped, not believed",
              balancer.sane_vram_gb(bad) is None)

    real_flag = balancer.GPU_EXECUTION
    try:
        balancer.GPU_EXECUTION = True
        check("with GPU execution on, a GPU node is sized by VRAM minus the OS reserve",
              balancer.max_layers_for(gpu_node, gb_per_layer=1.0)
              == int((24.0 - balancer.VRAM_OS_RESERVE_GB) * 0.75))
        # The reserve is the point: a card with no headroom left stutters the desktop it is
        # drawing. `local_gguf.GPU_HEADROOM_GB` reserves the same 1.5 GB on the local path.
        check("the VRAM branch reserves headroom, as the RAM branch always has",
              balancer.max_layers_for(gpu_node, gb_per_layer=1.0) < int(24.0 * 0.75))
        check("VRAM replaces system RAM, never adds to it",
              balancer.max_layers_for(gpu_node, gb_per_layer=1.0)
              != int((4.0 + 24.0) * 0.75))
        # A junk figure must not become a budget even once the flag is on.
        check("with the flag on, an absurd VRAM claim falls back to system RAM",
              balancer.max_layers_for({"node_id": "liar", "ram_free_gb": 4.0, "has_gpu": True,
                                       "gpu_vram_gb": 100000.0}, gb_per_layer=1.0) == 3)
    finally:
        balancer.GPU_EXECUTION = real_flag

    check("a CPU node is unaffected by the GPU arithmetic either way",
          balancer.max_layers_for({"node_id": "c", "ram_free_gb": 4.0}, gb_per_layer=1.0) == 3)
    check("a node reporting neither RAM nor VRAM is unconstrained, as before",
          balancer.max_layers_for({"node_id": "u"}, gb_per_layer=1.0) is None)

    balancer.ASSUMED_WEIGHT_BYTES = _real_bytes
    check("the real (pessimistic, fp32) weight assumption is restored",
          balancer.ASSUMED_WEIGHT_BYTES == 4.0)

    # ---------- balancer: a GPU is a tie-break, never a speed multiplier ----------
    equal = [{"node_id": "cpu", "ms_per_layer": 10.0},
             {"node_id": "gpu", "ms_per_layer": 10.0, "has_gpu": True}]
    a = balancer.solve(equal, 10)
    check("equal measured speeds -> equal split; a GPU does NOT inflate the share",
          [x["layers"] for x in a] == [5, 5], [x["layers"] for x in a])

    faster_cpu = [{"node_id": "cpu", "ms_per_layer": 5.0},
                  {"node_id": "gpu", "ms_per_layer": 20.0, "has_gpu": True}]
    a = balancer.solve(faster_cpu, 10)
    by = {x["node_id"]: x["layers"] for x in a}
    check("a measured-slow GPU node still gets fewer layers than a fast CPU node",
          by["cpu"] > by["gpu"], by)

    # The tie-break only fires when memory forces a layer to move and two candidates are
    # equally fast. Node 0 is over its cap; the CPU and GPU candidates tie on speed.
    tie = [{"node_id": "over", "ms_per_layer": 10.0, "ram_free_gb": 1.4},
           {"node_id": "cpu", "ms_per_layer": 10.0, "ram_free_gb": 100.0},
           {"node_id": "gpu", "ms_per_layer": 10.0, "ram_free_gb": 100.0, "has_gpu": True}]
    a = balancer.solve(tie, 9, gb_per_layer=1.0)
    by = {x["node_id"]: x["layers"] for x in a}
    check("displaced layers go to the GPU node when speeds tie",
          by["gpu"] > by["cpu"], by)
    check("the split still covers every layer exactly once",
          sum(by.values()) == 9 and a[-1]["layer_end"] == 8, a)

    check("assignments carry has_gpu for the dashboard/plan",
          all("has_gpu" in x for x in a))

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
