"""coordinator/test_gpu_capability.py — run: python -m coordinator.test_gpu_capability

Nodes report GPU capability; the coordinator stores it and the balancer may use it. The
properties worth pinning down are the ones a future reader is most likely to get wrong:

1. **VRAM is not capacity while the pipeline is CPU-only.** `common.py` materialises every
   shard into system RAM and selects no device, so a GPU node's VRAM holds nothing. Counting
   it would over-assign layers and OOM-kill a volunteer's machine. `balancer.GPU_EXECUTION`
   gates that arithmetic off, and there is a test asserting the gate holds.
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

    # A GPU node that re-registers from a build with no GPU detection must not silently lose
    # its card: has_gpu follows the latest report, but VRAM/name are COALESCEd like platform.
    register("gpu-node", has_gpu=True, vram=12.0, name="NVIDIA GeForce RTX 4070")
    n = models.get_node("gpu-node")
    check("re-registration keeps GPU details", n["has_gpu"] and n["gpu_vram_gb"] == 12.0)

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

    # ---------- balancer: VRAM is NOT capacity while the pipeline is CPU-only ----------
    check("GPU_EXECUTION is off (the pipeline has no CUDA path)",
          balancer.GPU_EXECUTION is False)

    gpu_node = {"node_id": "g", "ms_per_layer": 10.0, "ram_free_gb": 4.0,
                "has_gpu": True, "gpu_vram_gb": 24.0}
    check("VRAM does not raise the layer cap while GPU_EXECUTION is off",
          balancer.max_layers_for(gpu_node, gb_per_layer=1.0) == 3)   # 4 GB * 0.75 / 1

    real_flag = balancer.GPU_EXECUTION
    try:
        balancer.GPU_EXECUTION = True
        check("with GPU execution on, VRAM adds capacity (4 + 24) * 0.75",
              balancer.max_layers_for(gpu_node, gb_per_layer=1.0) == 21)
    finally:
        balancer.GPU_EXECUTION = real_flag

    check("a CPU node is unaffected by the GPU arithmetic either way",
          balancer.max_layers_for({"node_id": "c", "ram_free_gb": 4.0}, gb_per_layer=1.0) == 3)

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
