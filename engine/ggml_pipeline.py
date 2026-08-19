"""engine/ggml_pipeline.py — run the NETWORK path on llama.cpp instead of PyTorch ([P30]).

**The problem this exists to fix, with the numbers that justify it.** The network answers at
**1.36 tok/s** (measured end to end on 2026-08-19, coordinator `requests_served` 84 -> 85). That
is not "slow", it is unusable, and it is the single thing standing between NEURON and
TOKENOMICS §11.6's "answers under 30 s" gate. The cause is not the network: a real relayed hop
is 22.6 ms, about 7% of a token. The cause is the ENGINE — `node_server` computes its layers in
**PyTorch fp32**, measured at 3.09 tok/s against llama.cpp's 26.93 on the same machine.

**What this module does.** llama.cpp already knows how to split a model across machines: each
remote machine runs `ggml-rpc-server`, and the driver points a local `llama-server` at them with
`--rpc`, placing layers with `--tensor-split`. So the driver keeps the model file, the remote
machines contribute memory and compute, and NEURON keeps the decision that matters — WHO
computes WHICH layers — because `--tensor-split` is passed by us, from the coordinator's own
placement.

Measured across this PC and the Pavilion: **3.97 tok/s**, against 1.36 on the PyTorch path.

**And the honest limit, stated here rather than discovered later.** Splitting is not a speedup.
The same model on ONE of these machines runs at 32.11 tok/s — 8x faster than splitting it. A
distributed pipeline is slower than not distributing, always, because decode is sequential and
each extra machine adds a hop without removing work. Distribution buys CAPACITY: models that do
not fit on the machine in front of you. This module makes the capacity path usable, not fast.

**What it requires, which is a real change to what a driver is.** In ggml's RPC design the
CLIENT reads the model file and uploads tensors; the server holds no model of its own. With
mmap the driver does not need it in RAM, but it does need the whole file on DISK — where today
a node downloads only its own shard. Decided in PROBLEMS.md (2026-08-19): accept it, because
disk is the cheap resource and RAM is the binding one, and mark the boundary — this path cannot
serve a model no single machine can hold on disk. Only a full ggml embedding (Route 2) can.

**Transport.** Endpoints are `host:port` that must already be LOCAL to this process — a loopback
port bridged to the remote node through the [P52] authenticated channel (`agent/rpc_engine.py`
`bridge()`). ggml-rpc is a memory protocol and upstream says never to put it on an open network;
this module never opens a socket to a remote host itself, so it cannot be the thing that does.
"""
import json
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
import uuid

import requests

from safety import moderation

log = logging.getLogger("neuron.engine.ggml_pipeline")

BINARY_ENV = "NEURON_LLAMA_SERVER"
BINARY_NAMES = ("llama-server.exe",) if os.name == "nt" else ("llama-server",)
START_TIMEOUT_S = float(os.environ.get("NEURON_GGML_START_TIMEOUT_S", "600"))
# Standing the pipeline up means uploading every tensor to each remote machine. Measured at
# ~3 min for a 1.04 GB model over one hop, and it scales with model size and link speed, so a
# 180 s default timed out on the very first real run. It is a startup cost paid once per
# process, not per request — which is exactly why the server is kept alive rather than
# restarted per call.
BIND_HOST = "127.0.0.1"


def find_binary():
    """The llama-server executable, or None. None means "keep using PyTorch"."""
    explicit = os.environ.get(BINARY_ENV)
    if explicit and os.path.exists(explicit):
        return explicit
    here = os.path.dirname(os.path.abspath(__file__))
    roots = (here, os.path.dirname(here), os.path.join(os.path.dirname(here), "bin"))
    for name in BINARY_NAMES:
        for root in roots:
            cand = os.path.join(root, name)
            if os.path.exists(cand):
                return cand
        found = shutil.which(name)
        if found:
            return found
    return None


def _free_port():
    with socket.socket() as s:
        s.bind((BIND_HOST, 0))
        return s.getsockname()[1]


def split_from_layers(layer_counts):
    """Coordinator placement -> `--tensor-split`.

    This is the line that keeps the economics NEURON's. llama.cpp will happily decide placement
    by free memory if left alone; passing the split explicitly means the chain the coordinator
    built is the chain that runs, so emission still pays whoever actually held the layers.

    `--tensor-split` wants one number per device in device order, and the RPC devices come
    first, in the order given to `--rpc`. Ratios, not counts — llama.cpp normalises them — but
    counts ARE the ratio we want, so they are passed through unchanged.
    """
    return ",".join(str(int(c)) for c in layer_counts)


class GgmlPipeline:
    """A supervised local `llama-server` that computes across remote ggml devices.

    Started once and kept, deliberately. Standing the server up means uploading every tensor to
    the remote machines, which dominates a short request — measured at tens of seconds against
    a 6-second generation. A per-request process would spend all its time on setup and would
    have made the whole approach look slower than PyTorch when it is nearly 3x faster.
    """

    def __init__(self, model_path, endpoints, tensor_split=None, threads=None, binary=None):
        self.model_path = model_path
        self.endpoints = list(endpoints or [])
        self.tensor_split = tensor_split
        self.threads = threads or max(1, (os.cpu_count() or 4) - 1)
        self.binary = binary or find_binary()
        self.port = None
        self.proc = None
        self._lock = threading.Lock()

    @property
    def base_url(self):
        return f"http://{BIND_HOST}:{self.port}" if self.port else None

    def available(self):
        return bool(self.binary) and bool(self.model_path) and os.path.exists(self.model_path)

    def start(self):
        with self._lock:
            if self.proc and self.proc.poll() is None:
                return True
            if not self.available():
                log.info("ggml pipeline unavailable (binary=%s, model=%s) — staying on PyTorch",
                         self.binary, self.model_path)
                return False
            self.port = _free_port()
            cmd = [self.binary, "-m", self.model_path,
                   "--host", BIND_HOST, "--port", str(self.port),
                   "-t", str(self.threads), "--no-webui"]
            if self.endpoints:
                cmd += ["--rpc", ",".join(self.endpoints)]
            if self.tensor_split:
                cmd += ["--tensor-split", self.tensor_split]
            log.info("starting ggml pipeline across %d remote device(s), split=%s",
                     len(self.endpoints), self.tensor_split or "auto")
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, errors="replace")
            threading.Thread(target=self._drain, daemon=True).start()
            return self._wait_ready()

    def _drain(self):
        """llama-server's own log, relayed into ours.

        Not discarded: this is where a remote device failing to attach shows up, and [P51]'s
        lesson is that a subprocess writing to a pipe nobody reads is a failure that looks like
        silence. A windowed app has no console for it to fall out of.
        """
        try:
            for line in self.proc.stdout:
                line = line.rstrip()
                if line:
                    log.debug("llama-server: %s", line)
        except Exception:
            pass

    def _wait_ready(self):
        deadline = time.time() + START_TIMEOUT_S
        while time.time() < deadline:
            if self.proc.poll() is not None:
                log.error("llama-server exited with %s before becoming ready", self.proc.poll())
                return False
            try:
                r = requests.get(f"{self.base_url}/health", timeout=2)
                if r.status_code == 200:
                    log.info("ggml pipeline ready on %s", self.base_url)
                    return True
            except requests.RequestException:
                pass
            time.sleep(1.0)
        log.error("llama-server did not become ready within %.0fs", START_TIMEOUT_S)
        self.stop()
        return False

    def healthy(self):
        if not (self.proc and self.proc.poll() is None and self.port):
            return False
        try:
            return requests.get(f"{self.base_url}/health", timeout=2).status_code == 200
        except requests.RequestException:
            return False

    def stop(self):
        with self._lock:
            p, self.proc = self.proc, None
            if not p:
                return
            try:
                p.terminate()
                p.wait(timeout=10)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

    def stream(self, messages, max_new, request_id=None, node_ids=None,
               coordinator=None, wallet_id=None, cost_nrn=0.0):
        """Yield `neuron_driver.DRIVER.stream()`'s event shapes, so callers are unchanged.

        Output moderation runs against the FULL accumulated text rather than each delta, so a
        blocked phrase split across a token boundary is still caught — the same rule both other
        engines follow, and the reason it is repeated here rather than left to the caller.
        """
        request_id = request_id or ("ggml-" + uuid.uuid4().hex[:12])
        if not self.healthy() and not self.start():
            yield {"type": "error", "detail": "ggml pipeline unavailable",
                   "code": "no_ggml_engine"}
            return

        node_ids = list(node_ids or [])
        yield {"type": "meta", "request_id": request_id, "node_ids": node_ids,
               "nodes": len(node_ids), "cost_nrn": cost_nrn, "local": False,
               "engine": "ggml"}

        t0 = time.time()
        full, completion, finish = "", 0, "length"
        try:
            with requests.post(f"{self.base_url}/v1/chat/completions",
                               json={"messages": messages, "max_tokens": int(max_new),
                                     "temperature": 0.0, "stream": True},
                               stream=True, timeout=(10, 600)) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        continue
                    choice = (chunk.get("choices") or [{}])[0]
                    if choice.get("finish_reason") == "stop":
                        finish = "stop"
                    delta = (choice.get("delta") or {}).get("content")
                    if not delta:
                        continue
                    full += delta
                    completion += 1
                    verdict = moderation.check_text(full)
                    if verdict.blocked:
                        moderation.log_event("out", verdict.category, request_id, snippet=full)
                        moderation.report_violation(coordinator, wallet_id, "out",
                                                    verdict.category)
                        yield {"type": "error",
                               "detail": "This response was blocked by NEURON's acceptable-use "
                                         "policy (see SAFETY.md).",
                               "code": "content_policy_violation"}
                        return
                    yield {"type": "token", "text": delta}
        except requests.RequestException as e:
            yield {"type": "error", "detail": f"{e.__class__.__name__}: {e}"}
            return

        elapsed = max(time.time() - t0, 1e-6)
        yield {"type": "done", "completion_tokens": completion, "prompt_tokens": 0,
               "finish_reason": finish, "latency_ms": int(elapsed * 1000),
               "tok_per_s": round(completion / elapsed, 2), "text": full,
               "cost_nrn": cost_nrn}
