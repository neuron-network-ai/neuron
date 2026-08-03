# Contributing to NEURON

NEURON is a distributed AI inference network run on ordinary consumer machines. It is early
alpha, the network is small, and the code is honest about what it does and does not do. Help
is welcome — especially the unglamorous kind.

## Running the tests

Every suite is a plain script with no test framework. Run one directly:

```bash
python -m coordinator.test_open_join
```

Run all of them, and **judge by exit code, not by the last printed line**:

```bash
for t in coordinator/test_*.py agent/test_*.py; do
  m=$(echo "${t%.py}" | tr '/' '.')
  python -m "$m" >/dev/null 2>&1 && echo "ok   $m" || echo "FAIL $m"
done
python packaging/test_app_entry.py >/dev/null 2>&1 && echo "ok   packaging" || echo "FAIL packaging"
```

That warning is not decorative. A suite raises on the first bad assert and prints no summary,
so its output *ends* with `PASS` lines from the cases that already ran. Four suites stayed red
for weeks behind a `| tail -1` that looked green.

The `packaging/` test is run as a script rather than with `-m` on purpose: the directory has no
`__init__.py`, and `packaging` is also an installed third-party module, so `-m packaging.…`
resolves to the wrong thing and reports a failure that is not real.

`selftest_shard.py` proves the layer split is bit-exact against running the whole model on one
machine. It needs the model weights and three stages, so it is not part of the loop above, but
it must pass before anything touching the inference path is merged.

## Reporting a bug

Open an issue at **github.com/neuron-network-ai/neuron/issues**. Useful reports include:

- what you expected and what happened;
- your OS, Python version, and whether you installed the `.exe` or ran from source;
- the relevant part of `agent.log` (`%LOCALAPPDATA%\NEURON\agent.log` on a packaged install,
  `agent/agent.log` from source).

Please skim the log before pasting it. It contains your node id and local paths.

Security issues that would let someone execute code on a volunteer's machine, redirect
earnings, or read another user's prompts: contact the maintainer directly rather than opening
a public issue.

## Pull requests

1. Branch off `main-full`. That is the default branch and the real history — `main` is a
   snapshot and is currently behind.
2. Keep the change to one thing. A bug fix and a refactor in one PR is two reviews.
3. Add a test for the behaviour you changed, in the style of the suite next to it.
4. Run the suites above and say in the PR which ones you ran.
5. Describe what you *measured*, not what you expect. "2.1× on a real layer's Linears against
   PyTorch's BLAS GEMM" is a result; "much faster" is not.

Do not commit: model weights, `*.db`, `node_tokens.json`, `.venv/`, anything under
`blockchain/`, or the `neuronscript_*` kernel sources. All are gitignored, and the staged file
list is worth checking before each commit.

## Code style

Match the file you are editing. There is no linter and no formatter to satisfy.

The conventions that are real:

- 4-space indent, ~100-column lines, standard-library imports first.
- **Comments explain why, not what.** The valuable comment is the one recording why an obvious
  approach was rejected, or what broke last time. Several comments in this codebase are the
  only surviving record of a bug that cost a session.
- Say what is not true. If a number came from a synthetic benchmark, or a feature is wired but
  inert, the code should say so where someone will read it.
- Fail in the safe direction, and be explicit about which direction that is. "Cannot tell" is
  not "yes" — see `agent/gpu.py`, where unreadable GPU utilisation deliberately does not pause
  a node.

## What is welcome

- Bug fixes, especially ones a new user hits and the maintainers never will.
- Tests for code that has none. The tray and the app entry point each shipped four silent bugs
  behind zero tests.
- Platform coverage: Linux and macOS get far less real-world use than Windows.
- Documentation that is clearer or more honest, including corrections to claims in
  `whitepaper.md` or `README.md` that the code does not support.
- Measurements. A number from real hardware is worth more than an optimisation.

Please open an issue first for: anything that changes the NRN economics, the node trust model,
or the privacy property that only the driver sees plaintext. Those are decisions before they
are code.

## Two hard rules

**ARM compatibility in `agent/`.** Everything under `agent/` must run on ARM — phones and
single-board machines are a target. No x86 intrinsics, no AVX2 assumptions, no compiled
extension without a pure-Python fallback. Stick to Python plus `psutil`/`requests`/`ctypes`.

**`common.py` is not modified without an explicit instruction.** It holds the model sharding,
the KV cache and the TCP framing that `selftest_shard.py`'s bit-exactness depends on. Changes
there break the correctness proof the whole network rests on.

## License

Apache 2.0. Contributions are accepted under the same license.
