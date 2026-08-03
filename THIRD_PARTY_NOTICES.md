# Third-party notices

NEURON's own source is Apache License 2.0 — see [LICENSE](LICENSE).

The Windows installer is a PyInstaller bundle, so it **redistributes** the components
below. They stay under their own licences; NEURON's licence does not apply to them, and
theirs does not apply to NEURON. Every one is an OSI-approved open-source licence.

Full licence texts travel inside the installed application, under
`_internal/<package>.dist-info/`. This file is generated from that bundle by
`tools/gen_notices.py` — rerun it after any dependency change rather than editing by hand.

## Components requiring particular care

**pystray (LGPL-3.0)** — the system-tray icon, used only by `agent/tray.py`. It is the one
copyleft-with-teeth component in the bundle. NEURON's compliance rests on the whole
application being published as source under Apache 2.0: anyone may substitute a modified
pystray and rebuild, which is what LGPL-3.0 §4 asks for. The library is unmodified.

**certifi and tqdm (MPL-2.0)** — weak, file-level copyleft. The obligation attaches only to
the MPL-covered files themselves, which are unmodified here, and is satisfied by shipping
their licence text. It places no condition on NEURON's own code.

**llama.cpp / ggml (MIT)** — the local quantized engine (`engine/local_gguf.py`) loads
`llama.dll` and `ggml*.dll`. These native libraries are built from llama.cpp/ggml, whose
copyright is **separate** from the `llama-cpp-python` wrapper that vendors them:

> Copyright (c) 2023-2024 The ggml authors
>
> MIT License — https://github.com/ggerganov/llama.cpp/blob/master/LICENSE

## Full inventory

54 bundled distributions, as shipped in NEURON 0.18.0.

| Component | Version | Licence | Copyright |
|---|---|---|---|
| [accelerate](https://github.com/huggingface/accelerate) | 1.14.0 | Apache | author: The Hugging Face team |
| [anyio](https://github.com/agronholm/anyio) | 4.14.2 | MIT | Copyright (c) 2018 Alex Grönholm |
| [attrs](https://www.attrs.org/) | 26.1.0 | MIT | Copyright (c) 2015 Hynek Schlawack and the attrs contributors |
| [Authlib](https://github.com/authlib/authlib) | 1.7.2 | BSD-3-Clause | Copyright (c) 2017, Hsiaoming Yang |
| [bitarray](https://github.com/ilanschnell/bitarray) | 3.10.0 | PSF-2.0 | author: Ilan Schnell |
| [certifi](https://github.com/certifi/python-certifi) † | 2026.7.22 | MPL-2.0 | author: Kenneth Reitz |
| [ckzg](https://github.com/ethereum/c-kzg-4844) | 2.1.8 | Apache-2.0 | author: Ethereum Foundation |
| [click](https://github.com/pallets/click/) | 8.4.2 | BSD-3-Clause | Copyright 2014 Pallets |
| [cryptography](https://github.com/pyca/cryptography) | 49.0.0 | Apache-2.0 OR BSD-3-Clause | author: The Python Cryptographic Authority and individual contributors |
| [cytoolz](https://github.com/pytoolz/cytoolz) | 1.1.0 | BSD-3-Clause | Copyright (c) 2014-2022 Erik Welch |
| [diskcache](http://www.grantjenks.com/docs/diskcache/) | 5.6.3 | Apache 2.0 | Copyright 2016-2022 Grant Jenks |
| [eth_abi](https://github.com/ethereum/eth-abi) | 5.2.0 | MIT | Copyright (c) 2016-2020, 2022-2025 The Ethereum Foundation |
| [eth-account](https://github.com/ethereum/eth-account) | 0.13.7 | MIT | Copyright (c) 2019-2025 The Ethereum Foundation |
| [eth-hash](https://github.com/ethereum/eth-hash) | 0.8.0 | MIT | Copyright (c) 2018-2025 The Ethereum Foundation |
| [eth-keyfile](https://github.com/ethereum/eth-keyfile) | 0.8.1 | MIT | Copyright (c) 2017-2023 The Ethereum Foundation |
| [eth-keys](https://github.com/ethereum/eth-keys) | 0.7.0 | MIT | Copyright (c) 2017-2025 The Ethereum Foundation |
| [eth-rlp](https://github.com/ethereum/eth-rlp) | 2.2.0 | MIT | Copyright (c) 2018-2025 The Ethereum Foundation |
| [eth-typing](https://github.com/ethereum/eth-typing) | 6.0.0 | MIT | Copyright (c) 2018-2025 The Ethereum Foundation |
| [eth-utils](https://github.com/ethereum/eth-utils) | 6.0.0 | MIT | Copyright (c) 2017-2025 The Ethereum Foundation |
| [fake-useragent](https://github.com/fake-useragent/fake-useragent) | 2.2.0 | Apache-2.0 | author: Melroy van den Berg, Victor Kovtun |
| [fastapi](https://github.com/fastapi/fastapi) | 0.139.2 | MIT | Copyright (c) 2018 Sebastián Ramírez |
| [filelock](https://github.com/tox-dev/py-filelock) | 3.32.0 | MIT | Copyright (c) 2025 Bernát Gábor and contributors |
| [h2](https://github.com/python-hyper/h2/) | 4.4.0 | MIT | Copyright (c) 2015-2020 Cory Benfield and contributors |
| [hexbytes](https://github.com/ethereum/hexbytes) | 1.3.1 | MIT | Copyright (c) 2019-2025 The Ethereum Foundation |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | 0.36.2 | Apache | author: Hugging Face, Inc. |
| [itsdangerous](https://github.com/pallets/itsdangerous/) | 2.2.0 | BSD | Copyright 2011 Pallets |
| [Jinja2](https://github.com/pallets/jinja/) | 3.1.6 | BSD | Copyright 2007 Pallets |
| [llama_cpp_python](https://github.com/abetlen/llama-cpp-python) | 0.3.34 | MIT | Copyright (c) 2023 Andrei Betlen |
| [MarkupSafe](https://github.com/pallets/markupsafe/) | 3.0.3 | BSD-3-Clause | Copyright 2010 Pallets |
| [networkx](https://networkx.org/) | 3.6.1 | BSD-3-Clause | Copyright (c) 2004-2025, NetworkX Developers |
| [numpy](https://numpy.org) | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Copyright (c) 2005-2025, NumPy Developers. |
| [packaging](https://github.com/pypa/packaging) | 26.2 | Apache-2.0 OR BSD-2-Clause | author: Donald Stufft |
| [parsimonious](https://github.com/erikrose/parsimonious) | 0.10.0 | MIT | Copyright (c) 2012 Erik Rose |
| [pillow](https://python-pillow.github.io) | 12.3.0 | MIT-CMU | Copyright © 1997-2011 by Secret Labs AB |
| [psutil](https://github.com/giampaolo/psutil) | 7.2.2 | BSD-3-Clause | Copyright (c) 2009, Jay Loden, Dave Daeschler, Giampaolo Rodola |
| [pydantic](https://github.com/pydantic/pydantic) | 2.13.4 | MIT | Copyright (c) 2017 to present Pydantic Services Inc. and individual contributors. |
| [pystray](https://github.com/moses-palmer/pystray) ⚠ | 0.19.5 | LGPLv3 | author: Moses Palmér |
| [pyunormalize](https://github.com/mlodewijck/pyunormalize) | 17.0.0 | MIT | Copyright (c) 2021-2025, Marc Lodewijck |
| [PyYAML](https://pyyaml.org/) | 6.0.3 | MIT | Copyright (c) 2017-2021 Ingy döt Net |
| [regex](https://github.com/mrabarnett/mrab-regex) | 2026.7.19 | Apache-2.0 AND CNRI-Python | copyright (c) 1998-2001 by Secret Labs AB and licensed under CNRI's Python 1.6 |
| [requests](https://github.com/psf/requests) | 2.34.2 | Apache-2.0 | Copyright 2019 Kenneth Reitz |
| [rlp](https://github.com/ethereum/pyrlp) | 4.1.0 | MIT | Copyright (c) 2015 Jnnk, Vitalik Buterin |
| [safetensors](https://github.com/huggingface/safetensors) | 0.8.0 | Apache Software | author: Nicolas Patry, Luc Georges, Daniël De Kok |
| [setuptools](https://github.com/pypa/setuptools) | 65.5.0 | MIT | author: Python Packaging Authority |
| [sniffio](https://github.com/python-trio/sniffio) | 1.3.1 | MIT OR Apache-2.0 | author: "Nathaniel J. Smith" |
| [starlette](https://github.com/Kludex/starlette) | 1.3.1 | BSD-3-Clause | Copyright © 2018, [Encode OSS Ltd](https://www.encode.io/). |
| [sympy](https://sympy.org) | 1.14.0 | BSD | Copyright (c) 2006-2023 SymPy Development Team |
| [tokenizers](https://github.com/huggingface/tokenizers) | 0.19.1 | Apache Software | author: Anthony MOI |
| [toolz](https://github.com/pytoolz/toolz) | 1.1.0 | BSD-3-Clause | Copyright (c) 2013 Matthew Rocklin |
| [torch](https://pytorch.org/) | 2.4.1 | BSD-3 | Copyright (c) 2016- Facebook, Inc (Adam Paszke) |
| [tqdm](https://tqdm.github.io) † | 4.69.0 | MPL-2.0 AND MIT | Copyright (c) 2013 noamraph |
| [transformers](https://github.com/huggingface/transformers) | 4.44.2 | Apache 2.0 License | Copyright 2018- The Hugging Face team. All rights reserved. |
| [uvicorn](https://uvicorn.dev/) | 0.51.0 | BSD-3-Clause | Copyright © 2017-present, [Encode OSS Ltd](https://www.encode.io/). |
| [websockets](https://github.com/python-websockets/websockets) | 15.0.1 | BSD-3-Clause | author: Aymeric Augustin |

⚠ LGPL-3.0 — see above.  † MPL-2.0, weak copyleft — see above.

**Gap:** tokenizers ship without a licence text in the bundle. Apache-2.0 §4 requires the licence to travel with the redistribution, so `packaging/neuron-agent.spec` collects it explicitly — rerun this generator after the next build to confirm the row clears.

---

## Model weights are licensed separately

Model weights are **not** covered by any licence above, and are not redistributed by the
installer — each node downloads its own slice from the upstream repository. But *serving*
weights is distribution to end users, so a restricted licence binds the whole network
rather than one machine.

**This is enforced in code, not by convention.** `coordinator/model_registry.py` requires
every model to declare a licence and refuses any that is not in `PERMITTED_LICENSES`.
An absent or unrecognised licence is refused exactly like a restricted one, because
"we did not check" must not look like "we checked and it is fine". Before that gate,
`NEURON_EXTRA_MODELS` was an environment variable — restricted weights could enter the
catalog with no code change and no record. Adding one now takes a diff someone has to
justify. Covered by `coordinator/test_model_license_gate.py`.

- **Qwen2.5 at 0.5B / 1.5B / 7B / 14B / 32B** — Apache 2.0. This is what NEURON serves.
- **Qwen2.5 at 3B and 72B** — the separate Qwen and Qwen-Research licences, *not* Apache
  2.0. Reaching for a bigger Qwen looks like a size change and is actually a licence
  change; the gate refuses these.
- **Llama-family models** — the Llama Community Licence, which is *not* an open-source
  licence: acceptable-use policy, a 700M monthly-active-user ceiling, a required "Built
  with Llama" attribution, and a naming rule for derivatives. Refused by the gate.
  Benchmarking one locally is fine; serving one to users is a decision with obligations.

© 2026 NEURON Labs, Rotterdam
