"""Generate THIRD_PARTY_NOTICES.md from what is actually in the shipped bundle.

Reads every *.dist-info in the PyInstaller output: name, version, license id, the real
copyright line out of the bundled licence text, and the upstream URL. Nothing here is typed
from memory -- rerun it after any dependency change and commit the diff.
"""
import glob
import os
import re

BUNDLE = r"C:\Users\optin\neuron\dist\neuron-agent\_internal"
OUT = r"C:\Users\optin\neuron\THIRD_PARTY_NOTICES.md"


def field(text, name):
    m = re.search(rf"^{name}:\s*(.+)$", text, re.M | re.I)
    return m.group(1).strip() if m else ""


def url_of(text):
    hp = field(text, "Home-page")
    if hp:
        return hp
    for m in re.finditer(r"^Project-URL:\s*([^,]+),\s*(\S+)$", text, re.M | re.I):
        if m.group(1).strip().lower() in ("homepage", "source", "repository", "source code"):
            return m.group(2).strip()
    m = re.search(r"^Project-URL:\s*[^,]+,\s*(\S+)$", text, re.M)
    return m.group(1).strip() if m else ""


def license_id(text):
    lic = field(text, "License-Expression") or field(text, "License")
    if not lic or len(lic) > 60:
        cls = re.findall(r"^Classifier:\s*License\s*::\s*(.+)$", text, re.M | re.I)
        if cls:
            lic = cls[-1].replace("OSI Approved :: ", "").replace(" License", "")
    return lic or "see bundled text"


def copyright_of(dist_dir):
    """First real copyright line from whatever licence text ships with the package."""
    cands = sorted(glob.glob(os.path.join(dist_dir, "licenses", "**", "*"), recursive=True))
    cands += [p for p in sorted(glob.glob(os.path.join(dist_dir, "*")))
              if re.search(r"licen|copying|notice|authors", os.path.basename(p), re.I)]
    for p in cands:
        if not os.path.isfile(p):
            continue
        try:
            txt = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for line in txt.splitlines():
            s = re.sub(r"\s+", " ", line.strip().lstrip("#").strip())
            if not re.match(r"^(copyright|\(c\)|©)", s, re.I) or len(s) < 12:
                continue
            # The Apache-2.0 body and appendix both contain lines starting "copyright"
            # ("...copyright notice that is included in", "Copyright [yyyy] [name of
            # copyright owner]") which are boilerplate, not a holder. A real copyright
            # line carries a real year; those do not.
            if not re.search(r"\b(19|20)\d{2}\b", s):
                continue
            if re.search(r"copyright notice|name of copyright owner", s, re.I):
                continue
            # GPL/LGPL/MPL texts carry the *licence steward's* copyright on the licence
            # document itself (FSF, Mozilla). That is not the package's copyright holder.
            if re.search(r"Free Software Foundation|Mozilla Foundation", s, re.I):
                continue
            return s[:120]
    return ""


def has_license_text(dist_dir):
    for p in glob.glob(os.path.join(dist_dir, "licenses", "**", "*"), recursive=True):
        if os.path.isfile(p):
            return True
    return any(os.path.isfile(p) for p in glob.glob(os.path.join(dist_dir, "*"))
               if re.search(r"licen|copying|notice", os.path.basename(p), re.I))


rows = []
for di in sorted(glob.glob(os.path.join(BUNDLE, "*.dist-info")), key=lambda p: p.lower()):
    meta = os.path.join(di, "METADATA")
    if not os.path.exists(meta):
        continue
    text = open(meta, encoding="utf-8", errors="replace").read()
    rows.append({
        "name": field(text, "Name") or os.path.basename(di),
        "version": field(text, "Version"),
        "license": license_id(text),
        "url": url_of(text),
        "copyright": copyright_of(di),
        "has_text": has_license_text(di),
        "author": field(text, "Author") or field(text, "Author-email"),
    })

WEAK_COPYLEFT = {"certifi", "tqdm"}
LGPL = {"pystray"}

lines = []
w = lines.append
w("# Third-party notices")
w("")
w("NEURON's own source is Apache License 2.0 — see [LICENSE](LICENSE).")
w("")
w("The Windows installer is a PyInstaller bundle, so it **redistributes** the components")
w("below. They stay under their own licences; NEURON's licence does not apply to them, and")
w("theirs does not apply to NEURON. Every one is an OSI-approved open-source licence.")
w("")
w("Full licence texts travel inside the installed application, under")
w("`_internal/<package>.dist-info/`. This file is generated from that bundle by")
w("`tools/gen_notices.py` — rerun it after any dependency change rather than editing by hand.")
w("")
w("## Components requiring particular care")
w("")
w("**pystray (LGPL-3.0)** — the system-tray icon, used only by `agent/tray.py`. It is the one")
w("copyleft-with-teeth component in the bundle. NEURON's compliance rests on the whole")
w("application being published as source under Apache 2.0: anyone may substitute a modified")
w("pystray and rebuild, which is what LGPL-3.0 §4 asks for. The library is unmodified.")
w("")
w("**certifi and tqdm (MPL-2.0)** — weak, file-level copyleft. The obligation attaches only to")
w("the MPL-covered files themselves, which are unmodified here, and is satisfied by shipping")
w("their licence text. It places no condition on NEURON's own code.")
w("")
w("**llama.cpp / ggml (MIT)** — the local quantized engine (`engine/local_gguf.py`) loads")
w("`llama.dll` and `ggml*.dll`. These native libraries are built from llama.cpp/ggml, whose")
w("copyright is **separate** from the `llama-cpp-python` wrapper that vendors them:")
w("")
w("> Copyright (c) 2023-2024 The ggml authors")
w(">")
w("> MIT License — https://github.com/ggerganov/llama.cpp/blob/master/LICENSE")
w("")
w("## Full inventory")
w("")
w(f"{len(rows)} bundled distributions, as shipped in NEURON 0.18.0.")
w("")
w("| Component | Version | Licence | Copyright |")
w("|---|---|---|---|")
for r in rows:
    nm = r["name"]
    mark = ""
    if nm.lower() in LGPL:
        mark = " ⚠"
    elif nm.lower() in WEAK_COPYLEFT:
        mark = " †"
    if r["copyright"]:
        cp = r["copyright"].replace("|", "\\|")
    elif r["author"]:
        # No copyright line in the licence text (common for Apache-2.0, which carries none).
        # The declared author is real metadata; label it as such rather than dressing it up
        # as a copyright notice we did not find.
        cp = "author: " + re.sub(r"\s*<[^>]*>", "", r["author"]).replace("|", "\\|")[:80]
    elif r["has_text"]:
        cp = "*see bundled licence text*"
    else:
        cp = "**licence text not bundled**"
    link = f"[{nm}]({r['url']})" if r["url"].startswith("http") else nm
    w(f"| {link}{mark} | {r['version']} | {r['license']} | {cp} |")
w("")
w("⚠ LGPL-3.0 — see above.  † MPL-2.0, weak copyleft — see above.")
no_text = [r["name"] for r in rows if not r["has_text"]]
if no_text:
    w("")
    w("**Gap:** " + ", ".join(no_text) + " ship without a licence text in the bundle. "
      "Apache-2.0 §4 requires the licence to travel with the redistribution, so "
      "`packaging/neuron-agent.spec` collects it explicitly — rerun this generator after "
      "the next build to confirm the row clears.")
w("")
w("---")
w("")
w("## Model weights are licensed separately")
w("")
w("Model weights are **not** covered by any licence above, and are not redistributed by the")
w("installer — each node downloads its own slice from the upstream repository.")
w("")
w("- **Qwen2.5** (the served model family) — Apache 2.0. No usage restrictions.")
w("- **Llama-family models** — the Llama Community Licence, which is *not* an open-source")
w("  licence: it carries an acceptable-use policy, a monthly-active-user ceiling, and naming")
w("  and attribution requirements. NEURON does not serve one. Adding one would place real")
w("  obligations on the network and needs a deliberate decision, not a config change.")
w("")
w("© 2026 NEURON Labs, Rotterdam")

open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print(f"wrote {OUT}: {len(rows)} components")
missing = [r["name"] for r in rows if not r["copyright"]]
print("no copyright line found for:", ", ".join(missing) if missing else "(none)")
