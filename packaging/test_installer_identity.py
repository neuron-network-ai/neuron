"""packaging/test_installer_identity.py — the installer says who it is, in the one dialog
   a suspicious stranger actually opens

    python packaging/test_installer_identity.py

A volunteer downloaded NEURON-Setup-0.20.25.exe, did not trust 207 MB of unsigned executable
enough to double-click it, and opened Properties -> Details first. Windows told them:

    File description   NEURON Setup
    File version       0.0.0.0
    Product version    0.20.25
    Copyright          (blank)

The download had worked perfectly. The bytes were the published asset, all 217,140,114 of them.
What failed was the file's account of itself: Inno fills "Product version" from `AppVersion`
but defaults `VersionInfoVersion` to **0.0.0.0**, and leaves Company and Copyright empty unless
the .iss sets them. Every build this project has ever shipped was stamped that way.

**Why a cosmetic-looking field is not cosmetic here.** The installer is not code-signed
(PACKAGING.md: a certificate is the real fix and is not bought yet). With no signature, the
version resource is a large part of what a browser's download reputation check and Defender
have left to go on — and "version zero, no company, no copyright" is the profile of a file
nothing stands behind. The project asks strangers to trust an unsigned 200 MB binary and then
hands them a Properties dialog that argues against it.

So, three invariants, all checkable offline against the .iss:

  1. the version resource is populated at all — every field Windows shows in that dialog;
  2. `VersionInfoVersion` is a real version, NOT 0.0.0.0 — the specific default that caused this;
  3. none of it hardcodes a version number. `test_version_lockstep.py` already guards the one
     `#define AppVersion`; a second copy of "0.20.25" pasted into a VersionInfo line would sit
     there through every future release, and a file version that disagrees with the product
     version beside it is worse than the blank field it replaced.

And a fourth that is not about versions: **every directive VALUE must be ASCII.** The .iss has
no UTF-8 BOM, so Inno Setup reads it in the system codepage. An em dash in a comment is
harmless — there are several, and Inno never reads them — but the same character in
`VersionInfoCopyright` ships mojibake into the resource this file exists to fill in.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ISS = os.path.join(HERE, "neuron.iss")

# What Windows shows in Properties -> Details, and the directive behind each row.
REQUIRED = [
    ("VersionInfoVersion", "File version"),
    ("VersionInfoProductVersion", "Product version"),
    ("VersionInfoProductName", "Product name"),
    ("VersionInfoDescription", "File description"),
    ("VersionInfoCompany", "Company"),
    ("VersionInfoCopyright", "Copyright"),
]

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


def directives(src):
    """Every `Key=Value` in the .iss, ignoring comment lines (`;`) and section headers."""
    out = {}
    for line in src.splitlines():
        s = line.strip()
        if not s or s.startswith(";") or s.startswith("[") or s.startswith("#"):
            continue
        if "=" in s:
            k, _, v = s.partition("=")
            out.setdefault(k.strip(), v.strip())
    return out


def main():
    src = open(ISS, encoding="utf-8-sig").read()
    d = directives(src)

    app_version = re.search(r'#define\s+AppVersion\s+"([^"]+)"', src)
    check("neuron.iss declares #define AppVersion", bool(app_version))
    version = app_version.group(1) if app_version else None

    # 1. POPULATED AT ALL.
    for key, shown_as in REQUIRED:
        check(f'{key} is set (Properties shows it as "{shown_as}")',
              bool(d.get(key)),
              f"Windows leaves that row blank or zeroed, on an unsigned 200 MB installer")

    # 2. NOT THE DEFAULT. This is the whole bug: 0.0.0.0 is what Inno stamps when nobody says
    #    otherwise, so an absent directive and a wrong one look identical in the built file.
    fv = d.get("VersionInfoVersion", "")
    resolved = fv.replace("{#AppVersion}", version or "")
    check("VersionInfoVersion is not the 0.0.0.0 default",
          bool(fv) and not re.fullmatch(r"0(\.0)*", resolved),
          f"VersionInfoVersion={fv!r} -> {resolved!r}; that is the value the reported build had")
    check("VersionInfoVersion resolves to a numeric version Windows accepts",
          bool(re.fullmatch(r"\d+(\.\d+){1,3}", resolved)),
          f"{resolved!r} is not a version resource Inno will compile")

    # 3. DERIVED, NOT PASTED. A release bumps one define; anything that hardcodes the number
    #    here is a field that silently stops matching the build it is stamped on.
    if version:
        pasted = sorted(k for k, v in d.items()
                        if k.startswith("VersionInfo") and version in v)
        check("no VersionInfo line hardcodes the version instead of {#AppVersion}",
              not pasted,
              f"{', '.join(pasted)} literally contain {version!r}; the next release bumps "
              f"#define AppVersion and leaves these behind, so the file version and the product "
              f"version beside it disagree")

    # 4. ASCII VALUES. Comments may hold anything; a directive value may not, because this file
    #    has no BOM and Inno will read it in the system codepage.
    if src.startswith("﻿"):
        check("BOM present, so non-ASCII values would be safe", True)
    else:
        nonascii = sorted(k for k, v in d.items() if any(ord(c) > 127 for c in v))
        check("every directive value is ASCII (the file has no UTF-8 BOM)",
              not nonascii,
              f"{', '.join(nonascii)} contain non-ASCII; Inno reads a BOM-less .iss in the "
              f"system codepage, so those ship as mojibake in the version resource")

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
