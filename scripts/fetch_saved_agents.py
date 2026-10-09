"""Find the gitignored saved agents a readout-only rerun needs, wherever they are.

The historical hidden-8 held-out agents (`fullruns/l3_h8_heldout/agents/`) are not in the
repository. They are published with the paper's Zenodo record, and a reader who already has
them may hold them in a Drive folder. This resolves, in order: an existing local folder, a
Drive folder, then an archive URL (zip or tar), and verifies every file against the sha256
values recorded in the committed sensitivity artifact, so a rerun never scores the wrong
weights silently.

Usage (prints the resolved folder on the last line):
    python scripts/fetch_saved_agents.py --url https://zenodo.org/.../agents.zip
    python scripts/fetch_saved_agents.py --drive-dir /content/drive/MyDrive/ITASORL_agents/l3_h8_heldout
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_LOCAL = "fullruns/l3_h8_heldout/agents"
DEFAULT_REF = "artifacts/l0_audit/d045_world_samples_l3_h8_heldout.json"
NO_SOURCE = """\
No saved agents found. They are not in the git repository (19 MB, gitignored). Either:
  * pass --url with the agents archive of the paper's Zenodo record, or
  * upload the agent files to Drive (MyDrive/ITASORL_agents/l3_h8_heldout/) and pass --drive-dir.
The files are named agent_d0.45_s<seed>_survival.pt; their sha256 values are recorded in
artifacts/l0_audit/d045_world_samples_l3_h8_heldout.json and are checked after download."""


def _pattern(drift: float, arm: str) -> re.Pattern:
    return re.compile(rf"^agent_d{drift:.2f}_s(\d+)_{re.escape(arm)}\.pt$")


def find_agents_dir(root: Path, *, drift: float, arm: str) -> Path | None:
    """The folder under `root` (inclusive) holding the most matching agent files."""
    root = Path(root)
    if not root.is_dir():
        return None
    pat = _pattern(drift, arm)
    best, best_n = None, 0
    for dirpath, _dirs, files in os.walk(root):
        n = sum(1 for f in files if pat.match(f))
        if n > best_n:
            best, best_n = Path(dirpath), n
    return best


def extract_archive(archive: Path, out: Path) -> Path:
    archive, out = Path(archive), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(out, filter="data")
    else:
        raise SystemExit(f"{archive} is neither a zip nor a tar archive")
    return out


def download(url: str, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}", flush=True)
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as fh:  # noqa: S310
        shutil.copyfileobj(resp, fh)
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_agents(agents_dir: Path, ref: Path, *, drift: float, arm: str) -> dict:
    """Check every agent the artifact scored is present and byte-identical."""
    agents_dir = Path(agents_dir)
    with open(ref, encoding="utf-8") as fh:
        doc = json.load(fh)
    want = {f"agent_d{drift:.2f}_s{c['seed']}_{arm}.pt": c["agent_sha256"]
            for c in doc["cells"] if c.get("arm", arm) == arm}
    missing, mismatched = [], []
    for name, sha in sorted(want.items()):
        p = agents_dir / name
        if not p.is_file():
            missing.append(name)
        elif _sha256(p) != sha:
            mismatched.append(name)
    return {"ok": not missing and not mismatched, "n_checked": len(want),
            "missing": missing, "mismatched": mismatched}


def resolve(*, local_dir, drive_dir, url, work, drift: float = 0.45, arm: str = "survival") -> Path:
    for cand in (local_dir, drive_dir):
        if cand:
            found = find_agents_dir(Path(cand), drift=drift, arm=arm)
            if found is not None:
                print(f"using saved agents in {found}", flush=True)
                return found
    if url:
        work = Path(work)
        work.mkdir(parents=True, exist_ok=True)
        archive = download(url, work / url.rstrip("/").split("/")[-1].split("?")[0])
        found = find_agents_dir(extract_archive(archive, work / "extracted"), drift=drift, arm=arm)
        if found is not None:
            print(f"using saved agents extracted to {found}", flush=True)
            return found
        raise SystemExit(f"the archive at {url} holds no agent_d{drift:.2f}_s*_{arm}.pt files")
    raise SystemExit(NO_SOURCE)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--local-dir", default=DEFAULT_LOCAL)
    ap.add_argument("--drive-dir", default=None)
    ap.add_argument("--url", default=None)
    ap.add_argument("--work", default="fullruns/_downloaded_agents")
    ap.add_argument("--ref", default=DEFAULT_REF, help="artifact carrying the agent sha256 values")
    ap.add_argument("--drift", type=float, default=0.45)
    ap.add_argument("--arm", default="survival")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args(argv)
    found = resolve(local_dir=a.local_dir, drive_dir=a.drive_dir, url=a.url, work=a.work,
                    drift=a.drift, arm=a.arm)
    if not a.no_verify:
        if not os.path.exists(a.ref):
            raise SystemExit(f"no reference artifact at {a.ref}; pass --no-verify to skip the check")
        rep = verify_agents(found, Path(a.ref), drift=a.drift, arm=a.arm)
        if not rep["ok"]:
            raise SystemExit(f"saved agents do not match the committed artifact: missing {rep['missing']}, "
                             f"mismatched {rep['mismatched']}")
        print(f"verified {rep['n_checked']} agent files against {a.ref}", flush=True)
    print(found)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
