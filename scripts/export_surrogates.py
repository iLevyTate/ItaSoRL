"""Export the frozen L3 fingerprints the project used (revision step 15).

Each instance is retrained deterministically from its recipe (`train_g_motion`, world P,
fixed epochs) and serialized with `GMotion.to_npz`. A reader can load them with
`GMotion.from_npz` instead of retraining; `tests/test_reproducibility_package.py` checks
that a loaded file reproduces a fresh retrain exactly on this stack.

Usage:  python scripts/export_surrogates.py [--out artifacts/surrogates]
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import hashlib
import json
import os

from itasorl.surrogate_l3 import train_g_motion
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
INSTANCES = ((8, 0), (7, 0), (10, 1), (8, 2), (4, 0))   # hidden, G seed, as used in the runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="artifacts/surrogates")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    index = {"generated_by": "scripts/export_surrogates.py", "world": "WorldParams(k_land=1.5, "
             "k_water=1.5, gravity=0.4) [P]", "recipe": "train_g_motion defaults: 250 x 40 "
             "authentic scripted-policy transitions, 4 -> h -> h -> 2 ReLU MLP, Adam lr 1e-3, "
             "300 full-batch epochs, no early stopping",
             "device": "cpu (torch CPU kernels). The CPU runs (L3-H8-WM-CPU, L3-H8-NOWM-CPU, "
                       "L3-H10-GS1-CPU, the corrected C1/C2) trained G on CPU and match these files; "
                       "the GPU-published runs trained G on CUDA, whose last float bits can differ, "
                       "so their integrity gates regenerate G on CUDA", "files": {}}
    for hidden, seed in INSTANCES:
        name = f"gmotion_h{hidden}_s{seed}.npz"
        path = os.path.join(a.out, name)
        train_g_motion(hidden=hidden, seed=seed, params=P).to_npz(path)
        with open(path, "rb") as fh:
            index["files"][name] = {"hidden": hidden, "g_seed": seed,
                                    "sha256": hashlib.sha256(fh.read()).hexdigest()}
        print(f"wrote {path}")
    with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
