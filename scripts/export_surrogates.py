"""Export the frozen L3 fingerprints the project used (revision step 15).

Each instance is retrained from its recipe (`train_g_motion`, world P, fixed epochs) on one
CPU thread, as the run workers fit it, and serialized with `GMotion.to_npz`. A reader can
load them with `GMotion.from_npz` instead of retraining. `tests/test_reproducibility_package.py`
checks that a loaded file and a fresh single-thread retrain give outputs that agree within
np.allclose(atol=1e-6, rtol=1e-5), and that is the whole claim: a retrain is not bit-identical
across torch, numpy or CPU builds, or across thread counts. The committed files were exported
before this script recorded its stack, so their index.json names no versions. Every export
from now on records Python, torch and numpy versions, the OS and the thread count under
"stack".

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


def stack_record() -> dict:
    """The software stack and thread count an export ran on (recorded in index.json)."""
    import platform

    import numpy
    import torch
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": numpy.__version__, "os": platform.platform(),
            "torch_threads": torch.get_num_threads()}


def main() -> int:
    import torch
    torch.set_num_threads(1)     # the run workers fit G single-threaded; match their bits
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
                       "so their integrity gates regenerate G on CUDA",
             "reproduces": "a fresh single-thread CPU retrain agrees within np.allclose(atol=1e-6, "
                           "rtol=1e-5); bit identity is not claimed across stacks or thread counts",
             "stack": stack_record(), "files": {}}
    for hidden, seed in INSTANCES:
        name = f"gmotion_h{hidden}_s{seed}.npz"
        path = os.path.join(a.out, name)
        g = train_g_motion(hidden=hidden, seed=seed, params=P)
        g.to_npz(path)
        with open(path, "rb") as fh:
            index["files"][name] = {"hidden": hidden, "g_seed": seed,
                                    "params_fingerprint": g.params_fingerprint,
                                    "sha256": hashlib.sha256(fh.read()).hexdigest()}
        print(f"wrote {path}")
    with open(os.path.join(a.out, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
