#!/usr/bin/env bash
# Higher-power (n=10 seed) extension of Experiment B-v2, GPU-aware and RAM-guarded.
#
#   bash run_expB2_n10.sh
#
# run_expB2.py already auto-selects CUDA; this wrapper adds three safeguards the
# bare command lacks:
#   1. verifies a CUDA GPU is actually visible to torch (fails loudly if not),
#   2. refuses to start unless enough CPU system RAM is free (the episode buffers
#      live in system RAM, so a starved machine OOMs mid-run, as it did under the
#      concurrent ralph loop),
#   3. pins the run to the current commit and writes outputs to non-clobbering
#      _n10 names, restoring the git-tracked n=3 artifacts afterward.
set -euo pipefail
cd "$(dirname "$0")"

MIN_FREE_GB=${MIN_FREE_GB:-4}

# --- 1. require a CUDA GPU ----------------------------------------------------
if ! python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "ERROR: no CUDA GPU visible to torch. Aborting." >&2
  exit 1
fi
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0))"

# --- 2. memory guard ----------------------------------------------------------
free_ram_mb() {
  powershell.exe -NoProfile -Command \
    "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)" 2>/dev/null | tr -dc '0-9'
}

# TODO(you): choose the policy when free RAM < ${MIN_FREE_GB} GB.
#   Option A (fail fast, the default below): print why and exit, you rerun later.
#   Option B (wait and poll): loop until RAM frees (e.g. ralph finishes), then run.
# The tradeoff: fail-fast keeps your terminal free and never hangs, but you have
# to come back and rerun; wait-and-poll launches the moment RAM is available but
# ties up the shell and could wait indefinitely if nothing ever frees memory.
# Implement your preferred behavior in the block below (roughly 5-10 lines).
FREE_MB=$(free_ram_mb)
if [ "${FREE_MB:-0}" -lt $((MIN_FREE_GB * 1024)) ]; then
  echo "ERROR: only ${FREE_MB} MB RAM free (< ${MIN_FREE_GB} GB needed)." >&2
  echo "Free memory first (e.g. stop the ralph loop), then rerun." >&2
  exit 1
fi
echo "Free RAM: ${FREE_MB} MB -- OK"

# --- 3. run, pinned to the current commit, with safe output handling ----------
COMMIT=$(git rev-parse --short HEAD)
echo "Running n=10 extension pinned to HEAD=${COMMIT}"
trap 'git checkout -- expB2_results.json expB2_survival.png 2>/dev/null || true' EXIT

python run_expB2.py --seeds 0 1 2 3 4 5 6 7 8 9 2>&1 | tee "expB2_run_n10_${COMMIT}.log"

mv -f expB2_results.json "expB2_results_n10_${COMMIT}.json"
mv -f expB2_survival.png "expB2_survival_n10_${COMMIT}.png"
git checkout -- expB2_results.json expB2_survival.png
echo "Done. n=10 outputs: expB2_results_n10_${COMMIT}.json / expB2_survival_n10_${COMMIT}.png"
