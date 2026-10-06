#!/bin/bash
# Corrected-trainer confirmation runs C1 and C2, frozen in
# docs/specs/2026-10-06-corrected-trainer-confirmation-design.md (revision step 4).
#
# Usage, from a checkout of the frozen commit (a worktree keeps later edits away from the
# running workers):  bash scripts/revision/run_corrected_confirmation.sh <repo-root> [workers]
#   <repo-root> receives fullruns/corrected_* (gitignored) and artifacts/corrected_runs/*.
# Each run resumes from its own cells/ if interrupted (rerun the same command).
set -u
cd "$(dirname "$0")/../.." || exit 1
DEST="${1:?repo root for outputs}"
WORKERS="${2:-4}"
R="$DEST/fullruns"
A="$DEST/artifacts/corrected_runs"
SEEDS="0 1 2 3 4 5 6 7 8 9"
LOG="$R/corrected_confirmation.log"
mkdir -p "$R" "$A"
status() { echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }
export ITASORL_FOLDS=explicit
status "START (commit $(git rev-parse --short HEAD), workers=$WORKERS, cpus=$(nproc))"
python -c "import torch, numpy, sklearn; print('torch', torch.__version__, 'numpy', numpy.__version__, 'sklearn', sklearn.__version__)" | tee -a "$LOG"

run_one() {  # name, extra flags...
  local name="$1"; shift
  local out="$R/$name"
  mkdir -p "$out"
  local resume=""
  if ls "$out"/cells/cell_d*_s*.json >/dev/null 2>&1; then resume="--resume"; fi
  status "$name start $resume"
  python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 --seeds $SEEDS --updates 300 \
    --gae-bootstrap successor --budget-extend 450 --budget-snapshots 100 200 \
    --save-agents --dump-states "$out/states" --out-dir "$out" --workers "$WORKERS" \
    --device cpu $resume "$@" >> "$out/run.log" 2>&1
  status "$name train+eval exit=$?"
  python scripts/audit_behavior_mediation.py "$out/states" --json "$out/behavior_audit.json" \
    > "$out/behavior_audit.log" 2>&1
  status "$name behavior audit exit=$?"
  mkdir -p "$A/$name"
  cp "$out/expB2_results.json" "$out/behavior_audit.json" "$A/$name/" 2>/dev/null
  rm -rf "$A/$name/cells" && cp -r "$out/cells" "$A/$name/" 2>/dev/null
  status "$name copied to $A/$name"
}

run_one corrected_l3_h8_wm
run_one corrected_l3_h8_nowm --no-world-model
status "END"
