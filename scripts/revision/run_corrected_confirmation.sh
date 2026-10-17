#!/bin/bash
# Corrected-trainer confirmation runs C1 and C2, frozen in
# docs/specs/2026-10-06-corrected-trainer-confirmation-design.md (revision step 4).
#
# Usage, from a checkout of the frozen commit (a worktree keeps later edits away from the
# running workers):  bash scripts/revision/run_corrected_confirmation.sh [dest] [workers] [wm|nowm|both]
#   [dest] (default dist/retrain_<UTC date>, relative to the repository root) receives
#   fullruns/corrected_* (the raw run output) and corrected_runs/<run>/ (cells and summaries).
#   It is never artifacts/: the repository root and artifacts/ are refused, so the committed
#   artifacts/corrected_runs cells and the fullruns/corrected_* bundles stay as they are.
#   wm runs C1, nowm runs C2, both (the default) runs C1 then C2.
# Each run resumes from its own cells/ if interrupted (rerun the same command). A run is
# promoted to corrected_runs/<run>/ only when every python step exits 0 and the run has 20 cell
# files and both summaries, through a temporary folder and mv, so a failed or partial run leaves
# an earlier promotion as it was. The exit status is non-zero when any step fails.
set -u
set -o pipefail
cd "$(dirname "$0")/../.." || exit 1
DEST="${1:-dist/retrain_$(date -u +%Y%m%d)}"
WORKERS="${2:-4}"
WHICH="${3:-both}"
case "$WHICH" in
  wm|nowm|both) ;;
  *) echo "third argument must be wm, nowm or both, got '$WHICH'" >&2; exit 2 ;;
esac
mkdir -p "$DEST" || exit 1
ROOT_REAL="$(pwd -P)"
DEST_REAL="$(cd "$DEST" && pwd -P)" || exit 1
if [ "$DEST_REAL" = "$ROOT_REAL" ] || [ "$DEST_REAL" = "$ROOT_REAL/artifacts" ]; then
  echo "refusing dest '$DEST': it would write into the committed artifacts/ or fullruns/ bundles;" \
       "pass a fresh folder such as dist/retrain_<date>" >&2
  exit 2
fi
R="$DEST/fullruns"
A="$DEST/corrected_runs"
SEEDS="0 1 2 3 4 5 6 7 8 9"
LOG="$R/corrected_confirmation.log"
mkdir -p "$R" "$A" || exit 1
status() { echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }
export ITASORL_FOLDS=explicit
status "START (commit $(git rev-parse --short HEAD 2>/dev/null), workers=$WORKERS, runs=$WHICH, cpus=$(nproc))"
python -c "import torch, numpy, sklearn; print('torch', torch.__version__, 'numpy', numpy.__version__, 'sklearn', sklearn.__version__)" | tee -a "$LOG"
rc=$?
if [ "$rc" -ne 0 ]; then
  status "preflight exit=$rc: this python cannot import torch, numpy and sklearn; nothing run"
  exit "$rc"
fi

promote() {  # name, run output dir
  local name="$1" out="$2"
  local n
  n=$(ls "$out"/cells/cell_d*_s*.json 2>/dev/null | wc -l)
  if [ "$n" -ne 20 ] || [ ! -f "$out/expB2_results.json" ] || [ ! -f "$out/behavior_audit.json" ]; then
    status "$name NOT promoted: $n of 20 cell files, expB2_results.json and behavior_audit.json required"
    return 1
  fi
  local tmp="$A/.$name.incoming.$$" old="$A/.$name.previous.$$"
  mkdir -p "$tmp" \
    && cp "$out/expB2_results.json" "$out/behavior_audit.json" "$tmp/" \
    && cp -r "$out/cells" "$tmp/"
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    status "$name NOT promoted: copy exit=$rc"
    rm -rf "$tmp"
    return "$rc"
  fi
  if [ -e "$A/$name" ]; then
    mv "$A/$name" "$old" || { status "$name NOT promoted: could not move the earlier copy aside"; rm -rf "$tmp"; return 1; }
  fi
  if ! mv "$tmp" "$A/$name"; then
    [ -e "$old" ] && mv "$old" "$A/$name"
    status "$name NOT promoted: mv failed; the earlier copy is back in place"
    return 1
  fi
  rm -rf "$old"
  status "$name promoted to $A/$name"
}

run_one() {  # name, extra flags...
  local name="$1"; shift
  local out="$R/$name"
  mkdir -p "$out" || return 1
  local resume=""
  if ls "$out"/cells/cell_d*_s*.json >/dev/null 2>&1; then resume="--resume"; fi
  status "$name start $resume"
  python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 --seeds $SEEDS --updates 300 \
    --gae-bootstrap successor --budget-extend 450 --budget-snapshots 100 200 \
    --save-agents --dump-states "$out/states" --out-dir "$out" --workers "$WORKERS" \
    --device cpu $resume "$@" >> "$out/run.log" 2>&1
  local rc=$?
  status "$name train+eval exit=$rc"
  if [ "$rc" -ne 0 ]; then status "$name FAILED; nothing promoted"; return "$rc"; fi
  python scripts/audit_behavior_mediation.py "$out/states" --json "$out/behavior_audit.json" \
    > "$out/behavior_audit.log" 2>&1
  rc=$?
  status "$name behavior audit exit=$rc"
  if [ "$rc" -ne 0 ]; then status "$name FAILED; nothing promoted"; return "$rc"; fi
  promote "$name" "$out"
}

fail=0
if [ "$WHICH" != nowm ]; then run_one corrected_l3_h8_wm || fail=1; fi
if [ "$WHICH" != wm ]; then run_one corrected_l3_h8_nowm --no-world-model || fail=1; fi
status "END (exit $fail)"
exit "$fail"
