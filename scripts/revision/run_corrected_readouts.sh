#!/bin/bash
# Readout-only follow-ups of revision steps 5 to 12 on the saved agents of one corrected run
# (docs/specs/2026-10-06-primary-analysis-and-l0.md, -controlled-persistence-design.md,
# -texture-comparator-design.md; docs/REVISION_2026-10.md). No survival training except the
# per-seed retraining that the policy-controlled readout checks bit for bit.
#
# Usage, from a checkout of a fixed commit (a worktree keeps later edits away from the run):
#   bash scripts/revision/run_corrected_readouts.sh <repo-root> <run-name> [workers]
# Waits until run_corrected_confirmation.sh has copied <run-name>, then runs each readout at
# nice 10 so a training run that is still going keeps priority. Outputs go to
# <repo-root>/artifacts/{l0_audit,policy_controls,persistence,control_diagnostics,texture,
# population_readout}/<run-name>*.json; logs to <repo-root>/fullruns/readouts_<run-name>*.log.
set -u
cd "$(dirname "$0")/../.." || exit 1
DEST="${1:?repo root for outputs}"
NAME="${2:?run name, e.g. corrected_l3_h8_wm}"
W="${3:-2}"
RUN="$DEST/fullruns/$NAME"
A="$DEST/artifacts"
LOG="$DEST/fullruns/readouts_$NAME.log"
export ITASORL_FOLDS=explicit OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
status() { echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }
until grep -q "$NAME copied to" "$DEST/fullruns/corrected_confirmation.log" 2>/dev/null; do sleep 120; done
status "START $NAME (commit $(git rev-parse --short HEAD), workers=$W)"
WM=""
case "$NAME" in *nowm*) WM="--no-world-model" ;; esac
mkdir -p "$A"/{l0_audit,policy_controls,persistence,control_diagnostics,texture,population_readout}
step() {
  local tag="$1"; shift
  status "$tag start"
  nice -n 10 "$@" > "$DEST/fullruns/readouts_${NAME}_$tag.log" 2>&1
  status "$tag exit=$?"
}
step l0 python scripts/run_l0_audit.py --agents-dir "$RUN/agents" --out "$A/l0_audit/$NAME.json"
step policy python scripts/run_policy_controlled_readouts.py --run-dir "$RUN" $WM \
  --out "$A/policy_controls/$NAME.json" --workers "$W"
step persistence python scripts/run_persistence_readout.py --agents-dir "$RUN/agents" \
  --out "$A/persistence/$NAME.json" --workers "$W"
step controls python scripts/run_control_diagnostics.py --run-dir "$RUN" \
  --out "$A/control_diagnostics/$NAME.json" --workers "$W"
step texture_gn python scripts/run_texture_fresh_probe.py --agents-dir "$RUN/agents" --family gn \
  --param 0.01 --out "$A/texture/${NAME}_gn.json" --workers "$W"
step texture_qd python scripts/run_texture_fresh_probe.py --agents-dir "$RUN/agents" --family qd \
  --param 6.0 --out "$A/texture/${NAME}_qd.json" --workers "$W"
step population python scripts/validate_population_readout.py --run-dir "$RUN" \
  --out "$A/population_readout/$NAME.json"
status "END $NAME"
