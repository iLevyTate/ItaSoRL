#!/bin/bash
# DRAFT runner for docs/specs/2026-09-28-l3-hidden8-second-seed-design.md. Do not
# launch until the owner has frozen that spec.
#
# CPU cloud sandbox, same as the device control. Gate 0 at hidden 8 for G seeds
# 2, 3, 4 in that frozen order (capacity never moves); the first passing seed is
# appended to PREREGISTRATION_L3 section 12 before launch; then the organism run
# (published hidden 8 protocol, world-model auxiliary on, --l3-seed <selected>)
# and the behavior audit. Small result files go under artifacts/reviewer_gaps_runs/
# and are committed and pushed by the script itself.
#
# Usage (from the repo root):  bash scripts/reviewer_gaps/run_cloud_h8_new_seed.sh [workers]
set -u
cd "$(dirname "$0")/../.." || exit 1
WORKERS="${1:-$(python -c 'import os;print(max(1, (os.cpu_count() or 2) - 1))')}"
R=fullruns
A=artifacts/reviewer_gaps_runs
SEEDS="0 1 2 3 4 5 6 7 8 9"
SPEC=docs/specs/2026-09-28-l3-hidden8-second-seed-design.md
mkdir -p "$R" "$A"
LOG="$R/reviewer_gaps_chain.log"
status() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
status "START hidden-8 new-seed chain (commit $(git rev-parse --short HEAD), workers=$WORKERS, cpus=$(python -c 'import os;print(os.cpu_count())'))"
python -c "import torch, sklearn, numpy;print('torch', torch.__version__, 'sklearn', sklearn.__version__, 'numpy', numpy.__version__, 'cuda', torch.cuda.is_available())" | tee -a "$LOG"
export ITASORL_FOLDS=explicit

SEL=NONE
GS=0
for GS in 2 3 4; do
  mkdir -p "$R/l3_h8_gate0_seed$GS"
  status "STEP6 gate0 hidden 8, G seed $GS start"
  python scripts/run_expA_l3.py --g-seed $GS --hiddens 8 --floor-seeds 0 1 2 \
    --json "$R/l3_h8_gate0_seed$GS/calibration.json" --device cpu \
    > "$R/l3_h8_gate0_seed$GS/calibration.log" 2>&1
  status "STEP6 gate0 seed $GS exit=$?"
  SEL=$(python scripts/reviewer_gaps/select_instance.py "$R/l3_h8_gate0_seed$GS/calibration.json" 8)
  status "STEP6 selection at seed $GS: $SEL"
  python scripts/reviewer_gaps/prereg_append.py docs/PREREGISTRATION_L3.md \
    "$R/l3_h8_gate0_seed$GS/calibration.json" $GS "$SEL" \
    --title "HIDDEN-8 NEW-SEED INSTANCE" --spec "$SPEC" --order "G seeds 2, 3, 4 at hidden 8" >> "$LOG" 2>&1
  mkdir -p "$A/l3_h8_gate0_seed$GS"
  cp "$R/l3_h8_gate0_seed$GS/calibration.json" "$A/l3_h8_gate0_seed$GS/" 2>/dev/null
  if [ "$SEL" != "NONE" ]; then break; fi
done

if [ "$SEL" != "NONE" ]; then
  OUT="$R/l3_h8_gseed$GS"
  mkdir -p "$OUT"
  status "STEP7 organism run (hidden 8, G seed $GS) start"
  python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 --l3-seed $GS \
    --seeds $SEEDS --updates 300 --save-agents --workers "$WORKERS" --device cpu \
    --dump-states "$OUT/states" --out-dir "$OUT" > "$OUT/run.log" 2>&1
  status "STEP7 exit=$?"
  python scripts/audit_behavior_mediation.py "$OUT/states" \
    --json "$OUT/behavior_audit.json" > "$OUT/behavior_audit.log" 2>&1
  status "STEP7b behavior audit exit=$?"
  mkdir -p "$A/l3_h8_gseed$GS"
  cp "$OUT/expB2_results.json" "$OUT/behavior_audit.json" "$A/l3_h8_gseed$GS/" 2>/dev/null
  cp -r "$OUT/cells" "$A/l3_h8_gseed$GS/" 2>/dev/null
else
  status "STEP7 skipped: no hidden-8 candidate passed gate 0 at G seeds 2, 3, 4"
fi
status "END hidden-8 new-seed chain"

# Self-commit so the results survive even if no agent is awake to commit them.
if [ "${CHAIN_SELF_COMMIT:-1}" = "1" ]; then
  git add "$A" docs/PREREGISTRATION_L3.md 2>/dev/null
  git -c user.name="itasorl-cloud-chain" -c user.email="chain@itasorl.local" commit -q \
    -m "chore(reviewer-gaps): cloud run results, hidden-8 new-seed instance" \
    && git push -q origin HEAD 2>>"$LOG"
  status "SELF-COMMIT exit=$? ($(git rev-parse --short HEAD))"
fi
