#!/bin/bash
# Device control for the architecture baseline (spec addendum 2026-09-27 in
# docs/specs/2026-09-26-l3-architecture-baseline-design.md): rerun the PUBLISHED
# decoder-carrying L3 hidden = 8 protocol, unchanged, on the same CPU sandbox that
# produced the no-world-model baseline, then the behavior audit. Results are copied to
# artifacts/reviewer_gaps_runs/l3_h8_wm_cpu/ for commit.
#
# Usage (from the repo root):  bash scripts/reviewer_gaps/run_cloud_device_control.sh [workers]
set -u
cd "$(dirname "$0")/../.." || exit 1
WORKERS="${1:-$(python -c 'import os;print(max(1, (os.cpu_count() or 2) - 1))')}"
R=fullruns
A=artifacts/reviewer_gaps_runs
SEEDS="0 1 2 3 4 5 6 7 8 9"
mkdir -p "$R/l3_h8_wm_cpu" "$A/l3_h8_wm_cpu"
LOG="$R/reviewer_gaps_chain.log"
status() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
status "START device control (commit $(git rev-parse --short HEAD), workers=$WORKERS, cpus=$(python -c 'import os;print(os.cpu_count())'))"
python -c "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" | tee -a "$LOG"
status "STEP5 decoder-carrying organism run (CPU) start"
python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 \
  --seeds $SEEDS --updates 300 --save-agents --workers "$WORKERS" --device cpu \
  --dump-states "$R/l3_h8_wm_cpu/states" --out-dir "$R/l3_h8_wm_cpu" > "$R/l3_h8_wm_cpu/run.log" 2>&1
status "STEP5 exit=$?"
python scripts/audit_behavior_mediation.py "$R/l3_h8_wm_cpu/states" \
  --json "$R/l3_h8_wm_cpu/behavior_audit.json" > "$R/l3_h8_wm_cpu/behavior_audit.log" 2>&1
status "STEP5b behavior audit exit=$?"
cp "$R/l3_h8_wm_cpu/expB2_results.json" "$R/l3_h8_wm_cpu/behavior_audit.json" "$A/l3_h8_wm_cpu/" 2>/dev/null
cp -r "$R/l3_h8_wm_cpu/cells" "$A/l3_h8_wm_cpu/" 2>/dev/null
status "END device control"

# Self-commit so the results survive even if no agent is awake to commit them
# (the first device-control session stranded its results this way on 2026-09-27).
if [ "${CHAIN_SELF_COMMIT:-1}" = "1" ] && [ -f "$A/l3_h8_wm_cpu/expB2_results.json" ]; then
  git add "$A" 2>/dev/null
  git -c user.name="itasorl-cloud-chain" -c user.email="chain@itasorl.local" commit -q \
    -m "chore(reviewer-gaps): cloud run results, device control (decoder-carrying arm on CPU)" \
    && git push -q origin HEAD 2>>"$LOG"
  status "SELF-COMMIT exit=$? ($(git rev-parse --short HEAD))"
fi
