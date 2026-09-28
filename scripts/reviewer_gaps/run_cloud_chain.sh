#!/bin/bash
# Reviewer-gap experiment chain for a CPU cloud runner (no GPU assumed).
# Specs: docs/specs/2026-09-26-l3-architecture-baseline-design.md and
#        docs/specs/2026-09-26-l3-second-fingerprint-instance-design.md
#
# Steps, one after another, each logged under fullruns/ (gitignored) with a chain log:
#   2  no-world-model organism run (L3 hidden 8, n = 10) + behavior audit
#   3  gate 0 for a second fingerprint seed (frozen fallback: hiddens 8 7 9 10 at
#      G seed 1, then seed 2); selection appended to PREREGISTRATION_L3 sec 12
#   4  organism run on the selected instance + behavior audit
# Then the small result JSONs are copied under artifacts/reviewer_gaps_runs/ so they
# can be committed (state dumps and agent bundles stay in fullruns/, gitignored).
#
# Usage (from the repo root):  bash scripts/reviewer_gaps/run_cloud_chain.sh [workers]
set -u
cd "$(dirname "$0")/../.." || exit 1
WORKERS="${1:-$(python -c 'import os;print(max(1, (os.cpu_count() or 2) - 1))')}"
R=fullruns
A=artifacts/reviewer_gaps_runs
SEEDS="0 1 2 3 4 5 6 7 8 9"
mkdir -p "$R" "$A"
LOG="$R/reviewer_gaps_chain.log"
status() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
status "START cloud chain (commit $(git rev-parse --short HEAD), workers=$WORKERS, cpus=$(python -c 'import os;print(os.cpu_count())'))"
python -c "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" | tee -a "$LOG"

# ---- Step 2: architecture baseline (no world-model auxiliary) --------------
mkdir -p "$R/l3_h8_nowm"
status "STEP2 nowm organism run start"
python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 --no-world-model \
  --seeds $SEEDS --updates 300 --save-agents --workers "$WORKERS" --device cpu \
  --dump-states "$R/l3_h8_nowm/states" --out-dir "$R/l3_h8_nowm" > "$R/l3_h8_nowm/run.log" 2>&1
status "STEP2 exit=$?"
python scripts/audit_behavior_mediation.py "$R/l3_h8_nowm/states" \
  --json "$R/l3_h8_nowm/behavior_audit.json" > "$R/l3_h8_nowm/behavior_audit.log" 2>&1
status "STEP2b behavior audit exit=$?"
mkdir -p "$A/l3_h8_nowm"
cp "$R/l3_h8_nowm/expB2_results.json" "$R/l3_h8_nowm/behavior_audit.json" "$R/l3_h8_nowm/run.log" "$A/l3_h8_nowm/" 2>/dev/null
cp -r "$R/l3_h8_nowm/cells" "$A/l3_h8_nowm/" 2>/dev/null

# ---- Step 3: second fingerprint instance, gate 0 --------------------------
SEL=NONE
GS=0
for GS in 1 2; do
  mkdir -p "$R/l3_gate0_seed$GS"
  status "STEP3 gate0 G seed $GS start"
  python scripts/run_expA_l3.py --g-seed $GS --hiddens 8 7 9 10 --floor-seeds 0 1 2 \
    --json "$R/l3_gate0_seed$GS/calibration.json" --device cpu \
    > "$R/l3_gate0_seed$GS/calibration.log" 2>&1
  status "STEP3 gate0 seed $GS exit=$?"
  SEL=$(python scripts/reviewer_gaps/select_instance.py "$R/l3_gate0_seed$GS/calibration.json" 8 7 9 10)
  status "STEP3 selection at seed $GS: $SEL"
  python scripts/reviewer_gaps/prereg_append.py docs/PREREGISTRATION_L3.md \
    "$R/l3_gate0_seed$GS/calibration.json" $GS "$SEL" >> "$LOG" 2>&1
  mkdir -p "$A/l3_gate0_seed$GS"
  cp "$R/l3_gate0_seed$GS/calibration.json" "$R/l3_gate0_seed$GS/calibration.log" "$A/l3_gate0_seed$GS/" 2>/dev/null
  if [ "$SEL" != "NONE" ]; then break; fi
done

# ---- Step 4: organism run on the selected instance ------------------------
if [ "$SEL" != "NONE" ]; then
  OUT="$R/l3_h${SEL}_gseed$GS"
  mkdir -p "$OUT"
  status "STEP4 organism run (hidden $SEL, G seed $GS) start"
  python scripts/run_expB2.py --drift-mode l3 --l3-hidden $SEL --l3-seed $GS \
    --seeds $SEEDS --updates 300 --save-agents --workers "$WORKERS" --device cpu \
    --dump-states "$OUT/states" --out-dir "$OUT" > "$OUT/run.log" 2>&1
  status "STEP4 exit=$?"
  python scripts/audit_behavior_mediation.py "$OUT/states" \
    --json "$OUT/behavior_audit.json" > "$OUT/behavior_audit.log" 2>&1
  status "STEP4b behavior audit exit=$?"
  mkdir -p "$A/l3_h${SEL}_gseed$GS"
  cp "$OUT/expB2_results.json" "$OUT/behavior_audit.json" "$OUT/run.log" "$A/l3_h${SEL}_gseed$GS/" 2>/dev/null
  cp -r "$OUT/cells" "$A/l3_h${SEL}_gseed$GS/" 2>/dev/null
else
  status "STEP4 skipped: no gate-0 candidate passed at seeds 1 and 2"
fi
cp "$LOG" "$A/"
status "END cloud chain"
