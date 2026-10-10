#!/usr/bin/env bash
# One resume step of a goal-and-stakes run in a cloud session (spec: docs/specs/
# 2026-10-07-goal-and-stakes-design.md, "Compute and provenance").
#
#   scripts/cloud_run_step.sh <run-name> <max-minutes> [extra run_expB2 flags...]
#
# Pulls the run branch run/<run-name>, refuses to start if a lock younger than
# LOCK_MAX_MIN minutes exists on the branch, launches run_expB2 with --resume into
# artifacts/goal_stakes/<run-name>, commits and pushes every new cell as it lands, and
# exits 0 when all 20 cells exist (writing DONE), 3 while cells remain, 2 on a live lock.
set -euo pipefail
NAME="$1"; MAX_MIN="$2"; shift 2
BRANCH="run/${NAME}"
OUT="artifacts/goal_stakes/${NAME}"
LOCK="${OUT}/RUNNING.lock"
LOCK_MAX_MIN=${LOCK_MAX_MIN:-120}
N_CELLS=${N_CELLS:-20}

git fetch -q origin
git checkout -q "${BRANCH}" 2>/dev/null || git checkout -q -b "${BRANCH}" origin/main
git pull -q --ff-only origin "${BRANCH}" 2>/dev/null || true
mkdir -p "${OUT}/cells"

# find, not ls: ls exits 2 on an unmatched glob, and under set -e -o pipefail that
# aborted the step at `last=$(cells)` whenever no cell existed yet.
cells() { find "${OUT}/cells" -maxdepth 1 -name 'cell_d*_s*.json' 2>/dev/null | wc -l | tr -d ' '; }
if [ "$(cells)" -ge "${N_CELLS}" ]; then echo "status: all ${N_CELLS} cells present"; touch "${OUT}/DONE"; git add "${OUT}/DONE"; git commit -qm "${NAME}: DONE" || true; git push -q origin "${BRANCH}"; exit 0; fi
if [ -f "${LOCK}" ]; then
  age=$(( ( $(date +%s) - $(cat "${LOCK}") ) / 60 ))
  if [ "${age}" -lt "${LOCK_MAX_MIN}" ]; then echo "status: live lock (${age} min), skipping"; exit 2; fi
fi
date +%s > "${LOCK}"; git add "${LOCK}"; git commit -qm "${NAME}: lock $(date -u +%FT%TZ)" || true; git push -q origin "${BRANCH}"
echo "status: $(date -u +%FT%TZ) start, $(cells) cells present"

export ITASORL_FOLDS=explicit
( timeout "$((MAX_MIN * 60))" python scripts/run_expB2.py --drift-mode l3 --l3-hidden 8 --l3-seed 0 \
    --drifts 0.0 0.45 --seeds 0 1 2 3 4 5 6 7 8 9 --updates 300 --workers 4 --device cpu \
    --save-agents --resume --out-dir "${OUT}" "$@" 2>&1 | tee -a "${OUT}/run.log" ) &
RUN=$!
last=$(cells)
while kill -0 "${RUN}" 2>/dev/null; do
  sleep 300
  now=$(cells)
  if [ "${now}" -gt "${last}" ]; then
    git add "${OUT}/cells" "${OUT}/expB2_results.json" "${OUT}/run.log" 2>/dev/null || true
    git commit -qm "${NAME}: ${now}/${N_CELLS} cells" || true
    git push -q origin "${BRANCH}" || true
    last=${now}
  fi
done
wait "${RUN}" || true
git rm -q --cached "${LOCK}" 2>/dev/null || true; rm -f "${LOCK}"
git add -A "${OUT}" ; git commit -qm "${NAME}: step end, $(cells)/${N_CELLS} cells" || true
git push -q origin "${BRANCH}"
if [ "$(cells)" -ge "${N_CELLS}" ]; then touch "${OUT}/DONE"; git add "${OUT}/DONE"; git commit -qm "${NAME}: DONE" || true; git push -q origin "${BRANCH}"; exit 0; fi
exit 3
