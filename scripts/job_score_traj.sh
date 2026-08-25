#!/bin/bash
#SBATCH --job-name=score_traj
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Score a trajectory dir once, after sharded runs have all written into it.
# Depends on afterany (not afterok) so a single dead shard still yields a score
# over whatever completed -- the printed count shows how much was actually run.
#   TRAJ_DIR, GT, EVAL_OUT
set -euo pipefail
ROOT=${ROOT:-${ITER_ROOT}}
TRAJ_DIR=${TRAJ_DIR:?set TRAJ_DIR}
GT=${GT:?set GT}
EVAL_OUT=${EVAL_OUT:?set EVAL_OUT}

module load miniforge3 2>/dev/null || true
source activate ${CONDA_ENV:-${ROOT}/envs}
export PYTHONNOUSERSITE=1
cd "${ROOT}"

echo "scoring ${TRAJ_DIR}: $(ls "${TRAJ_DIR}"/run_*.json 2>/dev/null | wc -l) trajectories"
python analysis/match_eval.py \
  --traj-dir "${TRAJ_DIR}" \
  --gt-path "${GT}" \
  --output-file "${EVAL_OUT}"
echo "===== scored -> ${EVAL_OUT} ====="
