#!/bin/bash
#SBATCH --job-name=judge_all_bcp
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# LLM-as-a-judge SR over EVERY complete BrowseComp-Plus run dir, in ONE vLLM
# session (llm_judge_all.py loads the judge once, not once per directory).
# BCP's official metric is judge-based, not string match -- match_eval.py numbers
# under-count paraphrases and are only used for fast iteration.
#
# Auto-discovers any dir with >= MIN_RUNS trajectories, so re-running after new
# arms land (v2 replication, 4B/8B) picks them up. Skips dirs already judged.
set -euo pipefail
ROOT=${ROOT:-${ITER_ROOT}}
CONDA_ENV=${CONDA_ENV:-${ROOT}/envs}
MIN_RUNS=${MIN_RUNS:-830}
OUTD=${OUTD:-${ROOT}/experiments/llm_judge_bcp}
JUDGE=${JUDGE:-Qwen/Qwen3-30B-A3B-Thinking-2507}

module load miniforge3 2>/dev/null || true
module load cuda
source activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME=${HF_HOME:-${HF_HOME}}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=${TMPDIR:-${ROOT}/.jobtmp/${SLURM_JOB_ID:-$$}}
mkdir -p "${TMPDIR}" "${OUTD}"
cd "${ROOT}"

DIRS=()
for d in experiments/bcp/*/ experiments/backbone/bcp_*/ experiments/dedup_ablation/bcp_*/ experiments/dedup_v2/bcp_*/; do
  [ -d "$d" ] || continue
  n=$(ls "$d" 2>/dev/null | grep -c '\.json$' || true)
  [ "$n" -ge "${MIN_RUNS}" ] || continue
  label=$(echo "${d%/}" | sed 's|experiments/||; s|/|__|g')
  [ -f "${OUTD}/${label}.json" ] && { echo "skip (judged): ${label}"; continue; }
  DIRS+=("${label}=${d%/}")
done
# Partition the un-judged arms across parallel jobs: arms are independent, so
# N jobs each load the judge once and take every Nth arm. One job over 29 arms
# ran ~21 min/arm (~10h); this is the difference between 10h and ~2h.
if [ -n "${JOB_TOTAL:-}" ]; then
  MINE=(); i=0
  for e in "${DIRS[@]}"; do
    [ $(( i % JOB_TOTAL )) -eq "${JOB_INDEX:?set JOB_INDEX with JOB_TOTAL}" ] && MINE+=("$e")
    i=$((i+1))
  done
  DIRS=("${MINE[@]}")
  echo "partition ${JOB_INDEX}/${JOB_TOTAL}: ${#DIRS[@]} arms"
fi
echo "judging ${#DIRS[@]} run dirs with ${JUDGE}"
printf '  %s\n' "${DIRS[@]}"
[ ${#DIRS[@]} -eq 0 ] && { echo "nothing to judge"; exit 0; }

python analysis/llm_judge_all.py \
  --gt-path "${ROOT}/datasets/browsecomp-plus.tsv" \
  --qrel-path "${ROOT}/datasets/topics-qrels/qrel_evidence.txt" \
  --model-path "${JUDGE}" \
  --dirs "${DIRS[@]}" \
  --output-dir "${OUTD}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --batch-size "${BATCH:-512}"

echo "===== judge done -> ${OUTD} ====="
