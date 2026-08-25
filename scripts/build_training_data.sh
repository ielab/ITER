#!/bin/bash
set -euo pipefail

# cluster port of scripts/build_training_data.sh
# Builds v1 ([Memory]) and v2 ([Prev]) diversity training data from the 4 dedup
# trajectory sources, then merges the parts. Requires a live LLM judge served on
# JUDGE_API_URL (the job_cluster wrapper serves it). Run AFTER all 4 dedup traj
# jobs in runs/dedup_traj/ have finished.
#   ROOT         : repo root      (default: ${ITER_ROOT})
#   CONDA_ENV    : conda env path (default: ${ITER_ROOT}/envs)
#   JUDGE_API_URL: judge endpoint (default: http://127.0.0.1:6009/v1/chat/completions)
#   JUDGE_MODEL  : judge model id (default: auto)

ROOT=${ROOT:-${ITER_ROOT}}
CONDA_ENV=${CONDA_ENV:-${ITER_ROOT}/envs}

module load miniforge3 2>/dev/null || true
module load cuda
source activate "${CONDA_ENV}"
cd "${ROOT}"

export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

CORPUS="${ROOT}/data/corpus.jsonl"
TRAJ_BASE="${ROOT}/runs/dedup_traj"
OUT_DIR="${ROOT}/experiments/traj-aware/training_data"
JUDGE_API_URL=${JUDGE_API_URL:-http://127.0.0.1:6009/v1/chat/completions}
JUDGE_MODEL=${JUDGE_MODEL:-auto}
PARTS="${OUT_DIR}/parts"
mkdir -p "${PARTS}"

for src in bm25 qwen3-0.6b qwen3-4b qwen3-8b; do
  echo "===== $(date '+%H:%M:%S') building ${src} ====="
  python -u src/data_builder.py \
    --corpus-path "${CORPUS}" \
    --traj-dir "${TRAJ_BASE}/${src}" \
    --output-v1 "${PARTS}/${src}.v1.jsonl" \
    --output-v2 "${PARTS}/${src}.v2.jsonl" \
    --tokenizer-path Qwen/Qwen3-Embedding-0.6B \
    --judge-api-url "${JUDGE_API_URL}" \
    --judge-model "${JUDGE_MODEL}" \
    --max-workers 16
done

echo "===== merging parts ====="
cat "${PARTS}/"*.v1.jsonl > "${OUT_DIR}/training_data_v1.jsonl"
cat "${PARTS}/"*.v2.jsonl > "${OUT_DIR}/training_data_v2.jsonl"
echo "v1 samples: $(wc -l < "${OUT_DIR}/training_data_v1.jsonl")"
echo "v2 samples: $(wc -l < "${OUT_DIR}/training_data_v2.jsonl")"
