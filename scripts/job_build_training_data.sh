#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --gres=gpu:1
#SBATCH --job-name=diver_build_train_data
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# cluster port of bash/job/job_build_training_data.sh
#   1) serve the relevance judge (Qwen3-30B-A3B-Thinking) via vLLM on this GPU
#   2) run scripts/build_training_data.sh over the 4 dedup traj sources
#   3) stop the judge
# Run AFTER all 4 dedup traj jobs in runs/dedup_traj/ have finished.
#
# Usage (after `mkdir -p ${ITER_ROOT}/logs`):
#   sbatch scripts/job_build_training_data.sh

set -euo pipefail
ROOT=${ROOT:-${ITER_ROOT}}
CONDA_ENV=${CONDA_ENV:-${ITER_ROOT}/envs}
PORT=${PORT:-6009}
# Judge: local dir (with config.json) if present, else HF repo id from the cache.
JUDGE_LOCAL=${ROOT}/models/Qwen3-30B-A3B-Thinking-2507
JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH:-$([[ -f "${JUDGE_LOCAL}/config.json" ]] && echo "${JUDGE_LOCAL}" || echo "Qwen/Qwen3-30B-A3B-Thinking-2507")}
mkdir -p "${ROOT}/logs" "${ROOT}/experiments/traj-aware/logs"

module load miniforge3 2>/dev/null || true
module load cuda
source activate "${CONDA_ENV}"
cd "${ROOT}"

export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=${TMPDIR:-${ROOT}/.jobtmp/${SLURM_JOB_ID:-$$}}
mkdir -p "${TMPDIR}"
export TORCHINDUCTOR_CACHE_DIR=${TMPDIR}/torchinductor
export TRITON_CACHE_DIR=${TMPDIR}/triton
export VLLM_CACHE_ROOT=${TMPDIR}/vllm

# --- 1) start vLLM (judge) in the background, same conda env ---
VLLM_LOG=${ROOT}/experiments/traj-aware/logs/vllm_judge_${SLURM_JOB_ID:-$$}.log
python -m vllm.entrypoints.openai.api_server \
  --model "${JUDGE_MODEL_PATH}" --served-model-name "${JUDGE_MODEL_PATH}" \
  --host 0.0.0.0 --port "${PORT}" \
  --dtype bfloat16 --gpu-memory-utilization 0.92 --max-model-len 32768 \
  --enable-prefix-caching \
  > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT

echo "waiting for judge vLLM on :${PORT} ..."
for i in $(seq 1 100); do
  if curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q '"id"'; then
    echo "judge ready after ${i} checks"; break
  fi
  if ! kill -0 ${VLLM_PID} 2>/dev/null; then echo "judge vLLM died; see ${VLLM_LOG}"; exit 1; fi
  sleep 15
done
curl -s "http://localhost:${PORT}/v1/models" | grep -q '"id"' || { echo "judge not ready"; exit 1; }

# --- 2) build training data against the judge ---
JUDGE_API_URL="http://127.0.0.1:${PORT}/v1/chat/completions" \
JUDGE_MODEL="${JUDGE_MODEL_PATH}" \
ROOT="${ROOT}" CONDA_ENV="${CONDA_ENV}" \
bash "${ROOT}/scripts/build_training_data.sh"

echo "build done; stopping judge"
kill ${VLLM_PID} 2>/dev/null || true
echo "===== training data built ====="
