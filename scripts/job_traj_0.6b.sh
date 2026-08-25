#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128g
#SBATCH --gres=gpu:1
#SBATCH --job-name=diver_dedup_0.6b
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# cluster port of bash/job/job_traj_0.6b.sh
#   1) serve Tongyi-DeepResearch-30B-A3B via vLLM on this GPU
#   2) run scripts/build_training_traj.sh (FAISS qwen3-0.6b retriever)
#   3) stop vLLM
#
# Usage (after `mkdir -p ${ITER_ROOT}/logs`):
#   sbatch scripts/job_traj_0.6b.sh

set -euo pipefail
ROOT=${ROOT:-${ITER_ROOT}}
CONDA_ENV=${CONDA_ENV:-${ITER_ROOT}/envs}
TONGYI_LOCAL=${ROOT}/models/Tongyi-DeepResearch-30B-A3B
MODEL=${MODEL:-$([[ -f "${TONGYI_LOCAL}/config.json" ]] && echo "${TONGYI_LOCAL}" || echo "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B")}
PORT=${PORT:-6018}
mkdir -p "${ROOT}/logs" "${ROOT}/runs/dedup_traj/logs"

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

VLLM_LOG=${ROOT}/runs/dedup_traj/logs/vllm_0.6b_${SLURM_JOB_ID:-$$}.log
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" --served-model-name "${MODEL}" \
  --host 0.0.0.0 --port "${PORT}" \
  --dtype bfloat16 --gpu-memory-utilization 0.9 --max-model-len 98304 \
  --enable-prefix-caching \
  > "${VLLM_LOG}" 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT

echo "waiting for vLLM on :${PORT} ..."
for i in $(seq 1 100); do
  if curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q '"id"'; then
    echo "vLLM ready after ${i} checks"; break
  fi
  if ! kill -0 ${VLLM_PID} 2>/dev/null; then echo "vLLM died; see ${VLLM_LOG}"; exit 1; fi
  sleep 15
done
curl -s "http://localhost:${PORT}/v1/models" | grep -q '"id"' || { echo "vLLM not ready"; exit 1; }

SEARCHER_TYPE=faiss \
INDEX_PATH=${INDEX_PATH:-${ROOT}/data/indexes/qwen3e_0.6b/index-*.pkl} \
EMB_MODEL=${EMB_MODEL:-Qwen/Qwen3-Embedding-0.6B} \
OUT=${OUT:-${ROOT}/runs/dedup_traj/qwen3-0.6b} \
THREADS=${THREADS:-4} \
PORT="${PORT}" MODEL="${MODEL}" ROOT="${ROOT}" CONDA_ENV="${CONDA_ENV}" \
bash "${ROOT}/scripts/build_training_traj.sh"

kill ${VLLM_PID} 2>/dev/null || true
