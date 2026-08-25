#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH --job-name=diver_faiss_index
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# cluster port of bash/job/job_build_faiss_index.sh — thin sbatch wrapper that
# delegates to scripts/build_faiss_index.sh.
#
# Usage (after `mkdir -p ${ITER_ROOT}/logs`):
#   RETRIEVER=Qwen/Qwen3-Embedding-4B \
#   INDEX_DIR=${ITER_ROOT}/data/indexes/qwen3e_4b \
#   sbatch scripts/job_build_faiss_index.sh

set -euo pipefail
ROOT=${ROOT:-${ITER_ROOT}}
CONDA_ENV=${CONDA_ENV:-${ITER_ROOT}/envs}
mkdir -p "${ROOT}/logs"

RETRIEVER=${RETRIEVER:?set RETRIEVER} \
CORPUS_PATH=${CORPUS_PATH:-${ROOT}/data/corpus.jsonl} \
INDEX_DIR=${INDEX_DIR:?set INDEX_DIR} \
ROOT="${ROOT}" CONDA_ENV="${CONDA_ENV}" \
bash "${ROOT}/scripts/build_faiss_index.sh"

echo "===== index job done -> ${INDEX_DIR} ====="
