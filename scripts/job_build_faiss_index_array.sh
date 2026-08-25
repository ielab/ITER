#!/bin/bash
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --gres=gpu:1
#SBATCH --job-name=diver_faiss_shard
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Sharded FAISS encode as a SLURM array — NO pre-split files.
# Each task reads data/corpus.jsonl (memory-mapped from the shared Arrow cache,
# built once), takes its 1/NUM_SHARDS strided slice (balanced), and encodes it
# -> ${INDEX_DIR}/index-NNN.pkl  (NNN = SLURM_ARRAY_TASK_ID). faiss_searcher loads
# them all via the index-*.pkl glob. Normally launched by submit_faiss_index.sh
# (which warms the cache with shard 0 first). Set ENCODE_SORT=1 to also length-sort
# each slice (needs a tevatron with --encode_sort_by_length).
#
# Direct use (match --array count to NUM_SHARDS):
#   NUM_SHARDS=10 RETRIEVER=Qwen/Qwen3-Embedding-0.6B \
#   INDEX_DIR=${ITER_ROOT}/data/indexes/qwen3e_0.6b \
#   sbatch --array=0-9 scripts/job_build_faiss_index_array.sh

set -euo pipefail
ROOT=${ROOT:-${ITER_ROOT}}
CONDA_ENV=${CONDA_ENV:-${ITER_ROOT}/envs}
mkdir -p "${ROOT}/logs/indexing"

: "${SLURM_ARRAY_TASK_ID:?run me as an array job, e.g. sbatch --array=0-9 ...}"
NUM_SHARDS=${NUM_SHARDS:?set NUM_SHARDS (must equal the --array task count)}
RETRIEVER=${RETRIEVER:?set RETRIEVER}
INDEX_DIR=${INDEX_DIR:?set INDEX_DIR}
CORPUS_PATH=${CORPUS_PATH:-${ROOT}/data/corpus.jsonl}
SHARD=$(printf "%03d" "${SLURM_ARRAY_TASK_ID}")

RETRIEVER="${RETRIEVER}" \
CORPUS_PATH="${CORPUS_PATH}" \
INDEX_DIR="${INDEX_DIR}" \
OUT_PKL="${INDEX_DIR}/index-${SHARD}.pkl" \
NUM_SHARDS="${NUM_SHARDS}" \
SHARD_INDEX="${SLURM_ARRAY_TASK_ID}" \
ROOT="${ROOT}" CONDA_ENV="${CONDA_ENV}" \
bash "${ROOT}/scripts/build_faiss_index.sh"

echo "===== shard ${SHARD}/${NUM_SHARDS} done -> ${INDEX_DIR}/index-${SHARD}.pkl ====="
