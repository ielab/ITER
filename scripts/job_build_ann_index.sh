#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --gres=gpu:0
#SBATCH --partition=cpu
#SBATCH --job-name=diver_ann_index
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Build the prebuilt HNSW index for ONE model. CPU-only (faiss-cpu): --gres=gpu:0 +
# --partition=cpu keep it OFF the GPU nodes (omitting --gres alone still routes to a GPU
# partition on cluster). Reads index-*.pkl from INDEX_DIR and writes index.faiss +
# index.lookup.pkl there. Run once per model, after the FAISS encode array, before the
# faiss trajectory runs. Reused by traj + eval.
#
#   INDEX_DIR=${ITER_ROOT}/data/indexes/qwen3e_0.6b sbatch scripts/job_build_ann_index.sh
#
# SLOW: HNSW graph construction over 11.2M high-dim vectors is the cost — roughly
# ~3 h (0.6b), ~5-7 h (4b), ~10-12 h (8b) on 32 cores; hence --time=24:00:00. To go
# faster at a small recall cost, lower EF_CONSTRUCTION (e.g. 100) or M (INDEX_TYPE=HNSW16).
#
# RAM note: HNSW holds the full vectors, so bump --mem for the big-dim models:
#   0.6b ~50G (128g ok), 4b ~120G (use --mem=192g), 8b ~190G (use --mem=256g).
# Knobs: INDEX_TYPE (HNSW32), EF_CONSTRUCTION (200), EF_SEARCH (256).

set -euo pipefail
ROOT=${ROOT:-${ITER_ROOT}}
CONDA_ENV=${CONDA_ENV:-${ITER_ROOT}/envs}
INDEX_DIR=${INDEX_DIR:?set INDEX_DIR (the dir holding index-*.pkl)}
mkdir -p "${ROOT}/logs"

SHARDS="${INDEX_DIR}/index-*.pkl" \
OUT_DIR="${INDEX_DIR}" \
INDEX_TYPE=${INDEX_TYPE:-HNSW32} \
EF_CONSTRUCTION=${EF_CONSTRUCTION:-200} \
EF_SEARCH=${EF_SEARCH:-256} \
ROOT="${ROOT}" CONDA_ENV="${CONDA_ENV}" \
bash "${ROOT}/scripts/build_ann_index.sh"
