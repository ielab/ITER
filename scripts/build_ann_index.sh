#!/bin/bash
set -euo pipefail

# Build a prebuilt approximate FAISS index (HNSW by default) from the encoded
# .pkl shards, so the searcher does ms/query graph lookups instead of an exact
# flat scan. CPU-only (faiss-cpu); no GPU needed.
#   ROOT, CONDA_ENV  (defaults to the cluster paths)
#   SHARDS           glob of encode shards (required), e.g. ${INDEX_DIR}/index-*.pkl
#   OUT_DIR          where to write index.faiss + index.lookup.pkl (required)
#   INDEX_TYPE       faiss index_factory string (default HNSW32)
#   EF_CONSTRUCTION  HNSW build depth (default 200)
#   EF_SEARCH        HNSW query depth baked into the index (default 256)

ROOT=${ROOT:-${ITER_ROOT}}
CONDA_ENV=${CONDA_ENV:-${ITER_ROOT}/envs}
SHARDS=${SHARDS:?set SHARDS (glob of index-*.pkl)}
OUT_DIR=${OUT_DIR:?set OUT_DIR}
INDEX_TYPE=${INDEX_TYPE:-HNSW32}
EF_CONSTRUCTION=${EF_CONSTRUCTION:-200}
EF_SEARCH=${EF_SEARCH:-256}

module load miniforge3 2>/dev/null || true
module load cuda 2>/dev/null || true   # CPU partition (cpu) has no cuda module; not needed here
source activate "${CONDA_ENV}"
cd "${ROOT}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-16}}

python src/build_ann_index.py \
  --shards "${SHARDS}" \
  --out-dir "${OUT_DIR}" \
  --index-type "${INDEX_TYPE}" \
  --ef-construction "${EF_CONSTRUCTION}" \
  --ef-search "${EF_SEARCH}"

echo "===== ANN index done -> ${OUT_DIR}/index.faiss ====="
