#!/bin/bash
set -euo pipefail

# Driver (run on the LOGIN node, not via sbatch): submit a sharded FAISS encode.
# SHARDS = split count = number of jobs. Each task reads data/corpus.jsonl, takes
# its 1/SHARDS strided slice, and encodes -> index-NNN.pkl.
#
# Flow (WARM=1, the default for the first run):
#   1. a short CPU "warm" job parses corpus.jsonl into the shared Arrow cache once;
#   2. the encode array (ALL shards) starts after it (afterok) and runs CONCURRENTLY
#      — every task just memory-maps the cache, no "Generating train split".
# The cache is keyed by corpus.jsonl, so it is reused by EVERY model. After it
# exists, pass WARM=0 to skip the warm job and fire all shards immediately.
#
#   SHARDS=10 \
#   RETRIEVER=Qwen/Qwen3-Embedding-0.6B \
#   INDEX_DIR=${ITER_ROOT}/data/indexes/qwen3e_0.6b \
#   bash scripts/submit_faiss_index.sh
#
#   # later models, cache already warm:
#   SHARDS=16 WARM=0 RETRIEVER=...-4B INDEX_DIR=.../qwen3e_4b bash .../submit_faiss_index.sh
#
# Per-task time ~= (full-corpus encode time) / SHARDS. From a measured 0.6B run
# (batch 512, sdpa): ~12.8 h total -> ~1h20 per shard at SHARDS=10. Raise SHARDS
# (and/or BATCH_SIZE, and/or install flash-attn) to bring each task under your
# queue's fast-tier limit. Set TIME to comfortably exceed the real per-task time.
#
# Knobs: TIME (default 01:30:00), CORPUS_PATH, MODEL_TAG, WARM, plus anything
# build_faiss_index.sh reads (BATCH_SIZE, ATTN_IMPL, ENCODE_SORT).
# Load later with:  --index-path '${INDEX_DIR}/index-*.pkl'

ROOT=${ROOT:-${ITER_ROOT}}
CONDA_ENV=${CONDA_ENV:-${ITER_ROOT}/envs}
SHARDS=${SHARDS:?set SHARDS (number of shards = number of array tasks)}
RETRIEVER=${RETRIEVER:?set RETRIEVER}
INDEX_DIR=${INDEX_DIR:?set INDEX_DIR}
CORPUS_PATH=${CORPUS_PATH:-${ROOT}/data/corpus.jsonl}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${ROOT}/data/hf_cache}
TIME=${TIME:-01:30:00}
WARM=${WARM:-1}
JOBDIR=${ROOT}/scripts

if [[ "${SHARDS}" -lt 1 ]]; then
  echo "SHARDS must be >= 1." >&2
  exit 1
fi

MODEL_TAG=${MODEL_TAG:-$(basename "${INDEX_DIR}")}
LOGDIR=${LOGDIR:-${ROOT}/logs/indexing/${MODEL_TAG}}
mkdir -p "${LOGDIR}" "${INDEX_DIR}"
last=$((SHARDS - 1))

# DEBUG=1: warm the cache, then encode ONLY shard 0 (1/SHARDS of the corpus) to test
# the pipeline + measure it/s. Use a big SHARDS and a throwaway INDEX_DIR.
if [[ "${DEBUG:-0}" == "1" ]]; then
  ARRAY="0"
  echo "DEBUG: encoding ONLY shard 0 of ${SHARDS} (~1/${SHARDS} of the corpus) -> ${INDEX_DIR}"
else
  ARRAY="0-${last}"
fi

dep=""
if [[ "${WARM}" == "1" ]]; then
  wj=$(sbatch --parsable \
        --output="${LOGDIR}/warm-%j.txt" --error="${LOGDIR}/warm-%j-err.txt" \
        --export="ALL,ROOT=${ROOT},CONDA_ENV=${CONDA_ENV},CORPUS_PATH=${CORPUS_PATH},HF_DATASETS_CACHE=${HF_DATASETS_CACHE}" \
        "${JOBDIR}/job_warm_corpus_cache.sh")
  echo "warm cache   : ${wj}  (parses corpus once -> ${HF_DATASETS_CACHE})"
  dep="--dependency=afterok:${wj}"
fi

aj=$(sbatch --parsable ${dep} --array="${ARRAY}" --time="${TIME}" \
      --output="${LOGDIR}/split-%a.txt" --error="${LOGDIR}/split-%a-err.txt" \
      --export="ALL,ROOT=${ROOT},CONDA_ENV=${CONDA_ENV},RETRIEVER=${RETRIEVER},INDEX_DIR=${INDEX_DIR},CORPUS_PATH=${CORPUS_PATH},HF_DATASETS_CACHE=${HF_DATASETS_CACHE},NUM_SHARDS=${SHARDS}" \
      "${JOBDIR}/job_build_faiss_index_array.sh")
echo "encode array : ${aj}  (array=${ARRAY} of ${SHARDS}, time=${TIME} each)"
[[ -n "${dep}" ]] && echo "(array starts after warm ${wj}; all shards then memory-map the cache)"
echo "logs         : ${LOGDIR}/split-<n>.txt   ->  ${INDEX_DIR}/index-NNN.pkl"
echo "load later   : --index-path '${INDEX_DIR}/index-*.pkl'"
