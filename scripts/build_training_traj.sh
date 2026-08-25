#!/bin/bash
set -euo pipefail

# cluster port of scripts/build_training_traj.sh
# Runs dedup trajectory generation with Tongyi-DeepResearch-30B-A3B.
# Assumes vLLM is already serving on $PORT (the job_cluster wrapper does this).
#   ROOT         : repo root      (default: ${ITER_ROOT})
#   CONDA_ENV    : conda env path (default: ${ITER_ROOT}/envs)
# Required env: SEARCHER_TYPE, INDEX_PATH, OUT   (FAISS also needs EMB_MODEL)

ROOT=${ROOT:-${ITER_ROOT}}
CONDA_ENV=${CONDA_ENV:-${ITER_ROOT}/envs}
# MODEL: a local dir (with config.json) if present, else the HF repo id resolved from
# the HF cache (offline). Override with MODEL=<local path or HF name>.
TONGYI_LOCAL=${ROOT}/models/Tongyi-DeepResearch-30B-A3B
MODEL=${MODEL:-$([[ -f "${TONGYI_LOCAL}/config.json" ]] && echo "${TONGYI_LOCAL}" || echo "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B")}
CORPUS=${CORPUS:-${ROOT}/data/corpus.jsonl}
QUERY_FILE=${QUERY_FILE:-${ROOT}/datasets/topics-qrels/infoseekqa_train.tsv}
PORT=${PORT:-6018}
THREADS=${THREADS:-4}
SEARCHER_TYPE=${SEARCHER_TYPE:?set SEARCHER_TYPE}
INDEX_PATH=${INDEX_PATH:?set INDEX_PATH}
OUT=${OUT:?set OUT}
# Query sharding: this run handles strided shard SHARD_INDEX of NUM_SHARDS.
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_INDEX=${SHARD_INDEX:-0}
mkdir -p "${OUT}"

module load miniforge3 2>/dev/null || true
module load cuda
source activate "${CONDA_ENV}"
cd "${ROOT}"

export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export MAX_LLM_CALL_PER_RUN=50
# FAISS query encoder attention: sdpa needs no flash_attn (set flash_attention_2 if
# installed). Only matters for the faiss backends (bm25 loads no embedding model).
export FAISS_ATTN_IMPL=${FAISS_ATTN_IMPL:-sdpa}
# faiss-cpu flat search is memory-bandwidth bound; give it all allocated cores.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}

FAISS_ARGS=()
if [[ "${SEARCHER_TYPE}" == "faiss" ]]; then
  : "${EMB_MODEL:?set EMB_MODEL for faiss}"
  FAISS_ARGS=(
    --model-name "${EMB_MODEL}"
    --dataset-name "${CORPUS}"
    --pooling eos --normalize --torch-dtype float16
    --task-prefix 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'
    --max-length 512
  )
fi

echo "retriever=${SEARCHER_TYPE}  out=${OUT}"
python src/search_agent/tongyi_client.py \
  --output-dir "${OUT}" \
  --searcher-type "${SEARCHER_TYPE}" \
  --index-path "${INDEX_PATH}" \
  ${FAISS_ARGS[@]+"${FAISS_ARGS[@]}"} \
  --query "${QUERY_FILE}" \
  --model "${MODEL}" \
  --port "${PORT}" \
  --temperature 0.85 --top_p 0.95 --presence_penalty 1.1 \
  --num-threads "${THREADS}" \
  --snippet-max-tokens 64 \
  --k 10 \
  --dedup-search --dedup-pool-k 100 \
  --num-shards "${NUM_SHARDS}" --shard-index "${SHARD_INDEX}"

echo "===== done -> ${OUT} (shard ${SHARD_INDEX}/${NUM_SHARDS}) ====="
