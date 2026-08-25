#!/bin/bash
set -euo pipefail

# cluster port of scripts/build_faiss_index.sh
# Encodes the corpus into a dense embedding .pkl shard with tevatron. The FAISS
# flat index is built by the searcher when it loads these shards.
#   ROOT        : repo root      (default: ${ITER_ROOT})
#   CONDA_ENV   : conda env path (default: ${ITER_ROOT}/envs)
#   RETRIEVER   : path or HF name of the embedding model (required)
#                 e.g. Qwen/Qwen3-Embedding-0.6B  (baseline)
#                      ${ROOT}/models/LRAT-Qwen3-Embedding-0.6B  (LRAT)
#   CORPUS_PATH : corpus JSONL to encode (default: ${ROOT}/data/corpus.jsonl)
#   INDEX_DIR   : output directory for index-NNN.pkl (required)
#   OUT_PKL     : output shard file (default: ${INDEX_DIR}/index-000.pkl)
#   NUM_SHARDS  : if >1, encode only shard SHARD_INDEX of the corpus (tevatron
#                 strided shard) and length-sort it. No pre-split files needed;
#                 each task reads corpus.jsonl directly and takes its 1/N rows.
#   SHARD_INDEX : which shard this task encodes (required when NUM_SHARDS>1)

ROOT=${ROOT:-${ITER_ROOT}}
CONDA_ENV=${CONDA_ENV:-${ITER_ROOT}/envs}

module load miniforge3 2>/dev/null || true
module load cuda
source activate "${CONDA_ENV}"
cd "${ROOT}"

export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
# Shared, persistent Arrow cache: the corpus is parsed ONCE; every shard task then
# memory-maps it (no per-task "Generating train split"). Keep this path stable
# across tasks/models so the cache is reused.
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${ROOT}/data/hf_cache}
mkdir -p "${HF_DATASETS_CACHE}"
# Reduce fragmentation OOM (the encode allocates/frees large activation buffers).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RETRIEVER=${RETRIEVER:?set RETRIEVER}
CORPUS_PATH=${CORPUS_PATH:-${ROOT}/data/corpus.jsonl}
INDEX_DIR=${INDEX_DIR:?set INDEX_DIR}
# Output pkl name; the array job overrides this per shard (index-NNN.pkl).
OUT_PKL=${OUT_PKL:-${INDEX_DIR}/index-000.pkl}
DL_WORKERS=${DL_WORKERS:-8}
# Per-model batch defaults (512-token passages). The encode is BATCH-bound, not
# compute-bound: at batch 512 a 0.6B model runs the H100 at ~15% util, so bigger
# batches are the main speedup. These are safe under sdpa; raise BATCH_SIZE while
# watching nvidia-smi to fill 96 GB (with flash-attn you can go ~2x higher again).
if [[ -z "${BATCH_SIZE:-}" ]]; then
  case "${RETRIEVER}" in
    *0.6B*|*0.6b*) BATCH_SIZE=1024 ;;
    *4B*|*4b*)     BATCH_SIZE=256  ;;
    *8B*|*8b*)     BATCH_SIZE=128  ;;
    *)             BATCH_SIZE=256  ;;
  esac
fi
# Attention impl. sdpa (PyTorch built-in) needs no flash_attn and dispatches to a
# fused flash kernel on H100 fp16 -- not "much slower" per op. flash_attention_2 is
# more memory-efficient, so it allows ~2x larger batches (the real throughput win
# here); use it if flash_attn is installed. eager is the slow last resort.
ATTN_IMPL=${ATTN_IMPL:-sdpa}

# Optional: encode just one strided shard of the corpus. Lets N array tasks split
# the corpus with no pre-split files.
#   ENCODE_SORT=1 adds --encode_sort_by_length (the in-shard length sort that
#   minimises padding) -- needs a tevatron that has the flag (ITER's vendored one).
#   Default 0, because `import tevatron` may resolve to another checkout without it;
#   strided sharding alone still works everywhere.
NUM_SHARDS=${NUM_SHARDS:-1}
ENCODE_SORT=${ENCODE_SORT:-0}
SHARD_ARGS=()
if [[ "${NUM_SHARDS}" -gt 1 ]]; then
  SHARD_ARGS=(
    --dataset_number_of_shards "${NUM_SHARDS}"
    --dataset_shard_index "${SHARD_INDEX:?set SHARD_INDEX when NUM_SHARDS>1}"
  )
  [[ "${ENCODE_SORT}" == "1" ]] && SHARD_ARGS+=(--encode_sort_by_length)
fi

echo "encode | model=${RETRIEVER} batch_size=${BATCH_SIZE} dl_workers=${DL_WORKERS} attn=${ATTN_IMPL} shard=${SHARD_INDEX:-0}/${NUM_SHARDS}"
mkdir -p "$INDEX_DIR"

CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
  --model_name_or_path "$RETRIEVER" \
  --dataset_path "$CORPUS_PATH" \
  --encode_output_path "$OUT_PKL" \
  --passage_max_len 512 \
  --normalize \
  --pooling eos \
  --passage_prefix "" \
  --attn_implementation "$ATTN_IMPL" \
  --per_device_eval_batch_size "$BATCH_SIZE" \
  --dataloader_num_workers "$DL_WORKERS" \
  --padding_side left \
  --fp16 \
  ${SHARD_ARGS[@]+"${SHARD_ARGS[@]}"}

echo "===== FAISS shard done -> ${OUT_PKL} ====="
