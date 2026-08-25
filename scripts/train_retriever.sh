#!/bin/bash
set -euo pipefail

# cluster port of scripts/train_retriever.sh
# Fine-tunes Qwen3-Embedding-0.6B with the diversity-aware tiered-negative loss
# (vendored FlagEmbedding). wandb runs offline; sync later with
# `wandb sync ${WANDB_DIR}`.
#   ROOT      : repo root      (default: ${ITER_ROOT})
#   CONDA_ENV : conda env path (default: ${ITER_ROOT}/envs)
#   NUM_GPUS  : torchrun procs (default: 1)
# Required env (set by job_train_retriever/input_ablation.sh):
#   TRAIN_DATA, OUTPUT_DIR, RUN_NAME, QUERY_INSTRUCTION,
#   NEG_W_DIV, NEG_W_HARD, NEG_W_WEAK, DIV_NEG_CAP, HARD_NEG_CAP
#   QUERY_MAX_LEN (default 512 -- the redesign settings need up to 8192)
#
# Reverting this file has broken training 3x. Besides the tier flags below,
# --gradient_checkpointing_kwargs use_reentrant=false is REQUIRED: reentrant
# checkpointing under DDP dies with "Expected to mark a variable ready only once".
#
# TIERED NEGATIVES (do not revert to the 2-tier NEG_W_SEEN/NEG_W_UNSEEN form):
# the vendored FlagEmbedding takes neg_w_div / neg_w_hard / neg_w_weak. Also do
# NOT add --negatives_cross_device: gathering negatives across ranks discards
# the per-tier weights, silently turning this back into a flat-negative run.

ROOT=${ROOT:-${ITER_ROOT}}
CONDA_ENV=${CONDA_ENV:-${ITER_ROOT}/envs}
NUM_GPUS=${NUM_GPUS:-1}

module load miniforge3 2>/dev/null || true
module load cuda
source activate "${CONDA_ENV}"
cd "${ROOT}"

export PYTHONNOUSERSITE=1  # env is self-contained; ~/.local packages break it
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

# 2-socket NUMA: interleave memory across both nodes for ~2x aggregate RAM
# bandwidth (matches the cluster training-harness convention).
# Derive the rendezvous port from the job id: two ablation jobs sharing a node
# both bound 29500 and the second died with EADDRINUSE.
MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID:-29500} % 20000) ))
if command -v numactl >/dev/null 2>&1; then
  LAUNCHER=(numactl --interleave=all torchrun --nproc_per_node "${NUM_GPUS}" --master_port "${MASTER_PORT}")
else
  LAUNCHER=(torchrun --nproc_per_node "${NUM_GPUS}" --master_port "${MASTER_PORT}")
fi

"${LAUNCHER[@]}" \
  -m FlagEmbedding.finetune.embedder.decoder_only.base \
  --model_name_or_path Qwen/Qwen3-Embedding-0.6B \
  --train_data "${TRAIN_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --query_max_len "${QUERY_MAX_LEN:-512}" \
  --passage_max_len 512 \
  --pad_to_multiple_of 8 \
  --query_instruction_for_retrieval "${QUERY_INSTRUCTION}" \
  --query_instruction_format 'Instruct: {}\nQuery: {}' \
  --train_group_size 10 \
  --per_device_train_batch_size 32 \
  --learning_rate 1e-6 \
  --num_train_epochs 2 \
  --warmup_ratio 0.1 \
  --bf16 \
  --gradient_checkpointing \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --save_strategy epoch \
  --save_steps 500 \
  --logging_steps 1 \
  --overwrite_output_dir \
  --dataloader_drop_last True \
  --same_dataset_within_batch True \
  --temperature 0.02 \
  --sentence_pooling_method last_token \
  --normalize_embeddings True \
  --neg_w_div "${NEG_W_DIV}" \
  --neg_w_hard "${NEG_W_HARD}" \
  --neg_w_weak "${NEG_W_WEAK}" \
  --div_neg_cap "${DIV_NEG_CAP}" \
  --hard_neg_cap "${HARD_NEG_CAP}" \
  --report_to wandb \
  ${RESUME_FROM_CHECKPOINT:+--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}"}

echo "===== training done -> ${OUTPUT_DIR} ====="
