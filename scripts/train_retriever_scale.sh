#!/bin/bash
set -euo pipefail

# cluster port of scripts/train_retriever_4b.sh, generalized to any base model
# (4B / 8B). EXACT 0.6B diversity-aware recipe scaled to NPROC GPUs via
# DeepSpeed ZeRO-3 + CPU-offloaded optimizer (bash/config/ds_zero2.json).
#
# Identical to the 0.6B run: data, query format/instruction, 512/512, group 10,
# tier weights/caps, temperature 0.02, last_token pooling, lr 1e-6, warmup 0.1,
# 2 epochs, bf16 + grad checkpointing, negatives_cross_device OFF (each rank
# keeps its own 32x10 in-batch pool). Deviation (same as the colleague's 4B run):
# NPROC ranks x per-device batch 32 => effective batch 32*NPROC, steps shrink
# accordingly. Auto-resumes from the latest checkpoint if requeued.
#
# Required env: TRAIN_DATA, OUTPUT_DIR, RUN_NAME, QUERY_INSTRUCTION,
#   NEG_W_DIV, NEG_W_HARD, NEG_W_WEAK, DIV_NEG_CAP, HARD_NEG_CAP
# Optional: NUM_EPOCHS (2), BASE_MODEL (Qwen/Qwen3-Embedding-4B), NPROC (2)

ROOT=${ROOT:-${ITER_ROOT}}
CONDA_ENV=${CONDA_ENV:-${ITER_ROOT}/envs}
NUM_EPOCHS=${NUM_EPOCHS:-2}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-Embedding-4B}
# ---------------------------------------------------------------------------
# BATCH SIZING: EFF_BATCH is the primary knob, NOT per-device batch.
#
# Why: the number of optimizer steps is examples/EFF_BATCH, so a scale run at a
# different EFF_BATCH than the 0.6B ablation is NOT comparable -- it sees the
# same data but takes proportionally fewer updates at the same lr. The first
# 4b_i2 ran at EFF_BATCH=128 (32x2x2) => 1062 steps vs the 0.6B's 4244, i.e. a
# quarter of the updates, and had to be discarded.
#
# MICRO_BS is pinned at 32 because the in-batch negative pool is
# MICRO_BS x train_group_size (= 320); shrinking it changes the LOSS, not just
# the schedule, and is what collapsed the round-1 8B run.
#
# NPROC and GRAD_ACCUM are DERIVED, then asserted. Set EFF_BATCH (default 32 to
# match the 0.6B ablation); override NPROC only if you know it divides evenly.
# ---------------------------------------------------------------------------
MICRO_BS=${MICRO_BS:-32}
EFF_BATCH=${EFF_BATCH:-32}
if (( EFF_BATCH % MICRO_BS != 0 )); then
  echo "FATAL: EFF_BATCH=${EFF_BATCH} is not a multiple of MICRO_BS=${MICRO_BS}." >&2
  echo "       Lowering MICRO_BS would shrink the in-batch negative pool and change the loss." >&2
  exit 1
fi
FACTOR=$(( EFF_BATCH / MICRO_BS ))                 # = NPROC * GRAD_ACCUM
NPROC=${NPROC:-$(( FACTOR >= 2 ? 2 : 1 ))}         # 2 GPUs when it divides, else 1
if (( FACTOR % NPROC != 0 )); then
  echo "FATAL: NPROC=${NPROC} does not divide EFF_BATCH/MICRO_BS=${FACTOR}." >&2
  exit 1
fi
GRAD_ACCUM=$(( FACTOR / NPROC ))
ACTUAL=$(( MICRO_BS * NPROC * GRAD_ACCUM ))
if (( ACTUAL != EFF_BATCH )); then                 # belt-and-braces; must never fire
  echo "FATAL: derived effective batch ${ACTUAL} != requested ${EFF_BATCH}" >&2
  exit 1
fi
NEG_POOL=$(( MICRO_BS * ${TRAIN_GROUP_SIZE:-10} ))
echo "batch: EFF_BATCH=${EFF_BATCH} = MICRO_BS ${MICRO_BS} x NPROC ${NPROC} x GRAD_ACCUM ${GRAD_ACCUM}"
echo "       in-batch negative pool = ${NEG_POOL} (0.6B ablation: EFF_BATCH 32, pool 320)"
if (( MICRO_BS != 32 )); then
  echo "WARNING: MICRO_BS=${MICRO_BS} != 32 -> negative pool ${NEG_POOL} instead of 320."
  echo "         This CHANGES THE LOSS (InfoNCE denominator + in-batch tier weights),"
  echo "         so this run is not loss-identical to the 0.6B ablation. Recorded in RUN_NAME."
fi
MASTER_PORT=$(( 20000 + (${SLURM_JOB_ID:-29500} % 20000) ))

module load miniforge3 2>/dev/null || true
# cuda/12.9.1 = closest module to torch 2.10's cu128; DeepSpeed JIT-compiles its
# CPUAdam op on this node and needs nvcc. DS_SKIP_CUDA_CHECK tolerates the
# 12.8-vs-12.9 minor mismatch.
module load cuda/12.9.1
source activate "${CONDA_ENV}"
export PYTHONNOUSERSITE=1  # env is self-contained; ~/.local packages break it
export DS_SKIP_CUDA_CHECK=1
export TORCH_EXTENSIONS_DIR=${ROOT}/.cache/torch_extensions
mkdir -p "${TORCH_EXTENSIONS_DIR}"
cd "${ROOT}"

export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

# single-node multi-GPU NCCL settings validated on cluster by the gpt-oss-120b
# TP2 eval jobs
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

# activation-checkpoint boundaries dominate GPU memory at 352x512 tokens/step;
# expandable_segments reclaims allocator fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# auto-resume: pick up the latest checkpoint if a previous segment saved one
RESUME_ARGS=()
LATEST_CKPT=$(ls -d "${OUTPUT_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
if [ -n "${LATEST_CKPT}" ]; then
  echo "Resuming from ${LATEST_CKPT}"
  RESUME_ARGS=(--resume_from_checkpoint "${LATEST_CKPT}")
fi

torchrun --nproc_per_node "${NPROC}" --master_port "${MASTER_PORT}" \
  -m FlagEmbedding.finetune.embedder.decoder_only.base \
  --model_name_or_path "${BASE_MODEL}" \
  --deepspeed "${ROOT}/bash/config/ds_zero2.json" \
  --train_data "${TRAIN_DATA}" \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${RUN_NAME}" \
  --query_max_len "${QUERY_MAX_LEN:-512}" \
  --passage_max_len 512 \
  --pad_to_multiple_of 8 \
  --query_instruction_for_retrieval "${QUERY_INSTRUCTION}" \
  --query_instruction_format 'Instruct: {}\nQuery: {}' \
  --train_group_size 10 \
  --per_device_train_batch_size "${MICRO_BS}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --learning_rate 1e-6 \
  --num_train_epochs "${NUM_EPOCHS}" \
  --warmup_ratio 0.1 \
  --bf16 \
  --gradient_checkpointing \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --save_strategy steps \
  --save_steps 400 \
  --save_total_limit 2 \
  --logging_steps 1 \
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
  "${RESUME_ARGS[@]}"

echo "===== training done -> ${OUTPUT_DIR} ====="
