#!/bin/bash
#SBATCH --job-name=bcp_encode_one
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Encode BCP corpus with ONE retriever (LABEL + RETRIEVER env).
set -euo pipefail
LABEL=${LABEL:?set LABEL}; RETRIEVER=${RETRIEVER:?set RETRIEVER}; BATCH=${BATCH:-256}
module load miniforge3 2>/dev/null || true
module load cuda
source activate ${ITER_ROOT}/envs
export PYTHONNOUSERSITE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=${ITER_ROOT}/data/hf_cache HF_HOME=${HF_HOME}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ${ITER_ROOT}
mkdir -p data/indexes/bcp_${LABEL}
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
  --model_name_or_path "${RETRIEVER}" \
  --dataset_path data/browse-comp-plus-corpus.jsonl \
  --encode_output_path data/indexes/bcp_${LABEL}/index-000.pkl \
  --passage_max_len 512 --normalize --pooling eos --passage_prefix "" \
  --attn_implementation sdpa --per_device_eval_batch_size "${BATCH}" \
  --dataloader_num_workers 4 --padding_side left --fp16
echo "===== bcp encode ${LABEL} done ====="
