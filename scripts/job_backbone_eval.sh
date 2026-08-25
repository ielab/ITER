#!/bin/bash
#SBATCH --job-name=backbone_eval
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=12:00:00  # shards override this with a much shorter --time
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Cross-backbone evaluation of ONE retriever setting on ONE dataset.
#   BACKBONE = tongyi | qwen3.5-4b | qwen3.5-9b | qwen3.5-27b | qwen3.6-27b | gptoss-20b | gptoss-120b
#   DATASET  = infoseek | bcp
#   SETTING  = i2 (redesigned winner) | i0 (bare-query baseline) | lrat (external baseline, no dedup)
#
#   sbatch --gres=gpu:2 --export=ALL,BACKBONE=gptoss-120b,DATASET=bcp,SETTING=i2,PORT=6201 \
#          scripts/job_backbone_eval.sh
#
# The retriever trio (index / query-style / instruction / max-length) must mirror
# training; the backbone only changes the agent, never the retrieval config.

set -euo pipefail
ROOT=${ROOT:-${ITER_ROOT}}
BACKBONE=${BACKBONE:?set BACKBONE}
DATASET=${DATASET:?set DATASET=infoseek|bcp}
SETTING=${SETTING:?set SETTING=i2|i0|i9|lrat}
PORT=${PORT:-6200}
THREADS=${THREADS:-2}
MAX_TURN=${MAX_TURN:-50}
GPU_UTIL=${GPU_UTIL:-0.92}

module load miniforge3
module load cuda
source activate ${CONDA_ENV:-${ROOT}/envs}
export PYTHONNOUSERSITE=1
# Offline: every vLLM start otherwise hits the HF Hub API to resolve the
# model. A chain launcher fires ~90 shards at once and the burst trips the
# 2500-req/5min quota -> "429 Too Many Requests" and the shard dies in 42s
# (killed 7/16 BCP shards on 2026-08-13). Models are already in HF_HOME.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export FAISS_ATTN_IMPL=${FAISS_ATTN_IMPL:-sdpa}
cd "${ROOT}"

# ---------- backbone -> model id, client, vLLM flags ----------
EXTRA_VLLM=""; CLIENT=""; MAXLEN_SRV=98304
case "${BACKBONE}" in
  tongyi)       MODEL=Alibaba-NLP/Tongyi-DeepResearch-30B-A3B; CLIENT=tongyi;  EXTRA_VLLM="--trust-remote-code" ;;
  qwen3.5-4b)   MODEL=Qwen/Qwen3.5-4B;   CLIENT=qwen35; MAXLEN_SRV=131072; EXTRA_VLLM="--gdn-prefill-backend triton" ;;
  qwen3.5-9b)   MODEL=Qwen/Qwen3.5-9B;   CLIENT=qwen35; MAXLEN_SRV=131072; EXTRA_VLLM="--gdn-prefill-backend triton" ;;
  qwen3.5-27b)  MODEL=Qwen/Qwen3.5-27B;  CLIENT=qwen35; MAXLEN_SRV=131072; EXTRA_VLLM="--gdn-prefill-backend triton" ;;
  qwen3.6-27b)  MODEL=Qwen/Qwen3.6-27B;  CLIENT=qwen35; MAXLEN_SRV=131072; EXTRA_VLLM="--gdn-prefill-backend triton" ;;
  qwen3.8-27b)  MODEL=Qwen/Qwen3.8-27B;  CLIENT=qwen35; MAXLEN_SRV=131072; EXTRA_VLLM="--gdn-prefill-backend triton" ;;
  gptoss-20b)   MODEL=openai/gpt-oss-20b;  CLIENT=gptoss; MAXLEN_SRV=131072 ;;
  gptoss-120b)  MODEL=openai/gpt-oss-120b; CLIENT=gptoss; MAXLEN_SRV=131072; GPU_UTIL=0.80
                # GPTOSS_TP=1: experimental single-GPU serving (63G MXFP4 weights +
                # ~9G retriever leaves ~8-10G KV => max-num-seqs 2, 98k ctx)
                if [ "${GPTOSS_TP:-2}" = "1" ]; then
                  MAXLEN_SRV=98304
                  EXTRA_VLLM="--max-num-seqs 2 --enforce-eager"
                else
                  EXTRA_VLLM="--tensor-parallel-size 2 --max-num-seqs 16 --enforce-eager --disable-custom-all-reduce"
                  export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=0 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo
                fi ;;
  *) echo "Unknown BACKBONE=${BACKBONE}"; exit 1 ;;
esac

# ---------- retriever setting (must mirror training) ----------
case "${SETTING}" in
  i2)   RETRIEVER=${ROOT}/models/ablation/input_i2; STYLE=i2; MAX_LENGTH=8192
        INSTRUCTION='Given the main question, the current sub-query, and the sub-queries already tried in previous interactions, retrieve documents relevant to the current sub-query that provide NEW information not yet found.'
        IDX_WIKI=ablation_i2_wiki; IDX_BCP=bcp_redesign_i2; DEDUP=1 ;;
  i0)   RETRIEVER=${ROOT}/models/ablation/input_i0; STYLE=plain; MAX_LENGTH=512
        INSTRUCTION='Given a web search query, retrieve relevant passages that answer the query'
        IDX_WIKI=abl_i1_wiki; IDX_BCP=bcp_redesign_i0; DEDUP=1 ;;
  lrat) RETRIEVER=Yuqi-Zhou/LRAT-Qwen3-Embedding-0.6B; STYLE=plain; MAX_LENGTH=512
        INSTRUCTION='Given a web search query, retrieve relevant passages that answer the query'
        IDX_WIKI=lrat_wiki; IDX_BCP=bcp_lrat; DEDUP=0 ;;
  i9)   RETRIEVER=${ROOT}/models/ablation/input_i9; STYLE=i9; MAX_LENGTH=8192
        INSTRUCTION='Given the main question, the agent'"'"'s reasoning and the current sub-query it led to, and the sub-queries already tried in previous interactions, retrieve documents relevant to the current sub-query that provide NEW information not yet found.'
        IDX_WIKI=ablation_i9_wiki; IDX_BCP=bcp_redesign_i9; DEDUP=1 ;;
  agentir)  # external Tevatron/AgentIR-4B — served with ITS card prefix (no trailing
            # space after "Query:"), so submit with TASK_PREFIX_OVR; dedupoff config.
        RETRIEVER=${ROOT}/models/AgentIR-4B; STYLE=i8; MAX_LENGTH=8192
        INSTRUCTION='Given a user'"'"'s reasoning followed by a web search query, retrieve relevant passages that answer the query while incorporating the user'"'"'s reasoning'
        IDX_WIKI=agentir4b_wiki; IDX_BCP=bcp_agentir4b; DEDUP=0 ;;
  *) echo "Unknown SETTING=${SETTING}"; exit 1 ;;
esac

# Overrides so an arbitrary retriever (e.g. the UNTRAINED Qwen3-Embedding-0.6B,
# whose wiki index already exists as qwen3e_0.6b) can reuse this job.
RETRIEVER=${RETRIEVER_DIR:-${RETRIEVER}}
IDX_WIKI=${IDX_WIKI_OVR:-${IDX_WIKI}}
IDX_BCP=${IDX_BCP_OVR:-${IDX_BCP}}

# ---------- dataset ----------
case "${DATASET}" in
  infoseek) CORPUS=${ROOT}/data/corpus.jsonl; QUERY_FILE=${ROOT}/datasets/InfoSeek-Eval.tsv
            GT=${QUERY_FILE}; INDEX_PATH="${ROOT}/data/indexes/${IDX_WIKI}/index.faiss" ;;
  bcp)      CORPUS=${ROOT}/data/browse-comp-plus-corpus.jsonl; QUERY_FILE=${ROOT}/datasets/topics-qrels/queries.tsv
            GT=${ROOT}/datasets/browsecomp-plus.tsv; INDEX_PATH="${ROOT}/data/indexes/${IDX_BCP}/index-000.pkl" ;;
  *) echo "Unknown DATASET=${DATASET}"; exit 1 ;;
esac
# SHARD/NSHARD: run only this job's round-robin slice of the query file. All
# shards share ${OUT}; the client skips query_ids that already have a
# run_*.json (filenames are keyed on query_id), so shards never collide and a
# killed shard resumes exactly where it stopped. Scoring is deferred to the
# aggregator -- a per-shard match_eval would score 1/N and overwrite the real
# ${OUT}_eval.json.
if [ -n "${NSHARD:-}" ]; then
  SHARD=${SHARD:?set SHARD=0..NSHARD-1 when NSHARD is set}
  SLICE=${TMPDIR:-/tmp}/qslice_${SHARD}_of_${NSHARD}.tsv
  mkdir -p "$(dirname "${SLICE}")"
  awk -v s="${SHARD}" -v n="${NSHARD}" 'NR % n == s' "${QUERY_FILE}" > "${SLICE}"
  echo "shard ${SHARD}/${NSHARD}: $(wc -l < "${SLICE}") queries"
  QUERY_FILE=${SLICE}
fi
QUERY_FILE=${QUERY_FILE_OVR:-${QUERY_FILE}}

# BM25 support: SEARCHER_TYPE=bm25 + INDEX_PATH_OVR=<lucene dir>. A Lucene index
# is a DIRECTORY, hence -e not -f.
[ "${ONLINE:-0}" = "1" ] && GPU_UTIL=${ONLINE_GPU_UTIL:-0.78}   # FORCE headroom for in-process training
ONLINE_ARGS=()
if [ "${ONLINE:-0}" = "1" ]; then
  ONLINE_ARGS=(--online-update --online-lr "${ONLINE_LR:-1e-5}" --online-opt "${ONLINE_OPT:-adamw}" --online-lora-r "${ONLINE_LORA_R:-0}" --online-warmup-visits "${ONLINE_WARMUP:-0}" --online-gate "${ONLINE_GATE:-0}" --online-merge "${ONLINE_MERGE:-0}" --online-certainty "${ONLINE_CERT:-0}" --online-steps "${ONLINE_STEPS:-1}" --online-interp "${ONLINE_INTERP:-0}")
  echo "ONLINE per-question TTA enabled: lr=${ONLINE_LR:-1e-5}"
fi
SEARCHER_TYPE=${SEARCHER_TYPE:-faiss}
INDEX_PATH=${INDEX_PATH_OVR:-${INDEX_PATH}}
test -e "${INDEX_PATH}" || { echo "index missing: ${INDEX_PATH}"; exit 1; }

TASK_PREFIX=${TASK_PREFIX_OVR:-$'Instruct: '"${INSTRUCTION}"$'\nQuery: '}
OUT=${OUT:-${ROOT}/experiments/backbone/${DATASET}_${BACKBONE}_${SETTING}}
export TMPDIR=${TMPDIR:-/tmp/${SLURM_JOB_ID:-manual}}
mkdir -p "${TMPDIR}" "${ROOT}/logs/infoseek" "${OUT}"
export MAX_LLM_CALL_PER_RUN=${MAX_TURN}
export LRAT_STRONG_PROMPT=1

VLLM_LOG=${ROOT}/logs/infoseek/vllm_${BACKBONE}_${DATASET}_${SETTING}_${SLURM_JOB_ID:-manual}.log
( python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --served-model-name "${MODEL}" \
    --host 0.0.0.0 --port "${PORT}" \
    --dtype bfloat16 --gpu-memory-utilization "${GPU_UTIL}" --max-model-len "${MAXLEN_SRV}" \
    --enable-prefix-caching ${EXTRA_VLLM} \
    > "${VLLM_LOG}" 2>&1 ) &
VLLM_PID=$!
cleanup() { kill "${VLLM_PID}" 2>/dev/null || true; wait "${VLLM_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo "Waiting for ${MODEL} on port ${PORT}"
for i in $(seq 1 240); do
  curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q '"id"' && { echo "vLLM ready after ${i} checks"; break; }
  kill -0 "${VLLM_PID}" 2>/dev/null || { echo "vLLM died: ${VLLM_LOG}"; exit 1; }
  sleep 10
done
curl -s "http://localhost:${PORT}/v1/models" | grep -q '"id"'

# DEDUP_OVERRIDE decouples the harness from the retriever so the 2x2
# (ITER|LRAT) x (dedup|no-dedup) can be run: the per-setting defaults above
# pair ITER with dedup and LRAT without, which confounds the two factors.
DEDUP=${DEDUP_OVERRIDE:-${DEDUP}}
DEDUP_ARGS=(); [ "${DEDUP}" = "1" ] && DEDUP_ARGS=(--dedup-search --dedup-pool-k 100)
COMMON=(--output-dir "${OUT}" --searcher-type "${SEARCHER_TYPE}" --index-path "${INDEX_PATH}"
        --query "${QUERY_FILE}" --model "${MODEL}"
        --num-threads "${THREADS}" --snippet-max-tokens 64 --k 10 --query-style "${STYLE}")
# dense-retriever args are rejected by the client when searcher-type=bm25
if [ "${SEARCHER_TYPE}" != "bm25" ]; then
  COMMON+=(--model-name "${RETRIEVER}" --dataset-name "${CORPUS}"
           --pooling eos --normalize --torch-dtype float16
           --task-prefix "${TASK_PREFIX}" --max-length "${MAX_LENGTH}")
fi

echo "backbone=${BACKBONE} dataset=${DATASET} setting=${SETTING} style=${STYLE} dedup=${DEDUP}"
case "${CLIENT}" in
  tongyi) python src/search_agent/tongyi_client.py "${COMMON[@]}" --port "${PORT}" \
            --temperature 0.6 --top_p 0.95 --presence_penalty 1.1 ${DEDUP_ARGS[@]+"${DEDUP_ARGS[@]}"} ${ONLINE_ARGS[@]+"${ONLINE_ARGS[@]}"} ;;
  qwen35) python src/search_agent/qwen35_client.py "${COMMON[@]}" --port "${PORT}" \
            ${DEDUP_ARGS[@]+"${DEDUP_ARGS[@]}"} ${ONLINE_ARGS[@]+"${ONLINE_ARGS[@]}"} ;;
  gptoss) export URL="http://localhost:${PORT}/v1" API_KEY="EMPTY"
          python src/search_agent/gptoss_responses_client.py "${COMMON[@]}" \
            --get-document --query-template QUERY_TEMPLATE --strong \
            --max-iterations "${MAX_TURN}" --reasoning-effort medium ${DEDUP_ARGS[@]+"${DEDUP_ARGS[@]}"} ;;
esac

if [ -n "${NSHARD:-}" ]; then
  echo "shard ${SHARD}/${NSHARD} done; scoring deferred to the aggregator"
else
  python analysis/match_eval.py --traj-dir "${OUT}" --gt-path "${GT}" \
    --output-file "${OUT}_eval.json" || echo "match_eval failed; score later"
fi
echo "===== ${DATASET} ${BACKBONE} ${SETTING} done -> ${OUT} ====="
