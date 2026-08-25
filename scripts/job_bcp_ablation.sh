#!/bin/bash
#SBATCH --job-name=bcp_ablation
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# BrowseComp-Plus e2e (830 q) for ONE redesigned ablation retriever (i0..i5),
# Tongyi backbone, dedup harness. Mirrors job_infoseek_ablation.sh exactly —
# same retriever, query style, instruction and max-length trio — only the corpus
# (100K-doc BCP), the query file and the index differ.
# Run the BCP encode for the SAME setting first (job_bcp_encode_one.sh).
#
#   sbatch --export=ALL,SETTING=i4,PORT=6154 scripts/job_bcp_ablation.sh

set -euo pipefail
ROOT=${ROOT:-${ITER_ROOT}}
SETTING=${SETTING:?set SETTING=i0|i1|i2|i3|i4|i5}
MODEL=${MODEL:-Alibaba-NLP/Tongyi-DeepResearch-30B-A3B}
PORT=${PORT:-6150}
THREADS=${THREADS:-2}
MAX_TURN=${MAX_TURN:-50}

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
# The retriever shares this GPU with vLLM. 0.92 leaves ~6GB, which a 0.6B
# retriever fits in but a 4B/8B one does not (29764139 OOMed at 0.92).
case "${RETRIEVER_DIR:-}" in *scale_4b*|*AgentIR-4B*) GPU_UTIL=${GPU_UTIL:-0.78} ;; *scale_8b*) GPU_UTIL=${GPU_UTIL:-0.74} ;; esac
[ "${ONLINE:-0}" = "1" ] && GPU_UTIL=${GPU_UTIL:-0.78}

# ---- per-setting style trio (MUST mirror training; identical to the InfoSeek job) ----
case "${SETTING}" in
  i0) QUERY_STYLE=plain; MAX_LENGTH=512
      INSTRUCTION='Given a web search query, retrieve relevant passages that answer the query' ;;
  i1) QUERY_STYLE=i1; MAX_LENGTH=8192
      INSTRUCTION='Given the main question and the current sub-query, retrieve documents relevant to the current sub-query.' ;;
  i2) QUERY_STYLE=i2; MAX_LENGTH=8192
      INSTRUCTION='Given the main question, the current sub-query, and the sub-queries already tried in previous interactions, retrieve documents relevant to the current sub-query that provide NEW information not yet found.' ;;
  i3) QUERY_STYLE=i3; MAX_LENGTH=8192
      INSTRUCTION='Given the main question, the current sub-query, and previous interactions with the documents already visited, retrieve documents relevant to the current sub-query that provide NEW information beyond the visited documents.' ;;
  i4) QUERY_STYLE=i4; MAX_LENGTH=8192
      INSTRUCTION='Given the main question, the current sub-query, and previous interactions with the documents already visited and notes taken on them, retrieve documents relevant to the current sub-query that provide NEW information beyond the visited documents.' ;;
  i5) QUERY_STYLE=i5; MAX_LENGTH=8192
      INSTRUCTION='Given the main question, the current sub-query, and previous interactions with notes taken on the documents already read, retrieve documents relevant to the current sub-query that provide NEW information beyond what the notes cover.' ;;
  i6) QUERY_STYLE=i6; MAX_LENGTH=8192
      INSTRUCTION='Given the main question, the current sub-query, and previous interactions with the documents already visited, retrieve documents relevant to the current sub-query that provide NEW information beyond the visited documents.' ;;
  i7) QUERY_STYLE=i7; MAX_LENGTH=8192
      INSTRUCTION='Given the main question, the current sub-query, and previous interactions with the documents already visited and notes taken on them, retrieve documents relevant to the current sub-query that provide NEW information beyond the visited documents.' ;;
  i8)  # AgentIR format: pre-search reasoning + current subquery
    QUERY_STYLE=i8
    MAX_LENGTH=8192
    INSTRUCTION='Given the agent'"'"'s reasoning that led to the current sub-query, retrieve documents relevant to the current sub-query that provide NEW information beyond what the reasoning already covers.'
    ;;
  i9)  # i2 + Current Reasoning field
    QUERY_STYLE=i9
    MAX_LENGTH=8192
    INSTRUCTION='Given the main question, the agent'"'"'s reasoning and the current sub-query it led to, and the sub-queries already tried in previous interactions, retrieve documents relevant to the current sub-query that provide NEW information not yet found.'
    ;;
  *) echo "Unknown SETTING=${SETTING}"; exit 1 ;;
esac
# RETRIEVER_DIR / IDX_NAME let a scaled retriever reuse this job unchanged.
RETRIEVER=${RETRIEVER_DIR:-${ROOT}/models/ablation/input_${SETTING}}
INDEX_PATH="${ROOT}/data/indexes/${IDX_NAME:-bcp_redesign_${SETTING}}/index-000.pkl"
TASK_PREFIX=${TASK_PREFIX_OVR:-$'Instruct: '"${INSTRUCTION}"$'\nQuery: '}
CORPUS=${ROOT}/data/browse-comp-plus-corpus.jsonl
QUERY_FILE=${QUERY_FILE:-${ROOT}/datasets/topics-qrels/queries.tsv}
GT_FILE=${ROOT}/datasets/browsecomp-plus.tsv
OUT=${OUT:-${ROOT}/experiments/bcp/redesign_${SETTING}}
# SHARD/NSHARD: run only this job's slice of the query file (round-robin, so
# every shard gets a mix of easy/hard queries and they finish together). All
# shards write into the SAME ${OUT}; the clients skip query_ids that already
# have a run_*.json, so overlapping restarts are safe. Scoring is left to the
# aggregator job -- a per-shard match_eval would score 1/N of the set and
# overwrite the real ${OUT}_eval.json.
if [ -n "${NSHARD:-}" ]; then
  SHARD=${SHARD:?set SHARD=0..NSHARD-1 when NSHARD is set}
  SLICE=${TMPDIR:-/tmp}/queries_${SHARD}_of_${NSHARD}.tsv
  mkdir -p "$(dirname "${SLICE}")"
  awk -v s="${SHARD}" -v n="${NSHARD}" 'NR % n == s' "${QUERY_FILE}" > "${SLICE}"
  echo "shard ${SHARD}/${NSHARD}: $(wc -l < "${SLICE}") queries"
  QUERY_FILE=${SLICE}
fi

test -f "${INDEX_PATH}" || { echo "index not built yet: ${INDEX_PATH}"; exit 1; }
export TMPDIR=${TMPDIR:-/tmp/${SLURM_JOB_ID:-manual}}
mkdir -p "${TMPDIR}" "${ROOT}/logs/infoseek" "${OUT}"
export MAX_LLM_CALL_PER_RUN=${MAX_TURN}

VLLM_LOG=${ROOT}/logs/infoseek/vllm_bcp_redesign_${SETTING}_${SLURM_JOB_ID:-manual}.log
( python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --served-model-name "${MODEL}" \
    --host 0.0.0.0 --port "${PORT}" \
    --dtype bfloat16 --gpu-memory-utilization "${GPU_UTIL:-0.92}" --max-model-len 98304 \
    --enable-prefix-caching --trust-remote-code \
    > "${VLLM_LOG}" 2>&1 ) &
VLLM_PID=$!
cleanup() { kill "${VLLM_PID}" 2>/dev/null || true; wait "${VLLM_PID}" 2>/dev/null || true; }
trap cleanup EXIT

echo "Waiting for Tongyi-30B on port ${PORT}"
for i in $(seq 1 120); do
  curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q '"id"' && { echo "vLLM ready after ${i} checks"; break; }
  kill -0 "${VLLM_PID}" 2>/dev/null || { echo "vLLM died: ${VLLM_LOG}"; exit 1; }
  sleep 10
done
curl -s "http://localhost:${PORT}/v1/models" | grep -q '"id"'

DEDUP=${DEDUP_OVERRIDE:-1}
DEDUP_ARGS=(); [ "${DEDUP}" = "1" ] && DEDUP_ARGS=(--dedup-search --dedup-pool-k 100)
ONLINE_ARGS=()
if [ "${ONLINE:-0}" = "1" ]; then
  ONLINE_ARGS=(--online-update --online-lr "${ONLINE_LR:-1e-5}" --online-opt "${ONLINE_OPT:-adamw}" --online-lora-r "${ONLINE_LORA_R:-0}" --online-warmup-visits "${ONLINE_WARMUP:-0}" --online-gate "${ONLINE_GATE:-0}" --online-merge "${ONLINE_MERGE:-0}" --online-certainty "${ONLINE_CERT:-0}" --online-steps "${ONLINE_STEPS:-1}" --online-interp "${ONLINE_INTERP:-0}" --online-oracle-qrels "${ONLINE_ORACLE:-}" --online-oracle-mode "${ONLINE_ORACLE_MODE:-filter}")
  echo "ONLINE per-question TTA enabled: lr=${ONLINE_LR:-1e-5}"
fi
echo "dedup=${DEDUP}"
echo "BCP setting=${SETTING} retriever=${RETRIEVER} style=${QUERY_STYLE} max_len=${MAX_LENGTH}"
python src/search_agent/tongyi_client.py \
  --output-dir "${OUT}" \
  --searcher-type faiss \
  --index-path "${INDEX_PATH}" \
  --model-name "${RETRIEVER}" \
  --dataset-name "${CORPUS}" \
  --pooling eos --normalize --torch-dtype float16 \
  --task-prefix "${TASK_PREFIX}" \
  --max-length "${MAX_LENGTH}" \
  --query "${QUERY_FILE}" \
  --model "${MODEL}" \
  --port "${PORT}" \
  --temperature 0.6 --top_p 0.95 --presence_penalty 1.1 \
  --num-threads "${THREADS}" \
  --snippet-max-tokens 64 \
  --k 10 \
  --query-style "${QUERY_STYLE}" \
  ${DEDUP_ARGS[@]+"${DEDUP_ARGS[@]}"} \
  ${ONLINE_ARGS[@]+"${ONLINE_ARGS[@]}"}

if [ -n "${NSHARD:-}" ]; then
  echo "shard ${SHARD}/${NSHARD} done; scoring deferred to the aggregator"
else
  python analysis/match_eval.py \
    --traj-dir "${OUT}" \
    --gt-path "${GT_FILE}" \
    --output-file "${OUT}_eval.json" || echo "match_eval failed; score later"
fi

echo "===== bcp redesign ${SETTING} done -> ${OUT} ====="
