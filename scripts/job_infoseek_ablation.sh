#!/bin/bash
# ==================== ADJUST FOR YOUR CLUSTER (cluster) ====================
#SBATCH --job-name=infoseek_ablation
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=06:00:00  # shards override this with a much shorter --time
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# =========================================================================
#
# InfoSeek-Eval for ONE ablation retriever (redesigned i0..i5 suite), Tongyi
# backbone, dedup harness. Run job_build_index_ablation.sh for the SAME
# SETTING first.
#
#   sbatch --export=ALL,SETTING=i4 scripts/job_infoseek_ablation.sh
#
# The three style parameters MUST match how the model was trained
# (docs/query_input_redesign.md): --query-style <variant>, the variant's
# instruction as --task-prefix, and --max-length 8192 (full-history queries
# reach 6k tokens; 512 would truncate away everything the model learned from).
# i0 was trained on the bare sub-query -> --query-style plain + 512 suffice.

set -euo pipefail
ROOT=${ROOT:-${ITER_ROOT}}
SETTING=${SETTING:?set SETTING=i0|i1|i2|i3|i4|i5|i6|i7|i8|i9}
MODEL=${MODEL:-Alibaba-NLP/Tongyi-DeepResearch-30B-A3B}
PORT=${PORT:-6130}
THREADS=${THREADS:-2}
MAX_TURN=${MAX_TURN:-50}

module load miniforge3
module load cuda
source activate ${CONDA_ENV:-${ROOT}/envs}
# WITHOUT THIS vLLM SEGFAULTS AT STARTUP: ~/.local carries a pyarrow that wins
# over the env's. Cost a full round of i6/i7 + negative-ablation InfoSeek shards
# (all 36 died in 33s). job_bcp_ablation.sh and job_backbone_eval.sh already set it.
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
# true-online mode: leave VRAM headroom for query-encoder training
[ "${ONLINE:-0}" = "1" ] && GPU_UTIL=${GPU_UTIL:-0.78}

# ---- per-setting style trio (do not edit: must mirror training) ----
case "${SETTING}" in
  i0)
    QUERY_STYLE=plain
    MAX_LENGTH=512
    INSTRUCTION='Given a web search query, retrieve relevant passages that answer the query'
    ;;
  i1)
    QUERY_STYLE=i1
    MAX_LENGTH=8192
    INSTRUCTION='Given the main question and the current sub-query, retrieve documents relevant to the current sub-query.'
    ;;
  i2)
    QUERY_STYLE=i2
    MAX_LENGTH=8192
    INSTRUCTION='Given the main question, the current sub-query, and the sub-queries already tried in previous interactions, retrieve documents relevant to the current sub-query that provide NEW information not yet found.'
    ;;
  i3)
    QUERY_STYLE=i3
    MAX_LENGTH=8192
    INSTRUCTION='Given the main question, the current sub-query, and previous interactions with the documents already visited, retrieve documents relevant to the current sub-query that provide NEW information beyond the visited documents.'
    ;;
  i4)
    QUERY_STYLE=i4
    MAX_LENGTH=8192
    INSTRUCTION='Given the main question, the current sub-query, and previous interactions with the documents already visited and notes taken on them, retrieve documents relevant to the current sub-query that provide NEW information beyond the visited documents.'
    ;;
  i5)
    QUERY_STYLE=i5
    MAX_LENGTH=8192
    INSTRUCTION='Given the main question, the current sub-query, and previous interactions with notes taken on the documents already read, retrieve documents relevant to the current sub-query that provide NEW information beyond what the notes cover.'
    ;;
  i6)  # i3 with agent-view docs (64-token snippets, no tags)
    QUERY_STYLE=i6
    MAX_LENGTH=8192
    INSTRUCTION='Given the main question, the current sub-query, and previous interactions with the documents already visited, retrieve documents relevant to the current sub-query that provide NEW information beyond the visited documents.'
    ;;
  i7)  # i4 with agent-view docs; notes untagged, still 128
    QUERY_STYLE=i7
    MAX_LENGTH=8192
    INSTRUCTION='Given the main question, the current sub-query, and previous interactions with the documents already visited and notes taken on them, retrieve documents relevant to the current sub-query that provide NEW information beyond the visited documents.'
    ;;
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
# RETRIEVER_DIR / IDX_NAME let a scaled retriever (models/ablation/scale_4b_i2 +
# its own index) reuse this job unchanged; they default to the 0.6B ablation.
RETRIEVER=${RETRIEVER_DIR:-${ROOT}/models/ablation/input_${SETTING}}
# Prebuilt HNSW graph, NOT the raw index-*.pkl shards: the flat path loads every
# vector into RAM and OOMs this job's 110G at 11.2M docs.
INDEX_PATH="${ROOT}/data/indexes/${IDX_NAME:-ablation_${SETTING}_wiki}/index.faiss"
TASK_PREFIX=${TASK_PREFIX_OVR:-$'Instruct: '"${INSTRUCTION}"$'\nQuery: '}
CORPUS=${ROOT}/data/corpus.jsonl
QUERY_FILE=${QUERY_FILE_OVR:-${ROOT}/datasets/InfoSeek-Eval.tsv}
OUT=${OUT:-${ROOT}/experiments/infoseek/ablation_${SETTING}}

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

export TMPDIR=${TMPDIR:-/tmp/${SLURM_JOB_ID:-manual}}
mkdir -p "${TMPDIR}" "${ROOT}/logs/infoseek" "${OUT}"
export MAX_LLM_CALL_PER_RUN=${MAX_TURN}

VLLM_LOG=${ROOT}/logs/infoseek/vllm_ablation_${SETTING}_${SLURM_JOB_ID:-manual}.log
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

# dedup harness: default ON (historic behaviour); DEDUP_OVERRIDE=0 turns it off.
DEDUP=${DEDUP_OVERRIDE:-1}
DEDUP_ARGS=(); [ "${DEDUP}" = "1" ] && DEDUP_ARGS=(--dedup-search --dedup-pool-k 100)
echo "dedup=${DEDUP}"
ONLINE_ARGS=()
if [ "${ONLINE:-0}" = "1" ]; then
  # per-question test-time adaptation (fresh encoder copy per question)
  ONLINE_ARGS=(--online-update --online-lr "${ONLINE_LR:-1e-5}" --online-opt "${ONLINE_OPT:-adamw}" --online-lora-r "${ONLINE_LORA_R:-0}" --online-warmup-visits "${ONLINE_WARMUP:-0}" --online-gate "${ONLINE_GATE:-0}" --online-merge "${ONLINE_MERGE:-0}" --online-certainty "${ONLINE_CERT:-0}" --online-steps "${ONLINE_STEPS:-1}" --online-interp "${ONLINE_INTERP:-0}" --online-oracle-qrels "${ONLINE_ORACLE:-}")
  echo "ONLINE per-question TTA enabled: lr=${ONLINE_LR:-1e-5}"
fi
echo "setting=${SETTING}  retriever=${RETRIEVER}  query_style=${QUERY_STYLE}  max_length=${MAX_LENGTH}"
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
    --gt-path "${GT_FILE:-${QUERY_FILE}}" \
    --output-file "${OUT}_eval.json" || echo "match_eval failed; score later"
fi

echo "===== infoseek ablation ${SETTING} done -> ${OUT} ====="
