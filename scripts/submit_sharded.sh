#!/bin/bash
# Submit ANY of the three e2e evaluations as N parallel shards + one aggregator.
#
#   KIND=bcp      SETTING=i6 PORT0=6170 NSHARD=10 bash .../submit_sharded.sh
#   KIND=infoseek SETTING=i6 PORT0=6180 NSHARD=6  bash .../submit_sharded.sh
#   KIND=backbone BACKBONE=gptoss-120b DATASET=bcp SETTING=i2 PORT0=6420 \
#                 NSHARD=5 GPUS=2 bash .../submit_sharded.sh
#
# Optional: RETRIEVER_DIR, IDX_NAME, OUT, DEP ("--dependency=afterok:NNN").
#
# WALLTIME: shards are sized so each finishes in ~1-2 h, so they request 2-4 h
# rather than the 12-24 h a whole-set job needs. Short requests schedule sooner
# -- an over-long --time is not free, it costs queue priority.
#
# COST: one GPU (or GPUS each) and one vLLM copy of the agent per shard, for the
# shard's duration. NSHARD=10 means 10 GPUs at once.
set -euo pipefail
ROOT=${ROOT:-${ITER_ROOT}}
KIND=${KIND:?set KIND=bcp|infoseek|backbone}
NSHARD=${NSHARD:-10}
PORT0=${PORT0:?set PORT0}
GPUS=${GPUS:-1}
# PARTITION=gpu is assumed to be a short-walltime GPU partition whose queue
# always drains within 2h -- far better than waiting behind 7-day jobs on `gpu`.
# Size NSHARD so each shard's UNDONE queries finish inside TIME minus the vLLM
# startup (~15-25 min), or the shard is killed mid-slice (its finished
# trajectories survive; the rest need another pass).
PARTITION=${PARTITION:-}
QOS=${QOS:-}
CPUS=${CPUS:-8}
# RAM must hold the whole HNSW graph: 48G (0.6B) / 110G (4B) / 174G (8B), plus the
# retriever and vLLM. The job file asks for 110G, which OOM-killed all 12 4B
# InfoSeek shards. Scale it with the index.
case "${IDX_NAME:-}" in
  *scale_8b*) MEM=${MEM:-360G} ;;
  *scale_4b*) MEM=${MEM:-240G} ;;
  *)          MEM=${MEM:-110G} ;;
esac
THREADS=${THREADS:-2}
cd "${ROOT}"

case "${KIND}" in
  bcp)
    SETTING=${SETTING:?}; JOB=scripts/job_bcp_ablation.sh
    OUT=${OUT:-${ROOT}/experiments/bcp/redesign_${SETTING}}
    GT=${ROOT}/datasets/browsecomp-plus.tsv; TIME=${TIME:-04:00:00}
    TAG=bcp_${SETTING}
    EXTRA="SETTING=${SETTING}" ;;
  infoseek)
    SETTING=${SETTING:?}; JOB=scripts/job_infoseek_ablation.sh
    OUT=${OUT:-${ROOT}/experiments/infoseek/ablation_${SETTING}}
    GT=${ROOT}/datasets/InfoSeek-Eval.tsv; TIME=${TIME:-03:00:00}
    TAG=is_${SETTING}
    EXTRA="SETTING=${SETTING}" ;;
  backbone)
    BACKBONE=${BACKBONE:?}; DATASET=${DATASET:?}; SETTING=${SETTING:?}
    JOB=scripts/job_backbone_eval.sh
    OUT=${OUT:-${ROOT}/experiments/backbone/${DATASET}_${BACKBONE}_${SETTING}}
    GT=$([ "${DATASET}" = bcp ] && echo "${ROOT}/datasets/browsecomp-plus.tsv" \
                                || echo "${ROOT}/datasets/InfoSeek-Eval.tsv")
    TIME=${TIME:-05:00:00}
    TAG=${DATASET:0:3}_${BACKBONE}_${SETTING}
    EXTRA="BACKBONE=${BACKBONE},DATASET=${DATASET},SETTING=${SETTING}" ;;
  *) echo "Unknown KIND=${KIND}"; exit 1 ;;
esac
GT=${GT_OVR:-${GT}}

IDS=()
for i in $(seq 0 $((NSHARD - 1))); do
  j=$(sbatch --parsable ${DEP:-} --time="${TIME}" --gres=gpu:${GPUS} \
      ${PARTITION:+--partition=${PARTITION}} ${QOS:+--qos=${QOS}} --cpus-per-task=${CPUS} --mem=${MEM} \
      --job-name="${TAG}_s${i}" \
      --export=ALL,${EXTRA},PORT=$((PORT0 + i)),SHARD=${i},NSHARD=${NSHARD},THREADS=${THREADS},OUT=${OUT}${RETRIEVER_DIR:+,RETRIEVER_DIR=${RETRIEVER_DIR}}${IDX_NAME:+,IDX_NAME=${IDX_NAME}} \
      "${JOB}")
  IDS+=("${j}")
done
echo "${TAG}: ${NSHARD} shards x ${GPUS} gpu x ${CPUS} cpu x ${MEM}, --time ${TIME} ${PARTITION:+-p ${PARTITION}} threads=${THREADS}"
echo "  budget: $((NSHARD * CPUS * ${TIME%%:*} * 60)) cpu-min (express cap 54000)"
echo "  ids: ${IDS[*]}"

DEPS=$(IFS=:; echo "${IDS[*]}")
AGG=$(sbatch --parsable --dependency=afterany:${DEPS} --job-name="${TAG}_agg" \
      --export=ALL,TRAJ_DIR=${OUT},GT=${GT},EVAL_OUT=${OUT}_eval.json \
      scripts/job_score_traj.sh)
echo "${TAG}: aggregator=${AGG} -> ${OUT}_eval.json"
