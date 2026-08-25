# ITER — Interaction-Aware Retrieval for Agentic Search

This repository contains the full reproduction package for **ITER**, an agent
interaction-aware dense retriever for deep-research agents. ITER conditions
each retrieval on the agent's interaction history (main question + current
sub-query + previous sub-queries) and is trained with trajectory-relative
supervision — positives from document visits, tiered redundancy / hard / weak
negatives — collected in a **de-duplicated** search setting.

**Contents**

1. [Naming: paper ↔ code mapping](#1-naming-paper--code-mapping)
2. [Repository layout](#2-repository-layout)
3. [Setup](#3-setup)
4. [Data](#4-data)
5. [Reproduction, step by step](#5-reproduction-step-by-step)
6. [Evaluating a trained checkpoint only](#6-evaluating-a-trained-checkpoint-only)
7. [Baselines](#7-baselines)
8. [Analysis figures](#8-analysis-figures)
9. [Implementation notes](#9-implementation-notes)

---

## 1. Naming: paper ↔ code mapping

The method and its query representations were **renamed for the paper**; the
code keeps its original internal flags. Use this table to translate:

| paper name | code flag (`--query-style` / `SETTING`) | query content |
|---|---|---|
| ITER-i0 | `plain` / `SETTING=i0` | bare current sub-query |
| ITER-i1 | `i1` | + main question |
| **ITER-i2** (default) | `i2` | + previous sub-queries |
| ITER-i3 | `i5` | i2 + post-visit notes |
| ITER-i4 | `i6` | i2 + visited-doc snippets (64 tok) |
| ITER-i5 | `i7` | i2 + snippets + notes |
| ITER-i6 | `i8` | AgentIR format: pre-search reasoning + sub-query |
| ITER-i7 | `i9` | i2 + pre-search reasoning |

Negative-tier naming: the paper's **redundancy** tier is the code's
`div`/diversity tier (`NEG_W_DIV`); **hard** and **weak** keep their names.
Full details: [`docs/query_representations.md`](docs/query_representations.md).

---

## 2. Repository layout

```
src/
  search_agent/        agent clients (Tongyi / Qwen3.x / gpt-oss) with the
                       de-duplicated search tool; prompts; ReAct loop
  searcher/            FAISS (HNSW + flat) and BM25 searchers
  memory_utils.py      builds every query representation — shared by training
                       data extraction AND inference (byte-identical)
  data_builder.py      (query, positive, tiered-negatives) training examples
  index_builder.py     corpus encoding entry point
  build_ann_index.py   HNSW index construction from encoded shards
scripts/               SLURM pipeline scripts (see step-by-step below)
third_party/
  FlagEmbedding/       vendored trainer, modified: tiered negative weights
                       (--neg_w_div/--neg_w_hard/--neg_w_weak) + reasoning-length
                       instance weights
  tevatron/            vendored corpus encoder
analysis/              evaluation (match_eval, llm_judge_all) + figure scripts
docs/                  query representations, setup notes
```

---

## 3. Setup

Python 3.11, CUDA GPUs (H100-class assumed by the batch sizes; scale down as
needed). From the repository root:

```bash
export ITER_ROOT=$(pwd)          # every script reads this
pip install uv && uv venv envs && source envs/bin/activate
uv pip install vllm transformers accelerate faiss-cpu peft matplotlib
uv pip install -e third_party/FlagEmbedding    # trainer (tiered-negative mods)
uv pip install -e third_party/tevatron         # corpus encoder
# flash-attn: install the wheel for your torch/CUDA, or export
# FAISS_ATTN_IMPL=sdpa to run everything without it.
# BM25 indexing additionally needs Java 21+ (pyserini).
mkdir -p logs                                   # SLURM job logs land here
```

The scripts are written for SLURM (`sbatch`); each has its resource requests in
its header — adjust partition names and `YOUR_ACCOUNT` for your cluster. Every
script can also be run directly with `bash` on a machine with the right GPUs.

Agent backbones are served locally with vLLM and loaded from `$HF_HOME`
(scripts set `HF_HUB_OFFLINE=1`; download models first):
`Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`, `Qwen/Qwen3-Embedding-0.6B` (and
`-4B` for scaling), `Qwen/Qwen3-30B-A3B-Thinking-2507` (verifier + judge), and
for cross-backbone evaluation `Qwen/Qwen3.5-4B/9B/27B`, `Qwen/Qwen3.6-27B`,
`openai/gpt-oss-120b`.

## 4. Data

Place under `$ITER_ROOT`:

| path | content |
|---|---|
| `data/corpus.jsonl` | Wiki-25 512-token chunk corpus (11.2M chunks) — HF: `Lk123/wiki-25-512` |
| `data/browse-comp-plus-corpus.jsonl` | BrowseComp-Plus corpus (100,195 docs) |
| `datasets/InfoSeek-Eval.tsv` | 300 InfoSeek evaluation questions (id, question, answer) |
| `datasets/topics-qrels/queries.tsv` | 830 BrowseComp-Plus questions |
| `datasets/browsecomp-plus.tsv` | BCP reference answers |
| `datasets/topics-qrels/qrel_golds.txt`, `qrel_evidence.txt` | BCP gold/evidence qrels (TREC format) |
| `datasets/topics-qrels/infoseekqa_train.tsv` | 10k InfoSeek **training** questions (disjoint from eval) |

## 5. Reproduction, step by step

Each step lists the command, what it produces, and roughly how long it takes on
H100s. `SETTING` below is the **code flag** (Section 1).

### Step 1 — Build retrieval indexes

Encode the wiki corpus with the base (or later, trained) encoder and build the
HNSW index; the BCP corpus is small enough for a single flat index.

```bash
# wiki: sharded encode (10 array tasks, ~1.5 h) -> data/indexes/<IDX>/index-*.pkl
RETRIEVER=Qwen/Qwen3-Embedding-0.6B INDEX_DIR=$ITER_ROOT/data/indexes/base_wiki \
  bash scripts/submit_faiss_index.sh
# HNSW32 build over the shards (CPU, ~3.5 h at 64 cores) -> index.faiss + index.lookup.pkl
sbatch --export=ALL,INDEX_DIR=$ITER_ROOT/data/indexes/base_wiki scripts/job_build_ann_index.sh
# BCP: single-shard encode (~30 min) -> data/indexes/bcp_<LABEL>/index-000.pkl
sbatch --export=ALL,LABEL=base,RETRIEVER=Qwen/Qwen3-Embedding-0.6B scripts/job_bcp_encode_one.sh
# BM25 (for trajectory collection): scripts/build_bm25_index.sh
```

### Step 2 — Collect de-duplicated trajectories

Run Tongyi-DeepResearch-30B on the 10k training questions against 4 retrieval
backends (BM25, Qwen3-Embedding 0.6B/4B/8B zero-shot), with the de-duplicated
search tool active:

```bash
# one launch per retrieval backend; override the retrieval config per backend:
sbatch scripts/job_traj_0.6b.sh                                   # 0.6B dense backend
sbatch --export=ALL,INDEX_PATH='...',EMB_MODEL='...',OUT='...' scripts/job_traj_0.6b.sh
```

Output: `runs/dedup_traj/<backend>/run_<qid>_*.json` (one file per question:
full ReAct trace + per-search returned docids + visits). Keep trajectories
whose final answer matches the reference — 20,893 of 40,000 in the paper.

### Step 3 — Build training data (two stages)

**3a. Judged groups from trajectories:**

```bash
sbatch scripts/job_build_training_data.sh
```

This serves the Qwen3-30B-A3B-Thinking verifier on a local vLLM and runs
`src/data_builder.py` over each backend's trajectories. At every search step
with a visited-and-verified-relevant document it emits one training group:

- **positive** — the visited document (relevance verified from the agent's
  post-visit reasoning); the reasoning length sets the instance weight
  (paper Eq. 1–2);
- **negatives** — up to 3 redundancy (visited earlier, relevant), up to 3 hard
  (visited earlier, irrelevant), weak (returned at this step, never visited in
  the whole trajectory) to fill 9.

Output: `experiments/traj-aware/training_data/training_data_v1.jsonl` (and a
`_v2` variant with an alternative legacy query field — the positives/negatives
are identical).

**3b. Render the query representation for your chosen style:**

```bash
python src/rebuild_query_redesign.py     # edit V1_PATH / TRAJ_BASE / OUT_DIR / style list at the top
```

This rewrites each group's query string into the selected `--query-style`
representations using `src/memory_utils.py` — the same code that builds queries
at inference, guaranteeing byte-identical train/serve inputs. Because
positives/negatives are fixed by the trajectories, **one trajectory run serves
every query style** — only the query string differs. Output:
`train_data/redesign/<style>.jsonl`.

### Step 4 — Train the retriever

```bash
TRAIN_DATA=training_data/i2.jsonl OUTPUT_DIR=models/iter_i2 \
  QUERY_MAX_LEN=8192 NUM_GPUS=1 bash scripts/train_retriever.sh
```

Hyperparameters (fixed across all variants): Qwen3-Embedding-0.6B backbone,
lr 1e-6 (AdamW, 0.1 warmup), batch 32, 2 epochs, bf16, last-token pooling,
normalized embeddings, `passage_max_len` 512, temperature 0.02, tier weights
`(3.0, 1.0, 0.3)`. `QUERY_MAX_LEN=8192` is required for history-conditioned
styles (`plain` fits in 512). For the 4B scaling experiment use
`scripts/train_retriever_scale.sh` (same effective batch → same step count).
~9 h (0.6B) / ~35 h (4B) on one H100.

> **Important**: `passage_max_len` (512) must match the index encoder's
> passage length, and the training `query_instruction` must be byte-identical
> to the instruction in the eval scripts' `SETTING` case block.

### Step 5 — Re-index with the trained encoder

Repeat Step 1 with `RETRIEVER=$ITER_ROOT/models/iter_i2` (wiki + BCP).

### Step 6 — End-to-end evaluation

```bash
# InfoSeek-Eval (300 q, exact match), sharded 12 ways:
KIND=infoseek SETTING=i2 PORT0=6100 NSHARD=12 bash scripts/submit_sharded.sh
# BrowseComp-Plus (830 q), sharded 16 ways:
KIND=bcp SETTING=i2 PORT0=6200 NSHARD=16 bash scripts/submit_sharded.sh
```

`submit_sharded.sh` slices the query file round-robin, runs one vLLM+retriever
job per shard (shards share the output dir and skip already-finished query ids,
so restarts are safe), and queues an aggregator that scores the complete set
with `analysis/match_eval.py` → `<out>_eval.json`. Serving config: top-10
results, 64-token snippets, 50 tool-call turns, seed 2026, temp 0.6, top_p 0.95.

Cross-backbone transfer (five unseen agents):

```bash
KIND=backbone BACKBONE=qwen3.5-27b DATASET=bcp SETTING=i2 PORT0=6300 NSHARD=16 \
  bash scripts/submit_sharded.sh
# BACKBONE ∈ {qwen3.5-4b, qwen3.5-9b, qwen3.5-27b, qwen3.6-27b, gptoss-120b}
# gptoss-120b needs GPUS=2 per shard (or GPTOSS_TP=1 for single-GPU serving).
```

### Step 7 — Judge BrowseComp-Plus (official metric)

```bash
sbatch scripts/job_judge_all_bcp.sh
```

Sweeps every complete (≥830-run) BCP output directory with
Qwen3-30B-A3B-Thinking-2507 and writes `metrics` + per-query `details`
(verdict, reason, evidence recall) per arm. InfoSeek uses exact match — no
judge needed.

## 6. Evaluating a trained checkpoint only

If you only want to evaluate (skipping Steps 2–4): point
`RETRIEVER_DIR=<checkpoint>` at Step 5's indexing and Step 6's eval scripts —
`<RELEASED_ITER_0.6B_CHECKPOINT>` placeholders mark where released checkpoints
will slot in upon publication.

## 7. Baselines

| baseline | how |
|---|---|
| BM25 | `SETTING=i0` with `SEARCHER_TYPE=bm25` + the Lucene index |
| Base (untrained) | `SETTING=i0` with `RETRIEVER_DIR=Qwen/Qwen3-Embedding-0.6B` |
| LRAT | `SETTING=lrat` in `scripts/job_backbone_eval.sh` (public checkpoint, plain sub-query, no dedup) |
| AgentIR-4B | `SETTING=agentir` — served exactly per its model card: its own instruction prefix (`TASK_PREFIX_OVR`), `Reasoning:`/`Query:` input (= code flag `i8`), last-token pooling, its own indexes |

## 8. Analysis figures

After BCP arms are complete:

```bash
python analysis/plot_per_step_recall_retrievers.py    # cumulative search/visit recall per step
python analysis/plot_per_search_effectiveness.py      # per-search precision / nDCG@10
python analysis/retriever_recall_bcp.py               # retriever-only recall (no agent)
```

Edit the `ARMS` dict at the top of each script to point at your run
directories.

## 9. Implementation notes

- **Three-places-identical rule.** The query representation is constructed by
  `src/memory_utils.py` and must be byte-identical in training-data
  extraction, the training `query_instruction`, and inference. When adding a
  representation, change all three together.
- **De-duplicated search tool** (`src/search_agent/*_utils/tool_search.py`).
  Each search over-fetches K=100 candidates, reserves the k=10 ranked
  positions for first-time results, and lists previously returned documents
  under a `returned earlier` section that remains openable via
  `get_document`. This is both the deployment setting and the collection
  setting for training data.
- **Search tool input** is a single query string; agents that emit query
  arrays have only the first element served.
- **gpt-oss-120B serving**: default `--tensor-parallel-size 2` (2 GPUs per
  shard); `GPTOSS_TP=1` serves on one GPU (98k context, max 2 sequences) at
  ~70% throughput.
- Unless noted otherwise, all comparisons run with the dedup harness OFF
  (head-to-head retriever condition); the dedup-ON deployment results use
  `DEDUP_OVERRIDE=1`.

## License

MIT (see `LICENSE`). Vendored `third_party/` projects retain their upstream
licenses.
