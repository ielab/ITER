# Trajectory Construction

Training data comes from watching an agent search. Tongyi-DeepResearch answers
InfoSeek training questions, and every search, every document it opens and
everything it writes afterwards is saved as one JSON file per question.

The **de-duplicated interface is used here and only here**. Each search
over-fetches 100 candidates, gives the 10 ranked positions to documents this
trajectory has not seen, and lists previously returned ones under
`returned earlier`, still openable with `get_document`. Evaluation always serves
an unfiltered ranking.

## 1. Serve the agent

```bash
vllm serve Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
  --port <PORT> \
  --trust-remote-code \
  --max-model-len 98304 \
  --gpu-memory-utilization <GPU_MEM_UTIL>
```

## 2. Run the agent

| Argument | Meaning |
| --- | --- |
| `--searcher-type` | `bm25` or `faiss` |
| `--index-path` | Lucene directory, or the FAISS index / shard glob |
| `--model-name` | query-side encoder (dense only) |
| `--dataset-name` | corpus used for document lookup (dense only) |
| `--task-prefix` | instruction prefix for the encoder |
| `--query` | TSV of questions |
| `--output-dir` | one run JSON per question |
| `--dedup-search`, `--dedup-pool-k` | the de-duplicated interface |
| `--num-shards`, `--shard-index` | split the query file across parallel jobs |

```bash
python src/search_agent/tongyi_client.py \
  --output-dir /path/to/runs/dedup_traj/qwen3-0.6b \
  --searcher-type faiss \
  --index-path /path/to/index/wiki/index.faiss \
  --model-name Qwen/Qwen3-Embedding-0.6B \
  --dataset-name /path/to/corpus.jsonl \
  --pooling eos --normalize --torch-dtype float16 \
  --task-prefix 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ' \
  --max-length 512 \
  --query /path/to/infoseekqa_train.tsv \
  --model Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
  --port <PORT> \
  --temperature 0.85 --top_p 0.95 --presence_penalty 1.1 \
  --num-threads <THREADS> \
  --snippet-max-tokens 64 --k 10 \
  --dedup-search --dedup-pool-k 100
```

Collection runs at temperature 0.85, hotter than the 0.6 used for evaluation,
to widen the candidate distribution.

The paper runs this four times, once per retrieval backend — BM25 and zero-shot
Qwen3-Embedding at 0.6B, 4B and 8B — over 10,000 questions, and keeps the 20,893
trajectories whose final answer matches the reference.

## Output

One JSON per question, holding the full ReAct transcript plus, for every search:
the sub-query, the query string actually embedded (`retrieval_query`), the
docids returned, the ones hidden by de-duplication, and which documents the
agent opened.
