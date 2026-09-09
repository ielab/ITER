# Evaluation

An evaluation arm is one agent backbone searching one benchmark through one
retriever. Every retriever is served the same unfiltered top-10 with 64-token
snippets and a 50-call tool budget, so no comparison is confounded by the
interface.

## 1. Serve the agent

```bash
python run_eval.py --setting i7 --backbone tongyi --benchmark bcp --print-server
```

prints the vLLM command for that backbone, including the flags it needs
(`--trust-remote-code` for Tongyi, `--gdn-prefill-backend triton` for Qwen3.5,
tensor parallelism for gpt-oss).

## 2. Run the arm

| Argument | Meaning |
| --- | --- |
| `--setting` | which retriever and how it is queried; see the table below |
| `--backbone` | `tongyi`, `qwen3.5-4b/9b/27b`, `qwen3.6-27b`, `gptoss-120b` |
| `--benchmark` | `infoseek` or `bcp` |
| `--index` | FAISS index for this retriever and corpus |
| `--retriever` | override the setting's checkpoint |
| `--searcher` | `faiss` or `bm25` |
| `--port` | where vLLM is serving |
| `--dedup` | serve the de-duplicated interface (off for every paper number) |
| `--dry-run` | print the command instead of running it |

```bash
python run_eval.py \
  --setting i7 --backbone tongyi --benchmark bcp \
  --index /path/to/index/i7_bcp/index-000.pkl \
  --port <PORT>
```

`--setting` resolves the checkpoint, the query style, the instruction prefix and
the query length together, from `src/config.py`. They must agree with how the
retriever was trained, which is why they are not separate flags.

Before the arm starts, `--index` is checked against the `encoder.json` written
when it was built (see [index_construction.md](index_construction.md)). An index
from a different encoder is refused rather than silently retrieving badly;
`--ignore-index-check` overrides.

| setting | what it evaluates |
| --- | --- |
| `i0` … `i7` | ITER with each query representation |
| `i7` | the paper's default |
| `i7-4b` | the 4B scaling run |
| `base` | untrained Qwen3-Embedding-0.6B |
| `lrat` | LRAT's released checkpoint |
| `agentir` | AgentIR-4B, served with the instruction from its model card |

BM25 is `--setting i0 --searcher bm25` with `--index` pointing at the Lucene
directory.

Each arm's questions are independent, so the query file can be split and run in
parallel. Output filenames are keyed on the question id and the clients skip ids
that already have a result, so parallel slices can share one `--out` directory
and a killed slice resumes where it stopped. Score only once every slice is done.

## 3. Score

**InfoSeek-Eval** is exact match, and `run_eval.py` scores it automatically:

```bash
python analysis/match_eval.py \
  --traj-dir /path/to/runs/<ARM> \
  --gt-path /path/to/InfoSeek-Eval.tsv \
  --output-file /path/to/runs/<ARM>_eval.json
```

**BrowseComp-Plus** uses its official LLM judge. `llm_judge_all.py` loads the
judge once and sweeps many arms, which matters — it is about 20 minutes per arm.

| Argument | Meaning |
| --- | --- |
| `--dirs` | `label=directory` pairs, one per arm |
| `--gt-path` | `browsecomp-plus.tsv` |
| `--qrel-path` | `qrel_evidence.txt`, for evidence recall |
| `--model-path` | the judge, `Qwen/Qwen3-30B-A3B-Thinking-2507` |
| `--output-dir` | per-arm verdicts and metrics |

```bash
python analysis/llm_judge_all.py \
  --dirs iter_i7=/path/to/runs/bcp_tongyi_i7 lrat=/path/to/runs/bcp_tongyi_lrat \
  --gt-path /path/to/browsecomp-plus.tsv \
  --qrel-path /path/to/qrel_evidence.txt \
  --model-path Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --output-dir /path/to/runs/judged \
  --tensor-parallel-size <NUM_GPUS> \
  --gpu-memory-utilization <GPU_MEM_UTIL> \
  --batch-size 512
```

Each output file holds `metrics` (task success rate, average steps, evidence
search and visit recall) and `details` (per-question verdicts).
