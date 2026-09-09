# Training Data Construction

Two stages, deliberately separate: the positives and negatives come from what
the agent did and never change, while the query string depends on which
representation you are training. **One trajectory run therefore serves every
query representation.**

## 1. Positives and tiered negatives

A visit alone does not prove a document helped, so the reasoning the agent wrote
after reading it is passed to a verifier, and only documents judged relevant
become positives. Serve the verifier first:

```bash
vllm serve Qwen/Qwen3-30B-A3B-Thinking-2507 \
  --port <JUDGE_PORT> --gpu-memory-utilization <GPU_MEM_UTIL>
```

| Argument | Meaning |
| --- | --- |
| `--traj-dir` | one retrieval backend's trajectories |
| `--corpus-path` | corpus, for document text |
| `--output` | training groups |
| `--judge-api-url` | the verifier's chat-completions endpoint |
| `--max-workers` | concurrent verifier calls |

```bash
python -u src/data_builder.py \
  --corpus-path /path/to/corpus.jsonl \
  --traj-dir /path/to/runs/dedup_traj/<BACKEND> \
  --output /path/to/train_data/parts/<BACKEND>.jsonl \
  --tokenizer-path Qwen/Qwen3-Embedding-0.6B \
  --judge-api-url http://127.0.0.1:<JUDGE_PORT>/v1/chat/completions \
  --judge-model auto \
  --max-workers 16
```

Run once per backend, then concatenate the parts into one file.

Each group is one positive and nine negatives:

| tier | membership | weight |
| --- | --- | --- |
| redundancy (`div`) | visited before this search, judged relevant | 3.0, up to 3 |
| hard | visited before this search, judged irrelevant | 1.0, up to 3 |
| weak | returned now, never visited anywhere in the trajectory | 0.3, fills to 9 |

A document returned earlier and never opened is left **unlabelled** — not
opening something is not evidence that it is bad. The whole trajectory is
scanned before any labelling, so a document the agent opens later can never be
used as a negative earlier.

## 2. Render the query representation

| Argument | Meaning |
| --- | --- |
| `--v1` | the merged training groups from the previous step |
| `--traj-dir` | the same trajectories those groups were built from |
| `--corpus` | corpus, for document text |
| `--out` | one `training_data_<style>.jsonl` is written per style |
| `--backends` | subdirectories of `--traj-dir`, in the order `--v1` was built |
| `--styles` | which query styles to render (default: all) |

```bash
python src/rebuild_query_redesign.py \
  --v1 /path/to/train_data/training_data_v1.jsonl \
  --traj-dir /path/to/runs/dedup_traj \
  --corpus /path/to/corpus.jsonl \
  --out /path/to/train_data/redesign \
  --styles i0 i1 i2 i3 i4 i5 i6 i7
```

This rewrites each group's query string using `src/memory_utils.py` — the same
code that builds queries at inference time, so the training and serving inputs
are byte-identical. Output: `train_data/redesign/training_data_<style>.jsonl`.

The replay cannot rely on file order, so it rebuilds each candidate's **old**
query byte-for-byte and matches on `(query, positive, sorted negatives)`. Every
line of `--v1` must be claimed exactly once; if any is not, the script exits
non-zero rather than writing a misaligned training file.

Because the positives and negatives are copied across verbatim, the ablation
variants differ **only** in their query. Re-running `data_builder.py` per style
would re-run the verifier, and an LLM verifier does not return identical
verdicts twice — the variants would then differ in their labels too.

See [query_representations.md](query_representations.md) for what each style
contains.
