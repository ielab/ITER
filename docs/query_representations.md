# Query representations

Every representation is built by `src/memory_utils.py` and must be
byte-identical in three places: (1) training-data extraction from trajectories,
(2) the training `query_instruction`, and (3) live inference. All three call
that one function, and `run_train.py` / `run_eval.py` resolve the instruction
from it rather than carrying a copy.

## The ladder

| style | query contains | what it adds |
|---|---|---|
| `i0` | current sub-query | bare baseline (what LRAT gets) |
| `i1` | + main question | the overall task |
| **`i2`** | + previous sub-queries (numbered, oldest→newest) | **the default** — which directions were already explored |
| `i3` | i2 + visited-document snippets (64-token, untagged) | document text as memory |
| `i4` | i2 + post-visit notes per interaction | what was learned from what was read |
| `i5` | i2 + snippets + notes | both memory encodings |
| `i6` | pre-search reasoning + sub-query (AgentIR's format) | the agent's live intent, AgentIR packaging |
| `i7` | i2 + a `Current Reasoning:` line before `Current Subquery:` | reasoning added to the structured history |

These are the paper's numbers. There is no second numbering.

## Format rules

- Natural-language field names (`Main Question:`, `Current Subquery:`,
  `Previous Interactions:` …); history grouped per interaction (one search plus
  its visits), numbered oldest → newest.
- Empty fields render as `<empty>`; every field is one-lined.
- `i3`/`i5` show documents the way the agent saw them — the 64-token search
  snippet, untagged. `i4` carries notes only, tagged with `[docs_id:x]`, at 128
  tokens.
- The pre-search reasoning in `i6`/`i7` is the `<think>` of the turn that issues
  the search (digesting prior results and stating intent) — **not** the
  post-visit note used by `i4`/`i5`.
- `i6` uses AgentIR's raw template; `i7` injects the same signal into `i2`'s
  structure. So `i7` vs `i2` isolates the reasoning signal, and `i6` vs `i7`
  isolates the packaging plus the history.
- The off-the-shelf AgentIR-4B baseline is served with AgentIR's **own**
  instruction, not ours; see `config.SETTINGS["agentir"]`.

## Negative tiers

| paper name | code name / flag | members | weight |
|---|---|---|---|
| redundancy | `div` / `--neg_w_div` | visited before step t, judged relevant (already consumed) | 3.0 |
| hard | `hard` / `--neg_w_hard` | visited before step t, judged irrelevant | 1.0 |
| weak | `weak` / `--neg_w_weak` | returned at step t, never visited in the whole trajectory | 0.3 |

Documents returned earlier but never visited stay **unlabelled** — not opening
something is not evidence that it is bad. Each group is 1 positive plus up to 3
redundancy, up to 3 hard, and weak negatives filling the rest to 9.
