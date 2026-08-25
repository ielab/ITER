# ITER query representations (paper numbering) and code-flag mapping

Every query representation is built by `src/memory_utils.py` and must be
byte-identical in three places: (1) training-data extraction from trajectories,
(2) the training jsonl's `query_instruction`, and (3) live inference. The
per-setting instruction strings are in the eval scripts' `SETTING` case blocks.

## The ladder

| paper | code flag | query contains | adds |
|---|---|---|---|
| ITER-i0 | `plain` / `SETTING=i0` | current sub-query | bare baseline (LRAT-style) |
| ITER-i1 | `i1` | + main question | the overall task |
| **ITER-i2** | `i2` | + previous sub-queries (numbered, oldest→newest) | search directions already explored |
| ITER-i3 | `i5` | i2 + post-visit notes per interaction | what was learned from read docs |
| ITER-i4 | `i6` | i2 + visited-document snippets (64-token, untagged) | document text as memory |
| ITER-i5 | `i7` | i2 + snippets + notes | union of the two memory encodings |
| ITER-i6 | `i8` | pre-search reasoning + sub-query (AgentIR's exact format) | the agent's live intent, AgentIR packaging |
| ITER-i7 | `i9` | i2 + `Current Reasoning:` line before `Current Subquery:` | reasoning added to the structured history |

Two internal variants (old `i3`, `i4`: tagged 128-token document encodings) were
dropped from the paper; the internal `i5`–`i9` flags therefore map to paper
numbers shifted down by two. An additional internal `i10` (AgentIR's two fields
in our template) exists in the code but is not part of the paper's ladder.

## Format rules

- Natural-language field names (`Main Question:`, `Current Subquery:`,
  `Previous Interactions:` …); history grouped per interaction (one search +
  its visits), numbered oldest → newest.
- Empty fields render as `<empty>`; all fields one-lined.
- The pre-search reasoning in ITER-i6/i7 is the `<think>` of the turn that
  issues the search (digesting prior results and stating intent) — NOT the
  post-visit note used by ITER-i3/i5.
- ITER-i6 uses AgentIR's raw template and instruction-prefix style; ITER-i7
  injects the same signal into ITER-i2's structured format, so
  ITER-i7 vs ITER-i2 isolates the reasoning signal and ITER-i6 vs ITER-i7
  isolates the packaging + history.

## Negative tiers (paper §4.2.4)

| paper name | code name / env | members | weight |
|---|---|---|---|
| redundancy | `div` / `NEG_W_DIV` | visited before step t, judged relevant (already consumed) | 3.0 |
| hard | `hard` / `NEG_W_HARD` | visited before step t, judged irrelevant | 1.0 |
| weak | `weak` / `NEG_W_WEAK` | returned at step t, never visited in the whole trajectory | 0.3 |

Documents returned earlier but never visited stay **unlabeled** (no visit ⇒ no
utility observation). Groups: 1 positive + up to 3 redundancy + up to 3 hard,
weak fills the rest to 9 negatives.
