# Training

The retriever is fine-tuned with a weighted contrastive objective over the
training groups: an instance weight from the length of the agent's post-visit
reasoning, and a per-tier weight for each negative.

Training uses the vendored `third_party/FlagEmbedding`, which upstream does not
have — it adds `--neg_w_div / --neg_w_hard / --neg_w_weak` and the per-tier caps.

## Command

`run_train.py` resolves the query instruction for a setting from
`src/memory_utils.py`, so the string the model is trained with is by
construction the one inference will use. Every other hyperparameter comes from
`src/config.py`.

| Argument | Meaning |
| --- | --- |
| `--setting` | query representation, `i0`–`i7` (see [query_representations.md](query_representations.md)) |
| `--train-data` | `training_data_<style>.jsonl` from the previous step |
| `--out` | checkpoint directory |
| `--base-model` | defaults to `Qwen/Qwen3-Embedding-0.6B`; set for the 4B/8B scaling run |
| `--num-gpus` | `torchrun --nproc_per_node` |
| `--dry-run` | print the command instead of running it |

```bash
python run_train.py \
  --setting i7 \
  --train-data /path/to/train_data/redesign/training_data_i7.jsonl \
  --out /path/to/models/iter_i7 \
  --num-gpus <NUM_GPUS>
```

Run it with `--dry-run` first to see the full FlagEmbedding command.

## Hyperparameters

Fixed across every variant in the paper, and the defaults in `src/config.py`:

| | |
| --- | --- |
| backbone | Qwen3-Embedding-0.6B |
| learning rate | 1e-6, AdamW, 0.1 warmup |
| batch size | 32 |
| epochs | 2 |
| group size | 10 (1 positive, 9 negatives) |
| temperature | 0.02 |
| pooling | last token, normalized |
| passage max length | 512 |
| query max length | 512 for `i0`, 8192 for history-conditioned styles |
| tier weights | redundancy 3.0, hard 1.0, weak 0.3 |
| tier caps | up to 3 redundancy, then up to 3 hard |

Two things that have silently broken this training before:

- `--gradient_checkpointing_kwargs '{"use_reentrant": false}'` is required;
  reentrant checkpointing under DDP dies with *"Expected to mark a variable
  ready only once"*.
- Do **not** add `--negatives_cross_device`. Gathering negatives across ranks
  discards the per-tier weights, turning this back into a flat-negative run
  that trains without error and scores worse.

`--passage_max_len` (512) must match the value used when the index was encoded.
