#!/usr/bin/env python3
"""Fine-tune the retriever on trajectory-relative supervision.

Every hyperparameter is fixed by config.TRAINING; --setting selects the query
representation, which decides the instruction the model is trained with. That
instruction comes from src/memory_utils.py, the same place inference reads it,
so training and serving cannot drift apart.

    python run_train.py --setting i7 \
        --train-data train_data/redesign/training_data_i7.jsonl \
        --out models/iter_i7
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from config import INSTRUCTION_FORMAT, SETTINGS, TRAINING, task_prefix   # noqa: E402
from memory_utils import REDESIGN_INSTRUCTIONS                           # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", required=True, choices=sorted(SETTINGS))
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--out", required=True, help="checkpoint directory")
    ap.add_argument("--base-model", default=TRAINING["base_model"],
                    help="4B/8B for the scaling run")
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    setting = SETTINGS[args.setting]
    style = "i0" if setting.style == "plain" else setting.style
    t = TRAINING

    cmd = ["torchrun", "--nproc_per_node", str(args.num_gpus),
           "-m", "FlagEmbedding.finetune.embedder.decoder_only.base",
           "--model_name_or_path", args.base_model,
           "--train_data", args.train_data,
           "--output_dir", args.out,
           "--run_name", Path(args.out).name,
           # the query grows with the history; the passage never does
           "--query_max_len", str(setting.max_length),
           "--passage_max_len", str(t["passage_max_len"]),
           "--pad_to_multiple_of", "8",
           "--query_instruction_for_retrieval", REDESIGN_INSTRUCTIONS[style],
           "--query_instruction_format", INSTRUCTION_FORMAT,
           "--train_group_size", str(t["train_group_size"]),
           "--per_device_train_batch_size", str(t["per_device_train_batch_size"]),
           "--learning_rate", str(t["learning_rate"]),
           "--num_train_epochs", str(t["num_train_epochs"]),
           "--warmup_ratio", str(t["warmup_ratio"]),
           "--bf16", "--gradient_checkpointing",
           # reentrant checkpointing under DDP dies with
           # "Expected to mark a variable ready only once"
           "--gradient_checkpointing_kwargs", '{"use_reentrant": false}',
           "--save_strategy", "epoch", "--logging_steps", "1",
           "--overwrite_output_dir",
           "--dataloader_drop_last", "True",
           "--same_dataset_within_batch", "True",
           "--temperature", str(t["temperature"]),
           "--sentence_pooling_method", t["pooling"],
           "--normalize_embeddings", str(t["normalize"]),
           # the tiered negatives: redundancy pushes hardest, weak barely at all
           "--neg_w_div", str(t["neg_w_div"]),
           "--neg_w_hard", str(t["neg_w_hard"]),
           "--neg_w_weak", str(t["neg_w_weak"]),
           "--div_neg_cap", str(t["div_neg_cap"]),
           "--hard_neg_cap", str(t["hard_neg_cap"])]

    print(f"instruction: {REDESIGN_INSTRUCTIONS[style]}\n")
    print(" ".join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
