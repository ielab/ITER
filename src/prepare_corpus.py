#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Download + convert the corpus into the schema the ITER pipeline expects.

The `wiki-25-512` dataset ships with columns {"id", "contents"}, but every stage
of ITER (index_builder, tevatron encode, faiss_searcher, data_builder) reads
{"docid", "text"}. This loads the dataset (from the HF cache if present), renames
the columns, and writes a single JSONL corpus.

Sharding for the FAISS encode is NOT done here — the encode array shards
data/corpus.jsonl on the fly (tevatron strided shard + in-shard length sort), so
the same single corpus file serves every embedding model. See
bash/job_cluster/submit_faiss_index.sh.

Examples:
    python src/prepare_corpus.py
    python src/prepare_corpus.py --dataset Lk123/wiki-25-512 --output data/corpus.jsonl
"""
import argparse
import os

from datasets import load_dataset


def _allocated_cpus():
    """Cores actually allocated to this process (respects the SLURM cpuset),
    falling back to the machine total off-Linux."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS / Windows
        return os.cpu_count() or 8


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="Lk123/wiki-25-512",
                    help="HF dataset id (or local path) to load")
    ap.add_argument("--split", default="train")
    ap.add_argument("--output", default="data/corpus.jsonl", help="output JSONL path")
    ap.add_argument("--id-col", default="id", help="source column to rename to 'docid'")
    ap.add_argument("--text-col", default="contents", help="source column to rename to 'text'")
    ap.add_argument("--num-proc", type=int, default=_allocated_cpus(),
                    help="parallel workers for JSON writing (default: cores allocated to this job)")
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"loading {args.dataset} [{args.split}] | num_proc={args.num_proc}", flush=True)
    ds = load_dataset(args.dataset, split=args.split)

    rename = {}
    if args.id_col in ds.column_names:
        rename[args.id_col] = "docid"
    if args.text_col in ds.column_names:
        rename[args.text_col] = "text"
    if rename:
        ds = ds.rename_columns(rename)

    missing = {"docid", "text"} - set(ds.column_names)
    if missing:
        raise SystemExit(
            f"corpus is missing required columns {missing}; got {ds.column_names}. "
            f"Pass --id-col/--text-col to map the right source columns."
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    print(f"writing {len(ds):,} rows -> {args.output}", flush=True)
    ds.to_json(args.output, num_proc=args.num_proc, force_ascii=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
