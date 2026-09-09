#!/usr/bin/env python3
"""Record which encoder built an index, so evaluation can refuse a mismatch.

An index only makes sense with the encoder that produced it, at the same
passage length and pooling. Get that wrong and nothing raises: the retriever
just returns worse documents. This writes the encoder's settings next to the
index and lets run_eval.py check them before an arm starts.

Stamp an index that was built without it:

    python src/index_meta.py --index-dir /path/to/index/i2_bcp \
        --retriever ielabgroup/ITER-0.6B
"""

import argparse
import json
import os

META_NAME = "encoder.json"


def write_meta(index_dir, retriever):
    """Called after encoding; records what the index was built with."""
    from config import INDEXING

    meta = {"retriever": retriever,
            "passage_max_len": INDEXING["passage_max_len"],
            "pooling": INDEXING["pooling"],
            "normalize": INDEXING["normalize"]}
    os.makedirs(index_dir, exist_ok=True)
    path = os.path.join(index_dir, META_NAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return path


def check_meta(index_path, retriever):
    """Return a warning string if the index does not match `retriever`, else None.

    Unstamped indexes are not an error — older ones have no encoder.json — but
    they are reported, because an unchecked index is exactly the failure this
    file exists to catch.
    """
    from config import INDEXING

    index_dir = index_path if os.path.isdir(index_path) else os.path.dirname(index_path)
    path = os.path.join(index_dir, META_NAME)
    if not os.path.exists(path):
        return (f"{index_dir} has no {META_NAME}, so the encoder behind it cannot be "
                f"checked. Stamp it with: python src/index_meta.py "
                f"--index-dir {index_dir} --retriever <the encoder that built it>")

    with open(path, encoding="utf-8") as f:
        meta = json.load(f)

    problems = []
    if meta.get("retriever") != retriever:
        problems.append(f"built with {meta.get('retriever')}, but this arm serves {retriever}")
    for key in ("passage_max_len", "pooling", "normalize"):
        if meta.get(key) != INDEXING[key]:
            problems.append(f"{key} is {meta.get(key)} here, {INDEXING[key]} in config.INDEXING")
    if problems:
        return f"index {index_dir} does not match: " + "; ".join(problems)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--retriever", required=True, help="the encoder that built this index")
    args = ap.parse_args()
    print("wrote", write_meta(args.index_dir, args.retriever))


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
