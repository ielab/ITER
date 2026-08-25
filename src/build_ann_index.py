#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a prebuilt approximate FAISS index (HNSW by default) from the encoded
.pkl shards, ONCE, so the searcher loads a graph index (ms/query) instead of doing
an exact flat scan (~seconds/query) over the whole corpus.

The encode step (build_faiss_index.sh) writes index-NNN.pkl shards of
(reps, lookup) — raw normalised vectors + their docids. This concatenates them
(incrementally, to avoid a 2x memory spike), builds a faiss index via index_factory,
and writes:
  <out-dir>/index.faiss        the faiss index (graph + vectors)
  <out-dir>/index.lookup.pkl   the docid list aligned to faiss positions

Used by faiss_searcher for BOTH trajectory generation and eval (same backend, so
train/inference match). Vectors are normalised, so METRIC_INNER_PRODUCT = cosine.

Example (HNSW, the recommended default):
  python src/build_ann_index.py \
    --shards 'data/indexes/qwen3e_0.6b/index-*.pkl' \
    --out-dir data/indexes/qwen3e_0.6b \
    --index-type HNSW32 --ef-construction 200 --ef-search 256
"""
import argparse
import glob
import logging
import os
import pickle

import faiss
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True,
                    help="glob of encode .pkl shards, e.g. 'data/indexes/qwen3e_0.6b/index-*.pkl'")
    ap.add_argument("--out-dir", required=True, help="where to write index.faiss + index.lookup.pkl")
    ap.add_argument("--index-type", default="HNSW32",
                    help="faiss index_factory string (e.g. HNSW32, IVF65536,PQ64)")
    ap.add_argument("--ef-construction", type=int, default=200,
                    help="HNSW build-time depth (higher = better recall, slower build)")
    ap.add_argument("--ef-search", type=int, default=256,
                    help="HNSW query-time depth, baked into the index (higher = better recall)")
    ap.add_argument("--threads", type=int, default=0, help="faiss OMP threads (0 = all)")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.threads > 0:
        faiss.omp_set_num_threads(args.threads)

    shard_paths = sorted(glob.glob(args.shards))
    if not shard_paths:
        raise SystemExit(f"no shards match: {args.shards}")
    logger.info("Building '%s' index from %d shard(s)", args.index_type, len(shard_paths))

    index = None
    lookup = []
    total = 0
    for i, p in enumerate(shard_paths):
        with open(p, "rb") as f:
            reps, lk = pickle.load(f)
        reps = np.ascontiguousarray(reps, dtype=np.float32)
        if reps.ndim != 2 or reps.shape[0] == 0:
            logger.warning("shard %s is empty/malformed (shape=%s); skipping",
                           os.path.basename(p), reps.shape)
            continue
        if index is None:
            dim = reps.shape[1]
            index = faiss.index_factory(dim, args.index_type, faiss.METRIC_INNER_PRODUCT)
            if hasattr(index, "hnsw"):
                index.hnsw.efConstruction = args.ef_construction
                index.hnsw.efSearch = args.ef_search
            logger.info("index_factory(dim=%d, '%s', IP)", dim, args.index_type)
        if not index.is_trained:          # IVF/PQ need training; HNSW does not
            logger.info("training index on shard %d (%d vectors)", i, reps.shape[0])
            index.train(reps)
        index.add(reps)
        lookup += [str(x) for x in lk]
        total += reps.shape[0]
        logger.info("added shard %d/%d (%s): total=%d", i + 1, len(shard_paths),
                    os.path.basename(p), total)
        del reps

    if index is None or total == 0:
        raise SystemExit(f"no non-empty vectors found in shards: {args.shards}")

    os.makedirs(args.out_dir, exist_ok=True)
    index_path = os.path.join(args.out_dir, "index.faiss")
    lookup_path = os.path.join(args.out_dir, "index.lookup.pkl")
    faiss.write_index(index, index_path)
    with open(lookup_path, "wb") as f:
        pickle.dump(lookup, f)
    logger.info("done: %d vectors -> %s (+ %s)", total, index_path, lookup_path)


if __name__ == "__main__":
    main()
