"""Rebuild the training queries in the redesigned structured format.

Replays the 4 dedup trajectory sources and re-emits the training data as
every query style (docs/query_representations.md), reusing the
pos/neg of the existing v1 file — the LLM judge is NOT re-run.

Sample identity is recovered without relying on file order: for every replay
candidate the OLD v1 query is rebuilt byte-for-byte and the key
(query_v1, pos_id, sorted seen-negs, sorted weak-negs) is matched against the
v1 lines. Every v1 line must be claimed exactly once; candidates the judge had
rejected simply find no line.

Run from the repo root:
    python src/rebuild_query_redesign.py \
        --v1 train_data/training_data_v1.jsonl \
        --traj-dir runs/dedup_traj \
        --out train_data/redesign
"""
import argparse
import json
import multiprocessing
import os
import sys

from transformers import AutoTokenizer

from data_builder import (
    BROWSE_TOOLS,
    _get_docid_from_browse_step,
    _original_question,
    _reasoning_after,
    _search_query,
    _search_returned_docids,
    collect_docids_from_trajectories,
    load_all_trajectories,
    load_corpus_jsonl,
)
from config import COLLECTION_BACKENDS
from memory_utils import REDESIGN_INSTRUCTIONS, build_memory_query, build_redesign_query

ROOT = os.environ.get("ITER_ROOT", os.getcwd())

# Set from the command line in main(); the worker pool is forked afterwards and
# inherits them, which is why they live at module level.
V1_PATH = TRAJ_BASE = CORPUS_PATH = OUT_DIR = None
SOURCES = VARIANTS = None

_CORPUS = {}
_TOK = None


def _reasoning_before(steps, i):
    """Pre-search reasoning (i6/i7): consecutive reasoning steps right before
    the search call = the issuing turn's <think>. Joined with " " exactly like
    react_agent's set_current_thinking; a search preceded by another tool_call
    (parallel calls) gets "" -> rendered as "Empty"."""
    parts = []
    j = i - 1
    while j >= 0 and steps[j].get("type") == "reasoning":
        out = steps[j].get("output", "")
        if isinstance(out, list):
            out = " ".join(str(x) for x in out)
        parts.append(str(out).strip())
        j -= 1
    return " ".join(reversed(parts)).strip()


def replay_traj(traj):
    """Walk one trajectory exactly like data_builder.extract_pairs and return
    (key, {variant: query}) for every candidate emission point."""
    corpus, tok = _CORPUS, _TOK
    steps = traj["result"]
    question = _original_question(traj)

    global_visited = set()
    global_returned = set()
    for st in steps:
        if st.get("type") != "tool_call":
            continue
        if st.get("tool_name") == "search":
            global_returned.update(_search_returned_docids(st))
        elif st.get("tool_name") in BROWSE_TOOLS:
            d = _get_docid_from_browse_step(st)
            if d:
                global_visited.add(d)

    v1_reasonings = []
    interactions = []      # [{"query":.., "visits":[(docid, doc_text, reasoning)]}]
    visited_seen = []
    visited_set = set()
    snap = None
    out = []

    i = 0
    while i < len(steps):
        step = steps[i]

        if step.get("type") == "tool_call" and step.get("tool_name") == "search":
            sub_query = _search_query(step)
            snap = {
                "q_v1": build_memory_query(question, sub_query, v1_reasonings, tok),
                "sub_query": sub_query,
                "seen": list(visited_seen),
                "returned": _search_returned_docids(step),
                "inter": [{"query": g["query"], "visits": list(g["visits"])}
                          for g in interactions],
                "pre": _reasoning_before(steps, i),
            }
            interactions.append({"query": sub_query, "visits": []})
            i += 1
            continue

        if step.get("type") == "tool_call" and step.get("tool_name") in BROWSE_TOOLS:
            docid = _get_docid_from_browse_step(step)
            reasoning = _reasoning_after(steps, i)

            if docid is not None:
                first_time = docid not in visited_set

                if first_time and snap is not None and docid in global_returned and docid in corpus:
                    seen_ids = tuple(sorted(d for d in snap["seen"]
                                            if d != docid and d in corpus))
                    weak_ids = tuple(sorted(d for d in snap["returned"]
                                            if d not in global_visited and d != docid and d in corpus))
                    if seen_ids or weak_ids:
                        key = (snap["q_v1"], docid, seen_ids, weak_ids)
                        queries = {v: build_redesign_query(v, question, snap["sub_query"],
                                                           snap["inter"], tok,
                                                           pre_reasoning=snap["pre"])
                                   for v in VARIANTS}
                        out.append((key, queries))

                if reasoning:
                    v1_reasonings.append(reasoning)
                if interactions:
                    interactions[-1]["visits"].append(
                        (docid, corpus.get(docid, ""), reasoning))
                if first_time:
                    visited_set.add(docid)
                    visited_seen.append(docid)

            if i + 1 < len(steps) and steps[i + 1].get("type") == "reasoning":
                i += 2
            else:
                i += 1
            continue

        i += 1

    return out


def _init_worker():
    global _TOK
    _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v1", default=f"{ROOT}/train_data/training_data_v1.jsonl",
                    help="the training groups from data_builder.py")
    ap.add_argument("--traj-dir", default=f"{ROOT}/runs/dedup_traj",
                    help="holds one subdirectory per retrieval backend")
    ap.add_argument("--corpus", default=f"{ROOT}/data/corpus.jsonl")
    ap.add_argument("--out", default=f"{ROOT}/train_data/redesign",
                    help="one training_data_<style>.jsonl is written per style")
    ap.add_argument("--backends", nargs="+", default=COLLECTION_BACKENDS,
                    help="subdirectories of --traj-dir, in the order v1 was built")
    ap.add_argument("--styles", nargs="+", default=sorted(REDESIGN_INSTRUCTIONS,
                                                          key=lambda v: int(v[1:])),
                    help="query styles to render (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    return ap.parse_args()


def main():
    global V1_PATH, TRAJ_BASE, CORPUS_PATH, OUT_DIR, SOURCES, VARIANTS
    args = parse_args()
    V1_PATH, TRAJ_BASE = args.v1, args.traj_dir
    CORPUS_PATH, OUT_DIR = args.corpus, args.out
    SOURCES, VARIANTS = args.backends, args.styles

    os.makedirs(OUT_DIR, exist_ok=True)

    # index the v1 lines by content key
    v1_lines = []
    index = {}
    with open(V1_PATH, encoding="utf-8") as f:
        for li, line in enumerate(f):
            d = json.loads(line)
            key = (d["query"], d["pos_id"][0],
                   tuple(sorted(d["neg_diversity_id"] + d["neg_hard_id"])),
                   tuple(sorted(d["neg_weak_id"])))
            index.setdefault(key, []).append(li)
            v1_lines.append(d)
    n = len(v1_lines)
    dup_keys = sum(1 for v in index.values() if len(v) > 1)
    print(f"v1 lines: {n}  (keys with >1 line: {dup_keys})", flush=True)

    new_queries = [None] * n
    candidates = matched = 0

    global _CORPUS
    fork_ctx = multiprocessing.get_context("fork")
    for src in SOURCES:
        trajs = load_all_trajectories(f"{TRAJ_BASE}/{src}")
        needed = collect_docids_from_trajectories(trajs)
        _CORPUS = load_corpus_jsonl(CORPUS_PATH, filter_docids=needed)
        print(f"[{src}] trajs={len(trajs)} corpus={len(_CORPUS)}", flush=True)

        with fork_ctx.Pool(args.workers, initializer=_init_worker) as pool:
            for out in pool.imap_unordered(replay_traj, trajs, chunksize=16):
                for key, queries in out:
                    candidates += 1
                    lis = index.get(key)
                    if lis:
                        new_queries[lis.pop(0)] = queries
                        matched += 1
        print(f"[{src}] cumulative candidates={candidates} matched={matched}", flush=True)

    missing = [i for i, q in enumerate(new_queries) if q is None]
    print(f"matched {matched}/{n} lines; {candidates - matched} candidates had no line "
          f"(judge-rejected); missing lines: {len(missing)}", flush=True)
    if missing:
        print("FIRST MISSING LINES:", missing[:5], file=sys.stderr)
        sys.exit(1)

    for v in VARIANTS:
        path = f"{OUT_DIR}/training_data_{v}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for d, queries in zip(v1_lines, new_queries):
                rec = dict(d)
                rec["query"] = queries[v]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
