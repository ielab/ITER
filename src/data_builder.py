#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ITER diversity-aware training data builder (dedup trajectories).

For every NEW relevant visited doc, emit one positive-anchored sample. A doc is
a positive iff it is visited for the FIRST time, judged RELEVANT, and was surfaced
by some search in the trajectory. This captures BOTH:
  (a) the normal first-visit relevant doc, and
  (b) a doc revisited from the dedup `Already-seen` hint (historical
      search-not-visit that the agent comes back to read) -- exactly the doc LRAT
      wrongly labels as a negative.

Negatives are split into three tiers (for the stratified loss in FlagEmbedding):
  neg_diversity = docs visited before this one AND judged relevant (already in
                  memory but still relevant): the core diversity signal -- under
                  dedup these are exactly what the agent already read, so the
                  retriever must rank them DOWN to surface NEW relevant docs.
  neg_hard      = docs visited before this one but judged irrelevant: the agent
                  read them and found them useless (relevance hard negatives).
  neg_weak      = docs returned by the search IMMEDIATELY preceding this visit
                  that are NEVER visited anywhere in the whole trajectory
                  (current-step search-not-visit). The whole-trajectory check
                  kills the LRAT false-negative gamble: a doc revisited later can
                  never be a neg.
Negatives are VISITED-based, never RETURNED-based: the dedup harness suppresses
returned docs into the `Already-seen` hint and the hint is meant to resurface
returned-but-unread gold, so penalising returned docs would train the retriever
to bury that hint-gold. Historical search-not-visit left unread is in NO tier.

Two query forms are produced in one pass (shared pos/neg, query string differs):
  v1: [Q] + [Now] + [Memory]   ([Memory] = cleaned post-visit reasoning list)
  v2: [Q] + [Now] + [Prev]     ([Prev]   = prior sub-queries + their visited docs'
                                            title+snippet)

Relevance is decided by a live LLM judge over the agent's post-visit reasoning
(same as the original LRAT builder); requires --judge-api-url.

Output JSONL fields (per file):
  query, pos, pos_id, neg_diversity, neg_diversity_id, neg_hard, neg_hard_id,
  neg_weak, neg_weak_id, reasoning_len, reweight_rate

Example:
python src/data_builder.py \
  --corpus-path data/corpus.jsonl \
  --traj-dir runs/dedup_traj/bm25 \
  --output-v1 training_data/bm25.v1.jsonl \
  --output-v2 training_data/bm25.v2.jsonl \
  --tokenizer-path Qwen/Qwen3-Embedding-0.6B \
  --judge-api-url http://127.0.0.1:6009/v1/chat/completions \
  --judge-model auto --max-workers 32 --future-timeout 30
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import json
import math
import time
import logging
import argparse
import statistics
import multiprocessing
from typing import Dict, Any, List, Optional

import requests
from tqdm import tqdm
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError

from memory_utils import build_memory_query, build_prevdoc_query


logging.basicConfig(
    level=getattr(logging, os.getenv("LRAT_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

BROWSE_TOOLS = {"get_document", "visit"}


# -------------------------
# IO
# -------------------------
def load_corpus_jsonl(path: str, filter_docids: Optional[set] = None) -> Dict[str, str]:
    """Load corpus into {docid: text}. With filter_docids, keep only those docs."""
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading corpus: {os.path.basename(path)}"):
            if not line.strip():
                continue
            data = json.loads(line)
            docid = str(data["docid"])
            if filter_docids is None or docid in filter_docids:
                result[docid] = data["text"]
    return result


def load_all_trajectories(dir_path: str, verbose: bool = True) -> List[Dict[str, Any]]:
    trajectories = []
    total = loaded = failed = 0
    for fname in os.listdir(dir_path):
        fpath = os.path.join(dir_path, fname)
        if not os.path.isfile(fpath):
            continue
        total += 1
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                traj = json.load(f)
            if isinstance(traj, dict) and "result" in traj:
                trajectories.append(traj)
                loaded += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if verbose:
                logger.warning("Failed to read %s: %s", fname, e)
    if verbose:
        logger.info("Trajectory loading | total=%s loaded=%s skipped=%s", total, loaded, failed)
    return trajectories


# -------------------------
# Trajectory parsing helpers
# -------------------------
def _search_returned_docids(step: Dict[str, Any]) -> List[str]:
    """Docids a search returned. Prefer the structured trace; fall back to text."""
    rd = step.get("returned_docids")
    if isinstance(rd, list):
        return [str(d) for d in rd]
    docs = []
    for line in (step.get("output", "") or "").split("\n"):
        if line.startswith("DocID:"):
            docs.append(line.split(":", 1)[1].strip())
    return docs


def _search_query(step: Dict[str, Any]) -> str:
    """The agent's sub-query for a search. Prefer the trace's original_query."""
    oq = step.get("original_query")
    if isinstance(oq, str) and oq:
        return oq
    try:
        args = json.loads(step.get("arguments", "{}") or "{}")
    except Exception:
        return ""
    q = args.get("query", "")
    return q[0] if isinstance(q, list) and q else str(q)


def _get_docid_from_browse_step(step: Dict[str, Any]) -> Optional[str]:
    try:
        args = json.loads(step.get("arguments", "{}") or "{}")
    except Exception:
        return None
    if not isinstance(args, dict):
        return None
    doc = args.get("docid", None)
    if isinstance(doc, list):
        doc = doc[0] if doc else None
    return str(doc).split(":")[-1] if doc is not None else None


def _reasoning_after(steps: List[Dict[str, Any]], i: int) -> str:
    """Post-visit reasoning = the reasoning step right after the browse step."""
    if i + 1 < len(steps) and steps[i + 1].get("type") == "reasoning":
        out = steps[i + 1].get("output", "")
        if isinstance(out, list):
            return " ".join(str(x) for x in out)
        return str(out)
    return ""


def _token_len(tokenizer, text: str) -> Optional[int]:
    if tokenizer is None:
        return None
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return None


def _original_question(traj: Dict[str, Any]) -> str:
    meta = traj.get("metadata") or {}
    q = meta.get("query_source")
    if q:
        return str(q)
    for msg in traj.get("raw_messages", []):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return str(traj.get("query", "") or traj.get("question", "") or "")


def collect_docids_from_trajectories(trajectories: List[Dict[str, Any]]) -> set:
    docids = set()
    for traj in trajectories:
        for step in traj.get("result", []):
            if step.get("type") != "tool_call":
                continue
            if step.get("tool_name") == "search":
                docids.update(_search_returned_docids(step))
            elif step.get("tool_name") in BROWSE_TOOLS:
                d = _get_docid_from_browse_step(step)
                if d:
                    docids.add(d)
    return docids


# -------------------------
# Judge (live LLM relevance classifier over post-visit reasoning)
# -------------------------
JUDGE_PROMPT = r"""
You are an LLM judge. You will classify whether the AnalysisText suggests the browsing text is relevant or not relevant, using a bias aligned with typical browsing behavior: models often keep searching even when content is relevant, but they only say "not relevant" when it's clearly off-topic.

Input
AnalysisText: another model's analysis of a browsing page.
Decision rule (important)
Output NOT_RELEVANT only if the analysis contains a clear negative judgment of relevance (explicit or unmistakable), such as: "not relevant / irrelevant / unrelated / off-topic / doesn't help / cannot answer / no useful info," or it clearly concludes the content is about a different topic and provides no value for the task.
Otherwise output RELEVANT.
    - This includes cases where the analysis:
    - extracts useful facts/steps/details from the page,
    - says the page is partially helpful,
    - suggests using it as background/context,
    - recommends continuing to search for more sources (continuing search does not imply irrelevance).
Output (strict)
Return exactly one token:

RELEVANT
NOT_RELEVANT

Classify this:
AnalysisText:
{{ANALYSIS_TEXT}}
""".strip()


def judge_relevance(analysis_text: str, *, judge_api_url: str, judge_model: str,
                    headers: Optional[Dict[str, str]] = None,
                    retries: int = 2, sleep_sec: float = 1.0) -> bool:
    """True for RELEVANT, False for NOT_RELEVANT (False on persistent error)."""
    prompt = JUDGE_PROMPT.replace("{{ANALYSIS_TEXT}}", str(analysis_text))
    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "top_p": 0.95, "top_k": 50, "temperature": 0.8, "n": 1,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    for _ in range(retries + 1):
        try:
            resp = requests.post(judge_api_url, json=payload, headers=headers, timeout=3600)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if "</think>" in content:
                content = content.split("</think>", 1)[1].strip()
            first = content.split()[0] if content else ""
            return first == "RELEVANT"
        except Exception:
            time.sleep(sleep_sec)
    return False


def resolve_judge_model(judge_api_url: str, judge_model: str,
                        headers: Optional[Dict[str, str]] = None) -> str:
    """Resolve judge_model='auto' to the server's actually-served model id.

    vLLM requires the request 'model' to match its served-model-name; sending the
    literal 'auto' is rejected, which would make every judge call silently return
    False (i.e. label every doc NOT_RELEVANT). Query /v1/models instead, and raise
    loudly if it can't be resolved rather than mislabel the whole dataset.
    """
    if judge_model and judge_model != "auto":
        return judge_model
    models_url = judge_api_url.rsplit("/chat/completions", 1)[0] + "/models"
    resp = requests.get(models_url, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        raise RuntimeError(f"judge-model 'auto' but no models served at {models_url}")
    model_id = str(data[0]["id"])
    logger.info("Resolved judge-model 'auto' -> %s", model_id)
    return model_id


# -------------------------
# Sample extraction (memory-conditioned, tiered negatives, no reset)
# -------------------------
def extract_pairs(traj: Dict[str, Any], tokenizer, corpus: Dict[str, str], judge_fn) -> List[Dict[str, Any]]:
    steps = traj["result"]
    question = _original_question(traj)

    # whole-trajectory facts (no reset)
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

    # live state, updated as visits happen (feeds FUTURE search snapshots)
    v1_reasonings: List[str] = []        # raw post-visit reasoning of all visits so far
    v2_groups: List[Dict[str, Any]] = [] # [{"query":.., "docs":[docid,..]}] per search
    visited_seen: List[str] = []         # ordered unique: all docs visited so far
    visited_set = set()
    visited_sat: Dict[str, bool] = {}    # docid -> relevance at first visit (tier split)

    # The retriever query is built ONCE, at SEARCH time (frozen snapshot), matching
    # inference: memory holds only visits BEFORE the search is issued. Positives for
    # that query = new-relevant docs visited in the window after the search (incl.
    # hint-revisits of docs first surfaced by an earlier search).
    snap: Optional[Dict[str, Any]] = None

    samples = []
    i = 0
    while i < len(steps):
        step = steps[i]

        if step.get("type") == "tool_call" and step.get("tool_name") == "search":
            sub_query = _search_query(step)
            snap = {
                "q_v1": build_memory_query(question, sub_query, v1_reasonings, tokenizer),
                "q_v2": build_prevdoc_query(question, sub_query, list(v2_groups), corpus.get, tokenizer),
                "visited_seen": list(visited_seen),      # frozen: pre-search visits
                "returned": _search_returned_docids(step),
            }
            v2_groups.append({"query": sub_query, "docs": []})
            i += 1
            continue

        if step.get("type") == "tool_call" and step.get("tool_name") in BROWSE_TOOLS:
            docid = _get_docid_from_browse_step(step)
            reasoning = _reasoning_after(steps, i)

            if docid is not None:
                first_time = docid not in visited_set
                # only first-time visits need a relevance label (for the positive test)
                sat = bool(judge_fn(reasoning)) if first_time else False

                if first_time and sat and snap is not None and docid in global_returned and docid in corpus:
                    # split pre-search visited docs by their first-visit relevance
                    seen = snap["visited_seen"]
                    neg_div_ids = [d for d in seen
                                   if visited_sat.get(d) and d != docid and d in corpus]
                    neg_hard_ids = [d for d in seen
                                    if not visited_sat.get(d) and d != docid and d in corpus]
                    neg_weak_ids = [d for d in snap["returned"]
                                    if d not in global_visited and d != docid and d in corpus]
                    if neg_div_ids or neg_hard_ids or neg_weak_ids:
                        samples.append({
                            "query_v1": snap["q_v1"],
                            "query_v2": snap["q_v2"],
                            "pos": [corpus[docid]],
                            "pos_id": [docid],
                            "neg_diversity": [corpus[d] for d in neg_div_ids],
                            "neg_diversity_id": neg_div_ids,
                            "neg_hard": [corpus[d] for d in neg_hard_ids],
                            "neg_hard_id": neg_hard_ids,
                            "neg_weak": [corpus[d] for d in neg_weak_ids],
                            "neg_weak_id": neg_weak_ids,
                            "reasoning_len": _token_len(tokenizer, reasoning),
                        })

                # update live state for future snapshots (snapshot above stays frozen)
                if reasoning:
                    v1_reasonings.append(reasoning)
                if v2_groups:
                    v2_groups[-1]["docs"].append(docid)
                if first_time:
                    visited_set.add(docid)
                    visited_seen.append(docid)
                    visited_sat[docid] = sat

            # advance past the browse step (+ its trailing reasoning step)
            if i + 1 < len(steps) and steps[i + 1].get("type") == "reasoning":
                i += 2
            else:
                i += 1
            continue

        i += 1

    return samples


# -------------------------
# Reweighting (by post-visit reasoning length)
# -------------------------
def add_reweight_rate(samples: List[Dict[str, Any]]):
    all_lens = [float(s["reasoning_len"]) for s in samples
                if isinstance(s.get("reasoning_len"), (int, float)) and s["reasoning_len"] > 0]
    half_life = statistics.median(all_lens) if all_lens else 1.0
    ln2 = math.log(2.0)
    raws = []
    for s in samples:
        rl = s.get("reasoning_len")
        raws.append(1 - math.exp(-float(rl) * ln2 / half_life)
                    if isinstance(rl, (int, float)) and rl > 0 else None)
    mean_w = (sum(r for r in raws if r is not None) / len(all_lens)) if all_lens else 1.0
    for s, raw in zip(samples, raws):
        s["reweight_rate"] = (raw / mean_w) if isinstance(raw, float) and mean_w else 1.0
    return half_life, mean_w


# -------------------------
# Multiprocessing (fork; corpus shared copy-on-write via parent global)
# -------------------------
_GLOBALS: Dict[str, Any] = {}
_CORPUS: Dict[str, str] = {}


def _init_worker(tokenizer_path, judge_api_url, judge_model, headers):
    _GLOBALS["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_path)
    _GLOBALS["judge_api_url"] = judge_api_url
    _GLOBALS["judge_model"] = judge_model
    _GLOBALS["headers"] = headers


def _worker(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    judge_fn = lambda text: judge_relevance(
        text,
        judge_api_url=_GLOBALS["judge_api_url"],
        judge_model=_GLOBALS["judge_model"],
        headers=_GLOBALS["headers"],
    )
    return extract_pairs(traj, _GLOBALS["tokenizer"], _CORPUS, judge_fn)


# -------------------------
# Output
# -------------------------
def _write(path: str, samples: List[Dict[str, Any]], query_key: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            rec = {
                "query": s[query_key],
                "pos": s["pos"],
                "pos_id": s["pos_id"],
                "neg_diversity": s["neg_diversity"],
                "neg_diversity_id": s["neg_diversity_id"],
                "neg_hard": s["neg_hard"],
                "neg_hard_id": s["neg_hard_id"],
                "neg_weak": s["neg_weak"],
                "neg_weak_id": s["neg_weak_id"],
                "reasoning_len": s["reasoning_len"],
                "reweight_rate": s["reweight_rate"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -------------------------
# Main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path",    required=True)
    ap.add_argument("--traj-dir",       required=True)
    ap.add_argument("--output-v1",      required=True, help="v1 ([Memory]) training JSONL")
    ap.add_argument("--output-v2",      required=True, help="v2 ([Prev]) training JSONL")
    ap.add_argument("--tokenizer-path", required=True)
    ap.add_argument("--judge-api-url",  required=True)
    ap.add_argument("--judge-model",    default="auto")
    ap.add_argument("--judge-headers",  default="")
    ap.add_argument("--max-workers",    type=int, default=32)
    ap.add_argument("--future-timeout", type=int, default=30)
    ap.add_argument("--max-trajs",      type=int, default=0, help="0 = all")
    return ap.parse_args()


def main():
    args = parse_args()
    headers = (json.loads(args.judge_headers) if args.judge_headers.strip()
               else {"Content-Type": "application/json"})
    # Resolve 'auto' against the live server up front, so a bad value fails here
    # instead of silently labelling every doc NOT_RELEVANT inside the workers.
    args.judge_model = resolve_judge_model(args.judge_api_url, args.judge_model, headers)

    trajectories = load_all_trajectories(args.traj_dir)
    if args.max_trajs > 0:
        trajectories = trajectories[:args.max_trajs]
        logger.info("Capped to %d trajectories", len(trajectories))

    # Load only the docs these trajectories reference (full corpus is ~30 GB).
    needed = collect_docids_from_trajectories(trajectories)
    logger.info("Filtered corpus load: %d unique docids", len(needed))
    global _CORPUS
    _CORPUS = load_corpus_jsonl(args.corpus_path, filter_docids=needed)

    all_samples: List[Dict[str, Any]] = []
    fork_ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=args.max_workers,
        mp_context=fork_ctx,
        initializer=_init_worker,
        initargs=(args.tokenizer_path, args.judge_api_url, args.judge_model, headers),
    ) as ex:
        futures = {ex.submit(_worker, t): i for i, t in enumerate(trajectories)}
        for fu in tqdm(as_completed(futures), total=len(futures), desc="Trajectories"):
            idx = futures[fu]
            try:
                ss = fu.result(timeout=args.future_timeout)
            except TimeoutError:
                logger.warning("Timeout on trajectory %s", idx)
                continue
            except Exception as e:
                logger.warning("Error on trajectory %s: %r", idx, e)
                continue
            if ss:
                all_samples.extend(ss)

    half_life, mean_w = add_reweight_rate(all_samples)
    n = len(all_samples)
    avg_div = sum(len(s["neg_diversity_id"]) for s in all_samples) / n if n else 0
    avg_hard = sum(len(s["neg_hard_id"]) for s in all_samples) / n if n else 0
    avg_weak = sum(len(s["neg_weak_id"]) for s in all_samples) / n if n else 0
    logger.info(
        "Done | samples=%d half_life=%.1f mean_w=%.3f avg_neg_div=%.2f avg_neg_hard=%.2f avg_neg_weak=%.2f",
        n, half_life, mean_w, avg_div, avg_hard, avg_weak,
    )

    _write(args.output_v1, all_samples, "query_v1")
    _write(args.output_v2, all_samples, "query_v2")


if __name__ == "__main__":
    main()
