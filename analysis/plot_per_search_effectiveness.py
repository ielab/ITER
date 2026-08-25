"""Per-search retrieval effectiveness on BCP, by search index within the trajectory.

For each arm (retriever x backbone, dedupoff) and each search index t (1-based):
  precision@10  = mean over trajectories active at t of |top-10 ∩ qrel| / 10
  new-evidence  = same but counting only qrel docs NOT returned by any earlier search
Conditional means (only trajectories that reach t); t shown while n >= MIN_N.

Outputs: experiments/analysis/per_search_effectiveness.png (precision)
         experiments/analysis/per_search_new_evidence.png (novelty)
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = "${ITER_ROOT}"
E = os.path.join(ROOT, "experiments")
MIN_N = 50
MAX_T = 30

QRELS = defaultdict(set)
for line in open(os.path.join(ROOT, "datasets/topics-qrels/qrel_gold_evidence.txt")):
    f = line.split()
    if len(f) >= 4 and int(f[3]) > 0:
        QRELS[f[0]].add(f[2])

BB = lambda d: os.path.join(E, "backbone", d)
ARMS = {  # backbone -> {retriever: runs_dir}
    "Tongyi-30B": {"AgentIR-4B": os.path.join(E, "bcp", "agentir4b_dedupoff"),
                   "ITER i2": os.path.join(E, "dedup_ablation", "bcp_diver_dedupoff"),
                   "ITER i9": os.path.join(E, "bcp", "redesign_i9_dedupoff")},
    "Qwen3.5-4B": {"AgentIR-4B": BB("bcp_qwen3.5-4b_agentir"),
                   "ITER i2": BB("bcp_qwen3.5-4b_i2_dedupoff"),
                   "ITER i9": BB("bcp_qwen3.5-4b_i9_dedupoff")},
    "Qwen3.5-9B": {"AgentIR-4B": BB("bcp_qwen3.5-9b_agentir"),
                   "ITER i2": BB("bcp_qwen3.5-9b_i2_dedupoff"),
                   "ITER i9": BB("bcp_qwen3.5-9b_i9_dedupoff")},
    "Qwen3.5-27B": {"AgentIR-4B": BB("bcp_qwen3.5-27b_agentir"),
                    "ITER i2": BB("bcp_qwen3.5-27b_i2_dedupoff"),
                    "ITER i9": BB("bcp_qwen3.5-27b_i9_dedupoff")},
    "Qwen3.6-27B": {"AgentIR-4B": BB("bcp_qwen3.6-27b_agentir"),
                    "ITER i2": os.path.join(E, "dedup_ablation", "bcp_qwen3.6-27b_diver_dedupoff"),
                    "ITER i9": BB("bcp_qwen3.6-27b_i9_dedupoff")},
    "gpt-oss-120B": {"AgentIR-4B": BB("bcp_gptoss-120b_agentir"),
                     "ITER i2": BB("bcp_gptoss-120b_i2_dedupoff"),
                     "ITER i9": BB("bcp_gptoss-120b_i9_dedupoff")},
}

SEARCH_RE = re.compile(r"A search for .{0,400}? found \d+ results", re.S)
DOCID_RE = re.compile(r"DocID:(\d+)")


GPTOSS_DOCID_RE = re.compile(r'"docid":\s*"(\d+)"')


def per_search_lists(run):
    out = []
    if "raw_messages" in run:   # tongyi / qwen35 clients
        for m in run.get("raw_messages", []):
            if m.get("role") != "user":
                continue
            c = m.get("content")
            c = c if isinstance(c, str) else json.dumps(c)
            if "<tool_response>" not in c or "A search for" not in c:
                continue
            ids = DOCID_RE.findall(c)
            if ids:
                out.append(ids)
    else:                        # gptoss responses client: result item stream
        for item in run.get("result", []):
            if item.get("type") == "tool_call" and item.get("tool_name") == "search":
                ids = GPTOSS_DOCID_RE.findall(str(item.get("output", "")))
                if ids:
                    out.append(ids)
    return out


def ndcg_at_k(ids, rel, k=10):
    dcg = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ids[:k]) if d in rel)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal > 0 else 0.0


def arm_curves(runs_dir):
    """t -> per-trajectory values at search t:
    cumrec — CUMULATIVE recall: |union of all docs returned up to t ∩ qrels| / |qrels|
    ndcg   — per-search nDCG@10
    Trajectories that end before t keep their final cumulative recall (a finished
    agent's coverage doesn't vanish), so cumrec is over ALL qrel'd trajectories."""
    cumrec = defaultdict(list)
    ndcg = defaultdict(list)
    for fp in glob.glob(os.path.join(runs_dir, "run_*.json")):
        try:
            r = json.load(open(fp))
        except Exception:
            continue
        rel = QRELS.get(str(r.get("query_id")), set())
        if not rel:
            continue
        searches = per_search_lists(r)[:MAX_T]
        if not searches:
            continue
        seen = set()
        rec_t = []
        for ids in searches:
            seen.update(ids)
            rec_t.append(len(seen & rel) / len(rel))
        for t in range(1, MAX_T + 1):
            cumrec[t].append(rec_t[min(t, len(rec_t)) - 1])
        for t, ids in enumerate(searches, 1):
            ndcg[t].append(ndcg_at_k(ids, rel))
    return cumrec, ndcg


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = {}
    for bb, arms in ARMS.items():
        for name, d in arms.items():
            if not os.path.isdir(d):
                print("MISSING", bb, name, d)
                continue
            results[(bb, name)] = arm_curves(d)
            print("done", bb, name)

    COLORS = {"AgentIR-4B": "#d62728", "ITER i2": "#1f77b4", "ITER i9": "#2ca02c"}
    for metric_i, (title, fname) in enumerate(
            [("cumulative recall of gold ∪ evidence qrels (union of searches 1..t)", "per_search_cumulative_recall.png"),
             ("nDCG@10 of the search (gold ∪ evidence qrels)", "per_search_ndcg.png")]):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
        for ax, bb in zip(axes.flat, ARMS):
            for name in ARMS[bb]:
                if (bb, name) not in results:
                    continue
                curves = results[(bb, name)][metric_i]
                ts = sorted(t for t in curves if len(curves[t]) >= MIN_N)
                ys = [np.mean(curves[t]) for t in ts]
                ax.plot(ts, ys, label=name, color=COLORS[name], lw=1.8)
            ax.set_title(bb)
            ax.grid(alpha=0.3)
        for ax in axes[1]:
            ax.set_xlabel("search index within trajectory")
        for ax in axes[:, 0]:
            ax.set_ylabel(title.split(" of ")[0])
        axes[0, 0].legend()
        fig.suptitle(f"BCP (830 q, dedup off) — per-search {title}; mean over trajectories active at each step (n≥{MIN_N})")
        fig.tight_layout()
        out = os.path.join(E, "analysis", fname)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.savefig(out, dpi=150)
        print("wrote", out)


if __name__ == "__main__":
    main()
