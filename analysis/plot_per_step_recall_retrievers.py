"""Cumulative unified-qrel recall per search step on BCP, per backbone.

Three retrievers: AgentIR-4B vs ITER 0.6B i2 vs ITER 4B i2 (all dedupoff).
Figure 1: SEARCH recall — union of all returned docs up to search t.
Figure 2: VISIT recall  — union of successfully READ docs (get_document/visit)
          up to (and including the visits that follow) search t.
Qrels: gold ∪ evidence. Trajectories that end early keep their final value.
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = "${ITER_ROOT}"
E = os.path.join(ROOT, "experiments")
MAX_T = 30

QRELS = defaultdict(set)
for line in open(os.path.join(ROOT, "datasets/topics-qrels/qrel_gold_evidence.txt")):
    f = line.split()
    if len(f) >= 4 and int(f[3]) > 0:
        QRELS[f[0]].add(f[2])

BB = lambda d: os.path.join(E, "backbone", d)
ARMS = {
    "Tongyi-30B": {"AgentIR-4B": os.path.join(E, "bcp", "agentir4b_dedupoff"),
                   "0.6B i2": os.path.join(E, "dedup_ablation", "bcp_diver_dedupoff"),
                   "4B i2": os.path.join(E, "bcp", "scale_4b_i2_dedupoff")},
    "Qwen3.5-4B": {"AgentIR-4B": BB("bcp_qwen3.5-4b_agentir"),
                   "0.6B i2": BB("bcp_qwen3.5-4b_i2_dedupoff"),
                   "4B i2": BB("bcp_qwen3.5-4b_4bi2_dedupoff")},
    "Qwen3.5-9B": {"AgentIR-4B": BB("bcp_qwen3.5-9b_agentir"),
                   "0.6B i2": BB("bcp_qwen3.5-9b_i2_dedupoff"),
                   "4B i2": BB("bcp_qwen3.5-9b_4bi2_dedupoff")},
    "Qwen3.5-27B": {"AgentIR-4B": BB("bcp_qwen3.5-27b_agentir"),
                    "0.6B i2": BB("bcp_qwen3.5-27b_i2_dedupoff"),
                    "4B i2": BB("bcp_qwen3.5-27b_4bi2_dedupoff")},
    "Qwen3.6-27B": {"AgentIR-4B": BB("bcp_qwen3.6-27b_agentir"),
                    "0.6B i2": os.path.join(E, "dedup_ablation", "bcp_qwen3.6-27b_diver_dedupoff"),
                    "4B i2": BB("bcp_qwen3.6-27b_4bi2_dedupoff")},
    "gpt-oss-120B": {"AgentIR-4B": BB("bcp_gptoss-120b_agentir"),
                     "0.6B i2": BB("bcp_gptoss-120b_i2_dedupoff"),
                     "4B i2": BB("bcp_gptoss-120b_4bi2_dedupoff")},
}

DOCID_RE = re.compile(r"DocID:(\d+)")
GPTOSS_DOCID_RE = re.compile(r'"docid":\s*"(\d+)"')
CALL_RE = re.compile(r'"name":\s*"(get_document|visit)".{0,200}?"docid":\s*"?(\d+)"?', re.S)


def event_stream(run):
    """Yield ('search', [ids]) and ('visit', id) in trajectory order."""
    if "raw_messages" in run:
        msgs = run["raw_messages"]
        pending_visits = []
        for i, m in enumerate(msgs):
            role, c = m.get("role"), m.get("content")
            c = c if isinstance(c, str) else json.dumps(c)
            if role == "assistant":
                for tool, docid in CALL_RE.findall(c):
                    pending_visits.append(docid)
            elif role == "user" and "<tool_response>" in c:
                if "A search for" in c:
                    ids = DOCID_RE.findall(c)
                    if ids:
                        yield ("search", ids)
                elif pending_visits:
                    d = pending_visits.pop(0)
                    if not c.strip().startswith("<tool_response>\n[Document not found") \
                       and "[Document not found" not in c[:60] and '"error"' not in c[:200]:
                        yield ("visit", d)
    else:  # gptoss responses stream
        for item in run.get("result", []):
            if item.get("type") != "tool_call":
                continue
            out = str(item.get("output", ""))
            if item.get("tool_name") == "search":
                ids = GPTOSS_DOCID_RE.findall(out)
                if ids:
                    yield ("search", ids)
            elif item.get("tool_name") in ("get_document", "visit"):
                a = item.get("arguments") or {}
                if isinstance(a, str):
                    try:
                        a = json.loads(a)
                    except Exception:
                        a = {}
                d = str(a.get("docid", ""))
                if d and "[Document not found" not in out[:60] and '"error"' not in out[:200]:
                    yield ("visit", d)


def arm_curves(runs_dir):
    """t -> per-trajectory (search_recall, visit_recall) at search step t."""
    srec = defaultdict(list)
    vrec = defaultdict(list)
    for fp in glob.glob(os.path.join(runs_dir, "run_*.json")):
        try:
            r = json.load(open(fp))
        except Exception:
            continue
        rel = QRELS.get(str(r.get("query_id")), set())
        if not rel:
            continue
        seen_s, seen_v = set(), set()
        s_t, v_t = [], []
        for kind, payload in event_stream(r):
            if kind == "search":
                if len(s_t) >= MAX_T:
                    break
                seen_s.update(payload)
                s_t.append(len(seen_s & rel) / len(rel))
                v_t.append(len(seen_v & rel) / len(rel))
            else:
                seen_v.add(payload)
                if v_t:  # visits after search t count toward step t
                    v_t[-1] = len(seen_v & rel) / len(rel)
        if not s_t:
            continue
        for t in range(1, MAX_T + 1):
            srec[t].append(s_t[min(t, len(s_t)) - 1])
            vrec[t].append(v_t[min(t, len(v_t)) - 1])
    return srec, vrec


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
            n = len(results[(bb, name)][0].get(1, []))
            print(f"done {bb} / {name} (n={n})")

    COLORS = {"AgentIR-4B": "#d62728", "0.6B i2": "#1f77b4", "4B i2": "#2ca02c"}
    for metric_i, (label, fname) in enumerate(
            [("cumulative SEARCH recall (gold ∪ evidence)", "per_step_search_recall_3retrievers.png"),
             ("cumulative VISIT recall (gold ∪ evidence)", "per_step_visit_recall_3retrievers.png")]):
        fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True)
        for ax, bb in zip(axes.flat, ARMS):
            for name in ARMS[bb]:
                if (bb, name) not in results:
                    continue
                curves = results[(bb, name)][metric_i]
                ts = sorted(curves)
                ys = [np.mean(curves[t]) for t in ts]
                ax.plot(ts, ys, label=name, color=COLORS[name], lw=1.8)
            ax.set_title(bb)
            ax.grid(alpha=0.3)
        for ax in axes[1]:
            ax.set_xlabel("search index within trajectory")
        for ax in axes[:, 0]:
            ax.set_ylabel(label.split(" (")[0])
        axes[0, 0].legend()
        fig.suptitle(f"BCP (830 q, dedup off) — {label} up to search t; early-ending trajectories keep their final value")
        fig.tight_layout()
        out = os.path.join(E, "analysis", fname)
        fig.savefig(out, dpi=150)
        print("wrote", out)


if __name__ == "__main__":
    main()
