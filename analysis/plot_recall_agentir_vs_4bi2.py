"""Cumulative recall per search step on BCP: AgentIR-4B vs ITER 4B i2 (size-matched).

Two figures (search / visit), 6 backbone panels, gold ∪ evidence qrels.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_per_step_recall_retrievers import E, arm_curves  # noqa: E402

BB = lambda d: os.path.join(E, "backbone", d)
ARMS = {
    "Tongyi-30B": {"AgentIR-4B": os.path.join(E, "bcp", "agentir4b_dedupoff"),
                   "ITER 4B i2": os.path.join(E, "bcp", "scale_4b_i2_dedupoff")},
    "Qwen3.5-4B": {"AgentIR-4B": BB("bcp_qwen3.5-4b_agentir"),
                   "ITER 4B i2": BB("bcp_qwen3.5-4b_4bi2_dedupoff")},
    "Qwen3.5-9B": {"AgentIR-4B": BB("bcp_qwen3.5-9b_agentir"),
                   "ITER 4B i2": BB("bcp_qwen3.5-9b_4bi2_dedupoff")},
    "Qwen3.5-27B": {"AgentIR-4B": BB("bcp_qwen3.5-27b_agentir"),
                    "ITER 4B i2": BB("bcp_qwen3.5-27b_4bi2_dedupoff")},
    "Qwen3.6-27B": {"AgentIR-4B": BB("bcp_qwen3.6-27b_agentir"),
                    "ITER 4B i2": BB("bcp_qwen3.6-27b_4bi2_dedupoff")},
    "gpt-oss-120B": {"AgentIR-4B": BB("bcp_gptoss-120b_agentir"),
                     "ITER 4B i2": BB("bcp_gptoss-120b_4bi2_dedupoff")},
}


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

    COLORS = {"AgentIR-4B": "#d62728", "ITER 4B i2": "#1f77b4"}
    for metric_i, (label, fname) in enumerate(
            [("cumulative SEARCH recall (gold ∪ evidence)", "agentir_vs_4bi2_search_recall.png"),
             ("cumulative VISIT recall (gold ∪ evidence)", "agentir_vs_4bi2_visit_recall.png")]):
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
        fig.suptitle(f"BCP (830 q, dedup off) — {label} up to search t; AgentIR-4B vs ITER 4B i2 (size-matched)")
        fig.tight_layout()
        out = os.path.join(E, "analysis", fname)
        fig.savefig(out, dpi=150)
        print("wrote", out)


if __name__ == "__main__":
    main()
