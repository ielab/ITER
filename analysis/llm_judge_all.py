"""
Batch LLM-judge SR across many trajectory directories in ONE vLLM session (loads the
judge model once, not once per directory). Reuses evaluate.py's judge template /
GT loader / qrel loader / parse logic so results are directly comparable to any prior
single-directory evaluate.py run.

Usage:
  python scripts_evaluation/llm_judge_all.py \
    --gt-path datasets/browsecomp-plus.tsv \
    --qrel-path datasets/topics-qrels/qrel_evidence.txt \
    --model-path models/Qwen3-30B-A3B-Thinking-2507 \
    --dirs label1=path1 label2=path2 ... \
    --output-dir experiments/e2e_dedup/llm_judge_results
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import (  # noqa: E402
    DatasetLoader,
    JUDGE_TEMPLATE,
    chunked,
    load_qrel_evidence,
    parse_judge_result,
)


def build_queue(input_dir: Path, ground_truth: dict, qrel_data: dict):
    input_files = sorted(input_dir.glob("*.json"), key=lambda x: x.name)
    queue = []
    for json_file in input_files:
        try:
            run_data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        qid = str(run_data.get("query_id"))
        if qid.endswith(".0"):
            qid = qid[:-2]
        if qid not in ground_truth:
            continue
        gt_item = ground_truth[qid]

        model_response = ""
        result = run_data.get("result")
        if isinstance(result, list) and result:
            last_msg = result[-1]
            if isinstance(last_msg, dict) and last_msg.get("type") != "tool_call":
                out = last_msg.get("output", "") or ""
                if "<tool_call>" not in out:
                    model_response = out

        prompt = JUDGE_TEMPLATE.format(
            question=gt_item["question"], response=model_response, correct_answer=gt_item["answer"]
        )

        tool_call_counts = run_data.get("tool_call_counts", {}) or {}
        if not isinstance(tool_call_counts, dict):
            tool_call_counts = {}
        steps = sum(v for v in tool_call_counts.values() if isinstance(v, (int, float)))

        recall = 0.0
        gold_docs = set(qrel_data.get(qid, []))
        if gold_docs:
            retrieved = set(run_data.get("retrieved_docids", []) or [])
            recall = len(retrieved & gold_docs) / float(len(gold_docs))

        queue.append({
            "qid": qid, "prompt": prompt, "steps": int(steps), "recall": float(recall),
            "file": json_file.name, "model_response": model_response,
        })
    return queue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-path", required=True)
    ap.add_argument("--qrel-path", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dirs", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    ground_truth = DatasetLoader.load_tsv(Path(args.gt_path))
    qrel_data = load_qrel_evidence(Path(args.qrel_path))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = []
    for pair in args.dirs:
        label, path = pair.split("=", 1)
        q = build_queue(Path(path), ground_truth, qrel_data)
        print(f"{label}: {len(q)} judgeable trajectories from {path}")
        arms.append((label, q))

    print("Loading judge model:", args.model_path)
    from vllm import LLM, SamplingParams

    sampling_params = SamplingParams(temperature=0.0, max_tokens=8192, seed=2026)
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=24576,
        trust_remote_code=True,
    )

    summary = []
    for label, queue in arms:
        if not queue:
            continue
        results = []
        correct = 0
        # empty/no-answer trajectories are automatically wrong -- skip the (expensive)
        # judge call entirely for them instead of wasting a GPU inference on a
        # foregone conclusion
        judgeable = [m for m in queue if m["model_response"] and "tool_call" not in m["model_response"]
                     and "user_query" not in m["model_response"]]
        no_answer = [m for m in queue if m not in judgeable]
        for meta in no_answer:
            results.append({
                "query_id": meta["qid"], "correct": False, "steps": meta["steps"],
                "recall": meta["recall"], "judge_reason": "(skipped: no final answer text)", "file": meta["file"],
            })
        print(f"{label}: {len(no_answer)}/{len(queue)} skipped (no answer), {len(judgeable)} sent to judge")

        for batch in tqdm(list(chunked(judgeable, args.batch_size)), desc=label):
            convos = [[{"role": "user", "content": item["prompt"]}] for item in batch]
            outputs = llm.chat(convos, sampling_params)
            for meta, output in zip(batch, outputs):
                judge_text = output.outputs[0].text if output.outputs else ""
                is_correct = parse_judge_result(judge_text)
                correct += int(is_correct)
                results.append({
                    "query_id": meta["qid"], "correct": bool(is_correct), "steps": meta["steps"],
                    "recall": meta["recall"], "judge_reason": judge_text, "file": meta["file"],
                })

        total = len(results)
        sr = correct / float(total) if total else 0.0
        avg_steps = float(np.mean([r["steps"] for r in results])) if total else 0.0
        evidence_recall = float(np.mean([r["recall"] for r in results])) if total else 0.0

        with open(out_dir / f"{label}.json", "w", encoding="utf-8") as f:
            json.dump(
                {"metrics": {"Success Rate": sr, "Avg Steps": avg_steps, "Evidence Recall": evidence_recall},
                 "details": results},
                f, indent=2, ensure_ascii=False,
            )
        print(f"{label}: n={total} SR={sr:.4f} ({sr*100:.1f}%) EvidRecall={evidence_recall:.4f}")
        summary.append((label, total, sr, evidence_recall))

    print("\n===== summary =====")
    print(f"{'label':<20} {'n':>5} {'SR':>8} {'EvidRecall':>10}")
    for label, total, sr, evidence_recall in summary:
        print(f"{label:<20} {total:>5} {sr*100:>7.1f}% {evidence_recall:>10.4f}")


if __name__ == "__main__":
    main()
