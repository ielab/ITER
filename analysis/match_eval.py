"""
Lightweight string-match SR evaluation for BrowseComp-Plus trajectories.

Strategy:
  1. Runs ending in tool_call (hit step limit) → incorrect.
  2. Otherwise: strip <think>...</think>, normalize both gold and response
     (lowercase + remove accents + remove punctuation), then check
     if normalized gold is contained in normalized response.
"""

import argparse
import csv
import json
import os
import re
import unicodedata
from pathlib import Path


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_response(result: list) -> str:
    """Return final agent text output, or '' if run ended in tool_call."""
    if not result:
        return ""
    last = result[-1]
    out = last.get("output", "") or ""
    if "<tool_call>" in out or last.get("type") == "tool_call":
        return ""
    # strip <think>...</think>
    clean = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
    return clean


def load_gt(path: str) -> dict:
    gt = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if len(row) >= 3:
                gt[row[0].strip()] = row[2].strip()
    return gt


def evaluate(traj_dir: str, gt_path: str, output_file: str):
    gt = load_gt(gt_path)
    files = sorted(Path(traj_dir).glob("*.json"))

    results = []
    for fpath in files:
        data = json.load(open(fpath))
        qid = str(data.get("query_id", ""))
        if qid.endswith(".0"):
            qid = qid[:-2]
        if qid not in gt:
            continue

        gold = gt[qid]
        response = extract_response(data.get("result", []))

        tool_call_counts = data.get("tool_call_counts", {}) or {}
        steps = sum(v for v in tool_call_counts.values() if isinstance(v, (int, float)))

        if not response:
            correct = False
        else:
            correct = normalize(gold) in normalize(response)

        results.append({
            "query_id": qid,
            "correct": correct,
            "steps": int(steps),
            "gold": gold,
            "response_tail": response[-200:] if response else "",
            "file": fpath.name,
        })

    total = len(results)
    sr = sum(r["correct"] for r in results) / total if total else 0.0
    avg_steps = sum(r["steps"] for r in results) / total if total else 0.0

    print(f"n={total}  SR={sr:.4f} ({sr*100:.1f}%)  Avg Steps={avg_steps:.1f}")

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"metrics": {"SR": sr, "Avg Steps": avg_steps, "n": total},
                       "details": results}, f, indent=2, ensure_ascii=False)
        print(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()
    evaluate(args.traj_dir, args.gt_path, args.output_file)


if __name__ == "__main__":
    main()
