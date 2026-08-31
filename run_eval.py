#!/usr/bin/env python3
"""Run one evaluation arm: an agent backbone searching with one retriever.

Assumes the agent is already being served by vLLM on --port. Print the exact
server command with --print-server; starting, waiting for and killing that
server is scheduling, which every cluster does differently.

    python run_eval.py --setting i2 --backbone tongyi --benchmark bcp --print-server
    python run_eval.py --setting i2 --backbone tongyi --benchmark bcp \
        --index data/indexes/iter_i2_bcp/index-000.pkl
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from config import BACKBONES, BENCHMARKS, SERVING, SETTINGS    # noqa: E402
from index_meta import check_meta                              # noqa: E402

CLIENTS = {
    "tongyi": "src/search_agent/tongyi_client.py",
    "qwen35": "src/search_agent/qwen35_client.py",
    "gptoss": "src/search_agent/gptoss_responses_client.py",
}


def server_command(backbone, port, gpu_util):
    """The vLLM command that serves this backbone."""
    bb = BACKBONES[backbone]
    return ["python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", bb.model, "--served-model-name", bb.model,
            "--host", "0.0.0.0", "--port", str(port),
            "--dtype", "bfloat16", "--gpu-memory-utilization", str(gpu_util),
            "--max-model-len", str(bb.max_model_len),
            "--enable-prefix-caching"] + bb.vllm_flags


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", required=True, choices=sorted(SETTINGS))
    ap.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    ap.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    ap.add_argument("--index", help="FAISS index (default: data/indexes/<setting>_<benchmark>)")
    ap.add_argument("--retriever", help="override the setting's checkpoint")
    ap.add_argument("--searcher", default="faiss", choices=["faiss", "bm25"])
    ap.add_argument("--out", help="output directory (default: runs/<benchmark>_<backbone>_<setting>)")
    ap.add_argument("--port", type=int, default=6000, help="port vLLM is serving on")
    ap.add_argument("--gpu-util", type=float, default=0.92)
    ap.add_argument("--threads", type=int, default=2, help="concurrent questions")
    ap.add_argument("--dedup", action="store_true",
                    help="serve the de-duplicated harness (OFF for every paper number)")
    ap.add_argument("--print-server", action="store_true",
                    help="print the vLLM command for this backbone and exit")
    ap.add_argument("--ignore-index-check", action="store_true",
                    help="run even if the index was built by a different encoder")
    ap.add_argument("--dry-run", action="store_true", help="print the client command and exit")
    args = ap.parse_args()

    setting, backbone, bench = SETTINGS[args.setting], BACKBONES[args.backbone], BENCHMARKS[args.benchmark]

    if args.print_server:
        print(" ".join(server_command(args.backbone, args.port, args.gpu_util)))
        return

    index = args.index or f"data/indexes/{args.setting}_{args.benchmark}/index-000.pkl"
    out = args.out or f"runs/{args.benchmark}_{args.backbone}_{args.setting}"
    retriever = args.retriever or setting.retriever

    # An index built by a different encoder returns worse documents and raises
    # nothing, so check before spending the arm.
    if args.searcher != "bm25":
        problem = check_meta(index, retriever)
        if problem:
            print(f"WARNING: {problem}", file=sys.stderr)
            if not args.ignore_index_check:
                sys.exit("refusing to run; pass --ignore-index-check to override")

    # ---- arguments every client shares ----
    cmd = ["python", CLIENTS[backbone.client],
           "--output-dir", out,
           "--searcher-type", args.searcher,
           "--index-path", index,
           "--query", bench.queries,
           "--model", backbone.model,
           "--num-threads", str(args.threads),
           "--snippet-max-tokens", str(SERVING["snippet_max_tokens"]),
           "--k", str(SERVING["k"]),
           "--query-style", setting.style]

    # ---- dense retrieval: the encoder needs its own configuration ----
    if args.searcher != "bm25":
        cmd += ["--model-name", retriever,
                "--dataset-name", bench.corpus,
                "--pooling", "eos", "--normalize", "--torch-dtype", "float16",
                "--task-prefix", setting.prefix,
                "--max-length", str(setting.max_length)]

    if args.dedup:
        cmd += ["--dedup-search", "--dedup-pool-k", str(SERVING["dedup_pool_k"])]

    # ---- per-client generation settings ----
    if backbone.client == "tongyi":
        cmd += ["--port", str(args.port),
                "--temperature", str(SERVING["temperature"]),
                "--top_p", str(SERVING["top_p"]),
                "--presence_penalty", str(SERVING["presence_penalty"])]
    elif backbone.client == "qwen35":
        cmd += ["--port", str(args.port)]
    else:                                          # gpt-oss speaks the Responses API
        os.environ["URL"] = f"http://localhost:{args.port}/v1"
        os.environ["API_KEY"] = "EMPTY"
        cmd += ["--get-document", "--query-template", "QUERY_TEMPLATE", "--strong",
                "--max-iterations", str(SERVING["max_turns"]),
                "--reasoning-effort", "medium"]

    os.environ["MAX_LLM_CALL_PER_RUN"] = str(SERVING["max_turns"])
    os.environ["LRAT_STRONG_PROMPT"] = "1"

    print(" ".join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, check=True, cwd=ROOT)

    # ---- score: InfoSeek is exact match, BCP needs the official judge ----
    if bench.metric == "exact_match":
        subprocess.run(["python", "analysis/match_eval.py", "--traj-dir", out,
                        "--gt-path", bench.answers, "--output-file", f"{out}_eval.json"],
                       check=False, cwd=ROOT)
    else:
        print(f"\nBrowseComp-Plus needs its LLM judge on {out}; see docs/evaluate.md")


if __name__ == "__main__":
    main()
