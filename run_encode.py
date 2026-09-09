#!/usr/bin/env python3
"""Show (and optionally run) what ITER actually feeds its encoder.

The history-conditioned query is the part people get wrong when they reuse the
released checkpoint, so this prints the exact string, prefix included.

    python run_encode.py --setting i7 \
        --question "Which Ghanaian doctor sailed on the Copacabana?" \
        --previous "Ghanaian doctors educated in Scotland" \
                   "Belgian ship Copacabana passengers WWII" \
        --reasoning "I need to connect the doctor to the ship's passenger history." \
        --current "Ghanaian doctor Edinburgh clinic 1958"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import SETTINGS                                    # noqa: E402
from memory_utils import build_redesign_query                  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", default="i7", choices=sorted(SETTINGS),
                    help="query representation (default: i7, the paper's default)")
    ap.add_argument("--question", required=True, help="the user's main question")
    ap.add_argument("--current", required=True, help="the sub-query being issued now")
    ap.add_argument("--previous", nargs="*", default=[],
                    help="sub-queries already issued, oldest first")
    ap.add_argument("--reasoning", default="",
                    help="pre-search reasoning (only used by i6/i7)")
    ap.add_argument("--encode", metavar="MODEL",
                    help="also encode the query with this retriever and print its norm")
    args = ap.parse_args()

    setting = SETTINGS[args.setting]

    # ---- build the query exactly as the agent client does at search time ----
    interactions = [{"query": q, "visits": []} for q in args.previous]
    query = build_redesign_query(setting.style if setting.style != "plain" else "i0",
                                 args.question, args.current, interactions,
                                 tokenizer=None, pre_reasoning=args.reasoning)

    print("=== task prefix (prepended by the encoder) ===")
    print(setting.prefix)
    print("=== query ===")
    print(query)

    if args.encode:
        # ---- optional: prove the checkpoint loads and produces a vector ----
        import torch
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.encode)
        model = AutoModel.from_pretrained(args.encode, torch_dtype=torch.float16).eval()
        batch = tok(setting.prefix + query, return_tensors="pt",
                    truncation=True, max_length=setting.max_length)
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state[:, -1]      # last-token pooling
        vec = torch.nn.functional.normalize(hidden, dim=-1)
        print(f"=== embedding: dim {vec.shape[-1]}, norm {vec.norm().item():.4f} ===")


if __name__ == "__main__":
    main()
