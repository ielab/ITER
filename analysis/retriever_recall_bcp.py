#!/usr/bin/env python3
"""Retriever-ONLY recall on BrowseComp-Plus for the ablation models (no agent).

For each model: encode the 830 benchmark questions (model's own instruction;
structured styles use their FIRST-STEP query form with [Now]=question, since
there is no agent state to condition on), search its flat bcp_* index, report
macro-averaged recall@k and hit@k vs qrel_golds and qrel_evidence.

NOTE the structural limitation: this metric cannot exercise memory conditioning
at all — it measures single-shot ranking only.
"""
import argparse, json, pickle, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "tevatron" / "src"))
from tevatron.retriever.modeling import DenseModel  # noqa: E402

ILRAT = "Given a web search query, retrieve relevant passages that answer the query"
I0 = "Given the original question, the current sub-query, and notes from documents already read, retrieve documents containing NEW relevant information not yet found."
I2 = "Given the original question and the current sub-query, retrieve documents relevant to the current sub-query."
I3 = "Given the current sub-query and notes from documents already read, retrieve documents containing NEW relevant information not yet found."
I4 = "Given the original question, the current sub-query, notes from documents already read, and previous sub-queries with their visited documents, retrieve documents containing NEW relevant information not yet found."
I5 = "Given the original question, the current sub-query, and previous sub-queries with the documents already visited, retrieve documents containing NEW relevant information not yet found."

# label -> (model path/id, instruction, query_form)  form: plain | qnow
MODELS = {
    "lrat":      ("Yuqi-Zhou/LRAT-Qwen3-Embedding-0.6B", ILRAT, "plain"),
    "i0":        ("<RELEASED_ITER_0.6B_CHECKPOINT>", I0, "qnow"),
    "i1":        (str(ROOT / "models/ablation/input_i1"), ILRAT, "plain"),
    "i2":        (str(ROOT / "models/ablation/input_i2"), I2, "qnow"),
    "i3":        (str(ROOT / "models/ablation/input_i3"), I3, "now"),
    "i4":        (str(ROOT / "models/ablation/input_i4"), I4, "qnow"),
    "i5":        (str(ROOT / "models/ablation/input_i5"), I5, "qnow"),
    "4b_i0":     (str(ROOT / "models/ablation/scale_4b_i0"), I0, "qnow"),
    "4b_i4":     (str(ROOT / "models/ablation/scale_4b_i4"), I4, "qnow"),
    "8b_i0":     (str(ROOT / "models/ablation/scale_8b_i0"), I0, "qnow"),
    "8b_i4":     (str(ROOT / "models/ablation/scale_8b_i4"), I4, "qnow"),
}
INDEX_KEY = {"i0": "anchor_i0"}  # bcp index dir naming

def load_qrels(path):
    qrels = defaultdict(set)
    for line in open(path):
        parts = line.split()
        if len(parts) >= 4 and parts[3] != "0":
            qrels[parts[0]].add(parts[2])
    return qrels

def form_query(form, q):
    if form == "plain": return q
    if form == "now": return f"[Now] {q}"
    return f"[Q] {q}\n[Now] {q}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "experiments/bcp/retriever_recall_bcp.json"))
    args = ap.parse_args()

    queries = {}
    for line in open(ROOT / "datasets/topics-qrels/queries.tsv"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2: queries[parts[0]] = parts[1]
    gold = load_qrels(ROOT / "datasets/topics-qrels/qrel_golds.txt")
    evid = load_qrels(ROOT / "datasets/topics-qrels/qrel_evidence.txt")
    print(f"queries={len(queries)} gold-qids={len(gold)} evid-qids={len(evid)}")

    results = {}
    for label in args.models:
        path, instr, formtype = MODELS[label]
        idx_dir = ROOT / "data/indexes" / f"bcp_{INDEX_KEY.get(label, label)}"
        reps, lookup = pickle.load(open(idx_dir / "index-000.pkl", "rb"))
        reps = np.asarray(reps, dtype=np.float32)
        lookup = [str(x) for x in lookup]
        model = DenseModel.load(path, pooling="eos", normalize=True, torch_dtype=torch.float16).to("cuda").eval()
        tok = AutoTokenizer.from_pretrained(path, padding_side="left")
        if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
        prefix = f"Instruct: {instr}\nQuery: "
        qids = sorted(queries)
        texts = [prefix + form_query(formtype, queries[q]) for q in qids]
        embs = []
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i+64], padding=True, truncation=True, max_length=512, return_tensors="pt")
            enc = {k: v.to("cuda") for k, v in enc.items()}
            with torch.no_grad(), torch.amp.autocast("cuda"):
                embs.append(model.encode_query(enc).float().cpu().numpy())
        embs = np.concatenate(embs)
        scores = embs @ reps.T
        topk = np.argsort(-scores, axis=1)[:, :args.k]
        rg = rh = re_ = 0.0; ng = ne = 0
        for row, qid in enumerate(qids):
            docs = {lookup[j] for j in topk[row]}
            if gold.get(qid):
                rg += len(docs & gold[qid]) / len(gold[qid]); rh += bool(docs & gold[qid]); ng += 1
            if evid.get(qid):
                re_ += len(docs & evid[qid]) / len(evid[qid]); ne += 1
        results[label] = {"gold_recall@k": rg/max(ng,1), "gold_hit@k": rh/max(ng,1),
                          "evid_recall@k": re_/max(ne,1), "k": args.k, "n": ng}
        print(f"{label:8s} gold_recall@{args.k}={rg/max(ng,1):.4f} gold_hit@{args.k}={rh/max(ng,1):.4f} "
              f"evid_recall@{args.k}={re_/max(ne,1):.4f}")
        del model, reps; torch.cuda.empty_cache()
    json.dump(results, open(args.out, "w"), indent=1)
    print("wrote", args.out)

if __name__ == "__main__":
    main()
