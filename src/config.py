"""The experiment table for ITER.

Every setting reported in the paper is defined here: which query
representation it uses, what instruction prefixes its queries, how long its
inputs may be, and which agent backbones it was evaluated with. The run_*.py
entry points read this file and nothing else, so the numbers in the paper and
the commands in the README cannot drift apart.

The query representations i0-i7 are numbered exactly as in the paper.
"""

from dataclasses import dataclass, field
from typing import Optional

from memory_utils import REDESIGN_INSTRUCTIONS

# ---------------------------------------------------------------- query input

# The instruction that prefixes every query lives in ONE place,
# memory_utils.REDESIGN_INSTRUCTIONS, and is looked up from there.
INSTRUCTION_FORMAT = "Instruct: {}\nQuery: {}"


def task_prefix(style: str) -> str:
    """The encoder-side prefix for a query style ('Instruct: ...\\nQuery: ')."""
    # "plain" is the client's name for the bare sub-query; its instruction is i0's
    key = "i0" if style == "plain" else style
    return f"Instruct: {REDESIGN_INSTRUCTIONS[key]}\nQuery: "


# ------------------------------------------------------------------- settings

@dataclass
class Setting:
    """One row of the paper's tables: a retriever plus how it is queried."""
    style: str                              # --query-style passed to the agent client
    max_length: int                         # query truncation for the encoder
    retriever: str                          # HF id or local checkpoint path
    dedup: bool = True                      # de-duplicated harness (collection setting)
    task_prefix_override: Optional[str] = None   # external models with their own prefix

    @property
    def prefix(self) -> str:
        return self.task_prefix_override or task_prefix(self.style)


# `plain` is the client's name for the bare current sub-query, i0's style.
SETTINGS = {
    # --- ITER, one per query representation ---
    # Only the default (i7) is released; the ablation variants have to be
    # trained with run_train.py, which writes to models/iter_<setting>.
    "i0": Setting(style="plain", max_length=512,  retriever="models/iter_i0"),
    "i1": Setting(style="i1",    max_length=8192, retriever="models/iter_i1"),
    "i2": Setting(style="i2",    max_length=8192, retriever="models/iter_i2"),
    "i3": Setting(style="i3",    max_length=8192, retriever="models/iter_i3"),
    "i4": Setting(style="i4",    max_length=8192, retriever="models/iter_i4"),
    "i5": Setting(style="i5",    max_length=8192, retriever="models/iter_i5"),
    "i6": Setting(style="i6",    max_length=8192, retriever="models/iter_i6"),
    "i7": Setting(style="i7",    max_length=8192,
                  retriever="ielabgroup/ITER-Qwen3-Embedding-0.6B"),
    "i7-4b": Setting(style="i7", max_length=8192,
                     retriever="ielabgroup/ITER-Qwen3-Embedding-4B"),

    # --- baselines ---
    # BM25 is i0's style with a sparse searcher; see run_eval.py --searcher
    "base": Setting(style="plain", max_length=512, dedup=False,
                    retriever="Qwen/Qwen3-Embedding-0.6B"),
    "lrat": Setting(style="plain", max_length=512, dedup=False,
                    retriever="Yuqi-Zhou/LRAT-Qwen3-Embedding-0.6B"),
    # AgentIR is served exactly as its model card specifies: its own
    # instruction, and no trailing space after "Query:".
    "agentir": Setting(
        style="i6", max_length=8192, dedup=False, retriever="Tevatron/AgentIR-4B",
        task_prefix_override=(
            "Instruct: Given a user's reasoning followed by a web search query, "
            "retrieve relevant passages that answer the query while incorporating "
            "the user's reasoning\nQuery:")),
}

# ------------------------------------------------------------------ backbones

@dataclass
class Backbone:
    """A deep-research agent: which model to serve and which client drives it."""
    model: str
    client: str                             # tongyi | qwen35 | gptoss
    max_model_len: int = 262144
    vllm_flags: list = field(default_factory=list)


BACKBONES = {
    # Tongyi produced the training trajectories; the other five are unseen.
    "tongyi": Backbone("Alibaba-NLP/Tongyi-DeepResearch-30B-A3B", "tongyi",
                       max_model_len=98304, vllm_flags=["--trust-remote-code"]),
    "qwen3.5-4b":  Backbone("Qwen/Qwen3.5-4B",  "qwen35", vllm_flags=["--gdn-prefill-backend", "triton"]),
    "qwen3.5-9b":  Backbone("Qwen/Qwen3.5-9B",  "qwen35", vllm_flags=["--gdn-prefill-backend", "triton"]),
    "qwen3.5-27b": Backbone("Qwen/Qwen3.5-27B", "qwen35", vllm_flags=["--gdn-prefill-backend", "triton"]),
    "qwen3.6-27b": Backbone("Qwen/Qwen3.6-27B", "qwen35", vllm_flags=["--gdn-prefill-backend", "triton"]),
    "gptoss-120b": Backbone("openai/gpt-oss-120b", "gptoss",
                            vllm_flags=["--tensor-parallel-size", "2", "--max-num-seqs", "16",
                                        "--enforce-eager", "--disable-custom-all-reduce"]),
}

# ----------------------------------------------------------------- benchmarks

@dataclass
class Benchmark:
    queries: str
    answers: str
    corpus: str
    metric: str                             # exact_match | llm_judge


BENCHMARKS = {
    "infoseek": Benchmark("datasets/InfoSeek-Eval.tsv",
                          "datasets/InfoSeek-Eval.tsv",
                          "data/corpus.jsonl", "exact_match"),
    "bcp":      Benchmark("datasets/topics-qrels/queries.tsv",
                          "datasets/browsecomp-plus.tsv",
                          "data/browse-comp-plus-corpus.jsonl", "llm_judge"),
}

# -------------------------------------------------------------------- serving

# How the search tool is presented to the agent at evaluation time. Identical
# for every retriever, so no comparison is confounded by the interface.
SERVING = {
    "k": 10,                      # results per search
    "snippet_max_tokens": 64,     # snippet length in the tool output
    "max_turns": 50,              # tool-calling budget per question
    "dedup_pool_k": 100,          # candidates over-fetched before de-duplication
    "temperature": 0.6,
    "top_p": 0.95,
    "presence_penalty": 1.1,
}

# Trajectory collection ran hotter than evaluation, for candidate diversity.
COLLECTION = dict(SERVING, temperature=0.85, dedup=True)

# ------------------------------------------------------------------- training

# Fixed across every variant in the paper (Section 5.4, Table 5).
TRAINING = {
    "base_model": "Qwen/Qwen3-Embedding-0.6B",
    "learning_rate": 1e-6,
    "num_train_epochs": 2,
    "per_device_train_batch_size": 32,
    "warmup_ratio": 0.1,
    "train_group_size": 10,          # 1 positive + 9 negatives
    "passage_max_len": 512,
    "temperature": 0.02,
    "pooling": "last_token",
    "normalize": True,
    # tiered negatives: (redundancy, hard, weak) weights and per-tier caps
    "neg_w_div": 3.0,
    "neg_w_hard": 1.0,
    "neg_w_weak": 0.3,
    "div_neg_cap": 3,
    "hard_neg_cap": 3,
}

# -------------------------------------------------------------------- indexing

INDEXING = {
    "passage_max_len": 512,          # must match TRAINING["passage_max_len"]
    "pooling": "eos",
    "normalize": True,
    "index_type": "HNSW32",          # wiki only; BCP is small enough to stay flat
    "ef_construction": 200,
    "ef_search": 256,
}

# --------------------------------------------------------------------- verify

# The LLM that judges whether a visited document actually helped (positives),
# and the official BrowseComp-Plus answer judge.
VERIFIER_MODEL = "Qwen/Qwen3-30B-A3B-Thinking-2507"

# Retrieval backends used to collect the 4 x 10,000 training trajectories.
COLLECTION_BACKENDS = ["bm25", "qwen3-0.6b", "qwen3-4b", "qwen3-8b"]
