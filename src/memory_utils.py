#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared [Memory] construction for the diversity-aware retriever (v2).

Both the data builder (src/data_builder.py) and the inference-time
query builder (search_agent/.../tool_search.py) import this module so the
[Memory] block is byte-identical between training and inference.

Default retriever query:
    [Q]      original question
    [Now]    current sub-query
    [Memory] cleaned post-visit reasoning of all prior visits

[Memory] uses the agent's post-visit reasoning (the thinking step after each
get_document/visit), which carries both the extracted fact and the gap signal
("doc doesn't mention X, search for Y"). Cleaning is rule-based so it runs
identically at train and inference (no LLM judge available at inference).
"""

import re


# -------------------------
# Reasoning cleaning
# -------------------------

# tool / system mechanics noise: docid fetch failures, truncation, retries, ...
_NOISE = re.compile(
    r"\b(docid|doc id|doc ids|fetch|truncat|internal index|internal database|"
    r"search tool return|get those|those ids|try \d{4,}|not correct|limited fetch|"
    r"expects specific|different format|full path|the system|the tool)\b", re.I)

# explicit question-restatement markers
_QREST = re.compile(
    r"(the question[:\s]|the user (asks|mention|want)|question asks|asks[:\s])", re.I)

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text):
    return set(_WORD.findall(text.lower()))


def clean_reasoning(reasoning, question, max_overlap=0.6):
    """Keep facts + gap/next-intent; drop tool-noise and question-restatement.

    Deterministic (no LLM) so train and inference produce identical output.
    """
    q_tokens = _tokens(question or "")
    sentences = re.split(r"(?<=[.!?])\s+", (reasoning or "").strip())
    kept = []
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        # tool / system mechanics
        if _NOISE.search(s):
            continue
        # explicit question restatement
        if _QREST.search(s):
            continue
        # high token-overlap with the original question -> restatement
        st = _tokens(s)
        if st and q_tokens and len(st) > 6:
            if len(st & q_tokens) / len(st) > max_overlap:
                continue
        kept.append(s)
    return " ".join(kept)


# -------------------------
# [Memory] block builder
# -------------------------
def _truncate_tokens(tokenizer, text, n):
    """First n tokens of text, decoded back to a string."""
    if tokenizer is None or not text:
        return text or ""
    ids = tokenizer.encode(text, add_special_tokens=False)[:n]
    return tokenizer.decode(ids)


def _token_len(tokenizer, text):
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_memory_block(
    visited_reasonings,
    question,
    tokenizer,
    per_visit_tokens=64,
    budget_tokens=256,
    min_words=8,
):
    """Build the [Memory] text from prior visits' reasoning.

    visited_reasonings: list of raw reasoning strings, oldest -> newest.
    Most recent visits are kept first under the token budget. Visits whose
    cleaned reasoning is too thin (e.g. pure tool-noise) are dropped.
    Returns "" if nothing survives.
    """
    kept = []
    used = 0
    for raw in reversed(visited_reasonings):
        cleaned = clean_reasoning(raw, question)
        if len(cleaned.split()) < min_words:
            continue
        snippet = _truncate_tokens(tokenizer, cleaned, per_visit_tokens)
        slen = _token_len(tokenizer, snippet)
        if used + slen > budget_tokens:
            break
        kept.append(snippet)
        used += slen
    return " ; ".join(kept)


def build_memory_query(
    question,
    current_query,
    visited_reasonings,
    tokenizer,
    per_visit_tokens=64,
    budget_tokens=256,
    min_words=8,
):
    """v1 retriever query: [Q] + [Now] + [Memory] (cleaned post-visit reasoning)."""
    lines = [f"[Q] {question}", f"[Now] {current_query}"]
    memory = build_memory_block(
        visited_reasonings, question, tokenizer,
        per_visit_tokens=per_visit_tokens,
        budget_tokens=budget_tokens,
        min_words=min_words,
    )
    if memory:
        lines.append(f"[Memory] {memory}")
    return "\n".join(lines)


# -------------------------
# v2 [Prev] block: prior sub-queries + the docs visited under each (title + snippet)
# -------------------------
def _extract_title(passage_text):
    """First-line / front-matter title of a corpus passage (mirrors tool_search)."""
    title = ""
    if passage_text.startswith("---\ntitle:"):
        parts = passage_text.split("\n")
        if len(parts) > 1:
            title = parts[1].replace("title:", "").strip().strip('"')
    if not title:
        first = passage_text.split("\n")[0].strip()
        title = first[:50] + "..." if len(first) > 50 else first
    return title


def build_prevdoc_query(
    question,
    current_query,
    prev_groups,
    get_text,
    tokenizer,
    per_doc_tokens=48,
    budget_tokens=256,
):
    """v2 retriever query: [Q] + [Now] + [Prev].

    prev_groups: list of {"query": sub_query, "docs": [docid, ...]} for sub-queries
    STRICTLY before the current one, oldest -> newest. Each entry pairs a prior
    sub-query with the title+snippet of the docs visited under it. Groups with no
    visited docs are skipped. Newest groups are kept first under the token budget.

    get_text(docid) -> passage text or None. Lets the builder back this with a
    corpus dict and inference back it with the live searcher (byte-identical form).
    """
    lines = [f"[Q] {question}", f"[Now] {current_query}"]
    entries = []
    used = 0
    for g in reversed(prev_groups):
        docs = [(d, get_text(d)) for d in g["docs"]]
        docs = [(d, t) for d, t in docs if t]
        if not docs:
            continue
        doc_strs = []
        for d, text in docs:
            title = _extract_title(text)
            snip = _truncate_tokens(tokenizer, text, per_doc_tokens)
            doc_strs.append(f"{title}: {snip}" if title else snip)
        entry = f"{g['query']} -> " + " ; ".join(doc_strs)
        elen = _token_len(tokenizer, entry)
        if used + elen > budget_tokens:
            break
        entries.append(entry)
        used += elen
    if entries:
        lines.append("[Prev] " + " | ".join(entries))
    return "\n".join(lines)


# -------------------------
# Redesigned structured query (docs/query_input_redesign.md)
# -------------------------
# Constant template with <empty> placeholders; history grouped per interaction,
# ALL interactions kept (no budget); doc/note items truncated per-item and
# tagged [docs_id:x] so the doc<->note pairing is explicit.

REDESIGN_INSTRUCTIONS = {
    "i0": "Given a web search query, retrieve relevant passages that answer the query",
    "i1": "Given the main question and the current sub-query, retrieve documents relevant to the current sub-query.",
    "i2": "Given the main question, the current sub-query, and the sub-queries already tried in previous interactions, retrieve documents relevant to the current sub-query that provide NEW information not yet found.",
    "i3": "Given the main question, the current sub-query, and previous interactions with the documents already visited, retrieve documents relevant to the current sub-query that provide NEW information beyond the visited documents.",
    "i4": "Given the main question, the current sub-query, and previous interactions with the documents already visited and notes taken on them, retrieve documents relevant to the current sub-query that provide NEW information beyond the visited documents.",
    "i5": "Given the main question, the current sub-query, and previous interactions with notes taken on the documents already read, retrieve documents relevant to the current sub-query that provide NEW information beyond what the notes cover.",
    # i6/i7 = i3/i4 with the agent-view document representation (64-token
    # snippets, no [docs_id:x] tags) -- same fields, so same instructions
    "i6": "Given the main question, the current sub-query, and previous interactions with the documents already visited, retrieve documents relevant to the current sub-query that provide NEW information beyond the visited documents.",
    "i7": "Given the main question, the current sub-query, and previous interactions with the documents already visited and notes taken on them, retrieve documents relevant to the current sub-query that provide NEW information beyond the visited documents.",
    # i8/i9/i10 borrow AgentIR's PRE-search reasoning (arXiv:2603.04384): the
    # <think> of the turn issuing the search, uncleaned and untruncated.
    # i8  = AgentIR's own input format (reasoning + sub-query, raw newlines);
    # i9  = i2's structured format + a Current Reasoning field;
    # i10 = i8's two fields in OUR template (Current Reasoning + Current
    #       Subquery lines, one-lined, <empty>) -- i8 vs i10 isolates the
    #       packaging, i10 vs i9 isolates main question + history.
    # Instructions are OURS (family style: fields + target + NEW-information
    # clause) -- our training data's positives are novelty-based, so the
    # instruction states our objective, not AgentIR's "answer the query".
    # AgentIR's own instruction is only used on the off-the-shelf AgentIR-4B
    # line (query-style=reason, bash/job_agentir/).
    "i8": "Given the agent's reasoning that led to the current sub-query, retrieve documents relevant to the current sub-query that provide NEW information beyond what the reasoning already covers.",
    "i9": "Given the main question, the agent's reasoning and the current sub-query it led to, and the sub-queries already tried in previous interactions, retrieve documents relevant to the current sub-query that provide NEW information not yet found.",
    "i10": "Given the agent's reasoning and the current sub-query it led to, retrieve documents relevant to the current sub-query that provide NEW information beyond what the reasoning already covers.",
}

# per-variant rendering: (with_docs, with_notes, docid_tags, per_doc_tokens)
# i6/i7 mirror i3/i4 but drop the [docs_id:x] tags and cut documents to the
# 64-token search-snippet length the agent saw; notes stay at 128.
_VARIANT_CFG = {
    "i2": (False, False, True, 128),
    "i3": (True, False, True, 128),
    "i4": (True, True, True, 128),
    "i5": (False, True, True, 128),
    "i6": (True, False, False, 64),
    "i7": (True, True, False, 64),
    "i9": (False, False, True, 128),  # i2 fields + Current Reasoning line
}


def _oneline(text):
    """Collapse all whitespace runs (incl. newlines) to single spaces so a
    doc/note never breaks the one-line-per-field template."""
    return " ".join((text or "").split())


# forward-planning sentences: next-step ACTIONS, not doc content. The modal
# must be followed by an action verb -- "we need to search X" is a plan (drop),
# "we need a village with population 173" is a gap statement (keep).
_PLAN_VERBS = r"(?:search|verify|check|open|look|try|examine|find|confirm|refine|visit|browse|query|google|think|consider|explore|dig|investigate|scroll|request|ask|re-?get|re-?read|retrieve)"
_PLAN = re.compile(
    r"(\blet'?s\s+(?:(?:also|now|then|first|next|just|instead|quickly)\s+)?" + _PLAN_VERBS + r"\b"
    r"|\b(?:we|i)\s+(?:need\s+to|should|will|may\s+need\s+to|might|could|can)\s+(?:(?:also|now|then|first|next|just|instead|quickly)\s+)?" + _PLAN_VERBS + r"\b"
    r"|^(?:search|try\s+searching|use\s+search)\b"
    r"|\bnext,?\s+(?:search|step|we'?ll|let'?s)\b"
    r"|\bmaybe\s+(?:search|check|try|look)\b)", re.I)


def clean_note(reasoning, question, max_overlap=0.6):
    """Redesign-format note cleaning: clean_reasoning + drop forward-planning
    sentences. Relevance verdicts ("Not helpful") and gap statements survive.
    The old mem/docs paths keep clean_reasoning untouched -- already-deployed
    models were trained with it."""
    base = clean_reasoning(reasoning, question, max_overlap)
    kept = [s for s in re.split(r"(?<=[.!?])\s+", base)
            if s.strip() and not _PLAN.search(s)]
    return " ".join(kept)


def build_redesign_query(
    variant,
    main_question,
    current_subquery,
    interactions,
    tokenizer,
    per_note_tokens=128,
    pre_reasoning="",
):
    """Build the structured query for one variant.

    interactions: [{"query": sub_query, "visits": [(docid, doc_text, raw_reasoning)]}]
        oldest -> newest, EXCLUDING the current search. doc_text = clean corpus
        text (no tool-output header); raw_reasoning may be "".
    pre_reasoning (i8/i9/i10): the agent's <think> text of the turn issuing
        this search, raw. Empty -> "Empty" in i8 (AgentIR's convention),
        "<empty>" in i9/i10 (ours).
    """
    if variant == "i0":
        return current_subquery
    if variant == "i8":
        return f"Reasoning: {pre_reasoning or 'Empty'}\n\nQuery: {current_subquery}"
    if variant == "i10":
        # i8's two fields in our template: one-lined reasoning, <empty> rule
        reason = _oneline(pre_reasoning)
        return (f"Current Reasoning: {reason if reason else '<empty>'}\n"
                f"Current Subquery: {current_subquery}")
    lines = [f"Main Question: {main_question}"]
    if variant == "i9":
        # AgentIR reasoning borrowed into the i2 structure: uncleaned and
        # untruncated, whitespace-collapsed to keep one-line-per-field.
        # Reasoning BEFORE the subquery -- the thought leads to the search.
        reason = _oneline(pre_reasoning)
        lines.append(f"Current Reasoning: {reason if reason else '<empty>'}")
    lines.append(f"Current Subquery: {current_subquery}")
    if variant == "i1":
        return "\n".join(lines)

    with_docs, with_notes, tags, doc_cap = _VARIANT_CFG[variant]
    if not interactions:
        lines.append("Previous Interactions: <empty>")
        return "\n".join(lines)

    lines.append("Previous Interactions:")
    for k, it in enumerate(interactions, 1):
        lines.append(f"Previous SubQuery {k}: {it['query']}")
        if with_docs:
            docs = []
            for d, text, _ in it["visits"]:
                if text:
                    snip = _truncate_tokens(tokenizer, _oneline(text), doc_cap)
                    docs.append(f"[docs_id:{d}] {snip}" if tags else snip)
            lines.append("Visited Documents: " + (" ; ".join(docs) if docs else "<empty>"))
        if with_notes:
            notes = []
            for d, _, raw in it["visits"]:
                cleaned = clean_note(raw, main_question)
                if cleaned:
                    snip = _truncate_tokens(tokenizer, _oneline(cleaned), per_note_tokens)
                    notes.append(f"[docs_id:{d}] {snip}" if tags else snip)
            lines.append("Visited Document Notes: " + (" ; ".join(notes) if notes else "<empty>"))
    return "\n".join(lines)
