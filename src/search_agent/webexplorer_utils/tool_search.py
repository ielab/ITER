from typing import Union, List
from qwen_agent.tools.base import BaseTool, register_tool
from transformers import AutoTokenizer

from memory_utils import build_memory_query, build_prevdoc_query, build_redesign_query

# redesigned structured styles (docs/query_input_redesign.md); i0 == plain
REDESIGN_STYLES = ("i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "i10")


@register_tool("search", allow_overwrite=True)
class SearchToolHandler(BaseTool):
    name = "search"

    def __init__(
        self,
        searcher,
        snippet_max_tokens: int = 512,
        k: int = 5,
        query_style: str = "plain",  # plain | mem | docs | i1..i5 (redesigned; i0 == plain)
        found_budget: int = 256,
        found_per_doc: int = 32,
        dedup_search: bool = False,
        dedup_pool_k: int = 100,
        visited_penalty_gamma: float = 0.0,
        visited_penalty_max_docs: int = 32,
        coverage_mmr_mode: str = "off",
        coverage_mmr_lambda: float = 0.2,
        coverage_mmr_overfetch_k: int = 100,
        coverage_mmr_max_history_docs: int = 200,
        label_feedback_mode: bool = False,
    ):
        super().__init__()

        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.query_style = query_style
        self.found_budget = found_budget
        self.found_per_doc = found_per_doc
        # cross-step dedup: never re-return a docid already surfaced by an earlier
        # search in this trajectory (the agent can revisit it from history).
        self.dedup_search = dedup_search
        self.dedup_pool_k = dedup_pool_k
        self.visited_penalty_gamma = visited_penalty_gamma
        self.visited_penalty_max_docs = visited_penalty_max_docs
        if self.visited_penalty_gamma > 0 and not getattr(self.searcher, "supports_visited_penalty", False):
            raise ValueError("--visited-penalty-gamma requires a FAISS-compatible searcher")
        # LRAT coverage-MMR (post-retrieval rerank) + online label feedback
        self.coverage_mmr_mode = coverage_mmr_mode
        self.coverage_mmr_lambda = coverage_mmr_lambda
        self.coverage_mmr_overfetch_k = coverage_mmr_overfetch_k
        self.coverage_mmr_max_history_docs = coverage_mmr_max_history_docs
        if self.coverage_mmr_mode not in ("off", "fixed"):
            raise ValueError("--coverage-mmr-mode must be off or fixed")
        if self.coverage_mmr_mode != "off" and not getattr(self.searcher, "supports_coverage_mmr", False):
            raise ValueError("--coverage-mmr-mode requires a FAISS searcher")
        self.label_feedback_mode = label_feedback_mode
        self.original_question = ""
        self.found_docids = []
        self.searched_docids = []
        self.visited_docids = []
        self.previous_queries = []
        self.search_traces = []
        # online label feedback: helpful -> PRF pull, not_helpful -> MMR penalty pool
        self.positive_labeled_docids = []
        self.negative_labeled_docids = []
        # memory-conditioned query state (mirrors src/data_builder.py)
        self.visited_reasonings = []     # v1 [Memory]: post-visit reasoning, oldest -> newest
        self.query_groups = []           # v2 [Prev]: [{"query":.., "docs":[docid,..]}] per search
        # redesigned format: per-interaction visits with paired reasoning
        self.interactions = []           # [{"query":.., "visits":[[docid, reasoning], ...]}]

        self.description = f"Performs a search on a knowledge source: supply a single 'query' string; the tool retrieves the top {self.k} most relevant results."

        self.tokenizer = None
        if (snippet_max_tokens and snippet_max_tokens > 0) or query_style != "plain":
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

    def _truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not self.tokenizer:
            return text

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        return text

    def reset_trajectory(self, original_question: str):
        self.original_question = original_question or ""
        self.found_docids = []
        self.searched_docids = []
        self.visited_docids = []
        self.previous_queries = []
        self.search_traces = []
        self.positive_labeled_docids = []
        self.negative_labeled_docids = []
        self.visited_reasonings = []
        self.query_groups = []
        self.interactions = []

    def update_doc_labels(self, positive_docids: List[str], negative_docids: List[str]):
        for d in positive_docids:
            if d not in self.positive_labeled_docids:
                self.positive_labeled_docids.append(d)
        for d in negative_docids:
            if d not in self.negative_labeled_docids:
                self.negative_labeled_docids.append(d)

    def get_search_traces(self):
        return list(self.search_traces)

    def add_found_docids(self, docids: List[str]):
        for docid in docids:
            docid = str(docid)
            if docid not in self.found_docids:
                self.found_docids.append(docid)

    def add_previous_query(self, query: str):
        query = query or ""
        if query:
            self.previous_queries.append(query)

    def add_visit_reasoning(self, reasoning: str):
        """Append the agent's post-visit reasoning (v1 [Memory]). Called by the agent."""
        reasoning = (reasoning or "").strip()
        if reasoning:
            self.visited_reasonings.append(reasoning)
            # redesigned format: the reasoning arrives one turn after its visit --
            # pair it with the newest visit that has none yet
            for g in reversed(self.interactions):
                if g["visits"]:
                    if not g["visits"][-1][1]:
                        g["visits"][-1][1] = reasoning
                    break

    def add_visited_docid(self, docid: str):
        """Record a visited docid under the current sub-query group (v2 [Prev])."""
        docid = str(docid)
        if docid not in self.visited_docids:
            self.visited_docids.append(docid)
        if self.query_groups:
            self.query_groups[-1]["docs"].append(docid)
        if self.interactions:
            self.interactions[-1]["visits"].append([docid, ""])

    def _start_query_group(self, query: str):
        self.query_groups.append({"query": query, "docs": []})
        self.interactions.append({"query": query, "visits": []})

    def _build_styled_query(self, current_query: str) -> str:
        """Memory-conditioned retriever query (shared with src/data_builder.py).

        mem  (query_style="mem"):  [Q] + [Now] + [Memory] (visited reasonings)
        docs (query_style="docs"): [Q] + [Now] + [Prev]   (prior sub-queries +
                                          their visited docs' title+snippet)
        i1..i5: redesigned structured format (docs/query_input_redesign.md)
        """
        if self.query_style in REDESIGN_STYLES:
            get_text = lambda d: (self.searcher.get_document(d) or {}).get("text") or ""
            inter = [{"query": g["query"],
                      "visits": [(d, get_text(d), r) for d, r in g["visits"]]}
                     for g in self.interactions]
            return build_redesign_query(
                self.query_style, self.original_question, current_query,
                inter, self.tokenizer)
        if self.query_style == "docs":
            get_text = lambda d: (self.searcher.get_document(d) or {}).get("text")
            return build_prevdoc_query(
                self.original_question, current_query,
                list(self.query_groups), get_text, self.tokenizer)
        return build_memory_query(
            self.original_question, current_query,
            list(self.visited_reasonings), self.tokenizer)

    def _extract_title(self, passage_text: str) -> str:
        title = ""
        if passage_text.startswith("---\ntitle:"):
            lines = passage_text.split("\n")
            if len(lines) > 1:
                title = lines[1].replace("title:", "").strip().strip("\"")
        if not title:
            first_line = passage_text.split('\n')[0].strip()
            title = first_line[:50] + "..." if len(first_line) > 50 else first_line
        return title

    def _format_results(self, results: List[dict], max_tokens: int):
        formatted = []
        for r in results:
            passage_text = r["text"]
            title = self._extract_title(passage_text)
            snippet = self._truncate(passage_text, max_tokens)
            formatted_result = f"DocID:{r['docid']}\n[{title}]\n{snippet}"
            formatted.append(formatted_result)

        return formatted

    def _record(self, r: dict) -> dict:
        """Structured per-doc record for the trace: docid + retrieval score."""
        return {"docid": str(r.get("docid")), "score": r.get("score")}


    def search_with_searcher(self, query: str, k: int = None):
        try:
            if k is None:
                k = self.k

            found_docids_before_search = list(self.found_docids)
            previous_queries_before_search = list(self.previous_queries)
            retrieval_query = self._build_styled_query(query) if self.query_style != "plain" else query
            # start this search's group AFTER building the query (which used prior groups)
            if self.query_style != "plain":
                self._start_query_group(query)
            hidden = []
            visited_penalty_docids = list(self.visited_docids) if self.visited_penalty_gamma > 0 else []
            search_kwargs = {}
            if self.visited_penalty_gamma > 0:
                search_kwargs = {
                    "history_docids": visited_penalty_docids,
                    "visited_penalty_gamma": self.visited_penalty_gamma,
                    "visited_penalty_max_docs": self.visited_penalty_max_docs,
                }
            # LRAT coverage-MMR: penalty pool = confirmed not_helpful if available, else all searched
            if self.coverage_mmr_mode != "off":
                mmr_history = (
                    list(self.negative_labeled_docids)
                    if self.label_feedback_mode and self.negative_labeled_docids
                    else list(self.searched_docids)
                )
                search_kwargs.update({
                    "coverage_mmr_mode": self.coverage_mmr_mode,
                    "coverage_mmr_history_docids": mmr_history,
                    "coverage_mmr_lambda": self.coverage_mmr_lambda,
                    "coverage_mmr_overfetch_k": self.coverage_mmr_overfetch_k,
                    "coverage_mmr_max_history_docs": self.coverage_mmr_max_history_docs,
                })
            # LRAT label-PRF: visited docs (strong) + rated-helpful docs (weak) pull the query
            if self.label_feedback_mode:
                if self.found_docids:
                    search_kwargs["visited_prf_docids"] = list(self.found_docids)
                rated_only = [d for d in self.positive_labeled_docids if d not in self.found_docids]
                if rated_only:
                    search_kwargs["rated_prf_docids"] = rated_only
            if self.dedup_search:
                # over-fetch, drop docids already surfaced in this trajectory, keep top-k
                pool = self.searcher.search(retrieval_query, max(k, self.dedup_pool_k), **search_kwargs)
                seen = set(self.searched_docids)
                # docs that would have ranked in top-k but are now hidden as already-seen
                hidden = [r for r in pool[:k] if str(r.get("docid")) in seen]
                results = [r for r in pool if str(r.get("docid")) not in seen][:k]
            else:
                results = self.searcher.search(retrieval_query, k, **search_kwargs)

            if not results:
                self.search_traces.append({
                    "tool_name": "search",
                    "original_query": query,
                    "retrieval_query": retrieval_query,
                    "returned_docids": [],
                    "returned": [],
                    "hidden": [self._record(r) for r in hidden],
                    "found_docids_before_search": found_docids_before_search,
                    "previous_queries_before_search": previous_queries_before_search,
                    "visited_docids_before_search": visited_penalty_docids,
                    "visited_penalty_gamma": self.visited_penalty_gamma,
                    "visited_penalty_max_docs": self.visited_penalty_max_docs,
                    "k": k,
                    "query_style": self.query_style,
                })
                self.add_previous_query(query)
                return f"No results found for '{query}'. Try with a more general query.", []

            docids = []
            for r in results:
                if "docid" in r:
                    docids.append(str(r["docid"]))

            # remember what was surfaced, so later searches can dedup against it
            for d in docids:
                if d not in self.searched_docids:
                    self.searched_docids.append(d)

            self.search_traces.append({
                "tool_name": "search",
                "original_query": query,
                "retrieval_query": retrieval_query,
                "returned_docids": docids,                       # ids only (builder reads this)
                "returned": [self._record(r) for r in results],  # docid + score
                "hidden": [self._record(r) for r in hidden],     # dedup-suppressed: docid + score
                "found_docids_before_search": found_docids_before_search,
                "previous_queries_before_search": previous_queries_before_search,
                "visited_docids_before_search": visited_penalty_docids,
                "visited_penalty_gamma": self.visited_penalty_gamma,
                "visited_penalty_max_docs": self.visited_penalty_max_docs,
                "k": k,
                "query_style": self.query_style,
            })

            formatted_results = self._format_results(results, self.snippet_max_tokens)

            content = f"A search for '{query}' found {len(formatted_results)} results:\n\n## Web Results\n" + "\n\n".join(formatted_results)
            if hidden:
                hidden_str = " ; ".join(f"DocID:{r['docid']} [{self._extract_title(r['text'])}]" for r in hidden)
                content += ("\n\n## Returned earlier (relevant docs hidden because an earlier search already returned "
                            "them — you may NOT have read them yet; use get_document to read any)\n"
                            f"{hidden_str}")
            self.add_previous_query(query)
            return content, docids

        except Exception as e:
            self.search_traces.append({
                "tool_name": "search",
                "original_query": query,
                "retrieval_query": retrieval_query if "retrieval_query" in locals() else query,
                "returned_docids": [],
                "returned": [],
                "hidden": [],
                "found_docids_before_search": list(self.found_docids),
                "previous_queries_before_search": list(self.previous_queries),
                "visited_docids_before_search": list(self.visited_docids),
                "visited_penalty_gamma": self.visited_penalty_gamma,
                "visited_penalty_max_docs": self.visited_penalty_max_docs,
                "k": k if k is not None else self.k,
                "query_style": self.query_style,
                "error": str(e),
            })
            self.add_previous_query(query)
            return f"Search error for query '{query}': {str(e)}", []

    def call(self, params: Union[str, dict], **kwargs):
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field", None

        if not isinstance(query, str):
            if isinstance(query, list):
                query = query[0]
            else:
                return "[Search] Invalid request format: 'query' must be a string, not an array", None
        
        response, docids = self.search_with_searcher(query)
        
        return response, docids

@register_tool("get_document", allow_overwrite=True)
class GetDocumentToolHandler(BaseTool):
    name = "get_document"
    def __init__(self, searcher):
        super().__init__()
        self.searcher = searcher
        self.description = "Retrieve full document content based on provided docid(s)."
        self.document_max_tokens = 512
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

    def _truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not self.tokenizer:
            return text

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        return text
    
    def call(self, params: Union[str, dict], **kwargs):
        try:
            docid = params["docid"]
        except:
            return "[get_document] Invalid request format: Input must be a JSON object containing 'docid' field", None
    
        if isinstance(docid, list):
            docid = docid[0]

        results = []
        collected_docids = []

        try:
            content = self.searcher.get_document(docid)
            if not content:
                results.append(f"[Document not found] docid={docid}")
            else:
                truncted_content = self._truncate(content['text'], self.document_max_tokens)
                results.append(f"Document {docid}:\n{truncted_content}.")
                collected_docids.append(str(docid))
        except Exception as e:
            results.append(f"[Error retrieving {docid}]: {str(e)}")

        response_text = "\n\n".join(results)
        return response_text, collected_docids
