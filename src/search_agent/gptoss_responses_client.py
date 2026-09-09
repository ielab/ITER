import argparse
import csv
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompts import format_query, STRONG_SYSTEM_PROMPT, DEDUP_NOTICE
from searcher import SearcherType
from memory_utils import build_redesign_query

# redesigned structured styles (docs/query_input_redesign.md); i0 == plain
REDESIGN_STYLES = ("i1", "i2", "i3", "i4", "i5", "i6", "i7")
from utils import extract_retrieved_docids_from_result


logging.basicConfig(
    level=getattr(logging, os.getenv("LRAT_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_api_key() -> str:
    return os.getenv("API_KEYS") or os.getenv("API_KEY") or ""


def get_base_url() -> str:
    return os.getenv("URL") or os.getenv("BASE_URL") or ""


# Same ITER-featured tool handler as openai_client.py's SearchToolHandler (dedup,
# memory-conditioned query, coverage-MMR, on-the-fly feedback), except get_tool_definitions()
# returns the Responses API's flat tool schema (no "function" wrapper) instead of the
# chat-completions nested one -- everything else is identical, copy kept in sync by hand
# since the two protocols are otherwise unrelated (see gptoss_responses_client.py module
# docstring / job_debug_gptoss_smoke.sh header for why gpt-oss needs its own protocol).
class SearchToolHandler:
    def __init__(
        self,
        searcher,
        snippet_max_tokens: int | None = None,
        k: int = 5,
        include_get_document: bool = True,
        dedup_search: bool = False,
        dedup_pool_k: int = 100,
        query_style: str = "plain",
        original_question: str = "",
    ):
        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.document_max_tokens = 512
        self.include_get_document = include_get_document
        # cross-step dedup: never re-return a docid already surfaced earlier in this trajectory
        self.dedup_search = dedup_search
        self.dedup_pool_k = dedup_pool_k
        self.searched_docids = []

        # memory-conditioned query state (mirrors qwen35_utils/tool_search.py)
        self.query_style = query_style
        self.original_question = original_question
        # redesigned format: per-interaction visits with paired reasoning
        self.interactions = []           # [{"query":.., "visits":[[docid, reasoning], ...]}]
        # i6/i7 pre-search reasoning: the reasoning items of the turn issuing this search
        self.current_thinking = None

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

    def get_tool_definitions(self):
        return [
            {
                "type": "function",
                "name": "search",
                "description": self.searcher.search_description(self.k),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query to search the local knowledge base for relevant information",
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "get_document",
                "description": self.searcher.get_document_description(),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "docid": {"type": "string", "description": "Document ID to retrieve"}
                    },
                    "required": ["docid"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ]

    def execute_tool(self, tool_name: str, arguments: dict):
        if tool_name == "search":
            return self._search(arguments["query"])
        elif tool_name == "get_document":
            return self._get_document(arguments["docid"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def set_current_thinking(self, thinking: str):
        """Store the reasoning of the turn issuing the current search (pre-search)."""
        thinking = (thinking or "").strip()
        self.current_thinking = thinking or None

    def add_visit_reasoning(self, reasoning: str):
        """Attach the agent's post-visit reasoning to the document it just read."""
        reasoning = (reasoning or "").strip()
        if reasoning:
            # the reasoning arrives one turn after its visit -- pair it with the
            # newest visit that does not have one yet
            for g in reversed(self.interactions):
                if g["visits"]:
                    if not g["visits"][-1][1]:
                        g["visits"][-1][1] = reasoning
                    break

    def add_visited_docid(self, docid: str):
        """Record a visited docid under the current sub-query group."""
        if self.interactions:
            self.interactions[-1]["visits"].append([str(docid), ""])

    def _start_query_group(self, query: str):
        self.interactions.append({"query": query, "visits": []})

    def _build_styled_query(self, current_query: str) -> str:
        if self.query_style in REDESIGN_STYLES:
            get_text = lambda d: (self.searcher.get_document(d) or {}).get("text") or ""
            inter = [{"query": g["query"],
                      "visits": [(d, get_text(d), r) for d, r in g["visits"]]}
                     for g in self.interactions]
            return build_redesign_query(
                self.query_style, self.original_question, current_query,
                inter, self.tokenizer,
                pre_reasoning=self.current_thinking or "")
        raise ValueError(f"unknown query style: {self.query_style}")

    def _extract_title(self, text: str) -> str:
        title = ""
        if text.startswith("---\ntitle:"):
            lines = text.split("\n")
            if len(lines) > 1:
                title = lines[1].replace("title:", "").strip().strip('"')
        if not title:
            first = text.split("\n")[0].strip()
            title = first[:50] + "..." if len(first) > 50 else first
        return title

    def _search(self, query: str):
        hidden = []
        if self.query_style != "plain":
            retrieval_query = self._build_styled_query(query)
            self._start_query_group(query)
        else:
            retrieval_query = query

        if self.dedup_search:
            pool = self.searcher.search(retrieval_query, max(self.k, self.dedup_pool_k))
            seen = set(self.searched_docids)
            hidden = [c for c in pool[:self.k] if str(c.get("docid")) in seen]
            candidates = [c for c in pool if str(c.get("docid")) not in seen][:self.k]
        else:
            candidates = self.searcher.search(retrieval_query, self.k)

        for cand in candidates:
            cand["snippet"] = self._truncate(cand.get("text", ""), self.snippet_max_tokens)

        for cand in candidates:
            d = str(cand["docid"])
            if d not in self.searched_docids:
                self.searched_docids.append(d)

        results = []
        for cand in candidates:
            entry = {"docid": cand["docid"], "snippet": cand["snippet"]}
            if cand.get("score") is not None:
                entry["score"] = cand["score"]
            results.append(entry)

        if hidden:
            payload = {
                "results": results,
                "returned_earlier_note": ("Relevant docs hidden because an earlier search already "
                                          "returned them — you may NOT have read them yet; use "
                                          "get_document to read any."),
                "returned_earlier": [
                    {"docid": c["docid"], "title": self._extract_title(c.get("text", ""))}
                    for c in hidden
                ],
            }
            return json.dumps(payload, indent=2, ensure_ascii=False)

        return json.dumps(results, indent=2, ensure_ascii=False)

    def _get_document(self, docid: str):
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})
        if "text" in result:
            result["text"] = self._truncate(result["text"], self.document_max_tokens)
        return json.dumps(result, indent=2, ensure_ascii=False)


def _reasoning_text(item: dict) -> str:
    """Best-effort text extraction from a Responses API 'reasoning' output item."""
    summary = item.get("summary")
    if isinstance(summary, list) and summary:
        parts = [s.get("text", "") for s in summary if isinstance(s, dict)]
        return "\n".join(p for p in parts if p)
    parts = []
    for c in item.get("content", []) or []:
        if isinstance(c, dict) and c.get("type") in {"reasoning_text", "output_text", "text"}:
            t = str(c.get("text", "")).strip()
            if t:
                parts.append(t)
    return "\n".join(parts)


def handle_conversation(args, searcher, query_text, qid=None):
    tool_handler = SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=args.snippet_max_tokens,
        k=args.k,
        include_get_document=args.get_document,
        dedup_search=args.dedup_search,
        dedup_pool_k=args.dedup_pool_k,
        query_style=getattr(args, "query_style", "plain"),
        original_question=query_text,
    )
    system_prompt = STRONG_SYSTEM_PROMPT if getattr(args, "strong", False) else args.system
    if args.dedup_search:
        system_prompt += DEDUP_NOTICE
    use_memory = getattr(args, "query_style", "plain") != "plain"

    messages = [{"role": "user", "content": format_query(query_text, args.query_template)}]
    tools = tool_handler.get_tool_definitions()
    tool_usage_counts = {}
    # docids get_document'd in the PREVIOUS turn -> this turn's reasoning items are their
    # post-visit reasoning (Responses API separates hidden CoT into "reasoning" items,
    # unlike the ReAct clients where free-form text and the next action share one message)
    pending_visited_docids = []
    force_text_only = False
    status = "incomplete"

    client = OpenAI(base_url=get_base_url(), api_key=get_api_key(), timeout=1800.0)

    for i in range(args.max_iterations):
        is_last_round = (i == args.max_iterations - 2)

        request = {
            "model": args.model,
            "max_output_tokens": args.max_tokens,
            "input": messages,
            "truncation": "auto",
            "reasoning": {"effort": args.reasoning_effort, "summary": "detailed"},
            "instructions": system_prompt,
        }
        if not force_text_only:
            request["tools"] = tools

        response = None
        for attempt in range(5):
            try:
                response = client.responses.create(**request)
                break
            except Exception as e:
                logger.warning("Responses API call attempt %s failed: %s", attempt + 1, e)
                import time
                time.sleep(5)
        if response is None:
            break

        output_items = response.model_dump(mode="python")["output"]

        # vLLM's Harmony->Responses parser only classifies a "functions.X" recipient as
        # type="function_call" when the model used the "commentary" channel; if the model
        # (inconsistently) emits the same recipient from the "analysis" channel instead, it
        # falls through to the generic MCP-call branch (server_label="functions", type=
        # "mcp_call", never executed/routed to us). The recipient still names OUR tool, so
        # normalize it back into a function_call item before it enters the transcript --
        # otherwise it's silently dropped (never executed) and echoing an "mcp_call" item
        # back as input on the next turn isn't a recognized type server-side either.
        for it in output_items:
            if it.get("type") == "mcp_call" and it.get("server_label") == "functions":
                it["type"] = "function_call"
                it["call_id"] = it.pop("id")

        messages.extend(output_items)

        # drop a dangling trailing bare-reasoning item and retry (LRAT's own workaround
        # for truncated turns that end mid-thought with no message/function_call)
        if output_items and output_items[-1].get("type") == "reasoning" and not force_text_only:
            messages.pop()
            continue

        if use_memory and pending_visited_docids:
            reasoning_text = "\n".join(
                _reasoning_text(it) for it in output_items if it.get("type") == "reasoning"
            )
            tool_handler.add_visit_reasoning(reasoning_text)
            for d in pending_visited_docids:
                tool_handler.add_visited_docid(d)
            pending_visited_docids = []

        function_calls = [it for it in output_items if it.get("type") == "function_call"] if not force_text_only else []

        # pre-search reasoning: the reasoning items of THIS turn are the thought that
        # led to this turn's search call (the Responses-API counterpart of the ReAct
        # clients' <think>). Must be set before the tool is executed.
        if any((fc.get("name") or "").startswith("search") for fc in function_calls):
            tool_handler.set_current_thinking("\n".join(
                _reasoning_text(it) for it in output_items if it.get("type") == "reasoning"
            ))

        if not function_calls:
            status = "completed"
            break

        this_turn_visited = []
        for fc in function_calls:
            # same run-on-header model quirk as the mcp_call misclassification above: the
            # name/recipient occasionally gets extra "commentary"/"json"/"<|channel|>..."
            # text appended when the model chains tool calls without a clean separator.
            name = (fc.get("name") or "").split("<")[0]
            if name.startswith("search"):
                name = "search"
            elif name.startswith("get_document"):
                name = "get_document"
            call_id = fc.get("call_id")
            try:
                fargs = json.loads(fc.get("arguments") or "{}")
                tool_usage_counts[name] = tool_usage_counts.get(name, 0) + 1
                result_output = tool_handler.execute_tool(name, fargs)
                if name == "get_document":
                    this_turn_visited.append(str(fargs.get("docid")))
            except Exception as e:
                result_output = f'Error: invalid tool call ({e}). Provide a valid "name" and "arguments".'
            messages.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": result_output,
            })
        if use_memory:
            pending_visited_docids = this_turn_visited

        if is_last_round:
            messages.append({
                "role": "user",
                "content": "Retrieval complete. You are forbidden to call any tools now. "
                           "You must provide your final answer based on the above info.",
            })
            force_text_only = True

    _persist_response(
        args.output_dir, args.model, messages, tool_usage_counts, qid, status
    )
    return messages[-1]


def _persist_response(out_dir, model_name, messages, tool_counts, query_id, status):
    import datetime
    import re

    try:
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        safe_qid = re.sub(r"[^\w\-_]", "_", str(query_id)) if query_id else "single"
        final_filename = os.path.join(out_dir, f"run_{safe_qid}_{ts}.json")

        call_output_by_id = {}
        for item in messages or []:
            if isinstance(item, dict) and item.get("type") == "function_call_output":
                call_output_by_id[item.get("call_id")] = item.get("output")

        normalized_results = []
        for item in messages or []:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "function_call":
                raw_args = item.get("arguments") or "{}"
                try:
                    parsed_args = json.loads(raw_args)
                except Exception:
                    parsed_args = {"raw_string": raw_args, "error": "json_parse_failed"}
                normalized_results.append({
                    "type": "tool_call",
                    "id": item.get("call_id"),
                    "tool_name": item.get("name"),
                    "arguments": parsed_args,
                    "output": call_output_by_id.get(item.get("call_id")),
                })
            elif itype == "reasoning":
                normalized_results.append({
                    "type": "reasoning",
                    "output": _reasoning_text(item),
                })
            elif itype == "message":
                parts = item.get("content", []) or []
                text_chunks = [
                    str(p.get("text", "")) for p in parts
                    if isinstance(p, dict) and p.get("type") == "output_text"
                ]
                text = "\n".join(c for c in text_chunks if c).strip()
                if text:
                    normalized_results.append({"type": "output_text", "output": text})

        docids = extract_retrieved_docids_from_result(normalized_results)

        record = {
            "metadata": {"model": model_name, "timestamp": datetime.datetime.now().isoformat(), "status": status},
            "query_id": query_id,
            "tool_call_counts": tool_counts,
            "retrieved_docids": docids,
            "result": normalized_results,
        }
        with open(final_filename, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.exception("Failed to persist response: %s", e)


def main():
    parser = argparse.ArgumentParser(description="gpt-oss DeepSearch Client (Responses API)")
    parser.add_argument("--query", default="queries.tsv", help="Query text or TSV path")
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--max-tokens", type=int, default=10000, help="max_output_tokens")
    parser.add_argument("--output-dir", default="runs/gptoss")
    parser.add_argument("--system", default="You are a helpful assistant with search tools.")
    parser.add_argument("--query-template", default="QUERY_TEMPLATE")
    parser.add_argument("--max-iterations", type=int, default=50, help="Max loops for tool calling")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "high"])

    parser.add_argument("--searcher-type", required=True, choices=SearcherType.get_choices())
    parser.add_argument("--snippet-max-tokens", type=int, default=512)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--get-document", action="store_true")
    parser.add_argument("--strong", action="store_true", help="Use STRONG_SYSTEM_PROMPT")
    parser.add_argument("--query-style", choices=["plain", "i1", "i2", "i3", "i4", "i5", "i6", "i7"], default="plain")
    parser.add_argument("--dedup-search", action="store_true")
    parser.add_argument("--dedup-pool-k", type=int, default=100)

    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)
    args = parser.parse_args()

    searcher = searcher_class(args)

    if args.query.endswith(".tsv"):
        queries = []
        with open(args.query, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 2:
                    queries.append((row[0].strip(), row[1].strip()))

        processed_ids = set()
        out_dir = Path(args.output_dir).expanduser().resolve()
        if out_dir.exists():
            for json_path in out_dir.glob("run_*.json"):
                try:
                    with json_path.open("r", encoding="utf-8") as jf:
                        meta = json.load(jf)
                        qid_saved = meta.get("query_id")
                        if qid_saved:
                            processed_ids.add(str(qid_saved))
                except Exception:
                    continue

        remaining_queries = [q for q in queries if q[0] not in processed_ids]
        logger.info("Dataset total: %s", len(queries))
        logger.info("Already processed: %s", len(processed_ids))
        logger.info("Remaining to run: %s", len(remaining_queries))
        if not remaining_queries:
            logger.info("All queries in the TSV have been processed. Exiting.")
            return

        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = {
                executor.submit(handle_conversation, args, searcher, qtext, qid): qid
                for qid, qtext in remaining_queries
            }
            with tqdm(total=len(remaining_queries), desc="Processing Queries") as pbar:
                for future in as_completed(futures):
                    qid = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error("Query %s failed: %s", qid, e)
                    pbar.update(1)
    else:
        logger.info("Processing single query")
        handle_conversation(args, searcher, args.query, None)


if __name__ == "__main__":
    load_dotenv()
    main()
