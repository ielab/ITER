import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from prompts import format_query, STRONG_SYSTEM_PROMPT, DEDUP_NOTICE, LABEL_FEEDBACK_NOTICE
from searcher import SearcherType
from memory_utils import build_memory_query, build_prevdoc_query



logging.basicConfig(
    level=getattr(logging, os.getenv("LRAT_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def get_api_key() -> str:
    return os.getenv("API_KEYS") or os.getenv("API_KEY") or ""


def get_base_url() -> str:
    return os.getenv("URL") or os.getenv("BASE_URL") or ""


class SearchToolHandler:
    def __init__(
        self,
        searcher,
        snippet_max_tokens: int | None = None,
        k: int = 5,
        include_get_document: bool = True,
        dedup_search: bool = False,
        dedup_pool_k: int = 100,
        coverage_mmr_mode: str = "off",
        coverage_mmr_lambda: float = 0.2,
        coverage_mmr_overfetch_k: int = 100,
        label_feedback_mode: bool = False,
        no_prf: bool = False,
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

        # memory-conditioned query state (mirrors webexplorer_utils/tool_search.py)
        self.query_style = query_style
        self.original_question = original_question
        self.visited_reasonings = []     # v1 [Memory]: post-visit reasoning, oldest -> newest
        self.query_groups = []           # v2 [Prev]: [{"query":.., "docs":[docid,..]}] per search

        # coverage-MMR: penalize next results by similarity to already-seen docs
        self.coverage_mmr_mode = coverage_mmr_mode
        self.coverage_mmr_lambda = coverage_mmr_lambda
        self.coverage_mmr_overfetch_k = coverage_mmr_overfetch_k

        # on-the-fly feedback (intent-driven, hop-scoped Rocchio):
        #   same_goal (continue) -> PRF-pull toward window_visited + MMR-push from window not-visited
        #   new_goal  (pivot)    -> reset the window (start fresh for the new sub-goal)
        self.label_feedback_mode = label_feedback_mode
        self.no_prf = no_prf                # ablation: keep MMR-push, drop PRF-pull
        self.found_docids = []              # visited via get_document (global, all hops)
        self.window_searched = []           # surfaced since last pivot (positive+negative pool)
        self.window_visited = []            # visited since last pivot -> PRF positive pool
        self.feedback_events = []           # trace: {query, intent, n_prf, n_mmr} per search
        self.last_search_docids = []        # docids from the most recent search
        self.last_search_text = ""          # formatted result text of the most recent search

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
        # on-the-fly feedback (function-calling native): the agent declares whether this
        # search continues the same sub-goal or pivots to a new one; the harness turns the
        # PREVIOUS window's visit/no-visit behavior into PRF/MMR -- no extra turn or call.
        search_props = {
            "query": {
                "type": "string",
                "description": "Query to search the local knowledge base for relevant information",
            }
        }
        required = ["query"]
        if self.label_feedback_mode:
            search_props["intent"] = {
                "type": "string",
                "enum": ["same_goal", "new_goal"],
                "description": (
                    "Whether you have OBTAINED the fact your previous search was after:\n"
                    "  same_goal: you have NOT obtained it yet and are searching for it again -- "
                    "INCLUDING rephrasing, narrowing, or trying different keywords/angles for that "
                    "same missing fact. Use this whenever you keep searching because earlier "
                    "results did not give you the answer (even if you changed the wording).\n"
                    "  new_goal: you have ALREADY obtained that fact (e.g. you opened a document "
                    "and extracted it) and are now starting on a genuinely DIFFERENT fact.\n"
                    "Default to same_goal while still hunting the same missing fact; switch to "
                    "new_goal only once that fact is in hand."
                ),
            }
            required.append("intent")
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": self.searcher.search_description(self.k),
                    "parameters": {
                        "type": "object",
                        "properties": search_props,
                        "required": required,
                    },
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_document",
                    "description": self.searcher.get_document_description(),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "docid": {
                                "type": "string",
                                "description": "Document ID to retrieve",
                            }
                        },
                        "required": ["docid"],
                    },
                }
            }
        ]

    def execute_tool(self, tool_name: str, arguments: dict):
        if tool_name == "search":
            intent = arguments.get("intent", "same_goal") if self.label_feedback_mode else None
            if intent == "new_goal":
                # pivot: the previous sub-goal is resolved -> reset the feedback window
                self.window_searched = []
                self.window_visited = []
            return self._search(arguments["query"], intent=intent)
        elif tool_name == "get_document":
            return self._get_document(arguments["docid"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def add_visit_reasoning(self, reasoning: str):
        """Append the agent's post-visit reasoning (v1 [Memory]). Called by the caller loop."""
        reasoning = (reasoning or "").strip()
        if reasoning:
            self.visited_reasonings.append(reasoning)

    def add_visited_docid(self, docid: str):
        """Record a visited docid under the current sub-query group (v2 [Prev])."""
        if self.query_groups:
            self.query_groups[-1]["docs"].append(str(docid))

    def _start_query_group(self, query: str):
        self.query_groups.append({"query": query, "docs": []})

    def _build_styled_query(self, current_query: str) -> str:
        """Memory-conditioned retriever query (shared with src/data_builder.py).

        mem  (query_style="mem"):  [Q] + [Now] + [Memory] (visited reasonings)
        docs (query_style="docs"): [Q] + [Now] + [Prev]   (prior sub-queries +
                                          their visited docs' title+snippet)
        """
        if self.query_style == "docs":
            get_text = lambda d: (self.searcher.get_document(d) or {}).get("text")
            return build_prevdoc_query(
                self.original_question, current_query,
                list(self.query_groups), get_text, self.tokenizer)
        return build_memory_query(
            self.original_question, current_query,
            list(self.visited_reasonings), self.tokenizer)

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

    def _build_feedback_kwargs(self):
        """Hop-scoped Rocchio for a same_goal (continue) search.

        Within the current window: PRF-pull the query TOWARD visited docs (coherent, so a
        centroid pull), and MMR-push results AWAY from searched-but-not-visited docs
        (scattered, so a per-doc max penalty). Empty right after a pivot -> plain search.
        """
        kwargs = {}
        if not self.label_feedback_mode:
            # non-feedback modes keep the original behavior: MMR over all searched
            if self.coverage_mmr_mode != "off" and self.searched_docids:
                kwargs = {
                    "coverage_mmr_mode": self.coverage_mmr_mode,
                    "coverage_mmr_history_docids": list(self.searched_docids),
                    "coverage_mmr_lambda": self.coverage_mmr_lambda,
                    "coverage_mmr_overfetch_k": self.coverage_mmr_overfetch_k,
                }
            return kwargs
        # MMR push: away from this window's not-visited (skipped near-misses)
        not_visited = [d for d in self.window_searched if d not in set(self.window_visited)]
        if self.coverage_mmr_mode != "off" and not_visited:
            kwargs.update({
                "coverage_mmr_mode": self.coverage_mmr_mode,
                "coverage_mmr_history_docids": not_visited,
                "coverage_mmr_lambda": self.coverage_mmr_lambda,
                "coverage_mmr_overfetch_k": self.coverage_mmr_overfetch_k,
            })
        # PRF pull: toward this window's visited (what I read and am deepening)
        if self.window_visited and not self.no_prf:
            kwargs["visited_prf_docids"] = list(self.window_visited)
        return kwargs

    def _search(self, query: str, intent: str = None):
        hidden = []
        retrieval_query = self._build_styled_query(query) if self.query_style != "plain" else query
        # start this search's group AFTER building the query (which used prior groups)
        if self.query_style != "plain":
            self._start_query_group(query)
        mmr_kwargs = self._build_feedback_kwargs()
        if self.label_feedback_mode:
            not_visited = [d for d in self.window_searched if d not in set(self.window_visited)]
            self.feedback_events.append({
                "query": query, "intent": intent,
                "n_prf": len(self.window_visited), "n_mmr": len(not_visited),
            })
        if self.dedup_search:
            # over-fetch, drop docids already surfaced in this trajectory, keep top-k
            pool = self.searcher.search(retrieval_query, max(self.k, self.dedup_pool_k), **mmr_kwargs)
            seen = set(self.searched_docids)
            hidden = [c for c in pool[:self.k] if str(c.get("docid")) in seen]
            candidates = [c for c in pool if str(c.get("docid")) not in seen][:self.k]
        else:
            candidates = self.searcher.search(retrieval_query, self.k, **mmr_kwargs)

        for cand in candidates:
            text = cand.get("text", "")
            cand["snippet"] = self._truncate(text, self.snippet_max_tokens)

        # remember surfaced docids: global (dedup) + window (feedback pool)
        for cand in candidates:
            d = str(cand["docid"])
            if d not in self.searched_docids:
                self.searched_docids.append(d)
            if d not in self.window_searched:
                self.window_searched.append(d)

        results = []
        for cand in candidates:
            entry = {"docid": cand["docid"], "snippet": cand["snippet"]}
            if cand.get("score") is not None:
                entry["score"] = cand["score"]
            results.append(entry)

        # stash for the label-feedback rating call (handled in handle_conversation)
        self.last_search_docids = [str(c["docid"]) for c in candidates]
        self.last_search_text = json.dumps(results, indent=2, ensure_ascii=False)

        # dedup mode: when relevant docs were hidden (returned earlier), tell the agent
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

        # visited doc = PRF positive signal, both globally and in the current window
        d = str(docid)
        if d not in self.found_docids:
            self.found_docids.append(d)
        if d not in self.window_visited:
            self.window_visited.append(d)

        if "text" in result:
            result['text'] = self._truncate(result['text'], self.document_max_tokens)
        return json.dumps(result, indent=2, ensure_ascii=False)


def handle_conversation(args, searcher, query_text, qid=None):
    # per-query tool handler: dedup's searched_docids must never leak across threads
    tool_handler = SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=args.snippet_max_tokens,
        k=args.k,
        include_get_document=args.get_document,
        dedup_search=args.dedup_search,
        dedup_pool_k=args.dedup_pool_k,
        coverage_mmr_mode=args.coverage_mmr_mode,
        coverage_mmr_lambda=args.coverage_mmr_lambda,
        coverage_mmr_overfetch_k=args.coverage_mmr_overfetch_k,
        label_feedback_mode=args.label_feedback,
        no_prf=args.no_prf,
        query_style=getattr(args, "query_style", "plain"),
        original_question=query_text,
    )
    system_prompt = STRONG_SYSTEM_PROMPT if getattr(args, "strong", False) else args.system
    if args.dedup_search:
        system_prompt += DEDUP_NOTICE
    if args.label_feedback:
        system_prompt += LABEL_FEEDBACK_NOTICE
    tool_choice = 'auto'
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_query(query_text, args.query_template)}
    ]
    tools = tool_handler.get_tool_definitions()
    tool_usage_counts = {}
    total_usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "n_calls": 0}
    use_memory = getattr(args, "query_style", "plain") != "plain"
    # docids get_document'd in the PREVIOUS turn, awaiting this turn's content as their
    # post-visit reasoning (native function-calling puts reasoning+tool_calls in one
    # message, so "the reasoning after visiting doc X" is the NEXT assistant message's
    # content -- unlike the ReAct clients where it's a separate immediately-following step)
    pending_visited_docids = []
    # force_text_only: on the round right after "please answer now", omit the tools
    # schema entirely (not just tool_choice='none') so the model has no function to
    # call even if the provider doesn't strictly honor tool_choice -- MiMo does not,
    # and kept issuing search tool_calls straight through the forced round, so most
    # trajectories ran out the full turn budget with no final text answer.
    force_text_only = False

    # Assistant-prefill via continue_final_message (the Tongyi/WebExplorer fix for the
    # forced round) was tried here and DIDN'T work for this model/vLLM combo: two
    # different prefill texts both came back with exactly zero generated tokens on
    # 8-9/10 smoke queries (output length == prefill length, verbatim) -- a mechanical
    # incompatibility, not a prompt-design problem, so it's not used for this backbone.
    FORCED_ROUND_PREFILL = None

    for i in range(args.max_iterations):
        is_last_round = (i == args.max_iterations - 2)
        prefill = FORCED_ROUND_PREFILL if force_text_only else None
        response = None
        for i in range(5):
            try:
                # long timeout: local vLLM on long context (esp. MoE/gdn) can be slow;
                # a short default would falsely time out and trigger retry storms
                client = OpenAI(base_url=get_base_url(), api_key=get_api_key(), timeout=1800.0)
                if 'minimax' in args.model.lower():
                    extra_body={"reasoning_split": True}
                elif 'glm' in args.model.lower():
                    extra_body={
                        "thinking":{
                        "type":"enabled",
                        "clear_thinking": False
                    }}
                elif 'qwen3.5-4b' in args.model.lower():
                    # model card's recommended sampling for thinking-mode general tasks
                    extra_body={"top_k": 20}
                else:
                    extra_body={}
                call_messages = messages
                if prefill:
                    call_messages = messages + [{"role": "assistant", "content": prefill}]
                    extra_body["continue_final_message"] = True
                    extra_body["add_generation_prompt"] = False
                create_kwargs = dict(
                    model=args.model,
                    messages=call_messages,
                    temperature=args.temperature if args.temperature is not None else 1.0,
                    max_tokens=args.max_tokens,
                    extra_body=extra_body
                )
                if 'qwen3.5-4b' in args.model.lower():
                    create_kwargs["top_p"] = 0.95
                    create_kwargs["presence_penalty"] = 1.5
                if not force_text_only:
                    create_kwargs["tools"] = tools
                    create_kwargs["tool_choice"] = tool_choice
                response = client.chat.completions.create(**create_kwargs)
                break
            except Exception as e:
                if i < 5:
                    logger.warning("Chat completion attempt %s failed: %s", i + 1, e)
                    time.sleep(5)

        if response is None:
            break

        # accumulate token usage across every API call in this trajectory
        if response.usage:
            u = response.usage
            total_usage["input_tokens"] += u.prompt_tokens or 0
            total_usage["output_tokens"] += u.completion_tokens or 0
            total_usage["n_calls"] += 1
            ptd = getattr(u, "prompt_tokens_details", None)
            if ptd and getattr(ptd, "cached_tokens", None):
                total_usage["cached_input_tokens"] += ptd.cached_tokens

        res_msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        res_content = (prefill + (res_msg.content or "")) if prefill else (res_msg.content or "")

        # this turn's content is the post-visit reasoning for docs get_document'd LAST turn
        if use_memory and pending_visited_docids:
            tool_handler.add_visit_reasoning(res_content)
            for d in pending_visited_docids:
                tool_handler.add_visited_docid(d)
            pending_visited_docids = []

        # append as an explicit dict: mimo rejects the raw pydantic message
        # (its serialized tool_calls fail the API's tool_call_id validation)
        assistant_msg = {"role": "assistant", "content": res_content}
        # on the forced round, never act on tool_calls even if the server's parser
        # still produced them from leftover <tool_call> text in res_content -- the
        # prefill already commits this turn to being the final answer; executing a
        # "tool call" here would burn the last iteration with no text answer saved.
        if res_msg.tool_calls and not force_text_only:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in res_msg.tool_calls
            ]
        messages.append(assistant_msg)

        if res_msg.tool_calls and not force_text_only:
            # one tool message per tool_call, each carrying its tool_call_id — required
            # by the API even when a call fails (else "tool_call_id is not set").
            # MUST run before the is_last_round branch, so a tool_calls assistant
            # message is never followed by a non-tool message.
            did_search = False
            this_turn_visited = []
            for tool_call in res_msg.tool_calls:
                t_id = tool_call.id
                # vLLM's harmony tool parser (openai_tool_parser.py) splits the function
                # name off msg.recipient by "functions." with no further validation; when
                # gpt-oss emits back-to-back tool calls in one turn it sometimes runs the
                # next message's "<|channel|>..." header into the recipient field, leaking
                # it into the name (e.g. "search<|channel|>commentary") -- strip it here.
                t_name = tool_call.function.name.split("<")[0]
                try:
                    t_args = json.loads(tool_call.function.arguments)
                    tool_usage_counts[t_name] = tool_usage_counts.get(t_name, 0) + 1
                    result_output = tool_handler.execute_tool(t_name, t_args)
                    if t_name == "search":
                        did_search = True
                    elif t_name == "get_document":
                        this_turn_visited.append(str(t_args.get("docid")))
                except Exception as e:
                    result_output = f'Error: invalid tool call ({e}). Provide a valid "name" and "arguments".'
                messages.append({
                    "role": "tool",
                    "tool_call_id": t_id,
                    "name": t_name,
                    "content": result_output,
                })
            if use_memory:
                pending_visited_docids = this_turn_visited

        if is_last_round:
            messages.append({
                "role": "user",
                "content": "Retrieval complete. You are forbidden to call any tools now. "
                           "You must provide your final answer based on the above info."
            })
            tool_choice = 'none'
            force_text_only = True
            continue

        if finish_reason == "stop":
            break

    _persist_response(
        args.output_dir, args.model, messages, tool_usage_counts, qid, total_usage,
        tool_handler.feedback_events
    )
    return messages[-1]

def _persist_response(out_dir, model_name, messages, tool_counts, query_id, total_usage,
                      label_feedback_trace=None):
    import os
    import json
    import datetime
    import re

    try:
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        
        safe_qid = re.sub(r'[^\w\-_]', '_', str(query_id)) if query_id else "single"
        final_filename = os.path.join(out_dir, f"run_{safe_qid}_{ts}.json")

        T_START = "<think>"
        T_END = "</think>"
        think_pattern = rf"<{T_START}>(.*?)<{T_END}>"

        normalized_results = []
        
        for msg in messages:
            if hasattr(msg, 'model_dump'):
                m = msg.model_dump()
            elif isinstance(msg, dict):
                m = msg
            else:
                m = {"role": "unknown", "content": str(msg)}

            role = m.get("role")
            content = m.get("content") or ""

            if role == "assistant":
                found_think = re.search(think_pattern, content, re.DOTALL)
                if found_think:
                    normalized_results.append({
                        "type": "reasoning",
                        "output": found_think.group(1).strip()
                    })
                    content = re.sub(think_pattern, "", content, flags=re.DOTALL).strip()

                if content:
                    normalized_results.append({"type": "output_text", "output": content})

                if m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        raw_args = tc["function"].get("arguments", "{}")
                        if isinstance(raw_args, dict):
                            parsed_args = raw_args
                        else:
                            try:
                                parsed_args = json.loads(raw_args)
                            except:
                                parsed_args = {"raw_string": raw_args, "error": "json_parse_failed"}

                        normalized_results.append({
                            "type": "tool_call",
                            "id": tc.get("id"),
                            "tool_name": tc["function"].get("name"),
                            "arguments": parsed_args,
                            "output": None 
                        })

            elif role == "tool":
                res_id = m.get("tool_call_id")
                res_content = m.get("content")
                if isinstance(res_content, str) and len(res_content) > 200000:
                    res_content = res_content[:200000] + "...[truncated]"
                
                for entry in normalized_results:
                    if entry.get("type") == "tool_call" and entry.get("id") == res_id:
                        entry["output"] = res_content

        from utils import extract_retrieved_docids_from_result
        docids = extract_retrieved_docids_from_result(normalized_results)

        record = {
            "metadata": {
                "model": model_name,
                "timestamp": datetime.datetime.now().isoformat(),
                "usage": total_usage
            },
            "query_id": query_id,
            "tool_call_counts": tool_counts,
            "retrieved_docids": docids,
            "result": normalized_results,
            "label_feedback_trace": label_feedback_trace or [],
        }

        with open(final_filename, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.exception("Failed to persist response: %s", e)

def main():
    parser = argparse.ArgumentParser(description="MiniMax-M2 DeepSearch Client")
    parser.add_argument("--query", default="queries.tsv", help="Query text or TSV path")
    parser.add_argument("--model", default="MiniMax-M2.1", help="MiniMax Model Name")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--output-dir", default="runs/minimax_m2")
    parser.add_argument("--system", default="You are a helpful assistant with search tools.")
    parser.add_argument("--query-template", default="QUERY_TEMPLATE_NO_GET_DOCUMENT")
    parser.add_argument("--max-iterations", type=int, default=10, help="Max loops for tool calling")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=None)

    parser.add_argument("--searcher-type", required=True, choices=SearcherType.get_choices())
    parser.add_argument("--snippet-max-tokens", type=int, default=512)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--get-document", action="store_true")
    parser.add_argument("--strong", action="store_true",
                        help="Use STRONG_SYSTEM_PROMPT (general-model setting)")
    parser.add_argument("--query-style", choices=["plain", "mem", "docs"], default="plain",
                        help="Retriever query form: plain sub-query; mem = [Q]+[Now]+[Memory]; docs = [Q]+[Now]+[Prev]")
    parser.add_argument("--dedup-search", action="store_true",
                        help="Cross-step dedup: drop docids already surfaced earlier in this trajectory")
    parser.add_argument("--dedup-pool-k", type=int, default=100,
                        help="Over-fetch depth for --dedup-search before keeping top-k")
    parser.add_argument("--coverage-mmr-mode", default="off", choices=["off", "fixed"],
                        help="Cross-step MMR diversity mode (default: off)")
    parser.add_argument("--coverage-mmr-lambda", type=float, default=0.2,
                        help="MMR diversity weight lambda (default: 0.2)")
    parser.add_argument("--coverage-mmr-overfetch-k", type=int, default=100,
                        help="Over-fetch depth before MMR re-rank (default: 100)")
    parser.add_argument("--label-feedback", action="store_true",
                        help="Intent-driven on-the-fly feedback: same_goal -> PRF-pull visited + MMR-push not-visited; new_goal -> reset")
    parser.add_argument("--no-prf", action="store_true",
                        help="Ablation: disable the PRF-pull half; keep only intent-scoped MMR-push")

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
