# Code is mostly based on the original Alibaba-NLP/DeepResearch inference script
# Modified to use only our local search tool to adhere to BrowseComp-Plus evaluation

import json5
import logging
import os
import re
from typing import Dict, List, Optional, Union
from qwen_agent.llm.schema import Message
from qwen_agent.utils.utils import build_text_completion_prompt
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError
import tiktoken
from transformers import AutoTokenizer 
from datetime import datetime
from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import Message
from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
from qwen_agent.tools import BaseTool
import time

# label-feedback: parse "DocID:<id> -> helpful|not_helpful" rating lines
_LABEL_RE = re.compile(r'DocID[:\s]+(\d+)[^a-z\n]{0,30}(not[_\s]helpful|helpful)', re.IGNORECASE)


SYSTEM_PROMPT_SEARCH_ONLY = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform local web searches then returns a string of the top search results. Accepts a single query.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query."}}, "required": ["query"], "example": {"name": "search", "arguments": {"query": ["xxxx"]}}, "uniqueItems": true}}}
{"type": "function", "function": {"name": "get_document", "description": "Retrieve the full content one documents given document ID.", "parameters": {"type": "object", "properties": {"docid": {"type": "string", "description": "The document ID(s) to retrieve."}}, "required": ["docid"], "example": {"name": "get_document", "arguments": {"docid": xxx}}, "uniqueItems": true}}}


For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
"""

STRONG_SYSTEM_PROMPT_SEARCH_ONLY = """You are a meticulous research agent answering a hard, multi-constraint question. The answer is NOT in your memory; you must find it through search.

# How you must work
- Treat the question as several distinct clues/criteria. Every single criterion must be verified against retrieved documents before you answer.
- Do NOT answer from your own prior knowledge or guesses. If you are tempted to recall the answer, search to confirm it instead.
- Issue many focused searches, one clue at a time, reformulating queries when results are weak. Do not stop after one or two searches.
- A search result only shows a short snippet. Before you rely on a document, call get_document to read its full text and check the details.
- Only produce the final answer once every criterion is supported by evidence you actually retrieved. If anything is unverified, keep searching.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Perform local web searches then returns a string of the top search results. Accepts a single query.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query."}}, "required": ["query"], "example": {"name": "search", "arguments": {"query": ["xxxx"]}}, "uniqueItems": true}}}
{"type": "function", "function": {"name": "get_document", "description": "Retrieve the full content one documents given document ID.", "parameters": {"type": "object", "properties": {"docid": {"type": "string", "description": "The document ID(s) to retrieve."}}, "required": ["docid"], "example": {"name": "get_document", "arguments": {"docid": xxx}}, "uniqueItems": true}}}


For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
"""

DEDUP_NOTICE = """

# Retriever behavior
The search tool de-duplicates across steps: a document returned by an earlier search will NOT appear again in later search results (this keeps each search focused on new material). When a hidden document is relevant to the current search, it is listed under a "Returned earlier" section — returned means it appeared in a previous result list, NOT that you have read it. The short snippets shown in results are never enough to judge a document: before drawing conclusions from any document, read its full content with get_document using its DocID, whether it comes from the current results or the "Returned earlier" list."""

TRUNCATED_MESSAGE = """
--- Maximum Length Limit Reached ---
You have reached the maximum length limit. 
The response is truncated."""
FINAL_MESSAGE = """
--- Final Step Reached ---
Now you reach the final step. 
You are forbidden to call any tools.
You must offer your final answer now."""

from webexplorer_utils.tool_search import SearchToolHandler

OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

MAX_LLM_CALL_PER_RUN = int(os.getenv('MAX_LLM_CALL_PER_RUN', 50))


import random
import datetime


logger = logging.getLogger(__name__)


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")

class MultiTurnReactAgent(FnCallAgent):
    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 search_tool_handler: Optional[SearchToolHandler] = None,
                 get_document_handler: Optional[SearchToolHandler] = None,
                 **kwargs):

        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model"]
        self.search_tool = search_tool_handler
        self.get_document_tool = get_document_handler
        self.max_generation = 4096
        self.enable_thinking = True
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_local_path) 

    
    def call_server(self, msgs, planning_port, max_tries=10, prefill=None):
        """prefill: if set, hard-forces the assistant's reply to start with this exact
        string via vLLM's `continue_final_message` (assistant-prefill) -- the model
        physically cannot emit a <tool_call> first, unlike a text-only "please answer
        now" nudge which it can (and does) ignore. vLLM returns only the continuation,
        so the caller gets `prefill + continuation` back.
        """

        openai_api_key = "EMPTY"
        openai_api_base = f"http://127.0.0.1:{planning_port}/v1"

        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
            timeout=600.0,
        )

        base_sleep_time = 1

        call_msgs = msgs
        extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": self.enable_thinking  # 或 True
            }
        }
        if prefill:
            call_msgs = msgs + [{"role": "assistant", "content": prefill}]
            extra_body["continue_final_message"] = True
            extra_body["add_generation_prompt"] = False

        for attempt in range(max_tries):
            try:
                chat_response = client.chat.completions.create(
                    model=self.model,
                    messages=call_msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=0.6,
                    top_p=0.95,
                    logprobs=True,
                    max_tokens=self.max_generation,
                    seed=2026,
                    presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1),
                    extra_body=extra_body,
                )
                content = chat_response.choices[0].message.content
                finish_reason = chat_response.choices[0].finish_reason

                if content and content.strip():
                    if prefill:
                        content = prefill + content
                    return content.strip(), finish_reason
                else:
                    logger.warning(
                        "Attempt %s received an empty response. enable_thinking=%s message=%s",
                        attempt + 1,
                        self.enable_thinking,
                        chat_response.choices[0].message,
                    )

            except (APIError, APIConnectionError, APITimeoutError) as e:
                logger.warning("Attempt %s failed with an API/network error: %s", attempt + 1, e)
            except Exception as e:
                logger.warning("Attempt %s failed with an unexpected error: %s", attempt + 1, e)

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30) 
                
                logger.info("Retrying in %.2f seconds...", sleep_time)
                time.sleep(sleep_time)
            else:
                logger.error("All retry attempts have been exhausted. The call has failed.")
        
        return "vllm server error!!!", "error"

    def count_tokens(self, messages, model="gpt-4o"):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True
        )
        t_count = len(input_ids)
        return t_count

    def _rate_search_results(self, search_result_text: str, original_question: str,
                              docids: list, planning_port: int, messages: list):
        """Rating call using the agent's own conversation history — same agent, extra hidden step."""
        docid_lines = "\n".join(f"DocID:{d} ->" for d in docids)
        rating_user_msg = {
            "role": "user",
            "content": (
                f"The above search just returned the following results. "
                f"Please evaluate each one for its potential to help answer the research question. "
                f"Your feedback will be used to improve the future retriever behaviour.\n\n"
                f"{search_result_text}\n\n"
                f"For each DocID, complete the label — helpful or not_helpful:\n"
                f"  helpful     — contains any useful information or clues, even partial\n"
                f"  not_helpful — clearly off-topic or does not contribute at all\n\n"
                f"Output only the completed label lines, no explanation:\n{docid_lines}"
            ),
        }
        msgs = messages + [rating_user_msg]
        old_thinking = self.enable_thinking
        old_max_gen = self.max_generation
        self.enable_thinking = False
        self.max_generation = 256
        content, _ = self.call_server(msgs, planning_port)
        self.enable_thinking = old_thinking
        self.max_generation = old_max_gen
        positive, negative = [], []
        for m in _LABEL_RE.finditer(content):
            docid, label = m.group(1), m.group(2).lower()
            if 'not' in label:
                negative.append(docid)
            else:
                positive.append(docid)
        return positive, negative


    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        self.model=model
        try:
            question = data['item']['question']
        except:
            raw_msg = data['item']['messages'][1]["content"]
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg

        tool_call_counts = {}  # Only successful tool calls
        tool_call_counts_all = {}  # All tool calls (successful and failed)
        retrieved_docids = []
        tool_traces = []

        start_time = time.time()
        planning_port = data['planning_port']
        answer = data['item']['answer']
        self.user_prompt = question
        if self.search_tool:
            self.search_tool.reset_trajectory(question)
        if os.getenv("LRAT_STRONG_PROMPT") == "1":
            system_prompt = STRONG_SYSTEM_PROMPT_SEARCH_ONLY
        else:
            system_prompt = SYSTEM_PROMPT_SEARCH_ONLY
        if self.search_tool and getattr(self.search_tool, "dedup_search", False):
            system_prompt += DEDUP_NOTICE
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round = 0
        pending_visit = False  # a doc was just read; its post-visit reasoning is the next think
        while num_llm_calls_available > 0:
            round += 1

            if round == 1:
                self.max_generation = 4096
            elif round == 2:
                self.max_generation = 2048
            else:
                self.max_generation = 1024

            num_llm_calls_available -= 1
            content, finish_reason = self.call_server(messages, planning_port)

            self.enable_thinking = True

            if finish_reason == "length" and num_llm_calls_available > 0:
                messages.append({"role": "assistant", "content": content})

                truncated_msg = (
                    "ERROR: Your previous thought was too long and has been discarded. "
                    "Now, skip all reasoning and directly provide the <tool_call> or <answer>."
                )
                messages.append({"role": "user", "content": f"<tool_response>\n{truncated_msg}\n</tool_response>"})
                self.enable_thinking = False
                continue

            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]

            messages.append({"role": "assistant", "content": content.strip()})

            # capture post-visit reasoning for [Memory]: this turn's think reflects on
            # the doc read last turn (mirrors data_builder's _reasoning_after).
            if self.search_tool and pending_visit:
                thinks = re.findall(r'<think>(.*?)</think>', content, re.DOTALL)
                self.search_tool.add_visit_reasoning(" ".join(t.strip() for t in thinks))
                pending_visit = False

            if '<tool_call>' in content and '</tool_call>' in content:
                tool_call = content.split('<tool_call>')[1].split('</tool_call>')[0]
                try:
                    tool_call = json5.loads(tool_call)
                    tool_name = tool_call.get('name', '')
                    tool_args = tool_call.get('arguments', {})

                    tool_call_counts_all[tool_name] = tool_call_counts_all.get(tool_name, 0) + 1

                    result, docids = self.custom_call_tool(tool_name, tool_args)
                    if tool_name in ("get_document", "visit"):
                        pending_visit = True

                    # label-feedback: hidden rating step after a search (1 extra LLM call)
                    if (tool_name == 'search' and docids
                            and getattr(self.search_tool, 'label_feedback_mode', False)
                            and num_llm_calls_available > 0):
                        num_llm_calls_available -= 1
                        pos, neg = self._rate_search_results(
                            result, self.search_tool.original_question, docids, planning_port, messages)
                        if (pos or neg) and self.search_tool:
                            self.search_tool.update_doc_labels(pos, neg)

                    if docids is not None:
                        tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                        retrieved_docids.extend(docids)
                except:
                    tool_call_counts_all['invalid_json'] = tool_call_counts_all.get('invalid_json', 0) + 1
                    result = 'Error: Tool call is not a valid JSON. Tool call must contain a valid "name" and "arguments" field.'
                result = "<tool_response>\n" + result + "\n</tool_response>"
                messages.append({"role": "user", "content": result})
            else:
                termination = 'answer'
                break

            max_tokens = 26000
            token_count = self.count_tokens(messages)

            if num_llm_calls_available <= 0 or token_count > max_tokens:
                messages[-1]['content'] = TRUNCATED_MESSAGE + FINAL_MESSAGE
                self.max_generation = 6000
                content, finish_reason = self.call_server(
                    messages, planning_port,
                    prefill="Based on all the information gathered so far, my final answer is: ")
                messages.append({"role": "assistant", "content": content.strip()})

                if finish_reason == "length":
                    self.enable_thinking = False
                    content, finish_reason = self.call_server(
                        messages, planning_port,
                        prefill="Based on all the information gathered so far, my final answer is: ")
                
                prediction = messages[-1]['content']
                if '<tool_call>' in content:
                    termination = 'format error: generate an answer as token limit reached'
                else:
                    termination = 'generate an answer as token limit reached'

                result = {
                    "question": question,
                    "answer": answer,
                    "messages": messages,
                    "prediction": prediction,
                    "termination": termination,
                    "tool_call_counts": tool_call_counts,
                    "tool_call_counts_all": tool_call_counts_all,
                    "retrieved_docids": list(set(retrieved_docids)),
                    "tool_traces": self.search_tool.get_search_traces() if self.search_tool else tool_traces,
                }
                return result

        if '<tool_call>' not in messages[-1]['content']:
            prediction = messages[-1]['content']
            termination = 'answer'
        else:
            prediction = 'No answer found.'
            termination = 'answer not found'
            if num_llm_calls_available == 0:
                termination = 'exceed available llm calls'
        result = {
            "question": question,
            "answer": answer,
            "messages": messages,
            "prediction": prediction,
            "termination": termination,
            "tool_call_counts": tool_call_counts,
            "tool_call_counts_all": tool_call_counts_all,
            "retrieved_docids": list(set(retrieved_docids)),
            "tool_traces": self.search_tool.get_search_traces() if self.search_tool else tool_traces,
        }
        return result

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs): 
        if tool_name == "search" and self.search_tool:
            return self.search_tool.call(tool_args, **kwargs)
        elif tool_name == "get_document" and self.get_document_tool:
            result, docids = self.get_document_tool.call(tool_args, **kwargs)
            if self.search_tool and docids:
                self.search_tool.add_found_docids(docids)
                for d in docids:
                    self.search_tool.add_visited_docid(d)
            return result, docids
        else:
            return f"Error: Tool {tool_name} not found", None
