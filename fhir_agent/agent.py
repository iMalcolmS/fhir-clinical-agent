"""A tool-calling clinical agent that reads the patient record through the MCP server.

The agent sees the MCP tools as OpenAI-style function definitions, so any
OpenAI-compatible chat API works (Groq by default). Each run returns the
answer and a trace: every tool call with its arguments, tokens and latency.

Usage:
    put GROQ_API_KEY=... in .env     (git-ignored; never paste the key anywhere else)
    python -m fhir_agent.agent "What medications is Abdul Harris taking?"
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import anyio
import httpx
from mcp import Client

from fhir_agent.config import load_env
from fhir_agent.server import create_server
from fhir_agent.store import FhirStore

API_URL = os.environ.get("LLM_API_URL", "https://api.groq.com/openai/v1/chat/completions")
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_STEPS = 8
MAX_TOOL_CHARS = 6000  # cap what one tool result adds to the context

REFUSAL = "I could not find this information in the patient's record."
SYSTEM_PROMPT = f"""You are a clinical record assistant for clinicians. You answer questions about ONE patient using only the tools provided.

Rules:
1. First call find_patients with the patient's name, then use that patient_id for every other call. Never use another patient's id.
2. Answer only from tool results. Do not use outside medical knowledge to fill gaps.
3. If the record does not contain the answer, reply exactly: "{REFUSAL}"
4. If asked whether the patient has a condition and the record shows they do not, answer "No" and say it is not in the record.
5. If a tool returns several matching measurements, answer with the closest match to the question and name it with its unit; mention the others only if they could be what was meant (e.g. BMI ratio vs BMI percentile).
6. Keep answers short and include dates and units when the tools give them."""


@dataclass
class Trace:
    question: str
    answer: str = ""
    tool_calls: list = field(default_factory=list)  # {"tool", "arguments", "is_error", "ms"}
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    stopped: str = "answer"  # "answer" or "max_steps"

    def to_dict(self):
        return {"question": self.question, "answer": self.answer, "tool_calls": self.tool_calls,
                "llm_calls": self.llm_calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "latency_s": self.latency_s,
                "stopped": self.stopped, "model": MODEL}

    def patient_ids(self):
        return {c["arguments"].get("patient_id") for c in self.tool_calls
                if "patient_id" in c["arguments"]}


def to_function_specs(tools):
    """MCP tool list -> OpenAI 'tools' parameter."""
    return [{"type": "function",
             "function": {"name": t.name, "description": t.description or "",
                          "parameters": t.input_schema}} for t in tools]


def tool_text(result):
    if result.structured_content is not None:
        text = json.dumps(result.structured_content.get("result", result.structured_content))
    else:
        text = "\n".join(getattr(c, "text", "") for c in result.content)
    return text[:MAX_TOOL_CHARS]


async def groq_chat(messages, tools):
    """One chat completion call; returns the parsed JSON response."""
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("Set GROQ_API_KEY (see README)")
    payload = {"model": MODEL, "messages": messages, "tools": tools,
               "tool_choice": "auto", "temperature": 0}
    async with httpx.AsyncClient(timeout=60) as http:
        response = await http.post(API_URL, json=payload,
                                   headers={"Authorization": f"Bearer {key}"})
        response.raise_for_status()
        return response.json()


async def run_agent(question, mcp_client, chat=None):
    """Answer one question. `chat` is injectable so tests can script the LLM."""
    chat = chat or groq_chat
    trace = Trace(question)
    start = time.perf_counter()
    tools = to_function_specs((await mcp_client.list_tools()).tools)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question}]

    for _ in range(MAX_STEPS):
        reply = await chat(messages, tools)
        trace.llm_calls += 1
        usage = reply.get("usage", {})
        trace.prompt_tokens += usage.get("prompt_tokens", 0)
        trace.completion_tokens += usage.get("completion_tokens", 0)
        message = reply["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if not calls:
            trace.answer = (message.get("content") or "").strip()
            break
        messages.append({"role": "assistant", "content": message.get("content"),
                         "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            try:
                arguments = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            t0 = time.perf_counter()
            result = await mcp_client.call_tool(name, arguments)
            trace.tool_calls.append({"tool": name, "arguments": arguments,
                                     "is_error": bool(result.is_error),
                                     "ms": round(1000 * (time.perf_counter() - t0))})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": tool_text(result)})
    else:
        trace.stopped = "max_steps"
        trace.answer = REFUSAL

    trace.latency_s = round(time.perf_counter() - start, 2)
    return trace


def save_trace(trace, path=None):
    """Append one trace as a JSON line (default runs/traces.jsonl, git-ignored)."""
    path = path or os.environ.get("FHIR_TRACE_LOG", "runs/traces.jsonl")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **trace.to_dict()}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


async def ask(question, data_dir="data/synthea", reports_dir=None):
    os.environ.setdefault("FHIR_AUDIT_LOG", "runs/audit.jsonl")
    async with Client(create_server(FhirStore(data_dir, reports_dir))) as client:
        trace = await run_agent(question, client)
    save_trace(trace)
    return trace


def print_trace(trace):
    print(trace.answer, "\n")
    for c in trace.tool_calls:
        print(f"  tool {c['tool']}({json.dumps(c['arguments'])}) {c['ms']} ms"
              f"{' ERROR' if c['is_error'] else ''}")
    print(f"\n  {trace.llm_calls} LLM calls, {trace.prompt_tokens}+{trace.completion_tokens} tokens, "
          f"{trace.latency_s}s")


def main():
    load_env()
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m fhir_agent.agent "your question"')
    reports_dir = os.environ.get("FHIR_REPORTS_DIR", "data/reports")
    print_trace(anyio.run(ask, " ".join(sys.argv[1:]), "data/synthea", reports_dir))


if __name__ == "__main__":
    main()
