"""Run the agent on the evaluation set and score answers, tool use and patient isolation.

Metrics:
  pass rate          rule-based answer check (eval/scoring.py), overall and by category
  tool choice        the expected tool was called (questions that need one)
  patient isolation  every patient_id the agent used belongs to the asked-about patient
  over-refusal       answerable questions the agent refused
  cost               LLM calls, tokens and latency per question

The run resumes: questions already in eval/results.jsonl are skipped, so a
rate-limit stop on the free Groq tier does not lose work.

Usage:  python -m eval.run_eval [--sleep 4] [--limit N]
Writes: eval/results.jsonl, eval/report.md
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import anyio
import httpx
from mcp import Client

from eval.scoring import is_refusal, score
from fhir_agent.agent import MODEL, run_agent
from fhir_agent.config import load_env
from fhir_agent.server import create_server
from fhir_agent.store import FhirStore

QUESTIONS = Path("eval/questions.json")
RESULTS = Path("eval/results.jsonl")
REPORT = Path("eval/report.md")


async def run(questions, data_dir, sleep, results_path):
    done = set()
    if results_path.exists():
        done = {json.loads(line)["id"] for line in results_path.read_text().splitlines() if line}
    todo = [q for q in questions if q["id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to run")

    async with Client(create_server(FhirStore(data_dir))) as client:
        for i, q in enumerate(todo, 1):
            for attempt in range(4):
                try:
                    trace = await run_agent(q["question"], client)
                    break
                except httpx.HTTPStatusError as error:
                    if error.response.status_code != 429 or attempt == 3:
                        raise
                    wait = 20 * (attempt + 1)
                    print(f"   rate limited, waiting {wait}s")
                    await anyio.sleep(wait)
            passed, value = score(q["check"], trace.answer)
            used = trace.patient_ids()
            row = {
                **{k: q[k] for k in ("id", "patient_id", "category", "question", "expected_tool")},
                "check": q["check"], "answer": trace.answer, "passed": passed, "score": value,
                "refused": is_refusal(trace.answer),
                "tools": [c["tool"] for c in trace.tool_calls],
                "tool_ok": q["expected_tool"] is None or q["expected_tool"] in
                           [c["tool"] for c in trace.tool_calls],
                "isolated": used <= {q["patient_id"]},
                "llm_calls": trace.llm_calls, "prompt_tokens": trace.prompt_tokens,
                "completion_tokens": trace.completion_tokens, "latency_s": trace.latency_s,
                "stopped": trace.stopped,
            }
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[{i}/{len(todo)}] {'PASS' if passed else 'FAIL'} {q['category']:<14} {q['question']}")
            await anyio.sleep(sleep)


def pct(values):
    values = list(values)
    return f"{100 * sum(values) / len(values):.0f}%" if values else "–"


def report(rows, report_path=REPORT):
    answerable = [r for r in rows if r["check"]["type"] not in ("refuse", "negative")]
    needs_tool = [r for r in rows if r["expected_tool"]]
    lines = [
        "# Agent evaluation", "",
        f"`{MODEL}` on {len(rows)} questions about {len({r['patient_id'] for r in rows})} "
        "synthetic patients. Answers scored by rules; tools and patient ids read from the trace.", "",
        "| Metric | Value |", "|---|---|",
        f"| **Answer pass rate** | **{pct(r['passed'] for r in rows)}** |",
        f"| Expected tool called | {pct(r['tool_ok'] for r in needs_tool)} |",
        f"| Patient isolation (no other patient's id used) | {pct(r['isolated'] for r in rows)} |",
        f"| Answerable questions wrongly refused | {pct(r['refused'] for r in answerable)} |",
        f"| Mean tool calls per question | {statistics.mean(len(r['tools']) for r in rows):.1f} |",
        f"| Median latency | {statistics.median(r['latency_s'] for r in rows):.1f} s |",
        f"| Mean tokens per question | {statistics.mean(r['prompt_tokens'] + r['completion_tokens'] for r in rows):,.0f} |",
        "", "| Category | Questions | Pass rate | Expected tool called |", "|---|---|---|---|",
    ]
    for cat in sorted({r["category"] for r in rows}):
        group = [r for r in rows if r["category"] == cat]
        tool_group = [r for r in group if r["expected_tool"]]
        lines.append(f"| {cat} | {len(group)} | {pct(r['passed'] for r in group)} | "
                     f"{pct(r['tool_ok'] for r in tool_group)} |")
    failures = [r for r in rows if not r["passed"]]
    if failures:
        lines += ["", "## Failures", "", "| Category | Question | Tools called | Answer (start) |",
                  "|---|---|---|---|"]
        for r in failures:
            answer = r["answer"].replace("\n", " ").replace("|", "/")[:120]
            lines.append(f"| {r['category']} | {r['question']} | {', '.join(r['tools']) or '–'} | {answer} |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines[:16])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/synthea")
    parser.add_argument("--sleep", type=float, default=4.0, help="seconds between questions")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--questions", default=str(QUESTIONS))
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument("--report", default=str(REPORT))
    args = parser.parse_args()
    load_env()
    os.environ.setdefault("FHIR_AUDIT_LOG", "runs/audit.jsonl")

    results_path, report_path = Path(args.results), Path(args.report)
    questions = json.loads(Path(args.questions).read_text())[: args.limit]
    anyio.run(run, questions, args.data_dir, args.sleep, results_path)
    ids = {q["id"] for q in questions}
    rows = [json.loads(l) for l in results_path.read_text().splitlines() if l]
    print("\n" + report([r for r in rows if r["id"] in ids], report_path))


if __name__ == "__main__":
    main()
