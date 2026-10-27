"""Summarise agent traces: latency, tokens, tool use, errors and (optional) cost.

Reads the JSON lines written by the agent (runs/traces.jsonl) or by the
evaluation (eval/results*.jsonl). Cost is shown only if prices are set, in
US dollars per million tokens, because provider prices change:

    LLM_PRICE_IN=... LLM_PRICE_OUT=... python -m fhir_agent.trace_report eval/results_holdout.jsonl
"""

import argparse
import json
import os
import statistics
from collections import Counter
from pathlib import Path


def load(paths):
    rows = []
    for path in paths:
        rows += [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return rows


def tool_names(row):
    """Agent traces store tool call dicts; eval results store tool names."""
    calls = row.get("tool_calls", row.get("tools", []))
    return [c["tool"] if isinstance(c, dict) else c for c in calls]


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


def summarise(rows, price_in=None, price_out=None):
    latency = [r["latency_s"] for r in rows]
    tokens_in = [r["prompt_tokens"] for r in rows]
    tokens_out = [r["completion_tokens"] for r in rows]
    tools = Counter(t for r in rows for t in tool_names(r))
    errors = sum(1 for r in rows for c in r.get("tool_calls", [])
                 if isinstance(c, dict) and c.get("is_error"))
    out = {
        "questions": len(rows),
        "llm_calls_mean": round(statistics.mean(r["llm_calls"] for r in rows), 2),
        "tool_calls_mean": round(statistics.mean(len(tool_names(r)) for r in rows), 2),
        "latency_median_s": round(statistics.median(latency), 2),
        "latency_p95_s": round(percentile(latency, 0.95), 2),
        "prompt_tokens_mean": round(statistics.mean(tokens_in)),
        "completion_tokens_mean": round(statistics.mean(tokens_out)),
        "stopped_at_max_steps": sum(r.get("stopped") == "max_steps" for r in rows),
        "tool_errors": errors,
        "tool_use": dict(tools.most_common()),
    }
    if price_in is not None and price_out is not None:
        cost = (sum(tokens_in) * price_in + sum(tokens_out) * price_out) / 1e6
        out["cost_total_usd"] = round(cost, 4)
        out["cost_per_1000_questions_usd"] = round(1000 * cost / len(rows), 2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", default=["runs/traces.jsonl"])
    args = parser.parse_args()
    price_in, price_out = os.environ.get("LLM_PRICE_IN"), os.environ.get("LLM_PRICE_OUT")
    summary = summarise(load(args.paths),
                        float(price_in) if price_in else None,
                        float(price_out) if price_out else None)
    for key, value in summary.items():
        print(f"{key:<28} {value}")


if __name__ == "__main__":
    main()
