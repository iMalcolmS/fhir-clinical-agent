import json

import pytest
from mcp import Client

from fhir_agent.agent import REFUSAL, run_agent, to_function_specs
from fhir_agent.server import create_server
from fhir_agent.store import FhirStore

pytestmark = pytest.mark.anyio
MARIA = "d1e2f3a4-0000-1111-2222-333344445555"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def tool_call(i, name, args):
    return {"id": f"call_{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def scripted(*replies):
    """A fake LLM that returns the given messages in order and records what it saw."""
    seen = []

    async def chat(messages, tools):
        seen.append({"messages": list(messages), "tools": tools})
        message = replies[len(seen) - 1]
        return {"choices": [{"message": message}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10}}
    return chat, seen


async def test_tool_loop_and_trace(monkeypatch, tmp_path):
    monkeypatch.setenv("FHIR_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    chat, seen = scripted(
        {"content": None, "tool_calls": [tool_call(1, "find_patients", {"name": "Maria"})]},
        {"content": None, "tool_calls": [tool_call(2, "get_medications", {"patient_id": MARIA})]},
        {"content": "Atorvastatin 40 MG and Clopidogrel 75 MG."},
    )
    async with Client(create_server(FhirStore("tests/fixtures"))) as client:
        trace = await run_agent("What is Maria Lopes taking?", client, chat=chat)

    assert trace.answer.startswith("Atorvastatin")
    assert [c["tool"] for c in trace.tool_calls] == ["find_patients", "get_medications"]
    assert trace.patient_ids() == {MARIA}
    assert trace.llm_calls == 3 and trace.prompt_tokens == 300
    # the tool result was fed back to the model
    tool_messages = [m for m in seen[2]["messages"] if m["role"] == "tool"]
    assert "Atorvastatin" in tool_messages[-1]["content"]
    # every tool call was audited
    assert len((tmp_path / "audit.jsonl").read_text().splitlines()) == 2


async def test_tool_errors_are_returned_to_the_model(monkeypatch, tmp_path):
    monkeypatch.setenv("FHIR_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    chat, seen = scripted(
        {"content": None, "tool_calls": [tool_call(1, "get_conditions", {"patient_id": "wrong"})]},
        {"content": REFUSAL},
    )
    async with Client(create_server(FhirStore("tests/fixtures"))) as client:
        trace = await run_agent("Conditions?", client, chat=chat)
    assert trace.tool_calls[0]["is_error"] is True
    assert "find_patients" in seen[1]["messages"][-1]["content"]


async def test_stops_after_max_steps(monkeypatch, tmp_path):
    monkeypatch.setenv("FHIR_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    loop = {"content": None, "tool_calls": [tool_call(1, "find_patients", {"name": ""})]}
    chat, _ = scripted(*[loop] * 20)
    async with Client(create_server(FhirStore("tests/fixtures"))) as client:
        trace = await run_agent("Loop forever", client, chat=chat)
    assert trace.stopped == "max_steps" and trace.answer == REFUSAL


async def test_function_specs_match_mcp_tools():
    async with Client(create_server(FhirStore("tests/fixtures"))) as client:
        specs = to_function_specs((await client.list_tools()).tools)
    names = {s["function"]["name"] for s in specs}
    assert "get_medications" in names
    meds = next(s for s in specs if s["function"]["name"] == "get_medications")
    assert "patient_id" in meds["function"]["parameters"]["properties"]
