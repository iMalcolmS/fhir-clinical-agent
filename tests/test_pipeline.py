import json
from pathlib import Path

import pytest
from mcp import Client

from fhir_agent.agent import run_agent
from fhir_agent.pipeline import payload, report_from_response, resolve_patient, store_report
from fhir_agent.server import create_server
from fhir_agent.store import FhirStore
from fhir_agent.trace_report import summarise

pytestmark = pytest.mark.anyio
EXAMPLE = json.loads(Path("examples/triage_response.json").read_text())


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_report_from_response_accepts_both_shapes():
    assert report_from_response(EXAMPLE) is EXAMPLE["fhir"]
    assert report_from_response(EXAMPLE["fhir"]) is EXAMPLE["fhir"]
    with pytest.raises(ValueError):
        report_from_response({"resourceType": "Patient"})


async def test_triage_report_round_trip(monkeypatch, tmp_path):
    """A report from the triage API is stored, persisted and read back by the agent's tool."""
    monkeypatch.setenv("FHIR_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    store = FhirStore("tests/fixtures", tmp_path / "reports")
    async with Client(create_server(store)) as client:
        patient = await resolve_patient(client, "Maria")
        report_id = await store_report(client, patient["patient_id"], EXAMPLE["fhir"])
        result = await client.call_tool("get_imaging_reports", {"patient_id": patient["patient_id"]})

    reports = payload(result)
    triage = next(r for r in reports if r["id"] == report_id)
    assert triage["title"] == "Chest X-ray AI triage"
    assert "Effusion" in triage["conclusion"] and "Not a medical device" in triage["conclusion"]
    flagged = {f["finding"] for f in triage["findings"] if f["interpretation"] == "flagged"}
    assert flagged == {"Effusion", "Cardiomegaly"}
    # persisted: a fresh store sees the report too
    reloaded = FhirStore("tests/fixtures", tmp_path / "reports")
    ids = [e["resource"].get("id") for e in reloaded.bundle(patient["patient_id"])["entry"]]
    assert report_id in ids


async def test_agent_answers_from_stored_report(monkeypatch, tmp_path):
    monkeypatch.setenv("FHIR_AUDIT_LOG", str(tmp_path / "audit.jsonl"))
    store = FhirStore("tests/fixtures")
    replies = iter([
        {"content": None, "tool_calls": [{"id": "1", "type": "function", "function": {
            "name": "find_patients", "arguments": json.dumps({"name": "Maria"})}}]},
        None,  # filled in below once the patient id is known
        {"content": "Flagged for review: Effusion, Cardiomegaly (preliminary AI triage)."},
    ])
    seen = []

    async def chat(messages, tools):
        seen.append(messages[-1])
        reply = next(replies)
        if reply is None:
            pid = json.loads(messages[-1]["content"])[0]["patient_id"]
            reply = {"content": None, "tool_calls": [{"id": "2", "type": "function", "function": {
                "name": "get_imaging_reports", "arguments": json.dumps({"patient_id": pid})}}]}
        return {"choices": [{"message": reply}], "usage": {"prompt_tokens": 50, "completion_tokens": 5}}

    async with Client(create_server(store)) as client:
        patient = await resolve_patient(client, "Maria")
        await store_report(client, patient["patient_id"], EXAMPLE["fhir"])
        trace = await run_agent("What did Maria's chest X-ray triage show?", client, chat=chat)

    assert "Chest X-ray AI triage" in seen[-1]["content"]  # the report reached the model
    assert trace.patient_ids() == {patient["patient_id"]}
    assert "Effusion" in trace.answer


def test_trace_summary_handles_agent_and_eval_rows():
    agent_row = {"latency_s": 1.0, "prompt_tokens": 1000, "completion_tokens": 100, "llm_calls": 3,
                 "stopped": "answer",
                 "tool_calls": [{"tool": "find_patients", "is_error": False},
                                {"tool": "get_medications", "is_error": True}]}
    eval_row = {"latency_s": 3.0, "prompt_tokens": 3000, "completion_tokens": 300, "llm_calls": 3,
                "stopped": "answer", "tools": ["find_patients"]}
    s = summarise([agent_row, eval_row], price_in=1.0, price_out=2.0)
    assert s["questions"] == 2 and s["tool_errors"] == 1
    assert s["tool_use"] == {"find_patients": 2, "get_medications": 1}
    assert s["latency_median_s"] == 2.0
    assert s["cost_total_usd"] == round((4000 * 1.0 + 400 * 2.0) / 1e6, 4)
    assert "cost_total_usd" not in summarise([agent_row])
