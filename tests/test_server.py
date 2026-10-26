import json

import pytest
from mcp import Client

from fhir_agent.server import create_server
from fhir_agent.store import FhirStore

pytestmark = pytest.mark.anyio
FIXTURES = "tests/fixtures"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def payload(result):
    """Tool output as plain Python, whether returned as structured or text content."""
    if result.structured_content is not None:
        data = result.structured_content
        return data.get("result", data) if isinstance(data, dict) else data
    return json.loads(result.content[0].text)


async def _patient_id(client, name="maria"):
    found = payload(await client.call_tool("find_patients", {"name": name}))
    assert len(found) == 1
    return found[0]["patient_id"]


async def test_lists_all_tools():
    async with Client(create_server(FhirStore(FIXTURES))) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert {"find_patients", "get_medications", "get_conditions", "get_allergies",
            "get_observations", "get_imaging_reports", "add_imaging_report"} <= names


async def test_find_and_read_one_patient():
    async with Client(create_server(FhirStore(FIXTURES))) as client:
        pid = await _patient_id(client)
        meds = payload(await client.call_tool("get_medications", {"patient_id": pid}))
        conditions = payload(await client.call_tool("get_conditions", {"patient_id": pid}))
    assert meds and all(m["status"] == "active" for m in meds)
    assert conditions and all(c["status"] == "active" for c in conditions)


async def test_observation_search_suggests_names_when_nothing_matches():
    async with Client(create_server(FhirStore(FIXTURES))) as client:
        pid = await _patient_id(client)
        result = payload(await client.call_tool(
            "get_observations", {"patient_id": pid, "name": "no such measurement"}))
    assert result["matches"] == [] and result["available"]


async def test_unknown_patient_is_a_clear_error():
    async with Client(create_server(FhirStore(FIXTURES))) as client:
        result = await client.call_tool("get_conditions", {"patient_id": "nope"})
    assert result.is_error and "find_patients" in result.content[0].text


async def test_imaging_report_round_trip(tmp_path):
    report = {"resourceType": "DiagnosticReport", "status": "preliminary",
              "code": {"text": "Chest X-ray AI triage"}, "issued": "2026-10-24T10:00:00+00:00",
              "conclusion": "Flagged for radiologist review: Effusion",
              "contained": [{"resourceType": "Observation", "id": "obs-0",
                             "code": {"text": "Effusion"}, "valueQuantity": {"value": 0.81},
                             "interpretation": [{"text": "flagged"}]}]}
    store = FhirStore(FIXTURES, reports_dir=tmp_path)
    async with Client(create_server(store)) as client:
        pid = await _patient_id(client)
        stored = payload(await client.call_tool(
            "add_imaging_report", {"patient_id": pid, "report": report}))
        reports = payload(await client.call_tool("get_imaging_reports", {"patient_id": pid}))
    assert stored["stored"] is True
    assert reports[0]["findings"][0] == {"finding": "Effusion", "score": 0.81,
                                         "interpretation": "flagged"}
    # saved to disk, and reloaded by a new store
    assert len(list(tmp_path.glob("*.json"))) == 1
    reloaded = FhirStore(FIXTURES, reports_dir=tmp_path)
    assert any(e["resource"]["resourceType"] == "DiagnosticReport"
               for e in reloaded.bundle(pid)["entry"])


async def test_rejects_non_report_resources():
    async with Client(create_server(FhirStore(FIXTURES))) as client:
        pid = await _patient_id(client)
        result = await client.call_tool(
            "add_imaging_report", {"patient_id": pid, "report": {"resourceType": "Patient"}})
    assert result.is_error


def test_search_terms_expand_abbreviations():
    from fhir_agent.server import search_terms
    assert "body mass index" in search_terms("BMI")
    assert "bmi" in search_terms("body mass index")
    assert search_terms("cholesterol") == {"cholesterol"}
