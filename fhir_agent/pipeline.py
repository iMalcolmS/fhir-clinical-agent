"""End to end: chest X-ray triage API -> patient record -> clinical agent.

1. Get a FHIR DiagnosticReport from the chest-xray-triage service, either by
   posting an image to its /predict endpoint or from a saved response.
2. Attach it to the patient's record with the MCP tool add_imaging_report
   (the same call any MCP client would make; it is audited).
3. Ask the agent about it. The agent reads the report back through
   get_imaging_reports, like any other part of the record.

The triage API never sees who the patient is; the link between image and
patient is made here, on the record side, as a PACS/EHR integration would.

Usage:
    # offline, with the saved example response
    python -m fhir_agent.pipeline --patient "Abdul Harris" --response examples/triage_response.json

    # live, with the triage service running (docker run -p 7860:7860 ghcr.io/imalcolms/chest-xray-triage)
    python -m fhir_agent.pipeline --patient "Abdul Harris" --image chest.dcm --api http://localhost:7860

Add --no-agent to only store the report (no LLM call, no API key needed).
"""

import argparse
import json
import os
from pathlib import Path

import anyio
import httpx
from mcp import Client

from fhir_agent.agent import print_trace, run_agent, save_trace
from fhir_agent.config import load_env
from fhir_agent.server import create_server
from fhir_agent.store import FhirStore

DEFAULT_QUESTION = "What did the most recent chest X-ray AI triage report for {name} show?"


def triage_image(image_path, api_url, view=None):
    """POST an image to the triage API; returns its JSON response."""
    params = {"view": view} if view else {}
    with open(image_path, "rb") as f:
        response = httpx.post(f"{api_url.rstrip('/')}/predict", params=params,
                              files={"file": (Path(image_path).name, f)}, timeout=120)
    response.raise_for_status()
    return response.json()


def report_from_response(response):
    """Accept a full /predict response or a bare DiagnosticReport."""
    report = response.get("fhir", response)
    if report.get("resourceType") != "DiagnosticReport":
        raise ValueError("Expected a /predict response or a FHIR DiagnosticReport")
    return report


def payload(result):
    """The JSON value a tool returned (structured content if present, else its text)."""
    if result.structured_content is not None:
        return result.structured_content.get("result", result.structured_content)
    return json.loads("".join(getattr(c, "text", "") for c in result.content))


async def resolve_patient(client, name):
    matches = payload(await client.call_tool("find_patients", {"name": name}))
    if len(matches) != 1:
        found = ", ".join(m["name"] for m in matches) or "none"
        raise SystemExit(f"Need exactly one patient matching '{name}' (found: {found})")
    return matches[0]


async def store_report(client, patient_id, report):
    result = await client.call_tool("add_imaging_report",
                                    {"patient_id": patient_id, "report": report})
    if result.is_error:
        raise SystemExit("Could not store report: " +
                         " ".join(getattr(c, "text", "") for c in result.content))
    return payload(result)["report_id"]


async def run(args, report):
    store = FhirStore(args.data_dir, args.reports_dir)
    async with Client(create_server(store)) as client:
        patient = await resolve_patient(client, args.patient)
        report_id = await store_report(client, patient["patient_id"], report)
        print(f"Stored report {report_id} for {patient['name']}: {report.get('conclusion', '')}\n")
        if args.no_agent:
            return None
        question = args.question or DEFAULT_QUESTION.format(name=patient["name"])
        print(f"Q: {question}\n")
        trace = await run_agent(question, client)
        save_trace(trace)
        return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--patient", required=True, help="patient name (or part of it)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="DICOM/PNG/JPEG to send to the triage API")
    source.add_argument("--response", help="saved /predict response or DiagnosticReport JSON")
    parser.add_argument("--api", default="http://localhost:7860", help="triage API base URL")
    parser.add_argument("--view", choices=["AP", "PA"], help="override the DICOM view")
    parser.add_argument("--question", help="question for the agent (default: about the report)")
    parser.add_argument("--data-dir", default="data/synthea")
    parser.add_argument("--reports-dir", default=None,
                        help="also save the report here so it survives restarts (e.g. data/reports)")
    parser.add_argument("--no-agent", action="store_true", help="store the report only")
    args = parser.parse_args()
    load_env()
    os.environ.setdefault("FHIR_AUDIT_LOG", "runs/audit.jsonl")

    if args.image:
        response = triage_image(args.image, args.api, args.view)
    else:
        response = json.loads(Path(args.response).read_text())
    trace = anyio.run(run, args, report_from_response(response))
    if trace:
        print_trace(trace)


if __name__ == "__main__":
    main()
