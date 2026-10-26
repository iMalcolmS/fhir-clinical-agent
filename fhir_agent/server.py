"""MCP server exposing a patient's FHIR record as tools.

Any MCP client can use it: Claude Code, Claude Desktop, or the agent in this
repo. Every tool works on one patient at a time and needs that patient's id,
so an answer can never mix two patients' records. Every call is written to an
audit log (stderr, one JSON line), as a clinical system would require.

Run:   FHIR_DATA_DIR=data/synthea python -m fhir_agent.server
Add to Claude Code:
       claude mcp add fhir -e FHIR_DATA_DIR=$PWD/data/synthea -- $PWD/.venv/bin/python -m fhir_agent.server
"""

import json
import os
import sys
from datetime import datetime, timezone

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from fhir_agent.fhir import _concept_text, _date_only, extract_resources
from fhir_agent.store import FhirStore, UnknownPatient
from fhir_agent.views import (
    allergies,
    medication_list,
    observation_series,
    patient_header,
    problem_list,
)

INSTRUCTIONS = (
    "Tools over synthetic FHIR R4 patient records. Always resolve the patient first with "
    "find_patients, then pass that patient_id to every other tool. Answer only from tool "
    "results; if a tool returns nothing relevant, say the record does not contain it."
)


def _audit(tool, patient_id=None, **detail):
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "event": "tool_call", "tool": tool, "patient_id": patient_id, **detail}
    path = os.environ.get("FHIR_AUDIT_LOG")  # a file path, or stderr if unset
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    else:
        print(json.dumps(entry), file=sys.stderr, flush=True)  # stdout is the MCP channel


def create_server(store: FhirStore) -> MCPServer:
    server = MCPServer("fhir-clinical-tools", instructions=INSTRUCTIONS)

    def bundle_for(tool, patient_id):
        _audit(tool, patient_id)
        try:
            return store.bundle(patient_id)
        except UnknownPatient:
            raise ToolError(f"No patient with id '{patient_id}'. Use find_patients first.") from None

    @server.tool()
    def find_patients(name: str = "") -> list[dict]:
        """Find patients by (part of) their name. Empty name lists everyone."""
        _audit("find_patients", query=name)
        name = name.lower().strip()
        return [{"patient_id": pid, "name": full, "birth_date": birth}
                for pid, full, birth in store.patients() if name in full.lower()]

    @server.tool()
    def get_patient_summary(patient_id: str) -> dict:
        """Demographics, age, number of encounters and the date of the last one."""
        return patient_header(bundle_for("get_patient_summary", patient_id))

    @server.tool()
    def get_conditions(patient_id: str, active_only: bool = True) -> list[dict]:
        """Diagnoses with onset (and resolution) dates. Social findings are excluded."""
        rows = [p for p in problem_list(bundle_for("get_conditions", patient_id)) if not p["social"]]
        if active_only:
            rows = [p for p in rows if p["status"] == "active"]
        return [{k: v for k, v in p.items() if k != "social"} for p in rows]

    @server.tool()
    def get_medications(patient_id: str, active_only: bool = True) -> list[dict]:
        """Medications with first/last prescription date, count and latest status."""
        rows = medication_list(bundle_for("get_medications", patient_id))
        return [r for r in rows if r["status"] == "active"] if active_only else rows

    @server.tool()
    def get_allergies(patient_id: str) -> list[dict]:
        """Recorded allergies and intolerances. An empty list means none recorded."""
        return allergies(bundle_for("get_allergies", patient_id))

    @server.tool()
    def get_observations(patient_id: str, name: str, limit: int = 5) -> dict:
        """Latest readings of a measurement (e.g. 'blood pressure', 'BMI', 'HbA1c').

        Matches on part of the name. Several measurements can match (e.g. 'BMI'
        matches both the BMI ratio in kg/m2 and the BMI percentile used for
        children): they are listed closest match first, each with its unit.
        If nothing matches, returns the measurement names the record contains.
        """
        series = observation_series(bundle_for("get_observations", patient_id))
        names = [k for k in series if name.lower() in k.lower()]
        if not names:
            return {"matches": [], "available": sorted(series)}
        names.sort(key=len)  # shorter names carry fewer qualifiers ("[Percentile] Per age and sex")
        return {"matches": [{"measurement": k, "unit": series[k][-1]["unit"],
                             "latest_first": series[k][-limit:][::-1]} for k in names]}

    @server.tool()
    def get_imaging_reports(patient_id: str) -> list[dict]:
        """Radiology reports, including AI triage reports from the chest X-ray service."""
        reports = extract_resources(bundle_for("get_imaging_reports", patient_id), "DiagnosticReport")
        out = []
        for r in reports:
            out.append({
                "id": r.get("id"),
                "issued": _date_only(r.get("issued") or r.get("effectiveDateTime")),
                "title": _concept_text(r.get("code")),
                "status": r.get("status"),
                "conclusion": r.get("conclusion", ""),
                "findings": [{"finding": _concept_text(o.get("code")),
                              "score": o.get("valueQuantity", {}).get("value"),
                              "interpretation": _concept_text((o.get("interpretation") or [{}])[0])}
                             for o in r.get("contained", []) if o.get("resourceType") == "Observation"],
            })
        return sorted(out, key=lambda r: r["issued"], reverse=True)

    @server.tool()
    def add_imaging_report(patient_id: str, report: dict) -> dict:
        """Attach a FHIR DiagnosticReport (e.g. from the chest X-ray triage API) to a patient."""
        bundle_for("add_imaging_report", patient_id)
        try:
            report_id = store.add_report(patient_id, report)
        except ValueError as error:
            raise ToolError(str(error)) from None
        return {"stored": True, "report_id": report_id}

    return server


def main():
    data_dir = os.environ.get("FHIR_DATA_DIR", "data/synthea")
    reports_dir = os.environ.get("FHIR_REPORTS_DIR", "data/reports")
    store = FhirStore(data_dir, reports_dir)
    print(f"Loaded {len(store.patients())} patients from {data_dir}", file=sys.stderr)
    create_server(store).run("stdio")


if __name__ == "__main__":
    main()
