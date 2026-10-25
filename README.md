# FHIR clinical agent

[![CI](https://github.com/iMalcolmS/fhir-clinical-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/iMalcolmS/fhir-clinical-agent/actions/workflows/ci.yml)

An **MCP server** that gives an AI assistant safe, auditable access to a patient's FHIR R4 record, and (next) an agent that answers clinical questions with it. It is the second part of a clinical AI implementation series, after [chest-xray-triage](https://github.com/iMalcolmS/chest-xray-triage): the triage service writes FHIR `DiagnosticReport`s, and this server lets an assistant read them alongside the rest of the record.

> Synthetic data only (Synthea). Research prototype, not for clinical use.

## Tools

| Tool | Returns |
|---|---|
| `find_patients(name)` | Matching patients and their ids |
| `get_patient_summary(patient_id)` | Demographics, age, encounters |
| `get_conditions(patient_id, active_only)` | Diagnoses with onset dates (social findings excluded) |
| `get_medications(patient_id, active_only)` | Medications with first/last prescription and status |
| `get_allergies(patient_id)` | Allergies; an empty list means none recorded |
| `get_observations(patient_id, name, limit)` | Latest readings of a measurement; suggests names if none match |
| `get_imaging_reports(patient_id)` | Radiology reports, including AI triage reports |
| `add_imaging_report(patient_id, report)` | Stores a DiagnosticReport from the triage API |

Design choices:

- **One patient per call.** Every record tool needs a `patient_id`, so an answer cannot mix two patients.
- **Audit log.** Every call writes one JSON line (time, tool, patient id) to stderr.
- **Clear errors.** An unknown id returns "No patient with id … Use find_patients first", which lets an agent recover.

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
FHIR_DATA_DIR=data/synthea python -m fhir_agent.server   # stdio MCP server
```

Use it from Claude Code:

```bash
claude mcp add fhir -e FHIR_DATA_DIR=$PWD/data/synthea -e PYTHONPATH=$PWD -- $PWD/.venv/bin/python -m fhir_agent.server
```

## Data

`data/synthea/` holds 20 synthetic patients generated with [Synthea](https://github.com/synthetichealth/synthea), trimmed to the resource types used here (same set as [medical-rag-assistant](https://github.com/iMalcolmS/medical-rag-assistant)).

## Roadmap

- [x] MCP server with read tools, a write tool for imaging reports, audit log and tests
- [ ] Agent (tool calling) with an evaluation of tool choice, answers and refusals
- [ ] Cost and latency tracing; end-to-end demo with the chest X-ray triage service
