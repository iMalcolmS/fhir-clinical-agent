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

## Agent

`fhir_agent/agent.py` is a tool-calling agent that reaches the record only through the MCP tools. It works with any OpenAI-compatible chat API (Groq `openai/gpt-oss-120b` by default) and returns a trace of every tool call, token count and latency.

```bash
echo "GROQ_API_KEY=your_key" > .env        # git-ignored
python -m fhir_agent.agent "What medications is Abdul Harris taking?"
```

Rules in the system prompt: resolve the patient first and use only that id, answer only from tool results, and reply with a fixed refusal sentence when the record does not have the answer.

## Evaluation

`eval/build_eval_set.py` generates questions from the structured records (medications, allergies, latest BMI and blood pressure, diagnosis year, a condition the patient does *not* have, imaging reports, and information the record never holds). Each question names the tool a correct agent must call. `eval/run_eval.py` scores:

- **answer pass rate** (rule-based, reusing the medical-rag-assistant scoring),
- **expected tool called**,
- **patient isolation**: no tool call used another patient's id,
- **over-refusal**, tool calls, latency and tokens per question.

```bash
python -m eval.build_eval_set --patients 5
python -m eval.run_eval
```

## Data

`data/synthea/` holds 20 synthetic patients generated with [Synthea](https://github.com/synthetichealth/synthea), trimmed to the resource types used here (same set as [medical-rag-assistant](https://github.com/iMalcolmS/medical-rag-assistant)).

## Roadmap

- [x] MCP server with read tools, a write tool for imaging reports, audit log and tests
- [x] Agent (tool calling) with an evaluation of tool choice, answers and refusals
- [ ] Cost and latency tracing; end-to-end demo with the chest X-ray triage service
