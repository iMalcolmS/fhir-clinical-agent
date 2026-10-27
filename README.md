# FHIR clinical agent

[![CI](https://github.com/iMalcolmS/fhir-clinical-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/iMalcolmS/fhir-clinical-agent/actions/workflows/ci.yml)

An **MCP server** that gives an AI assistant safe, auditable access to a patient's FHIR R4 record, and a tool-calling **agent** that answers clinicians' questions with it, evaluated on held-out patients. It is the second part of a clinical AI implementation series, after [chest-xray-triage](https://github.com/iMalcolmS/chest-xray-triage): the triage service writes FHIR `DiagnosticReport`s, and this server lets an assistant read them alongside the rest of the record.

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
- **Audit log.** Every call writes one JSON line (time, tool, patient id) to stderr, or to the file in `FHIR_AUDIT_LOG`.
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
python -m eval.build_eval_set --patients 5                     # dev set
python -m eval.run_eval

python -m eval.build_eval_set --patients 5 --skip 5 --out eval/questions_holdout.json   # unseen patients
python -m eval.run_eval --questions eval/questions_holdout.json \
    --results eval/results_holdout.jsonl --report eval/report_holdout.md
```

## Results

`openai/gpt-oss-120b` on Groq, temperature 0. 37 questions per set, generated from 5 synthetic patients each; the held-out set uses 5 patients that were never looked at while fixing the agent.

| Metric | Dev set, first run | Dev set, after fixes | **Held-out patients** |
|---|--:|--:|--:|
| Answer pass rate | 95% | 100% | **100%** |
| Expected tool called | 97% | 100% | **100%** |
| Patient isolation | 100% | 100% | **100%** |
| Answerable questions wrongly refused | 0% | 0% | **0%** |
| Mean tool calls per question | 2.1 | 2.0 | 2.0 |
| Median latency | 1.1 s | 1.2 s | 1.9 s |
| Mean tokens per question | 2,849 | 2,844 | 2,838 |

Reports: [first run](eval/report_v1.md), [after fixes](eval/report.md), [held-out](eval/report_holdout.md).

**What the first run caught, and the fixes:**

1. *"What was the most recent BMI?"* was answered with the **BMI percentile** (97.9%), not the BMI in kg/m². Both observations match "BMI". `get_observations` now returns every match closest-first with its unit, and the system prompt says to answer with the closest match and name its unit.
2. Asked for a value the record never holds (an NHS number), the agent said so in its own words instead of the fixed refusal sentence. The answer was correct, so the scorer now accepts close refusal variants; the prompt still asks for the fixed sentence.
3. After fix 1 the agent missed BMI for one patient whose record names it "Body Mass Index" only. Searches now expand common clinical abbreviations (BMI, BP, HbA1c, HR, eGFR…).

The fixes are general (search and prompt), not per-question, and the held-out run on unseen patients scored 100% on every category. The sets are small (37 questions each), so this shows the agent is reliable on these question types, not that it is error-free.

## End to end with the chest X-ray triage service

```
chest X-ray ──► chest-xray-triage API ──► FHIR DiagnosticReport ──► add_imaging_report ──► patient record
  (DICOM)        (no patient identity)      (preliminary, flags)        (MCP, audited)              │
                                                                                                    ▼
                         clinician question ──► agent ──► get_imaging_reports ──► answer + trace
```

The triage API never sees who the patient is. The report is linked to the patient on the record side, as a PACS/EHR integration would do, and the agent then reads it like any other part of the record.

```bash
# offline: a saved /predict response from the triage API
python -m fhir_agent.pipeline --patient "Abdul Harris" --response examples/triage_response.json

# live: run the triage service, then send it a DICOM
docker run -p 7860:7860 ghcr.io/imalcolms/chest-xray-triage:latest
python -m fhir_agent.pipeline --patient "Abdul Harris" --image chest.dcm --api http://localhost:7860
```

`--no-agent` stores the report without calling the LLM; `--reports-dir data/reports` keeps it across restarts. `examples/triage_response.json` is in the API's exact response format with illustrative scores on a PA film (Effusion and Cardiomegaly flagged; Pneumonia not reported on PA films, where the model is too weak).

## Tracing and cost

Every agent run appends a trace (question, answer, each tool call with arguments and latency, LLM calls, tokens) to `runs/traces.jsonl`; tool calls are also in the audit log. Summarise traces or evaluation results:

```bash
python -m fhir_agent.trace_report                              # runs/traces.jsonl
python -m fhir_agent.trace_report eval/results_holdout.jsonl   # an evaluation run
LLM_PRICE_IN=... LLM_PRICE_OUT=... python -m fhir_agent.trace_report eval/results_holdout.jsonl  # + cost, $ per 1M tokens
```

It reports median and p95 latency, tokens, tool use per tool, tool errors, runs that hit the step limit and, if prices are given, cost per 1,000 questions. Prices are not hard-coded because they change.

## Data

`data/synthea/` holds 20 synthetic patients generated with [Synthea](https://github.com/synthetichealth/synthea), trimmed to the resource types used here (same set as [medical-rag-assistant](https://github.com/iMalcolmS/medical-rag-assistant)).

## Roadmap

- [x] MCP server with read tools, a write tool for imaging reports, audit log and tests
- [x] Agent (tool calling) with an evaluation of tool choice, answers and refusals
- [x] Eval-driven fixes and a held-out evaluation on unseen patients
- [x] Tracing and cost summary; end-to-end pipeline with the chest X-ray triage service
- [ ] Swap the in-memory store for a FHIR server (HAPI FHIR) behind the same three store methods
