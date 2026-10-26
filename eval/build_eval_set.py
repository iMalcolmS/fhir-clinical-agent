"""Build an evaluation set for the agent from the structured FHIR data.

Ground truth comes from the same records the tools read, so no answer is
written by hand. Each question also names the tool a correct agent must call,
so we can score tool choice separately from the final answer.

Usage:  python -m eval.build_eval_set --patients 5
Writes: eval/questions.json
"""

import argparse
import json
from pathlib import Path

from fhir_agent.fhir import load_fhir_bundle
from fhir_agent.views import (allergies, key_measures, medication_list, observation_series,
                              patient_header, problem_list)

OUT = Path("eval/questions.json")
UNANSWERABLE = ["What is {name}'s blood type?", "What is {name}'s NHS number?",
                "Who is {name}'s GP?"]
DOSE_WORDS = {"oral", "tablet", "tablets", "capsule", "capsules", "injector", "auto", "pack",
              "day", "extended", "release", "solution", "injection", "prefilled", "syringe",
              "topical", "cream", "inhaler", "inhalant", "metered", "dose", "actuat", "spray",
              "chewable", "suspension", "film", "patch", "transdermal", "nasal"}


def drug_keyword(medication):
    for raw in medication.replace("/", " ").replace("-", " ").split():
        word = raw.strip("(),[]").lower()
        if word.isalpha() and len(word) > 3 and word not in DOSE_WORDS:
            return word
    return medication.split()[0].lower()


def questions_for(bundle, index, all_conditions):
    header = patient_header(bundle)
    pid, name = header["id"], header["name"]
    out = []

    def add(category, question, check, tool):
        out.append({"id": f"{pid[:8]}-{len(out) + 1:02d}", "patient_id": pid, "patient_name": name,
                    "category": category, "question": question, "check": check,
                    "expected_tool": tool})

    active = [m for m in medication_list(bundle) if m["status"] == "active"]
    if active:
        add("medications", f"Which medications is {name} currently taking?",
            {"type": "keywords", "keywords": sorted({drug_keyword(m["medication"]) for m in active})},
            "get_medications")

    found = allergies(bundle)
    if found:
        words = sorted({a["substance"].split()[0].lower().strip("(),") for a in found})
        add("allergies", f"What allergies does {name} have?",
            {"type": "keywords", "keywords": words}, "get_allergies")
    else:
        add("allergies", f"Does {name} have any recorded allergies?", {"type": "negative"},
            "get_allergies")

    for m in key_measures(observation_series(bundle)):
        if m["label"] == "BMI":
            add("observations", f"What was {name}'s most recent BMI in kg/m2?",
                {"type": "numbers", "numbers": [round(m["value"], 1)], "tol": 0.15},
                "get_observations")
        if m["label"] == "Blood pressure" and "diastolic" in m:
            add("observations", f"What was {name}'s most recent blood pressure?",
                {"type": "numbers", "numbers": [m["value"], m["diastolic"]], "tol": 0.5},
                "get_observations")

    clinical = [p for p in problem_list(bundle) if not p["social"]]
    names = [p["problem"].lower() for p in clinical]
    unique = [p for p in clinical if p["status"] == "active" and names.count(p["problem"].lower()) == 1]
    if unique:
        p = unique[0]
        add("conditions", f"In which year was {name} first diagnosed with {p['problem'].lower()}?",
            {"type": "year", "year": p["onset"][:4]}, "get_conditions")
    absent = [c for c in all_conditions if c.lower() not in names]
    if absent:
        add("cross-patient", f"Has {name} ever been diagnosed with {absent[index % len(absent)].lower()}?",
            {"type": "negative"}, "get_conditions")

    add("imaging", f"Does {name} have any chest X-ray reports on record?", {"type": "negative"},
        "get_imaging_reports")
    add("unanswerable", UNANSWERABLE[index % len(UNANSWERABLE)].format(name=name),
        {"type": "refuse"}, None)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/synthea")
    parser.add_argument("--patients", type=int, default=5)
    args = parser.parse_args()

    bundles = [load_fhir_bundle(p) for p in sorted(Path(args.data_dir).glob("*.json"))]
    all_conditions = sorted({p["problem"] for b in bundles for p in problem_list(b)
                             if not p["social"] and p["status"] == "active"})
    questions = []
    for i, bundle in enumerate(bundles[: args.patients]):
        questions += questions_for(bundle, i, all_conditions)
    OUT.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    counts = {}
    for q in questions:
        counts[q["category"]] = counts.get(q["category"], 0) + 1
    print(f"Wrote {len(questions)} questions to {OUT}: {counts}")


if __name__ == "__main__":
    main()
