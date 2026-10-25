"""FHIR R4 helpers, shared with medical-rag-assistant (github.com/iMalcolmS/medical-rag-assistant)."""

import json
import re
from collections import defaultdict

UNKNOWN = "Unknown"


# --------------------------------------------------------------------------
# Low-level helpers
# --------------------------------------------------------------------------
def load_fhir_bundle(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_resources(bundle, resource_type):
    resources = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if resource.get("resourceType") == resource_type:
            resources.append(resource)
    return resources


def _concept_text(concept):
    """Human-readable label of a FHIR CodeableConcept."""
    if not concept:
        return UNKNOWN
    if concept.get("text"):
        return concept["text"]
    codings = concept.get("coding") or [{}]
    # Some concepts (e.g. clinicalStatus) only carry a code, no display
    return codings[0].get("display") or codings[0].get("code") or UNKNOWN


def _clean_name_part(part):
    """Synthea appends digits to names (e.g. 'Abdul218'); strip them for display."""
    return re.sub(r"\d+$", "", part)


def _date_only(value):
    """'2019-05-01T10:00:00+01:00' -> '2019-05-01'."""
    return value[:10] if value else UNKNOWN


# --------------------------------------------------------------------------
# Patient
# --------------------------------------------------------------------------
def get_patient_name(patient):
    names = patient.get("name") or [{}]
    official = next((n for n in names if n.get("use") == "official"), names[0])
    parts = [*official.get("given", []), official.get("family", "")]
    cleaned = [_clean_name_part(p) for p in parts if p]
    return " ".join(cleaned) or UNKNOWN


def patient_to_text(patient):
    return (
        f"Name: {get_patient_name(patient)} | Gender: {patient.get('gender', UNKNOWN)} | "
        f"Date of birth: {patient.get('birthDate', UNKNOWN)}"
    )


# --------------------------------------------------------------------------
# Clinical resources -> one line of text each
# --------------------------------------------------------------------------
def condition_to_text(condition):
    display = _concept_text(condition.get("code"))
    onset = _date_only(condition.get("onsetDateTime"))
    status = _concept_text(condition.get("clinicalStatus")).lower()
    line = f"{display} | onset: {onset} | status: {status}"
    if condition.get("abatementDateTime"):
        line += f" | resolved: {_date_only(condition['abatementDateTime'])}"
    return line


def _quantity_text(quantity):
    value = quantity.get("value")
    if value is None:
        return "?"
    if isinstance(value, float):
        value = round(value, 2)
    return f"{value} {quantity.get('unit', '')}".strip()


def observation_value_text(observation):
    """Handles single values, coded values and multi-part values (e.g. blood pressure)."""
    if "valueQuantity" in observation:
        return _quantity_text(observation["valueQuantity"])
    if "valueCodeableConcept" in observation:
        return _concept_text(observation["valueCodeableConcept"])
    if "valueString" in observation:
        return observation["valueString"]
    components = observation.get("component", [])
    if components:
        parts = []
        for comp in components:
            label = _concept_text(comp.get("code"))
            if "valueQuantity" in comp:
                parts.append(f"{label}: {_quantity_text(comp['valueQuantity'])}")
            elif "valueCodeableConcept" in comp:
                parts.append(f"{label}: {_concept_text(comp['valueCodeableConcept'])}")
        return "; ".join(parts) or "?"
    return "?"


def reference_index(bundle):
    """Map 'urn:uuid:...' and 'Type/id' references to the resources they point to."""
    index = {}
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        if entry.get("fullUrl"):
            index[entry["fullUrl"]] = resource
        if resource.get("resourceType") and resource.get("id"):
            index[f"{resource['resourceType']}/{resource['id']}"] = resource
    return index


def medication_name(med, references=None):
    """Drug name from medicationCodeableConcept, or from the Medication resource that
    medicationReference points to (newer Synthea versions use both)."""
    if med.get("medicationCodeableConcept"):
        return _concept_text(med["medicationCodeableConcept"])
    ref = (med.get("medicationReference") or {}).get("reference", "")
    if ref.startswith("#"):  # contained resource
        target = next((c for c in med.get("contained", []) if c.get("id") == ref[1:]), None)
    else:
        target = (references or {}).get(ref)
    if target:
        return _concept_text(target.get("code"))
    return (med.get("medicationReference") or {}).get("display", UNKNOWN)


def medication_to_text(med, references=None):
    display = medication_name(med, references)
    authored = _date_only(med.get("authoredOn"))
    return f"{display} | prescribed: {authored} | status: {med.get('status', 'unknown')}"


def summarise_medications(medications, references=None):
    """Group repeat prescriptions of the same drug into one line.

    Synthea creates a new MedicationRequest for every renewal, so the raw list
    repeats the same drug many times. One line per drug (first and last date,
    number of prescriptions, latest status) is shorter and easier to retrieve.
    """
    by_drug = defaultdict(list)
    for med in medications:
        name = medication_name(med, references)
        by_drug[name].append((med.get("authoredOn", ""), med.get("status", "unknown")))

    lines = []
    for name, entries in by_drug.items():
        entries.sort()
        first, last = _date_only(entries[0][0]), _date_only(entries[-1][0])
        period = first if first == last else f"{first} to {last}"
        count = len(entries)
        plural = "prescription" if count == 1 else "prescriptions"
        lines.append(f"{name} | {period} | {count} {plural} | latest status: {entries[-1][1]}")
    # Most recently prescribed first
    return sorted(lines, key=lambda line: line.split(" | ")[1][-10:], reverse=True)


def allergy_to_text(allergy):
    substance = _concept_text(allergy.get("code"))
    parts = [substance]
    if allergy.get("category"):
        parts.append(f"category: {', '.join(allergy['category'])}")
    if allergy.get("criticality"):
        parts.append(f"criticality: {allergy['criticality']}")
    reactions = [
        _concept_text(m)
        for r in allergy.get("reaction", [])
        for m in r.get("manifestation", [])
    ]
    if reactions:
        parts.append(f"reactions: {', '.join(reactions)}")
    recorded = allergy.get("recordedDate") or allergy.get("onsetDateTime")
    if recorded:
        parts.append(f"recorded: {_date_only(recorded)}")
    return " | ".join(parts)


# Synthea records social determinants of health (employment, education,
# housing, ...) as Condition resources. They are kept in their own document so
# that "Which diseases does the patient have?" is not answered with
# "full-time employment". Keyword list, not an official FHIR category.
SOCIAL_KEYWORDS = (
    "employment", "education", "labor force", "high school", "criminal record",
    "social isolation", "social contact", "housing", "homeless", "transport",
    "stress", "intimate partner", "violence", "refugee", "unemployed",
    "armed forces", "medication review due", "risk activity",
)


def is_social_finding(condition):
    name = _concept_text(condition.get("code")).lower()
    return any(keyword in name for keyword in SOCIAL_KEYWORDS)
