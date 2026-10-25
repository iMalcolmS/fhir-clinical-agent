"""Structured views of one FHIR bundle for the clinician dashboard.

The RAG pipeline answers free-text questions; this module gives the UI the
structured facts a clinician scans first: who the patient is, allergies,
active problems, current medication, vital-sign trends and the timeline.
All functions take a parsed bundle and return plain Python data, so they can
be unit-tested without Streamlit.
"""

from collections import defaultdict
from datetime import date

from fhir_agent.fhir import (
    _concept_text,
    _date_only,
    extract_resources,
    get_patient_name,
    is_social_finding,
    medication_name,
    reference_index,
)

# Vital signs and labs shown as headline tiles, in display order.
# Each entry: tile label -> list of observation names Synthea uses for it.
KEY_MEASURES = {
    "Blood pressure": ["Systolic Blood Pressure"],
    "BMI": ["Body mass index (BMI) [Ratio]", "Body Mass Index"],
    "Heart rate": ["Heart rate"],
    "HbA1c": ["Hemoglobin A1c/Hemoglobin.total in Blood"],
}


# --------------------------------------------------------------------------
# Patient banner
# --------------------------------------------------------------------------
def _age(birth, on):
    return on.year - birth.year - ((on.month, on.day) < (birth.month, birth.day))


def patient_header(bundle, today=None):
    patient = (extract_resources(bundle, "Patient") or [{}])[0]
    birth = patient.get("birthDate")
    deceased = _date_only(patient.get("deceasedDateTime")) if patient.get("deceasedDateTime") else None
    age = None
    if birth:
        end = date.fromisoformat(deceased) if deceased else (today or date.today())
        age = _age(date.fromisoformat(birth), end)
    encounters = sorted(_date_only(e.get("period", {}).get("start")) for e in extract_resources(bundle, "Encounter"))
    return {
        "id": patient.get("id", ""),
        "name": get_patient_name(patient),
        "gender": patient.get("gender", "unknown"),
        "birth_date": birth,
        "age": age,
        "deceased": deceased,
        "last_encounter": encounters[-1] if encounters else None,
        "encounter_count": len(encounters),
    }


def allergies(bundle):
    out = []
    for a in extract_resources(bundle, "AllergyIntolerance"):
        out.append({
            "substance": _concept_text(a.get("code")).replace(" (substance)", ""),
            "criticality": a.get("criticality", "unknown"),
            "category": ", ".join(a.get("category", [])) or "unknown",
        })
    return sorted(out, key=lambda a: (a["criticality"] != "high", a["substance"]))


# --------------------------------------------------------------------------
# Problems and medication
# --------------------------------------------------------------------------
def _clean_label(name):
    """'Essential hypertension (disorder)' -> 'Essential hypertension'."""
    for suffix in (" (disorder)", " (finding)", " (situation)"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def problem_list(bundle):
    problems = []
    for c in extract_resources(bundle, "Condition"):
        status = _concept_text(c.get("clinicalStatus")).lower()
        problems.append({
            "problem": _clean_label(_concept_text(c.get("code"))),
            "status": status,
            "onset": _date_only(c.get("onsetDateTime")),
            "resolved": _date_only(c.get("abatementDateTime")) if c.get("abatementDateTime") else None,
            "social": is_social_finding(c),
        })
    # Active first, then most recent onset first
    return sorted(problems, key=lambda p: (p["status"] != "active", _neg_date(p["onset"])))


def _neg_date(iso):
    """Sort key that puts the most recent ISO date first."""
    return "".join(chr(255 - ord(ch)) for ch in iso)


def medication_list(bundle):
    """One row per drug: first/last prescription, count and latest status."""
    references = reference_index(bundle)
    by_drug = defaultdict(list)
    for m in extract_resources(bundle, "MedicationRequest"):
        by_drug[medication_name(m, references)].append(
            (m.get("authoredOn", ""), m.get("status", "unknown"))
        )
    rows = []
    for drug, entries in by_drug.items():
        entries.sort()
        rows.append({
            "medication": drug,
            "status": entries[-1][1],
            "first": _date_only(entries[0][0]),
            "last": _date_only(entries[-1][0]),
            "prescriptions": len(entries),
        })
    return sorted(rows, key=lambda r: (r["status"] != "active", _neg_date(r["last"])))


# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------
def observation_series(bundle):
    """Numeric observations as {name: [{"date", "value", "unit"}]}, oldest first.

    Multi-part observations (blood pressure) are split into their components,
    e.g. "Systolic Blood Pressure" and "Diastolic Blood Pressure".
    """
    series = defaultdict(list)

    def add(name, quantity, when):
        if quantity and isinstance(quantity.get("value"), (int, float)):
            series[name].append({"date": when, "value": float(quantity["value"]), "unit": quantity.get("unit", "")})

    for obs in extract_resources(bundle, "Observation"):
        when = _date_only(obs.get("effectiveDateTime"))
        if "valueQuantity" in obs:
            add(_concept_text(obs.get("code")), obs["valueQuantity"], when)
        for comp in obs.get("component", []):
            add(_concept_text(comp.get("code")), comp.get("valueQuantity"), when)
    return {name: sorted(points, key=lambda p: p["date"]) for name, points in series.items()}


def key_measures(series):
    """Latest value, change from the previous reading and recent history for each headline tile."""
    tiles = []
    for label, names in KEY_MEASURES.items():
        name = next((n for n in names if n in series), None)
        if not name:
            continue
        points = series[name]
        latest = points[-1]
        tile = {
            "label": label,
            "value": latest["value"],
            "unit": latest["unit"],
            "date": latest["date"],
            "delta": latest["value"] - points[-2]["value"] if len(points) > 1 else None,
            "history": [p["value"] for p in points[-10:]],
        }
        if label == "Blood pressure" and "Diastolic Blood Pressure" in series:
            tile["diastolic"] = series["Diastolic Blood Pressure"][-1]["value"]
        tiles.append(tile)
    return tiles


# --------------------------------------------------------------------------
# Timeline
# --------------------------------------------------------------------------
def timeline_events(bundle):
    events = []
    for c in extract_resources(bundle, "Condition"):
        if is_social_finding(c):
            continue
        events.append({"date": _date_only(c.get("onsetDateTime")), "category": "Diagnosis",
                       "event": _clean_label(_concept_text(c.get("code")))})
    for med in medication_list(bundle):
        events.append({"date": med["first"], "category": "Medication", "event": med["medication"]})
    for e in extract_resources(bundle, "Encounter"):
        events.append({"date": _date_only(e.get("period", {}).get("start")), "category": "Encounter",
                       "event": _clean_label(_concept_text((e.get("type") or [{}])[0])).replace(" (procedure)", "")})
    return sorted((e for e in events if e["date"] != "Unknown"), key=lambda e: e["date"])
