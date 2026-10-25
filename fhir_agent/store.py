"""A small FHIR store over Synthea patient bundles.

Each Synthea bundle holds one patient and their records, so every resource in
a bundle belongs to that patient. Imaging reports written by the chest X-ray
triage service are added to the patient's bundle in memory and, optionally,
saved to disk so they survive a restart.

The store is deliberately simple. The tools in server.py only depend on
`patients`, `bundle` and `add_report`, so a real FHIR server (e.g. HAPI FHIR)
could be swapped in behind the same three methods.
"""

import json
import uuid
from pathlib import Path

from fhir_agent.fhir import extract_resources, get_patient_name, load_fhir_bundle


class UnknownPatient(KeyError):
    pass


class FhirStore:
    def __init__(self, data_dir, reports_dir=None):
        self._bundles = {}
        for path in sorted(Path(data_dir).glob("*.json")):
            bundle = load_fhir_bundle(path)
            patients = extract_resources(bundle, "Patient")
            if patients:
                self._bundles[patients[0]["id"]] = bundle
        self.reports_dir = Path(reports_dir) if reports_dir else None
        if self.reports_dir and self.reports_dir.exists():
            for path in sorted(self.reports_dir.glob("*.json")):
                report = json.loads(path.read_text())
                pid = report.get("subject", {}).get("reference", "").removeprefix("Patient/")
                if pid in self._bundles:
                    self._bundles[pid]["entry"].append({"resource": report})

    def patients(self):
        """[(id, name, birth date)] for every patient in the store."""
        out = []
        for pid, bundle in self._bundles.items():
            patient = extract_resources(bundle, "Patient")[0]
            out.append((pid, get_patient_name(patient), patient.get("birthDate")))
        return out

    def bundle(self, patient_id):
        try:
            return self._bundles[patient_id]
        except KeyError:
            raise UnknownPatient(patient_id) from None

    def add_report(self, patient_id, report):
        """Attach a DiagnosticReport to a patient; returns the stored resource id."""
        bundle = self.bundle(patient_id)
        if report.get("resourceType") != "DiagnosticReport":
            raise ValueError("Only DiagnosticReport resources can be added")
        report = dict(report)
        report["id"] = report.get("id") or str(uuid.uuid4())
        report["subject"] = {"reference": f"Patient/{patient_id}"}
        bundle["entry"].append({"resource": report})
        if self.reports_dir:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            (self.reports_dir / f"{report['id']}.json").write_text(json.dumps(report, indent=2))
        return report["id"]
