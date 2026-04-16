"""
Mock FHIR R4 Server — simulates FOCUS's environment
====================================================
Serves synthetic Patient demographics so the Patient Dashboard (Isa's app)
can be developed and tested without waiting on the real FOCUS FHIR server.

Implements a minimal FHIR R4 surface:
  GET /fhir/Patient              → Bundle of all patients
  GET /fhir/Patient/{id}         → single Patient resource
  GET /fhir/Observation?patient= → Bundle of vital-sign Observations for a patient

Patient IDs match those in .env (PATIENT_IDS) so the full stack links up:
  mock_app  →  inference_server  →  MQTT  →  fall_dashboard  →  Patient Dashboard
                                                                  ↑
                                                         this server (demographics)

Run:
  uvicorn focus_mock.fhir_server:app --host 0.0.0.0 --port 8003
  or via docker-compose (see infrastructure/docker-compose.yml — focus_mock service)

Namespace note:
  In production Kubernetes this server is replaced by the real FOCUS FHIR server
  living in their namespace. This mock is our-namespace-only and never ships.
"""

import os
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(
    title="Mock FHIR R4 Server (FOCUS simulation)",
    version="1.0.0",
    description="Synthetic patient demographics for local development. Not for production.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Seed data — one record per patient in .env PATIENT_IDS
# Extend this list when adding patients to the test setup.
# ---------------------------------------------------------------------------

_PATIENTS: dict[str, dict] = {
    "test_patient-001": {
        "resourceType": "Patient",
        "id": "test_patient-001",
        "identifier": [{"system": "urn:focus:patient", "value": "test_patient-001"}],
        "active": True,
        "name": [{"family": "Müller", "given": ["Hans"]}],
        "gender": "male",
        "birthDate": "1942-07-14",
        "address": [{"text": "Ward A, Room 3, Charité Berlin"}],
        "extension": [
            {
                "url": "http://hl7.org/fhir/StructureDefinition/patient-ward",
                "valueString": "Ward A",
            }
        ],
    },
    "test_patient-002": {
        "resourceType": "Patient",
        "id": "test_patient-002",
        "identifier": [{"system": "urn:focus:patient", "value": "test_patient-002"}],
        "active": True,
        "name": [{"family": "Schmidt", "given": ["Margarete"]}],
        "gender": "female",
        "birthDate": "1938-11-03",
        "address": [{"text": "Ward B, Room 7, Charité Berlin"}],
        "extension": [
            {
                "url": "http://hl7.org/fhir/StructureDefinition/patient-ward",
                "valueString": "Ward B",
            }
        ],
    },
}

# Vital-sign Observations per patient
# LOINC codes: 8302-2 = Body height, 29463-7 = Body weight, 8867-4 = Heart rate
_OBSERVATIONS: dict[str, list[dict]] = {
    "test_patient-001": [
        {
            "resourceType": "Observation",
            "id": "obs-001-height",
            "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": "8302-2", "display": "Body height"}]},
            "subject": {"reference": "Patient/test_patient-001"},
            "valueQuantity": {"value": 172, "unit": "cm", "system": "http://unitsofmeasure.org", "code": "cm"},
        },
        {
            "resourceType": "Observation",
            "id": "obs-001-weight",
            "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": "29463-7", "display": "Body weight"}]},
            "subject": {"reference": "Patient/test_patient-001"},
            "valueQuantity": {"value": 74, "unit": "kg", "system": "http://unitsofmeasure.org", "code": "kg"},
        },
        {
            "resourceType": "Observation",
            "id": "obs-001-hr",
            "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4", "display": "Heart rate"}]},
            "subject": {"reference": "Patient/test_patient-001"},
            "valueQuantity": {"value": 68, "unit": "beats/min", "system": "http://unitsofmeasure.org", "code": "/min"},
        },
    ],
    "test_patient-002": [
        {
            "resourceType": "Observation",
            "id": "obs-002-height",
            "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": "8302-2", "display": "Body height"}]},
            "subject": {"reference": "Patient/test_patient-002"},
            "valueQuantity": {"value": 158, "unit": "cm", "system": "http://unitsofmeasure.org", "code": "cm"},
        },
        {
            "resourceType": "Observation",
            "id": "obs-002-weight",
            "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": "29463-7", "display": "Body weight"}]},
            "subject": {"reference": "Patient/test_patient-002"},
            "valueQuantity": {"value": 61, "unit": "kg", "system": "http://unitsofmeasure.org", "code": "kg"},
        },
        {
            "resourceType": "Observation",
            "id": "obs-002-hr",
            "status": "final",
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4", "display": "Heart rate"}]},
            "subject": {"reference": "Patient/test_patient-002"},
            "valueQuantity": {"value": 72, "unit": "beats/min", "system": "http://unitsofmeasure.org", "code": "/min"},
        },
    ],
}

# ---------------------------------------------------------------------------
# FHIR bundle helper
# ---------------------------------------------------------------------------

def _bundle(resources: list[dict], total: int) -> dict:
    return {
        "resourceType": "Bundle",
        "type": "searchset",
        "total": total,
        "entry": [{"resource": r} for r in resources],
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "server": "mock-fhir", "patients": len(_PATIENTS)}


@app.get("/fhir/Patient")
def list_patients():
    """Return all patients as a FHIR Bundle."""
    patients = list(_PATIENTS.values())
    return _bundle(patients, len(patients))


@app.get("/fhir/Patient/{patient_id}")
def get_patient(patient_id: str):
    """Return a single FHIR R4 Patient resource."""
    patient = _PATIENTS.get(patient_id)
    if not patient:
        raise HTTPException(
            status_code=404,
            detail={
                "resourceType": "OperationOutcome",
                "issue": [{"severity": "error", "code": "not-found",
                            "diagnostics": f"Patient/{patient_id} not found"}],
            },
        )
    return patient


@app.get("/fhir/Observation")
def list_observations(patient: str = Query(..., description="Patient ID, e.g. test_patient-002")):
    """Return vital-sign Observations for a patient as a FHIR Bundle."""
    if patient not in _PATIENTS:
        raise HTTPException(
            status_code=404,
            detail={
                "resourceType": "OperationOutcome",
                "issue": [{"severity": "error", "code": "not-found",
                            "diagnostics": f"Patient/{patient} not found"}],
            },
        )
    obs = _OBSERVATIONS.get(patient, [])
    return _bundle(obs, len(obs))
