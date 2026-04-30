"""
FHIR Observation Builder for Fall Detection Results
====================================================
Converts a fall detection result dict into a FHIR R4 Observation resource.

FHIR resource type: Observation
  - category  : activity
  - code      : SNOMED 217082002 "Fall (event)"
  - subject   : Patient/<patient_id>
  - device    : Device/<device_id>
  - value     : valueBoolean (true = fall detected, false = no fall)
  - component : confidence score (probability 0.0–1.0)
  - component : model version string

SNOMED / LOINC codes used
  217082002   Fall (event)                     — SNOMED CT
  72514-3     Fall risk assessment score        — LOINC
  246075003   Causative agent                  — SNOMED (repurposed for algorithm)

Reference: HL7 FHIR R4  https://www.hl7.org/fhir/observation.html
"""

import uuid
from datetime import datetime, timezone
from typing import Optional


def build_fhir_observation(
    fall_detected: bool,
    confidence: float,
    model_version: str,
    patient_id: str,
    device_id: Optional[str],
    timestamp: Optional[str] = None,
    observation_id: Optional[str] = None,
    fhir_base_url: str = "",
) -> dict:
    """
    Build a FHIR R4 Observation resource representing one fall detection result.

    Parameters
    ----------
    fall_detected   : True if the model predicted a fall
    confidence      : XGBoost fall probability (0.0 – 1.0)
    model_version   : e.g. "v3"
    patient_id      : patient identifier — becomes Patient/<patient_id> reference
    device_id       : SmarKo device/wearable identifier (optional)
    timestamp       : ISO-8601 UTC string; defaults to now
    observation_id  : UUID for this resource; generated if not supplied
    fhir_base_url   : optional base URL prefix for references, e.g. "https://fhir.example.com"

    Returns
    -------
    dict  — FHIR R4 Observation (ready for json.dumps or POST to FHIR server)
    """
    obs_id  = observation_id or str(uuid.uuid4())
    ts      = timestamp or datetime.now(timezone.utc).isoformat()
    ref_base = fhir_base_url.rstrip("/") + "/" if fhir_base_url else ""

    observation = {
        "resourceType": "Observation",
        "id":           obs_id,
        "status":       "final",

        # Category: activity monitoring
        "category": [
            {
                "coding": [
                    {
                        "system":  "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code":    "activity",
                        "display": "Activity",
                    }
                ]
            }
        ],

        # What is being observed: fall event
        "code": {
            "coding": [
                {
                    "system":  "http://snomed.info/sct",
                    "code":    "217082002",
                    "display": "Fall (event)",
                }
            ],
            "text": "Fall detection",
        },

        # Who the observation is about
        "subject": {
            "reference": f"{ref_base}Patient/{patient_id}",
            "display":   patient_id,
        },

        # When the detection window was processed
        "effectiveDateTime": ts,

        # When this resource was created
        "issued": ts,

        # Primary result: was a fall detected?
        "valueBoolean": fall_detected,

        # Secondary components
        "component": [
            # 1. Confidence score (fall probability)
            {
                "code": {
                    "coding": [
                        {
                            "system":  "http://loinc.org",
                            "code":    "72514-3",
                            "display": "Fall risk assessment score",
                        }
                    ],
                    "text": "Fall confidence score",
                },
                "valueQuantity": {
                    "value":  round(confidence, 4),
                    "unit":   "probability",
                    "system": "http://unitsofmeasure.org",
                    "code":   "1",          # dimensionless
                },
            },
            # 2. Algorithm / model version
            {
                "code": {
                    "coding": [
                        {
                            "system":  "http://snomed.info/sct",
                            "code":    "246075003",
                            "display": "Causative agent",
                        }
                    ],
                    "text": "Detection algorithm version",
                },
                "valueString": f"SmarKo-FallDetection-{model_version}",
            },
        ],
    }

    # Device reference (optional — only if device_id is provided)
    if device_id:
        observation["device"] = {
            "reference": f"{ref_base}Device/{device_id}",
            "display":   device_id,
        }

    return observation
