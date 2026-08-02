import requests

FHIR_BASE = "http://localhost:8080/fhir"

def ensure_patient(patient_id):
    r = requests.get(f"{FHIR_BASE}/Patient/{patient_id}")
    if r.status_code == 404:
        patient = {
            "resourceType": "Patient",
            "id": patient_id
        }
        requests.post(
            f"{FHIR_BASE}/Patient",
            json=patient,
            headers={"Content-Type": "application/fhir+json"}
        )

ensure_patient("P1001")
