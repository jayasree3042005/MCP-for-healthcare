import os
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename
import requests
import pydicom
from dicomweb_client.api import DICOMwebClient
from requests.auth import HTTPBasicAuth
from requests import Session
import json
import uuid
from datetime import datetime
import time
import logging
from io import BytesIO
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Any

# Import healthcare utilities
from healthcare_utils import structure_patient_data, get_dicom_images_for_patient
from clinical_decision_support import generate_clinical_interpretation, format_clinical_report
from clinical_response_formatter import format_clinical_response, ClinicalResponseFormatter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
class Config:
    ORTHANC_URL = "http://localhost:8042"
    ORTHANC_AUTH = ("orthanc", "orthanc")
    FHIR_URL = "http://localhost:8080/fhir"
    FHIR_HDR = {"Content-Type": "application/fhir+json"}

    UPLOAD_FOLDER = "uploads"
    ALLOWED_DCM = {"dcm", "DCM"}
    ALLOWED_PATIENT_DATA = {"json", "hl7", "txt"}

# ─── APP SETUP ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = "replace-this-with-a-secure-random-key"

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ─── DICOMWEB CLIENT ─────────────────────────────────────────────────────────
session = Session()
session.auth = HTTPBasicAuth(*app.config["ORTHANC_AUTH"])
dicom_client = DICOMwebClient(
    url=f"{app.config['ORTHANC_URL']}/dicom-web",
    session=session
)

# Global variable to track conversion status
conversion_status = {}
CONVERSION_HISTORY_FILE = "uploads/conversion_history.json"

def load_conversion_history():
    """Load conversion history from persistent storage"""
    global conversion_status
    try:
        if os.path.exists(CONVERSION_HISTORY_FILE):
            with open(CONVERSION_HISTORY_FILE, 'r') as f:
                conversion_status = json.load(f)
            logger.info(f"Loaded {len(conversion_status)} conversions from history")
        else:
            conversion_status = {}
    except Exception as e:
        logger.error(f"Error loading conversion history: {e}")
        conversion_status = {}

def save_conversion_history():
    """Save conversion history to persistent storage"""
    try:
        os.makedirs(os.path.dirname(CONVERSION_HISTORY_FILE), exist_ok=True)
        with open(CONVERSION_HISTORY_FILE, 'w') as f:
            json.dump(conversion_status, f, indent=2)
        logger.info("Conversion history saved")
    except Exception as e:
        logger.error(f"Error saving conversion history: {e}")

# ─── IMPROVED FHIR FUNCTIONS ────────────────────────────────────────────────
def validate_fhir_resource(resource):
    """Validate FHIR resource structure"""
    required_fields = {
        'Patient': ['resourceType', 'id'],
        'Observation': ['resourceType', 'status', 'code', 'subject'],
        'DiagnosticReport': ['resourceType', 'status', 'code', 'subject']
    }
    
    resource_type = resource.get('resourceType')
    if not resource_type:
        return False, "Missing resourceType"
    
    if resource_type in required_fields:
        for field in required_fields[resource_type]:
            if field not in resource:
                return False, f"Missing required field: {field}"
    
    return True, "Valid"

def upload_fhir_resource_with_retry(resource, resource_type, max_retries=3):
    """Upload a single FHIR resource with enhanced debugging and proper HTTP methods"""
    
    # Validate resource first
    is_valid, validation_msg = validate_fhir_resource(resource)
    if not is_valid:
        logger.error(f"Invalid {resource_type} resource: {validation_msg}")
        return False, f"Validation failed: {validation_msg}"
    
    logger.info(f"Validated {resource_type} resource successfully")
    
    # Log the exact JSON being sent
    resource_json = json.dumps(resource, indent=2)
    logger.info(f"Uploading {resource_type} JSON:\n{resource_json}")
    
    for attempt in range(max_retries):
        try:
            # If the resource includes a client-specified id, prefer PUT
            resource_id = resource.get('id')
            if resource_id:
                url = f"{Config.FHIR_URL}/{resource_type}/{resource_id}"
                method = "PUT"
                logger.info(f"Client provided id present, using PUT to create/update {resource_type}/{resource_id}")
            else:
                # Check if resource exists first by searching a potential server-provided id (fallback)
                url = f"{Config.FHIR_URL}/{resource_type}"
                method = "POST"
                logger.info(f"No client id provided, using POST to create {resource_type}")
            
            logger.info(f"Attempting {method} to {url} (attempt {attempt + 1})")
            
            # Make the request
            if method == "PUT":
                response = requests.put(url, headers=Config.FHIR_HDR, json=resource, timeout=30)
            else:
                response = requests.post(url, headers=Config.FHIR_HDR, json=resource, timeout=30)
            
            # Log response details
            logger.info(f"FHIR {resource_type} {method} attempt {attempt + 1}: HTTP {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            logger.info(f"Response body: {response.text}")
            
            if response.status_code in (200, 201):
                # Try to obtain the authoritative id assigned by the server
                server_assigned_id = None
                try:
                    resp_json = response.json()
                    server_assigned_id = resp_json.get('id')
                except Exception:
                    resp_json = None

                # If server didn't return JSON id, check Location header
                if not server_assigned_id:
                    loc = response.headers.get('Location') or response.headers.get('location')
                    if loc:
                        server_assigned_id = loc.rstrip('/').split('/')[-1]

                # If we still don't have a server id and the client provided one, use it
                if not server_assigned_id and resource_id:
                    server_assigned_id = resource_id

                # Update the local resource id to the server-assigned id (in-place)
                if server_assigned_id:
                    resource['id'] = server_assigned_id
                    logger.info(f"Successfully uploaded {resource_type}; server id: {server_assigned_id}")
                else:
                    logger.info(f"Successfully uploaded {resource_type}; no id returned by server")

                # Verify the resource was created by querying it back using the authoritative id
                verify_id = resource.get('id')
                if verify_id:
                    verify_url = f"{Config.FHIR_URL}/{resource_type}/{verify_id}"
                    verify_response = requests.get(verify_url, headers=Config.FHIR_HDR, timeout=30)
                    if verify_response.status_code == 200:
                        logger.info(f"Verified {resource_type}/{verify_id} exists in FHIR server")
                    else:
                        logger.warning(f"Could not verify {resource_type}/{verify_id} after upload (HTTP {verify_response.status_code})")

                # For DiagnosticReport, also query by subject to confirm it's searchable
                if resource_type == "DiagnosticReport":
                    patient_ref = resource.get('subject', {}).get('reference', '')
                    if patient_ref:
                        search_url = f"{Config.FHIR_URL}/{resource_type}?subject={patient_ref}"
                        search_response = requests.get(search_url, headers=Config.FHIR_HDR, timeout=30)
                        logger.info(f"Search by subject response: HTTP {search_response.status_code}")
                        if search_response.status_code == 200:
                            try:
                                search_data = search_response.json()
                                total = search_data.get('total', 0)
                                logger.info(f"Found {total} DiagnosticReports for {patient_ref}")
                            except Exception:
                                logger.debug("Could not parse search response JSON")
                else:
                    logger.warning(f"Could not verify {resource_type}/{resource_id} after upload")
                
                return True, "Success"
            elif response.status_code == 400:
                # Bad request - parse the error for more details
                try:
                    error_response = response.json()
                    if 'issue' in error_response:
                        issues = error_response['issue']
                        error_details = []
                        for issue in issues:
                            diagnostics = issue.get('diagnostics', '')
                            severity = issue.get('severity', 'error')
                            error_details.append(f"{severity}: {diagnostics}")
                        error_detail = '; '.join(error_details)
                    else:
                        error_detail = response.text
                except:
                    error_detail = response.text if response.text else "Bad Request"
                
                logger.error(f"Bad request for {resource_type}: {error_detail}")
                return False, f"Bad request: {error_detail}"
            elif response.status_code == 422:
                # Unprocessable entity - validation error
                try:
                    error_response = response.json()
                    if 'issue' in error_response:
                        issues = error_response['issue']
                        error_details = []
                        for issue in issues:
                            diagnostics = issue.get('diagnostics', '')
                            error_details.append(diagnostics)
                        error_detail = '; '.join(error_details)
                    else:
                        error_detail = response.text
                except:
                    error_detail = response.text if response.text else "Validation Error"
                
                logger.error(f"Validation error for {resource_type}: {error_detail}")
                return False, f"Validation error: {error_detail}"
            else:
                # Other errors - retry
                error_detail = response.text if response.text else f"HTTP {response.status_code}"
                logger.warning(f"Upload failed for {resource_type} (attempt {attempt + 1}): {error_detail}")
                
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return False, f"Upload failed after {max_retries} attempts: {error_detail}"
                    
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for {resource_type} (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return False, f"Connection failed after {max_retries} attempts"
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout error for {resource_type} (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return False, f"Timeout after {max_retries} attempts"
        except Exception as e:
            logger.error(f"Unexpected error for {resource_type}: {str(e)}")
            return False, f"Unexpected error: {str(e)}"
    
    return False, "Maximum retries exceeded"

def test_fhir_server_connection():
    """Test if FHIR server is accessible"""
    try:
        response = requests.get(f"{Config.FHIR_URL}/metadata", timeout=10)
        return response.status_code == 200
    except:
        return False

# ─── HL7 TO FHIR CONVERSION ─────────────────────────────────────────────────
def convert_hl7_to_fhir(hl7_file_path, patient_id, clinical_text, diagnostic_content=None):
    """Convert HL7 data to FHIR resources with comprehensive diagnostic report content including discharge summary"""
    try:
        # Clean patient ID for FHIR
        patient_fhir_id = patient_id.replace(".", "-").replace("_", "-").replace(" ", "-")
        
        # Ensure ID is valid (alphanumeric and hyphens only)
        import re
        if not re.match(r'^[A-Za-z0-9\-]+$', patient_fhir_id):
            # If invalid, create a safe ID
            patient_fhir_id = f"patient-{abs(hash(patient_id)) % 100000}"
        
        # Create stable, deterministic IDs
        observation_id = f"obs-{abs(hash(f'{patient_id}-observation')) % 100000}"
        diagnostic_report_id = f"dr-{abs(hash(f'{patient_id}-diagnostic-report')) % 100000}"
        
        # Use proper ISO 8601 datetime format
        current_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        logger.info(f"Creating FHIR resources for patient {patient_id}")
        logger.info(f"Patient FHIR ID: {patient_fhir_id}")
        logger.info(f"DiagnosticReport ID: {diagnostic_report_id}")
        logger.info(f"DateTime: {current_time}")
        
        # Create Patient resource
        patient_resource = {
            "resourceType": "Patient",
            "id": patient_fhir_id,
            "identifier": [
                {
                    "use": "usual",
                    "system": "http://hospital.org/patients",
                    "value": patient_id
                }
            ],
            "name": [
                {
                    "use": "official",
                    "family": "Patient",
                    "given": [patient_id]
                }
            ],
            "gender": "unknown",
            "active": True
        }
        
        # LOINC code mappings for vital signs and laboratory tests
        loinc_mappings = {
            'heart_rate': {'code': '8867-4', 'display': 'Heart rate', 'unit': 'beats/minute', 'system_unit': '/min'},
            'systolic': {'code': '8480-6', 'display': 'Systolic blood pressure', 'unit': 'mmHg', 'system_unit': 'mm[Hg]'},
            'diastolic': {'code': '8462-4', 'display': 'Diastolic blood pressure', 'unit': 'mmHg', 'system_unit': 'mm[Hg]'},
            'oxygen_saturation': {'code': '2708-6', 'display': 'Oxygen saturation', 'unit': '%', 'system_unit': '%'},
            'temperature': {'code': '8310-5', 'display': 'Body temperature', 'unit': '°C', 'system_unit': 'Cel'},
            'wbc': {'code': '6690-2', 'display': 'WBC count', 'unit': 'K/uL', 'system_unit': '10*3/uL'},
            'hemoglobin': {'code': '718-7', 'display': 'Hemoglobin', 'unit': 'g/dL', 'system_unit': 'g/dL'},
            'platelets': {'code': '777-3', 'display': 'Platelets', 'unit': 'K/uL', 'system_unit': '10*3/uL'},
            'crp': {'code': '1988-5', 'display': 'C-reactive protein', 'unit': 'mg/L', 'system_unit': 'mg/L'},
            'procalcitonin': {'code': '33959-8', 'display': 'Procalcitonin', 'unit': 'ng/mL', 'system_unit': 'ng/mL'},
        }
        
        # Parse vital signs and labs from clinical text
        vitals_data = {}
        labs_data = {}
        
        # Extract numeric values from clinical text
        import re as regex_module
        clinical_text_lower = clinical_text.lower() if clinical_text else ""
        
        # Vital signs extraction
        hr_match = regex_module.search(r'heart\s+rate[:\s]+(\d+)', clinical_text_lower)
        if hr_match:
            vitals_data['heart_rate'] = float(hr_match.group(1))
        
        sbp_match = regex_module.search(r'systolic[:\s]+(\d+)', clinical_text_lower)
        if sbp_match:
            vitals_data['systolic'] = float(sbp_match.group(1))
        
        dbp_match = regex_module.search(r'diastolic[:\s]+(\d+)', clinical_text_lower)
        if dbp_match:
            vitals_data['diastolic'] = float(dbp_match.group(1))
        
        o2_match = regex_module.search(r'oxygen\s+saturation[:\s]+(\d+)', clinical_text_lower)
        if o2_match:
            vitals_data['oxygen_saturation'] = float(o2_match.group(1))
        
        temp_match = regex_module.search(r'temperature[:\s]+(\d+\.?\d*)', clinical_text_lower)
        if temp_match:
            vitals_data['temperature'] = float(temp_match.group(1))
        
        # Labs extraction
        wbc_match = regex_module.search(r'wbc[:\s]+(\d+\.?\d*)', clinical_text_lower)
        if wbc_match:
            labs_data['wbc'] = float(wbc_match.group(1))
        
        hgb_match = regex_module.search(r'hemoglobin[:\s]+(\d+\.?\d*)', clinical_text_lower)
        if hgb_match:
            labs_data['hemoglobin'] = float(hgb_match.group(1))
        
        plt_match = regex_module.search(r'platelets[:\s]+(\d+\.?\d*)', clinical_text_lower)
        if plt_match:
            labs_data['platelets'] = float(plt_match.group(1))
        
        crp_match = regex_module.search(r'crp[:\s]+(\d+\.?\d*)', clinical_text_lower)
        if crp_match:
            labs_data['crp'] = float(crp_match.group(1))
        
        procalcitonin_match = regex_module.search(r'procalcitonin[:\s]+(\d+\.?\d*)', clinical_text_lower)
        if procalcitonin_match:
            labs_data['procalcitonin'] = float(procalcitonin_match.group(1))
        
        # Create main clinical note observation (for reference)
        observation_resource = {
            "resourceType": "Observation",
            "id": observation_id,
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "survey",
                            "display": "Survey"
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "72133-2",
                        "display": "Clinical note"
                    }
                ]
            },
            "subject": {
                "reference": f"Patient/{patient_fhir_id}"
            },
            "effectiveDateTime": current_time,
            "valueString": clinical_text[:1000] if clinical_text else "No clinical text provided"
        }
        
        # Create individual observations for vital signs
        observations_list = []
        
        for vital_key, vital_value in vitals_data.items():
            if vital_value and vital_key in loinc_mappings:
                obs_id = f"obs-{vital_key}-{abs(hash(f'{patient_id}-{vital_key}')) % 100000}"
                vital_obs = {
                    "resourceType": "Observation",
                    "id": obs_id,
                    "status": "final",
                    "category": [
                        {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                    "code": "vital-signs",
                                    "display": "Vital Signs"
                                }
                            ]
                        }
                    ],
                    "code": {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": loinc_mappings[vital_key]['code'],
                                "display": loinc_mappings[vital_key]['display']
                            }
                        ]
                    },
                    "subject": {
                        "reference": f"Patient/{patient_fhir_id}"
                    },
                    "effectiveDateTime": current_time,
                    "valueQuantity": {
                        "value": vital_value,
                        "unit": loinc_mappings[vital_key]['unit'],
                        "system": "http://unitsofmeasure.org",
                        "code": loinc_mappings[vital_key]['system_unit']
                    }
                }
                observations_list.append(vital_obs)
                logger.info(f"Created observation: {vital_key} = {vital_value} {loinc_mappings[vital_key]['unit']}")
        
        # Create individual observations for lab results
        for lab_key, lab_value in labs_data.items():
            if lab_value and lab_key in loinc_mappings:
                obs_id = f"obs-{lab_key}-{abs(hash(f'{patient_id}-{lab_key}')) % 100000}"
                lab_obs = {
                    "resourceType": "Observation",
                    "id": obs_id,
                    "status": "final",
                    "category": [
                        {
                            "coding": [
                                {
                                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                                    "code": "laboratory",
                                    "display": "Laboratory"
                                }
                            ]
                        }
                    ],
                    "code": {
                        "coding": [
                            {
                                "system": "http://loinc.org",
                                "code": loinc_mappings[lab_key]['code'],
                                "display": loinc_mappings[lab_key]['display']
                            }
                        ]
                    },
                    "subject": {
                        "reference": f"Patient/{patient_fhir_id}"
                    },
                    "effectiveDateTime": current_time,
                    "valueQuantity": {
                        "value": lab_value,
                        "unit": loinc_mappings[lab_key]['unit'],
                        "system": "http://unitsofmeasure.org",
                        "code": loinc_mappings[lab_key]['system_unit']
                    }
                }
                observations_list.append(lab_obs)
                logger.info(f"Created observation: {lab_key} = {lab_value} {loinc_mappings[lab_key]['unit']}")
        
        # Build conclusion content from HL7 OBX segments
        conclusion_parts = []
        
        # Add diagnostic content if available
        if diagnostic_content:
            # Add discharge summary first if available (highest priority)
            if diagnostic_content.get('discharge_summary'):
                conclusion_parts.append("=== DISCHARGE SUMMARY ===")
                conclusion_parts.extend(diagnostic_content['discharge_summary'])
                conclusion_parts.append("")
            
            # Add findings from OBX segments
            if diagnostic_content.get('findings'):
                conclusion_parts.append("FINDINGS:")
                conclusion_parts.extend([f"- {finding}" for finding in diagnostic_content['findings']])
                conclusion_parts.append("")
            
            # Add impressions/conclusions
            if diagnostic_content.get('impressions'):
                conclusion_parts.append("IMPRESSION:")
                conclusion_parts.extend([f"- {impression}" for impression in diagnostic_content['impressions']])
                conclusion_parts.append("")
            
            # Add recommendations
            if diagnostic_content.get('recommendations'):
                conclusion_parts.append("RECOMMENDATIONS:")
                conclusion_parts.extend([f"- {rec}" for rec in diagnostic_content['recommendations']])
                conclusion_parts.append("")
            
            # Add diagnosis codes
            if diagnostic_content.get('diagnosis_codes'):
                conclusion_parts.append("DIAGNOSES:")
                for diag in diagnostic_content['diagnosis_codes']:
                    conclusion_parts.append(f"- {diag['code']}: {diag['description']}")
                conclusion_parts.append("")
            
            # Add procedure codes
            if diagnostic_content.get('procedure_codes'):
                conclusion_parts.append("PROCEDURES:")
                for proc in diagnostic_content['procedure_codes']:
                    conclusion_parts.append(f"- {proc['code']}: {proc['description']}")
                conclusion_parts.append("")
            
            # Add general report text
            if diagnostic_content.get('report_text'):
                conclusion_parts.append("ADDITIONAL NOTES:")
                conclusion_parts.extend([f"- {text}" for text in diagnostic_content['report_text']])
        
        # Ensure we always have conclusion content
        conclusion_text = '\n'.join(conclusion_parts).strip()
        if not conclusion_text:
            conclusion_text = clinical_text if clinical_text else "No clinical content available from HL7 OBX segments"
        
        # Create FHIR-compliant DiagnosticReport resource
        diagnostic_report = {
            "resourceType": "DiagnosticReport",
            "id": diagnostic_report_id,
            "status": "final",
            "code": {
                "text": "Report"
            },
            "subject": {
                "reference": f"Patient/{patient_fhir_id}"
            },
            "effectiveDateTime": current_time,
            "issued": current_time,
            "conclusion": conclusion_text
        }
        
        # Add category based on content type
        is_discharge = diagnostic_content and diagnostic_content.get('discharge_summary')
        if is_discharge:
            diagnostic_report["category"] = [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": "DS",
                            "display": "Discharge Summary"
                        }
                    ]
                }
            ]
            diagnostic_report["code"] = {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "18842-5",
                        "display": "Discharge summary"
                    }
                ],
                "text": "Discharge Summary Report"
            }
        else:
            diagnostic_report["category"] = [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": "LAB",
                            "display": "Laboratory"
                        }
                    ]
                }
            ]
            diagnostic_report["code"] = {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "11502-2",
                        "display": "Laboratory report"
                    }
                ],
                "text": "Clinical Report"
            }
        
        logger.info(f"DiagnosticReport conclusion length: {len(conclusion_text)}")
        logger.info(f"DiagnosticReport JSON structure complete")
        
        # Parse HL7 and create individual observations for vital signs and labs
        observations_list = []
        
        # Parse vital signs and create individual observations
        if diagnostic_content:
            hl7_text = '\n'.join(clinical_text) if isinstance(clinical_text, list) else clinical_text
            
            # Extract vital signs from clinical text
            vitals_data = {
                'HR': None, 'SBP': None, 'DBP': None, 'SpO2': None, 'Temp': None, 'Weight': None, 'Height': None,
                'WBC': None, 'Hemoglobin': None, 'Platelets': None, 'CRP': None, 'Procalcitonin': None
            }
            
            # Parse numeric values from HL7
            parts = hl7_text.split('\n')
            values = []
            for part in parts:
                try:
                    val = float(part.strip())
                    values.append(val)
                except:
                    pass
            
            # Map values to vitals (assuming order: HR, SBP, DBP, SpO2, Temp, Weight, Height, WBC, Hgb, Platelets, CRP, Procalcitonin)
            if len(values) >= 7:
                vitals_data['HR'] = values[0]
                vitals_data['SBP'] = values[1]
                vitals_data['DBP'] = values[2]
                vitals_data['SpO2'] = values[3]
                vitals_data['Temp'] = values[4]
                vitals_data['Weight'] = values[5]
                vitals_data['Height'] = values[6]
            if len(values) >= 12:
                vitals_data['WBC'] = values[7]
                vitals_data['Hemoglobin'] = values[8]
                vitals_data['Platelets'] = values[9]
                vitals_data['CRP'] = values[10]
                vitals_data['Procalcitonin'] = values[11]
            
            # Create Heart Rate observation
            if vitals_data['HR']:
                observations_list.append({
                    "resourceType": "Observation",
                    "id": f"obs-hr-{abs(hash(f'{patient_id}-HR')) % 100000}",
                    "status": "final",
                    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs", "display": "Vital Signs"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": "8867-4", "display": "Heart Rate"}]},
                    "subject": {"reference": f"Patient/{patient_fhir_id}"},
                    "effectiveDateTime": current_time,
                    "valueQuantity": {"value": vitals_data['HR'], "unit": "beats/minute", "system": "http://unitsofmeasure.org", "code": "/min"}
                })
            
            # Create BP observations (Systolic and Diastolic)
            if vitals_data['SBP']:
                observations_list.append({
                    "resourceType": "Observation",
                    "id": f"obs-sbp-{abs(hash(f'{patient_id}-SBP')) % 100000}",
                    "status": "final",
                    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs", "display": "Vital Signs"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic BP"}]},
                    "subject": {"reference": f"Patient/{patient_fhir_id}"},
                    "effectiveDateTime": current_time,
                    "valueQuantity": {"value": vitals_data['SBP'], "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}
                })
            
            if vitals_data['DBP']:
                observations_list.append({
                    "resourceType": "Observation",
                    "id": f"obs-dbp-{abs(hash(f'{patient_id}-DBP')) % 100000}",
                    "status": "final",
                    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs", "display": "Vital Signs"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": "8462-4", "display": "Diastolic BP"}]},
                    "subject": {"reference": f"Patient/{patient_fhir_id}"},
                    "effectiveDateTime": current_time,
                    "valueQuantity": {"value": vitals_data['DBP'], "unit": "mmHg", "system": "http://unitsofmeasure.org", "code": "mm[Hg]"}
                })
            
            # Create O2 Saturation observation
            if vitals_data['SpO2']:
                observations_list.append({
                    "resourceType": "Observation",
                    "id": f"obs-spo2-{abs(hash(f'{patient_id}-SpO2')) % 100000}",
                    "status": "final",
                    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs", "display": "Vital Signs"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": "2708-6", "display": "Oxygen Saturation"}]},
                    "subject": {"reference": f"Patient/{patient_fhir_id}"},
                    "effectiveDateTime": current_time,
                    "valueQuantity": {"value": vitals_data['SpO2'], "unit": "%", "system": "http://unitsofmeasure.org", "code": "%"}
                })
            
            # Create Temperature observation
            if vitals_data['Temp']:
                observations_list.append({
                    "resourceType": "Observation",
                    "id": f"obs-temp-{abs(hash(f'{patient_id}-Temp')) % 100000}",
                    "status": "final",
                    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs", "display": "Vital Signs"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": "8310-5", "display": "Body Temperature"}]},
                    "subject": {"reference": f"Patient/{patient_fhir_id}"},
                    "effectiveDateTime": current_time,
                    "valueQuantity": {"value": vitals_data['Temp'], "unit": "Celsius", "system": "http://unitsofmeasure.org", "code": "Cel"}
                })
            
            # Create Lab Result observations
            if vitals_data['WBC']:
                observations_list.append({
                    "resourceType": "Observation",
                    "id": f"obs-wbc-{abs(hash(f'{patient_id}-WBC')) % 100000}",
                    "status": "final",
                    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory", "display": "Laboratory"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": "6690-2", "display": "WBC Count"}]},
                    "subject": {"reference": f"Patient/{patient_fhir_id}"},
                    "effectiveDateTime": current_time,
                    "valueQuantity": {"value": vitals_data['WBC'], "unit": "K/uL", "system": "http://unitsofmeasure.org", "code": "10*3/uL"}
                })
            
            if vitals_data['Hemoglobin']:
                observations_list.append({
                    "resourceType": "Observation",
                    "id": f"obs-hgb-{abs(hash(f'{patient_id}-Hemoglobin')) % 100000}",
                    "status": "final",
                    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory", "display": "Laboratory"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": "718-7", "display": "Hemoglobin"}]},
                    "subject": {"reference": f"Patient/{patient_fhir_id}"},
                    "effectiveDateTime": current_time,
                    "valueQuantity": {"value": vitals_data['Hemoglobin'], "unit": "g/dL", "system": "http://unitsofmeasure.org", "code": "g/dL"}
                })
            
            if vitals_data['Platelets']:
                observations_list.append({
                    "resourceType": "Observation",
                    "id": f"obs-plt-{abs(hash(f'{patient_id}-Platelets')) % 100000}",
                    "status": "final",
                    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory", "display": "Laboratory"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": "777-3", "display": "Platelet Count"}]},
                    "subject": {"reference": f"Patient/{patient_fhir_id}"},
                    "effectiveDateTime": current_time,
                    "valueQuantity": {"value": vitals_data['Platelets'], "unit": "K/uL", "system": "http://unitsofmeasure.org", "code": "10*3/uL"}
                })
            
            if vitals_data['CRP']:
                observations_list.append({
                    "resourceType": "Observation",
                    "id": f"obs-crp-{abs(hash(f'{patient_id}-CRP')) % 100000}",
                    "status": "final",
                    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory", "display": "Laboratory"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": "30341-2", "display": "CRP"}]},
                    "subject": {"reference": f"Patient/{patient_fhir_id}"},
                    "effectiveDateTime": current_time,
                    "valueQuantity": {"value": vitals_data['CRP'], "unit": "mg/L", "system": "http://unitsofmeasure.org", "code": "mg/L"}
                })
            
            if vitals_data['Procalcitonin']:
                observations_list.append({
                    "resourceType": "Observation",
                    "id": f"obs-pct-{abs(hash(f'{patient_id}-Procalcitonin')) % 100000}",
                    "status": "final",
                    "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory", "display": "Laboratory"}]}],
                    "code": {"coding": [{"system": "http://loinc.org", "code": "33959-8", "display": "Procalcitonin"}]},
                    "subject": {"reference": f"Patient/{patient_fhir_id}"},
                    "effectiveDateTime": current_time,
                    "valueQuantity": {"value": vitals_data['Procalcitonin'], "unit": "ng/mL", "system": "http://unitsofmeasure.org", "code": "ng/mL"}
                })
        
        return {
            "patient": patient_resource,
            "observation": observation_resource,
            "observations_list": observations_list,
            "diagnostic_report": diagnostic_report
        }
        
    except Exception as e:
        logger.error(f"HL7 to FHIR conversion error: {str(e)}")
        raise Exception(f"HL7 to FHIR conversion failed: {str(e)}")

def upload_fhir_resources(fhir_resources, patient_id):
    """Upload FHIR resources to the FHIR server with comprehensive debugging and verification"""
    try:
        # Test server connection first
        if not test_fhir_server_connection():
            logger.error("FHIR server connection test failed")
            return {
                'patient': False,
                'observation': False,
                'diagnostic_report': False,
                'error': 'FHIR server is not accessible'
            }
        
        logger.info("FHIR server connection successful")
        results = {}
        detailed_errors = []
        
        # Upload Patient first
        logger.info(f"Uploading Patient resource for {patient_id}")
        success, message = upload_fhir_resource_with_retry(
            fhir_resources['patient'], 'Patient'
        )
        results['patient'] = success
        if not success:
            detailed_errors.append(f"Patient: {message}")
            logger.error(f"Patient upload failed: {message}")
        else:
            logger.info(f"Patient upload successful for {patient_id}")
            # Update dependent resources to reference the authoritative server Patient id
            try:
                server_patient_id = fhir_resources['patient'].get('id')
                if server_patient_id:
                    # Update DiagnosticReport subject reference
                    if 'diagnostic_report' in fhir_resources:
                        fhir_resources['diagnostic_report'].setdefault('subject', {})
                        fhir_resources['diagnostic_report']['subject']['reference'] = f"Patient/{server_patient_id}"
                    # Update Observation subject reference
                    if 'observation' in fhir_resources:
                        fhir_resources['observation'].setdefault('subject', {})
                        fhir_resources['observation']['subject']['reference'] = f"Patient/{server_patient_id}"
                    logger.info(f"Updated dependent resources to reference Patient/{server_patient_id}")
            except Exception as e:
                logger.warning(f"Could not update dependent resource references: {str(e)}")

            # Wait for Patient to be indexed before creating dependent resources
            time.sleep(1.5)
            
            # Verify Patient exists before creating dependent resources
            try:
                patient_id_clean = fhir_resources['patient']['id']
                verify_patient_url = f"{Config.FHIR_URL}/Patient/{patient_id_clean}"
                verify_patient_response = requests.get(verify_patient_url, headers=Config.FHIR_HDR, timeout=30)
                if verify_patient_response.status_code == 200:
                    logger.info(f"✅ Patient {patient_id_clean} verified and ready for dependent resources")
                else:
                    logger.warning(f"⚠️ Patient verification returned {verify_patient_response.status_code}, proceeding anyway")
            except Exception as pv:
                logger.warning(f"Could not verify Patient, proceeding: {str(pv)}")
        
        # Upload DiagnosticReport - this is the priority resource
        if results['patient']:
            logger.info(f"Uploading DiagnosticReport for {patient_id}")
            
            success, message = upload_fhir_resource_with_retry(
                fhir_resources['diagnostic_report'], 'DiagnosticReport'
            )
            results['diagnostic_report'] = success
            if not success:
                detailed_errors.append(f"DiagnosticReport: {message}")
                logger.error(f"DiagnosticReport upload failed: {message}")
            else:
                logger.info(f"DiagnosticReport upload successful for {patient_id}")
                
                # Additional verification: Query the FHIR server to confirm DiagnosticReport exists
                try:
                    dr_id = fhir_resources['diagnostic_report']['id']
                    verify_url = f"{Config.FHIR_URL}/DiagnosticReport/{dr_id}"
                    verify_response = requests.get(verify_url, headers=Config.FHIR_HDR, timeout=30)
                    
                    if verify_response.status_code == 200:
                        logger.info(f"✅ DiagnosticReport {dr_id} confirmed in FHIR server")
                        
                        # Also check total count
                        count_url = f"{Config.FHIR_URL}/DiagnosticReport"
                        count_response = requests.get(count_url, headers=Config.FHIR_HDR, timeout=30)
                        if count_response.status_code == 200:
                            count_data = count_response.json()
                            total = count_data.get('total', 0)
                            logger.info(f"Total DiagnosticReports in server: {total}")
                    else:
                        logger.warning(f"⚠️ Could not verify DiagnosticReport {dr_id} after upload")
                except Exception as ve:
                    logger.error(f"Error verifying DiagnosticReport: {str(ve)}")
        else:
            results['diagnostic_report'] = False
            detailed_errors.append("DiagnosticReport: Skipped due to Patient upload failure")
            logger.warning("DiagnosticReport skipped - Patient upload failed")
        
        # Upload Observation (lower priority)
        if results['patient']:
            logger.info(f"Uploading Observation for {patient_id}")
            success, message = upload_fhir_resource_with_retry(
                fhir_resources['observation'], 'Observation'
            )
            results['observation'] = success
            if not success:
                detailed_errors.append(f"Observation: {message}")
                logger.error(f"Observation upload failed: {message}")
            else:
                logger.info(f"Observation upload successful for {patient_id}")
        else:
            results['observation'] = False
            detailed_errors.append("Observation: Skipped due to Patient upload failure")
        
        # Upload individual observations from observations_list (vital signs and labs)
        if results['patient']:
            observations_list = fhir_resources.get('observations_list', [])
            if observations_list:
                logger.info(f"Uploading {len(observations_list)} individual observations (vitals and labs)")
                results['observations_list'] = []
                
                for obs in observations_list:
                    obs_id = obs.get('id', 'unknown')
                    obs_display = obs.get('code', {}).get('coding', [{}])[0].get('display', 'Unknown')
                    
                    logger.info(f"Uploading Observation: {obs_display} (ID: {obs_id})")
                    success, message = upload_fhir_resource_with_retry(obs, 'Observation')
                    
                    if success:
                        logger.info(f"✅ {obs_display} observation uploaded successfully")
                        results['observations_list'].append({
                            'id': obs_id,
                            'display': obs_display,
                            'success': True
                        })
                    else:
                        logger.error(f"❌ {obs_display} observation upload failed: {message}")
                        detailed_errors.append(f"{obs_display} Observation ({obs_id}): {message}")
                        results['observations_list'].append({
                            'id': obs_id,
                            'display': obs_display,
                            'success': False
                        })
            else:
                logger.info("No individual observations to upload (observations_list is empty)")
                results['observations_list'] = []
        else:
            results['observations_list'] = []
            logger.warning("Observations list: Skipped due to Patient upload failure")
        
        if detailed_errors:
            results['error'] = '; '.join(detailed_errors)
        
        logger.info(f"Final upload results for {patient_id}: {results}")
        return results
        
    except Exception as e:
        logger.error(f"FHIR upload error for {patient_id}: {str(e)}")
        return {
            'patient': False,
            'observation': False,
            'diagnostic_report': False,
            'error': f"Upload process failed: {str(e)}"
        }

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def allowed_file(filename, allowed_exts):
    return "." in filename and filename.rsplit(".", 1)[1] in allowed_exts

def parse_hl7_message(file_path):
    """Parse HL7 message and extract PatientID, clinical text, and diagnostic report content including discharge summary"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            hl7_text = f.read()
        
        # Manual HL7 parsing (reliable fallback method)
        # Normalize line endings
        hl7_text = hl7_text.replace('\r\n', '\n').replace('\r', '\n')
        
        patient_id = None
        clinical_text = []
        # patient demographics
        patient_info = {
            'id': None,
            'name': None,
            'dob': None,
            'age': None,
            'sex': None,
            'phone': None,
            'address': None
        }

        diagnostic_report_content = {
            'findings': [],
            'impressions': [],
            'recommendations': [],
            'procedure_codes': [],
            'diagnosis_codes': [],
            'report_text': [],
            'discharge_summary': []  # New field for discharge summary
        }
        
        lines = hl7_text.strip().split('\n')
        for line in lines:
            if not line.strip():
                continue
                
            fields = line.split('|')
            if len(fields) < 2:
                continue
                
            segment_type = fields[0].strip()
            
            if segment_type == 'PID' and len(fields) > 3:
                # PID segment - field 3 is patient ID
                raw_pid = fields[3].strip()
                # Handle composite patient ID (extract first part before ^)
                patient_id = raw_pid.split('^')[0] if '^' in raw_pid else raw_pid
                patient_info['id'] = patient_id

                # Patient name may be in field 5 (family^given^middle)
                try:
                    raw_name = fields[5].strip() if len(fields) > 5 else ''
                    if raw_name:
                        name_parts = raw_name.split('^')
                        # Compose readable name
                        if len(name_parts) >= 2:
                            patient_info['name'] = f"{name_parts[1]} {name_parts[0]}".strip()
                        else:
                            patient_info['name'] = raw_name
                except:
                    pass

                # DOB in field 7 (YYYYMMDD)
                try:
                    raw_dob = fields[7].strip() if len(fields) > 7 else ''
                    if raw_dob:
                        # format YYYYMMDD into YYYY-MM-DD
                        if len(raw_dob) >= 8:
                            dob = f"{raw_dob[0:4]}-{raw_dob[4:6]}-{raw_dob[6:8]}"
                            patient_info['dob'] = dob
                except:
                    pass

                # Sex in field 8
                try:
                    patient_info['sex'] = fields[8].strip() if len(fields) > 8 else None
                except:
                    pass

                # Phone may be in field 13 or 14 depending on HL7 flavor
                try:
                    phone = fields[13].strip() if len(fields) > 13 else ''
                    if phone:
                        patient_info['phone'] = phone
                except:
                    pass
                
            elif segment_type == 'OBX' and len(fields) > 5:
                # Observation segment - field 5 is observation value
                obs_value = fields[5].strip()
                if obs_value:
                    clinical_text.append(obs_value)

                    # Build a structured observation
                    obs_struct = {
                        'id': fields[3].strip() if len(fields) > 3 else None,
                        'value': obs_value,
                        'units': fields[6].strip() if len(fields) > 6 else None,
                        'status': fields[11].strip() if len(fields) > 11 else None
                    }

                    # Append to observations in diagnostic content
                    diagnostic_report_content.setdefault('observations', []).append(obs_struct)

                    # Check observation type in field 3 (observation identifier)
                    if len(fields) > 3:
                        obs_id = fields[3].strip().upper()

                        # Check for discharge summary content
                        if any(keyword in obs_id for keyword in ['DISCHARGE', 'DSCH', 'SUMMARY']):
                            diagnostic_report_content['discharge_summary'].append(obs_value)
                        elif 'FINDING' in obs_id or 'RESULT' in obs_id:
                            diagnostic_report_content['findings'].append(obs_value)
                        elif 'IMPRESSION' in obs_id or 'CONCLUSION' in obs_id:
                            diagnostic_report_content['impressions'].append(obs_value)
                        elif 'RECOMMENDATION' in obs_id or 'SUGGEST' in obs_id:
                            diagnostic_report_content['recommendations'].append(obs_value)
                        else:
                            diagnostic_report_content['report_text'].append(obs_value)
                    
            elif segment_type == 'NTE' and len(fields) > 3:
                # Notes segment - field 3 is comment
                note_text = fields[3].strip()
                if note_text:
                    clinical_text.append(note_text)
                    
                    # Check if this note contains discharge summary information
                    note_upper = note_text.upper()
                    if any(keyword in note_upper for keyword in ['DISCHARGE SUMMARY', 'DISCHARGE:', 'SUMMARY:']):
                        diagnostic_report_content['discharge_summary'].append(note_text)
                    else:
                        diagnostic_report_content['report_text'].append(note_text)
                    
            elif segment_type == 'DG1' and len(fields) > 3:
                # Diagnosis segment - field 3 is diagnosis code, field 4 is description
                if len(fields) > 3:
                    diag_code = fields[3].strip()
                    diag_desc = fields[4].strip() if len(fields) > 4 else ''
                    if diag_code:
                        diagnostic_report_content['diagnosis_codes'].append({
                            'code': diag_code,
                            'description': diag_desc
                        })
                        clinical_text.append(f"Diagnosis: {diag_code} - {diag_desc}")
                        
            elif segment_type == 'PR1' and len(fields) > 3:
                # Procedure segment - field 3 is procedure code, field 4 is description
                if len(fields) > 3:
                    proc_code = fields[3].strip()
                    proc_desc = fields[4].strip() if len(fields) > 4 else ''
                    if proc_code:
                        diagnostic_report_content['procedure_codes'].append({
                            'code': proc_code,
                            'description': proc_desc
                        })
                        clinical_text.append(f"Procedure: {proc_code} - {proc_desc}")
        
        # Join all clinical text
        report_text = '\n'.join(clinical_text) if clinical_text else "HL7 message processed"

        # Attach patient demographics into diagnostic_report_content for downstream use
        diagnostic_report_content['patient_info'] = patient_info

        # Also compute age from DOB if available
        try:
            if patient_info.get('dob'):
                from datetime import datetime, date
                dob_dt = datetime.strptime(patient_info['dob'], '%Y-%m-%d').date()
                today = date.today()
                age = today.year - dob_dt.year - ((today.month, today.day) < (dob_dt.month, dob_dt.day))
                patient_info['age'] = age
                diagnostic_report_content['patient_info'] = patient_info
        except:
            pass

        return patient_id, report_text, diagnostic_report_content
        
    except Exception as e:
        return None, f"Error parsing HL7: {str(e)}", None

def parse_json_patient_data(file_path):
    """Parse JSON file with patient data and extract PatientID and clinical text"""
    try:
        import json
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle FHIR Patient resource format
        if data.get("resourceType") == "Patient":
            patient_id = data.get("id")
            name_parts = []
            if "name" in data and len(data["name"]) > 0:
                name_obj = data["name"][0]
                if "given" in name_obj:
                    name_parts.extend(name_obj["given"])
                if "family" in name_obj:
                    name_parts.append(name_obj["family"])
            
            patient_name = " ".join(name_parts) if name_parts else "Unknown"
            report_text = f"FHIR Patient Resource - Name: {patient_name}, Gender: {data.get('gender', 'Unknown')}, Birth Date: {data.get('birthDate', 'Unknown')}"
            
            return patient_id, report_text
        
        # Handle custom patient data format
        patient_id = data.get("patientId") or data.get("PatientID") or data.get("patient_id") or data.get("id")
        
        # Get clinical text from various possible fields
        clinical_text = []
        
        if "clinicalText" in data:
            clinical_text.append(data["clinicalText"])
        if "clinical_text" in data:
            clinical_text.append(data["clinical_text"])
        if "diagnosis" in data:
            clinical_text.append(f"Diagnosis: {data['diagnosis']}")
        if "findings" in data:
            if isinstance(data["findings"], list):
                clinical_text.extend([f"Finding: {finding}" for finding in data["findings"]])
            else:
                clinical_text.append(f"Findings: {data['findings']}")
        if "reports" in data and isinstance(data["reports"], list):
            clinical_text.extend(data["reports"])
        if "conclusion" in data:
            clinical_text.append(f"Conclusion: {data['conclusion']}")
        if "recommendations" in data:
            clinical_text.append(f"Recommendations: {data['recommendations']}")
            
        report_text = '\n'.join(clinical_text) if clinical_text else "JSON patient data processed"
        
        return patient_id, report_text
        
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON format: {str(e)}"
    except Exception as e:
        return None, f"Error parsing JSON: {str(e)}"

def parse_patient_data_file(file_path):
    """Parse patient data file (JSON, HL7, or TXT) and extract PatientID, clinical text, and diagnostic content"""
    file_ext = file_path.split('.')[-1].lower()
    
    if file_ext == 'json':
        pid, report = parse_json_patient_data(file_path)
        return pid, report, None  # JSON doesn't have structured diagnostic content yet
    elif file_ext in ['hl7', 'txt']:
        # Try HL7 first, fallback to simple text format
        try:
            return parse_hl7_message(file_path)
        except:
            # Fallback to simple text format (first line = PatientID)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.read().splitlines()
                if lines:
                    patient_id = lines[0].strip()
                    report = '\n'.join(lines[1:]).strip() if len(lines) > 1 else "Text data processed"
                    return patient_id, report, None
                return None, "Empty file", None
            except Exception as e:
                return None, f"Error reading text file: {str(e)}", None
    else:
        return None, f"Unsupported file format: {file_ext}", None

def load_patient_mapping():
    """Load the DICOM to patient ID mapping from file"""
    mapping_file = os.path.join(Config.UPLOAD_FOLDER, "dicom_patient_map.json")
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, 'r') as f:
                return json.load(f)  # Returns {patient_id: [dicom_file_paths]}
        except Exception as e:
            logger.warning(f"Could not load patient mapping: {e}")
    return {}

# ─── ROUTES ──────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Test FHIR server connection at start
        if not test_fhir_server_connection():
            flash("FHIR server is not accessible. Please check the server status.", "error")
            return render_template("index.html")
        
        dicom_files = request.files.getlist("dicoms")
        hl7_files = request.files.getlist("hl7data")

        # 1) Process DICOM uploads and index by PatientID using mapping file
        # Load patient mapping to match DICOM files to logical patient IDs
        patient_mapping = load_patient_mapping()
        dicom_index = {}  # pid -> list of SOP Instance UIDs or filenames
        
        # Build reverse mapping: DICOM file path -> patient ID
        dicom_to_patient = {}
        for mapped_patient_id, mapped_dicom_files in patient_mapping.items():
            for dicom_path in mapped_dicom_files:
                # Normalize path separators and store just filename
                dicom_filename = os.path.basename(dicom_path)
                dicom_to_patient[dicom_filename] = mapped_patient_id
        
        for f in dicom_files:
            if f and allowed_file(f.filename, Config.ALLOWED_DCM):
                fname = secure_filename(f.filename)
                
                # Read DICOM data from uploaded file (without saving locally)
                f.seek(0)
                dicom_data = f.read()
                
                # upload to Orthanc
                try:
                    res = requests.post(
                        f"{Config.ORTHANC_URL}/instances",
                        auth=Config.ORTHANC_AUTH,
                        headers={"Content-Type": "application/dicom"},
                        data=dicom_data,
                        timeout=30
                    )
                    if res.status_code not in (200, 201):
                        flash(f"Failed to upload {fname} ({res.status_code}): {res.text[:100]}", "error")
                        continue
                except requests.exceptions.RequestException as e:
                    flash(f"Network error uploading {fname}: {str(e)}", "error")
                    continue

                # read header to get IDs from uploaded file
                try:
                    # Parse DICOM from bytes without saving to disk
                    from io import BytesIO
                    f.seek(0)
                    ds = pydicom.dcmread(BytesIO(f.read()), stop_before_pixels=True)
                    pid = ds.get("PatientID", None)
                    sop_uid = ds.get("SOPInstanceUID", None)
                    
                    # If file is in mapping, use the mapped patient ID
                    if fname in dicom_to_patient:
                        pid = dicom_to_patient[fname]
                    
                    # Validate required DICOM fields
                    if not pid:
                        flash(f"DICOM file {fname} missing PatientID and not in mapping", "warning")
                        continue
                    if not sop_uid:
                        flash(f"DICOM file {fname} missing SOPInstanceUID", "warning")
                        continue

                    # Store reference to DICOM file (from Orthanc, not local)
                    dicom_index.setdefault(pid, []).append(sop_uid)
                    flash(f"✅ Successfully uploaded DICOM to Orthanc: {fname} (Patient: {pid})", "success")
                        
                except Exception as e:
                    flash(f"Error reading DICOM file {fname}: {str(e)}", "error")
                    logger.error(f"DICOM read error: {e}")
                    continue

        # 2) Process Patient Data uploads with improved HL7 to FHIR conversion
        patient_data_index = {}
        conversion_results = []
        
        for f in hl7_files:
            if f and allowed_file(f.filename, Config.ALLOWED_PATIENT_DATA):
                fname = secure_filename(f.filename)
                path = os.path.join(Config.UPLOAD_FOLDER, fname)
                f.save(path)

                # Parse patient data file to extract PatientID, clinical text, and diagnostic content
                result = parse_patient_data_file(path)
                if len(result) == 3:
                    pid, report, diagnostic_content = result
                else:
                    pid, report = result
                    diagnostic_content = None
                
                if not pid:
                    flash(f"Could not extract PatientID from file: {fname} - {report}", "warning")
                    continue
                
                patient_data_index[pid] = report
                
                # Create conversion ID for tracking
                conversion_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                conversion_status[conversion_id] = {
                    "status": "processing",
                    "patient_id": pid,
                    "filename": fname,
                    "message": "Converting HL7 to FHIR...",
                    "timestamp": timestamp
                }
                
                try:
                    # Convert HL7 to FHIR with diagnostic content
                    conversion_status[conversion_id]["message"] = "Creating FHIR resources with diagnostic content..."
                    fhir_resources = convert_hl7_to_fhir(path, pid, report, diagnostic_content)
                    
                    # Upload to FHIR server
                    conversion_status[conversion_id]["message"] = "Uploading to FHIR server..."
                    upload_results = upload_fhir_resources(fhir_resources, pid)
                    
                    # Check results and provide detailed feedback
                    successful_uploads = [k for k, v in upload_results.items() if k != 'error' and v]
                    failed_uploads = [k for k, v in upload_results.items() if k != 'error' and not v]
                    
                    # Consider success if Patient and DiagnosticReport are uploaded (prioritize these)
                    critical_success = upload_results.get('patient', False) and upload_results.get('diagnostic_report', False)
                    
                    if critical_success and len(failed_uploads) == 0:  # All succeeded
                        conversion_status[conversion_id] = {
                            "status": "success",
                            "patient_id": pid,
                            "filename": fname,
                            "message": f"Successfully converted and uploaded all FHIR resources with diagnostic summary for Patient {pid}",
                            "timestamp": conversion_status[conversion_id].get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        }
                        flash(f"✅ Complete HL7 to FHIR conversion with diagnostic summary successful for Patient {pid}", "success")
                    elif critical_success:  # Patient and DiagnosticReport succeeded (most important)
                        error_detail = upload_results.get('error', 'Unknown error')
                        conversion_status[conversion_id] = {
                            "status": "success",
                            "patient_id": pid,
                            "filename": fname,
                            "message": f"Successfully uploaded Patient and DiagnosticReport with summary for Patient {pid}. Some optional resources failed: {', '.join(failed_uploads)}",
                            "timestamp": conversion_status[conversion_id].get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        }
                        flash(f"✅ HL7 to FHIR conversion with diagnostic summary successful for Patient {pid}. Diagnostic report uploaded successfully!", "success")
                    elif len(successful_uploads) > 0:  # Partial success
                        error_detail = upload_results.get('error', 'Unknown error')
                        conversion_status[conversion_id] = {
                            "status": "partial",
                            "patient_id": pid,
                            "filename": fname,
                            "message": f"Partial success for Patient {pid}. Uploaded: {', '.join(successful_uploads)}. Failed: {', '.join(failed_uploads)}. Error: {error_detail}",
                            "timestamp": conversion_status[conversion_id].get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        }
                        flash(f"⚠️ Partial HL7 to FHIR conversion for Patient {pid}. Uploaded: {', '.join(successful_uploads)}. Failed: {', '.join(failed_uploads)}. Details: {error_detail}", "warning")
                    else:  # All failed
                        error_detail = upload_results.get('error', 'All uploads failed')
                        conversion_status[conversion_id] = {
                            "status": "error",
                            "patient_id": pid,
                            "filename": fname,
                            "message": f"All uploads failed for Patient {pid}. Error: {error_detail}",
                            "timestamp": conversion_status[conversion_id].get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        }
                        flash(f"❌ HL7 to FHIR conversion failed for Patient {pid}. Error: {error_detail}", "error")
                    
                    conversion_results.append({
                        "conversion_id": conversion_id,
                        "patient_id": pid,
                        "filename": fname
                    })
                    
                except Exception as e:
                    error_msg = str(e)
                    conversion_status[conversion_id] = {
                        "status": "error",
                        "patient_id": pid,
                        "filename": fname,
                        "message": f"Conversion failed: {error_msg}",
                        "timestamp": conversion_status[conversion_id].get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    }
                    flash(f"❌ HL7 to FHIR conversion failed for {fname}: {error_msg}", "error")
                    logger.error(f"Conversion error for {fname}: {error_msg}")

        # 3) Match and queue AI ingestion
        pids_with_images = set(dicom_index)
        pids_with_patient_data = set(patient_data_index)
        matched = pids_with_images & pids_with_patient_data
        for pid in matched:
            sop_list = dicom_index[pid]
            report = patient_data_index[pid]
            # TODO: call your AI ingestion function, e.g.:
            # ai.ingest(pid, sop_list, report)
            flash(f"✅ Queued AI for PID {pid} ({len(sop_list)} images)", "success")

        # 4) Warn on unmatched only if there are significant mismatches
        # (suppress warnings if both DICOM and HL7 have been processed successfully)
        unmatched_images = pids_with_images - pids_with_patient_data
        unmatched_data = pids_with_patient_data - pids_with_images
        
        # Only warn if there are unmatched items and no matches at all
        if (unmatched_images or unmatched_data) and not matched:
            for pid in unmatched_images:
                logger.warning(f"No HL7/patient data match for DICOM Patient {pid}")
            for pid in unmatched_data:
                logger.warning(f"No DICOM match for patient data Patient {pid}")
        elif matched:
            # Unmatched items with some matches is okay
            if unmatched_images:
                for pid in unmatched_images:
                    logger.info(f"Optional: DICOM files exist for {pid} without matching HL7 data")
            if unmatched_data:
                for pid in unmatched_data:
                    logger.info(f"Optional: HL7 data exists for {pid} without matching DICOM files")

        # Store conversion results in session or pass to template
        if conversion_results:
            save_conversion_history()
            return render_template("index.html", conversion_results=conversion_results)

        save_conversion_history()
        return redirect(url_for("index"))

    return render_template("index.html")

# ─── NEW ROUTE FOR CONVERSION STATUS ───────────────────────────────────────
@app.route("/conversion-status/<conversion_id>")
def get_conversion_status(conversion_id):
    """Get the status of a conversion process"""
    status = conversion_status.get(conversion_id, {"status": "not_found"})
    return jsonify(status)

# ─── NEW ROUTES FOR VIEWING CONVERTED FHIR RESOURCES ──────────────────────
@app.route("/fhir-resources")
def view_fhir_resources():
    """View all FHIR resources in the server"""
    try:
        if not test_fhir_server_connection():
            flash("FHIR server is not accessible", "error")
            return render_template("fhir_resources.html", resources=None)
        
        # Get all Patients
        patients_response = requests.get(f"{Config.FHIR_URL}/Patient", timeout=30)
        patients = []
        if patients_response.status_code == 200:
            patients_data = patients_response.json()
            if 'entry' in patients_data:
                patients = [entry['resource'] for entry in patients_data['entry']]
        
        # Get all Observations
        observations_response = requests.get(f"{Config.FHIR_URL}/Observation", timeout=30)
        observations = []
        if observations_response.status_code == 200:
            observations_data = observations_response.json()
            if 'entry' in observations_data:
                observations = [entry['resource'] for entry in observations_data['entry']]
        
        # Get all DiagnosticReports
        reports_response = requests.get(f"{Config.FHIR_URL}/DiagnosticReport", timeout=30)
        reports = []
        if reports_response.status_code == 200:
            reports_data = reports_response.json()
            if 'entry' in reports_data:
                reports = [entry['resource'] for entry in reports_data['entry']]
        
        resources = {
            'patients': patients,
            'observations': observations,
            'diagnostic_reports': reports
        }
        
        return render_template("fhir_resources.html", resources=resources)
        
    except Exception as e:
        logger.error(f"Error fetching FHIR resources: {str(e)}")
        flash(f"Error fetching FHIR resources: {str(e)}", "error")
        return render_template("fhir_resources.html", resources=None)

@app.route("/fhir-resource/<resource_type>/<resource_id>")
def view_single_fhir_resource(resource_type, resource_id):
    """View a single FHIR resource"""
    try:
        if not test_fhir_server_connection():
            flash("FHIR server is not accessible", "error")
            return redirect(url_for("view_fhir_resources"))
        
        response = requests.get(f"{Config.FHIR_URL}/{resource_type}/{resource_id}", timeout=30)
        
        if response.status_code == 200:
            resource = response.json()
            return render_template("single_fhir_resource.html", 
                                 resource=resource, 
                                 resource_type=resource_type,
                                 resource_id=resource_id)
        else:
            flash(f"Resource not found: {resource_type}/{resource_id}", "error")
            return redirect(url_for("view_fhir_resources"))
            
    except Exception as e:
        logger.error(f"Error fetching FHIR resource: {str(e)}")
        flash(f"Error fetching FHIR resource: {str(e)}", "error")
        return redirect(url_for("view_fhir_resources"))

@app.route("/conversion-history")
def conversion_history():
    """View conversion history"""
    return render_template("conversion_history.html", conversions=conversion_status)

@app.route("/dicom-gallery")
def dicom_gallery():
    """View all DICOM files from Orthanc server"""
    # Fetch all DICOM files from Orthanc instead of local uploads
    dicom_files = []
    try:
        response = requests.get(
            f"{Config.ORTHANC_URL}/instances",
            auth=Config.ORTHANC_AUTH,
            timeout=30
        )
        if response.status_code == 200:
            instances = response.json()
            dicom_files = instances  # List of instance UIDs
    except Exception as e:
        logger.warning(f"Could not fetch DICOM files from Orthanc: {e}")
    
    return render_template("dicom_gallery.html", dicom_files=dicom_files)

@app.route("/api/all-dicom-files")
def get_all_dicom_files():
    """Get all DICOM files from Orthanc server"""
    try:
        response = requests.get(
            f"{Config.ORTHANC_URL}/instances",
            auth=Config.ORTHANC_AUTH,
            timeout=30
        )
        if response.status_code == 200:
            instances = response.json()
            # Fetch details for each instance
            dicom_files = []
            for instance_uuid in instances[:100]:  # Limit to 100 for performance
                try:
                    detail_response = requests.get(
                        f"{Config.ORTHANC_URL}/instances/{instance_uuid}",
                        auth=Config.ORTHANC_AUTH,
                        timeout=10
                    )
                    if detail_response.status_code == 200:
                        detail = detail_response.json()
                        dicom_files.append({
                            'uuid': instance_uuid,
                            'filename': detail.get('MainDicomTags', {}).get('SOPInstanceUID', instance_uuid),
                            'patient_id': detail.get('MainDicomTags', {}).get('PatientID', 'Unknown'),
                            'modality': detail.get('MainDicomTags', {}).get('Modality', 'DICOM'),
                            'study_date': detail.get('MainDicomTags', {}).get('StudyDate', 'N/A'),
                        })
                except:
                    pass
            return jsonify({"dicom_files": dicom_files})
        return jsonify({"dicom_files": []})
    except Exception as e:
        logger.error(f"Error fetching DICOM files from Orthanc: {e}")
        return jsonify({"dicom_files": []})

@app.route("/api/orthanc-debug")
def orthanc_debug():
    """Debug endpoint to check Orthanc connection and instances"""
    try:
        # Test basic Orthanc connectivity
        response = requests.get(
            f"{Config.ORTHANC_URL}/system",
            auth=Config.ORTHANC_AUTH,
            timeout=10
        )
        if response.status_code == 200:
            system_info = response.json()
            
            # Get instances count
            instances_response = requests.get(
                f"{Config.ORTHANC_URL}/instances",
                auth=Config.ORTHANC_AUTH,
                timeout=10
            )
            instances = instances_response.json() if instances_response.status_code == 200 else []
            
            return jsonify({
                "success": True,
                "orthanc_version": system_info.get("Version", "Unknown"),
                "instance_count": len(instances),
                "instances": instances[:10] if instances else [],
                "message": f"Orthanc is running with {len(instances)} instances"
            })
        return jsonify({"success": False, "error": f"Status {response.status_code}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ─── HL7 TO JSON CONVERSION ROUTES ────────────────────────────────────────────
@app.route("/api/hl7-to-json/<conversion_id>")
def get_hl7_json(conversion_id):
    """Get the HL7 file converted to JSON format"""
    try:
        if conversion_id not in conversion_status:
            return jsonify({"error": "Conversion not found"}), 404
        
        conv_info = conversion_status[conversion_id]
        patient_id = conv_info.get('patient_id')
        filename = conv_info.get('filename')
        
        if not filename or not patient_id:
            return jsonify({"error": "Invalid conversion info"}), 400
        
        file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        # Parse HL7 and convert to JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            hl7_text = f.read()
        
        # Parse HL7 into structured JSON
        hl7_lines = hl7_text.strip().split('\n')
        json_output = {
            "patient_id": patient_id,
            "filename": filename,
            "segments": []
        }
        
        for line in hl7_lines:
            if not line.strip():
                continue
            
            fields = line.split('|')
            segment_type = fields[0].strip() if fields else ""
            
            segment_data = {
                "type": segment_type,
                "fields": fields[1:] if len(fields) > 1 else []
            }
            json_output["segments"].append(segment_data)
        
        return jsonify(json_output)
    
    except Exception as e:
        logger.error(f"Error converting HL7 to JSON: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ─── DICOM TO PNG CONVERSION ROUTES ───────────────────────────────────────────
def apply_windowing(pixel_array, window_center=None, window_width=None):
    """Apply DICOM window/level adjustment to enhance image contrast"""
    try:
        # Use DICOM window settings if available, otherwise auto-detect
        if window_center is None:
            window_center = (pixel_array.max() + pixel_array.min()) / 2
        if window_width is None:
            window_width = pixel_array.max() - pixel_array.min()
        
        # Apply windowing formula
        windowed = np.clip(
            ((pixel_array - window_center) / window_width + 0.5) * 255,
            0, 255
        )
        return windowed.astype(np.uint8)
    except Exception as e:
        logger.warning(f"Windowing failed, using linear scaling: {e}")
        # Fallback to linear scaling
        arr = pixel_array.astype(float)
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max > arr_min:
            return ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        return np.zeros_like(pixel_array, dtype=np.uint8)

@app.route("/api/dicom-to-image/<dicom_filename>")
@app.route("/api/dicom-to-image/<dicom_filename>/<int:frame_number>")
def get_dicom_as_image(dicom_filename, frame_number=0):
    """
    Convert DICOM file to PNG/JPG image with optional frame selection.
    Supports window/level adjustment via query parameters:
    - window_center: center value for windowing
    - window_width: width value for windowing
    - format: output format (png, jpg)
    """
    try:
        file_path = os.path.join(Config.UPLOAD_FOLDER, dicom_filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "DICOM file not found"}), 404
        
        # Read DICOM file
        ds = pydicom.dcmread(file_path)
        
        # Get pixel data
        if not hasattr(ds, 'pixel_array'):
            return jsonify({"error": "DICOM file has no pixel data"}), 400
        
        pixel_array = ds.pixel_array
        
        # Handle multi-frame images
        if len(pixel_array.shape) == 3:
            # Multi-frame image (video/series)
            if frame_number >= pixel_array.shape[0]:
                return jsonify({"error": f"Frame {frame_number} out of range. Total frames: {pixel_array.shape[0]}"}), 400
            pixel_array = pixel_array[frame_number]
        
        # Get window/level settings from request or DICOM file
        window_center = request.args.get('window_center', type=float)
        window_width = request.args.get('window_width', type=float)
        
        # Try to get from DICOM file first
        if window_center is None and hasattr(ds, 'WindowCenter'):
            try:
                window_center = float(ds.WindowCenter) if isinstance(ds.WindowCenter, (list, str)) else ds.WindowCenter
            except:
                pass
        
        if window_width is None and hasattr(ds, 'WindowWidth'):
            try:
                window_width = float(ds.WindowWidth) if isinstance(ds.WindowWidth, (list, str)) else ds.WindowWidth
            except:
                pass
        
        # Apply windowing/normalization
        pixel_array_normalized = apply_windowing(pixel_array, window_center, window_width)
        
        # Convert to PIL Image
        if len(pixel_array_normalized.shape) == 2:
            # Grayscale image
            img = Image.fromarray(pixel_array_normalized, mode='L')
        elif len(pixel_array_normalized.shape) == 3:
            # RGB or multi-channel image
            if pixel_array_normalized.shape[2] == 3:
                img = Image.fromarray(pixel_array_normalized, mode='RGB')
            elif pixel_array_normalized.shape[2] == 4:
                img = Image.fromarray(pixel_array_normalized, mode='RGBA')
            else:
                # Take first channel
                img = Image.fromarray(pixel_array_normalized[:, :, 0], mode='L')
        else:
            return jsonify({"error": "Unsupported pixel array shape"}), 400
        
        # Get output format
        output_format = request.args.get('format', 'png').lower()
        if output_format not in ['png', 'jpg', 'jpeg']:
            output_format = 'png'
        
        # Convert to bytes and return
        img_io = BytesIO()
        save_format = 'JPEG' if output_format in ['jpg', 'jpeg'] else 'PNG'
        img.save(img_io, save_format, quality=95 if save_format == 'JPEG' else None)
        img_io.seek(0)
        
        mimetype = 'image/jpeg' if save_format == 'JPEG' else 'image/png'
        
        return send_file(
            img_io,
            mimetype=mimetype,
            as_attachment=False,
            download_name=f"{dicom_filename}.{output_format}"
        )
    
    except Exception as e:
        logger.error(f"Error converting DICOM to image: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/dicom-info/<dicom_filename>")
def get_dicom_info(dicom_filename):
    """
    Get DICOM metadata without pixel data.
    Returns patient info, modality, acquisition details, etc.
    """
    try:
        file_path = os.path.join(Config.UPLOAD_FOLDER, dicom_filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "DICOM file not found"}), 404
        
        # Read DICOM metadata only
        ds = pydicom.dcmread(file_path, stop_before_pixels=True)
        
        # Extract key metadata
        info = {
            "filename": dicom_filename,
            "patient_id": str(ds.get('PatientID', 'N/A')),
            "patient_name": str(ds.get('PatientName', 'N/A')),
            "modality": str(ds.get('Modality', 'Unknown')),
            "study_description": str(ds.get('StudyDescription', 'N/A')),
            "series_description": str(ds.get('SeriesDescription', 'N/A')),
            "study_date": str(ds.get('StudyDate', 'N/A')),
            "study_time": str(ds.get('StudyTime', 'N/A')),
            "sop_class_uid": str(ds.get('SOPClassUID', 'N/A')),
            "sop_instance_uid": str(ds.get('SOPInstanceUID', 'N/A')),
        }
        
        # Add window/level info if available
        if hasattr(ds, 'WindowCenter'):
            info['window_center'] = float(ds.WindowCenter) if isinstance(ds.WindowCenter, (list, str)) else ds.WindowCenter
        if hasattr(ds, 'WindowWidth'):
            info['window_width'] = float(ds.WindowWidth) if isinstance(ds.WindowWidth, (list, str)) else ds.WindowWidth
        
        # Add number of frames if multi-frame
        try:
            num_frames = ds.get('NumberOfFrames', 1)
            info['number_of_frames'] = int(num_frames)
        except:
            info['number_of_frames'] = 1
        
        # Get image dimensions
        if hasattr(ds, 'pixel_array'):
            arr = ds.pixel_array
            if len(arr.shape) == 2:
                info['width'] = arr.shape[1]
                info['height'] = arr.shape[0]
            elif len(arr.shape) == 3:
                info['frames'] = arr.shape[0]
                info['width'] = arr.shape[2]
                info['height'] = arr.shape[1]
        
        return jsonify({"success": True, "data": info})
    
    except Exception as e:
        logger.error(f"Error getting DICOM info: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ─── ORTHANC-SPECIFIC ENDPOINTS (fetch from server instead of local files) ────
@app.route("/api/orthanc-dicom-image/<instance_uuid>")
@app.route("/api/orthanc-dicom-image/<instance_uuid>/<int:frame_number>")
def get_orthanc_dicom_image(instance_uuid, frame_number=0):
    """
    Fetch DICOM image from Orthanc server and convert to PNG/JPG
    """
    try:
        # Fetch the complete DICOM from Orthanc
        response = requests.get(
            f"{Config.ORTHANC_URL}/instances/{instance_uuid}/file",
            auth=Config.ORTHANC_AUTH,
            timeout=30
        )
        
        if response.status_code != 200:
            return jsonify({"error": f"DICOM not found in Orthanc: {instance_uuid}"}), 404
        
        # Parse DICOM from bytes
        from io import BytesIO
        ds = pydicom.dcmread(BytesIO(response.content))
        
        # Get pixel data
        if not hasattr(ds, 'pixel_array'):
            return jsonify({"error": "DICOM file has no pixel data"}), 400
        
        pixel_array = ds.pixel_array
        
        # Handle multi-frame images
        if len(pixel_array.shape) == 3:
            if frame_number >= pixel_array.shape[0]:
                return jsonify({"error": f"Frame {frame_number} out of range"}), 400
            pixel_array = pixel_array[frame_number]
        
        # Get window/level settings
        window_center = request.args.get('window_center', type=float)
        window_width = request.args.get('window_width', type=float)
        
        if window_center is None and hasattr(ds, 'WindowCenter'):
            try:
                window_center = float(ds.WindowCenter) if isinstance(ds.WindowCenter, (list, str)) else ds.WindowCenter
            except:
                pass
        
        if window_width is None and hasattr(ds, 'WindowWidth'):
            try:
                window_width = float(ds.WindowWidth) if isinstance(ds.WindowWidth, (list, str)) else ds.WindowWidth
            except:
                pass
        
        # Apply windowing
        pixel_array_normalized = apply_windowing(pixel_array, window_center, window_width)
        
        # Convert to PIL Image
        if len(pixel_array_normalized.shape) == 2:
            img = Image.fromarray(pixel_array_normalized, mode='L')
        elif len(pixel_array_normalized.shape) == 3:
            if pixel_array_normalized.shape[2] == 3:
                img = Image.fromarray(pixel_array_normalized, mode='RGB')
            elif pixel_array_normalized.shape[2] == 4:
                img = Image.fromarray(pixel_array_normalized, mode='RGBA')
            else:
                img = Image.fromarray(pixel_array_normalized[:, :, 0], mode='L')
        else:
            return jsonify({"error": "Unsupported pixel array shape"}), 400
        
        # Output format
        output_format = request.args.get('format', 'png').lower()
        if output_format not in ['png', 'jpg', 'jpeg']:
            output_format = 'png'
        
        # Convert to output format
        img_io = BytesIO()
        if output_format in ['jpg', 'jpeg']:
            img.save(img_io, format='JPEG', quality=95)
            mime_type = 'image/jpeg'
        else:
            img.save(img_io, format='PNG')
            mime_type = 'image/png'
        
        img_io.seek(0)
        return send_file(img_io, mimetype=mime_type)
    
    except Exception as e:
        logger.error(f"Error fetching DICOM from Orthanc: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/orthanc-dicom-info/<instance_uuid>")
def get_orthanc_dicom_info(instance_uuid):
    """
    Get DICOM metadata from Orthanc
    """
    try:
        # Fetch instance details from Orthanc
        response = requests.get(
            f"{Config.ORTHANC_URL}/instances/{instance_uuid}",
            auth=Config.ORTHANC_AUTH,
            timeout=30
        )
        
        if response.status_code != 200:
            return jsonify({"success": False, "error": "Instance not found"}), 404
        
        instance_data = response.json()
        tags = instance_data.get('MainDicomTags', {})
        
        info = {
            'patient_id': tags.get('PatientID', 'Unknown'),
            'patient_name': tags.get('PatientName', 'Unknown'),
            'modality': tags.get('Modality', 'DICOM'),
            'study_date': tags.get('StudyDate', 'N/A'),
            'study_time': tags.get('StudyTime', 'N/A'),
            'study_description': tags.get('StudyDescription', 'N/A'),
            'series_number': tags.get('SeriesNumber', 'N/A'),
            'instance_number': tags.get('InstanceNumber', 'N/A'),
            'sop_uid': tags.get('SOPInstanceUID', instance_uuid),
            'series_uid': instance_data.get('ParentSeries', 'N/A'),
            'number_of_frames': 1,
            'uuid': instance_uuid
        }
        
        # Try to fetch full DICOM for additional info
        try:
            file_response = requests.get(
                f"{Config.ORTHANC_URL}/instances/{instance_uuid}/file",
                auth=Config.ORTHANC_AUTH,
                timeout=30
            )
            
            if file_response.status_code == 200:
                from io import BytesIO
                ds = pydicom.dcmread(BytesIO(file_response.content), stop_before_pixels=True)
                
                try:
                    num_frames = ds.get('NumberOfFrames', 1)
                    info['number_of_frames'] = int(num_frames)
                except:
                    pass
                
                if hasattr(ds, 'WindowCenter'):
                    try:
                        info['window_center'] = float(ds.WindowCenter) if isinstance(ds.WindowCenter, (list, str)) else ds.WindowCenter
                    except:
                        pass
                
                if hasattr(ds, 'WindowWidth'):
                    try:
                        info['window_width'] = float(ds.WindowWidth) if isinstance(ds.WindowWidth, (list, str)) else ds.WindowWidth
                    except:
                        pass
        except:
            pass
        
        return jsonify({"success": True, "data": info})
    
    except Exception as e:
        logger.error(f"Error getting Orthanc DICOM info: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/conversion-files/<conversion_id>")
def get_conversion_files(conversion_id):
    """Get list of DICOM and HL7 files for a conversion"""
    try:
        if conversion_id not in conversion_status:
            return jsonify({"error": "Conversion not found"}), 404
        
        conv_info = conversion_status[conversion_id]
        patient_id = conv_info.get('patient_id')
        filename = conv_info.get('filename')
        
        files_info = {
            "patient_id": patient_id,
            "hl7_file": filename,
            "dicom_files": []
        }
        
        # Load patient mapping to find DICOM files by mapped patient ID
        patient_mapping = load_patient_mapping()
        
        # Check if patient_id is directly in mapping
        if patient_id in patient_mapping:
            mapped_files = patient_mapping[patient_id]
            for dicom_path in mapped_files:
                dicom_filename = os.path.basename(dicom_path)
                if os.path.exists(os.path.join(Config.UPLOAD_FOLDER, dicom_filename)):
                    files_info["dicom_files"].append(dicom_filename)
        else:
            # Fallback: look for DICOM files in uploads folder by PatientID match
            for file in os.listdir(Config.UPLOAD_FOLDER):
                if file.lower().endswith('.dcm'):
                    file_path = os.path.join(Config.UPLOAD_FOLDER, file)
                    try:
                        ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                        file_patient_id = ds.get("PatientID", None)
                        if file_patient_id == patient_id:
                            files_info["dicom_files"].append(file)
                    except:
                        pass
        
        return jsonify(files_info)
    
    except Exception as e:
        logger.error(f"Error getting conversion files: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ─── DEMO/TEST ROUTES ─────────────────────────────────────────────────────────
@app.route("/api/populate-demo-data")
def populate_demo_data():
    """Populate conversion history with demo data for testing"""
    try:
        # Find existing HL7 and DICOM files
        demo_conversions = {}
        
        # Get list of HL7 files
        hl7_files = [f for f in os.listdir(Config.UPLOAD_FOLDER) if f.lower().endswith('.hl7')]
        dicom_files = [f for f in os.listdir(Config.UPLOAD_FOLDER) if f.lower().endswith('.dcm')]
        
        # Create demo conversions for existing files
        for hl7_file in hl7_files[:5]:  # Only first 5 for demo
            try:
                # Extract patient ID from filename
                if hl7_file.startswith('patient_'):
                    patient_id = hl7_file.replace('patient_', '').replace('.hl7', '')
                else:
                    patient_id = hl7_file.replace('_complete.hl7', '')
                
                demo_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                conversion_status[demo_id] = {
                    "status": "success",
                    "patient_id": patient_id,
                    "filename": hl7_file,
                    "message": f"✓ Demo conversion loaded from existing file",
                    "timestamp": timestamp
                }
                demo_conversions[demo_id] = patient_id
                logger.info(f"Added demo conversion: {patient_id} from {hl7_file}")
            except Exception as e:
                logger.warning(f"Could not create demo conversion for {hl7_file}: {e}")
                continue
        
        return jsonify({
            "success": True,
            "message": f"Populated {len(demo_conversions)} demo conversions",
            "conversions": demo_conversions,
            "dicom_files": dicom_files,
            "total_conversions": len(conversion_status)
        })
    
    except Exception as e:
        logger.error(f"Error populating demo data: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ─── AI CHAT ROUTES ───────────────────────────────────────────────────────────
# ─── PATIENT DASHBOARD & DICOM ROUTES ──────────────────────────────────────
from healthcare_utils import (
    structure_patient_data, 
    get_dicom_images_for_patient, 
    convert_dicom_to_png
)

@app.route("/patient/<patient_id>")
def patient_dashboard(patient_id):
    """Display comprehensive patient dashboard with vital signs, labs, DICOM images"""
    try:
        # Get patient from FHIR server
        patient_response = requests.get(f"{Config.FHIR_URL}/Patient/{patient_id}", timeout=30)
        
        if patient_response.status_code != 200:
            flash(f"Patient {patient_id} not found", "error")
            return redirect(url_for("view_fhir_resources"))
        
        patient_fhir = patient_response.json()
        
        # Extract basic patient info
        patient_name = 'Unknown'
        if patient_fhir.get('name') and len(patient_fhir['name']) > 0:
            name_obj = patient_fhir['name'][0]
            given = ' '.join(name_obj.get('given', [])) if name_obj.get('given') else ''
            family = name_obj.get('family', '')
            patient_name = f"{given} {family}".strip()
        
        patient_gender = patient_fhir.get('gender', 'Unknown')
        patient_dob = patient_fhir.get('birthDate', 'Unknown')
        patient_phone = 'Unknown'
        patient_email = 'Unknown'
        
        if patient_fhir.get('telecom'):
            for telecom in patient_fhir['telecom']:
                if telecom.get('system') == 'phone':
                    patient_phone = telecom.get('value', 'Unknown')
                elif telecom.get('system') == 'email':
                    patient_email = telecom.get('value', 'Unknown')
        
        # Get observations for this patient
        obs_response = requests.get(f"{Config.FHIR_URL}/Observation?subject=Patient/{patient_id}", timeout=30)
        observations = []
        hl7_text = ''
        if obs_response.status_code == 200:
            obs_data = obs_response.json()
            if 'entry' in obs_data:
                for entry in obs_data['entry']:
                    obs = entry['resource']
                    observations.append(obs)
                    if obs.get('valueString'):
                        hl7_text += obs['valueString'] + '\n'
        
        # Get DICOM images for this patient
        dicom_images = get_dicom_images_for_patient(
            Config.ORTHANC_URL, 
            Config.ORTHANC_AUTH,
            patient_id
        )
        
        # Structure all patient data
        patient_data = structure_patient_data(
            patient_id=patient_id,
            patient_name=patient_name,
            gender=patient_gender,
            dob=patient_dob,
            phone=patient_phone,
            email=patient_email,
            hl7_text=hl7_text,
            dicom_images=dicom_images,
            severity='Unknown'
        )
        
        return render_template("patient_dashboard.html", patient=patient_data)
    
    except Exception as e:
        logger.error(f"Error loading patient dashboard: {str(e)}")
        flash(f"Error loading patient data: {str(e)}", "error")
        return redirect(url_for("view_fhir_resources"))

@app.route("/patient/<patient_id>/clinical-report")
def patient_clinical_report(patient_id):
    """Generate clinical decision support report for a patient"""
    try:
        logger.info(f"Generating clinical report for patient {patient_id}")
        
        # Get patient from FHIR server
        patient_response = requests.get(f"{Config.FHIR_URL}/Patient/{patient_id}", timeout=30)
        
        if patient_response.status_code != 200:
            logger.warning(f"Patient {patient_id} not found in FHIR server")
            return jsonify({
                'error': f'Patient {patient_id} not found',
                'success': False
            }), 404
        
        # Get all observations for this patient
        obs_response = requests.get(f"{Config.FHIR_URL}/Observation?subject=Patient/{patient_id}", timeout=30)
        
        vitals = {}
        labs = {}
        observations = []
        imaging_findings = ''
        
        logger.info(f"Observations response status: {obs_response.status_code}")
        
        if obs_response.status_code == 200:
            obs_data = obs_response.json()
            if 'entry' in obs_data:
                logger.info(f"Found {len(obs_data['entry'])} observations")
                for entry in obs_data['entry']:
                    obs = entry.get('resource', {})
                    observations.append(obs)
                    
                    # Extract vital signs
                    code = obs.get('code', {}).get('coding', [{}])[0]
                    code_value = code.get('code', '')
                    display = code.get('display', '')
                    
                    # Try to extract numeric value
                    value = None
                    try:
                        if 'valueQuantity' in obs:
                            value = obs['valueQuantity'].get('value')
                        elif 'valueString' in obs:
                            # Try to parse from string
                            val_str = obs['valueString']
                            if val_str and isinstance(val_str, str):
                                try:
                                    # Extract first number from string
                                    import re
                                    numbers = re.findall(r'\d+\.?\d*', val_str)
                                    if numbers:
                                        value = float(numbers[0])
                                except:
                                    pass
                    except Exception as ve:
                        logger.debug(f"Could not extract value from observation: {str(ve)}")
                    
                    # Map to vital signs or labs based on LOINC code
                    if code_value == '8867-4' and value is not None:  # Heart rate
                        vitals['heart_rate'] = value
                        logger.debug(f"Extracted heart_rate: {value}")
                    elif code_value == '8480-6' and value is not None:  # Systolic BP
                        vitals['systolic'] = value
                        logger.debug(f"Extracted systolic: {value}")
                    elif code_value == '8462-4' and value is not None:  # Diastolic BP
                        vitals['diastolic'] = value
                        logger.debug(f"Extracted diastolic: {value}")
                    elif code_value == '2708-6' and value is not None:  # O2 sat
                        vitals['oxygen_saturation'] = value
                        logger.debug(f"Extracted oxygen_saturation: {value}")
                    elif code_value == '8310-5' and value is not None:  # Temperature
                        vitals['temperature'] = value
                        logger.debug(f"Extracted temperature: {value}")
                    elif code_value == '6690-2' and value is not None:  # WBC
                        labs['wbc'] = value
                        logger.debug(f"Extracted wbc: {value}")
                    elif code_value == '718-7' and value is not None:  # Hemoglobin
                        labs['hemoglobin'] = value
                        logger.debug(f"Extracted hemoglobin: {value}")
                    elif code_value == '777-3' and value is not None:  # Platelets
                        labs['platelets'] = value
                        logger.debug(f"Extracted platelets: {value}")
                    elif code_value == '1988-5' and value is not None:  # CRP
                        labs['crp'] = value
                        logger.debug(f"Extracted crp: {value}")
                    elif code_value == '33959-8' and value is not None:  # Procalcitonin
                        labs['procalcitonin'] = value
                        logger.debug(f"Extracted procalcitonin: {value}")
        
        # Try to get diagnostic report for imaging findings
        diag_response = requests.get(f"{Config.FHIR_URL}/DiagnosticReport?subject=Patient/{patient_id}", timeout=30)
        if diag_response.status_code == 200:
            diag_data = diag_response.json()
            if 'entry' in diag_data and len(diag_data['entry']) > 0:
                dr = diag_data['entry'][0].get('resource', {})
                imaging_findings = dr.get('conclusion', 'No findings documented')
                logger.info(f"Imaging findings: {imaging_findings[:100]}")
        
        # Prepare patient data for clinical interpretation
        patient_data = {
            'vital_signs': vitals,
            'lab_results': labs,
            'imaging_findings': imaging_findings,
            'symptoms': [],
            'diagnoses': [],
            'observations': observations
        }
        
        logger.info(f"Patient data prepared - vitals: {vitals}, labs: {labs}")
        
        # Generate clinical interpretation
        report_data = generate_clinical_interpretation(patient_data)
        
        logger.info(f"Clinical report generated - Severity: {report_data.get('severity_level')}, Urgency: {report_data.get('urgency_level')}")
        
        # Return as JSON
        return jsonify({
            'success': True,
            'patient_id': patient_id,
            'report': report_data,
            'raw_data': {
                'vitals': vitals,
                'labs': labs,
                'imaging': imaging_findings
            }
        })
    
    except Exception as e:
        logger.error(f"Error generating clinical report: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'Error generating report: {str(e)}',
            'success': False
        }), 500

@app.route("/get-dicom-image/<study_uid>/<series_uid>/<instance_uid>")
def get_dicom_image(study_uid, series_uid, instance_uid):
    """Retrieve DICOM image from Orthanc and convert to PNG"""
    try:
        # First, search for the instance by UID
        instances_url = f"{Config.ORTHANC_URL}/api/instances"
        instances_response = requests.get(instances_url, auth=HTTPBasicAuth(*Config.ORTHANC_AUTH))
        
        instance_id = None
        if instances_response.status_code == 200:
            instance_ids = instances_response.json()
            for iid in instance_ids:
                inst_url = f"{Config.ORTHANC_URL}/api/instances/{iid}"
                inst_response = requests.get(inst_url, auth=HTTPBasicAuth(*Config.ORTHANC_AUTH))
                if inst_response.status_code == 200:
                    inst_data = inst_response.json()
                    if inst_data.get('MainDicomTags', {}).get('SOPInstanceUID') == instance_uid:
                        instance_id = iid
                        break
        
        if not instance_id:
            instance_id = instance_uid
        
        # Get PNG preview from Orthanc
        preview_url = f"{Config.ORTHANC_URL}/api/instances/{instance_id}/preview"
        response = requests.get(preview_url, auth=HTTPBasicAuth(*Config.ORTHANC_AUTH), timeout=30)
        
        if response.status_code == 200:
            return send_file(
                BytesIO(response.content),
                mimetype='image/png',
                as_attachment=False
            )
        else:
            logger.error(f"Failed to get DICOM preview: {response.status_code}")
            from PIL import Image, ImageDraw
            img = Image.new('RGB', (512, 512), color='white')
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), 'DICOM Image Unavailable', fill='gray')
            img_bytes = BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            return send_file(img_bytes, mimetype='image/png')
    
    except Exception as e:
        logger.error(f"Error retrieving DICOM image: {str(e)}")
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (512, 512), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f'Error: {str(e)[:50]}', fill='red')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return send_file(img_bytes, mimetype='image/png')

# ─── AI CHAT ROUTES ───────────────────────────────────────────────────────────
@app.route("/ai-chat-page")
def ai_chat_page():
    """Render AI chat interface"""
    try:
        if not test_fhir_server_connection():
            flash("FHIR server is not accessible", "error")
            return render_template("ai_chat.html", patients=[])
        
        # Get all patients from FHIR server
        patients_response = requests.get(f"{Config.FHIR_URL}/Patient", timeout=30)
        patients = []
        if patients_response.status_code == 200:
            patients_data = patients_response.json()
            if 'entry' in patients_data:
                for entry in patients_data['entry']:
                    patient = entry['resource']
                    patient_info = {
                        'id': patient.get('id'),
                        'name': patient.get('name', [{}])[0].get('text', 'Unknown') if patient.get('name') else 'Unknown'
                    }
                    patients.append(patient_info)
        
        return render_template("ai_chat.html", patients=patients)
        
    except Exception as e:
        logger.error(f"Error loading AI chat page: {str(e)}")
        flash(f"Error loading AI chat: {str(e)}", "error")
        return render_template("ai_chat.html", patients=[])


@app.route('/analyze-image', methods=['POST'])
def analyze_image_route():
    """Accept an uploaded image (scan/xray/report) and return a simple automated analysis.
    This is a lightweight stub — replace `analyze_image` with a real ML model or external service.
    """
    try:
        patient_id = request.form.get('patient_id')
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})

        f = request.files['image']
        if f.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'})

        upload_dir = os.path.join('uploads', 'report_images')
        os.makedirs(upload_dir, exist_ok=True)
        fname = secure_filename(f.filename)
        # prefix with patient id and timestamp for uniqueness
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        saved_name = f"{patient_id or 'unknown'}_{timestamp}_{fname}"
        saved_path = os.path.join(upload_dir, saved_name)
        f.save(saved_path)

        analysis = analyze_image(saved_path, patient_id)
        # analysis expected to be a dict and include a human-readable text
        return jsonify({'success': True, 'analysis': analysis, 'analysis_text': analysis.get('analysis_text')})

    except Exception as e:
        logger.error(f"Image analysis error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


def analyze_image(file_path, patient_id=None):
    """Simple deterministic image analysis stub.
    Returns a dict with keys: finding, impression, recommendation, confidence, analysis_text
    Replace this with a call to a medical imaging model or an OCR+NLP pipeline for report images.
    """
    # Basic heuristic based on filename — placeholder for real model
    name = os.path.basename(file_path).lower()
    finding = 'No acute abnormality detected'
    impression = 'Image requires clinician review for formal interpretation.'
    recommendation = 'Clinical correlation recommended; if symptoms severe, obtain specialist review.'
    confidence = 0.45

    if any(k in name for k in ['xray', 'chest', 'cxr', 'cxr', 'pneumo']):
        finding = 'Patchy airspace opacities in the right lower zone (possible consolidation)'
        impression = 'Findings suggest focal consolidation; consider pneumonia or inflammatory process.'
        recommendation = 'Recommend chest radiograph correlation with clinical signs, consider antibiotic therapy if infection suspected.'
        confidence = 0.78
    elif any(k in name for k in ['ct', 'chestct', 'ctthorax']):
        finding = 'Interstitial markings with small subpleural nodules'
        impression = 'Consider interstitial lung disease vs. infectious/inflammatory causes.'
        recommendation = 'Consider high-resolution CT review by radiologist and correlate with labs.'
        confidence = 0.72
    elif any(k in name for k in ['report', 'ocr', 'scan']):
        finding = 'Textual report image uploaded; perform OCR for exact content'
        impression = 'Extracted text not available in this stub; integrate OCR for detailed parsing.'
        recommendation = 'Enable OCR pipeline (tesseract/pytesseract) to parse report text.'
        confidence = 0.5

    analysis_text = (
        f"Automated image review for patient {patient_id or 'N/A'}:\n"
        f"Finding: {finding}\n"
        f"Impression: {impression}\n"
        f"Recommendation: {recommendation}\n"
        f"Confidence: {int(confidence*100)}%"
    )

    return {
        'finding': finding,
        'impression': impression,
        'recommendation': recommendation,
        'confidence': confidence,
        'analysis_text': analysis_text
    }

@app.route("/get-dicom-images/<patient_id>")
def get_dicom_images(patient_id):
    """Fetch DICOM images for a patient from Orthanc"""
    try:
        # First check local mapped DICOMs in uploads/dcm_by_patient
        local_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'dcm_by_patient', patient_id)
        if os.path.isdir(local_dir):
            images = []
            for fn in sorted(os.listdir(local_dir))[:12]:
                # safe filename
                if '..' in fn or fn.startswith('.'):
                    continue
                images.append({
                    'instance_id': fn,
                    'series_id': fn,
                    'modality': 'DICOM',
                    'description': fn,
                    'image_url': f"/local-dicom-preview/{patient_id}/{fn}"
                })
            return jsonify({'success': True, 'images': images})

        # Query Orthanc for DICOM files
        # List all patients in Orthanc
        orthanc_url = f"{Config.ORTHANC_URL}/patients"
        response = requests.get(orthanc_url, auth=Config.ORTHANC_AUTH, timeout=30)
        
        dicom_images = []
        
        if response.status_code == 200:
            patients = response.json()
            
            # Search for patient in Orthanc
            for orthanc_patient_id in patients:
                # Get patient details
                patient_details_url = f"{Config.ORTHANC_URL}/patients/{orthanc_patient_id}"
                patient_response = requests.get(patient_details_url, auth=Config.ORTHANC_AUTH, timeout=30)
                
                if patient_response.status_code == 200:
                    patient_details = patient_response.json()
                    patient_main_dicom = patient_details.get('MainDicomTags', {})
                    
                    # Check if this matches our patient
                    orthanc_patient_name = patient_main_dicom.get('PatientName', '')
                    orthanc_patient_id_tag = patient_main_dicom.get('PatientID', '')
                    
                    # If matches (simplified matching), get studies
                    if patient_id.lower() in orthanc_patient_id_tag.lower() or patient_id.lower() in orthanc_patient_name.lower():
                        studies = patient_details.get('Studies', [])
                        
                        for study_id in studies:
                            study_url = f"{Config.ORTHANC_URL}/studies/{study_id}"
                            study_response = requests.get(study_url, auth=Config.ORTHANC_AUTH, timeout=30)
                            
                            if study_response.status_code == 200:
                                study_details = study_response.json()
                                series_list = study_details.get('Series', [])
                                
                                for series_id in series_list:
                                    series_url = f"{Config.ORTHANC_URL}/series/{series_id}"
                                    series_response = requests.get(series_url, auth=Config.ORTHANC_AUTH, timeout=30)
                                    
                                    if series_response.status_code == 200:
                                        series_details = series_response.json()
                                        instances = series_details.get('Instances', [])
                                        main_dicom = series_details.get('MainDicomTags', {})
                                        
                                        for instance_id in instances[:3]:  # Limit to 3 images per series
                                            dicom_images.append({
                                                'instance_id': instance_id,
                                                'series_id': series_id,
                                                'modality': main_dicom.get('Modality', 'Unknown'),
                                                'description': main_dicom.get('SeriesDescription', 'DICOM Image'),
                                                'image_url': f"/orthanc-image/{instance_id}"
                                            })
        
        return jsonify({'success': True, 'images': dicom_images[:12]})  # Limit to 12 images
        
    except Exception as e:
        logger.warning(f"Error fetching DICOM images: {str(e)}")
        return jsonify({'success': False, 'images': [], 'error': str(e)})

@app.route("/orthanc-image/<instance_id>")
def get_orthanc_image(instance_id):
    """Get DICOM image preview from Orthanc"""
    try:
        image_url = f"{Config.ORTHANC_URL}/api/instances/{instance_id}/preview"
        response = requests.get(image_url, auth=Config.ORTHANC_AUTH, timeout=30)
        
        if response.status_code == 200:
            return response.content, 200, {'Content-Type': 'image/png'}
        else:
            # Fallback to PNG if preview not available
            image_url = f"{Config.ORTHANC_URL}/api/instances/{instance_id}/rendered"
            response = requests.get(image_url, auth=Config.ORTHANC_AUTH, timeout=30)
            return response.content, 200, {'Content-Type': 'image/png'}
    except Exception as e:
        logger.error(f"Error getting DICOM image: {str(e)}")
        return jsonify({'error': 'Image not available'}), 404


@app.route('/local-dicom-preview/<patient_id>/<filename>')
def local_dicom_preview(patient_id, filename):
    """Render a local DICOM file from uploads/dcm_by_patient/<patient_id>/filename to PNG"""
    try:
        # sanitize
        if '..' in filename or filename.startswith('.'):
            return jsonify({'error': 'Invalid filename'}), 400
        path = os.path.join(app.config['UPLOAD_FOLDER'], 'dcm_by_patient', patient_id, filename)
        if not os.path.exists(path):
            return jsonify({'error': 'File not found'}), 404

        ds = pydicom.dcmread(path, force=True)
        arr = None
        try:
            arr = ds.pixel_array
        except Exception:
            # no pixel data
            return jsonify({'error': 'No pixel data in DICOM'}), 404

        # normalize to uint8
        img = arr
        if img.dtype != np.uint8:
            mn = img.min()
            mx = img.max()
            if mx - mn == 0:
                img8 = (img - mn).astype('uint8')
            else:
                img8 = ((img - mn) / (mx - mn) * 255.0).astype('uint8')
        else:
            img8 = img

        # If multi-channel or 2D, handle
        if img8.ndim == 3 and img8.shape[0] in (3,4):
            # channels first
            img8 = np.transpose(img8, (1,2,0))

        pil = Image.fromarray(img8)
        buf = BytesIO()
        pil.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        logger.error(f"Error rendering local DICOM: {e}")
        return jsonify({'error': str(e)}), 500

@app.route("/ai-chat", methods=["POST"])
@app.route("/api/ai-query", methods=["POST"])
def ai_chat():
    """Handle AI chat messages with structured clinical response formatting"""
    try:
        data = request.get_json()
        patient_id = data.get('patient_id')
        # Support both 'question' and 'message' field names for compatibility
        question = data.get('question') or data.get('message')
        history = data.get('history', [])
        dicom_uuid = data.get('dicom_uuid')
        
        if not patient_id or not question:
            return jsonify({'success': False, 'error': 'Missing patient_id or question'})
        
        # Fetch patient and related data from FHIR server
        patient_response = requests.get(f"{Config.FHIR_URL}/Patient/{patient_id}", timeout=30)
        if patient_response.status_code != 200:
            return jsonify({'success': False, 'error': f'Patient not found: {patient_id}'})
        
        patient_data = patient_response.json()
        
        # Get patient's diagnostic reports
        reports_response = requests.get(
            f"{Config.FHIR_URL}/DiagnosticReport?subject=Patient/{patient_id}",
            timeout=30
        )
        reports = []
        if reports_response.status_code == 200:
            reports_data = reports_response.json()
            if 'entry' in reports_data:
                reports = [entry['resource'] for entry in reports_data['entry']]
        
        # Get patient's observations
        obs_response = requests.get(
            f"{Config.FHIR_URL}/Observation?subject=Patient/{patient_id}",
            timeout=30
        )
        observations = []
        if obs_response.status_code == 200:
            obs_data = obs_response.json()
            if 'entry' in obs_data:
                observations = [entry['resource'] for entry in obs_data['entry']]
        
        # Try to load enhanced patient data
        enhanced_data = {}
        try:
            enhanced_file = os.path.join("uploads", f"patient_enhanced_{patient_id}.json")
            if os.path.exists(enhanced_file):
                with open(enhanced_file, 'r') as f:
                    enhanced_data = json.load(f).get('medicalRecord', {})
        except:
            pass
        
        # Try to load lab data
        lab_data = {}
        try:
            lab_file = os.path.join("uploads", f"lab_report_{patient_id}.json")
            if os.path.exists(lab_file):
                with open(lab_file, 'r') as f:
                    lab_data = json.load(f)
        except:
            pass
        
        # Detect question type and generate appropriate structured response
        question_lower = question.lower()
        ai_response = generate_structured_response(
            question_lower,
            patient_data,
            enhanced_data,
            reports,
            observations,
            lab_data,
            patient_id
        )
        
        return jsonify({'success': True, 'response': ai_response, 'is_structured': True})
        
    except Exception as e:
        logger.error(f"Error in AI chat: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


def generate_structured_response(
    question_lower: str,
    patient_data: Dict,
    enhanced_data: Dict,
    reports: List[Dict],
    observations: List[Dict],
    lab_data: Dict,
    patient_id: str
) -> str:
    """
    Generate a properly formatted structured clinical response based on question type
    
    Uses ClinicalResponseFormatter to ensure consistent formatting
    """
    
    formatter = ClinicalResponseFormatter()
    
    # Determine response type and generate accordingly
    
    # COMPLETE PATIENT PROFILE
    if any(word in question_lower for word in ['complete', 'full', 'summary', 'overview', 'all information', 'patient profile']):
        return formatter.format_complete_patient_report(
            patient_data,
            enhanced_data,
            reports,
            include_medications=True
        )['formatted_response']
    
    # JUST PATIENT INFO
    elif any(word in question_lower for word in ['patient info', 'demographics', 'personal', 'name', 'age', 'id', 'age group']):
        if any(word in question_lower for word in ['full', 'complete']):
            include_med = True
        else:
            include_med = False
        
        sections = [formatter.format_patient_info(patient_data, enhanced_data)]
        if include_med:
            sections.append(formatter.format_diagnosis(enhanced_data, reports))
        return "".join(sections)
    
    # DIAGNOSIS & CONDITION
    elif any(word in question_lower for word in ['diagnos', 'condition', 'illness', 'disease', 'problem', 'complaint', 'primary']):
        return format_clinical_response(
            patient_data,
            enhanced_data,
            reports,
            include_medications=True,
            include_labs=True
        )
    
    # FINDINGS & IMAGING
    elif any(word in question_lower for word in ['finding', 'report', 'imaging', 'image', 'scan', 'x-ray', 'ct', 'mri', 'dicom', 'modality']):
        sections = [
            formatter.format_patient_info(patient_data, enhanced_data),
            formatter.format_diagnosis(enhanced_data, reports),
            formatter.format_findings(reports, enhanced_data)
        ]
        return "".join(sections)
    
    # MEDICATIONS
    elif any(word in question_lower for word in ['medication', 'medicine', 'drug', 'prescri', 'therapy', 'treatment', 'drugs']):
        sections = [
            formatter.format_patient_info(patient_data, enhanced_data),
            formatter.format_medications(enhanced_data)
        ]
        return "".join(sections)
    
    # LAB RESULTS & VITALS
    elif any(word in question_lower for word in ['vital', 'lab', 'result', 'test', 'value', 'measurement', 'blood', 'oxygen', 'heart rate']):
        sections = [
            formatter.format_patient_info(patient_data, enhanced_data),
        ]
        if lab_data.get('result'):
            sections.append(formatter.format_lab_results(lab_data))
        else:
            sections.append("\n⚠️ No laboratory results available\n")
        return "".join(sections)
    
    # RISK & CONCERNS
    elif any(word in question_lower for word in ['risk', 'concern', 'danger', 'warning', 'alert', 'critical', 'severe', 'abnormal', 'anomal']):
        sections = [
            formatter.format_patient_info(patient_data, enhanced_data),
            formatter.format_risk_assessment(enhanced_data, lab_data)
        ]
        return "".join(sections)
    
    # DEFAULT: Generate full structured response
    else:
        return format_clinical_response(
            patient_data,
            enhanced_data,
            reports,
            lab_data,
            include_medications=True,
            include_labs=True
        )


def generate_fallback_response(question, patient_data, reports, observations):
    """Generate comprehensive responses with anomaly detection and treatment recommendations"""
    question_lower = question.lower()
    
    # Extract patient information
    patient_name = patient_data.get('name', [{}])[0].get('text', 'the patient') if patient_data.get('name') else 'the patient'
    patient_id = patient_data.get('id', 'Unknown')
    dob = patient_data.get('birthDate', 'Not available')
    gender = patient_data.get('gender', 'Not specified')
    age = patient_data.get('age', 'Not specified')
    
    # Try to load enhanced patient data for anomalies and treatment
    enhanced_data = {}
    try:
        enhanced_file = os.path.join("uploads", f"patient_enhanced_{patient_id}.json")
        if os.path.exists(enhanced_file):
            with open(enhanced_file, 'r') as f:
                enhanced_data = json.load(f).get('medicalRecord', {})
    except:
        pass
    
    # Try to load lab report data
    lab_data = {}
    try:
        lab_file = os.path.join("uploads", f"lab_report_{patient_id}.json")
        if os.path.exists(lab_file):
            with open(lab_file, 'r') as f:
                lab_data = json.load(f)
    except:
        pass
    
    # Initialize response with context
    response = ""
    
    # ANOMALIES & RISK ASSESSMENT
    if any(word in question_lower for word in ['anomal', 'risk', 'concern', 'danger', 'problem', 'abnormal', 'critical']):
        response = f"**Anomaly & Risk Assessment for {patient_name} (Age: {age}, {gender}):**\n\n"
        
        # Risk level
        risk_level = enhanced_data.get('riskLevel', 'Not specified')
        response += f"🚨 **Risk Level:** {risk_level}\n\n"
        
        # Anomalies detected
        anomalies = enhanced_data.get('anomalies', [])
        if anomalies:
            response += "**Detected Anomalies:**\n"
            for anomaly in anomalies:
                response += f"• {anomaly}\n"
            response += "\n"
        
        # Lab abnormalities
        if lab_data.get('result'):
            response += "**Abnormal Lab Results:**\n"
            for result in lab_data['result']:
                if result.get('status') != 'Normal':
                    response += f"• {result.get('test', 'Test')}: {result.get('value')} {result.get('unit')} (Normal: {result.get('normalRange')})\n"
            response += "\n"
        
        response += "**Clinical Assessment:**\n"
        if risk_level in ['High', 'Very High']:
            response += f"⚠️ This patient is at {risk_level} risk and requires:\n"
            response += "- Close monitoring and frequent follow-up\n"
            response += "- Possible hospital admission\n"
            response += "- Specialist consultation\n"
            response += "- Aggressive intervention if condition worsens\n"
        elif risk_level == 'Medium':
            response += "⚠️ This patient is at medium risk and requires:\n"
            response += "- Regular monitoring\n"
            response += "- Follow-up in 1-2 weeks\n"
            response += "- Possible specialist referral\n"
        else:
            response += "✓ This patient is at low risk but should maintain:\n"
            response += "- Regular check-ups\n"
            response += "- Conservative management\n"
            response += "- Symptom monitoring\n"
    
    # FINDINGS & REPORTS
    elif any(word in question_lower for word in ['finding', 'report', 'imaging', 'image', 'scan', 'x-ray', 'ct', 'mri']):
        response = f"**Findings & Imaging for {patient_name}:**\n\n"
        
        diagnosis = enhanced_data.get('diagnosis', 'Not specified')
        response += f"**Diagnosis:** {diagnosis}\n"
        response += f"**Modality:** {enhanced_data.get('modality', 'Not specified')}\n\n"
        
        if reports:
            for i, report in enumerate(reports[:3], 1):
                code = report.get('code', {}).get('text', 'Medical Report')
                conclusion = report.get('conclusion', 'No conclusion noted')
                status = report.get('status', 'Unknown')
                effective_date = report.get('effectiveDateTime', 'Date unknown')
                response += f"{i}. **{code}** (Status: {status})\n"
                response += f"   Date: {effective_date}\n"
                response += f"   Conclusion: {conclusion}\n\n"
        else:
            response += "No diagnostic imaging reports are currently available in the medical records.\n"
    
    # DIAGNOSIS
    elif any(word in question_lower for word in ['diagnos', 'condition', 'illness', 'disease', 'problem', 'complaint']):
        response = f"**Diagnoses & Conditions for {patient_name} (Age: {age}):**\n\n"
        
        diagnosis = enhanced_data.get('diagnosis', 'Not specified')
        response += f"**Primary Diagnosis:** {diagnosis}\n"
        response += f"**Risk Level:** {enhanced_data.get('riskLevel', 'Not specified')}\n\n"
        
        if reports:
            response += "**Diagnostic Reports:**\n"
            for report in reports[:3]:
                code = report.get('code', {}).get('text', 'Condition')
                conclusion = report.get('conclusion', '')
                if conclusion:
                    response += f"- {code}: {conclusion}\n"
        if observations:
            response += "\n**Clinical Observations:**\n"
            for obs in observations[:3]:
                code = obs.get('code', {}).get('text', 'Finding')
                value = obs.get('valueQuantity', {}).get('value', 'N/A')
                unit = obs.get('valueQuantity', {}).get('unit', '')
                response += f"- {code}: {value} {unit}\n"
        
        # Add anomalies if present
        anomalies = enhanced_data.get('anomalies', [])
        if anomalies:
            response += "\n**Associated Anomalies:**\n"
            for anomaly in anomalies:
                response += f"• {anomaly}\n"
        
        if not reports and not observations and not anomalies:
            response += "No specific diagnostic information is currently available.\n"
    
    # TREATMENT & THERAPY
    elif any(word in question_lower for word in ['treatment', 'therapy', 'medication', 'medicine', 'drug', 'prescri', 'manage', 'interven']):
        response = f"**Treatment & Medication Plan for {patient_name}:**\n\n"
        
        treatment_plan = enhanced_data.get('treatmentPlan', [])
        medications = enhanced_data.get('medications', [])
        
        if treatment_plan:
            response += "**Treatment Plan:**\n"
            for treatment in treatment_plan:
                response += f"• {treatment}\n"
            response += "\n"
        
        if medications:
            response += "**Prescribed Medications:**\n"
            for med in medications:
                response += f"• {med}\n"
            response += "\n"
        
        if reports:
            response += "**Clinical Recommendations:**\n"
            for report in reports[:2]:
                conclusion = report.get('conclusion', '')
                if conclusion:
                    response += f"- {conclusion}\n"
        
        response += "\n**Important Notes:**\n"
        response += "⚠️ **This is NOT a prescription.** All treatment decisions must be made by the treating physician.\n"
        response += "- Medications should be individualized for the patient\n"
        response += "- Follow-up appointments are critical for monitoring\n"
        response += "- Report side effects immediately\n"
        response += "- Consult with specialist if condition does not improve\n"

    # VITAL SIGNS & OBSERVATIONS
    elif any(word in question_lower for word in ['vital', 'observation', 'test', 'lab', 'result', 'value', 'measurement']):
        response = f"**Lab Results & Test Values for {patient_name}:**\n\n"
        
        # Display lab data if available
        if lab_data.get('result'):
            response += "**Laboratory Tests:**\n"
            abnormal_count = 0
            for result in lab_data['result']:
                test_name = result.get('test', 'Test')
                value = result.get('value', 'N/A')
                unit = result.get('unit', '')
                normal_range = result.get('normalRange', '')
                status = result.get('status', 'Normal')
                
                if status != 'Normal':
                    response += f"⚠️ **{test_name}**: {value} {unit}\n"
                    abnormal_count += 1
                else:
                    response += f"✓ **{test_name}**: {value} {unit}\n"
                
                if normal_range:
                    response += f"   Reference Range: {normal_range}\n"
                response += f"   Status: {status}\n"
            
            if abnormal_count > 0:
                response += f"\n**Summary:** {abnormal_count} abnormal result(s) detected - requires clinical attention\n"
        
        elif observations:
            response += "**Vital Signs & Observations:**\n"
            for obs in observations[:10]:
                code = obs.get('code', {}).get('text', 'Test')
                value = obs.get('valueQuantity', {}).get('value', 'N/A')
                unit = obs.get('valueQuantity', {}).get('unit', '')
                ref_range = obs.get('referenceRange', [{}])[0].get('text', '')
                status = obs.get('status', 'unknown')
                
                response += f"- **{code}**: {value} {unit}\n"
                if ref_range:
                    response += f"  Reference Range: {ref_range}\n"
                response += f"  Status: {status}\n"
        else:
            response += "No observation data currently available in the system.\n"
    
    # MEDICAL HISTORY
    elif any(word in question_lower for word in ['history', 'previous', 'past', 'before', 'prior', 'background', 'chronic']):
        response = f"**Medical History & Background for {patient_name}:**\n\n"
        response += f"**Demographics:**\n"
        response += f"- Patient ID: {patient_id}\n"
        response += f"- Name: {patient_name}\n"
        response += f"- Father's Name: {enhanced_data.get('fatherName', 'Not specified')}\n"
        response += f"- Date of Birth: {dob}\n"
        response += f"- Age: {age}\n"
        response += f"- Gender: {gender}\n"
        response += f"- Mobile: {enhanced_data.get('mobileNumber', 'Not specified')}\n"
        response += f"- Email: {enhanced_data.get('email', 'Not specified')}\n"
        response += f"- Address: {enhanced_data.get('address', 'Not specified')}\n\n"
        
        response += f"**Current Condition:**\n"
        response += f"- Diagnosis: {enhanced_data.get('diagnosis', 'Not specified')}\n"
        response += f"- Risk Level: {enhanced_data.get('riskLevel', 'Not specified')}\n\n"
        
        if reports:
            response += "**Medical Records:**\n"
            for report in reports:
                code = report.get('code', {}).get('text', 'Report')
                date = report.get('effectiveDateTime', 'Date unknown')
                response += f"- {code} ({date})\n"
        
        if observations:
            response += "\n**Historical Test Results:**\n"
            for obs in observations[:5]:
                code = obs.get('code', {}).get('text', 'Test')
                date = obs.get('effectiveDateTime', 'Date unknown')
                value = obs.get('valueQuantity', {}).get('value', '')
                response += f"- {code}: {value} ({date})\n"
    
    # CONCERNS & RISKS
    elif any(word in question_lower for word in ['concern', 'risk', 'danger', 'warning', 'alert', 'critical', 'severe', 'abnormal']):
        response = f"**Risk Assessment & Clinical Concerns for {patient_name}:**\n\n"
        
        risk_level = enhanced_data.get('riskLevel', 'Not specified')
        response += f"🚨 **Overall Risk Level:** {risk_level}\n\n"
        
        response += "**Identified Issues:**\n"
        abnormal_found = False
        
        # Check for abnormal lab results
        if lab_data.get('result'):
            for result in lab_data['result']:
                if result.get('status') != 'Normal':
                    response += f"⚠️ {result.get('test', 'Test')}: {result.get('value')} (Status: {result.get('status')})\n"
                    abnormal_found = True
        
        # Check for anomalies in enhanced data
        anomalies = enhanced_data.get('anomalies', [])
        if anomalies:
            for anomaly in anomalies:
                response += f"• {anomaly}\n"
                abnormal_found = True
        
        # Check observations
        if observations:
            for obs in observations[:5]:
                status = obs.get('status', '').lower()
                if 'abnormal' in status or 'critical' in status or 'high' in status:
                    code = obs.get('code', {}).get('text', 'Finding')
                    value = obs.get('valueQuantity', {}).get('value', '')
                    response += f"⚠️ {code}: {value} (Status: {status})\n"
                    abnormal_found = True
        
        if not abnormal_found:
            response += "- No critical abnormalities detected in current records\n"
        
        response += f"\n**Recommendations based on {risk_level} Risk:**\n"
        if risk_level in ['High', 'Very High']:
            response += "🔴 URGENT: This patient requires:\n"
            response += "  • Immediate specialist consultation\n"
            response += "  • Frequent monitoring (daily/weekly)\n"
            response += "  • Possible hospital admission for observation\n"
            response += "  • Aggressive treatment intervention\n"
            response += "  • Follow-up within 24-48 hours\n"
        elif risk_level == 'Medium':
            response += "🟡 MODERATE: This patient requires:\n"
            response += "  • Specialist consultation within 1 week\n"
            response += "  • Regular monitoring (weekly)\n"
            response += "  • Follow-up within 7-14 days\n"
            response += "  • May need intervention depending on progression\n"
        else:
            response += "🟢 LOW: This patient should maintain:\n"
            response += "  • Regular check-ups every 3-6 months\n"
            response += "  • Conservative management\n"
            response += "  • Monitor for symptom changes\n"
            response += "  • Follow routine prevention measures\n"
    
    # PATIENT DEMOGRAPHICS
    elif any(word in question_lower for word in ['age', 'born', 'gender', 'name', 'male', 'female', 'patient info', 'demographic', 'contact']):
        response = f"**Patient Information & Demographics:**\n\n"
        response += f"**Personal Information:**\n"
        response += f"- Name: {patient_name}\n"
        response += f"- Patient ID: {patient_id}\n"
        response += f"- Father's Name: {enhanced_data.get('fatherName', 'Not specified')}\n"
        response += f"- Date of Birth: {dob}\n"
        response += f"- Age: {age} years\n"
        response += f"- Gender: {gender}\n\n"
        
        response += f"**Contact Information:**\n"
        response += f"- Mobile Number: {enhanced_data.get('mobileNumber', 'Not specified')}\n"
        response += f"- Email: {enhanced_data.get('email', 'Not specified')}\n"
        response += f"- Address: {enhanced_data.get('address', 'Not specified')}\n\n"
        
        response += f"**Medical Summary:**\n"
        response += f"- Total Diagnostic Reports: {len(reports)}\n"
        response += f"- Total Test Results: {len(observations)}\n"
        response += f"- Lab Tests Performed: {len(lab_data.get('result', []))}\n"
        response += f"- Risk Level: {enhanced_data.get('riskLevel', 'Not specified')}\n"
    
    # GENERAL / ANY OTHER QUESTION
    else:
        response = f"**Patient Analysis - {patient_name}**\n\n"
        response += f"📋 **Quick Overview:**\n"
        response += f"- Name: {patient_name} | ID: {patient_id}\n"
        response += f"- Age: {age} | DOB: {dob} | Gender: {gender}\n"
        response += f"- Contact: {enhanced_data.get('mobileNumber', 'N/A')}\n"
        response += f"- Diagnosis: {enhanced_data.get('diagnosis', 'Not specified')}\n"
        response += f"- Risk Level: {enhanced_data.get('riskLevel', 'Not specified')}\n\n"
        
        # Latest findings
        if reports:
            response += "**Latest Findings:**\n"
            for report in reports[:2]:
                code = report.get('code', {}).get('text', 'Report')
                conclusion = report.get('conclusion', 'No conclusion')
                response += f"• {code}: {conclusion}\n"
            response += "\n"
        
        # Treatment plan
        treatment_plan = enhanced_data.get('treatmentPlan', [])
        if treatment_plan:
            response += "**Current Treatment Plan:**\n"
            for treatment in treatment_plan[:2]:
                response += f"• {treatment}\n"
            response += "\n"
        
        # Medications
        medications = enhanced_data.get('medications', [])
        if medications:
            response += "**Current Medications:**\n"
            for med in medications[:3]:
                response += f"• {med}\n"
            response += "\n"
        
        # Lab results summary
        if lab_data.get('result'):
            abnormal = [r for r in lab_data['result'] if r.get('status') != 'Normal']
            if abnormal:
                response += f"**Lab Results:** {len(abnormal)} abnormal test(s) detected\n"
                for result in abnormal[:3]:
                    response += f"• {result.get('test')}: {result.get('value')} ({result.get('status')})\n"
                response += "\n"
        
        response += "**How I Can Help:**\n"
        response += "• Ask about **findings** or **diagnosis**\n"
        response += "• Ask about **medication** or **treatment** plans\n"
        response += "• Inquire about **lab results** or **vital signs**\n"
        response += "• Ask about **risk level** or **anomalies**\n"
        response += "• Request **medical history** or **demographics**\n"
    
    # Add professional disclaimer at the end
    if not response.endswith("\n"):
        response += "\n"
    response += "\n---\n**⚠️ Medical Disclaimer:** This analysis is based on available medical records. Always consult with qualified medical professionals for clinical decisions, diagnoses, and treatment recommendations."
    
    return response

@app.route('/api/patient-primary-dicom/<patient_id>')
def get_patient_primary_dicom(patient_id):
    """Get the primary DICOM image for a patient for AI chat display"""
    try:
        # Get DICOM images for the patient
        dicom_images = get_dicom_images_for_patient(
            Config.ORTHANC_URL,
            Config.ORTHANC_AUTH,
            patient_id
        )
        
        if not dicom_images:
            return jsonify({'success': False, 'error': 'No DICOM images found for patient'})
        
        # Return the first (primary) DICOM image
        primary_dicom = dicom_images[0]
        
        return jsonify({
            'success': True,
            'dicom': {
                'instance_id': primary_dicom.get('orthanc_instance_id') or primary_dicom.get('instance_id'),
                'modality': primary_dicom.get('modality', 'Unknown'),
                'study_date': primary_dicom.get('date', 'Unknown'),
                'description': primary_dicom.get('description', 'Medical Image'),
                # Provide an explicit image URL for frontend consumers. If Orthanc instance id
                # is available, use the orthanc-dicom-image endpoint; otherwise, serve a local
                # preview from uploads via local_dicom_preview
                'image_url': (f"/api/orthanc-dicom-image/{primary_dicom.get('orthanc_instance_id')}?format=png"
                              if primary_dicom.get('orthanc_instance_id')
                              else f"/local-dicom-preview/{patient_id}/{primary_dicom.get('instance_id')}")
            }
        })
        
    except Exception as e:
        logger.error(f"Error fetching primary DICOM for patient {patient_id}: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# ─── KIDNEY CANCER ANALYSIS ROUTES ───────────────────────────────────────────
from kidney_analyzer import get_analyzer as get_kidney_analyzer

@app.route('/kidney-analysis')
def kidney_analysis_page():
    """Render the kidney cancer analysis page"""
    analyzer = get_kidney_analyzer()
    return render_template('kidney_analysis.html', model_ready=analyzer.is_ready())

@app.route('/kidney-analyze', methods=['POST'])
def kidney_analyze():
    """Analyze an uploaded kidney CT image"""
    try:
        analyzer = get_kidney_analyzer()
        if not analyzer.is_ready():
            return jsonify({
                'success': False,
                'error': 'Model not trained yet. Please run: python train_kidney_model.py'
            })

        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image uploaded'})

        f = request.files['image']
        if f.filename == '':
            return jsonify({'success': False, 'error': 'Empty filename'})

        # Save uploaded image
        upload_dir = os.path.join('uploads', 'kidney_analysis')
        os.makedirs(upload_dir, exist_ok=True)
        fname = secure_filename(f.filename)
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        saved_name = f"kidney_{timestamp}_{fname}"
        saved_path = os.path.join(upload_dir, saved_name)
        f.save(saved_path)

        # Run classification
        classification = analyzer.classify(saved_path)
        if 'error' in classification:
            return jsonify({'success': False, 'error': classification['error']})

        # Get medical suggestions
        suggestions = analyzer.get_medical_suggestions(
            classification['predicted_class'],
            classification['confidence']
        )

        return jsonify({
            'success': True,
            'classification': classification,
            'suggestions': suggestions
        })

    except Exception as e:
        logger.error(f"Kidney analysis error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# ─── RUN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load conversion history on startup
    load_conversion_history()
    app.run(debug=True)
    
print("Kidney analysis API started")