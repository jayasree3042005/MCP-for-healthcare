"""
Healthcare Data Processing Utilities
- Structured observation generation from HL7
- DICOM image retrieval and conversion
- Patient data structuring
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)

# ─── VITAL SIGNS MAPPING ──────────────────────────────────────────────────
VITAL_SIGNS_MAPPING = {
    '8867-4': {'name': 'Heart Rate', 'unit': 'beats/min', 'normal_range': '60-100'},
    '8480-6': {'name': 'Systolic BP', 'unit': 'mmHg', 'normal_range': '90-120'},
    '8462-4': {'name': 'Diastolic BP', 'unit': 'mmHg', 'normal_range': '60-80'},
    '2708-6': {'name': 'Oxygen Saturation', 'unit': '%', 'normal_range': '95-100'},
    '8310-5': {'name': 'Body Temperature', 'unit': '°C', 'normal_range': '36.5-37.5'},
    '3141-9': {'name': 'Body Weight', 'unit': 'kg', 'normal_range': 'Varies'},
    '8302-2': {'name': 'Body Height', 'unit': 'cm', 'normal_range': 'Varies'},
}

# ─── LAB TESTS MAPPING ────────────────────────────────────────────────────
LAB_TESTS_MAPPING = {
    '6690-2': {'name': 'WBC', 'unit': 'K/uL', 'normal_range': '4.5-11.0'},
    '718-7': {'name': 'Hemoglobin', 'unit': 'g/dL', 'normal_range': '12.0-17.5'},
    '777-3': {'name': 'Platelets', 'unit': 'K/uL', 'normal_range': '150-400'},
    '30341-2': {'name': 'CRP', 'unit': 'mg/L', 'normal_range': '<3.0'},
    '33959-8': {'name': 'Procalcitonin', 'unit': 'ng/mL', 'normal_range': '<0.1'},
}


def parse_vital_signs_from_hl7(hl7_text: str) -> List[Dict[str, Any]]:
    """
    Extract vital signs from HL7 OBX segments
    Returns structured vital sign observations
    """
    vital_signs = []
    
    lines = hl7_text.split('\n')
    for line in lines:
        if line.startswith('OBX'):
            fields = line.split('|')
            if len(fields) >= 8:
                try:
                    # OBX[3] = observation id
                    # OBX[5] = observation value
                    # OBX[6] = units
                    obs_id = fields[3].split('^')[0] if '^' in fields[3] else fields[3]
                    obs_value = fields[5].strip()
                    obs_unit = fields[6].strip() if len(fields) > 6 else ''
                    
                    if obs_id in VITAL_SIGNS_MAPPING:
                        mapping = VITAL_SIGNS_MAPPING[obs_id]
                        
                        # Determine if abnormal
                        abnormal = False
                        warning = False
                        
                        try:
                            value = float(obs_value)
                            range_str = mapping['normal_range']
                            
                            if '-' in range_str and range_str != 'Varies':
                                range_parts = range_str.split('-')
                                min_val = float(range_parts[0])
                                max_val = float(range_parts[1])
                                
                                if value < min_val or value > max_val:
                                    # Check severity
                                    if obs_id == '8310-5' and value > 39:  # High fever
                                        abnormal = True
                                    elif obs_id == '2708-6' and value < 92:  # Low O2
                                        abnormal = True
                                    elif value < min_val * 0.8 or value > max_val * 1.2:
                                        abnormal = True
                                    else:
                                        warning = True
                        except:
                            pass
                        
                        vital_signs.append({
                            'loinc_code': obs_id,
                            'name': mapping['name'],
                            'value': obs_value,
                            'unit': obs_unit or mapping['unit'],
                            'reference': mapping['normal_range'],
                            'abnormal': abnormal,
                            'warning': warning,
                            'timestamp': datetime.now().isoformat()
                        })
                except Exception as e:
                    logger.warning(f"Error parsing vital sign: {e}")
                    continue
    
    return vital_signs


def parse_lab_results_from_hl7(hl7_text: str) -> List[Dict[str, Any]]:
    """
    Extract lab results from HL7 OBX segments
    Returns structured lab result observations
    """
    lab_results = []
    
    lines = hl7_text.split('\n')
    for line in lines:
        if line.startswith('OBX'):
            fields = line.split('|')
            if len(fields) >= 8:
                try:
                    obs_id = fields[3].split('^')[0] if '^' in fields[3] else fields[3]
                    obs_value = fields[5].strip()
                    obs_unit = fields[6].strip() if len(fields) > 6 else ''
                    obs_range = fields[7].strip() if len(fields) > 7 else ''
                    abnormal_flag = fields[8].strip() if len(fields) > 8 else ''
                    
                    if obs_id in LAB_TESTS_MAPPING:
                        mapping = LAB_TESTS_MAPPING[obs_id]
                        
                        # Determine status
                        status = 'Normal'
                        if abnormal_flag in ['H', 'L', '>L', '<H']:
                            status = 'Abnormal'
                        elif abnormal_flag == '':
                            # Try to infer from value vs range
                            try:
                                value = float(obs_value)
                                if obs_range and '-' in obs_range:
                                    range_parts = obs_range.split('-')
                                    min_val = float(range_parts[0])
                                    max_val = float(range_parts[1])
                                    
                                    if value < min_val or value > max_val:
                                        status = 'Abnormal'
                                    elif abs(value - min_val) < 0.2 or abs(value - max_val) < 0.2:
                                        status = 'Warning'
                            except:
                                pass
                        else:
                            status = 'Warning'
                        
                        lab_results.append({
                            'loinc_code': obs_id,
                            'name': mapping['name'],
                            'value': obs_value,
                            'unit': obs_unit or mapping['unit'],
                            'reference': obs_range or mapping['normal_range'],
                            'status': status,
                            'timestamp': datetime.now().isoformat()
                        })
                except Exception as e:
                    logger.warning(f"Error parsing lab result: {e}")
                    continue
    
    return lab_results


def parse_diagnoses_from_hl7(hl7_text: str) -> List[Dict[str, Any]]:
    """Extract diagnoses from HL7 DG1 segments"""
    diagnoses = []
    
    lines = hl7_text.split('\n')
    for line in lines:
        if line.startswith('DG1'):
            fields = line.split('|')
            if len(fields) >= 5:
                try:
                    # DG1[3] = diagnosis code
                    # DG1[4] = diagnosis description
                    diag_code = fields[3].strip()
                    diag_desc = fields[4].strip() if len(fields) > 4 else ''
                    
                    diagnoses.append({
                        'code': diag_code,
                        'description': diag_desc,
                        'severity': 'Unspecified'
                    })
                except Exception as e:
                    logger.warning(f"Error parsing diagnosis: {e}")
                    continue
    
    return diagnoses


def parse_medications_from_hl7(hl7_text: str) -> List[Dict[str, Any]]:
    """
    Extract medications from HL7 NTE segments (typically in format)
    "MEDICATIONS: Drug1 dose route freq duration; Drug2..."
    """
    medications = []
    
    lines = hl7_text.split('\n')
    for line in lines:
        if line.startswith('NTE'):
            fields = line.split('|')
            if len(fields) >= 4:
                note_text = fields[3].strip() if len(fields) > 3 else ''
                
                if 'MEDICATIONS:' in note_text.upper():
                    # Extract medication part
                    med_section = note_text.split('MEDICATIONS:')[-1].strip()
                    med_items = med_section.split(';')
                    
                    for med_item in med_items:
                        med_item = med_item.strip()
                        if med_item:
                            # Parse medication details
                            parts = med_item.split()
                            if len(parts) >= 2:
                                medications.append({
                                    'name': parts[0],
                                    'dose': parts[1] if len(parts) > 1 else 'Unknown',
                                    'route': parts[2] if len(parts) > 2 else 'PO',
                                    'frequency': parts[3] if len(parts) > 3 else 'Unknown',
                                    'duration': parts[4] if len(parts) > 4 else 'Ongoing',
                                    'indication': ''
                                })
    
    return medications


def parse_clinical_notes_from_hl7(hl7_text: str) -> Dict[str, str]:
    """Extract findings, impression, and discharge summary from NTE segments"""
    notes = {
        'findings': '',
        'impression': '',
        'discharge_summary': ''
    }
    
    lines = hl7_text.split('\n')
    for line in lines:
        if line.startswith('NTE'):
            fields = line.split('|')
            if len(fields) >= 4:
                note_text = fields[3].strip() if len(fields) > 3 else ''
                
                if 'FINDINGS:' in note_text.upper():
                    notes['findings'] = note_text.split('FINDINGS:')[-1].split('IMPRESSION:')[0].strip()
                elif 'IMPRESSION:' in note_text.upper():
                    notes['impression'] = note_text.split('IMPRESSION:')[-1].split('DISCHARGE')[0].strip()
                elif 'DISCHARGE' in note_text.upper():
                    notes['discharge_summary'] = note_text.split('DISCHARGE')[-1].strip()
    
    return notes


def get_dicom_images_for_patient(orthanc_url: str, orthanc_auth: Tuple[str, str], 
                                  patient_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve DICOM images from Orthanc for a specific patient
    """
    images = []
    
    try:
        # First, check for local files under uploads/dcm_by_patient/<patient_id>/
        local_dir = os.path.join(os.getcwd(), 'uploads', 'dcm_by_patient', patient_id)
        if os.path.isdir(local_dir):
            for fname in sorted(os.listdir(local_dir)):
                if fname.lower().endswith('.dcm'):
                    fpath = os.path.join(local_dir, fname)
                    stat = os.stat(fpath)
                    images.append({
                        'filename': fname,
                        'instance_id': fname,
                        'study_id': '',
                        'series_id': '',
                        'study_uid': '',
                        'series_uid': '',
                        'instance_uid': fname,
                        'modality': 'DCM',
                        'date': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d'),
                        'orthanc_instance_id': ''
                    })
            if images:
                return images

        # Fallback to Orthanc search if local files not found
        search_url = f"{orthanc_url}/api/patients"
        response = requests.get(search_url, auth=HTTPBasicAuth(*orthanc_auth))
        
        if response.status_code == 200:
            patient_list = response.json()
            
            for orthanc_patient_id in patient_list:
                # Get patient details
                patient_url = f"{orthanc_url}/api/patients/{orthanc_patient_id}"
                patient_response = requests.get(patient_url, auth=HTTPBasicAuth(*orthanc_auth))
                
                if patient_response.status_code == 200:
                    patient_data = patient_response.json()
                    
                    # Check if this is our patient
                    if patient_data.get('MainDicomTags', {}).get('PatientID', '').lower() == patient_id.lower() or \
                       str(orthanc_patient_id).lower() == patient_id.lower():
                        
                        # Get all studies for this patient
                        for study_id in patient_data.get('Studies', []):
                            study_url = f"{orthanc_url}/api/studies/{study_id}"
                            study_response = requests.get(study_url, auth=HTTPBasicAuth(*orthanc_auth))
                            
                            if study_response.status_code == 200:
                                study_data = study_response.json()
                                study_uid = study_data.get('MainDicomTags', {}).get('StudyInstanceUID', study_id)
                                
                                # Get all series for this study
                                for series_id in study_data.get('Series', []):
                                    series_url = f"{orthanc_url}/api/series/{series_id}"
                                    series_response = requests.get(series_url, auth=HTTPBasicAuth(*orthanc_auth))
                                    
                                    if series_response.status_code == 200:
                                        series_data = series_response.json()
                                        series_uid = series_data.get('MainDicomTags', {}).get('SeriesInstanceUID', series_id)
                                        modality = series_data.get('MainDicomTags', {}).get('Modality', 'Unknown')
                                        
                                        # Get all instances for this series
                                        for instance_id in series_data.get('Instances', []):
                                            instance_url = f"{orthanc_url}/api/instances/{instance_id}"
                                            instance_response = requests.get(instance_url, auth=HTTPBasicAuth(*orthanc_auth))
                                            
                                            if instance_response.status_code == 200:
                                                instance_data = instance_response.json()
                                                instance_uid = instance_data.get('MainDicomTags', {}).get('SOPInstanceUID', instance_id)
                                                
                                                images.append({
                                                    'filename': f"{modality}_{instance_id}.dcm",
                                                    'instance_id': instance_id,
                                                    'study_id': study_id,
                                                    'series_id': series_id,
                                                    'study_uid': study_uid,
                                                    'series_uid': series_uid,
                                                    'instance_uid': instance_uid,
                                                    'modality': modality,
                                                    'date': study_data.get('MainDicomTags', {}).get('StudyDate', 'Unknown'),
                                                    'orthanc_instance_id': instance_id
                                                })
    
    except Exception as e:
        logger.error(f"Error retrieving DICOM images: {e}")
    
    return images


def convert_dicom_to_png(orthanc_url: str, orthanc_auth: Tuple[str, str], 
                        instance_id: str) -> bytes:
    """
    Convert DICOM instance to PNG image using Orthanc
    """
    try:
        render_url = f"{orthanc_url}/api/instances/{instance_id}/preview"
        response = requests.get(render_url, auth=HTTPBasicAuth(*orthanc_auth))
        
        if response.status_code == 200:
            return response.content
        else:
            logger.error(f"Failed to retrieve DICOM preview: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error converting DICOM to PNG: {e}")
        return None


def structure_patient_data(patient_id: str, patient_name: str, gender: str, 
                          dob: str, phone: str, email: str,
                          hl7_text: str, dicom_images: List[Dict[str, Any]],
                          severity: str = 'Unknown') -> Dict[str, Any]:
    """
    Structure all patient data into a comprehensive patient object
    """
    
    # Parse clinical data from HL7
    vital_signs = parse_vital_signs_from_hl7(hl7_text)
    lab_results = parse_lab_results_from_hl7(hl7_text)
    diagnoses = parse_diagnoses_from_hl7(hl7_text)
    medications = parse_medications_from_hl7(hl7_text)
    clinical_notes = parse_clinical_notes_from_hl7(hl7_text)
    
    # Calculate age from DOB
    try:
        dob_date = datetime.strptime(dob, '%Y-%m-%d')
        age = (datetime.now() - dob_date).days // 365
    except:
        age = 'Unknown'
    
    patient_data = {
        'id': patient_id,
        'name': patient_name,
        'age': age,
        'gender': gender,
        'dob': dob,
        'phone': phone,
        'email': email,
        'severity': severity,
        'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'updated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'vital_signs': vital_signs,
        'lab_results': lab_results,
        'diagnoses': diagnoses,
        'medications': medications,
        'findings': clinical_notes.get('findings', ''),
        'impression': clinical_notes.get('impression', ''),
        'discharge_summary': clinical_notes.get('discharge_summary', ''),
        'dicom_images': dicom_images
    }
    
    return patient_data
