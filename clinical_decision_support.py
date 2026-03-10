"""
Clinical Decision Support System
Generates structured medical explanations for patient assessment
"""

from typing import Dict, List, Any
from datetime import datetime

def generate_clinical_interpretation(patient_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Generate a comprehensive clinical interpretation report based on patient data.
    
    Args:
        patient_data: Dictionary containing patient vitals, labs, imaging, symptoms
    
    Returns:
        Dictionary with formatted clinical interpretation sections
    """
    
    # Extract vital signs
    vitals = patient_data.get('vital_signs', {})
    labs = patient_data.get('lab_results', {})
    imaging = patient_data.get('imaging_findings', '')
    symptoms = patient_data.get('symptoms', [])
    diagnoses = patient_data.get('diagnoses', [])
    
    # Parse numeric values
    heart_rate = vitals.get('heart_rate')
    systolic_bp = vitals.get('systolic')
    diastolic_bp = vitals.get('diastolic')
    oxygen_sat = vitals.get('oxygen_saturation')
    temperature = vitals.get('temperature')
    
    wbc = labs.get('wbc')
    hemoglobin = labs.get('hemoglobin')
    crp = labs.get('crp')
    
    # Clinical Assessment Logic
    has_fever = temperature and float(temperature) > 38.0
    low_oxygen = oxygen_sat and float(oxygen_sat) < 94
    elevated_inflammatory = (crp and float(crp) > 10) or (wbc and float(wbc) > 11)
    respiratory_distress = heart_rate and float(heart_rate) > 100
    
    # Determine severity
    severity_score = 0
    if has_fever: severity_score += 1
    if low_oxygen: severity_score += 2
    if elevated_inflammatory: severity_score += 1
    if respiratory_distress: severity_score += 1
    
    severity_level = "Mild" if severity_score <= 1 else "Moderate" if severity_score <= 3 else "Severe"
    urgency_level = "Non-urgent" if severity_score <= 1 else "Urgent" if severity_score <= 3 else "Emergency"
    
    # Build report sections
    clinical_interpretation = _build_clinical_interpretation(
        has_fever, temperature, oxygen_sat, imaging, diagnoses, severity_level
    )
    
    urgency_assessment = _build_urgency_assessment(
        low_oxygen, oxygen_sat, respiratory_distress, severity_level, urgency_level
    )
    
    monitoring_plan = _build_monitoring_plan(
        severity_score, oxygen_sat, temperature, elevated_inflammatory
    )
    
    imaging_findings = _build_imaging_summary(imaging, diagnoses)
    
    risk_table = _build_risk_assessment_table(
        temperature, oxygen_sat, severity_level, imaging, has_fever, urgency_level
    )
    
    warning_signs = _build_warning_signs_list()
    
    return {
        'clinical_interpretation': clinical_interpretation,
        'urgency_assessment': urgency_assessment,
        'monitoring_plan': monitoring_plan,
        'imaging_findings': imaging_findings,
        'risk_table': risk_table,
        'warning_signs': warning_signs,
        'severity_level': severity_level,
        'urgency_level': urgency_level,
        'generated_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


def _build_clinical_interpretation(has_fever: bool, temperature: float, 
                                   oxygen_sat: float, imaging: str, 
                                   diagnoses: List[str], severity: str) -> str:
    """Build the clinical interpretation section"""
    
    interpretation = "CLINICAL INTERPRETATION\n"
    interpretation += "─" * 50 + "\n\n"
    
    findings = []
    
    if diagnoses:
        findings.append(f"The clinical presentation is consistent with: {', '.join(diagnoses)}")
    else:
        findings.append("Based on current findings, the patient appears to have a respiratory condition.")
    
    if has_fever:
        findings.append(f"The patient has a fever (temperature {temperature}°C), suggesting an active inflammatory response or infection.")
    else:
        findings.append("The patient's temperature is currently normal or low.")
    
    if oxygen_sat and float(oxygen_sat) >= 95:
        findings.append(f"Oxygen saturation is adequate at {oxygen_sat}%, indicating reasonable oxygenation on room air.")
    elif oxygen_sat and float(oxygen_sat) >= 90:
        findings.append(f"Oxygen saturation is mildly reduced at {oxygen_sat}%, suggesting mild hypoxemia.")
    else:
        findings.append(f"Oxygen saturation is significantly reduced at {oxygen_sat}%, indicating hypoxemia.")
    
    if imaging:
        findings.append(f"Imaging findings show: {imaging}")
    
    findings.append(f"Overall, the clinical status appears {severity.lower()}.")
    
    interpretation += "\n".join(findings)
    interpretation += "\n"
    
    return interpretation


def _build_urgency_assessment(low_oxygen: bool, oxygen_sat: float, 
                             respiratory_distress: bool, severity: str,
                             urgency_level: str) -> str:
    """Build the urgency assessment section"""
    
    assessment = "DOES THE PATIENT REQUIRE URGENT INTERVENTION?\n"
    assessment += "─" * 50 + "\n\n"
    
    if urgency_level == "Emergency":
        assessment += "⚠️ YES - IMMEDIATE ATTENTION REQUIRED\n\n"
        assessment += "The patient shows signs requiring urgent or emergent care:\n"
        if low_oxygen:
            assessment += f"  • Oxygen saturation is {oxygen_sat}%, which is below safe levels\n"
        if respiratory_distress:
            assessment += "  • Elevated heart rate suggests respiratory distress\n"
        assessment += "  • Risk of rapid deterioration is significant\n"
        assessment += "\nRECOMMENDATION: Seek immediate emergency care or hospital admission.\n"
    elif urgency_level == "Urgent":
        assessment += "⚠️ YES - URGENT CARE NEEDED\n\n"
        assessment += "The patient shows concerning signs that require prompt evaluation:\n"
        if low_oxygen:
            assessment += f"  • Oxygen saturation is {oxygen_sat}%, which is below ideal levels\n"
        if respiratory_distress:
            assessment += "  • Signs of respiratory stress are present\n"
        assessment += "  • Close monitoring is essential\n"
        assessment += "\nRECOMMENDATION: Hospital admission or urgent outpatient evaluation is advised.\n"
    else:
        assessment += "✓ NO - Currently stable\n\n"
        assessment += "The patient appears clinically stable at this time:\n"
        if oxygen_sat and float(oxygen_sat) >= 95:
            assessment += f"  • Oxygen saturation is adequate at {oxygen_sat}%\n"
        assessment += "  • Vital signs are relatively stable\n"
        assessment += "  • No immediate signs of severe respiratory distress\n"
        assessment += "\nRECOMMENDATION: Close follow-up and monitoring at home, with clear warning signs to watch for.\n"
    
    assessment += f"\nCURRENT STATUS: {severity} condition\n"
    
    return assessment


def _build_monitoring_plan(severity_score: int, oxygen_sat: float, 
                          temperature: float, elevated_inflammatory: bool) -> str:
    """Build the monitoring and follow-up plan"""
    
    plan = "MONITORING AND FOLLOW-UP\n"
    plan += "─" * 50 + "\n\n"
    
    if severity_score >= 4:
        plan += "INTENSIVE MONITORING REQUIRED\n"
        plan += "  • Continuous or frequent vital sign monitoring\n"
        plan += "  • Daily blood oxygen checks\n"
        plan += "  • Hospital-based care recommended\n"
        plan += "  • Risk of rapid deterioration is significant\n\n"
    elif severity_score >= 2:
        plan += "REGULAR MONITORING REQUIRED\n"
        plan += "  • Check oxygen saturation twice daily\n"
        plan += "  • Monitor temperature at least once daily\n"
        plan += "  • Watch for worsening symptoms\n"
        plan += "  • Follow-up assessment within 24-48 hours\n\n"
    else:
        plan += "ROUTINE MONITORING\n"
        plan += "  • Check oxygen saturation once daily\n"
        plan += "  • Monitor temperature\n"
        plan += "  • General symptom awareness\n"
        plan += "  • Follow-up assessment within 3-5 days\n\n"
    
    plan += "RISK FACTORS FOR DETERIORATION:\n"
    if oxygen_sat and float(oxygen_sat) < 95:
        plan += "  • Oxygen saturation trending downward could indicate progression\n"
    if temperature and float(temperature) > 39:
        plan += "  • Persistently high fever may indicate worsening infection\n"
    if elevated_inflammatory:
        plan += "  • Elevated inflammatory markers suggest ongoing infection or inflammation\n"
    
    plan += "\nFOLLOW-UP:\n"
    plan += "  • Repeat imaging if clinical deterioration occurs\n"
    plan += "  • Repeat lab work in 3-5 days if hospitalized; 1-2 weeks if outpatient\n"
    plan += "  • Clear discharge/transition plan should be established\n"
    
    return plan


def _build_imaging_summary(imaging: str, diagnoses: List[str]) -> str:
    """Build imaging findings summary"""
    
    summary = "IMAGING FINDINGS SUMMARY\n"
    summary += "─" * 50 + "\n\n"
    
    if imaging:
        summary += imaging + "\n\n"
    
    if diagnoses:
        summary += "Associated Clinical Diagnoses:\n"
        for diagnosis in diagnoses:
            summary += f"  • {diagnosis}\n"
    else:
        summary += "Imaging findings are being reviewed for classification.\n"
    
    return summary


def _build_risk_assessment_table(temperature: float, oxygen_sat: float, 
                                severity: str, imaging: str, 
                                has_fever: bool, urgency: str) -> str:
    """Build the risk assessment table"""
    
    table = "PATIENT RISK ASSESSMENT SUMMARY\n"
    table += "─" * 50 + "\n\n"
    table += "┌─────────────────────────────────┬──────────────────────────────┐\n"
    table += "│ Assessment Factor               │ Status                       │\n"
    table += "├─────────────────────────────────┼──────────────────────────────┤\n"
    
    fever_status = "Present (High)" if has_fever else "Absent or Low"
    table += f"│ Fever                           │ {fever_status:<28} │\n"
    
    table += f"│ Current Clinical Status         │ {severity + ' condition':<28} │\n"
    
    danger_level = "High" if urgency == "Emergency" else "Moderate" if urgency == "Urgent" else "Low"
    table += f"│ Danger Level                    │ {danger_level:<28} │\n"
    
    o2_status = f"{oxygen_sat}%" if oxygen_sat else "Unknown"
    table += f"│ Oxygen Saturation               │ {o2_status:<28} │\n"
    
    imaging_status = "Abnormal findings" if imaging else "Normal or pending"
    table += f"│ Imaging Findings                │ {imaging_status:<28} │\n"
    
    pneumonia_status = "Likely" if "consolidation" in imaging.lower() or "pneumonia" in imaging.lower() else "Not evident"
    table += f"│ Pneumonia/Consolidation Status  │ {pneumonia_status:<28} │\n"
    
    care_level = "Hospital care" if danger_level in ["High", "Moderate"] else "Home care with monitoring"
    table += f"│ Recommended Care Level          │ {care_level:<28} │\n"
    
    disposition = "Hospitalized" if danger_level in ["High", "Moderate"] else "Outpatient/Home"
    table += f"│ Disposition                     │ {disposition:<28} │\n"
    
    table += "└─────────────────────────────────┴──────────────────────────────┘\n"
    
    return table


def _build_warning_signs_list() -> str:
    """Build list of warning signs to monitor"""
    
    warnings = "WHAT TO WATCH FOR - WARNING SIGNS\n"
    warnings += "─" * 50 + "\n\n"
    warnings += "IMMEDIATE EMERGENCY SIGNS (Seek emergency care immediately):\n"
    warnings += "  🔴 Severe shortness of breath or inability to speak in full sentences\n"
    warnings += "  🔴 Oxygen saturation drops below 90%\n"
    warnings += "  🔴 Chest pain or pressure\n"
    warnings += "  🔴 Confusion, difficulty concentrating, or severe fatigue\n"
    warnings += "  🔴 Loss of consciousness\n"
    warnings += "  🔴 Persistent coughing up blood\n\n"
    
    warnings += "URGENT WARNING SIGNS (Contact healthcare provider same day):\n"
    warnings += "  🟠 Worsening cough or productive cough with discolored sputum\n"
    warnings += "  🟠 Temperature rises above 39°C (102°F)\n"
    warnings += "  🟠 Persistent high fever lasting more than 3-5 days\n"
    warnings += "  🟠 Oxygen saturation below 94%\n"
    warnings += "  🟠 New onset or worsening shortness of breath\n"
    warnings += "  🟠 Persistent chest discomfort\n"
    warnings += "  🟠 Severe fatigue or weakness\n\n"
    
    warnings += "MONITORING SIGNS (Keep records and report to healthcare provider):\n"
    warnings += "  🟡 Daily temperature trends\n"
    warnings += "  🟡 Oxygen saturation at rest and with activity\n"
    warnings += "  🟡 Frequency and character of cough\n"
    warnings += "  🟡 Energy levels and ability to perform daily activities\n"
    warnings += "  🟡 Any new symptoms or symptom changes\n"
    
    return warnings


def format_clinical_report(interpretation_data: Dict[str, str]) -> str:
    """
    Format the complete clinical report as a readable string for display
    
    Args:
        interpretation_data: Dictionary from generate_clinical_interpretation()
    
    Returns:
        Formatted multi-section report string
    """
    
    report = "=" * 60 + "\n"
    report += "CLINICAL DECISION SUPPORT REPORT\n"
    report += f"Generated: {interpretation_data.get('generated_timestamp', '')}\n"
    report += "=" * 60 + "\n\n"
    
    report += interpretation_data.get('clinical_interpretation', '') + "\n\n"
    report += interpretation_data.get('urgency_assessment', '') + "\n\n"
    report += interpretation_data.get('monitoring_plan', '') + "\n\n"
    report += interpretation_data.get('imaging_findings', '') + "\n\n"
    report += interpretation_data.get('risk_table', '') + "\n\n"
    report += interpretation_data.get('warning_signs', '') + "\n\n"
    
    report += "=" * 60 + "\n"
    report += "DISCLAIMER:\n"
    report += "This report is for clinical decision support and educational purposes only.\n"
    report += "It does not constitute medical advice or a final diagnosis.\n"
    report += "All findings should be reviewed and confirmed by a qualified healthcare provider.\n"
    report += "=" * 60 + "\n"
    
    return report
