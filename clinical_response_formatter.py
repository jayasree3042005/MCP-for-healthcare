"""
Clinical Response Formatter
Formats AI responses to match the clinical dashboard format requirements
"""

from typing import Dict, List, Any, Optional
import json


class ClinicalResponseFormatter:
    """Formats medical data into structured KEY-VALUE clinical format"""
    
    @staticmethod
    def format_patient_info(patient_data: Dict[str, Any], enhanced_data: Optional[Dict] = None) -> str:
        """Format patient information in structured KEY-VALUE format"""
        
        # Extract basic patient info
        patient_id = patient_data.get('id', 'N/A')
        name = "N/A"
        if patient_data.get('name'):
            name = patient_data['name'][0].get('text', 'N/A') if isinstance(patient_data['name'], list) else patient_data['name']
        
        dob = patient_data.get('birthDate', 'N/A')
        gender = patient_data.get('gender', 'N/A')
        
        # Calculate age if DOB available
        age = "N/A"
        if dob and dob != 'N/A':
            try:
                from datetime import datetime
                birth_date = datetime.strptime(dob[:10], '%Y-%m-%d')
                today = datetime.now()
                age = str((today - birth_date).days // 365)
            except:
                age = "N/A"
        
        # Get contact from enhanced data
        contact = "N/A"
        if enhanced_data:
            mobile = enhanced_data.get('mobileNumber', '')
            email = enhanced_data.get('email', '')
            if mobile and email:
                contact = f"{mobile} | {email}"
            elif mobile:
                contact = mobile
            elif email:
                contact = email
        
        formatted = "==============================\n"
        formatted += "🩺 PATIENT INFORMATION\n"
        formatted += "==============================\n\n"
        formatted += f"Patient ID        : {patient_id}\n"
        formatted += f"Name              : {name}\n"
        formatted += f"Age               : {age}\n"
        formatted += f"Gender            : {gender}\n"
        formatted += f"Contact           : {contact}\n"
        
        return formatted
    
    @staticmethod
    def format_diagnosis(enhanced_data: Optional[Dict], reports: List[Dict]) -> str:
        """Format diagnosis section"""
        
        diagnosis = "N/A"
        risk_level = "N/A"
        
        if enhanced_data:
            diagnosis = enhanced_data.get('diagnosis', 'Not specified')
            risk_level = enhanced_data.get('riskLevel', 'Not specified')
        
        # Get from reports if available
        if not diagnosis or diagnosis == 'N/A':
            if reports and len(reports) > 0:
                diagnosis = reports[0].get('code', {}).get('text', 'Not specified')
        
        formatted = "\n==============================\n"
        formatted += "📋 DIAGNOSIS\n"
        formatted += "==============================\n\n"
        formatted += f"Diagnosis         : {diagnosis}\n"
        formatted += f"Risk Level        : {risk_level}\n"
        
        return formatted
    
    @staticmethod
    def format_findings(reports: List[Dict], enhanced_data: Optional[Dict] = None) -> str:
        """Format latest findings section"""
        
        formatted = "\n==============================\n"
        formatted += "🔍 LATEST FINDINGS\n"
        formatted += "==============================\n\n"
        
        findings = []
        
        # Get findings from reports
        if reports:
            for report in reports[:3]:
                conclusion = report.get('conclusion', '')
                if conclusion:
                    findings.append(conclusion)
        
        # Get anomalies from enhanced data
        if enhanced_data:
            anomalies = enhanced_data.get('anomalies', [])
            for anomaly in anomalies[:5]:
                if anomaly not in findings:
                    findings.append(anomaly)
        
        if findings:
            for finding in findings[:5]:
                # Clean up finding text
                finding_text = str(finding).strip()
                if not finding_text.startswith('-'):
                    formatted += f"- {finding_text}\n"
                else:
                    formatted += f"{finding_text}\n"
        else:
            formatted += "- No significant findings currently documented\n"
        
        return formatted
    
    @staticmethod
    def format_medications(enhanced_data: Optional[Dict]) -> str:
        """Format medications section"""
        
        formatted = "\n==============================\n"
        formatted += "💊 CURRENT MEDICATIONS\n"
        formatted += "==============================\n\n"
        
        medications = []
        if enhanced_data and enhanced_data.get('medications'):
            medications = enhanced_data.get('medications', [])
        
        if medications:
            for i, med in enumerate(medications[:5], 1):
                if isinstance(med, dict):
                    med_name = med.get('name', med.get('medicine', 'Unknown'))
                    dosage = med.get('dosage', 'Not specified')
                    frequency = med.get('frequency', 'Not specified')
                    duration = med.get('duration', 'Not specified')
                else:
                    med_name = str(med)
                    dosage = "Not specified"
                    frequency = "Not specified"
                    duration = "Not specified"
                
                formatted += f"- Name     : {med_name}\n"
                formatted += f"  Dosage   : {dosage}\n"
                formatted += f"  Frequency: {frequency}\n"
                formatted += f"  Duration : {duration}\n"
                if i < len(medications[:5]):
                    formatted += "\n"
        else:
            formatted += "- No medications currently prescribed\n"
        
        return formatted
    
    @staticmethod
    def format_image_analysis(analysis_data: Dict[str, Any]) -> str:
        """Format DICOM image analysis"""
        
        formatted = "\n==============================\n"
        formatted += "🧠 IMAGE ANALYSIS\n"
        formatted += "==============================\n\n"
        
        observation = analysis_data.get('observation', 'Not analyzed')
        abnormalities = analysis_data.get('abnormalities', 'None detected')
        severity = analysis_data.get('severity', 'Not specified')
        recommendation = analysis_data.get('recommendation', 'Continue monitoring')
        
        formatted += f"Observation   : {observation}\n"
        formatted += f"Abnormalities : {abnormalities}\n"
        formatted += f"Severity      : {severity}\n"
        formatted += f"Recommendation: {recommendation}\n"
        
        return formatted
    
    @staticmethod
    def format_complete_patient_report(
        patient_data: Dict[str, Any],
        enhanced_data: Optional[Dict] = None,
        reports: Optional[List[Dict]] = None,
        include_medications: bool = True,
        image_analysis: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Format complete patient report as structured clinical format
        
        Returns:
            Dictionary with formatted sections and a flag to indicate formatting style
        """
        
        if not reports:
            reports = []
        
        sections = []
        
        # Add patient info
        sections.append(ClinicalResponseFormatter.format_patient_info(patient_data, enhanced_data))
        
        # Add diagnosis
        sections.append(ClinicalResponseFormatter.format_diagnosis(enhanced_data, reports))
        
        # Add findings
        sections.append(ClinicalResponseFormatter.format_findings(reports, enhanced_data))
        
        # Add medications if requested
        if include_medications:
            sections.append(ClinicalResponseFormatter.format_medications(enhanced_data))
        
        # Add image analysis if provided
        if image_analysis:
            sections.append(ClinicalResponseFormatter.format_image_analysis(image_analysis))
        
        formatted_text = "".join(sections)
        
        return {
            'formatted_response': formatted_text,
            'is_structured': True,
            'sections': {
                'patient_info': patient_data,
                'enhanced_data': enhanced_data or {},
                'reports': reports,
                'image_analysis': image_analysis
            }
        }
    
    @staticmethod
    def format_lab_results(lab_data: Dict[str, Any]) -> str:
        """Format lab results in structured format"""
        
        formatted = "\n==============================\n"
        formatted += "🧪 LABORATORY RESULTS\n"
        formatted += "==============================\n\n"
        
        if not lab_data or not lab_data.get('result'):
            return formatted + "- No laboratory results available\n"
        
        abnormal_count = 0
        for result in lab_data.get('result', []):
            test_name = result.get('test', 'Unknown Test')
            value = result.get('value', 'N/A')
            unit = result.get('unit', '')
            status = result.get('status', 'Normal')
            normal_range = result.get('normalRange', '')
            
            status_icon = "⚠️" if status != 'Normal' else "✓"
            formatted += f"{status_icon} {test_name}         : {value} {unit}\n"
            
            if normal_range:
                formatted += f"  Range        : {normal_range}\n"
            if status != 'Normal':
                formatted += f"  Status       : {status}\n"
                abnormal_count += 1
            
            formatted += "\n"
        
        if abnormal_count > 0:
            formatted += f"⚠️ ALERT: {abnormal_count} abnormal result(s) require clinical attention\n"
        
        return formatted
    
    @staticmethod
    def format_risk_assessment(enhanced_data: Optional[Dict], lab_data: Optional[Dict] = None) -> str:
        """Format risk assessment and anomalies"""
        
        risk_level = "Not specified"
        if enhanced_data:
            risk_level = enhanced_data.get('riskLevel', 'Not specified')
        
        formatted = "\n==============================\n"
        formatted += "🚨 RISK ASSESSMENT & ANOMALIES\n"
        formatted += "==============================\n\n"
        formatted += f"Overall Risk Level: {risk_level}\n\n"
        
        # Anomalies
        if enhanced_data and enhanced_data.get('anomalies'):
            formatted += "Detected Anomalies:\n"
            for anomaly in enhanced_data.get('anomalies', []):
                formatted += f"- {anomaly}\n"
            formatted += "\n"
        
        # Abnormal lab results
        if lab_data and lab_data.get('result'):
            abnormal_labs = [r for r in lab_data['result'] if r.get('status') != 'Normal']
            if abnormal_labs:
                formatted += "Abnormal Lab Findings:\n"
                for result in abnormal_labs:
                    formatted += f"- {result.get('test', 'Test')}: {result.get('value')} {result.get('unit')} "
                    formatted += f"(Expected: {result.get('normalRange', 'N/A')})\n"
        
        return formatted
    
    @staticmethod
    def convert_response_to_structured(
        response_text: str,
        patient_data: Dict[str, Any],
        enhanced_data: Optional[Dict] = None,
        reports: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Convert a free-text AI response to structured format or augment with structure
        """
        
        if not reports:
            reports = []
        
        # If response already looks structured, return it as is
        if "======" in response_text or "🩺" in response_text:
            return {
                'response': response_text,
                'is_structured': True,
                'format': 'already_structured'
            }
        
        # Otherwise, create a structured response based on content
        return {
            'response': response_text,
            'is_structured': False,
            'format': 'free_text',
            'recommendation': 'Consider reformatting with clinical_response_formatter.format_complete_patient_report()'
        }


# Formatting helper function for quick use
def format_clinical_response(
    patient_data: Dict[str, Any],
    enhanced_data: Optional[Dict] = None,
    reports: Optional[List[Dict]] = None,
    lab_data: Optional[Dict] = None,
    include_medications: bool = True,
    include_labs: bool = False,
    include_risk: bool = False
) -> str:
    """
    Quickly format clinical response with multiple sections
    
    Args:
        patient_data: Patient FHIR resource
        enhanced_data: Enhanced patient data with diagnosis, anomalies, etc.
        reports: List of diagnostic reports
        lab_data: Lab results data
        include_medications: Include medication section
        include_labs: Include lab results section
        include_risk: Include risk assessment section
    
    Returns:
        Formatted clinical response string
    """
    
    formatter = ClinicalResponseFormatter()
    sections = []
    
    # Always include patient info
    sections.append(formatter.format_patient_info(patient_data, enhanced_data))
    
    # Diagnosis
    sections.append(formatter.format_diagnosis(enhanced_data, reports or []))
    
    # Findings
    sections.append(formatter.format_findings(reports or [], enhanced_data))
    
    # Medications
    if include_medications:
        sections.append(formatter.format_medications(enhanced_data))
    
    # Labs
    if include_labs and lab_data:
        sections.append(formatter.format_lab_results(lab_data))
    
    # Risk
    if include_risk:
        sections.append(formatter.format_risk_assessment(enhanced_data, lab_data))
    
    return "".join(sections)
