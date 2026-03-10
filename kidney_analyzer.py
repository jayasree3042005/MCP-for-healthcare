"""
Kidney Cancer Analyzer Module
==============================
Provides classification, segmentation visualization (heatmap),
and AI-generated medical suggestions for kidney CT images.

Uses scikit-learn + OpenCV (Python 3.14 compatible).
"""

import os
import json
import base64
import pickle
import numpy as np
import cv2
from io import BytesIO
from PIL import Image


# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "kidney_classifier.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "kidney_scaler.pkl")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
IMG_SIZE = (128, 128)


def extract_features(image_path):
    """
    Extract features from a kidney CT image (must match training pipeline).
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = []

    # 1. HOG features
    win_size = IMG_SIZE
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    n_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, n_bins)
    hog_features = hog.compute(gray).flatten()
    step = max(1, len(hog_features) // 200)
    hog_subsampled = hog_features[::step][:200]
    features.extend(hog_subsampled)

    # 2. Color histogram
    hist_gray = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
    hist_gray = hist_gray / (hist_gray.sum() + 1e-8)
    features.extend(hist_gray)

    for i in range(3):
        hist_c = cv2.calcHist([img], [i], None, [16], [0, 256]).flatten()
        hist_c = hist_c / (hist_c.sum() + 1e-8)
        features.extend(hist_c)

    # 3. Texture features
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    features.extend([
        laplacian_var,
        sobel_mag.mean(),
        sobel_mag.std(),
        sobel_mag.max(),
    ])

    # 4. Statistical features
    features.extend([
        gray.mean(),
        gray.std(),
        float(np.median(gray)),
        float(np.percentile(gray, 25)),
        float(np.percentile(gray, 75)),
        gray.min().item(),
        gray.max().item(),
    ])

    # 5. Region-based features
    h, w = gray.shape
    center = gray[h//4:3*h//4, w//4:3*w//4]
    periphery_mask = np.ones_like(gray, dtype=bool)
    periphery_mask[h//4:3*h//4, w//4:3*w//4] = False
    periphery = gray[periphery_mask]
    features.extend([
        center.mean(),
        center.std(),
        periphery.mean(),
        periphery.std(),
        center.mean() - periphery.mean(),
    ])

    # 6. Contour features
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        circularity = 4 * np.pi * area / (perimeter**2 + 1e-8)
        features.extend([area, perimeter, circularity, len(contours)])
    else:
        features.extend([0, 0, 0, 0])

    return np.array(features, dtype=np.float32)


class KidneyAnalyzer:
    """Kidney CT image classifier with segmentation visualization."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.class_names = None
        self._load_model()

    def _load_model(self):
        """Load the trained model, scaler, and class names."""
        try:
            if (os.path.exists(MODEL_PATH) and
                os.path.exists(SCALER_PATH) and
                os.path.exists(CLASS_NAMES_PATH)):

                with open(MODEL_PATH, "rb") as f:
                    self.model = pickle.load(f)
                with open(SCALER_PATH, "rb") as f:
                    self.scaler = pickle.load(f)
                with open(CLASS_NAMES_PATH, "r") as f:
                    self.class_names = json.load(f)
                print(f"✅ Kidney model loaded: {len(self.class_names)} classes - {self.class_names}")
            else:
                print("⚠️  Kidney model not found. Please run: python train_kidney_model.py")
        except Exception as e:
            print(f"❌ Error loading kidney model: {e}")
            self.model = None

    def is_ready(self):
        """Check if the model is loaded and ready."""
        return self.model is not None and self.class_names is not None and self.scaler is not None

    def classify(self, image_path):
        """
        Classify a kidney CT image.

        Returns:
            dict with keys: predicted_class, confidence, all_scores
        """
        if not self.is_ready():
            return {"error": "Model not loaded. Run train_kidney_model.py first."}

        features = extract_features(image_path)
        if features is None:
            return {"error": "Could not read image file."}

        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Predict probabilities
        probabilities = self.model.predict_proba(features_scaled)[0]
        predicted_idx = np.argmax(probabilities)
        predicted_class = self.class_names[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        all_scores = {
            self.class_names[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(len(self.class_names))
        }

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "all_scores": all_scores
        }

    @staticmethod
    def get_medical_suggestions(predicted_class, confidence):
        """
        Generate AI-based medical suggestions based on classification results.

        Returns:
            dict with severity, description, treatments, follow_up, lifestyle, disclaimer
        """
        suggestions = {
            "Normal": {
                "severity": "Low",
                "severity_color": "#22c55e",
                "description": (
                    "The CT scan appears to show a normal kidney with no significant abnormalities detected. "
                    "The renal parenchyma shows normal density and architecture."
                ),
                "treatments": [
                    "No treatment required — kidney appears healthy",
                    "Continue routine health screenings as recommended",
                    "Maintain adequate hydration (2–3 liters/day)",
                ],
                "follow_up": [
                    "Routine annual health check-up",
                    "Repeat imaging only if symptoms develop",
                    "Monitor blood pressure regularly",
                ],
                "lifestyle": [
                    "Maintain a balanced diet low in sodium",
                    "Stay physically active with regular exercise",
                    "Limit alcohol consumption and avoid smoking",
                    "Keep blood sugar levels in check",
                ],
            },
            "Cyst": {
                "severity": "Low to Moderate",
                "severity_color": "#f59e0b",
                "description": (
                    "The CT scan indicates the presence of a renal cyst. Simple kidney cysts are very common "
                    "and usually benign. However, complex cysts may require further evaluation using the "
                    "Bosniak classification system to rule out malignancy."
                ),
                "treatments": [
                    "Simple cysts (Bosniak I–II): Typically no treatment needed, observation only",
                    "Symptomatic cysts: Percutaneous aspiration or sclerotherapy",
                    "Complex cysts (Bosniak III–IV): Surgical evaluation may be recommended",
                    "Pain management with analgesics if symptomatic",
                ],
                "follow_up": [
                    "Follow-up ultrasound or CT in 6–12 months to monitor cyst size",
                    "Bosniak classification assessment for complex-appearing cysts",
                    "Renal function tests (BUN, Creatinine, GFR)",
                    "Urinalysis to check for hematuria",
                ],
                "lifestyle": [
                    "Stay well-hydrated to support kidney function",
                    "Low-sodium diet to reduce fluid retention",
                    "Avoid contact sports if cyst is large",
                    "Report any flank pain, blood in urine, or fever immediately",
                ],
            },
            "Tumor": {
                "severity": "High",
                "severity_color": "#ef4444",
                "description": (
                    "The CT scan suggests the presence of a renal mass/tumor. This finding requires urgent "
                    "medical evaluation. Renal cell carcinoma (RCC) accounts for approximately 90% of kidney "
                    "cancers. Early detection and treatment significantly improve outcomes."
                ),
                "treatments": [
                    "Partial nephrectomy: Surgical removal of the tumor preserving kidney function (preferred for small tumors <4cm)",
                    "Radical nephrectomy: Complete kidney removal for larger tumors",
                    "Ablation therapy (cryoablation/radiofrequency) for small tumors in non-surgical candidates",
                    "Targeted therapy: Sunitinib, Pazopanib, or Cabozantinib for advanced/metastatic RCC",
                    "Immunotherapy: Nivolumab + Ipilimumab combination for intermediate/poor-risk metastatic RCC",
                    "Active surveillance: For small renal masses (<2cm) in elderly or comorbid patients",
                ],
                "follow_up": [
                    "URGENT: Consult urologist/oncologist within 1–2 weeks",
                    "Contrast-enhanced CT or MRI for detailed characterization",
                    "Chest CT to evaluate for pulmonary metastases",
                    "Complete metabolic panel and renal function tests",
                    "Biopsy may be recommended for tissue diagnosis",
                    "Staging workup (TNM staging) to guide treatment",
                ],
                "lifestyle": [
                    "Seek emotional support — counseling or support groups",
                    "Maintain nutritious diet rich in fruits and vegetables",
                    "Moderate physical activity as tolerated",
                    "Avoid smoking — increases RCC risk and worsens outcomes",
                    "Follow oncology team's recommendations closely",
                ],
            },
            "Stone": {
                "severity": "Moderate",
                "severity_color": "#f97316",
                "description": (
                    "The CT scan indicates the presence of kidney stones (nephrolithiasis). "
                    "Kidney stones are solid mineral deposits that can cause significant pain and "
                    "urinary tract complications. Size, location, and composition guide treatment decisions."
                ),
                "treatments": [
                    "Small stones (<5mm): Conservative management with hydration and pain control (alpha-blockers like Tamsulosin)",
                    "Medium stones (5–10mm): Extracorporeal Shock Wave Lithotripsy (ESWL)",
                    "Large stones (>10mm): Ureteroscopy with laser lithotripsy",
                    "Staghorn calculi: Percutaneous Nephrolithotomy (PCNL)",
                    "Pain management: NSAIDs (Ketorolac, Ibuprofen) or opioids for acute renal colic",
                ],
                "follow_up": [
                    "24-hour urine analysis for stone-forming risk factors",
                    "Stone composition analysis if passed or surgically removed",
                    "Follow-up imaging in 4–6 weeks to confirm stone passage",
                    "Renal function tests (Creatinine, BUN)",
                    "Urine culture if infection suspected",
                ],
                "lifestyle": [
                    "Increase fluid intake to at least 2.5–3 liters/day",
                    "Reduce sodium intake (<2,300 mg/day)",
                    "Limit animal protein consumption",
                    "Reduce oxalate-rich foods (spinach, chocolate, nuts) if calcium oxalate stones",
                    "Increase citrate intake (lemon water)",
                    "Avoid excessive vitamin C supplementation",
                ],
            },
        }

        result = suggestions.get(predicted_class, suggestions["Normal"])
        result["predicted_class"] = predicted_class
        result["confidence"] = confidence
        result["disclaimer"] = (
            "⚠️ DISCLAIMER: This is an AI-assisted analysis for educational and screening purposes only. "
            "It is NOT a substitute for professional medical diagnosis. Always consult a qualified "
            "healthcare provider (urologist/nephrologist) for proper evaluation and treatment planning."
        )

        return result


# Singleton instance
_analyzer_instance = None


def get_analyzer():
    """Get or create the singleton KidneyAnalyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = KidneyAnalyzer()
    return _analyzer_instance
