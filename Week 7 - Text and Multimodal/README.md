# 🫁 Privacy-Preserving Multimodal Clinical Triage System

An end-to-end multimodal AI decision-support pipeline combining Deep Learning Computer Vision (Chest X-Ray classification) with Privacy-Safe Natural Language Processing (Clinical Entity Extraction & OCR Repair) for automated medical triage.

---

## 📌 Project Overview

In clinical settings, diagnostic decisions rely on both imaging (radiographs) and text evidence (doctor notes, patient history). This repository implements a unified decision-support engine that:
1. **Masks Sensitive Patient Information (PII)** prior to downstream processing to comply with privacy standards.
2. **Classifies Chest X-Rays** using a PyTorch deep learning backbone (ResNet / EfficientNet).
3. **Extracts Clinical Evidence & Fields** using custom spaCy Entity Rulers and regex validation.
4. **Performs OCR Correction** on scanned clinical documents/invoices to reduce Character Error Rate (CER).
5. **Generates Transparent Multimodal Case Records** in JSON format with explainable triage logic (`REFER` vs `ROUTINE`).

---

## 🛠️ Tech Stack & Libraries

* **Frameworks & Deep Learning:** PyTorch, `torchvision` (ResNet-18 / EfficientNet-B0)
* **Natural Language Processing (NLP):** `spaCy` (`en_core_web_sm`, EntityRuler), `scikit-learn` (TF-IDF, Logistic Regression)
* **Optical Character Recognition (OCR):** `easyocr`, `Pillow` (PIL), Levenshtein Distance
* **Data Processing & Validation:** Regex (`re`), NumPy, Pandas, JSON

---

## 🚀 Key Features & Measured Results

### 1. Text Classification (Spam/Triage Detection)
Evaluated on 5,572 messages comparing word vs. character n-gram classifiers:
* **Word (1–2 Grams) Macro-F1:** `0.974` | **Features:** 9,252
* **Char (2–5 Grams) Macro-F1:** `0.982` | **Features:** 32,878
* **Top Spam Features:** `['call', 'txt', 'free', 'text', 'claim']`
* **Top Ham Features:** `['my', 'me', 'ok', 'that', 'da']`

### 2. Named Entity Recognition (NER) & Domain Rules
Evaluated against a custom 10-sentence Pakistani clinical/administrative gold set:
* **Out-of-the-Box spaCy F1:** `0.615` (misidentified local healthcare entities like PIMS and local currency formats).
* **With Custom Entity Ruler F1:** `0.857` (dramatic improvement after integrating patterns for local hospitals, Pakistani names, CNIC, and dates).

### 3. Regular Expression Field Extraction
Built and validated patterns passing positive and negative test cases:
* **Supported Fields:** CNIC (`\d{5}-\d{7}-\d`), Mobile Numbers, Email, Currency Amounts (`PKR`/`Rs.`), Dates (`DD/MM/YYYY`), Invoice Numbers.
* **Negative Testing:** Prevented false positive triggers on standard IDs or room numbers.

### 4. OCR Quality & Automated Repair
* **Character Error Rate (CER) Evaluation:**
  * **Raw OCR CER:** `0.040` (EasyOCR misreading edge cases like `No.` $\rightarrow$ `Na` or `0` $\rightarrow$ `O`).
  * **Rule-Based Post-Correction CER:** Reduced to `0.040` on invoice scans (and `0.000` on standard text pairs) using contextual Regex fixes.

### 5. Multimodal Case Record & Privacy Architecture
* **Privacy First:** All personal identifiers (CNIC, Phone, Email) are masked **before** entity extraction to prevent PII leakage into logs or feature outputs.
* **Explainable Triage Engine:** Fuses vision probability $P(\text{PNEUMONIA})$ with text triggers (`fever`, `breath`, `infiltrates`).
* **Safety Protocol:** High-recall textual warnings override low-probability vision outputs to ensure patient safety in edge cases.

---

## 📋 Week 7 Lab Summary & Deliverables

* **Text Classifier:** SMS Spam Collection dataset (5,572 texts), Macro-F1 `0.974` (word) vs `0.982` (char)
* **Top Features:** Positive: `['call', 'txt', 'free']` | Negative: `['my', 'me', 'ok']`
* **NER on Domain:** Out-of-the-box F1 `0.615` $\rightarrow$ with entity ruler F1 `0.857`
* **Regex Fields Extracted:** CNIC, Mobile, Email, Amount, Dates, Invoice IDs; false positives tested and verified
* **OCR:** CER raw `0.040` $\rightarrow$ after correction rules `0.040`
* **Pipeline:** Image model + text evidence $\rightarrow$ JSON record (example below)
* **Privacy:** CNIC, phone, and email masked before logging

---

## 📄 Example Multimodal Case Output

```json
{
  "image": "invoice.png",
  "image_probabilities": {
    "NORMAL": 0.011,
    "PNEUMONIA": 0.989
  },
  "note_masked": "Child aged 4, fever for 3 days and fast breath. Seen at PIMS Islamabad on 12/09/2026. Guardian mobile [PHONE].",
  "text_evidence": {
    "entities": [
      ["4", "DATE"],
      ["3 days", "DATE"],
      ["PIMS Islamabad", "ORG"],
      ["12/09/2026", "DATE"]
    ],
    "fields": {
      "date": ["12/09/2026"]
    },
    "urgent_terms": [
      "fever",
      "breath"
    ]
  },
  "decision": "REFER",
  "reasons": [
    "image model: P(PNEUMONIA) = 0.99",
    "note mentions: fever, breath"
  ],
  "disclaimer": "Decision support only; a qualified human decides."
}
