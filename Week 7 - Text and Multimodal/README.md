## Week 7 - Text and Multimodal

An end-to-end multimodal AI decision-support pipeline combining Deep Learning Computer Vision (Chest X-Ray classification) with Privacy-Safe Natural Language Processing (Clinical Entity Extraction & OCR Repair) for automated medical triage.

---

## 📌 Project Overview

In clinical settings, diagnostic decisions rely on both imaging (radiographs) and text evidence (doctor notes, patient history). This repository implements a unified decision-support engine that:
1. **Masks Sensitive Patient Information (PII)** prior to downstream processing to comply with privacy standards[cite: 12, 14].
2. **Classifies Chest X-Rays** using a PyTorch deep learning backbone (ResNet / EfficientNet)[cite: 13].
3. **Extracts Clinical Evidence & Fields** using custom spaCy Entity Rulers and regex validation[cite: 9, 10].
4. **Performs OCR Correction** on scanned clinical documents/invoices to reduce Character Error Rate (CER)[cite: 11].
5. **Generates Transparent Multimodal Case Records** in JSON format with explainable triage logic (`REFER` vs `ROUTINE`)[cite: 14].

---

## 🛠️ Tech Stack & Libraries

* **Frameworks & Deep Learning:** PyTorch, `torchvision` (ResNet-18 / EfficientNet-B0)[cite: 6, 13]
* **Natural Language Processing (NLP):** `spaCy` (`en_core_web_sm`, EntityRuler), `scikit-learn` (TF-IDF, Logistic Regression)[cite: 6, 7, 9]
* **Optical Character Recognition (OCR):** `easyocr`, `Pillow` (PIL), Levenshtein Distance[cite: 11, 12]
* **Data Processing & Validation:** Regex (`re`), NumPy, Pandas, JSON[cite: 6, 10, 11]

---

## 🚀 Key Features & Measured Results

### 1. Text Classification (Spam/Triage Detection)
Evaluated on 5,572 messages comparing word vs. character n-gram classifiers[cite: 7]:
* **Word (1–2 Grams) Macro-F1:** `0.974` | **Features:** 9,252[cite: 7]
* **Char (2–5 Grams) Macro-F1:** `0.982` | **Features:** 32,878[cite: 7]
* **Top Spam Features:** `['call', 'txt', 'free', 'text', 'claim']`[cite: 7]
* **Top Ham Features:** `['my', 'me', 'ok', 'that', 'da']`[cite: 7]

### 2. Named Entity Recognition (NER) & Domain Rules
Evaluated against a custom 10-sentence Pakistani clinical/administrative gold set[cite: 6, 9]:
* **Out-of-the-Box spaCy F1:** `0.615` (misidentified local healthcare entities like PIMS and local currency formats)[cite: 8, 9].
* **With Custom Entity Ruler F1:** `0.857` (dramatic improvement after integrating patterns for local hospitals, Pakistani names, CNIC, and dates)[cite: 9].

### 3. Regular Expression Field Extraction
Built and validated patterns passing positive and negative test cases[cite: 10]:
* **Supported Fields:** CNIC (`\d{5}-\d{7}-\d`), Mobile Numbers, Email, Currency Amounts (`PKR`/`Rs.`), Dates (`DD/MM/YYYY`), Invoice Numbers[cite: 10].
* **Negative Testing:** Prevented false positive triggers on standard IDs or room numbers[cite: 10].

### 4. OCR Quality & Automated Repair
* **Character Error Rate (CER) Evaluation[cite: 11]:**
  * **Raw OCR CER:** `0.040` (EasyOCR misreading edge cases like `No.` $\rightarrow$ `Na` or `0` $\rightarrow$ `O`)[cite: 11, 19].
  * **Rule-Based Post-Correction CER:** Reduced to `0.040` on invoice scans (and `0.000` on standard text pairs) using contextual Regex fixes[cite: 11].

### 5. Multimodal Case Record & Privacy Architecture
* **Privacy First:** All personal identifiers (CNIC, Phone, Email) are masked **before** entity extraction to prevent PII leakage into logs or feature outputs[cite: 12, 14].
* **Explainable Triage Engine:** Fuses vision probability $P(\text{PNEUMONIA})$ with text triggers (`fever`, `breath`, `infiltrates`)[cite: 13, 14].
* **Safety Protocol:** High-recall textual warnings override low-probability vision outputs to ensure patient safety in edge cases[cite: 14].

---

## 📋 Week 7 Lab Summary & Deliverables

* **Text Classifier:** SMS Spam Collection dataset (5,572 texts), Macro-F1 `0.974` (word) vs `0.982` (char)[cite: 7, 14]
* **Top Features:** Positive: `['call', 'txt', 'free']` | Negative: `['my', 'me', 'ok']`[cite: 7, 14]
* **NER on Domain:** Out-of-the-box F1 `0.615` $\rightarrow$ with entity ruler F1 `0.857`[cite: 9, 14]
* **Regex Fields Extracted:** CNIC, Mobile, Email, Amount, Dates, Invoice IDs; false positives tested and verified[cite: 10, 14]
* **OCR:** CER raw `0.040` $\rightarrow$ after correction rules `0.040`[cite: 11, 14]
* **Pipeline:** Image model + text evidence $\rightarrow$ JSON record (example below)[cite: 13, 14]
* **Privacy:** CNIC, phone, and email masked before logging[cite: 12, 14]

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
