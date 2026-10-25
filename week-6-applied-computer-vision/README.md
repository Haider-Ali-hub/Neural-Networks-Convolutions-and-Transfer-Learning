# Week 6: Applied Computer Vision — Chest X-Ray (Pneumonia) Classification

**Course:** Introduction to Applied AI  
**Program:** BS Electronics (7th Semester), Quaid-i-Azam University  
**Author:** **Haider Ali Asghar**  
**Roll No:** 04102313042  

---

## 📌 Project Overview
This repository contains the complete end-to-end implementation and lab documentation for **Lab 6: Applied Computer Vision in Your Own Domain**. The project evaluates transfer learning, data leakage prevention, class imbalance handling, decision thresholding, and shortcut analysis on the medical **Chest X-Ray Images (Pneumonia)** dataset.

---

## 🛠️ System & Environment Setup
* **Environment:** Kaggle Notebook
* **Hardware Accelerator:** NVIDIA T4 GPU
* **Backbone Architecture:** ResNet18 (Pretrained on ImageNet)
* **Input Specifications:** Grayscale (`GRAYSCALE = True`), resized to standard dimensions ($224 \times 224$)
* **Data Splitting Strategy:** 6-fold `StratifiedGroupKFold` to prevent patient-level data leakage

---

## **Datasheet and Model Card**


**Who benefits:** Radiologists and clinical triage teams in pediatric screening settings. **Why it matters:** Early and accurate diagnosis of pneumonia reduces diagnostic delays, prevents critical treatment delays, and minimizes unnecessary antibiotic usage or diagnostic harms.

**Dataset:** Chest X-Ray Images (Pneumonia) by Paul Mooney (Kaggle dataset link); 5,216 total training images, 2 classes (NORMAL, PNEUMONIA); labeled by expert radiologist evaluations.

**Data audit:** Imbalance ratio 2.9; exact duplicates 62 files (across 30 groups); test images with a near-duplicate in train 0 (out of 624 test images); split by patient (patient-level grouping using `StratifiedGroupKFold`), 0 shared groups between train and validation.

**Imbalance handling:** Weighted loss (weighted CrossEntropyLoss), weights: NORMAL = 1.935, PNEUMONIA = 0.674.

**Results (test, used once):** Macro-F1 0.878; weakest per-class recall: NORMAL recall = 0.726; for binary: sensitivity 0.990 and specificity 0.795 at threshold 0.803 (set on val for target recall $\ge 0.95$).

**Grad-CAM:** The model looks at central pulmonary fields for true positives, but also focuses on upper sternum structures, outer abdominal borders, and peripheral 'R' orientation letters on false positives; shortcut risk: shortcut suspected (checked for dependence on peripheral text markers and non-diagnostic anatomical borders).

**Limitations:** Pediatric single-center retrospective dataset, potential demographic/imaging equipment bias, sensitivity to peripheral artifacts. Not for autonomous medical diagnosis or standalone clinical decision-making without specialist supervision.

---

## 📊 Dataset Audit & Leakage Analysis

* **Class Distribution:**
  * Total Training Images: **5,216** (NORMAL: **1,341**, PNEUMONIA: **3,875**)
  * Imbalance Ratio: **2.9** (Pneumonia cases are ~2.9x more frequent than Normal cases)
* **Duplicates Audit:**
  * Exact MD5 Duplicates: **62 files** across **30 duplicate groups**
  * Near-Duplicates: **0 test images** breached the strict fingerprint threshold (`d_min < 0.10`) against train
* **Split Counts:**
  * **Train:** 4,354 images | **Validation:** 862 images | **Test:** 624 images
  * **Group Isolation:** 0 groups shared between train and validation sets, ensuring strictly non-overlapping evaluation.

---

## ⚙️ Imbalance Handling & Loss Function
* **Class Weights Calculation:**  
  Calculated using inverse class frequency $W = \frac{N}{K \times n_c}$:
  * **NORMAL:** `1.935` (Calculated as $\frac{4354}{2 \times 1125}$)
  * **PNEUMONIA:** `0.674` (Calculated as $\frac{4354}{2 \times 3229}$)
* **Data Augmentations:**
  * Applied conservative spatial/intensity transforms: `RandomResizedCrop` (scale 0.8–1.0), `RandomRotation` ($\pm 10^\circ$), and subtle `ColorJitter` (brightness/contrast=0.15).
  * *Rationale:* Preserves anatomical structure and avoids distorting medical diagnostic indicators.

---

## 📈 Two-Phase Training Performance

The model was trained using a two-phase strategy to adapt features to the medical imaging domain, evaluated via **Macro-F1**:

| Phase | Description | Epochs | Best Validation Macro-F1 |
| :--- | :--- | :---: | :---: |
| **Phase 1** | Head-Only Training (Classifier head unfrozen) | 2 | **0.875** |
| **Phase 2** | Fine-Tuning (Unfrozen classifier head + `layer4`) | 3 | **0.964** |

*Fine-tuning the final residual block improved validation Macro-F1 significantly from `0.875` to `0.964`.*

---

## 🎯 Test Set Evaluation & Decision Thresholding

* **Default Argmax Evaluation (Test Set):**
  * **Test Macro-F1:** `0.878`
  * **Weakest Class:** `NORMAL` (F1-score: `0.835`, Sensitivity: `0.726`)
* **Validation Threshold Optimization:**
  * Target Recall: $\ge 0.95$
  * Optimal Threshold Selected on Validation: **`0.803`**
* **Final Test Metrics at Selected Threshold (`0.803`):**
  * **Test AUC:** `0.975`
  * **Sensitivity (Recall):** **`0.990`** (386 TP / 4 FN) — *Target achieved*
  * **Specificity:** **`0.795`** (186 TN / 48 FP)
  * **PPV at Test Prevalence (0.625):** `0.889`
  * **PPV at Realistic Prevalence (0.050):** `0.203`

> **Clinical Interpretation:** While test sensitivity is exceptional (`99.0%`), dropping prevalence to a realistic 5% screening setting causes PPV to fall from `88.9%` to `20.3%` due to false positive weighting, demonstrating the necessity of secondary clinical verification.

---

## 🔍 Grad-CAM Shortcut Analysis

* **Observations:** Heatmaps for correctly classified normal cases focused on legitimate central mediastinum and pulmonary regions. However, false positive activations highlighted non-pathological features such as upper sternum structures, outer abdominal borders, and peripheral anatomical marker letters ('R').
* **Overall Verdict:** **`Shortcut Suspected`**
* **Proposed Solution:** Implement border/corner cropping (10–15% margin removal) to isolate lung fields and remove text tags before retraining.

---

## 📁 Repository Structure
```text
.
├── week-6-applied-computer-vision/
│   ├── README.md                                    # Comprehensive lab documentation
│   └── Week 6 – Applied Computer Vision.ipynb       # Executed Kaggle notebook with outputs
