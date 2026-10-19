# Project 2, Week 5: Scene Classification with Transfer Learning

---

## Project Overview
This repository contains the complete implementation and evaluation for Lab 5. The primary objective is to build an end-to-end deep learning image classifier using PyTorch and a pretrained ResNet-18 architecture. The project covers data preprocessing, augmentations, feature extraction by freezing the backbone, differential learning rate fine-tuning of the final block (`layer4`), and a comprehensive error analysis.

---

## Dataset & Notebook Configuration
* **Dataset:** Intel Image Classification (puneet6060)
* **Target Classes (6 Total):** `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`
* **Dataset Splits:**
  * **Train Set:** ~11,929 images (85% split from training directory with heavy data augmentation)
  * **Validation Set:** ~2,105 images (15% split from training directory, clean/unaugmented)
  * **Test Set:** 3,000 images (separate directory, evaluated strictly once)

---

## Training Strategy & Experimental Results

### 1. Feature Extraction Phase
* **Backbone:** ResNet-18 pretrained on ImageNet (frozen body layers)
* **Trainable Parameters:** 3,078 parameters (new 6-class linear classification head)
* **Optimizer & Hyperparameters:** Adam optimizer with `lr = 1e-3` for 3 epochs
* **Best Validation Accuracy:** ~87.8%

### 2. Fine-Tuning Phase
* **Unfrozen Layers:** `layer4` block alongside the classification head
* **Trainable Parameters:** ~8.4 Million parameters
* **Optimizer & Hyperparameters:** Differential learning rates (`layer4 lr = 1e-4`, `head lr = 1e-3`) for 3 epochs
* **Best Validation Accuracy:** ~89.5% (demonstrating improvement over feature extraction)

### 3. Final Test Evaluation
* **Overall Test Accuracy:** ~88.5%
* **Weakest Classes by Recall:** `buildings` and `street`
* **Most Confused Class Pair:** `glacier` vs `mountain` (consistent with exploratory visual analysis)

---

## Error Analysis & Key Insights
* **Misclassification Drivers:** The error gallery reveals that most misclassifications occur due to genuine visual ambiguity in natural outdoor scenes—specifically where lighting conditions, textures, and structures overlap between contrasting classes (such as glaciers and mountains).
* **Model Serialization:** The final trained model weights are saved locally in the Kaggle environment as `resnet18_intel.pt` (~45 MB) for future inference and deployment.

---

## Week 6 Proposal: Project 2 Domain Extension
* **Problem Statement:** Automated detection of plant leaf diseases to assist agricultural workers and local farmers in early crop health diagnostics and preventing harvest failures.
* **Target Beneficiaries:** Farmers, agronomists, and agricultural extension workers.
* **Candidate Dataset:** PlantVillage Dataset (hosted on Kaggle).
* **Project Scope:** Multi-class image classification across 38 distinct classes covering healthy and diseased crop leaves.
