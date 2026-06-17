# ASL Sign Detection using ResNet50

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)](#)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-purple.svg)](#)
[![CPU](https://img.shields.io/badge/CPU-Intel%20i5--13420H-blue.svg)](#)
[![RAM](https://img.shields.io/badge/RAM-8GB-lightgrey.svg)](#)
[![Batch Size](https://img.shields.io/badge/Batch%20Size-16-red.svg)](#)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](#)

A deep learning project focused on American Sign Language (ASL) alphabet recognition using a transfer learning approach with the ResNet50 architecture.

---

## 📌 Project Overview & Objectives

* [cite_start]**Goal:** Create an image classification pipeline capable of recognizing hand gestures representing the American Sign Language alphabets (A-Z)[cite: 3].
* [cite_start]**Core Tasks:** Handle variations in real-world image environments such as lighting conditions, hand orientation, and custom backgrounds [cite: 5] [cite_start]to extract reliable classification features[cite: 12].
* **Application:** Bridges communication gaps by creating accessible visual tools for translating static hand signals into textual alphabet characters.

---

## 💻 Hardware Configuration
[cite_start]Due to specific physical resource limits[cite: 112], the model was compiled and executed using the following localized environment parameters:
* **Processor:** Intel i5-13420H CPU
* **Memory:** 8GB RAM
* [cite_start]**Execution Boundary:** Fixed optimization window bounded to 15 epochs[cite: 22, 34].

---

## 📊 About the Dataset

* [cite_start]**Classes:** 26 separate classes representing the letters of the alphabet (A-Z)[cite: 3].
* [cite_start]**Format:** Raw structural image directories organized into `asl_alphabet_train` and `asl_alphabet_test`[cite: 101].
* [cite_start]**Validation Split:** Implemented as a clean **80/20 train-validation split** directly within the data pipeline configuration [cite: 92][cite_start], ensuring 20% of the data remains completely unseen during immediate model parameter corrections[cite: 94].

---

## 🛠️ Machine Learning Pipeline & Architecture

### Data Preprocessing & Cleaning
* [cite_start]**Duplicate Removal:** Automatically screens and deletes duplicate sample files to prevent data redundancy or artificial validation score leaks[cite: 20].
* [cite_start]**Augmentation & Scaling:** Uses an `ImageDataGenerator` setup to dynamically apply data augmentations to help the model learn variations in lighting and positioning[cite: 18, 20].

### Why ResNet50?
* [cite_start]**Feature Extraction:** Pre-trained on the ImageNet dataset containing over 1 million images[cite: 11]. [cite_start]The underlying convolutional layers automatically identify general structural markers like edges and hand silhouettes[cite: 12].
* [cite_start]**Overfitting Prevention:** Added custom dense layers on top of the frozen base [cite: 115] [cite_start]with deep Dropout rates ($0.4$ and $0.3$) [cite: 22, 27, 28] [cite_start]and an input scaling dimension of $(224, 224, 3)$ [cite: 22] [cite_start]to manage generalization[cite: 13].

---

## ⚙️ Hyperparameters

| Hyperparameter | Value |
| :--- | :--- |
| **Input Shape** | [cite_start]`(224, 224, 3)` [cite: 22] |
| **Optimizer** | [cite_start]`Adam` [cite: 22] |
| **Learning Rate** | [cite_start]`0.0005` [cite: 22, 29] |
| **Dropout Rates** | [cite_start]`0.4` (Feature layer) / `0.3` (Dense layer) [cite: 22, 27, 28] |
| **Dense Units** | [cite_start]`448` [cite: 22, 26] |
| **Loss Function** | [cite_start]`categorical_crossentropy` [cite: 22] |
| **Batch Size** | [cite_start]`16` [cite: 22] |
| **Epochs** | [cite_start]`15` [cite: 22, 34] |

---

## 📈 Model Evaluation Strategy

[cite_start]Performance is validated through several complementary metrics generated during testing cycles[cite: 18]:
* [cite_start]**Accuracy:** Captures the raw proportion of correctly identified sign letters[cite: 77, 80].
* [cite_start]**Precision & Recall (F1-Score):** Used to verify structural performance balance across potentially unbalanced [cite: 82] [cite_start]or structurally overlapping gesture subsets (e.g., distinguishing visually similar signs like "M" and "N")[cite: 85, 114].
* [cite_start]**Top-K Accuracy:** Evaluates if the actual targets reside within the highest likelihood score categories output by the model[cite: 77, 84].
* [cite_start]**Dynamic Callbacks:** Training includes `EarlyStopping` alongside `ReduceLROnPlateau` to decrease the optimizer's step size automatically if validation progress hits an active bottleneck[cite: 20, 22, 116, 117].

---

## 📂 Project Structure

```text
├── .gitignore               # Explicitly ignores datasets, cache, and heavy model files
├── class_indices.json       # Maps text target labels (A-Z) to model array indexes
├── data_loader.py           # Handles image loading, duplicate removal, and augmentation
├── model_builder 1.py       # Defines the ResNet50 transfer learning structure
└── train_model 1.py         # Handles the execution loops, callbacks, and metrics
