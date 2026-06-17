# ASL Sign Detection using ResNet50

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](#)
[![Model](https://img.shields.io/badge/Model-ResNet50-brightgreen.svg)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)](#)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-purple.svg)](#)
[![Batch Size](https://img.shields.io/badge/Batch%20Size-16-red.svg)](#)
[![Status](https://img.shields.io/badge/Status-Complete-green.svg)](#)

An end-to-end deep learning repository for American Sign Language (ASL) alphabet recognition utilizing a transfer learning approach built on top of the ResNet50 architecture.

---

## 📌 Project Overview & Objectives

* **Core Goal:** Develop a robust computer vision classification pipeline capable of translating static hand gestures representing the ASL alphabet (A-Z) into text characters.
* **Challenge Addressed:** Real-world hand gestures exhibit severe variations in lighting conditions, camera angles, hand orientation, and backgrounds. This project leverages deep residual features to maintain translation consistency.
* **Application Impact:** Forms the architectural backbone for assistive translation applications, bridging communication gaps for the deaf and hard-of-hearing community.

---

## 📊 Dataset & Validation Protocol

* **Target Classes:** 26 separate categorical classes mapped directly to the individual alphabet letters (A-Z).
* **Data Organization:** Organized into explicit `asl_alphabet_train` and `asl_alphabet_test` directories.
* **Validation Split Strategy:** Implemented a clean **80/20 Train-Validation split** via `ImageDataGenerator(validation_split=0.2)`. 20% of the training data is fully isolated from parameter optimization to ensure unbiased performance estimation. 
* **Note on Cross-Validation:** Standard K-Fold Cross-Validation was bypassed due to the heavy computational overhead and resource-intensive training times typical of large-scale deep image data.

---

## 🛠️ Deep Learning Architecture & Pipeline

### 1. Data Processing Lifecycle (`data_loader.py`)
* **Deduplication:** Incorporates `remove_duplicates_from_folder()` to purge duplicate image profiles, preventing validation data leakage and artificial score inflation.
* **Dynamic Augmentation:** Leverages `ImageDataGenerator` scaling configurations to inject structural changes (rotations, brightness shifts) to simulate environmental real-world variables.

### 2. Neural Network Design (`model_builder.py`)
* **Base Feature Extractor:** Employs **ResNet50** pre-trained on ImageNet ($1\text{M}+$ images). The deep convolutional structures are locked to draw low-level primitives (edges, shapes) and mid-level textures without destroying weight states.
* **Custom Classification Head:** Integrates a dense feature reduction funnel optimized via manual parameter grid tests:
  * Feature Layer Dropout: `0.4` 
  * Dense Structural Layer: `448` Units
  * Core Classification Dropout: `0.3` 
  * Final Softmax Output Layer: 26 target units corresponding to the alphabet index mappings.

---

## ⚙️ Hyperparameter Summary

| Hyperparameter | Choice / Value | Rationale |
| :--- | :--- | :--- |
| **Input Tensor Shape** | `(224, 224, 3)` | Expected native image size for ResNet50 input vectors. |
| **Optimizer** | `Adam` | Adaptive gradient step calculation for smoother convergence. |
| **Learning Rate** | `0.0005` (Tuned) | Manually cut from `0.001` to steady weight step adjustments. |
| **Loss Function** | `categorical_crossentropy` | Standard optimization loss for multi-class labels. |
| **Batch Size** | `16` | Constrained window to fit memory maps on 8GB RAM limits. |
| **Epoch Limit** | `15` | Extended parameter cycle for model stabilization. |

---

## 📈 Model Evaluation Strategy & Metrics

A multi-dimensional approach was integrated to evaluate performance beyond raw classification rates:
* **Categorical Accuracy:** Tracks the total global conversion rate across standard profiles.
* **Top-K Accuracy (`top_k_categorical_accuracy`):** Validates if the proper letter selection appears within the model's highest probability scores, crucial when distinguishing between highly similar gestures.
* **Precision, Recall, & F1-Score:** Extracted via `classification_report()` to identify performance drops in minor or imbalanced classes.
* **Active Optimization Callbacks:** Combines `EarlyStopping` (halting degradation runs) with `ReduceLROnPlateau` to automatically scale down the learning rate if the validation loss hits an ongoing plateau.

---

## 🔬 Limitations, Observations & Future Scope

### Key Observations
* **Optimizer Stability:** The introduction of the `ReduceLROnPlateau` callback successfully rescued validation loss volatility midway through the training runs.
* **Overfitting Prevention:** High dropout rules combined with `EarlyStopping` reliably restricted the model from memorizing the local dataset noise.

### Project Limitations
* **Visual Overlap:** Certain letters (such as **"M" and "N"**) show high structural hand shape similarity, causing minor cross-class confusion.
* **Frozen Base Constraint:** Keeping the primary ResNet50 backplane locked protected resource footprints but limited specific deep-layer fine-tuning capabilities.

### Suggested Enhancements
1. **Partial Unfreezing:** Unfreeze the top layers of the ResNet50 architecture for high-level domain adaptation fine-tuning.
2. **Architecture Alternatives:** Test more computationally lightweight, modern networks like `EfficientNet` for faster convergence.
3. **Training Expansion:** Run the optimized training profile across 20-25 epochs if hardware or GPU acceleration becomes available.

---

## 📂 Project Structure

```text
├── .gitignore               # Excludes datasets, tuning logs, and large model binaries
├── .EADME.md                # Project documentation
├── class_indices.json       # Structural metadata binding integer indices to A-Z labels
├── data_loader.py           # Preprocessing pipeline, duplicate cleaner, and image loader
├── model_builder.py         # Defines the customized ResNet50 architecture and tuner layout
└── train_model.py           # Primary runtime script orchestrating loops, callbacks, and metrics
