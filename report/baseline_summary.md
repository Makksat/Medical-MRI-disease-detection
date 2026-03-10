# Baseline Experiments Summary

## Project Overview

This project develops deep learning pipelines for **MRI-based neurological disease detection and lesion segmentation** using **PyTorch and MONAI**.

Three baseline tasks were implemented:

- **Alzheimer MRI classification** (OASIS dataset)
- **Parkinson's disease MRI classification** (OpenNeuro datasets)
- **Multiple sclerosis lesion segmentation** (MSSEG 2016 dataset)

The goal of this stage was to build **complete end-to-end pipelines**, including:

- MRI data loading
- preprocessing
- model training
- validation and evaluation
- visualization
- experiment logging

These baselines serve as the foundation for future work such as **federated learning and improved interpretability**.

---

# 1. Alzheimer MRI Classification (OASIS)

## Dataset

The **OASIS dataset** provides structural **T1-weighted MRI scans** used to study brain aging and dementia.

Task:

- Binary classification (disease vs control)

Preprocessing included:

- intensity normalization
- spatial resampling
- tensor conversion using MONAI transforms

## Model

A **3D convolutional neural network** was trained using PyTorch and MONAI.

Configuration:

| Component | Setting |
|----------|--------|
| Framework | PyTorch + MONAI |
| Task | Binary classification |
| Loss | Binary Cross Entropy |
| Optimizer | Adam |
| Input | 3D MRI volumes |

## Interpretability

**Occlusion Sensitivity** was applied to identify brain regions influencing predictions.

## Results

Artifacts generated:

- ROC curves
- confusion matrix
- training curves
- occlusion maps

Stored in:
_figures/oasis/
results/oasis/_


---

# 2. Parkinson's Disease MRI Classification

## Dataset

MRI scans were obtained from **OpenNeuro datasets**.

Task:

- Binary classification  
  Parkinson's patients vs healthy controls.

## Model

The same **3D CNN architecture** was used for classification.

Configuration:

| Component | Setting |
|----------|--------|
| Framework | PyTorch + MONAI |
| Loss | Binary Cross Entropy |
| Optimizer | Adam |
| Input | 3D MRI |

## Interpretability

**Grad-CAM** was used to visualize brain regions contributing to predictions.

## Results

Generated artifacts:

- ROC curves
- confusion matrix
- training curves
- Grad-CAM visualizations

Stored in:

_figures/parkinson/
results/parkinson/_

---

# 3. Multiple Sclerosis Lesion Segmentation (MSSEG 2016)

## Dataset

The **MSSEG 2016 dataset** contains multi-modal MRI scans with expert lesion annotations.

Modalities used:

- FLAIR
- T1
- T2
- DP

These four modalities were combined as **input channels**.

## Model

Segmentation was performed using a **MONAI 3D U-Net**.

Configuration:

| Parameter | Value |
|-----------|------|
| Architecture | 3D U-Net |
| Input channels | 4 |
| Loss | Dice-based |
| Optimizer | Adam |

## Baseline Results

Single training split:

**Best Dice ≈ 0.0297**

3-fold cross-validation:

| Fold | Dice |
|-----|------|
| 1 | 0.0105 |
| 2 | 0.0605 |
| 3 | 0.0288 |

Mean Dice: **0.0333**

Although segmentation performance is low, the goal of this stage was to validate the **full segmentation pipeline**.

Generated artifacts:

- training curves
- prediction examples
- cross-validation statistics

Stored in:
_figures/msseg/
results/msseg/_


---

# Key Achievements

The baseline stage successfully established working pipelines for:

- MRI classification (Alzheimer, Parkinson)
- MRI lesion segmentation (Multiple Sclerosis)

All pipelines support:

- dataset loading
- preprocessing
- model training
- evaluation
- visualization
- experiment logging

---

# Next Steps

Future work will focus on:

### Model improvements

- improved segmentation loss functions
- better patch sampling
- hyperparameter tuning

### Federated Learning

Simulation of **multi-center MRI training without centralized data sharing** using frameworks such as:

- Flower
- NVFlare

### Interpretability

Additional explainability methods:

- Grad-CAM extensions
- occlusion sensitivity
- feature attribution techniques
