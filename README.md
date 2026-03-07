# Medical MRI Disease Detection with MONAI

## Overview
This project develops a reusable 3D deep learning pipeline for MRI-based disease detection using MONAI and PyTorch.  
The pipeline is currently applied to:
- OASIS cross-sectional MRI for cognitive impairment detection (CDR > 0)
- OpenNeuro ds005892 for Parkinson’s disease vs healthy control classification

The long-term goal is to extend the same framework to Multiple Sclerosis and Federated Learning.

## Objectives
- Build a reproducible 3D MRI classification pipeline
- Evaluate performance using ROC AUC, accuracy, confusion matrix, and training curves
- Add interpretability using occlusion sensitivity and Grad-CAM
- Reuse the same pipeline across multiple neurological diseases
- Later compare centralized and federated learning settings

## Datasets
### 1. OASIS Cross-Sectional
- Task: cognitive impairment detection (proxy)
- Label rule: CDR > 0
- Modality: T1-weighted MRI

### 2. OpenNeuro ds005892
- Task: Parkinson’s disease vs healthy control
- Label rule:
  - Control = 0
  - PD-NC / PD-MCI = 1
- Modality: T1-weighted MRI

## Model
- MONAI DenseNet121 (3D)
- Input size: 128 x 128 x 128
- Loss: CrossEntropyLoss with class weights
- Optimizer: AdamW
- Mixed precision training (AMP)

## Results
### OASIS
- Validation AUC: ~0.94
- Includes ROC, confusion matrix, curves, occlusion sensitivity

### Parkinson (ds005892)
- Validation AUC: ~0.89
- Validation Accuracy: ~0.91
- Includes ROC, confusion matrix, curves, Grad-CAM

## Repository Structure
- `notebooks/` — training notebooks
- `results/` — CSV and JSON outputs
- `figures/` — plots and explainability images
- `report/` — methodology and results notes

## Next Steps
- Add Multiple Sclerosis dataset
- Refactor shared code into reusable Python modules
- Add federated learning experiments
- Prepare final internship report and presentation
