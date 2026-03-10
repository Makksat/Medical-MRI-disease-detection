# Medical MRI Disease Detection with MONAI

A modular neuroimaging project for **3D MRI disease detection and lesion segmentation** using **MONAI** and **PyTorch**. The repository currently includes centralized learning baselines for:

- **Alzheimer-related cognitive impairment proxy** on **OASIS Cross-Sectional**
- **Parkinson’s disease vs healthy control** on **OpenNeuro ds005892**
- **Multiple sclerosis lesion segmentation** baseline on **MSSEG 2016**

The longer-term internship direction is to extend the same framework to **federated learning** and **interpretability** across multiple neurological disorders.

## Project status

This repository already contains working baseline pipelines and result artifacts. At this stage, it should be presented as a **baseline research repository** rather than a finished production system.

### Current baseline status

| Task | Dataset | Model | Main metric | Status |
|---|---|---:|---:|---|
| Cognitive impairment proxy | OASIS Cross-Sectional | 3D DenseNet121 | Validation AUC ~0.94 | Baseline complete |
| Parkinson vs control | OpenNeuro ds005892 | 3D DenseNet121 | Validation AUC ~0.89, Accuracy ~0.91 | Baseline complete |
| MS lesion segmentation | MSSEG 2016 | 3D U-Net (4-channel) | Mean 3-fold Dice ~0.0333 | Baseline complete, needs improvement |

## Objectives

- Build a reusable **3D MRI deep learning pipeline** with MONAI
- Compare classification performance across neurological disease datasets
- Establish a working **MS lesion segmentation baseline**
- Add **explainability** tools such as occlusion sensitivity and Grad-CAM
- Extend the project toward **federated learning** for distributed medical imaging research
- Prepare a reproducible technical foundation for the internship report and presentation

## Datasets

### 1) OASIS Cross-Sectional
- **Task:** cognitive impairment detection proxy
- **Target definition:** `CDR > 0`
- **Modality:** T1-weighted structural MRI
- **Use in this repo:** 3D classification baseline

### 2) OpenNeuro ds005892
- **Task:** Parkinson’s disease vs healthy control classification
- **Label definition:**
  - `Control = 0`
  - `PD-NC / PD-MCI = 1`
- **Modality:** T1-weighted structural MRI
- **Use in this repo:** 3D classification baseline with Grad-CAM visualization

### 3) MSSEG 2016
- **Task:** multiple sclerosis lesion segmentation
- **Modalities:** FLAIR, T1, T2, DP
- **Use in this repo:** 4-channel 3D segmentation baseline with single split and 3-fold cross-validation

## Methods

### Classification pipeline
- **Framework:** MONAI + PyTorch
- **Backbone:** 3D DenseNet121
- **Input shape:** 128 × 128 × 128
- **Loss:** CrossEntropyLoss with class weighting
- **Optimizer:** AdamW
- **Training:** mixed precision (AMP)
- **Evaluation:** ROC AUC, accuracy, confusion matrix, training curves
- **Interpretability:** occlusion sensitivity and Grad-CAM

### Segmentation pipeline
- **Framework:** MONAI + PyTorch
- **Backbone:** 3D U-Net
- **Input channels:** 4 MRI modalities (FLAIR, T1, T2, DP)
- **Task:** voxel-wise lesion segmentation
- **Evaluation:** Dice score, validation curves, prediction visualization, 3-fold cross-validation

## Results

### OASIS baseline
- Validation AUC: **~0.94**
- Includes ROC curve, confusion matrix, training curves, and occlusion sensitivity

### Parkinson baseline
- Validation AUC: **~0.89**
- Validation accuracy: **~0.91**
- Includes ROC curve, confusion matrix, training curves, and Grad-CAM

### MSSEG baseline
Single-split baseline:
- Best validation Dice: **~0.0297**

3-fold cross-validation:
- Fold 1 best Dice: **0.0105**
- Fold 2 best Dice: **0.0605**
- Fold 3 best Dice: **0.0288**
- Mean Dice: **0.0333**
- Std Dice: **0.0207**

Interpretation:
- The MSSEG pipeline now runs **end-to-end**
- The current segmentation quality is **weak**, so this should be treated as a **baseline result**
- The next iteration should focus on loss design, threshold tuning, and sampling strategy

## Repository structure

```text
Medical-MRI-disease-detection/
├── docs/                      # project notes, internship planning, future roadmap
├── figures/                   # plots, CAMs, ROC curves, prediction examples
├── notebooks/                 # Colab / Jupyter notebooks
├── report/                    # report drafts and supporting material
├── results/                   # CSV, JSON, and lightweight result artifacts
├── src/                       # reusable Python code (transforms, utils, training helpers)
├── .gitignore
├── README.md
└── requirements.txt
```

## Recommended repository cleanup

To keep the repository readable:

- Keep **datasets and model weights out of Git**
- Keep only **lightweight result artifacts** in `results/` and `figures/`
- Use `src/` for code that you want to reuse across notebooks
- Store the cleaned baseline notebook in `notebooks/`
- Put methodology notes and internship planning into `docs/`

## Suggested files to keep in GitHub now

### In `notebooks/`
- `oasis_parkinson_msseg_baseline.ipynb`

### In `results/`
- `msseg_cv_results.csv`
- `msseg_cv_summary.json`
- lightweight summary CSV/JSON files for OASIS and Parkinson if available

### In `figures/`
- ROC curves
- confusion matrices
- Grad-CAM examples
- `msseg_prediction_example.png`
- `msseg_val_dice.png`

### Do not upload
- raw MRI datasets
- extracted dataset folders
- `.nii` / `.nii.gz` data
- very large checkpoints unless there is a strong reason
- temporary Colab outputs and caches

## Reproducibility

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

Most experiments were developed in **Google Colab**, so some paths may need to be adapted for a local environment.

## Roadmap

### Near-term
- Clean notebooks and move reusable code into `src/`
- Improve MSSEG baseline with better loss functions
- Add a compact experiment log
- Prepare polished report figures and tables

### Internship extensions
- Add **federated learning** experiments with Flower or NVFlare
- Compare centralized vs federated setups
- Expand interpretability for classification and segmentation
- Standardize preprocessing and evaluation across all diseases

## Limitations

- The Alzheimer task currently uses a **proxy label** rather than a full diagnostic pipeline
- The Parkinson and MS pipelines are still in the **baseline stage**
- The repository is notebook-heavy and should gradually be refactored into reusable modules

## Citation and acknowledgment

If you use this repository structure or ideas, please cite the original datasets and MONAI in your report/presentation.

## Contact

Maintained by **Maksat Kaparov** as part of an internship project on MRI-based neurological disease analysis.
