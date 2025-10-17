# MathProj Textile Failure Prediction

Comprehensive predictive-maintenance study for automated looms. The project explores multiple modelling paradigms—Tensor Dynamic Mode Decomposition (DMD), Higher-Order SVD (HO-SVD), Support Vector Machines (SVM) from scratch, and Reproducing Kernel Hilbert Space (RKHS) classifiers—to anticipate textile machine failures using multivariate telemetry.

## Repository layout

```text
.
├── DMD Tensor/               # Tensor-DMD feature extraction + classifier pipeline
├── HO-SVD/                   # Higher-order SVD subspace classification workflow
├── RKHS/                     # Random kitchen sinks / RKHS-based classifier
├── SVM/                      # From-scratch SVM experiments and tuned model artefacts
├── textile_machine_data.csv  # Source telemetry dataset
├── requirements.txt          # Common Python dependencies
└── README.md                 # (this file)
```

Each subfolder contains a `README.md` or inline documentation for method-specific details, CLI flags, and artefacts.

## Prerequisites

- Python 3.10+
- Recommended: create a virtual environment (`python -m venv .venv`) and install dependencies via `pip install -r requirements.txt`.
- Optional: MiKTeX/TeX Live if you plan to compile the accompanying IEEE-style report.

## Quick start

```powershell
# Activate environment (Windows PowerShell)
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt

# Run the Tensor-DMD pipeline with hyperparameter tuning and artefact export
python "DMD Tensor/DMD_Tensor.py" --data-path textile_machine_data.csv --grid-search --compare-classifiers --output-dir artifacts

# Execute the SVM-from-scratch workflow
python SVM/svm.py

# Run HO-SVD subspace classifier
python "HO-SVD/HO-SVD.py"
```

All pipelines default to the shared dataset and produce metrics plus artefacts (confusion matrices, ROC curves, trained models) under their respective `artifacts/` directories.

## Pipelines at a glance

| Pipeline | Purpose | Key outputs |
|----------|---------|-------------|
| Tensor-DMD | Transforms sliding windows into DMD features (eigenvalues, modal energies, reconstruction error) and feeds scikit-learn classifiers. Supports grid search, cross-validation, ROC/feature-influence artefacts. | `DMD Tensor/artifacts/…` with metrics, plots, joblib models. |
| HO-SVD | Builds class-specific higher-order SVD bases and classifies windows via subspace reconstruction error. | Reconstruction-error metrics, confusion matrices, plots under `HO-SVD/`. |
| SVM | Custom SVM pipeline with iterative feature sanitisation, PCA, and extensive hyperparameter sweeps (including class-weighted GridSearchCV). | `SVM/` outputs accuracy reports, tuned model (`best_svc.joblib`), feature diagnostics. |
| RKHS | Random Kitchen Sinks approximations for kernel methods, sharing preprocessing and evaluation protocols. | RKHS pipeline artefacts under `RKHS/`. |

## Dataset description

- `textile_machine_data.csv`: multivariate sensor readings (temperature, vibration, power consumption, etc.) enriched with machine type and `Failure` labels.
- Preprocessing steps (common across pipelines): drop identifiers/timestamps, impute missing values, standardise continuous features, one-hot encode categories, and optionally apply PCA.
- Sliding-window tensorisation ensures each sample covers a temporal history with the label at the window’s terminal timestamp.

## Reproducing results

1. Install requirements and run the desired pipeline(s) as shown in **Quick start**.
2. Inspect generated artefacts:
   - `artifacts/<classifier>/metrics.json` for scalar metrics.
   - Plot images (`confusion_matrix.png`, `roc_curve.png`, `decision_boundary.png`).
   - Feature importance tables (`feature_influence.csv`) and trained models (`trained_model.joblib`).
3. Use the stored JSON/CSV outputs to populate the IEEE-style report or presentation materials.

## Reporting and presentation

- The LaTeX manuscript (IEEEtran format) covers: Abstract, Introduction, Dataset, Methodology (DMD, HO-SVD, SVM, RKHS), Experiments, Conclusion, and Acknowledgments.
- Slides focus on: telemetry overview, PCA + tensorisation pipeline, DMD feature extraction, classifier comparison, key metrics, and interpretability artefacts.

## Contributing

This is a collaborative college project. Please coordinate changes with the team before merging, and keep artefact directories clean (commit only curated outputs or update `.gitignore` for bulky files).

---

For questions or reproducibility issues, reach out to the project maintainers via the contact information in the manuscript author section.
