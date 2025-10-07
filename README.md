# Tensor DMD Textile Failure Prediction

This project implements a **tensor-based Dynamic Mode Decomposition (DMD)** feature
extractor on top of the provided textile machine telemetry. Sliding temporal
windows are lifted into a tensor representation, dominant spatio-temporal modes
are distilled via DMD, and their descriptors feed a downstream classifier to
predict *Failure* vs *No Failure* states.

## How it works

1. **Pre-processing** – The CSV is chronologically ordered, machine identifiers
   and timestamps are discarded, numerical sensors are standardised, and
   machine types are one-hot encoded.
2. **Tensor windowing** – Consecutive snapshots (default: 12 timesteps) form a
   third-order tensor capturing the evolving machine state.
3. **Dynamic Mode Decomposition** – Each tensor window is decomposed, exposing
   dominant eigen-dynamics, modal amplitudes, and reconstruction residuals.
4. **Classification** – The engineered features feed configurable classifiers
   (logistic regression, gradient boosting, or random forest) that report
   accuracy, F1, ROC-AUC, and full diagnostics.

## Quick start

```powershell
# Create & activate a virtual environment (optional but recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Train and evaluate the DMD-based classifier
python DMD_Tensor.py --data-path textile_machine_data.csv
```

### Optional arguments

- `--window-size` *(int, default: 12)* – timesteps per tensor window.
- `--dmd-rank` *(int, default: 8)* – truncated rank for the DMD computation.
- `--top-modes` *(int, default: 6)* – number of dominant modes exported as
  explicit features.
- `--test-size` *(float, default: 0.2)* – evaluation split ratio.
- `--classifier {logreg|gradient_boosting|random_forest}` – choose the downstream
   estimator.
- `--grid-search` – enable cross-validated hyperparameter tuning for the chosen
   classifier.
- `--export-features path/to/file.csv` – persist the engineered DMD feature
   table for further analysis.
- `--export-model path/to/model.joblib` – serialise the fitted pipeline for
   deployment (requires `joblib`).

## Outputs

Running the script prints a JSON diagnostics block containing:

- Accuracy, precision, recall, F1, ROC-AUC.
- Confusion matrix and textual classification report.
- 5-fold cross-validated ROC-AUC mean and standard deviation.

When export flags are supplied, the engineered feature matrix and the trained
model artefacts are written to disk alongside the full configuration for
reproducibility.

## Next steps

- Complement with additional classifiers (e.g., gradient boosting or SVM).
- Incorporate streaming inference by processing windows in real-time.
- Enrich the feature space with domain-specific health indicators or thresholds.
