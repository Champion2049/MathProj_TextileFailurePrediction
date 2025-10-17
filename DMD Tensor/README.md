# Tensor DMD Textile Failure Prediction

This project implements a **tensor-based Dynamic Mode Decomposition (DMD)** feature
extractor on top of the provided textile machine telemetry. Sliding temporal
windows are lifted into a tensor representation, dominant spatio-temporal modes
are distilled via DMD, and their descriptors feed a downstream classifier to
predict *Failure* vs *No Failure* states. Every stage (PCA + DMD) is derived
step-by-step from first principles with rich inline commentary.

## How it works

1. **Pre-processing** – The CSV is chronologically ordered, machine identifiers
   and timestamps are discarded, numerical sensors are standardised, and
   machine types are one-hot encoded.
2. **PCA from scratch** – The centred feature matrix undergoes an eigenvalue
   analysis of its covariance matrix to retain the dominant principal components
   (by variance threshold or explicit count).
3. **Tensor windowing** – Consecutive snapshots (default: 12 timesteps) form a
   third-order tensor capturing the evolving machine state.
4. **Dynamic Mode Decomposition** – Each tensor window is decomposed, exposing
   dominant eigen-dynamics, modal amplitudes, and reconstruction residuals.
5. **Classification** – The engineered features feed configurable classifiers
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
- `--compare-classifiers` – evaluate all supported classifiers (logreg,
  gradient boosting, random forest) with identical features and print a
  side-by-side comparison table.
- `--no-pca` – disable the handcrafted PCA stage.
- `--pca-components` / `--pca-variance` – control the dimensionality preserved by
   the PCA projection (defaults to 95% cumulative variance).
- `--output-dir path/to/folder` – store evaluation artefacts in the specified
  directory (confusion matrix, ROC curve, decision landscape, ranked feature
  influence table, metrics.json, trained model, etc.).
- `--grid-search` – enable cross-validated hyperparameter tuning for the chosen
   classifier.
- `--export-features path/to/file.csv` – persist the engineered DMD feature
   table for further analysis.
- `--export-model path/to/model.joblib` – serialise the fitted pipeline for
   deployment (requires `joblib`). When combined with `--compare-classifiers`,
   the filename is suffixed with the classifier name to avoid overwrites.

## Outputs

Running the script prints a console diagnostics summary containing:

- Accuracy, precision, recall, F1, ROC-AUC.
- Confusion matrix (also saved as an image when `--output-dir` is provided),
   textual classification report, and the top contributing features driving the
   predictions.
- 5-fold cross-validated ROC-AUC mean and standard deviation.

When export flags are supplied, the engineered feature matrix and the trained
model artefacts are written to disk alongside the full configuration for
reproducibility. Providing `--output-dir` additionally saves:

- `confusion_matrix.png` – hold-out confusion matrix visualisation.
- `roc_curve.png` – ROC plot with the AUC annotated.
- `decision_boundary.png` – two-dimensional decision landscape (first two
   feature axes).
- `feature_influence.csv` – ranked feature importance / coefficient table.
- `metrics.json` – structured metrics bundle for downstream reporting.
- `trained_model.joblib` – fitted pipeline and configuration snapshot.

## Next steps

- Incorporate streaming inference by processing windows in real-time.
- Enrich the feature space with domain-specific health indicators or thresholds.
- Extend interpretability with SHAP or partial dependence analysis when
   stronger explainability is required.
