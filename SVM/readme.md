
````markdown
# Textile Machine Failure Prediction

Predicts textile machine failures using an SVM model with preprocessing, PCA, and regularization.

---

## Dataset
- CSV: `textile_machine_data.csv`
- Target: `Failure` (0 = no failure, 1 = failure)
- Features used for final model: `Vibration, RPM, Power_Consumption, Yarn_Tension`

---

## Requirements
- Python 3.x
- Libraries:
  ```bash
  numpy
  # Textile Machine Failure Prediction (SVM)

  This folder contains an SVM pipeline for predicting textile machine failures from sensor data.

  Overview
  --------
  - Dataset: `textile_machine_data.csv` (repo root)
  - Target column: `Failure` (0 = no failure, 1 = failure)
  - Final features used by the cleaned model: `Vibration, RPM, Power_Consumption, Yarn_Tension`

  Requirements
  ------------
  - Python 3.x
  - Install required packages (see `requirements.txt` in repo root):

  ```powershell
  pip install -r requirements.txt
  ```

  Quick start
  -----------
  From the repository root run:

  ```powershell
  python SVM/svm.py
  ```

  This script will:
  - Load `textile_machine_data.csv` from the repo root.
  - Run preprocessing (drop leaking/highly-correlated features, small Gaussian noise), PCA, and an SVC pipeline.
  - Perform cross-validation and a small grid search for regularization (C).
  - Save artifacts: `feature_importance.png` (in `SVM/`) and `best_svc.joblib` (in repo root).

  Notes about paths and running
  ----------------------------
  - The script expects the CSV at `../textile_machine_data.csv` when run from `SVM/`, or `textile_machine_data.csv` when run from repo root. Running `python SVM/svm.py` from repo root is the recommended approach.
  - The trained model is saved to `best_svc.joblib` in the repo root.

  Demo: load the saved model and run predictions
  --------------------------------------------
  Example (run from repo root):

  ```python
  import joblib
  import pandas as pd

  # Load model (saved to repo root by the script)
  model = joblib.load('best_svc.joblib')

  # Prepare new data (use the same 4 features the model expects)
  df = pd.read_csv('textile_machine_data.csv')
  X_new = df[['Vibration', 'RPM', 'Power_Consumption', 'Yarn_Tension']].iloc[:5]

  # Predict
  predictions = model.predict(X_new)
  print(predictions)
  ```

  What changed and why
  ----------------------
  - Fixed malformed Markdown fences and inconsistent code blocks.
  - Clarified the correct run command and file paths.
  - Pointed to `requirements.txt` for installation instead of listing packages inline.
  - Added explicit note about where artifacts are saved (`feature_importance.png` in `SVM/`, `best_svc.joblib` in repo root).

  If you want, I can also:
  - Add a short `predict.py` script that accepts a CSV and outputs predictions.
  - Save the permutation importance table as a CSV.
  - Create a small README at the repo root describing all scripts.
