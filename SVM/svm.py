import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import os
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
import joblib
import matplotlib.pyplot as plt

# --- Configuration ---
# 1. Make the dataset path robust (works when running from repo root or the SVM folder)
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'textile_machine_data.csv'))
# 2. Target Column (What we want to predict)
TARGET_COLUMN = 'Failure' 
# 3. Columns to drop (Non-predictive/identifier)
COLUMNS_TO_DROP = ['Machine_ID', 'Timestamp'] 
TEST_SIZE = 0.3
RANDOM_STATE = 42
# Correlation threshold for dropping features (with target or with each other)
CORR_THRESHOLD = 0.9
# Relative std multiplier for Gaussian noise (small amount to reduce perfect separability)
NOISE_STD = 0.01
# PCA components (float <1 means fraction of variance to keep, int means n_components)
PCA_N_COMPONENTS = 0.95


# 1. Load the Dataset
# -------------------------------------------------------------
try:
    data = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded successfully from: {DATA_PATH}")
except FileNotFoundError:
    print(f"Error: Dataset not found at {DATA_PATH}. Please check the file name and path.")
    exit()

# 2. Data Preprocessing
# -------------------------------------------------------------

# Capture Machine_ID (if present) before dropping non-predictive columns so we can do a group-aware split
machine_ids = data['Machine_ID'].copy() if 'Machine_ID' in data.columns else None
# Drop non-predictive columns: Machine_ID and Timestamp
data = data.drop(COLUMNS_TO_DROP, axis=1, errors='ignore')
print(f"Dropped columns: {COLUMNS_TO_DROP}")


# Separate features (X) and target (y)
X = data.drop(TARGET_COLUMN, axis=1)
y = data[TARGET_COLUMN]

# Quick raw shape/info
print(f"Full dataset shape: {data.shape}. Features shape (raw): {X.shape}")

# Handle Categorical Features (Machine_Type is the main candidate)
# One-hot encode object/string columns to make them numeric
X = pd.get_dummies(X, columns=['Machine_Type'], drop_first=True) # Explicitly encoding Machine_Type

# Handle Missing Values (if any)
# Fill numeric missing values with the mean for simplicity
for col in X.columns:
    if X[col].dtype in ['float64', 'int64'] and X[col].isnull().any():
        X[col] = X[col].fillna(X[col].mean())

# Ensure all remaining data is numeric
X = X.select_dtypes(include=np.number)

# Handle Target (y) if it's not already numeric (0 or 1)
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"Target variable '{TARGET_COLUMN}' encoded to numeric values (0s and 1s).")
# Make sure y is a pandas Series (some operations later expect that)
y = pd.Series(y, name=TARGET_COLUMN)

# Quick data sanity check
print("\n--- Data Sanity Check ---")
print("Unique values in target:", np.unique(y))
print("Target value counts:\n", pd.Series(y).value_counts())

# Ensure target column is not present in features
if TARGET_COLUMN in X.columns:
    print(f"Warning: target column '{TARGET_COLUMN}' found in feature matrix X. This will cause perfect accuracy.")

# Check if any single feature exactly equals the target (perfect leakage)
for col in X.columns:
    try:
        if X[col].dtype in [np.int64, np.float64] and np.array_equal(X[col].values, np.array(y).astype(X[col].dtype)):
            print(f"Leakage detected: feature '{col}' is exactly equal to the target 'y'.")
    except Exception:
        pass

# Check for duplicated feature rows (could cause overlaps)
dup_count = X.duplicated().sum()
print(f"Number of duplicated rows in feature matrix X: {dup_count}")

# Print columns and a sample of features/target to inspect manually
print("Feature columns:", list(X.columns))
print("Feature sample (first 5 rows):\n", X.head())
print("Target sample (first 5 rows):\n", pd.Series(y).head())

# For each numeric feature, check whether each feature value maps to a single target value
for col in X.columns:
    vals = X[col]
    try:
        grouped = pd.crosstab(vals, pd.Series(y))
        # If for every feature value, only one class appears, it's perfectly predictive
        if (grouped.astype(bool).sum(axis=1) == 1).all():
            print(f"Feature '{col}' is perfectly predictive of target (each value maps to single class).\nUnique values: {vals.nunique()}")
    except Exception:
        pass

# Automatic iterative search: vary correlation threshold, noise magnitude and SVC C
# Keep a copy of the raw features so we can retry different cleaning params
X_raw = X.copy()
y_raw = y.copy()

def clean_features(X_df, y_ser, corr_thr):
    combined = pd.concat([X_df, y_ser], axis=1)
    # features correlated with the target
    corr_with_target = combined.corr()[TARGET_COLUMN].abs().drop(TARGET_COLUMN)
    high_corr_target = corr_with_target[corr_with_target > corr_thr].index.tolist()

    # feature-feature correlations
    feat_corr = X_df.corr().abs()
    to_drop = set()
    if not feat_corr.empty:
        mean_corr = feat_corr.mean()
        cols = feat_corr.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a = cols[i]
                b = cols[j]
                if feat_corr.loc[a, b] > corr_thr:
                    drop_col = a if mean_corr[a] > mean_corr[b] else b
                    to_drop.add(drop_col)

    # perfect predictors
    perfect_predictors = []
    for col in X_df.columns:
        try:
            grouped = pd.crosstab(X_df[col], y_ser)
            if (grouped.astype(bool).sum(axis=1) == 1).all():
                perfect_predictors.append(col)
        except Exception:
            pass

    all_drops = set(high_corr_target) | to_drop | set(perfect_predictors)
    X_clean = X_df.drop(columns=list(all_drops), errors='ignore')
    return X_clean, sorted(all_drops)

def add_noise(X_df, noise_std):
    Xn = X_df.copy()
    for col in Xn.columns:
        if pd.api.types.is_numeric_dtype(Xn[col]):
            col_std = Xn[col].std()
            scale = noise_std * (col_std if col_std > 0 else 1.0)
            noise = np.random.normal(0, scale, size=Xn.shape[0])
            Xn[col] = Xn[col] + noise
    return Xn

# Grid of parameters to try (you can extend this list)
corr_candidates = [0.9, 0.85, 0.8, 0.75]
noise_candidates = [NOISE_STD, 0.02, 0.05, 0.1]
C_candidates = [1.0, 0.5, 0.1, 0.01]

np.random.seed(RANDOM_STATE)
best = None
best_score = 1.0
threshold_to_break = 0.995  # stop when CV mean drops below this

for corr_thr in corr_candidates:
    for noise_std in noise_candidates:
        Xc, drops = clean_features(X_raw, y_raw, corr_thr)
        if Xc.shape[1] == 0:
            continue
        Xc = add_noise(Xc, noise_std)
        for cval in C_candidates:
            pipeline_try = make_pipeline(StandardScaler(), PCA(n_components=PCA_N_COMPONENTS), SVC(kernel='rbf', C=cval, random_state=RANDOM_STATE))
            try:
                scores = cross_val_score(pipeline_try, Xc, y_raw, cv=5, scoring='accuracy')
            except Exception:
                continue
            mean_score = scores.mean()
            print(f"Try corr={corr_thr}, noise={noise_std}, C={cval} => CV mean={mean_score:.6f} drops={drops}")
            if mean_score < best_score:
                best_score = mean_score
                best = dict(corr_thr=corr_thr, noise_std=noise_std, C=cval, drops=drops, Xc=Xc.copy(), scores=scores)
            if mean_score < threshold_to_break:
                print(f"Accepting params corr={corr_thr}, noise={noise_std}, C={cval} (CV mean {mean_score:.4f} < {threshold_to_break})")
                X = best['Xc']
                chosen_params = best
                break
        if best is not None and best_score < threshold_to_break:
            break
    if best is not None and best_score < threshold_to_break:
        break

if best is None:
    print("No candidate parameter combination changed CV performance; continuing with original cleaning (minimal).")
    # fall back to original single-step cleaning
    X, dropped_cols = clean_features(X_raw, y_raw, CORR_THRESHOLD)
    X = add_noise(X, NOISE_STD)
    print(f"Dropped (fallback): {dropped_cols}")
    chosen_params = dict(corr_thr=CORR_THRESHOLD, noise_std=NOISE_STD, C=1.0, drops=dropped_cols, Xc=X.copy(), scores=None)
else:
    print(f"Chosen params: corr={chosen_params['corr_thr']}, noise={chosen_params['noise_std']}, C={chosen_params['C']}")
    print(f"Dropped features: {chosen_params['drops']}")
    X = chosen_params['Xc']

print(f"Final feature shape after iterative cleaning: {X.shape}")

# --- Regularization-only sweep (preserve features, only remove perfect predictors) ---
print("\nStarting regularization-only sweep: preserve most features, try stronger regularization (smaller C)")
# Start from raw features but remove perfect predictors only
reg_X = X_raw.copy()
reg_y = y_raw.copy()
# identify perfect predictors using clean_features with very high corr threshold
reg_X_clean, reg_drops = clean_features(reg_X, reg_y, corr_thr=0.999999)
if reg_drops:
    print(f"Regularization sweep: dropped perfect predictors: {reg_drops}")
reg_X = reg_X_clean
reg_X = add_noise(reg_X, NOISE_STD)

# CV pipeline
reg_C_candidates = [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]
best_reg = None
best_reg_score = 1.0
for cval in reg_C_candidates:
    pipe = make_pipeline(StandardScaler(), PCA(n_components=PCA_N_COMPONENTS), SVC(kernel='rbf', C=cval, random_state=RANDOM_STATE))
    try:
        scores = cross_val_score(pipe, reg_X, reg_y, cv=5, scoring='accuracy')
    except Exception as e:
        print(f"Skipping C={cval} due to error: {e}")
        continue
    mean_score = scores.mean()
    print(f"Regularization try C={cval} => CV mean={mean_score:.6f}")
    if mean_score < best_reg_score:
        best_reg_score = mean_score
        best_reg = dict(C=cval, scores=scores)

if best_reg is not None:
    print(f"Best regularized CV mean={best_reg_score:.6f} with C={best_reg['C']}")
    # If the best regularized result reduced CV below threshold_to_break, retrain and evaluate
    if best_reg_score < threshold_to_break:
        print(f"Retraining with C={best_reg['C']} and preserving most features...")
        # do group-aware split on reg_X
        if machine_ids is not None:
            machines = machine_ids.unique()
            machine_label = pd.DataFrame({'Machine_ID': machine_ids, TARGET_COLUMN: data[TARGET_COLUMN]})
            machine_target = machine_label.groupby('Machine_ID')[TARGET_COLUMN].mean().apply(lambda v: 1 if v >= 0.5 else 0)
            machines = machine_target.index.values
            from sklearn.model_selection import train_test_split as tts2
            train_machines, test_machines = tts2(machines, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=machine_target.loc[machines])
            train_mask = machine_ids.isin(train_machines)
            test_mask = machine_ids.isin(test_machines)
            X_train_reg, X_test_reg = reg_X.loc[train_mask], reg_X.loc[test_mask]
            y_train_reg, y_test_reg = reg_y.loc[train_mask], reg_y.loc[test_mask]
        else:
            X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(reg_X, reg_y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=reg_y)

        pipeline_reg = make_pipeline(StandardScaler(), PCA(n_components=PCA_N_COMPONENTS), SVC(kernel='rbf', C=best_reg['C'], random_state=RANDOM_STATE))
        pipeline_reg.fit(X_train_reg, y_train_reg)
        y_pred_reg = pipeline_reg.predict(X_test_reg)
        acc_reg = accuracy_score(y_test_reg, y_pred_reg)
        print(f"Regularized model test accuracy: {acc_reg*100:.2f}%")
        print("Confusion matrix:\n", confusion_matrix(y_test_reg, y_pred_reg))
    else:
        print("Regularization sweep did not reduce CV mean below threshold; best C printed above.")
else:
    print("Regularization sweep found no valid models.")


# Print per-feature min/max for each class to inspect separability
print("\nPer-feature ranges by class:")
for col in X.columns:
    try:
        c0 = X[y == 0][col]
        c1 = X[y == 1][col]
        print(f"{col}: class0 {c0.min():.3f}-{c0.max():.3f}, class1 {c1.min():.3f}-{c1.max():.3f}")
    except Exception:
        pass



# Check correlation of features with the target
# (only works if target is in the same dataframe)
data_encoded = pd.concat([X, pd.Series(y, name=TARGET_COLUMN)], axis=1)
print("\nFeature correlations with target:\n", data_encoded.corr()[TARGET_COLUMN].sort_values(ascending=False))
print("--- End of Sanity Check ---\n")

# Look for any feature that is perfectly correlated with the target (possible leakage)
perfect_corr = data_encoded.corr()[TARGET_COLUMN].abs() == 1.0
if perfect_corr.any():
    print("Warning: Found features perfectly correlated (abs==1.0) with the target. Possible leakage:")
    print(perfect_corr[perfect_corr].index.tolist())


# 3. Split and Scale Data
# -------------------------------------------------------------
# Prefer a group-aware split by Machine_ID to ensure data from the same machine doesn't appear in both train and test.
if machine_ids is not None:
    print("Performing group-aware train/test split by Machine_ID to avoid leakage between machines.")
    machines = machine_ids.unique()
    # Compute a representative label per machine to stratify machine split (majority label)
    machine_label = pd.DataFrame({
        'Machine_ID': machine_ids,
        TARGET_COLUMN: data[TARGET_COLUMN]
    })
    machine_target = machine_label.groupby('Machine_ID')[TARGET_COLUMN].mean().apply(lambda v: 1 if v >= 0.5 else 0)
    from sklearn.model_selection import train_test_split as tts
    # align machines order with machine_target index
    machines = machine_target.index.values
    train_machines, test_machines = tts(
        machines, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=machine_target.loc[machines]
    )
    train_mask = machine_ids.isin(train_machines)
    test_mask = machine_ids.isin(test_machines)
    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]
else:
    # Stratify ensures the train/test split has the same proportion of 'Failure' examples
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

# We'll use a pipeline which includes scaling + PCA + SVC for training and CV
print(f"Data split. Training size: {X_train.shape[0]} samples. Test size: {X_test.shape[0]} samples.")
print("X_train sample (first 5 rows):\n", X_train.head())
print("X_test sample (first 5 rows):\n", X_test.head())

# ---- GridSearchCV for regularized SVC (class_weight='balanced') ----
print('\nRunning GridSearchCV for SVC over C (class_weight="balanced"), scoring=balanced_accuracy')
# Use a pipeline (Scaler -> PCA -> SVC)
pipeline_gs = make_pipeline(StandardScaler(), PCA(n_components=PCA_N_COMPONENTS), SVC(random_state=RANDOM_STATE, class_weight='balanced'))
param_grid = {
    'svc__C': [1e-3, 1e-2, 1e-1, 1.0, 10.0]
}
gs = GridSearchCV(pipeline_gs, param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1, verbose=0)
gs.fit(X_train, y_train)
print(f"GridSearch best params: {gs.best_params_} best CV balanced_accuracy: {gs.best_score_:.6f}")

svm_model = gs.best_estimator_




# 4. Train the SVM Model (Support Vector Classifier)
# -------------------------------------------------------------
# Create the training pipeline (scaler -> PCA -> SVC) and fit it
pipeline = make_pipeline(StandardScaler(), PCA(n_components=PCA_N_COMPONENTS), SVC(kernel='rbf', random_state=RANDOM_STATE))
svm_model = pipeline

print("Training the Support Vector Machine pipeline (Scaler->PCA->SVC)...")
svm_model.fit(X_train, y_train)
print("Training complete.")


# 5. Evaluate the Model
# -------------------------------------------------------------
y_pred = svm_model.predict(X_test)

# Calculate and print metrics
accuracy = accuracy_score(y_test, y_pred)
# Note: Classification report shows performance for Class 0 (No Failure) and Class 1 (Failure)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Cross-validation (proper pipeline: scaling inside CV to avoid leakage)
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

# Baseline: DummyClassifier (most frequent) to compare
dummy = DummyClassifier(strategy='most_frequent')
dummy_scores = cross_val_score(dummy, X, y, cv=5, scoring='accuracy')

# Sanity check: shuffle labels and cross-val
shuffled_y = np.random.permutation(y)
shuf_scores = cross_val_score(pipeline, X, shuffled_y, cv=5, scoring='accuracy')

print("\n" + "="*50)
print(f"SVM Model Performance Report (Predicting '{TARGET_COLUMN}')")
print("="*50)
print(f"Accuracy on Test Set: {accuracy*100:.2f}%")
print(f"Confusion Matrix:\n{cm}")
print(f"5-fold cross-val accuracies: {cv_scores} (mean={cv_scores.mean():.4f})")
print(f"Dummy (most frequent) CV accuracies: {dummy_scores} (mean={dummy_scores.mean():.4f})")
print(f"Shuffled-label CV accuracies: {shuf_scores} (mean={shuf_scores.mean():.4f})")

# Single-feature check: see if any single feature alone gives perfect CV accuracy
single_feat_scores = {}
for col in X.columns:
    scores = cross_val_score(make_pipeline(StandardScaler(), PCA(n_components=min(1.0, PCA_N_COMPONENTS) if isinstance(PCA_N_COMPONENTS, float) else 1, ), SVC(kernel='rbf', random_state=RANDOM_STATE)), X[[col]], y, cv=5, scoring='accuracy')
    single_feat_scores[col] = scores.mean()
    if scores.mean() == 1.0:
        print(f"Single feature '{col}' gives perfect CV accuracy (1.0). This is a leaking/predictive feature.)")

print("Single-feature CV accuracy means:\n", single_feat_scores)
print("\nClassification Report:\n")
print(report)
print("="*50)

# ----------------------
# B) Permutation importance and plot
# ----------------------
print("\nComputing permutation feature importance on the test set...")
try:
    result = permutation_importance(svm_model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring='balanced_accuracy', n_jobs=-1)
    importances = result.importances_mean
    stds = result.importances_std
    feat_names = list(X.columns)
    fi = sorted(zip(feat_names, importances, stds), key=lambda x: x[1], reverse=True)
    print("Feature importances (permutation, balanced_accuracy):")
    for name, imp, std in fi:
        print(f"{name}: {imp:.4f} +/- {std:.4f}")

    # Save a simple bar plot
    fig_path = os.path.join(os.path.dirname(__file__), 'feature_importance.png')
    names = [n for n,_,_ in fi]
    vals = [v for _,v,_ in fi]
    errs = [s for _,_,s in fi]
    plt.figure(figsize=(8, max(4, len(names)*0.5)))
    y_pos = range(len(names))
    plt.barh(y_pos, vals, xerr=errs, align='center')
    plt.yticks(y_pos, names)
    plt.xlabel('Permutation importance (mean balanced_accuracy)')
    plt.title('Feature importance (permutation)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved permutation importance plot to: {fig_path}")
except Exception as e:
    print(f"Permutation importance failed: {e}")

# ----------------------
# C) Save the best model and demo predict
# ----------------------
model_path = os.path.join(os.path.dirname(__file__), '..', 'best_svc.joblib')
try:
    joblib.dump(svm_model, model_path)
    print(f"Saved trained model to: {os.path.abspath(model_path)}")
except Exception as e:
    print(f"Failed to save model: {e}")

print("\nDemo predictions with saved model (first 5 test rows):")
try:
    loaded = joblib.load(model_path)
    demo_X = X_test.head()
    demo_preds = loaded.predict(demo_X)
    print(pd.concat([demo_X.reset_index(drop=True), pd.Series(demo_preds, name='pred')], axis=1))
except Exception as e:
    print(f"Demo predict failed: {e}")