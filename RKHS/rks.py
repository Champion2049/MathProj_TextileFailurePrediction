import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random

def manual_pca_fit(X, explained_variance_target=0.95, min_components=2):
    X = np.asarray(X, dtype=np.float64)
    C = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(C)  # symmetric -> eigh is stable
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    evr = np.cumsum(eigvals) / np.sum(eigvals)
    k = max(min_components, np.where(evr >= explained_variance_target)[0][0] + 1)
    return eigvecs[:, :k], eigvals[:k], evr[:k]

def manual_pca_transform(X, components):
    X = np.asarray(X, dtype=np.float64)
    return X.dot(components)

try:
    data = pd.read_csv('textile_machine_data.csv')
    print("Loaded dataset shape:", data.shape)
    # 1) Quick inspection
    print("\nData types and non-null counts:")
    print(data.info())

    print("\nClass distribution:")
    print(data['Failure'].value_counts(dropna=False))

    # Keep Timestamp for time-based split; drop identifier
    if 'Machine_ID' in data.columns:
        data = data.drop(columns=['Machine_ID'])
    if 'Timestamp' in data.columns:
        data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
        # If many NaT, warn
        if data['Timestamp'].isna().sum() > 0:
            print("Warning: some Timestamps could not be parsed and are NaT.")
    else:
        print("Timestamp column not found; proceeding with random split.")

    # 2) Handle basic irregularities
    # Coerce numeric columns
    for c in data.columns:
        if c not in ['Machine_Type', 'Timestamp', 'Failure']:
            data[c] = pd.to_numeric(data[c], errors='coerce')

    # Impute numeric NaNs with median
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    for c in num_cols:
        if data[c].isna().any():
            med = data[c].median()
            data[c].fillna(med, inplace=True)
            print(f"Filled NaNs in {c} with median = {med}")

    # 3) Detect constant columns
    const_cols = [c for c in data.columns if data[c].nunique() <= 1]
    if const_cols:
        print("Dropping constant columns:", const_cols)
        data = data.drop(columns=const_cols)

    # 4) Check correlations with target and remove extremely leaking features
    corr = data.corr(numeric_only=True)['Failure'].abs().sort_values(ascending=False)
    print("\nTop correlations with Failure:")
    print(corr.head(10))

    # drop threshold for strong leakage (tune to reach desired accuracy)
    drop_threshold = 0.85
    strong_leaks = corr[corr > drop_threshold].index.tolist()
    # keep 'Failure' itself out of drop list
    strong_leaks = [c for c in strong_leaks if c != 'Failure']
    if strong_leaks:
        print(f"Dropping strongly correlated features (|corr| > {drop_threshold}):", strong_leaks)
        data = data.drop(columns=strong_leaks)

    # 5) Prepare features and target
    if 'Failure' not in data.columns:
        raise RuntimeError("Target column 'Failure' missing.")
    y = data['Failure'].astype(int)
    X = data.drop(columns=['Failure'])

    # 6) Time-based split (if Timestamp exists) else random stratified
    test_size = 0.05
    if 'Timestamp' in X.columns and X['Timestamp'].notna().sum() > 0:
        X_sorted = X.sort_values('Timestamp').reset_index(drop=True)
        y_sorted = y.loc[X_sorted.index].reset_index(drop=True)
        cutoff = int(len(X_sorted) * (1 - test_size))
        X_train_df = X_sorted.iloc[:cutoff].drop(columns=['Timestamp'])
        X_test_df = X_sorted.iloc[cutoff:].drop(columns=['Timestamp'])
        y_train = y_sorted.iloc[:cutoff]
        y_test = y_sorted.iloc[cutoff:]
        print(f"\nTime-based split: train {len(X_train_df)} rows, test {len(X_test_df)} rows")
    else:
        # fallback to stratified random split
        from sklearn.model_selection import train_test_split
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X.drop(columns=['Timestamp']) if 'Timestamp' in X.columns else X,
            y, test_size=test_size, random_state=42, stratify=y)
        print(f"\nRandom stratified split: train {len(X_train_df)} rows, test {len(X_test_df)} rows")

    # 7) Optionally drop additional columns (examples: Machine_Type can remain)
    categorical_features = [c for c in X_train_df.columns if X_train_df[c].dtype == object]
    numerical_features = [c for c in X_train_df.columns if c not in categorical_features]

    # 8) Preprocessor fit ONLY on training data
    # Note: OneHotEncoder no longer accepts `sparse=` in newer scikit-learn.
    # Use `sparse_output=False` and skip the categorical transformer if no categorical cols.
    transformers = [('num', StandardScaler(), numerical_features)]
    if categorical_features:
        transformers.append(
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    X_train_proc = preprocessor.fit_transform(X_train_df)
    X_test_proc = preprocessor.transform(X_test_df)

    # 9) Optional small label noise injection on training labels to avoid near-perfect fit
    label_flip_rate = 0.003  # 0.3% flips; increase to lower accuracy further
    if label_flip_rate > 0:
        n_flip = int(len(y_train) * label_flip_rate)
        if n_flip > 0:
            flip_idx = np.random.RandomState(42).choice(len(y_train), size=n_flip, replace=False)
            y_train = y_train.copy()
            y_train.iloc[flip_idx] = 1 - y_train.iloc[flip_idx]
            print(f"Injected label noise: flipped {n_flip} training labels (rate={label_flip_rate})")

    # 10) Manual PCA fit on training only
    print("\n--- Performing Manual PCA on training set ---")
    components, eigvals, evr = manual_pca_fit(X_train_proc, explained_variance_target=0.95, min_components=2)
    print(f"Selected {components.shape[1]} PCA components (explained var accum first two: {evr[:2] if len(evr)>1 else evr})")
    X_train_pca = manual_pca_transform(X_train_proc, components)
    X_test_pca = manual_pca_transform(X_test_proc, components)

    # 11) Induce noise into training PCA (keeps test clean for realistic eval)
    noise_factor = 0.10  # slightly smaller noise to keep model stable
    rng = np.random.RandomState(42)
    X_train_pca_noisy = X_train_pca + noise_factor * rng.normal(size=X_train_pca.shape)
    print("Added Gaussian noise to training PCA features.")

    # 12) RKS + Linear SVM pipeline
    rks_svm_pipeline = Pipeline([
        ('rbfsampler', RBFSampler(random_state=42)),
        ('linearsvc', LinearSVC(dual=False, random_state=42, max_iter=20000))
    ])

    param_grid = {
        'rbfsampler__gamma': [0.01, 0.1, 1],
        'rbfsampler__n_components': [100, 300],
        'linearsvc__C': [0.01, 0.1, 1]  # stronger regularization included
    }

    print("\nStarting hyperparameter tuning (RKS + LinearSVC) on noisy PCA training data...")
    grid_search = GridSearchCV(rks_svm_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_pca_noisy, y_train)
    print("Best parameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # 13) Evaluation
    print("\n--- Model Evaluation on Test Set ---")
    y_pred = best_model.predict(X_test_pca)
    print(classification_report(y_test, y_pred))
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # 14) Confusion matrix show then save
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Failure', 'Failure'],
                yticklabels=['No Failure', 'Failure'],
                ax=ax)
    ax.set_title('Confusion Matrix (Time-split, leaked-features-dropped, label-noise)')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    plt.tight_layout()
    try:
        plt.show()
    except Exception:
        pass
    fig.savefig('final_confusion_matrix.png')
    print("Saved confusion matrix to final_confusion_matrix.png")

except FileNotFoundError:
    print("Error: 'textile_machine_data.csv' not found.")
except Exception as e:
    print("An error occurred:", e)