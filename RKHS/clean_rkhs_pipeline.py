"""
Clean RKHS pipeline
- Inspect dataset for irregularities/biases
- Clean data: drop identifiers, coerce numerics, impute, remove constant cols, detect and (optionally) drop leaking features
- Split train/test (stratified)
- Manual PCA (fit on train only)
- RKHS: compute RBF kernel on PCA projection
- Train & evaluate: LogisticRegression on PCA coords, SVC(kernel='precomputed') on RKHS kernel
- Print diagnostics and save small report

Run: python clean_rkhs_pipeline.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Utilities ----------

def manual_pca(X, k=2):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    C = (X_centered.T @ X_centered) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    V_k = eigvecs[:, :k]
    Z = X_centered @ V_k
    explained_variance_ratio = eigvals[:k] / np.sum(eigvals)
    return Z, eigvals[:k], explained_variance_ratio, X_mean, V_k


def rbf_kernel(Z, sigma=1.0):
    Z = np.asarray(Z, dtype=float)
    sq_dists = np.sum(Z**2, axis=1).reshape(-1, 1) + np.sum(Z**2, axis=1) - 2 * (Z @ Z.T)
    return np.exp(-sq_dists / (2 * sigma**2))

# ---------- Load & Inspect ----------
print('Loading dataset...')
# dataset is at repository root
df = pd.read_csv('textile_machine_data.csv')
print('Shape:', df.shape)
print('\nColumns and dtypes:')
print(df.dtypes)
print('\nClass counts:')
print(df['Failure'].value_counts())

# ---------- Cleaning & Irregularity handling ----------
# Drop obvious identifiers and timestamp
for c in ['Machine_ID', 'Timestamp']:
    if c in df.columns:
        df = df.drop(columns=[c])

# One-hot encode Machine_Type (if exists)
if 'Machine_Type' in df.columns:
    df = pd.get_dummies(df, columns=['Machine_Type'], drop_first=True)

# Coerce numeric columns, impute with median for robustness
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
# Keep target separate
if 'Failure' not in numeric_cols and 'Failure' in df.columns:
    numeric_cols.append('Failure')

features = [c for c in numeric_cols if c != 'Failure']
X_df = df[features].apply(pd.to_numeric, errors='coerce')
# If many NaNs, consider dropping rows, but we'll impute median per-column
nan_counts = X_df.isnull().sum()
if nan_counts.sum() > 0:
    print('\nImputing numeric NaNs with column median:')
    print(nan_counts[nan_counts>0])
    X_df = X_df.fillna(X_df.median())

# Remove constant columns
const_cols = [c for c in X_df.columns if X_df[c].nunique() <= 1]
if const_cols:
    print('\nDropping constant columns:', const_cols)
    X_df = X_df.drop(columns=const_cols)

# Recompute features list
features = list(X_df.columns)
X = X_df.values.astype(float)
y = pd.to_numeric(df['Failure'], errors='coerce').astype(int).values

# Detect potential leakage: features exactly equal to target or near-perfect correlation
corr = pd.concat([X_df, pd.Series(y, name='Failure')], axis=1).corr()['Failure'].abs().sort_values(ascending=False)
print('\nTop correlations with Failure:')
print(corr.head(10))

# If any feature correlates > 0.995, warn and drop (configurable)
leak_threshold = 0.995
leaky = corr[(corr > leak_threshold) & (corr.index != 'Failure')].index.tolist()
if leaky:
    print('\nWarning: found near-perfectly correlated features (will drop):', leaky)
    X_df = X_df.drop(columns=leaky)
    features = list(X_df.columns)
    X = X_df.values.astype(float)

# Optional: remove top-n correlated features to make task harder (commented)
# high_corr_thresh = 0.9
# drop_more = corr[(corr>high_corr_thresh) & (corr.index!='Failure')].index.tolist()
# print('Dropping highly correlated features:', drop_more)
# X_df = X_df.drop(columns=drop_more)

# ---------- Train/Test split (stratified) ----------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
print('\nTrain/test sizes:', X_train_raw.shape, X_test_raw.shape)

# ---------- Standardize (fit on train) ----------
scaler = StandardScaler().fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# ---------- Manual PCA (fit on train) ----------
k = min(5, X_train.shape[1])
Z_train, eigvals, evr, X_mean, V_k = manual_pca(X_train, k=k)
# Transform test using train mean and components
X_test_centered = X_test - X_mean
Z_test = X_test_centered @ V_k
print('\nManual PCA explained variance ratio (train):', evr)

# ---------- RKHS Kernel on PCA projection ----------
sigma = np.std(Z_train) if np.std(Z_train)>0 else 1.0
K_train = rbf_kernel(Z_train, sigma=sigma)
K_test = np.exp(-np.sum((Z_test[:,None,:]-Z_train[None,:,:])**2, axis=2)/(2*sigma**2))

# ---------- Train & Evaluate classifiers ----------
# 1) Logistic on PCA coords
clf_pca = LogisticRegression(max_iter=2000)
clf_pca.fit(Z_train, y_train)
y_pred_pca = clf_pca.predict(Z_test)

# 2) Kernel SVM using precomputed kernel
svc = SVC(kernel='precomputed')
svc.fit(K_train, y_train)
y_pred_k = svc.predict(K_test)

# Metrics
def print_metrics(y_true, y_pred, label):
    print(f"\n--- {label} ---")
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print('Precision:', precision_score(y_true, y_pred))
    print('Recall:', recall_score(y_true, y_pred))
    print('F1:', f1_score(y_true, y_pred))
    print('\nClassification report:')
    print(classification_report(y_true, y_pred))
    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))

print_metrics(y_test, y_pred_pca, 'Logistic on PCA')
print_metrics(y_test, y_pred_k, 'Kernel SVM on RKHS')

# Cross-validated checks (to ensure no leak)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pipe_raw = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000))])
print('\nCross-validated accuracy (raw features):', cross_val_score(pipe_raw, X, y, cv=cv, scoring='accuracy').mean())

pipe_pca = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=k)), ('clf', LogisticRegression(max_iter=2000))])
print('Cross-validated accuracy (PCA pipeline):', cross_val_score(pipe_pca, X, y, cv=cv, scoring='accuracy').mean())

# Permutation test on RKHS (shuffle labels) to ensure model isn't memorizing
from sklearn.utils import shuffle
n_perm = 30
perm_acc = []
for i in range(n_perm):
    y_sh = shuffle(y_train, random_state=42+i)
    svc.fit(K_train, y_sh)
    perm_acc.append(svc.score(K_test, y_test))
print('\nPermutation test mean accuracy (RKHS SVM):', np.mean(perm_acc))

# ---------- Plots: ROC and Confusion Matrices ----------
# Prepare scores for ROC curves
try:
    # Logistic regression: use predict_proba if available
    if hasattr(clf_pca, "predict_proba"):
        scores_pca = clf_pca.predict_proba(Z_test)[:, 1]
    else:
        scores_pca = clf_pca.decision_function(Z_test)
except Exception:
    scores_pca = clf_pca.decision_function(Z_test)

try:
    scores_k = svc.decision_function(K_test)
except Exception:
    # fallback: use predicted labels (will produce a degenerate ROC)
    scores_k = svc.predict(K_test)

fpr_p, tpr_p, _ = roc_curve(y_test, scores_pca)
roc_auc_p = auc(fpr_p, tpr_p)
fpr_k, tpr_k, _ = roc_curve(y_test, scores_k)
roc_auc_k = auc(fpr_k, tpr_k)

# ROC plot
plt.figure(figsize=(8, 6))
plt.plot(fpr_p, tpr_p, label=f'Logistic PCA (AUC = {roc_auc_p:.3f})', lw=2)
plt.plot(fpr_k, tpr_k, label=f'RKHS SVM (AUC = {roc_auc_k:.3f})', lw=2)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.tight_layout()
try:
    plt.show()
except Exception:
    pass
plt.savefig('roc_curve.png')
print("Saved ROC curve to roc_curve.png")

# Confusion matrix heatmaps (Logistic)
cm_p = confusion_matrix(y_test, y_pred_pca)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_p, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'])
ax.set_title('Confusion Matrix - Logistic on PCA')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
try:
    plt.show()
except Exception:
    pass
fig.savefig('confusion_matrix_logistic_pca.png')
print("Saved confusion matrix to confusion_matrix_logistic_pca.png")

# Confusion matrix heatmap (RKHS SVM)
cm_k = confusion_matrix(y_test, y_pred_k)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_k, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'])
ax.set_title('Confusion Matrix - RKHS SVM')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
try:
    plt.show()
except Exception:
    pass
fig.savefig('confusion_matrix_rkhs_svm.png')
print("Saved confusion matrix to confusion_matrix_rkhs_svm.png")

print('\nDone.')
