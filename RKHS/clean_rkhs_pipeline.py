import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
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
df = pd.read_csv('textile_machine_data.csv')

# ---------- Cleaning & Irregularity handling ----------
features_to_drop = [
    'Machine_ID', 'Timestamp', 'Operating_Hours', 
    'Noise_Level', 'Bearing_Temperature', 'Temperature', 'Motor_Current'
]
df = df.drop(columns=features_to_drop)
print(f"Dropped columns: {features_to_drop}")


# One-hot encode Machine_Type
if 'Machine_Type' in df.columns:
    df = pd.get_dummies(df, columns=['Machine_Type'], drop_first=True)

# Process features
features = [c for c in df.columns if c != 'Failure']
X = df[features].values.astype(float)
y = df['Failure'].values.astype(int)

# ---------- Train/Test split (stratified) ----------
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
print('\nTrain/test sizes:', X_train_raw.shape, X_test_raw.shape)

# ---------- Standardize (fit on train) ----------
scaler = StandardScaler().fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# ---------- Manual PCA for Evaluation (k=5) ----------
k_eval = min(5, X_train.shape[1])
Z_train_eval, _, evr_eval, X_mean_eval, V_k_eval = manual_pca(X_train, k=k_eval)
X_test_centered_eval = X_test - X_mean_eval
Z_test_eval = X_test_centered_eval @ V_k_eval
print('\nManual PCA (for evaluation) explained variance ratio (train):', evr_eval)


# ---------- RKHS Kernel on PCA projection ----------
sigma = np.std(Z_train_eval) if np.std(Z_train_eval) > 0 else 1.0
K_train = rbf_kernel(Z_train_eval, sigma=sigma)
K_test = np.exp(-np.sum((Z_test_eval[:, None, :] - Z_train_eval[None, :, :])**2, axis=2) / (2 * sigma**2))

# ---------- Train & Evaluate classifiers ----------
# 1) Logistic on PCA coords
clf_pca = LogisticRegression(max_iter=2000, random_state=42)
clf_pca.fit(Z_train_eval, y_train)
y_pred_pca = clf_pca.predict(Z_test_eval)

# 2) Kernel SVM using precomputed kernel
svc = SVC(kernel='precomputed', probability=True, random_state=42)
svc.fit(K_train, y_train)
y_pred_k = svc.predict(K_test)

# Metrics function
def print_metrics(y_true, y_pred, label):
    print(f"\n--- {label} ---")
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))

print_metrics(y_test, y_pred_pca, 'Logistic on PCA')
print_metrics(y_test, y_pred_k, 'Kernel SVM on RKHS')

# ---------- Plots: ROC and Confusion Matrices ----------
scores_pca = clf_pca.predict_proba(Z_test_eval)[:, 1]
scores_k = svc.predict_proba(K_test)[:, 1]

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
plt.savefig('final_roc_curve.png')
print("\nSaved ROC curve to final_roc_curve.png")

# Confusion matrix heatmaps
cm_p = confusion_matrix(y_test, y_pred_pca)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_p, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'])
ax.set_title('Confusion Matrix - Logistic on PCA')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
fig.savefig('final_confusion_matrix_logistic.png')
print("Saved confusion matrix to final_confusion_matrix_logistic.png")

cm_k = confusion_matrix(y_test, y_pred_k)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_k, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['No Failure', 'Failure'], yticklabels=['No Failure', 'Failure'])
ax.set_title('Confusion Matrix - RKHS SVM')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
fig.savefig('final_confusion_matrix_rkhs.png')
print("Saved confusion matrix to final_confusion_matrix_rkhs.png")

# ---------- NEW: Decision Boundary Plot ----------
print("\nGenerating decision boundary plot...")
# Step 1: Get 2D PCA for visualization
k_viz = 2
Z_train_viz, _, _, _, _ = manual_pca(X_train, k=k_viz)

# Step 2: Train models on the 2D data
clf_viz = LogisticRegression(random_state=42).fit(Z_train_viz, y_train)
# We use a standard RBF SVM here for visualization, as it's a direct analogy to the RKHS method
svc_viz = SVC(kernel='rbf', gamma='auto', random_state=42).fit(Z_train_viz, y_train)

# Step 3: Create a meshgrid to plot the decision surface
x_min, x_max = Z_train_viz[:, 0].min() - 1, Z_train_viz[:, 0].max() + 1
y_min, y_max = Z_train_viz[:, 1].min() - 1, Z_train_viz[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Step 4: Plot the decision boundaries
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot for Logistic Regression
Z_logistic = clf_viz.predict(np.c_[xx.ravel(), yy.ravel()])
Z_logistic = Z_logistic.reshape(xx.shape)
axes[0].contourf(xx, yy, Z_logistic, alpha=0.4, cmap=plt.cm.coolwarm)
scatter = axes[0].scatter(Z_train_viz[:, 0], Z_train_viz[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
axes[0].set_title('Logistic Regression Decision Boundary')
axes[0].set_xlabel('Principal Component 1')
axes[0].set_ylabel('Principal Component 2')
axes[0].legend(handles=scatter.legend_elements()[0], labels=['No Failure', 'Failure'])


# Plot for RBF SVM
Z_svm = svc_viz.predict(np.c_[xx.ravel(), yy.ravel()])
Z_svm = Z_svm.reshape(xx.shape)
axes[1].contourf(xx, yy, Z_svm, alpha=0.4, cmap=plt.cm.coolwarm)
scatter = axes[1].scatter(Z_train_viz[:, 0], Z_train_viz[:, 1], c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
axes[1].set_title('RBF SVM Decision Boundary (Visual Analogy for RKHS)')
axes[1].set_xlabel('Principal Component 1')
axes[1].set_ylabel('Principal Component 2')
axes[1].legend(handles=scatter.legend_elements()[0], labels=['No Failure', 'Failure'])

plt.tight_layout()
fig.savefig('decision_boundaries.png')
print("Saved decision boundary plot to decision_boundaries.png")


print('\nDone.')