import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns



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


df = pd.read_csv('textile_machine_data.csv')

#droppping Machine_ID, and Timestamp since they are not needed, and might cause a data leakage
#dropping Operating_Hours, Noise_Level, Bearing_Temperature, Temperature, Motor_Current since they are the five most highly correlated features (high correlation with Failure)

features_to_drop = [
    'Machine_ID', 'Timestamp', 'Operating_Hours', 
    'Noise_Level', 'Bearing_Temperature', 'Temperature', 'Motor_Current'
]
df = df.drop(columns=features_to_drop)
print(f"Dropped columns: {features_to_drop}")


# One-hot encode Machine_Type
if 'Machine_Type' in df.columns:
    df = pd.get_dummies(df, columns=['Machine_Type'], drop_first=True)

#convert all the numerical features to float and the target to int
features = [c for c in df.columns if c != 'Failure']
X = df[features].values.astype(float)
y = df['Failure'].values.astype(int)

# the class proportions are biased if we do a normal split. 
# A stratified split ensures that the training and testing sets have the same proportion of classes (or labels) as the original dataset. 
# imbalanced datasets, where some classes are underrepresented.
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
print('\nTrain/test sizes:', X_train_raw.shape, X_test_raw.shape)


scaler = StandardScaler().fit(X_train_raw)
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# pca for evaluation (k=5)
k_eval = min(5, X_train.shape[1])
Z_train_eval, _, evr_eval, X_mean_eval, V_k_eval = manual_pca(X_train, k=k_eval)
X_test_centered_eval = X_test - X_mean_eval
Z_test_eval = X_test_centered_eval @ V_k_eval
print('\nManual PCA (for evaluation) explained variance ratio (train):', evr_eval)


# rkhs kernel 
# there are different types of kernels like
# - Linear Kernel
# - Polynomial Kernel
# - Sigmoid Kernel
# here we use RBF kernel, that is
# K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
sigma = np.std(Z_train_eval) if np.std(Z_train_eval) > 0 else 1.0
K_train = rbf_kernel(Z_train_eval, sigma=sigma)
K_test = np.exp(-np.sum((Z_test_eval[:, None, :] - Z_train_eval[None, :, :])**2, axis=2) / (2 * sigma**2))


# hyperparameter tuning with cross-validation
# hyperparameter values that give the best model performance. 
#hyperparameters are parameters that are set before the learning process begins and are not learned from the data itself.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 1) Logistic Regression on PCA coordinates  
pipe_log = Pipeline([('clf', LogisticRegression(max_iter=2000, random_state=42))])
param_grid_log = {'clf__C': [0.01, 0.1, 1, 10]}
grid_log = GridSearchCV(pipe_log, param_grid_log, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
grid_log.fit(Z_train_eval, y_train)
best_log = grid_log.best_estimator_
print("\nBest Logistic parameters (via GridSearchCV):", grid_log.best_params_)

# 2) RBF SVM on PCA coords 
pipe_svc = Pipeline([('svc', SVC(kernel='rbf', probability=True, random_state=42))])
param_grid_svc = {'svc__C': [0.1, 1, 10], 'svc__gamma': ['scale', 'auto', 0.01, 0.1]}
grid_svc = GridSearchCV(pipe_svc, param_grid_svc, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
grid_svc.fit(Z_train_eval, y_train)
best_svc = grid_svc.best_estimator_
print("Best SVC parameters (via GridSearchCV):", grid_svc.best_params_)

# Predictions on test set using best estimators(the set of hyperparameter values
# that produced the highest cross‑validated score during your search)
y_pred_pca = best_log.predict(Z_test_eval)
try:
    scores_pca = best_log.predict_proba(Z_test_eval)[:, 1]
except Exception:
    scores_pca = best_log.decision_function(Z_test_eval)

y_pred_k = best_svc.predict(Z_test_eval)
scores_k = best_svc.predict_proba(Z_test_eval)[:, 1]

#cross-validated per-fold F1 scores (on training data, for the selected best models)
cv_scores_log = cross_val_score(best_log, Z_train_eval, y_train, cv=cv, scoring='f1', n_jobs=-1)
cv_scores_svc = cross_val_score(best_svc, Z_train_eval, y_train, cv=cv, scoring='f1', n_jobs=-1)

print(f"\nCV F1 (Logistic) per-fold: {cv_scores_log}")
print(f"CV F1 (Logistic) mean={cv_scores_log.mean():.4f}, std={cv_scores_log.std():.4f}")
print(f"\nCV F1 (SVC) per-fold: {cv_scores_svc}")
print(f"CV F1 (SVC)     mean={cv_scores_svc.mean():.4f}, std={cv_scores_svc.std():.4f}")

print(f"\nGridSearch best cross-validated F1 (Logistic): {grid_log.best_score_:.4f}")
print(f"GridSearch best cross-validated F1 (SVC):      {grid_svc.best_score_:.4f}")


#classification reports
def print_metrics(y_true, y_pred, label):
    print(f"\n--- {label} ---")
    print('Accuracy:', accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))

print_metrics(y_test, y_pred_pca, 'Logistic on PCA')
print_metrics(y_test, y_pred_k, 'Kernel SVM on RKHS')

# ROC plots: Receiver Operating Characteristic - AUC plots: Area Under Curve

fpr_p, tpr_p, _ = roc_curve(y_test, scores_pca)
roc_auc_p = auc(fpr_p, tpr_p)
fpr_k, tpr_k, _ = roc_curve(y_test, scores_k)
roc_auc_k = auc(fpr_k, tpr_k)


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

#Decision boundary plot
print("\nDecision boundary plot")

# Step 1: Reduce the data to 2D PCA for visualization
k_viz = 2
Z_train_viz, _, _, _, _ = manual_pca(X_train, k=k_viz)

# Step 2: Train 2 classifiers on the 2D data
clf_viz = LogisticRegression(random_state=42).fit(Z_train_viz, y_train)

# We use a standard RBF SVM here for visualization
svc_viz = SVC(kernel='rbf', gamma='auto', random_state=42).fit(Z_train_viz, y_train)

# Step 3: Build a dense grid that covers teh 2D PCA area
x_min, x_max = Z_train_viz[:, 0].min() - 1, Z_train_viz[:, 0].max() + 1
y_min, y_max = Z_train_viz[:, 1].min() - 1, Z_train_viz[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))


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


print('\nImplementation of a complete RKHS pipeline is done')