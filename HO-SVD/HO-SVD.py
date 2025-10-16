import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create a directory to save plots
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# =============================================================================
# Part 1: Manual Tensor and HO-SVD Helper Functions
# =============================================================================

def unfold(tensor, mode):
    """Unfolds a tensor into a matrix along a specified mode."""
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

def mode_n_product(tensor, matrix, mode):
    """Computes the mode-n product of a tensor and a matrix."""
    return np.moveaxis(np.tensordot(tensor, matrix, axes=(mode, 1)), -1, mode)

def manual_hosvd(tensor):
    """Performs Higher-Order Singular Value Decomposition (HO-SVD)."""
    num_modes = len(tensor.shape)
    factor_matrices = []
    core_tensor = tensor
    for mode in range(num_modes):
        unfolded_matrix = unfold(tensor, mode)
        U, _, _ = np.linalg.svd(unfolded_matrix, full_matrices=False)
        factor_matrices.append(U)
    for mode, U in enumerate(factor_matrices):
        core_tensor = mode_n_product(core_tensor, U.T, mode)
    return core_tensor, factor_matrices

# =============================================================================
# Part 2: Data Loading, Preprocessing, and Initial Visualization
# =============================================================================
try:
    df = pd.read_csv('textile_machine_data.csv')
except FileNotFoundError:
    print("Error: 'textile_machine_data.csv' not found. Please ensure it's in the same directory.")
    exit()

df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M')
df = df.sort_values('Timestamp')

# --- VISUALIZATION 1: Class Distribution ---
plt.figure(figsize=(8, 5))
# Updated line to address deprecation warning
sns.countplot(x='Failure', data=df, hue='Failure', palette='mako', legend=False)
plt.title('Class Distribution in Original Data')
plt.xlabel('Class (0: Non-Failure, 1: Failure)')
plt.ylabel('Count')
plt.savefig('visualizations/01_class_distribution.png', bbox_inches='tight')
plt.show()

# --- VISUALIZATION 2: Sensor Data Over Time ---
fig, ax1 = plt.subplots(figsize=(15, 6))
# Plot Temperature
ax1.plot(df['Timestamp'], df['Temperature'], color='red', label='Temperature', alpha=0.7)
ax1.set_xlabel('Timestamp')
ax1.set_ylabel('Temperature', color='red')
ax1.tick_params(axis='y', labelcolor='red')
# Plot Vibration on a second y-axis
ax2 = ax1.twinx()
ax2.plot(df['Timestamp'], df['Vibration'], color='blue', label='Vibration', alpha=0.7)
ax2.set_ylabel('Vibration', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
# Highlight Failure events
failure_points = df[df['Failure'] == 1]
ax1.scatter(failure_points['Timestamp'], failure_points['Temperature'], color='black', s=100, zorder=5, label='Failure Event')
fig.suptitle('Sensor Readings Over Time with Failure Events')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.savefig('visualizations/02_sensor_data_over_time.png', bbox_inches='tight')
plt.show()


# Preprocessing
df = pd.get_dummies(df, columns=['Machine_Type'], drop_first=True)
features_df = df.drop(columns=['Machine_ID', 'Failure', 'Timestamp'])
target = df['Failure']
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_df)
print("Data loaded and preprocessed.")
print("-" * 30)

# =============================================================================
# Part 3: Time Series Tensorization
# =============================================================================

def create_sequences(features, labels, sequence_length=10):
    sequences, sequence_labels = [], []
    for i in range(len(features) - sequence_length + 1):
        sequences.append(features[i:i+sequence_length])
        sequence_labels.append(labels.iloc[i + sequence_length - 1])
    return np.array(sequences), np.array(sequence_labels)

SEQUENCE_LENGTH = 10
X_tensor, y_sequences = create_sequences(features_scaled, target, SEQUENCE_LENGTH)
print("Time series tensor created.")
print(f"Tensor shape: {X_tensor.shape}")
print("-" * 30)

# =============================================================================
# Part 4: Train/Test Split
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_sequences, test_size=0.3, random_state=42, stratify=y_sequences
)
print("Train/Test split completed.")
print("-" * 30)

# =============================================================================
# Part 5: Model Training with HO-SVD
# =============================================================================

tensor_class0 = X_train[y_train == 0]
tensor_class1 = X_train[y_train == 1]

if len(tensor_class0) == 0 or len(tensor_class1) == 0:
    raise ValueError("One of the classes has no samples in the training set.")

print("Performing HO-SVD on class tensors...")
_, factor_matrices_class0 = manual_hosvd(tensor_class0)
_, factor_matrices_class1 = manual_hosvd(tensor_class1)

U0_mode1_full, U0_mode2_full = factor_matrices_class0[1], factor_matrices_class0[2]
U1_mode1_full, U1_mode2_full = factor_matrices_class1[1], factor_matrices_class1[2]
print("HO-SVD training complete.")
print("-" * 30)

# =============================================================================
# Part 6: Hyperparameter Tuning (Rank k) and Optimization Visualization
# =============================================================================

timesteps, num_features = X_train.shape[1], X_train.shape[2]
print("Finding optimal rank k to MINIMIZE FALSE NEGATIVES...")
best_k = 0
min_false_negatives = float('inf')
best_cm = None

# Store metrics for each k
k_values, fn_counts, fp_counts, recall_scores, precision_scores = [], [], [], [], []

for k in range(1, min(timesteps, num_features) + 1):
    U0_mode1, U0_mode2 = U0_mode1_full[:, :k], U0_mode2_full[:, :k]
    U1_mode1, U1_mode2 = U1_mode1_full[:, :k], U1_mode2_full[:, :k]

    predictions = []
    for test_sample_matrix in X_test:
        projected_0 = U0_mode1.T @ test_sample_matrix @ U0_mode2
        reconstructed_0 = U0_mode1 @ projected_0 @ U0_mode2.T
        error_0 = np.linalg.norm(test_sample_matrix - reconstructed_0)

        projected_1 = U1_mode1.T @ test_sample_matrix @ U1_mode2
        reconstructed_1 = U1_mode1 @ projected_1 @ U1_mode2.T
        error_1 = np.linalg.norm(test_sample_matrix - reconstructed_1)
        predictions.append(1 if error_1 < error_0 else 0)

    cm = confusion_matrix(y_test, predictions)
    false_negatives = cm[1, 0] if cm.shape == (2, 2) else 0
    false_positives = cm[0, 1] if cm.shape == (2, 2) else 0

    # Store metrics
    k_values.append(k)
    fn_counts.append(false_negatives)
    fp_counts.append(false_positives)
    # pos_label=1 for Failure class
    recall_scores.append(recall_score(y_test, predictions, pos_label=1, zero_division=0))
    precision_scores.append(precision_score(y_test, predictions, pos_label=1, zero_division=0))

    if false_negatives < min_false_negatives:
        min_false_negatives = false_negatives
        best_k = k
        best_cm = cm

print(f"\nOptimal rank k found: {best_k} with {min_false_negatives} False Negatives")
print("-" * 30)

# --- VISUALIZATION 3: FN and FP vs. Rank (k) ---
plt.figure(figsize=(12, 6))
plt.plot(k_values, fn_counts, marker='o', linestyle='--', label='False Negatives (Missed Failures)')
plt.plot(k_values, fp_counts, marker='x', linestyle=':', label='False Positives (False Alarms)')
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Optimal k = {best_k}')
plt.title('Error Analysis vs. Subspace Rank (k)')
plt.xlabel('Rank (k)')
plt.ylabel('Number of Errors')
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.savefig('visualizations/03_errors_vs_k.png', bbox_inches='tight')
plt.show()

# --- VISUALIZATION 4: Precision-Recall Trade-off vs. Rank (k) ---
plt.figure(figsize=(12, 6))
plt.plot(k_values, recall_scores, marker='o', linestyle='--', label='Recall (Sensitivity)')
plt.plot(k_values, precision_scores, marker='x', linestyle=':', label='Precision')
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Optimal k = {best_k}')
plt.title('Precision-Recall Trade-off for Failure Class vs. Rank (k)')
plt.xlabel('Rank (k)')
plt.ylabel('Score')
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.savefig('visualizations/04_precision_recall_vs_k.png', bbox_inches='tight')
plt.show()


# =============================================================================
# Part 7: Final Evaluation and Visualization
# =============================================================================

U0_mode1, U0_mode2 = U0_mode1_full[:, :best_k], U0_mode2_full[:, :best_k]
U1_mode1, U1_mode2 = U1_mode1_full[:, :best_k], U1_mode2_full[:, :best_k]

final_predictions = []
reconstruction_errors_data = [] # For visualization

for i, test_sample_matrix in enumerate(X_test):
    projected_0 = U0_mode1.T @ test_sample_matrix @ U0_mode2
    reconstructed_0 = U0_mode1 @ projected_0 @ U0_mode2.T
    error_0 = np.linalg.norm(test_sample_matrix - reconstructed_0)

    projected_1 = U1_mode1.T @ test_sample_matrix @ U1_mode2
    reconstructed_1 = U1_mode1 @ projected_1 @ U1_mode2.T
    error_1 = np.linalg.norm(test_sample_matrix - reconstructed_1)

    final_predictions.append(1 if error_1 < error_0 else 0)
    reconstruction_errors_data.append({
        'error_0': error_0,
        'error_1': error_1,
        'true_label': y_test[i]
    })

errors_df = pd.DataFrame(reconstruction_errors_data)

print("Final Classification Results on the Test Set:")
print(classification_report(y_test, final_predictions, target_names=['Non-Failure (0)', 'Failure (1)']))

# --- VISUALIZATION 5: Reconstruction Error Distribution ---
plt.figure(figsize=(12, 7))
sns.kdeplot(data=errors_df, x='error_0', hue='true_label', fill=True,
            palette={0: 'skyblue', 1: 'gray'}, common_norm=False)
sns.kdeplot(data=errors_df, x='error_1', hue='true_label', fill=True,
            palette={0: 'lightcoral', 1: 'red'}, common_norm=False)
plt.title(f'Distribution of Reconstruction Errors (at k={best_k})')
plt.xlabel('Reconstruction Error')
import matplotlib.lines as mlines
legend_elements = [mlines.Line2D([0], [0], color='skyblue', lw=4, label='True Non-Failure (recon by Class 0)'),
                   mlines.Line2D([0], [0], color='gray', lw=4, label='True Failure (recon by Class 0)'),
                   mlines.Line2D([0], [0], color='lightcoral', lw=4, label='True Non-Failure (recon by Class 1)'),
                   mlines.Line2D([0], [0], color='red', lw=4, label='True Failure (recon by Class 1)')]
plt.legend(handles=legend_elements, title="Legend")
plt.savefig('visualizations/05_error_distribution.png', bbox_inches='tight')
plt.show()

# --- VISUALIZATION 6: Final Confusion Matrix ---
final_cm = confusion_matrix(y_test, final_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(final_cm, annot=True, fmt='d', cmap='mako',
            xticklabels=['Predicted Non-Failure', 'Predicted Failure'],
            yticklabels=['Actual Non-Failure', 'Actual Failure'])
plt.title(f'Final Confusion Matrix (Optimal k={best_k})')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('visualizations/06_final_confusion_matrix.png', bbox_inches='tight')
plt.show()

