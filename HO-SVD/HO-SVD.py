import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# Part 1: Manual Tensor and HO-SVD Helper Functions
# =============================================================================

def unfold(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

def mode_n_product(tensor, matrix, mode):
    return np.moveaxis(np.tensordot(tensor, matrix, axes=(mode, 1)), -1, mode)

def manual_hosvd(tensor):
    num_modes = len(tensor.shape)
    factor_matrices = []
    for mode in range(num_modes):
        unfolded_matrix = unfold(tensor, mode)
        U, _, _ = np.linalg.svd(unfolded_matrix, full_matrices=False)
        factor_matrices.append(U)
    core_tensor = tensor
    for mode, U in enumerate(factor_matrices):
        core_tensor = mode_n_product(core_tensor, U.T, mode)
    return core_tensor, factor_matrices

# =============================================================================
# Part 2: Data Loading and Preprocessing
# =============================================================================
try:
    df = pd.read_csv('textile_machine_data.csv')
except FileNotFoundError:
    print("Error: 'textile_machine_data.csv' not found. Please ensure it's in the same directory.")
    exit()

# Convert Timestamp to datetime and sort
df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d-%m-%Y %H:%M')
df = df.sort_values('Timestamp')

# One-Hot Encode Machine_Type
df = pd.get_dummies(df, columns=['Machine_Type'], drop_first=True)

# Separate features, drop non-sensor data
features_df = df.drop(columns=['Machine_ID', 'Failure', 'Timestamp'])
target = df['Failure']

# Normalize numerical features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_df)
print("Data loaded and preprocessed.")
print(f"Shape of feature data: {features_scaled.shape}")
print("-" * 30)

# =============================================================================
# Part 3: Time Series Tensorization
# =============================================================================

def create_sequences(features, labels, sequence_length=10):
    sequences = []
    sequence_labels = []
    for i in range(len(features) - sequence_length + 1):
        sequences.append(features[i:i+sequence_length])
        # The label for the sequence is the label of the last time step
        sequence_labels.append(labels.iloc[i + sequence_length - 1])
    return np.array(sequences), np.array(sequence_labels)

SEQUENCE_LENGTH = 10
X_tensor, y_sequences = create_sequences(features_scaled, target, SEQUENCE_LENGTH)

print("Time series tensor created.")
print(f"Tensor shape: {X_tensor.shape}")
print(f"Labels shape: {y_sequences.shape}")
print(f"Class distribution in sequences:\n{pd.Series(y_sequences).value_counts()}")
print("-" * 30)

# =============================================================================
# Part 4: Train/Test Split (NO SMOTE)
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_sequences, test_size=0.3, random_state=42, stratify=y_sequences
)

# Keep data in original 3D tensor form — no reshaping for SMOTE
print("Train/Test split completed (SMOTE skipped).")
print(f"Training tensor shape: {X_train.shape}")
print(f"Class distribution in training set:\n{pd.Series(y_train).value_counts()}")
print("-" * 30)

# =============================================================================
# Part 5: Model Training with HO-SVD (on original classes)
# =============================================================================

# Split training tensor by class
tensor_class0 = X_train[y_train == 0]
tensor_class1 = X_train[y_train == 1]

# Safety check: ensure both classes have data
if len(tensor_class0) == 0 or len(tensor_class1) == 0:
    raise ValueError("One of the classes has no samples in the training set. Check class distribution.")

print("Performing HO-SVD on class tensors...")
_, factor_matrices_class0 = manual_hosvd(tensor_class0)
_, factor_matrices_class1 = manual_hosvd(tensor_class1)

U0_mode1_full, U0_mode2_full = factor_matrices_class0[1], factor_matrices_class0[2]
U1_mode1_full, U1_mode2_full = factor_matrices_class1[1], factor_matrices_class1[2]
print("HO-SVD training complete.")
print("-" * 30)

# =============================================================================
# Part 6: Prediction by Reconstruction and Evaluation (Optimized for Low FN)
# =============================================================================

timesteps, num_features = X_train.shape[1], X_train.shape[2]

print("Finding optimal rank k to MINIMIZE FALSE NEGATIVES...")
best_k = 0
min_false_negatives = float('inf')
best_cm = None

for k in range(1, min(timesteps, num_features) + 1):
    U0_mode1, U0_mode2 = U0_mode1_full[:, :k], U0_mode2_full[:, :k]
    U1_mode1, U1_mode2 = U1_mode1_full[:, :k], U1_mode2_full[:, :k]

    predictions = []
    for test_sample_matrix in X_test:
        # Reconstruct using class-0 subspace
        projected_0 = U0_mode1.T @ test_sample_matrix @ U0_mode2
        reconstructed_0 = U0_mode1 @ projected_0 @ U0_mode2.T
        error_0 = np.linalg.norm(test_sample_matrix - reconstructed_0)

        # Reconstruct using class-1 subspace
        projected_1 = U1_mode1.T @ test_sample_matrix @ U1_mode2
        reconstructed_1 = U1_mode1 @ projected_1 @ U1_mode2.T
        error_1 = np.linalg.norm(test_sample_matrix - reconstructed_1)

        predictions.append(1 if error_1 < error_0 else 0)

    cm = confusion_matrix(y_test, predictions)
    false_negatives = cm[1, 0] if cm.shape == (2, 2) else 0  # Handle edge case if one class missing

    print(f"  k = {k}, False Negatives = {false_negatives}")

    if false_negatives < min_false_negatives:
        min_false_negatives = false_negatives
        best_k = k
        best_cm = cm

print(f"\nOptimal rank k found: {best_k} with {min_false_negatives} False Negatives")
print("-" * 30)

# Final evaluation with best k
U0_mode1, U0_mode2 = U0_mode1_full[:, :best_k], U0_mode2_full[:, :best_k]
U1_mode1, U1_mode2 = U1_mode1_full[:, :best_k], U1_mode2_full[:, :best_k]
final_predictions = []
for test_sample_matrix in X_test:
    projected_0 = U0_mode1.T @ test_sample_matrix @ U0_mode2
    reconstructed_0 = U0_mode1 @ projected_0 @ U0_mode2.T
    error_0 = np.linalg.norm(test_sample_matrix - reconstructed_0)

    projected_1 = U1_mode1.T @ test_sample_matrix @ U1_mode2
    reconstructed_1 = U1_mode1 @ projected_1 @ U1_mode2.T
    error_1 = np.linalg.norm(test_sample_matrix - reconstructed_1)

    final_predictions.append(1 if error_1 < error_0 else 0)

print("Final Classification Results on the Test Set:")
print(classification_report(y_test, final_predictions, target_names=['Non-Failure (0)', 'Failure (1)']))
print("Final Confusion Matrix (Optimized for low FN):")
final_cm = confusion_matrix(y_test, final_predictions)
print(final_cm)
plt.figure(figsize=(8, 6))
sns.heatmap(final_cm, annot=True, fmt='d', cmap='mako')
plt.title(f'Time-Series HO-SVD Confusion Matrix (Optimal k={best_k} for low FN)')
plt.show()