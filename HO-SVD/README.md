# Predictive Maintenance for Textile Machines using Tensor Decomposition (HO-SVD)

## Overview

This project implements a novel predictive maintenance model for textile machinery using a tensor-based approach. Instead of traditional feature engineering on time-series data, this model leverages **Higher-Order Singular Value Decomposition (HO-SVD)** to classify machine states as either "Failure" or "Non-Failure".

The core idea is to treat sequences of multivariate sensor data as tensors and learn the principal subspaces for each class. A new time-series sequence is then classified based on which subspace can reconstruct it with the lowest error. The model is specifically optimized to **minimize False Negatives**, as missing a real failure is typically the most costly error in an industrial setting.

## Key Features

-   **Tensor-Based Classification:** Utilizes HO-SVD, a form of tensor decomposition, to capture the complex, high-dimensional relationships in time-series sensor data.
-   **Time-Series Tensorization:** Converts raw tabular data into 3D tensors `(sequences, timesteps, features)` suitable for advanced analysis.
-   **Optimized for Predictive Maintenance:** The hyperparameter tuning process is explicitly designed to find the model configuration that minimizes False Negatives (missed failures).
-   **Rich Visualization Suite:** Automatically generates and saves a series of plots to help understand the data, the model's optimization process, and the final results.
-   **Pure Python/NumPy Implementation:** The core tensor logic is implemented from scratch using NumPy, making the underlying mechanics clear and easy to follow.

---

## How It Works

The classification pipeline follows these steps:

1.  **Data Preprocessing:** The `textile_machine_data.csv` is loaded, timestamps are parsed, and categorical features are one-hot encoded. All sensor features are standardized.
2.  **Tensorization:** The time-series data is transformed into overlapping sequences, creating a 3rd-order tensor `(Samples x Timesteps x Features)`.
3.  **Class-Based Subspace Learning:**
    -   The training tensors are separated by class (one tensor for all "Failure" sequences, one for "Non-Failure").
    -   HO-SVD is performed independently on each class tensor. This yields a set of orthogonal factor matrices (`U` matrices) for each class, which define the principal subspaces that best represent the patterns of that class.
4.  **Hyperparameter Tuning (Rank `k`):**
    -   The model iterates through different subspace ranks (`k`). The rank determines the complexity of the learned patterns.
    -   For each `k`, the model makes predictions on the test set.
    -   The `k` that results in the **lowest number of False Negatives** is selected as the optimal rank.
5.  **Classification by Reconstruction:**
    -   To classify a new test sample (a matrix), it is projected onto the learned subspaces of both Class 0 and Class 1 using the optimal factor matrices.
    -   It is then reconstructed from each projection.
    -   The reconstruction error (Frobenius norm of the difference) is calculated for both cases.
    -   The sample is assigned to the class that yields the **smaller reconstruction error**.

---

## Visualizations Generated

The script automatically creates a `visualizations/` directory and saves the following plots as `.png` files:

1.  **`01_class_distribution.png`**: A bar chart showing the distribution of "Failure" and "Non-Failure" classes in the dataset.
2.  **`02_sensor_data_over_time.png`**: A time-series plot of Temperature and Vibration, with actual failure events highlighted.
3.  **`03_errors_vs_k.png`**: Shows how the number of False Negatives and False Positives changes as the subspace rank `k` increases. Crucial for understanding the optimization process.
4.  **`04_precision_recall_vs_k.png`**: Illustrates the trade-off between Precision and Recall for the "Failure" class across different values of `k`.
5.  **`05_error_distribution.png`**: A density plot showing the distribution of reconstruction errors for both classes, which helps visualize the separability of the classes.
6.  **`06_final_confusion_matrix.png`**: A heatmap of the final confusion matrix using the best `k`, providing a clear summary of model performance.

---

## Prerequisites

This script requires the following Python libraries. You can install them using pip:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib