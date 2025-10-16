"""Tensor-based Dynamic Mode Decomposition (DMD) classifier for textile failure prediction.

This script ingests the provided textile machine telemetry, constructs
tensor snapshots over sliding temporal windows, extracts DMD-driven
features, and trains a binary classifier to distinguish failure versus
non-failure regimes.

Usage (from the project root)::

	python DMD_Tensor.py --data-path C:/Users/mechi/Documents/GitHub/MathProj_TextileFailurePrediction/textile_machine_data.csv --classifier gradient_boosting --grid-search --output-dir artifacts   

Optional arguments expose the window length, DMD rank, target classifier,
and export paths for engineered features and the fitted model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike, NDArray
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	classification_report,
	confusion_matrix,
	precision_recall_fscore_support,
	roc_auc_score,
	roc_curve,
	ConfusionMatrixDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DMDTensorConfig:
	"""Bundle runtime configuration for the Tensor DMD classifier."""

	data_path: Path
	window_size: int = 12
	dmd_rank: int = 8
	top_modes: int = 6
	test_size: float = 0.2
	random_state: int = 42
	classifier: str = "logreg"
	grid_search: bool = False
	compare_classifiers: bool = False
	use_pca: bool = True
	pca_components: Optional[int] = None
	pca_variance_ratio: float = 0.95
	output_dir: Optional[Path] = None
	export_features_path: Optional[Path] = None
	export_model_path: Optional[Path] = None


@dataclass
class PCAProjection:
	"""Container for the ingredients of a PCA projection learned from scratch."""

	mean_vector: NDArray[np.float64]
	components: NDArray[np.float64]
	explained_variance: NDArray[np.float64]
	explained_variance_ratio: NDArray[np.float64]


@dataclass
class EvaluationArtifacts:
	"""Cache the evaluation-time arrays needed for interpretability artefacts."""

	y_test: NDArray[np.int64]
	y_pred: NDArray[np.int64]
	y_proba: NDArray[np.float64]
	X_test: NDArray[np.float64]
	feature_names: Sequence[str]


class PCAFromScratch:
	r"""Principal Component Analysis solved via covariance eigen-decomposition.

	Step-by-step outline:
	1. Centre the feature matrix so every column has zero empirical mean.
	2. Estimate the covariance matrix using the unbiased formula
	   (1 / (N - 1)) * X_c^T X_c where X_c denotes the centred data.
	3. Solve the eigenvalue problem to obtain principal directions and the
	   amount of variance each direction explains.
	4. Project the centred data onto the leading eigenvectors to form the
	   decorrelated principal components.
	"""

	def __init__(
		self,
		n_components: Optional[int] = None,
		variance_ratio: Optional[float] = None,
	) -> None:
		if n_components is not None and n_components <= 0:
			raise ValueError("n_components must be positive if supplied")
		if variance_ratio is not None and not (0.0 < variance_ratio <= 1.0):
			raise ValueError("variance_ratio must lie in (0, 1]")

		self.n_components = n_components
		self.variance_ratio = variance_ratio
		self.projection_: Optional[PCAProjection] = None

	# ------------------------------------------------------------------
	# Fitting phase
	# ------------------------------------------------------------------
	def fit(self, X: NDArray[np.float64]) -> "PCAFromScratch":
		if X.ndim != 2:
			raise ValueError("PCA expects a 2D matrix of shape (n_samples, n_features)")

		# 1. Centre the data so every feature has zero empirical mean.
		mean_vector = X.mean(axis=0)
		X_centered = X - mean_vector

		# 2. Estimate the covariance matrix via the unbiased estimator.
		#    Sigma = (1 / (N-1)) * X_c^T X_c
		n_samples = X.shape[0]
		if n_samples < 2:
			raise ValueError("At least two samples are required to compute covariance")
		covariance_matrix = (X_centered.T @ X_centered) / (n_samples - 1)

		# 3. Decompose the symmetric covariance matrix. eigh() exploits symmetry
		#    and guarantees sorted real eigenvalues / eigenvectors.
		eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

		# 4. Sort the eigenpairs from largest eigenvalue to smallest to prioritise
		#    directions with maximal variance conservation.
		sort_indices = np.argsort(eigenvalues)[::-1]
		eigenvalues = eigenvalues[sort_indices]
		eigenvectors = eigenvectors[:, sort_indices]

		# 5. Determine how many components to keep either by explicit count or
		#    cumulative explained variance threshold.
		if self.n_components is not None:
			k = min(self.n_components, eigenvectors.shape[1])
		elif self.variance_ratio is not None:
			cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
			k = int(np.searchsorted(cumulative, self.variance_ratio) + 1)
		else:
			k = eigenvectors.shape[1]

		components = eigenvectors[:, :k]
		explained_variance = eigenvalues[:k]
		explained_variance_ratio = explained_variance / eigenvalues.sum()

		self.projection_ = PCAProjection(
			mean_vector=mean_vector.astype(np.float64),
			components=components.astype(np.float64),
			explained_variance=explained_variance.astype(np.float64),
			explained_variance_ratio=explained_variance_ratio.astype(np.float64),
		)
		return self

	# ------------------------------------------------------------------
	# Transformation phase
	# ------------------------------------------------------------------
	def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
		if self.projection_ is None:
			raise RuntimeError("PCA must be fitted before calling transform")

		# Project the centred observations onto the retained principal directions.
		X_centered = X - self.projection_.mean_vector
		return X_centered @ self.projection_.components

	def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
		return self.fit(X).transform(X)



# ---------------------------------------------------------------------------
# DMD feature extraction utilities
# ---------------------------------------------------------------------------


class TensorDMDFeatureExtractor:
	"""Compute Tensor-DMD descriptors for a sequence of multivariate states."""

	def __init__(self, rank: int = 8, top_modes: int = 6) -> None:
		if rank <= 0:
			raise ValueError("rank must be a positive integer")
		if top_modes <= 0:
			raise ValueError("top_modes must be a positive integer")
		self.rank = rank
		self.top_modes = top_modes

	def transform(self, window: NDArray[np.float64]) -> NDArray[np.float64]:
		r"""Compute Dynamic Mode Decomposition features for a time window.

		Method overview:

		1. Build two snapshot matrices, X and Y, containing successive observations
		   that differ by one timestep so that Y represents the future state of X.
		2. Perform a rank-r singular value decomposition (SVD) of X to isolate the
		   dominant spatial patterns in an orthonormal basis.
		3. Form the reduced linear model A_tilde = U_r.T @ Y @ V_r @ Sigma_r^{-1},
		   which describes the dynamics in the low-dimensional subspace.
		4. Solve the eigenvalue problem for A_tilde to obtain characteristic
		   frequencies/growth rates and lift the corresponding modes back to the
		   original feature space.
		5. Summarise the modal behaviour (eigenvalues, amplitudes, reconstruction
		   error) together with simple window statistics to provide classifier-ready
		   features.
		"""

		if window.ndim != 2:
			raise ValueError("window must be 2-dimensional")
		if window.shape[0] < 3:
			raise ValueError("window must contain at least 3 timesteps for DMD")

		# STEP 1 -------------------------------------------------------------
		# Build the pair of snapshot matrices X and Y by shifting the temporal
		# window by one time-step. Each column captures the state at time k.
		X = window[:-1].T  # shape: (n_features, window_size - 1)
		Y = window[1:].T   # shape: (n_features, window_size - 1)

		try:
			# STEP 2 ---------------------------------------------------------
			# Compute the compact SVD of X = U Σ V^H. The singular values quantify
			# the energy carried by each orthogonal direction in the data.
			U, singular_vals, Vh = np.linalg.svd(X, full_matrices=False)
		except np.linalg.LinAlgError:
			# In degenerate cases fall back to zeros while keeping output shape stable
			return np.zeros(self._feature_length(window.shape[1]), dtype=np.float64)

		r = int(min(self.rank, len(singular_vals), X.shape[0], X.shape[1]))
		if r == 0:
			return np.zeros(self._feature_length(window.shape[1]), dtype=np.float64)

		# Retain the leading r singular components.
		U_r = U[:, :r]
		S_r = singular_vals[:r]
		V_r = Vh.conj().T[:, :r]

		# STEP 3 -------------------------------------------------------------
		# Assemble the reduced-order linear operator (A_tilde) that advances the
		# system dynamics in the subspace spanned by U_r.
		S_inv = np.diag(1.0 / S_r)
		A_tilde = U_r.T @ Y @ V_r @ S_inv
		eigvals, eigvecs = np.linalg.eig(A_tilde)

		# STEP 4 -------------------------------------------------------------
		# Sort the eigenpairs by their magnitude to emphasise dynamically dominant
		# modes. Pull the DMD modes back into the original feature space.
		order = np.argsort(-np.abs(eigvals))
		eigvals = eigvals[order]
		eigvecs = eigvecs[:, order]

		modes = Y @ V_r @ S_inv @ eigvecs

		# STEP 5 -------------------------------------------------------------
		# Quantify the fidelity of the rank-r approximation by reconstructing X
		# and comparing the residual energy with the original signal.
		X_rank_r = U_r @ np.diag(S_r) @ V_r.T
		recon_error = np.linalg.norm(X - X_rank_r) / (np.linalg.norm(X) + 1e-8)

		# Modal amplitudes from least squares fit to the initial state
		try:
			amplitudes = np.linalg.lstsq(modes, X[:, 0], rcond=None)[0]
		except np.linalg.LinAlgError:
			amplitudes = np.zeros(r, dtype=np.complex128)

		features: List[float] = []

		features.extend(self._pad(np.real(eigvals), self.top_modes))
		features.extend(self._pad(np.imag(eigvals), self.top_modes))
		features.extend(self._pad(np.abs(eigvals), self.top_modes))
		features.extend(self._pad(np.angle(eigvals), self.top_modes))

		singular_energy = (S_r ** 2) / (singular_vals[:r] ** 2).sum()
		features.extend(self._pad(singular_energy, self.top_modes))

		features.append(recon_error)

		features.extend(self._pad(np.real(amplitudes), self.top_modes))
		features.extend(self._pad(np.imag(amplitudes), self.top_modes))

		# STEP 6 -------------------------------------------------------------
		# Temporal summary statistics complement the modal decomposition and
		# capture slower drifts not fully described by oscillatory modes.
		feature_means = window.mean(axis=0)
		feature_stds = window.std(axis=0)
		feature_trends = window[-1] - window[0]

		features.extend(feature_means.tolist())
		features.extend(feature_stds.tolist())
		features.extend(feature_trends.tolist())

		# Global window descriptors
		features.append(float(window.mean()))
		features.append(float(window.std()))
		features.append(float(np.linalg.norm(window[-1] - window[-2])))

		return np.asarray(features, dtype=np.float64)

	def feature_names(self, base_feature_count: int) -> List[str]:
		names: List[str] = []
		for suffix in ("eig_real", "eig_imag", "eig_abs", "eig_phase",
					   "sv_energy", "amp_real", "amp_imag"):
			names.extend([f"{suffix}_{i}" for i in range(self.top_modes)])
		names.append("recon_error")
		for suffix in ("mean", "std", "trend"):
			names.extend([f"{suffix}_{i}" for i in range(base_feature_count)])
		names.extend(["window_mean", "window_std", "last_step_delta_norm"])
		return names

	def _pad(self, array: Sequence[float], length: int) -> List[float]:
		padded = list(array)[:length]
		if len(padded) < length:
			padded.extend([0.0] * (length - len(padded)))
		return padded

	def _feature_length(self, base_feature_count: int) -> int:
		return len(self.feature_names(base_feature_count))


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Dataset not found at {path}")

	df = pd.read_csv(path)
	if "Timestamp" in df.columns:
		df["Timestamp"] = pd.to_datetime(df["Timestamp"])
		df = df.sort_values("Timestamp").reset_index(drop=True)
	return df


def build_feature_matrix(df: pd.DataFrame) -> Tuple[NDArray[np.float64], NDArray[np.int64], List[str]]:
	df = df.copy()

	drop_cols = [col for col in ("Machine_ID", "Timestamp") if col in df.columns]
	if drop_cols:
		df = df.drop(columns=drop_cols)

	feature_cols = [
		"Temperature",
		"Vibration",
		"RPM",
		"Power_Consumption",
		"Operating_Hours",
		"Motor_Current",
		"Bearing_Temperature",
		"Noise_Level",
		"Yarn_Tension",
	]

	cat_cols = ["Machine_Type"] if "Machine_Type" in df.columns else []
	target_col = "Failure"

	if target_col not in df.columns:
		raise ValueError("Dataset must include a 'Failure' column for supervision")

	transformers = []
	if feature_cols:
		# Standardise continuous sensors so each contributes comparably (zero mean / unit variance).
		transformers.append(("num", StandardScaler(), feature_cols))
	if cat_cols:
		transformers.append((
			"cat",
			OneHotEncoder(handle_unknown="ignore"),
			cat_cols,
		))

	column_transformer = ColumnTransformer(transformers)
	feature_array = column_transformer.fit_transform(df)

	# Capture generated feature names for interpretability
	feature_names: List[str] = []
	if feature_cols:
		feature_names.extend(feature_cols)
	if cat_cols:
		encoder: OneHotEncoder = column_transformer.named_transformers_["cat"]
		feature_names.extend(encoder.get_feature_names_out(cat_cols).tolist())

	if hasattr(feature_array, "toarray"):
		feature_array = feature_array.toarray()

	labels = df[target_col].astype(int).to_numpy()
	return feature_array.astype(np.float64), labels, feature_names


def window_tensor(
	feature_matrix: NDArray[np.float64],
	labels: NDArray[np.int64],
	window_size: int,
	extractor: TensorDMDFeatureExtractor,
) -> Tuple[NDArray[np.float64], NDArray[np.int64], List[str]]:
	if window_size < 3:
		raise ValueError("window_size must be >= 3")

	samples: List[NDArray[np.float64]] = []
	sample_labels: List[int] = []

	for start in range(0, feature_matrix.shape[0] - window_size + 1):
		end = start + window_size
		window = feature_matrix[start:end, :]
		label = int(labels[end - 1])
		# Each sliding window encodes a short-term dynamical regime. DMD transforms
		# this tensor slice into modal features that the classifier can ingest.
		features = extractor.transform(window)
		samples.append(features)
		sample_labels.append(label)

	if not samples:
		raise ValueError(
			"Insufficient data to form any windows. Reduce window size or collect more data."
		)

	feature_names = extractor.feature_names(feature_matrix.shape[1])
	return np.vstack(samples), np.asarray(sample_labels, dtype=np.int64), feature_names


# ---------------------------------------------------------------------------
# Modelling utilities
# ---------------------------------------------------------------------------


def make_classifier(name: str, random_state: int) -> Tuple[Pipeline, dict]:
	name = name.lower()
	if name == "logreg":
		pipeline = Pipeline([
			("scaler", StandardScaler()),
			("clf", LogisticRegression(
				max_iter=2000,
				class_weight="balanced",
				solver="lbfgs",
				random_state=random_state,
			)),
		])
		param_grid = {
			"clf__C": [0.1, 1.0, 5.0],
			"clf__penalty": ["l2"],
		}
	elif name == "gradient_boosting":
		pipeline = Pipeline([
			("clf", GradientBoostingClassifier(random_state=random_state)),
		])
		param_grid = {
			"clf__n_estimators": [150, 250],
			"clf__learning_rate": [0.05, 0.1],
			"clf__max_depth": [2, 3],
		}
	elif name == "random_forest":
		pipeline = Pipeline([
			("clf", RandomForestClassifier(
				n_estimators=300,
				class_weight="balanced",
				random_state=random_state,
				n_jobs=-1,
			)),
		])
		param_grid = {
			"clf__max_depth": [None, 8, 12],
			"clf__min_samples_split": [2, 5, 10],
			"clf__min_samples_leaf": [1, 2, 4],
		}
	else:
		raise ValueError(f"Unsupported classifier '{name}'")

	return pipeline, param_grid


def compute_feature_influence(
	pipeline: Pipeline,
	feature_names: Sequence[str],
) -> List[Dict[str, float]]:
	"""Estimate feature influence using model-dependent attributes."""

	if not feature_names:
		return []

	clf = pipeline.named_steps.get("clf")
	if clf is None:
		return []

	importance: Optional[np.ndarray] = None
	signed_weights: Optional[np.ndarray] = None

	if hasattr(clf, "coef_"):
		coef = np.asarray(clf.coef_)
		if coef.ndim == 2:
			coef = coef[0]
		signed_weights = coef.astype(float, copy=True)
		scaler = pipeline.named_steps.get("scaler")
		if scaler is not None and hasattr(scaler, "scale_"):
			scale = np.asarray(scaler.scale_)
			if scale.shape[0] == signed_weights.shape[0]:
				# Rescale coefficients back to the original feature scale for readability.
				signed_weights = signed_weights / np.where(scale == 0, 1.0, scale)
		importance = np.abs(signed_weights)
	elif hasattr(clf, "feature_importances_"):
		importance = np.asarray(getattr(clf, "feature_importances_"), dtype=float)

	if importance is None:
		return []

	importance = importance.astype(float)
	if signed_weights is None:
		signed_weights = importance.copy()

	order = np.argsort(importance)[::-1]
	results: List[Dict[str, float]] = []
	for idx in order:
		if idx >= len(feature_names):
			continue
		results.append({
			"feature": str(feature_names[idx]),
			"score": float(importance[idx]),
			"weight": float(signed_weights[idx]),
		})
	return results


def save_metrics_json(metrics: dict, destination: Path) -> None:
	"""Persist metrics to JSON with numpy-friendly serialisation."""

	def _default(obj: object) -> object:
		if isinstance(obj, (np.generic,)):
			return obj.item()
		raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")

	with destination.open("w", encoding="utf-8") as fh:
		json.dump(metrics, fh, indent=2, default=_default)


def save_roc_curve_plot(
	artifacts: EvaluationArtifacts,
	destination: Path,
	classifier_label: str,
) -> None:
	"""Render and store ROC curve using held-out predictions."""

	fpr, tpr, _ = roc_curve(artifacts.y_test, artifacts.y_proba)
	fig, ax = plt.subplots(figsize=(5.0, 4.0))
	ax.plot(fpr, tpr, label=f"{classifier_label} (AUC={roc_auc_score(artifacts.y_test, artifacts.y_proba):.3f})")
	ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
	ax.set_xlabel("False Positive Rate")
	ax.set_ylabel("True Positive Rate")
	ax.set_title("ROC Curve")
	ax.legend(loc="lower right")
	ax.grid(alpha=0.3)
	plt.tight_layout()
	fig.savefig(destination, dpi=300)
	plt.close(fig)


def save_decision_boundary_plot(
	artifacts: EvaluationArtifacts,
	pipeline: Pipeline,
	destination: Path,
	classifier_label: str,
) -> None:
	"""Generate a 2D decision boundary on the leading feature axes when feasible."""

	if artifacts.X_test.shape[1] < 2:
		print("Decision boundary plot skipped: need at least two features.")
		return

	x_coords = artifacts.X_test[:, 0]
	y_coords = artifacts.X_test[:, 1]
	x_margin = (x_coords.max() - x_coords.min()) * 0.05 or 1.0
	y_margin = (y_coords.max() - y_coords.min()) * 0.05 or 1.0
	x_min, x_max = x_coords.min() - x_margin, x_coords.max() + x_margin
	y_min, y_max = y_coords.min() - y_margin, y_coords.max() + y_margin

	grid_x, grid_y = np.meshgrid(
		np.linspace(x_min, x_max, 200),
		np.linspace(y_min, y_max, 200),
	)
	baseline = artifacts.X_test.mean(axis=0)
	grid_points = np.tile(baseline, (grid_x.size, 1))
	grid_points[:, 0] = grid_x.ravel()
	grid_points[:, 1] = grid_y.ravel()
	probabilities = pipeline.predict_proba(grid_points)[:, 1]
	probabilities = probabilities.reshape(grid_x.shape)

	fig, ax = plt.subplots(figsize=(5.5, 4.5))
	contour = ax.contourf(grid_x, grid_y, probabilities, levels=20, cmap="coolwarm", alpha=0.7)
	plt.colorbar(contour, ax=ax, label="Failure probability")
	_ = ax.scatter(
		x_coords,
		y_coords,
		c=artifacts.y_test,
		cmap="coolwarm",
		edgecolor="k",
		s=35,
		label="Test samples",
	)
	ax.set_xlabel(str(artifacts.feature_names[0]))
	ax.set_ylabel(str(artifacts.feature_names[1]))
	ax.set_title(f"Decision Landscape: {classifier_label}")
	ax.legend(loc="upper right")
	plt.tight_layout()
	fig.savefig(destination, dpi=300)
	plt.close(fig)


def save_feature_influence_table(top_features: List[Dict[str, float]], destination: Path) -> None:
	"""Persist ranked feature influences to CSV for auditability."""

	if not top_features:
		return
	frame = pd.DataFrame(top_features)
	frame.to_csv(destination, index=False)


def train_and_evaluate(
	X: NDArray[np.float64],
	y: NDArray[np.int64],
	cfg: DMDTensorConfig,
	feature_names: Sequence[str],
) -> Tuple[Pipeline, dict, EvaluationArtifacts]:
	"""Train the chosen classifier and report an extensive metrics dictionary."""
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=cfg.test_size,
		stratify=y,
		random_state=cfg.random_state,
	)

	pipeline, param_grid = make_classifier(cfg.classifier, cfg.random_state)

	best_params = None
	if cfg.grid_search:
		search = GridSearchCV(
			pipeline,
			param_grid=param_grid,
			cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state),
			scoring="roc_auc",
			n_jobs=-1,
		)
		search.fit(X_train, y_train)
		pipeline = search.best_estimator_
		best_params = search.best_params_
	else:
		pipeline.fit(X_train, y_train)

	y_pred = pipeline.predict(X_test)
	y_proba = pipeline.predict_proba(X_test)[:, 1]

	precision, recall, f1, _ = precision_recall_fscore_support(
		y_test, y_pred, average="binary", zero_division=0
	)

	metrics = {
		"accuracy": float((y_pred == y_test).mean()),
		"precision": float(precision),
		"recall": float(recall),
		"f1": float(f1),
		"roc_auc": float(roc_auc_score(y_test, y_proba)),
		"confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
		"classification_report": classification_report(y_test, y_pred),
	}

	if best_params is not None:
		metrics["best_params"] = best_params

	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.random_state)
	cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
	metrics["cv_roc_auc_mean"] = float(cv_scores.mean())
	metrics["cv_roc_auc_std"] = float(cv_scores.std())

	artifacts = EvaluationArtifacts(
		y_test=y_test,
		y_pred=y_pred,
		y_proba=y_proba,
		X_test=X_test,
		feature_names=feature_names,
	)

	return pipeline, metrics, artifacts


def pretty_print_metrics(metrics: dict) -> None:
	"""Render evaluation results in a concise, human-friendly layout."""

	def fmt(value: object) -> str:
		if isinstance(value, (float, np.floating)):
			return f"{value:.4f}"
		if isinstance(value, (int, np.integer)):
			return f"{int(value)}"
		return str(value)

	classifier_label = metrics.get("classifier", "Tensor DMD classifier")
	print(f"=== {classifier_label.title()} Performance Snapshot ===")

	ordered_stats = [
		("Accuracy", "accuracy"),
		("Precision", "precision"),
		("Recall", "recall"),
		("F1 score", "f1"),
		("ROC AUC", "roc_auc"),
		("CV ROC AUC (mean)", "cv_roc_auc_mean"),
		("CV ROC AUC (std)", "cv_roc_auc_std"),
	]

	print("\nPerformance summary:")
	for label, key in ordered_stats:
		if key in metrics:
			print(f"  {label:<20}: {fmt(metrics[key])}")

	if "confusion_matrix" in metrics:
		cm = metrics["confusion_matrix"]
		print("\nConfusion matrix (rows=true, cols=pred):")
		print("              pred=0   pred=1")
		print(f"  true=0     {cm[0][0]:7d}   {cm[0][1]:7d}")
		print(f"  true=1     {cm[1][0]:7d}   {cm[1][1]:7d}")

	if "best_params" in metrics:
		print("\nBest hyperparameters:")
		for param, value in metrics["best_params"].items():
			print(f"  {param}: {value}")

	if metrics.get("top_features"):
		print("\nTop contributing features:")
		for entry in metrics["top_features"][:5]:
			direction = "positive" if entry["weight"] >= 0 else "negative"
			print(f"  {entry['feature']:<30}: {fmt(entry['score'])} ({direction})")

	if "pca" in metrics:
		pca_info = metrics["pca"]
		print("\nPCA summary:")
		if "n_components" in pca_info:
			print(f"  Components kept       : {pca_info['n_components']}")
		if "cumulative_explained_variance" in pca_info:
			print(
				f"  Cumulative variance   : {fmt(pca_info['cumulative_explained_variance'])}"
			)
		ratios = pca_info.get("explained_variance_ratio", [])
		if ratios:
			sample = ", ".join(f"{r:.4f}" for r in ratios[:5])
			suffix = " ..." if len(ratios) > 5 else ""
			print(f"  Leading variance ratios: {sample}{suffix}")

	if "classification_report" in metrics:
		print("\nClassification report:")
		print(metrics["classification_report"].rstrip())


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def parse_args() -> DMDTensorConfig:
	parser = argparse.ArgumentParser(description="Tensor DMD textile failure classifier")
	parser.add_argument(
		"--data-path",
		type=Path,
		default=Path("textile_machine_data.csv"),
		help="Path to the textile telemetry CSV file.",
	)
	parser.add_argument(
		"--window-size",
		type=int,
		default=12,
		help="Number of sequential timesteps per tensor window.",
	)
	parser.add_argument(
		"--dmd-rank",
		type=int,
		default=8,
		help="Truncation rank for the DMD decomposition.",
	)
	parser.add_argument(
		"--top-modes",
		type=int,
		default=6,
		help="How many dominant modes to retain as explicit features.",
	)
	parser.add_argument(
		"--test-size",
		type=float,
		default=0.2,
		help="Hold-out fraction for evaluation.",
	)
	parser.add_argument(
		"--classifier",
		choices=["logreg", "gradient_boosting", "random_forest"],
		default="logreg",
		help="Classifier to train on top of DMD features.",
	)
	parser.add_argument(
		"--compare-classifiers",
		action="store_true",
		help="Run the experiment for every supported classifier and summarize the results.",
	)
	parser.add_argument(
		"--grid-search",
		action="store_true",
		help="Enable cross-validated hyperparameter tuning for the chosen classifier.",
	)
	parser.add_argument(
		"--no-pca",
		dest="use_pca",
		action="store_false",
		help="Disable the PCA dimensionality reduction stage.",
	)
	parser.set_defaults(use_pca=True)
	parser.add_argument(
		"--pca-components",
		type=int,
		default=None,
		help="Explicit number of principal components to retain (overrides variance).",
	)
	parser.add_argument(
		"--pca-variance",
		type=float,
		default=0.95,
		help="Cumulative explained variance ratio threshold for PCA (ignored if components specified).",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Directory where evaluation artefacts (confusion matrix, model) will be stored.",
	)
	parser.add_argument(
		"--export-features",
		type=Path,
		default=None,
		help="Optional path to save the engineered DMD features as CSV.",
	)
	parser.add_argument(
		"--export-model",
		type=Path,
		default=None,
		help="Optional path to persist the trained model (joblib).",
	)

	args = parser.parse_args()

	return DMDTensorConfig(
		data_path=args.data_path,
		window_size=args.window_size,
		dmd_rank=args.dmd_rank,
		top_modes=args.top_modes,
		test_size=args.test_size,
		classifier=args.classifier,
		grid_search=args.grid_search,
		compare_classifiers=args.compare_classifiers,
		use_pca=args.use_pca,
		pca_components=args.pca_components,
		pca_variance_ratio=args.pca_variance,
		output_dir=args.output_dir,
		export_features_path=args.export_features,
		export_model_path=args.export_model,
	)


def main(cfg: Optional[DMDTensorConfig] = None) -> None:
	if cfg is None:
		cfg = parse_args()

	df = load_dataset(cfg.data_path)
	feature_matrix, labels, base_feature_names = build_feature_matrix(df)

	pca_projection: Optional[PCAProjection] = None
	if cfg.use_pca:
		pca = PCAFromScratch(
			n_components=cfg.pca_components,
			variance_ratio=cfg.pca_variance_ratio,
		)
		feature_matrix = pca.fit_transform(feature_matrix)
		pca_projection = pca.projection_
		base_feature_names = [f"pca_component_{i}" for i in range(feature_matrix.shape[1])]

	extractor = TensorDMDFeatureExtractor(rank=cfg.dmd_rank, top_modes=cfg.top_modes)
	X, y, dmd_feature_names = window_tensor(
		feature_matrix,
		labels,
		window_size=cfg.window_size,
		extractor=extractor,
	)

	if cfg.export_features_path:
		features_df = pd.DataFrame(X, columns=dmd_feature_names)
		features_df["Failure"] = y
		cfg.export_features_path.parent.mkdir(parents=True, exist_ok=True)
		features_df.to_csv(cfg.export_features_path, index=False)
		print(f"Saved engineered features to {cfg.export_features_path}")

	artifact_dir = cfg.output_dir
	if artifact_dir:
		artifact_dir.mkdir(parents=True, exist_ok=True)

	classifiers_to_run = [cfg.classifier]
	if cfg.compare_classifiers:
		classifiers_to_run = ["logreg", "gradient_boosting", "random_forest"]

	comparison_records: List[dict] = []

	for classifier_name in classifiers_to_run:
		local_cfg = replace(cfg, classifier=classifier_name, compare_classifiers=False)
		model, metrics, artifacts = train_and_evaluate(X, y, local_cfg, dmd_feature_names)
		if pca_projection is not None:
			metrics["pca"] = {
				"n_components": int(pca_projection.components.shape[1]),
				"explained_variance_ratio": pca_projection.explained_variance_ratio.tolist(),
				"cumulative_explained_variance": float(pca_projection.explained_variance_ratio.cumsum()[-1]),
			}
		metrics["classifier"] = classifier_name
		top_features = compute_feature_influence(model, dmd_feature_names)
		if top_features:
			metrics["top_features"] = top_features[:10]

		pretty_print_metrics(metrics)
		comparison_records.append({"classifier": classifier_name, "metrics": metrics})

		if artifact_dir:
			run_dir = artifact_dir / classifier_name if cfg.compare_classifiers else artifact_dir
			run_dir.mkdir(parents=True, exist_ok=True)

			cm_raw = metrics.get("confusion_matrix")
			if cm_raw is not None:
				cm = np.asarray(cm_raw)
				if cm.ndim == 2:
					cm_path = run_dir / "confusion_matrix.png"
					fig, ax = plt.subplots(figsize=(4.5, 4.0))
					disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No failure", "Failure"])
					disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
					ax.set_title("Confusion Matrix")
					plt.tight_layout()
					fig.savefig(cm_path, dpi=300)
					plt.close(fig)
					print(f"Saved confusion matrix to {cm_path}")

			roc_path = run_dir / "roc_curve.png"
			save_roc_curve_plot(artifacts, roc_path, classifier_name)
			print(f"Saved ROC curve to {roc_path}")

			decision_path = run_dir / "decision_boundary.png"
			save_decision_boundary_plot(artifacts, model, decision_path, classifier_name)
			if decision_path.exists():
				print(f"Saved decision landscape to {decision_path}")

			if metrics.get("top_features"):
				feature_path = run_dir / "feature_influence.csv"
				save_feature_influence_table(metrics["top_features"], feature_path)
				print(f"Saved feature influence table to {feature_path}")

			metrics_path = run_dir / "metrics.json"
			save_metrics_json(metrics, metrics_path)
			print(f"Saved metrics summary to {metrics_path}")

			model_path = run_dir / "trained_model.joblib"
			joblib.dump({"model": model, "config": asdict(local_cfg), "feature_names": dmd_feature_names}, model_path)
			print(f"Saved trained model to {model_path}")

		if cfg.export_model_path:
			target_path = cfg.export_model_path
			if cfg.compare_classifiers:
				target_path = target_path.with_name(f"{target_path.stem}_{classifier_name}{target_path.suffix}")
			target_path.parent.mkdir(parents=True, exist_ok=True)
			joblib.dump({"model": model, "config": asdict(local_cfg), "feature_names": dmd_feature_names}, target_path)
			print(f"Persisted trained model to {target_path}")

	if cfg.compare_classifiers:
		comparison_table = []
		for record in comparison_records:
			metrics = record["metrics"]
			comparison_table.append({
				"classifier": record["classifier"],
				"roc_auc": metrics.get("roc_auc"),
				"cv_mean": metrics.get("cv_roc_auc_mean"),
				"cv_std": metrics.get("cv_roc_auc_std"),
				"accuracy": metrics.get("accuracy"),
				"f1": metrics.get("f1"),
			})
		best_model = max(comparison_records, key=lambda item: item["metrics"].get("roc_auc", 0.0))
		print("\n=== Classifier comparison (ROC AUC focus) ===")
		for row in comparison_table:
			print(
				f"  {row['classifier']:<20} | ROC-AUC={row['roc_auc']:.4f} | CV={row['cv_mean']:.4f}±{row['cv_std']:.4f} | Acc={row['accuracy']:.4f} | F1={row['f1']:.4f}"
			)
		print(f"\nBest performer (by ROC AUC): {best_model['classifier']}")
		if artifact_dir:
			comparison_path = artifact_dir / "classifier_comparison.json"
			with comparison_path.open("w", encoding="utf-8") as fh:
				json.dump(comparison_table, fh, indent=2)
			print(f"Stored classifier comparison to {comparison_path}")


if __name__ == "__main__":
	main()
