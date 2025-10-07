"""Tensor-based Dynamic Mode Decomposition (DMD) classifier for textile failure prediction.

This script ingests the provided textile machine telemetry, constructs
tensor snapshots over sliding temporal windows, extracts DMD-driven
features, and trains a binary classifier to distinguish failure versus
non-failure regimes.

Usage (from the project root)::

	python DMD_Tensor.py --data-path textile_machine_data.csv

Optional arguments expose the window length, DMD rank, target classifier,
and export paths for engineered features and the fitted model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	classification_report,
	confusion_matrix,
	precision_recall_fscore_support,
	roc_auc_score,
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
	use_pca: bool = True
	pca_components: Optional[int] = None
	pca_variance_ratio: float = 0.95
	export_features_path: Optional[Path] = None
	export_model_path: Optional[Path] = None


@dataclass
class PCAProjection:
	"""Container for the ingredients of a PCA projection learned from scratch."""

	mean_vector: NDArray[np.float64]
	components: NDArray[np.float64]
	explained_variance: NDArray[np.float64]
	explained_variance_ratio: NDArray[np.float64]


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
		r"""Return a 1D feature vector for the provided window following DMD.

		The algorithm unfolds as:

		1. Arrange consecutive measurements as two time-shifted snapshot matrices
		   $X$ and $Y$ such that $Y \approx A X$, where $A$ captures the linear
		   evolution across one time-step.
		2. Compress $X$ via a rank-$r$ Singular Value Decomposition (SVD) to obtain
		   an orthonormal basis that explains the dominant energy in the data.
		3. Build the reduced operator $\tilde{A} = U_r^\top Y V_r \Sigma_r^{-1}$
		   which is the similarity transform of $A$ expressed in the low-dimensional
		   coordinates.
		4. Solve the eigen decomposition of $\tilde{A}$ to recover eigenvalues
		   (temporal frequencies / growth rates) and eigenvectors. Lift the modes
		   back to the original space to obtain DMD modes.
		5. Derive descriptive statistics (eigenvalue parts, modal amplitudes,
		   reconstruction error, and basic temporal summaries) that serve as the
		   engineered features for classification.
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
		# Assemble the reduced-order linear operator \tilde{A} that advances the
		# system dynamics in the subspace spanned by U_r.
		S_inv = np.diag(1.0 / S_r)
		A_tilde = U_r.T @ Y @ V_r @ S_inv
		eigvals, eigvecs = np.linalg.eig(A_tilde)

		# STEP 4 -------------------------------------------------------------
		# Sort the eigenpairs by magnitude |λ| to prioritise dynamically dominant
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


def train_and_evaluate(
	X: NDArray[np.float64],
	y: NDArray[np.int64],
	cfg: DMDTensorConfig,
) -> Tuple[Pipeline, dict]:
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

	return pipeline, metrics


def pretty_print_metrics(metrics: dict) -> None:
	"""Render evaluation results in a concise, human-friendly layout."""

	def fmt(value: object) -> str:
		if isinstance(value, (float, np.floating)):
			return f"{value:.4f}"
		if isinstance(value, (int, np.integer)):
			return f"{int(value)}"
		return str(value)

	print("=== Tensor DMD Textile Failure Classification ===")

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
		use_pca=args.use_pca,
		pca_components=args.pca_components,
		pca_variance_ratio=args.pca_variance,
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

	model, metrics = train_and_evaluate(X, y, cfg)
	if pca_projection is not None:
		metrics["pca"] = {
			"n_components": int(pca_projection.components.shape[1]),
			"explained_variance_ratio": pca_projection.explained_variance_ratio.tolist(),
			"cumulative_explained_variance": float(pca_projection.explained_variance_ratio.cumsum()[-1]),
		}

	pretty_print_metrics(metrics)

	if cfg.export_features_path:
		features_df = pd.DataFrame(X, columns=dmd_feature_names)
		features_df["Failure"] = y
		features_df.to_csv(cfg.export_features_path, index=False)
		print(f"Saved engineered features to {cfg.export_features_path}")

	if cfg.export_model_path:
		try:
			import joblib
		except ModuleNotFoundError as exc:  # pragma: no cover
			raise ModuleNotFoundError(
				"joblib is required to export the trained model. Install it via 'pip install joblib'."
			) from exc

		model_dir = cfg.export_model_path.parent
		model_dir.mkdir(parents=True, exist_ok=True)
		joblib.dump({"model": model, "config": asdict(cfg), "feature_names": dmd_feature_names}, cfg.export_model_path)
		print(f"Persisted trained model to {cfg.export_model_path}")


if __name__ == "__main__":
	main()
