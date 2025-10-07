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
	export_features_path: Optional[Path] = None
	export_model_path: Optional[Path] = None


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
		"""Return a 1D feature vector for the provided window.

		Parameters
		----------
		window:
			Array of shape (window_size, n_features) describing sequential
			telemetry snapshots. The last row represents the most recent
			observation in the window.
		"""

		if window.ndim != 2:
			raise ValueError("window must be 2-dimensional")
		if window.shape[0] < 3:
			raise ValueError("window must contain at least 3 timesteps for DMD")

		# Separate snapshots for DMD (columns are successive system states)
		X = window[:-1].T  # shape: (n_features, window_size - 1)
		Y = window[1:].T   # shape: (n_features, window_size - 1)

		try:
			U, singular_vals, Vh = np.linalg.svd(X, full_matrices=False)
		except np.linalg.LinAlgError:
			# In degenerate cases fall back to zeros while keeping output shape stable
			return np.zeros(self._feature_length(window.shape[1]), dtype=np.float64)

		r = int(min(self.rank, len(singular_vals), X.shape[0], X.shape[1]))
		if r == 0:
			return np.zeros(self._feature_length(window.shape[1]), dtype=np.float64)

		U_r = U[:, :r]
		S_r = singular_vals[:r]
		V_r = Vh.conj().T[:, :r]

		S_inv = np.diag(1.0 / S_r)
		A_tilde = U_r.T @ Y @ V_r @ S_inv
		eigvals, eigvecs = np.linalg.eig(A_tilde)

		# Sort modes by magnitude (energy dominance)
		order = np.argsort(-np.abs(eigvals))
		eigvals = eigvals[order]
		eigvecs = eigvecs[:, order]

		modes = Y @ V_r @ S_inv @ eigvecs

		# Rank-r reconstruction of X and residual energy
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

		# Temporal summary statistics (per-feature) to capture slower dynamics
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
		export_features_path=args.export_features,
		export_model_path=args.export_model,
	)


def main(cfg: Optional[DMDTensorConfig] = None) -> None:
	if cfg is None:
		cfg = parse_args()

	df = load_dataset(cfg.data_path)
	feature_matrix, labels, base_feature_names = build_feature_matrix(df)

	extractor = TensorDMDFeatureExtractor(rank=cfg.dmd_rank, top_modes=cfg.top_modes)
	X, y, dmd_feature_names = window_tensor(
		feature_matrix,
		labels,
		window_size=cfg.window_size,
		extractor=extractor,
	)

	model, metrics = train_and_evaluate(X, y, cfg)

	print("=== Tensor DMD Textile Failure Classification ===")
	print(json.dumps(metrics, indent=2))

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
