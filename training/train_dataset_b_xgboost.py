"""Train an isolated XGBoost model using only the approved Kaggle Dataset B.

The script writes only dataset_b_* artifacts. It uses grouped partitions,
keeps calibration separate from fitting, and leaves the held-out test split
untouched until final evaluation.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "external" / "UsedCars_Combined.csv"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "dataset_b_xgboost_model.json"
PREPROCESSOR_PATH = ARTIFACTS / "dataset_b_preprocessor.joblib"
METADATA_PATH = ARTIFACTS / "dataset_b_metadata.json"
METRICS_PATH = ARTIFACTS / "dataset_b_metrics.json"
CALIBRATION_PATH = ARTIFACTS / "dataset_b_calibration.json"
SCHEMA_PATH = ARTIFACTS / "dataset_b_feature_schema.json"

SEED = 2026
TARGET = "price_lakhs"
FEATURES = [
    "brand",
    "model_family",
    "manufacturing_year",
    "kms_driven",
    "fuel_type",
    "city",
]
CATEGORICAL = ["brand", "model_family", "fuel_type", "city"]
NUMERIC = ["manufacturing_year", "kms_driven"]


def _read_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(DATA_PATH)
    required = {
        "Id",
        "year",
        "brand",
        "full_model_name",
        "model_name",
        "price",
        "distance_travelled(kms)",
        "fuel_type",
        "city",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"Dataset B is missing required columns: {missing}")

    frame = raw.rename(
        columns={
            "year": "manufacturing_year",
            "distance_travelled(kms)": "kms_driven",
        }
    ).copy()
    frame["price_lakhs"] = pd.to_numeric(frame["price"], errors="coerce") / 100000.0
    frame["model_family"] = (
        frame["model_name"].astype("string").str.strip().str.lower()
    )
    for column in CATEGORICAL:
        frame[column] = frame[column].astype("string").fillna("__missing__").str.strip()
    frame["manufacturing_year"] = pd.to_numeric(
        frame["manufacturing_year"], errors="coerce"
    )
    frame["kms_driven"] = pd.to_numeric(frame["kms_driven"], errors="coerce")

    invalid = (
        frame[TARGET].isna()
        | ~np.isfinite(frame[TARGET])
        | (frame[TARGET] <= 0)
        | frame["manufacturing_year"].isna()
        | frame["kms_driven"].isna()
        | (frame["manufacturing_year"] < 1980)
        | (frame["manufacturing_year"] > 2025)
        | (frame["kms_driven"] < 0)
    )
    frame = frame.loc[~invalid].reset_index(drop=True)
    frame["signature"] = (
        frame[
            [
                "brand",
                "model_family",
                "full_model_name",
                "manufacturing_year",
                "kms_driven",
                "fuel_type",
                "city",
            ]
        ]
        .astype(str)
        .agg("|".join, axis=1)
    )
    quality = {
        "raw_rows": int(len(raw)),
        "raw_columns": list(raw.columns),
        "raw_missing_values": {
            str(k): int(v) for k, v in raw.isna().sum().items()
        },
        "raw_exact_duplicates": int(raw.duplicated().sum()),
        "rows_removed_invalid": int(invalid.sum()),
        "usable_rows": int(len(frame)),
        "target_raw_min_rupees": float(raw["price"].min()),
        "target_raw_max_rupees": float(raw["price"].max()),
        "target_lakhs_median": float(frame[TARGET].median()),
        "target_lakhs_skew": float(frame[TARGET].skew()),
        "repeated_signature_groups": int(
            (frame["signature"].value_counts() > 1).sum()
        ),
        "repeated_signature_rows": int(
            frame["signature"].value_counts().loc[
                lambda values: values > 1
            ].sum()
        ),
    }
    return frame, quality


def _split_groups(
    frame: pd.DataFrame,
    group_column: str,
    seed: int,
) -> dict[str, np.ndarray]:
    indices = np.arange(len(frame))

    def take(
        available: np.ndarray, fraction: float, random_state: int
    ) -> tuple[np.ndarray, np.ndarray]:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=fraction, random_state=random_state
        )
        left, right = next(
            splitter.split(
                frame.iloc[available],
                groups=frame.iloc[available][group_column],
            )
        )
        return available[left], available[right]

    train_val_cal, test = take(indices, 0.15, seed)
    train_val, calibration = take(train_val_cal, 0.117647, seed + 1)
    train, validation = take(train_val, 0.176471, seed + 2)
    return {
        "train": np.sort(train),
        "validation": np.sort(validation),
        "calibration": np.sort(calibration),
        "test": np.sort(test),
    }


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC),
            ("categorical", categorical, CATEGORICAL),
        ]
    )


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0.0)
    relative = np.abs(predicted - actual) / np.maximum(np.abs(actual), 1e-8)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "median_absolute_error": float(
            median_absolute_error(actual, predicted)
        ),
        "within_10_percent": float(np.mean(relative <= 0.10)),
        "within_20_percent": float(np.mean(relative <= 0.20)),
        "within_30_percent": float(np.mean(relative <= 0.30)),
        "rows": int(len(actual)),
    }


def _baseline(actual: np.ndarray, training_target: np.ndarray) -> dict[str, float]:
    median = float(np.median(training_target))
    result = _metrics(actual, np.full(len(actual), median))
    result["prediction_lakhs"] = median
    return result


def _fit_model(
    frame: pd.DataFrame,
    splits: dict[str, np.ndarray],
    params: dict[str, Any],
) -> tuple[Pipeline, float, dict[str, float]]:
    preprocessor = _preprocessor()
    train = frame.iloc[splits["train"]]
    validation = frame.iloc[splits["validation"]]
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        n_jobs=2,
        random_state=SEED,
        **params,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(
        train[FEATURES],
        np.log1p(train[TARGET].to_numpy()),
    )
    validation_prediction = np.expm1(pipeline.predict(validation[FEATURES]))
    return pipeline, float(mean_absolute_error(validation[TARGET], validation_prediction)), _metrics(
        validation[TARGET].to_numpy(), validation_prediction
    )


def _evaluate_protocol(
    frame: pd.DataFrame,
    protocol: str,
    group_column: str,
    seed: int,
) -> tuple[dict[str, Any], Pipeline | None]:
    splits = _split_groups(frame, group_column, seed)
    candidates = [
        {
            "n_estimators": 250,
            "max_depth": 2,
            "learning_rate": 0.03,
            "min_child_weight": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 10.0,
        },
        {
            "n_estimators": 350,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.9,
            "reg_alpha": 0.1,
            "reg_lambda": 10.0,
        },
    ]
    trained: list[tuple[float, Pipeline, dict[str, Any], dict[str, float]]] = []
    for params in candidates:
        pipeline, validation_mae, validation_metrics = _fit_model(
            frame, splits, params
        )
        trained.append((validation_mae, pipeline, params, validation_metrics))
    _, pipeline, selected_params, validation_metrics = min(
        trained, key=lambda item: item[0]
    )

    calibration = frame.iloc[splits["calibration"]]
    calibration_prediction = np.expm1(pipeline.predict(calibration[FEATURES]))
    calibration_residuals = np.abs(
        calibration[TARGET].to_numpy() - calibration_prediction
    )
    rank = int(np.ceil((len(calibration_residuals) + 1) * 0.90))
    quantile = float(np.sort(calibration_residuals)[min(rank, len(calibration_residuals)) - 1])
    test = frame.iloc[splits["test"]]
    test_prediction = np.expm1(pipeline.predict(test[FEATURES]))
    test_metrics = _metrics(test[TARGET].to_numpy(), test_prediction)
    lower = np.maximum(test_prediction - quantile, 0.0)
    upper = test_prediction + quantile
    coverage = float(
        np.mean(
            (test[TARGET].to_numpy() >= lower)
            & (test[TARGET].to_numpy() <= upper)
        )
    )
    split_audit = {
        name: {
            "rows": int(len(indices)),
            "unique_manufacturers": int(frame.iloc[indices]["brand"].nunique()),
            "unique_model_families": int(
                frame.iloc[indices]["model_family"].nunique()
            ),
            "unique_signatures": int(frame.iloc[indices]["signature"].nunique()),
        }
        for name, indices in splits.items()
    }
    result = {
        "protocol": protocol,
        "group_column": group_column,
        "seed": seed,
        "target": "log1p(price_lakhs), reported after expm1",
        "features": FEATURES,
        "selected_hyperparameters": selected_params,
        "validation_metrics": validation_metrics,
        "baseline_test_metrics": _baseline(
            test[TARGET].to_numpy(), frame.iloc[splits["train"]][TARGET].to_numpy()
        ),
        "test_metrics": test_metrics,
        "calibration": {
            "nominal_coverage": 0.90,
            "rows": int(len(calibration)),
            "quantile_lakhs": quantile,
            "empirical_test_coverage": coverage,
            "average_interval_width_lakhs": float(np.mean(upper - lower)),
        },
        "split_audit": split_audit,
    }
    return result, pipeline


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Approved Dataset B was not found: {DATA_PATH}")
    frame, quality = _read_dataset()
    known, known_pipeline = _evaluate_protocol(
        frame, "known_model_families", "signature", SEED
    )
    unseen, _ = _evaluate_protocol(
        frame, "unseen_model_families", "model_family", SEED
    )
    if known_pipeline is None:
        raise RuntimeError("Known-family training did not produce a model.")

    ARTIFACTS.mkdir(exist_ok=True)
    known_pipeline.named_steps["model"].save_model(MODEL_PATH)
    joblib.dump(known_pipeline.named_steps["preprocessor"], PREPROCESSOR_PATH)
    METRICS_PATH.write_text(
        json.dumps(
            {
                "dataset_quality": quality,
                "known_model_families": known,
                "unseen_model_families": unseen,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    CALIBRATION_PATH.write_text(
        json.dumps(
            {
                "protocol": known["protocol"],
                "nominal_coverage": known["calibration"]["nominal_coverage"],
                "calibration_rows": known["calibration"]["rows"],
                "quantile_lakhs": known["calibration"]["quantile_lakhs"],
                "empirical_test_coverage": known["calibration"][
                    "empirical_test_coverage"
                ],
                "average_interval_width_lakhs": known["calibration"][
                    "average_interval_width_lakhs"
                ],
                "limitation": (
                    "This interval is calibrated for the known-family grouped "
                    "protocol and is not guaranteed for unseen model families."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    SCHEMA_PATH.write_text(
        json.dumps(
            {
                "features": FEATURES,
                "categorical_features": CATEGORICAL,
                "numeric_features": NUMERIC,
                "target_raw_column": "price",
                "target_column": TARGET,
                "price_conversion": "price_lakhs = price / 100000",
                "excluded_columns": [
                    "Id",
                    "full_model_name",
                    "signature",
                    "price",
                ],
                "unknown_category_behavior": "OneHotEncoder(handle_unknown='ignore')",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    METADATA_PATH.write_text(
        json.dumps(
            {
                "dataset": "Kaggle Used Car Data by Sanjeet Singh Naik",
                "source": "data/external/UsedCars_Combined.csv",
                "license": "CC0 Public Domain",
                "dataset_fingerprint_sha256": _fingerprint(DATA_PATH),
                "random_seed": SEED,
                "package_versions": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scikit_learn": sklearn.__version__,
                    "xgboost": xgb.__version__,
                    "joblib": joblib.__version__,
                },
                "training_rows": quality["usable_rows"],
                "model_artifact": MODEL_PATH.name,
                "preprocessor_artifact": PREPROCESSOR_PATH.name,
                "metrics_artifact": METRICS_PATH.name,
                "calibration_artifact": CALIBRATION_PATH.name,
                "feature_schema_artifact": SCHEMA_PATH.name,
                "target_transformation": "log1p during fitting; expm1 for reported prices",
                "test_policy": "Test partitions were not used for model selection or calibration quantiles.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"known": known, "unseen": unseen}, indent=2))


if __name__ == "__main__":
    main()
