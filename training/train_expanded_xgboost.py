"""Train isolated expanded-data XGBoost experiments.

This experiment does not touch production models or artifacts. Original and
Kaggle targets remain separate because their source price semantics differ.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             median_absolute_error, r2_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from .data_cleaning import clean_data
    from .data_source_policy import reject_restricted_dataset
except ImportError:
    from data_cleaning import clean_data
    from data_source_policy import reject_restricted_dataset

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_PATH = ROOT / "data" / "external" / "UsedCars_Combined.csv"
ORIGINAL_PATH = ROOT / "data" / "original" / "Used Car Dataset.csv"
ARTIFACTS = ROOT / "artifacts"

BASE_NUMERIC = ["manufacturing_year", "kms_driven"]
BASE_CATEGORICAL = ["brand", "model_name", "full_model_name", "fuel_type"]
PARAMS = [
    {"n_estimators": 250, "max_depth": 3, "learning_rate": 0.03,
     "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.9,
     "reg_alpha": 0.1, "reg_lambda": 8},
    {"n_estimators": 350, "max_depth": 2, "learning_rate": 0.03,
     "min_child_weight": 3, "subsample": 0.9, "colsample_bytree": 0.9,
     "reg_alpha": 0.1, "reg_lambda": 10},
    {"n_estimators": 250, "max_depth": 3, "learning_rate": 0.02,
     "min_child_weight": 8, "subsample": 0.85, "colsample_bytree": 0.8,
     "reg_alpha": 0.2, "reg_lambda": 12},
]


def _norm(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    relative = np.abs(predicted - actual) / np.maximum(np.abs(actual), 1e-8)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "median_absolute_error": float(median_absolute_error(actual, predicted)),
        "within_10_percent": float(np.mean(relative <= 0.10)),
        "within_20_percent": float(np.mean(relative <= 0.20)),
        "within_30_percent": float(np.mean(relative <= 0.30)),
    }


def _load_kaggle() -> pd.DataFrame:
    raw = pd.read_csv(KAGGLE_PATH)
    result = pd.DataFrame({
        "source_row_id": raw["Id"].astype(str),
        "brand": raw["brand"].astype("string"),
        "model_name": raw["model_name"].astype("string"),
        "full_model_name": raw["full_model_name"].astype("string"),
        "manufacturing_year": pd.to_numeric(raw["year"], errors="coerce"),
        "fuel_type": raw["fuel_type"].astype("string"),
        "kms_driven": pd.to_numeric(raw["distance_travelled(kms)"], errors="coerce"),
        "city": raw["city"].astype("string"),
        "price_lakhs": pd.to_numeric(raw["price"], errors="coerce") / 100000.0,
    })
    result["model_family_key"] = result["model_name"].map(_norm)
    result["signature_key"] = (
        result["brand"].map(_norm) + "|" + result["full_model_name"].map(_norm)
        + "|" + result["manufacturing_year"].astype("string")
        + "|" + result["price_lakhs"].round(8).astype("string")
        + "|" + result["kms_driven"].astype("string")
        + "|" + result["fuel_type"].map(_norm) + "|" + result["city"].map(_norm)
    )
    valid = (
        result["manufacturing_year"].between(1980, 2023)
        & result["kms_driven"].gt(0)
        & result["price_lakhs"].gt(0)
        & result["model_family_key"].ne("")
    )
    return result.loc[valid].reset_index(drop=True)


def _load_original() -> pd.DataFrame:
    original, _ = clean_data(ORIGINAL_PATH)
    return pd.DataFrame({
        "source_row_id": original.index.astype(str),
        "brand": original["car_name"].map(
            lambda value: str(value).split()[1]
            if len(str(value).split()) > 1 and re.match(r"^\d{4}\s+", str(value))
            else str(value).split()[0]
        ),
        "model_name": original["car_name"].map(
            lambda value: " ".join(str(value).split()[1:3])
            if re.match(r"^\d{4}\s+", str(value))
            else " ".join(str(value).split()[:2])
        ),
        "full_model_name": original["car_name"],
        "manufacturing_year": original["manufacturing_year"],
        "fuel_type": original["fuel_type"],
        "kms_driven": original["kms_driven"],
        "price_lakhs": original["price"],
    }).reset_index(drop=True)


def _split(
    frame: pd.DataFrame, protocol: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_column = "signature_key" if protocol == "known_model_families" else "model_family_key"
    groups = frame[group_column].to_numpy()
    if protocol == "known_model_families":
        family_signature_counts = frame.groupby("model_family_key")[
            "signature_key"
        ].nunique()
        eligible_families = set(family_signature_counts[family_signature_counts > 1].index)
        eligible = frame[frame["model_family_key"].isin(eligible_families)]
    else:
        eligible = frame
    for seed in range(42, 142):
        first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        eligible_train_pos, eligible_test_pos = next(
            first.split(eligible, groups=eligible[group_column].to_numpy())
        )
        test = eligible.iloc[eligible_test_pos].copy()
        remaining = frame.drop(index=test.index)
        second = GroupShuffleSplit(
            n_splits=1, test_size=0.1765, random_state=seed + 1000
        )
        if protocol == "known_model_families":
            remaining_family_counts = remaining.groupby("model_family_key")[
                "signature_key"
            ].nunique()
            validation_families = set(
                remaining_family_counts[remaining_family_counts > 1].index
            )
            validation_pool = remaining[
                remaining["model_family_key"].isin(validation_families)
            ]
        else:
            validation_pool = remaining
        if validation_pool.empty:
            continue
        train_pos, validation_pos = next(
            second.split(validation_pool, groups=validation_pool[group_column].to_numpy())
        )
        validation = validation_pool.iloc[validation_pos].copy()
        train = remaining.drop(index=validation.index).copy()
        if protocol == "known_model_families":
            train_families = set(train["model_family_key"])
            if not set(validation["model_family_key"]).issubset(train_families):
                continue
            if not set(test["model_family_key"]).issubset(train_families):
                continue
        else:
            train_families = set(train["model_family_key"])
            if set(validation["model_family_key"]) & train_families:
                continue
            if set(test["model_family_key"]) & train_families:
                continue
        return train, validation, test
    raise RuntimeError(f"Could not construct a valid {protocol} split.")


def _preprocessor(include_city: bool) -> ColumnTransformer:
    numerical = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = BASE_CATEGORICAL + (["city"] if include_city else [])
    encoded = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("numeric", numerical, BASE_NUMERIC),
        ("categorical", encoded, categorical),
    ])


def _fit(
    params: dict[str, Any], target_mode: str, x_train: Any, y_train: np.ndarray,
    x_validation: Any, y_validation: np.ndarray,
) -> tuple[xgb.XGBRegressor, float]:
    model = xgb.XGBRegressor(
        objective="reg:squarederror", tree_method="hist", n_jobs=2,
        random_state=42, early_stopping_rounds=30, **params,
    )
    train_y = np.log1p(y_train) if target_mode == "log1p" else y_train
    valid_y = np.log1p(y_validation) if target_mode == "log1p" else y_validation
    model.fit(x_train, train_y, eval_set=[(x_validation, valid_y)], verbose=False)
    prediction = model.predict(x_validation)
    if target_mode == "log1p":
        prediction = np.expm1(prediction)
    return model, float(mean_absolute_error(y_validation, prediction))


def _predict(model: xgb.XGBRegressor, matrix: Any, target_mode: str) -> np.ndarray:
    prediction = model.predict(matrix)
    return np.maximum(np.expm1(prediction), 0) if target_mode == "log1p" else np.maximum(prediction, 0)


def _split_summary(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> dict[str, Any]:
    def summary(frame: pd.DataFrame) -> dict[str, int]:
        return {
            "rows": int(len(frame)),
            "unique_manufacturers": int(frame["brand"].nunique()),
            "unique_model_families": int(frame["model_family_key"].nunique()),
            "unique_signatures": int(frame["signature_key"].nunique()),
        }
    return {"train": summary(train), "validation": summary(validation), "test": summary(test)}


def _manufacturer_metrics(
    test: pd.DataFrame, prediction: np.ndarray, minimum_count: int = 10
) -> dict[str, Any]:
    frame = test[["brand"]].copy()
    frame["actual"] = test["price_lakhs"].to_numpy()
    frame["prediction"] = prediction
    output: dict[str, Any] = {}
    for brand, group in frame.groupby("brand"):
        if len(group) < minimum_count:
            continue
        output[str(brand)] = {
            "test_rows": int(len(group)),
            **_metrics(group["actual"].to_numpy(), group["prediction"].to_numpy()),
        }
    return output


def _run_protocol(
    data: pd.DataFrame, protocol: str, include_city: bool
) -> dict[str, Any]:
    train, validation, test = _split(data, protocol)
    features = BASE_NUMERIC + BASE_CATEGORICAL + (["city"] if include_city else [])
    processor = _preprocessor(include_city)
    x_train = processor.fit_transform(train[features])
    x_validation = processor.transform(validation[features])
    x_test = processor.transform(test[features])
    y_train = train["price_lakhs"].to_numpy(float)
    y_validation = validation["price_lakhs"].to_numpy(float)
    y_test = test["price_lakhs"].to_numpy(float)

    best: tuple[float, xgb.XGBRegressor, str, dict[str, Any]] | None = None
    tuning: list[dict[str, Any]] = []
    for target_mode in ("raw", "log1p"):
        for index, params in enumerate(PARAMS, 1):
            candidate, score = _fit(
                params, target_mode, x_train, y_train, x_validation, y_validation
            )
            tuning.append({
                "target_mode": target_mode, "candidate": index,
                "validation_mae": score, "params": params,
            })
            if best is None or score < best[0]:
                best = (score, candidate, target_mode, params)
    if best is None:
        raise RuntimeError("No expanded-data candidate was trained.")

    validation_mae, selected, target_mode, params = best
    test_prediction = _predict(selected, x_test, target_mode)
    return {
        "protocol": protocol,
        "include_city": include_city,
        "target": "price_lakhs",
        "test_metrics": _metrics(y_test, test_prediction),
        "validation_mae_for_selection": validation_mae,
        "selected_target_mode": target_mode,
        "selected_params": params,
        "tuning": tuning,
        "split_summary": _split_summary(train, validation, test),
        "manufacturer_metrics_min_10_test_rows": _manufacturer_metrics(test, test_prediction),
        "test_model_families_unseen_in_train": int(
            len(set(test["model_family_key"]) - set(train["model_family_key"]))
        ),
        "test_row_count": int(len(test)),
    }


def main() -> None:
    reject_restricted_dataset(ORIGINAL_PATH)
    ARTIFACTS.mkdir(exist_ok=True)
    kaggle = _load_kaggle()
    original = _load_original()
    results: list[dict[str, Any]] = []
    for protocol in ("known_model_families", "unseen_model_families"):
        for include_city in (False, True):
            results.append(_run_protocol(kaggle, protocol, include_city))

    kaggle_metrics = {
        "source": "UsedCars_Combined.csv",
        "price_definition": "Kaggle raw price converted from INR rupees to INR lakhs; listing/asking semantics unconfirmed.",
        "rows": int(len(kaggle)),
        "protocols": results,
    }
    original_metrics = {
        "source": "Used Car Dataset.csv",
        "price_definition": "Original price(in lakhs), preserved separately; recorded-price semantics unconfirmed.",
        "rows_cleaned": int(len(original)),
        "comparison_note": "Original target was not pooled with Kaggle target. Existing production XGBoost metrics are reported separately for reference.",
        "existing_production_xgboost_metrics": json.loads(
            (ARTIFACTS / "xgboost_metrics.json").read_text(encoding="utf-8")
        ),
    }
    comparison = {
        "experiment": "expanded Kaggle-only XGBoost safeguards",
        "dataset_c_excluded": True,
        "production_artifacts_modified": False,
        "test_touched_during_tuning": False,
        "target_separation": {
            "original_target": "price(in lakhs)",
            "kaggle_target": "price / 100000",
            "pooled_training": False,
        },
        "kaggle_results": kaggle_metrics,
        "original_reference": original_metrics,
    }
    (ARTIFACTS / "xgboost_expanded_kaggle_metrics.json").write_text(
        json.dumps(kaggle_metrics, indent=2), encoding="utf-8"
    )
    (ARTIFACTS / "xgboost_original_common_metrics.json").write_text(
        json.dumps(original_metrics, indent=2), encoding="utf-8"
    )
    (ARTIFACTS / "expanded_model_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "kaggle_rows": len(kaggle), "original_cleaned_rows": len(original),
        "protocol_runs": len(results),
        "artifacts": [
            "expanded_model_comparison.json",
            "xgboost_expanded_kaggle_metrics.json",
            "xgboost_original_common_metrics.json",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
