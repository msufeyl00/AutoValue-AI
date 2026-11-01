"""Five-seed, leakage-controlled evaluation for the expanded-data experiment.

This script writes only artifacts/expanded_repeated_evaluation and never
changes production models, preprocessors, calibration files, or application
code.
"""

from __future__ import annotations

import hashlib
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
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ARTIFACTS / "expanded_repeated_evaluation"
KAGGLE_PATH = ROOT / "data" / "external" / "UsedCars_Combined.csv"
ORIGINAL_PATH = ROOT / "data" / "original" / "Used Car Dataset.csv"
SEEDS = [42, 123, 2026, 3407, 7777]
PROTECTED = [
    "xgboost_car_price_model.json",
    "xgboost_preprocessor.joblib",
    "xgboost_calibration.json",
    "xgboost_calibrated_car_price_model.json",
    "xgboost_calibrated_preprocessor.joblib",
    "xgboost_v2_car_price_model.json",
    "xgboost_v2_preprocessor.joblib",
]
NUMERIC = ["manufacturing_year", "kms_driven"]
CATEGORICAL = ["brand", "model_name", "full_model_name", "fuel_type"]
CONFIG = {
    "n_estimators": 250,
    "max_depth": 3,
    "learning_rate": 0.03,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 8,
}


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
    frame = pd.DataFrame({
        "source_row_id": raw["Id"].astype(str),
        "brand": raw["brand"].astype("string"),
        "model_name": raw["model_name"].astype("string"),
        "full_model_name": raw["full_model_name"].astype("string"),
        "manufacturing_year": pd.to_numeric(raw["year"], errors="coerce"),
        "fuel_type": raw["fuel_type"].astype("string"),
        "kms_driven": pd.to_numeric(raw["distance_travelled(kms)"], errors="coerce"),
        "city": raw["city"].astype("string"),
        "price_lakhs": pd.to_numeric(raw["price"], errors="coerce") / 100000,
    })
    frame["family_key"] = frame["model_name"].map(_norm)
    frame["signature_key"] = (
        frame["brand"].map(_norm) + "|" + frame["full_model_name"].map(_norm)
        + "|" + frame["manufacturing_year"].astype("string")
        + "|" + frame["price_lakhs"].round(8).astype("string")
        + "|" + frame["kms_driven"].astype("string")
        + "|" + frame["fuel_type"].map(_norm) + "|" + frame["city"].map(_norm)
    )
    valid = (
        frame["manufacturing_year"].between(1980, 2023)
        & frame["kms_driven"].gt(0)
        & frame["price_lakhs"].gt(0)
        & frame["family_key"].ne("")
    )
    return frame.loc[valid].reset_index(drop=True)


def _load_original() -> pd.DataFrame:
    cleaned, _ = clean_data(ORIGINAL_PATH)
    frame = pd.DataFrame({
        "source_row_id": cleaned.index.astype(str),
        "brand": cleaned["car_name"].map(
            lambda value: str(value).split()[1]
            if re.match(r"^\d{4}\s+", str(value))
            else str(value).split()[0]
        ),
        "model_name": cleaned["car_name"].map(
            lambda value: " ".join(str(value).split()[1:3])
            if re.match(r"^\d{4}\s+", str(value))
            else " ".join(str(value).split()[:2])
        ),
        "full_model_name": cleaned["car_name"].astype("string"),
        "manufacturing_year": cleaned["manufacturing_year"],
        "fuel_type": cleaned["fuel_type"].astype("string"),
        "kms_driven": cleaned["kms_driven"],
        "price_lakhs": cleaned["price"],
        "city": pd.Series(["Unknown"] * len(cleaned)),
    }).reset_index(drop=True)
    frame["family_key"] = frame["model_name"].map(_norm)
    frame["signature_key"] = (
        frame["brand"].map(_norm) + "|" + frame["full_model_name"].map(_norm)
        + "|" + frame["manufacturing_year"].astype("string")
        + "|" + frame["price_lakhs"].round(8).astype("string")
        + "|" + frame["kms_driven"].astype("string")
        + "|" + frame["fuel_type"].map(_norm)
    )
    return frame


def _split(frame: pd.DataFrame, protocol: str, seed: int) -> dict[str, pd.DataFrame]:
    group_column = "signature_key" if protocol == "known_families" else "family_key"
    if protocol == "known_families":
        counts = frame.groupby("family_key")["signature_key"].nunique()
        eligible_families = set(counts[counts >= 4].index)
        rng = np.random.default_rng(seed)
        reserved_train_indices: list[int] = []
        holdout_indices: list[int] = []
        for family, group in frame.groupby("family_key", sort=True):
            signatures = group["signature_key"].unique().tolist()
            rng.shuffle(signatures)
            if family not in eligible_families:
                reserved_train_indices.extend(group.index.tolist())
                continue
            reserved_signature = signatures.pop()
            reserved_train_indices.extend(
                group.loc[group["signature_key"] == reserved_signature].index.tolist()
            )
            holdout_indices.extend(
                group.loc[group["signature_key"].isin(signatures)].index.tolist()
            )
        reserved_train = frame.loc[reserved_train_indices]
        eligible = frame.loc[holdout_indices]
        for attempt in range(100):
            random_state = seed * 100 + attempt
            _, test_pos = next(GroupShuffleSplit(
                n_splits=1, test_size=0.1765, random_state=random_state
            ).split(eligible, groups=eligible[group_column]))
            test = eligible.iloc[test_pos]
            remaining = eligible.drop(index=test.index)
            _, calibration_pos = next(GroupShuffleSplit(
                n_splits=1, test_size=0.2143, random_state=random_state + 1
            ).split(remaining, groups=remaining[group_column]))
            calibration = remaining.iloc[calibration_pos]
            remaining = remaining.drop(index=calibration.index)
            _, validation_pos = next(GroupShuffleSplit(
                n_splits=1, test_size=0.2778, random_state=random_state + 2
            ).split(remaining, groups=remaining[group_column]))
            validation = remaining.iloc[validation_pos]
            train = pd.concat([reserved_train, remaining.drop(index=validation.index)])
            return {
                "train": train.copy(), "validation": validation.copy(),
                "calibration": calibration.copy(), "test": test.copy(),
            }
        raise RuntimeError(f"No valid {protocol} split for seed {seed}")
    else:
        eligible = frame
    for attempt in range(100):
        random_state = seed * 100 + attempt
        _, test_pos = next(GroupShuffleSplit(
            n_splits=1, test_size=0.15, random_state=random_state
        ).split(eligible, groups=eligible[group_column]))
        test = eligible.iloc[test_pos]
        remaining = frame.drop(index=test.index)
        _, calibration_pos = next(GroupShuffleSplit(
            n_splits=1, test_size=0.1765, random_state=random_state + 1
        ).split(remaining, groups=remaining[group_column]))
        calibration = remaining.iloc[calibration_pos]
        remaining = remaining.drop(index=calibration.index)
        _, validation_pos = next(GroupShuffleSplit(
            n_splits=1, test_size=0.2143, random_state=random_state + 2
        ).split(remaining, groups=remaining[group_column]))
        validation = remaining.iloc[validation_pos]
        train = remaining.drop(index=validation.index)
        train_families = set(train["family_key"])
        if protocol == "known_families":
            valid = (
                set(test["family_key"]).issubset(train_families)
                and set(calibration["family_key"]).issubset(train_families)
                and set(validation["family_key"]).issubset(train_families)
            )
        else:
            valid = not (
                (set(test["family_key"]) & train_families)
                or (set(calibration["family_key"]) & train_families)
                or (set(validation["family_key"]) & train_families)
            )
        if valid:
            return {
                "train": train.copy(), "validation": validation.copy(),
                "calibration": calibration.copy(), "test": test.copy(),
            }
    raise RuntimeError(f"No valid {protocol} split for seed {seed}")


def _preprocessor(include_city: bool) -> ColumnTransformer:
    categories = CATEGORICAL + (["city"] if include_city else [])
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, categories),
    ])


def _fit_predict(
    split: dict[str, pd.DataFrame], include_city: bool
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    features = NUMERIC + CATEGORICAL + (["city"] if include_city else [])
    processor = _preprocessor(include_city)
    train = split["train"]
    validation = split["validation"]
    x_train = processor.fit_transform(train[features])
    x_validation = processor.transform(validation[features])
    model = xgb.XGBRegressor(
        objective="reg:squarederror", tree_method="hist", n_jobs=2,
        random_state=42, early_stopping_rounds=30, **CONFIG,
    )
    y_train = np.log1p(train["price_lakhs"].to_numpy(float))
    y_validation = np.log1p(validation["price_lakhs"].to_numpy(float))
    model.fit(x_train, y_train, eval_set=[(x_validation, y_validation)], verbose=False)
    def predict(frame: pd.DataFrame) -> np.ndarray:
        return np.maximum(np.expm1(model.predict(processor.transform(frame[features]))), 0)
    return predict(split["calibration"]), predict(split["test"]), float(
        mean_absolute_error(validation["price_lakhs"], predict(validation))
    ), {
        "features": features,
        "config": CONFIG,
        "target_mode": "log1p",
        "best_iteration": int(model.best_iteration),
        "transformed_feature_count": int(model.get_booster().num_features()),
    }


def _conformal(
    calibration_actual: np.ndarray, calibration_prediction: np.ndarray,
    test_actual: np.ndarray, test_prediction: np.ndarray,
) -> dict[str, float]:
    residuals = np.abs(calibration_actual - calibration_prediction)
    rank = int(np.ceil((len(residuals) + 1) * 0.90))
    quantile = float(np.sort(residuals)[min(rank - 1, len(residuals) - 1)])
    lower = np.maximum(0, test_prediction - quantile)
    upper = test_prediction + quantile
    return {
        "nominal_coverage": 0.90,
        "calibration_rows": int(len(residuals)),
        "calibration_quantile_lakhs": quantile,
        "empirical_test_coverage": float(np.mean(
            (test_actual >= lower) & (test_actual <= upper)
        )),
        "average_interval_width_lakhs": float(np.mean(upper - lower)),
    }


def _audit(split: dict[str, pd.DataFrame]) -> dict[str, Any]:
    def one(frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(frame)),
            "source_row_ids": frame["source_row_id"].tolist(),
            "unique_manufacturers": int(frame["brand"].nunique()),
            "unique_model_families": int(frame["family_key"].nunique()),
            "unique_signatures": int(frame["signature_key"].nunique()),
        }
    sets = {name: set(frame["signature_key"]) for name, frame in split.items()}
    family_sets = {name: set(frame["family_key"]) for name, frame in split.items()}
    return {
        "partitions": {name: one(frame) for name, frame in split.items()},
        "signature_overlap": {
            f"{left}_{right}": len(sets[left] & sets[right])
            for left in sets for right in sets if left < right
        },
        "family_overlap_with_train": {
            name: len(family_sets[name] & family_sets["train"])
            for name in ("validation", "calibration", "test")
        },
    }


def _hashes() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in PROTECTED:
        path = ARTIFACTS / name
        if not path.exists():
            result[name] = None
            continue
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def main() -> None:
    reject_restricted_dataset(ORIGINAL_PATH)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = _hashes()
    datasets = {"dataset_b": _load_kaggle(), "dataset_a": _load_original()}
    runs: list[dict[str, Any]] = []
    for seed in SEEDS:
        for protocol in ("known_families", "unseen_families"):
            for include_city in (False, True):
                if protocol == "unseen_families" and len(datasets["dataset_b"]) == 0:
                    continue
                data = datasets["dataset_b"]
                split = _split(data, protocol, seed)
                calibration_prediction, test_prediction, validation_mae, model_info = _fit_predict(
                    split, include_city
                )
                calibration_actual = split["calibration"]["price_lakhs"].to_numpy(float)
                test_actual = split["test"]["price_lakhs"].to_numpy(float)
                name = f"dataset_b_{protocol}_{'city' if include_city else 'no_city'}"
                run = {
                    "dataset": "dataset_b",
                    "protocol": protocol,
                    "include_city": include_city,
                    "seed": seed,
                    "validation_mae_for_selection": validation_mae,
                    "test_metrics": _metrics(test_actual, test_prediction),
                    "conformal": _conformal(
                        calibration_actual, calibration_prediction,
                        test_actual, test_prediction,
                    ),
                    "model": model_info,
                    "split_audit": _audit(split),
                    "prediction_file": f"{name}_seed_{seed}_predictions.csv",
                }
                pd.DataFrame({
                    "source_row_id": split["test"]["source_row_id"].to_numpy(),
                    "actual_price_lakhs": test_actual,
                    "predicted_price_lakhs": test_prediction,
                }).to_csv(OUTPUT / run["prediction_file"], index=False)
                (OUTPUT / f"{name}_seed_{seed}.json").write_text(
                    json.dumps(run, indent=2), encoding="utf-8"
                )
                runs.append(run)
        split = _split(datasets["dataset_a"], "known_families", seed)
        calibration_prediction, test_prediction, validation_mae, model_info = _fit_predict(
            split, False
        )
        calibration_actual = split["calibration"]["price_lakhs"].to_numpy(float)
        test_actual = split["test"]["price_lakhs"].to_numpy(float)
        name = "dataset_a_common_baseline"
        run = {
            "dataset": "dataset_a",
            "protocol": "common_feature_baseline_known_families",
            "include_city": False,
            "seed": seed,
            "validation_mae_for_selection": validation_mae,
            "test_metrics": _metrics(test_actual, test_prediction),
            "conformal": _conformal(
                calibration_actual, calibration_prediction,
                test_actual, test_prediction,
            ),
            "model": model_info,
            "split_audit": _audit(split),
            "prediction_file": f"{name}_seed_{seed}_predictions.csv",
        }
        pd.DataFrame({
            "source_row_id": split["test"]["source_row_id"].to_numpy(),
            "actual_price_lakhs": test_actual,
            "predicted_price_lakhs": test_prediction,
        }).to_csv(OUTPUT / run["prediction_file"], index=False)
        (OUTPUT / f"{name}_seed_{seed}.json").write_text(
            json.dumps(run, indent=2), encoding="utf-8"
        )
        runs.append(run)

    after = _hashes()
    if before != after:
        raise RuntimeError("Protected production artifact hash changed.")
    metric_names = [
        "mae", "rmse", "r2", "median_absolute_error",
        "within_10_percent", "within_20_percent", "within_30_percent",
    ]
    aggregate: dict[str, Any] = {}
    for key in sorted({(
        run["dataset"], run["protocol"], run["include_city"]
    ) for run in runs}):
        selected = [
            run for run in runs
            if (run["dataset"], run["protocol"], run["include_city"]) == key
        ]
        aggregate["|".join(map(str, key))] = {
            "seeds": [run["seed"] for run in selected],
            "metrics_mean": {
                metric: float(np.mean([run["test_metrics"][metric] for run in selected]))
                for metric in metric_names
            },
            "metrics_std": {
                metric: float(np.std([run["test_metrics"][metric] for run in selected], ddof=1))
                for metric in metric_names
            },
            "conformal_mean": {
                metric: float(np.mean([run["conformal"][metric] for run in selected]))
                for metric in ("empirical_test_coverage", "average_interval_width_lakhs")
            },
            "conformal_std": {
                metric: float(np.std([run["conformal"][metric] for run in selected], ddof=1))
                for metric in ("empirical_test_coverage", "average_interval_width_lakhs")
            },
        }
    metadata = {
        "experiment": "five-seed expanded evaluation",
        "generated_by": "training/run_repeated_expanded_evaluation.py",
        "seeds": SEEDS,
        "expected_evaluations": 25,
        "actual_evaluations": len(runs),
        "dataset_c_used": False,
        "production_artifacts_modified": False,
        "test_used_for_tuning": False,
        "calibration_test_separation": True,
        "protected_hashes_before": before,
        "protected_hashes_after": after,
        "price_definitions": {
            "dataset_a": "Original price(in lakhs), preserved separately.",
            "dataset_b": "Kaggle price divided by 100000 into INR lakhs.",
        },
        "note": "Expanded-model conformal results are experimental and not production calibration artifacts.",
    }
    (OUTPUT / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    (OUTPUT / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (OUTPUT / "protected_hashes.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "evaluations": len(runs),
        "output_directory": str(OUTPUT),
        "protected_hashes_unchanged": before == after,
    }, indent=2))


if __name__ == "__main__":
    main()
