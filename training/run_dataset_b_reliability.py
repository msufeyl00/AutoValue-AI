"""Five-seed reliability evaluation for the Dataset B XGBoost candidate."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from train_dataset_b_xgboost import (
    CATEGORICAL,
    DATA_PATH,
    FEATURES,
    NUMERIC,
    TARGET,
    _read_dataset,
    _split_groups,
)
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "dataset_b_reliability"
SEEDS = [42, 123, 2026, 3407, 7777]
MIN_DIAGNOSTIC_ROWS = 10
PARAMETER_CANDIDATES = [
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


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
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
        "median_absolute_error": float(median_absolute_error(actual, predicted)),
        "within_10_percent": float(np.mean(relative <= 0.10)),
        "within_20_percent": float(np.mean(relative <= 0.20)),
        "within_30_percent": float(np.mean(relative <= 0.30)),
        "test_rows": int(len(actual)),
    }


def _fit(
    frame: pd.DataFrame,
    splits: dict[str, np.ndarray],
    parameters: dict[str, Any],
    seed: int,
) -> tuple[Pipeline, dict[str, float]]:
    train = frame.iloc[splits["train"]]
    validation = frame.iloc[splits["validation"]]
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        n_jobs=2,
        random_state=seed,
        **parameters,
    )
    pipeline = Pipeline([("preprocessor", _preprocessor()), ("model", model)])
    pipeline.fit(train[FEATURES], np.log1p(train[TARGET].to_numpy()))
    prediction = np.expm1(pipeline.predict(validation[FEATURES]))
    return pipeline, _metrics(validation[TARGET].to_numpy(), prediction)


def _overlap_audit(
    frame: pd.DataFrame, splits: dict[str, np.ndarray], group_column: str
) -> dict[str, Any]:
    groups = {
        name: set(frame.iloc[index][group_column].astype(str))
        for name, index in splits.items()
    }
    overlaps = {
        f"{left}_vs_{right}": len(groups[left] & groups[right])
        for left, right in (
            ("train", "validation"),
            ("train", "calibration"),
            ("train", "test"),
            ("validation", "calibration"),
            ("validation", "test"),
            ("calibration", "test"),
        )
    }
    return {
        "group_column": group_column,
        "overlap_counts": overlaps,
        "partitions": {
            name: {
                "rows": int(len(index)),
                "unique_manufacturers": int(frame.iloc[index]["brand"].nunique()),
                "unique_model_families": int(
                    frame.iloc[index]["model_family"].nunique()
                ),
                "unique_signatures": int(
                    frame.iloc[index]["signature"].nunique()
                ),
            }
            for name, index in splits.items()
        },
    }


def _evaluate(
    frame: pd.DataFrame, protocol: str, group_column: str, seed: int
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    splits = _split_groups(frame, group_column, seed)
    candidates = []
    for parameters in PARAMETER_CANDIDATES:
        pipeline, validation_metrics = _fit(frame, splits, parameters, seed)
        candidates.append((validation_metrics["mae"], pipeline, parameters, validation_metrics))
    _, pipeline, selected_parameters, validation_metrics = min(
        candidates, key=lambda item: item[0]
    )

    calibration = frame.iloc[splits["calibration"]]
    calibration_prediction = np.maximum(
        np.expm1(pipeline.predict(calibration[FEATURES])), 0.0
    )
    residuals = np.abs(calibration[TARGET].to_numpy() - calibration_prediction)
    rank = int(np.ceil((len(residuals) + 1) * 0.90))
    quantile = float(np.sort(residuals)[min(rank, len(residuals)) - 1])

    test = frame.iloc[splits["test"]].copy()
    prediction = np.maximum(np.expm1(pipeline.predict(test[FEATURES])), 0.0)
    actual = test[TARGET].to_numpy(dtype=float)
    lower = np.maximum(prediction - quantile, 0.0)
    upper = prediction + quantile
    test_metrics = _metrics(actual, prediction)
    baseline_prediction = np.full(
        len(test), float(np.median(frame.iloc[splits["train"]][TARGET]))
    )
    baseline_metrics = _metrics(actual, baseline_prediction)
    test["prediction_lakhs"] = prediction
    test["actual_lakhs"] = actual
    test["error_lakhs"] = prediction - actual
    test["absolute_error_lakhs"] = np.abs(prediction - actual)
    test["relative_error"] = np.abs(prediction - actual) / np.maximum(actual, 1e-8)
    test["interval_lower_lakhs"] = lower
    test["interval_upper_lakhs"] = upper
    test["protocol"] = protocol
    test["seed"] = seed
    test["partition"] = "test"

    result = {
        "protocol": protocol,
        "group_column": group_column,
        "seed": seed,
        "selected_hyperparameters": selected_parameters,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_metrics,
        "baseline_beaten": bool(test_metrics["mae"] < baseline_metrics["mae"]),
        "calibration_rows": int(len(calibration)),
        "calibration_quantile_lakhs": quantile,
        "nominal_coverage": 0.90,
        "empirical_test_coverage": float(
            np.mean((actual >= lower) & (actual <= upper))
        ),
        "average_interval_width_lakhs": float(np.mean(upper - lower)),
    }
    return result, test, _overlap_audit(frame, splits, group_column)


def _aggregate(records: pd.DataFrame) -> dict[str, Any]:
    metric_columns = [
        "mae",
        "rmse",
        "r2",
        "median_absolute_error",
        "within_10_percent",
        "within_20_percent",
        "within_30_percent",
        "test_rows",
        "calibration_rows",
        "empirical_test_coverage",
        "average_interval_width_lakhs",
    ]
    aggregate: dict[str, Any] = {}
    for protocol, group in records.groupby("protocol", sort=True):
        aggregate[protocol] = {
            "mean": {column: float(group[column].mean()) for column in metric_columns},
            "std": {column: float(group[column].std(ddof=1)) for column in metric_columns},
            "min": {column: float(group[column].min()) for column in metric_columns},
            "max": {column: float(group[column].max()) for column in metric_columns},
            "baseline_beaten_seeds": int(group["baseline_beaten"].sum()),
            "seed_count": int(len(group)),
        }
    return aggregate


def _diagnostic_table(predictions: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    grouped = predictions.groupby(column, dropna=False)
    rows = []
    for value, group in grouped:
        if len(group) < MIN_DIAGNOSTIC_ROWS:
            continue
        rows.append(
            {
                column: str(value),
                "rows": int(len(group)),
                "mae_lakhs": float(group["absolute_error_lakhs"].mean()),
                "median_absolute_error_lakhs": float(
                    group["absolute_error_lakhs"].median()
                ),
                "mean_signed_error_lakhs": float(group["error_lakhs"].mean()),
                "actual_median_lakhs": float(group["actual_lakhs"].median()),
            }
        )
    return sorted(rows, key=lambda row: row["mae_lakhs"], reverse=True)


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_report(
    records: pd.DataFrame,
    predictions: pd.DataFrame,
    aggregate: dict[str, Any],
    quality: dict[str, Any],
) -> None:
    lines = [
        "# Dataset B reliability report",
        "",
        f"- Dataset rows used: **{quality['usable_rows']}**",
        f"- Seeds: **{', '.join(str(seed) for seed in SEEDS)}**",
        "- Target: `log1p(price_lakhs)` during fitting, inverted with `expm1`.",
        "- Test data was excluded from model selection and calibration quantile calculation.",
        "",
        "## Aggregate test metrics",
        "",
        "| Protocol | MAE mean ± std | RMSE mean ± std | R² mean ± std | Within 20% mean ± std | Coverage mean ± std | Width mean ± std |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for protocol, values in aggregate.items():
        mean = values["mean"]
        std = values["std"]
        lines.append(
            f"| {protocol} | {mean['mae']:.3f} ± {std['mae']:.3f} | "
            f"{mean['rmse']:.3f} ± {std['rmse']:.3f} | "
            f"{mean['r2']:.3f} ± {std['r2']:.3f} | "
            f"{mean['within_20_percent']:.3f} ± {std['within_20_percent']:.3f} | "
            f"{mean['empirical_test_coverage']:.3f} ± {std['empirical_test_coverage']:.3f} | "
            f"{mean['average_interval_width_lakhs']:.3f} ± {std['average_interval_width_lakhs']:.3f} |"
        )
    all_predictions = predictions.copy()
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"Manufacturer diagnostics include only groups with at least {MIN_DIAGNOSTIC_ROWS} pooled test rows.",
            "",
            "### Manufacturers with largest pooled MAE",
            "",
            "| Manufacturer | Rows | MAE (lakhs) | Median AE (lakhs) | Signed error (lakhs) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in _diagnostic_table(all_predictions, "brand")[:10]:
        lines.append(
            f"| {row['brand']} | {row['rows']} | {row['mae_lakhs']:.3f} | "
            f"{row['median_absolute_error_lakhs']:.3f} | {row['mean_signed_error_lakhs']:.3f} |"
        )
    lines.extend(
        [
            "",
            "### Model families with largest pooled MAE",
            "",
            f"Model-family diagnostics include only groups with at least {MIN_DIAGNOSTIC_ROWS} pooled test rows.",
            "",
            "| Model family | Rows | MAE (lakhs) | Median AE (lakhs) | Signed error (lakhs) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in _diagnostic_table(all_predictions, "model_family")[:10]:
        lines.append(
            f"| {row['model_family']} | {row['rows']} | {row['mae_lakhs']:.3f} | "
            f"{row['median_absolute_error_lakhs']:.3f} | {row['mean_signed_error_lakhs']:.3f} |"
        )
    signed_by_price = all_predictions.assign(
        price_band=pd.qcut(
            all_predictions["actual_lakhs"], q=3, duplicates="drop"
        )
    ).groupby("price_band", observed=True)["error_lakhs"].agg(["count", "mean"])
    lines.extend(
        [
            "",
            "### Price-band bias",
            "",
            "| Actual-price band | Rows | Mean signed error (lakhs) |",
            "|---|---:|---:|",
        ]
    )
    for band, row in signed_by_price.iterrows():
        lines.append(f"| {band} | {int(row['count'])} | {row['mean']:.3f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Seed variability must be judged from the saved mean, standard deviation, minimum, and maximum rather than one split.",
            "- Coverage and interval width are reported separately for known-family and unseen-family protocols; unseen-family intervals are not automatically valid for deployment.",
            "- Low-sample manufacturer and model-family groups are excluded from diagnostics and should not support subgroup claims.",
            "- This reliability stage does not modify or promote the current candidate model.",
        ]
    )
    (OUTPUT / "reliability_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    frame, quality = _read_dataset()
    records: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    split_audits: dict[str, Any] = {}
    for seed in SEEDS:
        for protocol, group_column in (
            ("known_model_families", "signature"),
            ("unseen_model_families", "model_family"),
        ):
            result, prediction, audit = _evaluate(
                frame, protocol, group_column, seed
            )
            records.append(result)
            predictions.append(prediction)
            split_audits[f"{protocol}_seed_{seed}"] = audit

    record_frame = pd.DataFrame(records)
    metric_columns = [
        "mae",
        "rmse",
        "r2",
        "median_absolute_error",
        "within_10_percent",
        "within_20_percent",
        "within_30_percent",
    ]
    flat_records = pd.DataFrame(
        [
            {
                "protocol": row["protocol"],
                "seed": row["seed"],
                **{metric: row["test_metrics"][metric] for metric in metric_columns},
                "test_rows": row["test_metrics"]["test_rows"],
                "calibration_rows": row["calibration_rows"],
                "empirical_test_coverage": row["empirical_test_coverage"],
                "average_interval_width_lakhs": row[
                    "average_interval_width_lakhs"
                ],
                "baseline_mae": row["baseline_test_metrics"]["mae"],
                "baseline_beaten": row["baseline_beaten"],
                "selected_n_estimators": row["selected_hyperparameters"][
                    "n_estimators"
                ],
                "selected_max_depth": row["selected_hyperparameters"]["max_depth"],
            }
            for row in records
        ]
    )
    prediction_frame = pd.concat(predictions, ignore_index=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    flat_records.to_csv(OUTPUT / "per_seed_metrics.csv", index=False)
    prediction_frame.to_csv(OUTPUT / "prediction_results.csv", index=False)
    (OUTPUT / "aggregate_metrics.json").write_text(
        json.dumps(_aggregate(flat_records), indent=2), encoding="utf-8"
    )
    (OUTPUT / "split_audits.json").write_text(
        json.dumps(split_audits, indent=2), encoding="utf-8"
    )
    (OUTPUT / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "dataset": "Kaggle Used Car Data by Sanjeet Singh Naik",
                "source": str(DATA_PATH.relative_to(ROOT)),
                "license": "CC0 Public Domain",
                "dataset_fingerprint_sha256": _fingerprint(DATA_PATH),
                "seeds": SEEDS,
                "protocols": {
                    "known_model_families": "grouped by repeated vehicle signature",
                    "unseen_model_families": "grouped by model_family",
                },
                "features": FEATURES,
                "categorical_features": CATEGORICAL,
                "numeric_features": NUMERIC,
                "parameter_candidates": PARAMETER_CANDIDATES,
                "package_versions": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "scikit_learn": sklearn.__version__,
                    "xgboost": xgb.__version__,
                },
                "current_candidate_preserved": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(flat_records, prediction_frame, _aggregate(flat_records), quality)
    print(flat_records.to_string(index=False))


if __name__ == "__main__":
    main()
