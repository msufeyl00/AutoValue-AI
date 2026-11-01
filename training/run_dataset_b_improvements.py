"""Leakage-controlled Dataset B model improvement experiments."""

from __future__ import annotations

import hashlib
import json
import platform
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from train_dataset_b_xgboost import (
    DATA_PATH,
    TARGET,
    _read_dataset,
    _split_groups,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "dataset_b_improvements"
SEEDS = [42, 123, 2026, 3407, 7777]
MIN_SUBGROUP_ROWS = 10
BASE_NUMERIC = ["manufacturing_year", "kms_driven"]
BASE_CATEGORICAL = ["brand", "model_family", "fuel_type", "city"]
PARAMS = {
    "baseline": {
        "n_estimators": 350, "max_depth": 3, "learning_rate": 0.03,
        "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.9,
        "reg_alpha": 0.1, "reg_lambda": 10.0,
    },
    "tuned_log": {
        "n_estimators": 250, "max_depth": 2, "learning_rate": 0.03,
        "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.9,
        "reg_alpha": 0.1, "reg_lambda": 10.0,
    },
}
CANDIDATES = {
    "current_baseline": {"target": "log", "objective": "reg:squarederror",
                         "features": "base", "params": "baseline"},
    "raw_price": {"target": "raw", "objective": "reg:squarederror",
                  "features": "base", "params": "baseline"},
    "log_price_tuned": {"target": "log", "objective": "reg:squarederror",
                        "features": "base", "params": "tuned_log"},
    "pseudo_huber": {"target": "log", "objective": "reg:pseudohubererror",
                     "features": "base", "params": "baseline"},
    "interaction_features": {"target": "log", "objective": "reg:squarederror",
                             "features": "interaction", "params": "baseline"},
    "full_model_name": {"target": "log", "objective": "reg:squarederror",
                        "features": "full_name", "params": "baseline"},
}


def _features(frame: pd.DataFrame, mode: str) -> tuple[list[str], list[str], list[str]]:
    data = frame
    data["brand_model_family"] = (
        data["brand"].astype(str).str.lower() + "__" +
        data["model_family"].astype(str).str.lower()
    )
    categorical = list(BASE_CATEGORICAL)
    if mode in {"interaction", "full_name"}:
        categorical.append("brand_model_family")
    if mode == "full_name":
        categorical.append("full_model_name")
    return BASE_NUMERIC + categorical, BASE_NUMERIC, categorical


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]), categorical),
    ])


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0.0)
    relative = np.abs(predicted - actual) / np.maximum(actual, 1e-8)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "median_absolute_error": float(median_absolute_error(actual, predicted)),
        "within_10_percent": float(np.mean(relative <= 0.10)),
        "within_20_percent": float(np.mean(relative <= 0.20)),
        "within_30_percent": float(np.mean(relative <= 0.30)),
        "rows": int(len(actual)),
    }


def _target_values(frame: pd.DataFrame, target: str) -> np.ndarray:
    values = frame[TARGET].to_numpy(dtype=float)
    return np.log1p(values) if target == "log" else values


def _inverse(values: np.ndarray, target: str) -> np.ndarray:
    return np.expm1(values) if target == "log" else np.maximum(values, 0.0)


def _price_bands(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    quantiles = train[TARGET].quantile([1 / 3, 2 / 3]).to_numpy()
    bands = pd.cut(
        test[TARGET],
        bins=[-np.inf, quantiles[0], quantiles[1], np.inf],
        labels=["low", "median", "high"],
    )
    return pd.DataFrame({"band": bands, "actual": test[TARGET].to_numpy()})


def _fit(
    frame: pd.DataFrame,
    splits: dict[str, np.ndarray],
    specification: dict[str, Any],
    seed: int,
) -> tuple[Pipeline, dict[str, float], list[str], list[str]]:
    features, numeric, categorical = _features(frame, specification["features"])
    train = frame.iloc[splits["train"]]
    validation = frame.iloc[splits["validation"]]
    model = xgb.XGBRegressor(
        objective=specification["objective"], eval_metric="mae",
        tree_method="hist", n_jobs=2, random_state=seed,
        **PARAMS[specification["params"]],
    )
    pipeline = Pipeline([("preprocessor", _preprocessor(numeric, categorical)),
                         ("model", model)])
    pipeline.fit(train[features], _target_values(train, specification["target"]))
    validation_prediction = _inverse(
        pipeline.predict(validation[features]), specification["target"]
    )
    return pipeline, _metrics(validation[TARGET].to_numpy(), validation_prediction), features, numeric + categorical


def _evaluate(
    frame: pd.DataFrame,
    candidate: str,
    protocol: str,
    group_column: str,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    specification = CANDIDATES[candidate]
    splits = _split_groups(frame, group_column, seed)
    partition_groups = {
        name: set(frame.iloc[index][group_column].astype(str))
        for name, index in splits.items()
    }
    overlap_count = sum(
        len(partition_groups[left] & partition_groups[right])
        for left, right in [
            ("train", "validation"),
            ("train", "calibration"),
            ("train", "test"),
            ("validation", "calibration"),
            ("validation", "test"),
            ("calibration", "test"),
        ]
    )
    if overlap_count:
        raise RuntimeError(
            f"Group leakage detected for {candidate}/{protocol}/seed {seed}: "
            f"{overlap_count} overlapping groups."
        )
    try:
        pipeline, validation_metrics, features, schema = _fit(
            frame, splits, specification, seed
        )
    except xgb.core.XGBoostError as exc:
        if specification["objective"] != "reg:pseudohubererror":
            raise
        return {
            "candidate": candidate, "protocol": protocol, "seed": seed,
            "status": "unsupported_objective", "error": str(exc),
        }, pd.DataFrame()

    calibration = frame.iloc[splits["calibration"]]
    calibration_prediction = _inverse(
        pipeline.predict(calibration[features]), specification["target"]
    )
    residuals = np.abs(calibration[TARGET].to_numpy() - calibration_prediction)
    rank = int(np.ceil((len(residuals) + 1) * 0.90))
    quantile = float(np.sort(residuals)[min(rank, len(residuals)) - 1])

    test = frame.iloc[splits["test"]].copy()
    prediction = _inverse(pipeline.predict(test[features]), specification["target"])
    actual = test[TARGET].to_numpy(dtype=float)
    lower = np.maximum(prediction - quantile, 0.0)
    upper = prediction + quantile
    metrics = _metrics(actual, prediction)
    baseline = np.full(len(test), float(np.median(frame.iloc[splits["train"]][TARGET])))
    baseline_metrics = _metrics(actual, baseline)
    bands = _price_bands(frame.iloc[splits["train"]], test)
    subgroup: dict[str, dict[str, float]] = {}
    for band in ["low", "median", "high"]:
        mask = bands["band"].to_numpy() == band
        if int(mask.sum()) >= MIN_SUBGROUP_ROWS:
            subgroup[band] = {
                "rows": int(mask.sum()),
                "mae": float(mean_absolute_error(actual[mask], prediction[mask])),
                "bias": float(np.mean(prediction[mask] - actual[mask])),
            }
    test["prediction_lakhs"] = prediction
    test["actual_lakhs"] = actual
    test["error_lakhs"] = prediction - actual
    test["absolute_error_lakhs"] = np.abs(prediction - actual)
    test["interval_lower_lakhs"] = lower
    test["interval_upper_lakhs"] = upper
    test["candidate"] = candidate
    test["protocol"] = protocol
    test["seed"] = seed
    result = {
        "candidate": candidate, "protocol": protocol, "seed": seed, "status": "ok",
        "target_mode": specification["target"], "objective": specification["objective"],
        "feature_mode": specification["features"], "features": features,
        "group_overlap_count": overlap_count,
        "validation_metrics": validation_metrics, "test_metrics": metrics,
        "baseline_test_metrics": baseline_metrics,
        "baseline_beaten": bool(metrics["mae"] < baseline_metrics["mae"]),
        "subgroup_metrics": subgroup,
        "calibration_rows": int(len(calibration)),
        "calibration_quantile_lakhs": quantile, "nominal_coverage": 0.90,
        "empirical_test_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "average_interval_width_lakhs": float(np.mean(upper - lower)),
        "split_rows": {name: int(len(index)) for name, index in splits.items()},
        "feature_schema": schema,
        "pipeline": pipeline,
    }
    return result, test


def _flatten(results: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {k: result[k] for k in [
            "candidate", "protocol", "seed", "status", "target_mode",
            "objective", "feature_mode", "calibration_rows",
            "calibration_quantile_lakhs", "nominal_coverage",
            "empirical_test_coverage", "average_interval_width_lakhs",
            "baseline_beaten", "group_overlap_count",
        ]}
        for prefix, values in [
            ("validation", result.get("validation_metrics", {})),
            ("test", result.get("test_metrics", {})),
            ("baseline", result.get("baseline_test_metrics", {})),
        ]:
            for key, value in values.items():
                row[f"{prefix}_{key}"] = value
        for band, values in result.get("subgroup_metrics", {}).items():
            row[f"{band}_rows"] = values["rows"]
            row[f"{band}_mae"] = values["mae"]
            row[f"{band}_bias"] = values["bias"]
        rows.append(row)
    return pd.DataFrame(rows)


def _aggregate(metrics: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    numeric = [
        "validation_mae", "test_mae", "test_rmse", "test_r2",
        "test_median_absolute_error", "test_within_10_percent",
        "test_within_20_percent", "test_within_30_percent",
        "empirical_test_coverage", "average_interval_width_lakhs",
        "high_mae", "high_bias", "median_mae", "median_bias", "low_mae", "low_bias",
    ]
    valid = metrics[metrics["status"] == "ok"]
    for (candidate, protocol), group in valid.groupby(["candidate", "protocol"]):
        output[f"{candidate}__{protocol}"] = {
            "seed_count": int(len(group)),
            "mean": {column: float(group[column].mean()) for column in numeric if column in group},
            "std": {column: float(group[column].std(ddof=1)) for column in numeric if column in group},
            "min": {column: float(group[column].min()) for column in numeric if column in group},
            "max": {column: float(group[column].max()) for column in numeric if column in group},
            "baseline_beaten_seeds": int(group["baseline_beaten"].sum()),
        }
    return output


def _select_candidate(metrics: pd.DataFrame) -> str:
    valid = metrics[metrics["status"] == "ok"].copy()
    summary = valid.groupby("candidate").agg(
        validation_mae=("validation_mae", "mean"),
    )
    return str(summary.sort_values(["validation_mae"]).index[0])


def _write_report(metrics: pd.DataFrame, aggregate: dict[str, Any], selected: str) -> None:
    lines = [
        "# Dataset B improvement report", "",
        f"Selected by validation-only ranking: **{selected}**.",
        "Historical five-seed test results were not used as tuning feedback.",
        "All reported test metrics reuse the established grouped exploratory splits and require a future locked evaluation.",
        "",
        "## Candidate comparison",
        "",
        "| Candidate | Protocol | Validation MAE mean | Test MAE mean | Test RMSE mean | High-price MAE mean | High-price bias mean | Coverage mean | Width mean | Baseline wins |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, values in aggregate.items():
        candidate, protocol = key.split("__", 1)
        mean = values["mean"]
        lines.append(
            f"| {candidate} | {protocol} | {mean.get('validation_mae', float('nan')):.3f} | "
            f"{mean.get('test_mae', float('nan')):.3f} | {mean.get('test_rmse', float('nan')):.3f} | "
            f"{mean.get('high_mae', float('nan')):.3f} | {mean.get('high_bias', float('nan')):.3f} | "
            f"{mean.get('empirical_test_coverage', float('nan')):.3f} | "
            f"{mean.get('average_interval_width_lakhs', float('nan')):.3f} | "
            f"{values['baseline_beaten_seeds']}/{values['seed_count']} |"
        )
    lines.extend([
        "", "## Limitations", "",
        "- Expensive vehicles remain difficult because premium brands and families have sparse, heterogeneous samples.",
        "- Conformal intervals are marginal empirical intervals for these grouped protocols; they do not guarantee individual coverage.",
        "- Unseen-family performance and interval widths are especially sensitive to which families are held out.",
        "- A new independent dataset or locked future evaluation is required before deployment claims.",
    ])
    (OUTPUT / "improvement_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    frame, quality = _read_dataset()
    results: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for seed in SEEDS:
        for candidate in CANDIDATES:
            for protocol, group_column in [
                ("known_model_families", "signature"),
                ("unseen_model_families", "model_family"),
            ]:
                result, prediction = _evaluate(
                    frame, candidate, protocol, group_column, seed
                )
                results.append(result)
                if not prediction.empty:
                    predictions.append(prediction)

    metrics = _flatten(results)
    aggregate = _aggregate(metrics)
    selected = _select_candidate(metrics)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT / "per_seed_metrics.csv", index=False)
    metrics.groupby("candidate", as_index=False).mean(numeric_only=True).to_csv(
        OUTPUT / "candidate_comparison.csv", index=False
    )
    pd.concat(predictions, ignore_index=True).to_csv(
        OUTPUT / "high_price_error_analysis.csv", index=False
    )
    (OUTPUT / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    ablation = {
        name: {
            "feature_mode": spec["features"],
            "features": next(
                (
                    results[i]["features"]
                    for i in range(len(results))
                    if results[i]["candidate"] == name
                    and results[i].get("status") == "ok"
                ),
                [],
            ),
        }
        for name, spec in CANDIDATES.items()
    }
    (OUTPUT / "feature_ablation_report.json").write_text(
        json.dumps(ablation, indent=2, default=str), encoding="utf-8"
    )
    calibration = metrics.groupby(["candidate", "protocol"]).agg(
        coverage_mean=("empirical_test_coverage", "mean"),
        coverage_std=("empirical_test_coverage", "std"),
        width_mean=("average_interval_width_lakhs", "mean"),
        width_std=("average_interval_width_lakhs", "std"),
    ).reset_index()
    calibration.to_json(OUTPUT / "calibration_comparison.json", orient="records", indent=2)
    _write_report(metrics, aggregate, selected)

    selected_result = next(
        result for result in results
        if result.get("candidate") == selected
        and result.get("protocol") == "known_model_families"
        and result.get("seed") == 2026
        and result.get("status") == "ok"
    )
    pipeline = selected_result["pipeline"]
    pipeline.named_steps["model"].save_model(
        OUTPUT / "selected_dataset_b_xgboost_model.json"
    )
    joblib.dump(
        pipeline.named_steps["preprocessor"],
        OUTPUT / "selected_dataset_b_preprocessor.joblib",
    )
    (OUTPUT / "experiment_metadata.json").write_text(
        json.dumps({
            "dataset": "Kaggle Used Car Data by Sanjeet Singh Naik",
            "source": str(DATA_PATH.relative_to(ROOT)),
            "license": "CC0 Public Domain",
            "dataset_fingerprint_sha256": _fingerprint(DATA_PATH),
            "rows_used": quality["usable_rows"],
            "seeds": SEEDS,
            "candidates": CANDIDATES,
            "selected_candidate": selected,
            "selection_basis": "validation metrics only; test metrics are exploratory evidence",
            "package_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__, "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__, "xgboost": xgb.__version__,
            },
        }, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print(f"\nselected_candidate={selected}")


if __name__ == "__main__":
    main()
