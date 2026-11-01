# AutoValue AI

AutoValue AI is a used-car price prediction application with a React/Vite
interface and a modular FastAPI backend. The public repository contains an
isolated Dataset B XGBoost experiment and does not publish raw datasets,
legacy Dataset A models, or Dataset C-derived files.

## Problem statement

Used-car prices vary with age, distance travelled, manufacturer, model family,
fuel type, and city. AutoValue AI provides an estimate in Indian rupees lakhs
and an empirically calibrated interval to support exploration. The interval is
not a guaranteed selling-price range.

## Features

- React dashboard with a legacy prediction interface retained for local
  compatibility.
- Separate **Dataset B Experimental Model** interface with six supported
  fields.
- FastAPI health, model information, metrics, and prediction endpoints.
- Dataset B XGBoost model with grouped evaluation and split-conformal calibration.
- Unknown categorical values handled safely by the fitted preprocessor.
- Feature flags keep the Dataset B endpoint and UI disabled by default.

## Architecture

```text
                         +----------------------+
                         | React/Vite frontend  |
                         | production +        |
                         | experimental UI     |
                         +----------+-----------+
                                    |
                         POST /api/experimental/
                         dataset-b/predict
                                    |
                         +----------v-----------+
                         | FastAPI routes       |
                         | validation + flags  |
                         +----------+-----------+
                                    |
                         +----------v-----------+
                         | Dataset B predictor  |
                         | XGBoost + preprocessor|
                         +----------+-----------+
                                    |
                         +----------v-----------+
                         | Model bundle         |
                         | JSON + joblib +      |
                         | calibration config  |
                         +----------------------+
```

The legacy `/api/predict` path is not replaced by the Dataset B model. If the
legacy local artifacts are absent, that route returns HTTP 503 rather than
silently using a different feature schema.

## Technology stack

- Python 3.12
- FastAPI, Uvicorn, Pydantic
- XGBoost, scikit-learn, pandas, NumPy, joblib
- React 18, Vite, Axios, Recharts
- pytest

## Dataset and licensing

The approved training source is Dataset B: **Used Car Data by Sanjeet Singh
Naik**, available from [Kaggle](https://www.kaggle.com/datasets/sanjeetsinghnaik/used-car-information).
The Kaggle listing reports a CC0 Public Domain licence.

The raw CSV is intentionally excluded from Git. To run training experiments:

1. Download the approved dataset from Kaggle.
2. Save it as `data/external/UsedCars_Combined.csv`.
3. Review [DATA_SOURCES.md](DATA_SOURCES.md) and confirm the source and licence.

Dataset A and Dataset C are excluded because their reuse permissions were not
verified. Do not copy their datasets or derived models into this repository.

## Machine-learning methodology

The experimental Dataset B model uses:

- `manufacturing_year` and `kms_driven` as numeric features.
- `brand`, `model_family`, `fuel_type`, `city`, and a
  `brand_model_family` interaction as categorical features.
- `price / 100000` as the target price in lakhs.
- `log1p(price_lakhs)` during fitting and `expm1` for inference.
- One-hot encoding with unknown categories ignored.
- Grouped train, validation, calibration, and test partitions.
- XGBoost histogram training on CPU.
- A separate calibration partition for nominal 90% split-conformal intervals.

The selected experimental bundle was evaluated on 1,725 usable Dataset B rows.
On the known-family protocol it reported test MAE **3.966 lakhs**, RMSE
**7.361 lakhs**, R² **0.712**, and 61.96% of predictions within 30%.
On the unseen-family protocol it reported test MAE **4.867 lakhs**, RMSE
**11.760 lakhs**, R² **0.374**, and 49.82% within 30%.

These are grouped exploratory results on the available dataset, not a
deployment guarantee. The known-family calibration reported approximately
96.08% empirical test coverage with an average interval width of 20.68 lakhs.
Unseen-family intervals were substantially wider in the reliability study.

## Limitations

- The Dataset B model is experimental and not production-ready.
- The dataset is relatively small and may not represent every market segment.
- Unseen model families and premium vehicles can have large errors.
- Conformal coverage is marginal and depends on representative calibration and
  deployment data; it does not guarantee coverage for an individual vehicle.
- The model does not use engine capacity, mileage, seats, transmission,
  ownership, or insurance because Dataset B does not reliably provide them.
- No price estimate should be treated as an appraisal or guaranteed sale price.

## Installation

From the repository root on Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm.cmd ci
Copy-Item .env.example .env
```

The committed model bundle is under
`artifacts/dataset_b_improvements/`, so the experimental inference service
does not require the raw CSV at startup.

## Run locally

By default both experimental flags are false:

```powershell
py -3.12 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

Open `http://127.0.0.1:5173`.

To enable the isolated experimental interface for local testing only, set both
flags before starting the relevant process:

```powershell
$env:ENABLE_DATASET_B_EXPERIMENTAL="true"
$env:VITE_ENABLE_DATASET_B_EXPERIMENTAL="true"
```

The backend flag is the security boundary; the frontend flag only controls
visibility. Keep both defaults false in project configuration.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Backend and legacy model availability |
| GET | `/api/model-info` | Legacy model metadata when available |
| GET | `/api/metrics` | Legacy model metrics when available |
| POST | `/api/predict` | Legacy 11-field prediction; returns 503 without local legacy artifacts |
| POST | `/api/experimental/dataset-b/predict` | Dataset B six-field prediction; disabled by default |

Dataset B accepts exactly `brand`, `model_family`, `manufacturing_year`,
`kms_driven`, `fuel_type`, and `city`. It returns an estimated price in lakhs,
lower and upper experimental bounds, a model identifier, and an unknown-family
warning when applicable.

## Validation

```powershell
npm.cmd run build
py -3.12 -m pytest backend\tests -q
```

## Screenshots

The experimental result view is validated locally through the browser with the
estimate, interval, model identifier, loading state, and disabled-endpoint
error state. Screenshots are kept outside the repository to avoid publishing
unnecessary user or machine metadata.

## Future enhancements

- Independent locked evaluation on a new licensed dataset.
- More robust coverage analysis for unseen model families.
- Better premium-vehicle and high-price calibration.
- Automated frontend interaction tests.
- Production deployment only after independent validation and a documented
  model governance review.
