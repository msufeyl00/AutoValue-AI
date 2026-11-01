# AutoValue AI data sources

## Approved source

### Dataset B — Used Car Data by Sanjeet Singh Naik

- Canonical local file: `data/external/UsedCars_Combined.csv`
- Source URL: `https://www.kaggle.com/datasets/sanjeetsinghnaik/used-car-information`
- Licence: **CC0 1.0 Universal / Public Domain Dedication**, as reported by the Kaggle dataset listing.
- Retained files: only the combined CSV is used as the approved source.
- Excluded alternatives: city-specific CSVs and `data/external/Dataset.csv` are redundant or contain engineered ranking fields and are not retained for training.
- Attribution: retain the dataset name, author, Kaggle URL, and CC0 notice in published dataset documentation.

## Excluded sources

### Dataset A — `Used Car Dataset.csv`

- Location: `data/original/Used Car Dataset.csv`
- Status: removed from the active project because a verified reuse licence or permission was not established.
- Existing ANN, original XGBoost, calibrated XGBoost, and XGBoost V2 artifacts were trained from this source. They remain local-only backups for continuity but are not cleared for publication, deployment, or GitHub.
- These models must be retrained from Dataset B before any public release or production claim.

### Dataset C — `in3us/Used-Car-Market-Analysis`

- Location: `data/external/dataset_c_github/used_cars.csv`
- Source URL: `https://github.com/in3us/Used-Car-Market-Analysis`
- Status: removed because the repository did not declare an explicit reuse licence or permission.
- Dataset C reports and derived files were removed from the active project.
- It must not be used for training, redistribution, or commercial reuse unless permission is obtained.

## Model and artifact policy

The current FastAPI artifacts are historical local backups trained from Dataset A. They are intentionally excluded from GitHub and must not be represented as licence-cleared models. Retrain the ANN, production XGBoost, calibration artifacts, and any V2 model using Dataset B before publishing or deploying a cleared model.

The expanded Dataset B evaluation outputs are retained only when their metadata identifies Dataset B and contains no Dataset A or Dataset C records. Mixed-source reports and Dataset A prediction outputs are removed or regenerated.

## Git history

Restricted datasets and derived files were present in earlier commits, including `91f3386` and `872559e`. Removing current files and excluding future commits does not erase those historical blobs. A repository history rewrite and force-push would be required to remove them from Git history; no history rewrite or push was performed.
