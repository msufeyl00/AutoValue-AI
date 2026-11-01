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

The legacy FastAPI artifacts are historical local backups trained from Dataset
A. They remain local-only fallbacks, are intentionally excluded from GitHub,
and must not be represented as licence-cleared models. They are distinct from
the committed Dataset B experimental bundle under
`artifacts/dataset_b_improvements/`, which was trained only from the approved
Dataset B source and includes its matching preprocessor and calibration
configuration.

The committed Dataset B evaluation outputs are retained only when their
metadata identifies Dataset B and contains no Dataset A or Dataset C records.
Mixed-source reports and Dataset A prediction outputs are excluded. The
Dataset B model remains experimental and is not a production-readiness claim.

## Git history


The repository was initialized with a clean Git history after removing
the previous local history containing restricted-source files.

The clean initial commit was published to:

https://github.com/msufeyl00/AutoValue-AI

The public main branch contains the reviewed project source code,
Dataset B experimental model bundle, and approved evaluation reports.

Restricted Dataset A and Dataset C files, derived reports, and
Dataset A-dependent legacy models are excluded from public Git history.