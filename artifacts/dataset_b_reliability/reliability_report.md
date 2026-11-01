# Dataset B reliability report

- Dataset rows used: **1725**
- Seeds: **42, 123, 2026, 3407, 7777**
- Target: `log1p(price_lakhs)` during fitting, inverted with `expm1`.
- Test data was excluded from model selection and calibration quantile calculation.

## Aggregate test metrics

| Protocol | MAE mean ± std | RMSE mean ± std | R² mean ± std | Within 20% mean ± std | Coverage mean ± std | Width mean ± std |
|---|---:|---:|---:|---:|---:|---:|
| known_model_families | 4.579 ± 0.578 | 10.753 ± 2.744 | 0.508 ± 0.146 | 0.457 ± 0.035 | 0.922 ± 0.040 | 19.479 ± 3.338 |
| unseen_model_families | 4.968 ± 0.763 | 10.969 ± 2.786 | 0.450 ± 0.117 | 0.332 ± 0.060 | 0.926 ± 0.052 | 26.197 ± 12.556 |

## Diagnostics

Manufacturer diagnostics include only groups with at least 10 pooled test rows.

### Manufacturers with largest pooled MAE

| Manufacturer | Rows | MAE (lakhs) | Median AE (lakhs) | Signed error (lakhs) |
|---|---:|---:|---:|---:|
| Volvo | 30 | 19.669 | 24.389 | -19.103 |
| MINI | 34 | 19.473 | 13.045 | -19.473 |
| Mercedes-Benz | 221 | 12.190 | 7.676 | -6.902 |
| Land Rover | 24 | 11.679 | 2.937 | -10.936 |
| Jaguar | 45 | 11.622 | 8.897 | -10.260 |
| Ford | 80 | 10.238 | 6.510 | -7.466 |
| BMW | 114 | 7.662 | 4.708 | -2.687 |
| MG | 33 | 6.274 | 6.567 | -6.272 |
| Audi | 126 | 5.957 | 3.357 | -4.541 |
| Chevrolet | 14 | 4.723 | 3.332 | 4.723 |

### Model families with largest pooled MAE

Model-family diagnostics include only groups with at least 10 pooled test rows.

| Model family | Rows | MAE (lakhs) | Median AE (lakhs) | Signed error (lakhs) |
|---|---:|---:|---:|---:|
| cooper | 19 | 24.206 | 13.372 | -24.206 |
| s-class | 25 | 21.196 | 10.991 | -21.128 |
| xc60 | 10 | 19.338 | 26.286 | -19.338 |
| gle | 16 | 18.307 | 14.709 | -18.307 |
| 7 | 13 | 16.514 | 11.748 | -16.514 |
| f-pace | 13 | 14.897 | 16.171 | -14.897 |
| countryman | 15 | 13.478 | 11.953 | -13.478 |
| endeavour | 41 | 12.286 | 12.556 | -12.286 |
| q7 | 14 | 9.832 | 5.146 | -9.378 |
| gla | 11 | 9.082 | 8.325 | 9.082 |

### Price-band bias

| Actual-price band | Rows | Mean signed error (lakhs) |
|---|---:|---:|
| (0.624, 5.75] | 914 | 1.647 |
| (5.75, 12.5] | 907 | 0.375 |
| (12.5, 147.0] | 896 | -8.360 |

## Interpretation

- Seed variability must be judged from the saved mean, standard deviation, minimum, and maximum rather than one split.
- Coverage and interval width are reported separately for known-family and unseen-family protocols; unseen-family intervals are not automatically valid for deployment.
- Low-sample manufacturer and model-family groups are excluded from diagnostics and should not support subgroup claims.
- This reliability stage does not modify or promote the current candidate model.
