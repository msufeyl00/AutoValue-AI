# Dataset B improvement report

Selected by validation-only ranking: **interaction_features**.
Historical five-seed test results were not used as tuning feedback.
All reported test metrics reuse the established grouped exploratory splits and require a future locked evaluation.

## Candidate comparison

| Candidate | Protocol | Validation MAE mean | Test MAE mean | Test RMSE mean | High-price MAE mean | High-price bias mean | Coverage mean | Width mean | Baseline wins |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| current_baseline | known_model_families | 5.486 | 4.579 | 10.753 | 9.394 | -7.618 | 0.922 | 19.479 | 5/5 |
| current_baseline | unseen_model_families | 7.076 | 4.834 | 10.640 | 13.145 | -10.755 | 0.916 | 24.830 | 5/5 |
| full_model_name | known_model_families | 5.485 | 4.567 | 10.752 | 9.362 | -7.626 | 0.926 | 19.722 | 5/5 |
| full_model_name | unseen_model_families | 7.130 | 4.856 | 10.662 | 13.210 | -10.854 | 0.914 | 24.699 | 5/5 |
| interaction_features | known_model_families | 5.470 | 4.560 | 10.742 | 9.344 | -7.632 | 0.925 | 19.741 | 5/5 |
| interaction_features | unseen_model_families | 7.061 | 4.816 | 10.596 | 13.059 | -10.714 | 0.919 | 24.726 | 5/5 |
| log_price_tuned | known_model_families | 6.334 | 5.528 | 11.796 | 11.417 | -10.594 | 0.919 | 24.457 | 5/5 |
| log_price_tuned | unseen_model_families | 7.554 | 5.172 | 11.359 | 14.003 | -12.970 | 0.928 | 28.144 | 5/5 |
| pseudo_huber | known_model_families | 5.375 | 4.525 | 10.909 | 9.550 | -6.997 | 0.925 | 20.288 | 5/5 |
| pseudo_huber | unseen_model_families | 7.399 | 4.713 | 10.627 | 13.180 | -10.730 | 0.921 | 26.088 | 5/5 |
| raw_price | known_model_families | 5.631 | 5.188 | 10.613 | 8.841 | -4.512 | 0.894 | 20.133 | 5/5 |
| raw_price | unseen_model_families | 8.157 | 5.194 | 10.156 | 11.551 | -7.416 | 0.905 | 25.992 | 5/5 |

## Limitations

- Expensive vehicles remain difficult because premium brands and families have sparse, heterogeneous samples.
- Conformal intervals are marginal empirical intervals for these grouped protocols; they do not guarantee individual coverage.
- Unseen-family performance and interval widths are especially sensitive to which families are held out.
- A new independent dataset or locked future evaluation is required before deployment claims.
