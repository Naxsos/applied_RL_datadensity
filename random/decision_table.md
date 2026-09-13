# Decision Table

| method | zone_visit_rate | true_return_mean | robustness_score | fits_local | cost_vs_forecast_err_corr | wall_clock_train_s | weighted_total |
|---|---|---|---|---|---|---|---|
| lagr | 0.878 | -146.597 | 0.999 | 1.000 | nan | 984.513 | 0.594 |
| baseline | 0.765 | -400.791 | 0.000 | 1.000 | nan | 1030.508 | 0.439 |
| ens | 0.886 | -411.023 | 0.876 | 1.000 | 0.409 | 807.683 | 0.355 |
| bnn | 0.882 | -415.465 | 0.928 | 1.000 | 0.054 | 897.767 | 0.271 |

**Winner (highest weighted_total): `lagr`** — carry into the full benchmark vs the paper baseline.