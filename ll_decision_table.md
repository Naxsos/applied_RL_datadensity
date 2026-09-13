# LL Decision Table

| method | knob | value | n_runs | strict_landing | landing_success | zone_visit | zone_depth | return | crash | timeout | episode_len | wall_clock_s | weighted_total |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | p | 10.000 | 6 | 0.937 | 0.995 | 0.258 | 0.01042 | 260.0 | 0.022 | 0.042 | 232.5 | 2479.4 | 0.935 |
| lagr | epsilon | 0.010 | 6 | 0.844 | 0.979 | 0.253 | 0.00953 | 243.6 | 0.001 | 0.155 | 278.0 | 2565.3 | 0.904 |
| baseline | p | 2.000 | 6 | 0.818 | 0.968 | 0.273 | 0.01051 | 238.3 | 0.026 | 0.157 | 287.4 | 2392.2 | 0.591 |
| baseline | p | 30.000 | 6 | 0.692 | 0.908 | 0.289 | 0.01248 | 201.2 | 0.023 | 0.285 | 337.1 | 2570.6 | 0.010 |

**Best LL setting by weighted_total:** `baseline` with `p=10.000`.