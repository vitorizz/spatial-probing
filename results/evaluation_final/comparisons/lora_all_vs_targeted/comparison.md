# Baseline / adapter comparison

Baseline: lora_targeted. Adapter: lora_all.

Rates are percentages; changes and confidence intervals are percentage points (adapter minus baseline).

| Dataset | Metric | N | Groups | Baseline (%) | Adapter (%) | Change (pp) | CI for change (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| whatsup | accuracy | 408 | 102 | 93.38 | 100.00 | +6.62 | 95% [+4.41, +8.83] |
| whatsup | lateral_accuracy | 204 | 102 | 100.00 | 100.00 | +0.00 | 95% [+0.00, +0.00] |
| whatsup | depth_accuracy | 204 | 102 | 86.76 | 100.00 | +13.24 | 95% [+8.82, +17.66] |
| vsr/zeroshot/test | accuracy | 1222 | 715 | 82.08 | 80.44 | -1.64 | 95% [-3.45, +0.25] |
| pope/random | accuracy | 3000 | 500 | 90.97 | 90.67 | -0.30 | 95% [-0.87, +0.23] |
| pope/random | precision | 3000 | 500 | 97.90 | 98.11 | +0.21 | 95% [-0.24, +0.66] |
| pope/random | recall | 3000 | 500 | 83.73 | 82.93 | -0.80 | 95% [-1.80, +0.20] |
| pope/random | f1 | 3000 | 500 | 90.26 | 89.88 | -0.38 | 95% [-1.03, +0.24] |
| pope/random | yes_ratio | 3000 | 500 | 42.77 | 42.27 | -0.50 | 95% [-1.03, +0.07] |

Paired percentile bootstrap resamples image groups (VSR/POPE) or object-pair groups (What'sUp).
Each draw recomputes metrics from all questions in the sampled groups. JSON records resample counts and seeds.
Intervals describe evaluation-group sampling, not variation across training runs.
An absent interval means fewer than two contributing groups or fewer than two valid resamples.
See JSON for any recorded environment differences between the runs.
