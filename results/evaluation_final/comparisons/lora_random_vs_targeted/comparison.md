# Baseline / adapter comparison

Baseline: lora_targeted. Adapter: lora_random.

Rates are percentages; changes and confidence intervals are percentage points (adapter minus baseline).

| Dataset | Metric | N | Groups | Baseline (%) | Adapter (%) | Change (pp) | CI for change (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| whatsup | accuracy | 408 | 102 | 93.38 | 100.00 | +6.62 | 95% [+4.41, +8.83] |
| whatsup | lateral_accuracy | 204 | 102 | 100.00 | 100.00 | +0.00 | 95% [+0.00, +0.00] |
| whatsup | depth_accuracy | 204 | 102 | 86.76 | 100.00 | +13.24 | 95% [+8.82, +17.66] |
| vsr/zeroshot/test | accuracy | 1222 | 715 | 82.08 | 82.32 | +0.25 | 95% [-1.26, +1.71] |
| pope/random | accuracy | 3000 | 500 | 90.97 | 89.87 | -1.10 | 95% [-1.60, -0.63] |
| pope/random | precision | 3000 | 500 | 97.90 | 98.23 | +0.33 | 95% [+0.03, +0.70] |
| pope/random | recall | 3000 | 500 | 83.73 | 81.20 | -2.53 | 95% [-3.53, -1.60] |
| pope/random | f1 | 3000 | 500 | 90.26 | 88.91 | -1.36 | 95% [-1.96, -0.78] |
| pope/random | yes_ratio | 3000 | 500 | 42.77 | 41.33 | -1.43 | 95% [-1.93, -0.93] |

Paired percentile bootstrap resamples image groups (VSR/POPE) or object-pair groups (What'sUp).
Each draw recomputes metrics from all questions in the sampled groups. JSON records resample counts and seeds.
Intervals describe evaluation-group sampling, not variation across training runs.
An absent interval means fewer than two contributing groups or fewer than two valid resamples.
See JSON for any recorded environment differences between the runs.
