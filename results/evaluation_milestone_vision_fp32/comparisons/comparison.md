# Baseline / adapter comparison

Rates are percentages; changes and confidence intervals are percentage points (adapter minus baseline).

| Dataset | Metric | N | Groups | Baseline (%) | Adapter (%) | Change (pp) | CI for change (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| whatsup | accuracy | 408 | 102 | 50.00 | 80.39 | +30.39 | 95% [+26.72, +34.07] |
| whatsup | lateral_accuracy | 204 | 102 | 99.51 | 100.00 | +0.49 | 95% [+0.00, +1.47] |
| whatsup | depth_accuracy | 204 | 102 | 0.49 | 60.78 | +60.29 | 95% [+52.94, +67.65] |
| vsr/zeroshot/test | accuracy | 1222 | 715 | 81.83 | 82.08 | +0.25 | 95% [-1.49, +1.97] |
| pope/random | accuracy | 3000 | 500 | 88.27 | 90.73 | +2.47 | 95% [+1.83, +3.10] |
| pope/random | precision | 3000 | 500 | 98.81 | 98.19 | -0.62 | 95% [-1.10, -0.21] |
| pope/random | recall | 3000 | 500 | 77.47 | 83.00 | +5.53 | 95% [+4.40, +6.67] |
| pope/random | f1 | 3000 | 500 | 86.85 | 89.96 | +3.11 | 95% [+2.35, +3.87] |
| pope/random | yes_ratio | 3000 | 500 | 39.20 | 42.27 | +3.07 | 95% [+2.50, +3.63] |

Paired percentile bootstrap resamples image groups (VSR/POPE) or object-pair groups (What'sUp).
Each draw recomputes metrics from all questions in the sampled groups. JSON records resample counts and seeds.
Intervals describe evaluation-group sampling, not variation across training runs.
An absent interval means fewer than two contributing groups or fewer than two valid resamples.
See JSON for any recorded environment differences between the runs.
