# Baseline / adapter comparison

Baseline: base. Adapter: lora_random.

Rates are percentages; changes and confidence intervals are percentage points (adapter minus baseline).

| Dataset | Metric | N | Groups | Baseline (%) | Adapter (%) | Change (pp) | CI for change (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| whatsup | accuracy | 408 | 102 | 50.00 | 100.00 | +50.00 | 95% [+49.26, +50.74] |
| whatsup | lateral_accuracy | 204 | 102 | 99.51 | 100.00 | +0.49 | 95% [+0.00, +1.47] |
| whatsup | depth_accuracy | 204 | 102 | 0.49 | 100.00 | +99.51 | 95% [+98.53, +100.00] |
| vsr/zeroshot/test | accuracy | 1222 | 715 | 81.83 | 82.32 | +0.49 | 95% [-1.52, +2.46] |
| pope/random | accuracy | 3000 | 500 | 88.27 | 89.87 | +1.60 | 95% [+1.07, +2.13] |
| pope/random | precision | 3000 | 500 | 98.81 | 98.23 | -0.58 | 95% [-1.05, -0.18] |
| pope/random | recall | 3000 | 500 | 77.47 | 81.20 | +3.73 | 95% [+2.73, +4.80] |
| pope/random | f1 | 3000 | 500 | 86.85 | 88.91 | +2.06 | 95% [+1.43, +2.73] |
| pope/random | yes_ratio | 3000 | 500 | 39.20 | 41.33 | +2.13 | 95% [+1.60, +2.67] |

Paired percentile bootstrap resamples image groups (VSR/POPE) or object-pair groups (What'sUp).
Each draw recomputes metrics from all questions in the sampled groups. JSON records resample counts and seeds.
Intervals describe evaluation-group sampling, not variation across training runs.
An absent interval means fewer than two contributing groups or fewer than two valid resamples.
See JSON for any recorded environment differences between the runs.
