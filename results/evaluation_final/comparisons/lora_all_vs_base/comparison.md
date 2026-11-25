# Baseline / adapter comparison

Baseline: base. Adapter: lora_all.

Rates are percentages; changes and confidence intervals are percentage points (adapter minus baseline).

| Dataset | Metric | N | Groups | Baseline (%) | Adapter (%) | Change (pp) | CI for change (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| whatsup | accuracy | 408 | 102 | 50.00 | 100.00 | +50.00 | 95% [+49.26, +50.74] |
| whatsup | lateral_accuracy | 204 | 102 | 99.51 | 100.00 | +0.49 | 95% [+0.00, +1.47] |
| whatsup | depth_accuracy | 204 | 102 | 0.49 | 100.00 | +99.51 | 95% [+98.53, +100.00] |
| vsr/zeroshot/test | accuracy | 1222 | 715 | 81.83 | 80.44 | -1.39 | 95% [-3.57, +0.74] |
| pope/random | accuracy | 3000 | 500 | 88.27 | 90.67 | +2.40 | 95% [+1.80, +3.03] |
| pope/random | precision | 3000 | 500 | 98.81 | 98.11 | -0.70 | 95% [-1.22, -0.24] |
| pope/random | recall | 3000 | 500 | 77.47 | 82.93 | +5.47 | 95% [+4.27, +6.67] |
| pope/random | f1 | 3000 | 500 | 86.85 | 89.88 | +3.04 | 95% [+2.30, +3.83] |
| pope/random | yes_ratio | 3000 | 500 | 39.20 | 42.27 | +3.07 | 95% [+2.43, +3.70] |

Paired percentile bootstrap resamples image groups (VSR/POPE) or object-pair groups (What'sUp).
Each draw recomputes metrics from all questions in the sampled groups. JSON records resample counts and seeds.
Intervals describe evaluation-group sampling, not variation across training runs.
An absent interval means fewer than two contributing groups or fewer than two valid resamples.
See JSON for any recorded environment differences between the runs.
