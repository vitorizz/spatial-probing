# Baseline / adapter comparison

Baseline: base. Adapter: lora_targeted.

Rates are percentages; changes and confidence intervals are percentage points (adapter minus baseline).

| Dataset | Metric | N | Groups | Baseline (%) | Adapter (%) | Change (pp) | CI for change (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| whatsup | accuracy | 408 | 102 | 50.00 | 93.38 | +43.38 | 95% [+41.18, +45.59] |
| whatsup | lateral_accuracy | 204 | 102 | 99.51 | 100.00 | +0.49 | 95% [+0.00, +1.47] |
| whatsup | depth_accuracy | 204 | 102 | 0.49 | 86.76 | +86.27 | 95% [+81.86, +90.69] |
| vsr/zeroshot/test | accuracy | 1222 | 715 | 81.83 | 82.08 | +0.25 | 95% [-1.61, +1.96] |
| pope/random | accuracy | 3000 | 500 | 88.27 | 90.97 | +2.70 | 95% [+2.00, +3.37] |
| pope/random | precision | 3000 | 500 | 98.81 | 97.90 | -0.91 | 95% [-1.48, -0.42] |
| pope/random | recall | 3000 | 500 | 77.47 | 83.73 | +6.27 | 95% [+5.00, +7.47] |
| pope/random | f1 | 3000 | 500 | 86.85 | 90.26 | +3.42 | 95% [+2.60, +4.24] |
| pope/random | yes_ratio | 3000 | 500 | 39.20 | 42.77 | +3.57 | 95% [+2.93, +4.20] |

Paired percentile bootstrap resamples image groups (VSR/POPE) or object-pair groups (What'sUp).
Each draw recomputes metrics from all questions in the sampled groups. JSON records resample counts and seeds.
Intervals describe evaluation-group sampling, not variation across training runs.
An absent interval means fewer than two contributing groups or fewer than two valid resamples.
See JSON for any recorded environment differences between the runs.
