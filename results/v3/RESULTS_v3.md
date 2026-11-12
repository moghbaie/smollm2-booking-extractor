# Results v3 — date fixes (calendar prompt + wider date data)

Both models: SmolLM2-1.7B-Instruct, fallback trainer, lr 1e-4, 3 epochs, LoRA r16, seed 42. v2_big = hf_big_lr1e-4.

| set | v2_big | v3_big | paired diff [CI95] | dates v2 → v3 [CI95 of diff] |
|---|---|---|---|---|
| test | 86.33% | 95.0% | +8.67 [5.67, 12.0] | 84.53% → 94.34% (+9.81 [6.04, 13.96]) |
| stress | 70.67% | 77.0% | +6.33 [2.33, 10.67] | 73.86% → 81.82% (+7.95 [3.41, 12.5]) |
| old_test | 96.67% | 97.0% | +0.33 [-1.67, 2.33] | 95.22% → 95.69% (+0.48 [-2.39, 3.35]) |
| old_stress | 80.33% | 85.0% | +4.67 [0.67, 8.67] | 81.46% → 91.22% (+9.76 [4.39, 15.12]) |

| stress group | n | v2 | v3 | diff [CI95] |
|---|---|---|---|---|
| both | 105 | 65.71 | 71.43 | +5.71 [-1.9, 13.33] |
| new_wording | 106 | 66.98 | 75.47 | +8.49 [1.89, 15.09] |
| noise_only | 89 | 80.9 | 85.39 | +4.49 [-2.25, 11.24] |

Prompt length: 154 → 299 tokens (mean).

| model | ifeval | hellaswag |
|---|---|---|
| 1.7B untouched | 47.13 ± 2.15 | 71.72 ± 0.45 |
| v2_big (hf_big_lr1e-4) | 49.91 ± 2.15 | 71.53 ± 0.45 |
| v3_big | 47.69 ± 2.15 | 71.74 ± 0.45 |

**Release:** publish `v3_big` (it needs the calendar prompt)