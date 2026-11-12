# SmolLM2 booking extractor

LoRA fine-tunes of SmolLM2-Instruct that turn a booking request into JSON:

```
Today is Wednesday, 2026-11-11.
Request: can u move my court 2 slot to next fri at 4:30pm
→ {"intent": "reschedule", "date": "2026-11-13", "time": "16:30", "resource": "Court 2", "needs_clarification": false}
```

`intent` is one of book / reschedule / cancel / check. Anything the request does not state is `null`, and
`needs_clarification` is true when a field the intent needs is missing.

Every run is a Colab notebook with checks at each step (data, training, scoring, general-ability benchmarks),
built by a `build_notebook_*.py` script.

## Results

**Release model: `v3_big`** (SmolLM2-1.7B-Instruct, all-linear LoRA r16, calendar prompt).

Exact match (all five fields right) on the same 300-row sets. *test* uses wording families never seen in
training; *stress* adds held-out wordings and typos on top of that.

| model | trainer | data | test | stress |
|---|---|---|---|---|
| 360M | transformers (`hf_small_v2`) | v2 | 88.67 | 67.67 |
| 1.7B | Soup (q/v LoRA) | v2 | 93.67 | 79.00 |
| 1.7B | transformers (`hf_big_lr1e-4`) | v2 | 96.67 | 80.33 |
| **1.7B** | **transformers (`v3_big`)** | **v3** | **97.00** | **85.00** |

v3 on its own new sets, against v2's model on the same rows (paired bootstrap CI95):

| set | v2 model | v3 model | diff | date field |
|---|---|---|---|---|
| v3 test | 86.33 | 95.00 | +8.67 [5.67, 12.0] | 84.5 → 94.3 |
| v3 stress | 70.67 | 77.00 | +6.33 [2.33, 10.67] | 73.9 → 81.8 |
| v2 test | 96.67 | 97.00 | +0.33 [−1.67, 2.33] | 95.2 → 95.7 |
| v2 stress | 80.33 | 85.00 | +4.67 [0.67, 8.67] | 81.5 → 91.2 |

General ability (lm-eval; IFEval prompt-level strict, HellaSwag acc_norm, ± stderr):

| model | IFEval | HellaSwag |
|---|---|---|
| 360M untouched | 33.09 ± 2.02 | 56.74 ± 0.49 |
| 360M tuned (run 1, `ep_3`) | 31.61 ± 2.00 | 57.04 ± 0.49 |
| 1.7B untouched | 47.13 ± 2.15 | 71.72 ± 0.45 |
| 1.7B tuned, v2 (`hf_big_lr1e-4`) | 49.91 ± 2.15 | 71.53 ± 0.45 |
| 1.7B tuned, v3 (`v3_big`) | 47.69 ± 2.15 | 71.74 ± 0.45 |

## What changed between runs, and what each change did

| run | notebook | change | effect |
|---|---|---|---|
| 1 | `soup_sprint_colab.ipynb` | 360M, Soup, v1 data, small hyperparameter study (`ep_3` won) | test 86.67, but stress only 44.0: it had memorised the training wordings |
| 2 | `soup_sprint_v2_colab.ipynb` | v2 data: more wordings, typos, fillers, mixed case; held-out wording tiers kept for stress only | the 360M stayed weak on unseen wording |
| 3 | `soup_sprint_1p7b_colab.ipynb` | 1.7B, same v2 data, Soup | big jump on stress; IFEval/HellaSwag rise with the base model |
| 4 | `soup_sprint_fallback_colab.ipynb` | own transformers trainer (`scripts/train_hf.py`): LoRA on all linear layers, loss on the answer only | Soup's default LoRA only touches q/v; all-linear adds +3.0 [0.67, 5.33] on test at 1.7B. Size effect with the same trainer: test +8.0 [4.7, 11.3], stress +12.7 [7.7, 17.7] |
| 5 | `soup_sprint_v3_colab.ipynb` | v3 data: 12 new date wordings, date typos and shorthand, and a 21-day calendar in the prompt | dates were most remaining errors; v3 fixes most of them with no loss on v2's sets |

Main findings:

- **Model size mattered most** for unseen wording (stress), and it is also the only thing that moved
  IFEval and HellaSwag (+14 / +15 points from 360M to 1.7B). Fine-tuning itself leaves them unchanged:
  every tuned model is within one standard error of its untouched base.
- **LoRA placement mattered**: all-linear beat q/v-only at equal settings.
- **Dates needed both data and prompt**: a calendar in the prompt turns "next Friday" into a lookup;
  wider date wording and noise teach the model to use it.

## Known limits

- Day counting is still the weak spot: "in two weeks" / "a fortnight" (56–65% on stress) and
  "N days from now" (63%). The model can also produce a date that does not exist (seen once: 2026-02-29).
- Synthetic English data, eight fixed resource names, one time zone-free "today".
- The v3 model needs the calendar prompt (mean 299 tokens vs 154 before, ≈ 8% slower).
- Check the output before acting on it.

## Layout

| path | what |
|---|---|
| `scripts/generate_data*.py` | rule-based data generators (v1, v2, v3) with template-family splits |
| `scripts/check_data*.py` | data checks run in each notebook |
| `scripts/train_hf.py` | transformers + peft trainer (all-linear LoRA, answer-only loss, replay) |
| `scripts/predict.py`, `evaluate.py`, `breakdown.py` | greedy prediction, exact match with bootstrap CI95, per-group and per-form breakdowns |
| `scripts/sprint.py` | notebook helpers: config, train (Soup or hf backend), predict, score, plots |
| `build_notebook*.py` | build each Colab notebook; the `.ipynb` files hold the run outputs |
| `tests/` | pytest suite for generators, scoring, prediction and the trainer |
| `results/` | scores copied from the Colab runs (`fallback/`, `v3/`) |

Run the tests with `pytest tests`.

## Reproduce

Open a notebook in Colab on an A100 (40 GB is enough for the 1.7B; a 22 GB L4 runs out of memory at
batch 16), mount Drive and run all cells. Data is generated from fixed seeds, so it is identical on every run.
