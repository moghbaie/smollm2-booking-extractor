# Colab Sprint Plan: SmolLM2-360M JSON Extraction with Soup

Nov 7, 2026 · @Mehrnoosh

## Fastest timeline: 2 days

On Colab this project takes about **2 days: 12–20 hours of your time and only 2–4 GPU hours.** The model is small (360M parameters) and the training set is 2,000 short examples, so each training run takes minutes on an A100 or H100. Your own time goes into reading results and making decisions. The data generator, evaluator, tests and notebook are already written, which removes the slowest 20–30 hours from the earlier estimate.

Everything runs from one notebook, `soup_sprint_colab.ipynb`. It writes every script to your Google Drive and has a check cell at the end of each phase.

| When | Phases | Your time | GPU time |
| --- | --- | --- | --- |
| Day 1, morning | 0 Setup · 1 Data | 1.5–3 h | \~15 min |
| Day 1, midday | 2 Baseline | 1–2 h | \~10 min |
| Day 1, afternoon | 3 Smoke run · 4 Full SFT | 2–3 h | \~30 min |
| Day 2, morning | 5 Hyperparameter study (9 runs) | 2–4 h, mostly waiting | 1–1.5 h |
| Day 2, afternoon | 6 Final test, general-skills check, export | 2–3 h | 0.5–1 h |
| Day 2, evening | Write-up and results log | 2–3 h | — |
| **Total** |  | **≈ 12–20 h** | **≈ 2–4 h** |

These GPU times are my estimates from the data size; Phase 3 measures the real speed. The absolute fastest is a single 8–10 hour day, if nothing breaks. Expect setup problems or a data fix to add half a day.

## Which GPU and plan

Use **Colab Pro ($9.99/month) with an A100**. For a 360M model that should be nearly as fast as an H100 at under a third of the units per hour. The strongest GPUs, the H100 and the G4 (RTX PRO 6000, 96 GB), need Pro+ at $49.99 a month: Google said in August 2026 that both are Pro+ only "for the time being" ([GitHub issue](https://github.com/googlecolab/colabtools/issues/6076)).

| GPU | Plan needed | Units per hour | Whole project (2–4 GPU h) | Notes |
| --- | --- | --- | --- | --- |
| H100 80 GB | Pro+ | \~18 (one measurement) | \~36–72 units | Fastest on paper; a 360M model can't keep it busy |
| G4 RTX PRO 6000 96 GB | Pro+ (for now) | 8.71 | \~17–35 units | Newest architecture (Blackwell), bf16 |
| **A100 40/80 GB** | **Pro** | **5.4** | **\~11–22 units** | **Recommended.** bf16; pick 80 GB with the High-RAM switch |
| L4 24 GB | Pro | 1.71 | \~4–7 units | About 2× slower than an A100; still under 1 day of GPU time |

Unit rates are independent measurements, since Google doesn't publish them ([McCormick, March 2026](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/); [Fastero, 2026](https://fastero.com/blog/google-colab-pro)). Check the live rate in Colab's Resources panel. Colab Pro's 100 units cover this project several times over, and unused units last 90 days.

**Why a bigger GPU barely helps here:** each training step on a 360M model with 16 short examples finishes too fast to use an H100's capacity, so loading, tokenising and saving dominate. My estimate is that the H100 would save around 15–30 minutes over the whole project compared with an A100. Choose the H100 only if you already have Pro+, or later move up to SmolLM2-1.7B.

## Phase 0 — Colab setup (≈ 30 min)

Goal: a GPU runtime with Soup installed and everything saved to Google Drive, so a disconnect costs nothing.

1. Open [colab.research.google.com](https://colab.research.google.com), then File → Upload notebook → `soup_sprint_colab.ipynb`.
2. Runtime → Change runtime type → **A100 GPU** (or H100 on Pro+), High-RAM on.
3. Run the Phase 0 cells. They mount Drive, create `MyDrive/soup-sprint/`, install `soup-cli[train,eval]`, and run `soup doctor`.

**Checks**

| # | Check | Pass when |
| --- | --- | --- |
| 0.1 | GPU | `nvidia-smi` shows an A100/H100/L4; the check cell confirms bf16 support |
| 0.2 | Soup | `soup --version` prints a version (written against v0.77.2); `soup doctor` shows no errors |
| 0.3 | Drive | `MyDrive/soup-sprint/` exists and `reports/pip_freeze.txt` was written |

In Colab, `soup train` needs `--yes`, because notebook cells have no interactive terminal to confirm in. The notebook already adds it.

## Phase 1 — Generate and check the data (≈ 1–2 h)

Goal: a training set and fixed evaluation sets with labels computed by rules, and no overlap between them.

**The task.** Turn a booking request into JSON:

```json
{"intent": "reschedule", "date": "2026-12-07", "time": "18:00", "resource": null, "needs_clarification": false}
```

- `intent` is one of `book`, `reschedule`, `cancel`, `check`.
- `null` means the request doesn't state that value.
- `needs_clarification` is `true` when a value the intent needs is missing. Required fields: book needs date, time and resource; reschedule needs date and time; cancel needs date; check needs date and resource.
- Each prompt includes a "today" date, so relative dates like "tomorrow" and "in 5 days" have one correct answer.

**Where the data comes from:**

| File | Rows | Source | Used for |
| --- | --- | --- | --- |
| `data/train.jsonl` | 2,000 | `scripts/generate_data.py`, from 24 training templates | Training (Soup holds back 10% for validation loss) |
| `data/eval/val.jsonl` | 300 | 8 *different* templates | Choosing hyperparameters |
| `data/eval/test.jsonl` | 300 | 8 more unseen templates | Final score, run once |
| `data/eval/stress.jsonl` | 150 | Test templates plus noise: capitals, typos, filler, unseen date/time/room wording | Robustness check |
| `data/replay.jsonl` | 300 | [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk), `smol-magpie-ultra-short` rows | Mixed in at 10% to reduce forgetting; never scored |

The split is by **template family**, so validation and test contain sentence patterns the model never saw in training. Every label comes from rules. The replay rows are general chat that SmolLM2 was originally trained on. They're there to keep its general ability, not as something to learn.

**Checks** (the notebook runs `pytest` and `scripts/check_data.py`)

| # | Check | Pass when |
| --- | --- | --- |
| 1.1 | Script tests | 14 tests pass (evaluator edge cases and replay filter) |
| 1.2 | Structure | Every training row is user → assistant with a non-empty answer |
| 1.3 | Labels | All 2,750 targets pass the JSON schema |
| 1.4 | No leakage | 0 shared prompts between any two splits; template families disjoint |
| 1.5 | Coverage | All 4 intents in every split; missing-field share 10–60% (generated: train 28%, val 29%, test 52%, stress 47%) |
| 1.6 | Checksums | Files match the SHA-256 values in `data/manifest.json` |
| 1.7 | Soup format | `soup data validate` reports 2,000/2,000 rows valid (chatml), and the replay file is valid too |
| 1.8 | Your eyes | Read the 10 random rows the check cell prints; each label should be obviously right |

## Phase 2 — Evaluator and baseline (≈ 1–2 h)

Goal: score the untouched SmolLM2-360M-Instruct on the validation set, so every later number has something to beat.

`soup infer` generates the answers with greedy decoding (`--temperature 0`), and `scripts/evaluate.py` scores them. Soup's built-in `eval custom` only does exact, regex or contains matching, which can't check a JSON schema, so the project uses its own scorer.

| Metric | Meaning |
| --- | --- |
| **exact\_match** (primary) | Valid against the schema AND all 5 fields correct, with a 95% bootstrap confidence interval |
| schema\_valid | Exactly the 5 keys, with correct types and formats |
| parse\_strict | The reply is only a JSON object, with no other text |
| parse\_lenient | A JSON object can be found somewhere in the reply |
| field\_\* | Accuracy of each field on its own |
| by group | exact\_match per intent, and for complete vs missing-field requests |

With `--pred-b` the scorer also gives a **paired** difference between two models on the same prompts, with its own confidence interval. Every failed row goes to `reports/*.errors.jsonl` for reading.

**Checks**

| # | Check | Pass when |
| --- | --- | --- |
| 2.1 | Baseline scored | `reports/base_val.json` exists; numbers copied to the Results log |
| 2.2 | Repeatable | The rerun gives exactly the same exact\_match (greedy decoding) |
| 2.3 | Plausible | parse\_lenient > 0 (the model answers something). A low exact\_match is expected; small instruct models rarely produce exact JSON unprompted |
| 2.4 | Read failures | You can name the top 2–3 failure types from the errors file (extra text, wrong date arithmetic, `"null"` as text, …) |

## Phase 3 — Smoke run (≈ 30 min)

Goal: prove that training, saving and inference work, using 200 rows and 1 epoch, before the real run.

The notebook writes `configs/smoke.yaml` from the main config: 200 rows, 1 epoch, no replay. It runs `soup train --dry-run` first, which checks the config and data without training, then the real run. Then it scores the result on the validation set.

**Checks**

| # | Check | Pass when |
| --- | --- | --- |
| 3.1 | Dry run | `--dry-run` passes |
| 3.2 | Loss | Finite (no NaN), clearly lower at the end than the start |
| 3.3 | Adapter | `runs/smoke/` exists and `soup infer` loads it |
| 3.4 | Learning signal | parse\_strict well above the baseline after only 200 examples |
| 3.5 | Speed | Note the seconds per step; multiply out to check the Phase 4 time estimate |

## Phase 4 — Full SFT run (≈ 1 h, \~10 min of GPU)

Goal: the first complete fine-tuned model, compared against the baseline on the validation set.

```yaml
# configs/sft.yaml (checked against Soup v0.77.2's config schema)
base: HuggingFaceTB/SmolLM2-360M-Instruct
task: sft
data:
  train: ./data/train.jsonl
  format: chatml
  val_split: 0.1
  max_length: 512
  replay: ./data/replay.jsonl     # general chat mixed in to limit forgetting
  replay_ratio: 0.1
training:
  epochs: 2
  lr: 1.0e-4
  batch_size: 16
  gradient_accumulation_steps: 1
  seed: 42
  eval_steps: 25
  scheduler: cosine
  warmup_ratio: 0.03
  quantization: none              # Soup defaults to 4-bit; not needed for 360M
  auto_mixed_precision: true      # bf16 on A100 / H100 / L4
  lora: {r: 16, alpha: 32, dropout: 0.05}
output: ./runs/sft_v1
```

Some of these settings differ from Soup's defaults on purpose:

- `quantization: none`, because Soup defaults to 4-bit and a 360M model doesn't need it.
- `train_on_responses_only` is left at its default (on), so the loss covers only the JSON answer, not the long instruction.
- `eval_steps: 25` logs validation loss often enough to see overfitting.

**Checks**

| # | Check | Pass when |
| --- | --- | --- |
| 4.1 | Curves (TensorBoard) | Training loss falls; validation loss flattens. If validation loss rises in epoch 2, note it for Phase 5 |
| 4.2 | Validation exact\_match | Clearly above the baseline, with the paired CI95 lower bound above 0 |
| 4.3 | Format | parse\_strict ≥ 95% |
| 4.4 | Missing-field group | Not far below the complete group (if it is, the model is guessing values instead of using null) |
| 4.5 | Read 20 errors | Each failure has a nameable cause |

## Phase 5 — Hyperparameter study (≈ 2–4 h, \~1–1.5 h of GPU)

Goal: see the lesson's effects for yourself (learning rate, epochs, capacity) and choose a final model using the **validation set only**.

The study changes one factor at a time. That's 9 runs, each taking minutes:

| Round | Vary | Values | Kept fixed |
| --- | --- | --- | --- |
| 1 | Learning rate | 5e-5 (the course's AdamW starting point), 1e-4, 2e-4 | 2 epochs, rank 16 |
| 2 | Epochs | 1, 3 (2 already ran in round 1) | best lr, rank 16 |
| 3 | LoRA rank | 8, 32 (alpha = 2 × rank) | best lr and epochs |
| 4 | Seed | 7, 123 (42 already ran) | best of everything; measures run-to-run noise |

The notebook gives each run its own config and output folder through a small `run()` helper. It avoids `soup sweep` here because in Soup 0.77.2 sweep runs all share one output folder. If Colab disconnects, re-running a round skips any run whose adapter is already on Drive. Results collect in `reports/hparam_study.csv`.

**Checks**

| # | Check | Pass when |
| --- | --- | --- |
| 5.1 | All runs scored | 9 rows in the study table, each with exact\_match and its CI95 |
| 5.2 | Learning rate behaves as the lesson says | 2e-4 learns fastest but may be noisier; 5e-5 may underfit in 2 epochs |
| 5.3 | Epochs | Validation loss is lowest near the best epoch count; 3 epochs don't help if validation loss was already rising |
| 5.4 | Noise | The 3 seeds agree within a few points. Differences in rounds 1–3 smaller than that are ties: pick the cheaper setting |
| 5.5 | Frozen choice | `BEST_RUN` is written down **before** any test-set run |

## Phase 6 — Final test, regression check, export (≈ 2–3 h)

Goal: one honest final score, a check that general skills survived, and a model you can run locally.

1. **Test and stress sets, run once.** Generate answers on `test` and `stress` with both the chosen model and the base model, then score them as a pair: `--pred-b` gives the paired difference and its CI95.
2. **General-skills regression.** Run `soup eval benchmark --benchmarks hellaswag,ifeval` (lm-evaluation-harness) on the base model and on the tuned adapter. IFEval measures instruction following, the skill most at risk; HellaSwag is a quick general check. Don't measure regression with SmolTalk prompts, because the base model was trained on them.
3. **Export (optional).** Merge the adapter and export a GGUF file with `soup merge` and `soup export --format gguf --quant q4_k_m`. A 360M GGUF runs easily on your GTX 1650 laptop, or even without a GPU.
4. **Stop paying.** Runtime → Disconnect and delete runtime.

**Acceptance checks (set before you see the test results)**

| # | Check | Pass when |
| --- | --- | --- |
| 6.1 | Task win | Test exact\_match beats the base model by ≥ 10 points, and the paired CI95 lies entirely above 0 |
| 6.2 | Format | parse\_strict ≥ 95% on test |
| 6.3 | Robustness | Stress-set exact\_match within \~15 points of test (bigger gap = memorised phrasings) |
| 6.4 | No forgetting | IFEval and HellaSwag each drop by ≤ 5 points from the base model |
| 6.5 | Honest process | Test was run once, after `BEST_RUN` was frozen |

If 6.1 fails, read the errors file before changing anything. Most fixes are in the data, such as missing phrasings or ambiguous labels, not in the hyperparameters.

## Results log

One row per evaluated model. All scores are % on the **validation** set, except the last two rows.

| Run | lr | Epochs | Rank | Seed | exact\_match (CI95) | schema\_valid | parse\_strict | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base model | — | — | — | — |  |  |  |  |
| smoke | 1e-4 | 1 | 16 | 42 |  |  |  | 200 rows |
| sft\_v1 | 1e-4 | 2 | 16 | 42 |  |  |  |  |
| lr\_5e-05 | 5e-5 | 2 | 16 | 42 |  |  |  |  |
| lr\_0.0002 | 2e-4 | 2 | 16 | 42 |  |  |  |  |
| ep\_1 | best | 1 | 16 | 42 |  |  |  |  |
| ep\_3 | best | 3 | 16 | 42 |  |  |  |  |
| r\_8 | best | best | 8 | 42 |  |  |  |  |
| r\_32 | best | best | 32 | 42 |  |  |  |  |
| best\_seed7 / best\_seed123 | best | best | best | 7 / 123 |  |  |  | noise check |
| **Final: test** (vs base) |  |  |  |  |  |  |  | paired diff CI95: |
| **Final: stress** (vs base) |  |  |  |  |  |  |  | IFEval / HellaSwag Δ: |

## What was tested before handing this over

- **Tested here:** I ran the data generator and all 23 data checks, which pass. The 14 unit tests for the evaluator and replay filter pass. I ran the evaluator end to end on simulated outputs, including the paired comparison. Every config the notebook writes (smoke, the learning-rate, epoch and rank runs, the seeds) passes Soup v0.77.2's own config schema. `soup data validate` accepts the training file. Every notebook cell compiles.
- **Not tested (needs Colab):** GPU training, `soup infer` speed, the replay download (Hugging Face isn't reachable from my workspace), `soup eval benchmark`, and the adapter folder name (`adapter_config.json`) that the skip-if-done logic relies on.

## Sources

- [Soup repository](https://github.com/MakazhanAlpamys/Soup) and its CLI help, v0.77.2: commands, config schema, `infer` input format
- [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
- [smol-smoltalk dataset](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk): replay data; the `source` column and its values
- [Colab H100/G4 availability (GitHub issue #6076)](https://github.com/googlecolab/colabtools/issues/6076)
- [Colab GPU unit rates, Chris McCormick (March 2026)](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/) and [Fastero (2026)](https://fastero.com/blog/google-colab-pro)
- [DeepLearning.AI lesson: hyperparameters (part 1)](<https://learn.deeplearning.ai/courses/fine-tuning-and-reinforcement-learning-for-llms-intro-to-post-training/lesson/cnd19h/fine-tuning%3A-hyperparameters-%26-hyperparameter-tuning-(part-1)>)
