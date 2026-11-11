"""Build soup_sprint_colab.ipynb from the scripts/ and tests/ in this folder.

Each script is embedded as a %%writefile cell, so the notebook alone recreates the project on Google Drive.
Usage: python build_notebook.py   (re-run after editing any script)
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = [
    "scripts/common.py", "scripts/generate_data.py", "scripts/fetch_replay.py", "scripts/check_data.py",
    "scripts/evaluate.py", "scripts/predict.py", "scripts/train_hf.py", "scripts/sprint.py",
    "tests/test_data.py", "tests/test_replay.py", "tests/test_evaluate.py", "tests/test_predict.py",
    "tests/test_sprint.py", "tests/test_train_hf.py",
]

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


# ===================================================================== intro
md("""
# Soup sprint: SmolLM2-360M → booking-request JSON

Fine-tune `HuggingFaceTB/SmolLM2-360M-Instruct` with **Soup** (LoRA SFT) to turn booking requests into JSON:

```json
{"intent": "reschedule", "date": "2026-12-07", "time": "18:00", "resource": null, "needs_clarification": false}
```

**How to use:** Runtime → Change runtime type → **A100** (High-RAM on). Then run the cells top to bottom.
Everything is written to `MyDrive/soup-sprint/`, so after a disconnect you re-run Phase 0's cells and continue:
finished trainings and predictions are skipped automatically.

**Checks:** every phase ends with a check cell. Each check prints ✅ PASS, ❌ FAIL or 👀 REVIEW (needs your eyes) and
is saved to `reports/checks.json`. Hard checks stop the notebook so nothing builds on a broken step.

| Phase | What | Checks |
|---|---|---|
| 0 | Colab setup, install Soup, write scripts to Drive | 0.1–0.3 |
| 1 | Generate data, fetch replay, tests, data checks | 1.1–1.8 |
| 2 | Baseline: untouched model on val | 2.1–2.4 |
| 3 | Smoke run: 200 rows, 1 epoch | 3.1–3.5 |
| 4 | Full SFT run `sft_v1` | 4.1–4.5 |
| 5 | Hyperparameter study (9 runs), freeze `BEST_RUN` | 5.1–5.5 |
| 6 | Test + stress (once), general-skills check, export | 6.1–6.5 |

**Where this notebook differs from the original plan** (Soup 0.77.2 docs do not cover these):
`soup infer` → `scripts/predict.py` (greedy, loads the LoRA adapter);
`soup eval benchmark` → `lm-eval` directly;
`soup train --yes/--dry-run` → flags are used only if `soup train --help` lists them, else `soup plan`;
adapter folder → found by searching for `adapter_config.json`.
Check 3.4 uses exact_match/schema_valid instead of parse_strict, because the base model already replies with bare JSON.
""")

# ===================================================================== phase 0
md("""
## Phase 0 — Colab setup (≈ 30 min)

**Steps:** 1) confirm the GPU, 2) mount Drive and create `MyDrive/soup-sprint/`, 3) install `soup-cli[train,eval]`,
4) write all scripts and tests to Drive, 5) run the check cell.

| # | Check | Pass when |
|---|---|---|
| 0.1 | GPU | A GPU is visible and supports bf16 (A100 / H100 / L4) |
| 0.2 | Soup | `soup --version` prints a version (written against 0.77.2); `soup doctor` exits 0 |
| 0.3 | Drive | `MyDrive/soup-sprint/` exists and `reports/pip_freeze.txt` was written |
""")
code("""
!nvidia-smi
import sys; print("python", sys.version)
""")
code("""
import os
from google.colab import drive
drive.mount("/content/drive")

PROJECT = "/content/drive/MyDrive/soup-sprint"
for d in ["scripts", "tests", "data/eval", "configs", "runs", "reports/preds", "logs", "exports"]:
    os.makedirs(os.path.join(PROJECT, d), exist_ok=True)
os.chdir(PROJECT)
print("working in", os.getcwd())
""")
code("""
# ~2-4 min. soup-cli needs Python >= 3.11
!pip -q install "soup-cli[train,eval]" pyyaml pytest tabulate
# Colab preinstalls torchao 0.10, which recent peft refuses to load (needs > 0.16).
# Neither Soup nor this project uses torchao (no quantization_aware training), so remove it.
!pip uninstall -y -q torchao
!soup --version
""")
md("""
### Write the project files to Drive

Each cell below writes one file. They are the same scripts that were tested locally
(35 unit tests, data checks, and an end-to-end CPU run of train → predict → score).
""")
for rel in FILES:
    with open(os.path.join(HERE, rel), encoding="utf-8") as f:
        body = f.read()
    code(f"%%writefile {rel}\n{body}")

code("""
# Load the helpers (re-run this cell after any reconnect)
import importlib, json, os, sys, subprocess
os.chdir(PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "scripts"))
import sprint; importlib.reload(sprint)
from sprint import check
""")
code("""
# ---- Phase 0 checks
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
bf16 = bool(gpu) and torch.cuda.is_bf16_supported()
check("0.1", bool(gpu) and bf16, f"gpu={gpu}, bf16={bf16}", hard=True)
if gpu and not any(k in gpu for k in ("A100", "H100", "L4", "RTX PRO", "G4")):
    print("note: not the GPU the plan assumes; timings will differ")

rc, ver = sprint.sh(["soup", "--version"], check_rc=False)
rc_doc, doc = sprint.sh(["soup", "doctor"], check_rc=False)
check("0.2", rc == 0 and rc_doc == 0,
      f"soup --version -> {ver.strip()[-40:]!r} (plan: 0.77.2); soup doctor exit {rc_doc}")
if "0.77.2" not in ver:
    print("note: Soup version differs from 0.77.2; flags are detected from --help, so this is usually fine")

freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
open("reports/pip_freeze.txt", "w").write(freeze)
check("0.3", os.path.isdir(PROJECT) and os.path.getsize("reports/pip_freeze.txt") > 0,
      f"{PROJECT} exists; pip_freeze.txt has {len(freeze.splitlines())} packages", hard=True)
""")

# ===================================================================== phase 1
md("""
## Phase 1 — Generate and check the data (≈ 1–2 h)

**Steps:** 1) generate train/val/test/stress with rule-computed labels (split by template family),
2) fetch 300 replay rows from `smol-smoltalk` (`smol-magpie-ultra-short`), 3) run the unit tests,
4) run the data checks, 5) `soup data validate`, 6) read 10 random rows yourself.

| # | Check | Pass when |
|---|---|---|
| 1.1 | Script tests | all `pytest` tests pass |
| 1.2 | Structure | every training row is user → assistant with a non-empty answer |
| 1.3 | Labels | all 2,750 targets pass the JSON schema |
| 1.4 | No leakage | 0 shared prompts between splits; template families disjoint |
| 1.5 | Coverage | all 4 intents in every split; missing-field share 10–60% |
| 1.6 | Checksums | files match `data/manifest.json` |
| 1.7 | Soup format | `soup data validate` accepts train (2,000 rows) and replay |
| 1.8 | Your eyes | the 10 printed rows all have obviously right labels |
""")
code("""
!python scripts/generate_data.py --out data
!python scripts/fetch_replay.py --out data/replay.jsonl --n 300
""")
code("""
# ---- 1.1 unit tests
import re
rc, out = sprint.sh([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], check_rc=False)
m = re.search(r"(\\d+) passed", out)
check("1.1", rc == 0 and m is not None, f"pytest exit {rc}; {out.strip().splitlines()[-1]}", hard=True)
""")
code("""
# ---- 1.2-1.6, replay, and 10 rows for 1.8
rc, out = sprint.sh([sys.executable, "scripts/check_data.py", "--data", "data", "--reports", "reports"], check_rc=False)
for r in json.load(open("reports/data_checks.json")):
    check(r["id"], r["ok"], f"{r['name']}: {r['detail']}")
check("1.x", rc == 0, "all data checks passed" if rc == 0 else "data checks failed — fix before training", hard=True)
""")
code("""
# ---- 1.7 Soup accepts the files
rc_t, out_t = sprint.sh(["soup", "data", "validate", "data/train.jsonl"], check_rc=False)
rc_r, out_r = sprint.sh(["soup", "data", "validate", "data/replay.jsonl"], check_rc=False)
ok = rc_t == 0 and rc_r == 0
check("1.7", ok if not ok else (True if "2000" in out_t or "2,000" in out_t else None),
      f"train exit {rc_t}, replay exit {rc_r}; confirm above that 2000/2000 rows are valid (chatml)", hard=not ok)
""")
code("""
# ---- 1.8 your eyes: read the 10 rows printed by the data-check cell above.
# Set to True once every label is obviously right (False = stop and fix the generator).
LABELS_LOOK_RIGHT = None
check("1.8", LABELS_LOOK_RIGHT, "10 random rows read by eye")
""")

# ===================================================================== phase 2
md("""
## Phase 2 — Evaluator and baseline (≈ 1–2 h)

**Steps:** 1) greedy answers from the untouched model on val (`scripts/predict.py`), 2) score with `scripts/evaluate.py`,
3) rerun and compare, 4) read the failures.

Metrics: **exact_match** (schema-valid and all 5 fields right, with a bootstrap CI95), schema_valid, parse_strict
(reply is only JSON), parse_lenient (JSON found somewhere), per-field accuracy, and exact_match by intent and
complete vs missing-field. Failures go to `reports/<run>_<split>.errors.jsonl` with named error types.

| # | Check | Pass when |
|---|---|---|
| 2.1 | Baseline scored | `reports/base_val.json` exists and is in the Results log |
| 2.2 | Repeatable | a rerun gives identical replies and the same exact_match |
| 2.3 | Plausible | parse_lenient > 0 |
| 2.4 | Read failures | you can name the top 2–3 failure types |
""")
code("""
sprint.predict("base", "val")
base_val = sprint.score("base", "val")
sprint.log_result("base", "val", base_val, notes="untouched SmolLM2-360M-Instruct")
""")
code("""
# ---- Phase 2 checks
check("2.1", os.path.exists("reports/base_val.json"), f"exact_match={base_val['metrics']['exact_match']}%")

rerun = sprint.predict("base", "val", force=True, out="reports/preds/base_val_rerun.pred.jsonl")
import evaluate as ev
a, b = ev.load_preds(sprint.pred_path("base", "val")), ev.load_preds(rerun)
same = a == b
em_rerun = ev.evaluate(ev.load_jsonl("data/eval/val.jsonl"), b)[0]["metrics"]["exact_match"]
check("2.2", same and em_rerun == base_val["metrics"]["exact_match"],
      f"identical replies={same}; exact_match {base_val['metrics']['exact_match']} vs {em_rerun}")

check("2.3", base_val["metrics"]["parse_lenient"] > 0, f"parse_lenient={base_val['metrics']['parse_lenient']}%")
sprint.show_errors("base", "val", n=10)
check("2.4", None, f"top error types: {list(base_val['error_types'].items())[:5]} — name the top 2-3")
""")

# ===================================================================== phase 3
md("""
## Phase 3 — Smoke run (≈ 30 min)

**Steps:** 1) write `configs/smoke.yaml` (200 rows, 1 epoch, no replay), 2) `soup doctor --config` and a dry run
(`soup train --dry-run` if it exists, else `soup plan`), 3) train, 4) predict + score on val, 5) measure speed.

If `soup train` fails and `ALLOW_HF_FALLBACK` is on, the run is repeated with `scripts/train_hf.py`
(same config, plain transformers + peft) and **every later run uses the same backend** (`reports/backend.txt`).

| # | Check | Pass when |
|---|---|---|
| 3.1 | Dry run | the dry run / `soup plan` passes |
| 3.2 | Loss | finite, and lower at the end than at the start |
| 3.3 | Adapter | `runs/smoke/` has an adapter and `predict.py` loads it |
| 3.4 | Learning signal | exact_match clearly above base (≥ +10 points) and schema_valid ≥ base |
| 3.5 | Speed | seconds per step measured; Phase 4 time estimated |
""")
code("""
sprint.ALLOW_HF_FALLBACK = True
sprint.set_backend("soup")

with open("data/train.jsonl") as f, open("data/train_smoke.jsonl", "w") as g:
    for i, line in enumerate(f):
        if i < 200:
            g.write(line)
smoke_base = sprint.deep_merge(sprint.SFT_CONFIG, {"data": {"train": "./data/train_smoke.jsonl"}, "training": {"epochs": 1}})
del smoke_base["data"]["replay"], smoke_base["data"]["replay_ratio"]
smoke_cfg_path, smoke_cfg = sprint.make_config("smoke", base_cfg=smoke_base)
print(open(smoke_cfg_path).read())
""")
code("""
# ---- 3.1 config validation
rc_doc, out_doc = sprint.sh(["soup", "doctor", "--config", smoke_cfg_path], check_rc=False)
how, rc_dry, out_dry = sprint.dry_run(smoke_cfg_path)
ok = rc_dry == 0
check("3.1", ok, f"{how} exit {rc_dry}; soup doctor --config exit {rc_doc} (it lists settings Soup ignores)")
if not ok:
    if sprint.ALLOW_HF_FALLBACK:
        print("Soup rejected the config. Read the output above; if it is an unknown key (e.g. auto_mixed_precision),"
              " remove it from sprint.SFT_CONFIG and re-run from the cell above. Otherwise training falls back to train_hf.py.")
    else:
        raise sprint.CheckFailed("dry run failed")
""")
code("""
# ---- train the smoke run (falls back to train_hf.py if soup train fails)
import shutil
try:
    smoke_meta = sprint.train("smoke", smoke_cfg_path, smoke_cfg)
except RuntimeError as e:
    print(e)
    if sprint.get_backend() == "soup" and sprint.ALLOW_HF_FALLBACK:
        print("\\n*** soup train failed -> switching to the transformers+peft fallback for ALL runs ***\\n")
        sprint.set_backend("hf")
        shutil.rmtree("runs/smoke", ignore_errors=True)
        smoke_meta = sprint.train("smoke", smoke_cfg_path, smoke_cfg)
    else:
        raise
print(smoke_meta)
""")
code("""
# ---- Phase 3 checks
import math
h = sprint.loss_history("runs/smoke")
losses = [l for _, l in h["train"]]
k = max(1, len(losses) // 4)
start, end = (sum(losses[:k]) / k, sum(losses[-k:]) / k) if losses else (None, None)
check("3.2", bool(losses) and all(math.isfinite(l) for l in losses) and end < start,
      f"{len(losses)} loss points from {h['source']}; start≈{start}, end≈{end}")

sprint.predict("smoke", "val", adapter_run="smoke")
smoke_val = sprint.score("smoke", "val", vs="base")
check("3.3", sprint.find_adapter("runs/smoke") is not None, f"adapter at {sprint.find_adapter('runs/smoke')}; predict.py loaded it")

bm, sm = base_val["metrics"], smoke_val["metrics"]
check("3.4", sm["exact_match"] >= bm["exact_match"] + 10 and sm["schema_valid"] >= bm["schema_valid"],
      f"exact_match {bm['exact_match']} -> {sm['exact_match']}, schema_valid {bm['schema_valid']} -> {sm['schema_valid']}, "
      f"parse_strict {bm['parse_strict']} -> {sm['parse_strict']}")

steps = h["global_step"] or 1
secs = h["train_runtime"] or smoke_meta.get("seconds")
sec_per_step = secs / steps if secs else None
n_rows = 1800 + round(0.1 / 0.9 * 1800)          # train after val split + replay
full_steps = math.ceil(n_rows / 16) * 2
est = f"{sec_per_step * full_steps / 60:.1f} min" if sec_per_step else "unknown"
check("3.5", sec_per_step is not None, f"{sec_per_step and round(sec_per_step, 2)} s/step over {steps} steps; "
      f"Phase 4 ≈ {full_steps} steps ≈ {est} (+ model load/save); backend={sprint.get_backend()}")
sprint.log_result("smoke", "val", smoke_val, sprint.settings(smoke_cfg), notes=f"200 rows; backend={sprint.get_backend()}")
""")

# ===================================================================== phase 4
md("""
## Phase 4 — Full SFT run (≈ 1 h, ~10 min of GPU)

`configs/sft_v1.yaml` is the plan's config: 2 epochs, lr 1e-4, batch 16, LoRA r16/α32, cosine, warmup 0.03,
replay 10%, no quantization, `eval_steps: 25`, loss on the JSON answer only (Soup's default).
`sft_v1` is also the lr = 1e-4 point of the Phase 5 learning-rate round.

| # | Check | Pass when |
|---|---|---|
| 4.1 | Curves | training loss falls; validation loss flattens (rising in epoch 2 is noted for Phase 5) |
| 4.2 | Val exact_match | clearly above base, paired CI95 lower bound > 0 |
| 4.3 | Format | parse_strict ≥ 95% |
| 4.4 | Missing-field group | within 15 points of the complete group |
| 4.5 | Read 20 errors | each failure has a nameable cause |
""")
code("""
sft_row = sprint.run("sft_v1", {}, round_="1-lr", notes="plan config")
print(open("configs/sft_v1.yaml").read())
""")
code("""
sprint.plot_losses(["sft_v1"], title="sft_v1", save_as="loss_sft_v1.png")
# Optional live view:  %load_ext tensorboard  then  %tensorboard --logdir runs
""")
code("""
# ---- Phase 4 checks
sft_val = json.load(open("reports/sft_v1_val.json"))
h = sprint.loss_history("runs/sft_v1")
tr = [l for _, l in h["train"]]; va = [l for _, l in h["eval"]]
falls = len(tr) > 1 and tr[-1] < tr[0]
rising = len(va) > 2 and va[-1] > min(va) * 1.02
check("4.1", falls if not rising else None,
      f"train loss {tr[0] if tr else None} -> {tr[-1] if tr else None}; "
      f"val loss min {min(va) if va else None}, last {va[-1] if va else None}"
      + ("  — val loss rose at the end: note for Phase 5 (epochs)" if rising else ""))
p = sft_val["paired_a_minus_b"]
check("4.2", p["exact_match_diff"] > 0 and p["ci95"][0] > 0,
      f"exact_match {sft_val['metrics']['exact_match']}% vs base: {p['exact_match_diff']:+} points, CI95={p['ci95']}")
check("4.3", sft_val["metrics"]["parse_strict"] >= 95, f"parse_strict={sft_val['metrics']['parse_strict']}%")
g = sft_val["groups"]
gap = g["complete"]["exact_match"] - g["missing_field"]["exact_match"]
check("4.4", gap <= 15, f"complete={g['complete']['exact_match']}%, missing_field={g['missing_field']['exact_match']}% (gap {gap:.1f})")
sprint.show_errors("sft_v1", "val", n=20)
check("4.5", None, f"error types: {list(sft_val['error_types'].items())[:6]} — name the cause of each of the 20 above")
""")

# ===================================================================== phase 5
md("""
## Phase 5 — Hyperparameter study (≈ 2–4 h, ~1–1.5 h of GPU)

One factor at a time, **validation set only**. Each run gets its own config and `runs/<name>/`; re-running a
cell skips any run whose adapter is already on Drive. Results collect in `reports/hparam_study.csv`.

| Round | Vary | Values | Kept fixed |
|---|---|---|---|
| 1 | learning rate | 5e-5, 1e-4 (= sft_v1), 2e-4 | 2 epochs, rank 16 |
| 2 | epochs | 1, 3 (2 already ran) | best lr, rank 16 |
| 3 | LoRA rank | 8, 32 (alpha = 2 × rank) | best lr and epochs |
| 4 | seed | 7, 123 (42 already ran) | best of everything → run-to-run noise |

**Choosing inside a round:** the best exact_match wins, but any setting within `TIE_POINTS` of the best counts as a
tie and the cheaper one is chosen (fewer epochs, smaller rank). Round 4 measures the real noise; if it is bigger
than `TIE_POINTS`, revisit the earlier picks with that number.

| # | Check | Pass when |
|---|---|---|
| 5.1 | All runs scored | 9 rows in the study table, each with exact_match and CI95 |
| 5.2 | Learning rate | 2e-4 learns fastest (may be noisier); 5e-5 may underfit in 2 epochs |
| 5.3 | Epochs | val loss is lowest near the best epoch count |
| 5.4 | Noise | the 3 seeds agree within a few points (≤ 5) |
| 5.5 | Frozen choice | `BEST_RUN` written to `reports/best_run.json` **before** any test-set run |
""")
code("""
TIE_POINTS = 2.0  # differences smaller than this are ties (update after round 4)

def study():
    import pandas as pd
    return pd.read_csv("reports/hparam_study.csv")

def pick(names, prefer=None):
    \"\"\"Best exact_match among `names`; within TIE_POINTS, prefer the smallest value of `prefer`.\"\"\"
    df = study().set_index("run").loc[names]
    best = df["exact_match"].max()
    ties = df[df["exact_match"] >= best - TIE_POINTS]
    choice = ties.sort_values(prefer).index[0] if prefer else df["exact_match"].idxmax()
    print(df[["lr", "epochs", "rank", "seed", "exact_match", "ci_lo", "ci_hi", "min_val_loss", "last_val_loss"]])
    print(f"-> pick {choice} (ties within {TIE_POINTS} pts: {list(ties.index)})")
    return choice

def name_lr(lr):
    return f"lr_{lr}"
""")
md("### Round 1 — learning rate")
code("""
for lr in [5e-5, 2e-4]:
    sprint.run(name_lr(lr), {"training": {"lr": lr}}, round_="1-lr")
R1 = ["lr_5e-05", "sft_v1", "lr_0.0002"]
best_r1 = pick(R1)
BEST_LR = float(study().set_index("run").loc[best_r1, "lr"])
sprint.plot_losses(R1, title="Round 1: learning rate", save_as="loss_round1.png")
print("BEST_LR =", BEST_LR)
""")
md("### Round 2 — epochs")
code("""
for ep in [1, 3]:
    sprint.run(f"ep_{ep}", {"training": {"lr": BEST_LR, "epochs": ep}}, round_="2-epochs")
R2 = ["ep_1", best_r1, "ep_3"]
best_r2 = pick(R2, prefer="epochs")
BEST_EPOCHS = int(study().set_index("run").loc[best_r2, "epochs"])
sprint.plot_losses(R2, title="Round 2: epochs", save_as="loss_round2.png")
print("BEST_EPOCHS =", BEST_EPOCHS)
""")
md("### Round 3 — LoRA rank")
code("""
for r in [8, 32]:
    sprint.run(f"r_{r}", {"training": {"lr": BEST_LR, "epochs": BEST_EPOCHS, "lora": {"r": r, "alpha": 2 * r}}},
               round_="3-rank")
R3 = ["r_8", best_r2, "r_32"]
best_r3 = pick(R3, prefer="rank")
BEST_RANK = int(study().set_index("run").loc[best_r3, "rank"])
print("BEST_RANK =", BEST_RANK, "| best config so far:", best_r3)
""")
md("### Round 4 — seeds (run-to-run noise)")
code("""
best_over = {"training": {"lr": BEST_LR, "epochs": BEST_EPOCHS, "lora": {"r": BEST_RANK, "alpha": 2 * BEST_RANK}}}
for seed in [7, 123]:
    sprint.run(f"best_seed{seed}", sprint.deep_merge(best_over, {"training": {"seed": seed}}), round_="4-seed",
               notes="noise check")
R4 = [best_r3, "best_seed7", "best_seed123"]
df4 = study().set_index("run").loc[R4]
NOISE = float(df4["exact_match"].max() - df4["exact_match"].min())
print(df4[["seed", "exact_match", "ci_lo", "ci_hi"]]); print(f"seed spread = {NOISE:.1f} points")
""")
code("""
# ---- Phase 5 checks
df = study()
nine = ["sft_v1", "lr_5e-05", "lr_0.0002", "ep_1", "ep_3", "r_8", "r_32", "best_seed7", "best_seed123"]
have = [n for n in nine if n in set(df["run"])]
check("5.1", len(have) == 9 and df.set_index("run").loc[have, ["ci_lo", "ci_hi"]].notna().all().all(),
      f"{len(have)}/9 study runs scored: missing {sorted(set(nine) - set(have))}")
s = df.set_index("run")
check("5.2", None, "exact_match / final train loss by lr: " + ", ".join(
      f"{n}: {s.loc[n, 'exact_match']}% / {s.loc[n, 'final_train_loss']:.3f}" for n in R1))
check("5.3", None, "val loss (min / last) by epochs: " + ", ".join(
      f"{s.loc[n, 'epochs']}ep: {s.loc[n, 'min_val_loss']:.3f} / {s.loc[n, 'last_val_loss']:.3f}" for n in R2))
check("5.4", NOISE <= 5, f"seed spread {NOISE:.1f} points (TIE_POINTS was {TIE_POINTS})"
      + ("; earlier picks that differ by less than this are ties" if NOISE > TIE_POINTS else ""))
print(df.to_string())
""")
md("""
### Freeze `BEST_RUN` — before any test-set run

`BEST_RUN` defaults to the round-3 winner (seed 42). Change it here if the tables above argue for something else
(e.g. a cheaper setting that ties). Once the test set has been run, this cell refuses to change it.
""")
code("""
BEST_RUN = best_r3   # edit if you choose differently

import time
if os.path.exists("reports/test_lock.json"):
    frozen = json.load(open("reports/best_run.json"))
    assert frozen["run"] == BEST_RUN, f"test set already run with {frozen['run']}; BEST_RUN cannot change now"
    print("already frozen:", frozen)
else:
    row = study().set_index("run").loc[BEST_RUN]
    frozen = {"run": BEST_RUN, "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "lr": float(row["lr"]), "epochs": int(row["epochs"]), "rank": int(row["rank"]), "seed": int(row["seed"]),
              "val_exact_match": float(row["exact_match"]), "seed_noise_points": NOISE}
    json.dump(frozen, open("reports/best_run.json", "w"), indent=2)
    print("frozen:", frozen)
check("5.5", os.path.exists("reports/best_run.json"), f"BEST_RUN={BEST_RUN} frozen at {frozen['frozen_at']}")
""")

# ===================================================================== phase 6
md("""
## Phase 6 — Final test, general-skills check, export (≈ 2–3 h)

After a reconnect, run Phase 0's install and *Load the helpers* cells, then continue here.

**Steps:** 1) answers on **test** and **stress** from `BEST_RUN` and from the base model, scored as a pair (run once),
2) IFEval + HellaSwag with `lm-eval` on base and tuned, 3) optional merge + GGUF export, 4) results summary,
5) **Runtime → Disconnect and delete runtime**.

**Acceptance checks (set before seeing test results)**

| # | Check | Pass when |
|---|---|---|
| 6.1 | Task win | test exact_match beats base by ≥ 10 points and the paired CI95 is entirely above 0 |
| 6.2 | Format | parse_strict ≥ 95% on test |
| 6.3 | Robustness | stress exact_match within ~15 points of test |
| 6.4 | No forgetting | IFEval and HellaSwag each drop ≤ 5 points from base |
| 6.5 | Honest process | test was run once, after `BEST_RUN` was frozen |

If 6.1 fails, read the errors file before changing anything — most fixes are in the data, not the hyperparameters.
""")
code("""
import time
frozen = json.load(open("reports/best_run.json"))
BEST_RUN = frozen["run"]
lock_path = "reports/test_lock.json"
if not os.path.exists(lock_path):
    json.dump({"run": BEST_RUN, "first_test_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "best_frozen_at": frozen["frozen_at"]}, open(lock_path, "w"), indent=2)
lock = json.load(open(lock_path))
print("lock:", lock)

final = {}
for split in ["test", "stress"]:
    sprint.predict("base", split)                              # skipped if already done
    sprint.predict(BEST_RUN, split, adapter_run=BEST_RUN)      # skipped if already done
    final[split] = sprint.score(BEST_RUN, split, vs="base")
    sprint.log_result(f"FINAL {BEST_RUN}", split, final[split], frozen, notes="vs base, paired")
""")
code("""
# ---- 6.1-6.3, 6.5
t, st = final["test"], final["stress"]
p = t["paired_a_minus_b"]
check("6.1", p["exact_match_diff"] >= 10 and p["ci95"][0] > 0,
      f"test exact_match {t['metrics']['exact_match']}% vs base {t['b']['metrics']['exact_match']}%: "
      f"{p['exact_match_diff']:+} points, CI95={p['ci95']}")
check("6.2", t["metrics"]["parse_strict"] >= 95, f"test parse_strict={t['metrics']['parse_strict']}%")
gap = t["metrics"]["exact_match"] - st["metrics"]["exact_match"]
check("6.3", gap <= 15, f"test {t['metrics']['exact_match']}% vs stress {st['metrics']['exact_match']}% (gap {gap:.1f})")
check("6.5", lock["run"] == BEST_RUN and lock["best_frozen_at"] == frozen["frozen_at"],
      f"test first run at {lock['first_test_at']} with {lock['run']}; BEST_RUN frozen at {frozen['frozen_at']}")
if p["exact_match_diff"] < 10:
    sprint.show_errors(BEST_RUN, "test", n=20)
""")
md("""
### General-skills regression (lm-eval)

IFEval (instruction following, with the chat template) and HellaSwag (general, log-likelihood, no template),
base vs base + adapter. Set `LM_EVAL_LIMIT` to e.g. 500 for a faster, noisier estimate.
""")
code("""
!pip -q install "lm-eval[ifeval]"
""")
code("""
import glob
LM_EVAL_LIMIT = None
TASKS = {"ifeval": ("prompt_level_strict_acc,none", True), "hellaswag": ("acc_norm,none", False)}

def lm_eval(tag, adapter_run=None):
    scores = {}
    for task, (metric, chat) in TASKS.items():
        out = f"reports/lm_eval/{tag}_{task}"
        if not glob.glob(f"{out}/**/results_*.json", recursive=True):
            args = f"pretrained={sprint.BASE},dtype=bfloat16"
            if adapter_run:
                args += f",peft={sprint.find_adapter(os.path.join('runs', adapter_run))}"
            cmd = [sys.executable, "-m", "lm_eval", "--model", "hf", "--model_args", args, "--tasks", task,
                   "--batch_size", "auto", "--output_path", out, "--seed", "0"]
            if chat:
                cmd.append("--apply_chat_template")
            if LM_EVAL_LIMIT:
                cmd += ["--limit", str(LM_EVAL_LIMIT)]
            sprint.sh(cmd, log=f"logs/lm_eval_{tag}_{task}.log")
        res = json.load(open(sorted(glob.glob(f"{out}/**/results_*.json", recursive=True))[-1]))
        scores[task] = round(100 * res["results"][task][metric], 2)
    return scores

lm_base = lm_eval("base")
lm_best = lm_eval("best", adapter_run=BEST_RUN)
print("base:", lm_base, "\\nbest:", lm_best)
json.dump({"base": lm_base, "best": lm_best, "run": BEST_RUN}, open("reports/lm_eval_summary.json", "w"), indent=2)
""")
code("""
# ---- 6.4
drops = {k: round(lm_base[k] - lm_best[k], 2) for k in TASKS}
check("6.4", all(d <= 5 for d in drops.values()),
      "drop in points (base - tuned): " + ", ".join(f"{k} {lm_base[k]} -> {lm_best[k]} ({d:+})" for k, d in drops.items()))
""")
md("""
### Export (optional)

Merges the adapter into the base model and exports a `q4_k_m` GGUF (runs on a GTX 1650 laptop or CPU).
Uses `soup merge` / `soup export`; if `soup merge` fails, peft's `merge_and_unload` is used instead.
""")
code("""
import glob
RUN_EXPORT = False
if RUN_EXPORT:
    merged = "runs/merged"
    rc, _ = sprint.sh(["soup", "merge", "--adapter", sprint.find_adapter(f"runs/{BEST_RUN}"), "--output", merged],
                      check_rc=False)
    if rc != 0:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        m = AutoModelForCausalLM.from_pretrained(sprint.BASE, torch_dtype=torch.bfloat16)
        m = PeftModel.from_pretrained(m, sprint.find_adapter(f"runs/{BEST_RUN}")).merge_and_unload()
        m.save_pretrained(merged); AutoTokenizer.from_pretrained(sprint.BASE).save_pretrained(merged)
    rc, _ = sprint.sh(["soup", "export", "--model", merged, "--format", "gguf", "--quant", "q4_k_m",
                       "--output", "exports"], check_rc=False)
    ggufs = glob.glob("exports/**/*.gguf", recursive=True) + glob.glob(f"{merged}/**/*.gguf", recursive=True)
    check("6.x", bool(ggufs), f"GGUF files: {ggufs} (soup export exit {rc})")
""")
md("### Results log and check summary")
code("""
lm = json.load(open("reports/lm_eval_summary.json")) if os.path.exists("reports/lm_eval_summary.json") else None
log = sprint.table("results_log")
lines = ["# Results — SmolLM2-360M booking JSON", "",
         f"BEST_RUN: `{BEST_RUN}` (frozen {frozen['frozen_at']}); backend: {sprint.get_backend()}", "",
         "All scores are % on the validation set except the FINAL rows.", "", log.to_markdown(index=False), "",
         "## Hyperparameter study", "", sprint.table("hparam_study").to_markdown(index=False), ""]
if lm:
    lines += ["## General skills (lm-eval)", "", "| task | base | tuned | Δ |", "|---|---|---|---|"]
    lines += [f"| {k} | {lm['base'][k]} | {lm['best'][k]} | {lm['best'][k] - lm['base'][k]:+.2f} |" for k in lm["base"]]
open("reports/RESULTS.md", "w").write("\\n".join(lines))
print("\\n".join(lines))
print()
sprint.checks_summary()
""")
md("""
**Done → Runtime → Disconnect and delete runtime** to stop using compute units.
Everything (data, configs, adapters, predictions, reports, `RESULTS.md`) is in `MyDrive/soup-sprint/`.
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm",
                                                                    "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(HERE, "soup_sprint_colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"wrote {out}: {len(cells)} cells")
