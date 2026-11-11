"""Build soup_sprint_v2_colab.ipynb: the second run (noisy, more varied training data; new test and stress sets).

Each script is embedded as a %%writefile cell, so the notebook alone recreates the project on Google Drive.
Usage: python build_notebook_v2.py   (re-run after editing any script)
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = [
    "scripts/common.py", "scripts/generate_data.py", "scripts/generate_data_v2.py", "scripts/fetch_replay.py",
    "scripts/check_data.py", "scripts/check_data_v2.py", "scripts/evaluate.py", "scripts/breakdown.py",
    "scripts/predict.py", "scripts/train_hf.py", "scripts/sprint.py",
    "tests/test_data.py", "tests/test_data_v2.py", "tests/test_replay.py", "tests/test_evaluate.py",
    "tests/test_predict.py", "tests/test_sprint.py", "tests/test_train_hf.py",
]

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


# ===================================================================== intro
md("""
# Soup sprint v2: does noisier, more varied training data fix the stress gap?

**Run 1 result:** `ep_3` scored 86.7% on test but 44.0% on stress, a 42.7-point gap. The stress errors were
mostly wording the model never saw ("rm B", "quarter to seven", "two weeks from today"), plus typos and fillers.

**This run** trains two models with the **same settings** as `ep_3` (lr 2e-4, 3 epochs, LoRA r16, seed 42), both with **Soup**:

| Run | Training data | Why |
|---|---|---|
| `ep3_soup` | v1's 2,000 rows (regenerated, identical) | Reference: `ep_3` again, but on Soup (run 1's `ep_3` used the fallback trainer) |
| `noisy_v2` | the same 2,000 rows **+ 1,000 augmented rows** | Noise (typos, fillers, case, missing punctuation) and new wording |

Both are scored as a **pair** on **new** val, test and stress sets.

**How the "do / don't" list is applied**

| Rule | How |
|---|---|
| Train from the base model, on old + new data | `noisy_v2` starts from SmolLM2; train = v1 rows + augmented rows |
| Training noise differs from stress noise | Stress uses its own fillers and a keyboard-neighbour typo that training never sees |
| Hold some wording out | Three wording tiers: *base* (v1), *extra* (training sees it), *heldout* (**stress only**) |
| New test and stress sets | 8 new template families + new seed (2027). v1's test templates moved into training |
| Paired comparison, clean and stress | `noisy_v2` vs `ep3_soup` on the same rows, with CI95, also per stress group |
| One trainer | Soup only. The fallback is off: if Soup fails, the notebook stops |
| Don't tune on stress | No choices are made in this notebook. Test and stress run once (lock file) |

Everything is written to `MyDrive/soup-sprint-v2/`; run 1's folder `MyDrive/soup-sprint/` is only read.

| Phase | What | Checks |
|---|---|---|
| 0 | Setup | 0.1–0.3 |
| 1 | Data v1 (regenerated) + v2, checks | 1.1–1.8, 1.H, 1.V |
| 2 | Train `ep3_soup` and `noisy_v2` with Soup | 2.1–2.3 |
| 3 | Validation (look, don't choose) | 3.1–3.2 |
| 4 | Test + stress, once; per-group breakdown | 4.1–4.6, 4.B |
| 5 | General skills (IFEval, HellaSwag) | 5.1 |
| 6 | Results summary | — |
""")

# ===================================================================== phase 0
md("""
## Phase 0 — Setup (≈ 10 min)

| # | Check | Pass when |
|---|---|---|
| 0.1 | GPU | A GPU is visible and supports bf16 |
| 0.2 | Soup | `soup --version` and `soup doctor` exit 0 |
| 0.3 | Drive | `MyDrive/soup-sprint-v2/` exists and `reports/pip_freeze.txt` was written |
""")
code("""
!nvidia-smi
import sys; print("python", sys.version)
""")
code("""
import os
from google.colab import drive
drive.mount("/content/drive")

PROJECT = "/content/drive/MyDrive/soup-sprint-v2"
V1_PROJECT = "/content/drive/MyDrive/soup-sprint"      # run 1: read only
for d in ["scripts", "tests", "data/eval", "data_v1/eval", "configs", "runs", "reports/preds", "logs"]:
    os.makedirs(os.path.join(PROJECT, d), exist_ok=True)
os.chdir(PROJECT)
print("working in", os.getcwd(), "| run 1 folder found:", os.path.isdir(V1_PROJECT))
""")
code("""
# ~2-4 min
!pip -q install "soup-cli[train,eval]" pyyaml pytest tabulate
# Colab preinstalls torchao 0.10, which recent peft refuses to load (needs > 0.16). Neither Soup nor this project uses it.
!pip uninstall -y -q torchao
!soup --version
""")
md("### Write the project files to Drive")
for rel in FILES:
    with open(os.path.join(HERE, rel), encoding="utf-8") as f:
        body = f.read()
    code(f"%%writefile {rel}\n{body}")
code("""
# Load the helpers (re-run this cell after any reconnect)
import importlib, json, os, sys, subprocess, shutil
os.chdir(PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "scripts"))
import sprint; importlib.reload(sprint)
import evaluate as ev; importlib.reload(ev)
import breakdown as bd; importlib.reload(bd)
from sprint import check
""")
code("""
# ---- Phase 0 checks
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
bf16 = bool(gpu) and torch.cuda.is_bf16_supported()
check("0.1", bool(gpu) and bf16, f"gpu={gpu}, bf16={bf16}", hard=True)
rc, ver = sprint.sh(["soup", "--version"], check_rc=False)
rc_doc, _ = sprint.sh(["soup", "doctor"], check_rc=False, quiet=True)
check("0.2", rc == 0 and rc_doc == 0, f"soup --version exit {rc}; soup doctor exit {rc_doc}", hard=True)
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
open("reports/pip_freeze.txt", "w").write(freeze)
check("0.3", os.path.getsize("reports/pip_freeze.txt") > 0, f"{PROJECT}; {len(freeze.splitlines())} packages", hard=True)
""")

# ===================================================================== phase 1
md("""
## Phase 1 — Data (≈ 15 min)

1. **`data_v1/`**: v1's training data, regenerated with seed 1234 (identical to run 1), and run 1's replay file.
2. **`data/`**: the v2 data from `scripts/generate_data_v2.py` (seed 2027):

| File | Rows | Contents |
|---|---|---|
| `train.jsonl` | 3,000 | v1's 2,000 rows + 1,000 augmented rows (v1 train + retired v1 test templates; noise and *extra* wording), shuffled |
| `eval/val.jsonl` | 300 | v1's val templates; about half clean, half augmented (`group` = clean / augmented) |
| `eval/test.jsonl` | 300 | **8 new templates**, clean, v1-style wording |
| `eval/stress.jsonl` | 300 | the new test templates + **stress noise** and/or **held-out wording** (`group` = noise_only / new_wording / both) |
| `replay.jsonl` | 400 | general chat (Soup needs 333 for 3,000 rows at 10%) |

**Wording tiers** (each row records its tier and form in `forms`):

| Value | base (v1) | extra (training sees it) | heldout (stress only) |
|---|---|---|---|
| date | today, tomorrow, in 5 days, next Friday, December 7, 2026-12-07 … | a week from today, in five days, Dec. 7th, December 7th | two weeks from today, 5 days from now, 7th Dec, Dec 7, tmrw |
| time | 18:00, 6pm, 6 pm, 6 o'clock in the evening, noon | half past six, quarter to seven, 6.30 p.m., midday | 1830h, 6:30 in the evening, 6:30PM, 18h30 |
| resource | Room B, room B | rm B, room-B, studio two, studio no. 2, laboratory 3 | roomB, room #B, studio number two, conference room B |

| # | Check | Pass when |
|---|---|---|
| 1.1 | Tests | all `pytest` tests pass (v1 + v2 generator + per-group scorer) |
| 1.V | v1 reproduced | `data_v1/train.jsonl` matches run 1's checksum; replay copied from run 1 |
| 1.2 | Structure | 3,000 chat rows; all 2,000 v1 rows present unchanged |
| 1.3 | Labels | all 3,900 labels pass the schema |
| 1.4 | No leakage | no shared text; template families separate; test families are new |
| 1.H | Held out | held-out wording, neighbour typos and stress fillers appear **only** in stress |
| 1.5 | Coverage | 4 intents everywhere; missing share 10–60%; every stress group ≥ 60 rows |
| 1.6 | Checksums | files match `data/manifest.json` |
| 1.7 | Soup format | `soup data validate` accepts train (3,000) and replay |
| 1.8 | Your eyes | the printed rows (2 per stress group) have obviously right labels |
""")
code("""
!python scripts/generate_data.py --out data_v1 --seed 1234
src = os.path.join(V1_PROJECT, "data", "replay.jsonl")
if os.path.exists(src):
    shutil.copy(src, "data_v1/replay.jsonl"); print("copied run 1's replay file")
else:
    !python scripts/fetch_replay.py --out data_v1/replay.jsonl --n 300
!python scripts/generate_data_v2.py --out data --seed 2027 --v1-seed 1234
!python scripts/fetch_replay.py --out data/replay.jsonl --n 400
""")
code("""
# ---- 1.1 tests, 1.V v1 reproduced
import re
rc, out = sprint.sh([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], check_rc=False)
check("1.1", rc == 0 and re.search(r"(\\d+) passed", out) is not None, f"pytest exit {rc}; {out.strip().splitlines()[-1]}", hard=True)

ours = json.load(open("data_v1/manifest.json"))["files"]["train.jsonl"]["sha256"]
v1_manifest = os.path.join(V1_PROJECT, "data", "manifest.json")
theirs = json.load(open(v1_manifest))["files"]["train.jsonl"]["sha256"] if os.path.exists(v1_manifest) else "eafd6b9bbb69"
check("1.V", ours.startswith(theirs[:12]) and (len(theirs) == 12 or ours == theirs),
      f"data_v1/train.jsonl sha256 {ours[:12]}… vs run 1 {theirs[:12]}…", hard=True)
""")
code("""
# ---- 1.2-1.6, 1.H, replay, and rows for 1.8
rc, out = sprint.sh([sys.executable, "scripts/check_data_v2.py", "--data", "data", "--v1-data", "data_v1",
                     "--reports", "reports"], check_rc=False)
for r in json.load(open("reports/data_checks.json")):
    check(r["id"], r["ok"], f"{r['name']}: {r['detail']}")
check("1.x", rc == 0, "all data checks passed" if rc == 0 else "data checks failed — fix before training", hard=True)
""")
code("""
# ---- 1.7 Soup accepts the files
rc_t, out_t = sprint.sh(["soup", "data", "validate", "data/train.jsonl"], check_rc=False)
rc_r, _ = sprint.sh(["soup", "data", "validate", "data/replay.jsonl"], check_rc=False)
check("1.7", rc_t == 0 and rc_r == 0 and ("3000/3000" in out_t or "3,000" in out_t),
      f"train exit {rc_t} (expect 3000/3000 valid), replay exit {rc_r}", hard=True)
""")
code("""
# ---- 1.8 your eyes: read the rows printed by the data-check cell above.
# Set to True once every label is obviously right (False = stop and fix the generator).
LABELS_LOOK_RIGHT = None
check("1.8", LABELS_LOOK_RIGHT, "rows read by eye (augmented train, val, test, 2 per stress group)")
""")

# ===================================================================== phase 2
md("""
## Phase 2 — Train two models with Soup (≈ 20 min, ~15 min of GPU)

Both use run 1's winning settings: lr 2e-4, 3 epochs, LoRA r16/α32, seed 42, batch 16, cosine, replay 10%.
The only difference is the training file. **The fallback trainer is off**, so both runs use the same trainer.

| # | Check | Pass when |
|---|---|---|
| 2.1 | Dry runs | `soup train --dry-run` passes for both configs |
| 2.2 | Same trainer | both runs trained with backend `soup` |
| 2.3 | Curves | training loss finite and falling; validation loss logged |
""")
code("""
sprint.ALLOW_HF_FALLBACK = False
sprint.set_backend("soup")
HP = {"training": {"lr": 2e-4, "epochs": 3, "seed": 42, "lora": {"r": 16, "alpha": 32, "dropout": 0.05}}}
RUNS = {
    "ep3_soup": sprint.deep_merge(HP, {"data": {"train": "./data_v1/train.jsonl", "replay": "./data_v1/replay.jsonl"}}),
    "noisy_v2": sprint.deep_merge(HP, {"data": {"train": "./data/train.jsonl", "replay": "./data/replay.jsonl"}}),
}
CFG = {name: sprint.make_config(name, over) for name, over in RUNS.items()}
for name, (path, _) in CFG.items():
    print(f"---- {path}"); print(open(path).read())
    how, rc, _ = sprint.dry_run(path)
    check(f"2.1{'ab'[list(RUNS).index(name)]}", rc == 0, f"{name}: {how} exit {rc}", hard=True)
""")
code("""
# ---- train (skipped if the adapter is already on Drive; stops on any Soup failure)
META = {name: sprint.train(name, path, cfg) for name, (path, cfg) in CFG.items()}
for m in META.values():
    print(m)
""")
code("""
# ---- Phase 2 checks
import math
backends = {n: m.get("backend") for n, m in META.items()}
check("2.2", all(b == "soup" for b in backends.values()), f"backends={backends}")
ok, detail = True, []
for n in RUNS:
    h = sprint.loss_history(f"runs/{n}")
    tr = [l for _, l in h["train"]]; va = [l for _, l in h["eval"]]
    good = len(tr) > 1 and all(math.isfinite(l) for l in tr) and tr[-1] < tr[0] and len(va) > 0
    ok &= good
    detail.append(f"{n}: train {tr[0]:.3f}->{tr[-1]:.3f}, val min {min(va) if va else None}" if tr else f"{n}: no loss logged")
check("2.3", ok, "; ".join(detail))
sprint.plot_losses(list(RUNS), title="ep3_soup vs noisy_v2", save_as="loss_v2.png")
""")

# ===================================================================== phase 3
md("""
## Phase 3 — Validation (look, don't choose)

Both models are already fixed, so nothing here is used to pick anything. Val shows the effect on familiar
templates: the **clean** half (v1-style) and the **augmented** half (training-style noise and wording).

| # | Check | Pass when |
|---|---|---|
| 3.1 | Clean val held | `noisy_v2` − `ep3_soup` on clean val ≥ −3 points |
| 3.2 | Augmented val | 👀 how much `noisy_v2` gains on augmented val (expected: clearly positive) |
""")
code("""
def group_table(split, a, b):
    gold = ev.load_jsonl(f"data/eval/{split}.jsonl")
    res = bd.by_group(gold, ev.load_preds(sprint.pred_path(a, split)), ev.load_preds(sprint.pred_path(b, split)))
    pb = bd.by_group(gold, ev.load_preds(sprint.pred_path(b, split)))
    print(f"{split}: {a} vs {b}")
    print(f"  {'group':12s} {'n':>4s} {b:>10s} {a:>10s}   diff [CI95]")
    for g, r in res.items():
        p = r["paired_a_minus_b"]
        print(f"  {g:12s} {r['n']:4d} {pb[g]['exact_match']:9.1f}% {r['exact_match']:9.1f}%   {p['exact_match_diff']:+.1f} {p['ci95']}")
    return res

for name in ["base", *RUNS]:
    sprint.predict(name, "val", adapter_run=None if name == "base" else name)
sprint.score("base", "val")
val_ref = sprint.score("ep3_soup", "val", vs="base")
val_new = sprint.score("noisy_v2", "val", vs="ep3_soup")
for n, rep in [("ep3_soup", val_ref), ("noisy_v2", val_new)]:
    sprint.log_result(n, "val", rep, sprint.settings(CFG[n][1]), notes=f"train={RUNS[n]['data']['train']}")
val_groups = group_table("val", "noisy_v2", "ep3_soup")
""")
code("""
# ---- Phase 3 checks
d = val_groups["clean"]["paired_a_minus_b"]
check("3.1", d["exact_match_diff"] >= -3, f"clean val: {d['exact_match_diff']:+} points, CI95={d['ci95']}")
d = val_groups["augmented"]["paired_a_minus_b"]
check("3.2", None, f"augmented val: {d['exact_match_diff']:+} points, CI95={d['ci95']}")
sprint.show_errors("noisy_v2", "val", n=10)
""")

# ===================================================================== phase 4
md("""
## Phase 4 — Test and stress, once (≈ 10 min)

**Acceptance checks, set before seeing any results:**

| # | Check | Pass when |
|---|---|---|
| 4.1 | Robustness gain | stress: `noisy_v2` − `ep3_soup`, paired CI95 **entirely above 0** |
| 4.2 | Clean accuracy held | test: `noisy_v2` − `ep3_soup` ≥ −3 points |
| 4.3 | Gap | `noisy_v2`: test − stress ≤ 15 points (run 1: 42.7) |
| 4.4 | Format | `noisy_v2` parse_strict ≥ 95% on test and stress |
| 4.5 | Honest process | lock written once; test/stress not re-run with other models after it |
| 4.6 | Per group | 👀 stress by group (noise_only / new_wording / both) and field accuracy by wording tier |
| 4.B | Trainer effect | 👀 if run 1's `ep_3` adapter is on Drive: `ep_3` (fallback trainer) vs `ep3_soup` (Soup), same data and settings |

Expected: noise_only should improve most (training saw that kind of noise). new_wording measures real
generalisation to wording no training row contains; a smaller gain there is normal. If 4.1 fails, read
`reports/noisy_v2_stress.errors.jsonl` before changing anything, and remember the next try needs a new test set.
""")
code("""
import time
V1_EP3 = os.path.join(V1_PROJECT, "runs", "ep_3")
MODELS = {"base": None, "ep3_soup": "ep3_soup", "noisy_v2": "noisy_v2"}
if sprint.find_adapter(V1_EP3):
    MODELS["ep3_hf_v1"] = V1_EP3          # run 1's winner, for the trainer comparison (4.B) only
lock_path = "reports/test_lock.json"
if not os.path.exists(lock_path):
    json.dump({"models": list(MODELS), "first_test_at": time.strftime("%Y-%m-%d %H:%M:%S")}, open(lock_path, "w"), indent=2)
lock = json.load(open(lock_path))
print("lock:", lock)
assert set(MODELS) <= set(lock["models"]), f"test/stress already run with {lock['models']}; don't add models now"

final = {}
for split in ["test", "stress"]:
    for name, adapter in MODELS.items():
        sprint.predict(name, split, adapter_run=adapter)          # skipped if already done
    sprint.score("base", split)
    ref = sprint.score("ep3_soup", split, vs="base")
    final[split] = sprint.score("noisy_v2", split, vs="ep3_soup")
    sprint.log_result("FINAL ep3_soup", split, ref, sprint.settings(CFG["ep3_soup"][1]), notes="vs base")
    sprint.log_result("FINAL noisy_v2", split, final[split], sprint.settings(CFG["noisy_v2"][1]), notes="vs ep3_soup")
""")
code("""
# ---- per-group breakdown
stress_groups = group_table("stress", "noisy_v2", "ep3_soup")
gold_st = ev.load_jsonl("data/eval/stress.jsonl")
tiers = {n: bd.field_by_tier(gold_st, ev.load_preds(sprint.pred_path(n, "stress"))) for n in ["ep3_soup", "noisy_v2"]}
print("\\nstress field accuracy by wording tier")
print(f"  {'field/tier':18s} {'n':>4s} {'ep3_soup':>9s} {'noisy_v2':>9s}")
for k in tiers["noisy_v2"]:
    print(f"  {k:18s} {tiers['noisy_v2'][k]['n']:4d} {tiers['ep3_soup'][k]['accuracy']:8.1f}% {tiers['noisy_v2'][k]['accuracy']:8.1f}%")
json.dump({"stress_groups": stress_groups, "field_by_tier": tiers}, open("reports/stress_breakdown.json", "w"), indent=2)
""")
code("""
# ---- Phase 4 checks
t, s = final["test"], final["stress"]
ps, pt = s["paired_a_minus_b"], t["paired_a_minus_b"]
check("4.1", ps["ci95"][0] > 0, f"stress: noisy_v2 {s['metrics']['exact_match']}% vs ep3_soup {s['b']['metrics']['exact_match']}%: "
      f"{ps['exact_match_diff']:+} points, CI95={ps['ci95']}")
check("4.2", pt["exact_match_diff"] >= -3, f"test: noisy_v2 {t['metrics']['exact_match']}% vs ep3_soup {t['b']['metrics']['exact_match']}%: "
      f"{pt['exact_match_diff']:+} points, CI95={pt['ci95']}")
gap = t["metrics"]["exact_match"] - s["metrics"]["exact_match"]
gap_ref = t["b"]["metrics"]["exact_match"] - s["b"]["metrics"]["exact_match"]
check("4.3", gap <= 15, f"noisy_v2 gap {gap:.1f} (ep3_soup gap {gap_ref:.1f}; run 1: 42.7)")
check("4.4", t["metrics"]["parse_strict"] >= 95 and s["metrics"]["parse_strict"] >= 95,
      f"parse_strict test {t['metrics']['parse_strict']}%, stress {s['metrics']['parse_strict']}%")
check("4.5", set(MODELS) <= set(lock["models"]), f"test first run at {lock['first_test_at']} with {lock['models']}")
check("4.6", None, "stress by group: " + ", ".join(
      f"{g} {r['paired_a_minus_b']['exact_match_diff']:+.1f} {r['paired_a_minus_b']['ci95']}" for g, r in stress_groups.items()))
if "ep3_hf_v1" in MODELS:
    diffs = []
    for split in ["test", "stress"]:
        gold = ev.load_jsonl(f"data/eval/{split}.jsonl")
        rep, _ = ev.evaluate(gold, ev.load_preds(sprint.pred_path("ep3_hf_v1", split)), ev.load_preds(sprint.pred_path("ep3_soup", split)))
        p = rep["paired_a_minus_b"]
        diffs.append(f"{split}: ep_3 (fallback) {rep['metrics']['exact_match']}% vs ep3_soup {rep['b']['metrics']['exact_match']}% "
                     f"= {p['exact_match_diff']:+} {p['ci95']}")
    check("4.B", None, "; ".join(diffs) + " — a CI95 that excludes 0 means the trainer itself changes results")
sprint.show_errors("noisy_v2", "stress", n=15)
""")

# ===================================================================== phase 5
md("""
## Phase 5 — General skills (≈ 40 min; optional)

IFEval and HellaSwag for `noisy_v2` vs base. Base results are copied from run 1 when they exist (same model,
same settings), which saves about 20 minutes.

| # | Check | Pass when |
|---|---|---|
| 5.1 | No forgetting | IFEval and HellaSwag each drop ≤ 5 points from base |
""")
code("""
RUN_LM_EVAL = True
if RUN_LM_EVAL:
    !pip -q install "lm-eval[ifeval]"
""")
code("""
import glob
TASKS = {"ifeval": ("prompt_level_strict_acc,none", True), "hellaswag": ("acc_norm,none", False)}

def lm_eval(tag, adapter_run=None):
    scores = {}
    for task, (metric, chat) in TASKS.items():
        out = f"reports/lm_eval/{tag}_{task}"
        v1_out = os.path.join(V1_PROJECT, out)
        if tag == "base" and not glob.glob(f"{out}/**/results_*.json", recursive=True) and os.path.isdir(v1_out):
            shutil.copytree(v1_out, out, dirs_exist_ok=True); print(f"copied run 1's {out}")
        if not glob.glob(f"{out}/**/results_*.json", recursive=True):
            args = f"pretrained={sprint.BASE},dtype=bfloat16"
            if adapter_run:
                args += f",peft={sprint.find_adapter(os.path.join('runs', adapter_run))}"
            cmd = [sys.executable, "-m", "lm_eval", "--model", "hf", "--model_args", args, "--tasks", task,
                   "--batch_size", "auto", "--output_path", out, "--seed", "0"]
            if chat:
                cmd.append("--apply_chat_template")
            sprint.sh(cmd, log=f"logs/lm_eval_{tag}_{task}.log", quiet=True)
        res = json.load(open(sorted(glob.glob(f"{out}/**/results_*.json", recursive=True))[-1]))
        scores[task] = round(100 * res["results"][task][metric], 2)
    return scores

if RUN_LM_EVAL:
    lm_base, lm_new = lm_eval("base"), lm_eval("noisy_v2", adapter_run="noisy_v2")
    json.dump({"base": lm_base, "noisy_v2": lm_new}, open("reports/lm_eval_summary.json", "w"), indent=2)
    drops = {k: round(lm_base[k] - lm_new[k], 2) for k in TASKS}
    check("5.1", all(d <= 5 for d in drops.values()),
          "drop in points (base - noisy_v2): " + ", ".join(f"{k} {lm_base[k]} -> {lm_new[k]} ({d:+})" for k, d in drops.items()))
""")

# ===================================================================== phase 6
md("### Results summary")
code("""
lm = json.load(open("reports/lm_eval_summary.json")) if os.path.exists("reports/lm_eval_summary.json") else None
bdj = json.load(open("reports/stress_breakdown.json"))
lines = ["# Results v2 — noisy, more varied training data", "",
         "Both models: lr 2e-4, 3 epochs, LoRA r16, seed 42, Soup. Val/test/stress are new (seed 2027).", "",
         sprint.table("results_log").to_markdown(index=False), "",
         "## Stress by group (noisy_v2 − ep3_soup, paired)", "", "| group | n | noisy_v2 | diff | CI95 |", "|---|---|---|---|---|"]
lines += [f"| {g} | {r['n']} | {r['exact_match']} | {r['paired_a_minus_b']['exact_match_diff']:+} | {r['paired_a_minus_b']['ci95']} |"
          for g, r in bdj["stress_groups"].items()]
lines += ["", "## Stress field accuracy by wording tier", "", "| field/tier | n | ep3_soup | noisy_v2 |", "|---|---|---|---|"]
lines += [f"| {k} | {v['n']} | {bdj['field_by_tier']['ep3_soup'][k]['accuracy']} | {v['accuracy']} |"
          for k, v in bdj["field_by_tier"]["noisy_v2"].items()]
if lm:
    lines += ["", "## General skills (lm-eval)", "", "| task | base | noisy_v2 | Δ |", "|---|---|---|---|"]
    lines += [f"| {k} | {lm['base'][k]} | {lm['noisy_v2'][k]} | {lm['noisy_v2'][k] - lm['base'][k]:+.2f} |" for k in lm["base"]]
open("reports/RESULTS_v2.md", "w", encoding="utf-8").write("\\n".join(lines))
print("\\n".join(lines))
print()
sprint.checks_summary()
""")
md("""
**Done → Runtime → Disconnect and delete runtime** to stop using compute units.
Everything is in `MyDrive/soup-sprint-v2/`. The new adapter is `runs/noisy_v2/`.
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm",
                                                                    "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(HERE, "soup_sprint_v2_colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"wrote {out}: {len(cells)} cells")
