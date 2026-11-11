"""Build soup_sprint_1p7b_colab.ipynb: SmolLM2-1.7B-Instruct on the v2 data, compared with run 1's 360M `ep_3`
on the booking task and on IFEval / HellaSwag.

Each script is embedded as a %%writefile cell, so the notebook alone recreates the project on Google Drive.
Usage: python build_notebook_1p7b.py   (re-run after editing any script)
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
# Soup sprint 1.7B: does a 4.7× bigger model beat the 360M on the same data?

**Main question:** does `SmolLM2-1.7B-Instruct` (≈ 4.7× the parameters) score higher on the general-skill
benchmarks **IFEval** and **HellaSwag** than run 1's 360M (IFEval 31.6, HellaSwag 57.0 after tuning), while still
doing the booking task at least as well? It also reports the booking scores and the cost (speed, adapter size).

| | 360M `ep_3` (run 1, already trained) | 1.7B (this notebook) |
|---|---|---|
| Base model | `SmolLM2-360M-Instruct` | `SmolLM2-1.7B-Instruct` |
| Training data | v1: 2,000 clean rows + 10% replay | **v2: the same 2,000 rows + 1,000 augmented rows** (noise, extra wording) + 10% replay |
| Settings | lr 2e-4, 3 epochs, LoRA r16/α32, seed 42 | 3 epochs, r16/α32, seed 42; **lr 1e-4 or 2e-4, chosen on val** |
| Trainer | fallback (`train_hf.py`) | Soup (no fallback) |

**Read the comparison with care:** the two models differ in **size, training data and trainer**, so a difference
between them is the effect of all three together, not of size alone. The fairest part is the **clean test set**
(v1-style wording, which both models' training covered); on **stress**, the 1.7B also had the benefit of the
augmented rows. The comparison with the **untouched 1.7B** (5.1) is clean.

Both are scored on the v2 evaluation sets (new templates, seed 2027), which neither model has seen.

**Order of work:** generate data → smoke run → two lr runs → choose on **val only** and freeze → test and stress
**once** → paired comparison with `ep_3` and with the untouched 1.7B → cost (speed, adapter size) → IFEval and HellaSwag.

Everything is written to `MyDrive/soup-sprint-1p7b/`; run 1's folder `MyDrive/soup-sprint/` is only read
(for its `ep_3` adapter, its replay file and its data checksum).

| Phase | What | GPU time | Checks |
|---|---|---|---|
| 0 | Setup | — | 0.1–0.3 |
| 1 | Generate v2 data, `ep_3` predictions | ~5 min | 1.1–1.6, 1.H, 1.V, 1.C |
| 2 | Baseline: untouched 1.7B on val | ~2 min | 2.1 |
| 3 | Smoke run (200 rows, 1 epoch) | ~3 min | 3.1–3.3 |
| 4 | lr 1e-4 and 2e-4, choose on val, freeze | ~60–80 min | 4.1–4.4 |
| 5 | Test + stress once; vs 360M and vs 1.7B base | ~5 min | 5.1–5.7 |
| 6 | **General skills (IFEval, HellaSwag): the main result** | ~60–90 min | 6.1–6.3 |
| 7 | Results summary | — | — |
""")

# ===================================================================== phase 0
md("""
## Phase 0 — Setup (≈ 10 min)

Use an **A100** (40 or 80 GB) with High-RAM. A 1.7B model with LoRA in bf16 needs about 4 GB of weights plus
activations, so batch 16 fits comfortably; an L4 (24 GB) also works, about 2× slower.

| # | Check | Pass when |
|---|---|---|
| 0.1 | GPU | A GPU is visible, supports bf16, and has ≥ 20 GB of memory |
| 0.2 | Soup | `soup --version` and `soup doctor` exit 0 |
| 0.3 | Drive | `MyDrive/soup-sprint-1p7b/` exists and `reports/pip_freeze.txt` was written |
""")
code("""
!nvidia-smi
import sys; print("python", sys.version)
""")
code("""
import os
from google.colab import drive
drive.mount("/content/drive")

PROJECT = "/content/drive/MyDrive/soup-sprint-1p7b"
V1_PROJECT = "/content/drive/MyDrive/soup-sprint"      # run 1: read only
for d in ["scripts", "tests", "data/eval", "data_v1", "configs", "runs", "reports/preds", "logs"]:
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
import importlib, json, os, re, sys, subprocess, shutil, time
os.chdir(PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "scripts"))
import sprint; importlib.reload(sprint)
import evaluate as ev; importlib.reload(ev)
import breakdown as bd; importlib.reload(bd)
from sprint import check

BIG = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
SMALL = "HuggingFaceTB/SmolLM2-360M-Instruct"
SPLITS = ["val", "test", "stress"]

def predict(name, split, adapter=None, base=BIG, force=False, out=None):
    \"\"\"Greedy answers with an explicit base model (scripts/predict.py --base); records the seconds taken.\"\"\"
    out = out or sprint.pred_path(name, split)
    if os.path.exists(out) and not force:
        print(f"skip predict {name}/{split}: exists"); return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [sys.executable, "scripts/predict.py", "--base", base, "--input", f"data/eval/{split}.jsonl",
           "--output", out, "--batch-size", "32"]
    if adapter:
        cmd += ["--adapter", adapter]
    _, log = sprint.sh(cmd)
    m = re.search(r"(\\d+) rows in ([\\d.]+)s", log)
    if m:
        path = "reports/speed.json"
        speed = json.load(open(path)) if os.path.exists(path) else {}
        speed[os.path.basename(out)] = {"rows": int(m.group(1)), "seconds": float(m.group(2)), "base": base}
        json.dump(speed, open(path, "w"), indent=2)
    return out
""")
code("""
# ---- Phase 0 checks
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
mem = torch.cuda.get_device_properties(0).total_memory / 2**30 if gpu else 0
bf16 = bool(gpu) and torch.cuda.is_bf16_supported()
check("0.1", bool(gpu) and bf16 and mem >= 20, f"gpu={gpu}, {mem:.0f} GB, bf16={bf16}", hard=True)
rc, ver = sprint.sh(["soup", "--version"], check_rc=False)
rc_doc, _ = sprint.sh(["soup", "doctor"], check_rc=False, quiet=True)
check("0.2", rc == 0 and rc_doc == 0, f"soup --version exit {rc}; soup doctor exit {rc_doc}", hard=True)
freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
open("reports/pip_freeze.txt", "w").write(freeze)
check("0.3", os.path.getsize("reports/pip_freeze.txt") > 0, f"{PROJECT}; {len(freeze.splitlines())} packages", hard=True)
""")

# ===================================================================== phase 1
md("""
## Phase 1 — Data, and the 360M predictions (≈ 5 min)

1. **`data_v1/`**: v1's training data, regenerated (seed 1234), only to confirm it matches run 1 exactly.
2. **`data/`**: the v2 data from `scripts/generate_data_v2.py` (seed 2027): v1's 2,000 rows + 1,000 augmented
   training rows; new val (300), test (300) and stress (300) sets; 400 replay rows.
3. `ep_3`'s answers on the new val, test and stress sets (360M base + run 1's adapter).

| # | Check | Pass when |
|---|---|---|
| 1.1 | Tests | all `pytest` tests pass |
| 1.2–1.6, 1.H | Data checks | the v2 data checks pass (structure, labels, leakage, held-out wording, coverage, checksums) |
| 1.V | v1 reproduced | `data_v1/train.jsonl` matches run 1's checksum, so the 2,000 shared rows are the ones `ep_3` trained on |
| 1.C | 360M comparison ready | `ep_3` predictions exist for val, test and stress |
""")
code("""
!python scripts/generate_data.py --out data_v1 --seed 1234
!python scripts/generate_data_v2.py --out data --seed 2027 --v1-seed 1234
!python scripts/fetch_replay.py --out data/replay.jsonl --n 400
""")
code("""
# ---- 1.1 tests, data checks, 1.V
rc, out = sprint.sh([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], check_rc=False)
check("1.1", rc == 0 and re.search(r"(\\d+) passed", out) is not None, f"pytest exit {rc}; {out.strip().splitlines()[-1]}", hard=True)
rc, out = sprint.sh([sys.executable, "scripts/check_data_v2.py", "--data", "data", "--v1-data", "data_v1",
                     "--reports", "reports"], check_rc=False, quiet=True)
for r in json.load(open("reports/data_checks.json")):
    check(r["id"], r["ok"], f"{r['name']}: {r['detail']}")
check("1.x", rc == 0, "all data checks passed" if rc == 0 else "data checks failed", hard=True)
ours = json.load(open("data_v1/manifest.json"))["files"]["train.jsonl"]["sha256"]
v1_manifest = os.path.join(V1_PROJECT, "data", "manifest.json")
theirs = json.load(open(v1_manifest))["files"]["train.jsonl"]["sha256"] if os.path.exists(v1_manifest) else "eafd6b9bbb69"
check("1.V", ours.startswith(theirs[:12]) and (len(theirs) == 12 or ours == theirs),
      f"data_v1/train.jsonl sha256 {ours[:12]}… vs run 1 {theirs[:12]}…", hard=True)
""")
code("""
# ---- 1.C run 1's ep_3 (360M) on the new sets (no choice uses these before Phase 5)
V1_EP3 = os.path.join(V1_PROJECT, "runs", "ep_3")
if sprint.find_adapter(V1_EP3):
    for split in SPLITS:
        predict("ep3_360m", split, adapter=V1_EP3, base=SMALL)
COMPARE = all(os.path.exists(sprint.pred_path("ep3_360m", s)) for s in SPLITS)
check("1.C", True if COMPARE else None,
      "ep_3 predictions ready for val/test/stress" if COMPARE else f"no adapter under {V1_EP3}: "
      "continuing without the 360M comparison (5.3/5.4 will be skipped)")
""")

# ===================================================================== phase 2
md("""
## Phase 2 — Baseline: the untouched 1.7B on val (≈ 3 min)

| # | Check | Pass when |
|---|---|---|
| 2.1 | Baseline scored | `reports/base_1p7b_val.json` exists; parse_lenient > 0 |
""")
code("""
predict("base_1p7b", "val")
base_val = sprint.score("base_1p7b", "val")
sprint.log_result("base_1p7b", "val", base_val, notes="untouched SmolLM2-1.7B-Instruct")
check("2.1", base_val["metrics"]["parse_lenient"] > 0,
      f"exact_match={base_val['metrics']['exact_match']}%, schema_valid={base_val['metrics']['schema_valid']}%, "
      f"parse_lenient={base_val['metrics']['parse_lenient']}%")
sprint.show_errors("base_1p7b", "val", n=5)
""")

# ===================================================================== phase 3
md("""
## Phase 3 — Smoke run (200 rows, 1 epoch, ≈ 3 min)

Catches out-of-memory errors and config problems before the long runs.

| # | Check | Pass when |
|---|---|---|
| 3.1 | Dry run | `soup train --dry-run` passes with the 1.7B base |
| 3.2 | Trains | the adapter is saved, backend is `soup`, loss is finite |
| 3.3 | Speed | seconds per step measured; the time of one full run is estimated |
""")
code("""
sprint.ALLOW_HF_FALLBACK = False
sprint.set_backend("soup")
HP = {"base": BIG, "training": {"epochs": 3, "seed": 42, "lora": {"r": 16, "alpha": 32, "dropout": 0.05}},
      "data": {"train": "./data/train.jsonl", "replay": "./data/replay.jsonl"}}

with open("data/train.jsonl") as f, open("data/train_smoke.jsonl", "w") as g:
    for i, line in enumerate(f):
        if i < 200:
            g.write(line)
smoke_base = sprint.deep_merge(sprint.SFT_CONFIG, sprint.deep_merge(HP, {
    "data": {"train": "./data/train_smoke.jsonl"}, "training": {"epochs": 1, "lr": 2e-4}}))
del smoke_base["data"]["replay"], smoke_base["data"]["replay_ratio"]
smoke_path, smoke_cfg = sprint.make_config("smoke_1p7b", base_cfg=smoke_base)
print(open(smoke_path).read())
how, rc, _ = sprint.dry_run(smoke_path)
check("3.1", rc == 0, f"{how} exit {rc}", hard=True)
""")
code("""
import math
smoke_meta = sprint.train("smoke_1p7b", smoke_path, smoke_cfg)
h = sprint.loss_history("runs/smoke_1p7b")
losses = [l for _, l in h["train"]]
check("3.2", smoke_meta.get("backend") == "soup" and bool(losses) and all(math.isfinite(l) for l in losses),
      f"backend={smoke_meta.get('backend')}; {len(losses)} loss points: {[round(l, 3) for l in losses[:5]]}", hard=True)
steps = h["global_step"] or 1
secs = h["train_runtime"] or smoke_meta.get("seconds")
n_rows = 2700 + round(0.1 / 0.9 * 2700)       # train after the 10% val split, plus replay
full_steps = math.ceil(n_rows / 16) * 3
est = secs / steps * full_steps / 60 if secs else None
check("3.3", est is not None, f"{secs / steps:.2f} s/step over {steps} steps; one full run ≈ {full_steps} steps ≈ {est:.0f} min"
      if est else "no timing found")
""")

# ===================================================================== phase 4
md("""
## Phase 4 — Learning rate: 1e-4 vs 2e-4, chosen on val (≈ 60–80 min)

Larger models often prefer a lower learning rate, so the 360M's 2e-4 is checked against 1e-4.
Everything else stays the same. **The choice uses val only**; test and stress are not touched until it is frozen.

**Rule (set now):** the higher val exact_match wins. If the two are within `TIE_POINTS` (2), choose 1e-4,
the more conservative setting.

| # | Check | Pass when |
|---|---|---|
| 4.1 | Both runs | both trained with Soup and scored on val, with CI95 |
| 4.2 | Learning | each beats the untouched 1.7B on val (paired CI95 above 0) |
| 4.3 | Curves | 👀 validation loss is not rising at the end (overfitting) |
| 4.4 | Frozen | `BEST_1P7B` written to `reports/best_run.json` before any test-set run |
""")
code("""
TIE_POINTS = 2.0
LRS = {"big_lr1e-4": 1e-4, "big_lr2e-4": 2e-4}
VAL, META = {}, {}
for name, lr in LRS.items():
    path, cfg = sprint.make_config(name, sprint.deep_merge(HP, {"training": {"lr": lr}}))
    META[name] = sprint.train(name, path, cfg)          # skipped if the adapter is already on Drive
    predict(name, "val", adapter=f"runs/{name}")
    VAL[name] = sprint.score(name, "val", vs="base_1p7b")
    sprint.log_result(name, "val", VAL[name], sprint.settings(cfg), notes="1.7B")
sprint.plot_losses(list(LRS), title="1.7B: learning rate", save_as="loss_1p7b.png")
""")
code("""
# ---- Phase 4 checks and the frozen choice
em = {n: VAL[n]["metrics"]["exact_match"] for n in LRS}
check("4.1", all(META[n].get("backend") == "soup" for n in LRS),
      "; ".join(f"{n}: {em[n]}% CI95={VAL[n]['metrics']['exact_match_ci95']} ({META[n].get('backend')})" for n in LRS))
check("4.2", all(VAL[n]["paired_a_minus_b"]["ci95"][0] > 0 for n in LRS),
      "; ".join(f"{n} vs base: {VAL[n]['paired_a_minus_b']['exact_match_diff']:+} {VAL[n]['paired_a_minus_b']['ci95']}" for n in LRS))
curves = {}
for n in LRS:
    va = [l for _, l in sprint.loss_history(f"runs/{n}")["eval"]]
    curves[n] = f"val loss min {min(va):.4f}, last {va[-1]:.4f}" if va else "no val loss"
check("4.3", None, "; ".join(f"{n}: {c}" for n, c in curves.items()))

if os.path.exists("reports/test_lock.json"):
    frozen = json.load(open("reports/best_run.json"))
    print("already frozen:", frozen)
else:
    a, b = em["big_lr1e-4"], em["big_lr2e-4"]
    best = "big_lr1e-4" if (a >= b or b - a < TIE_POINTS) else "big_lr2e-4"
    frozen = {"run": best, "lr": LRS[best], "val_exact_match": em[best], "val_all": em,
              "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(frozen, open("reports/best_run.json", "w"), indent=2)
    print("frozen:", frozen)
BEST_1P7B = frozen["run"]
check("4.4", os.path.exists("reports/best_run.json"), f"BEST_1P7B={BEST_1P7B} frozen at {frozen['frozen_at']}")
""")

# ===================================================================== phase 5
md("""
## Phase 5 — Test and stress, once (≈ 5 min)

**Acceptance checks, set before seeing any results:**

| # | Check | Pass when |
|---|---|---|
| 5.1 | Task win | test: 1.7B tuned beats the untouched 1.7B by ≥ 10 points, paired CI95 above 0 |
| 5.2 | Format | parse_strict ≥ 95% on test and stress |
| 5.3 | Size effect, clean | 👀 test: 1.7B − `ep_3` (paired). CI95 above 0 = the bigger model is better; including 0 = no evidence |
| 5.4 | Size effect, stress | 👀 stress: 1.7B − `ep_3` (paired), overall and per group (noise_only / new_wording / both) |
| 5.5 | Gap | 1.7B: test − stress ≤ 15 points |
| 5.6 | Honest process | test/stress run once, after `BEST_1P7B` was frozen |
| 5.7 | Cost | 👀 seconds per 300 rows and adapter size, 1.7B vs 360M |

Expected: new_wording ("rm B", "1830h", "tmrw") is where a bigger model should help most, since it knows more
English from pretraining. If 5.3 and 5.4 both include 0, the 360M is the better choice: same accuracy, about 4.7× cheaper.
""")
code("""
frozen = json.load(open("reports/best_run.json"))
BEST_1P7B = frozen["run"]
lock_path = "reports/test_lock.json"
if not os.path.exists(lock_path):
    json.dump({"run": BEST_1P7B, "first_test_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "best_frozen_at": frozen["frozen_at"]}, open(lock_path, "w"), indent=2)
lock = json.load(open(lock_path))
print("lock:", lock)

final, vs_small = {}, {}
for split in ["test", "stress"]:
    predict("base_1p7b", split)
    predict(BEST_1P7B, split, adapter=f"runs/{BEST_1P7B}")
    if COMPARE:
        vs_small[split] = sprint.score(BEST_1P7B, split, vs="ep3_360m")
        sprint.log_result("FINAL 1.7B vs ep_3 (360M)", split, vs_small[split], notes=f"{BEST_1P7B}; paired vs run 1's ep_3 (360M)")
    final[split] = sprint.score(BEST_1P7B, split, vs="base_1p7b")          # saved report = vs the 1.7B base
    sprint.log_result("FINAL 1.7B vs base", split, final[split], notes=f"{BEST_1P7B}; paired vs untouched 1.7B")
""")
code("""
# ---- per-group breakdown (stress)
gold_st = ev.load_jsonl("data/eval/stress.jsonl")
P = {n: ev.load_preds(sprint.pred_path(n, "stress")) for n in [BEST_1P7B] + (["ep3_360m"] if COMPARE else [])}
groups = bd.by_group(gold_st, P[BEST_1P7B], P.get("ep3_360m"))
small_groups = bd.by_group(gold_st, P["ep3_360m"]) if COMPARE else {}
print(f"  {'group':12s} {'n':>4s} {'360M':>8s} {'1.7B':>8s}   diff [CI95]")
for g, r in groups.items():
    p = r.get("paired_a_minus_b")
    print(f"  {g:12s} {r['n']:4d} {small_groups.get(g, {}).get('exact_match', float('nan')):7.1f}% {r['exact_match']:7.1f}%   "
          + (f"{p['exact_match_diff']:+.1f} {p['ci95']}" if p else "—"))
tiers = {n: bd.field_by_tier(gold_st, p) for n, p in P.items()}
print("\\nstress field accuracy by wording tier")
for k, v in tiers[BEST_1P7B].items():
    print(f"  {k:18s} n={v['n']:4d}  360M {tiers.get('ep3_360m', {}).get(k, {}).get('accuracy', float('nan')):6.1f}%  1.7B {v['accuracy']:6.1f}%")
json.dump({"stress_groups": groups, "ep3_360m_groups": small_groups, "field_by_tier": tiers},
          open("reports/stress_breakdown.json", "w"), indent=2)
""")
code("""
# ---- 5.7 cost: inference time on the same 300 val rows, and adapter size
def adapter_mb(run_dir):
    d = sprint.find_adapter(run_dir)
    f = os.path.join(d, "adapter_model.safetensors") if d else None
    return round(os.path.getsize(f) / 2**20, 1) if f and os.path.exists(f) else None

predict(BEST_1P7B, "val", adapter=f"runs/{BEST_1P7B}", force=True, out="reports/preds/timing_1p7b_val.pred.jsonl")
if sprint.find_adapter(V1_EP3):
    predict("ep3_360m", "val", adapter=V1_EP3, base=SMALL, force=True, out="reports/preds/timing_360m_val.pred.jsonl")
speed = json.load(open("reports/speed.json"))
t_big = speed.get("timing_1p7b_val.pred.jsonl", {}).get("seconds")
t_small = speed.get("timing_360m_val.pred.jsonl", {}).get("seconds")
COST = {"seconds_300_rows": {"1.7B": t_big, "360M": t_small},
        "adapter_mb": {"1.7B": adapter_mb(f"runs/{BEST_1P7B}"), "360M": adapter_mb(V1_EP3)}}
json.dump(COST, open("reports/cost.json", "w"), indent=2)
print(COST)
""")
code("""
# ---- Phase 5 checks
t, s = final["test"], final["stress"]
p = t["paired_a_minus_b"]
check("5.1", p["exact_match_diff"] >= 10 and p["ci95"][0] > 0,
      f"test {t['metrics']['exact_match']}% vs untouched 1.7B {t['b']['metrics']['exact_match']}%: {p['exact_match_diff']:+} {p['ci95']}")
check("5.2", t["metrics"]["parse_strict"] >= 95 and s["metrics"]["parse_strict"] >= 95,
      f"parse_strict test {t['metrics']['parse_strict']}%, stress {s['metrics']['parse_strict']}%")
if COMPARE:
    for cid, split in [("5.3", "test"), ("5.4", "stress")]:
        r = vs_small[split]; q = r["paired_a_minus_b"]
        verdict = "1.7B better" if q["ci95"][0] > 0 else ("360M better" if q["ci95"][1] < 0 else "no clear difference")
        extra = ("; by group: " + ", ".join(f"{g} {v['paired_a_minus_b']['exact_match_diff']:+.1f} {v['paired_a_minus_b']['ci95']}"
                                             for g, v in groups.items())) if split == "stress" else ""
        check(cid, None, f"{split}: 1.7B {r['metrics']['exact_match']}% vs 360M {r['b']['metrics']['exact_match']}%: "
                         f"{q['exact_match_diff']:+} {q['ci95']} -> {verdict}{extra}")
gap = t["metrics"]["exact_match"] - s["metrics"]["exact_match"]
check("5.5", gap <= 15, f"1.7B gap test - stress = {gap:.1f} points")
check("5.6", lock["run"] == BEST_1P7B and lock["best_frozen_at"] == frozen["frozen_at"],
      f"test first run at {lock['first_test_at']} with {lock['run']}; frozen at {frozen['frozen_at']}")
c = COST
ratio = f"{c['seconds_300_rows']['1.7B'] / c['seconds_300_rows']['360M']:.1f}× slower" if all(c["seconds_300_rows"].values()) else "n/a"
check("5.7", None, f"300 val rows: 1.7B {c['seconds_300_rows']['1.7B']}s vs 360M {c['seconds_300_rows']['360M']}s ({ratio}); "
                   f"adapter 1.7B {c['adapter_mb']['1.7B']} MB vs 360M {c['adapter_mb']['360M']} MB")
sprint.show_errors(BEST_1P7B, "stress", n=10)
""")

# ===================================================================== phase 6
md("""
## Phase 6 — General skills: IFEval and HellaSwag (≈ 60–90 min) — the main goal

Run 1's 360M scored IFEval 33.1 → 31.6 and HellaSwag 56.7 → 57.0 (untouched → tuned). Those are copied from
run 1's folder; this phase measures the untouched and the tuned 1.7B with the **same tasks, metrics and settings**.

**What to expect:** these benchmarks measure what the model learned in **pretraining and general instruction
tuning**. Fine-tuning on booking JSON does not teach them, so the tuned 1.7B should score close to the untouched
1.7B. Any gain over the 360M comes from the bigger base model, not from our fine-tuning. The useful result is
the pair "1.7B is better at general skills **and** still keeps them after our fine-tuning".

| # | Check | Pass when |
|---|---|---|
| 6.1 | Gain over the 360M | tuned 1.7B beats run 1's tuned 360M (`ep_3`) on IFEval **and** HellaSwag, each by more than 1.96 × the combined standard error |
| 6.2 | No forgetting | tuned 1.7B drops ≤ 5 points from the untouched 1.7B on each |
| 6.3 | Where the gain comes from | 👀 base-vs-base (pretraining) and tuned-vs-base (our fine-tuning), for both sizes |
""")
code("""
RUN_LM_EVAL = True   # the main result of this notebook; set False only to skip it for now
if RUN_LM_EVAL:
    !pip -q install "lm-eval[ifeval]"
""")
code("""
import glob, math
TASKS = {"ifeval": ("prompt_level_strict_acc,none", True), "hellaswag": ("acc_norm,none", False)}

def lm_eval(tag, base=BIG, adapter_dir=None, copy_from=None):
    \"\"\"{task: (score %, stderr %)}. copy_from: an earlier run's reports/lm_eval/<tag> prefix to reuse.\"\"\"
    scores = {}
    for task, (metric, chat) in TASKS.items():
        out = f"reports/lm_eval/{tag}_{task}"
        if copy_from and not glob.glob(f"{out}/**/results_*.json", recursive=True) and os.path.isdir(f"{copy_from}_{task}"):
            shutil.copytree(f"{copy_from}_{task}", out, dirs_exist_ok=True); print("copied", f"{copy_from}_{task}")
        if not glob.glob(f"{out}/**/results_*.json", recursive=True):
            if copy_from:
                return None                       # run 1 results not found: nothing to compare with
            args = f"pretrained={base},dtype=bfloat16" + (f",peft={adapter_dir}" if adapter_dir else "")
            cmd = [sys.executable, "-m", "lm_eval", "--model", "hf", "--model_args", args, "--tasks", task,
                   "--batch_size", "auto", "--output_path", out, "--seed", "0"]
            if chat:
                cmd.append("--apply_chat_template")
            sprint.sh(cmd, log=f"logs/lm_eval_{tag}_{task}.log", quiet=True)
        res = json.load(open(sorted(glob.glob(f"{out}/**/results_*.json", recursive=True))[-1]))["results"][task]
        se = res.get(metric.replace(",", "_stderr,"))
        scores[task] = (round(100 * res[metric], 2), round(100 * se, 2) if isinstance(se, (int, float)) else None)
    return scores

if RUN_LM_EVAL:
    LM = {
        "360M untouched": lm_eval("v1_base_360m", copy_from=os.path.join(V1_PROJECT, "reports/lm_eval/base")),
        "360M tuned (ep_3)": lm_eval("v1_ep3_360m", copy_from=os.path.join(V1_PROJECT, "reports/lm_eval/best")),
        "1.7B untouched": lm_eval("base_1p7b"),
        "1.7B tuned": lm_eval("best_1p7b", adapter_dir=sprint.find_adapter(f"runs/{BEST_1P7B}")),
    }
    json.dump(LM, open("reports/lm_eval_summary.json", "w"), indent=2)
    print(f"{'model':20s} " + "  ".join(f"{t:>16s}" for t in TASKS))
    for m, sc in LM.items():
        print(f"{m:20s} " + ("  ".join(f"{sc[t][0]:8.2f} ± {sc[t][1] or float('nan'):5.2f}" for t in TASKS) if sc else "not available"))
""")
code("""
# ---- Phase 6 checks
def z(a, b):
    \"\"\"(difference, z) for two independent scores with standard errors.\"\"\"
    (sa, ea), (sb, eb) = a, b
    se = math.sqrt((ea or 0) ** 2 + (eb or 0) ** 2)
    return round(sa - sb, 2), (round((sa - sb) / se, 2) if se else None)

if RUN_LM_EVAL:
    big, big0, small, small0 = LM["1.7B tuned"], LM["1.7B untouched"], LM["360M tuned (ep_3)"], LM["360M untouched"]
    if small:
        gains = {t: z(big[t], small[t]) for t in TASKS}
        check("6.1", all(d > 0 and zz is not None and zz > 1.96 for d, zz in gains.values()),
              "tuned 1.7B - tuned 360M: " + ", ".join(f"{t} {big[t][0]} vs {small[t][0]} ({d:+}, z={zz})" for t, (d, zz) in gains.items()))
    else:
        check("6.1", None, "run 1's lm-eval results not found in " + V1_PROJECT + "; compare by hand with IFEval 31.61, HellaSwag 57.04")
    drops = {t: round(big0[t][0] - big[t][0], 2) for t in TASKS}
    check("6.2", all(d <= 5 for d in drops.values()),
          "drop (untouched - tuned 1.7B): " + ", ".join(f"{t} {big0[t][0]} -> {big[t][0]} ({-d:+})" for t, d in drops.items()))
    parts = [f"{t}: pretraining (1.7B vs 360M untouched) {z(big0[t], small0[t])[0] if small0 else 'n/a':+}, "
             f"our fine-tuning (1.7B tuned vs untouched) {z(big[t], big0[t])[0]:+}" for t in TASKS]
    check("6.3", None, "; ".join(parts))
""")

# ===================================================================== phase 7
md("### Results summary")
code("""
lm = json.load(open("reports/lm_eval_summary.json")) if os.path.exists("reports/lm_eval_summary.json") else None
bdj = json.load(open("reports/stress_breakdown.json"))
cost = json.load(open("reports/cost.json"))
lines = ["# Results — SmolLM2-1.7B vs 360M on the v2 data", "",
         f"BEST_1P7B: `{BEST_1P7B}` (lr {frozen['lr']}, chosen on val: {frozen['val_all']}); Soup; 3 epochs, LoRA r16, seed 42.", "",
         sprint.table("results_log").to_markdown(index=False), "",
         "## Stress by group", "", "| group | n | 360M | 1.7B | diff (1.7B − 360M) | CI95 |", "|---|---|---|---|---|---|"]
for g, r in bdj["stress_groups"].items():
    q = r.get("paired_a_minus_b")
    small = bdj["ep3_360m_groups"].get(g, {}).get("exact_match", "—")
    lines.append(f"| {g} | {r['n']} | {small} | {r['exact_match']} | {q['exact_match_diff'] if q else '—'} | {q['ci95'] if q else '—'} |")
lines += ["", "## Cost", "", "| | 360M | 1.7B |", "|---|---|---|",
          f"| seconds per 300 rows | {cost['seconds_300_rows']['360M']} | {cost['seconds_300_rows']['1.7B']} |",
          f"| adapter size (MB) | {cost['adapter_mb']['360M']} | {cost['adapter_mb']['1.7B']} |"]
if lm:
    lines += ["", "## General skills (lm-eval; score ± standard error)", "",
              "| model | " + " | ".join(TASKS) + " |", "|---|" + "---|" * len(TASKS)]
    lines += [f"| {m} | " + " | ".join(f"{sc[t][0]} ± {sc[t][1]}" for t in TASKS) + " |" for m, sc in lm.items() if sc]
open("reports/RESULTS_1p7b.md", "w", encoding="utf-8").write("\\n".join(lines))
print("\\n".join(lines))
print()
sprint.checks_summary()
""")
md("""
**Done → Runtime → Disconnect and delete runtime** to stop using compute units.
Everything is in `MyDrive/soup-sprint-1p7b/`. The tuned adapter is `runs/<BEST_1P7B>/`.
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm",
                                                                    "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(HERE, "soup_sprint_1p7b_colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"wrote {out}: {len(cells)} cells")
