"""Build soup_sprint_v3_colab.ipynb: better date data + a calendar in the prompt, 1.7B with the fallback trainer,
compared (paired) with the current release candidate hf_big_lr1e-4.

Scripts are embedded as %%writefile cells (the v3 generator is new). The helper functions predict, lm_eval, z,
adapter_mb and the constants BIG, SMALL, TASKS are loaded from soup_sprint_1p7b_colab.ipynb, not redefined.
Usage: python build_notebook_v3.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = [
    "scripts/common.py", "scripts/generate_data.py", "scripts/generate_data_v2.py", "scripts/generate_data_v3.py",
    "scripts/fetch_replay.py", "scripts/check_data.py", "scripts/check_data_v3.py", "scripts/evaluate.py",
    "scripts/breakdown.py", "scripts/predict.py", "scripts/train_hf.py", "scripts/sprint.py",
    "tests/test_data.py", "tests/test_data_v2.py", "tests/test_data_v3.py", "tests/test_replay.py",
    "tests/test_evaluate.py", "tests/test_predict.py", "tests/test_sprint.py", "tests/test_train_hf.py",
]
cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n")})


# ===================================================================== intro
md("""
# Soup sprint v3: fixing dates (better data + a calendar in the prompt)

**Why:** the release candidate `hf_big_lr1e-4` (1.7B, fallback trainer, v2 data) scores 96.7% on test and 80.3% on
stress, but dates cause all 10 test errors and 38 of its 59 stress errors. Two causes: date wording it never saw
("tmrw", numeric dates, "this Friday") and date arithmetic ("two weeks from today" off by one).

**What changes (v3):**

| | v2 (`hf_big_lr1e-4`) | v3 (`v3_big`, this notebook) |
|---|---|---|
| Prompt | instruction + `Today is <Weekday>, <date>.` + request | the same + **`Calendar:` the next 21 days with weekdays** |
| Date wording in training | 13 forms | **21 forms**: + "on Friday", "this Friday", "in a week", "in two weeks", "on December 7, 2026", "on Monday the 7th", "on 17/12", "on the 17th" |
| Date noise | none (only whole-sentence case changes) | **typos inside date words** ("Decmber", "tomorow") and **shorthand** ("tmr", "Fri"), words ≥ 5 letters |
| Training rows | 3,000 | **4,500**: the same 3,000 (new prompt) + 1,500 date-focused rows |
| Held out for stress | 5 date forms | **9**: + "this coming Friday", "in a fortnight", "on 12/17", "on 17 Dec 2026"; + stress-only shorthand ("2moro", "Thurs") and the keyboard-neighbour typo |
| Model and training | 1.7B, fallback, lr 1e-4, 3 epochs, LoRA r16, seed 42 | **identical** |

**How it is judged:** both models answer the same rows, each with its own prompt format (`*_v2prompt.jsonl` copies
for the v2 model), and are compared as pairs on:
- the **new v3 test and stress sets** (8 new template families, seed 3031);
- the **old v2 test and stress sets**, to make sure nothing got worse. No v3 training row repeats any of their requests.

Everything is written to `MyDrive/soup-sprint-v3/`. Read only: `MyDrive/soup-sprint-1p7b/` (the v2 model, its
predictions and lm-eval results, and `soup_sprint_1p7b_colab.ipynb` for the helper functions).

| Phase | What | GPU time | Checks |
|---|---|---|---|
| 0 | Setup, write scripts, load helpers | — | 0.1–0.3 |
| 1 | Generate v3 data, tests, data checks, token lengths | ~3 min | 1.1–1.8, 1.H, 1.D, 1.C, 1.O, 1.L |
| 2 | Untouched 1.7B on val; train `v3_big`; val comparison | ~45–70 min | 2.0–2.4 |
| 3 | Test, stress, old test, old stress: once | ~8 min | 3.1–3.9 |
| 4 | IFEval and HellaSwag for `v3_big` | ~30–45 min | 4.1–4.2 |
| 5 | Results and release recommendation | — | — |
""")

# ===================================================================== phase 0
md("""
## Phase 0 — Setup (≈ 10 min)

| # | Check | Pass when |
|---|---|---|
| 0.1 | GPU | bf16 GPU with ≥ 20 GB (A100 recommended) |
| 0.2 | v2 model | `hf_big_lr1e-4` adapter found in `MyDrive/soup-sprint-1p7b/runs/` |
| 0.3 | Helpers | the 7 definitions were loaded from `soup_sprint_1p7b_colab.ipynb` |
""")
code("""
!nvidia-smi
import sys; print("python", sys.version)
""")
code("""
import os
from google.colab import drive
drive.mount("/content/drive")

PROJECT = "/content/drive/MyDrive/soup-sprint-v3"
P1P7B = "/content/drive/MyDrive/soup-sprint-1p7b"          # read only: the v2 model and its results
for d in ["scripts", "tests", "data/eval", "configs", "runs", "reports/preds", "logs"]:
    os.makedirs(os.path.join(PROJECT, d), exist_ok=True)
os.chdir(PROJECT)
print("working in", os.getcwd(), "| 1.7B folder found:", os.path.isdir(P1P7B))
""")
code("""
# ~3-5 min; same packages as the earlier runs
!pip -q install "soup-cli[train,eval]" pyyaml pytest tabulate "lm-eval[ifeval]"
!pip uninstall -y -q torchao
""")
md("### Write the project files to Drive")
for rel in FILES:
    with open(os.path.join(HERE, rel), encoding="utf-8") as f:
        body = f.read()
    code(f"%%writefile {rel}\n{body}")
code("""
# Load the project's modules, then the helper functions from soup_sprint_1p7b_colab.ipynb (re-run after a reconnect)
import ast, glob, importlib, json, math, os, re, shutil, subprocess, sys, time
os.chdir(PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "scripts"))
import sprint; importlib.reload(sprint)
import evaluate as ev; importlib.reload(ev)
import breakdown as bd; importlib.reload(bd)
from sprint import check

HELPERS = ["BIG", "SMALL", "TASKS", "predict", "lm_eval", "z", "adapter_mb"]
NB_NAME = "soup_sprint_1p7b_colab.ipynb"

def find_notebook():
    for p in [os.path.join(P1P7B, NB_NAME), f"/content/drive/MyDrive/Colab Notebooks/{NB_NAME}", f"/content/{NB_NAME}"]:
        if os.path.exists(p):
            return p
    hits = glob.glob(f"/content/drive/MyDrive/**/{NB_NAME}", recursive=True)
    return hits[0] if hits else None

def load_from_notebook(path, names):
    \"\"\"Run only the top-level `def`s and assignments called `names` from the notebook's code cells.\"\"\"
    found = {}
    for c in json.load(open(path, encoding="utf-8"))["cells"]:
        if c["cell_type"] != "code":
            continue
        src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        if src.lstrip().startswith("%%"):
            continue
        src = "\\n".join(re.sub(r"^(\\s*)[!%].*", r"\\1pass", l) for l in src.splitlines())
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                found[node.name] = node
            elif (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                  and node.targets[0].id in names):
                found[node.targets[0].id] = node
    missing = sorted(set(names) - set(found))
    if missing:
        raise RuntimeError(f"not found in {path}: {missing}")
    order = [n for n in names if not isinstance(found[n], ast.FunctionDef)] + [n for n in names if isinstance(found[n], ast.FunctionDef)]
    exec(compile(ast.Module(body=[found[n] for n in order], type_ignores=[]), path, "exec"), globals())
    return order

NB_1P7B = find_notebook()
assert NB_1P7B, f"put {NB_NAME} in {P1P7B} or MyDrive/Colab Notebooks and re-run this cell"
LOADED = load_from_notebook(NB_1P7B, HELPERS)
print("loaded from", NB_1P7B, ":", LOADED)

V2_RUN = "v2_big"                                              # the release candidate, under a short name here
V2_ADAPTER = os.path.join(P1P7B, "runs", "hf_big_lr1e-4")
V3_RUN = "v3_big"

def pair(gold_split, a, a_split, b, b_split):
    \"\"\"Paired report: model a (its predictions on a_split) vs model b (on b_split); same rows, gold from gold_split.\"\"\"
    gold = ev.load_jsonl(f"data/eval/{gold_split}.jsonl")
    rep, errors = ev.evaluate(gold, ev.load_preds(sprint.pred_path(a, a_split)), ev.load_preds(sprint.pred_path(b, b_split)))
    with open(f"reports/{a}_{gold_split}.errors.jsonl", "w", encoding="utf-8") as f:   # read by sprint.show_errors
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\\n")
    return rep

def date_pair(gold_split, a, a_split, b, b_split):
    \"\"\"Date-field accuracy on rows that state a date: a, b and the paired difference with CI95.\"\"\"
    gold = [r for r in ev.load_jsonl(f"data/eval/{gold_split}.jsonl") if r["label"]["date"] is not None]
    pa, pb = ev.load_preds(sprint.pred_path(a, a_split)), ev.load_preds(sprint.pred_path(b, b_split))
    ca = [float(ev.score_row(pa.get(r["id"], ""), r["label"])["field_date"]) for r in gold]
    cb = [float(ev.score_row(pb.get(r["id"], ""), r["label"])["field_date"]) for r in gold]
    d = [x - y for x, y in zip(ca, cb)]
    lo, hi = ev.bootstrap_ci(d)
    return {"n": len(d), "a": ev.pct(sum(ca) / len(ca)), "b": ev.pct(sum(cb) / len(cb)),
            "diff": ev.pct(sum(d) / len(d)), "ci95": [ev.pct(lo), ev.pct(hi)]}
""")
code("""
# ---- Phase 0 checks
import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
mem = torch.cuda.get_device_properties(0).total_memory / 2**30 if gpu else 0
check("0.1", bool(gpu) and torch.cuda.is_bf16_supported() and mem >= 20, f"gpu={gpu}, {mem:.0f} GB", hard=True)
check("0.2", sprint.find_adapter(V2_ADAPTER) is not None, f"v2 adapter: {sprint.find_adapter(V2_ADAPTER)}", hard=True)
check("0.3", set(LOADED) == set(HELPERS), f"{len(LOADED)} helpers from {NB_1P7B}", hard=True)
""")

# ===================================================================== phase 1
md("""
## Phase 1 — v3 data (≈ 3 min)

`scripts/generate_data_v3.py` (seed 3031) writes `data/train.jsonl` (4,500 rows), `data/eval/{val,test,stress}.jsonl`
(new), `data/eval/old_{test,stress}.jsonl` (v2's sets with the calendar prompt) and a `*_v2prompt.jsonl` copy of
each eval set for the v2 model. The calendar rule: "next / this / on Friday" = the first Friday after today.

| # | Check | Pass when |
|---|---|---|
| 1.1 | Tests | all `pytest` tests pass |
| 1.2–1.6 | Structure, labels, leakage, coverage, checksums | as in v2; no v3 row repeats any v2 request |
| 1.H | Held out | held-out wording, neighbour typos, stress fillers and stress shorthand appear only in stress |
| 1.D | Date coverage | every training date form ≥ 10 rows; every held-out form in stress; date noise in ≥ 200 training rows |
| 1.C | Calendar | every prompt's calendar is right (21 days, weekdays); the label date is listed whenever it falls inside it |
| 1.O | Old sets | `old_test` / `old_stress` are exactly v2's rows |
| 1.L | Length | the longest training example fits in `max_length` 512 tokens (nothing is cut off) |
| 1.8 | Your eyes | the printed rows have obviously right dates |
""")
code("""
!python scripts/generate_data_v3.py --out data --seed 3031
!python scripts/fetch_replay.py --out data/replay.jsonl --n 500
""")
code("""
# ---- 1.1 tests and the data checks
rc, out = sprint.sh([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], check_rc=False, quiet=True)
check("1.1", rc == 0, f"pytest exit {rc}; {out.strip().splitlines()[-1]}", hard=True)
rc, out = sprint.sh([sys.executable, "scripts/check_data_v3.py", "--data", "data", "--reports", "reports"], check_rc=False)
for r in json.load(open("reports/data_checks.json")):
    check(r["id"], r["ok"], f"{r['name']}: {r['detail']}")
check("1.x", rc == 0, "all data checks passed" if rc == 0 else "data checks failed — fix before training", hard=True)
""")
code("""
# ---- 1.L token lengths (the calendar makes prompts longer)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(BIG)
train_rows = ev.load_jsonl("data/train.jsonl")
n_tok = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])
lengths = [n_tok(tok.apply_chat_template(r["messages"], tokenize=False)) for r in train_rows]
v2_prompt_len = [len(tok(r["prompt"])["input_ids"]) for r in ev.load_jsonl("data/eval/test_v2prompt.jsonl")]
v3_prompt_len = [len(tok(r["prompt"])["input_ids"]) for r in ev.load_jsonl("data/eval/test.jsonl")]
LEN = {"max_train_tokens": max(lengths), "mean_prompt_v2": round(sum(v2_prompt_len) / len(v2_prompt_len)),
       "mean_prompt_v3": round(sum(v3_prompt_len) / len(v3_prompt_len))}
json.dump(LEN, open("reports/token_lengths.json", "w"), indent=2)
check("1.L", LEN["max_train_tokens"] <= 512,
      f"longest training example {LEN['max_train_tokens']} tokens (limit 512); mean prompt {LEN['mean_prompt_v2']} -> "
      f"{LEN['mean_prompt_v3']} tokens with the calendar", hard=True)
""")
code("""
# ---- 1.8 your eyes: read the rows printed by the data-check cell above.
LABELS_LOOK_RIGHT = None
check("1.8", LABELS_LOOK_RIGHT, "rows read by eye (dates in particular)")
""")

# ===================================================================== phase 2
md("""
## Phase 2 — Train `v3_big` (≈ 45–70 min)

Same settings as `hf_big_lr1e-4`: SmolLM2-1.7B-Instruct, fallback trainer, lr 1e-4, 3 epochs, LoRA r16/α32 on all
linear layers, seed 42, batch 16, cosine, 10% replay. Only the data (and so the prompt) differs. Nothing is chosen
in this notebook, so val is only for looking.

| # | Check | Pass when |
|---|---|---|
| 2.0 | Baseline | the untouched 1.7B scored on v3 val (with the calendar prompt) |
| 2.1 | Trainer | `v3_big` trained with backend `hf` |
| 2.2 | Curves | training loss finite and falling; validation loss logged |
| 2.3 | Learning | `v3_big` beats the untouched 1.7B on val, paired CI95 above 0 |
| 2.4 | Val vs v2 | 👀 `v3_big` vs `v2_big` on val (paired, each with its own prompt), by group, and date accuracy |
""")
code("""
predict("base_1p7b", "val")
base_val = sprint.score("base_1p7b", "val")
check("2.0", base_val["metrics"]["parse_lenient"] > 0, f"untouched 1.7B on v3 val: exact_match {base_val['metrics']['exact_match']}%")
""")
code("""
sprint.ALLOW_HF_FALLBACK = False
sprint.set_backend("hf")
cfg_path, CFG = sprint.make_config(V3_RUN, {"base": BIG, "data": {"train": "./data/train.jsonl", "replay": "./data/replay.jsonl"},
                                             "training": {"lr": 1e-4, "epochs": 3, "seed": 42,
                                                          "lora": {"r": 16, "alpha": 32, "dropout": 0.05}}})
print(open(cfg_path).read())
META = sprint.train(V3_RUN, cfg_path, CFG)        # skipped if the adapter is already on Drive
print(META)
sprint.plot_losses([V3_RUN], title="v3_big", save_as="loss_v3.png")
""")
code("""
# ---- val: v3_big (calendar prompt) vs v2_big (its own prompt), same rows
predict(V3_RUN, "val", adapter=f"runs/{V3_RUN}")
predict(V2_RUN, "val_v2prompt", adapter=V2_ADAPTER)
VAL_BASE = sprint.score(V3_RUN, "val", vs="base_1p7b")
VAL = pair("val", V3_RUN, "val", V2_RUN, "val_v2prompt")
VAL_G = bd.by_group(ev.load_jsonl("data/eval/val.jsonl"), ev.load_preds(sprint.pred_path(V3_RUN, "val")),
                    ev.load_preds(sprint.pred_path(V2_RUN, "val_v2prompt")))
VAL_D = date_pair("val", V3_RUN, "val", V2_RUN, "val_v2prompt")
h = sprint.loss_history(f"runs/{V3_RUN}")
tr = [l for _, l in h["train"]]; va = [l for _, l in h["eval"]]
k = max(1, len(tr) // 4)
check("2.1", META.get("backend") == "hf", f"backend={META.get('backend')}, {META.get('seconds')} s")
check("2.2", len(tr) > 1 and all(math.isfinite(l) for l in tr) and sum(tr[-k:]) / k < sum(tr[:k]) / k and len(va) > 0,
      f"train {sum(tr[:k]) / k:.3f} -> {sum(tr[-k:]) / k:.3f}; val loss min {min(va) if va else None}")
p = VAL_BASE["paired_a_minus_b"]
check("2.3", p["ci95"][0] > 0, f"v3_big {VAL_BASE['metrics']['exact_match']}% vs untouched: {p['exact_match_diff']:+} {p['ci95']}")
q = VAL["paired_a_minus_b"]
check("2.4", None, f"val: v3 {VAL['metrics']['exact_match']}% vs v2 {VAL['b']['metrics']['exact_match']}% = {q['exact_match_diff']:+} {q['ci95']}; "
      + ", ".join(f"{g} {r['paired_a_minus_b']['exact_match_diff']:+.1f}" for g, r in VAL_G.items())
      + f"; dates {VAL_D['b']}% -> {VAL_D['a']}% ({VAL_D['diff']:+} {VAL_D['ci95']})")
""")

# ===================================================================== phase 3
md("""
## Phase 3 — Test and stress, once (≈ 8 min)

**Acceptance checks, set before seeing any results** (all paired: `v3_big` − `v2_big` on the same rows):

| # | Check | Pass when |
|---|---|---|
| 3.1 | Dates fixed | v3 stress, rows that state a date: date accuracy difference with CI95 **above 0** |
| 3.2 | Stress better | v3 stress exact match: CI95 **above 0** |
| 3.3 | Test not worse | v3 test exact match: difference ≥ −2 points |
| 3.4 | Old sets not worse | old test ≥ −2 points and old stress ≥ −3 points (v2's own sets, where v2 had the home advantage) |
| 3.5 | Format | `v3_big` parse_strict ≥ 95% on all four sets |
| 3.6 | Gap | `v3_big`: v3 test − v3 stress ≤ 15 points |
| 3.7 | Honest process | test and stress run once; lock written before |
| 3.8 | Per date form | 👀 date accuracy per form (and "+noise") on v3 test and stress, both models |
| 3.9 | Cost | 👀 prompt length and seconds per 300 rows, v2 vs v3 |

**Release rule (set now):** publish `v3_big` if 3.1, 3.2, 3.3, 3.4 and 3.5 pass and Phase 4 shows no forgetting;
otherwise keep `hf_big_lr1e-4`.
""")
code("""
lock_path = "reports/test_lock_v3.json"
if not os.path.exists(lock_path):
    json.dump({"models": [V3_RUN, V2_RUN], "first_test_at": time.strftime("%Y-%m-%d %H:%M:%S")}, open(lock_path, "w"), indent=2)
LOCK = json.load(open(lock_path)); print("lock:", LOCK)

for s in ["test", "stress", "old_test", "old_stress"]:
    predict(V3_RUN, s, adapter=f"runs/{V3_RUN}")
for s in ["test", "stress"]:
    predict(V2_RUN, f"{s}_v2prompt", adapter=V2_ADAPTER)
for s in ["test", "stress"]:                     # v2_big's answers on its own sets already exist in the 1.7B folder
    src = os.path.join(P1P7B, "reports", "preds", f"hf_big_lr1e-4_{s}.pred.jsonl")
    dst = sprint.pred_path(V2_RUN, f"old_{s}_v2prompt")
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy(src, dst); print("copied", src)
    predict(V2_RUN, f"old_{s}_v2prompt", adapter=V2_ADAPTER)   # skipped when copied

CMP, DATES = {}, {}
for s in ["test", "stress", "old_test", "old_stress"]:
    CMP[s] = pair(s, V3_RUN, s, V2_RUN, f"{s}_v2prompt")
    DATES[s] = date_pair(s, V3_RUN, s, V2_RUN, f"{s}_v2prompt")
    m, b, q = CMP[s]["metrics"], CMP[s]["b"]["metrics"], CMP[s]["paired_a_minus_b"]
    print(f"{s:11s} v3 {m['exact_match']:6.2f}% {m['exact_match_ci95']}  v2 {b['exact_match']:6.2f}%  diff {q['exact_match_diff']:+.2f} {q['ci95']}"
          f"  | dates (n={DATES[s]['n']}) v2 {DATES[s]['b']}% -> v3 {DATES[s]['a']}% ({DATES[s]['diff']:+} {DATES[s]['ci95']})")
    sprint.log_result(f"FINAL v3 vs v2", s, {**CMP[s], "vs": V2_RUN}, notes="paired; each model with its own prompt")
json.dump({"exact_match": CMP, "dates": DATES}, open("reports/compare_v3.json", "w"), indent=2)
""")
code("""
# ---- breakdowns: stress groups, date by wording tier, date by form
def preds(name, split):
    return ev.load_preds(sprint.pred_path(name, split))

gold_st, gold_te = ev.load_jsonl("data/eval/stress.jsonl"), ev.load_jsonl("data/eval/test.jsonl")
G = bd.by_group(gold_st, preds(V3_RUN, "stress"), preds(V2_RUN, "stress_v2prompt"))
G2 = bd.by_group(gold_st, preds(V2_RUN, "stress_v2prompt"))
print(f"  {'stress group':12s} {'n':>4s} {'v2':>7s} {'v3':>7s}   diff [CI95]")
for g, r in G.items():
    q = r["paired_a_minus_b"]
    print(f"  {g:12s} {r['n']:4d} {G2[g]['exact_match']:6.1f}% {r['exact_match']:6.1f}%   {q['exact_match_diff']:+.1f} {q['ci95']}")
TIER = {n: bd.field_by_tier(gold_st, p) for n, p in [(V2_RUN, preds(V2_RUN, "stress_v2prompt")), (V3_RUN, preds(V3_RUN, "stress"))]}
print("\\n  stress field/tier      n      v2      v3")
for k_ in TIER[V3_RUN]:
    print(f"  {k_:18s} {TIER[V3_RUN][k_]['n']:4d} {TIER[V2_RUN][k_]['accuracy']:6.1f}% {TIER[V3_RUN][k_]['accuracy']:6.1f}%")
FORM = {}
for split, gold, v2s in [("test", gold_te, "test_v2prompt"), ("stress", gold_st, "stress_v2prompt")]:
    FORM[split] = {"v2": bd.field_by_form(gold, preds(V2_RUN, v2s)), "v3": bd.field_by_form(gold, preds(V3_RUN, split))}
    print(f"\\n  date accuracy by form — {split}")
    for f_, r in FORM[split]["v3"].items():
        print(f"    {f_:34s} n={r['n']:3d}  v2 {FORM[split]['v2'][f_]['accuracy']:6.1f}%  v3 {r['accuracy']:6.1f}%")
json.dump({"stress_groups": G, "stress_groups_v2": G2, "tiers": TIER, "date_forms": FORM},
          open("reports/breakdown_v3.json", "w"), indent=2)
""")
code("""
# ---- Phase 3 checks
def d_(s): return CMP[s]["paired_a_minus_b"]
check("3.1", DATES["stress"]["ci95"][0] > 0,
      f"stress dates: v2 {DATES['stress']['b']}% -> v3 {DATES['stress']['a']}% ({DATES['stress']['diff']:+} {DATES['stress']['ci95']}); "
      f"test dates: {DATES['test']['b']}% -> {DATES['test']['a']}% ({DATES['test']['diff']:+} {DATES['test']['ci95']})")
check("3.2", d_("stress")["ci95"][0] > 0, f"stress: {d_('stress')['exact_match_diff']:+} {d_('stress')['ci95']}")
check("3.3", d_("test")["exact_match_diff"] >= -2, f"test: {d_('test')['exact_match_diff']:+} {d_('test')['ci95']}")
check("3.4", d_("old_test")["exact_match_diff"] >= -2 and d_("old_stress")["exact_match_diff"] >= -3,
      f"old test: {d_('old_test')['exact_match_diff']:+} {d_('old_test')['ci95']}; old stress: {d_('old_stress')['exact_match_diff']:+} {d_('old_stress')['ci95']}")
check("3.5", all(CMP[s]["metrics"]["parse_strict"] >= 95 for s in CMP),
      "parse_strict: " + ", ".join(f"{s} {CMP[s]['metrics']['parse_strict']}%" for s in CMP))
gap = CMP["test"]["metrics"]["exact_match"] - CMP["stress"]["metrics"]["exact_match"]
check("3.6", gap <= 15, f"v3 test - v3 stress = {gap:.1f} points (v2 on the same sets: "
      f"{CMP['test']['b']['metrics']['exact_match'] - CMP['stress']['b']['metrics']['exact_match']:.1f})")
check("3.7", set(LOCK["models"]) == {V3_RUN, V2_RUN}, f"test first run at {LOCK['first_test_at']}")
worst = sorted(((r["accuracy"], f_) for f_, r in FORM["stress"]["v3"].items() if r["n"] >= 5))[:4]
check("3.8", None, "weakest stress date forms for v3: " + ", ".join(f"{f_} {a}%" for a, f_ in worst))
speed = json.load(open("reports/speed.json")) if os.path.exists("reports/speed.json") else {}
sec = lambda f: speed.get(f, {}).get("seconds")
check("3.9", None, f"mean prompt {LEN['mean_prompt_v2']} -> {LEN['mean_prompt_v3']} tokens; 300 test rows: "
      f"v2 {sec(f'{V2_RUN}_test_v2prompt.pred.jsonl')}s, v3 {sec(f'{V3_RUN}_test.pred.jsonl')}s; "
      f"adapter {adapter_mb(f'runs/{V3_RUN}')} MB")
sprint.show_errors(V3_RUN, "stress", n=12)
""")

# ===================================================================== phase 4
md("""
## Phase 4 — IFEval and HellaSwag (≈ 30–45 min)

Only `v3_big` is evaluated; the untouched 1.7B and `v2_big` are copied from the 1.7B folder.

| # | Check | Pass when |
|---|---|---|
| 4.1 | No forgetting | `v3_big` drops ≤ 5 points from the untouched 1.7B on each |
| 4.2 | vs v2 | 👀 `v3_big` vs `v2_big` on each |
""")
code("""
RUN_LM_EVAL = True
if RUN_LM_EVAL:
    LM = {"1.7B untouched": lm_eval("base_1p7b", copy_from=os.path.join(P1P7B, "reports/lm_eval/base_1p7b")),
          "v2_big (hf_big_lr1e-4)": lm_eval("hf_best_1p7b", copy_from=os.path.join(P1P7B, "reports/lm_eval/hf_best_1p7b")),
          "v3_big": lm_eval("v3_big", adapter_dir=sprint.find_adapter(f"runs/{V3_RUN}"))}
    json.dump(LM, open("reports/lm_eval_summary_v3.json", "w"), indent=2)
    for m, sc in LM.items():
        print(f"{m:24s} " + ("  ".join(f"{t} {sc[t][0]} ± {sc[t][1]}" for t in TASKS) if sc else "not available"))
    base17, new, old = LM["1.7B untouched"], LM["v3_big"], LM["v2_big (hf_big_lr1e-4)"]
    drops = {t: round(base17[t][0] - new[t][0], 2) for t in TASKS} if base17 else {}
    check("4.1", bool(drops) and all(d <= 5 for d in drops.values()),
          ", ".join(f"{t} {base17[t][0]} -> {new[t][0]} ({-d:+})" for t, d in drops.items()) if drops else "base results missing")
    check("4.2", None, ", ".join(f"{t}: v3 - v2 {z(new[t], old[t])[0]:+} (z={z(new[t], old[t])[1]})" for t in TASKS) if old else "v2 results missing")
""")

# ===================================================================== phase 5
md("### Results and release recommendation")
code("""
checks_all = json.load(open("reports/checks.json"))
need = ["3.1", "3.2", "3.3", "3.4", "3.5", "4.1"]
release = all(checks_all.get(c, {}).get("status") == "PASS" for c in need)
lines = ["# Results v3 — date fixes (calendar prompt + wider date data)", "",
         f"Both models: SmolLM2-1.7B-Instruct, fallback trainer, lr 1e-4, 3 epochs, LoRA r16, seed 42. v2_big = hf_big_lr1e-4.", "",
         "| set | v2_big | v3_big | paired diff [CI95] | dates v2 → v3 [CI95 of diff] |", "|---|---|---|---|---|"]
for s in ["test", "stress", "old_test", "old_stress"]:
    m, b, q, d = CMP[s]["metrics"], CMP[s]["b"]["metrics"], CMP[s]["paired_a_minus_b"], DATES[s]
    lines.append(f"| {s} | {b['exact_match']}% | {m['exact_match']}% | {q['exact_match_diff']:+} {q['ci95']} | "
                 f"{d['b']}% → {d['a']}% ({d['diff']:+} {d['ci95']}) |")
lines += ["", "| stress group | n | v2 | v3 | diff [CI95] |", "|---|---|---|---|---|"]
lines += [f"| {g} | {r['n']} | {G2[g]['exact_match']} | {r['exact_match']} | {r['paired_a_minus_b']['exact_match_diff']:+} {r['paired_a_minus_b']['ci95']} |"
          for g, r in G.items()]
lines += ["", f"Prompt length: {LEN['mean_prompt_v2']} → {LEN['mean_prompt_v3']} tokens (mean).", ""]
if RUN_LM_EVAL:
    lines += ["| model | " + " | ".join(TASKS) + " |", "|---|" + "---|" * len(TASKS)]
    lines += [f"| {m} | " + " | ".join(f"{sc[t][0]} ± {sc[t][1]}" for t in TASKS) + " |" for m, sc in LM.items() if sc]
lines += ["", "**Release:** " + ("publish `v3_big` (it needs the calendar prompt)" if release else
                                "keep `hf_big_lr1e-4`: " + ", ".join(c for c in need if checks_all.get(c, {}).get("status") != "PASS") + " did not pass")]
open("reports/RESULTS_v3.md", "w", encoding="utf-8").write("\\n".join(lines))
print("\\n".join(lines)); print()
sprint.checks_summary()
""")
md("""
**Done → Runtime → Disconnect and delete runtime.** Results: `MyDrive/soup-sprint-v3/reports/RESULTS_v3.md`;
adapter: `runs/v3_big/`.
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(HERE, "soup_sprint_v3_colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"wrote {out}: {len(cells)} cells")
