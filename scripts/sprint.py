"""Notebook helpers: shell runs, PASS/FAIL checks, configs, training, prediction, scoring, result tables.

Import in the notebook after `os.chdir(PROJECT)`:  import sys; sys.path.insert(0, "scripts"); import sprint
Everything is written under the project folder (on Google Drive in Colab), so a disconnect loses nothing.
"""
from __future__ import annotations

import copy
import csv
import glob
import json
import os
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev  # noqa: E402
from predict import find_adapter  # noqa: E402

BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
ALLOW_HF_FALLBACK = True   # if `soup train` fails in Phase 3, train with scripts/train_hf.py instead
PREDICT_LIMIT = None       # rows per split for predict(); None = all (only lower it for local tests)
PREDICT_BATCH = 32

# configs/sft.yaml from the sprint plan (Phase 4)
SFT_CONFIG = {
    "base": BASE,
    "task": "sft",
    "data": {
        "train": "./data/train.jsonl",
        "format": "chatml",
        "val_split": 0.1,
        "max_length": 512,
        "replay": "./data/replay.jsonl",
        "replay_ratio": 0.1,
    },
    "training": {
        "epochs": 2,
        "lr": 1.0e-4,
        "batch_size": 16,
        "gradient_accumulation_steps": 1,
        "seed": 42,
        "eval_steps": 25,
        "scheduler": "cosine",
        "warmup_ratio": 0.03,
        "quantization": "none",
        "auto_mixed_precision": True,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05},
    },
    "output": "./runs/sft_v1",
}


def P(*parts: str) -> str:
    """Path inside the project folder (the current working directory)."""
    return os.path.normpath(os.path.join(os.getcwd(), *parts))


# ---------------------------------------------------------------- checks
class CheckFailed(AssertionError):
    pass


def _load_checks() -> dict:
    path = P("reports", "checks.json")
    return json.load(open(path)) if os.path.exists(path) else {}


def check(cid: str, ok, detail: str = "", hard: bool = False) -> bool:
    """Record a check. ok=True/False -> PASS/FAIL; ok=None -> REVIEW (needs your judgement).
    hard=True raises on FAIL so the notebook stops before building on a broken step."""
    status = "REVIEW" if ok is None else ("PASS" if ok else "FAIL")
    icon = {"PASS": "✅", "FAIL": "❌", "REVIEW": "👀"}[status]
    print(f"{icon} {status:6s} {cid:5s} {detail}")
    checks = _load_checks()
    checks[cid] = {"status": status, "detail": detail, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(P("reports"), exist_ok=True)
    with open(P("reports", "checks.json"), "w") as f:
        json.dump(checks, f, indent=2)
    if hard and status == "FAIL":
        raise CheckFailed(f"{cid}: {detail}")
    return bool(ok)


def checks_summary() -> None:
    checks = _load_checks()
    for cid in sorted(checks, key=lambda c: [(0, int(x), "") if x.isdigit() else (1, 0, x) for x in c.split(".")]):
        c = checks[cid]
        print(f"{c['status']:6s} {cid:5s} {c['detail'][:110]}")
    counts = {s: sum(c["status"] == s for c in checks.values()) for s in ("PASS", "FAIL", "REVIEW")}
    print(counts)


# ---------------------------------------------------------------- shell
def sh(cmd, log: str | None = None, check_rc: bool = True, quiet: bool = False) -> tuple[int, str]:
    """Run a command in the project folder, stream its output, optionally tee to a log file."""
    shell = isinstance(cmd, str)
    print("$", cmd if shell else " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, cwd=os.getcwd(), env={**os.environ, "PYTHONUNBUFFERED": "1"})
    lines = []
    logf = open(log, "a", encoding="utf-8") if log else None
    for line in proc.stdout:
        lines.append(line)
        if not quiet:
            print(line, end="")
        if logf:
            logf.write(line)
    proc.wait()
    if logf:
        logf.close()
    out = "".join(lines)
    if check_rc and proc.returncode != 0:
        raise RuntimeError(f"command failed (exit {proc.returncode}):\n" + "".join(lines[-40:]))
    return proc.returncode, out


_HELP: dict[str, str] = {}


def soup_help(*sub: str) -> str:
    key = " ".join(sub)
    if key not in _HELP:
        _, _HELP[key] = sh(["soup", *sub, "--help"], check_rc=False, quiet=True)
    return _HELP[key]


def soup_supports(flag: str, *sub: str) -> bool:
    return flag in soup_help(*sub)


# ---------------------------------------------------------------- configs
def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def make_config(name: str, overrides: dict | None = None, base_cfg: dict | None = None) -> tuple[str, dict]:
    import yaml
    cfg = deep_merge(base_cfg or SFT_CONFIG, overrides or {})
    cfg["output"] = f"./runs/{name}"
    os.makedirs(P("configs"), exist_ok=True)
    path = os.path.join("configs", f"{name}.yaml")
    with open(P(path), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path, cfg


def settings(cfg: dict) -> dict:
    t = cfg["training"]
    return {"lr": t["lr"], "epochs": t["epochs"], "rank": t["lora"]["r"], "seed": t["seed"]}


# ---------------------------------------------------------------- training
def get_backend() -> str:
    path = P("reports", "backend.txt")
    return open(path).read().strip() if os.path.exists(path) else "soup"


def set_backend(name: str) -> None:
    os.makedirs(P("reports"), exist_ok=True)
    with open(P("reports", "backend.txt"), "w") as f:
        f.write(name)


def dry_run(cfg_path: str) -> tuple[str, int, str]:
    """Validate a config without training: `soup train --dry-run` if it exists, else `soup plan`."""
    if soup_supports("--dry-run", "train"):
        cmd = ["soup", "train", "--config", cfg_path, "--dry-run"]
    elif "Usage" in soup_help("plan"):
        cmd = ["soup", "plan", "--config", cfg_path]
    else:
        return "none", 1, "neither `soup train --dry-run` nor `soup plan` is available"
    rc, out = sh(cmd, check_rc=False)
    return " ".join(cmd[:3]), rc, out


def train_cmd(cfg_path: str, backend: str) -> list[str]:
    if backend == "hf":
        return [sys.executable, "scripts/train_hf.py", "--config", cfg_path]
    cmd = ["soup", "train", "--config", cfg_path]
    if soup_supports("--yes", "train"):
        cmd.append("--yes")  # notebook cells cannot answer an interactive confirmation
    if soup_supports("--tensorboard", "train"):
        cmd.append("--tensorboard")
    return cmd


def train(name: str, cfg_path: str, cfg: dict, force: bool = False) -> dict:
    """Train unless an adapter for this run is already on Drive. Returns run metadata."""
    out = P(cfg["output"])
    meta_path = os.path.join(out, "sprint_meta.json")
    if not force and find_adapter(out):
        print(f"skip training {name}: adapter already at {find_adapter(out)}")
        return json.load(open(meta_path)) if os.path.exists(meta_path) else {"name": name, "skipped": True}
    backend = get_backend()
    os.makedirs(P("logs"), exist_ok=True)
    log = P("logs", f"{name}.log")
    t0 = time.time()
    rc, _ = sh(train_cmd(cfg_path, backend), log=log, check_rc=False)
    if rc != 0:
        raise RuntimeError(f"training {name} failed with backend={backend} (exit {rc}); see {log}")
    adapter = find_adapter(out)
    if adapter is None:
        raise RuntimeError(f"training {name} finished but no adapter_config.json under {out}; ls: {os.listdir(out)}")
    meta = {"name": name, "backend": backend, "seconds": round(time.time() - t0, 1), "adapter": adapter,
            **settings(cfg)}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def loss_history(out_dir: str) -> dict:
    """Training / validation loss from trainer_state.json (top level or newest checkpoint), else rewind.jsonl."""
    out_dir = P(out_dir) if not os.path.isabs(out_dir) else out_dir
    states = glob.glob(os.path.join(out_dir, "**", "trainer_state.json"), recursive=True)
    if states:
        def rank(p):
            rel = os.path.relpath(p, out_dir)
            step = next((int(x.split("-")[1]) for x in rel.split(os.sep) if x.startswith("checkpoint-")), 10**9)
            return step
        path = max(states, key=rank)
        st = json.load(open(path))
        hist = st.get("log_history", [])
        res = {"source": path, "global_step": st.get("global_step"),
               "train": [(h["step"], h["loss"]) for h in hist if "loss" in h],
               "eval": sorted({h["step"]: h["eval_loss"] for h in hist if "eval_loss" in h}.items()),
               "train_runtime": next((h["train_runtime"] for h in hist if "train_runtime" in h), None)}
        return res
    rewind = os.path.join(out_dir, "rewind.jsonl")
    if os.path.exists(rewind):
        rows = [json.loads(line) for line in open(rewind) if line.strip()]
        losses = [r["loss"] for r in rows if isinstance(r.get("loss"), (int, float))]
        return {"source": rewind, "global_step": len(losses), "train": list(enumerate(losses, 1)),
                "eval": [], "train_runtime": None}
    return {"source": None, "global_step": None, "train": [], "eval": [], "train_runtime": None}


def plot_losses(names: list[str], title: str = "loss", save_as: str | None = None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for n in names:
        h = loss_history(os.path.join("runs", n))
        if h["train"]:
            ax[0].plot(*zip(*h["train"]), label=n)
        if h["eval"]:
            ax[1].plot(*zip(*h["eval"]), marker="o", label=n)
    for a, t in zip(ax, ["training loss", "validation loss"]):
        a.set_title(t); a.set_xlabel("step"); a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    if save_as:
        fig.savefig(P("reports", save_as), dpi=120)
    plt.show()
    return fig


# ---------------------------------------------------------------- prediction and scoring
def pred_path(name: str, split: str) -> str:
    return P("reports", "preds", f"{name}_{split}.pred.jsonl")


def predict(name: str, split: str, adapter_run: str | None = None, force: bool = False, out: str | None = None) -> str:
    """Greedy answers for data/eval/<split>.jsonl. adapter_run=None means the base model."""
    out = out or pred_path(name, split)
    if os.path.exists(out) and not force:
        print(f"skip predict {name}/{split}: {out} exists")
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [sys.executable, "scripts/predict.py", "--input", f"data/eval/{split}.jsonl", "--output", out,
           "--batch-size", str(PREDICT_BATCH)]
    if adapter_run:
        cmd += ["--adapter", os.path.join("runs", adapter_run)]
    if PREDICT_LIMIT:
        cmd += ["--limit", str(PREDICT_LIMIT)]
    sh(cmd)
    return out


def score(name: str, split: str, vs: str | None = None) -> dict:
    """Score reports/preds/<name>_<split> (optionally paired against <vs>); saves report + errors."""
    gold = ev.load_jsonl(P("data", "eval", f"{split}.jsonl"))
    preds = ev.load_preds(pred_path(name, split))
    if PREDICT_LIMIT:
        gold = gold[:PREDICT_LIMIT]
    preds_b = ev.load_preds(pred_path(vs, split)) if vs else None
    report, errors = ev.evaluate(gold, preds, preds_b)
    report.update({"name": name, "split": split, "vs": vs})
    base = P("reports", f"{name}_{split}")
    with open(base + ".json", "w") as f:
        json.dump(report, f, indent=2)
    with open(base + ".errors.jsonl", "w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    m = report["metrics"]
    line = (f"{name}/{split}: exact_match={m['exact_match']}% CI95={m['exact_match_ci95']} "
            f"schema_valid={m['schema_valid']}% parse_strict={m['parse_strict']}% parse_lenient={m['parse_lenient']}%")
    if vs:
        p = report["paired_a_minus_b"]
        line += f"\n  vs {vs}: {p['exact_match_diff']:+} points, CI95={p['ci95']}"
    print(line)
    print("  groups:", {g: v["exact_match"] for g, v in report["groups"].items()})
    print("  top errors:", list(report["error_types"].items())[:6])
    return report


def show_errors(name: str, split: str, n: int = 20, seed: int = 0) -> None:
    import random
    path = P("reports", f"{name}_{split}.errors.jsonl")
    rows = ev.load_jsonl(path) if os.path.exists(path) else []
    print(f"{len(rows)} failed rows in {path}; showing {min(n, len(rows))}")
    for e in random.Random(seed).sample(rows, min(n, len(rows))):
        print(f"- [{','.join(e['error_types'])}] today={e['today']} | {e['request']}\n"
              f"    gold: {json.dumps(e['gold'])}\n    got : {e['response'][:300]!r}")


# ---------------------------------------------------------------- tables
def _upsert_csv(path: str, row: dict, keys: tuple[str, ...]) -> None:
    rows = list(csv.DictReader(open(path))) if os.path.exists(path) else []
    rows = [r for r in rows if tuple(r.get(k) for k in keys) != tuple(str(row[k]) for k in keys)]
    rows.append({k: str(v) for k, v in row.items()})
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def log_result(run: str, split: str, report: dict, s: dict | None = None, notes: str = "") -> None:
    """One row per evaluated model in reports/results_log.csv (the plan's Results log)."""
    m = report["metrics"]
    s = s or {}
    row = {"run": run, "split": split, "lr": s.get("lr", "—"), "epochs": s.get("epochs", "—"),
           "rank": s.get("rank", "—"), "seed": s.get("seed", "—"),
           "exact_match": m["exact_match"], "ci95": f"[{m['exact_match_ci95'][0]}, {m['exact_match_ci95'][1]}]",
           "schema_valid": m["schema_valid"], "parse_strict": m["parse_strict"], "notes": notes}
    if "paired_a_minus_b" in report:
        p = report["paired_a_minus_b"]
        row["paired_diff"] = f"{p['exact_match_diff']:+} [{p['ci95'][0]}, {p['ci95'][1]}] vs {report.get('vs')}"
    _upsert_csv(P("reports", "results_log.csv"), row, ("run", "split"))


def table(name: str = "results_log"):
    import pandas as pd
    path = P("reports", f"{name}.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


# ---------------------------------------------------------------- one study run
def run(name: str, overrides: dict | None = None, notes: str = "", round_: str = "") -> dict:
    """Config -> train (skipped if done) -> predict val -> score vs base -> study table + results log."""
    cfg_path, cfg = make_config(name, overrides)
    meta = train(name, cfg_path, cfg)
    predict(name, "val", adapter_run=name)
    report = score(name, "val", vs="base")
    hist = loss_history(cfg["output"])
    s = settings(cfg)
    m = report["metrics"]
    row = {"round": round_, "run": name, **s, "exact_match": m["exact_match"],
           "ci_lo": m["exact_match_ci95"][0], "ci_hi": m["exact_match_ci95"][1],
           "schema_valid": m["schema_valid"], "parse_strict": m["parse_strict"],
           "missing_group": report["groups"].get("missing_field", {}).get("exact_match"),
           "final_train_loss": hist["train"][-1][1] if hist["train"] else None,
           "min_val_loss": min(v for _, v in hist["eval"]) if hist["eval"] else None,
           "last_val_loss": hist["eval"][-1][1] if hist["eval"] else None,
           "train_seconds": meta.get("seconds"), "backend": meta.get("backend", get_backend())}
    _upsert_csv(P("reports", "hparam_study.csv"), row, ("run",))
    log_result(name, "val", report, s, notes)
    return row
