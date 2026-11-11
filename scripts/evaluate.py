"""Score model replies against gold labels.

Metrics (all in %): exact_match (primary, with bootstrap CI95), schema_valid, parse_strict,
parse_lenient, field_<name>, and exact_match by intent and by complete/missing group.
With --pred-b, also the paired difference exact_match(A) - exact_match(B) on the same rows, with CI95.

Usage:
  python scripts/evaluate.py --gold data/eval/val.jsonl --pred reports/base_val.pred.jsonl \
      --out reports/base_val.json [--pred-b reports/other.pred.jsonl]
Prediction files are JSONL with {"id": ..., "response": "..."}; missing ids score as failures.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIELDS, INTENTS, schema_errors  # noqa: E402

N_BOOT = 2000


# ---------------------------------------------------------------- parsing
def parse_strict(text: str):
    """The reply is a JSON object and nothing else (surrounding whitespace allowed)."""
    try:
        obj = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None
    return obj if isinstance(obj, dict) else None


def parse_lenient(text: str):
    """First JSON object found anywhere in the reply (e.g. inside ```json fences or after prose)."""
    obj = parse_strict(text)
    if obj is not None:
        return obj
    if not isinstance(text, str):
        return None
    dec = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def same_value(a, b) -> bool:
    """Equal with the same type, so True != 1 and "null" != None."""
    return type(a) is type(b) and a == b


# ---------------------------------------------------------------- per-row scoring
def score_row(response: str, gold: dict) -> dict:
    strict = parse_strict(response)
    obj = parse_lenient(response)
    errs = schema_errors(obj) if obj is not None else ["no_json"]
    fields = {f: obj is not None and f in obj and same_value(obj[f], gold[f]) for f in FIELDS}
    exact = not errs and all(fields.values())
    return {
        "parse_strict": strict is not None,
        "parse_lenient": obj is not None,
        "schema_valid": not errs,
        "exact_match": exact,
        **{f"field_{f}": v for f, v in fields.items()},
        "_parsed": obj,
        "_schema_errors": errs,
    }


def error_types(response: str, gold: dict, s: dict) -> list[str]:
    """Nameable failure causes, for reading errors (checks 2.4 / 4.5)."""
    if s["exact_match"]:
        return []
    types = []
    if not s["parse_lenient"]:
        return ["no_json"]
    if not s["parse_strict"]:
        types.append("extra_text")
    obj = s["_parsed"]
    if any(isinstance(obj.get(f), str) and obj[f].strip().lower() in ("null", "none", "") for f in FIELDS):
        types.append("null_as_text")
    if any(e.startswith(("missing_keys", "extra_keys")) for e in s["_schema_errors"]):
        types.append("wrong_keys")
    for f in FIELDS:
        if not s[f"field_{f}"]:
            if f in obj and obj[f] is not None and gold[f] is None:
                types.append(f"{f}_invented")       # value given where request had none
            elif (f not in obj or obj[f] is None) and gold[f] is not None:
                types.append(f"{f}_missed")
            else:
                types.append(f"{f}_wrong")
    return types or ["other"]


# ---------------------------------------------------------------- statistics
def bootstrap_ci(values: list[float], n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return (means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1])


def pct(x: float) -> float:
    return round(100 * x, 2)


def summarise(rows: list[dict], scores: list[dict]) -> dict:
    n = len(rows)
    out = {"n": n, "metrics": {}, "groups": {}}
    keys = ["exact_match", "schema_valid", "parse_strict", "parse_lenient"] + [f"field_{f}" for f in FIELDS]
    for k in keys:
        out["metrics"][k] = pct(sum(s[k] for s in scores) / n) if n else float("nan")
    em = [float(s["exact_match"]) for s in scores]
    lo, hi = bootstrap_ci(em)
    out["metrics"]["exact_match_ci95"] = [pct(lo), pct(hi)]
    groups = {f"intent={i}": [j for j, r in enumerate(rows) if r["label"]["intent"] == i] for i in INTENTS}
    groups["complete"] = [j for j, r in enumerate(rows) if not r["label"]["needs_clarification"]]
    groups["missing_field"] = [j for j, r in enumerate(rows) if r["label"]["needs_clarification"]]
    for g, idx in groups.items():
        if idx:
            out["groups"][g] = {"n": len(idx), "exact_match": pct(sum(em[j] for j in idx) / len(idx))}
    return out


def paired_diff(scores_a: list[dict], scores_b: list[dict]) -> dict:
    d = [float(a["exact_match"]) - float(b["exact_match"]) for a, b in zip(scores_a, scores_b)]
    lo, hi = bootstrap_ci(d)
    return {"exact_match_diff": pct(sum(d) / len(d)), "ci95": [pct(lo), pct(hi)],
            "a_only_correct": sum(x > 0 for x in d), "b_only_correct": sum(x < 0 for x in d)}


# ---------------------------------------------------------------- io
def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_preds(path: str) -> dict[str, str]:
    return {r["id"]: r.get("response") or "" for r in load_jsonl(path)}


def evaluate(gold_rows: list[dict], preds: dict[str, str], preds_b: dict[str, str] | None = None) -> tuple[dict, list[dict]]:
    responses = [preds.get(r["id"], "") for r in gold_rows]
    scores = [score_row(resp, r["label"]) for resp, r in zip(responses, gold_rows)]
    report = summarise(gold_rows, scores)
    report["missing_predictions"] = sum(r["id"] not in preds for r in gold_rows)
    errors, counter = [], Counter()
    for r, resp, s in zip(gold_rows, responses, scores):
        types = error_types(resp, r["label"], s)
        counter.update(types)
        if types:
            errors.append({"id": r["id"], "template": r.get("template"), "request": r.get("request"),
                           "today": r.get("today"), "gold": r["label"], "response": resp, "error_types": types})
    report["error_types"] = dict(counter.most_common())
    if preds_b is not None:
        scores_b = [score_row(preds_b.get(r["id"], ""), r["label"]) for r in gold_rows]
        report["b"] = summarise(gold_rows, scores_b)
        report["paired_a_minus_b"] = paired_diff(scores, scores_b)
    return report, errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--pred-b", default=None, help="second model's predictions for a paired comparison")
    ap.add_argument("--out", required=True, help="report JSON; errors go to <out minus .json>.errors.jsonl")
    args = ap.parse_args()

    gold = load_jsonl(args.gold)
    report, errors = evaluate(gold, load_preds(args.pred), load_preds(args.pred_b) if args.pred_b else None)
    report.update({"gold": args.gold, "pred": args.pred, "pred_b": args.pred_b})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    err_path = re.sub(r"\.json$", "", args.out) + ".errors.jsonl"
    with open(err_path, "w", encoding="utf-8") as f:
        for e in errors:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    m = report["metrics"]
    print(f"n={report['n']}  exact_match={m['exact_match']}% CI95={m['exact_match_ci95']}  "
          f"schema_valid={m['schema_valid']}%  parse_strict={m['parse_strict']}%  parse_lenient={m['parse_lenient']}%")
    print("fields: " + "  ".join(f"{f}={m['field_' + f]}%" for f in FIELDS))
    print("groups: " + "  ".join(f"{g}={v['exact_match']}% (n={v['n']})" for g, v in report["groups"].items()))
    print(f"top error types: {list(report['error_types'].items())[:6]}")
    if "paired_a_minus_b" in report:
        p = report["paired_a_minus_b"]
        print(f"paired A-B: {p['exact_match_diff']:+} points, CI95={p['ci95']}  "
              f"(A-only correct {p['a_only_correct']}, B-only correct {p['b_only_correct']})")
    print(f"report -> {args.out}   errors ({len(errors)}) -> {err_path}")


if __name__ == "__main__":
    main()
