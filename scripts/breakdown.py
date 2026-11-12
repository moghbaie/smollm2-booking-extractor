"""Scores per row group (e.g. stress: noise_only / new_wording / both), and field accuracy per wording tier.

Used by the v2 notebook. `gold_rows` are eval rows with `group` and `forms`; preds are {id: response}.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate as ev  # noqa: E402


def _scores(gold_rows, preds):
    return [ev.score_row(preds.get(r["id"], ""), r["label"]) for r in gold_rows]


def by_group(gold_rows: list[dict], preds: dict, preds_b: dict | None = None, key: str = "group") -> dict:
    """{group: {n, exact_match, ci95[, paired_a_minus_b]}}; paired = A - B on the same rows."""
    idx = defaultdict(list)
    for i, r in enumerate(gold_rows):
        idx[r.get(key, "all")].append(i)
    sa = _scores(gold_rows, preds)
    sb = _scores(gold_rows, preds_b) if preds_b is not None else None
    out = {}
    for g in sorted(idx):
        em = [float(sa[i]["exact_match"]) for i in idx[g]]
        lo, hi = ev.bootstrap_ci(em)
        res = {"n": len(em), "exact_match": ev.pct(sum(em) / len(em)), "ci95": [ev.pct(lo), ev.pct(hi)]}
        if sb is not None:
            res["paired_a_minus_b"] = ev.paired_diff([sa[i] for i in idx[g]], [sb[i] for i in idx[g]])
        out[g] = res
    return out


def field_by_tier(gold_rows: list[dict], preds: dict) -> dict:
    """Accuracy of date/time/resource, split by the wording tier used for that value (base/extra/heldout)."""
    s = _scores(gold_rows, preds)
    acc = defaultdict(list)
    for r, sc in zip(gold_rows, s):
        for slot, form in r.get("forms", {}).items():
            acc[f"{slot}/{form.split(':')[0]}"].append(float(sc[f"field_{slot}"]))
    return {k: {"n": len(v), "accuracy": ev.pct(sum(v) / len(v))} for k, v in sorted(acc.items())}


def field_by_form(gold_rows: list[dict], preds: dict, slot: str = "date") -> dict:
    """Accuracy of one field per wording form ("extra:in_two_weeks", ...); "+noise" marks rows whose value was noised."""
    s = _scores(gold_rows, preds)
    acc = defaultdict(list)
    for r, sc in zip(gold_rows, s):
        form = r.get("forms", {}).get(slot)
        if form:
            noised = any(o.startswith(f"{slot}_") for o in r.get("noise", []))
            acc[form + ("+noise" if noised else "")].append(float(sc[f"field_{slot}"]))
    return {k: {"n": len(v), "accuracy": ev.pct(sum(v) / len(v))} for k, v in sorted(acc.items())}
