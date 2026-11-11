"""Tests for the evaluator (Phase 2)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from evaluate import bootstrap_ci, error_types, evaluate, parse_lenient, parse_strict, score_row  # noqa: E402

GOLD = {"intent": "reschedule", "date": "2026-12-07", "time": "18:00", "resource": None, "needs_clarification": False}
GOOD = json.dumps(GOLD)


def test_exact_reply_scores_everything():
    s = score_row(GOOD, GOLD)
    assert s["parse_strict"] and s["parse_lenient"] and s["schema_valid"] and s["exact_match"]


def test_whitespace_is_still_strict():
    assert score_row("\n  " + GOOD + "\n", GOLD)["parse_strict"]


def test_prose_and_fences_are_lenient_only():
    for text in [f"Here you go: {GOOD}", f"```json\n{GOOD}\n```", f"{GOOD} Hope this helps!"]:
        s = score_row(text, GOLD)
        assert not s["parse_strict"] and s["parse_lenient"] and s["exact_match"], text
        assert error_types(text, GOLD, s) == []  # exact_match counts lenient parses


def test_lenient_skips_non_object_braces():
    assert parse_lenient("set {x} then " + GOOD) == GOLD
    assert parse_lenient("no json here") is None
    assert parse_strict("[1, 2]") is None and parse_strict("") is None


def test_null_as_text_is_not_null():
    s = score_row(json.dumps({**GOLD, "resource": "null"}), GOLD)
    assert not s["field_resource"] and not s["exact_match"] and s["schema_valid"]
    assert "null_as_text" in error_types("", GOLD, s)


def test_bool_types_must_match():
    s = score_row(json.dumps({**GOLD, "needs_clarification": 0}), GOLD)
    assert not s["schema_valid"] and not s["field_needs_clarification"]


def test_extra_key_fails_schema_but_fields_can_match():
    s = score_row(json.dumps({**GOLD, "note": "x"}), GOLD)
    assert all(s[f"field_{f}"] for f in GOLD) and not s["schema_valid"] and not s["exact_match"]
    assert "wrong_keys" in error_types("", GOLD, s)


def test_error_type_names():
    invented = score_row(json.dumps({**GOLD, "resource": "Room A"}), GOLD)
    assert "resource_invented" in error_types("", GOLD, invented)
    missed = score_row(json.dumps({**GOLD, "date": None}), GOLD)
    assert "date_missed" in error_types("", GOLD, missed)
    wrong = score_row(json.dumps({**GOLD, "date": "2026-12-08"}), GOLD)
    assert "date_wrong" in error_types("", GOLD, wrong)
    assert error_types("blah", GOLD, score_row("blah", GOLD)) == ["no_json"]


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    vals = [1.0] * 70 + [0.0] * 30
    lo, hi = bootstrap_ci(vals)
    assert lo < 0.7 < hi and (lo, hi) == bootstrap_ci(vals)
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)


def test_evaluate_groups_missing_preds_and_pairing():
    rows = [{"id": f"r{i}", "label": {**GOLD, "intent": "cancel" if i % 2 else "reschedule"}} for i in range(10)]
    preds_a = {r["id"]: json.dumps(r["label"]) for r in rows}
    preds_b = {r["id"]: "I can't do that" for r in rows[:5]}  # 5 rows missing entirely
    report, errors = evaluate(rows, preds_a, preds_b)
    assert report["metrics"]["exact_match"] == 100.0 and errors == []
    assert report["b"]["metrics"]["exact_match"] == 0.0
    assert report["paired_a_minus_b"]["exact_match_diff"] == 100.0
    assert report["paired_a_minus_b"]["ci95"] == [100.0, 100.0]
    assert set(report["groups"]) >= {"intent=cancel", "intent=reschedule", "complete"}
    report_b, errors_b = evaluate(rows, preds_b)
    assert report_b["missing_predictions"] == 5 and len(errors_b) == 10
