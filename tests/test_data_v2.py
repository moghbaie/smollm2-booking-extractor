"""Tests for the v2 data generator and the per-group scorer."""
import datetime as dt
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import generate_data as v1  # noqa: E402
import generate_data_v2 as g  # noqa: E402
from breakdown import by_group, field_by_tier  # noqa: E402
from common import RESOURCES, label_errors, label_to_text  # noqa: E402

TODAY = dt.date(2026, 11, 11)  # a Wednesday


@pytest.fixture(scope="module")
def data():
    return g.generate(2027, 1234)


def test_tiers_do_not_share_forms():
    for forms in (g.DATE_FORMS, g.TIME_FORMS, g.RESOURCE_FORMS):
        a, b, c = (set(forms[t]) for t in g.TIERS)
        assert not (a & b or a & c or b & c)


@pytest.mark.parametrize("form,days", [("tmrw", 1), ("week_from_today", 7), ("two_weeks", 14)])
def test_relative_dates(form, days):
    text, iso = g.date_text(TODAY, form, random.Random(0))
    assert iso == (TODAY + dt.timedelta(days=days)).isoformat()


@pytest.mark.parametrize("form", ["in_n_days_words", "days_from_now"])
def test_n_days_matches_text(form):
    words = {v: k for k, v in v1.NUM_WORDS.items()}
    for seed in range(30):
        text, iso = g.date_text(TODAY, form, random.Random(seed))
        tok = text.split()[1] if form == "in_n_days_words" else text.split()[0]
        n = words[tok] if form == "in_n_days_words" else int(tok)
        assert iso == (TODAY + dt.timedelta(days=n)).isoformat()


@pytest.mark.parametrize("h,m,form,text", [
    (18, 30, "military", "at 1830h"), (7, 5, "h_h", "at 7h05"), (18, 30, "ampm_caps", "at 6:30PM"),
    (18, 0, "h12_part", "at 6 in the evening"), (11, 45, "quarter", "at quarter to twelve in the morning"),
    (16, 45, "quarter", "at quarter to five in the afternoon"), (12, 30, "half_past", "at half past twelve in the afternoon"),
    (9, 15, "dotted", "at 9.15 a.m."), (12, 0, "midday", "at midday"), (13, 0, "midday", None),
])
def test_time_text(h, m, form, text):
    assert g.time_text(h, m, form) == text


def test_every_resource_has_a_form_in_every_tier():
    for res in RESOURCES:
        for tier in g.TIERS:
            assert any(g.resource_text(res, f) for f in g.RESOURCE_FORMS[tier]), (res, tier)
    assert g.resource_text("Studio 2", "number_spelled") == "studio number two"
    assert g.resource_text("Room B", "conference") == "conference room B"


@pytest.mark.parametrize("op", ["swap", "drop", "double", "neighbor"])
def test_typo_always_changes_the_word(op):
    rng = random.Random(0)
    for w in ["booking", "reservation", "cancel", "Room", "please"]:
        assert g.typo(w, op, rng) != w


def test_noise_never_touches_slot_values():
    rng = random.Random(0)
    segs = [("Please book ", False), ("Studio 2", True), (" ", False), ("tomorrow", True), (" please.", False)]
    for _ in range(200):
        text, ops = g.apply_noise(segs, rng, "stress", force=True)
        assert ops
        low = text.lower()
        assert "studio 2" in low and "tomorrow" in low


def test_generate_is_deterministic(data):
    again = g.generate(2027, 1234)
    assert [r["request"] for r in again["stress"]] == [r["request"] for r in data["stress"]]


def test_v1_rows_kept_unchanged(data):
    v1_train = v1.generate(1234)["train"]
    ours = {label_to_text(r["label"]) + r["prompt"] for r in data["train"] if r["group"] == "v1_original"}
    assert len(ours) == 2000 and all(label_to_text(r["label"]) + r["prompt"] in ours for r in v1_train)


def test_heldout_only_in_stress_and_labels_valid(data):
    for split in ["train", "val", "test"]:
        for r in data[split]:
            assert not any(f.startswith("heldout:") for f in r["forms"].values())
            assert "typo:neighbor" not in r["noise"]
    groups = {r["group"] for r in data["stress"]}
    assert groups == {"noise_only", "new_wording", "both"}
    for r in data["stress"]:
        has_h = any(f.startswith("heldout:") for f in r["forms"].values())
        assert (r["group"] in ("new_wording", "both")) == has_h
        assert (r["group"] in ("noise_only", "both")) == bool(r["noise"])
    for split in data:
        assert all(not label_errors(r["label"]) for r in data[split])


def test_breakdown_groups_and_tiers():
    gold = [{"id": "a", "group": "x", "forms": {"date": "heldout:tmrw"},
             "label": {"intent": "cancel", "date": "2026-11-12", "time": None, "resource": None, "needs_clarification": False}},
            {"id": "b", "group": "y", "forms": {"date": "base:today"},
             "label": {"intent": "cancel", "date": "2026-11-11", "time": None, "resource": None, "needs_clarification": False}}]
    right = {r["id"]: label_to_text(r["label"]) for r in gold}
    wrong = {"a": right["b"], "b": right["b"]}
    res = by_group(gold, right, wrong)
    assert res["x"]["exact_match"] == 100.0 and res["x"]["paired_a_minus_b"]["exact_match_diff"] == 100.0
    assert res["y"]["paired_a_minus_b"]["exact_match_diff"] == 0.0
    assert field_by_tier(gold, wrong) == {"date/base": {"n": 1, "accuracy": 100.0}, "date/heldout": {"n": 1, "accuracy": 0.0}}
