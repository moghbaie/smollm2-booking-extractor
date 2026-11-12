"""Tests for the v3 data generator: new date forms, date noise, calendar prompt, splits."""
import datetime as dt
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import generate_data_v2 as v2  # noqa: E402
import generate_data_v3 as g  # noqa: E402
from breakdown import field_by_form  # noqa: E402
from common import build_prompt, label_errors, label_to_text  # noqa: E402
from generate_data import next_weekday  # noqa: E402

TODAY = dt.date(2026, 11, 11)  # a Wednesday


@pytest.fixture(scope="module")
def data():
    return g.generate(3031, 2027, 1234)


def test_tiers_disjoint_and_extend_v2():
    a, b, c = (set(g.DATE_FORMS[t]) for t in ["base", "extra", "heldout"])
    assert not (a & b or a & c or b & c)
    assert set(v2.DATE_FORMS["heldout"]) <= c and set(v2.DATE_FORMS["extra"]) <= b


@pytest.mark.parametrize("form", ["weekday_bare", "this_weekday", "coming_weekday"])
def test_weekday_forms_follow_the_next_weekday_rule(form):
    for seed in range(20):
        text, iso = g.date_text(TODAY, form, random.Random(seed))
        day = text.split()[-1]
        w = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day)
        assert iso == next_weekday(TODAY, w).isoformat()


@pytest.mark.parametrize("form,days", [("in_a_week", 7), ("in_two_weeks", 14), ("fortnight", 14)])
def test_relative_week_forms(form, days):
    assert g.date_text(TODAY, form, random.Random(0))[1] == (TODAY + dt.timedelta(days=days)).isoformat()


@pytest.mark.parametrize("form", ["weekday_ordinal", "day_only"])
def test_day_of_month_forms_stay_inside_the_calendar(form):
    for seed in range(40):
        text, iso = g.date_text(TODAY, form, random.Random(seed))
        d = dt.date.fromisoformat(iso)
        assert 1 <= (d - TODAY).days < g.CAL_DAYS
        assert text.endswith(g.ordinal(d.day))
        if form == "weekday_ordinal":
            assert text.split()[1] == ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][d.weekday()]


@pytest.mark.parametrize("form", ["numeric_dm", "numeric_md"])
def test_numeric_dates_are_unambiguous(form):
    for seed in range(40):
        text, iso = g.date_text(TODAY, form, random.Random(seed))
        d = dt.date.fromisoformat(iso)
        a, b = (int(x) for x in text.split()[1].split("/"))
        assert d.day > 12 and ((a, b) == (d.day, d.month) if form == "numeric_dm" else (a, b) == (d.month, d.day))


def test_year_forms():
    for form in ["month_day_year", "day_abbr_year"]:
        text, iso = g.date_text(TODAY, form, random.Random(1))
        assert text.endswith(iso[:4])


def test_date_noise_keeps_digits_and_changes_text():
    rng = random.Random(0)
    for style in ["train", "stress"]:
        for surface in ["on Friday the 17th", "two weeks from today", "on December 7, 2026", "tomorrow"]:
            out, ops = g.apply_date_noise(surface, rng, style)
            assert ops and out != surface
            assert [c for c in out if c.isdigit()] == [c for c in surface if c.isdigit()]


def test_short_words_get_no_typos():
    rng = random.Random(0)
    for _ in range(50):
        out, ops = g.apply_date_noise("tmrw", rng, "stress")
        assert out == "tmrw" and ops == []


def test_stress_shorthand_never_matches_training_shorthand():
    train = {v.lower().strip(".") for v in g.SHORTHAND["train"].values()}
    stress = {v.lower().strip(".") for v in g.SHORTHAND["stress"].values()}
    assert not train & stress


def test_calendar_prompt():
    p = g.build_prompt_v3("Book Room A tomorrow.", TODAY)
    assert "Today is Wednesday, 2026-11-11." in p
    assert "Calendar: 2026-11: Thu 12, Fri 13, " in p and "Mon 30 | 2026-12: Tue 01, Wed 02.\n" in p   # today + 21 days
    assert p.endswith("Request: Book Room A tomorrow.")


def test_calendar_parses_back_to_the_right_dates():
    for today in [TODAY, dt.date(2026, 12, 20), dt.date(2027, 2, 15)]:   # month and year boundaries, February
        days = g.calendar_dates(g.build_prompt_v3("x", today))
        want = [today + dt.timedelta(days=i) for i in range(1, g.CAL_DAYS + 1)]
        assert [iso for _, iso in days] == [d.isoformat() for d in want]
        assert [wd for wd, _ in days] == [d.strftime("%a") for d in want]


def test_to_in_is_cleaned():
    assert g.fix_to_in("Could we move it to in two weeks?") == "Could we move it in two weeks?"
    assert g.fix_to_in("MOVE IT UNTIL IN A FORTNIGHT") == "MOVE IT IN A FORTNIGHT"


def test_generate_splits(data):
    assert len(data["train"]) == 4500 and sum(r["source"] == "v2" for r in data["train"]) == 3000
    for split in ["train", "val", "test"]:
        for r in data[split]:
            assert not any(f.startswith("heldout:") for f in r["forms"].values())
            assert not any(o.endswith(":neighbor") for o in r["noise"])
            assert "Calendar: " in r["prompt"]
    for split in data:
        assert all(not label_errors(r["label"]) for r in data[split])
    for r in data["test_v2prompt"][:20]:
        assert r["prompt"] == build_prompt(r["request"], dt.date.fromisoformat(r["today"]))


def test_old_sets_are_v2_and_not_trained_on(data):
    old = v2.generate()
    assert [r["request"] for r in data["old_test"]] == [r["request"] for r in old["test"]]
    trained = {v2.norm(r["request"]) for r in data["train"]}
    assert not trained & {v2.norm(r["request"]) for r in data["old_stress"]}


def test_generate_is_deterministic(data):
    again = g.generate(3031, 2027, 1234)
    assert [r["request"] for r in again["stress"]] == [r["request"] for r in data["stress"]]


def test_field_by_form():
    gold = [{"id": "a", "forms": {"date": "extra:in_a_week"}, "noise": ["date_short"],
             "label": {"intent": "cancel", "date": "2026-11-18", "time": None, "resource": None, "needs_clarification": False}},
            {"id": "b", "forms": {"date": "base:today"}, "noise": [],
             "label": {"intent": "cancel", "date": "2026-11-11", "time": None, "resource": None, "needs_clarification": False}}]
    preds = {"a": label_to_text(gold[1]["label"]), "b": label_to_text(gold[1]["label"])}
    assert field_by_form(gold, preds) == {"base:today": {"n": 1, "accuracy": 100.0},
                                          "extra:in_a_week+noise": {"n": 1, "accuracy": 0.0}}
