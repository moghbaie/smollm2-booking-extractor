"""Tests for the task rules and the data generator (Phase 1)."""
import datetime as dt
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import generate_data as g  # noqa: E402
from common import RESOURCES, label_errors, needs_clarification, schema_errors  # noqa: E402

SAT = dt.date(2026, 11, 7)  # a Saturday


# ---------------------------------------------------------------- rules
def test_next_weekday_is_strictly_after_today():
    assert g.next_weekday(SAT, 5) == dt.date(2026, 11, 14)  # next Saturday: +7, never today
    assert g.next_weekday(SAT, 6) == dt.date(2026, 11, 8)   # next Sunday: tomorrow
    assert g.next_weekday(SAT, 0) == dt.date(2026, 11, 9)   # next Monday


def test_needs_clarification_rules():
    base = {"date": "2026-12-07", "time": "18:00", "resource": "Room A"}
    assert not needs_clarification({"intent": "book", **base})
    assert needs_clarification({"intent": "book", **base, "resource": None})
    assert not needs_clarification({"intent": "reschedule", **base, "resource": None})
    assert needs_clarification({"intent": "reschedule", **base, "time": None})
    assert not needs_clarification({"intent": "cancel", "date": "2026-12-07", "time": None, "resource": None})
    assert needs_clarification({"intent": "cancel", "date": None, "time": "18:00", "resource": "Room A"})
    assert needs_clarification({"intent": "check", **base, "resource": None})
    assert not needs_clarification({"intent": "check", **base, "time": None})


def test_schema_errors_edge_cases():
    good = {"intent": "book", "date": "2026-12-07", "time": "18:00", "resource": "Room A", "needs_clarification": False}
    assert schema_errors(good) == []
    assert "bad_date" in schema_errors({**good, "date": "null"})        # "null" as text is wrong
    assert "bad_date" in schema_errors({**good, "date": "2026-02-30"})  # not a real day
    assert "bad_time" in schema_errors({**good, "time": "6pm"})
    assert "bad_time" in schema_errors({**good, "time": "24:00"})
    assert "bad_needs_clarification" in schema_errors({**good, "needs_clarification": 0})
    assert "bad_intent" in schema_errors({**good, "intent": "Book"})
    assert any(e.startswith("extra_keys") for e in schema_errors({**good, "room": "A"}))
    assert any(e.startswith("missing_keys") for e in schema_errors({k: v for k, v in good.items() if k != "time"}))
    assert schema_errors([good]) == ["not_an_object"]
    assert label_errors({**good, "needs_clarification": True}) == ["inconsistent_needs_clarification"]


# ---------------------------------------------------------------- surfaces -> labels
MONTHS = {m.lower(): i + 1 for i, m in enumerate(g.MONTHS)}
WORDS = {v: k for k, v in g.NUM_WORDS.items()}
WEEKDAYS = {d.lower(): i for i, d in enumerate(g.WEEKDAYS)}


def parse_date(surface: str, today: dt.date) -> dt.date:
    """Independent re-implementation of the date wording, used to cross-check labels."""
    s = surface.lower().removeprefix("on ").replace(".", "")
    rel = {"today": 0, "tomorrow": 1, "the day after tomorrow": 2, "a week from today": 7, "two weeks from today": 14}
    if s in rel:
        return today + dt.timedelta(days=rel[s])
    if m := re.fullmatch(r"in (\w+) days", s):
        n = int(m.group(1)) if m.group(1).isdigit() else WORDS[m.group(1)]
        return today + dt.timedelta(days=n)
    if m := re.fullmatch(r"next (\w+)", s):
        delta = (WEEKDAYS[m.group(1)] - today.weekday()) % 7 or 7
        return today + dt.timedelta(days=delta)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return dt.date.fromisoformat(s)
    tokens = s.replace("the ", "").replace(" of ", " ").split()
    day = next(int(re.sub(r"\D", "", t)) for t in tokens if re.match(r"\d", t))
    month = next(v for k, v in MONTHS.items() for t in tokens if t.isalpha() and k.startswith(t))
    for year in (today.year, today.year + 1):  # next occurrence after today
        try:
            d = dt.date(year, month, day)
        except ValueError:  # 29 February in a non-leap year
            continue
        if d > today:
            return d
    raise ValueError(surface)


def parse_time(surface: str) -> str:
    s = surface.lower().removeprefix("at ")
    if s in ("noon", "midday"):
        return "12:00"
    if m := re.fullmatch(r"(\d\d):(\d\d)", s):
        return s
    if m := re.fullmatch(r"(\d\d)(\d\d)h", s):
        return f"{m.group(1)}:{m.group(2)}"
    pm_word = "evening" in s or "afternoon" in s
    if m := re.fullmatch(r"(half past|quarter past|quarter to) (\w+) in the \w+", s):
        h = WORDS[m.group(2)]
        h = h % 12 + (12 if pm_word else 0)
        if m.group(1) == "quarter to":
            h -= 1
            return f"{h:02d}:45"
        return f"{h:02d}:{30 if m.group(1) == 'half past' else 15:02d}"
    if m := re.fullmatch(r"(\d+) o'clock in the \w+", s):
        return f"{int(m.group(1)) % 12 + (12 if pm_word else 0):02d}:00"
    m = re.fullmatch(r"(\d+)(?:[:.](\d\d))? ?(am|pm|a\.m\.|p\.m\.)", s)
    assert m, surface
    h = int(m.group(1)) % 12 + (12 if m.group(3).startswith("p") else 0)
    return f"{h:02d}:{m.group(2) or '00'}"


def test_date_labels_match_wording():
    rng = random.Random(0)
    for i in range(3000):
        today = dt.date(2026, 1, 1) + dt.timedelta(days=rng.randrange(730))
        surface, iso = g.date_surface(today, rng, stress=bool(i % 2))
        assert parse_date(surface, today).isoformat() == iso, (today, surface, iso)


def test_time_labels_match_wording():
    rng = random.Random(1)
    for i in range(3000):
        surface, iso = g.time_surface(rng, stress=bool(i % 2))
        assert parse_time(surface) == iso, (surface, iso)


def test_resource_wording_maps_to_known_resource():
    rng = random.Random(2)
    for i in range(500):
        surface, res = g.resource_surface(rng, stress=bool(i % 2))
        assert res in RESOURCES and surface


# ---------------------------------------------------------------- rendering and noise
def test_render_drops_missing_optional_segments():
    t = "Please move my booking[ for {resource}][ to {date}][ {time}]."
    segs = g.render(t, {"date": "on December 7", "time": None, "resource": None})
    assert g.clean("".join(s for s, _ in segs)) == "Please move my booking to December 7."
    segs = g.render(t, {"date": "in 3 days", "time": "at 6pm"})
    assert g.clean("".join(s for s, _ in segs)) == "Please move my booking in 3 days at 6pm."


def test_noise_never_touches_slot_values():
    rng = random.Random(3)
    segs = [("Please cancel my booking ", False), ("Room A", True), (" ", False), ("on December 7", True), (".", False)]
    for _ in range(300):
        text = g.add_noise(segs, rng).lower()
        assert "room a" in text and "on december 7" in text


def test_generation_is_deterministic_and_complete():
    a, b = g.generate(seed=99), g.generate(seed=99)
    assert a == b
    assert {k: len(v) for k, v in a.items()} == {"train": 2000, "val": 300, "test": 300, "stress": 150}
    for rows in a.values():
        for r in rows:
            assert label_errors(r["label"]) == []
            assert r["request"] in r["prompt"]
