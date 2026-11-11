"""Shared task definition: fields, rules, schema and prompt format."""
from __future__ import annotations

import datetime as dt
import re

INTENTS = ["book", "reschedule", "cancel", "check"]
FIELDS = ["intent", "date", "time", "resource", "needs_clarification"]
REQUIRED = {
    "book": ["date", "time", "resource"],
    "reschedule": ["date", "time"],
    "cancel": ["date"],
    "check": ["date", "resource"],
}
RESOURCES = ["Room A", "Room B", "Room C", "Studio 1", "Studio 2", "Lab 3", "Court 1", "Court 2"]

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")

INSTRUCTION = (
    "Extract the booking request below as JSON. Reply with only a JSON object with exactly these keys: "
    '"intent" (one of "book", "reschedule", "cancel", "check"), '
    '"date" ("YYYY-MM-DD" or null), "time" ("HH:MM", 24-hour, or null), '
    '"resource" (room or facility name, or null), '
    '"needs_clarification" (true if a value the intent needs is missing, else false). '
    "Use null for anything the request does not state."
)


def needs_clarification(rec: dict) -> bool:
    return any(rec.get(f) is None for f in REQUIRED[rec["intent"]])


def build_prompt(request: str, today: dt.date) -> str:
    return f"{INSTRUCTION}\nToday is {today.strftime('%A')}, {today.isoformat()}.\nRequest: {request}"


def label_to_text(label: dict) -> str:
    import json
    return json.dumps({k: label[k] for k in FIELDS}, ensure_ascii=False)


def schema_errors(obj) -> list[str]:
    """Return a list of problems; empty list means schema-valid."""
    errs: list[str] = []
    if not isinstance(obj, dict):
        return ["not_an_object"]
    keys = set(obj)
    if keys != set(FIELDS):
        missing = set(FIELDS) - keys
        extra = keys - set(FIELDS)
        if missing:
            errs.append(f"missing_keys:{sorted(missing)}")
        if extra:
            errs.append(f"extra_keys:{sorted(extra)}")
    if "intent" in obj and obj["intent"] not in INTENTS:
        errs.append("bad_intent")
    if "date" in obj and obj["date"] is not None:
        d = obj["date"]
        ok = isinstance(d, str) and bool(_DATE_RE.match(d))
        if ok:
            try:
                dt.date.fromisoformat(d)
            except ValueError:
                ok = False
        if not ok:
            errs.append("bad_date")
    if "time" in obj and obj["time"] is not None:
        t = obj["time"]
        if not (isinstance(t, str) and _TIME_RE.match(t)):
            errs.append("bad_time")
    if "resource" in obj and obj["resource"] is not None:
        r = obj["resource"]
        if not (isinstance(r, str) and r.strip()):
            errs.append("bad_resource")
    if "needs_clarification" in obj and not isinstance(obj["needs_clarification"], bool):
        errs.append("bad_needs_clarification")
    return errs


def label_errors(label: dict) -> list[str]:
    """Schema check plus the consistency rule, used on generated gold labels."""
    errs = schema_errors(label)
    if not errs and label["needs_clarification"] != needs_clarification(label):
        errs.append("inconsistent_needs_clarification")
    if not errs and label["resource"] is not None and label["resource"] not in RESOURCES:
        errs.append("unknown_resource")
    return errs
