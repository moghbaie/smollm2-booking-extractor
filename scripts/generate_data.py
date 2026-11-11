"""Generate train / val / test / stress splits with rule-computed labels.

Splits are separated by template family: every split uses its own sentence templates.
Usage: python scripts/generate_data.py --out data --seed 1234
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FIELDS, INTENTS, REQUIRED, RESOURCES, build_prompt, label_to_text, needs_clarification  # noqa: E402

# ---------------------------------------------------------------- templates
# "[ ... {slot} ...]" is an optional segment, dropped when the slot is missing.
# Date surfaces include their own preposition ("on December 7", "tomorrow"); time surfaces include "at".
TEMPLATES = {
    "train": {
        "book": [
            "Please book[ {resource}][ {date}][ {time}].",
            "I'd like to reserve[ {resource}][ {date}][ {time}].",
            "Can you get me[ {resource}][ {date}][ {time}]?",
            "Book[ {resource}] for me[ {date}][ {time}] please.",
            "I need a booking[ for {resource}][ {date}][ {time}], can you set that up?",
            "Reserve[ {resource}][ {time}][ {date}] for our team.",
        ],
        "reschedule": [
            "Please move my booking[ for {resource}][ to {date}][ {time}].",
            "Can I change my reservation[ of {resource}][ to {date}][ {time}]?",
            "Reschedule my[ {resource}] slot[ to {date}][ {time}].",
            "I need to shift my appointment[ in {resource}][ to {date}][ {time}].",
            "Push my booking[ for {resource}] back[ to {date}][ {time}].",
            "Could we move it[ to {date}][ {time}] instead?[ It's the {resource} booking.]",
        ],
        "cancel": [
            "Please cancel my booking[ for {resource}][ {date}][ {time}].",
            "Cancel the[ {resource}] reservation[ {date}][ {time}].",
            "I won't need my booking[ of {resource}][ {date}][ {time}] anymore, please cancel it.",
            "Drop my reservation[ {date}][ {time}][ for {resource}].",
            "Can you call off my booking[ in {resource}][ {date}][ {time}]?",
            "Remove my[ {resource}] slot[ {date}][ {time}] from the calendar.",
        ],
        "check": [
            "Is there availability[ in {resource}][ {date}][ {time}]?",
            "Can you check availability[ for {resource}][ {date}][ {time}]?",
            "Do you have any openings[ in {resource}][ {date}][ {time}]?",
            "What's the availability[ of {resource}][ {date}][ {time}]?",
            "Any free slots[ for {resource}][ {date}][ {time}]?",
            "I want to know whether there is space[ in {resource}][ {date}][ {time}].",
        ],
    },
    "val": {
        "book": [
            "Could I grab a slot[ in {resource}][ {date}][ {time}]?",
            "Hold[ {resource}] for us[ {date}][ {time}].",
        ],
        "reschedule": [
            "Something came up, can we do[ {date}][ {time}] instead for my[ {resource}] reservation?",
            "Let's change the time of my booking[ in {resource}][ to {date}][ {time}].",
        ],
        "cancel": [
            "We no longer need the booking[ for {resource}][ {date}][ {time}], scrap it.",
            "Kindly delete my reservation[ of {resource}][ {date}][ {time}].",
        ],
        "check": [
            "Would there be room[ in {resource}][ {date}][ {time}]?",
            "Look up if anything is open[ in {resource}][ {date}][ {time}].",
        ],
    },
    "test": {
        "book": [
            "We would like a reservation[ for {resource}][ {date}][ {time}] please.",
            "Set up a booking[ for {resource}][ {date}][ {time}] if possible.",
        ],
        "reschedule": [
            "Is it possible to postpone my[ {resource}] booking[ until {date}][ {time}]?",
            "I'd prefer[ {date}][ {time}] for my[ {resource}] reservation instead of the current one.",
        ],
        "cancel": [
            "Our plans changed so please release the[ {resource}] booking[ {date}][ {time}].",
            "Forget about my reservation[ for {resource}][ {date}][ {time}], it's cancelled.",
        ],
        "check": [
            "Is there a vacancy[ for {resource}][ {date}][ {time}]?",
            "Please find out if a spot is available[ in {resource}][ {date}][ {time}].",
        ],
    },
}
TEMPLATES["stress"] = TEMPLATES["test"]  # same families as test, plus noise and unseen wording

SPLIT_CFG = {  # rows, probability that >=1 required field is missing, use unseen surface forms
    "train": dict(n=2000, p_missing=0.28, stress=False),
    "val": dict(n=300, p_missing=0.30, stress=False),
    "test": dict(n=300, p_missing=0.50, stress=False),
    "stress": dict(n=150, p_missing=0.50, stress=True),
}

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
             9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def ordinal(n: int) -> str:
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


# ---------------------------------------------------------------- date rules
def next_weekday(today: dt.date, weekday: int) -> dt.date:
    """Rule: 'next <Weekday>' = the first such day strictly after today."""
    delta = (weekday - today.weekday()) % 7
    return today + dt.timedelta(days=delta or 7)


def date_surface(today: dt.date, rng: random.Random, stress: bool) -> tuple[str, str]:
    base_forms = ["today", "tomorrow", "day_after", "in_n_days", "next_weekday",
                  "month_day", "day_month", "iso", "ordinal_of"]
    stress_forms = ["week_from_today", "two_weeks", "in_n_days_words", "abbr_dot", "day_abbr"]
    form = rng.choice(stress_forms if (stress and rng.random() < 0.7) else base_forms)
    if form == "today":
        return "today", today.isoformat()
    if form == "tomorrow":
        return "tomorrow", (today + dt.timedelta(days=1)).isoformat()
    if form == "day_after":
        return "the day after tomorrow", (today + dt.timedelta(days=2)).isoformat()
    if form == "in_n_days":
        n = rng.randint(2, 14)
        return f"in {n} days", (today + dt.timedelta(days=n)).isoformat()
    if form == "next_weekday":
        w = rng.randrange(7)
        return f"next {WEEKDAYS[w]}", next_weekday(today, w).isoformat()
    if form == "week_from_today":
        return "a week from today", (today + dt.timedelta(days=7)).isoformat()
    if form == "two_weeks":
        return "two weeks from today", (today + dt.timedelta(days=14)).isoformat()
    if form == "in_n_days_words":
        n = rng.randint(2, 10)
        return f"in {NUM_WORDS[n]} days", (today + dt.timedelta(days=n)).isoformat()
    # absolute dates: always within the next 1..90 days, so "December 7" has one meaning
    target = today + dt.timedelta(days=rng.randint(1, 90))
    m, d = MONTHS[target.month - 1], target.day
    surface = {
        "month_day": f"on {m} {d}",
        "day_month": f"on {d} {m}",
        "iso": f"on {target.isoformat()}",
        "ordinal_of": f"on the {ordinal(d)} of {m}",
        "abbr_dot": f"on {m[:3]}. {ordinal(d)}",
        "day_abbr": f"on {ordinal(d)} {m[:3]}",
    }[form]
    return surface, target.isoformat()


# ---------------------------------------------------------------- time rules
def _h12(h: int) -> tuple[int, str]:
    return (h % 12 or 12), ("am" if h < 12 else "pm")


def _part_of_day(h: int) -> str:
    return "in the morning" if h < 12 else ("in the afternoon" if h < 17 else "in the evening")


def time_surface(rng: random.Random, stress: bool) -> tuple[str, str]:
    h = rng.randint(7, 21)
    m = rng.choice([0, 0, 0, 30, 30, 15, 45])
    iso = f"{h:02d}:{m:02d}"
    h12, ap = _h12(h)
    base = ["h24", "ampm", "ampm_space", "oclock", "noon"]
    strs = ["half_past", "quarter", "military", "dotted", "midday"]
    order = (strs if (stress and rng.random() < 0.7) else base)[:]
    rng.shuffle(order)
    for form in order + ["h24"]:
        if form == "h24":
            return f"at {iso}", iso
        if form == "ampm":
            return (f"at {h12}{ap}" if m == 0 else f"at {h12}:{m:02d}{ap}"), iso
        if form == "ampm_space":
            return (f"at {h12} {ap}" if m == 0 else f"at {h12}:{m:02d} {ap}"), iso
        if form == "oclock" and m == 0:
            return f"at {h12} o'clock {_part_of_day(h)}", iso
        if form == "noon" and h == 12 and m == 0:
            return "at noon", iso
        if form == "midday" and h == 12 and m == 0:
            return "at midday", iso
        if form == "half_past" and m == 30:
            return f"at half past {NUM_WORDS[h12]} {_part_of_day(h)}", iso
        if form == "quarter" and m == 15:
            return f"at quarter past {NUM_WORDS[h12]} {_part_of_day(h)}", iso
        if form == "quarter" and m == 45:
            nh12, _ = _h12(h + 1)
            return f"at quarter to {NUM_WORDS[nh12]} {_part_of_day(h + 1)}", iso
        if form == "military":
            return f"at {h:02d}{m:02d}h", iso
        if form == "dotted":
            return f"at {h12}.{m:02d} {'a.m.' if ap == 'am' else 'p.m.'}", iso
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- resource rules
def resource_surface(rng: random.Random, stress: bool) -> tuple[str, str]:
    res = rng.choice(RESOURCES)
    kind, ident = res.split(" ")
    if stress and rng.random() < 0.7:
        word = NUM_WORDS[int(ident)] if ident.isdigit() else ident
        forms = {
            "Room": [f"rm {ident}", f"room-{ident}"],
            "Studio": [f"studio {word}", f"studio no. {ident}"],
            "Lab": [f"laboratory {ident}", f"lab no. {ident}"],
            "Court": [f"court no. {ident}", f"court {word}"],
        }[kind]
        return rng.choice(forms), res
    return rng.choice([res, res, f"{kind.lower()} {ident}"]), res


# ---------------------------------------------------------------- rendering
_SEG_RE = re.compile(r"\[([^\]]*)\]")
_SLOT_RE = re.compile(r"\{(\w+)\}")


def render(template: str, surfaces: dict) -> list[tuple[str, bool]]:
    """Render to segments [(text, is_slot)]. Missing slots drop their optional segment."""
    pieces: list[tuple[str, bool]] = []
    pos = 0
    for m in _SEG_RE.finditer(template):
        pieces.extend(_render_plain(template[pos:m.start()], surfaces))
        inner = m.group(1)
        slots = _SLOT_RE.findall(inner)
        if all(surfaces.get(s) is not None for s in slots):
            pieces.extend(_render_plain(inner, surfaces))
        pos = m.end()
    pieces.extend(_render_plain(template[pos:], surfaces))
    return pieces


def _render_plain(text: str, surfaces: dict) -> list[tuple[str, bool]]:
    out, pos = [], 0
    for m in _SLOT_RE.finditer(text):
        out.append((text[pos:m.start()], False))
        out.append((surfaces[m.group(1)], True))
        pos = m.end()
    out.append((text[pos:], False))
    return [p for p in out if p[0]]


def clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,?!])", r"\1", text)
    # "to on December 7" -> "to December 7";  "to in 3 days" -> "in 3 days"
    text = re.sub(r"\b(to|until) on\b", r"\1", text, flags=re.I)
    text = re.sub(r"\b(?:to|until) (in \w+ days)\b", r"\1", text, flags=re.I)
    return text[0].upper() + text[1:] if text else text


# ---------------------------------------------------------------- noise (stress only)
FILLERS_PRE = ["hey,", "hi there!", "um so", "hello,", "ok so", "quick one:"]
FILLERS_POST = ["thanks!", "cheers", "asap pls", "thank you so much", "ty"]


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 4 or not word.isalpha():
        return word
    i = rng.randrange(len(word) - 1)
    op = rng.choice(["swap", "drop", "double"])
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "drop":
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]


def add_noise(segments: list[tuple[str, bool]], rng: random.Random) -> str:
    segs = [list(s) for s in segments]
    if rng.random() < 0.5:  # 1-2 typos, only in template words (never inside a slot value)
        cand = [(i, j) for i, (t, slot) in enumerate(segs) if not slot
                for j, w in enumerate(t.split(" ")) if len(w) >= 4 and w.isalpha()]
        for i, j in rng.sample(cand, k=min(len(cand), rng.randint(1, 2))):
            words = segs[i][0].split(" ")
            words[j] = _typo(words[j], rng)
            segs[i][0] = " ".join(words)
    text = clean("".join(t for t, _ in segs))
    if rng.random() < 0.4:
        text = f"{rng.choice(FILLERS_PRE)} {text[0].lower()}{text[1:]}"
    if rng.random() < 0.4:
        text = f"{text} {rng.choice(FILLERS_POST)}"
    r = rng.random()
    if r < 0.15:
        text = text.upper()
    elif r < 0.35:
        text = text.lower()
    return text


# ---------------------------------------------------------------- sampling
def sample_presence(intent: str, p_missing: float, rng: random.Random) -> dict:
    req = REQUIRED[intent]
    present = {f: True for f in ["date", "time", "resource"]}
    for f in ["date", "time", "resource"]:
        if f not in req:
            present[f] = rng.random() < 0.5  # optional fields: 50/50
    if rng.random() < p_missing:
        k = 1 if (len(req) == 1 or rng.random() < 0.75) else rng.randint(2, len(req))
        for f in rng.sample(req, k):
            present[f] = False
    return present


def make_row(split: str, rng: random.Random, start: dt.date) -> dict:
    cfg = SPLIT_CFG[split]
    intent = rng.choice(INTENTS)
    t_idx = rng.randrange(len(TEMPLATES[split][intent]))
    template = TEMPLATES[split][intent][t_idx]
    today = start + dt.timedelta(days=rng.randrange(730))
    present = sample_presence(intent, cfg["p_missing"], rng)
    label = {"intent": intent, "date": None, "time": None, "resource": None}
    surfaces = {}
    if present["date"]:
        surfaces["date"], label["date"] = date_surface(today, rng, cfg["stress"])
    if present["time"]:
        surfaces["time"], label["time"] = time_surface(rng, cfg["stress"])
    if present["resource"]:
        surfaces["resource"], label["resource"] = resource_surface(rng, cfg["stress"])
    label["needs_clarification"] = needs_clarification(label)
    segments = render(template, surfaces)
    request = add_noise(segments, rng) if cfg["stress"] else clean("".join(t for t, _ in segments))
    family = f"{'test' if split == 'stress' else split}/{intent}/{t_idx}"
    return {
        "split": split, "template": family, "today": today.isoformat(), "request": request,
        "prompt": build_prompt(request, today), "label": {k: label[k] for k in FIELDS},
    }


def generate(seed: int = 1234) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    start = dt.date(2026, 1, 1)
    seen: set[str] = set()
    out: dict[str, list[dict]] = {}
    for split in ["train", "val", "test", "stress"]:
        rows, tries = [], 0
        while len(rows) < SPLIT_CFG[split]["n"]:
            tries += 1
            if tries > 200_000:
                raise RuntimeError(f"could not generate enough unique rows for {split}")
            row = make_row(split, rng, start)
            key = re.sub(r"\W+", " ", row["request"].lower()).strip()
            if key in seen:  # no request text is shared across (or within) splits
                continue
            seen.add(key)
            row["id"] = f"{split}-{len(rows):04d}"
            rows.append(row)
        out[split] = rows
    return out


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    splits = generate(args.seed)
    # train.jsonl holds only chat messages (what Soup trains on); metadata goes to a sidecar file
    train_chat = [{"messages": [{"role": "user", "content": r["prompt"]},
                                {"role": "assistant", "content": label_to_text(r["label"])}]}
                  for r in splits["train"]]
    files = {
        "train.jsonl": train_chat,
        "train_meta.jsonl": splits["train"],
        "eval/val.jsonl": splits["val"],
        "eval/test.jsonl": splits["test"],
        "eval/stress.jsonl": splits["stress"],
    }
    for rel, rows in files.items():
        write_jsonl(os.path.join(args.out, rel), rows)
    manifest = {"seed": args.seed,
                "files": {rel: {"rows": len(rows), "sha256": sha256(os.path.join(args.out, rel))}
                          for rel, rows in files.items()}}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    for rel, info in manifest["files"].items():
        print(f"{rel:22s} {info['rows']:5d} rows  sha256={info['sha256'][:12]}…")


if __name__ == "__main__":
    main()
