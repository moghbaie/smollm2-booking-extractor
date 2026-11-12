"""Generate the v3 data: wider date wording, typos and shorthand inside dates, and a calendar in the prompt.

What changed from v2:
- Prompt: after "Today is ...", a compact 21-day calendar ("Calendar: 2026-11: Thu 12, Fri 13, ... | 2026-12: Tue 01.")
  so the model looks relative dates up instead of counting. Every row (train and eval) uses this prompt.
- Date wording: 8 new forms that training sees ("on Friday", "this Friday", "in a week", "in two weeks",
  "on December 7, 2026", "on Monday the 7th", "on 17/12", "on the 17th") and 4 new held-out forms that only
  stress uses ("this coming Friday", "in a fortnight", "on 12/17", "on 17 Dec 2026"), on top of v2's tiers.
- Date noise: typos inside date words ("Decmber", "tomorow") and shorthand ("tmr", "Fri"); stress has its own
  shorthand ("2moro", "Thurs") and the keyboard-neighbour typo that training never sees.
- train = v2's 3,000 training rows (same text and labels, new prompt) + 1,500 date-focused rows.
- New test and stress sets from 8 new template families ("test3"). v2's test and stress sets are kept as
  `old_test` / `old_stress` (calendar prompt) to check that nothing got worse; none of their requests is trained on.
- `*_v2prompt.jsonl` copies of the eval sets carry the v2 prompt, so the v2 model can be scored on the same rows.

Usage: python scripts/generate_data_v3.py --out data --seed 3031
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_data as v1  # noqa: E402
import generate_data_v2 as v2  # noqa: E402
from common import FIELDS, INSTRUCTION, INTENTS, build_prompt, label_to_text, needs_clarification  # noqa: E402
from generate_data import MONTHS, NUM_WORDS, WEEKDAYS, next_weekday, ordinal, render  # noqa: E402

# ---------------------------------------------------------------- prompt with a calendar
CAL_DAYS = 21
INSTRUCTION_V3 = INSTRUCTION + " Use the calendar to turn relative dates into YYYY-MM-DD."


def calendar_line(today: dt.date) -> str:
    """Compact calendar of the next CAL_DAYS days: each month once, then weekday + day.
    "Calendar: 2026-11: Thu 12, Fri 13, ..., Mon 30 | 2026-12: Tue 01, Wed 02."
    (Full ISO dates per day cost ~14 tokens each with SmolLM2's digit-by-digit tokenizer; this costs ~5.)"""
    groups: list[tuple[str, list[str]]] = []
    for i in range(1, CAL_DAYS + 1):
        d = today + dt.timedelta(days=i)
        month = f"{d.year}-{d.month:02d}"
        if not groups or groups[-1][0] != month:
            groups.append((month, []))
        groups[-1][1].append(f"{WEEKDAYS[d.weekday()][:3]} {d.day:02d}")
    return "Calendar: " + " | ".join(f"{m}: {', '.join(days)}" for m, days in groups) + "."


_CAL_RE = re.compile(r"Calendar: (.*)\.\n")


def calendar_dates(prompt: str) -> list[tuple[str, str]]:
    """Parse a prompt's calendar back into [(weekday abbreviation, ISO date), ...]; [] if there is none."""
    m = _CAL_RE.search(prompt)
    if not m:
        return []
    out = []
    for group in m.group(1).split(" | "):
        month, days = group.split(": ", 1)
        for item in days.split(", "):
            wd, day = item.split(" ")
            out.append((wd, f"{month}-{day}"))
    return out


def build_prompt_v3(request: str, today: dt.date) -> str:
    return (f"{INSTRUCTION_V3}\nToday is {WEEKDAYS[today.weekday()]}, {today.isoformat()}.\n"
            f"{calendar_line(today)}\nRequest: {request}")


# ---------------------------------------------------------------- templates
TEST3_TEMPLATES = {  # 8 new families, used only by v3 test and stress
    "book": [
        "Can you lock in a booking[ for {resource}][ {date}][ {time}] for me?",
        "I'd like to secure a spot[ in {resource}][ {date}][ {time}].",
    ],
    "reschedule": [
        "Please shift my[ {resource}] reservation over[ to {date}][ {time}].",
        "Could my booking[ for {resource}] be changed over[ to {date}][ {time}]?",
    ],
    "cancel": [
        "Please scratch my[ {resource}] reservation[ {date}][ {time}].",
        "I need to back out of my booking[ for {resource}][ {date}][ {time}].",
    ],
    "check": [
        "Is anything still free[ in {resource}][ {date}][ {time}]?",
        "Could you tell me about openings[ in {resource}][ {date}][ {time}]?",
    ],
}
SPLIT_CFG = {
    "train_aug3": dict(n=1500, p_missing=0.28),
    "val": dict(n=300, p_missing=0.30),
    "test": dict(n=300, p_missing=0.50),
    "stress": dict(n=300, p_missing=0.50),
}
P_DATE_BOOST = 0.6  # if sampling left the date out, put it back with this probability (date-focused data)

# ---------------------------------------------------------------- date wording
NEW_DATE_EXTRA = ["weekday_bare", "this_weekday", "in_a_week", "in_two_weeks", "month_day_year", "weekday_ordinal",
                  "numeric_dm", "day_only"]
NEW_DATE_HELDOUT = ["coming_weekday", "fortnight", "numeric_md", "day_abbr_year"]
DATE_FORMS = {
    "base": list(v2.DATE_FORMS["base"]),
    "extra": v2.DATE_FORMS["extra"] + NEW_DATE_EXTRA,
    "heldout": v2.DATE_FORMS["heldout"] + NEW_DATE_HELDOUT,
}


def _target(today: dt.date, rng: random.Random, lo: int, hi: int, day_over_12: bool = False) -> dt.date:
    while True:
        t = today + dt.timedelta(days=rng.randint(lo, hi))
        if not day_over_12 or t.day > 12:
            return t


def date_text(today: dt.date, form: str, rng: random.Random) -> tuple[str, str]:
    """(surface text, ISO date). v2 forms are delegated to generate_data_v2.date_text."""
    if form not in NEW_DATE_EXTRA + NEW_DATE_HELDOUT:
        return v2.date_text(today, form, rng)
    if form in ("weekday_bare", "this_weekday", "coming_weekday"):
        w = rng.randrange(7)
        lead = {"weekday_bare": "on", "this_weekday": "this", "coming_weekday": "this coming"}[form]
        return f"{lead} {WEEKDAYS[w]}", next_weekday(today, w).isoformat()   # rule: the first such day after today
    if form == "in_a_week":
        return "in a week", (today + dt.timedelta(days=7)).isoformat()
    if form in ("in_two_weeks", "fortnight"):
        text = "in two weeks" if form == "in_two_weeks" else rng.choice(["in a fortnight", "a fortnight from today"])
        return text, (today + dt.timedelta(days=14)).isoformat()
    if form in ("weekday_ordinal", "day_only"):
        t = _target(today, rng, 1, CAL_DAYS - 1)           # inside the calendar, so "the 17th" has one meaning
        text = f"on {WEEKDAYS[t.weekday()]} the {ordinal(t.day)}" if form == "weekday_ordinal" else f"on the {ordinal(t.day)}"
        return text, t.isoformat()
    if form in ("numeric_dm", "numeric_md"):
        t = _target(today, rng, 1, 90, day_over_12=True)   # day > 12: day/month order is unambiguous
        return (f"on {t.day}/{t.month}" if form == "numeric_dm" else f"on {t.month}/{t.day}"), t.isoformat()
    t = _target(today, rng, 1, 90)
    if form == "month_day_year":
        return f"on {MONTHS[t.month - 1]} {t.day}, {t.year}", t.isoformat()
    return f"on {t.day} {MONTHS[t.month - 1][:3]} {t.year}", t.isoformat()   # day_abbr_year


def pick_date(today: dt.date, tier: str, rng: random.Random) -> tuple[str, str, str]:
    form = rng.choice(DATE_FORMS[tier])
    text, iso = date_text(today, form, rng)
    return text, iso, form


# ---------------------------------------------------------------- noise inside the date
SHORTHAND = {
    "train": {"tomorrow": "tmr", "Monday": "Mon", "Tuesday": "Tue", "Wednesday": "Wed", "Thursday": "Thu",
              "Friday": "Fri", "Saturday": "Sat", "Sunday": "Sun", "September": "Sept"},
    # stress forms never coincide with a training form, even with punctuation added
    "stress": {"tomorrow": "2moro", "Monday": "Mo", "Tuesday": "Tues", "Wednesday": "Weds", "Thursday": "Thurs",
               "Friday": "Fr", "Saturday": "Sa", "Sunday": "Su"},
}
MIN_TYPO_LEN = 5   # shorter words lose their meaning with a typo ("tmrw" -> "mrw", "July" -> "Juy")
DATE_TYPO_OPS = {"train": ["swap", "drop", "double"], "stress": ["neighbor", "swap", "drop"]}


def apply_date_noise(text: str, rng: random.Random, style: str) -> tuple[str, list[str]]:
    """One shorthand or one typo inside the date words; digits are never touched. Returns (text, ops)."""
    words = text.split(" ")
    short = [i for i, w in enumerate(words) if w in SHORTHAND[style]]
    if short and rng.random() < 0.5:
        i = rng.choice(short)
        words[i] = SHORTHAND[style][words[i]]
        return " ".join(words), ["date_short"]
    cand = [i for i, w in enumerate(words) if len(w) >= MIN_TYPO_LEN and w.isalpha()]
    if not cand:
        return text, []
    i = rng.choice(cand)
    op = rng.choice(DATE_TYPO_OPS[style])
    words[i] = v2.typo(words[i], op, rng)
    return " ".join(words), [f"date_typo:{op}"]


# ---------------------------------------------------------------- rows
_TO_IN = re.compile(r"\b(?:to|until) (in)\b", re.IGNORECASE)


def fix_to_in(text: str) -> str:
    """'move it to in two weeks' -> 'move it in two weeks' (v1's clean() only handled 'to in N days')."""
    return _TO_IN.sub(r"\1", text)


def recipe(split: str, rng: random.Random) -> dict:
    aug = dict(date_tier=lambda: "extra" if rng.random() < 0.7 else "base",
               tier=lambda: "extra" if rng.random() < 0.5 else "base",
               noise="train", noise_p=0.6, date_noise="train", date_noise_p=0.35, force=False, group="augmented")
    clean = dict(date_tier=lambda: "base", tier=lambda: "base", noise=None, date_noise=None, group="clean")
    if split == "train_aug3":
        return aug
    if split == "val":
        return aug if rng.random() < 0.5 else clean
    if split == "test":   # clean text; dates in any wording training covers
        return dict(clean, date_tier=lambda: "extra" if rng.random() < 0.5 else "base")
    kind = rng.choice(["noise_only", "new_wording", "both"])  # stress
    seen = lambda: "extra" if rng.random() < 0.5 else "base"  # noqa: E731
    held = lambda: "heldout"  # noqa: E731
    return dict(date_tier=seen if kind == "noise_only" else held, tier=seen if kind == "noise_only" else held,
                noise=None if kind == "new_wording" else "stress", noise_p=1.0, force=True,
                date_noise=None if kind == "new_wording" else "stress", date_noise_p=0.5, group=kind)


def make_row(split: str, rng: random.Random, start: dt.date) -> dict | None:
    cfg, rec = SPLIT_CFG[split], recipe(split, rng)
    templates = v2.TRAIN_AUG_TEMPLATES if split == "train_aug3" else (
        v1.TEMPLATES["val"] if split == "val" else TEST3_TEMPLATES)
    intent = rng.choice(INTENTS)
    t_idx = rng.randrange(len(templates[intent]))
    if split == "train_aug3":
        own = t_idx < v2.N_V1_TRAIN[intent]
        family = f"{'train' if own else 'v1test'}/{intent}/{t_idx if own else t_idx - v2.N_V1_TRAIN[intent]}"
    else:
        family = f"{'val' if split == 'val' else 'test3'}/{intent}/{t_idx}"
    today = start + dt.timedelta(days=rng.randrange(730))
    present = v1.sample_presence(intent, cfg["p_missing"], rng)
    if not present["date"] and rng.random() < P_DATE_BOOST:
        present["date"] = True
    label = {"intent": intent, "date": None, "time": None, "resource": None}
    surfaces, forms, noise = {}, {}, []
    if present["date"]:
        tier = rec["date_tier"]()
        surfaces["date"], label["date"], f = pick_date(today, tier, rng)
        forms["date"] = f"{tier}:{f}"
        if rec["date_noise"] and rng.random() < rec["date_noise_p"]:
            surfaces["date"], ops = apply_date_noise(surfaces["date"], rng, rec["date_noise"])
            noise += ops
    if present["time"]:
        tier = rec["tier"]()
        surfaces["time"], label["time"], f = v2.pick_time(tier, rng)
        forms["time"] = f"{tier}:{f}"
    if present["resource"]:
        tier = rec["tier"]()
        surfaces["resource"], label["resource"], f = v2.pick_resource(tier, rng)
        forms["resource"] = f"{tier}:{f}"
    label["needs_clarification"] = needs_clarification(label)
    segments = render(templates[intent][t_idx], surfaces)
    if rec["noise"] and rng.random() < rec["noise_p"]:
        request, ops = v2.apply_noise(segments, rng, rec["noise"], force=rec["force"])
        noise += ops
    else:
        request = v2.clean("".join(t for t, _ in segments))
    request = fix_to_in(request)
    group = rec["group"]
    if split == "stress":
        group = v2.stress_group(bool(noise), any(v.startswith("heldout:") for v in forms.values()))
        if group is None:
            return None
    return {
        "split": "train" if split == "train_aug3" else split, "template": family, "today": today.isoformat(),
        "request": request, "prompt": build_prompt_v3(request, today), "label": {k: label[k] for k in FIELDS},
        "group": group, "noise": noise, "forms": forms, "source": "v3",
    }


def reprompt(rows: list[dict], v2_prompt: bool = False) -> list[dict]:
    """Same rows with the v3 (calendar) prompt, or with the v2 prompt."""
    out = []
    for r in rows:
        today = dt.date.fromisoformat(r["today"])
        p = build_prompt(r["request"], today) if v2_prompt else build_prompt_v3(r["request"], today)
        out.append({**r, "prompt": p})
    return out


def generate(seed: int = 3031, v2_seed: int = 2027, v1_seed: int = 1234) -> dict[str, list[dict]]:
    old = v2.generate(v2_seed, v1_seed)
    seen = {v2.norm(r["request"]) for s in old.values() for r in s}       # never repeat any v2 request
    rng = random.Random(seed)
    start = dt.date(2026, 1, 1)
    out: dict[str, list[dict]] = {}
    for split in ["train_aug3", "val", "test", "stress"]:
        rows, tries = [], 0
        while len(rows) < SPLIT_CFG[split]["n"]:
            tries += 1
            if tries > 300_000:
                raise RuntimeError(f"could not generate enough unique rows for {split}")
            row = make_row(split, rng, start)
            if row is None or v2.norm(row["request"]) in seen:
                continue
            seen.add(v2.norm(row["request"]))
            row["id"] = f"{split}-{len(rows):04d}"
            rows.append(row)
        out[split] = rows
    train_old = [{**r, "source": "v2"} for r in reprompt(old["train"])]
    train = train_old + out.pop("train_aug3")
    rng.shuffle(train)
    out["train"] = train
    out["old_test"], out["old_stress"] = reprompt(old["test"]), reprompt(old["stress"])
    for s in ["val", "test", "stress"]:
        out[f"{s}_v2prompt"] = reprompt(out[s], v2_prompt=True)
    for s in ["test", "stress"]:
        out[f"old_{s}_v2prompt"] = old[s]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=3031)
    ap.add_argument("--v2-seed", type=int, default=2027)
    ap.add_argument("--v1-seed", type=int, default=1234)
    args = ap.parse_args()
    splits = generate(args.seed, args.v2_seed, args.v1_seed)
    train_chat = [{"messages": [{"role": "user", "content": r["prompt"]},
                                {"role": "assistant", "content": label_to_text(r["label"])}]}
                  for r in splits["train"]]
    files = {"train.jsonl": train_chat, "train_meta.jsonl": splits["train"],
             **{f"eval/{s}.jsonl": rows for s, rows in splits.items() if s != "train"}}
    for rel, rows in files.items():
        v1.write_jsonl(os.path.join(args.out, rel), rows)
    groups = {s: {g: sum(r.get("group") == g for r in splits[s]) for g in sorted({str(r.get("group")) for r in splits[s]})}
              for s in ["train", "val", "test", "stress"]}
    manifest = {"version": 3, "seed": args.seed, "v2_seed": args.v2_seed, "v1_seed": args.v1_seed, "groups": groups,
                "files": {rel: {"rows": len(rows), "sha256": v1.sha256(os.path.join(args.out, rel))}
                          for rel, rows in files.items()}}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    for rel, info in manifest["files"].items():
        print(f"{rel:28s} {info['rows']:5d} rows  sha256={info['sha256'][:12]}…")
    print("groups:", groups)


if __name__ == "__main__":
    main()
