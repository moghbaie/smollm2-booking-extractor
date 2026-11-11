"""Generate the v2 data: v1 training rows + noisy, more varied training rows, and NEW val/test/stress sets.

What changed from v1:
- train = the 2,000 v1 training rows, unchanged, + 1,000 augmented rows (noise and extra wordings).
- Wording comes in three tiers. "base" = v1 wording. "extra" = new wording that training sees.
  "heldout" = wording that only the stress set uses, so stress still measures generalisation.
- Noise also has a training style and a stress style: the stress fillers are different and stress adds a
  keyboard-neighbour typo that training never sees.
- test and stress use 8 NEW template families. v1's test templates have been looked at, so they move into training.
- Every row records what was done to it (`group`, `noise`, `forms`), so stress can be scored per group:
  noise_only, new_wording, both.

Usage: python scripts/generate_data_v2.py --out data --seed 2027 --v1-seed 1234
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
from common import FIELDS, INTENTS, build_prompt, label_to_text, needs_clarification  # noqa: E402
from generate_data import MONTHS, NUM_WORDS, WEEKDAYS, next_weekday, ordinal, render  # noqa: E402


def clean(text: str) -> str:
    """v1 clean(), plus: a sentence ending in "a.m." / "p.m." keeps a single period ("10.30 a.m.." -> "10.30 a.m.")."""
    return re.sub(r"(\s[ap])\.m\.\.", r"\1.m.", v1.clean(text))

# ---------------------------------------------------------------- templates
# 8 new families, used only by test and stress. Nobody has seen rows from them.
NEW_TEST_TEMPLATES = {
    "book": [
        "Could you pencil me in[ for {resource}][ {date}][ {time}]?",
        "Put me down for a slot[ in {resource}][ {date}][ {time}].",
    ],
    "reschedule": [
        "My[ {resource}] booking needs to be moved[ to {date}][ {time}].",
        "Can my[ {resource}] reservation be switched[ to {date}][ {time}]?",
    ],
    "cancel": [
        "Take my[ {resource}] booking[ {date}][ {time}] off the schedule, we can't make it.",
        "I have to withdraw my reservation[ for {resource}][ {date}][ {time}].",
    ],
    "check": [
        "Tell me whether anything is free[ in {resource}][ {date}][ {time}].",
        "Can you see if there's an opening[ in {resource}][ {date}][ {time}]?",
    ],
}
# augmented training rows use v1's train templates + v1's (retired) test templates
TRAIN_AUG_TEMPLATES = {i: v1.TEMPLATES["train"][i] + v1.TEMPLATES["test"][i] for i in INTENTS}
N_V1_TRAIN = {i: len(v1.TEMPLATES["train"][i]) for i in INTENTS}

SPLIT_CFG = {
    "train_aug": dict(n=1000, p_missing=0.28),
    "val": dict(n=300, p_missing=0.30),
    "test": dict(n=300, p_missing=0.50),
    "stress": dict(n=300, p_missing=0.50),
}

# ---------------------------------------------------------------- wording tiers
DATE_FORMS = {
    "base": ["today", "tomorrow", "day_after", "in_n_days", "next_weekday", "month_day", "day_month", "iso", "ordinal_of"],
    "extra": ["week_from_today", "in_n_days_words", "abbr_dot", "month_day_ordinal"],
    "heldout": ["two_weeks", "days_from_now", "day_abbr", "abbr_nodot", "tmrw"],
}
TIME_FORMS = {
    "base": ["h24", "ampm", "ampm_space", "oclock", "noon"],
    "extra": ["half_past", "quarter", "dotted", "midday"],
    "heldout": ["military", "h12_part", "ampm_caps", "h_h"],
}
RESOURCE_FORMS = {
    "base": ["canonical", "lower_kind"],
    "extra": ["short", "hyphen", "word", "long"],
    "heldout": ["nospace", "hash", "number_spelled", "conference"],
}
TIERS = ["base", "extra", "heldout"]


def date_text(today: dt.date, form: str, rng: random.Random) -> tuple[str, str]:
    rel = {"today": ("today", 0), "tomorrow": ("tomorrow", 1), "day_after": ("the day after tomorrow", 2),
           "week_from_today": ("a week from today", 7), "two_weeks": ("two weeks from today", 14), "tmrw": ("tmrw", 1)}
    if form in rel:
        text, n = rel[form]
        return text, (today + dt.timedelta(days=n)).isoformat()
    if form in ("in_n_days", "in_n_days_words", "days_from_now"):
        n = rng.randint(2, 10 if form == "in_n_days_words" else 14)
        text = (f"in {NUM_WORDS[n]} days" if form == "in_n_days_words"
                else f"{n} days from now" if form == "days_from_now" else f"in {n} days")
        return text, (today + dt.timedelta(days=n)).isoformat()
    if form == "next_weekday":
        w = rng.randrange(7)
        return f"next {WEEKDAYS[w]}", next_weekday(today, w).isoformat()
    # absolute dates: always within the next 1..90 days, so "December 7" has one meaning
    target = today + dt.timedelta(days=rng.randint(1, 90))
    m, d = MONTHS[target.month - 1], target.day
    text = {
        "month_day": f"on {m} {d}",
        "day_month": f"on {d} {m}",
        "iso": f"on {target.isoformat()}",
        "ordinal_of": f"on the {ordinal(d)} of {m}",
        "month_day_ordinal": f"on {m} {ordinal(d)}",
        "abbr_dot": f"on {m[:3]}. {ordinal(d)}",
        "day_abbr": f"on {ordinal(d)} {m[:3]}",
        "abbr_nodot": f"on {m[:3]} {d}",
    }[form]
    return text, target.isoformat()


def _h12(h: int) -> tuple[int, str]:
    return (h % 12 or 12), ("am" if h < 12 else "pm")


def _part_of_day(h: int) -> str:
    return "in the morning" if h < 12 else ("in the afternoon" if h < 17 else "in the evening")


def time_text(h: int, m: int, form: str) -> str | None:
    """Surface text for h:m in `form`, or None when the form does not apply to this time."""
    h12, ap = _h12(h)
    hm12 = f"{h12}" if m == 0 else f"{h12}:{m:02d}"
    part = _part_of_day(h)  # always the part of day of the actual time (11:45 is "in the morning")
    if form == "h24":
        return f"at {h:02d}:{m:02d}"
    if form == "ampm":
        return f"at {hm12}{ap}"
    if form == "ampm_space":
        return f"at {hm12} {ap}"
    if form == "oclock":
        return f"at {h12} o'clock {part}" if m == 0 else None
    if form in ("noon", "midday"):
        return f"at {form}" if (h, m) == (12, 0) else None
    if form == "half_past":
        return f"at half past {NUM_WORDS[h12]} {part}" if m == 30 else None
    if form == "quarter":
        if m == 15:
            return f"at quarter past {NUM_WORDS[h12]} {part}"
        if m == 45:
            return f"at quarter to {NUM_WORDS[_h12(h + 1)[0]]} {part}"
        return None
    if form == "dotted":
        return f"at {h12}.{m:02d} {'a.m.' if ap == 'am' else 'p.m.'}"
    if form == "military":
        return f"at {h:02d}{m:02d}h"
    if form == "h12_part":
        return f"at {hm12} {part}"
    if form == "ampm_caps":
        return f"at {hm12}{ap.upper()}"
    if form == "h_h":
        return f"at {h}h{m:02d}"
    raise ValueError(form)


def resource_text(res: str, form: str) -> str | None:
    """Surface text for a canonical resource ("Studio 2") in `form`, or None when it does not apply."""
    kind, ident = res.split(" ")
    low = kind.lower()
    word = NUM_WORDS[int(ident)] if ident.isdigit() else None
    if form == "canonical":
        return res
    if form == "lower_kind":
        return f"{low} {ident}"
    if form == "short":
        return f"rm {ident}" if kind == "Room" else f"{low} no. {ident}"
    if form == "hyphen":
        return f"{low}-{ident}"
    if form == "word":
        return f"{low} {word}" if word else None
    if form == "long":
        return f"laboratory {ident}" if kind == "Lab" else None
    if form == "nospace":
        return f"{low}{ident}"
    if form == "hash":
        return f"{low} #{ident}"
    if form == "number_spelled":
        return f"{low} number {word}" if word else None
    if form == "conference":
        return f"conference room {ident}" if kind == "Room" else None
    raise ValueError(form)


def pick_date(today, tier, rng):
    form = rng.choice(DATE_FORMS[tier])
    text, iso = date_text(today, form, rng)
    return text, iso, form


def pick_time(tier, rng):
    h, m = rng.randint(7, 21), rng.choice([0, 0, 0, 30, 30, 15, 45])
    for form in rng.sample(TIME_FORMS[tier], len(TIME_FORMS[tier])):
        text = time_text(h, m, form)
        if text:
            return text, f"{h:02d}:{m:02d}", form
    raise AssertionError("every tier has a form that always applies")


def pick_resource(tier, rng):
    res = rng.choice(v1.RESOURCES)
    for form in rng.sample(RESOURCE_FORMS[tier], len(RESOURCE_FORMS[tier])):
        text = resource_text(res, form)
        if text:
            return text, res, form
    raise AssertionError("every tier has a form that always applies")


# ---------------------------------------------------------------- noise
TRAIN_PRE = ["hey,", "hi there!", "um so", "hello,", "ok so", "quick one:"]
TRAIN_POST = ["thanks!", "cheers", "asap pls", "thank you so much", "ty"]
STRESS_PRE = ["yo", "good morning!", "sorry to bother you, but", "hiya,", "so basically"]
STRESS_POST = ["thx!!", "much appreciated", "pls and thank you", "cheers mate", ":)"]
NOISE_STYLE = {
    "train": dict(typo_ops=["swap", "drop", "double"], pre=TRAIN_PRE, post=TRAIN_POST, strip_punct=True),
    "stress": dict(typo_ops=["neighbor", "swap", "drop"], pre=STRESS_PRE, post=STRESS_POST, strip_punct=False),
}
NOISE_P = {"typo": 0.5, "pre": 0.35, "post": 0.35, "case": 0.3, "strip_punct": 0.2}
_KEY_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
NEIGHBORS = {c: (row[i - 1] if i else "") + (row[i + 1] if i + 1 < len(row) else "")
             for row in _KEY_ROWS for i, c in enumerate(row)}


def typo(word: str, op: str, rng: random.Random) -> str:
    """Return `word` with one typo of kind `op`; always differs from `word` (len >= 4, letters only)."""
    for _ in range(20):
        i = rng.randrange(len(word) - 1)
        if op == "swap":
            out = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        elif op == "drop":
            out = word[:i] + word[i + 1:]
        elif op == "double":
            out = word[:i] + word[i] + word[i:]
        elif op == "neighbor":
            c = word[i]
            n = rng.choice(NEIGHBORS.get(c.lower(), c.lower()) or c.lower())
            out = word[:i] + (n.upper() if c.isupper() else n) + word[i + 1:]
        else:
            raise ValueError(op)
        if out != word:
            return out
    return word[:-1]  # practically unreachable; still a change


def apply_noise(segments: list[tuple[str, bool]], rng: random.Random, style: str, force: bool = False) -> tuple[str, list[str]]:
    """Noise in template words only (slot values keep their text, except for whole-sentence case changes)."""
    st = NOISE_STYLE[style]
    allowed = [op for op in NOISE_P if op != "strip_punct" or st["strip_punct"]]
    ops = [op for op in allowed if rng.random() < NOISE_P[op]]
    if force and not ops:
        ops = [rng.choice(allowed)]
    applied: list[str] = []
    segs = [list(s) for s in segments]
    if "typo" in ops:
        cand = [(i, j) for i, (t, slot) in enumerate(segs) if not slot
                for j, w in enumerate(t.split(" ")) if len(w) >= 4 and w.isalpha()]
        for i, j in rng.sample(cand, k=min(len(cand), rng.randint(1, 2))):
            words = segs[i][0].split(" ")
            op = rng.choice(st["typo_ops"])
            words[j] = typo(words[j], op, rng)
            segs[i][0] = " ".join(words)
            applied.append(f"typo:{op}")
    text = clean("".join(t for t, _ in segs))
    if "pre" in ops:
        text = f"{rng.choice(st['pre'])} {text[0].lower()}{text[1:]}"
        applied.append("filler_pre")
    if "post" in ops:
        text = f"{text} {rng.choice(st['post'])}"
        applied.append("filler_post")
    if "strip_punct" in ops and text[-1] in ".?!":
        text = text.rstrip(".?!")
        applied.append("strip_punct")
    if "case" in ops:
        upper = rng.random() < 0.4
        text = text.upper() if upper else text.lower()
        applied.append("upper" if upper else "lower")
    return text, applied


# ---------------------------------------------------------------- rows
def recipe(split: str, rng: random.Random) -> dict:
    """How to build one row: wording tier per slot and noise style."""
    aug = dict(tier=lambda: "extra" if rng.random() < 0.6 else "base", noise="train", noise_p=0.7, force=False)
    if split == "train_aug":
        return {**aug, "group": "augmented"}
    if split == "val":
        if rng.random() < 0.5:
            return {**aug, "group": "augmented"}
        return dict(tier=lambda: "base", noise=None, group="clean")
    if split == "test":
        return dict(tier=lambda: "base", noise=None, group="clean")
    kind = rng.choice(["noise_only", "new_wording", "both"])  # stress
    seen_tier = lambda: "extra" if rng.random() < 0.5 else "base"  # noqa: E731
    return dict(tier=(seen_tier if kind == "noise_only" else lambda: "heldout"),
                noise=(None if kind == "new_wording" else "stress"), noise_p=1.0, force=True, group=kind)


def stress_group(has_noise: bool, has_heldout: bool) -> str | None:
    if has_noise and has_heldout:
        return "both"
    return "noise_only" if has_noise else ("new_wording" if has_heldout else None)


def make_row(split: str, rng: random.Random, start: dt.date) -> dict | None:
    cfg, rec = SPLIT_CFG[split], recipe(split, rng)
    templates = TRAIN_AUG_TEMPLATES if split == "train_aug" else (
        v1.TEMPLATES["val"] if split == "val" else NEW_TEST_TEMPLATES)
    intent = rng.choice(INTENTS)
    t_idx = rng.randrange(len(templates[intent]))
    if split == "train_aug":
        own = t_idx < N_V1_TRAIN[intent]
        family = f"{'train' if own else 'v1test'}/{intent}/{t_idx if own else t_idx - N_V1_TRAIN[intent]}"
    else:
        family = f"{'val' if split == 'val' else 'test2'}/{intent}/{t_idx}"
    today = start + dt.timedelta(days=rng.randrange(730))
    present = v1.sample_presence(intent, cfg["p_missing"], rng)
    label = {"intent": intent, "date": None, "time": None, "resource": None}
    surfaces, forms = {}, {}
    if present["date"]:
        tier = rec["tier"]()
        surfaces["date"], label["date"], f = pick_date(today, tier, rng)
        forms["date"] = f"{tier}:{f}"
    if present["time"]:
        tier = rec["tier"]()
        surfaces["time"], label["time"], f = pick_time(tier, rng)
        forms["time"] = f"{tier}:{f}"
    if present["resource"]:
        tier = rec["tier"]()
        surfaces["resource"], label["resource"], f = pick_resource(tier, rng)
        forms["resource"] = f"{tier}:{f}"
    label["needs_clarification"] = needs_clarification(label)
    segments = render(templates[intent][t_idx], surfaces)
    noise: list[str] = []
    if rec["noise"] and rng.random() < rec["noise_p"]:
        request, noise = apply_noise(segments, rng, rec["noise"], force=rec["force"])
    else:
        request = clean("".join(t for t, _ in segments))
    group = rec["group"]
    if split == "stress":
        group = stress_group(bool(noise), any(v.startswith("heldout:") for v in forms.values()))
        if group is None:
            return None  # e.g. a heldout-wording row whose slots were all missing: draw again
    return {
        "split": "train" if split == "train_aug" else split, "template": family, "today": today.isoformat(),
        "request": request, "prompt": build_prompt(request, today), "label": {k: label[k] for k in FIELDS},
        "group": group, "noise": noise, "forms": forms,
    }


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def generate(seed: int = 2027, v1_seed: int = 1234) -> dict[str, list[dict]]:
    """train = v1's 2,000 training rows (unchanged) + augmented rows, shuffled; plus new val/test/stress."""
    v1_train = v1.generate(v1_seed)["train"]
    for r in v1_train:
        r.update(group="v1_original", noise=[], forms={})
    rng = random.Random(seed)
    start = dt.date(2026, 1, 1)
    seen = {norm(r["request"]) for r in v1_train}
    out: dict[str, list[dict]] = {}
    for split in ["train_aug", "val", "test", "stress"]:
        rows, tries = [], 0
        while len(rows) < SPLIT_CFG[split]["n"]:
            tries += 1
            if tries > 200_000:
                raise RuntimeError(f"could not generate enough unique rows for {split}")
            row = make_row(split, rng, start)
            if row is None or norm(row["request"]) in seen:  # no request text shared across or within splits
                continue
            seen.add(norm(row["request"]))
            row["id"] = f"{split}-{len(rows):04d}"
            rows.append(row)
        out[split] = rows
    train = v1_train + out.pop("train_aug")
    rng.shuffle(train)  # mix old and new rows before Soup takes its 10% validation split
    out["train"] = train
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--v1-seed", type=int, default=1234)
    args = ap.parse_args()
    splits = generate(args.seed, args.v1_seed)
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
        v1.write_jsonl(os.path.join(args.out, rel), rows)
    groups = {s: {g: sum(r["group"] == g for r in splits[s]) for g in sorted({r["group"] for r in splits[s]})}
              for s in ["train", "val", "test", "stress"]}
    manifest = {"version": 2, "seed": args.seed, "v1_seed": args.v1_seed, "groups": groups,
                "files": {rel: {"rows": len(rows), "sha256": v1.sha256(os.path.join(args.out, rel))}
                          for rel, rows in files.items()}}
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    for rel, info in manifest["files"].items():
        print(f"{rel:22s} {info['rows']:5d} rows  sha256={info['sha256'][:12]}…")
    print("groups:", groups)


if __name__ == "__main__":
    main()
