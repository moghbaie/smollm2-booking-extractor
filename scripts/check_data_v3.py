"""Phase 1 data checks for the v3 data; prints rows to read for check 1.8.

Exit code 1 if any check fails. Writes a summary to <reports>/data_checks.json.
Usage: python scripts/check_data_v3.py --data data --reports reports
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_data_v2 as v2  # noqa: E402
import generate_data_v3 as v3  # noqa: E402
from check_data import chat_ok, load_jsonl, norm  # noqa: E402
from common import INTENTS, build_prompt, label_errors  # noqa: E402
from generate_data import WEEKDAYS, sha256  # noqa: E402

NEW_EVAL = ["val", "test", "stress"]
OLD_EVAL = ["old_test", "old_stress"]
N_TRAIN = 3000 + v3.SPLIT_CFG["train_aug3"]["n"]


def calendar_ok(prompt: str, today: dt.date) -> bool:
    """The calendar lists exactly the next CAL_DAYS days, each with its correct weekday."""
    days = v3.calendar_dates(prompt)
    want = [today + dt.timedelta(days=i) for i in range(1, v3.CAL_DAYS + 1)]
    return len(days) == len(want) and all(w == WEEKDAYS[d.weekday()][:3] and iso == d.isoformat()
                                          for (w, iso), d in zip(days, want))


def run_checks(data: str) -> list[dict]:
    results: list[dict] = []

    def check(cid: str, name: str, ok: bool, detail: str) -> None:
        results.append({"id": cid, "name": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {cid:5s} {name:30s} {detail}")

    train = load_jsonl(os.path.join(data, "train.jsonl"))
    meta = load_jsonl(os.path.join(data, "train_meta.jsonl"))
    ev = {s: load_jsonl(os.path.join(data, "eval", f"{s}.jsonl"))
          for s in NEW_EVAL + OLD_EVAL + [f"{s}_v2prompt" for s in NEW_EVAL] + [f"{s}_v2prompt" for s in OLD_EVAL]}

    # 1.2 structure ------------------------------------------------------------------------------------
    bad = [i for i, r in enumerate(train) if not chat_ok(r)]
    aligned = len(train) == len(meta) and all(r["messages"][0]["content"] == m["prompt"] for r, m in zip(train, meta))
    src = Counter(m["source"] for m in meta)
    check("1.2", "structure", not bad and aligned and len(train) == N_TRAIN and src["v2"] == 3000,
          f"{len(train)} train rows ({src['v2']} from v2, {src['v3']} new), {len(bad)} malformed, meta aligned={aligned}")

    # 1.3 labels ---------------------------------------------------------------------------------------
    labels = {"train": [json.loads(r["messages"][1]["content"]) for r in train], **{s: [r["label"] for r in v] for s, v in ev.items()}}
    invalid = [(s, i) for s, v in labels.items() for i, lab in enumerate(v) if label_errors(lab)]
    match = all(a == m["label"] for a, m in zip(labels["train"], meta))
    check("1.3", "labels pass schema", not invalid and match,
          f"{sum(len(v) for v in labels.values())} labels, {len(invalid)} invalid {invalid[:3]}, meta match={match}")

    # 1.4 leakage --------------------------------------------------------------------------------------
    rows = {"train": meta, **{s: ev[s] for s in NEW_EVAL + OLD_EVAL}}
    req = {s: {norm(r["request"]) for r in v} for s, v in rows.items()}
    shared = {f"{a}&{b}": len(req[a] & req[b]) for a, b in itertools.combinations(rows, 2)
              if not {a, b} <= set(OLD_EVAL)}
    fams = {s: {r["template"] for r in v} for s, v in rows.items()}
    disjoint = not (fams["train"] & (fams["val"] | fams["test"] | fams["old_test"]) or fams["val"] & fams["test"])
    new_fams = all(f.startswith("test3/") for f in fams["test"]) and fams["stress"] <= fams["test"]
    check("1.4", "no leakage", sum(shared.values()) == 0 and disjoint and new_fams,
          f"shared requests={shared}, families disjoint={disjoint}, test families new and stress⊆test={new_fams}")

    # 1.H held-out wording, stress noise and stress shorthand only in stress ----------------------------
    stress_short = set(v3.SHORTHAND["stress"].values())
    def leaks(v):
        h = sum(any(f.startswith("heldout:") for f in r.get("forms", {}).values()) for r in v)
        n = sum(any(o.endswith(":neighbor") for o in r.get("noise", [])) for r in v)
        f = sum(any(r["request"].lower().startswith(x + " ") for x in v2.STRESS_PRE)
                or any(r["request"].lower().endswith(" " + x) for x in v2.STRESS_POST) for r in v)
        s = sum(any(w.strip(".,?!:;") in stress_short for w in r["request"].split()) for r in v)
        return h, n, f, s
    seen = {s: leaks(rows[s]) for s in ["train", "val", "test"]}
    share = sum(any(f.startswith("heldout:") for f in r["forms"].values()) for r in ev["stress"]) / len(ev["stress"])
    check("1.H", "held-out only in stress", all(x == (0, 0, 0, 0) for x in seen.values()) and share >= 0.5,
          f"(held-out wording, neighbour typos, stress fillers, stress shorthand) per split={seen}; stress held-out share={share:.0%}")

    # 1.D date coverage ----------------------------------------------------------------------------------
    def form_counts(v, tier):
        return Counter(r["forms"]["date"].split(":")[1] for r in v if r["forms"].get("date", "").startswith(tier + ":"))
    tr_new = [m for m in meta if m["source"] == "v3"]
    tr_forms = Counter()
    for t in ["base", "extra"]:
        tr_forms.update(form_counts(tr_new, t))
    st_held = form_counts(ev["stress"], "heldout")
    miss_train = [f for f in v3.DATE_FORMS["base"] + v3.DATE_FORMS["extra"] if tr_forms[f] < 10]
    miss_stress = [f for f in v3.DATE_FORMS["heldout"] if st_held[f] < 3]
    date_noise = Counter(o.split(":")[0] for m in tr_new for o in m["noise"] if o.startswith("date"))
    st_noise = Counter(o.split(":")[0] for r in ev["stress"] for o in r["noise"] if o.startswith("date"))
    check("1.D", "date coverage", not miss_train and not miss_stress and sum(date_noise.values()) >= 200 and sum(st_noise.values()) >= 20,
          f"train forms <10 rows: {miss_train}; stress held-out forms <3 rows: {miss_stress}; "
          f"date noise train={dict(date_noise)}, stress={dict(st_noise)}")

    # 1.C calendar --------------------------------------------------------------------------------------
    all_rows = meta + [r for s in NEW_EVAL + OLD_EVAL for r in ev[s]]
    bad_cal = sum(not calendar_ok(r["prompt"], dt.date.fromisoformat(r["today"])) for r in all_rows)
    in_cal = [r for r in all_rows if r["label"]["date"] and 1 <= (dt.date.fromisoformat(r["label"]["date"]) - dt.date.fromisoformat(r["today"])).days <= v3.CAL_DAYS]
    listed = sum(r["label"]["date"] in {iso for _, iso in v3.calendar_dates(r["prompt"])} for r in in_cal)
    v2p_ok = all(r["prompt"] == build_prompt(r["request"], dt.date.fromisoformat(r["today"]))
                 for s in ev if s.endswith("_v2prompt") for r in ev[s])
    same_rows = all([(a["id"], a["request"]) for a in ev[s]] == [(b["id"], b["request"]) for b in ev[f"{s}_v2prompt"]]
                    for s in NEW_EVAL + OLD_EVAL)
    check("1.C", "calendar and prompt copies", bad_cal == 0 and listed == len(in_cal) and v2p_ok and same_rows,
          f"{len(all_rows)} prompts, {bad_cal} with a wrong calendar; label date listed in {listed}/{len(in_cal)} rows "
          f"where it falls inside the calendar; v2-prompt copies correct={v2p_ok}, same rows={same_rows}")

    # 1.O old sets are v2's test/stress, unchanged ---------------------------------------------------------
    old = v2.generate()
    same_old = all([(r["id"], r["request"], r["label"]) for r in ev[f"old_{s}"]] ==
                   [(r["id"], r["request"], r["label"]) for r in old[s]] for s in ["test", "stress"])
    check("1.O", "old sets = v2 test/stress", same_old, f"old_test and old_stress identical to v2's rows: {same_old}")

    # 1.5 coverage ---------------------------------------------------------------------------------------
    cov, ok = {}, True
    for s, v in labels.items():
        if s.endswith("_v2prompt") or s.startswith("old_"):
            continue
        share_m = sum(l["needs_clarification"] for l in v) / len(v)
        cov[s] = f"{len({l['intent'] for l in v})} intents, missing={share_m:.0%}"
        ok &= {l["intent"] for l in v} == set(INTENTS) and 0.10 <= share_m <= 0.60
    groups = Counter(r["group"] for r in ev["stress"])
    ok &= len(groups) == 3 and min(groups.values()) >= 60
    check("1.5", "coverage", ok, "; ".join(f"{s}: {c}" for s, c in cov.items()) + f"; stress groups={dict(groups)}")

    # 1.6 checksums ----------------------------------------------------------------------------------------
    manifest = json.load(open(os.path.join(data, "manifest.json")))
    mism = [rel for rel, info in manifest["files"].items()
            if not os.path.exists(os.path.join(data, rel)) or sha256(os.path.join(data, rel)) != info["sha256"]]
    check("1.6", "checksums", not mism, f"{len(manifest['files'])} files in manifest, mismatched={mism}")

    # replay -------------------------------------------------------------------------------------------------
    rpath = os.path.join(data, "replay.jsonl")
    need = round(0.1 / 0.9 * N_TRAIN * 0.9)   # the fallback trainer: replay = r/(1-r) x (train after the 10% val split)
    if os.path.exists(rpath):
        replay = load_jsonl(rpath)
        rbad = sum(not chat_ok(r) for r in replay)
        check("1.R", "replay file", rbad == 0 and len(replay) >= need, f"{len(replay)} rows (need {need}), {rbad} malformed")
    else:
        check("1.R", "replay file", False, f"{rpath} missing (run fetch_replay.py)")

    # 1.8 rows to read ----------------------------------------------------------------------------------------
    rng = random.Random()
    noisy = [m for m in tr_new if any(o.startswith("date") for o in m["noise"])]
    pick = [("train", r) for r in rng.sample(noisy, 3)] + [("train", r) for r in rng.sample(tr_new, 2)]
    pick += [("test", r) for r in rng.sample(ev["test"], 2)]
    for g in ["noise_only", "new_wording", "both"]:
        pick += [("stress", r) for r in rng.sample([r for r in ev["stress"] if r["group"] == g], 2)]
    print("\n1.8  Read these rows — each date should be obviously right (calendar rule: 'next/this/on Friday' = "
          "the first Friday after today):")
    for s, r in pick:
        d = dt.date.fromisoformat(r["today"])
        print(f"[{s:6s} {r['group']:11s}] today={r['today']} ({WEEKDAYS[d.weekday()][:3]}) | {r['request']}\n"
              f"         date form={r['forms'].get('date')} noise={r['noise']}\n         -> {json.dumps(r['label'])}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()
    results = run_checks(args.data)
    os.makedirs(args.reports, exist_ok=True)
    with open(os.path.join(args.reports, "data_checks.json"), "w") as f:
        json.dump(results, f, indent=2)
    failed = [r["id"] for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} data checks passed" + (f"; FAILED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
