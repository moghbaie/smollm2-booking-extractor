"""Phase 1 data checks 1.2-1.6 (+ replay), and prints 10 random rows for check 1.8.

Exit code 1 if any check fails. Writes a summary to reports/data_checks.json.
Usage: python scripts/check_data.py --data data --reports reports
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INTENTS, label_errors  # noqa: E402
from generate_data import sha256  # noqa: E402

EVAL_SPLITS = ["val", "test", "stress"]


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def chat_ok(row: dict) -> bool:
    msgs = row.get("messages")
    return (isinstance(msgs, list) and len(msgs) == 2
            and [m.get("role") for m in msgs] == ["user", "assistant"]
            and all(isinstance(m.get("content"), str) and m["content"].strip() for m in msgs))


def run_checks(data: str) -> list[dict]:
    results: list[dict] = []

    def check(cid: str, name: str, ok: bool, detail: str) -> None:
        results.append({"id": cid, "name": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {cid:5s} {name:28s} {detail}")

    train = load_jsonl(os.path.join(data, "train.jsonl"))
    meta = load_jsonl(os.path.join(data, "train_meta.jsonl"))
    evals = {s: load_jsonl(os.path.join(data, "eval", f"{s}.jsonl")) for s in EVAL_SPLITS}

    # 1.2 structure -------------------------------------------------------------
    bad = [i for i, r in enumerate(train) if not chat_ok(r)]
    aligned = len(train) == len(meta) and all(
        r["messages"][0]["content"] == m["prompt"] for r, m in zip(train, meta))
    check("1.2", "structure", not bad and aligned and len(train) == 2000,
          f"{len(train)} train rows, {len(bad)} malformed, meta aligned={aligned}")

    # 1.3 labels ----------------------------------------------------------------
    labels: dict[str, list[dict]] = {"train": []}
    parse_fail = 0
    for r in train:
        try:
            labels["train"].append(json.loads(r["messages"][1]["content"]))
        except (json.JSONDecodeError, KeyError, IndexError):
            parse_fail += 1
    for s in EVAL_SPLITS:
        labels[s] = [r["label"] for r in evals[s]]
    total = sum(len(v) for v in labels.values())
    invalid = [(s, i, label_errors(lab)) for s, v in labels.items() for i, lab in enumerate(v) if label_errors(lab)]
    train_match = all(lab == m["label"] for lab, m in zip(labels["train"], meta))
    check("1.3", "labels pass schema", parse_fail == 0 and not invalid and train_match and total == 2750,
          f"{total} labels, {parse_fail} unparseable, {len(invalid)} invalid {invalid[:3]}, meta match={train_match}")

    # 1.4 leakage ---------------------------------------------------------------
    rows = {"train": meta, **evals}
    prompts = {s: {r["prompt"] for r in v} for s, v in rows.items()}
    requests = {s: {norm(r["request"]) for r in v} for s, v in rows.items()}
    shared = {f"{a}&{b}": len(prompts[a] & prompts[b]) + len(requests[a] & requests[b])
              for a, b in itertools.combinations(rows, 2)}
    fams = {s: {r["template"] for r in v} for s, v in rows.items()}
    fam_disjoint = not (fams["train"] & fams["val"] or fams["train"] & fams["test"] or fams["val"] & fams["test"])
    stress_in_test = fams["stress"] <= fams["test"]
    check("1.4", "no leakage", sum(shared.values()) == 0 and fam_disjoint and stress_in_test,
          f"shared prompts/requests={shared}, families disjoint={fam_disjoint}, stress⊆test={stress_in_test}")

    # 1.5 coverage --------------------------------------------------------------
    cov = {}
    ok = True
    for s, v in labels.items():
        intents = {lab["intent"] for lab in v}
        share = sum(lab["needs_clarification"] for lab in v) / len(v)
        cov[s] = f"{len(intents)} intents, missing={share:.0%}"
        ok &= intents == set(INTENTS) and 0.10 <= share <= 0.60
    check("1.5", "coverage", ok, "; ".join(f"{s}: {c}" for s, c in cov.items()))

    # 1.6 checksums -------------------------------------------------------------
    manifest = json.load(open(os.path.join(data, "manifest.json")))
    mism = [rel for rel, info in manifest["files"].items()
            if not os.path.exists(os.path.join(data, rel)) or sha256(os.path.join(data, rel)) != info["sha256"]]
    check("1.6", "checksums", not mism, f"{len(manifest['files'])} files in manifest, mismatched={mism}")

    # replay (structure only; never scored) --------------------------------------
    rpath = os.path.join(data, "replay.jsonl")
    if os.path.exists(rpath):
        replay = load_jsonl(rpath)
        need = round(0.1 / 0.9 * len(train))  # Soup: n_replay = round(r/(1-r) * n_new)
        rbad = sum(not chat_ok(r) for r in replay)
        overlap = len({r["messages"][0]["content"] for r in replay} & prompts["train"])
        check("1.R", "replay file", rbad == 0 and len(replay) >= need and overlap == 0,
              f"{len(replay)} rows (Soup needs {need} at ratio 0.1), {rbad} malformed, {overlap} overlap with train")
    else:
        check("1.R", "replay file", False, f"{rpath} missing (run fetch_replay.py)")

    # 1.8 rows to read by eye ----------------------------------------------------
    rng = random.Random()
    pool = [("train", m) for m in meta] + [(s, r) for s in EVAL_SPLITS for r in evals[s]]
    print("\n1.8  Read these 10 random rows — each label should be obviously right:")
    for s, r in rng.sample(pool, 10):
        print(f"[{s:6s}] today={r['today']} ({__import__('datetime').date.fromisoformat(r['today']).strftime('%a')})"
              f" | {r['request']}\n         -> {json.dumps(r['label'])}")
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
