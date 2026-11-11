"""Phase 1 data checks for the v2 data (1.2-1.6, held-out wording, replay); prints rows to read for check 1.8.

Exit code 1 if any check fails. Writes a summary to reports/data_checks.json.
Usage: python scripts/check_data_v2.py --data data --v1-data data_v1 --reports reports
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_data import chat_ok, load_jsonl, norm  # noqa: E402
from common import INTENTS, label_errors  # noqa: E402
from generate_data import sha256  # noqa: E402
from generate_data_v2 import SPLIT_CFG, STRESS_POST, STRESS_PRE  # noqa: E402

EVAL_SPLITS = ["val", "test", "stress"]
N_V1_TRAIN = 2000
N_TRAIN = N_V1_TRAIN + SPLIT_CFG["train_aug"]["n"]


def run_checks(data: str, v1_data: str | None) -> list[dict]:
    results: list[dict] = []

    def check(cid: str, name: str, ok: bool, detail: str) -> None:
        results.append({"id": cid, "name": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {cid:5s} {name:28s} {detail}")

    train = load_jsonl(os.path.join(data, "train.jsonl"))
    meta = load_jsonl(os.path.join(data, "train_meta.jsonl"))
    evals = {s: load_jsonl(os.path.join(data, "eval", f"{s}.jsonl")) for s in EVAL_SPLITS}

    # 1.2 structure: chat rows, aligned metadata, and every v1 training row present unchanged ------------
    bad = [i for i, r in enumerate(train) if not chat_ok(r)]
    aligned = len(train) == len(meta) and all(
        r["messages"][0]["content"] == m["prompt"] for r, m in zip(train, meta))
    v1_detail, v1_ok = "v1 rows not compared (no --v1-data)", True
    if v1_data:
        v1_rows = load_jsonl(os.path.join(v1_data, "train.jsonl"))
        ours = {json.dumps(r, sort_keys=True) for r in train}
        missing = sum(json.dumps(r, sort_keys=True) not in ours for r in v1_rows)
        v1_ok = len(v1_rows) == N_V1_TRAIN and missing == 0
        v1_detail = f"{len(v1_rows) - missing}/{len(v1_rows)} v1 rows present unchanged"
    n_orig = sum(m["group"] == "v1_original" for m in meta)
    check("1.2", "structure", not bad and aligned and len(train) == N_TRAIN and n_orig == N_V1_TRAIN and v1_ok,
          f"{len(train)} train rows ({n_orig} v1 + {len(train) - n_orig} augmented), {len(bad)} malformed, "
          f"meta aligned={aligned}; {v1_detail}")

    # 1.3 labels ---------------------------------------------------------------------------------------
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
    expected = N_TRAIN + sum(SPLIT_CFG[s]["n"] for s in EVAL_SPLITS)
    invalid = [(s, i, label_errors(lab)) for s, v in labels.items() for i, lab in enumerate(v) if label_errors(lab)]
    train_match = all(lab == m["label"] for lab, m in zip(labels["train"], meta))
    check("1.3", "labels pass schema", parse_fail == 0 and not invalid and train_match and total == expected,
          f"{total} labels, {parse_fail} unparseable, {len(invalid)} invalid {invalid[:3]}, meta match={train_match}")

    # 1.4 leakage: no shared text; template families separate; test/stress families are new ------------
    rows = {"train": meta, **evals}
    prompts = {s: {r["prompt"] for r in v} for s, v in rows.items()}
    requests = {s: {norm(r["request"]) for r in v} for s, v in rows.items()}
    shared = {f"{a}&{b}": len(prompts[a] & prompts[b]) + len(requests[a] & requests[b])
              for a, b in itertools.combinations(rows, 2)}
    fams = {s: {r["template"] for r in v} for s, v in rows.items()}
    fam_disjoint = not (fams["train"] & fams["val"] or fams["train"] & fams["test"] or fams["val"] & fams["test"])
    stress_in_test = fams["stress"] <= fams["test"]
    new_fams = all(f.startswith("test2/") for f in fams["test"])
    check("1.4", "no leakage", sum(shared.values()) == 0 and fam_disjoint and stress_in_test and new_fams,
          f"shared prompts/requests={shared}, families disjoint={fam_disjoint}, stress⊆test={stress_in_test}, "
          f"test families new={new_fams}")

    # 1.H held-out wording and noise appear only in stress ----------------------------------------------
    def leaks(v):
        n_form = sum(any(f.startswith("heldout:") for f in r.get("forms", {}).values()) for r in v)
        n_typo = sum("typo:neighbor" in r.get("noise", []) for r in v)
        n_fill = sum(any(r["request"].lower().startswith(f + " ") for f in STRESS_PRE)
                     or any(r["request"].lower().endswith(" " + f) for f in STRESS_POST) for r in v)
        return n_form, n_typo, n_fill
    seen = {s: leaks(rows[s]) for s in ["train", "val", "test"]}
    st = evals["stress"]
    share_heldout = sum(any(f.startswith("heldout:") for f in r["forms"].values()) for r in st) / len(st)
    check("1.H", "held-out only in stress", all(x == (0, 0, 0) for x in seen.values()) and share_heldout >= 0.5,
          f"(held-out wording, neighbour typos, stress fillers) per split={seen}; "
          f"stress rows with held-out wording={share_heldout:.0%}")

    # 1.5 coverage -------------------------------------------------------------------------------------
    cov, ok = {}, True
    for s, v in labels.items():
        intents = {lab["intent"] for lab in v}
        share = sum(lab["needs_clarification"] for lab in v) / len(v)
        cov[s] = f"{len(intents)} intents, missing={share:.0%}"
        ok &= intents == set(INTENTS) and 0.10 <= share <= 0.60
    gcount = {s: {g: sum(r["group"] == g for r in rows[s]) for g in sorted({r["group"] for r in rows[s]})}
              for s in ["val", "stress"]}
    ok &= all(n >= 60 for n in gcount["stress"].values()) and len(gcount["stress"]) == 3
    ok &= all(n >= 100 for n in gcount["val"].values()) and len(gcount["val"]) == 2
    check("1.5", "coverage", ok, "; ".join(f"{s}: {c}" for s, c in cov.items()) + f"; groups={gcount}")

    # 1.6 checksums ------------------------------------------------------------------------------------
    manifest = json.load(open(os.path.join(data, "manifest.json")))
    mism = [rel for rel, info in manifest["files"].items()
            if not os.path.exists(os.path.join(data, rel)) or sha256(os.path.join(data, rel)) != info["sha256"]]
    check("1.6", "checksums", not mism, f"{len(manifest['files'])} files in manifest, mismatched={mism}")

    # replay (structure only; never scored) ------------------------------------------------------------
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

    # 1.8 rows to read by eye: 3 augmented train rows, 2 val, 2 test, 2 per stress group --------------
    rng = random.Random()
    pick = [("train", r) for r in rng.sample([m for m in meta if m["group"] == "augmented"], 3)]
    pick += [("val", r) for r in rng.sample(evals["val"], 2)] + [("test", r) for r in rng.sample(evals["test"], 2)]
    for g in ["noise_only", "new_wording", "both"]:
        pick += [("stress", r) for r in rng.sample([r for r in st if r["group"] == g], 2)]
    print("\n1.8  Read these rows — each label should be obviously right:")
    for s, r in pick:
        day = __import__("datetime").date.fromisoformat(r["today"]).strftime("%a")
        print(f"[{s:6s} {r['group']:11s}] today={r['today']} ({day}) | {r['request']}\n"
              f"         forms={r['forms']} noise={r['noise']}\n         -> {json.dumps(r['label'])}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--v1-data", default=None, help="v1 data folder, to confirm its 2,000 training rows are included")
    ap.add_argument("--reports", default="reports")
    args = ap.parse_args()
    results = run_checks(args.data, args.v1_data)
    os.makedirs(args.reports, exist_ok=True)
    with open(os.path.join(args.reports, "data_checks.json"), "w") as f:
        json.dump(results, f, indent=2)
    failed = [r["id"] for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} data checks passed" + (f"; FAILED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
