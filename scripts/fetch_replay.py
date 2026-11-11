"""Download general-chat replay rows from HuggingFaceTB/smol-smoltalk (smol-magpie-ultra-short).

Replay rows are mixed into training to limit forgetting; they are never scored.
Usage: python scripts/fetch_replay.py --out data/replay.jsonl --n 300
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATASET = "HuggingFaceTB/smol-smoltalk"
SOURCE = "smol-magpie-ultra-short"


def filter_replay_row(row: dict, max_chars: int = 1800) -> dict | None:
    """First user -> assistant exchange of a SOURCE chat (magpie chats are usually multi-turn);
    system prompts dropped; None = reject."""
    if row.get("source") != SOURCE:
        return None
    msgs = [m for m in (row.get("messages") or []) if m.get("role") != "system"][:2]
    if len(msgs) != 2 or msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
        return None
    user = (msgs[0].get("content") or "").strip()
    asst = (msgs[1].get("content") or "").strip()
    if not user or not asst or len(user) + len(asst) > max_chars:
        return None
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": asst}]}


def select_replay(rows, n: int, seed: int = 0, max_chars: int = 1800) -> list[dict]:
    kept = [r for r in (filter_replay_row(x, max_chars) for x in rows) if r is not None]
    if len(kept) < n:
        raise RuntimeError(f"only {len(kept)} replay rows passed the filter, need {n}")
    return random.Random(seed).sample(kept, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/replay.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset  # imported here so the tests don't need `datasets`
    ds = load_dataset(DATASET, split="train")
    if "source" not in ds.column_names:
        raise RuntimeError(f"{DATASET} has no 'source' column: {ds.column_names}")
    ds = ds.filter(lambda s: s == SOURCE, input_columns="source")
    print(f"{len(ds)} rows with source={SOURCE}")
    if len(ds) == 0:
        raise RuntimeError(f"no rows with source={SOURCE}; values seen: {sorted(set(load_dataset(DATASET, split='test')['source']))}")
    rows = select_replay(ds, args.n, args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # record the file in the data manifest next to it
    from generate_data import sha256
    data_dir = os.path.dirname(args.out) or "."
    mpath = os.path.join(data_dir, "manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {"files": {}}
    rel = os.path.relpath(args.out, data_dir).replace(os.sep, "/")
    manifest["files"][rel] = {"rows": len(rows), "sha256": sha256(args.out)}
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
