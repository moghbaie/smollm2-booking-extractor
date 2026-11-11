"""Tests for notebook helpers and the fallback trainer's data handling (Phases 3-5)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import sprint  # noqa: E402
from train_hf import split_and_mix  # noqa: E402


@pytest.fixture()
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_deep_merge_keeps_untouched_keys():
    merged = sprint.deep_merge(sprint.SFT_CONFIG, {"training": {"lr": 2e-4, "lora": {"r": 8, "alpha": 16}}})
    assert merged["training"]["lr"] == 2e-4 and merged["training"]["lora"] == {"r": 8, "alpha": 16, "dropout": 0.05}
    assert merged["training"]["epochs"] == 2 and merged["data"] == sprint.SFT_CONFIG["data"]
    assert sprint.SFT_CONFIG["training"]["lr"] == 1e-4  # base config not mutated


def test_make_config_sets_own_output_folder(in_tmp):
    import yaml
    path, cfg = sprint.make_config("lr_0.0002", {"training": {"lr": 2e-4}})
    on_disk = yaml.safe_load(open(path))
    assert on_disk == cfg and cfg["output"] == "./runs/lr_0.0002"
    assert sprint.settings(cfg) == {"lr": 2e-4, "epochs": 2, "rank": 16, "seed": 42}


def test_loss_history_prefers_newest_trainer_state(in_tmp):
    for step, loss in [(10, 2.0), (20, 1.0)]:
        d = in_tmp / "runs" / "x" / f"checkpoint-{step}"
        d.mkdir(parents=True)
        hist = [{"step": s, "loss": loss + 1 / s} for s in range(5, step + 1, 5)] + [{"step": step, "eval_loss": loss}]
        json.dump({"global_step": step, "log_history": hist}, open(d / "trainer_state.json", "w"))
    h = sprint.loss_history("runs/x")
    assert h["global_step"] == 20 and h["eval"] == [(20, 1.0)] and len(h["train"]) == 4


def test_loss_history_falls_back_to_rewind(in_tmp):
    d = in_tmp / "runs" / "y"
    d.mkdir(parents=True)
    with open(d / "rewind.jsonl", "w") as f:
        for loss in [3.0, 2.0, 1.5]:
            f.write(json.dumps({"loss": loss, "tokens": 100}) + "\n")
    h = sprint.loss_history("runs/y")
    assert h["train"] == [(1, 3.0), (2, 2.0), (3, 1.5)]
    assert sprint.loss_history("runs/none")["source"] is None


def test_check_records_and_hard_fail(in_tmp):
    assert sprint.check("9.1", True, "ok")
    sprint.check("9.2", None, "look at it")
    with pytest.raises(sprint.CheckFailed):
        sprint.check("9.3", False, "broken", hard=True)
    saved = json.load(open("reports/checks.json"))
    assert [saved[c]["status"] for c in ("9.1", "9.2", "9.3")] == ["PASS", "REVIEW", "FAIL"]


def test_upsert_replaces_same_run(in_tmp):
    os.makedirs("reports")
    sprint._upsert_csv("reports/t.csv", {"run": "a", "x": 1}, ("run",))
    sprint._upsert_csv("reports/t.csv", {"run": "b", "x": 2}, ("run",))
    sprint._upsert_csv("reports/t.csv", {"run": "a", "x": 3}, ("run",))
    rows = list(__import__("csv").DictReader(open("reports/t.csv")))
    assert sorted((r["run"], r["x"]) for r in rows) == [("a", "3"), ("b", "2")]


def test_split_and_mix_matches_soup_replay_formula():
    train = [{"i": i} for i in range(2000)]
    replay = [{"r": i} for i in range(300)]
    tr, val, n_replay = split_and_mix(train, replay, 0.1, 0.1, seed=42)
    assert len(val) == 200 and n_replay == round(0.1 / 0.9 * 1800) == 200 and len(tr) == 2000
    assert not ({id(x) for x in val} & {id(x) for x in tr})
    tr2, val2, n2 = split_and_mix(train, [], 0.1, 0.1, seed=42)
    assert n2 == 0 and len(tr2) == 1800 and val2 == val
