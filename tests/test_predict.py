"""Tests for adapter discovery (the layout of Soup's output folder is not documented)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from predict import find_adapter  # noqa: E402


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


def test_top_level_adapter_wins(tmp_path):
    touch(tmp_path / "adapter_config.json")
    touch(tmp_path / "checkpoint-50" / "adapter_config.json")
    assert find_adapter(str(tmp_path)) == str(tmp_path)


def test_final_subfolder_beats_checkpoints(tmp_path):
    touch(tmp_path / "checkpoint-50" / "adapter_config.json")
    touch(tmp_path / "final" / "adapter_config.json")
    assert find_adapter(str(tmp_path)) == str(tmp_path / "final")


def test_newest_checkpoint_when_only_checkpoints(tmp_path):
    for n in (9, 50, 100):
        touch(tmp_path / f"checkpoint-{n}" / "adapter_config.json")
    assert find_adapter(str(tmp_path)) == str(tmp_path / "checkpoint-100")


def test_none_when_no_adapter(tmp_path):
    touch(tmp_path / "rewind.jsonl")
    assert find_adapter(str(tmp_path)) is None
