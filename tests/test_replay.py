"""Tests for the replay filter (Phase 1)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from fetch_replay import SOURCE, filter_replay_row, select_replay  # noqa: E402


def row(msgs, source=SOURCE):
    return {"source": source, "messages": msgs}


U = {"role": "user", "content": "What is 2+2?"}
A = {"role": "assistant", "content": "4."}


def test_keeps_single_turn_and_drops_system():
    out = filter_replay_row(row([{"role": "system", "content": "be nice"}, U, A]))
    assert out == {"messages": [U, A]}


def test_rejects_other_sources():
    assert filter_replay_row(row([U, A], source="smol-rewrite")) is None


def test_multi_turn_keeps_first_exchange_and_rejects_wrong_order():
    later = {"role": "user", "content": "And 3+3?"}
    assert filter_replay_row(row([U, A, later, A])) == {"messages": [U, A]}
    assert filter_replay_row(row([A, U])) is None
    assert filter_replay_row(row([U])) is None


def test_rejects_empty_and_too_long():
    assert filter_replay_row(row([U, {"role": "assistant", "content": "   "}])) is None
    assert filter_replay_row(row([U, {"role": "assistant", "content": "x" * 2500}])) is None


def test_select_is_seeded_and_checks_supply():
    rows = [row([{"role": "user", "content": f"q{i}"}, A]) for i in range(50)]
    assert select_replay(rows, 10, seed=0) == select_replay(rows, 10, seed=0)
    with pytest.raises(RuntimeError):
        select_replay(rows, 51)
