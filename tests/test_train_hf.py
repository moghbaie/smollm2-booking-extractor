"""Loss masking of the fallback trainer: only the assistant reply is trained on.
Needs `transformers` and the SmolLM2 tokenizer (downloaded once); skipped otherwise."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
transformers = pytest.importorskip("transformers")

from train_hf import build_example  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    try:
        return transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    except OSError as e:  # offline
        pytest.skip(f"tokenizer unavailable: {e}")


def test_only_reply_tokens_have_labels(tok):
    reply = '{"intent": "cancel", "date": null, "time": null, "resource": null, "needs_clarification": true}'
    ex = build_example(tok, [{"role": "user", "content": "Cancel it."}, {"role": "assistant", "content": reply}], 512)
    trained = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
    assert tok.decode(trained).startswith(reply) and "<|im_end|>" in tok.decode(trained)
    assert "Cancel it." not in tok.decode(trained)
    assert all(l in (-100, t) for t, l in zip(ex["input_ids"], ex["labels"]))


def test_truncation_keeps_lengths_aligned(tok):
    ex = build_example(tok, [{"role": "user", "content": "x " * 600}, {"role": "assistant", "content": "ok"}], 64)
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"]) == 64
