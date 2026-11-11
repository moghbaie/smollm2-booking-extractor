"""Fallback trainer: plain transformers + peft, reading the same YAML config as `soup train`.

Used only if `soup train` fails in Phase 3 and ALLOW_HF_FALLBACK is on. It copies Soup's settings:
val_split from train, replay mixed into train only (n_replay = round(r/(1-r) * n_train)),
loss on the assistant reply only, LoRA, cosine schedule, bf16 when available.

Usage: python scripts/train_hf.py --config configs/sft_v1.yaml [--max-rows N]
Writes the adapter and trainer_state.json to the config's `output` folder.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_example(tok, messages: list[dict], max_length: int) -> dict:
    """Token ids for a whole chat, with labels = -100 everywhere except the final assistant reply."""
    prompt_text = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(messages, tokenize=False)
    if not full_text.startswith(prompt_text):
        raise ValueError("chat template: prompt is not a prefix of the full conversation")
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    reply_ids = tok(full_text[len(prompt_text):], add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + reply_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + reply_ids)[:max_length]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def split_and_mix(train_rows: list[dict], replay_rows: list[dict], val_split: float, replay_ratio: float,
                  seed: int) -> tuple[list[dict], list[dict], int]:
    rng = random.Random(seed)
    rows = train_rows[:]
    rng.shuffle(rows)
    n_val = int(round(len(rows) * val_split))
    val, train = rows[:n_val], rows[n_val:]
    n_replay = 0
    if replay_rows and replay_ratio > 0:
        n_replay = min(len(replay_rows), round(replay_ratio / (1 - replay_ratio) * len(train)))
        train = train + rng.sample(replay_rows, n_replay)
        rng.shuffle(train)
    return train, val, n_replay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-rows", type=int, default=None, help="for quick local tests only")
    args = ap.parse_args()

    import torch
    import yaml
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    cfg = yaml.safe_load(open(args.config))
    data, tr, lora = cfg["data"], cfg["training"], cfg["training"].get("lora", {})
    out = cfg["output"]
    seed = int(tr.get("seed", 42))
    set_seed(seed)

    tok = AutoTokenizer.from_pretrained(cfg["base"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    train_rows = load_jsonl(data["train"])[: args.max_rows]
    replay_rows = load_jsonl(data["replay"]) if data.get("replay") else []
    train_rows, val_rows, n_replay = split_and_mix(train_rows, replay_rows, float(data.get("val_split", 0.1)),
                                                   float(data.get("replay_ratio", 0.0)), seed)
    max_len = int(data.get("max_length", 512))
    train_ds = [build_example(tok, r["messages"], max_len) for r in train_rows]
    val_ds = [build_example(tok, r["messages"], max_len) for r in val_rows]
    print(f"train={len(train_ds)} (incl. {n_replay} replay)  val={len(val_ds)}")

    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dkey = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    model = AutoModelForCausalLM.from_pretrained(cfg["base"], **{dkey: torch.bfloat16 if bf16 else torch.float32})
    r = int(lora.get("r", 16))
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=int(lora.get("alpha", 2 * r)),
                                             lora_dropout=float(lora.get("dropout", 0.05)),
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (n - len(seq))  # noqa: E731
        return {"input_ids": torch.tensor([pad(b["input_ids"], tok.pad_token_id) for b in batch]),
                "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
                "labels": torch.tensor([pad(b["labels"], -100) for b in batch])}

    try:
        import tensorboard  # noqa: F401
        report_to = ["tensorboard"]
    except ImportError:
        report_to = []
    bs = int(tr.get("batch_size", 16))
    ga = int(tr.get("gradient_accumulation_steps", 1))
    epochs = float(tr.get("epochs", 2))
    total_steps = math.ceil(len(train_ds) / (bs * ga)) * epochs
    # warmup as steps: transformers 5 dropped `warmup_ratio`
    warmup_steps = math.ceil(float(tr.get("warmup_ratio", 0.03)) * total_steps)
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", os.path.join(out, "tb"))  # transformers 5
    kwargs = dict(
        output_dir=out, num_train_epochs=epochs, learning_rate=float(tr.get("lr", 1e-4)),
        per_device_train_batch_size=bs, per_device_eval_batch_size=bs, gradient_accumulation_steps=ga,
        lr_scheduler_type=tr.get("scheduler", "cosine"), warmup_steps=warmup_steps,
        weight_decay=float(tr.get("weight_decay", 0.01)), seed=seed, bf16=bf16,
        eval_strategy="steps" if val_ds else "no", eval_steps=int(tr.get("eval_steps") or 25),
        logging_steps=1, save_strategy="no", report_to=report_to, logging_dir=os.path.join(out, "tb"),
        remove_unused_columns=False, use_cpu=not use_cuda,
    )
    # keep only arguments this transformers version accepts (v4 and v5 differ)
    import inspect
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    dropped = sorted(set(kwargs) - accepted)
    if dropped:
        print(f"note: this transformers version has no TrainingArguments {dropped}; skipped")
    targs = TrainingArguments(**{k: v for k, v in kwargs.items() if k in accepted})
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds or None,
                      data_collator=collate)
    trainer.train()
    if val_ds:
        trainer.evaluate()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    trainer.state.save_to_json(os.path.join(out, "trainer_state.json"))
    steps = trainer.state.global_step
    print(f"done: {steps} steps (expected ~{total_steps:.0f}) -> {out}")


if __name__ == "__main__":
    main()
