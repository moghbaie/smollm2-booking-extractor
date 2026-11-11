"""Greedy (deterministic) generation for an eval split, with the base model or base + LoRA adapter.

Used instead of `soup infer` because Soup 0.77.2 does not document loading a PEFT adapter
or greedy decoding for the transformers backend.

Usage:
  python scripts/predict.py --input data/eval/val.jsonl --output reports/base_val.pred.jsonl
  python scripts/predict.py --adapter runs/sft_v1 --input data/eval/val.jsonl --output reports/sft_v1_val.pred.jsonl
Output JSONL: {"id", "response", "tokens_generated"}.
"""
from __future__ import annotations

import argparse
import json
import os
import time

BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"


def find_adapter(path: str) -> str | None:
    """Folder holding adapter_config.json: the run folder itself, else the shallowest non-checkpoint
    subfolder, else the newest checkpoint-N. None if there is no adapter."""
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return path
    found = []
    for root, _dirs, files in os.walk(path):
        if "adapter_config.json" in files:
            rel = os.path.relpath(root, path)
            parts = rel.split(os.sep)
            ckpt = next((int(p.split("-")[1]) for p in parts if p.startswith("checkpoint-") and p.split("-")[1].isdigit()), None)
            found.append((ckpt is not None, -(ckpt or 0), len(parts), root))
    return sorted(found)[0][3] if found else None


def dtype_kwarg(dtype) -> dict:
    """transformers >= 4.56 takes `dtype=`; older versions only know `torch_dtype=`."""
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def load_model(base: str, adapter: str | None, dtype_name: str = "auto"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype_name == "auto":
        dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    else:
        dtype = getattr(torch, dtype_name)
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, **dtype_kwarg(dtype)).to(device)
    if adapter:
        from peft import PeftModel
        adapter_dir = find_adapter(adapter)
        if adapter_dir is None:
            raise FileNotFoundError(f"no adapter_config.json under {adapter}")
        model = PeftModel.from_pretrained(model, adapter_dir)
        print(f"loaded adapter from {adapter_dir}")
    model.eval()
    return tok, model, device


def generate(tok, model, device, prompts: list[str], batch_size: int = 32, max_new_tokens: int = 96) -> list[tuple[str, int]]:
    import torch

    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
             for p in prompts]
    out: list[tuple[str, int]] = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        for row in new:
            n = int((row != tok.pad_token_id).sum())
            out.append((tok.decode(row, skip_special_tokens=True).strip(), n))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--adapter", default=None, help="training output folder (searched for adapter_config.json)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", default="auto")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()][: args.limit]
    tok, model, device = load_model(args.base, args.adapter, args.dtype)
    t0 = time.time()
    results = generate(tok, model, device, [r["prompt"] for r in rows], args.batch_size, args.max_new_tokens)
    dt_s = time.time() - t0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r, (resp, n) in zip(rows, results):
            f.write(json.dumps({"id": r["id"], "response": resp, "tokens_generated": n}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows in {dt_s:.1f}s on {device} -> {args.output}")


if __name__ == "__main__":
    main()
