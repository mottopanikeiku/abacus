"""Reference token ids and logits from Hugging Face transformers, in float32.

usage: uv run tools/reference.py MODEL_DIR OUT_DIR

Writes OUT_DIR/ref.json (prompt ids, greedy continuations, tokenizer cases) and
OUT_DIR/logits.npy, float32 [prompt, step, vocab]: the logits each greedy token
was chosen from.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STEPS = 32

PROMPTS = [
    ("chat", "Explain how a CPU cache works, in a few sentences."),
    ("chat", "Write a haiku about the sea, then explain it line by line."),
    ("chat", "What is the capital of Australia? Give some history."),
    ("raw",
     "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of "
     "foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, "
     "it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had "
     "everything before us, we had nothing before us, we were all going direct to Heaven, we were all going "
     "direct the other way - in short, the period was so far like the present period, that some of its "
     "noisiest authorities insisted on its being received, for good or for evil, in the superlative degree "
     "of comparison only. There were a king with a large jaw and a queen with a plain face, on the throne "
     "of England; there were a king with a large jaw and a queen with a fair face, on the throne of France. "
     "In both countries it was clearer than crystal to the lords of the State preserves of loaves and "
     "fishes, that things in general were settled for ever."),
]

TOKENIZER_CASES = [
    "Hello, world! It's 2026 and I'm testing   spaces.\nNew line",
    "  leading spaces and trailing  ",
    "tabs\tand\r\nwindows newlines\n\n\nthree",
    "Numbers: 3.14159, 1,000,000 and the 42nd",
    "CamelCase snake_case kebab-case SHOUTING",
    "I'VE DONE IT, they'll say. We'd've.",
    "Symbols: @#$%^&*()[]{}<>?/\\|~`",
    "def f(x):\n    return x ** 2  # square\n",
    "Café, naïve, résumé, Zürich",
    "中文测试 日本語のテスト",
    "emoji 🙂 and arrows → done",
]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    model_dir, out = Path(sys.argv[1]).expanduser(), Path(sys.argv[2])
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager")
    model.eval()

    prompts, logits = [], []
    for kind, text in PROMPTS:
        if kind == "chat":
            text = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(text)
        seq, steps = list(ids), []
        with torch.no_grad():
            for _ in range(STEPS):  # no KV cache: every step is a fresh full forward pass
                row = model(torch.tensor([seq])).logits[0, -1]
                steps.append(row.numpy())
                seq.append(int(row.argmax()))
        prompts.append({"text": text, "ids": ids, "greedy": seq[len(ids):]})
        logits.append(np.stack(steps))
        print(f"{len(ids)} prompt tokens -> {tok.decode(seq[len(ids):])!r}")

    cases = [{"text": t, "ids": tok.encode(t)} for t in TOKENIZER_CASES + [p["text"] for p in prompts]]
    out.mkdir(parents=True, exist_ok=True)
    (out / "ref.json").write_text(json.dumps({"model": model_dir.name, "prompts": prompts, "tokenizer": cases}))
    np.save(out / "logits.npy", np.stack(logits).astype(np.float32))


if __name__ == "__main__":
    main()
