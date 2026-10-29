"""Tokenize text locally; all model arithmetic runs in the C++ engine.

Use --chat for the pinned snapshot's chat template. Generation stops at the
snapshot's configured EOS tokens. Stdout contains only text; --stats optionally
writes engine metadata as JSON to stderr (or to a supplied file).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def stop_token_ids(tokenizer, snapshot: Path) -> list[int]:
    """Prefer generation EOS (possibly multiple IDs), then tokenizer EOS."""
    config = snapshot / "generation_config.json"
    value = json.loads(config.read_text()).get("eos_token_id") if config.exists() else None
    if value is None:
        value = tokenizer.eos_token_id
    values = value if isinstance(value, list) else [value]
    if not values or any(type(token) is not int or not 0 <= token < len(tokenizer) for token in values):
        raise ValueError("tokenizer/generation configuration must provide valid EOS token IDs")
    return list(dict.fromkeys(values))


def prompt_token_ids(tokenizer, prompt: str, chat: bool) -> list[int]:
    if chat:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
        )
    else:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not ids:
        raise ValueError("prompt must produce at least one token")
    return ids


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="BF16 snapshot or quantized engine directory")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Original pinned Hugging Face snapshot")
    parser.add_argument("--engine", type=Path, default=Path("build/cpu-decode"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--chat", action="store_true", help="Apply the snapshot's chat template and assistant prefix")
    parser.add_argument("--steps", type=int, default=32, help="Maximum generated tokens, including EOS")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--kernel", choices=["auto", "scalar", "simd256", "simd512", "simd512x4", "vnni", "vnni16"], default="auto",
                        help="auto selects supported FP32 arithmetic; vnni/vnni16 explicitly quantize activations")
    parser.add_argument("--kv", choices=["f16", "f32", "i8", "i8-centered"], default="f16")
    parser.add_argument("--lookup", type=int, default=0, help="Maximum lookup draft tokens; 0 disables speculation")
    parser.add_argument("--ngram", type=int, default=4)
    parser.add_argument("--prefill-batch", type=int, default=8)
    parser.add_argument("--affinity", choices=["strict", "unpinned"], default="strict")
    parser.add_argument("--cpu-set", help="Explicit comma-separated CPU IDs (strict affinity only)")
    parser.add_argument("--stats", nargs="?", const="-", metavar="PATH",
                        help="Write JSON metadata to PATH; omit PATH or use - for stderr")
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
        ids = prompt_token_ids(tokenizer, args.prompt, args.chat)
        stops = stop_token_ids(tokenizer, args.tokenizer)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    command = [str(args.engine), "generate", "--model", str(args.model),
               "--tokens", ",".join(map(str, ids)), "--steps", str(args.steps),
               "--stop-tokens", ",".join(map(str, stops)), "--threads", str(args.threads),
               "--kernel", args.kernel, "--kv", args.kv, "--lookup", str(args.lookup),
               "--ngram", str(args.ngram), "--prefill-batch", str(args.prefill_batch),
               "--affinity", args.affinity]
    if args.cpu_set is not None:
        command += ["--cpu-set", args.cpu_set]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        parser.exit(completed.returncode, completed.stderr)
    result = json.loads(completed.stdout)
    print(tokenizer.decode(result["generated_tokens"], skip_special_tokens=True))
    if args.stats is not None:
        encoded = json.dumps(result, indent=2) + "\n"
        if args.stats == "-":
            sys.stderr.write(encoded)
        else:
            path = Path(args.stats)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(encoded)


if __name__ == "__main__":
    main()
