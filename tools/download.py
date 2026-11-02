#!/usr/bin/env python3
"""Download a Qwen3 checkpoint from Hugging Face: config, tokenizer and weights.

usage: python3 tools/download.py Qwen/Qwen3-0.6B models/Qwen3-0.6B

Standard library only. Files already there are skipped, so an interrupted
download picks up at the file it stopped in.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def fetch(url, path):
    part = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url) as r, open(part, "wb") as f:
        size, done = int(r.headers.get("Content-Length", 0)), 0
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            if size > 1e8:
                print(f"\r{path.name}: {done / 1e9:.2f} of {size / 1e9:.2f} GB", end="", file=sys.stderr)
    if size > 1e8:
        print(file=sys.stderr)
    part.rename(path)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    repo, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{repo}/resolve/main/"
    try:  # bigger models come in shards listed by an index
        with urllib.request.urlopen(base + "model.safetensors.index.json") as r:
            index = r.read()
        (out / "model.safetensors.index.json").write_bytes(index)
        weights = sorted(set(json.loads(index)["weight_map"].values()))
    except urllib.error.HTTPError:
        weights = ["model.safetensors"]
    for name in ["config.json", "tokenizer.json", "tokenizer_config.json", *weights]:
        if not (out / name).exists():
            fetch(base + name, out / name)


if __name__ == "__main__":
    main()
