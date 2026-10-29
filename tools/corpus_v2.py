#!/usr/bin/env python3
"""Pinned, book-disjoint public-domain calibration and held-out token windows.

Preparation fixes the selection before any candidate runs. Only short excerpts
and token IDs are published. The committed excerpts and manifest are the source
of truth: Project Gutenberg has since changed the pinned Frankenstein file, so a
fresh download no longer matches its recorded SHA-256, and none is needed.
"""
import os
import sys

if not __package__:
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from tools.download_model import FILES, MODEL_ID, REVISION, file_hash

SOURCES = {
    "calibration": {
        "id": "pg84", "title": "Frankenstein; or, the Modern Prometheus",
        "author": "Mary Wollstonecraft Shelley", "publication_year": 1818,
        "url": "https://www.gutenberg.org/ebooks/84.txt.utf-8",
        "source_page": "https://www.gutenberg.org/ebooks/84",
        "sha256": "7810cd483cffcf2cc8a1d8f0d5807931e69d4f48cd14149b8c76f88af82fead3",
        "body_anchor": "\nLetter 1\n\n_To Mrs. Saville, England._",
    },
    "heldout": {
        "id": "pg1342", "title": "Pride and Prejudice", "author": "Jane Austen",
        "publication_year": 1813,
        "url": "https://www.gutenberg.org/ebooks/1342.txt.utf-8",
        "source_page": "https://www.gutenberg.org/ebooks/1342",
        "sha256": "3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75",
        "body_anchor": "It is a truth universally acknowledged,",
    },
}
LICENSE = "Public domain in the United States; check local copyright law elsewhere"
LICENSE_URL = "https://www.gutenberg.org/policy/license.html"
POLICY = {
    "version": 1, "priming_tokens": 256, "scored_tokens_per_window": 256,
    "calibration_windows": 2, "heldout_windows": 8,
    "selection": "First consecutive nonoverlapping 513-token blocks from each pinned body anchor; no shuffle, filtering or candidate-dependent selection",
    "tokenization": "Pinned Qwen tokenizer; add_special_tokens=False; no chat template or normalization",
    "partition": "Different books and authors; calibration only chooses format, heldout never tunes it",
    "format_selection": "Among 32F16,64F32,64F16,128F16 (<=8.5 matrix bits/weight), minimize calibration mean KL, then maximize top1, then minimize cross entropy, then label",
}
FORMAT_CHOICES = ((32, "f16"), (64, "f32"), (64, "f16"), (128, "f16"))
CORPUS_SHA256 = "05ee7119f5b639e88c174001628127b82bb840752fedea1c3e0b99db51d119f4"
ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "results/v2"  # Committed manifest and excerpts that match every pin.


def protect_destination(path: Path, root: Path = ROOT) -> Path:
    """Do not let any v2 writer modify archived results, including symlinks."""
    destination = path.resolve()
    result_root = (root / "results").resolve()
    if destination.is_relative_to(result_root) and not destination.is_relative_to(result_root / "v2"):
        raise ValueError("v2 output must not overwrite v1 results")
    return destination


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value, exclusive=False) -> None:
    protect_destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def window_alignment(window: dict) -> tuple[list[int], list[int], list[int]]:
    """Row j predicts tokens[j+1], never the token just consumed."""
    tokens = window["tokens"]
    start = POLICY["priming_tokens"]
    expected = start + POLICY["scored_tokens_per_window"] + 1
    if len(tokens) != expected or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Invalid window token IDs/length")
    return tokens[:-1], list(range(start, len(tokens) - 1)), tokens[start + 1:]


def validate_manifest(data: dict, directory: Path | None = None) -> None:
    if data["model_id"] != MODEL_ID or data["revision"] != REVISION or data["policy"] != POLICY:
        raise ValueError("Corpus tokenizer pin or predetermined policy changed")
    ids = set()
    for split, source in SOURCES.items():
        stored = data["sources"][split]
        if any(stored.get(key) != value for key, value in source.items()):
            raise ValueError("Corpus source pin changed")
        if stored.get("license") != LICENSE or stored.get("license_url") != LICENSE_URL:
            raise ValueError("Corpus license missing/changed")
        windows = [window for window in data["windows"] if window["split"] == split]
        if len(windows) != POLICY[f"{split}_windows"]:
            raise ValueError("Protected corpus partition/window count changed")
        excerpt_tokens = []
        for index, window in enumerate(windows):
            if (window["id"] != f"{split}-{index:02d}" or window["id"] in ids
                    or window["source_id"] != source["id"] or window["source_token_start"] != index * 513):
                raise ValueError("Protected corpus partition/selection changed")
            ids.add(window["id"])
            window_alignment(window)
            if digest_json(window["tokens"]) != window["tokens_sha256"]:
                raise ValueError("Corpus token hash mismatch")
            excerpt_tokens.extend(window["tokens"])
        if digest_json(excerpt_tokens) != stored["excerpt_tokens_sha256"]:
            raise ValueError("Corpus excerpt token hash mismatch")
        if directory is not None:
            name = stored["excerpt"]
            if Path(name).name != name or file_hash(directory / name) != stored["excerpt_sha256"]:
                raise ValueError("Corpus excerpt text hash mismatch")
    if len(ids) != len(data["windows"]):
        raise ValueError("Unknown/duplicate corpus split")


def load_manifest(path: Path) -> dict:
    if file_hash(path) != CORPUS_SHA256:
        raise ValueError("Protected corpus manifest hash changed")
    data = json.loads(path.read_text())
    validate_manifest(data, path.parent)
    return data


def check_tokenizer(model: Path, data: dict, directory: Path) -> None:
    """Require the pinned tokenizer files, then re-derive every window from its excerpt text."""
    from transformers import AutoTokenizer

    # Tokenizer-only verification avoids loading model weights during preparation.
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        size, algorithm, expected = FILES[name]
        path = model / name
        if not path.is_file():
            raise ValueError(f"Pinned tokenizer file missing: {name}")
        if path.stat().st_size != size or file_hash(path, algorithm) != expected:
            raise ValueError("Pinned tokenizer checksum mismatch")
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, trust_remote_code=False, use_fast=True)
    for split, stored in data["sources"].items():
        excerpt = (directory / stored["excerpt"]).read_text(encoding="utf-8")
        tokens = [token for window in data["windows"] if window["split"] == split for token in window["tokens"]]
        if tokenizer.encode(excerpt, add_special_tokens=False) != tokens:
            raise ValueError("Committed excerpt no longer tokenizes to the pinned windows")


def prepare(model: Path | None, output_dir: Path) -> dict:
    """Copy the hash-pinned committed corpus; with a pinned tokenizer, also re-derive its windows.

    Needs no network. The excerpt text must tokenize back to exactly the committed token windows,
    including when an existing corpus is reused.
    """
    protect_destination(output_dir)
    destination = output_dir / "corpus.json"
    source = output_dir if destination.exists() else COMMITTED
    data = load_manifest(source / "corpus.json")
    if model is not None:
        check_tokenizer(model, data, source)
    if source == output_dir:
        return data
    output_dir.mkdir(parents=True, exist_ok=True)
    for stored in data["sources"].values():
        excerpt = output_dir / stored["excerpt"]
        protect_destination(excerpt)
        if excerpt.resolve() != (COMMITTED / stored["excerpt"]).resolve():
            shutil.copyfile(COMMITTED / stored["excerpt"], excerpt)
    protect_destination(destination)
    with destination.open("xb") as stream:
        stream.write((COMMITTED / "corpus.json").read_bytes())
    return load_manifest(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="Pinned snapshot; when given, also re-tokenize the committed excerpts")
    parser.add_argument("--output-dir", type=Path, default=Path("results/v2"))
    args = parser.parse_args()
    data = prepare(args.model, args.output_dir)
    print(json.dumps({"corpus": str(args.output_dir / "corpus.json"), "sha256": file_hash(args.output_dir / "corpus.json"), "positions": {split: POLICY[f"{split}_windows"] * 256 for split in SOURCES}}, indent=2))


if __name__ == "__main__":
    main()
