#!/usr/bin/env python3
"""Convert a Hugging Face Qwen3 checkpoint directory into one abacus model file.

usage: python3 tools/export.py MODEL_DIR OUT.bin

Standard library only. Weights are written as bf16, the dtype Qwen3 ships in.
"""
import array
import json
import struct
import sys
from pathlib import Path


def byte_decoder():
    """Inverse of GPT-2's bytes_to_unicode: printable character -> byte."""
    bs = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def read_safetensors(model_dir):
    tensors = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for name, info in header.items():
            if name != "__metadata__":
                start, end = info["data_offsets"]
                tensors[name] = (path, 8 + n + start, end - start, info["dtype"])
    return tensors


def bf16_bytes(tensor):
    path, offset, size, dtype = tensor
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read(size)
    if dtype == "BF16":
        return data
    if dtype != "F32":
        sys.exit(f"unsupported dtype {dtype}")
    bits = array.array("I", data)  # round to nearest even
    return array.array("H", [(u + 0x7FFF + ((u >> 16) & 1)) >> 16 for u in bits]).tobytes()


def tokenizer_section(model_dir):
    tj = json.loads((model_dir / "tokenizer.json").read_text())
    vocab, dec = tj["model"]["vocab"], byte_decoder()
    added = {t["content"]: t["id"] for t in tj["added_tokens"]}
    n_tokens = max([*vocab.values(), *added.values()]) + 1
    pieces = [b""] * n_tokens
    for s, i in vocab.items():
        pieces[i] = bytes(dec[ch] for ch in s)
    for s, i in added.items():
        pieces[i] = s.encode()
    merges = [m.split(" ") if isinstance(m, str) else m for m in tj["model"]["merges"]]
    out = [struct.pack("<3i", n_tokens, len(added), len(merges))]
    out += [struct.pack("<H", len(p)) + p for p in pieces]
    out += [struct.pack(f"<{len(added)}i", *sorted(added.values()))]
    out += [struct.pack("<3i", vocab[a], vocab[b], vocab[a + b]) for a, b in merges]
    return b"".join(out), added


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    model_dir, out_path = Path(sys.argv[1]).expanduser(), sys.argv[2]
    cfg = json.loads((model_dir / "config.json").read_text())
    if cfg.get("model_type") != "qwen3":
        sys.exit(f"expected a qwen3 model, got {cfg.get('model_type')}")
    tensors = read_safetensors(model_dir)
    tok, added = tokenizer_section(model_dir)
    tied = cfg.get("tie_word_embeddings", False) or "lm_head.weight" not in tensors
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    rope_theta = cfg.get("rope_theta") or cfg["rope_parameters"]["rope_theta"]  # newer transformers nest it
    header = b"abacus\0\0" + struct.pack(
        "<10i2f2i", 1, cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"],
        cfg["num_attention_heads"], cfg["num_key_value_heads"], head_dim, cfg["vocab_size"],
        cfg["max_position_embeddings"], int(tied), rope_theta, cfg["rms_norm_eps"],
        added["<|endoftext|>"], added["<|im_end|>"])

    names = ["model.embed_tokens.weight"]
    for l in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{l}."
        names += [p + n + ".weight" for n in (
            "input_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "self_attn.q_norm", "self_attn.k_norm", "post_attention_layernorm",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")]
    names.append("model.norm.weight")
    if not tied:
        names.append("lm_head.weight")

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(tok)
        for name in names:
            f.write(b"\0" * (-f.tell() % 64))  # every tensor starts on a 64-byte boundary
            f.write(bf16_bytes(tensors[name]))
        size = f.tell()
    print(f"wrote {out_path}: {size / 1e9:.2f} GB, {len(names)} tensors")


if __name__ == "__main__":
    main()
