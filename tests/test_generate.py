"""Deterministic tokenizer, text-wrapper, and synthetic native CLI behavior."""
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from tools.generate import prompt_token_ids, stop_token_ids

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tokenizer_snapshot(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    vocabulary = {name: index for index, name in enumerate(
        ("zero", "hello", "<eos>", "<other-eos>", "<user>", "<assistant>", "prompt", "<unk>")
    )}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<eos>", unk_token="<unk>",
        additional_special_tokens=["<other-eos>", "<user>", "<assistant>"],
        chat_template="{% for message in messages %}<user> {{ message['content'] }} {% endfor %}"
                      "{% if add_generation_prompt %}<assistant> zero{% endif %}",
    )
    snapshot = tmp_path / "tokenizer"
    tokenizer.save_pretrained(snapshot)
    (snapshot / "generation_config.json").write_text(json.dumps({"eos_token_id": [3, 2, 3]}))
    return tokenizer, snapshot


def test_prompt_uses_configured_template_and_assistant_prefix(tokenizer_snapshot):
    tokenizer, _ = tokenizer_snapshot
    assert prompt_token_ids(tokenizer, "prompt", chat=True) == [4, 6, 5, 0]
    assert prompt_token_ids(tokenizer, "prompt", chat=False) == [6]
    with pytest.raises(ValueError, match="at least one token"):
        prompt_token_ids(tokenizer, "", chat=False)
    tokenizer.chat_template = None
    with pytest.raises(ValueError):
        prompt_token_ids(tokenizer, "prompt", chat=True)


def test_generation_eos_precedes_tokenizer_and_preserves_zero(tokenizer_snapshot):
    tokenizer, snapshot = tokenizer_snapshot
    assert stop_token_ids(tokenizer, snapshot) == [3, 2]
    config = snapshot / "generation_config.json"
    config.write_text(json.dumps({"eos_token_id": 0}))
    assert stop_token_ids(tokenizer, snapshot) == [0]
    config.write_text(json.dumps({"eos_token_id": None}))
    assert stop_token_ids(tokenizer, snapshot) == [2]
    config.unlink()
    assert stop_token_ids(tokenizer, snapshot) == [2]
    tokenizer.eos_token = None
    with pytest.raises(ValueError, match="valid EOS"):
        stop_token_ids(tokenizer, snapshot)


@pytest.mark.parametrize("value", [[], -1, 8, True, "2", [2, -1], [2, False]])
def test_invalid_generation_eos_is_rejected(tokenizer_snapshot, value):
    tokenizer, snapshot = tokenizer_snapshot
    (snapshot / "generation_config.json").write_text(json.dumps({"eos_token_id": value}))
    with pytest.raises(ValueError, match="valid EOS"):
        stop_token_ids(tokenizer, snapshot)


@pytest.fixture
def engine():
    path = Path(os.environ.get("CPU_DECODE_ENGINE", ROOT / "build/cpu-decode")).resolve()
    if not path.is_file():
        if "CPU_DECODE_ENGINE" in os.environ:
            pytest.fail(f"Explicitly requested native engine does not exist: {path}")
        pytest.skip("Build cpu-decode to exercise synthetic native CLI behavior")
    return path


@pytest.fixture
def synthetic_model(tmp_path):
    """The real tiny Qwen forward predicts (token + 1) % 8, with margin."""
    model = tmp_path / "model"
    model.mkdir()
    width = 8
    (model / "config.json").write_text(json.dumps({
        "model_type": "qwen2", "tie_word_embeddings": True, "hidden_act": "silu",
        "hidden_size": width, "intermediate_size": width, "num_hidden_layers": 1,
        "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": width,
        "max_position_embeddings": 16, "rms_norm_eps": 1e-6, "rope_theta": 10000,
    }))
    tensors = {}

    def add(name, shape, fill=0):
        count = 1
        for dimension in shape:
            count *= dimension
        data = [fill] * count
        tensors[name] = (shape, data)
        return data

    embedding = add("model.embed_tokens.weight", [width, width])
    for index in range(width):
        embedding[index * width + index] = 1
    add("model.norm.weight", [width], 1)
    base = "model.layers.0."
    add(base + "input_layernorm.weight", [width], 1)
    add(base + "post_attention_layernorm.weight", [width], 1)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        rows = width // 2 if name in ("k_proj", "v_proj") else width
        add(base + "self_attn." + name + ".weight", [rows, width])
        if name != "o_proj":
            add(base + "self_attn." + name + ".bias", [rows])
    for name in ("gate_proj", "up_proj"):
        data = add(base + "mlp." + name + ".weight", [width, width])
        for index in range(width):
            data[index * width + index] = 1
    down = add(base + "mlp.down_proj.weight", [width, width])
    for index in range(width):
        down[((index + 1) % width) * width + index] = 1
    header, payload = {}, bytearray()
    for name, (shape, values) in tensors.items():
        offset = len(payload)
        payload.extend(struct.pack("<" + "H" * len(values), *(0x3F80 if value else 0 for value in values)))
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, len(payload)]}
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (model / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return model


def native(engine, model, command, *options):
    return subprocess.run(
        [str(engine), command, "--model", str(model), "--threads", "1", "--kernel", "scalar",
         "--affinity", "unpinned", *options], text=True, capture_output=True, cwd=ROOT,
    )


@pytest.mark.parametrize("steps", [0, 1])
def test_full_prompt_needs_no_extra_generation_slot(engine, synthetic_model, steps):
    completed = native(engine, synthetic_model, "generate", "--tokens", ",".join(["0"] * 16),
                       "--steps", str(steps), "--stop-tokens", "1")
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["generated_tokens"] == ([1] if steps else [])
    assert result["generated_token_count"] == steps
    assert result["finish_reason"] == ("stop" if steps else "length")
    assert result["lookup"]["forward_tokens"] == 0


def test_lookup_matches_fixed_and_stop_truncated_cli(engine, synthetic_model):
    prompt = "0,1,2,3,4,5,6,7,0"
    for lookup in (0, 4):
        for stops, expected, finish in (([], list(range(1, 8)) + [0], "length"),
                                        (["--stop-tokens", "3,6"], [1, 2, 3], "stop")):
            completed = native(engine, synthetic_model, "generate", "--tokens", prompt,
                               "--steps", "8", "--lookup", str(lookup), *stops)
            assert completed.returncode == 0, completed.stderr
            result = json.loads(completed.stdout)
            assert result["generated_tokens"] == expected
            assert result["generated_token_count"] == len(expected)
            assert result["finish_reason"] == finish
            assert result["steps"] == 8


def test_timing_keeps_fixed_steps_and_consumed_capacity(engine, synthetic_model):
    timed = native(engine, synthetic_model, "time-generate", "--tokens", "0,1,2,3,4,5,6,7,0",
                   "--steps", "8", "--lookup", "4", "--repeats", "1")
    assert timed.returncode == 0, timed.stderr
    assert json.loads(timed.stdout)["samples"][0]["generated_tokens"] == list(range(1, 8)) + [0]
    bench = native(engine, synthetic_model, "bench", "--tokens", "0", "--context", "15",
                   "--steps", "1", "--repeats", "1")
    assert bench.returncode == 0, bench.stderr
    assert json.loads(bench.stdout)["samples"][0]["generated_tokens"] == [1]
    overflow = native(engine, synthetic_model, "bench", "--tokens", "0", "--context", "16",
                      "--steps", "1", "--repeats", "1")
    assert overflow.returncode != 0
    assert "capacity" in overflow.stderr


@pytest.mark.parametrize("command, option", [
    ("time-generate", "--stop-tokens"), ("bench", "--stop-tokens"),
    ("bench", "--lookup"), ("bench", "--prefill-batch"), ("bench", "--ngram"),
])
def test_timing_rejects_generation_only_options(engine, tmp_path, command, option):
    completed = native(engine, tmp_path / "nonexistent", command, option, "1")
    assert completed.returncode != 0
    assert option in completed.stderr
    assert "cannot open" not in completed.stderr


@pytest.mark.parametrize("stats", [None, "-", "file"])
def test_text_wrapper_outputs_only_text_and_optional_json(engine, synthetic_model, tokenizer_snapshot, tmp_path, stats):
    _, snapshot = tokenizer_snapshot
    command = [sys.executable, "-m", "tools.generate", "--engine", str(engine),
               "--model", str(synthetic_model), "--tokenizer", str(snapshot),
               "--prompt", "prompt", "--chat", "--threads", "1", "--steps", "8",
               "--lookup", "4", "--prefill-batch", "1", "--kv", "f32", "--affinity", "unpinned"]
    output = tmp_path / "stats" / "generation.json"
    if stats is not None:
        command += ["--stats"] if stats == "-" else ["--stats", str(output)]
    completed = subprocess.run(command, text=True, capture_output=True, cwd=ROOT)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "hello\n"
    if stats is None:
        assert completed.stderr == ""
    else:
        result = json.loads(completed.stderr if stats == "-" else output.read_text())
        if stats == "file":
            assert completed.stderr == ""
        assert result["prompt_tokens"] == [4, 6, 5, 0]
        assert result["generated_tokens"] == [1, 2]
        assert result["generated_token_count"] == 2
        assert result["finish_reason"] == "stop"
        assert result["stop_tokens"] == [3, 2]
        assert result["activation_dtype"] == "float32"
        assert result["kv_dtype"] == "f32"
