"""Write a tiny random Qwen3 checkpoint with the real Qwen3 tokenizer (for CI).

usage: uv run tools/tiny.py OUT_DIR
"""
import sys

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=128, intermediate_size=384, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=64, max_position_embeddings=4096,
                      rope_theta=1e6, rms_norm_eps=1e-6, tie_word_embeddings=True)
    model = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
    model.save_pretrained(sys.argv[1])
    AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B").save_pretrained(sys.argv[1])


if __name__ == "__main__":
    main()
