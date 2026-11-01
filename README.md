# abacus

A small CPU inference engine for Qwen3-0.6B, in C.

```sh
make
make model                                  # downloads Qwen3-0.6B (1.5 GB) and converts it
./abacus model.bin -p "Why is the sky blue?"
```

`make check` compares it with Hugging Face transformers (needs [uv](https://docs.astral.sh/uv/)).
