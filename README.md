# abacus

A CPU inference engine for Qwen3-0.6B in about 1,000 lines of C. It starts as a naive float32 loop and gets faster one commit at a time, and every commit is raced against llama.cpp.

![Decode and prefill tokens/s for every commit, llama.cpp Q8_0 dashed](results/ladder.svg)

Where it lands against llama.cpp Q8_0 with the same int8 weights: prefill is 1.5–1.7× faster at 1, 4 and 8 threads, and decode is a tie, 4–5% ahead at 4 and 8 threads and 4% behind on one thread. llama.cpp's 4-bit Q4_0 still decodes faster at 8 threads (208 vs 185 tok/s), since it reads half the bytes.

## Run it

```sh
make
make model                                  # downloads Qwen3-0.6B (1.5 GB) and converts it, needs python3
./abacus model.bin -p "Why is the sky blue?"
```

`-t N` sets threads (default: all cores), `--temp 0` makes it greedy, `--raw` skips the chat template.

## The ladder

Each line is one commit. Numbers are median decode tokens/s at 8 threads, or prefill tokens/s where the line says so, from the final benchmark run.

1. [scalar fp32 engine: 2.7 tok/s](https://github.com/mottopanikeiku/abacus/commit/4b4c4f19ad62a130c8a08e3ed4b4af0e6da03fa6)
2. [threads: 2.7 -> 20 tok/s](https://github.com/mottopanikeiku/abacus/commit/d134f2310e20e630c40534b048a88fe2d086fc8c)
3. [neon fp32 dot: 20 -> 46 tok/s](https://github.com/mottopanikeiku/abacus/commit/55d0323cfacadc7d65d9cd3ce3d0e361a515e908)
4. [bf16 weights: 46 -> 68 tok/s](https://github.com/mottopanikeiku/abacus/commit/e078d48f2b74a7537bf22bdb10a114c2f36e4a1c)
5. [int8 weights, sdot: 68 -> 100 tok/s](https://github.com/mottopanikeiku/abacus/commit/8ec0e7bdd6d706a652a9cf45b98be4496fc83be2)
6. [spin-wait pool: 115 -> 208 prefill tok/s](https://github.com/mottopanikeiku/abacus/commit/b29302de63ef2e0ee8cd2fa6cb6eaa4483ec5d60)
7. [batched prefill: 208 -> 466 prefill tok/s](https://github.com/mottopanikeiku/abacus/commit/266bd929440a34712b006b1c3109d297bedb4cb2)
8. [i8mm prefill: 466 -> 584 prefill tok/s](https://github.com/mottopanikeiku/abacus/commit/f2555673404a7174b251d056f3d67d893d0fec67)
9. [4-row interleave: 584 -> 635 prefill tok/s](https://github.com/mottopanikeiku/abacus/commit/f3e7e391d0b984ce79ca0350f93ed4aa0c8a7d84)
10. [fused qkv, gate/up: 169 -> 173 tok/s](https://github.com/mottopanikeiku/abacus/commit/7f10e34cf039c1c90e776103007252b1bd045b61)
11. [parallel norms, swiglu: 636 -> 930 prefill tok/s](https://github.com/mottopanikeiku/abacus/commit/2e48ed27e936ded2941e1066ca68983151cec3bd)
12. [rope table: 180 -> 185 tok/s](https://github.com/mottopanikeiku/abacus/commit/e95a153f9eb5a1b92bfad1b4190f72ddc40c9a10)
13. [8x4 i8mm tiles: 938 -> 1051 prefill tok/s](https://github.com/mottopanikeiku/abacus/commit/d44797cbaf6dbfcc6d53ead731bdeb38b142d197)

Every commit passes the same check against Hugging Face transformers in float32 (`make check`): identical greedy tokens and logits within 0.002 for the float commits; for int8, the same next token as the reference at 90% or more of positions and mean KL at most 0.02. The int8 engine lands at 97–98% and 0.006–0.008.

## Results

Qwen3-0.6B, 128-token prompt, 128 generated tokens, median of 5 interleaved runs on a 16-core AWS Graviton4 (Neoverse-V2). Threads are pinned to the first N cores for both engines.

| Threads | abacus decode | llama.cpp Q8_0 | ratio | abacus prefill | llama.cpp Q8_0 | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45 | 47 | 0.96x | 165 | 97 | 1.70x |
| 4 | 127 | 120 | 1.05x | 582 | 385 | 1.51x |
| 8 | 185 | 177 | 1.04x | 1051 | 672 | 1.56x |

Both read the same int8 weights: blocks of 32 values with one f16 scale, 8.5 bits per weight. Every commit's numbers with 95% intervals, and llama.cpp Q4_0 for reference, are in [results/table.md](results/table.md).

## How it works

| File | Lines | What |
|---|---:|---|
| [src/model.c](src/model.c) | 316 | load the model, run the transformer on a batch of tokens |
| [src/matmul.c](src/matmul.c) | 188 | int8 matrix products: SDOT for one token, SMMLA for four |
| [src/tokenizer.c](src/tokenizer.c) | 183 | byte-level BPE with Qwen's pre-tokenizer rules |
| [src/main.c](src/main.c) | 187 | chat, benchmark and check modes |
| [src/pool.c](src/pool.c) | 64 | a spinning thread pool |

Weights are quantized to int8 at load and interleaved in groups of four rows, so one token streams a single array and SMMLA finds its row pairs side by side. A prompt runs as one batch of up to 128 tokens, so each weight is read once per batch. Decoding at 1 to 8 threads runs at the machine's memory bandwidth, which is why the decode gap to llama.cpp is small; prefill is compute-bound, and that is where the kernels pull ahead.

NEON with dotprod and i8mm is the fast path (Graviton3/4, Apple M2 and later). Other CPUs, including x86, get a plain C fallback; CI checks both on a tiny random Qwen3.

## Reproduce

```sh
uv run tools/reference.py models/Qwen3-0.6B ref    # transformers reference, once
uv run bench/run.py --threads 1,4,8 --reps 5     # check and time every commit and llama.cpp
```

`bench/run.py` builds each engine commit in its own git worktree, checks it, builds llama.cpp at a pinned commit (`LLAMA_COMMIT`) with its Q8_0 and Q4_0 conversions, then times everything in a fresh random order per repetition. `bench/report.py` redraws the chart and [results/table.md](results/table.md) from [results/raw.jsonl](results/raw.jsonl).

MIT license. Model weights are Qwen's, Apache-2.0.
