Qwen3-0.6B, 128-token prefill and 128-token decode. Median tokens/s of 5 interleaved runs in each of 2 separate benchmark jobs (a job may start at a later commit); brackets are 95% bootstrap intervals.

## Decode tokens/s

| # | Commit | 1 thr | 2 thr | 4 thr |
|---:|---|---:|---:|---:|
| 13 | 8x4 i8mm tiles | 29.4 [28.8, 30.4] | 55.9 [54.2, 57.9] | 104 [99.0, 107] |
| 14 | avx2 kernels | 29.3 [28.9, 30.2] | 54.1 [53.7, 58.1] | 99.5 [97.6, 106] |
| 15 | vnni kernels | 29.4 [28.8, 30.7] | 56.0 [53.9, 57.8] | 103 [98.3, 106] |
| 16 | runtime i8mm, macos | 29.4 [29.0, 30.6] | 54.8 [53.9, 58.3] | 103 [98.9, 106] |
| 17 | 4-bit weights | 29.2 [28.7, 30.4] | 55.7 [53.3, 58.3] | 102 [99.2, 106] |
| 18 | parallel load | 29.1 [28.5, 30.1] | 55.0 [53.7, 57.9] | 103 [98.4, 106] |
| 19 | hadamard rotation | 29.1 [28.5, 30.2] | 55.4 [53.3, 57.3] | 100 [95.9, 105] |
| 20 | sdot prefill tiles | 28.9 [28.6, 30.2] | 54.8 [53.4, 57.0] | 100 [97.0, 105] |
| 17 | 4-bit weights --q4 | 30.7 [30.5, 31.1] | 57.0 [56.0, 58.1] | 105 [102, 107] |
| 18 | parallel load --q4 | 30.7 [30.6, 31.2] | 57.0 [56.2, 58.4] | 104 [102, 107] |
| 19 | hadamard rotation --q4 | 30.8 [30.3, 31.0] | 56.9 [56.3, 58.1] | 104 [101, 108] |
| 20 | sdot prefill tiles --q4 | 30.7 [30.4, 31.1] | 56.8 [56.0, 57.7] | 105 [101, 107] |
|  | llama.cpp Q4_0 | 23.1 [22.9, 23.6] | 41.0 [40.5, 41.8] | 74.0 [73.2, 75.8] |
|  | llama.cpp Q8_0 | 28.6 [27.9, 29.8] | 50.8 [49.0, 52.2] | 87.2 [84.6, 89.6] |

## Prefill tokens/s

| # | Commit | 1 thr | 2 thr | 4 thr |
|---:|---|---:|---:|---:|
| 13 | 8x4 i8mm tiles | 114 [113, 114] | 209 [209, 211] | 397 [394, 401] |
| 14 | avx2 kernels | 114 [113, 114] | 210 [208, 211] | 398 [396, 400] |
| 15 | vnni kernels | 114 [113, 114] | 210 [208, 211] | 395 [392, 400] |
| 16 | runtime i8mm, macos | 114 [113, 114] | 210 [209, 211] | 397 [395, 398] |
| 17 | 4-bit weights | 112 [112, 113] | 207 [206, 209] | 393 [390, 396] |
| 18 | parallel load | 112 [112, 113] | 208 [207, 209] | 392 [387, 396] |
| 19 | hadamard rotation | 110 [110, 111] | 204 [203, 205] | 382 [381, 388] |
| 20 | sdot prefill tiles | 110 [110, 111] | 204 [203, 206] | 385 [383, 386] |
| 17 | 4-bit weights --q4 | 98.4 [98.1, 98.6] | 183 [182, 185] | 350 [349, 351] |
| 18 | parallel load --q4 | 98.5 [98.3, 98.7] | 184 [182, 184] | 343 [330, 350] |
| 19 | hadamard rotation --q4 | 98.4 [98.1, 98.7] | 183 [182, 184] | 348 [345, 350] |
| 20 | sdot prefill tiles --q4 | 98.3 [98.0, 98.5] | 183 [182, 184] | 350 [348, 351] |
|  | llama.cpp Q4_0 | 87.1 [86.9, 87.2] | 173 [172, 173] | 340 [337, 341] |
|  | llama.cpp Q8_0 | 78.9 [78.9, 79.0] | 157 [157, 157] | 309 [307, 310] |

## abacus vs llama.cpp

The ratio of abacus to llama.cpp tokens/s within each repetition, median and 95% interval over the repetitions.

| Threads | Decode: sdot prefill tiles vs llama.cpp Q8_0 | Prefill: sdot prefill tiles vs llama.cpp Q8_0 | Decode: sdot prefill tiles --q4 vs llama.cpp Q4_0 | Prefill: sdot prefill tiles --q4 vs llama.cpp Q4_0 |
|---:|---:|---:|---:|---:|
| 1 | 1.02x [1.01, 1.03] | 1.39x [1.39, 1.40] | 1.33x [1.31, 1.33] | 1.13x [1.13, 1.13] |
| 2 | 1.09x [1.07, 1.10] | 1.30x [1.29, 1.31] | 1.38x [1.37, 1.39] | 1.06x [1.05, 1.06] |
| 4 | 1.16x [1.12, 1.17] | 1.25x [1.24, 1.26] | 1.40x [1.37, 1.42] | 1.03x [1.02, 1.04] |

## Load seconds (model.bin, quantization included)

| # | Commit | 1 thr | 2 thr | 4 thr |
|---:|---|---:|---:|---:|
| 13 | 8x4 i8mm tiles | 0.90 [0.90, 0.91] | 0.90 [0.90, 0.91] | 0.90 [0.89, 0.91] |
| 14 | avx2 kernels | 0.91 [0.90, 0.91] | 0.91 [0.90, 0.92] | 0.90 [0.90, 0.91] |
| 15 | vnni kernels | 0.90 [0.90, 0.91] | 0.91 [0.90, 0.91] | 0.90 [0.89, 0.91] |
| 16 | runtime i8mm, macos | 0.91 [0.90, 0.91] | 0.91 [0.90, 0.92] | 0.90 [0.89, 0.91] |
| 17 | 4-bit weights | 1.72 [1.71, 1.73] | 1.71 [1.71, 1.73] | 1.71 [1.70, 1.72] |
| 18 | parallel load | 1.82 [1.81, 1.83] | 0.97 [0.96, 0.97] | 0.56 [0.56, 0.57] |
| 19 | hadamard rotation | 2.09 [2.08, 2.09] | 1.10 [1.09, 1.11] | 0.63 [0.62, 0.63] |
| 20 | sdot prefill tiles | 2.07 [2.07, 2.09] | 1.09 [1.09, 1.10] | 0.63 [0.62, 0.63] |
| 17 | 4-bit weights --q4 | 2.06 [2.03, 2.07] | 2.05 [2.03, 2.08] | 2.04 [2.03, 2.07] |
| 18 | parallel load --q4 | 2.21 [2.18, 2.22] | 1.15 [1.14, 1.17] | 0.64 [0.64, 0.65] |
| 19 | hadamard rotation --q4 | 2.09 [2.07, 2.12] | 1.09 [1.08, 1.11] | 0.62 [0.61, 0.62] |
| 20 | sdot prefill tiles --q4 | 2.07 [2.05, 2.09] | 1.09 [1.08, 1.10] | 0.61 [0.61, 0.62] |

## Job to job

Each benchmark job on its own, for the newest commit and llama.cpp, first as ratios within a repetition, then as tokens/s. "apart" marks a value whose interval doesn't overlap with another job's.

| Decode | Threads | job 1 | job 2 |
|---|---:|---:|---:|
| sdot prefill tiles vs llama.cpp Q8_0 | 1 | 1.02x [1.01, 1.06] | 1.01x [1.01, 1.03] |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 1 | 1.32x [1.31, 1.41] | 1.33x [1.29, 1.33] |
| sdot prefill tiles vs llama.cpp Q8_0 | 2 | 1.09x [1.04, 1.10] | 1.09x [1.06, 1.11] |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 2 | 1.38x [1.37, 1.40] | 1.38x [1.36, 1.40] |
| sdot prefill tiles vs llama.cpp Q8_0 | 4 | 1.14x [1.09, 1.16] | 1.17x [1.13, 1.22] |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 4 | 1.39x [1.33, 1.41] | 1.42x [1.39, 1.45] |

| Decode | Threads | job 1 | job 2 |
|---|---:|---:|---:|
| sdot prefill tiles | 1 | 28.6 [28.4, 28.8] apart | 30.2 [29.1, 30.3] apart |
| sdot prefill tiles --q4 | 1 | 30.5 [30.2, 30.8] | 31.1 [30.6, 31.4] |
| llama.cpp Q4_0 | 1 | 22.9 [21.8, 23.3] | 23.6 [23.0, 24.2] |
| llama.cpp Q8_0 | 1 | 27.9 [27.0, 28.4] apart | 29.8 [28.8, 30.0] apart |
| sdot prefill tiles | 2 | 53.5 [52.1, 54.0] apart | 57.0 [55.7, 57.8] apart |
| sdot prefill tiles --q4 | 2 | 56.1 [55.7, 56.2] apart | 57.6 [57.5, 58.1] apart |
| llama.cpp Q4_0 | 2 | 40.7 [39.8, 40.9] apart | 41.7 [41.2, 42.1] apart |
| llama.cpp Q8_0 | 2 | 49.0 [48.8, 50.1] apart | 52.1 [51.5, 52.7] apart |
| sdot prefill tiles | 4 | 97.3 [92.2, 97.8] apart | 104 [103, 106] apart |
| sdot prefill tiles --q4 | 4 | 102 [97.9, 103] apart | 107 [106, 108] apart |
| llama.cpp Q4_0 | 4 | 73.5 [72.7, 73.7] apart | 75.7 [74.4, 76.7] apart |
| llama.cpp Q8_0 | 4 | 84.6 [83.0, 87.9] | 89.6 [86.5, 91.0] |

| Prefill | Threads | job 1 | job 2 |
|---|---:|---:|---:|
| sdot prefill tiles vs llama.cpp Q8_0 | 1 | 1.39x [1.39, 1.39] apart | 1.40x [1.40, 1.40] apart |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 1 | 1.13x [1.13, 1.13] | 1.13x [1.13, 1.13] |
| sdot prefill tiles vs llama.cpp Q8_0 | 2 | 1.29x [1.29, 1.30] apart | 1.31x [1.30, 1.31] apart |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 2 | 1.05x [1.05, 1.06] apart | 1.06x [1.06, 1.07] apart |
| sdot prefill tiles vs llama.cpp Q8_0 | 4 | 1.25x [1.23, 1.32] | 1.25x [1.22, 1.26] |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 4 | 1.03x [1.02, 1.06] | 1.03x [1.02, 1.05] |

| Prefill | Threads | job 1 | job 2 |
|---|---:|---:|---:|
| sdot prefill tiles | 1 | 110 [109, 110] apart | 111 [110, 111] apart |
| sdot prefill tiles --q4 | 1 | 98.0 [97.8, 98.2] apart | 98.5 [98.4, 98.7] apart |
| llama.cpp Q4_0 | 1 | 86.9 [86.9, 87.0] apart | 87.2 [87.1, 87.3] apart |
| llama.cpp Q8_0 | 1 | 78.9 [78.8, 79.0] | 79.0 [78.9, 79.0] |
| sdot prefill tiles | 2 | 203 [202, 204] apart | 206 [205, 206] apart |
| sdot prefill tiles --q4 | 2 | 182 [182, 182] apart | 184 [183, 185] apart |
| llama.cpp Q4_0 | 2 | 172 [172, 173] apart | 173 [173, 173] apart |
| llama.cpp Q8_0 | 2 | 157 [157, 157] apart | 157 [157, 158] apart |
| sdot prefill tiles | 4 | 384 [380, 386] | 386 [381, 389] |
| sdot prefill tiles --q4 | 4 | 350 [347, 350] | 351 [347, 352] |
| llama.cpp Q4_0 | 4 | 339 [329, 341] | 341 [336, 343] |
| llama.cpp Q8_0 | 4 | 307 [291, 309] | 310 [308, 313] |

Jobs: 1 = 41527509779, 2 = 41527517661.
