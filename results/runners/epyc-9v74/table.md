Qwen3-0.6B, 128-token prefill and 128-token decode. Median tokens/s of 5 interleaved runs in each of 3 separate benchmark jobs (a job may start at a later commit); brackets are 95% bootstrap intervals.

## Decode tokens/s

| # | Commit | 1 thr | 2 thr | 4 thr |
|---:|---|---:|---:|---:|
| 13 | 8x4 i8mm tiles | 1.46 [1.46, 1.46] | 1.51 [1.51, 1.51] | 2.99 [2.99, 3.00] |
| 14 | avx2 kernels | 39.3 [38.6, 39.8] | 47.8 [47.5, 49.0] | 51.3 [51.1, 53.2] |
| 15 | vnni kernels | 43.2 [43.0, 43.6] | 48.1 [47.9, 49.5] | 51.4 [51.2, 53.2] |
| 16 | runtime i8mm, macos | 43.2 [43.1, 43.5] | 48.8 [47.9, 49.6] | 52.5 [51.3, 53.7] |
| 17 | 4-bit weights | 43.1 [43.0, 43.6] | 47.9 [47.8, 49.5] | 51.5 [51.2, 53.5] |
| 18 | parallel load | 43.2 [42.9, 43.7] | 48.2 [47.7, 49.3] | 52.3 [51.2, 53.5] |
| 19 | hadamard rotation | 43.0 [42.4, 43.2] | 47.5 [47.4, 48.7] | 51.1 [50.8, 52.9] |
| 20 | sdot prefill tiles | 42.7 [42.2, 43.1] | 47.5 [47.4, 47.6] | 50.7 [49.4, 50.9] |
| 17 | 4-bit weights --q4 | 53.8 [53.5, 54.3] | 59.1 [58.7, 59.3] | 75.8 [74.6, 78.0] |
| 18 | parallel load --q4 | 53.9 [53.6, 54.1] | 58.9 [58.4, 59.2] | 74.4 [74.0, 77.3] |
| 19 | hadamard rotation --q4 | 53.6 [53.1, 53.8] | 58.4 [58.1, 59.1] | 74.1 [73.7, 77.2] |
| 20 | sdot prefill tiles --q4 | 53.7 [52.9, 53.8] | 58.3 [58.0, 58.6] | 74.0 [73.5, 74.4] |
|  | llama.cpp Q4_0 | 47.7 [47.5, 47.8] | 50.2 [49.7, 50.3] | 79.6 [78.8, 81.5] |
|  | llama.cpp Q8_0 | 31.6 [31.2, 32.3] | 34.7 [33.7, 34.7] | 51.3 [50.2, 53.3] |

## Prefill tokens/s

| # | Commit | 1 thr | 2 thr | 4 thr |
|---:|---|---:|---:|---:|
| 13 | 8x4 i8mm tiles | 1.97 [1.97, 1.97] | 2.05 [2.05, 2.05] | 4.09 [4.08, 4.09] |
| 14 | avx2 kernels | 99.6 [99.4, 99.7] | 107 [107, 107] | 208 [207, 209] |
| 15 | vnni kernels | 124 [124, 124] | 134 [134, 134] | 259 [259, 259] |
| 16 | runtime i8mm, macos | 124 [124, 124] | 134 [134, 135] | 259 [258, 260] |
| 17 | 4-bit weights | 122 [122, 122] | 133 [133, 134] | 258 [256, 258] |
| 18 | parallel load | 121 [121, 122] | 132 [132, 133] | 256 [255, 256] |
| 19 | hadamard rotation | 121 [121, 122] | 131 [131, 132] | 254 [254, 254] |
| 20 | sdot prefill tiles | 122 [121, 122] | 132 [131, 133] | 253 [253, 254] |
| 17 | 4-bit weights --q4 | 123 [123, 123] | 133 [133, 134] | 257 [254, 258] |
| 18 | parallel load --q4 | 123 [122, 123] | 132 [132, 133] | 256 [255, 256] |
| 19 | hadamard rotation --q4 | 123 [123, 123] | 132 [132, 132] | 256 [255, 256] |
| 20 | sdot prefill tiles --q4 | 123 [123, 123] | 132 [132, 133] | 256 [255, 256] |
|  | llama.cpp Q4_0 | 145 [145, 145] | 159 [159, 159] | 312 [312, 313] |
|  | llama.cpp Q8_0 | 119 [119, 119] | 117 [117, 118] | 231 [231, 232] |

## abacus vs llama.cpp

The ratio of abacus to llama.cpp tokens/s within each repetition, median and 95% interval over the repetitions.

| Threads | Decode: sdot prefill tiles vs llama.cpp Q8_0 | Prefill: sdot prefill tiles vs llama.cpp Q8_0 | Decode: sdot prefill tiles --q4 vs llama.cpp Q4_0 | Prefill: sdot prefill tiles --q4 vs llama.cpp Q4_0 |
|---:|---:|---:|---:|---:|
| 1 | 1.37x [1.35, 1.38] | 1.02x [1.02, 1.02] | 1.12x [1.11, 1.13] | 0.85x [0.84, 0.85] |
| 2 | 1.40x [1.37, 1.42] | 1.12x [1.12, 1.13] | 1.17x [1.16, 1.18] | 0.83x [0.83, 0.84] |
| 4 | 0.99x [0.97, 1.04] | 1.10x [1.09, 1.11] | 0.93x [0.92, 0.95] | 0.82x [0.82, 0.82] |

## Load seconds (model.bin, quantization included)

| # | Commit | 1 thr | 2 thr | 4 thr |
|---:|---|---:|---:|---:|
| 13 | 8x4 i8mm tiles | 3.82 [3.80, 3.85] | 3.82 [3.80, 3.85] | 3.84 [3.80, 3.85] |
| 14 | avx2 kernels | 3.70 [3.66, 3.75] | 3.70 [3.66, 3.79] | 3.71 [3.65, 3.71] |
| 15 | vnni kernels | 3.71 [3.65, 3.75] | 3.70 [3.66, 3.73] | 3.71 [3.70, 3.75] |
| 16 | runtime i8mm, macos | 3.70 [3.65, 3.71] | 3.70 [3.65, 3.79] | 3.71 [3.65, 3.79] |
| 17 | 4-bit weights | 3.46 [3.45, 3.52] | 3.46 [3.44, 3.47] | 3.46 [3.46, 3.49] |
| 18 | parallel load | 3.28 [3.27, 3.28] | 2.21 [2.21, 2.21] | 1.17 [1.16, 1.18] |
| 19 | hadamard rotation | 4.25 [4.24, 4.25] | 2.62 [2.61, 2.65] | 1.38 [1.37, 1.38] |
| 20 | sdot prefill tiles | 4.25 [4.24, 4.27] | 2.61 [2.61, 2.63] | 1.37 [1.37, 1.39] |
| 17 | 4-bit weights --q4 | 4.54 [4.53, 4.55] | 4.53 [4.53, 4.56] | 4.54 [4.53, 4.54] |
| 18 | parallel load --q4 | 4.59 [4.59, 4.60] | 3.04 [3.04, 3.05] | 1.58 [1.58, 1.58] |
| 19 | hadamard rotation --q4 | 4.76 [4.76, 4.76] | 3.04 [3.04, 3.04] | 1.59 [1.58, 1.59] |
| 20 | sdot prefill tiles --q4 | 4.76 [4.76, 4.78] | 3.04 [3.04, 3.07] | 1.58 [1.58, 1.59] |

## Job to job

Each benchmark job on its own, for the newest commit and llama.cpp, first as ratios within a repetition, then as tokens/s. "apart" marks a value whose interval doesn't overlap with another job's.

| Decode | Threads | job 2 | job 3 |
|---|---:|---:|---:|
| sdot prefill tiles vs llama.cpp Q8_0 | 1 | 1.37x [1.36, 1.38] | 1.35x [1.31, 1.39] |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 1 | 1.13x [1.12, 1.13] | 1.11x [1.11, 1.13] |
| sdot prefill tiles vs llama.cpp Q8_0 | 2 | 1.37x [1.36, 1.40] apart | 1.42x [1.41, 1.44] apart |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 2 | 1.16x [1.15, 1.18] | 1.17x [1.17, 1.19] |
| sdot prefill tiles vs llama.cpp Q8_0 | 4 | 1.00x [0.99, 1.08] | 0.99x [0.94, 1.04] |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 4 | 0.93x [0.93, 0.94] | 0.93x [0.92, 0.96] |

| Decode | Threads | job 2 | job 3 |
|---|---:|---:|---:|
| sdot prefill tiles | 1 | 43.1 [42.8, 43.2] apart | 42.2 [41.4, 42.7] apart |
| sdot prefill tiles --q4 | 1 | 53.7 [53.7, 53.9] | 52.9 [52.4, 53.8] |
| llama.cpp Q4_0 | 1 | 47.8 [47.7, 47.8] apart | 47.5 [47.4, 47.6] apart |
| llama.cpp Q8_0 | 1 | 31.5 [31.1, 31.6] | 31.5 [30.4, 31.6] |
| sdot prefill tiles | 2 | 47.4 [47.4, 47.5] | 47.6 [46.9, 47.8] |
| sdot prefill tiles --q4 | 2 | 58.4 [58.0, 58.9] | 58.2 [57.9, 59.0] |
| llama.cpp Q4_0 | 2 | 50.3 [49.8, 50.3] | 49.7 [49.6, 49.9] |
| llama.cpp Q8_0 | 2 | 34.7 [33.8, 34.9] | 33.7 [32.9, 33.9] |
| sdot prefill tiles | 4 | 50.8 [50.6, 50.9] | 50.2 [47.7, 51.1] |
| sdot prefill tiles --q4 | 4 | 74.3 [73.9, 74.4] | 73.6 [72.6, 74.5] |
| llama.cpp Q4_0 | 4 | 79.5 [78.8, 79.8] | 78.7 [77.3, 79.9] |
| llama.cpp Q8_0 | 4 | 51.1 [47.2, 51.3] | 50.8 [48.3, 51.5] |

| Prefill | Threads | job 2 | job 3 |
|---|---:|---:|---:|
| sdot prefill tiles vs llama.cpp Q8_0 | 1 | 1.02x [1.02, 1.02] | 1.02x [1.02, 1.03] |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 1 | 0.84x [0.84, 0.85] | 0.85x [0.85, 0.85] |
| sdot prefill tiles vs llama.cpp Q8_0 | 2 | 1.12x [1.12, 1.13] | 1.12x [1.11, 1.14] |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 2 | 0.83x [0.83, 0.84] | 0.83x [0.83, 0.85] |
| sdot prefill tiles vs llama.cpp Q8_0 | 4 | 1.10x [1.08, 1.11] | 1.10x [1.09, 1.17] |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 4 | 0.82x [0.82, 0.82] | 0.82x [0.65, 0.82] |

| Prefill | Threads | job 2 | job 3 |
|---|---:|---:|---:|
| sdot prefill tiles | 1 | 122 [122, 122] apart | 121 [120, 122] apart |
| sdot prefill tiles --q4 | 1 | 123 [123, 123] | 123 [123, 123] |
| llama.cpp Q4_0 | 1 | 145 [145, 146] apart | 145 [145, 145] apart |
| llama.cpp Q8_0 | 1 | 119 [119, 120] | 118 [118, 119] |
| sdot prefill tiles | 2 | 131 [131, 133] | 132 [130, 132] |
| sdot prefill tiles --q4 | 2 | 132 [132, 134] | 132 [132, 134] |
| llama.cpp Q4_0 | 2 | 159 [159, 160] | 159 [156, 159] |
| llama.cpp Q8_0 | 2 | 117 [117, 117] | 117 [117, 118] |
| sdot prefill tiles | 4 | 253 [251, 257] | 254 [253, 254] |
| sdot prefill tiles --q4 | 4 | 256 [255, 256] | 255 [203, 256] |
| llama.cpp Q4_0 | 4 | 312 [311, 313] | 312 [311, 313] |
| llama.cpp Q8_0 | 4 | 232 [229, 232] | 231 [217, 231] |

Jobs: 1 = 41521230537, 2 = 41527536846, 3 = 41536364427.
