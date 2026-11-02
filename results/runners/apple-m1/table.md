Qwen3-0.6B, 128-token prefill and 128-token decode. Median tokens/s of 5 interleaved runs in each of 2 separate benchmark jobs (a job may start at a later commit); brackets are 95% bootstrap intervals.

## Decode tokens/s

| # | Commit | 1 thr | 2 thr | 3 thr |
|---:|---|---:|---:|---:|
| 16 | runtime i8mm, macos | 51.9 [37.5, 64.5] | 59.1 [42.6, 71.8] | 54.3 [50.1, 67.8] |
| 17 | 4-bit weights | 51.5 [39.7, 64.6] | 63.7 [49.4, 71.3] | 59.0 [49.8, 68.6] |
| 18 | parallel load | 53.5 [38.4, 64.4] | 63.8 [44.2, 73.1] | 52.3 [46.6, 71.3] |
| 19 | hadamard rotation | 52.2 [37.4, 64.0] | 63.2 [48.7, 71.1] | 54.3 [45.6, 74.9] |
| 20 | sdot prefill tiles | 54.3 [36.9, 65.0] | 57.6 [41.8, 70.0] | 61.0 [47.8, 73.0] |
| 17 | 4-bit weights --q4 | 56.0 [43.1, 68.8] | 90.9 [55.6, 98.8] | 92.6 [60.2, 102] |
| 18 | parallel load --q4 | 57.0 [45.4, 68.1] | 80.3 [51.2, 97.8] | 62.0 [56.5, 95.1] |
| 19 | hadamard rotation --q4 | 59.0 [47.1, 68.1] | 91.1 [57.9, 102] | 75.3 [50.7, 94.4] |
| 20 | sdot prefill tiles --q4 | 56.3 [47.6, 68.0] | 84.7 [55.8, 101] | 65.8 [59.3, 98.7] |
|  | llama.cpp Q4_0 | 60.4 [50.1, 69.1] | 96.7 [69.8, 109] | 80.2 [74.0, 121] |
|  | llama.cpp Q8_0 | 54.7 [42.8, 68.2] | 66.0 [48.8, 76.7] | 62.5 [48.7, 74.9] |

## Prefill tokens/s

| # | Commit | 1 thr | 2 thr | 3 thr |
|---:|---|---:|---:|---:|
| 16 | runtime i8mm, macos | 109 [82.6, 129] | 206 [136, 233] | 211 [168, 327] |
| 17 | 4-bit weights | 118 [103, 128] | 221 [166, 233] | 280 [174, 346] |
| 18 | parallel load | 122 [99.0, 128] | 226 [179, 235] | 240 [175, 344] |
| 19 | hadamard rotation | 109 [77.8, 128] | 206 [139, 237] | 260 [158, 346] |
| 20 | sdot prefill tiles | 135 [118, 157] | 256 [181, 285] | 290 [216, 399] |
| 17 | 4-bit weights --q4 | 87.1 [72.1, 101] | 168 [115, 188] | 235 [137, 270] |
| 18 | parallel load --q4 | 89.3 [67.2, 100] | 158 [121, 187] | 195 [148, 279] |
| 19 | hadamard rotation --q4 | 85.2 [68.9, 100] | 178 [119, 186] | 176 [146, 279] |
| 20 | sdot prefill tiles --q4 | 131 [108, 149] | 241 [171, 269] | 284 [191, 396] |
|  | llama.cpp Q4_0 | 205 [162, 238] | 425 [353, 468] | 491 [328, 662] |
|  | llama.cpp Q8_0 | 197 [161, 223] | 409 [270, 432] | 501 [402, 604] |

## abacus vs llama.cpp

The ratio of abacus to llama.cpp tokens/s within each repetition, median and 95% interval over the repetitions.

| Threads | Decode: sdot prefill tiles vs llama.cpp Q8_0 | Prefill: sdot prefill tiles vs llama.cpp Q8_0 | Decode: sdot prefill tiles --q4 vs llama.cpp Q4_0 | Prefill: sdot prefill tiles --q4 vs llama.cpp Q4_0 |
|---:|---:|---:|---:|---:|
| 1 | 0.93x [0.88, 0.98] | 0.70x [0.68, 0.76] | 1.00x [0.88, 1.07] | 0.64x [0.61, 0.67] |
| 2 | 0.89x [0.83, 0.94] | 0.65x [0.56, 0.70] | 0.89x [0.78, 0.94] | 0.57x [0.49, 0.59] |
| 3 | 0.99x [0.93, 1.06] | 0.64x [0.52, 0.76] | 0.81x [0.76, 0.86] | 0.59x [0.46, 0.90] |

## Load seconds (model.bin, quantization included)

| # | Commit | 1 thr | 2 thr | 3 thr |
|---:|---|---:|---:|---:|
| 16 | runtime i8mm, macos | 1.12 [0.91, 1.76] | 1.00 [0.81, 1.85] | 1.18 [0.79, 2.05] |
| 17 | 4-bit weights | 1.68 [1.29, 2.24] | 1.46 [1.24, 1.99] | 1.56 [1.28, 1.74] |
| 18 | parallel load | 1.57 [1.25, 2.41] | 0.83 [0.71, 1.32] | 0.78 [0.47, 1.06] |
| 19 | hadamard rotation | 1.64 [1.42, 2.28] | 0.92 [0.76, 1.45] | 0.80 [0.61, 1.02] |
| 20 | sdot prefill tiles | 1.88 [1.49, 2.89] | 0.92 [0.75, 1.51] | 0.91 [0.52, 1.17] |
| 17 | 4-bit weights --q4 | 2.09 [1.90, 2.88] | 2.09 [1.80, 2.78] | 2.02 [1.85, 3.17] |
| 18 | parallel load --q4 | 2.39 [1.93, 3.32] | 1.11 [1.02, 1.98] | 1.02 [0.76, 1.27] |
| 19 | hadamard rotation --q4 | 2.53 [1.98, 3.24] | 1.18 [1.02, 1.65] | 1.20 [0.81, 1.45] |
| 20 | sdot prefill tiles --q4 | 2.44 [1.95, 3.51] | 1.24 [1.02, 1.70] | 1.13 [0.68, 1.41] |

## Job to job

Each benchmark job on its own, for the newest commit and llama.cpp, first as ratios within a repetition, then as tokens/s. "apart" marks a value whose interval doesn't overlap with another job's.

| Decode | Threads | job 1 | job 2 |
|---|---:|---:|---:|
| sdot prefill tiles vs llama.cpp Q8_0 | 1 | 0.88x [0.68, 0.96] | 0.95x [0.92, 1.09] |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 1 | 1.02x [0.75, 1.13] | 0.98x [0.88, 1.06] |
| sdot prefill tiles vs llama.cpp Q8_0 | 2 | 0.84x [0.64, 1.10] | 0.92x [0.82, 0.96] |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 2 | 0.78x [0.59, 1.12] | 0.92x [0.86, 0.96] |
| sdot prefill tiles vs llama.cpp Q8_0 | 3 | 1.02x [0.93, 1.10] | 0.95x [0.72, 1.14] |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 3 | 0.80x [0.49, 1.29] | 0.81x [0.73, 0.90] |

| Decode | Threads | job 1 | job 2 |
|---|---:|---:|---:|
| sdot prefill tiles | 1 | 36.9 [34.0, 45.3] apart | 65.0 [63.4, 66.2] apart |
| sdot prefill tiles --q4 | 1 | 49.0 [41.5, 51.3] apart | 67.7 [61.4, 69.0] apart |
| llama.cpp Q4_0 | 1 | 50.2 [43.3, 55.9] apart | 69.1 [65.0, 69.9] apart |
| llama.cpp Q8_0 | 1 | 42.8 [41.5, 49.9] apart | 68.2 [59.5, 70.8] apart |
| sdot prefill tiles | 2 | 41.8 [37.8, 51.7] apart | 70.0 [63.6, 73.6] apart |
| sdot prefill tiles --q4 | 2 | 55.8 [50.4, 76.7] apart | 101 [92.8, 104] apart |
| llama.cpp Q4_0 | 2 | 69.8 [66.3, 85.5] apart | 108 [108, 110] apart |
| llama.cpp Q8_0 | 2 | 48.8 [46.9, 59.1] apart | 76.7 [72.8, 78.1] apart |
| sdot prefill tiles | 3 | 49.1 [40.3, 66.6] | 71.4 [46.4, 75.1] |
| sdot prefill tiles --q4 | 3 | 59.8 [37.1, 69.0] | 98.7 [59.3, 108] |
| llama.cpp Q4_0 | 3 | 74.0 [53.4, 78.8] apart | 121 [81.5, 122] apart |
| llama.cpp Q8_0 | 3 | 48.7 [43.3, 60.8] apart | 74.9 [64.3, 78.5] apart |

| Prefill | Threads | job 1 | job 2 |
|---|---:|---:|---:|
| sdot prefill tiles vs llama.cpp Q8_0 | 1 | 0.76x [0.45, 0.97] | 0.70x [0.68, 0.71] |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 1 | 0.67x [0.47, 0.72] | 0.62x [0.59, 0.65] |
| sdot prefill tiles vs llama.cpp Q8_0 | 2 | 0.56x [0.54, 0.76] | 0.66x [0.64, 0.67] |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 2 | 0.54x [0.38, 0.67] | 0.58x [0.56, 0.58] |
| sdot prefill tiles vs llama.cpp Q8_0 | 3 | 0.62x [0.42, 0.88] | 0.66x [0.39, 0.83] |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 3 | 0.64x [0.33, 1.20] | 0.59x [0.36, 0.60] |

| Prefill | Threads | job 1 | job 2 |
|---|---:|---:|---:|
| sdot prefill tiles | 1 | 119 [80.3, 126] apart | 157 [145, 157] apart |
| sdot prefill tiles --q4 | 1 | 110 [77.6, 122] apart | 149 [141, 151] apart |
| llama.cpp Q4_0 | 1 | 162 [156, 182] apart | 238 [229, 246] apart |
| llama.cpp Q8_0 | 1 | 164 [127, 180] apart | 223 [214, 225] apart |
| sdot prefill tiles | 2 | 189 [140, 236] apart | 285 [276, 288] apart |
| sdot prefill tiles --q4 | 2 | 171 [143, 215] apart | 268 [267, 271] apart |
| llama.cpp Q4_0 | 2 | 362 [310, 387] apart | 468 [464, 477] apart |
| llama.cpp Q8_0 | 2 | 270 [253, 389] apart | 432 [428, 438] apart |
| sdot prefill tiles | 3 | 236 [179, 340] | 399 [196, 425] |
| sdot prefill tiles --q4 | 3 | 247 [176, 391] | 396 [147, 398] |
| llama.cpp Q4_0 | 3 | 328 [227, 574] | 662 [408, 676] |
| llama.cpp Q8_0 | 3 | 402 [203, 524] | 604 [484, 627] |

Jobs: 1 = 41527513710, 2 = 41527521464.
