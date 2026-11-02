Qwen3-0.6B, 128-token prefill and 128-token decode. Median tokens/s of 5 interleaved runs in each of 8 separate benchmark jobs (a job may start at a later commit); brackets are 95% bootstrap intervals.

## Decode tokens/s

| # | Commit | 1 thr | 2 thr | 4 thr |
|---:|---|---:|---:|---:|
| 13 | 8x4 i8mm tiles | 1.46 [1.46, 1.46] | 1.50 [1.50, 1.50] | 2.96 [2.90, 2.97] |
| 14 | avx2 kernels | 27.3 [26.9, 28.2] | 36.6 [35.9, 37.8] | 49.3 [48.5, 50.5] |
| 15 | vnni kernels | 27.3 [26.9, 28.2] | 36.8 [35.9, 37.7] | 48.9 [47.6, 51.6] |
| 16 | runtime i8mm, macos | 27.1 [26.7, 28.4] | 36.7 [35.8, 37.5] | 49.5 [47.6, 51.1] |
| 17 | 4-bit weights | 27.3 [26.7, 28.2] | 36.5 [35.9, 37.1] | 48.7 [48.2, 51.5] |
| 18 | parallel load | 27.6 [26.5, 28.4] | 36.6 [35.7, 37.9] | 49.7 [48.8, 51.0] |
| 19 | hadamard rotation | 28.1 [27.3, 28.8] | 37.2 [36.5, 38.4] | 50.0 [49.2, 51.7] |
| 20 | sdot prefill tiles | 27.8 [27.2, 28.7] | 36.7 [36.0, 38.6] | 49.7 [48.9, 50.8] |
| 17 | 4-bit weights --q4 | 35.2 [34.3, 36.2] | 42.6 [42.3, 43.0] | 67.2 [65.3, 67.9] |
| 18 | parallel load --q4 | 35.2 [34.0, 35.8] | 42.6 [42.1, 43.1] | 67.5 [66.1, 68.1] |
| 19 | hadamard rotation --q4 | 35.9 [35.3, 36.4] | 43.0 [42.5, 43.5] | 67.4 [66.5, 68.6] |
| 20 | sdot prefill tiles --q4 | 35.5 [35.1, 36.8] | 42.7 [42.5, 43.4] | 67.6 [66.9, 69.2] |
|  | llama.cpp Q4_0 | 34.6 [34.2, 35.3] | 38.6 [38.4, 38.9] | 70.3 [69.1, 70.9] |
|  | llama.cpp Q8_0 | 24.9 [24.2, 25.3] | 32.4 [32.2, 32.6] | 53.9 [52.5, 55.1] |

## Prefill tokens/s

| # | Commit | 1 thr | 2 thr | 4 thr |
|---:|---|---:|---:|---:|
| 13 | 8x4 i8mm tiles | 1.98 [1.98, 1.98] | 2.03 [2.03, 2.03] | 3.98 [3.97, 3.99] |
| 14 | avx2 kernels | 81.4 [81.2, 81.7] | 88.8 [88.5, 88.8] | 172 [172, 173] |
| 15 | vnni kernels | 81.4 [81.2, 81.8] | 88.7 [88.6, 88.9] | 172 [171, 173] |
| 16 | runtime i8mm, macos | 81.3 [81.1, 81.4] | 88.7 [88.4, 88.8] | 172 [172, 173] |
| 17 | 4-bit weights | 80.3 [80.0, 80.6] | 87.2 [86.9, 87.3] | 169 [168, 169] |
| 18 | parallel load | 80.3 [80.1, 80.6] | 87.2 [87.1, 87.3] | 169 [168, 169] |
| 19 | hadamard rotation | 79.9 [79.7, 80.3] | 86.3 [86.1, 86.6] | 168 [167, 168] |
| 20 | sdot prefill tiles | 80.0 [79.5, 80.3] | 86.5 [86.1, 86.6] | 168 [167, 169] |
| 17 | 4-bit weights --q4 | 81.3 [81.1, 81.4] | 91.1 [91.0, 91.3] | 175 [174, 176] |
| 18 | parallel load --q4 | 81.0 [80.8, 81.2] | 91.2 [91.1, 91.3] | 176 [176, 177] |
| 19 | hadamard rotation --q4 | 81.2 [81.1, 81.5] | 91.1 [91.0, 91.3] | 177 [176, 177] |
| 20 | sdot prefill tiles --q4 | 81.4 [81.1, 81.7] | 91.1 [91.0, 91.3] | 176 [176, 178] |
|  | llama.cpp Q4_0 | 88.3 [88.1, 88.3] | 94.4 [94.2, 94.5] | 186 [186, 187] |
|  | llama.cpp Q8_0 | 77.3 [77.1, 77.5] | 79.1 [79.0, 79.2] | 156 [155, 156] |

## abacus vs llama.cpp

The ratio of abacus to llama.cpp tokens/s within each repetition, median and 95% interval over the repetitions.

| Threads | Decode: sdot prefill tiles vs llama.cpp Q8_0 | Prefill: sdot prefill tiles vs llama.cpp Q8_0 | Decode: sdot prefill tiles --q4 vs llama.cpp Q4_0 | Prefill: sdot prefill tiles --q4 vs llama.cpp Q4_0 |
|---:|---:|---:|---:|---:|
| 1 | 1.13x [1.13, 1.15] | 1.03x [1.03, 1.04] | 1.04x [1.03, 1.04] | 0.92x [0.92, 0.93] |
| 2 | 1.15x [1.12, 1.18] | 1.09x [1.09, 1.09] | 1.11x [1.11, 1.12] | 0.97x [0.97, 0.97] |
| 4 | 0.93x [0.93, 0.94] | 1.07x [1.07, 1.08] | 0.97x [0.96, 0.98] | 0.95x [0.95, 0.95] |

## Load seconds (model.bin, quantization included)

| # | Commit | 1 thr | 2 thr | 4 thr |
|---:|---|---:|---:|---:|
| 13 | 8x4 i8mm tiles | 4.60 [4.60, 4.61] | 4.60 [4.60, 4.60] | 4.60 [4.60, 4.61] |
| 14 | avx2 kernels | 4.44 [4.44, 4.44] | 4.44 [4.43, 4.45] | 4.44 [4.43, 4.44] |
| 15 | vnni kernels | 4.44 [4.43, 4.45] | 4.45 [4.44, 4.46] | 4.44 [4.44, 4.45] |
| 16 | runtime i8mm, macos | 4.44 [4.43, 4.44] | 4.44 [4.43, 4.45] | 4.44 [4.43, 4.45] |
| 17 | 4-bit weights | 4.08 [4.08, 4.08] | 4.08 [4.08, 4.09] | 4.08 [4.08, 4.09] |
| 18 | parallel load | 4.13 [4.13, 4.13] | 2.77 [2.77, 2.77] | 1.47 [1.46, 1.47] |
| 19 | hadamard rotation | 4.75 [4.72, 4.76] | 3.20 [3.18, 3.20] | 1.67 [1.67, 1.68] |
| 20 | sdot prefill tiles | 4.73 [4.72, 4.75] | 3.20 [3.18, 3.21] | 1.68 [1.68, 1.68] |
| 17 | 4-bit weights --q4 | 5.34 [5.33, 5.34] | 5.34 [5.33, 5.35] | 5.34 [5.34, 5.35] |
| 18 | parallel load --q4 | 5.49 [5.49, 5.50] | 3.65 [3.65, 3.65] | 1.90 [1.90, 1.91] |
| 19 | hadamard rotation --q4 | 5.35 [5.35, 5.36] | 3.54 [3.54, 3.54] | 1.84 [1.84, 1.85] |
| 20 | sdot prefill tiles --q4 | 5.36 [5.35, 5.36] | 3.54 [3.54, 3.54] | 1.84 [1.84, 1.85] |

## Job to job

Each benchmark job on its own, for the newest commit and llama.cpp, first as ratios within a repetition, then as tokens/s. "apart" marks a value whose interval doesn't overlap with another job's.

| Decode | Threads | job 2 | job 3 | job 4 | job 5 | job 6 | job 7 | job 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sdot prefill tiles vs llama.cpp Q8_0 | 1 | 1.18x [1.02, 1.29] | 1.11x [1.10, 1.13] | 1.13x [1.10, 1.19] | 1.13x [1.12, 1.17] | 1.13x [1.07, 1.16] | 1.14x [1.08, 1.21] | 1.14x [1.12, 1.17] |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 1 | 1.04x [0.99, 1.12] | 1.02x [0.99, 1.04] | 1.03x [1.02, 1.07] | 1.04x [1.03, 1.04] | 1.05x [1.03, 1.06] | 1.04x [0.96, 1.08] | 1.04x [1.02, 1.05] |
| sdot prefill tiles vs llama.cpp Q8_0 | 2 | 1.09x [1.00, 1.17] apart | 1.11x [1.10, 1.12] apart | 1.15x [1.13, 1.17] apart | 1.19x [1.18, 1.20] apart | 1.18x [1.17, 1.20] apart | 1.19x [1.11, 1.20] apart | 1.13x [1.12, 1.15] apart |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 2 | 1.09x [1.03, 1.16] apart | 1.10x [1.10, 1.11] apart | 1.11x [1.11, 1.12] apart | 1.12x [1.12, 1.13] apart | 1.11x [1.11, 1.12] apart | 1.12x [1.11, 1.13] apart | 1.10x [1.09, 1.10] apart |
| sdot prefill tiles vs llama.cpp Q8_0 | 4 | 0.90x [0.83, 1.03] | 0.93x [0.93, 0.94] | 0.94x [0.90, 0.94] | 0.93x [0.61, 1.46] | 0.93x [0.92, 0.95] | 0.92x [0.86, 0.96] | 0.94x [0.93, 0.97] |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 4 | 0.95x [0.90, 0.99] | 0.96x [0.95, 0.99] | 0.97x [0.96, 0.99] | 0.98x [0.96, 0.99] | 0.97x [0.96, 0.98] | 0.96x [0.70, 1.00] | 0.97x [0.94, 0.99] |

| Decode | Threads | job 2 | job 3 | job 4 | job 5 | job 6 | job 7 | job 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sdot prefill tiles | 1 | 24.8 [21.9, 25.6] apart | 27.1 [26.9, 27.2] apart | 28.3 [26.6, 28.7] apart | 29.2 [29.0, 29.3] apart | 29.8 [28.5, 30.0] apart | 28.5 [27.4, 29.1] apart | 27.3 [26.7, 27.8] apart |
| sdot prefill tiles --q4 | 1 | 32.3 [29.2, 35.3] apart | 35.1 [34.4, 35.5] apart | 36.5 [34.7, 36.9] apart | 37.3 [36.9, 37.4] apart | 37.5 [37.1, 38.0] apart | 35.4 [33.4, 36.8] apart | 35.1 [34.5, 35.7] apart |
| llama.cpp Q4_0 | 1 | 31.6 [28.7, 33.0] apart | 34.3 [34.0, 34.6] apart | 35.0 [33.5, 35.7] apart | 35.8 [35.7, 36.0] apart | 36.1 [35.5, 36.4] apart | 34.6 [33.9, 35.0] apart | 33.9 [33.6, 34.4] apart |
| llama.cpp Q8_0 | 1 | 20.9 [19.8, 21.6] apart | 24.5 [23.8, 24.6] apart | 24.2 [24.1, 25.5] apart | 25.7 [24.9, 26.1] apart | 26.5 [25.8, 26.6] apart | 25.3 [22.8, 25.4] apart | 23.9 [23.7, 24.2] apart |
| sdot prefill tiles | 2 | 32.3 [29.8, 33.7] apart | 35.9 [35.7, 36.2] apart | 37.3 [36.7, 38.6] apart | 39.2 [38.9, 39.3] apart | 39.4 [38.8, 39.9] apart | 38.3 [35.6, 39.0] apart | 36.0 [35.7, 36.5] apart |
| sdot prefill tiles --q4 | 2 | 39.7 [39.1, 41.6] apart | 42.5 [42.3, 42.7] apart | 42.7 [42.5, 43.5] apart | 44.0 [43.8, 44.1] apart | 44.1 [43.9, 44.3] apart | 43.3 [42.5, 43.4] apart | 42.2 [42.1, 42.3] apart |
| llama.cpp Q4_0 | 2 | 37.6 [35.8, 38.1] apart | 38.4 [38.2, 38.7] apart | 38.5 [38.4, 39.0] apart | 39.2 [38.9, 39.3] apart | 39.6 [39.3, 39.7] apart | 38.8 [37.9, 38.9] apart | 38.5 [38.3, 38.6] apart |
| llama.cpp Q8_0 | 2 | 29.7 [27.9, 30.5] apart | 32.4 [32.2, 32.6] apart | 32.5 [32.4, 33.0] apart | 32.9 [32.7, 33.0] apart | 33.2 [33.0, 33.3] apart | 32.3 [32.0, 32.5] apart | 32.0 [31.7, 32.1] apart |
| sdot prefill tiles | 4 | 43.7 [39.9, 47.1] apart | 48.9 [47.8, 49.7] apart | 50.1 [46.4, 51.9] apart | 53.3 [35.2, 53.6] apart | 53.4 [53.0, 53.5] apart | 50.2 [47.0, 50.8] apart | 49.2 [48.5, 50.0] apart |
| sdot prefill tiles --q4 | 4 | 61.9 [59.1, 65.1] apart | 67.2 [66.4, 68.3] apart | 69.2 [65.2, 69.3] apart | 70.5 [69.5, 70.9] apart | 71.2 [70.9, 71.6] apart | 67.6 [48.3, 68.2] apart | 67.4 [63.2, 67.5] apart |
| llama.cpp Q4_0 | 4 | 65.3 [63.0, 68.4] apart | 70.2 [67.7, 71.1] apart | 70.2 [67.2, 71.2] apart | 71.9 [71.2, 72.1] apart | 73.5 [72.8, 73.5] apart | 68.8 [67.9, 70.8] apart | 68.3 [67.3, 69.8] apart |
| llama.cpp Q8_0 | 4 | 48.0 [45.5, 50.7] apart | 52.5 [51.5, 53.4] apart | 54.6 [49.5, 55.4] apart | 57.2 [36.6, 57.5] apart | 57.1 [56.6, 57.3] apart | 54.5 [53.0, 54.9] apart | 52.2 [50.8, 52.7] apart |

| Prefill | Threads | job 2 | job 3 | job 4 | job 5 | job 6 | job 7 | job 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sdot prefill tiles vs llama.cpp Q8_0 | 1 | 1.04x [1.02, 1.04] apart | 1.04x [1.03, 1.06] apart | 1.04x [1.04, 1.04] apart | 1.04x [1.03, 1.12] apart | 1.04x [1.03, 1.35] apart | 1.02x [1.01, 1.02] apart | 1.03x [1.02, 1.03] apart |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 1 | 0.92x [0.91, 0.93] apart | 0.92x [0.91, 0.92] apart | 0.93x [0.92, 0.95] apart | 0.93x [0.93, 0.93] apart | 0.93x [0.93, 0.95] apart | 0.92x [0.92, 0.92] apart | 0.92x [0.92, 0.92] apart |
| sdot prefill tiles vs llama.cpp Q8_0 | 2 | 1.10x [1.09, 1.10] apart | 1.10x [1.09, 1.10] apart | 1.09x [1.09, 1.10] apart | 1.09x [1.09, 1.10] apart | 1.09x [1.09, 1.10] apart | 1.09x [1.08, 1.09] apart | 1.09x [1.08, 1.09] apart |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 2 | 0.97x [0.95, 0.97] apart | 0.97x [0.96, 0.97] apart | 0.97x [0.96, 0.97] apart | 0.97x [0.97, 0.97] apart | 0.97x [0.97, 0.97] apart | 0.97x [0.96, 0.97] apart | 0.97x [0.96, 0.97] apart |
| sdot prefill tiles vs llama.cpp Q8_0 | 4 | 1.07x [1.01, 1.26] | 1.08x [0.62, 1.09] | 1.08x [1.07, 1.14] | 1.07x [1.07, 1.08] | 1.07x [1.07, 1.08] | 1.08x [1.06, 1.08] | 1.07x [1.02, 1.47] |
| sdot prefill tiles --q4 vs llama.cpp Q4_0 | 4 | 0.94x [0.92, 0.96] | 0.95x [0.94, 0.96] | 0.95x [0.94, 0.95] | 0.95x [0.93, 0.97] | 0.95x [0.94, 0.96] | 0.95x [0.94, 0.95] | 0.95x [0.94, 0.95] |

| Prefill | Threads | job 2 | job 3 | job 4 | job 5 | job 6 | job 7 | job 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sdot prefill tiles | 1 | 79.5 [78.9, 79.7] apart | 80.2 [79.4, 80.4] apart | 80.5 [80.1, 80.7] apart | 80.5 [79.6, 80.9] apart | 80.8 [80.0, 81.3] apart | 78.7 [78.4, 79.4] apart | 79.5 [78.5, 79.7] apart |
| sdot prefill tiles --q4 | 1 | 80.4 [79.0, 81.5] apart | 80.9 [80.6, 81.7] apart | 81.5 [81.1, 81.7] apart | 82.0 [81.9, 82.2] apart | 82.2 [82.2, 82.3] apart | 81.3 [80.7, 81.5] apart | 81.0 [80.8, 81.2] apart |
| llama.cpp Q4_0 | 1 | 87.7 [86.7, 88.2] apart | 88.2 [88.0, 88.4] apart | 88.1 [85.6, 88.3] apart | 88.4 [88.4, 88.5] apart | 88.6 [87.0, 88.7] apart | 88.2 [88.0, 88.3] apart | 88.0 [87.9, 88.3] apart |
| llama.cpp Q8_0 | 1 | 76.5 [76.3, 77.1] apart | 77.5 [75.9, 77.5] apart | 77.5 [77.4, 77.9] apart | 77.6 [71.9, 78.0] apart | 77.9 [60.0, 78.0] apart | 77.6 [77.2, 77.7] apart | 77.2 [77.0, 77.4] apart |
| sdot prefill tiles | 2 | 85.9 [85.5, 86.4] apart | 86.6 [86.1, 86.7] apart | 86.6 [86.4, 86.8] apart | 86.8 [86.5, 86.9] apart | 86.9 [86.5, 87.1] apart | 86.1 [85.6, 86.5] apart | 85.9 [85.6, 86.1] apart |
| sdot prefill tiles --q4 | 2 | 90.6 [90.2, 91.0] apart | 91.0 [90.7, 91.1] apart | 91.0 [90.8, 91.3] apart | 91.5 [91.3, 91.7] apart | 91.8 [91.7, 91.9] apart | 91.3 [90.9, 91.6] apart | 91.0 [90.9, 91.1] apart |
| llama.cpp Q4_0 | 2 | 94.2 [93.7, 94.6] apart | 94.1 [93.9, 94.2] apart | 94.3 [94.2, 94.5] apart | 94.6 [94.5, 94.7] apart | 94.8 [94.6, 94.8] apart | 94.4 [94.2, 94.6] apart | 94.2 [93.6, 94.5] apart |
| llama.cpp Q8_0 | 2 | 78.3 [78.2, 78.8] apart | 79.0 [78.5, 79.3] apart | 79.1 [78.9, 79.4] apart | 79.3 [79.2, 79.3] apart | 79.4 [79.2, 79.5] apart | 79.0 [79.0, 79.2] apart | 79.1 [78.6, 79.2] apart |
| sdot prefill tiles | 4 | 166 [156, 167] apart | 169 [97.2, 169] apart | 168 [167, 169] apart | 169 [167, 169] apart | 169 [169, 170] apart | 167 [165, 168] apart | 166 [159, 167] apart |
| sdot prefill tiles --q4 | 4 | 175 [172, 176] apart | 178 [174, 178] apart | 176 [174, 178] apart | 178 [175, 178] apart | 178 [177, 179] apart | 176 [175, 177] apart | 176 [174, 176] apart |
| llama.cpp Q4_0 | 4 | 185 [183, 187] apart | 187 [186, 187] apart | 186 [186, 187] apart | 187 [185, 188] apart | 188 [187, 188] apart | 186 [185, 186] apart | 186 [185, 186] apart |
| llama.cpp Q8_0 | 4 | 155 [132, 155] apart | 156 [154, 157] apart | 155 [148, 156] apart | 157 [157, 157] apart | 157 [157, 158] apart | 155 [155, 157] apart | 155 [114, 155] apart |

Jobs: 1 = 41521222126, 2 = 41527525238, 3 = 41527529309, 4 = 41527533349, 5 = 41536359883, 6 = 41536368627, 7 = 41536372331, 8 = 41536376622.
