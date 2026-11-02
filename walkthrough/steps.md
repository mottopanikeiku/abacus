# The ladder, step by step

One section per engine commit, oldest first. `walkthrough/build.py` turns this
file, the git history and the saved benchmark results into the walkthrough
page; the comment under each heading picks the lines of that commit's diff to
show.

## scalar fp32 engine
<!-- show src/model.c "static void matmul" -->
The starting point: the whole model in float32, one token at a time, on one
thread. Every matrix-vector product is a plain double loop. It runs Qwen3's
layers as transformers does (RMSNorm, grouped-query attention with a norm on
each query and key head, rotary embeddings, a SwiGLU MLP) and its logits match
transformers to within 0.002, which is the check every later step has to pass.

## threads
<!-- show src/model.c "static void matmul_rows" -->
Nearly all the time goes into matrix-vector products, and the rows of a
product don't depend on each other. A small thread pool splits the rows of
every product, and the attention heads, across cores.

## neon fp32 dot
<!-- show src/model.c "vfmaq_f32(s0" -->
Each row is a dot product, and with a single running sum every multiply-add
waits for the one before it. Four 4-wide NEON accumulators keep sixteen sums
in flight and do four multiply-adds per instruction.

## bf16 weights
<!-- show src/model.c "static float dot_bf16" -->
Qwen3 ships in bfloat16, and the engine widened it to float32 at load, so every
token streamed twice the bytes it needed. Now the weights stay bf16 in the
mapped file and are widened in registers: a bf16 is the top half of a float32,
so widening is a 16-bit shift.

## int8 weights, sdot
<!-- show src/matmul.c "vdotq_s32" -->
Decoding reads every weight once per token, so bytes per weight set the speed.
Weights become int8 in blocks of 32 with one scale per block (the numbers of
llama.cpp's Q8_0), activations are quantized the same way before each product,
and SDOT multiplies 16 byte pairs per instruction into int32 sums. From here
the check asks for the same next token and a small KL divergence instead of
matching logits.

## spin-wait pool
<!-- show src/pool.c "== seen) cpu_relax()" -->
A forward pass hands the pool a few hundred small jobs per token, and waking a
sleeping thread through a condition variable costs more than many of the jobs.
Idle workers now spin on an atomic counter instead of sleeping.

## batched prefill
<!-- show src/model.c "forward_batch" -->
The prompt still went through the model one token at a time, reading every
weight once per token. Now up to 128 prompt tokens go through each layer
together, and each weight read from memory is used for all of them.

## i8mm prefill
<!-- show src/matmul.c "A 4x4 tile is four of them" -->
SMMLA, from the i8mm extension, multiplies a 2x8 int8 matrix by the transpose
of another: two rows times two tokens in one instruction, twice the work of an
SDOT. The batched product runs in tiles of 4 rows by 4 tokens, four SMMLAs per
step.

## 4-row interleave
<!-- show src/matmul.c "(r % 4) * 8" -->
Weights are reordered at load so that each block of 32 columns holds 8 bytes of
row 0, then 8 of rows 1, 2 and 3, and so on. Decoding streams one array instead
of four rows at once, and SMMLA finds the row pairs it multiplies already side
by side.

## fused qkv, gate/up
<!-- show src/model.c "wqkv" -->
q, k and v are three products of the same input, and so are gate and up. They
sit back to back in the file, so each set loads as one taller matrix and runs as
one product: one quantization of the input and one trip through the pool
instead of three or two.

## parallel norms, swiglu
<!-- show src/model.c "for_each_token" -->
With fast products, the work still left on one thread shows: the norms, rotary
embeddings, cache writes and SwiGLU. They now run on the pool as well, split by
token, or by element when there is only one token.

## rope table
<!-- show src/model.c "rope_cos" -->
Rotary embeddings called cosf and sinf for every pair of every head of every
token. A table of both, filled once at load, replaces the calls.

## 8x4 i8mm tiles
<!-- show src/matmul.c "acc[4][2]" -->
The SMMLA kernel grows to 8 rows by 4 tokens, so each input register loaded
feeds four SMMLAs instead of two, and the int32 sums for a whole block stay in
registers.

## avx2 kernels
<!-- show src/x86.c "_mm256_maddubs_epi16" -->
x86 had only the plain C fallback. AVX2 has no int8 by int8 dot product, so
the kernel uses the usual trick: VPMADDUBSW multiplies unsigned bytes by signed
ones, so the weights hand their signs to the inputs, |w| times (x times the
sign of w), and VPMADDWD adds the int16 pairs into int32. The kernels now sit
in a table that is filled at startup from what the CPU reports.

## vnni kernels
<!-- show src/x86.c "_mm256_dpbusd_avx_epi32" -->
VNNI's VPDPBUSD does the multiply, both additions and the accumulation in one
instruction. AVX-VNNI (Alder Lake, Zen 5) and AVX-512 VNNI (Ice Lake, Zen 4)
encode it differently, so the x86 kernels become one body compiled three
times, each with its own dot product inlined. CI runs every version under
Intel's CPU emulator.

## runtime i8mm, macos
<!-- show src/neon.c "getauxval" -->
The NEON kernels were compiled in only when the compiler targeted dotprod and
i8mm, so each binary was tied to the CPU it was built for: built on a Graviton4
it needs i8mm, which an M1 or a Graviton2 lacks, and built for plain arm64 it
never uses SDOT. Each kernel now switches its extension on for itself, and the
engine asks the OS at startup which ones the CPU has: getauxval on Linux,
sysctl on macOS.

## 4-bit weights
<!-- show src/neon.c "vshlq_n_s8" -->
With --q4, layer weights take 4 bits each in blocks of 32 with one scale (the
numbers of llama.cpp's Q4_0), two to a byte. They are packed so that one AND or
one shift turns a register of them into int8s sixteen times too large; the
SDOT, SMMLA and AVX2 kernels run on those unchanged and divide by 16 at the
end. Decoding streams 4.5 bits per weight instead of 8.5.

## parallel load
<!-- show src/matmul.c "pool_run(quantize_groups" -->
Weights are quantized as the model loads, and that ran on one thread: fine for
0.6B, slow for 4B. The pool now starts before loading and each thread
quantizes its share of the row groups.

## hadamard rotation
<!-- show src/matmul.c "static void rotate(float *x)" -->
Qwen3-1.7B failed the int8 check with a mean KL of 0.026 against the limit of
0.02, so its hidden states after every layer went next to those of
transformers. The error grew just as it did in a simulation that quantizes the
inputs of every product as well as the weights, and well beyond what
quantizing the weights alone costs, because a few inputs are thousands of
times larger than the rest of their block of 32 and the block's scale rounds
the other 31 towards zero. Every block of 32 weights and of inputs now goes
through the same 32x32 Hadamard matrix before it is quantized. The matrix is
orthogonal, so each block's dot product stays the same while a large value
gets spread over the whole block, and the kernels don't change at all. Mean
KL on Qwen3-1.7B drops to 0.005. The 4-bit weights stay as they were, because
in the same simulation rotating them made the model less accurate.
