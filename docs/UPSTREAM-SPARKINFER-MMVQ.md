> **Filing note.** This is written to be pasted as a GitHub issue on SparkInfer. Everything under "Measured" was measured on the pinned commit `5347b27`; everything under "Root cause" is quoted from that commit's source. Line numbers are `5347b27`.

---

# `launch_mmvq_rows` refuses `M > 8` with no chunking loop, so packed decode falls back to one row at a time above 8 concurrent sequences

## Summary

On a GGUF checkpoint whose projections are Q4_K, `Qwen35Model::decode_packed()` fails for every batch wider than 8 rows. The first Q4_K projection of the batch — `attn_qkv.weight`, `n_out=8192, K=2048`, on layer 0 of a recurrent block — is handed to `kernels::launch_mmvq_rows`, which forwards to `launch_mmvq_q4k_rows`, which returns `false` for `M > 8`. Unlike the FP8 and NVFP4 branches beside it, the mmvq branch of `proj()` has no per-row fallback, so `dflash_verify_short_run` declines at layer 0 and the engine advances every one of the batch's sequences through `forward_token()` individually. Aggregate throughput at 16 and 32 concurrent sequences then lands at **0.90x of the single-sequence rate** — 452.3 and 456.1 tok/s against 503.2 at batch 1. The FP32 sibling `launch_mmvq_rows_f32`, seven lines below in the same file, chunks `M` into groups of 8 for exactly this case; `launch_gemv_rows2`, `launch_gemv_nvfp4_rows_dp4a` and `launch_gemv_nvfp4_rows_dp4a2` do the same. This one does not, which reads as an omission rather than a design choice. The 8-row limit is **not** a hardware or arithmetic constraint: `MMAX` is a template parameter that sizes only a per-thread accumulator array and one shared-memory buffer, the launcher already dispatches `MMAX` ∈ {4, 6, 8}, and the surrounding runtime is already sized for 32 rows (`kVerifyMaxRows = 32`, `kQwen35MaxPackedRows = 32`, every verify arena buffer allocated at `NA = 32`). Adding the chunking loop recovers **2.7x** of aggregate throughput on the measured checkpoint. One further launcher must be chunked in the same patch or the wider batch is silently *wrong*, not merely slow — see "The patch is not complete without the shared expert".

Model: `unsloth/Qwen3.6-35B-A3B-GGUF`, `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (22.13 GB, 40 layers, `full_attention_interval` 4 → 30 recurrent layers, hidden 2048, `linear_qkvdim` 8192, MoE 8-of-256 experts, per-expert ffn 512). Box: RTX 5090, sm_120, 170 SMs, 1792 GB/s, CUDA 13.3, driver 595.84.

## Reproduction

No patch and no instrumentation needed — the runtime prints its own cause on stderr.

```bash
MODEL=/root/models/qwen36-moe/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
BIN=.../build/runtime/qwen3_gguf_cb_bench

# argv: <model> <concurrency> <prompt_len> <max_new> <long_prefill>

# A. default packed-row cap (32). Expect the two stderr lines below, and
#    agg_tok_s at or below the single-sequence rate.
$BIN $MODEL 16 128 64 4096   2> c16-default.err
$BIN $MODEL 32 128 64 4096   2> c32-default.err

# B. identical command with the packed-row cap set to the width the GEMV accepts.
#    Nothing else changes: the engine already splits a batch wider than the cap
#    into chunks of the cap (inference_engine.cpp:458-469).
SPARKINFER_PACKED_MAX_ROWS=8 $BIN $MODEL 16 128 64 4096  2> c16-cap8.err
SPARKINFER_PACKED_MAX_ROWS=8 $BIN $MODEL 32 128 64 4096  2> c32-cap8.err

# C. single-sequence reference
$BIN $MODEL 1 128 64 4096
```

Arm A prints, once per declined step:

```
[dflash-verify] mmvq_rows refused type=12 N=16 n_out=8192 K=2048
[dflash-verify] declined at layer=0 (linear_attn=1) N=16
```

`type=12` is Q4_K (`gemm.h:196`, "Q8_0=8, Q4_K=12, Q6_K=14"). `n_out=8192, K=2048` is `blk.N.attn_qkv.weight`, the Gated-DeltaNet in-projection on every recurrent layer — `lqkv = s.linear_qkvdim` (`qwen35_prefill.cpp:201`, `= 2*linear_qdim + linear_vdim`, `qwen35.cpp:518`). Arm B prints neither line.

## Measured

`SPARKINFER_PACKED_MAX_ROWS` is the only variable between the two columns.

| | default cap (32) | `SPARKINFER_PACKED_MAX_ROWS=8` | ratio |
|---|--:|--:|--:|
| c=16 aggregate | 452.8 tok/s | **1204.4 tok/s** | **2.66x** |
| c=32 aggregate | 455.6 tok/s | **1226.8 tok/s** | **2.69x** |
| c=16 packed forwards | 127 of 2173 (5.8%) | 254 of 269 (94.4%) | |
| c=32 packed forwards | 127 of 4205 (3.0%) | 508 of 522 (97.3%) | |

The counters are *forward passes*, not tokens: at c=16 with the default cap, 2046 of 2173 forwards through the model were single-row. Isolated one-run-per-width sweep, same binary, same box, minutes apart:

| concurrency | packed forwards | max rows seen | aggregate | vs batch 1 (503.2 tok/s) |
|---|--:|--:|--:|--:|
| 1 | — | 1 | 503.2 tok/s | 1.00x |
| 4 | 129 / 138 (93%) | 5 | 907.5 tok/s | 1.80x |
| 8 | 133 / 151 (88%) | 9 | 2456.0 tok/s | 4.88x |
| 16 | 127 / 2173 (6%) | 17 | **452.3 tok/s** | **0.899x** |
| 32 | 127 / 4205 (3%) | 32 | **456.1 tok/s** | **0.906x** |

**The 0.90x is the cleanest confirmation of the diagnosis.** "Decode 16 sequences one row at a time" predicts an aggregate equal to the single-sequence rate minus scheduling overhead. 452.3 / 503.2 = 0.899 and 456.1 / 503.2 = 0.906. That is what the code path says should happen, to within 10%.

Two caveats stated up front, because they change how much weight each number carries:

- `agg_tok_s` is `decode_tokens / wall_s` over the whole run including each stream's prefill (`qwen3_gguf_cb_bench.cpp:218-232`), so figures at different concurrencies are not perfectly comparable. **The controlled claim is the cap A/B (2.66x / 2.69x): same concurrency, same command, one env var.** The often-quoted "5.4x cliff" (2456 at c=8 vs 452.3 at c=16) compares two different runs and is the weaker of the two.
- The 2456 tok/s c=8 point implies a 3.26 ms 8-row step, whereas the c=32 capped run implies 6.52 ms per 8-row chunk (26.08 ms / 4 chunks). That factor of two is not explained by the GEMV and is not something this report diagnoses; it is flagged in "Open questions" below.

## Root cause

**1. The kernel launcher refuses.** `kernels/csrc/cuda/gemm/gemv.cu:3816-3843`, `launch_mmvq_q4k_rows`:

```c
    if (M < 1 || M > 8 || N < 1) return false;                     // :3824
    if (K != 2048 && K != 4096 && K != 5120 && K != 6144) return false;
    ...
    #define SI_Q4K_ROWS_DISPATCH(KB) \
        do { \
            if (M <= 6) si_mmvq_q4k_rows_exact_kernel<__nv_bfloat16, KB, 6, SI_Q4K_OROWS, 1><<<...>>>(q, w, out, M, N); \
            else        si_mmvq_q4k_rows_exact_kernel<__nv_bfloat16, KB, 8, SI_Q4K_OROWS, 1><<<...>>>(q, w, out, M, N); \
        } while (0)
```

The identical guard appears in the two sibling types: `launch_mmvq_q6k_rows` at `:3846` and `launch_mmvq_q80_rows` at `:3861`, both `if (M < 1 || M > 8 || N < 1 || ...) return false;`.

**2. The dispatcher does not chunk; its `_f32` twin does.** `gemv.cu:3880-3900`, verbatim and adjacent:

```c
bool launch_mmvq_rows(int qtype, const void* q81, const void* W, void* y,
                      int M, int N, int K, cudaStream_t stream) {
    if (qtype == 12) return launch_mmvq_q4k_rows(q81, W, y, M, N, K, stream);
    if (qtype == 14) return launch_mmvq_q6k_rows(q81, W, y, M, N, K, stream);
    if (qtype == 8)  return launch_mmvq_q80_rows(q81, W, y, M, N, K, stream);
    return false;
}
bool launch_mmvq_rows_f32(int qtype, const void* q81, const void* W, float* y,
                          int M, int N, int K, cudaStream_t stream) {
    if (M < 1 || N < 1) return false;
    if (M > 8) {   // chunk: see launch_gemv_nvfp4_rows_dp4a. q81 rows are (K>>5) blocks apart.
        for (int r0 = 0; r0 < M; r0 += 8) {
            const int m = (M - r0) < 8 ? (M - r0) : 8;
            if (!launch_mmvq_rows_f32(qtype,
                                      reinterpret_cast<const si_block_q8_1*>(q81)
                                          + (size_t)r0 * (size_t)(K >> 5),
                                      W, y + (size_t)r0 * N, m, N, K, stream)) return false;
        }
        return true;
    }
```

Three other multi-row launchers in the same file carry the same loop — `launch_gemv_rows2` at `:2885`, `launch_gemv_nvfp4_rows_dp4a2` at `:3335`, `launch_gemv_nvfp4_rows_dp4a` at `:3436`. (For accuracy: `launch_gemv_nvfp4_rows` at `:3513` does **not** chunk — `if (M < 2 || M > 8) return false;` at `:3522` — nor does `launch_gemv_fp8_rows` at `:2958`, nor `launch_gemv_rows_t` at `:2861`. Those three are survivable because their callers loop per row; see point 4.)

**3. The caller turns a declined launch into a declined forward.** `runtime/src/models/qwen35_prefill.cpp:3223-3232`, inside `proj()`:

```c
        quant_rows(in, k);
        if (!kernels::launch_mmvq_rows(type, q81, w, out, N, no, k, st)) {
            fprintf(stderr, "[dflash-verify] mmvq_rows refused type=%d N=%d n_out=%d K=%d\n",
                    type, N, no, k);
            return false;
        }
        return true;
```

**4. The two branches immediately above it in the same lambda do have a fallback.** `:3189-3192` (FP8) and `:3213-3216` (NVFP4):

```c
            if (kernels::launch_gemv_fp8_rows(in, w, out, N, no, k, st)) return true;
            for (int r = 0; r < N; ++r)
                kernels::launch_gemv_fp8(in + (size_t)r * k, w, out + (size_t)r * no, no, k, st);
            return true;
```

So an NVFP4 or FP8 checkpoint at `M > 8` gets slow (full weight traffic per row) but keeps working; a Q4_K/Q6_K/Q8_0 checkpoint declines. That asymmetry is why this shows up as a total loss of batching on GGUF and not on the ModelOpt NVFP4 export.

**5. From there the whole batch is abandoned.** `qwen35_prefill.cpp:3506` is the call that fails first:

```c
                supported = proj(xn, w.wqkv, w.wqkv_type, rq, lqkv, H);
```

`supported == false` breaks the layer loop, `:4029` prints `[dflash-verify] declined at layer=%d (linear_attn=%d) N=%d start=%d`, `abandon_capture()` runs and the function returns `-1` (`:4034-4035`). `Qwen35Model::decode_packed` returns `consumed == n`, i.e. `false` (`qwen35.cpp:3485-3487`). `ContinuousBatchEngine` then advances the chunk row by row (`inference_engine.cpp:470-478`):

```c
        if (m >= 2)
            ok = model_->decode_packed(toks.data(), pos.data(), seqs.data(), (int)m, out.data());
        if (!ok) {
            for (size_t i = 0; i < m; i++) {
                Job* j = live[off + i];
                model_->activate_session(j->seq_id);
                out[i] = model_->forward_token(...);
            }
        }
```

## Is 8 a real kernel constraint? No — it is a dispatcher tier

This is the question that decides whether the fix is small or large, so here is the whole of what `MMAX` does. `gemv.cu:2006-2059`:

```c
template <typename OutT, int NSUPER, int MMAX, int OROWS, int GRP>
__global__ void si_mmvq_q4k_rows_exact_kernel(const si_block_q8_1* __restrict__ q,
                                              const unsigned char* __restrict__ W,
                                              OutT* __restrict__ y, int M, int N) {
    constexpr int NW = 4, WS = 32, vdr = 2, qi = 32;
    ...
    __shared__ float partial[GRP][OROWS][MMAX][NW - 1][WS];        // :2018
    ...
        float tmp[OROWS][MMAX];                                     // :2024
```

`MMAX` appears in exactly those two declarations and in the `#pragma unroll` bounds that walk them. It is not a tiling width, not a warp-mapping parameter, not an MMA shape. The per-`K` walk (`:2028-2032`) and the two-stage four-warp fold (`:2033-2056`) are independent of it; `si_vec_dot_q4_K_tiled` (`:1674-1677`) takes the accumulators by reference as `float (&acc)[OROWS][R]` and its per-row temporaries (`u0[2], u1[2], dot2[2], d8[2]`) live inside the `for r` body, while the weight decode `si_q4k_packet pk[OROWS]` (`:1682`) does not depend on `R` at all. So the only quantity that scales with row count is `2 x MMAX` floats per thread plus `768 x MMAX` bytes of shared memory per CTA (`GRP=1`, `OROWS=2` — `SI_Q4K_OROWS` is `2` at `:1990`).

Concretely, at `GRP=1, OROWS=2, NW=4, WS=32`:

| `MMAX` | smem/CTA | accumulators/thread | measured registers (bf16, `NSUPER=8`) |
|---|--:|--:|--:|
| 6 | 4608 B | 12 | **54**, `STACK:0 LOCAL:0` |
| 8 | 6144 B | 16 | **60**, `STACK:0 LOCAL:0` |
| 16 | 12288 B | 32 | ~84 (extrapolated at +3 reg/row) |
| 32 | 24576 B | 64 | ~130 (extrapolated) |

Registers are from `cuobjdump -res-usage` on the shipped object of this commit (`build/kernels/csrc/cuda/gemm/CMakeFiles/si_gemm.dir/gemv.cu.o`), instantiations `<__nv_bfloat16, 8, 6, 2, 1>` and `<__nv_bfloat16, 8, 8, 2, 1>`. Going from 6 rows to 8 costs 6 registers with no spill; nothing here is near a wall at 16. The 48 KB static shared-memory limit is not reached until `MMAX = 64`.

The launcher itself already proves the parameter is free: it dispatches `MMAX = 6` or `8` by row count (`:3833-3836`), and `launch_mmvq_rows_f32` additionally dispatches `MMAX = 4` and `OROWS = 4` for the LM head (`:3906-3915`). The explicit instantiation list at `:2062-2087` carries `<*, {8,16,20,24}, {4,6,8}, {2,4}, 1>`.

**Conclusion: the fix is "add the loop", not "write a wider kernel".** A wider kernel is a separate, optional improvement, quantified below.

One honest counterweight, because it is in your own source and a reader will find it. `launch_gemv_nvfp4_rows_dp4a` (`:3429-3435`) records a measured attempt at exactly the wide instantiation:

```c
    // Wider than the instantiated row tiers: serve it as chunks of 8 rather than instantiating
    // R=9..16. Measured on RTX 5090 at the real decode shapes, a 16-row instantiation costs
    // t(16) = 1.002 * 2*t(8) with 495 spill stores -- `uint4 xg[GPT][R]` alone is 128 registers
    // at R=16 -- so a wider template buys nothing over re-reading the weights, and chunking keeps
    // every row on the tuned R<=8 kernel.
```

That result does not transfer to this kernel. `gemv_nvfp4_rows_dp4a_kernel` stages a `uint4 xg[GPT][R]` activation block in registers, so its register cost scales with `R x GPT`; `si_mmvq_q4k_rows_exact_kernel` re-reads the activation from global per `r` and its only `R`-scaled state is `tmp[OROWS][MMAX]`. Measured: 54 → 60 registers for 6 → 8 rows, no spills. Whether a 16- or 32-row Q4_K instantiation actually pays is untested and would need one benchmark; the chunking loop does not depend on the answer.

## Does it affect quantization types other than Q4_K?

Yes — every type `launch_mmvq_rows` dispatches, and one more path beside it:

| type | launcher | guard | caller fallback |
|---|---|---|---|
| 12 Q4_K | `launch_mmvq_q4k_rows` | `gemv.cu:3824` `M > 8` | none → declines |
| 14 Q6_K | `launch_mmvq_q6k_rows` | `gemv.cu:3846` `M > 8` | none → declines |
| 8 Q8_0 | `launch_mmvq_q80_rows` | `gemv.cu:3861` `M > 8` | none → declines |
| 0 bf16 | `launch_gemv_rows` → `launch_gemv_rows_t` | `gemv.cu:2861` `M > 8` | none (`qwen35_prefill.cpp:3166-3168`, `:3262`) → declines |
| 109 NVFP4 | `launch_gemv_nvfp4_rows` | `gemv.cu:3522` `M > 8` | per-row loop `:3214-3216` → correct, slow |
| 108 FP8 | `launch_gemv_fp8_rows` | `gemv.cu:2958` `M > 8` | per-row loop `:3190-3192` → correct, slow |

The single chunk in `launch_mmvq_rows` covers the first three at once. The bf16 `type == 0` path is a separate one-line omission of the same shape (its two-matrix neighbour `launch_gemv_rows2` already chunks at `:2885`) and is included in the patch below; the checkpoint measured here does not exercise it.

## The patch is not complete without the shared expert

This is the part that matters most for correctness, and it is why the one-hunk patch should not be merged alone.

`kernels/csrc/cuda/moe/expert_ffn_q4k.cu:2326-2331`:

```c
void launch_shared_expert_q8_mmvq_rows(
    const void* input_q8, const void* gate_q, const void* up_q, const void* down_q,
    const float* dw, void* output, float* h_scratch, void* h_q8_buf,
    int hidden, int ffn, int rows, cudaStream_t stream) {
    if (!input_q8 || !gate_q || !up_q || !down_q || !dw || !output || !h_scratch ||
        !h_q8_buf || hidden != 2048 || ffn != 512 || rows < 1 || rows > 8) return;
```

It returns `void`. At `rows > 8` it launches nothing, reports nothing, and its caller (`qwen35_prefill.cpp:4007-4009`) cannot tell. The `output` buffer is `shared`, an arena allocation (`qwen35_prefill.cpp:2965`) whose `Arena::alloc` reuses device memory without zeroing it (`qwen35_prefill.cpp:57-74`), and `shared` is summed into the residual stream two statements later by `launch_add_rmsnorm3_q8_rows` (`:4015-4016`). So today the refusal at layer 0 is what *protects* this: the batch never gets that far. Chunk `launch_mmvq_rows` alone and a 16-row packed step would proceed past layer 0 and add a stale buffer as the shared-expert contribution on every MoE layer — wrong output, no diagnostic. Both hunks, or neither.

(The routed-expert path is fine: `launch_down_q4k_mmvq_splitk_rows` also declines above 8 rows, `expert_ffn_q4k.cu:1732`, but its caller documents and takes a per-token fallback grid — correct, just slower.)

## The minimal patch

Bit-exactness is preserved by construction, the same argument `launch_mmvq_rows_f32` already relies on: `MMAX` bounds array sizes only, the per-row `kbx` accumulation order and the four-warp fold are identical at every instantiated width, and each row's result depends on no other row. A chunked call is bit-identical to one wide call and to `M` separate one-row calls — which is what DSpark's losslessness gate requires. No allocation changes: `kVerifyMaxRows = 32` (`qwen35_prefill.cpp:89`), every verify arena buffer is allocated at `NA = kVerifyMaxRows` (`:2833`), `q81` included (`:3008`), and `kQwen35MaxPackedRows = 32` (`qwen35.h:14`).

```diff
--- a/kernels/csrc/cuda/gemm/gemv.cu
+++ b/kernels/csrc/cuda/gemm/gemv.cu
@@ -3880,6 +3880,21 @@
 bool launch_mmvq_rows(int qtype, const void* q81, const void* W, void* y,
                       int M, int N, int K, cudaStream_t stream) {
+    if (M < 1 || N < 1) return false;
+    // chunk: same construction as launch_mmvq_rows_f32 below and launch_gemv_nvfp4_rows_dp4a.
+    // q81 rows are (K>>5) blocks apart (llama_q8_1_bytes, :3721) and y is [M,N] row-major, so a
+    // chunk is a pointer offset. Bit-identical to a single wide call: MMAX bounds tmp[]/partial[]
+    // only, and every row's accumulation order and four-warp fold are independent of M.
+    if (M > 8) {
+        for (int r0 = 0; r0 < M; r0 += 8) {
+            const int m = (M - r0) < 8 ? (M - r0) : 8;
+            if (!launch_mmvq_rows(qtype,
+                                  reinterpret_cast<const si_block_q8_1*>(q81)
+                                      + (size_t)r0 * (size_t)(K >> 5),
+                                  W, reinterpret_cast<__nv_bfloat16*>(y) + (size_t)r0 * N,
+                                  m, N, K, stream)) return false;
+        }
+        return true;
+    }
     if (qtype == 12) return launch_mmvq_q4k_rows(q81, W, y, M, N, K, stream);
     if (qtype == 14) return launch_mmvq_q6k_rows(q81, W, y, M, N, K, stream);
     if (qtype == 8)  return launch_mmvq_q80_rows(q81, W, y, M, N, K, stream);
     return false;
 }
```

```diff
--- a/kernels/csrc/cuda/moe/expert_ffn_q4k.cu
+++ b/kernels/csrc/cuda/moe/expert_ffn_q4k.cu
@@ -2326,7 +2326,22 @@
 void launch_shared_expert_q8_mmvq_rows(
     const void* input_q8, const void* gate_q, const void* up_q, const void* down_q,
     const float* dw, void* output, float* h_scratch, void* h_q8_buf,
     int hidden, int ffn, int rows, cudaStream_t stream) {
     if (!input_q8 || !gate_q || !up_q || !down_q || !dw || !output || !h_scratch ||
-        !h_q8_buf || hidden != 2048 || ffn != 512 || rows < 1 || rows > 8) return;
+        !h_q8_buf || hidden != 2048 || ffn != 512 || rows < 1) return;
+    // Chunk rather than no-op. This returns void, so a silent decline above 8 rows leaves
+    // `output` holding whatever the arena last put there and the caller adds it to the
+    // residual stream. Chunks run in order on one stream, so h_q8_buf is safe to reuse.
+    if (rows > 8) {
+        for (int r0 = 0; r0 < rows; r0 += 8) {
+            const int m = (rows - r0) < 8 ? (rows - r0) : 8;
+            launch_shared_expert_q8_mmvq_rows(
+                static_cast<const unsigned char*>(input_q8)
+                    + (size_t)r0 * llama_q8_1_bytes(hidden),
+                gate_q, up_q, down_q, dw + r0,
+                static_cast<__nv_bfloat16*>(output) + (size_t)r0 * hidden,
+                h_scratch + (size_t)r0 * ffn, h_q8_buf, hidden, ffn, m, stream);
+        }
+        return;
+    }
```

Optional third hunk, the same omission on the bf16 path (not exercised by this checkpoint; its two-matrix neighbour `launch_gemv_rows2` already has the loop at `:2885`):

```diff
--- a/kernels/csrc/cuda/gemm/gemv.cu
+++ b/kernels/csrc/cuda/gemm/gemv.cu
@@ -2879,6 +2879,15 @@
 bool launch_gemv_rows(const void* x, const void* W, void* y,
                       int M, int N, int K, cudaStream_t stream) {
+    if (M > 8 && N >= 1 && !(K & 7)) {            // chunk: see launch_gemv_rows2 below
+        for (int r0 = 0; r0 < M; r0 += 8) {
+            const int m = (M - r0) < 8 ? (M - r0) : 8;
+            if (!launch_gemv_rows(reinterpret_cast<const __nv_bfloat16*>(x) + (size_t)r0 * K, W,
+                                  reinterpret_cast<__nv_bfloat16*>(y) + (size_t)r0 * N,
+                                  m, N, K, stream)) return false;
+        }
+        return true;
+    }
     return launch_gemv_rows_t<__nv_bfloat16, 8>(x, W,
         reinterpret_cast<__nv_bfloat16*>(y), M, N, K, stream);
 }
```

A cheap hardening that would have made this a one-line diagnosis: give `launch_shared_expert_q8_mmvq_rows` a `bool` return, and give the mmvq branch of `proj()` the same per-row fallback loop the FP8 and NVFP4 branches already have, so a refused shape costs throughput instead of the whole batch.

## What the patch does not fix

1. **It does not recover the whole gap, because chunking re-reads the weights once per chunk.** Four chunks of 8 at 32 sequences is four passes over the row-shared weights where one 32-row kernel would be one. Measured: the cap recovers 2.66-2.69x, against a 5.43x gap between the c=8 and c=16 runs.
2. **A genuinely wide-row kernel is worth much less than the loop on *this* checkpoint, and that is a property of MoE.** Taking measured step times against 1792 GB/s and fitting `bytes(rows) = D + rows x P` from two points (batch-1 step 1.987 ms → 3.56 GB; 8-row step) gives, at 32 rows in one pass instead of four chunks of 8:
   - fitted from the c=32-capped 8-row chunk (6.52 ms): `D = 2.40 GB`, `P = 1.16 GB/row` → 26.08 ms → 22.06 ms, i.e. 1227 → **1451 tok/s (+18%)**;
   - fitted from the c=8 run's 8-row step (3.26 ms): `D = 3.24 GB`, `P = 0.33 GB/row` → 13.03 ms → 7.61 ms, i.e. 2456 → **4205 tok/s (+71%)**.
   
   So somewhere between **+18% and +71%** on top of what the loop already delivers. Both are inferences from a two-point linear fit on bandwidth-implied bytes, and MoE decode steps are not bandwidth-saturated, so treat them as a bracket rather than a prediction. The reason the range sits so far below 4x is that 8-of-256 routing makes most of a step's traffic per-row: the shared term `D` is only 2.4-3.2 GB of an 11.7 GB 8-row step. **On a dense Q4_K checkpoint, where `D` is nearly the whole step, the same wide kernel would be worth substantially more.** Order of work: land the loop, then decide about the kernel with a measurement.
3. **The other `M > 8` refusals stay.** `launch_gemv_nvfp4_rows` (`:3522`) and `launch_gemv_fp8_rows` (`:2958`) still decline and their callers still fall back to a per-row loop that re-reads the entire weight matrix once per row (`qwen35_prefill.cpp:3190-3192`, `:3214-3216`). Correct, but the full per-row traffic — chunking those two would be a strict improvement over the loop they have. The routed-expert down projection (`expert_ffn_q4k.cu:1732`) is in the same category.
4. **It does not change the graph-tier or arena story** — those are already 32-wide and need no work.
5. **The 8-row chunk cost itself is unexplained.** 3.26 ms in the c=8 run against 6.52 ms in the c=32-capped run for the same 8-row work. Whatever accounts for that factor of two (per-row engine callbacks, scheduling, KV/state growth, or a measurement artefact of `agg_tok_s` including prefill) is untouched by this patch and is probably the next thing worth profiling after it lands.

## Open questions for maintainers

- Was the `M > 8` bound on `launch_mmvq_q4k_rows` chosen for the speculative-verify block width (DSpark `block_size` 7 plus one, per the `kVerifyMaxRows` comment at `qwen35_prefill.cpp:84`), and simply not revisited when continuous-batch decode began reusing the same forward at up to 32 rows?
- Is there a reason `proj()`'s mmvq branch has no per-row fallback where its FP8 and NVFP4 neighbours do?
- Has a 16- or 32-row `si_mmvq_q4k_rows_exact_kernel` instantiation ever been benchmarked? The NVFP4 note at `:3429-3435` measured one for a kernel with very different register scaling; this one's only `R`-scaled state is `tmp[OROWS][MMAX]`, and 6 → 8 rows costs 6 registers with no spill.

---

*Measured on SparkInfer `5347b27` against `unsloth/Qwen3.6-35B-A3B-GGUF` on an RTX 5090 (CUDA 13.3, driver 595.84). Raw counters and per-arm data available on request. No SparkInfer source was modified to produce any number in this report.*