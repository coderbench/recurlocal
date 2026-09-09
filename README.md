# RecurLocal

**RecurLocal is a feasibility-first CUDA library for software-directed locality of mutable recurrent neural state in hybrid LLM inference.**

> Keep recurrent AI state close to compute.

RecurLocal is intentionally **not** another inference engine, KV-cache manager, or model quantizer. Its first technical question is narrower:

> Can explicit L2 residency hints plus layer-ahead state prefetch reduce recurrent-state HBM traffic enough to improve real hybrid-LLM decode throughput?

The project starts with a small, measurable primitive and is designed to integrate with runtimes such as SparkInfer.

## Why this exists

Hybrid LLMs increasingly combine full attention with recurrent / linear-attention layers. Qwen3.8-27B, for example, has 64 language layers with a repeating pattern of three linear-attention layers followed by one full-attention layer. Its recurrent state uses FP32 and has 48 value heads of dimension 128×128, which is 3 MiB of matrix state per recurrent layer.

That state is mutable and repeatedly read/written during decode. RecurLocal experiments with the CUDA memory hierarchy rather than changing model math:

- reserve a bounded persisting-L2 set-aside when supported;
- mark hot recurrent-state windows as persisting;
- pre-touch the next recurrent layer's state asynchronously;
- rotate the hot window according to known layer execution order;
- measure whether this improves **end-to-end** decode, not just a microbenchmark.

CUDA's access-policy windows are hints, not placement guarantees. RecurLocal therefore treats every policy as an experimentally measured optimization, never as an assumed win.

## Non-goals

RecurLocal v0 does **not**:

- implement Gated DeltaNet / KDA / Mamba math;
- replace SparkInfer, vLLM, or SGLang;
- page recurrent state to CPU/NVMe;
- quantize recurrent state;
- change model outputs;
- claim any speedup before hardware measurements exist.

## Architecture

```text
Inference runtime
      |
      | recurrent layer N is about to run
      v
+---------------------------+
|         RecurLocal        |
|                           |
| locality planner          |
| persisting-L2 window      |
| next-layer pre-touch      |
| cache-window rotation     |
+-------------+-------------+
              |
              v
      CUDA memory hierarchy
        L2 <----> HBM
              |
              v
        recurrent kernel
```

## First real result

One RTX 5090, CUDA 13.3. **Qwen3.8-27B** (NVFP4, 64 layers, 48 of them recurrent) on a
**pinned SparkInfer commit** (`5347b27c`), through the adapter in
[`integrations/sparkinfer/`](integrations/sparkinfer/). Control and candidate are the same
binary — the hook is inert unless `RECURLOCAL` names a mode — run interleaved on the same box.
Control: **96.08 tok/s** at ctx 128, batch 1. Noise floor **0.023%**.

| mode | real end-to-end decode |
|---|--:|
| `baseline` — hook on, no policy | -0.01% |
| `persist` | **+0.13%** |
| `prefetch` | -1.29% |
| `combined` | -1.17% |

Output is token-exact against the unhooked runtime under greedy replay.

**All of the persist gain is an artefact of how the window is delivered, not of the policy.**
Production decode is a captured CUDA graph, and a stream access-policy window is host-side
state a graph never records:

| window delivery | gain |
|---|--:|
| `stream` (default; hand the window back for the runtime to attach) | **-0.019%** |
| `capture_node` (set the attribute on the node the capture just recorded) | **+0.129%** |

Under the safe path the policy never reaches the replayed graph, so there is nothing to
measure. `capture_node` reaches it by mutating a graph mid-capture, which CUDA does not
document as supported — a documentation gap rather than an observed defect: it has not failed
in any run taken, including twelve isolated runs at 32 sequences. It is opt-in, counted, and
self-disabling after a first detected invalidation.

(An earlier version of this README blamed a 32-sequence throughput collapse on that mutation.
**That was wrong**: the arms that collapsed were `baseline` and `prefetch`, which never arm the
mechanism at all. The collapse is a runtime fallback of unknown cause — see
[`docs/OPTIMIZATION-SURFACES.md`](docs/OPTIMIZATION-SURFACES.md).)

The settled part is the boundary: **a locality library can compute a persisting window, but
under graph decode it cannot deliver one without the runtime attaching it at its own launch
site.** Pre-touch has no such problem — its kernels and events are recorded into the graph
like any other work.

### The scored result

The candidate is `prefetch` with `token_end` joins and the `ptx_l2` walk — the best
configuration that applies a real policy without mutating a graph mid-capture. 3 interleaved
pairs, batch 1 at three contexts plus concurrency 4 and 16, token-exact output. The pre-touch
really ran (188 launches at batch 1) and the concurrency arms really went through
`decode_packed` (129/138 tokens packed at c=4):

```
$ python3 eval/decide.py --real results/rtx5090-real.json
verdict: reject   weighted gain -0.488%   impact none   significant false
  batch1         +0.016%  (w=0.40)
  concurrency16  -1.311%  (w=0.20)
  concurrency4   -0.667%  (w=0.20)
  unresolved: ['batch1/ctx16384']
```

It costs more than it saves, and the cost grows with concurrency exactly as the axis sweeps
predicted. An earlier scored run reported +0.053% for `persist` with safe window delivery;
that was a **null candidate** which applied no policy at all — 192 windows computed, none
attached — and `eval/real_eval.py` now refuses such a run by name.

**That is a rejection under the project's own gate**, and the reason is arithmetic rather
than implementation:

```
recurrent state per token   48 x (3 MiB fp32 + 60 KiB bf16) x 2  =  294 MiB
decode step                 10.41 ms x 1792 GB/s                 =   18.6 GB
recurrent share                                                      1.65%
```

Qwen3.8-27B is a **dense** hybrid: every weight is read every token, so at batch 1 the
recurrent state is 1.65% of the memory traffic. Making it *free* would be worth 1.65% — below
the 2% floor the go/no-go table rejects at, before any policy is chosen.
`eval/traffic_budget.py` computes this from the pinned geometry, and it is worth running
before optimizing for any new model or concurrency.

The one thing that did move is not locality at all. Production decode is a captured CUDA graph
replayed per token, so every fork and join the pre-touch needs is a permanent graph node:

| pre-touch ordering | gain |
|---|--:|
| join per layer (v0.1 shape) | -1.29% |
| one join per token | **-0.09%** |

1.20 points of the 1.29 were the ordering, not the memory. The synthetic benchmark cannot see
this, because it does not capture a graph — and that is one of four axes on which it has now
disagreed with the real model.

### Concurrency is where the room is, and it is still not captured

Weights are read once per decode step whatever the batch; recurrent state once per sequence.
So the share of traffic RecurLocal can address grows with concurrency — and every policy
tracks the wrong way:

| | ceiling | control | `persist` | `prefetch` |
|---|--:|--:|--:|--:|
| batch 1 | 1.65% | 96.1 tok/s | **+0.13%** | -1.29% |
| concurrency 4 | 2.88% | 329 tok/s | +0.06% | -2.19% |
| concurrency 16 | 6.73% | 769 tok/s | -0.21% | **-5.88%** |
| concurrency 32 | — | 1262 tok/s | -1.13% | *unresolved* |

By 16 sequences there is 6.7% genuinely on the table and the library captures none of it. That
is the open problem this work hands over, and it is much better posed than the one it started
from: the hot-set accounting is now correct, the concurrent decode path is instrumented and
proven to be the one measured, the ceiling at any batch size is computable, and the mechanism
meant to capture it demonstrably does not.
[`docs/OPTIMIZATION-SURFACES.md`](docs/OPTIMIZATION-SURFACES.md) has the full matrix.

## First measured result (synthetic)

One RTX 5090, CUDA 13.3, synthetic recurrent-state benchmark — 48 layers x 3 MiB, 32 timed
tokens after 4 warm-up, 15 interleaved repeats. Full data in
[`results/rtx5090-synthetic.json`](results/rtx5090-synthetic.json).

| mode | gain vs baseline | spread |
|---|--:|--:|
| persist | +10.1% | 0.9% |
| prefetch | **+15.9%** | 4.7% |
| combined | +10.4% | 4.6% |

Tuning the prefetch distance, which v0.1 had hard-coded to one layer, reaches **+21.0% at
distance 6**. Correctness held across all 39 strategy x distance configurations: every one
produced a bit-identical final state.

**This is a synthetic locality benchmark, not a model speedup.** No model and no runtime were
involved. It was enough to justify building the real integration, and that integration has
since contradicted it on four separate axes — including this headline. Read the real section
above first; this one explains mechanism and decides nothing.

### The batch-1 policy does not survive batch N

Every number above is one sequence. Under concurrency the picture inverts
([`results/rtx5090-surfaces.json`](results/rtx5090-surfaces.json)):

| sequences | 1 | 4 | 16 | 32 |
|---|--:|--:|--:|--:|
| persist | +9.9% | **-11.1%** | -1.8% | -0.6% |
| prefetch | +13.9% | +7.3% | **+22.9%** | -4.8% |
| combined | +9.9% | -0.4% | -17.6% | **-21.0%** |

`persist` turns harmful at four sequences — and the v0.1 planner did not consider that
oversubscribed at all, because its hot-set model counted one layer's state and ignored that
the other 47 layers' are equally live. **That accounting is now fixed**
(`--hot-set-model token_footprint`, with the old model kept as the control); what the
corrected number should make the policy *do* at each concurrency is still open. Separately,
64 MiB of streaming weight traffic against a 48 MiB set-aside costs **two thirds of
throughput**.

Three more that overturn stated assumptions: **kernel-integrated prefetch is 30-33 points
worse** than a separate stream (section 34.4 and the v0.5 roadmap assume the opposite);
**state layout alone is a 2.3x effect** and flips which policy wins; and the `cliff` hot-set
policy scores **+22.8% where every other policy scores -17.4%** at 16 sequences.

These are the open problems, not the settled results — and the real integration has since
shown that several of them are artefacts of a benchmark that does not capture a CUDA graph.
[`docs/OPTIMIZATION-SURFACES.md`](docs/OPTIMIZATION-SURFACES.md) maps every surface on both
benchmarks, the flag that isolates each, and where the two disagree.

## Go / no-go gate

| Real Qwen3.8 / SparkInfer result | Decision |
|---|---|
| <2% end-to-end gain | reject the project |
| 2–4% | probably reject |
| 4–7% | promising |
| 7–10% | strong candidate |
| >10% | expand immediately |

**The gate has been run and the result is in the first band.** The scored measurement lives
in `results/rtx5090-real.json` and the verdict is derived from it by `eval/decide.py --real`,
not asserted — the numbers are quoted once, above, from that command's actual output.

Section 21 says the project should be willing to fail this test. On Qwen3.8-27B, on this
hardware, with these policies, it fails.

What that does and does not mean:

- It does **not** mean the mechanism is broken. The pre-touch demonstrably runs, and the
  persisting window is verifiably present in a replayed graph when it is attached.
- It does mean the mechanism is aimed at 1.65% of the problem at batch 1 on a dense hybrid,
  and that the policies do not convert the 6.7% that concurrency puts on the table.
- The next honest experiment is not more tuning of these policies. It is a model whose weight
  traffic per token is smaller — a sparse MoE, where the same recurrent state is a much larger
  share — or a policy that survives the concurrency scaling this one does not.

No synthetic result should be marketed as a model speedup.

This table is executable rather than advisory. `eval/decide.py` applies it, along with the
impact tiers and the weighted workload score, as a deterministic function of measurements:

```bash
python3 eval/decide.py --synthetic eval-result.json   # reports, never tiers
python3 eval/decide.py --real real-result.json        # scores the real workload matrix
```

The synthetic path exists to say no: a locality microbenchmark is reported but never
labelled, because the only number that decides this project is real end-to-end decode. The
real path gates on bit-identical output, blocks any workload regressing more than 2%, and
combines the matrix with a weighted geometric mean so one cherry-picked win cannot carry a
PR that loses elsewhere.

## Using it from a runtime

RecurLocal is a component, not an application. It installs a CMake package:

```cmake
find_package(RecurLocal 0.1 REQUIRED)
target_link_libraries(your_runtime PRIVATE RecurLocal::recurlocal_cuda)
```

Construction and every entry point are `noexcept` and report `cudaError_t`; the controller
detects CUDA Graph capture and hands the access-policy window back for the caller to attach to
its kernel node, because a stream attribute is not recorded into a graph. See
`integrations/sparkinfer/README.md` for the full contract.

## Build: CPU-only

```bash
cmake -S . -B build -DRECURLOCAL_BUILD_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/recur_local_info
```

## Build: CUDA

```bash
cmake -S . -B build \
  -DRECURLOCAL_BUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Synthetic CUDA benchmark

```bash
./build/recur_local_cuda_bench baseline
./build/recur_local_cuda_bench persist
./build/recur_local_cuda_bench prefetch
./build/recur_local_cuda_bench combined

python3 eval/run_eval.py --binary ./build/recur_local_cuda_bench
```

The default benchmark emulates 48 recurrent layers with 3 MiB of state per layer, over 32 timed tokens preceded by 4 untimed warm-up tokens. It is a memory-locality experiment, **not** a faithful GDN model benchmark.

Benchmark options: `--layers`, `--tokens`, `--warmup-tokens`, `--inner-iters`, `--state-bytes`, `--device`.

The evaluator runs every mode `--repeats` times (default 3), interleaved so drift does not land on one mode, and compares median timings. It refuses to report a result unless every run of every mode produced the same final-state checksum, and marks the run unstable when the run-to-run spread is wider than the difference being measured — the bands below start at 2%, which a single run per mode cannot resolve. Pre-touch work is joined into the timed region, so prefetch overhead counts against prefetch modes.

Every result records what produced it — GPU, compute capability, driver and runtime version,
commit and whether the tree was dirty, and the observed graphics clock. Pass
`--pin-clock-mhz auto` (needs privileges) to lock the clock so an absolute time is
reproducible off the box, not merely same-box comparable.

## Integration contract

A runtime needs only the lifecycle hook shown in `integrations/sparkinfer/README.md`. The
SparkInfer integration is 77 lines of insertions against a pinned commit, and CI asserts it
deletes nothing.

The CUDA code is tested: `tests/test_cuda_controller.cu` runs 115 device-side checks over the
controller, the graph-capture state machine and every pre-touch strategy, and
`scripts/sanitize.sh` keeps it memcheck/initcheck/synccheck/racecheck clean. Writing those
found nine real defects, including a persisting-L2 set-aside that was never given back and a
failing pre-touch that could strand a host runtime's graph capture — see the CHANGELOG.

`stats()` reports windows applied, windows *deferred* under graph capture, windows actually
attached to a captured graph node, layers where the hot set was oversubscribed, and pre-touch
volume — so a null end-to-end result can be explained rather than guessed at. That mattered
here: without the node attachment the persisting policy would have been absent from every
graph replay, `windows_attached_to_node` would have read 0, and `persist` would have measured
its cost with none of its effect.

## Contribution model

There are deliberately no bounty-style optimization issues required. Profile `main`, find a bottleneck, and move the frontier.

[`docs/OPTIMIZATION-SURFACES.md`](docs/OPTIMIZATION-SURFACES.md) maps every surface, the flag
that isolates it, and the current frontier on each — including the two that are measured and
open, and the two that cannot be competed on yet because nothing measures them.

## References

- Qwen3.8-27B config: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
- CUDA L2 cache control: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html
- SparkInfer: https://github.com/gittensor-ai-lab/sparkinfer

## License

MIT
