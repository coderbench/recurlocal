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

## Go / no-go gate

| Real Qwen3.8 / SparkInfer result | Decision |
|---|---|
| <2% end-to-end gain | reject the project |
| 2–4% | probably reject |
| 4–7% | promising |
| 7–10% | strong candidate |
| >10% | expand immediately |

No synthetic result should be marketed as a model speedup.

## Build: CPU-only

```bash
cmake -S . -B build -DRECURLLOCAL_BUILD_CUDA=OFF
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/recur_local_info
```

## Build: CUDA

```bash
cmake -S . -B build \
  -DRECURLLOCAL_BUILD_CUDA=ON \
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

The default benchmark emulates 48 recurrent layers with 3 MiB of state per layer. It is a memory-locality experiment, **not** a faithful GDN model benchmark.

## Integration contract

A runtime needs only the lifecycle hook shown in `integrations/sparkinfer/README.md`.

## Contribution model

There are deliberately no bounty-style optimization issues required. Profile `main`, find a bottleneck, and move the frontier.

Useful directions include cache-window sizing, hit-ratio policy, prefetch distance, pre-touch kernels, stream scheduling, batch-aware hot sets, CUDA Graph integration, device-specific policy, and instrumentation.

## References

- Qwen3.8-27B config: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json
- CUDA L2 cache control: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html
- SparkInfer: https://github.com/gittensor-ai-lab/sparkinfer

## License

MIT
