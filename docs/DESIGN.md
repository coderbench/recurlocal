# Design

## Core hypothesis

Recurrent state differs from streamed model weights: it is relatively small per layer, mutable, revisited every decode token, and accessed in a predictable layer order.

RecurLocal tests whether explicit cache policy plus next-layer warm-up can exploit this reuse.

## CUDA semantics

CUDA exposes stream-level access-policy windows with `cudaAccessPropertyPersisting` and `cudaAccessPropertyStreaming`, plus device-reported limits for L2 set-aside and access-window size.

These are hints. Oversubscription can cause thrashing, so all policies must be measured.

## Prefetch semantics

There is no general CUDA API that guarantees an arbitrary `cudaMalloc` region is placed into L2. The v0 prototype uses an asynchronous read-only pre-touch kernel. It may help or hurt; the evaluator decides.

## Correctness

RecurLocal v0 does not change model arithmetic or state representation. A lossy/compressed state representation would belong to a different evaluation contract.
