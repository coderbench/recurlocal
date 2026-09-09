# SparkInfer integration

RecurLocal should hook immediately before an existing recurrent / linear-attention layer reads its mutable recurrent state.

Do **not** fork SparkInfer's GDN math. Keep SparkInfer responsible for the model computation and expose only a narrow locality hook.

Conceptual integration:

```cpp
for (layer : layers) {
    if (layer.is_recurrent()) {
        recurlocal.before_layer(
            recurrent_state_ptr,
            recurrent_state_bytes,
            next_recurrent_state_ptr,
            next_recurrent_state_count,
            has_next_recurrent_layer);

        run_existing_sparkinfer_recurrent_kernel(...);
        recurlocal.after_layer();
    } else {
        run_existing_layer(...);
    }
}
```

The integration needs the current/next state pointer and size, device id, compute stream, and an optional low-priority prefetch stream.

## Correctness rule

The locality layer must not change state values, precision, or model math. The pre-touch path is read-only and best-effort.

## Real evaluation

Record the exact SparkInfer commit, model/checkpoint digest, CUDA toolkit, driver, GPU, prompts, context lengths, concurrency, and RecurLocal commit. Never benchmark against a moving `main` in a release claim.
