# The Transit Graph

The Transit Graph is the object this project is named after. Everything a planner can do is a
consequence of knowing, before a kernel runs, what it and the kernels after it are going to
read.

```text
order ------------------------------------------------------------->

 K1          K2          K3          K4          K5        | K1'
 W1(r)     S_a(rw)     W2(r)       S_b(rw)     KV(r,w)     | W1(r)
                                                           |
   +-------------------- wrap edge --------------------------+
              S_a is next read at K2 of the NEXT token
```

## What it computes

For every pair of consecutive uses of a tensor where the second one reads, the graph records
a `TransitEdge` carrying the reuse distance in **three currencies**:

| metric | question it answers | when it is the right one |
|---|---|---|
| `Ordinal` | how many kernels run in between | cheap, engine-independent, wrong as a residency predictor whenever kernels differ in size |
| `Bytes` | how much *other* traffic flows in between | **residency**: a line survives to its next use only if the cache is bigger than this |
| `Time` | how many nanoseconds in between | **prefetch timing**: a prefetch must be issued far enough ahead to complete |

These are not three views of one number. They disagree, and the disagreement is the point:
residency is a bytes question and prefetch timing is a time question, so a planner that does
both reads both. `--reuse-metric` is a swept axis rather than a constant somebody picked.

## The factor of two

A `ReadWrite` use moves its bytes **twice** — read in, modified value written back. This is
not a refinement; it is the factor of two in every ceiling this project has ever quoted:

```text
48 recurrent layers x (3 MiB fp32 + 60 KiB bf16) x 2  =  294 MiB per token
```

A model that counted a read-modify-write once would report half the removable traffic and
halve every ceiling with it. `tests/test_core.cpp` pins it.

KV is the counter-example and the traces model it accordingly: in decode, KV is a large
**read** of the whole block plus a small **write** of one token's append — not a
read-modify-write of the block. Getting that wrong does not make a comparison noisy, it makes
it meaningless, because bytes-saved-per-byte-held is exactly the quantity the admission rules
discriminate on.

## The loop

`set_cyclic(true)` says the recorded window is one iteration of a loop, so the last use of a
tensor is followed by its first. **A decode token is such a loop and a prefill is not.**

A recurrent state is used once per token. Without the wrap edge it has no reuse at all, every
planner correctly declines it, and the whole recurrent surface reads as unreachable. That is a
wrong answer that looks exactly like a right one, which is why it is a test rather than a
convention.

## What it says about this project's own numbers

The graph reproduces the published bound from a recorded trace, rather than from a
hand-written geometry file:

```console
$ tensortransit inspect tests/golden/trace_recurrent.json
  traffic  18500000000 B per iteration

per role:
  role                  tensors     uses          bytes      removable
  recurrent_state            96       96      153944064      307888128
  model_weight               64       64    18192111872    18192111872

ceilings [MODEL, not measurements]:
  policy scope          unlimited on this device     resident     held B
  recurrent_state         +1.692%        +0.685%          41%   62914560
```

- removable recurrent traffic **307,888,128 B = 293.6 MiB** — published as 294 MiB
- footprint **153,944,064 B = 146.8 MiB** — published as 146.8 MiB
- residency **41%** against the device's 60 MiB — published as 41%
- traffic ceiling **+1.692%**, persist ceiling **+0.685%** — published as 1.69% and 0.68%

That agreement is asserted in `tests/test_golden.cpp`. If the graph ever stops reproducing it,
the graph is wrong — the published figures were measured before any of this code existed.

## "Unbounded" is a broken question, not a large number

Over a cyclic window **every** tensor is re-read next iteration, weights included. So a cache
of unlimited size removes the whole step and the unlimited-cache ceiling saturates. `inspect`
prints `unbounded` there rather than the `+0.000%` that `f/(1-f)` collapses to at `f = 1`,
which would read as "there is nothing here" when the truth is the opposite.

**The column that binds is the device one.** It is a density-greedy fill of the device's
persisting budget, not a proportional scaling of the unlimited figure — those differ whenever
candidates differ in density, and a read-modify-written state saves two bytes per byte held
where a streamed weight saves one.

## Cost

`build()` is prefix-sum based, so "traffic strictly between use *i* and use *j*" is one
subtraction rather than a scan. The analysis happens once and the plan is reused across
tokens; a graph that recomputed itself per query would put its cost inside the decode loop
that the plan cache exists to keep it out of.

## Privacy

A trace carries tensor ids, sizes, roles, kernel ids, ordering and timing. It carries **no
pointers** — a device address from another process is meaningless on the machine that reads
it — and there is no field in `TraceMetadata`, in the writer, or in
`schemas/trace.schema.json` that can hold a prompt, a token, or a request body. Spec section
45 is enforced by construction rather than by policy.
