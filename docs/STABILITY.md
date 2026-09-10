# API and ABI stability

TensorTransit is embedded by inference runtimes, and its contract is a set of enumerator
values other people switch on. This file says what a consumer may rely on across versions, what it
may not, and which of those claims the build actually checks rather than merely asserts.

It is a v0.2 prototype and the promises below are scoped to that honestly: the API is stable,
the ABI is not offered at all, and the supported way to consume this library is to build it
from source in the same build as the runtime that embeds it.

## 1. Surfaces and their tiers

| Surface | Tier | Meaning |
|---|---|---|
| `include/tensortransit/version.h` | **Stable** | Macros and two functions; additive only. |
| `include/tensortransit/recurrent.h` | **Stable API, unstable ABI** | Enumerator names, enumerator values, struct field names and meanings are stable. Struct *sizes* are not. |
| `include/tensortransit/cuda_recurrent.h` | **Stable API, unstable ABI** | Same, plus everything below about `ControllerStats`. |
| `include/tensortransit/sparkinfer.h` | **Stable API, unstable ABI** | The five-call hook surface. This is the one a real runtime patches against. |
| `include/tensortransit/tensor.h`, `graph.h`, `plan.h`, `planner.h`, `executor.h`, `device.h`, `runtime.h`, `trace.h`, `ceiling.h` | **Stable API, unstable ABI** | The 0.2 core. Enumerator names and values, struct field names and meanings are stable; struct sizes are not. |
| CMake package `TensorTransit`, targets `TensorTransit::tensortransit`, `::tensortransit_cuda`, `::tensortransit_sparkinfer` | **Stable** | Names, namespace, and the `TensorTransit_VERSION` a `find_package` sets. |
| CMake package `RecurLocal` and the `RecurLocal::*` targets | **Deprecated, still stable** | Section 6a. Removed no earlier than 0.3.0. |
| Everything in section 7 | **Not stable** | Changes without notice, in a patch release. |

Everything under `include/` is public and installed by `install(DIRECTORY include/ ...)`.
Everything under `src/` and `executors/` is implementation. As of 0.2.0 there is exactly one
implementation header, `planners/common.h`: it is not installed, it is reachable only from the
build tree, and nothing outside `planners/` may include it. `schemas/*.json`, in contrast,
ARE part of the contract — see section 5a.

## 2. Enums: names versus values

An enum in a consumer's `switch` is **source-compatible** if the enumerator *names* are
stable. It is **ABI-compatible** only if the enumerator *values* are stable, because a value
is what actually crosses a compiled boundary.

RecurLocal guarantees **both**, for every public enumerator, with one asymmetry stated in
section 3: values are stable in the sense that an existing enumerator never changes its
numeric value, but a *newer* library may return a value an older consumer has never heard of.

Every public enum is a scoped enumeration (`enum class`). A scoped enumeration has a fixed
underlying type of `int` unless another is given, so `sizeof` is 4 on every supported
platform and **adding an enumerator can never change it**. As of this release each one also
says `: int` outright, so the guarantee is greppable rather than inferred from the standard,
and every one round-trips through `to_string`/`parse_*` under `tests/test_planner.cpp`.

Which enums cross the ABI boundary, and how:

| Enum | Crosses as | ABI-relevant |
|---|---|---|
| `LocalityMode` | field of `PlannerConfig` (passed **by value**); parameter/return of `to_string`/`parse_mode` | **yes** |
| `PreTouchStrategy` | field of `PlannerConfig` **and** of `LayerPlan`; **direct parameter** of `pre_touch_async`, `pre_touch_bytes_async`, `pre_touch_rows_async` | **yes, three ways** |
| `HotSetPolicy` | field of `PlannerConfig` | **yes** |
| `SetAsidePolicy` | field of `PlannerConfig` | **yes** |
| `PrefetchSchedule` | field of `PlannerConfig` | **yes** |
| `HotSetModel` | field of `PlannerConfig` and of `LayerPlan` | **yes** |
| `WindowScope` | field of both; **parameter** of `resolve_window_region` | **yes** |
| `WindowTarget` | field of both; **parameter** of `select_window_segment` | **yes** |
| `PreTouchCoverage` | field of both; **parameter** of `pre_touch_covers` | **yes** |
| `PrefetchJoin` | field of `PlannerConfig` | **yes** |
| `WindowAttach` | field of `PlannerConfig` | **yes** |
| `StateKind` | field of `StateSegment` and of `CudaLocalityController::RowSet`; **parameter** of `pre_touch_covers` | **yes** |
| `sparkinfer::StateKernel` | **parameter** of `after_launch` — the only RecurLocal enum a patched runtime writes by hand | **yes** |

There is no public enum that is source-only. Every one of them is in a struct passed by
value, in a function signature, or both.

**What you may rely on, per enum:**

- The **name** of an enumerator that exists today will not change or be reused for a
  different meaning. Removal follows section 6.
- The **numeric value** of an enumerator that exists today will not change. New enumerators
  are appended after the last one, so existing values are pinned by construction.
- The **string** an enumerator round-trips through (`to_string` / `parse_*`) will not change
  for an enumerator that exists today. Those strings are the sweep-axis and environment-
  variable vocabulary (`RECURLOCAL_PRE_TOUCH=vec4_ldcg`, `RECURLOCAL_WINDOW_ATTACH=stream`,
  …), so renaming one silently re-labels every result file that names it. Covered by the
  round-trip tests in `tests/test_planner.cpp`.
- The **underlying type is `int`** and `sizeof` is 4, on every platform this library builds
  for.

**What you may not rely on:** that the set is closed. See the next section.

## 3. Adding an enumerator

The project's convention guarantees this happens often: a mechanism is an enumerator plus an
implementation, selectable at run time, never a file replacement (`CONTRIBUTING.md`, "Adding
a mechanism"). So the rule has to be explicit.

**A new enumerator is appended after the current last enumerator of its enum. It is never
inserted and never given an explicit value.**

Consequences, precisely:

- **To ABI: nothing.** The enum's underlying type is fixed at `int`; appending changes no
  existing value and no `sizeof`. A pre-built consumer that never sees the new value
  continues to work.
- **To a consumer's exhaustive `switch`: a `-Wswitch` warning, not a break.** A `switch` over
  a scoped enum with no `default:` that does not handle the new enumerator still compiles and
  still runs; GCC and Clang warn under `-Wall`. The library's own `to_string` functions are
  written this way deliberately — no `default:`, an unreachable trailing `return` — so that
  adding an enumerator and forgetting its string is a compiler warning in RecurLocal's own
  build (`-Wall -Wextra`, `CMakeLists.txt:49`) rather than a silent mislabel.
- **To an *older* consumer at run time:** a newer library may hand back an enumerator the
  consumer's build has never heard of, in a `LayerPlan` field. A consumer that must be robust
  across versions should treat an unrecognised value as "an axis I do not model" and fall
  through to its default handling — not assert. This is the one direction in which enum
  *values* are not a two-way contract, and it is why the `to_string` functions all return
  `"unknown"` for an out-of-range value rather than naming a plausible one.
- **To the release number: a minor bump.** Appending an enumerator is a feature. See
  section 5.

Adding an enumerator obliges four things in the same commit, all of them already project
convention: the value, its `to_string`/`parse_*` pair and round-trip test, the implementation
behind that value only, and a row in the relevant sweep-axis table.

## 4. Struct layout

Six structs cross the boundary: `PlannerConfig`, `LayerPlan`, `DeviceCaps`,
`RecurrentGeometry`, `StateSegment` (`planner.h`), and `LayerActions`, `ControllerStats`,
`CudaLocalityController::RowSet` (`cuda_api.h`), plus `sparkinfer::GdnStateLayout` and
`GdnPackedLayout`. All are aggregates with NSDMI defaults; all are trivially copyable and
standard-layout.

**The rule for adding a field: append it after the current last member. Never insert one
between existing members, and never drop one into interior padding.**

What each does:

| Change | Source compatibility | ABI |
|---|---|---|
| Append a field at the end | Kept. Existing designated/aggregate initialisation still compiles; the new field takes its NSDMI default. | **Broken.** `sizeof` usually grows, and a pre-built consumer passing the old struct by value under-supplies the new one. Usually caught by the `static_assert` on `sizeof`; see the tail-padding hole below, which it does not catch. |
| Insert a field between members | Kept. | **Broken, silently, in the worst way.** Every following field's offset moves. If the field lands in interior padding, `sizeof` does not even change. |
| Reorder, retype, or rename a field | **Broken.** | **Broken.** |
| Change a field's *default* | Kept, ABI kept — but it changes the measured behaviour of every caller that did not set it explicitly. Treat it as a behaviour change and record it in the changelog with the measurement that motivated it. |

`PlannerConfig` has interior padding at byte offsets 4, 52 and 92 on LP64. A four-byte enum
dropped into the hole at 92 leaves `sizeof(PlannerConfig)` at 104 and moves no existing
offset, and an old caller's value-initialised `PlannerConfig{}` (padding zeroed) would hand
the new library enumerator 0 of an axis it has never heard of — a wrong measurement with no
symptom. **This is why the rule is "append after the last member" rather than "keep `sizeof`
stable", and why `planner.h` asserts an `offsetof` table and not only sizes.**

**The one case the asserts do not catch, stated rather than papered over: tail padding.**
`StateSegment` ends with a four-byte `StateKind` at offset 32 and has `sizeof` 40 — four bytes
of tail padding. Appending a four-byte enum to it leaves `sizeof` at **40** and moves no
existing offset, so the size assert passes, the `offsetof` table passes, and a pre-built
consumer's value-initialised `StateSegment{}` hands the new library enumerator 0 of an axis it
has never heard of. Verified: `struct` with the extra member measures 40, identical.
`PlannerConfig` and `LayerPlan` fall into the same trap on every *second* four-byte append.

There is no portable compile-time member count, so no assert closes this. The rule for that
case is enforced by review: **an append that does not change `sizeof` still requires a minor
version bump and a changelog entry**, and a reviewer who sees a new field and an unchanged
assert should treat that as the hazard rather than as a convenience.

Current LP64 sizes, asserted in the headers:

| Struct | `sizeof` |
|---|--:|
| `PlannerConfig` | 104 |
| `LayerPlan` | 72 |
| `DeviceCaps` | 24 |
| `RecurrentGeometry` | 32 |
| `StateSegment` | 40 |
| `WindowRegion` | 16 |

### 4.1 `PlannerConfig` is passed by value, and the remedy is source-only distribution

`LocalityPlanner(DeviceCaps caps, PlannerConfig config)` (`planner.h:325`),
`CudaLocalityController(int device, PlannerConfig config)` (`cuda_api.h:123`) and
`initialize(int device, PlannerConfig config)` (`cuda_api.h:128`) all take `PlannerConfig`
**by value**. Adding a field to it is therefore an ABI break for any consumer whose object
files were compiled against the older header, whatever the linkage.

**RecurLocal's answer is not a versioned struct. It is: there is no binary distribution, and
the build enforces it.**

- RecurLocal is a **static archive only**. `add_library(recurlocal STATIC ...)` is
  unconditional; `BUILD_SHARED_LIBS=ON` does not produce a shared library. There is no
  `.so`, therefore no `SOVERSION`, and this file does not promise one. If a shared build is
  ever offered it will carry `SOVERSION` equal to `${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}`
  while the major version is 0, and this section will be rewritten.
- **The supported way to consume RecurLocal is to build it from source, from the same tree,
  in the same build, as the runtime that embeds it.** That is exactly what
  `adapters/sparkinfer/build.sh` does: it wipes and reinstalls RecurLocal from scratch on
  every integration build before configuring SparkInfer against it.
- **A stale mix is not yet caught.** The `static_assert`s on struct size and field offset in
  `planner.h` fire in the CONSUMER's build, so a consumer whose header disagrees with this
  release's layout fails to compile. They do not catch the other direction — a consumer whose
  header is current but whose archive is stale. Section 9 names the hardening that would, and
  says plainly that it is not done.

**Why not a size/version field in `PlannerConfig`.** The `cbSize`-style trick does work for a
104-byte struct — it is passed in memory on SysV x86-64, so a callee that reads the size
field first can avoid touching bytes the caller did not supply. It is rejected anyway: it
would oblige the library to carry a compatibility path for every field it ever adds, forever,
to serve an ABI it has just declared it does not offer, for a project with one real consumer
that rebuilds from source on every run. **Why not an opaque handle.** It would replace
`PlannerConfig cfg{}; cfg.hit_ratio = …;` — the aggregate that the SparkInfer adapter and
every test builds — with a setter per axis, and a new axis would then be a new *function*
rather than a new field, which is a worse tax on the one operation this project performs
constantly. A v0.1 prototype's honest position is the narrow one: **build from source, and let
the linker prove you did.**

## 5. Versioning

RecurLocal follows semantic versioning, with the pre-1.0 clause taken seriously rather than
used as an excuse.

The version has a single source of truth: `include/recurlocal/version.h`, which
`CMakeLists.txt` parses to set `project(... VERSION ...)`, so a consumer's compile-time
`RECURLOCAL_VERSION_*` and the version `find_package(RecurLocal)` reports cannot disagree.
`write_basic_package_version_file(... COMPATIBILITY SameMinorVersion)` means
`find_package(RecurLocal 0.1 REQUIRED)` accepts any 0.1.x and rejects 0.2.0. **Pin the minor
version in your `find_package` call.** A bare `find_package(RecurLocal REQUIRED)` accepts
anything and gives you none of this.

**While the major version is 0:**

| Release | May contain |
|---|---|
| **0.Y.Z → 0.Y.(Z+1)** (patch) | Bug fixes, measurement fixes, new telemetry fields, documentation. **No** new enumerator, **no** new struct field, **no** signature change. Source- and ABI-compatible in both directions. |
| **0.Y.Z → 0.(Y+1).0** (minor) | New enumerators, new struct fields, new functions, new default values, deprecations. **Source-compatible** for existing code (modulo a `-Wswitch` warning). **ABI-breaking** — the inline-namespace tag is bumped, so a stale link fails loudly. |
| **0.(Y+1).0** | Also where a removal previously announced under section 6 lands. |

In semver terms, pre-1.0 the **minor** position carries what 1.x would put in the major
position. That is the honest statement of where this project is: it adds an enumerator most
weeks, and pretending each one is a major release would make the number meaningless.

Compile-time tests a consumer can write:

```c++
#include "recurlocal/version.h"
#if RECURLOCAL_VERSION_AT_LEAST(0, 2, 0)
    cfg.set_aside_policy = recurlocal::SetAsidePolicy::Residency;
#endif
```

`RECURLOCAL_VERSION_MAJOR/MINOR/PATCH`, `RECURLOCAL_VERSION_STRING`,
`RECURLOCAL_VERSION_NUMBER` (`MAJOR*10000 + MINOR*100 + PATCH`, so minor and patch must stay
under 100) and `RECURLOCAL_VERSION_AT_LEAST(maj,min,pat)` are all available. `RECURLOCAL_ABI`
is the inline-namespace tag as an integer, for a consumer that wants to assert it.
`recurlocal::version_string()` and `version_number()` report what was actually **linked**,
which is the pair to check when you did not build both halves yourself;
`tests/test_planner.cpp` asserts they agree with the macros.

**At 1.0**, if it happens, the major position takes over ABI breaks and the minor position
becomes source-additive-only. Nothing in this file is a commitment to reach 1.0.

## 5a. Serialization schemas

Three documents have a declared, versioned schema, and each carries its version in its own
first field so a reader can refuse a document it does not understand rather than
misinterpreting one:

| document | schema | version field | written by |
|---|---|---|---|
| trace | `schemas/trace.schema.json` | `schema_version` | `write_trace()` |
| plan | `schemas/plan.schema.json` | `plan_schema_version` | `TransitPlan::to_json()` |
| evaluation result | `eval/real_result_schema.json` | `schema_version` | `eval/real_eval.py` |

A breaking change increments the version, and `read_trace()` refuses a version it does not
know **by number, with the number in the message** — it does not attempt a best-effort parse.
`eval/test_schemas.py` validates every committed golden trace and every plan the CLI produces
against these, in the same `ctest` run as the planner tests, because a schema nothing is
checked against is documentation.

Two things the schemas encode that are contract rather than shape:

- **A trace carries no pointers, no prompts, no tokens and no request bodies.** There is no
  field for any of them in `TraceMetadata`, in the writer, or in the schema. Spec section 45
  is enforced by construction rather than by policy.
- **A plan declares `"cost_model": {"basis": "model"}`.** Everything under `cost_model` is an
  output of a model whose assumptions are in `docs/evaluation.md`, never a measurement, and
  `eval/test_schemas.py` fails a plan that omits the marker. The moment a prediction and a
  measurement appear in the same table, somebody will quote whichever is larger.

Note that files in `results/` are **not** all evaluator artifacts. Some are hand-written
roll-ups and they declare no `schema_version`; the schema test lists them rather than
validating them, because a narrative summary sitting next to an authoritative artifact and
looking like one is its own hazard.

## 6. Deprecation

An enumerator or a field is removed over **two minor releases**, never in one.

1. **Release 0.Y** — announce. The symbol keeps working and keeps its value. It is marked
   `[[deprecated("…, removed in 0.(Y+2); use X")]]`, the changelog's *Deprecated* section
   names it and its replacement, and the doc comment says what measurement retired it. A
   deprecated `parse_*` string keeps parsing; `to_string` keeps emitting it, so old result
   files stay readable.
2. **Release 0.(Y+1)** — still present, still warning. This is the release a consumer is
   expected to notice it in.
3. **Release 0.(Y+2)** — removed. The enumerator's numeric value is **retired, not reused**:
   every enumerator after it keeps its value, and the gap stays. Reusing a retired value
   would make an old result file's recorded value name a different mechanism, which is the
   one failure this project cannot tolerate.

A removed `parse_*` string throws `std::invalid_argument` like any other unknown string —
loudly, which is the existing convention (`eval` scripts and the SparkInfer adapter both
refuse to run rather than fall back to a default under the requested label).

**Currently deprecated:**

| Symbol | Deprecated in | Removed no earlier than | Replacement |
|---|---|---|---|
| CMake options `RECURLLOCAL_BUILD_*` (double-L misspelling) | 0.2.0 | 0.4.0 | `TENSORTRANSIT_BUILD_*` |
| CMake options `RECURLOCAL_BUILD_CUDA`, `RECURLOCAL_BUILD_TESTS` | 0.2.0 | 0.4.0 | `TENSORTRANSIT_BUILD_*` |
| Preprocessor macros `RECURLLOCAL_WITH_CUDA`, `RECURLOCAL_WITH_CUDA` | 0.2.0 | 0.4.0 | `TENSORTRANSIT_WITH_CUDA` |
| Headers `include/recurlocal/*.h` | 0.2.0 | 0.3.0 | `include/tensortransit/*.h` |
| Namespace `recurlocal` | 0.2.0 | 0.3.0 | `tensortransit` (it is an alias, not a copy) |
| Macros `RECURLOCAL_VERSION_*` | 0.2.0 | 0.3.0 | `TENSORTRANSIT_VERSION_*` |
| CMake package `RecurLocal`, targets `RecurLocal::*` | 0.2.0 | 0.3.0 | `TensorTransit`, `TensorTransit::*` |
| Environment prefix `RECURLOCAL_` read by the adapter | 0.2.0 | 0.4.0 | `TENSORTRANSIT_` |

Nothing else is deprecated. `WindowAttach::CaptureNode` in particular is **not**: it is now
the better-documented of the two node-attach modes, not the worse one — see
`docs/OPTIMIZATION-SURFACES.md`.

## 6a. The 0.1 -> 0.2 migration, and the two names that are NOT moving

RecurLocal became the recurrent-state workload inside TensorTransit in 0.2.0. The migration is
additive: every 0.1 name still resolves, `tests/test_compat.cpp` includes **only** the
deprecated headers so the shim breaking is a build failure here rather than a link failure in
somebody else's tree, and CI consumes the installed package under both package names.

One deliberate asymmetry in the two `find_package` version files: `TensorTransit` uses
`SameMinorVersion` (the honest answer for a library whose ABI is not offered), and
`RecurLocal` uses `SameMajorVersion`, so an existing `find_package(RecurLocal 0.1 REQUIRED)`
keeps resolving until the shim is removed. That is the only place they differ and it is the
reason they differ.

**Two names are staying put, and both for the same reason: the reproduction path.**

- **The `RECURLOCAL_STATS ` stderr line prefix.** `eval/run_from_base.sh` runs the instrument
  from the BASE commit against a candidate's binary — that is the whole point of it, so a
  one-line change to a noise floor or an estimator cannot ride in on a submission. The base
  commit's parser knows only this prefix. Renaming the emitted line would make every
  base-commit evaluator report "the binary is unhooked" against a perfectly good build, which
  is a false accusation aimed at a contributor. The reader accepts both spellings; the writer
  keeps the one the old readers know.
- **The `RECURLOCAL_*` environment names.** Every command line in `results/*.json` sets them,
  and those commands are how a published number is reproduced. They are read second, after
  the `TENSORTRANSIT_*` spelling.

The corollary is a hazard, and it is guarded on the other side: an evaluator that scrubbed
only `RECURLOCAL*` from the control environment would let an operator with
`export TENSORTRANSIT=combined` compare the candidate against itself and measure ~0%.
`eval/real_eval.py::scrubbed_environment` removes **both** prefixes and is asserted directly
rather than only through an end-to-end run — nothing about a contaminated control looks wrong
in the output.

## 7. Explicitly not stable

None of the following is part of the contract. All of it may change in a patch release,
without deprecation.

- **Adapter telemetry JSON.** The object `tensortransit::sparkinfer::write_stats_json()` writes,
  and the `RECURLOCAL_STATS ` line it prints to stderr at exit. Field names, nesting, and the
  set of counters change whenever a measurement needs a new one — two were added in the
  release that introduced `WindowAttach::CaptureNodeStrict`. It carries **no** `schema_version`
  and is validated against **no** schema; `eval/real_result_schema.json` types it as
  `["object","null"]` deliberately. Read it for provenance, not programmatically.
- **`ControllerStats` field order and `sizeof`.** Counters are grouped by meaning, not
  appended, and have been inserted mid-struct. It is a debugging surface reached through
  `stats()`, and it is the one struct in the public headers whose layout is expected to churn.
  Copy out the fields you need; do not `memcpy` it, serialise it, or store it across a rebuild.
- **`eval/*.py`.** `decide.py`, `real_eval.py`, `real_sweep.py`, `sweep.py`,
  `traffic_budget.py`, `run_eval.py` — their command-line flags, their internals, and their
  thresholds. They are the project's instrument. Their **output** formats are separately
  versioned by `schema_version` in `eval/result_schema.json` and
  `eval/real_result_schema.json`; those integers are the contract, the scripts are not. The
  decision bands they encode are covered by `ctest` because drift there silently redefines
  every past verdict.
- **`workloads/recurrent/synthetic/cuda_bench.cu`** and `recur_local_cuda_bench`'s flags and output. The synthetic
  benchmark is not the score; it has disagreed with the real model on three axes.
- **`adapters/sparkinfer/tensortransit-hook.patch` and `pin.json`.** The patch is written
  against one SparkInfer commit and is expected to be rewritten whenever upstream moves.
  Changing any value in `pin.json` invalidates every `real-result.json` produced before the
  change, which is a *measurement* contract, not an API one. The **adapter header**
  `include/recurlocal/sparkinfer.h` is stable per section 1; the patch that calls it is not.
- **Anything under `src/`.** There are no headers there and no exported symbols that are not
  declared in `include/recurlocal/`.
- **`results/*.json` and `configs/*.json`** as inputs to anything but the scripts that read
  them.

## 8. What the build enforces

Every claim above maps to a check, so this file cannot quietly become false:

| Claim | Enforced by |
|---|---|
| Enum underlying type is `int`, `sizeof` 4 | `: int` written on every declaration in `planner.h` |
| Public struct sizes | `static_assert` in the headers (LP64-guarded), so a consumer's own build checks too — **except an append absorbed by tail padding**, section 4 |
| Public struct field offsets | `offsetof` asserts in `planner.h`, LP64-guarded |
| Enumerator ↔ string round-trip | round-trip tests in `tests/test_planner.cpp` |
| A new enumerator without a string | `-Wall` `-Wswitch` on RecurLocal's own `to_string` |
| Header version ≡ package version | `CMakeLists.txt` parses `version.h`; `tests/test_planner.cpp` asserts the linked functions agree with the macros |
| Consumer header vintage ≡ linked archive vintage | **not enforced** — see section 9 |
| `find_package` version gate | `COMPATIBILITY SameMinorVersion` in `write_basic_package_version_file` |
```

---

---

## 9. Where this contract is weaker than it should be

Stated here rather than left for a reader to discover, because a stability document that
overstates its own enforcement is worse than none.

- **The ABI vintage is not in the mangled names.** Section 4.1 says the supported mode is to
  build from source. Today that is a convention plus a build script
  (`adapters/sparkinfer/build.sh` wipes and reinstalls RecurLocal before configuring
  SparkInfer, so a header/archive mismatch is impossible for the one real consumer by
  accident of the script rather than by construction). The hardening is an inline namespace
  tagged with the vintage — `namespace recurlocal { inline namespace v0 { ... } }` — spelled
  once in `version.h` and applied to all nine declaration and definition sites. It is
  transparent to name lookup, so no consumer source changes, and it puts the vintage into
  every mangled symbol so a stale mix is an undefined reference at link instead of a silently
  mismatched struct. It is not done, and until it is, "build from source" is advice.

- **`tests/test_abi.cpp` does not exist.** The struct-size and offset checks live as
  `static_assert`s in `planner.h` itself, which is stronger in one way — a consumer's own
  build runs them — and weaker in another: there is no per-enum underlying-type assertion and
  no offset table for the `cuda_api.h` structs, because those are declared unstable anyway.

- **Nothing checks the deprecation table.** Section 6's schedule is enforced by review.

