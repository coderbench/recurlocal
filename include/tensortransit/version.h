#pragma once

// Single source of truth for the project version: CMake parses these values rather
// than declaring its own, so a consumer's compile-time and package-time versions
// cannot disagree.
//
// 0.2.0 is the RecurLocal -> TensorTransit generalization. The recurrent locality
// policy is unchanged and every number measured under 0.1.0 still reproduces; what is
// new is that the policy is now one planner among several behind a tensor-role-agnostic
// core. See docs/roadmap.md and docs/STABILITY.md section 8.
//
// 0.2.1 puts that core into the measured path. The SparkInfer adapter and the synthetic
// benchmark drove the 0.1 controller directly until now, so a planner a contributor wrote
// moved no measured number; both route through Registry -> Graph -> Planner -> Executor,
// with the 0.1 controller kept as TENSORTRANSIT_ENGINE=v0 so the migration is an A/B rather
// than an assertion. It also registers KV, replaces the linear cost model, and replaces the
// impact bands with the continuous Frontier Gain. Nothing removed; every 0.1 name still
// resolves and every 0.2.0 API is source-compatible.
#define TENSORTRANSIT_VERSION_MAJOR 0
#define TENSORTRANSIT_VERSION_MINOR 2
#define TENSORTRANSIT_VERSION_PATCH 1
#define TENSORTRANSIT_VERSION_STRING "0.2.1"

// Comparable as one integer, which requires MINOR and PATCH to stay under 100. If either
// ever needs three digits the multipliers must change together with every comparison.
#define TENSORTRANSIT_VERSION_NUMBER                       \
    (TENSORTRANSIT_VERSION_MAJOR * 10000 +                 \
     TENSORTRANSIT_VERSION_MINOR * 100 + TENSORTRANSIT_VERSION_PATCH)

// What a consumer writes instead of comparing the pieces by hand. An enumerator or a struct
// field added in 0.3.0 is guarded with
//     #if TENSORTRANSIT_VERSION_AT_LEAST(0, 3, 0)
// so one source tree can build against two vintages of this header.
#define TENSORTRANSIT_VERSION_AT_LEAST(maj, min, pat) \
    (TENSORTRANSIT_VERSION_NUMBER >= ((maj) * 10000 + (min) * 100 + (pat)))

namespace tensortransit {

// Version of the library actually linked, for consumers that load it dynamically or
// vendor it and need to check what they got.
const char* version_string() noexcept;
int version_number() noexcept;

}  // namespace tensortransit
