#pragma once

// Single source of truth for the project version: CMake parses these values rather
// than declaring its own, so a consumer's compile-time and package-time versions
// cannot disagree.
#define RECURLOCAL_VERSION_MAJOR 0
#define RECURLOCAL_VERSION_MINOR 1
#define RECURLOCAL_VERSION_PATCH 0
#define RECURLOCAL_VERSION_STRING "0.1.0"

// Comparable as one integer, which requires MINOR and PATCH to stay under 100. If either
// ever needs three digits the multipliers must change together with every comparison.
#define RECURLOCAL_VERSION_NUMBER \
    (RECURLOCAL_VERSION_MAJOR * 10000 + RECURLOCAL_VERSION_MINOR * 100 + RECURLOCAL_VERSION_PATCH)

// What a consumer writes instead of comparing the pieces by hand. An enumerator or a struct
// field added in 0.2.0 is guarded with
//     #if RECURLOCAL_VERSION_AT_LEAST(0, 2, 0)
// so one source tree can build against two vintages of this header.
#define RECURLOCAL_VERSION_AT_LEAST(maj, min, pat) \
    (RECURLOCAL_VERSION_NUMBER >= ((maj) * 10000 + (min) * 100 + (pat)))

namespace recurlocal {

// Version of the library actually linked, for consumers that load it dynamically or
// vendor it and need to check what they got.
const char* version_string() noexcept;
int version_number() noexcept;

} // namespace recurlocal
