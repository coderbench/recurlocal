#pragma once

// Single source of truth for the project version: CMake parses these values rather
// than declaring its own, so a consumer's compile-time and package-time versions
// cannot disagree.
#define RECURLOCAL_VERSION_MAJOR 0
#define RECURLOCAL_VERSION_MINOR 1
#define RECURLOCAL_VERSION_PATCH 0
#define RECURLOCAL_VERSION_STRING "0.1.0"

#define RECURLOCAL_VERSION_NUMBER \
    (RECURLOCAL_VERSION_MAJOR * 10000 + RECURLOCAL_VERSION_MINOR * 100 + RECURLOCAL_VERSION_PATCH)

namespace recurlocal {

// Version of the library actually linked, for consumers that load it dynamically or
// vendor it and need to check what they got.
const char* version_string() noexcept;
int version_number() noexcept;

} // namespace recurlocal
