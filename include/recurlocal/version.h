#pragma once
// Compatibility shim. RecurLocal is now the recurrent-state workload inside TensorTransit;
// this header keeps a source tree that included the old path building unchanged.
//
// Deprecated in 0.2.0. It will be removed no earlier than 0.3.0, and docs/STABILITY.md
// section 8 says what that removal will and will not break.
#include "tensortransit/version.h"

#define RECURLOCAL_VERSION_MAJOR TENSORTRANSIT_VERSION_MAJOR
#define RECURLOCAL_VERSION_MINOR TENSORTRANSIT_VERSION_MINOR
#define RECURLOCAL_VERSION_PATCH TENSORTRANSIT_VERSION_PATCH
#define RECURLOCAL_VERSION_STRING TENSORTRANSIT_VERSION_STRING
#define RECURLOCAL_VERSION_NUMBER TENSORTRANSIT_VERSION_NUMBER
#define RECURLOCAL_VERSION_AT_LEAST(maj, min, pat) TENSORTRANSIT_VERSION_AT_LEAST(maj, min, pat)

// The whole of the old namespace, by alias rather than by re-declaration: an alias cannot
// drift from what it names, where a second set of declarations could and eventually would.
namespace tensortransit {}
namespace recurlocal = tensortransit;
