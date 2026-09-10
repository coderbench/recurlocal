// The 0.1 surface still exists and still names the same things.
//
// A compatibility promise nothing exercises is a compatibility hope. This file includes ONLY
// the deprecated headers and uses ONLY the deprecated namespace, so it fails to build the
// moment either stops working -- which is the point: the failure has to be a build error in
// this repository rather than a link error in somebody else's.
#include "recurlocal/cuda_api.h"
#include "recurlocal/planner.h"
#include "recurlocal/sparkinfer.h"
#include "recurlocal/version.h"

#include <cstdio>
#include <iostream>
#include <string>

static int g_failures = 0;
// Counted, and printed on success. A suite that says only "passed" cannot be
// distinguished from one whose checks were all compiled out, and the repo manifest
// used to carry a hand-typed total that nothing regenerated.
static int g_checks = 0;
#define CHECK(cond) do { ++g_checks; if (!(cond)) { \
    std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_failures; } } while (0)

static void test_the_old_namespace_still_names_the_planner() {
    // `recurlocal` is an ALIAS of `tensortransit`, not a second set of declarations, so
    // these are the same entities and cannot drift apart.
    recurlocal::DeviceCaps caps{96ull << 20, 64ull << 20, 32ull << 20};
    recurlocal::PlannerConfig config;
    config.mode = recurlocal::LocalityMode::Combined;
    config.persisting_budget_fraction = 0.5;
    recurlocal::LocalityPlanner planner(caps, config);
    CHECK(planner.recommended_l2_set_aside() == (32ull << 20));

    const recurlocal::LayerPlan plan = planner.plan_for_layer(3ull << 20, true);
    CHECK(plan.use_persisting_window);

    // The same type reached through both names is one type.
    tensortransit::LayerPlan* same = &const_cast<recurlocal::LayerPlan&>(plan);
    CHECK(same != nullptr);
}

static void test_the_old_version_macros_still_expand() {
    CHECK(std::string(RECURLOCAL_VERSION_STRING) == TENSORTRANSIT_VERSION_STRING);
    CHECK(RECURLOCAL_VERSION_NUMBER == TENSORTRANSIT_VERSION_NUMBER);
    // A consumer guarding on the 0.1 macro must still compile against 0.2.
    CHECK(RECURLOCAL_VERSION_AT_LEAST(0, 1, 0));
    CHECK(std::string(recurlocal::version_string()) == TENSORTRANSIT_VERSION_STRING);
}

static void test_the_old_enum_values_have_not_moved() {
    // Enumerator VALUES are what cross a compiled boundary. A consumer built against the 0.1
    // header and linked against 0.2 reads these numbers, so they are the ABI.
    CHECK(static_cast<int>(recurlocal::LocalityMode::Baseline) == 0);
    CHECK(static_cast<int>(recurlocal::LocalityMode::Persist) == 1);
    CHECK(static_cast<int>(recurlocal::LocalityMode::Prefetch) == 2);
    CHECK(static_cast<int>(recurlocal::LocalityMode::Combined) == 3);
    CHECK(static_cast<int>(recurlocal::PreTouchStrategy::Scalar) == 0);
    CHECK(static_cast<int>(recurlocal::HotSetPolicy::Proportional) == 0);
    CHECK(static_cast<int>(recurlocal::HotSetPolicy::Quota) == 4);
    CHECK(static_cast<int>(recurlocal::WindowAttach::Stream) == 0);
    CHECK(static_cast<int>(recurlocal::WindowAttach::CaptureNodeStrict) == 2);
    CHECK(static_cast<int>(recurlocal::StateKind::Matrix) == 0);
}

static void test_the_adapter_surface_is_reachable_under_both_names() {
    // Without CUDA the adapter compiles away to inline no-ops; with it, these are the real
    // five calls. Either way the NAMES have to resolve, because that is what a patched
    // runtime writes by hand.
    CHECK(recurlocal::sparkinfer::enabled() == tensortransit::sparkinfer::enabled());
    CHECK(std::string(recurlocal::sparkinfer::mode_name()) ==
          tensortransit::sparkinfer::mode_name());
    CHECK(static_cast<int>(recurlocal::sparkinfer::StateKernel::Conv) == 0);
    CHECK(static_cast<int>(recurlocal::sparkinfer::StateKernel::Gdn) == 1);
}

int main() {
    test_the_old_namespace_still_names_the_planner();
    test_the_old_version_macros_still_expand();
    test_the_old_enum_values_have_not_moved();
    test_the_adapter_surface_is_reachable_under_both_names();

    if (g_failures) { std::cout << g_failures << " compatibility check(s) failed\n"; return 1; }
    std::cout << "compatibility tests passed (" << g_checks << " checks)\n";
    return 0;
}
