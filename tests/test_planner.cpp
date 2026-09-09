#include "recurlocal/planner.h"
#include "recurlocal/version.h"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace recurlocal;

// assert() disappears under NDEBUG, which would leave a Release build reporting a
// green test run with nothing checked. These checks are always compiled in.
static int g_failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_failures; } } while (0)
#define CHECK_NEAR(a, b) CHECK(std::abs((a) - (b)) < 1e-9)

static constexpr std::size_t MiB = 1024ull * 1024ull;
static DeviceCaps blackwell_like() { return DeviceCaps{96 * MiB, 64 * MiB, 32 * MiB}; }
static PlannerConfig base_config() {
    PlannerConfig cfg;
    cfg.mode = LocalityMode::Combined;
    cfg.persisting_budget_fraction = 0.5;
    cfg.hit_ratio = 0.8;
    return cfg;
}

template <class Fn> static bool throws_invalid_argument(Fn fn) {
    try { fn(); } catch (const std::invalid_argument&) { return true; } catch (...) { return false; }
    return false;
}

static void test_set_aside_and_modes() {
    LocalityPlanner p(blackwell_like(), base_config());
    CHECK(p.recommended_l2_set_aside() == 32 * MiB);

    const auto combined = p.plan_for_layer(3 * MiB, true);
    CHECK(combined.use_persisting_window);
    CHECK(combined.prefetch_next);
    CHECK(combined.hot_window_bytes == 3 * MiB);
    CHECK_NEAR(combined.hit_ratio, 0.8);

    PlannerConfig cfg = base_config();
    cfg.mode = LocalityMode::Baseline;
    const auto baseline = LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true);
    CHECK(!baseline.use_persisting_window);
    CHECK(!baseline.prefetch_next);
    CHECK(baseline.hot_window_bytes == 0);

    cfg.mode = LocalityMode::Persist;
    const auto persist = LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true);
    CHECK(persist.use_persisting_window);
    CHECK(!persist.prefetch_next);

    cfg.mode = LocalityMode::Prefetch;
    const auto prefetch = LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true);
    CHECK(!prefetch.use_persisting_window);
    CHECK(prefetch.prefetch_next);
}

static void test_window_limits() {
    PlannerConfig cfg = base_config();
    cfg.max_hot_window_bytes = 1 * MiB;
    CHECK(LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true).hot_window_bytes == 1 * MiB);

    // The device access-policy window is a hard ceiling on the request.
    DeviceCaps narrow = blackwell_like();
    narrow.access_policy_max_window_bytes = 1 * MiB;
    CHECK(LocalityPlanner(narrow, base_config()).plan_for_layer(3 * MiB, true).hot_window_bytes == 1 * MiB);

    // A zero-byte state is not a hot set.
    const auto empty = LocalityPlanner(blackwell_like(), base_config()).plan_for_layer(0, true);
    CHECK(!empty.use_persisting_window);
    CHECK(empty.hot_window_bytes == 0);
}

static void test_device_capability_edges() {
    // Device without persisting-L2 support: no set-aside can be reserved, so no
    // persisting window may be planned, but layer-ahead prefetch is unaffected.
    DeviceCaps no_persist = blackwell_like();
    no_persist.persisting_l2_max_bytes = 0;
    LocalityPlanner p(no_persist, base_config());
    CHECK(p.recommended_l2_set_aside() == 0);
    const auto plan = p.plan_for_layer(3 * MiB, true);
    CHECK(!plan.use_persisting_window);
    CHECK(plan.hot_window_bytes == 0);
    CHECK(plan.prefetch_next);

    // Device reporting no access-policy window support.
    DeviceCaps no_window = blackwell_like();
    no_window.access_policy_max_window_bytes = 0;
    CHECK(!LocalityPlanner(no_window, base_config()).plan_for_layer(3 * MiB, true).use_persisting_window);

    // A zero budget fraction opts out of persisting behaviour entirely.
    PlannerConfig off = base_config();
    off.persisting_budget_fraction = 0.0;
    LocalityPlanner po(blackwell_like(), off);
    CHECK(po.recommended_l2_set_aside() == 0);
    CHECK(!po.plan_for_layer(3 * MiB, true).use_persisting_window);

    // A full budget fraction may not exceed what the device reports.
    PlannerConfig full = base_config();
    full.persisting_budget_fraction = 1.0;
    CHECK(LocalityPlanner(blackwell_like(), full).recommended_l2_set_aside() == 64 * MiB);
}

static void test_validate_reports_without_throwing() {
    // An integrating runtime may be built without exceptions, so a bad config has to be
    // detectable before anything is constructed.
    CHECK(validate(base_config()) == nullptr);

    PlannerConfig c = base_config(); c.hit_ratio = 1.1;
    CHECK(validate(c) != nullptr);
    c = base_config(); c.hit_ratio = -0.1;                  CHECK(validate(c) != nullptr);
    c = base_config(); c.persisting_budget_fraction = 1.5;  CHECK(validate(c) != nullptr);
    c = base_config(); c.persisting_budget_fraction = -0.5; CHECK(validate(c) != nullptr);
    c = base_config(); c.prefetch_distance = kMaxPrefetchDistance + 1; CHECK(validate(c) != nullptr);
    c = base_config(); c.prefetch_distance = -1;            CHECK(validate(c) != nullptr);

    // Boundaries are valid, not rejected.
    c = base_config(); c.hit_ratio = 0.0;                   CHECK(validate(c) == nullptr);
    c = base_config(); c.hit_ratio = 1.0;                   CHECK(validate(c) == nullptr);
    c = base_config(); c.persisting_budget_fraction = 0.0;  CHECK(validate(c) == nullptr);
    c = base_config(); c.persisting_budget_fraction = 1.0;  CHECK(validate(c) == nullptr);
    c = base_config(); c.prefetch_distance = 0;             CHECK(validate(c) == nullptr);
    c = base_config(); c.prefetch_distance = kMaxPrefetchDistance; CHECK(validate(c) == nullptr);
}

static void test_prefetch_distance_is_an_open_axis() {
    // Distance is a contributor-tunable axis, not a fixed 0/1 switch: the plan carries the
    // distance so the runtime knows which layer's state to hand over.
    for (int d = 1; d <= kMaxPrefetchDistance; ++d) {
        PlannerConfig cfg = base_config();
        cfg.prefetch_distance = d;
        const auto plan = LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true);
        CHECK(plan.prefetch_next);
        CHECK(plan.prefetch_distance == d);
    }
    PlannerConfig off = base_config();
    off.prefetch_distance = 0;
    const auto none = LocalityPlanner(blackwell_like(), off).plan_for_layer(3 * MiB, true);
    CHECK(!none.prefetch_next);
    CHECK(none.prefetch_distance == 0);
}

static void test_pre_touch_strategy_round_trips() {
    const PreTouchStrategy all[] = {PreTouchStrategy::Scalar, PreTouchStrategy::Vec4,
                                    PreTouchStrategy::Vec4Ldcg, PreTouchStrategy::PtxL2,
                                    PreTouchStrategy::WarpTile, PreTouchStrategy::Partial};
    for (auto s : all) {
        CHECK(parse_pre_touch_strategy(to_string(s)) == s);
        PlannerConfig cfg = base_config();
        cfg.pre_touch = s;
        // The chosen strategy has to reach the plan, or a contributor's kernel is selected
        // in config and silently ignored at the call site.
        CHECK(LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true).pre_touch == s);
    }
    CHECK(throws_invalid_argument([] { parse_pre_touch_strategy("vec8"); }));
    CHECK(throws_invalid_argument([] { parse_pre_touch_strategy(nullptr); }));
}

static void test_version_is_consistent() {
    CHECK(std::strcmp(version_string(), RECURLOCAL_VERSION_STRING) == 0);
    CHECK(version_number() == RECURLOCAL_VERSION_NUMBER);
    CHECK(version_number() > 0);
}

static void test_default_constructed_planner_is_inert() {
    // Default construction exists so a controller can hold one before it knows the
    // device. With no caps it must plan nothing rather than guess.
    LocalityPlanner p;
    CHECK(p.recommended_l2_set_aside() == 0);
    const auto plan = p.plan_for_layer(3 * MiB, true);
    CHECK(!plan.use_persisting_window);
    CHECK(plan.hot_window_bytes == 0);
}

static void test_concurrent_hot_set() {
    LocalityPlanner p(blackwell_like(), base_config());  // 32 MiB budget, hit_ratio 0.8

    // Hot set inside the budget keeps the full requested hit ratio.
    CHECK_NEAR(p.plan_for_layer(3 * MiB, true, 8 * MiB).hit_ratio, 0.8);

    // Exactly at the budget is still not oversubscribed.
    CHECK_NEAR(p.plan_for_layer(2 * MiB, true, 30 * MiB).hit_ratio, 0.8);

    // Oversubscribed: back off rather than pretend everything stays resident, and say so.
    const auto over = p.plan_for_layer(3 * MiB, true, 61 * MiB);
    CHECK(over.use_persisting_window);
    CHECK(over.hit_ratio_reduced);
    CHECK_NEAR(over.hit_ratio, 0.8 * (32.0 / 64.0));
    CHECK(!p.plan_for_layer(3 * MiB, true, 8 * MiB).hit_ratio_reduced);

    // Backing off is monotonic in the amount of contention.
    double previous = 1.0;
    for (std::size_t other = 0; other <= 256; other += 32) {
        const double ratio = p.plan_for_layer(3 * MiB, true, other * MiB).hit_ratio;
        CHECK(ratio <= previous + 1e-12);
        CHECK(ratio >= 0.05 - 1e-12);
        previous = ratio;
    }

    // Severe oversubscription is floored, never driven to zero.
    CHECK_NEAR(p.plan_for_layer(3 * MiB, true, 100000 * MiB).hit_ratio, 0.05);
}

static void test_prefetch_gating() {
    PlannerConfig cfg = base_config();
    CHECK(!LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, false).prefetch_next);

    cfg.prefetch_distance = 0;
    CHECK(!LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true).prefetch_next);
}

static void test_determinism() {
    LocalityPlanner p(blackwell_like(), base_config());
    const auto first = p.plan_for_layer(3 * MiB, true, 40 * MiB);
    for (int i = 0; i < 64; ++i) {
        const auto again = p.plan_for_layer(3 * MiB, true, 40 * MiB);
        CHECK(again.hot_window_bytes == first.hot_window_bytes);
        CHECK(again.use_persisting_window == first.use_persisting_window);
        CHECK(again.prefetch_next == first.prefetch_next);
        CHECK_NEAR(again.hit_ratio, first.hit_ratio);
    }
}

static void test_config_validation() {
    const auto caps = blackwell_like();
    CHECK(throws_invalid_argument([&] { PlannerConfig c = base_config(); c.hit_ratio = 1.1; LocalityPlanner(caps, c); }));
    CHECK(throws_invalid_argument([&] { PlannerConfig c = base_config(); c.hit_ratio = -0.1; LocalityPlanner(caps, c); }));
    CHECK(throws_invalid_argument([&] { PlannerConfig c = base_config(); c.persisting_budget_fraction = 1.5; LocalityPlanner(caps, c); }));
    CHECK(throws_invalid_argument([&] { PlannerConfig c = base_config(); c.persisting_budget_fraction = -0.5; LocalityPlanner(caps, c); }));
    CHECK(throws_invalid_argument([&] { PlannerConfig c = base_config(); c.prefetch_distance = kMaxPrefetchDistance + 1; LocalityPlanner(caps, c); }));
    CHECK(throws_invalid_argument([&] { PlannerConfig c = base_config(); c.prefetch_distance = -1; LocalityPlanner(caps, c); }));
}

static void test_hot_set_policies_differ_under_pressure() {
    // 32 MiB budget, hit_ratio 0.8. A 3 MiB window against 61 MiB of other hot state is
    // half-subscribed, so every policy sees the same pressure and must react differently.
    auto plan_with = [](HotSetPolicy policy) {
        PlannerConfig cfg = base_config();
        cfg.hot_set_policy = policy;
        return LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true, 61 * MiB);
    };

    // Every policy must agree that the hot set is oversubscribed; they differ only in
    // what they do about it. Telemetry stays comparable across policies that way.
    for (auto policy : {HotSetPolicy::Proportional, HotSetPolicy::Fixed,
                        HotSetPolicy::Sqrt, HotSetPolicy::Cliff}) {
        CHECK(plan_with(policy).hot_set_oversubscribed);
    }

    const auto fixed = plan_with(HotSetPolicy::Fixed);
    CHECK(fixed.use_persisting_window);
    CHECK(!fixed.hit_ratio_reduced);
    CHECK_NEAR(fixed.hit_ratio, 0.8);            // the naive control: asks for everything

    const auto proportional = plan_with(HotSetPolicy::Proportional);
    CHECK_NEAR(proportional.hit_ratio, 0.8 * 0.5);

    const auto sqrt_policy = plan_with(HotSetPolicy::Sqrt);
    CHECK(sqrt_policy.hit_ratio > proportional.hit_ratio);   // gentler backoff
    CHECK(sqrt_policy.hit_ratio < fixed.hit_ratio);

    const auto cliff = plan_with(HotSetPolicy::Cliff);
    CHECK(!cliff.use_persisting_window);          // refuses to thrash a shared cache
    CHECK(cliff.hot_window_bytes == 0);

    // Below the budget every policy behaves identically, so a policy change cannot
    // silently alter the uncontended case.
    for (auto policy : {HotSetPolicy::Proportional, HotSetPolicy::Fixed,
                        HotSetPolicy::Sqrt, HotSetPolicy::Cliff}) {
        PlannerConfig cfg = base_config();
        cfg.hot_set_policy = policy;
        const auto easy = LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true, 8 * MiB);
        CHECK(!easy.hot_set_oversubscribed);
        CHECK(easy.use_persisting_window);
        CHECK_NEAR(easy.hit_ratio, 0.8);
    }
}

static void test_min_hit_ratio_is_the_floor() {
    PlannerConfig cfg = base_config();
    cfg.min_hit_ratio = 0.25;
    const auto plan = LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true, 100000 * MiB);
    CHECK_NEAR(plan.hit_ratio, 0.25);

    CHECK(validate(cfg) == nullptr);

    // A floor above the requested ratio is clamped, not rejected: hit_ratio = 0 means
    // "do not persist", and the default floor must not turn that into a config error.
    PlannerConfig above = base_config();
    above.min_hit_ratio = 0.9;                       // above hit_ratio 0.8
    CHECK(validate(above) == nullptr);
    CHECK_NEAR(LocalityPlanner(blackwell_like(), above)
                   .plan_for_layer(3 * MiB, true, 100000 * MiB).hit_ratio, 0.8);

    PlannerConfig bad = base_config(); bad.min_hit_ratio = -0.1;
    CHECK(validate(bad) != nullptr);
}

static void test_policy_names_round_trip() {
    for (auto p : {HotSetPolicy::Proportional, HotSetPolicy::Fixed,
                   HotSetPolicy::Sqrt, HotSetPolicy::Cliff})
        CHECK(parse_hot_set_policy(to_string(p)) == p);
    CHECK(throws_invalid_argument([] { parse_hot_set_policy("greedy"); }));
    CHECK(throws_invalid_argument([] { parse_hot_set_policy(nullptr); }));
}

static void test_prefetch_schedules_vary_with_depth() {
    auto distances = [](PrefetchSchedule sched, int base) {
        PlannerConfig cfg = base_config();
        cfg.prefetch_schedule = sched;
        cfg.prefetch_distance = base;
        LocalityPlanner p(blackwell_like(), cfg);
        std::vector<int> out;
        for (int layer = 0; layer < 16; ++layer)
            out.push_back(p.plan_for_layer(3 * MiB, true, 0, layer).prefetch_distance);
        return out;
    };

    // Uniform is the v0.1 behaviour and must stay exactly that.
    for (int d : distances(PrefetchSchedule::Uniform, 4)) CHECK(d == 4);

    // Ramp reaches further with depth, never past the configured maximum.
    const auto ramp = distances(PrefetchSchedule::Ramp, 4);
    CHECK(ramp.front() < ramp.back());
    for (int d : ramp) CHECK(d >= 1 && d <= 4);

    // Alternating skips every other layer; Sparse skips three in four but goes deeper.
    const auto alt = distances(PrefetchSchedule::Alternating, 4);
    for (std::size_t i = 0; i < alt.size(); ++i) CHECK(alt[i] == (i % 2 == 0 ? 4 : 0));
    const auto sparse = distances(PrefetchSchedule::Sparse, 4);
    for (std::size_t i = 0; i < sparse.size(); ++i) CHECK(sparse[i] == (i % 4 == 0 ? 8 : 0));

    // A caller that cannot supply a layer index must still get defined, uniform behaviour
    // rather than silently landing on layer 0's entry in the schedule.
    PlannerConfig cfg = base_config();
    cfg.prefetch_schedule = PrefetchSchedule::Alternating;
    cfg.prefetch_distance = 4;
    LocalityPlanner p(blackwell_like(), cfg);
    for (int layer : {-1, -5}) CHECK(p.plan_for_layer(3 * MiB, true, 0, layer).prefetch_distance == 4);

    // A schedule never invents prefetching where the mode disables it.
    PlannerConfig off = cfg;
    off.mode = LocalityMode::Persist;
    LocalityPlanner po(blackwell_like(), off);
    for (int layer = 0; layer < 8; ++layer)
        CHECK(!po.plan_for_layer(3 * MiB, true, 0, layer).prefetch_next);
}

static void test_schedule_names_round_trip() {
    for (auto sch : {PrefetchSchedule::Uniform, PrefetchSchedule::Ramp,
                     PrefetchSchedule::Alternating, PrefetchSchedule::Sparse})
        CHECK(parse_prefetch_schedule(to_string(sch)) == sch);
    CHECK(throws_invalid_argument([] { parse_prefetch_schedule("adaptive"); }));
    CHECK(throws_invalid_argument([] { parse_prefetch_schedule(nullptr); }));
}

static void test_mode_names() {
    const LocalityMode modes[] = {LocalityMode::Baseline, LocalityMode::Persist,
                                  LocalityMode::Prefetch, LocalityMode::Combined};
    for (auto mode : modes) CHECK(parse_mode(to_string(mode)) == mode);
    CHECK(throws_invalid_argument([] { parse_mode("Combined"); }));
    CHECK(throws_invalid_argument([] { parse_mode(""); }));
    CHECK(throws_invalid_argument([] { parse_mode(nullptr); }));
}

// The bug this fixes, as a test: 48 recurrent layers of 3 MiB is 144 MiB of state that a
// single token revisits in full, and the v0.1 accounting called that "3 MiB" because it
// counted only the layer about to run. `persist` measured -11% at four concurrent sequences
// while the planner reported no oversubscription at all.
static void test_token_footprint_sees_what_current_layer_misses() {
    RecurrentGeometry geo;
    geo.recurrent_layers = 48;
    geo.bytes_per_layer = 3 * MiB;
    geo.sequences = 1;

    PlannerConfig cfg = base_config();
    cfg.hot_set_policy = HotSetPolicy::Fixed;   // isolate the accounting from the policy
    cfg.hot_set_model = HotSetModel::CurrentLayer;
    const auto old_model = LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true, geo);
    CHECK(!old_model.hot_set_oversubscribed);
    CHECK(old_model.hot_set_bytes == 3 * MiB);
    CHECK(old_model.hot_set_model == HotSetModel::CurrentLayer);

    cfg.hot_set_model = HotSetModel::TokenFootprint;
    const auto fixed = LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true, geo);
    CHECK(fixed.hot_set_oversubscribed);
    CHECK(fixed.hot_set_bytes == 144 * MiB);        // every layer, not one
    CHECK(fixed.hot_set_budget_bytes == 32 * MiB);
    CHECK(fixed.hot_set_model == HotSetModel::TokenFootprint);

    // And it scales with concurrency, which is the axis the old model was blind to.
    geo.sequences = 4;
    const auto batched = LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true, geo);
    CHECK(batched.hot_set_bytes == 4 * 144 * MiB);
}

// A footprint model reports an ABSOLUTE hot set, so widening the window must not inflate it:
// the window covers state the footprint already counted. Getting this wrong would make a
// whole-allocation window look like twice the pressure of a per-layer one on the same state.
static void test_footprint_does_not_double_count_a_widened_window() {
    RecurrentGeometry geo;
    geo.recurrent_layers = 48;
    geo.bytes_per_layer = 3 * MiB;
    PlannerConfig cfg = base_config();
    cfg.hot_set_model = HotSetModel::TokenFootprint;
    LocalityPlanner p(blackwell_like(), cfg);

    const auto slice = p.plan_for_layer(3 * MiB, true, geo);          // WindowScope::Layer
    const auto whole = p.plan_for_layer(144 * MiB, true, geo);        // WindowScope::Allocation
    CHECK(slice.hot_set_bytes == 144 * MiB);
    CHECK(whole.hot_set_bytes == 144 * MiB);
    CHECK_NEAR(slice.hit_ratio, whole.hit_ratio);

    // ...and where the window is the larger of the two, it is the hot set: a window is
    // bytes the policy is actively asking the cache to hold, whatever the model counted.
    RecurrentGeometry tiny;
    tiny.recurrent_layers = 2;
    tiny.bytes_per_layer = 1 * MiB;                       // 2 MiB footprint
    const auto window_dominates = p.plan_for_layer(8 * MiB, true, tiny);
    CHECK(window_dominates.hot_window_bytes == 8 * MiB);
    CHECK(window_dominates.hot_set_bytes == 8 * MiB);
    CHECK(!window_dominates.hot_set_oversubscribed);       // 8 MiB fits the 32 MiB set-aside
}

// ReuseWindow adds what passes through L2 between two visits to the same state. Recurrent
// state is a small fraction of a decode step's traffic, so leaving the weights out
// understates the pressure even when every recurrent layer is counted.
static void test_reuse_window_counts_streamed_traffic() {
    RecurrentGeometry geo;
    geo.recurrent_layers = 48;
    geo.bytes_per_layer = 3 * MiB;
    geo.streamed_bytes_per_token = 18000 * MiB / 1000;   // ~18 MiB of weights per token

    PlannerConfig cfg = base_config();
    cfg.hot_set_model = HotSetModel::TokenFootprint;
    const auto footprint = LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true, geo);
    cfg.hot_set_model = HotSetModel::ReuseWindow;
    const auto reuse = LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true, geo);
    CHECK(reuse.hot_set_bytes == footprint.hot_set_bytes + geo.streamed_bytes_per_token);
    CHECK(reuse.hit_ratio < footprint.hit_ratio);   // proportional back-off sees more pressure
}

// A runtime that cannot describe its own geometry must not be given a number invented for
// it. The planner falls back to the v0.1 accounting and reports which model it applied.
static void test_hot_set_model_degrades_visibly_without_geometry() {
    PlannerConfig cfg = base_config();
    cfg.hot_set_model = HotSetModel::TokenFootprint;
    const auto p = LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true, RecurrentGeometry{});
    CHECK(p.hot_set_model == HotSetModel::CurrentLayer);
    CHECK(p.hot_set_bytes == 3 * MiB);
}

// The geometry-aware overload must not change what CurrentLayer decides; it is the control
// the other models are measured against, so it has to stay the v0.1 answer exactly.
static void test_current_layer_model_matches_the_legacy_call() {
    RecurrentGeometry geo;
    geo.recurrent_layers = 48;
    geo.bytes_per_layer = 3 * MiB;
    PlannerConfig cfg = base_config();
    cfg.hot_set_model = HotSetModel::CurrentLayer;
    LocalityPlanner p(blackwell_like(), cfg);
    for (int layer = 0; layer < 8; ++layer) {
        const auto legacy = p.plan_for_layer(3 * MiB, true, 0, layer);
        const auto geometric = p.plan_for_layer(3 * MiB, true, geo, layer);
        CHECK(legacy.hot_window_bytes == geometric.hot_window_bytes);
        CHECK_NEAR(legacy.hit_ratio, geometric.hit_ratio);
        CHECK(legacy.prefetch_next == geometric.prefetch_next);
        CHECK(legacy.hot_set_oversubscribed == geometric.hot_set_oversubscribed);
    }
}

// Under the corrected accounting the hot set is far larger than the budget, so the policies
// must still be distinguishable there - otherwise fixing the model would collapse the
// hot-set surface into a single answer and there would be nothing left to compete on.
static void test_policies_still_separate_under_the_corrected_model() {
    RecurrentGeometry geo;
    geo.recurrent_layers = 48;
    geo.bytes_per_layer = 3 * MiB;
    PlannerConfig cfg = base_config();
    cfg.hot_set_model = HotSetModel::TokenFootprint;
    cfg.min_hit_ratio = 0.0;

    cfg.hot_set_policy = HotSetPolicy::Fixed;
    const auto fixed = LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true, geo);
    cfg.hot_set_policy = HotSetPolicy::Proportional;
    const auto prop = LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true, geo);
    cfg.hot_set_policy = HotSetPolicy::Sqrt;
    const auto sq = LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true, geo);
    cfg.hot_set_policy = HotSetPolicy::Cliff;
    const auto cliff = LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true, geo);

    CHECK(fixed.hit_ratio > sq.hit_ratio);
    CHECK(sq.hit_ratio > prop.hit_ratio);
    CHECK(prop.hit_ratio > 0.0);
    CHECK(!cliff.use_persisting_window);
}

static void test_new_axis_names_round_trip() {
    for (auto m : {HotSetModel::CurrentLayer, HotSetModel::TokenFootprint, HotSetModel::ReuseWindow})
        CHECK(parse_hot_set_model(to_string(m)) == m);
    for (auto sc : {WindowScope::Layer, WindowScope::Allocation, WindowScope::Ahead})
        CHECK(parse_window_scope(to_string(sc)) == sc);
    for (auto t : {WindowTarget::Matrix, WindowTarget::Conv, WindowTarget::Widest, WindowTarget::Narrowest})
        CHECK(parse_window_target(to_string(t)) == t);
    for (auto c : {PreTouchCoverage::Matrix, PreTouchCoverage::Conv, PreTouchCoverage::Both})
        CHECK(parse_pre_touch_coverage(to_string(c)) == c);
    CHECK(throws_invalid_argument([] { parse_hot_set_model("footprint"); }));
    CHECK(throws_invalid_argument([] { parse_window_scope(nullptr); }));
    CHECK(throws_invalid_argument([] { parse_window_target("both"); }));
    CHECK(throws_invalid_argument([] { parse_pre_touch_coverage("all"); }));
    for (auto j : {PrefetchJoin::PerLayer, PrefetchJoin::TokenEnd})
        CHECK(parse_prefetch_join(to_string(j)) == j);
    CHECK(throws_invalid_argument([] { parse_prefetch_join("deferred"); }));
}

// distance_for_layer is what a runtime asks before it can hand over state N layers ahead,
// so it has to agree with the distance the plan then reports for the same index.
static void test_distance_for_layer_agrees_with_the_plan() {
    PlannerConfig cfg = base_config();
    cfg.prefetch_distance = 4;
    for (auto sch : {PrefetchSchedule::Uniform, PrefetchSchedule::Ramp,
                     PrefetchSchedule::Alternating, PrefetchSchedule::Sparse}) {
        cfg.prefetch_schedule = sch;
        LocalityPlanner p(blackwell_like(), cfg);
        for (int layer = 0; layer < 12; ++layer) {
            const int d = p.distance_for_layer(layer);
            const auto plan = p.plan_for_layer(3 * MiB, true, 0, layer);
            CHECK(plan.prefetch_next == (d > 0));
            CHECK(plan.prefetch_distance == (d > 0 ? d : 0));
        }
    }
    // A mode that does not prefetch must report no distance, or a runtime would gather
    // state the controller then declines to touch.
    cfg.mode = LocalityMode::Persist;
    CHECK(LocalityPlanner(blackwell_like(), cfg).distance_for_layer(0) == 0);
}

// The plan must carry the axes forward, because the controller resolves them into pointers
// and a silently defaulted target would put the window on the wrong state.
static void test_plan_carries_the_window_axes() {
    PlannerConfig cfg = base_config();
    cfg.window_scope = WindowScope::Allocation;
    cfg.window_target = WindowTarget::Conv;
    cfg.pre_touch_coverage = PreTouchCoverage::Both;
    const auto p = LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true);
    CHECK(p.window_scope == WindowScope::Allocation);
    CHECK(p.window_target == WindowTarget::Conv);
    CHECK(p.pre_touch_coverage == PreTouchCoverage::Both);
}

// A hybrid layer's two states, laid out the way a runtime holds them: one allocation per
// state, indexed by absolute layer.
static void fill_segments(StateSegment out[2], unsigned char* state_base, unsigned char* conv_base,
                          int layer, int n_layers) {
    const std::size_t state_stride = 3 * MiB, conv_stride = 60 * 1024;
    out[0] = StateSegment{state_base + layer * state_stride, state_stride,
                          state_base, n_layers * state_stride, StateKind::Matrix};
    out[1] = StateSegment{conv_base + layer * conv_stride, conv_stride,
                          conv_base, n_layers * conv_stride, StateKind::Conv};
}

static void test_window_target_picks_the_right_state() {
    auto* state_base = reinterpret_cast<unsigned char*>(0x10000000ull);
    auto* conv_base = reinterpret_cast<unsigned char*>(0x80000000ull);
    StateSegment segs[2];
    fill_segments(segs, state_base, conv_base, 5, 64);

    CHECK(select_window_segment(segs, 2, WindowTarget::Matrix)->kind == StateKind::Matrix);
    CHECK(select_window_segment(segs, 2, WindowTarget::Conv)->kind == StateKind::Conv);
    CHECK(select_window_segment(segs, 2, WindowTarget::Widest)->kind == StateKind::Matrix);
    CHECK(select_window_segment(segs, 2, WindowTarget::Narrowest)->kind == StateKind::Conv);
    CHECK(select_window_segment(nullptr, 0, WindowTarget::Matrix) == nullptr);

    // A model that carries no convolution state must still get a window, not nothing.
    StateSegment only_matrix[1] = {segs[0]};
    CHECK(select_window_segment(only_matrix, 1, WindowTarget::Conv) == only_matrix);

    // A zero-sized segment is not a candidate; asking for it must not hand back a window
    // over a null pointer.
    StateSegment empty[1] = {StateSegment{nullptr, 0, nullptr, 0, StateKind::Matrix}};
    CHECK(select_window_segment(empty, 1, WindowTarget::Widest) == nullptr);
}

static void test_window_scope_never_leaves_the_allocation() {
    auto* state_base = reinterpret_cast<unsigned char*>(0x10000000ull);
    auto* conv_base = reinterpret_cast<unsigned char*>(0x80000000ull);
    const int n_layers = 64;
    StateSegment segs[2];

    fill_segments(segs, state_base, conv_base, 5, n_layers);
    CHECK(resolve_window_region(segs[0], WindowScope::Layer, 4).bytes == 3 * MiB);
    CHECK(resolve_window_region(segs[0], WindowScope::Layer, 4).ptr == segs[0].ptr);

    const auto whole = resolve_window_region(segs[0], WindowScope::Allocation, 4);
    CHECK(whole.ptr == state_base);
    CHECK(whole.bytes == (std::size_t)n_layers * 3 * MiB);

    // Ahead reaches distance+1 slices...
    CHECK(resolve_window_region(segs[0], WindowScope::Ahead, 3).bytes == 4 * 3 * MiB);
    CHECK(resolve_window_region(segs[0], WindowScope::Ahead, 0).bytes == 3 * MiB);

    // ...but never past the end of the allocation. The last layer has nothing ahead of it,
    // and a window that ran on would cover memory the runtime does not own.
    fill_segments(segs, state_base, conv_base, n_layers - 1, n_layers);
    CHECK(resolve_window_region(segs[0], WindowScope::Ahead, 8).bytes == 3 * MiB);
    fill_segments(segs, state_base, conv_base, n_layers - 3, n_layers);
    CHECK(resolve_window_region(segs[0], WindowScope::Ahead, 8).bytes == 3 * 3 * MiB);

    // A caller that cannot describe the allocation gets the slice, not a guess - for BOTH
    // widening scopes. Ahead used to extrapolate to (distance+1) slices here, hinting memory
    // the caller never said it owned; the slice is the only extent we actually know.
    StateSegment bare{segs[0].ptr, segs[0].bytes, nullptr, 0, StateKind::Matrix};
    CHECK(resolve_window_region(bare, WindowScope::Allocation, 4).bytes == segs[0].bytes);
    CHECK(resolve_window_region(bare, WindowScope::Ahead, 4).bytes == segs[0].bytes);
}

static void test_pre_touch_coverage_selects_states() {
    CHECK(pre_touch_covers(PreTouchCoverage::Matrix, StateKind::Matrix));
    CHECK(!pre_touch_covers(PreTouchCoverage::Matrix, StateKind::Conv));
    CHECK(pre_touch_covers(PreTouchCoverage::Conv, StateKind::Conv));
    CHECK(!pre_touch_covers(PreTouchCoverage::Conv, StateKind::Matrix));
    CHECK(pre_touch_covers(PreTouchCoverage::Both, StateKind::Matrix));
    CHECK(pre_touch_covers(PreTouchCoverage::Both, StateKind::Conv));
    // An unclassified state is recurrent state too: the default must be to touch it, not
    // to silently leave a model's third buffer cold.
    CHECK(pre_touch_covers(PreTouchCoverage::Matrix, StateKind::Other));
    CHECK(pre_touch_covers(PreTouchCoverage::Both, StateKind::Other));
}

// Qwen3.5/3.8: three recurrent layers then one full-attention layer, 64 layers, interval 4.
// If this walk is wrong the pre-touch warms a buffer the next recurrent layer will not read,
// and nothing anywhere reports it - the run is still correct, just useless.
static void test_recurrent_layer_walk() {
    CHECK(is_recurrent_layer(0, 4));
    CHECK(is_recurrent_layer(1, 4));
    CHECK(is_recurrent_layer(2, 4));
    CHECK(!is_recurrent_layer(3, 4));      // every 4th layer is full attention
    CHECK(is_recurrent_layer(4, 4));
    CHECK(!is_recurrent_layer(63, 4));
    int recurrent = 0;
    for (int i = 0; i < 64; ++i) if (is_recurrent_layer(i, 4)) ++recurrent;
    CHECK(recurrent == 48);

    // Distance is counted in RECURRENT layers, so it must step over the attention layer.
    CHECK(recurrent_layer_ahead(1, 1, 64, 4) == 2);
    CHECK(recurrent_layer_ahead(2, 1, 64, 4) == 4);      // 3 is full attention
    CHECK(recurrent_layer_ahead(2, 2, 64, 4) == 5);
    // From layer 0 the recurrent layers ahead are 1, 2, 4, 5, 6, 8, 9, 10, ...
    CHECK(recurrent_layer_ahead(0, 3, 64, 4) == 4);
    CHECK(recurrent_layer_ahead(0, 8, 64, 4) == 10);

    // The end of the stack, where there is nothing ahead to warm.
    CHECK(recurrent_layer_ahead(62, 1, 64, 4) == -1);
    CHECK(recurrent_layer_ahead(60, 4, 64, 4) == -1);
    CHECK(recurrent_layer_ahead(60, 2, 64, 4) == 62);

    // Degenerate inputs a runtime can actually produce.
    CHECK(recurrent_layer_ahead(0, 0, 64, 4) == -1);     // distance 0 is "do not prefetch"
    CHECK(recurrent_layer_ahead(0, 1, 0, 4) == -1);
    CHECK(recurrent_layer_ahead(0, 1, 64, 0) == 1);      // no interval: every layer recurrent
    CHECK(!is_recurrent_layer(-1, 4));
}

// The driver does not promise to honour a set-aside request: an RTX 5090 returns 18 MiB for
// a 15 MiB ask, and starts from a non-zero default. Budgeting against the request rather than
// the grant makes every oversubscription decision wrong by that rounding.
static void test_budget_follows_the_granted_set_aside_not_the_request() {
    PlannerConfig cfg = base_config();
    cfg.hot_set_policy = HotSetPolicy::Fixed;
    LocalityPlanner p(blackwell_like(), cfg);
    CHECK(p.recommended_l2_set_aside() == 32 * MiB);
    CHECK(p.granted_l2_set_aside() == 0);
    CHECK(p.effective_l2_budget() == 32 * MiB);      // nothing measured yet -> the request

    // A hot set of 40 MiB oversubscribes a 32 MiB request...
    CHECK(p.plan_for_layer(8 * MiB, true, 32 * MiB).hot_set_oversubscribed);
    // ...but NOT a device that actually granted 48 MiB.
    p.set_granted_l2_set_aside(48 * MiB);
    CHECK(p.effective_l2_budget() == 48 * MiB);
    CHECK(!p.plan_for_layer(8 * MiB, true, 32 * MiB).hot_set_oversubscribed);
    CHECK(p.plan_for_layer(8 * MiB, true, 64 * MiB).hot_set_oversubscribed);

    // And a device that granted LESS than asked must tighten, not keep the optimistic number.
    p.set_granted_l2_set_aside(16 * MiB);
    CHECK(p.plan_for_layer(8 * MiB, true, 16 * MiB).hot_set_oversubscribed);
}

// Telemetry that reads zero when the truth is 147 MiB is worse than no telemetry: a reader
// concludes nothing was competing for L2. The accounting describes the workload, so it must
// be reported whatever policy happens to be enabled.
static void test_hot_set_accounting_is_reported_in_every_mode() {
    RecurrentGeometry geo;
    geo.recurrent_layers = 48;
    geo.bytes_per_layer = 3 * MiB;
    for (auto mode : {LocalityMode::Baseline, LocalityMode::Persist,
                      LocalityMode::Prefetch, LocalityMode::Combined}) {
        PlannerConfig cfg = base_config();
        cfg.mode = mode;
        cfg.hot_set_model = HotSetModel::TokenFootprint;
        const auto p = LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true, geo);
        CHECK(p.hot_set_bytes == 144 * MiB);        // the workload, not the policy
        CHECK(p.hot_set_budget_bytes == 32 * MiB);
        CHECK(p.hot_set_oversubscribed);
    }
}

int main() {
    test_set_aside_and_modes();
    test_window_limits();
    test_device_capability_edges();
    test_validate_reports_without_throwing();
    test_version_is_consistent();
    test_default_constructed_planner_is_inert();
    test_concurrent_hot_set();
    test_prefetch_gating();
    test_determinism();
    test_config_validation();
    test_prefetch_distance_is_an_open_axis();
    test_pre_touch_strategy_round_trips();
    test_hot_set_policies_differ_under_pressure();
    test_min_hit_ratio_is_the_floor();
    test_policy_names_round_trip();
    test_prefetch_schedules_vary_with_depth();
    test_schedule_names_round_trip();
    test_mode_names();
    test_token_footprint_sees_what_current_layer_misses();
    test_reuse_window_counts_streamed_traffic();
    test_footprint_does_not_double_count_a_widened_window();
    test_hot_set_model_degrades_visibly_without_geometry();
    test_current_layer_model_matches_the_legacy_call();
    test_policies_still_separate_under_the_corrected_model();
    test_new_axis_names_round_trip();
    test_plan_carries_the_window_axes();
    test_distance_for_layer_agrees_with_the_plan();
    test_window_target_picks_the_right_state();
    test_window_scope_never_leaves_the_allocation();
    test_pre_touch_coverage_selects_states();
    test_recurrent_layer_walk();
    test_budget_follows_the_granted_set_aside_not_the_request();
    test_hot_set_accounting_is_reported_in_every_mode();

    if (g_failures) { std::cout << g_failures << " planner check(s) failed\n"; return 1; }
    std::cout << "planner tests passed\n";
    return 0;
}
