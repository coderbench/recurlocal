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
                        HotSetPolicy::Sqrt, HotSetPolicy::Cliff, HotSetPolicy::Quota}) {
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
                        HotSetPolicy::Sqrt, HotSetPolicy::Cliff, HotSetPolicy::Quota}) {
        PlannerConfig cfg = base_config();
        cfg.hot_set_policy = policy;
        const auto easy = LocalityPlanner(blackwell_like(), cfg).plan_for_layer(3 * MiB, true, 8 * MiB);
        CHECK(!easy.hot_set_oversubscribed);
        CHECK(easy.use_persisting_window);
        CHECK_NEAR(easy.hit_ratio, 0.8);
    }
}

static void test_quota_admits_whole_layers_instead_of_shaving_every_hit_ratio() {
    // The regime this policy exists for: a footprint a little larger than the set-aside.
    // 32 MiB budget, a 2 MiB window per layer, 30 layers = 60 MiB declared hot. Every other
    // policy answers by asking all 30 layers for a reduced hit ratio; quota keeps 16 of them
    // whole and declines the rest, because a cache line is resident or it is not.
    auto plan_at = [](HotSetPolicy policy, int layer) {
        PlannerConfig cfg = base_config();
        cfg.hot_set_policy = policy;
        // additive accounting: 58 MiB declared beside this layer's own 2 MiB window is a
        // 60 MiB hot set against a 32 MiB budget.
        return LocalityPlanner(blackwell_like(), cfg)
                   .plan_for_layer(2 * MiB, true, 58 * MiB, layer);
    };

    int admitted = 0;
    for (int layer = 0; layer < 30; ++layer) {
        const auto plan = plan_at(HotSetPolicy::Quota, layer);
        if (plan.use_persisting_window) {
            ++admitted;
            // An admitted layer gets the FULL hit ratio. That is the whole point: the budget
            // is spent on fewer layers rather than diluted across all of them.
            CHECK_NEAR(plan.hit_ratio, 0.8);
            CHECK(plan.hot_window_bytes == 2 * MiB);
            CHECK(!plan.hit_ratio_reduced);
        } else {
            CHECK(plan.hot_window_bytes == 0);
            CHECK(plan.hit_ratio_reduced);
        }
    }
    // 32 MiB of budget holds 16 whole 2 MiB windows, and quota admits exactly that many --
    // never more (which would thrash) and never fewer (which would waste the set-aside).
    CHECK(admitted == 16);

    // Proportional spends the same budget on every layer at a reduced ratio, so its total
    // requested residency is the same while no layer is whole. The two are genuinely
    // different requests, which is what makes this an A/B rather than a retuning.
    for (int layer = 0; layer < 30; ++layer) {
        const auto prop = plan_at(HotSetPolicy::Proportional, layer);
        CHECK(prop.use_persisting_window);
        CHECK(prop.hit_ratio < 0.8);
    }
}

static void test_quota_admits_the_same_layers_on_every_token() {
    // A window that moved between tokens would evict exactly the state it kept last time,
    // which is worse than installing none. The choice must be a pure function of the layer
    // ordinal.
    PlannerConfig cfg = base_config();
    cfg.hot_set_policy = HotSetPolicy::Quota;
    LocalityPlanner planner(blackwell_like(), cfg);
    for (int layer = 0; layer < 30; ++layer) {
        const bool first = planner.plan_for_layer(2 * MiB, true, 58 * MiB, layer)
                               .use_persisting_window;
        for (int token = 0; token < 4; ++token)
            CHECK(planner.plan_for_layer(2 * MiB, true, 58 * MiB, layer)
                      .use_persisting_window == first);
    }
}

static void test_quota_spends_a_fixed_budget_however_many_layers_want_it() {
    // The hot set counts every sequence, so four times the footprint against the same cache
    // must admit the same NUMBER of windows over four times as many units -- a quarter of the
    // fraction. A policy that ignored that would install four times the windows the device
    // can hold and be indistinguishable from `fixed`.
    auto admitted = [](std::size_t other, int units) {
        PlannerConfig cfg = base_config();
        cfg.hot_set_policy = HotSetPolicy::Quota;
        LocalityPlanner planner(blackwell_like(), cfg);
        int n = 0;
        for (int layer = 0; layer < units; ++layer)
            if (planner.plan_for_layer(2 * MiB, true, other, layer).use_persisting_window)
                ++n;
        return n;
    };
    const int one_seq  = admitted(58 * MiB, 30);    //  60 MiB hot, 30 units
    const int four_seq = admitted(238 * MiB, 120);  // 240 MiB hot, 120 units
    // 32 MiB of set-aside holds 16 whole 2 MiB windows either way.
    CHECK(one_seq == 16);
    CHECK(four_seq == 16);
    // So the admitted fraction fell from 16/30 to 16/120 as the footprint grew.
    CHECK(four_seq * 30 < one_seq * 120);
}

static void test_quota_is_inert_when_everything_fits(void) {
    // Under the budget there is nothing to ration, and quota must not decline a window that
    // the device could simply hold -- otherwise it would be a regression on every small model.
    PlannerConfig cfg = base_config();
    cfg.hot_set_policy = HotSetPolicy::Quota;
    LocalityPlanner planner(blackwell_like(), cfg);
    for (int layer = 0; layer < 8; ++layer) {
        const auto plan = planner.plan_for_layer(2 * MiB, true, 14 * MiB, layer);
        CHECK(plan.use_persisting_window);
        CHECK_NEAR(plan.hit_ratio, 0.8);
        CHECK(!plan.hot_set_oversubscribed);
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
                   HotSetPolicy::Sqrt, HotSetPolicy::Cliff, HotSetPolicy::Quota})
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


// ---------------------------------------------------------------------------------------
// SetAsidePolicy: how large a set-aside to ask for once the workload is known.
// ---------------------------------------------------------------------------------------

// The two checkpoints this repository has measured, as geometry. Both formulas are the
// adapter's own (recurlocal_sparkinfer.cpp) and both reproduce the published footprints.
static RecurrentGeometry moe_geometry(int sequences) {
    RecurrentGeometry g;
    g.recurrent_layers = 30;
    g.bytes_per_layer = 2146304;   // 32*128*128*4 + 3*8192*2, per configs/qwen3.6-...json
    g.sequences = sequences;
    return g;
}
static RecurrentGeometry dense_geometry(int sequences) {
    RecurrentGeometry g;
    g.recurrent_layers = 48;
    g.bytes_per_layer = 3 * MiB + 60 * 1024;
    g.sequences = sequences;
    return g;
}
static DeviceCaps rtx5090() { return DeviceCaps{96 * MiB, 60 * MiB, 128 * MiB}; }

static void test_fixed_set_aside_is_the_shipped_behaviour_whatever_the_workload() {
    // The control has to be exactly what every number in this repository was measured under,
    // or an A/B against it measures the change AND a shifted baseline.
    PlannerConfig cfg = base_config();
    cfg.persisting_budget_fraction = 0.75;
    LocalityPlanner p(rtx5090(), cfg);
    CHECK(cfg.set_aside_policy == SetAsidePolicy::Fixed);   // and it is the default
    for (int seq : {1, 4, 16, 32}) {
        CHECK(p.recommended_l2_set_aside(moe_geometry(seq)) == p.recommended_l2_set_aside());
        CHECK(p.recommended_l2_set_aside(dense_geometry(seq)) == p.recommended_l2_set_aside());
    }
    CHECK(p.recommended_l2_set_aside() == 45 * MiB);
}

static void test_a_workload_aware_policy_with_no_workload_is_the_shipped_behaviour() {
    // A rule that cannot see the workload must not invent one. Every degenerate geometry
    // falls back to the constant fraction rather than to zero, which would silently turn
    // `persist` into `baseline` for any caller that never declared its geometry.
    for (auto pol : {SetAsidePolicy::FitFootprint, SetAsidePolicy::Residency}) {
        PlannerConfig cfg = base_config();
        cfg.set_aside_policy = pol;
        LocalityPlanner p(rtx5090(), cfg);
        const auto fixed = p.recommended_l2_set_aside();
        CHECK(p.recommended_l2_set_aside(RecurrentGeometry{}) == fixed);
        RecurrentGeometry g = moe_geometry(1);
        g.recurrent_layers = 0;   CHECK(p.recommended_l2_set_aside(g) == fixed);
        g = moe_geometry(1); g.bytes_per_layer = 0; CHECK(p.recommended_l2_set_aside(g) == fixed);
    }
}

static void test_fit_footprint_never_reserves_more_than_the_footprint_can_use() {
    // The unarguable half: reserving cache to hold bytes that do not exist takes capacity
    // from the stream and buys nothing. This is the regime a device with more persisting L2
    // than the model needs would be in, and nothing in the library had an opinion about it.
    PlannerConfig cfg = base_config();
    cfg.set_aside_policy = SetAsidePolicy::FitFootprint;
    RecurrentGeometry tiny = moe_geometry(1);
    tiny.recurrent_layers = 2;                       // 4.09 MiB of footprint
    const auto footprint = LocalityPlanner::token_footprint_bytes(tiny);
    LocalityPlanner p(rtx5090(), cfg);
    CHECK(p.recommended_l2_set_aside(tiny) == footprint);
    CHECK(p.recommended_l2_set_aside(tiny) < p.recommended_l2_set_aside());

    // ...and never more than the device will grant, whatever the footprint asks for.
    LocalityPlanner q(rtx5090(), cfg);
    CHECK(q.recommended_l2_set_aside(moe_geometry(32)) == 60 * MiB);
}

static void test_fit_footprint_reaches_the_setting_the_measurement_calls_best() {
    // The defect this policy exists for: at batch 1 on the sparse-MoE checkpoint the measured
    // optimum is the FULL 60 MiB (+1.63%), and the shipped constant asks for 45 MiB (+1.28%).
    // A policy that still multiplied by persisting_budget_fraction could not get there without
    // the caller ALSO changing the constant it was introduced to replace.
    PlannerConfig cfg = base_config();
    cfg.set_aside_policy = SetAsidePolicy::FitFootprint;
    cfg.persisting_budget_fraction = 0.75;           // left at the shipped default on purpose
    LocalityPlanner p(rtx5090(), cfg);
    CHECK(p.recommended_l2_set_aside(moe_geometry(1)) == 60 * MiB);
    CHECK(p.recommended_l2_set_aside() == 45 * MiB); // the constant is untouched beside it
}

static void test_residency_declines_where_the_footprint_cannot_be_held() {
    // The measured facts this encodes: the persist family PAYS at a resident fraction of
    // 0.977 (MoE, batch 1) and does NOT at 0.478 (the same model, four sequences). The
    // default threshold sits between them, so the policy admits the first and declines the
    // second -- which is the whole content of the rule.
    PlannerConfig cfg = base_config();
    cfg.set_aside_policy = SetAsidePolicy::Residency;
    LocalityPlanner p(rtx5090(), cfg);
    CHECK(cfg.min_residency == 0.50);

    CHECK(p.achievable_residency(moe_geometry(1)) > 0.97);
    CHECK(p.recommended_l2_set_aside(moe_geometry(1)) == 60 * MiB);

    CHECK(p.achievable_residency(moe_geometry(4)) < 0.50);
    CHECK(p.recommended_l2_set_aside(moe_geometry(4)) == 0);
    CHECK(p.recommended_l2_set_aside(moe_geometry(16)) == 0);
    CHECK(p.recommended_l2_set_aside(moe_geometry(32)) == 0);
}

static void test_residency_gives_up_the_dense_models_small_real_gain_and_says_so() {
    // A falsifiable prediction, kept as a test so it cannot quietly stop being one: the dense
    // checkpoint at batch 1 sits at a resident fraction of 0.409 and measures a RESOLVED
    // +0.10%. The default threshold declines it. If a sweep of --axis min-residency shows the
    // crossover is below 0.409, this test is what has to change, deliberately.
    PlannerConfig cfg = base_config();
    cfg.set_aside_policy = SetAsidePolicy::Residency;
    LocalityPlanner p(rtx5090(), cfg);
    const double rho = p.achievable_residency(dense_geometry(1));
    CHECK(rho > 0.40 && rho < 0.42);
    CHECK(p.recommended_l2_set_aside(dense_geometry(1)) == 0);

    PlannerConfig lower = cfg;
    lower.min_residency = 0.40;                       // below the dense model's 0.409
    CHECK(LocalityPlanner(rtx5090(), lower).recommended_l2_set_aside(dense_geometry(1)) == 60 * MiB);
}

static void test_residency_thresholds_of_zero_and_one_are_the_two_corners() {
    PlannerConfig cfg = base_config();
    cfg.set_aside_policy = SetAsidePolicy::Residency;
    cfg.min_residency = 0.0;                          // admit anything: FitFootprint exactly
    LocalityPlanner none(rtx5090(), cfg);
    PlannerConfig fit = cfg; fit.set_aside_policy = SetAsidePolicy::FitFootprint;
    LocalityPlanner fitp(rtx5090(), fit);
    for (int seq : {1, 4, 32})
        CHECK(none.recommended_l2_set_aside(moe_geometry(seq))
              == fitp.recommended_l2_set_aside(moe_geometry(seq)));

    cfg.min_residency = 1.0;                          // only a footprint that fits whole
    LocalityPlanner all(rtx5090(), cfg);
    CHECK(all.recommended_l2_set_aside(moe_geometry(1)) == 0);   // 0.977 is not 1.0
    RecurrentGeometry small = moe_geometry(1);
    small.recurrent_layers = 4;                                  // 8.2 MiB, fits whole
    CHECK(all.recommended_l2_set_aside(small) == LocalityPlanner::token_footprint_bytes(small));
}

static void test_the_footprint_saturates_instead_of_wrapping() {
    // A hot set that wraps to a small number is worse than no accounting at all: it reports
    // that everything fits and applies a policy sized for a workload that does not exist.
    RecurrentGeometry g;
    g.recurrent_layers = 48;
    g.bytes_per_layer = static_cast<std::size_t>(-1) / 2;
    g.sequences = 1024;
    CHECK(LocalityPlanner::token_footprint_bytes(g) == static_cast<std::size_t>(-1));

    PlannerConfig cfg = base_config();
    cfg.set_aside_policy = SetAsidePolicy::Residency;
    LocalityPlanner p(rtx5090(), cfg);
    CHECK(p.achievable_residency(g) > 0.0);
    CHECK(p.achievable_residency(g) < 1e-9);
    CHECK(p.recommended_l2_set_aside(g) == 0);
}

static void test_a_negative_sequence_count_counts_as_one() {
    // The runtime declares this; a runtime that declares nonsense must get defined behaviour
    // rather than a footprint of zero, which would report that everything fits.
    RecurrentGeometry g = moe_geometry(-5);
    CHECK(LocalityPlanner::token_footprint_bytes(g)
          == LocalityPlanner::token_footprint_bytes(moe_geometry(1)));
    CHECK(LocalityPlanner::token_footprint_bytes(moe_geometry(0))
          == LocalityPlanner::token_footprint_bytes(moe_geometry(1)));
}

static void test_set_aside_policy_names_round_trip() {
    for (auto p : {SetAsidePolicy::Fixed, SetAsidePolicy::FitFootprint, SetAsidePolicy::Residency})
        CHECK(parse_set_aside_policy(to_string(p)) == p);
    CHECK(std::strcmp(to_string(SetAsidePolicy::FitFootprint), "fit_footprint") == 0);
    bool threw = false;
    try { parse_set_aside_policy("adaptive"); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

static void test_window_attach_names_round_trip_including_strict() {
    for (auto a : {WindowAttach::Stream, WindowAttach::CaptureNode, WindowAttach::CaptureNodeStrict})
        CHECK(parse_window_attach(to_string(a)) == a);
    CHECK(std::strcmp(to_string(WindowAttach::CaptureNodeStrict), "capture_node_strict") == 0);
    // Appended, never inserted: a consumer that persisted the integer value of CaptureNode
    // must still get CaptureNode back. This is the whole ABI promise for these enums.
    CHECK(static_cast<int>(WindowAttach::Stream) == 0);
    CHECK(static_cast<int>(WindowAttach::CaptureNode) == 1);
    CHECK(static_cast<int>(WindowAttach::CaptureNodeStrict) == 2);
}

static void test_set_aside_policy_values_are_appended_not_inserted() {
    CHECK(static_cast<int>(SetAsidePolicy::Fixed) == 0);
    CHECK(static_cast<int>(SetAsidePolicy::FitFootprint) == 1);
    CHECK(static_cast<int>(SetAsidePolicy::Residency) == 2);
    // Fixed is 0 so a zero-initialised PlannerConfig is the shipped behaviour.
    CHECK(PlannerConfig{}.set_aside_policy == SetAsidePolicy::Fixed);
}

static void test_the_stability_contract_holds_where_the_code_can_check_it() {
    // docs/STABILITY.md is a promise about symbols other people switch on. These are the
    // parts of it a test can hold to: every enumerator's numeric value is pinned by being
    // appended and never inserted, every one round-trips through its string, and no
    // to_string returns a real enumerator's name for a value outside its enumeration.
    CHECK(static_cast<int>(LocalityMode::Baseline) == 0);
    CHECK(static_cast<int>(HotSetPolicy::Proportional) == 0);
    CHECK(static_cast<int>(HotSetPolicy::Quota) == 4);         // appended, not inserted
    CHECK(static_cast<int>(HotSetModel::CurrentLayer) == 0);
    CHECK(static_cast<int>(PreTouchStrategy::Scalar) == 0);
    CHECK(static_cast<int>(WindowScope::Layer) == 0);
    CHECK(static_cast<int>(WindowTarget::Matrix) == 0);
    CHECK(static_cast<int>(PrefetchJoin::PerLayer) == 0);
    CHECK(static_cast<int>(StateKind::Matrix) == 0);
    // Scoped enums have a fixed underlying type, so appending can never move sizeof - which
    // is what makes "a new mechanism is a new enumerator" cost a consumer nothing.
    CHECK(sizeof(LocalityMode) == sizeof(int));
    CHECK(sizeof(SetAsidePolicy) == sizeof(int));
    CHECK(sizeof(WindowAttach) == sizeof(int));

    // An out-of-range value must never be labelled as a real enumerator. to_string(SetAsidePolicy)
    // returned "fixed" - the CONTROL arm's own name - which would put the control's label on a
    // candidate nobody can identify.
    CHECK(std::strcmp(to_string(static_cast<SetAsidePolicy>(99)), "unknown") == 0);
    CHECK(std::strcmp(to_string(static_cast<WindowAttach>(99)), "unknown") == 0);

    // A default-constructed config is the shipped behaviour on every axis, so a consumer that
    // sets one field does not silently opt into four others.
    const PlannerConfig d{};
    CHECK(d.set_aside_policy == SetAsidePolicy::Fixed);
    CHECK(d.hot_set_policy == HotSetPolicy::Proportional);
    CHECK(d.window_attach == WindowAttach::Stream);
    CHECK(d.hot_set_model == HotSetModel::CurrentLayer);
}

static void test_min_residency_is_validated() {
    PlannerConfig cfg = base_config();
    cfg.min_residency = 1.5;
    CHECK(validate(cfg) != nullptr);
    cfg.min_residency = -0.1;
    CHECK(validate(cfg) != nullptr);
    CHECK(throws_invalid_argument([&] { LocalityPlanner(rtx5090(), cfg); }));
    cfg.min_residency = 0.5;
    CHECK(validate(cfg) == nullptr);
}

// ---------------------------------------------------------------------------------------
// Device-capability matrix. RecurLocal has only ever run on one device; DeviceCaps is
// queried rather than hardcoded, but nothing checked what the planner does when the numbers
// come back different. These fabricate the caps and need no GPU.
// ---------------------------------------------------------------------------------------

static void test_a_device_with_no_persisting_l2_asks_for_no_window() {
    // Not an error. Some devices report zero, and a driver may refuse the limit outright.
    // The plan must simply not ask an integrating runtime to install a window it cannot back.
    for (auto pol : {SetAsidePolicy::Fixed, SetAsidePolicy::FitFootprint, SetAsidePolicy::Residency}) {
        PlannerConfig cfg = base_config();
        cfg.mode = LocalityMode::Persist;
        cfg.hot_set_model = HotSetModel::TokenFootprint;
        cfg.set_aside_policy = pol;
        LocalityPlanner p(DeviceCaps{40 * MiB, 0, 32 * MiB}, cfg);
        CHECK(p.recommended_l2_set_aside() == 0);
        CHECK(p.recommended_l2_set_aside(moe_geometry(1)) == 0);
        CHECK(p.achievable_residency(moe_geometry(1)) == 0.0);
        const auto plan = p.plan_for_layer(2 * MiB, true, moe_geometry(1), 0);
        CHECK(!plan.use_persisting_window);
        CHECK(plan.hot_window_bytes == 0);
    }
}

static void test_a_persisting_capacity_too_small_to_hold_one_window_is_declined() {
    // A device whose whole set-aside is smaller than one layer's state can hold nothing
    // across a token. Installing a window there is pure overhead, and the shipped
    // proportional policy would install one anyway at the floor hit ratio.
    PlannerConfig cfg = base_config();
    cfg.mode = LocalityMode::Persist;
    cfg.hot_set_model = HotSetModel::TokenFootprint;
    cfg.set_aside_policy = SetAsidePolicy::Residency;
    LocalityPlanner p(DeviceCaps{4 * MiB, 64 * 1024, 4 * MiB}, cfg);
    CHECK(p.achievable_residency(moe_geometry(1)) < 0.01);
    CHECK(p.recommended_l2_set_aside(moe_geometry(1)) == 0);
}

static void test_a_persisting_capacity_larger_than_the_footprint_reserves_only_what_is_needed() {
    // The opposite corner, and the one no measurement has ever reached: a device with more
    // persisting L2 than the model's whole recurrent footprint. Fixed reserves a fraction of
    // the DEVICE, so it over-reserves; the footprint-aware policies reserve the footprint.
    DeviceCaps big{512 * MiB, 256 * MiB, 128 * MiB};
    const auto footprint = LocalityPlanner::token_footprint_bytes(moe_geometry(1));

    PlannerConfig fixed = base_config();
    fixed.persisting_budget_fraction = 0.75;
    CHECK(LocalityPlanner(big, fixed).recommended_l2_set_aside(moe_geometry(1)) == 192 * MiB);

    for (auto pol : {SetAsidePolicy::FitFootprint, SetAsidePolicy::Residency}) {
        PlannerConfig cfg = fixed;
        cfg.set_aside_policy = pol;
        LocalityPlanner p(big, cfg);
        CHECK(p.achievable_residency(moe_geometry(1)) == 1.0);
        CHECK(p.recommended_l2_set_aside(moe_geometry(1)) == footprint);
        CHECK(p.recommended_l2_set_aside(moe_geometry(1)) < 192 * MiB);
    }
}

static void test_fabricated_datacentre_devices_behave_as_their_arithmetic_says() {
    // A100 (40 MiB L2, 30 MiB persisting) and H100 (50 MiB L2, 40 MiB persisting), against
    // the two checkpoints this repository has measured. Nothing here is a measurement; these
    // pin the POLICY the planner would apply, which is what a consumer switching on these
    // enums depends on.
    struct Dev { const char* name; DeviceCaps caps; };
    const Dev devs[] = {
        {"A100", DeviceCaps{40 * MiB, 30 * MiB, 128 * MiB}},
        {"H100", DeviceCaps{50 * MiB, 40 * MiB, 128 * MiB}},
    };
    PlannerConfig cfg = base_config();
    cfg.set_aside_policy = SetAsidePolicy::Residency;
    for (const auto& d : devs) {
        LocalityPlanner p(d.caps, cfg);
        // The batch-1 MoE footprint is 61.4 MiB, so the two devices land either side of the
        // threshold and the rule does not answer the same on both: the A100's 30 MiB holds
        // 0.489 of it and is declined, the H100's 40 MiB holds 0.651 and is admitted. That is
        // the point of a device-capability matrix - the surface this library found on an
        // RTX 5090 exists on one of these and not the other, and the arithmetic says which.
        const bool holds_enough = d.caps.persisting_l2_max_bytes * 2 >= 61 * MiB;
        CHECK((p.achievable_residency(moe_geometry(1)) >= 0.5) == holds_enough);
        CHECK((p.recommended_l2_set_aside(moe_geometry(1)) > 0) == holds_enough);
        // Concurrency multiplies the footprint, so both decline from four sequences on.
        for (int seq : {4, 16, 32}) {
            CHECK(p.achievable_residency(moe_geometry(seq)) < 0.5);
            CHECK(p.recommended_l2_set_aside(moe_geometry(seq)) == 0);
        }
        // FitFootprint still reserves the whole capacity there, which is the honest
        // difference between the two rules: one asks whether it can pay, the other does not.
        PlannerConfig fit = cfg;
        fit.set_aside_policy = SetAsidePolicy::FitFootprint;
        CHECK(LocalityPlanner(d.caps, fit).recommended_l2_set_aside(moe_geometry(1))
              == d.caps.persisting_l2_max_bytes);
    }
}

static void test_a_window_cap_smaller_than_the_state_still_clamps() {
    // accessPolicyMaxWindowSize is 128 MiB on an RTX 5090 and has never bound. On a device
    // where it does, the window must be the cap and not the state.
    PlannerConfig cfg = base_config();
    cfg.mode = LocalityMode::Persist;
    LocalityPlanner p(DeviceCaps{96 * MiB, 60 * MiB, 512 * 1024}, cfg);
    const auto plan = p.plan_for_layer(2 * MiB, true, moe_geometry(1), 0);
    CHECK(plan.hot_window_bytes == 512 * 1024);
    CHECK(plan.use_persisting_window);
}

static void test_a_device_that_allows_no_window_at_all_installs_none() {
    PlannerConfig cfg = base_config();
    cfg.mode = LocalityMode::Persist;
    LocalityPlanner p(DeviceCaps{96 * MiB, 60 * MiB, 0}, cfg);
    const auto plan = p.plan_for_layer(2 * MiB, true, moe_geometry(1), 0);
    CHECK(!plan.use_persisting_window);
    CHECK(plan.hot_window_bytes == 0);
    // ...and it still reports the accounting, so a null result can be read rather than guessed.
    CHECK(plan.hot_set_budget_bytes == p.effective_l2_budget());
    CHECK(plan.hot_set_bytes > 0);
}

static void test_quota_without_an_ordinal_says_it_fell_back_to_fixed() {
    // Quota rations by layer ordinal. A caller who supplies none cannot be rationed - and
    // used to get a full-hit-ratio window on every layer with hit_ratio_reduced=false, i.e.
    // HotSetPolicy::Fixed exactly, while the telemetry reported the policy it was asked for
    // and denied that anything had backed off. Selecting an enumerator has to either do
    // something or say that it did not.
    PlannerConfig cfg = base_config();
    cfg.mode = LocalityMode::Persist;
    cfg.hot_set_model = HotSetModel::TokenFootprint;
    cfg.hot_set_policy = HotSetPolicy::Quota;
    LocalityPlanner p(rtx5090(), cfg);
    const auto plan = p.plan_for_layer(2 * MiB, true, moe_geometry(32), -1);
    CHECK(plan.hot_set_policy == HotSetPolicy::Fixed);       // what it DID
    CHECK(plan.use_persisting_window);
    CHECK_NEAR(plan.hit_ratio, cfg.hit_ratio);

    // With an ordinal it rations, and says so.
    const auto rationed = p.plan_for_layer(2 * MiB, true, moe_geometry(32), 1);
    CHECK(rationed.hot_set_policy == HotSetPolicy::Quota);

    // Every other policy is ordinal-independent, so none of them ever degrades.
    for (auto pol : {HotSetPolicy::Proportional, HotSetPolicy::Fixed, HotSetPolicy::Sqrt,
                     HotSetPolicy::Cliff}) {
        PlannerConfig c = cfg;
        c.hot_set_policy = pol;
        CHECK(LocalityPlanner(rtx5090(), c)
                  .plan_for_layer(2 * MiB, true, moe_geometry(32), -1).hot_set_policy == pol);
    }
}

static void test_quota_admits_the_budget_once_not_once_per_period() {
    // The spread pattern selects `admitted` of `period` ordinals. Deriving the period from a
    // BYTE count instead of a layer count made it smaller than the number of ordinals the
    // runtime walks whenever the window spanned more than one layer's slice - which
    // WindowScope::Ahead produces today - and the pattern then repeated, admitting a multiple
    // of the budget. The regression is silent: every window still looks individually correct.
    PlannerConfig cfg = base_config();
    cfg.mode = LocalityMode::Persist;
    cfg.hot_set_model = HotSetModel::TokenFootprint;
    cfg.hot_set_policy = HotSetPolicy::Quota;
    LocalityPlanner p(rtx5090(), cfg);

    // A window eight layers wide against a 30-layer model: 8 byte-units, 30 ordinals.
    const std::size_t wide = 16 * MiB;
    std::size_t admitted_bytes = 0;
    for (int layer = 0; layer < 30; ++layer) {
        const auto plan = p.plan_for_layer(wide, true, moe_geometry(4), layer);
        if (plan.use_persisting_window) admitted_bytes += plan.hot_window_bytes;
    }
    CHECK(admitted_bytes <= p.effective_l2_budget());
}

static void test_quota_still_admits_when_the_hot_set_has_saturated() {
    // `hot` saturates to SIZE_MAX by design - "a wrong answer here must never be a small
    // one". The old unit count then computed `hot + window - 1`, which wrapped, making the
    // period zero and Quota decline every layer: the largest representable hot set producing
    // the smallest possible admission, the exact inversion the saturating arithmetic exists
    // to prevent, and Quota silently becoming Cliff.
    RecurrentGeometry huge;
    huge.recurrent_layers = 48;
    huge.bytes_per_layer = static_cast<std::size_t>(-1) / 8;
    huge.sequences = 64;
    CHECK(LocalityPlanner::token_footprint_bytes(huge) == static_cast<std::size_t>(-1));

    PlannerConfig cfg = base_config();
    cfg.mode = LocalityMode::Persist;
    cfg.hot_set_model = HotSetModel::TokenFootprint;
    cfg.hot_set_policy = HotSetPolicy::Quota;
    LocalityPlanner p(rtx5090(), cfg);
    // Nothing can be held, so declining everything is the RIGHT answer here - but it must be
    // reached by the arithmetic rather than by an overflow, so the same geometry one order of
    // magnitude smaller must also decline, and a budget that holds whole layers must admit.
    int admitted = 0;
    for (int layer = 0; layer < 48; ++layer)
        if (p.plan_for_layer(3 * MiB, true, huge, layer).use_persisting_window) ++admitted;
    // The invariant the overflow broke: if the budget holds whole windows, admit that many.
    // 30 MiB of set-aside holds ten 3 MiB windows, and a hot set of SIZE_MAX does not change
    // how many the device can keep -- it changes how little of the workload they cover.
    CHECK(admitted == static_cast<int>(p.effective_l2_budget() / (3 * MiB)));
    CHECK(admitted > 0);

    // The published figure, reproduced: 60 MiB of set-aside over a 61.41 MiB footprint admits
    // 29 of 30 layers whole and declines one.
    PlannerConfig full = cfg;
    full.persisting_budget_fraction = 1.0;
    LocalityPlanner q(rtx5090(), full);
    int ok = 0;
    const std::size_t layer_state = moe_geometry(1).bytes_per_layer;   // 2.047 MiB, not 2 MiB
    for (int layer = 0; layer < 30; ++layer)
        if (q.plan_for_layer(layer_state, true, moe_geometry(1), layer).use_persisting_window) ++ok;
    CHECK(ok == 29);
    CHECK(60 * MiB / layer_state == 29);   // the arithmetic the figure comes from
}

static void test_a_backoff_never_asks_for_more_residency_than_the_set_aside_holds() {
    // hitRatio x num_bytes is what the driver is told to keep resident. Proportional's
    // un-floored arithmetic satisfies that <= budget by construction; the min_hit_ratio clamp
    // is the only thing that can break it, and on a device whose persisting L2 is smaller
    // than one layer's state it asked for 77x the reservation.
    const DeviceCaps tiny{6 * MiB, 512 * 1024, 128 * MiB};
    PlannerConfig cfg = base_config();
    cfg.mode = LocalityMode::Persist;
    cfg.hot_set_model = HotSetModel::TokenFootprint;
    cfg.persisting_budget_fraction = 0.5;            // 256 KiB of budget
    cfg.min_hit_ratio = 0.25;
    cfg.hot_set_policy = HotSetPolicy::Proportional;
    LocalityPlanner p(tiny, cfg);
    const auto plan = p.plan_for_layer(3 * MiB, true, moe_geometry(1), 0);
    CHECK(plan.use_persisting_window);
    CHECK(plan.hit_ratio * static_cast<double>(plan.hot_window_bytes)
          <= static_cast<double>(p.effective_l2_budget()) + 1.0);
    // On a device where the budget comfortably exceeds one window the floor is untouched, so
    // no number this repository has published moves.
    LocalityPlanner big(rtx5090(), cfg);
    const auto unchanged = big.plan_for_layer(2 * MiB, true, moe_geometry(32), 0);
    CHECK_NEAR(unchanged.hit_ratio, cfg.min_hit_ratio);
}

static void test_the_oversubscription_flag_describes_the_workload_not_the_device() {
    // Same 61.4 MiB footprint, three devices and configs that all end with no window. The
    // flag used to read FALSE on the device with no persisting L2 at all - the maximally
    // oversubscribed case - and TRUE on one that merely refused the access-policy window.
    PlannerConfig cfg = base_config();
    cfg.mode = LocalityMode::Persist;
    cfg.hot_set_model = HotSetModel::TokenFootprint;

    const auto no_persist = LocalityPlanner(DeviceCaps{96 * MiB, 0, 128 * MiB}, cfg)
                                .plan_for_layer(2 * MiB, true, moe_geometry(1), 0);
    const auto no_window = LocalityPlanner(DeviceCaps{96 * MiB, 60 * MiB, 0}, cfg)
                               .plan_for_layer(2 * MiB, true, moe_geometry(1), 0);
    PlannerConfig declined = cfg;
    declined.persisting_budget_fraction = 0.0;
    const auto caller_declined = LocalityPlanner(rtx5090(), declined)
                                     .plan_for_layer(2 * MiB, true, moe_geometry(1), 0);

    for (const auto* plan : {&no_persist, &no_window, &caller_declined}) {
        CHECK(!plan->use_persisting_window);
        CHECK(plan->hot_set_bytes > 60 * MiB);
        CHECK(plan->hot_set_oversubscribed);
    }
}

static void test_a_set_aside_request_is_one_the_driver_could_grant() {
    // A set-aside is carved OUT of L2, so it cannot exceed L2 - and `l2_bytes` was queried,
    // printed, and read by no policy at all, so a device whose two numbers disagree got a
    // request for a set-aside larger than its entire cache with nothing to say so.
    const struct { const char* name; DeviceCaps caps; } matrix[] = {
        {"rtx5090",      rtx5090()},
        {"a100",         DeviceCaps{40 * MiB, 30 * MiB, 128 * MiB}},
        {"h100",         DeviceCaps{50 * MiB, 40 * MiB, 128 * MiB}},
        {"tiny_persist", DeviceCaps{6 * MiB, 512 * 1024, 128 * MiB}},
        {"huge_persist", DeviceCaps{512 * MiB, 512 * MiB, 128 * MiB}},
        {"reports_zero", DeviceCaps{0, 0, 0}},
        // An emulator, a MIG slice or a stubbed query: more persisting L2 than L2.
        {"incoherent",   DeviceCaps{16 * MiB, 64 * MiB, 128 * MiB}},
    };
    for (const auto& d : matrix) {
        for (double f : {0.0, 0.05, 0.5, 0.75, 1.0}) {
            PlannerConfig cfg = base_config();
            cfg.persisting_budget_fraction = f;
            LocalityPlanner p(d.caps, cfg);
            const auto want = p.recommended_l2_set_aside();
            CHECK(want <= d.caps.persisting_l2_max_bytes);
            if (d.caps.l2_bytes) CHECK(want <= d.caps.l2_bytes);
            CHECK(p.effective_l2_budget() == want);
        }
    }
}

static void test_the_most_capable_device_representable_still_persists() {
    // static_cast<std::size_t> of a double that rounds to 2^64 is undefined, and the observed
    // answer was 0 - persist silently OFF on a device with the largest capacity the type can
    // express. Real hardware cannot reach it; a device-capability test can, and a policy that
    // is undefined on an input a test can construct is a policy nobody can check.
    constexpr auto kMax = static_cast<std::size_t>(-1);
    PlannerConfig cfg = base_config();
    cfg.persisting_budget_fraction = 1.0;
    LocalityPlanner p(DeviceCaps{kMax, kMax, 128 * MiB}, cfg);
    CHECK(p.recommended_l2_set_aside() == kMax);
    cfg.persisting_budget_fraction = 0.5;
    CHECK(LocalityPlanner(DeviceCaps{kMax, kMax, 128 * MiB}, cfg).recommended_l2_set_aside() > 0);
}


static void test_a_backoff_that_changed_nothing_is_not_reported_as_a_reduction() {
    // `hit_ratio_reduced` means "the requested hit ratio was cut". Where min_hit_ratio sits
    // at or above hit_ratio the clamp returns the request unchanged, and reporting a cut that
    // did not happen makes the one telemetry field that explains a null result lie about it.
    PlannerConfig cfg = base_config();
    cfg.mode = LocalityMode::Persist;
    cfg.hot_set_model = HotSetModel::TokenFootprint;
    cfg.hit_ratio = 0.2;
    cfg.min_hit_ratio = 0.9;          // floor above the request: the clamp cannot move it
    for (auto pol : {HotSetPolicy::Proportional, HotSetPolicy::Sqrt}) {
        cfg.hot_set_policy = pol;
        const auto plan = LocalityPlanner(rtx5090(), cfg)
                              .plan_for_layer(2 * MiB, true, moe_geometry(32), 0);
        CHECK(plan.hot_set_oversubscribed);
        CHECK_NEAR(plan.hit_ratio, 0.2);
        CHECK(!plan.hit_ratio_reduced);
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
    test_quota_admits_whole_layers_instead_of_shaving_every_hit_ratio();
    test_quota_admits_the_same_layers_on_every_token();
    test_quota_spends_a_fixed_budget_however_many_layers_want_it();
    test_quota_is_inert_when_everything_fits();
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
    test_fixed_set_aside_is_the_shipped_behaviour_whatever_the_workload();
    test_a_workload_aware_policy_with_no_workload_is_the_shipped_behaviour();
    test_fit_footprint_never_reserves_more_than_the_footprint_can_use();
    test_fit_footprint_reaches_the_setting_the_measurement_calls_best();
    test_residency_declines_where_the_footprint_cannot_be_held();
    test_residency_gives_up_the_dense_models_small_real_gain_and_says_so();
    test_residency_thresholds_of_zero_and_one_are_the_two_corners();
    test_the_footprint_saturates_instead_of_wrapping();
    test_a_negative_sequence_count_counts_as_one();
    test_set_aside_policy_names_round_trip();
    test_min_residency_is_validated();
    test_the_stability_contract_holds_where_the_code_can_check_it();
    test_window_attach_names_round_trip_including_strict();
    test_set_aside_policy_values_are_appended_not_inserted();
    test_a_device_with_no_persisting_l2_asks_for_no_window();
    test_a_persisting_capacity_too_small_to_hold_one_window_is_declined();
    test_a_persisting_capacity_larger_than_the_footprint_reserves_only_what_is_needed();
    test_fabricated_datacentre_devices_behave_as_their_arithmetic_says();
    test_a_window_cap_smaller_than_the_state_still_clamps();
    test_a_device_that_allows_no_window_at_all_installs_none();
    test_quota_without_an_ordinal_says_it_fell_back_to_fixed();
    test_quota_admits_the_budget_once_not_once_per_period();
    test_quota_still_admits_when_the_hot_set_has_saturated();
    test_a_backoff_never_asks_for_more_residency_than_the_set_aside_holds();
    test_the_oversubscription_flag_describes_the_workload_not_the_device();
    test_a_set_aside_request_is_one_the_driver_could_grant();
    test_the_most_capable_device_representable_still_persists();
    test_a_backoff_that_changed_nothing_is_not_reported_as_a_reduction();

    if (g_failures) { std::cout << g_failures << " planner check(s) failed\n"; return 1; }
    std::cout << "planner tests passed\n";
    return 0;
}
