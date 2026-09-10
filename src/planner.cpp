#include "recurlocal/planner.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace recurlocal {

const char* version_string() noexcept { return RECURLOCAL_VERSION_STRING; }
int version_number() noexcept { return RECURLOCAL_VERSION_NUMBER; }

const char* validate(const PlannerConfig& config) noexcept {
    if (!(config.persisting_budget_fraction >= 0.0 && config.persisting_budget_fraction <= 1.0))
        return "persisting_budget_fraction must be in [0,1]";
    if (!(config.hit_ratio >= 0.0 && config.hit_ratio <= 1.0))
        return "hit_ratio must be in [0,1]";
    if (config.prefetch_distance < 0 || config.prefetch_distance > kMaxPrefetchDistance)
        return "prefetch_distance must be in [0,8]";
    if (!(config.min_hit_ratio >= 0.0 && config.min_hit_ratio <= 1.0))
        return "min_hit_ratio must be in [0,1]";
    if (!(config.min_residency >= 0.0 && config.min_residency <= 1.0))
        return "min_residency must be in [0,1]";
    return nullptr;
}

LocalityPlanner::LocalityPlanner(DeviceCaps caps, PlannerConfig config)
    : caps_(caps), config_(config) {
    if (const char* problem = validate(config_)) throw std::invalid_argument(problem);
}

std::size_t LocalityPlanner::recommended_l2_set_aside() const noexcept {
    if (!caps_.persisting_l2_max_bytes || config_.persisting_budget_fraction <= 0.0) return 0;
    // The whole capacity, without going through a double. static_cast<std::size_t> of a
    // double that rounds to 2^64 is undefined, and a fabricated DeviceCaps with
    // persisting_l2_max_bytes = SIZE_MAX reaches it: the observed answer was 0, which turns
    // persist off on the most capable device representable. Real hardware cannot get there
    // (cudaDeviceProp reports an int) but a device-capability test can, and a policy that is
    // undefined on an input a test can construct is a policy nobody can check.
    std::size_t requested = caps_.persisting_l2_max_bytes;
    if (config_.persisting_budget_fraction < 1.0)
        requested = static_cast<std::size_t>(
            static_cast<double>(caps_.persisting_l2_max_bytes) * config_.persisting_budget_fraction);
    requested = std::min(requested, caps_.persisting_l2_max_bytes);
    // A set-aside is carved OUT of L2, so it cannot exceed L2. Until now `l2_bytes` was
    // queried by the controller, printed by the info tool and the benchmark, and read by no
    // policy at all - so a device whose two numbers disagree (an emulator, a MIG slice, a
    // stubbed query, a fabricated fixture) got a request for a set-aside larger than its
    // entire cache, with nothing to say so. Zero still means "not reported", not "no cache".
    if (caps_.l2_bytes) requested = std::min(requested, caps_.l2_bytes);
    return requested;
}

namespace {
// Distance for one layer under a schedule. Depends only on the layer index so a caller
// never has to know how many recurrent layers remain.
int scheduled_distance(PrefetchSchedule schedule, int base, int layer_index) noexcept {
    if (layer_index < 0 || base <= 0) return base;
    switch (schedule) {
        case PrefetchSchedule::Uniform:     return base;
        case PrefetchSchedule::Ramp:        return std::min(base, 1 + layer_index / 8);
        case PrefetchSchedule::Alternating: return (layer_index % 2 == 0) ? base : 0;
        case PrefetchSchedule::Sparse:      return (layer_index % 4 == 0)
                                                   ? std::min(kMaxPrefetchDistance, base * 2) : 0;
    }
    return base;
}
} // namespace

namespace {
// Bytes that must stay resident for a recurrent state to still be there when its layer
// comes round again. Saturating, because a wrong answer here must never be a small one.
std::size_t saturating_mul(std::size_t a, std::size_t b) noexcept {
    if (!a || !b) return 0;
    constexpr auto kMax = static_cast<std::size_t>(-1);
    return (a > kMax / b) ? kMax : a * b;
}
std::size_t saturating_add(std::size_t a, std::size_t b) noexcept {
    constexpr auto kMax = static_cast<std::size_t>(-1);
    return (a > kMax - b) ? kMax : a + b;
}
} // namespace

std::size_t LocalityPlanner::token_footprint_bytes(const RecurrentGeometry& g) noexcept {
    if (!g.valid()) return 0;
    const auto seqs = static_cast<std::size_t>(g.sequences > 0 ? g.sequences : 1);
    return saturating_mul(
        saturating_mul(g.bytes_per_layer, static_cast<std::size_t>(g.recurrent_layers)), seqs);
}

double LocalityPlanner::achievable_residency(const RecurrentGeometry& g) const noexcept {
    const auto footprint = token_footprint_bytes(g);
    if (!footprint || !caps_.persisting_l2_max_bytes) return 0.0;
    if (caps_.persisting_l2_max_bytes >= footprint) return 1.0;
    return static_cast<double>(caps_.persisting_l2_max_bytes) / static_cast<double>(footprint);
}

std::size_t LocalityPlanner::recommended_l2_set_aside(const RecurrentGeometry& g) const noexcept {
    const auto fixed = recommended_l2_set_aside();
    if (config_.set_aside_policy == SetAsidePolicy::Fixed) return fixed;
    // A workload-aware rule with no workload is not a rule. Degrading to the shipped constant
    // is the only honest thing to do here, and it is what keeps a caller that never declares
    // its geometry on exactly the behaviour it had.
    const auto footprint = token_footprint_bytes(g);
    if (!footprint || !caps_.persisting_l2_max_bytes) return fixed;
    if (config_.persisting_budget_fraction <= 0.0) return 0;

    if (config_.set_aside_policy == SetAsidePolicy::Residency &&
        achievable_residency(g) < config_.min_residency)
        return 0;   // it cannot hold enough of the footprint to be worth the cache it takes

    // Take what the footprint can use, and no more than the device will give.
    //
    // `persisting_budget_fraction` is deliberately NOT applied here. It is the dial
    // SetAsidePolicy::Fixed turns, and turning both would leave the workload-aware rules
    // unable to reach the setting the workload wants without the caller ALSO changing the
    // constant they were introduced to replace - which is the whole defect. One dial per
    // policy; the fraction is documented as read by Fixed only.
    return std::min(footprint, caps_.persisting_l2_max_bytes);
}

int LocalityPlanner::distance_for_layer(int layer_index) const noexcept {
    if (config_.mode != LocalityMode::Prefetch && config_.mode != LocalityMode::Combined) return 0;
    return scheduled_distance(config_.prefetch_schedule, config_.prefetch_distance, layer_index);
}

bool is_recurrent_layer(int layer, int full_attn_interval) noexcept {
    if (layer < 0) return false;
    // interval <= 0 means "not a hybrid stack"; every layer is then recurrent, which is the
    // right answer for a pure recurrent model and harmless for a pure attention one (the
    // caller will not be asking).
    if (full_attn_interval <= 0) return true;
    return ((layer + 1) % full_attn_interval) != 0;
}

int recurrent_layer_ahead(int layer, int distance, int layer_count,
                          int full_attn_interval) noexcept {
    if (distance <= 0 || layer_count <= 0) return -1;
    int remaining = distance;
    for (int i = layer + 1; i < layer_count; ++i) {
        if (!is_recurrent_layer(i, full_attn_interval)) continue;
        if (--remaining == 0) return i;
    }
    return -1;
}

const StateSegment* select_window_segment(const StateSegment* segments, int count,
                                          WindowTarget target) noexcept {
    if (!segments || count <= 0) return nullptr;
    const StateSegment* best = nullptr;
    for (int i = 0; i < count; ++i) {
        const StateSegment& seg = segments[i];
        if (!seg.ptr || !seg.bytes) continue;
        switch (target) {
            case WindowTarget::Matrix: if (seg.kind == StateKind::Matrix) return &seg; break;
            case WindowTarget::Conv:   if (seg.kind == StateKind::Conv) return &seg; break;
            case WindowTarget::Widest:    if (!best || seg.bytes > best->bytes) best = &seg; break;
            case WindowTarget::Narrowest: if (!best || seg.bytes < best->bytes) best = &seg; break;
        }
    }
    // Matrix and Conv fall through to the first usable segment rather than declining to
    // act: a model that does not carry the requested state should still get a window over
    // the state it does carry.
    if (!best && (target == WindowTarget::Matrix || target == WindowTarget::Conv))
        for (int i = 0; i < count; ++i)
            if (segments[i].ptr && segments[i].bytes) return &segments[i];
    return best;
}

WindowRegion resolve_window_region(const StateSegment& seg, WindowScope scope,
                                   int prefetch_distance) noexcept {
    const auto* slice = static_cast<const unsigned char*>(seg.ptr);
    const auto* base = static_cast<const unsigned char*>(seg.base);
    const bool have_base = base && seg.base_bytes && slice >= base && slice < base + seg.base_bytes;
    switch (scope) {
        case WindowScope::Layer:
            break;
        case WindowScope::Allocation:
            // The whole recurrent-state array. Under a token-scale reuse distance this is
            // the only region whose residency is worth anything: a window over one layer's
            // slice has moved on 47 times before that slice is read again.
            if (have_base) return {seg.base, seg.base_bytes};
            break;
        case WindowScope::Ahead: {
            // Reaching past this layer's slice is only safe where the caller told us how big
            // the allocation is. With no base declared, the slice is the ONLY memory we know
            // belongs to the runtime, so widening would hint - and, via Ahead's use in the
            // pre-touch sizing, potentially read - memory nobody said we owned.
            if (!have_base) return {seg.ptr, seg.bytes};
            const std::size_t reach = prefetch_distance > 0
                                    ? static_cast<std::size_t>(prefetch_distance) + 1 : 1;
            std::size_t span = seg.bytes * reach;
            const auto remaining = static_cast<std::size_t>(base + seg.base_bytes - slice);
            if (span > remaining) span = remaining;
            return {seg.ptr, span};
        }
    }
    return {seg.ptr, seg.bytes};
}

bool pre_touch_covers(PreTouchCoverage coverage, StateKind kind) noexcept {
    switch (coverage) {
        case PreTouchCoverage::Matrix: return kind != StateKind::Conv;
        case PreTouchCoverage::Conv:   return kind == StateKind::Conv;
        case PreTouchCoverage::Both:   return true;
    }
    return true;
}

LayerPlan LocalityPlanner::plan_for_layer(std::size_t current_state_bytes,
                                           bool has_next_recurrent_layer,
                                           const RecurrentGeometry& geometry,
                                           int layer_index) const noexcept {
    // The geometry-aware form differs from the legacy one only in what it counts as hot,
    // so it computes that here and hands the result to the shared body as the caller's
    // declared bytes, minus the window the body will add back.
    std::size_t declared = 0;
    HotSetModel applied = HotSetModel::CurrentLayer;
    if (config_.hot_set_model != HotSetModel::CurrentLayer && geometry.valid()) {
        applied = config_.hot_set_model;
        const auto seqs = static_cast<std::size_t>(geometry.sequences > 0 ? geometry.sequences : 1);
        // Every recurrent layer's state is equally live across a token: the reuse distance
        // for layer 0's state is the whole rest of the token's layer walk. Counting only
        // the layer about to run is what let `persist` be applied at concurrencies where
        // the working set was already three times the set-aside.
        std::size_t footprint = saturating_mul(
            saturating_mul(geometry.bytes_per_layer,
                           static_cast<std::size_t>(geometry.recurrent_layers)), seqs);
        if (applied == HotSetModel::ReuseWindow)
            footprint = saturating_add(footprint, geometry.streamed_bytes_per_token);
        declared = footprint;
    }
    // The footprint models give an ABSOLUTE hot set - everything live across a token -
    // where CurrentLayer gives bytes to ADD to the window. Passing the footprint as
    // something to add would double-count this layer's slice; subtracting the slice first
    // would get it wrong the other way as soon as WindowScope widens the window past one
    // slice. So the two are kept as different quantities rather than made to look alike.
    auto p = plan_for_layer_impl(current_state_bytes, has_next_recurrent_layer,
                                 applied == HotSetModel::CurrentLayer ? 0 : declared,
                                 applied == HotSetModel::CurrentLayer,
                                 layer_index,
                                 geometry.recurrent_layers > 0
                                     ? static_cast<std::size_t>(geometry.recurrent_layers) : 0u);
    p.hot_set_model = applied;
    return p;
}

LayerPlan LocalityPlanner::plan_for_layer(std::size_t current_state_bytes,
                                           bool has_next_recurrent_layer,
                                           std::size_t concurrently_hot_bytes,
                                           int layer_index) const noexcept {
    return plan_for_layer_impl(current_state_bytes, has_next_recurrent_layer,
                               concurrently_hot_bytes, /*additive=*/true, layer_index);
}

LayerPlan LocalityPlanner::plan_for_layer_impl(std::size_t current_state_bytes,
                                               bool has_next_recurrent_layer,
                                               std::size_t hot_bytes, bool additive,
                                               int layer_index,
                                               std::size_t ordinal_period) const noexcept {
    LayerPlan p{};
    p.hot_set_policy = config_.hot_set_policy;
    p.window_scope = config_.window_scope;
    p.window_target = config_.window_target;
    p.pre_touch_coverage = config_.pre_touch_coverage;
    const bool use_persist = config_.mode == LocalityMode::Persist || config_.mode == LocalityMode::Combined;
    const bool use_prefetch = config_.mode == LocalityMode::Prefetch || config_.mode == LocalityMode::Combined;

    // Without a reserved L2 set-aside the hardware has nothing to hold a persisting
    // window in, so the plan must not ask an integrating runtime to install one.
    const auto budget = effective_l2_budget();
    // Report the accounting whatever the mode. It used to be written only inside the persist
    // guard, so a prefetch-only run emitted hot_set_bytes=0, hot_set_budget_bytes=0 next to
    // hot_set_model="token_footprint" - a reader would conclude nothing was competing for L2
    // when 147 MiB was. The numbers describe the workload, not the policy that happens to be
    // enabled.
    p.hot_set_budget_bytes = budget;
    p.hot_set_bytes = additive ? saturating_add(hot_bytes, current_state_bytes)
                               : std::max(hot_bytes, current_state_bytes);
    // Not `budget > 0 &&`. A device with no persisting L2, or a caller who set the budget
    // fraction to zero, is the MAXIMALLY oversubscribed case, and the flag read false there
    // while reading true for the same 148 MiB workload on a device that merely refused the
    // access-policy window - opposite answers for identical outcomes. The comment above says
    // these numbers describe the workload; now they do. (Telemetry contract change: a
    // zero-budget arm that used to emit hot_set_oversubscribed=0 now emits the truth.)
    p.hot_set_oversubscribed = p.hot_set_bytes > budget;

    if (use_persist && current_state_bytes && caps_.access_policy_max_window_bytes && budget) {
        std::size_t window = std::min(current_state_bytes, caps_.access_policy_max_window_bytes);
        if (config_.max_hot_window_bytes) window = std::min(window, config_.max_hot_window_bytes);
        p.hot_window_bytes = window;
        p.use_persisting_window = window > 0;
        p.hit_ratio = config_.hit_ratio;
        // Additive: the caller declared bytes that compete with the window this layer wants.
        // Absolute: the caller's model already counts everything live, and the window can
        // never be the smaller of the two without the accounting being wrong.
        const std::size_t hot = additive ? saturating_add(hot_bytes, window)
                                         : std::max(hot_bytes, window);
        p.hot_set_bytes = hot;               // refine: the window is what we actually ask for
        p.hot_set_oversubscribed = hot > budget;   // budget is non-zero inside this branch
        if (hot > budget) {
            const double share = static_cast<double>(budget) / static_cast<double>(hot);
            // What the driver is told to keep resident is hitRatio x num_bytes. Proportional's
            // un-floored arithmetic satisfies that <= budget by construction; the min_hit_ratio
            // clamp is the only thing that can break it, and on a device whose persisting L2 is
            // smaller than one layer's state it broke it by 77x - asking 157 KiB resident
            // against a 2 KiB set-aside. The floor may not raise a request above what the
            // reservation can physically hold.
            const double budget_ratio = window ? static_cast<double>(budget) / static_cast<double>(window)
                                               : 0.0;
            const double floor_ratio = std::min(config_.min_hit_ratio, config_.hit_ratio);
            switch (config_.hot_set_policy) {
                case HotSetPolicy::Fixed:
                    break;  // ask for everything and let the hardware sort it out
                // `hit_ratio_reduced` means the requested hit ratio was CUT, and it is the one
                // telemetry field that explains a null persist result. Setting it whenever the
                // policy ran made it report a cut that did not happen - with min_hit_ratio at
                // or above hit_ratio the clamp returns the request unchanged - so a reader
                // chasing "why did persist do nothing" was sent to a back-off that never
                // occurred. It now reports the outcome rather than the branch.
                case HotSetPolicy::Proportional:
                    p.hit_ratio = std::clamp(config_.hit_ratio * share, floor_ratio, config_.hit_ratio);
                    if (budget_ratio < p.hit_ratio) p.hit_ratio = budget_ratio;
                    p.hit_ratio_reduced = p.hit_ratio < config_.hit_ratio;
                    break;
                case HotSetPolicy::Sqrt:
                    p.hit_ratio = std::clamp(config_.hit_ratio * std::sqrt(share), floor_ratio, config_.hit_ratio);
                    p.hit_ratio_reduced = p.hit_ratio < config_.hit_ratio;
                    break;
                case HotSetPolicy::Cliff:
                    p.use_persisting_window = false;
                    p.hot_window_bytes = 0;
                    p.hit_ratio = 0.0;
                    p.hit_ratio_reduced = true;
                    break;
                case HotSetPolicy::Quota: {
                    // Ration whole layers over the ordinals the caller will actually walk.
                    //
                    // This used to count window-sized UNITS of the hot set and spread the
                    // admitted ones over that many indices - `(i * admissible) % units`. The
                    // pattern is right for one period of `units` ordinals and wrong past it:
                    // where the hot set holds fewer units than the model has recurrent layers
                    // (any window wider than one layer's slice, which WindowScope::Ahead
                    // produces today) the period repeats and admits several times the budget.
                    // And `hot + window - 1` wrapped whenever `hot` had saturated, making
                    // `units` zero and Quota decline everything - the largest representable
                    // hot set producing the smallest possible admission, the exact inversion
                    // the saturating arithmetic exists to prevent.
                    //
                    // The period is the number of recurrent layers, which only the
                    // geometry-aware caller knows. Without it Quota cannot ration at all, and
                    // says so through `p.hot_set_policy` rather than silently behaving as
                    // Fixed while reporting that nothing backed off.
                    std::size_t period = ordinal_period;
                    if (!period && window) {
                        // No declared layer count: fall back to counting window-sized units of
                        // the hot set, which is what this policy always did. That is exactly
                        // right while the window is one layer's slice - the unit count IS the
                        // layer count - and it is the reason the fallback is kept rather than
                        // replaced. Written as a division plus a remainder rather than the
                        // old `(hot + window - 1) / window`, whose addition wrapped whenever
                        // `hot` had saturated.
                        period = hot / window + ((hot % window) ? 1u : 0u);
                    }
                    if (!period || layer_index < 0) {
                        // Quota rations by ordinal. A caller who supplies none cannot be
                        // rationed, and used to get a full-hit-ratio window on every layer
                        // while the telemetry reported no back-off - selecting the policy had
                        // no effect and nothing said so.
                        p.hot_set_policy = HotSetPolicy::Fixed;
                        break;
                    }
                    // How many WHOLE windows the set-aside holds. Counted in windows and not
                    // in ordinal shares of the hot set: an ordinal's share is hot/period,
                    // which is the same thing only while the window is one layer's slice. Where
                    // it is wider, admitting a share-derived count spends `admitted x window`
                    // bytes against a budget sized for `admitted x hot/period`, and
                    // over-subscribes by their ratio.
                    const std::size_t admitted = std::min(period, budget / window);
                    // Spread them evenly rather than taking a prefix, so no contiguous run of
                    // the model is left uncovered, and decide from the ordinal alone so every
                    // token makes the identical choice - a window that moved between tokens
                    // would evict exactly the state it kept. Reducing the ordinal modulo the
                    // period is what stops the pattern repeating past one period and admitting
                    // several times the budget.
                    const auto i = static_cast<std::size_t>(layer_index) % period;
                    const bool admit = admitted > 0 && (i * admitted) % period < admitted;
                    if (!admit) {
                        p.use_persisting_window = false;
                        p.hot_window_bytes = 0;
                        p.hit_ratio = 0.0;
                    }
                    // Declining this layer's window IS the back-off, so it is reported as one.
                    p.hit_ratio_reduced = !admit;
                    break;
                }
            }
        }
    }
    const int distance = scheduled_distance(config_.prefetch_schedule,
                                            config_.prefetch_distance, layer_index);
    p.prefetch_next = use_prefetch && distance > 0 && has_next_recurrent_layer;
    if (p.prefetch_next) {
        p.prefetch_distance = distance;
        p.pre_touch = config_.pre_touch;
    }
    return p;
}

const char* to_string(LocalityMode mode) noexcept {
    switch (mode) {
        case LocalityMode::Baseline: return "baseline";
        case LocalityMode::Persist: return "persist";
        case LocalityMode::Prefetch: return "prefetch";
        case LocalityMode::Combined: return "combined";
    }
    return "unknown";
}

const char* to_string(PreTouchStrategy strategy) noexcept {
    switch (strategy) {
        case PreTouchStrategy::Scalar: return "scalar";
        case PreTouchStrategy::Vec4: return "vec4";
        case PreTouchStrategy::Vec4Ldcg: return "vec4_ldcg";
        case PreTouchStrategy::PtxL2: return "ptx_l2";
        case PreTouchStrategy::WarpTile: return "warp_tile";
        case PreTouchStrategy::Partial: return "partial";
    }
    return "unknown";
}

PreTouchStrategy parse_pre_touch_strategy(const char* text) {
    if (!text) throw std::invalid_argument("pre-touch strategy is null");
    std::string s(text);
    if (s == "scalar") return PreTouchStrategy::Scalar;
    if (s == "vec4") return PreTouchStrategy::Vec4;
    if (s == "vec4_ldcg") return PreTouchStrategy::Vec4Ldcg;
    if (s == "ptx_l2") return PreTouchStrategy::PtxL2;
    if (s == "warp_tile") return PreTouchStrategy::WarpTile;
    if (s == "partial") return PreTouchStrategy::Partial;
    throw std::invalid_argument("unknown pre-touch strategy: " + s);
}

const char* to_string(SetAsidePolicy policy) noexcept {
    switch (policy) {
        case SetAsidePolicy::Fixed: return "fixed";
        case SetAsidePolicy::FitFootprint: return "fit_footprint";
        case SetAsidePolicy::Residency: return "residency";
    }
    // Not "fixed". Every other to_string here returns "unknown" for a value outside the
    // enumeration, and returning the CONTROL arm's own name would put a sweep's control
    // label on a candidate nobody can identify.
    return "unknown";
}

SetAsidePolicy parse_set_aside_policy(const char* text) {
    const std::string t = text ? text : "";
    if (t == "fixed") return SetAsidePolicy::Fixed;
    if (t == "fit_footprint") return SetAsidePolicy::FitFootprint;
    if (t == "residency") return SetAsidePolicy::Residency;
    throw std::invalid_argument("unknown set-aside policy: " + t);
}

const char* to_string(HotSetPolicy policy) noexcept {
    switch (policy) {
        case HotSetPolicy::Proportional: return "proportional";
        case HotSetPolicy::Fixed: return "fixed";
        case HotSetPolicy::Sqrt: return "sqrt";
        case HotSetPolicy::Cliff: return "cliff";
        case HotSetPolicy::Quota: return "quota";
    }
    return "unknown";
}

HotSetPolicy parse_hot_set_policy(const char* text) {
    if (!text) throw std::invalid_argument("hot-set policy is null");
    std::string s(text);
    if (s == "proportional") return HotSetPolicy::Proportional;
    if (s == "fixed") return HotSetPolicy::Fixed;
    if (s == "sqrt") return HotSetPolicy::Sqrt;
    if (s == "cliff") return HotSetPolicy::Cliff;
    if (s == "quota") return HotSetPolicy::Quota;
    throw std::invalid_argument("unknown hot-set policy: " + s);
}

const char* to_string(PrefetchSchedule schedule) noexcept {
    switch (schedule) {
        case PrefetchSchedule::Uniform: return "uniform";
        case PrefetchSchedule::Ramp: return "ramp";
        case PrefetchSchedule::Alternating: return "alternating";
        case PrefetchSchedule::Sparse: return "sparse";
    }
    return "unknown";
}

PrefetchSchedule parse_prefetch_schedule(const char* text) {
    if (!text) throw std::invalid_argument("prefetch schedule is null");
    std::string s(text);
    if (s == "uniform") return PrefetchSchedule::Uniform;
    if (s == "ramp") return PrefetchSchedule::Ramp;
    if (s == "alternating") return PrefetchSchedule::Alternating;
    if (s == "sparse") return PrefetchSchedule::Sparse;
    throw std::invalid_argument("unknown prefetch schedule: " + s);
}

const char* to_string(HotSetModel model) noexcept {
    switch (model) {
        case HotSetModel::CurrentLayer: return "current_layer";
        case HotSetModel::TokenFootprint: return "token_footprint";
        case HotSetModel::ReuseWindow: return "reuse_window";
    }
    return "unknown";
}

HotSetModel parse_hot_set_model(const char* text) {
    if (!text) throw std::invalid_argument("hot-set model is null");
    std::string s(text);
    if (s == "current_layer") return HotSetModel::CurrentLayer;
    if (s == "token_footprint") return HotSetModel::TokenFootprint;
    if (s == "reuse_window") return HotSetModel::ReuseWindow;
    throw std::invalid_argument("unknown hot-set model: " + s);
}

const char* to_string(WindowScope scope) noexcept {
    switch (scope) {
        case WindowScope::Layer: return "layer";
        case WindowScope::Allocation: return "allocation";
        case WindowScope::Ahead: return "ahead";
    }
    return "unknown";
}

WindowScope parse_window_scope(const char* text) {
    if (!text) throw std::invalid_argument("window scope is null");
    std::string s(text);
    if (s == "layer") return WindowScope::Layer;
    if (s == "allocation") return WindowScope::Allocation;
    if (s == "ahead") return WindowScope::Ahead;
    throw std::invalid_argument("unknown window scope: " + s);
}

const char* to_string(WindowTarget target) noexcept {
    switch (target) {
        case WindowTarget::Matrix: return "matrix";
        case WindowTarget::Conv: return "conv";
        case WindowTarget::Widest: return "widest";
        case WindowTarget::Narrowest: return "narrowest";
    }
    return "unknown";
}

WindowTarget parse_window_target(const char* text) {
    if (!text) throw std::invalid_argument("window target is null");
    std::string s(text);
    if (s == "matrix") return WindowTarget::Matrix;
    if (s == "conv") return WindowTarget::Conv;
    if (s == "widest") return WindowTarget::Widest;
    if (s == "narrowest") return WindowTarget::Narrowest;
    throw std::invalid_argument("unknown window target: " + s);
}

const char* to_string(WindowAttach attach) noexcept {
    switch (attach) {
        case WindowAttach::Stream: return "stream";
        case WindowAttach::CaptureNode: return "capture_node";
        case WindowAttach::CaptureNodeStrict: return "capture_node_strict";
    }
    return "unknown";
}

WindowAttach parse_window_attach(const char* text) {
    if (!text) throw std::invalid_argument("window attach is null");
    std::string s(text);
    if (s == "stream") return WindowAttach::Stream;
    if (s == "capture_node") return WindowAttach::CaptureNode;
    if (s == "capture_node_strict") return WindowAttach::CaptureNodeStrict;
    throw std::invalid_argument("unknown window attach: " + s);
}

const char* to_string(PrefetchJoin join) noexcept {
    switch (join) {
        case PrefetchJoin::PerLayer: return "per_layer";
        case PrefetchJoin::TokenEnd: return "token_end";
    }
    return "unknown";
}

PrefetchJoin parse_prefetch_join(const char* text) {
    if (!text) throw std::invalid_argument("prefetch join is null");
    std::string s(text);
    if (s == "per_layer") return PrefetchJoin::PerLayer;
    if (s == "token_end") return PrefetchJoin::TokenEnd;
    throw std::invalid_argument("unknown prefetch join: " + s);
}

const char* to_string(PreTouchCoverage coverage) noexcept {
    switch (coverage) {
        case PreTouchCoverage::Matrix: return "matrix";
        case PreTouchCoverage::Conv: return "conv";
        case PreTouchCoverage::Both: return "both";
    }
    return "unknown";
}

PreTouchCoverage parse_pre_touch_coverage(const char* text) {
    if (!text) throw std::invalid_argument("pre-touch coverage is null");
    std::string s(text);
    if (s == "matrix") return PreTouchCoverage::Matrix;
    if (s == "conv") return PreTouchCoverage::Conv;
    if (s == "both") return PreTouchCoverage::Both;
    throw std::invalid_argument("unknown pre-touch coverage: " + s);
}

LocalityMode parse_mode(const char* text) {
    if (!text) throw std::invalid_argument("mode is null");
    std::string s(text);
    if (s == "baseline") return LocalityMode::Baseline;
    if (s == "persist") return LocalityMode::Persist;
    if (s == "prefetch") return LocalityMode::Prefetch;
    if (s == "combined") return LocalityMode::Combined;
    throw std::invalid_argument("unknown mode: " + s);
}

} // namespace recurlocal
