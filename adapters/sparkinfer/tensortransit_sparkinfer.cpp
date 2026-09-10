#include "tensortransit/sparkinfer.h"

#if defined(TENSORTRANSIT_WITH_CUDA) || defined(RECURLOCAL_WITH_CUDA) || \
    defined(RECURLLOCAL_WITH_CUDA)
#include "tensortransit/cuda_recurrent.h"
#include "transit_backend.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

namespace tensortransit {
namespace sparkinfer {
namespace {

// Every knob is an environment variable so one binary can run the whole matrix against one
// model load. Nothing here has a compiled-in default that differs from RecurLocal's own.
// Every setting is readable under both spellings: TENSORTRANSIT_X is preferred and
// RECURLOCAL_X is the deprecated 0.1 name, checked second.
//
// Keeping the old names working is not politeness. Every result in results/ was produced by
// a command line that sets them, and `eval/run_from_base.sh` re-runs the instrument from the
// BASE commit precisely so a scoring change cannot ride in on a submission -- so a rename
// that made those commands mean nothing would silently invalidate every published number and
// the reproduction path with it.
//
// The corollary is a real hazard and it is guarded on the other side: an evaluator that
// scrubs only RECURLOCAL* from the control environment would let an operator with
// TENSORTRANSIT=combined exported compare the candidate against itself and measure ~0%.
// eval/real_eval.py scrubs BOTH prefixes, and a test asserts it.
const char* env_lookup(const char* name) noexcept {
    if (!name) return nullptr;
    // "RECURLOCAL_FOO" -> try "TENSORTRANSIT_FOO" first; "RECURLOCAL" -> "TENSORTRANSIT".
    static constexpr const char kOld[] = "RECURLOCAL";
    static constexpr std::size_t kOldLen = sizeof(kOld) - 1;
    if (std::strncmp(name, kOld, kOldLen) == 0) {
        std::string preferred = "TENSORTRANSIT";
        preferred += name + kOldLen;
        if (const char* v = std::getenv(preferred.c_str()); v && *v) return v;
    }
    const char* v = std::getenv(name);
    return (v && *v) ? v : nullptr;
}
const char* env_or(const char* name, const char* fallback) noexcept {
    const char* v = env_lookup(name);
    return v ? v : fallback;
}
double env_double(const char* name, double fallback) noexcept {
    const char* v = env_lookup(name);
    if (!v) return fallback;
    char* end = nullptr;
    const double parsed = std::strtod(v, &end);
    return (end && end != v) ? parsed : fallback;
}
long long env_int(const char* name, long long fallback) noexcept {
    const char* v = env_lookup(name);
    if (!v) return fallback;
    char* end = nullptr;
    const long long parsed = std::strtoll(v, &end, 10);
    return (end && end != v) ? parsed : fallback;
}

// A parse failure must not be silent: a typo in a sweep script would otherwise run the
// default configuration under the label of the one that was asked for, and the result file
// would record a measurement of something else.
template <class Fn>
bool parse_or_fail(const char* name, const char* fallback, Fn fn, const char** why) noexcept {
    const char* text = env_or(name, fallback);
    try { fn(text); return true; }
    catch (const std::exception& e) {
        static std::string message;
        message = std::string(name) + "=" + text + ": " + e.what();
        *why = message.c_str();
        return false;
    }
}

struct Adapter {
    bool configured = false;       // env has been read
    bool usable = false;           // mode != off and the config parsed
    bool initialised = false;      // the controller owns a device and streams RIGHT NOW
    // ...and whether it ever did. shutdown() clears `initialised`, and the stats snapshot is
    // printed at exit - i.e. AFTER the model destructor has shut us down - so reporting only
    // the live flag told a reader the hook had never run when it had bracketed every layer.
    bool ever_initialised = false;
    cudaStream_t bound_stream = nullptr;   // the one model this process's adapter serves
    std::size_t l2_set_aside_at_init = 0;
    bool broken = false;           // initialisation failed; stay out of the way
    std::string mode = "off";
    const char* config_error = nullptr;

    PlannerConfig config{};
    CudaLocalityController controller;
    cudaStream_t prefetch_stream = nullptr;
    GdnStateLayout layout{};

    // Which engine drives the step. `v0` is CudaLocalityController -- the 0.1 policy, and
    // what every published number in results/ was measured under. `transit` is
    // TensorRegistry -> TransitGraph -> ITransitPlanner -> CudaTransitExecutor, which is the
    // surface a contributor can actually move.
    //
    // Both live in one binary and are selected by one environment variable, because the only
    // way to establish that the second reproduces the first is to run them against one model
    // load on one box in one interleaved A/B. A replacement would have made that impossible
    // to check, and "the migration changed nothing" would have been an assertion.
    bool use_transit = true;
    transit::Engine transit_engine;
    transit::Settings transit_settings{};
    transit::KvGeometry kv{};
    bool kv_declared = false;

    // Set for a packed (concurrent) step; cleared for a single-sequence one. The two
    // decode paths are different code in SparkInfer and produce different state shapes, so
    // the adapter tracks which one it is inside rather than guessing from the layout.
    bool packed = false;
    // The packed entry point borrows begin_token() for its stream setup, then overwrites the
    // sequence count. Declaring the geometry inside that borrowed call would size the
    // reservation for one sequence and then again for N, twice per token, forever.
    bool declare_suppressed = false;
    // The widest geometry the run ever declared. The LAST one is not representative: a
    // concurrent run's tail chunk is unpacked, so a run that spent 80% of its tokens at four
    // sequences reports one at exit.
    int max_geometry_sequences = 1;
    std::size_t max_geometry_bytes_per_layer = 0;
    GdnPackedLayout packed_layout{};

    int recurrent_layers = 0;
    RecurrentGeometry geometry{};
    // Recurrent-layer ordinal within this token, which is what the prefetch schedule is
    // indexed by. Not the absolute layer index: a hybrid model's recurrent layers are not
    // contiguous, and a schedule that skipped every other *absolute* layer would skip
    // whichever ones the full-attention interval happened to align with.
    int recurrent_ordinal = 0;
    // The absolute layer index the last before_layer()/before_attention_layer() opened.
    // after_layer() carries no argument -- the 0.1 hook never needed one -- and the transit
    // engine closes a kernel by id, so the id has to be remembered rather than guessed.
    int last_layer = -1;
    // Which state kernel the pending window was aimed at, so after_launch() attaches it to
    // the launch that actually reads those bytes.
    StateKernel pending_kernel = StateKernel::Gdn;
    bool window_pending = false;

    std::uint64_t tokens = 0;
    // Which decode path the run actually used. A runtime can decline to batch for reasons
    // that have nothing to do with locality - a row set that changed, an unsupported shape -
    // and a concurrency measurement that silently ran the single-sequence path is not a
    // concurrency measurement. Counted so the result file can say which one it was.
    std::uint64_t tokens_packed = 0;
    std::uint64_t layers_packed = 0;
    int max_rows_seen = 0;
};

// One adapter per process, deliberately: the hook has to be reachable from SparkInfer's
// static call sites without threading a handle through them. That is fine for one model on
// one stream - which is what every path in this integration is - and WRONG for two.
//
// The singleton holds cross-call mutable state (the layer ordinal, the pending window and its
// target kernel, the layout, the packed flag). Two Qwen35Model instances in one process would
// interleave their layer walks through it, and SparkInfer's mutex does not help because it is
// per-model. The result would not be a crash, it would be quietly wrong numbers - the worst
// failure mode for a measurement harness.
//
// So rather than a mutex, which would prevent the data race and preserve the wrong logic,
// this detects the second instance and refuses. A loud refusal is a bug report; a silent
// interleave is a bad result someone publishes.
Adapter& adapter() noexcept { static Adapter a; return a; }

// This layer's two state slices, as segments. Both, always: the accounting has to see the
// whole recurrent footprint even when only one of them is being windowed or pre-touched.
int segments_for(const GdnStateLayout& l, int layer, StateSegment out[2]) noexcept {
    int n = 0;
    if (l.lin_state && l.lin_state_stride) {
        auto* base = static_cast<unsigned char*>(l.lin_state);
        out[n].ptr = base + static_cast<std::size_t>(layer) * l.lin_state_stride;
        out[n].bytes = l.lin_state_stride;
        out[n].base = base;
        out[n].base_bytes = static_cast<std::size_t>(l.n_layers) * l.lin_state_stride;
        out[n].kind = StateKind::Matrix;
        ++n;
    }
    if (l.lin_conv_state && l.lin_conv_stride) {
        auto* base = static_cast<unsigned char*>(l.lin_conv_state);
        out[n].ptr = base + static_cast<std::size_t>(layer) * l.lin_conv_stride;
        out[n].bytes = l.lin_conv_stride;
        out[n].base = base;
        out[n].base_bytes = static_cast<std::size_t>(l.n_layers) * l.lin_conv_stride;
        out[n].kind = StateKind::Conv;
        ++n;
    }
    return n;
}

// Aim the deferred window at the kernel that reads the state it covers. WindowTarget's
// Widest and Narrowest are resolved by byte count, so ask the resolved region which
// allocation it fell in rather than re-deriving the choice here.
StateKernel pending_kernel_for(const GdnStateLayout& l, const void* region) noexcept {
    if (!region || !l.lin_conv_state) return StateKernel::Gdn;
    const auto* r = static_cast<const unsigned char*>(region);
    const auto* conv = static_cast<const unsigned char*>(l.lin_conv_state);
    const std::size_t conv_bytes = static_cast<std::size_t>(l.n_layers) * l.lin_conv_stride;
    return (r >= conv && r < conv + conv_bytes) ? StateKernel::Conv : StateKernel::Gdn;
}

void begin_common(Adapter& a, int n_layers, int full_attn_interval,
                  std::size_t bytes_per_layer, int sequences) noexcept {
    a.recurrent_layers = 0;
    for (int i = 0; i < n_layers; ++i)
        if (is_recurrent_layer(i, full_attn_interval)) ++a.recurrent_layers;

    a.geometry.recurrent_layers = a.recurrent_layers;
    a.geometry.bytes_per_layer = bytes_per_layer;
    a.geometry.sequences = sequences;
    // What else passes through L2 between two visits to the same state. The adapter cannot
    // know it - it is the model's weight and activation traffic per token - so it is
    // declared by whoever runs the benchmark, and left at zero (ReuseWindow == footprint)
    // rather than invented.
    a.geometry.streamed_bytes_per_token =
        static_cast<std::size_t>(env_int("RECURLOCAL_STREAMED_BYTES_PER_TOKEN", 0));

    a.recurrent_ordinal = 0;
    a.window_pending = false;
    // The transit engine opens its own step inside declare_geometry_now(), which runs after
    // the packed path has finalised the sequence count. Calling begin_sequence() here as well
    // would advance the 0.1 controller's per-token state in a run it is not driving.
    if (!a.use_transit) a.controller.begin_sequence();
    ++a.tokens;
}

// The step geometry the transit engine registers and records, in whichever shape this step
// is running. Row 0's addresses on the packed path: an access-policy window needs an address
// the host can name and there is exactly one, while the FOOTPRINT scales with `sequences`,
// which the engine is told separately.
transit::StepGeometry step_geometry(const Adapter& a) noexcept {
    transit::StepGeometry g{};
    if (a.packed) {
        g.lin_state = a.packed_layout.host_lin_state_row0;
        g.lin_state_stride = a.packed_layout.lin_state_stride;
        g.lin_conv_state = a.packed_layout.host_lin_conv_row0;
        g.lin_conv_stride = a.packed_layout.lin_conv_stride;
        g.n_layers = a.packed_layout.n_layers;
        g.full_attn_interval = a.packed_layout.full_attn_interval;
    } else {
        g.lin_state = a.layout.lin_state;
        g.lin_state_stride = a.layout.lin_state_stride;
        g.lin_conv_state = a.layout.lin_conv_state;
        g.lin_conv_stride = a.layout.lin_conv_stride;
        g.n_layers = a.layout.n_layers;
        g.full_attn_interval = a.layout.full_attn_interval;
    }
    g.sequences = a.geometry.sequences > 0 ? a.geometry.sequences : 1;
    g.streamed_bytes_per_token = a.geometry.streamed_bytes_per_token;
    return g;
}

// Roles as a comma-separated list of the names tensor.h parses. An unrecognised name is a
// failure, not a silently dropped role: a sweep that asked for "kv_cache" and got "nothing"
// would report a KV arm that never ran.
bool parse_roles(const char* text, RoleMask* out, std::string* bad) noexcept {
    RoleMask mask = RoleMask::none();
    std::string token;
    auto flush = [&]() -> bool {
        if (token.empty()) return true;
        TensorRole role{};
        if (!parse_tensor_role(token.c_str(), &role)) { *bad = token; return false; }
        mask.add(role);
        token.clear();
        return true;
    };
    for (const char* p = text; *p; ++p) {
        if (*p == ',' || *p == '+' || *p == ' ') { if (!flush()) return false; }
        else token.push_back(*p);
    }
    if (!flush()) return false;
    *out = mask;
    return true;
}

// The TensorTransit engine's dials. Everything the 0.1 engine already reads keeps its
// meaning -- hit ratio, budget fraction, prefetch distance, window attach, the mode -- so one
// command line drives both engines and an engine A/B is an engine A/B rather than a
// configuration change wearing one.
bool configure_transit(Adapter& a) noexcept {
    transit::Settings s{};
    TransitPlannerConfig cfg{};
    cfg.hit_ratio = a.config.hit_ratio;
    cfg.min_hit_ratio = a.config.min_hit_ratio;
    cfg.budget_fraction = a.config.persisting_budget_fraction;
    cfg.prefetch_distance = a.config.prefetch_distance;
    cfg.reuse_metric = ReuseMetric::Bytes;
    // ONE window per kernel, because that is what the hardware delivers: CUDA binds a single
    // access-policy window to a stream, a launch, or a graph node. A plan marking two regions
    // before one kernel would have the second silently replace the first while the telemetry
    // counted two. `widest` then selects the fp32 matrix state over the bf16 convolution
    // window on this model, which is what WindowTarget::Matrix -- the 0.1 default, and what
    // every number in results/ was measured under -- selects by a different route.
    cfg.max_windows_per_kernel = 1;
    cfg.window_preference = WindowPreference::Widest;

    const bool persist = a.config.mode == LocalityMode::Persist ||
                         a.config.mode == LocalityMode::Combined;
    const bool prefetch = a.config.mode == LocalityMode::Prefetch ||
                          a.config.mode == LocalityMode::Combined;
    cfg.persist_roles = persist ? RoleMask::of(TensorRole::RecurrentState) : RoleMask::none();
    cfg.prefetch_enabled = prefetch;
    cfg.prefetch_roles = prefetch ? RoleMask::of(TensorRole::RecurrentState) : RoleMask::none();
    cfg.stream_roles = RoleMask::none();
    s.planner = a.config.mode == LocalityMode::Baseline ? "baseline" : "recurrent_v0";

    // A preset is the specification's second proof track in one word. It sets the roles and
    // the admission rule TOGETHER, so the five arms cannot drift apart between runs.
    if (const char* preset_name = env_lookup("TENSORTRANSIT_PRESET")) {
        PolicyPreset preset{};
        if (!parse_policy_preset(preset_name, &preset)) {
            static std::string why;
            why = std::string("TENSORTRANSIT_PRESET=") + preset_name + ": unknown preset";
            a.config_error = why.c_str();
            return false;
        }
        cfg = preset_config(preset, cfg);
        cfg.max_windows_per_kernel = 1;
        cfg.window_preference = WindowPreference::Widest;
        s.planner = preset_planner(preset);
    }

    struct Parse { const char* name; bool (*fn)(Adapter&, const char*, TransitPlannerConfig&,
                                                transit::Settings&); };
    static const Parse kParsers[] = {
        {"TENSORTRANSIT_PLANNER", [](Adapter&, const char* v, TransitPlannerConfig&,
                                     transit::Settings& st) {
             for (const char* const* n = planner_names(); *n; ++n)
                 if (std::strcmp(v, *n) == 0) { st.planner = v; return true; }
             return false;
         }},
        {"TENSORTRANSIT_ADMISSION", [](Adapter&, const char* v, TransitPlannerConfig& c,
                                       transit::Settings&) {
             return parse_admission_rule(v, &c.admission);
         }},
        {"TENSORTRANSIT_REUSE_METRIC", [](Adapter&, const char* v, TransitPlannerConfig& c,
                                          transit::Settings&) {
             return parse_reuse_metric(v, &c.reuse_metric);
         }},
        {"TENSORTRANSIT_WINDOW_BINDING", [](Adapter&, const char* v, TransitPlannerConfig& c,
                                            transit::Settings&) {
             return parse_window_binding(v, &c.window_binding);
         }},
        {"TENSORTRANSIT_WINDOW_PREFERENCE", [](Adapter&, const char* v, TransitPlannerConfig& c,
                                               transit::Settings&) {
             return parse_window_preference(v, &c.window_preference);
         }},
        {"TENSORTRANSIT_PREFETCH_TIMING", [](Adapter&, const char* v, TransitPlannerConfig& c,
                                             transit::Settings&) {
             return parse_prefetch_timing(v, &c.prefetch_timing);
         }},
        {"TENSORTRANSIT_PERSIST_ROLES", [](Adapter&, const char* v, TransitPlannerConfig& c,
                                           transit::Settings&) {
             std::string bad;
             return parse_roles(v, &c.persist_roles, &bad);
         }},
        {"TENSORTRANSIT_PREFETCH_ROLES", [](Adapter&, const char* v, TransitPlannerConfig& c,
                                            transit::Settings&) {
             std::string bad;
             if (!parse_roles(v, &c.prefetch_roles, &bad)) return false;
             c.prefetch_enabled = !c.prefetch_roles.empty();
             return true;
         }},
        {"TENSORTRANSIT_STREAM_ROLES", [](Adapter&, const char* v, TransitPlannerConfig& c,
                                          transit::Settings&) {
             std::string bad;
             return parse_roles(v, &c.stream_roles, &bad);
         }},
    };
    for (const Parse& parser : kParsers) {
        const char* value = env_lookup(parser.name);
        if (!value) continue;
        if (!parser.fn(a, value, cfg, s)) {
            static std::string why;
            why = std::string(parser.name) + "=" + value + ": unrecognised value";
            a.config_error = why.c_str();
            return false;
        }
    }

    cfg.role_floor_share = env_double("TENSORTRANSIT_ROLE_FLOOR_SHARE", cfg.role_floor_share);
    cfg.max_reuse_distance_budgets =
        env_double("TENSORTRANSIT_MAX_REUSE_BUDGETS", cfg.max_reuse_distance_budgets);
    cfg.max_windows_per_kernel =
        static_cast<int>(env_int("TENSORTRANSIT_MAX_WINDOWS_PER_KERNEL",
                                 cfg.max_windows_per_kernel));
    cfg.prefetch_min_bytes = static_cast<std::size_t>(
        env_int("TENSORTRANSIT_PREFETCH_MIN_BYTES",
                static_cast<long long>(cfg.prefetch_min_bytes)));
    cfg.prefetch_join_at_end = a.config.prefetch_join == PrefetchJoin::TokenEnd;

    if (const char* problem = validate(cfg)) {
        a.config_error = problem;
        return false;
    }
    s.planner_config = cfg;
    s.attach = a.config.window_attach;
    s.budget_fraction = a.config.persisting_budget_fraction;
    s.register_kv = env_int("TENSORTRANSIT_REGISTER_KV", 1) != 0;
    a.transit_settings = s;
    return true;
}

void configure_once() noexcept {
    Adapter& a = adapter();
    if (a.configured) return;
    a.configured = true;

    a.mode = env_or("RECURLOCAL", "off");
    if (a.mode == "off" || a.mode == "0") return;

    const char* why = nullptr;
    PlannerConfig cfg{};
    bool ok = parse_or_fail("RECURLOCAL", "off",
                            [&](const char* t) { cfg.mode = parse_mode(t); }, &why);
    ok = ok && parse_or_fail("RECURLOCAL_PRE_TOUCH", "vec4",
                             [&](const char* t) { cfg.pre_touch = parse_pre_touch_strategy(t); }, &why);
    ok = ok && parse_or_fail("RECURLOCAL_HOT_SET_POLICY", "proportional",
                             [&](const char* t) { cfg.hot_set_policy = parse_hot_set_policy(t); }, &why);
    ok = ok && parse_or_fail("RECURLOCAL_PREFETCH_SCHEDULE", "uniform",
                             [&](const char* t) { cfg.prefetch_schedule = parse_prefetch_schedule(t); }, &why);
    ok = ok && parse_or_fail("RECURLOCAL_HOT_SET_MODEL", "token_footprint",
                             [&](const char* t) { cfg.hot_set_model = parse_hot_set_model(t); }, &why);
    ok = ok && parse_or_fail("RECURLOCAL_WINDOW_SCOPE", "layer",
                             [&](const char* t) { cfg.window_scope = parse_window_scope(t); }, &why);
    ok = ok && parse_or_fail("RECURLOCAL_WINDOW_TARGET", "matrix",
                             [&](const char* t) { cfg.window_target = parse_window_target(t); }, &why);
    ok = ok && parse_or_fail("RECURLOCAL_PRE_TOUCH_COVERAGE", "both",
                             [&](const char* t) { cfg.pre_touch_coverage = parse_pre_touch_coverage(t); }, &why);
    ok = ok && parse_or_fail("RECURLOCAL_PREFETCH_JOIN", "per_layer",
                             [&](const char* t) { cfg.prefetch_join = parse_prefetch_join(t); }, &why);
    ok = ok && parse_or_fail("RECURLOCAL_WINDOW_ATTACH", "stream",
                             [&](const char* t) { cfg.window_attach = parse_window_attach(t); }, &why);
    ok = ok && parse_or_fail("RECURLOCAL_SET_ASIDE_POLICY", "fixed",
                             [&](const char* t) { cfg.set_aside_policy = parse_set_aside_policy(t); }, &why);
    if (!ok) {
        // Loud, immediately, and not only in the stats: a configuration the adapter cannot
        // parse leaves the hook off, and a sweep would then measure the control while
        // labelling it as the candidate it asked for. That is worse than a crash.
        a.config_error = why;
        std::fprintf(stderr, "[recurlocal] DISABLED - bad configuration: %s\n", why);
        return;
    }

    cfg.prefetch_distance = static_cast<int>(env_int("RECURLOCAL_PREFETCH_DISTANCE", 1));
    cfg.hit_ratio = env_double("RECURLOCAL_HIT_RATIO", 0.70);
    cfg.persisting_budget_fraction = env_double("RECURLOCAL_BUDGET_FRACTION", 0.75);
    cfg.min_hit_ratio = env_double("RECURLOCAL_MIN_HIT_RATIO", 0.05);
    cfg.min_residency = env_double("RECURLOCAL_MIN_RESIDENCY", 0.50);
    cfg.max_hot_window_bytes = static_cast<std::size_t>(env_int("RECURLOCAL_MAX_WINDOW_BYTES", 0));
    if (const char* problem = validate(cfg)) {
        a.config_error = problem;
        std::fprintf(stderr, "[recurlocal] DISABLED - invalid planner config: %s\n", problem);
        return;
    }

    a.config = cfg;

    // Engine selection LAST, so it reads a validated 0.1 config: every dial the two engines
    // share is parsed once, in one place, and cannot mean two things.
    const std::string engine = env_or("TENSORTRANSIT_ENGINE", "transit");
    if (engine == "transit") {
        a.use_transit = true;
    } else if (engine == "v0" || engine == "recurlocal") {
        a.use_transit = false;
    } else {
        static std::string why;
        why = "TENSORTRANSIT_ENGINE=" + engine + ": expected transit or v0";
        a.config_error = why.c_str();
        std::fprintf(stderr, "[recurlocal] DISABLED - %s\n", a.config_error);
        return;
    }
    if (a.use_transit && !configure_transit(a)) {
        std::fprintf(stderr, "[recurlocal] DISABLED - bad transit configuration: %s\n",
                     a.config_error ? a.config_error : "unknown");
        return;
    }
    a.usable = true;
}

} // namespace

bool enabled() noexcept { configure_once(); return adapter().usable; }

const char* mode_name() noexcept {
    configure_once();
    return adapter().usable ? to_string(adapter().config.mode) : "off";
}

void declare_geometry_now(Adapter& a) noexcept;

bool begin_token(const GdnStateLayout& layout) noexcept {
    Adapter& a = adapter();
    configure_once();
    if (!a.usable || a.broken) return false;
    if (!layout.compute || layout.n_layers <= 0) return false;

    if (!a.initialised) {
        // Everything here is illegal or wasteful inside a CUDA Graph capture - reserving
        // the L2 set-aside, allocating the pre-touch scratch, creating a stream - which is
        // why begin_token() is specified to be called before capture begins.
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) { cudaGetLastError(); a.broken = true; return false; }
        // Non-blocking so the pre-touch never serialises against the legacy default stream,
        // and at the compute stream's own priority: a prefetch that outranks the kernel it
        // is meant to help takes SMs away from it.
        //
        // Created BEFORE either engine initialises, because the transit executor binds both
        // streams in one call and a null prefetch stream there means Prefetch actions are
        // skipped rather than run on the critical path wearing a prefetch's counter.
        int least = 0, greatest = 0;
        cudaDeviceGetStreamPriorityRange(&least, &greatest);
        int compute_priority = least;
        cudaStreamGetPriority(layout.compute, &compute_priority);
        if (cudaStreamCreateWithPriority(&a.prefetch_stream, cudaStreamNonBlocking,
                                         compute_priority) != cudaSuccess) {
            cudaGetLastError(); a.broken = true; return false;
        }
        if (a.use_transit) {
            if (!a.transit_engine.initialize(device, layout.compute, a.prefetch_stream,
                                             a.transit_settings)) {
                std::fprintf(stderr, "[recurlocal] DISABLED - transit engine: %s\n",
                             a.transit_engine.error() ? a.transit_engine.error() : "unknown");
                a.broken = true;
                return false;
            }
        } else {
            if (a.controller.initialize(device, a.config) != cudaSuccess) { a.broken = true; return false; }
            if (a.controller.bind_streams(layout.compute, a.prefetch_stream) != cudaSuccess) {
                a.broken = true; return false;
            }
        }
        a.initialised = true;
        a.ever_initialised = true;
        a.l2_set_aside_at_init = a.use_transit ? a.transit_engine.set_aside_at_init_bytes()
                                               : a.controller.l2_set_aside_bytes();
        // The line PREFIX stays `RECURLOCAL_STATS` on purpose, and it is the one name in
        // this migration that is not being moved forward.
        //
        // `eval/run_from_base.sh` runs the instrument from the BASE commit against a
        // candidate's binary -- that is the whole point of it, so a one-line change to a
        // noise floor or an estimator cannot ride in on a submission. The base commit's
        // parser knows only this prefix. Renaming the emitted line would make every
        // base-commit evaluator report "the binary is unhooked" against a perfectly good
        // build, which is a false accusation aimed at the contributor. The reader accepts
        // both spellings; the writer keeps the one the old readers know.
        //
        // TENSORTRANSIT_STATS=1 turns it on, as does RECURLOCAL_STATS=1.
        // runtime to call for them. A benchmark harness reads them off stderr, and a run
        // whose stats line is missing was a run with no hook - which is exactly what the
        // control arm should look like in a result file.
        if (const char* want = env_lookup("RECURLOCAL_STATS"); want && want[0] != '0')
            std::atexit([] {
                std::fputs("RECURLOCAL_STATS ", stderr);
                write_stats_json(stderr);
                std::fputc('\n', stderr);
            });
    }

    a.packed = false;
    // A second model instance would share this singleton's per-layer state with the first.
    if (a.bound_stream && a.bound_stream != layout.compute) {
        if (!a.broken) {
            std::fprintf(stderr,
                "[recurlocal] DISABLED - a second compute stream (%p, was %p) reached the hook. "
                "The adapter holds one model's walk state per process; sharing it would "
                "interleave two models' layers and produce quietly wrong numbers.\n",
                (void*)layout.compute, (void*)a.bound_stream);
        }
        a.broken = true;
        return false;
    }
    a.bound_stream = layout.compute;

    a.layout = layout;
    begin_common(a, layout.n_layers, layout.full_attn_interval,
                 layout.lin_state_stride + layout.lin_conv_stride, /*sequences=*/1);
    // Not when the packed path is borrowing this entry point for its stream setup: it is
    // about to overwrite the sequence count, and declaring here would size the reservation
    // for one sequence and then again for N, on every single token.
    if (!a.declare_suppressed) declare_geometry_now(a);
    return true;
}

// The set-aside was requested at initialize() from the config alone, before anything was
// known about the workload. A workload-aware SetAsidePolicy sizes it from the footprint, so
// it has to be told the footprint - and told it ONCE the geometry is final. Declaring it
// inside begin_common() ran before the packed path had overwritten the sequence count, so a
// concurrent step declared `sequences = 1` and every workload-aware rule sized the
// reservation for a workload that was not running. The controller ignores this under the
// default policy and whenever the answer has not changed.
void declare_geometry_now(Adapter& a) noexcept {
    // One call site, two engines, and the same contract: the geometry is final HERE, after
    // the packed path has overwritten the sequence count. Declaring inside begin_common()
    // sized the reservation for one sequence and then again for N, twice per token, forever.
    if (a.use_transit) { a.transit_engine.begin_token(step_geometry(a), a.kv); return; }
    a.controller.declare_geometry(a.geometry);
}

bool begin_token_packed(const GdnPackedLayout& layout) noexcept {
    Adapter& a = adapter();
    // The single-sequence layout is what carries the compute stream through to
    // initialisation, and the packed path runs on the same stream, so reuse that entry
    // point rather than duplicating the setup.
    GdnStateLayout as_single;
    as_single.lin_state = layout.host_lin_state_row0;
    as_single.lin_state_stride = layout.lin_state_stride;
    as_single.lin_conv_state = layout.host_lin_conv_row0;
    as_single.lin_conv_stride = layout.lin_conv_stride;
    as_single.n_layers = layout.n_layers;
    as_single.full_attn_interval = layout.full_attn_interval;
    as_single.compute = layout.compute;
    a.declare_suppressed = true;
    const bool ok = begin_token(as_single);
    a.declare_suppressed = false;
    if (!ok) return false;

    a.packed = true;
    a.packed_layout = layout;
    ++a.tokens_packed;
    if (layout.rows > a.max_rows_seen) a.max_rows_seen = layout.rows;
    // At concurrency the sequence count is not a declaration, it is observed: this is how
    // many sequences the runtime is actually decoding in this step.
    a.geometry.sequences = layout.rows > 0 ? layout.rows : 1;
    if (a.geometry.sequences > a.max_geometry_sequences) {
        a.max_geometry_sequences = a.geometry.sequences;
        a.max_geometry_bytes_per_layer = a.geometry.bytes_per_layer;
    }
    declare_geometry_now(a);
    return true;
}

// Packed step: the window still goes on a host-nameable allocation (row 0's), because an
// access-policy window is an address range and there is only one to give. What changes is
// the pre-touch, which walks every sequence's slice in one launch per state.
void before_layer_packed(Adapter& a, int layer) noexcept {
    const GdnPackedLayout& l = a.packed_layout;
    StateSegment window[2];
    GdnStateLayout row0;
    row0.lin_state = l.host_lin_state_row0;
    row0.lin_state_stride = l.lin_state_stride;
    row0.lin_conv_state = l.host_lin_conv_row0;
    row0.lin_conv_stride = l.lin_conv_stride;
    row0.n_layers = l.n_layers;
    const int window_count = segments_for(row0, layer, window);

    const int distance = a.controller.planner().distance_for_layer(a.recurrent_ordinal);
    const int ahead = recurrent_layer_ahead(layer, distance, l.n_layers, l.full_attn_interval);

    CudaLocalityController::RowSet rows[2];
    int row_sets = 0;
    if (ahead >= 0) {
        if (l.device_lin_state && l.lin_state_stride) {
            rows[row_sets].device_bases = l.device_lin_state;
            rows[row_sets].rows = l.rows;
            rows[row_sets].byte_offset = static_cast<std::size_t>(ahead) * l.lin_state_stride;
            rows[row_sets].bytes = l.lin_state_stride;
            rows[row_sets].kind = StateKind::Matrix;
            ++row_sets;
        }
        if (l.device_lin_conv && l.lin_conv_stride) {
            rows[row_sets].device_bases = l.device_lin_conv;
            rows[row_sets].rows = l.rows;
            rows[row_sets].byte_offset = static_cast<std::size_t>(ahead) * l.lin_conv_stride;
            rows[row_sets].bytes = l.lin_conv_stride;
            rows[row_sets].kind = StateKind::Conv;
            ++row_sets;
        }
    }

    LayerActions actions{};
    a.controller.before_layer(window_count ? window : nullptr, window_count,
                              row_sets ? rows : nullptr, row_sets,
                              /*has_next_recurrent_layer=*/ahead >= 0,
                              a.geometry, &actions);
    ++a.recurrent_ordinal;
    a.window_pending = actions.window_requires_launch_attribute;
    a.pending_kernel = pending_kernel_for(row0, actions.window_region);
}

void before_layer(int layer) noexcept {
    Adapter& a = adapter();
    if (!a.initialised || a.broken) return;
    if (a.use_transit) {
        if (a.packed) ++a.layers_packed;
        ++a.recurrent_ordinal;
        a.last_layer = layer;
        a.transit_engine.before_layer(layer);
        return;
    }
    if (a.packed) { ++a.layers_packed; before_layer_packed(a, layer); return; }

    StateSegment current[2];
    const int current_count = segments_for(a.layout, layer, current);
    if (current_count == 0) return;

    // The runtime, not the planner, has to produce the state the schedule wants to reach,
    // so ask how far ahead this recurrent layer prefetches before gathering anything.
    const int distance = a.controller.planner().distance_for_layer(a.recurrent_ordinal);
    StateSegment next[2];
    int next_count = 0;
    const int ahead = recurrent_layer_ahead(layer, distance, a.layout.n_layers,
                                            a.layout.full_attn_interval);
    if (ahead >= 0) next_count = segments_for(a.layout, ahead, next);

    LayerActions actions{};
    a.controller.before_layer(current, current_count,
                              next_count ? next : nullptr, next_count,
                              /*has_next_recurrent_layer=*/ahead >= 0,
                              a.geometry, &actions);
    ++a.recurrent_ordinal;

    a.window_pending = actions.window_requires_launch_attribute;
    a.pending_kernel = pending_kernel_for(a.layout, actions.window_region);
}

void after_launch(StateKernel which) noexcept {
    Adapter& a = adapter();
    if (!a.initialised || a.broken) return;
    if (a.use_transit) { a.transit_engine.after_launch(which); return; }
    if (!a.window_pending) return;
    if (which != a.pending_kernel) return;
    a.window_pending = false;
    a.controller.attach_window_to_captured_node();
}

void after_layer() noexcept {
    Adapter& a = adapter();
    if (!a.initialised || a.broken) return;
    a.window_pending = false;
    if (a.use_transit) { a.transit_engine.after_layer(a.last_layer); return; }
    a.controller.after_layer();
}

// The paged KV pools, held until the next declare_geometry_now() reads them. Recorded rather
// than acted on immediately because the geometry is only final after the packed path has
// settled the sequence count, and a KV footprint declared against the wrong row count is the
// same accounting error the recurrent side already made once.
void declare_kv_cache(const KvCacheLayout& kv) noexcept {
    Adapter& a = adapter();
    configure_once();
    if (!a.usable || a.broken) return;
    transit::KvGeometry g{};
    g.k_pool = kv.k_pool;
    g.v_pool = kv.v_pool;
    g.pool_bytes = kv.pool_bytes;
    g.layer_stride_bytes = kv.layer_stride_bytes;
    g.live_bytes_per_layer = kv.live_bytes_per_layer;
    g.kv_slots = kv.kv_slots;
    g.rows = kv.rows > 0 ? kv.rows : 1;
    a.kv = g;
    a.kv_declared = g.valid();
}

// A FULL-ATTENTION layer. The 0.1 engine has no policy for one and never sees it; routing
// these through before_layer()/after_layer() would change what every published number was
// measured under, so they are a separate pair rather than a flag.
void before_attention_layer(int layer) noexcept {
    Adapter& a = adapter();
    if (!a.initialised || a.broken || !a.use_transit) return;
    a.last_layer = layer;
    a.transit_engine.before_layer(layer);
}

void after_attention_layer(int layer) noexcept {
    Adapter& a = adapter();
    if (!a.initialised || a.broken || !a.use_transit) return;
    a.transit_engine.after_layer(layer);
}

void end_token() noexcept {
    Adapter& a = adapter();
    if (!a.initialised || a.broken) return;
    if (a.use_transit) { a.transit_engine.end_token(); return; }
    a.controller.end_sequence();
}

void write_transit_stats_json(std::FILE* out) noexcept;

void write_stats_json(std::FILE* out) noexcept {
    if (!out) return;
    Adapter& a = adapter();
    configure_once();
    // One stats line, two engines, the SAME field names.
    //
    // Not tidiness: `eval/real_eval.py` refuses a candidate whose telemetry reads
    // windows_applied 0, windows_attached_to_node 0, pre_touch_launches 0 -- the first scored
    // run in this repository was exactly that -- and `eval/run_from_base.sh` runs the
    // evaluator from the BASE commit, whose parser knows only these names. An engine that
    // emitted its own vocabulary would be reported as unhooked by every base-commit
    // evaluator against a perfectly good build, which is a false accusation aimed at the
    // contributor. So the transit engine's counters are mapped onto the 0.1 line, and what
    // has no 0.1 counterpart goes in the "transit" block below rather than being squeezed
    // into a field that means something else.
    ControllerStats s = a.controller.stats();
    if (a.use_transit) {
        const ExecutorStats x = a.transit_engine.total_executor_stats();
        const transit::Counters& t = a.transit_engine.counters();
        s = ControllerStats{};
        s.layers = t.layers;
        s.windows_applied = x.persist_applied;
        s.windows_deferred_to_caller = x.persist_deferred;
        s.windows_attached_to_node = x.persist_attached_to_node;
        s.window_nodes_attached = x.persist_nodes_attached;
        // The transit executor has no CaptureNodeStrict mode, so it never declines an attach
        // for ambiguity. Reporting the wrong-kernel skips here instead would put a different
        // question's answer under this name; they are in the transit block.
        s.window_attach_ambiguous = 0;
        s.window_attach_failures = x.actions_failed;
        s.capture_invalidations = x.capture_invalidations;
        s.released_during_capture = x.released_during_capture;
        s.hit_ratio_reduced = t.hit_ratio_reduced;
        s.hot_set_oversubscribed = t.hot_set_oversubscribed;
        s.pre_touch_launches = x.prefetch_applied;
        s.pre_touch_segments = x.prefetch_applied;
        s.pre_touch_bytes = x.prefetch_bytes;
        s.pre_touch_skipped = x.prefetch_skipped;
        s.compute_stream_priority = a.controller.stats().compute_stream_priority;
        s.prefetch_stream_priority = a.controller.stats().prefetch_stream_priority;
    }
    const auto& c = a.config;
    std::fprintf(out,
        "{\"adapter\":\"tensortransit-sparkinfer\","
        "\"mode\":\"%s\",\"initialised\":%s,\"ever_initialised\":%s,"
        "\"broken\":%s,\"config_error\":%s,"
        "\"config\":{\"pre_touch\":\"%s\",\"pre_touch_coverage\":\"%s\",\"prefetch_distance\":%d,"
        "\"prefetch_schedule\":\"%s\",\"prefetch_join\":\"%s\",\"window_attach\":\"%s\","
        "\"hot_set_model\":\"%s\","
        "\"hot_set_policy\":\"%s\",\"window_scope\":\"%s\",\"window_target\":\"%s\","
        "\"set_aside_policy\":\"%s\",\"min_residency\":%.4f,"
        "\"hit_ratio\":%.4f,\"budget_fraction\":%.4f},"
        "\"geometry\":{\"recurrent_layers\":%d,\"bytes_per_layer\":%zu,\"sequences\":%d,"
        "\"max_sequences_declared\":%d,\"bytes_per_layer_at_max\":%zu,"
        "\"streamed_bytes_per_token\":%zu},"
        "\"l2_set_aside_bytes\":%zu,\"l2_set_aside_at_init_bytes\":%zu,"
        "\"stats\":{\"tokens\":%llu,\"tokens_packed\":%llu,\"layers\":%llu,"
        "\"layers_packed\":%llu,\"max_rows_seen\":%d,\"windows_applied\":%llu,"
        "\"windows_deferred_to_caller\":%llu,\"windows_attached_to_node\":%llu,"
        "\"window_nodes_attached\":%llu,\"window_attach_ambiguous\":%llu,"
        "\"window_attach_failures\":%llu,\"capture_invalidations\":%llu,"
        "\"released_during_capture\":%llu,\"hit_ratio_reduced\":%llu,"
        "\"hot_set_oversubscribed\":%llu,"
        "\"pre_touch_launches\":%llu,\"pre_touch_segments\":%llu,\"pre_touch_bytes\":%llu,"
        "\"pre_touch_skipped\":%llu,\"compute_stream_priority\":%d,"
        "\"prefetch_stream_priority\":%d}",
        a.usable ? to_string(c.mode) : "off",
        a.initialised ? "true" : "false", a.ever_initialised ? "true" : "false",
        a.broken ? "true" : "false",
        a.config_error ? "\"set\"" : "null",
        to_string(c.pre_touch), to_string(c.pre_touch_coverage), c.prefetch_distance,
        to_string(c.prefetch_schedule), to_string(c.prefetch_join), to_string(c.window_attach),
        to_string(c.hot_set_model),
        to_string(c.hot_set_policy), to_string(c.window_scope), to_string(c.window_target),
        to_string(c.set_aside_policy), c.min_residency,
        c.hit_ratio, c.persisting_budget_fraction,
        a.geometry.recurrent_layers, a.geometry.bytes_per_layer, a.geometry.sequences,
        a.max_geometry_sequences, a.max_geometry_bytes_per_layer,
        a.geometry.streamed_bytes_per_token,
        // The LARGEST set-aside held during the run. Reporting only the init-time value made
        // every resize by a workload-aware policy invisible to every sweep; reporting the
        // value in force at exit reports 0, because shutdown has already given the partition
        // back by the time an atexit handler runs.
        a.use_transit ? a.transit_engine.set_aside_peak_bytes()
                      : a.controller.l2_set_aside_peak_bytes(),
        a.ever_initialised ? a.l2_set_aside_at_init
                           : (a.use_transit ? a.transit_engine.set_aside_peak_bytes()
                                            : a.controller.l2_set_aside_bytes()),
        (unsigned long long)a.tokens, (unsigned long long)a.tokens_packed,
        (unsigned long long)s.layers, (unsigned long long)a.layers_packed, a.max_rows_seen,
        (unsigned long long)s.windows_applied, (unsigned long long)s.windows_deferred_to_caller,
        (unsigned long long)s.windows_attached_to_node,
        (unsigned long long)s.window_nodes_attached, (unsigned long long)s.window_attach_ambiguous,
        (unsigned long long)s.window_attach_failures,
        (unsigned long long)s.capture_invalidations,
        (unsigned long long)s.released_during_capture, (unsigned long long)s.hit_ratio_reduced,
        (unsigned long long)s.hot_set_oversubscribed, (unsigned long long)s.pre_touch_launches,
        (unsigned long long)s.pre_touch_segments, (unsigned long long)s.pre_touch_bytes,
        (unsigned long long)s.pre_touch_skipped,
        s.compute_stream_priority, s.prefetch_stream_priority);
    write_transit_stats_json(out);
    std::fputc('}', out);
    if (a.config_error) std::fprintf(stderr, "[recurlocal] bad configuration: %s\n", a.config_error);
}

// What the TensorTransit engine did that the 0.1 line has no field for. Emitted as its own
// object, appended to the same line, so an older reader ignores it and a newer one gets the
// plan digest, the planner name, the recompile behaviour and the decline census -- which is
// the difference between "the policy did not help" and "the policy was never admitted".
void write_transit_stats_json(std::FILE* out) noexcept {
    Adapter& a = adapter();
    if (!out || !a.use_transit) return;
    const transit::Engine& e = a.transit_engine;
    const ExecutorStats x = e.total_executor_stats();
    const RuntimeStats r = e.total_runtime_stats();
    const TransitPlannerConfig& cfg = a.transit_settings.planner_config;
    static const char* const kDecline[] = {"none", "no_reuse", "budget_exhausted", "too_large",
                                           "reuse_too_far", "role_excluded",
                                           "below_min_hit_ratio", "not_supported"};
    std::fprintf(out,
        ",\"transit\":{\"engine\":\"transit\",\"planner\":\"%s\","
        "\"admission\":\"%s\",\"reuse_metric\":\"%s\",\"window_binding\":\"%s\","
        "\"window_preference\":\"%s\",\"max_windows_per_kernel\":%d,"
        "\"plan_digest\":\"%016llx\","
        "\"recurrent_tensors\":%d,\"kv_tensors\":%d,\"kv_declared\":%s,"
        "\"committed_bytes\":%zu,\"predicted_saved_bytes\":%zu,"
        "\"step_traffic_bytes\":%zu,"
        "\"compiles\":%llu,\"plan_reuses\":%llu,\"reuse_rate\":%.6f,"
        "\"geometry_rebuilds\":%llu,\"last_recompile_reason\":\"%s\","
        "\"planner_ns\":%llu,\"executor_ns\":%llu,\"record_ns\":%llu,"
        "\"attach_calls\":%llu,\"attach_skipped_wrong_kernel\":%llu,"
        "\"stale_tensor_refs\":%llu,\"actions_skipped\":%llu,"
        "\"stream_applied\":%llu,\"clear_applied\":%llu,"
        "\"executor_steps\":%llu,\"executor_kernels\":%llu,\"declines\":{",
        e.planner_name().c_str(), to_string(cfg.admission), to_string(cfg.reuse_metric),
        to_string(cfg.window_binding), to_string(cfg.window_preference),
        cfg.max_windows_per_kernel,
        (unsigned long long)e.plan_digest(),
        e.recurrent_layers(), e.kv_layers(), a.kv_declared ? "true" : "false",
        e.committed_bytes(), e.predicted_saved_bytes(), e.step_traffic_bytes(),
        (unsigned long long)r.compiles, (unsigned long long)r.plan_reuses, r.reuse_rate(),
        (unsigned long long)e.counters().recompiles_geometry, to_string(r.last_reason),
        (unsigned long long)r.compile_ns, (unsigned long long)x.host_ns,
        (unsigned long long)r.record_ns,
        (unsigned long long)e.counters().attach_calls,
        (unsigned long long)e.counters().attach_skipped_wrong_kernel,
        (unsigned long long)x.stale_tensor_refs, (unsigned long long)x.actions_skipped,
        (unsigned long long)x.stream_applied, (unsigned long long)x.clear_applied,
        (unsigned long long)x.steps, (unsigned long long)x.kernels);
    for (std::size_t i = 0; i < e.declines_size() && i < 8; ++i)
        std::fprintf(out, "%s\"%s\":%llu", i ? "," : "", kDecline[i],
                     (unsigned long long)e.declines()[i]);
    std::fputs("}}", out);   // declines, then the transit object itself
}

void shutdown() noexcept {
    Adapter& a = adapter();
    if (!a.initialised) return;
    // Both engines hand the device-wide L2 partition back BEFORE the streams they borrowed
    // are destroyed. The other order leaves the executor probing a destroyed stream inside
    // libcuda, which is a crash in the teardown path of an optional optimization layer.
    if (a.use_transit) a.transit_engine.shutdown();
    else a.controller.reset();
    if (a.prefetch_stream) { cudaStreamDestroy(a.prefetch_stream); a.prefetch_stream = nullptr; }
    a.initialised = false;
}

} // namespace sparkinfer
} // namespace tensortransit
#endif
