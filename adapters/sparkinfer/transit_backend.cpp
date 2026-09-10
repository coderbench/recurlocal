#include "transit_backend.h"

#if defined(TENSORTRANSIT_WITH_CUDA) || defined(RECURLOCAL_WITH_CUDA) || \
    defined(RECURLLOCAL_WITH_CUDA)
#include <algorithm>
#include <cstring>

#include "tensortransit/trace.h"

namespace tensortransit {
namespace sparkinfer {
namespace transit {
namespace {

// The graph's kernel ids are the runtime's own absolute layer indices, offset by one so 0
// stays the invalid id. That is what makes `before_layer(L)` a lookup rather than a search,
// and what puts a hybrid model's attention layers in the right ORDER between two recurrent
// ones -- reuse distance in bytes is a question about what runs in between, and a graph that
// recorded only the recurrent layers would answer it with the attention traffic missing.
KernelId kernel_for_layer(int layer) noexcept {
    return static_cast<KernelId>(layer) + 1;
}

}  // namespace

bool Engine::Signature::operator==(const Signature& o) const noexcept {
    return lin_state == o.lin_state && lin_conv == o.lin_conv &&
           lin_state_stride == o.lin_state_stride && lin_conv_stride == o.lin_conv_stride &&
           k_pool == o.k_pool && v_pool == o.v_pool && kv_live == o.kv_live &&
           n_layers == o.n_layers && full_attn_interval == o.full_attn_interval &&
           sequences == o.sequences && kv_slots == o.kv_slots && streamed == o.streamed;
}

Engine::~Engine() { shutdown(); }

bool Engine::initialize(int device, cudaStream_t compute, cudaStream_t prefetch,
                        const Settings& settings) noexcept {
    settings_ = settings;
    device_ = device;
    compute_ = compute;

    DeviceProfile profile{};
    if (query_device_profile(device, &profile) != cudaSuccess) {
        error_ = "query_device_profile failed";
        return false;
    }
    runtime_.set_device_profile(profile);
    if (!runtime_.set_planner(settings_.planner.c_str(), settings_.planner_config)) {
        // A silent fall back to baseline would report a measurement of the wrong planner,
        // which is the failure this whole layer exists to make impossible.
        error_ = "unknown planner name";
        return false;
    }

    // The set-aside the config asks for, resolved against what the device will grant. Asked
    // for HERE and not by the planner: it is a device-wide, context-lifetime reservation and
    // the executor is the only object that gives it back.
    std::size_t want = profile.persisting_l2_max_bytes;
    if (settings_.budget_fraction < 1.0)
        want = static_cast<std::size_t>(static_cast<double>(want) * settings_.budget_fraction);
    if (executor_.initialize(device, want) != cudaSuccess) {
        error_ = "executor initialize failed";
        return false;
    }
    set_aside_at_init_ = executor_.set_aside_bytes();

    if (prefetch && executor_.bind_streams(compute, prefetch) != cudaSuccess) {
        error_ = "bind_streams failed";
        return false;
    }
    if (!prefetch) {
        // Without a second stream a Prefetch action would run on the compute stream, which is
        // not a prefetch: it is extra work on the critical path wearing a prefetch's counter.
        // The executor already skips it, and this records why.
        error_ = nullptr;
    }
    executor_.set_registry(&runtime_.registry());
    if (settings_.planner_config.prefetch_enabled && settings_.scratch_floats) {
        if (cudaMalloc(reinterpret_cast<void**>(&scratch_),
                       settings_.scratch_floats * sizeof(float)) == cudaSuccess)
            executor_.set_scratch(scratch_, settings_.scratch_floats);
        else
            cudaGetLastError();
    }
    runtime_.set_executor(&executor_);
    initialised_ = true;
    return true;
}

namespace {
// Sum every counter. Written out rather than memcpy'd so that a field appended to
// ExecutorStats fails to compile here instead of silently going uncounted.
void accumulate(ExecutorStats* into, const ExecutorStats& add) noexcept {
    into->steps += add.steps;
    into->kernels += add.kernels;
    into->actions_applied += add.actions_applied;
    into->actions_skipped += add.actions_skipped;
    into->actions_failed += add.actions_failed;
    into->persist_applied += add.persist_applied;
    into->persist_bytes += add.persist_bytes;
    into->stream_applied += add.stream_applied;
    into->prefetch_applied += add.prefetch_applied;
    into->prefetch_bytes += add.prefetch_bytes;
    into->clear_applied += add.clear_applied;
    into->events_recorded += add.events_recorded;
    into->events_waited += add.events_waited;
    into->persist_deferred += add.persist_deferred;
    into->persist_attached_to_node += add.persist_attached_to_node;
    into->persist_nodes_attached += add.persist_nodes_attached;
    into->stale_tensor_refs += add.stale_tensor_refs;
    into->host_ns += add.host_ns;
    into->capture_invalidations += add.capture_invalidations;
    into->released_during_capture += add.released_during_capture;
    into->prefetch_skipped += add.prefetch_skipped;
}
void accumulate(RuntimeStats* into, const RuntimeStats& add) noexcept {
    into->compiles += add.compiles;
    into->plan_reuses += add.plan_reuses;
    into->steps += add.steps;
    into->compile_ns += add.compile_ns;
    into->record_ns += add.record_ns;
    into->last_reason = add.last_reason;
}
}  // namespace

ExecutorStats Engine::total_executor_stats() const noexcept {
    ExecutorStats total = retired_executor_;
    accumulate(&total, executor_.stats());
    return total;
}

RuntimeStats Engine::total_runtime_stats() const noexcept {
    RuntimeStats total = retired_runtime_;
    accumulate(&total, runtime_.stats());
    return total;
}

void Engine::rebuild(const StepGeometry& geometry, const KvGeometry& kv) noexcept {
    // Retire the counters BEFORE reset() clears them. Everything below this line is about a
    // new plan; everything above it already happened and still has to be reportable.
    accumulate(&retired_executor_, executor_.stats());
    accumulate(&retired_runtime_, runtime_.stats());
    // Registry and graph are rebuilt together: a tensor set and a kernel sequence that
    // disagreed would produce uses the graph drops into `unresolved_uses`, and a plan built
    // from a graph that quietly lost half its demand.
    runtime_.reset();
    executor_.set_registry(&runtime_.registry());
    runtime_.set_executor(&executor_);
    window_kind_.assign(static_cast<std::size_t>(geometry.n_layers), -1);
    window_reduced_.assign(static_cast<std::size_t>(geometry.n_layers), false);
    recurrent_layers_ = 0;
    kv_layers_ = 0;

    auto* state_base = static_cast<unsigned char*>(geometry.lin_state);
    auto* conv_base = static_cast<unsigned char*>(geometry.lin_conv_state);
    const std::size_t state_alloc =
        static_cast<std::size_t>(geometry.n_layers) * geometry.lin_state_stride;
    const std::size_t conv_alloc =
        static_cast<std::size_t>(geometry.n_layers) * geometry.lin_conv_stride;

    struct LayerTensors {
        TensorId matrix = kInvalidTensorId;
        TensorId conv = kInvalidTensorId;
        TensorId k = kInvalidTensorId;
        TensorId v = kInvalidTensorId;
        TensorId weight = kInvalidTensorId;
    };
    std::vector<LayerTensors> per_layer(static_cast<std::size_t>(geometry.n_layers));

    // Per-layer weight traffic, only when the operator declared a step figure. Registering a
    // fabricated one would put an invented denominator into every ceiling this graph feeds.
    const std::size_t weight_per_layer =
        geometry.streamed_bytes_per_token && geometry.n_layers
            ? geometry.streamed_bytes_per_token / static_cast<std::size_t>(geometry.n_layers)
            : 0;

    int kv_slot = 0;
    for (int layer = 0; layer < geometry.n_layers; ++layer) {
        LayerTensors& t = per_layer[static_cast<std::size_t>(layer)];
        if (is_recurrent_layer(layer, geometry.full_attn_interval)) {
            ++recurrent_layers_;
            if (state_base && geometry.lin_state_stride) {
                TensorDesc d{};
                d.ptr = state_base + static_cast<std::size_t>(layer) * geometry.lin_state_stride;
                d.bytes = geometry.lin_state_stride;
                d.base = state_base;
                d.base_bytes = state_alloc;
                d.role = TensorRole::RecurrentState;
                d.mutable_data = true;
                d.request_local = true;
                d.device = device_;
                t.matrix = runtime_.register_tensor(d).id;
            }
            if (conv_base && geometry.lin_conv_stride) {
                TensorDesc d{};
                d.ptr = conv_base + static_cast<std::size_t>(layer) * geometry.lin_conv_stride;
                d.bytes = geometry.lin_conv_stride;
                d.base = conv_base;
                d.base_bytes = conv_alloc;
                d.role = TensorRole::RecurrentState;
                d.mutable_data = true;
                d.request_local = true;
                d.device = device_;
                t.conv = runtime_.register_tensor(d).id;
            }
        } else if (settings_.register_kv && kv.valid()) {
            // Paged KV lives on the FULL-ATTENTION layers only; the recurrent layers carry
            // state instead and never touch these pools.
            const std::size_t offset =
                static_cast<std::size_t>(kv_slot) * kv.layer_stride_bytes;
            if (offset + kv.live_bytes_per_layer <= kv.pool_bytes) {
                TensorDesc d{};
                d.bytes = kv.live_bytes_per_layer;
                d.base_bytes = kv.pool_bytes;
                d.role = TensorRole::KVCache;
                d.mutable_data = true;
                d.request_local = true;
                d.device = device_;
                d.ptr = static_cast<const unsigned char*>(kv.k_pool) + offset;
                d.base = kv.k_pool;
                t.k = runtime_.register_tensor(d).id;
                if (kv.v_pool) {
                    d.ptr = static_cast<const unsigned char*>(kv.v_pool) + offset;
                    d.base = kv.v_pool;
                    t.v = runtime_.register_tensor(d).id;
                }
                ++kv_layers_;
            }
            ++kv_slot;
        }
        if (weight_per_layer) {
            // A stand-in for the weights this layer streams. It has no reuse within a token,
            // which is exactly the point: it is the denominator and the interference, not a
            // candidate. Its ADDRESS is fabricated, so device = -1 -- the executor refuses to
            // let a synthetic address reach the driver, and that check lives in one place.
            TensorDesc d{};
            d.ptr = reinterpret_cast<const void*>(static_cast<std::uintptr_t>(layer) + 1);
            d.bytes = weight_per_layer;
            d.role = TensorRole::ModelWeight;
            d.model_global = true;
            d.device = -1;
            t.weight = runtime_.register_tensor(d).id;
        }
    }

    runtime_.begin_recording();
    for (int layer = 0; layer < geometry.n_layers; ++layer) {
        const LayerTensors& t = per_layer[static_cast<std::size_t>(layer)];
        KernelEvent kernel{};
        kernel.id = kernel_for_layer(layer);
        kernel.order = static_cast<std::uint64_t>(layer);
        runtime_.record_kernel(kernel);

        auto use = [&](TensorId tensor, AccessKind access, std::size_t bytes) {
            if (tensor == kInvalidTensorId) return;
            TensorUse u{};
            u.tensor = tensor;
            u.kernel = kernel.id;
            u.access = access;
            u.bytes = bytes;
            runtime_.record_use(u);
        };
        // Recurrent state is read once and written once per layer -- the pinned runtime's own
        // kernel comment says so -- which is what ReadWrite prices.
        use(t.matrix, AccessKind::ReadWrite, 0);
        use(t.conv, AccessKind::ReadWrite, 0);
        // KV is read whole and appended to by one token. Declaring the append as a write over
        // the whole slice would double the KV traffic in every ceiling; the read is the term
        // that matters and the one a cache can serve.
        use(t.k, AccessKind::Read, 0);
        use(t.v, AccessKind::Read, 0);
        use(t.weight, AccessKind::Read, 0);
    }
    // A decode token IS one iteration of a loop: a recurrent state written at layer i is read
    // again at layer i of the NEXT token, and that edge is the only one it has. A graph that
    // did not close the loop would report the whole recurrent surface as unreused, every
    // planner would correctly decline all of it, and the wrong answer would look right.
    runtime_.end_recording(/*cyclic=*/true);

    // The reuse distance a survival term reads is "bytes of OTHER traffic between two uses",
    // and on this model most of that is weights: 18.5 GB a step against ~350 MB of state and
    // KV. The adapter cannot know the figure -- it is the runtime's weight and activation
    // traffic -- so it is declared by whoever runs the benchmark and left at zero rather than
    // invented. Left at zero, though, every reuse distance in this graph is roughly fifty
    // times too short, and a cost model that reads it will think everything survives.
    //
    // That does not touch a MEASUREMENT, and it does not touch the default planner:
    // `recurrent_v0` uses TokenFootprint accounting, which counts the resident footprint and
    // ignores streamed bytes entirely. It does touch any planner whose admission depends on
    // survival. So: say so, once, rather than let a plan be optimistic in silence.
    if (!geometry.streamed_bytes_per_token && settings_.planner != "recurrent_v0" &&
        settings_.planner_config.cost_model != CostModel::Linear && !warned_no_step_traffic_) {
        warned_no_step_traffic_ = true;
        std::fprintf(stderr,
            "[recurlocal] NOTE: planner=%s with cost_model=residency and no declared step "
            "traffic. TENSORTRANSIT_STREAMED_BYTES_PER_TOKEN is 0, so this graph's reuse "
            "distances count only the state and KV this adapter registers (~%zu B) and not "
            "the model's weight traffic. Survival is therefore overestimated and every "
            "candidate looks more admissible than it is. Set it to the measured step traffic "
            "(18500000000 for Qwen3.8-27B at batch 1) for a plan whose predictions mean "
            "anything. Measurements are unaffected.\n",
            settings_.planner.c_str(), runtime_.graph().step_traffic_bytes());
    }

    RuntimeState state{};
    state.phase = RuntimePhase::Decode;
    state.active_requests = geometry.sequences > 0 ? geometry.sequences : 1;
    state.granted_budget_bytes = executor_.set_aside_bytes();
    runtime_.compile(state);
    snapshot_plan();
    maybe_write_trace();

    // Which state kind holds the window at each layer, so `after_launch` is a lookup. The
    // rule is the v0 engine's: ask which allocation the resolved region fell in, rather than
    // re-deriving the choice and letting the two drift.
    for (const TransitAction& action : runtime_.plan().actions()) {
        // Persist AND Stream: both set the one access-policy window a kernel node carries,
        // and both are host-side stream state that a capture never records -- so both have to
        // reach the node, and both need to know WHICH launch reads the bytes they cover.
        if (action.kind != TransitActionKind::Persist &&
            action.kind != TransitActionKind::Stream) continue;
        if (action.before_kernel == kInvalidKernelId) continue;
        const auto layer = static_cast<std::size_t>(action.before_kernel - 1);
        if (layer >= window_kind_.size()) continue;
        const auto* p = static_cast<const unsigned char*>(action.ptr);
        const bool is_state = state_base && p >= state_base && p < state_base + state_alloc;
        const bool is_conv = conv_base && p >= conv_base && p < conv_base + conv_alloc;
        // Anything that is neither recurrent allocation is KV, and the attention kernel is
        // what reads it.
        const StateKernel which = is_conv ? StateKernel::Conv
                                          : (is_state ? StateKernel::Gdn
                                                      : StateKernel::Attention);
        window_kind_[layer] = static_cast<signed char>(which);
        window_reduced_[layer] = action.kind == TransitActionKind::Persist &&
                                 action.hit_ratio < settings_.planner_config.hit_ratio;
    }
    // The live set the plan would have to hold against the budget it was given. This is the
    // 0.1 `hot_set_oversubscribed` question asked of the whole graph rather than of one
    // layer's slice, which is the accounting error the Transit Graph exists to make
    // impossible -- every recurrent layer's state is equally live across a token.
    const PlanCostModel& cost = runtime_.plan().cost();
    oversubscribed_ = cost.budget_bytes != 0 && cost.peak_live_bytes > cost.budget_bytes;
}

void Engine::maybe_write_trace() noexcept {
    // Once per process, after the first compile: the graph is stable across tokens by
    // construction (that is what the plan cache keys on), so a second write would be the same
    // file and a write per token would put file I/O on the decode path.
    if (trace_written_ || settings_.trace_out.empty()) return;
    trace_written_ = true;
    TraceMetadata meta{};
    meta.device = device_;
    meta.model = settings_.trace_model;
    meta.runtime = "SparkInfer";
    meta.runtime_commit = settings_.trace_runtime_commit;
    meta.phase = "decode";
    meta.active_requests = runtime_.plan().cost().budget_bytes ? recurrent_layers_ : 1;
    meta.cyclic = true;
    meta.notes = "recorded from the live SparkInfer adapter";
    std::string error;
    if (!write_trace_file(settings_.trace_out, runtime_.graph(), runtime_.registry(), meta,
                          &error)) {
        std::fprintf(stderr, "[recurlocal] could not write trace to %s: %s\n",
                     settings_.trace_out.c_str(), error.c_str());
    } else {
        std::fprintf(stderr, "[recurlocal] wrote trace %s (%zu tensors, %zu kernels)\n",
                     settings_.trace_out.c_str(), runtime_.registry().size(),
                     runtime_.graph().kernels().size());
    }
}

void Engine::snapshot_plan() noexcept {
    const TransitPlan& plan = runtime_.plan();
    plan_digest_ = plan.digest();
    committed_bytes_ = plan.cost().committed_bytes;
    predicted_saved_ = plan.cost().predicted_saved_bytes;
    step_traffic_ = plan.cost().step_traffic_bytes;
    for (std::size_t i = 0; i < kDeclineReasons; ++i) declines_[i] = 0;
    for (const TransitDecline& d : plan.declines()) {
        const auto slot = static_cast<std::size_t>(d.reason);
        if (slot < kDeclineReasons) ++declines_[slot];
    }
}

void Engine::begin_token(const StepGeometry& geometry, const KvGeometry& kv) noexcept {
    if (!initialised_ || !geometry.valid()) return;
    ++counters_.tokens;

    Signature signature{};
    signature.lin_state = geometry.lin_state;
    signature.lin_conv = geometry.lin_conv_state;
    signature.lin_state_stride = geometry.lin_state_stride;
    signature.lin_conv_stride = geometry.lin_conv_stride;
    signature.n_layers = geometry.n_layers;
    signature.full_attn_interval = geometry.full_attn_interval;
    signature.sequences = geometry.sequences;
    signature.streamed = geometry.streamed_bytes_per_token;
    if (settings_.register_kv && kv.valid()) {
        signature.k_pool = kv.k_pool;
        signature.v_pool = kv.v_pool;
        signature.kv_live = kv.live_bytes_per_layer;
        signature.kv_slots = kv.kv_slots;
    }

    if (!have_signature_ || signature != signature_) {
        rebuild(geometry, kv);
        signature_ = signature;
        have_signature_ = true;
        ++counters_.recompiles_geometry;
    }
    counters_.compiles = runtime_.stats().compiles;
    counters_.plan_reuses = runtime_.stats().plan_reuses;
    pending_layer_ = -1;
    runtime_.begin_step();
}

void Engine::before_layer(int layer) noexcept {
    if (!initialised_ || layer < 0) return;
    pending_layer_ = layer;
    const auto slot = static_cast<std::size_t>(layer);
    if (slot < window_kind_.size() && window_kind_[slot] >= 0) {
        ++counters_.layers;
        if (window_reduced_[slot]) ++counters_.hit_ratio_reduced;
        if (oversubscribed_) ++counters_.hot_set_oversubscribed;
    }
    runtime_.before_kernel(kernel_for_layer(layer));
}

void Engine::after_launch(StateKernel which) noexcept {
    if (!initialised_) return;
    if (settings_.attach == WindowAttach::Stream) return;
    if (pending_layer_ < 0) return;
    const auto layer = static_cast<std::size_t>(pending_layer_);
    if (layer >= window_kind_.size() || window_kind_[layer] < 0) return;
    // Only the launch that reads the bytes the window covers. Attaching on the other one
    // would mark a kernel node for memory it never touches, which spends set-aside on
    // nothing and reports a delivered policy for it.
    if (static_cast<StateKernel>(window_kind_[layer]) != which) {
        ++counters_.attach_skipped_wrong_kernel;
        return;
    }
    ++counters_.attach_calls;
    executor_.attach_window_to_captured_node();
}

void Engine::after_layer(int layer) noexcept {
    if (!initialised_ || layer < 0) return;
    runtime_.after_kernel(kernel_for_layer(layer));
    pending_layer_ = -1;
}

void Engine::end_token() noexcept {
    if (!initialised_) return;
    runtime_.end_step();
}

void Engine::shutdown() noexcept {
    if (!initialised_) return;
    runtime_.set_executor(nullptr);
    executor_.release();
    if (scratch_) {
        cudaFree(scratch_);
        scratch_ = nullptr;
    }
    initialised_ = false;
}

}  // namespace transit
}  // namespace sparkinfer
}  // namespace tensortransit
#endif
