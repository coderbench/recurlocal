#pragma once
// The SparkInfer adapter's TensorTransit engine.
//
// This is the file that puts the 0.2 core into the measured path. Before it, the adapter
// drove `CudaLocalityController` -- the 0.1 policy -- directly, and `TransitRuntime`,
// `TensorRegistry`, `TransitGraph`, `ITransitPlanner` and `CudaTransitExecutor` were
// reachable only from the CLI and the tests. A contributor who wrote a planner therefore
// changed nothing about the number the evaluator prints, which made the entire competition
// surface decorative.
//
// It is a SECOND engine rather than a replacement, selected by TENSORTRANSIT_ENGINE, because
// the two have to be comparable in one process against one model load: `v0` is the policy
// every published number in this repository was measured under, and `transit` has to be shown
// to reproduce it before anything else it says can be believed. See docs/STABILITY.md.
//
// NOT a public header: it is not installed and nothing outside adapters/sparkinfer includes
// it.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#if defined(TENSORTRANSIT_WITH_CUDA) || defined(RECURLOCAL_WITH_CUDA) || \
    defined(RECURLLOCAL_WITH_CUDA)
#include <cuda_runtime_api.h>

#include "tensortransit/cuda_executor.h"
#include "tensortransit/recurrent.h"
#include "tensortransit/runtime.h"
#include "tensortransit/sparkinfer.h"

namespace tensortransit {
namespace sparkinfer {
namespace transit {

// The recurrent-state geometry of one decode step, as the adapter already computes it from
// the runtime's own config. Addresses are row 0's on the packed path: an access-policy
// window needs an address the host can name and there is exactly one to give, while the
// FOOTPRINT scales with `sequences`, which is what the planner is told separately.
struct StepGeometry {
    void* lin_state = nullptr;
    std::size_t lin_state_stride = 0;
    void* lin_conv_state = nullptr;
    std::size_t lin_conv_stride = 0;
    int n_layers = 0;
    int full_attn_interval = 0;
    int sequences = 1;
    // Weight and activation traffic through L2 between two visits to the same state. The
    // adapter cannot know it, so it is declared by whoever runs the benchmark and left at
    // zero rather than invented -- at zero no ModelWeight tensor is registered at all.
    std::size_t streamed_bytes_per_token = 0;

    bool valid() const noexcept {
        return n_layers > 0 && (lin_state_stride != 0 || lin_conv_stride != 0);
    }
};

// The paged KV pools of one decode step.
//
// Optional, and the reason it exists: no adapter exposed KV to the registry, so the
// specification's second proof track -- does one planner arbitrating a shared budget across
// two tensor classes beat two independent policies -- could not be measured at all. Not
// "had not been": could not be. A runtime that never declares this simply has no KV in the
// registry and every KV-scoped planner declines with `no_reuse`, which is the truth.
struct KvGeometry {
    const void* k_pool = nullptr;
    const void* v_pool = nullptr;
    std::size_t pool_bytes = 0;          // bytes of ONE pool (K and V are sized alike)
    std::size_t layer_stride_bytes = 0;  // between consecutive slots' sub-pools
    // Bytes ONE attention layer's K (or V) actually reads in this step, over every live
    // sequence. Not the slot stride: the pool is sized for the longest context the server
    // will ever hold and a decode step at 128 tokens reads a thousandth of it. Declaring the
    // stride would put a KV footprint two orders of magnitude too large into every ceiling.
    std::size_t live_bytes_per_layer = 0;
    int kv_slots = 0;   // pool slots actually allocated
    int rows = 1;       // sequences whose blocks the live bytes cover

    bool valid() const noexcept {
        return k_pool != nullptr && kv_slots > 0 && live_bytes_per_layer != 0;
    }
};

// Everything the engine is configured with. Parsed from the environment by the adapter, so
// this header carries no getenv and stays testable.
struct Settings {
    std::string planner = "recurrent_v0";
    TransitPlannerConfig planner_config{};
    // How a window reaches a kernel under CUDA Graph capture. Shared with the v0 engine's
    // enumerator on purpose: the two engines must be selectable with the same command line
    // or the A/B is comparing two configurations rather than two engines.
    WindowAttach attach = WindowAttach::Stream;
    double budget_fraction = 0.75;
    // Register KV when the runtime declares it. Off makes the engine recurrent-only, which
    // is what reproduces the 0.1 policy exactly.
    bool register_kv = true;
    // Prefetch needs scratch for the (discarded) pre-touch results.
    std::size_t scratch_floats = 1024;
    // Where to write the recorded Transit Graph, once, after the first compile. Empty
    // disables it.
    //
    // This is what turns the offline planning loop from a synthetic exercise into a real one.
    // tests/golden/*.json carry the MEASURED recurrent geometry of Qwen3.8-27B and a
    // SYNTHETIC KV block size, with the weight traffic divided evenly across layers -- so
    // every comparison run against them is sharp about the recurrent half and approximate
    // about everything else. A trace recorded here has the runtime's real KV slice sizes, its
    // real layer interleaving, and its real per-layer demand.
    std::string trace_out;
    std::string trace_model;
    std::string trace_runtime_commit;
};

// What the engine did, in the currency the evaluation harness already parses.
struct Counters {
    std::uint64_t tokens = 0;
    std::uint64_t compiles = 0;
    std::uint64_t plan_reuses = 0;
    std::uint64_t recompiles_geometry = 0;
    std::uint64_t attach_calls = 0;
    std::uint64_t attach_skipped_wrong_kernel = 0;
    // Bracketed kernels that carry a persisting window, and the two 0.1 telemetry lines the
    // evaluation harness has read since the first result file. `hit_ratio_reduced` is a
    // window admitted at less than the configured ratio -- the hot-set policy shaving it --
    // and `hot_set_oversubscribed` is a step whose live set exceeds the budget. Counted per
    // bracketed layer, which is what the 0.1 engine counts, so the two engines' stats lines
    // mean the same thing and an A/B between them is readable.
    std::uint64_t layers = 0;
    std::uint64_t hit_ratio_reduced = 0;
    std::uint64_t hot_set_oversubscribed = 0;
    // Traces written. More than one is normal and is the point: the first graph an engine
    // builds has no KV in it, because the runtime declares its pools after it opens the token.
    std::uint64_t traces_written = 0;
};

// One engine instance. Not synchronised: the adapter holds exactly one, for one model on one
// stream, and refuses a second model rather than interleaving two layer walks through it.
class Engine {
public:
    Engine() noexcept = default;
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Reserves the set-aside and binds the streams. `prefetch` is borrowed and may be null,
    // in which case Prefetch actions are skipped rather than silently run on the compute
    // stream. False on failure; `error()` says why.
    bool initialize(int device, cudaStream_t compute, cudaStream_t prefetch,
                    const Settings& settings) noexcept;
    bool initialised() const noexcept { return initialised_; }
    const char* error() const noexcept { return error_; }

    // Declares the step's geometry and compiles if it changed. Called once per token, from
    // OUTSIDE graph capture. Cheap on the steady path: one signature compare.
    void begin_token(const StepGeometry& geometry, const KvGeometry& kv) noexcept;

    // Bracket one layer, by ABSOLUTE layer index -- the same index the state allocations are
    // indexed by, and the one the recorded graph orders its kernels on.
    void before_layer(int layer) noexcept;
    void after_layer(int layer) noexcept;
    // The launch that just recorded a node. Attaches a deferred window only when the state
    // it covers is the state this kernel reads, which is the same rule the v0 engine applies.
    void after_launch(StateKernel which) noexcept;
    void end_token() noexcept;

    // Give the device back: drop the window, release the set-aside, free the scratch.
    void shutdown() noexcept;

    const TransitRuntime& runtime() const noexcept { return runtime_; }
    const CudaTransitExecutor& executor() const noexcept { return executor_; }
    // Executor and runtime counters for the WHOLE run, not for the current plan.
    //
    // `TransitRuntime::reset()` clears the executor's stats along with the plan, which is
    // right for a runtime being reused for a different model and wrong for telemetry an
    // evaluator reads at process exit: a geometry rebuild -- which a growing KV context
    // causes every block -- would erase the windows already delivered, and the run would
    // report itself as a NULL CANDIDATE having applied a policy to every layer. That is not
    // hypothetical; it is what the first run of this engine reported.
    ExecutorStats total_executor_stats() const noexcept;
    RuntimeStats total_runtime_stats() const noexcept;
    const Counters& counters() const noexcept { return counters_; }
    std::size_t set_aside_peak_bytes() const noexcept { return executor_.set_aside_peak_bytes(); }
    std::size_t set_aside_at_init_bytes() const noexcept { return set_aside_at_init_; }
    const std::string& planner_name() const noexcept { return settings_.planner; }
    std::uint64_t plan_digest() const noexcept { return plan_digest_; }
    // Counts by decline reason, in DeclineReason order, from the plan in force.
    const std::uint64_t* declines() const noexcept { return declines_; }
    std::size_t declines_size() const noexcept { return kDeclineReasons; }
    int recurrent_layers() const noexcept { return recurrent_layers_; }
    int kv_layers() const noexcept { return kv_layers_; }
    std::size_t committed_bytes() const noexcept { return committed_bytes_; }
    std::size_t predicted_saved_bytes() const noexcept { return predicted_saved_; }
    std::size_t step_traffic_bytes() const noexcept { return step_traffic_; }

private:
    static constexpr std::size_t kDeclineReasons = 8;

    // Identity of the geometry a plan was built for. A recompile that fires every token puts
    // the planner on the critical path, which is the single most likely way this layer costs
    // more than it saves -- so the signature is explicit rather than inferred.
    struct Signature {
        const void* lin_state = nullptr;
        const void* lin_conv = nullptr;
        std::size_t lin_state_stride = 0;
        std::size_t lin_conv_stride = 0;
        const void* k_pool = nullptr;
        const void* v_pool = nullptr;
        std::size_t kv_live = 0;
        int n_layers = 0;
        int full_attn_interval = 0;
        int sequences = 0;
        int kv_slots = 0;
        std::size_t streamed = 0;
        bool operator==(const Signature& o) const noexcept;
        bool operator!=(const Signature& o) const noexcept { return !(*this == o); }
    };

    void rebuild(const StepGeometry& geometry, const KvGeometry& kv) noexcept;
    void maybe_write_trace(int sequences) noexcept;
    void snapshot_plan() noexcept;

    TransitRuntime runtime_;
    CudaTransitExecutor executor_;
    Settings settings_{};
    Signature signature_{};
    // Which state kind the plan's window for this layer covers, indexed by absolute layer.
    // Empty means no window there. Precomputed at compile time so `after_launch` is a lookup
    // rather than a search through the plan on every recurrent kernel.
    std::vector<signed char> window_kind_;
    // Whether that layer's window was admitted below the configured hit ratio. Precomputed
    // for the same reason.
    std::vector<bool> window_reduced_;
    bool oversubscribed_ = false;
    // Counters retired by a rebuild, so the totals above span the run.
    ExecutorStats retired_executor_{};
    RuntimeStats retired_runtime_{};
    float* scratch_ = nullptr;
    cudaStream_t compute_ = nullptr;
    bool initialised_ = false;
    bool have_signature_ = false;
    bool warned_no_step_traffic_ = false;
    const char* error_ = nullptr;
    int device_ = 0;
    int pending_layer_ = -1;
    std::size_t set_aside_at_init_ = 0;
    std::uint64_t plan_digest_ = 0;
    std::uint64_t declines_[kDeclineReasons] = {};
    int recurrent_layers_ = 0;
    int kv_layers_ = 0;
    std::size_t committed_bytes_ = 0;
    std::size_t predicted_saved_ = 0;
    std::size_t step_traffic_ = 0;
    Counters counters_{};
};

}  // namespace transit
}  // namespace sparkinfer
}  // namespace tensortransit
#endif
