#pragma once
// RecurLocal <-> SparkInfer adapter.
//
// SparkInfer owns model execution; this file owns nothing but the decision of where the
// recurrent state should live while that execution runs. It is deliberately the entire
// integration surface: five calls, no SparkInfer type in the signature, no model math.
//
// Why an adapter and not a patch that calls the controller directly: the hook has to know
// things the controller cannot see and the runtime should not have to spell out - which of
// the model's layers are recurrent, that there are two recurrent states and not one, how
// far ahead the configured schedule wants to reach, and which of the two states the window
// was aimed at. Keeping that here holds the diff against SparkInfer to a handful of lines,
// which is what makes it re-appliable to a moving upstream.
//
// Everything is off unless RECURLOCAL is set, so a build carrying this adapter and a build
// without it are the same binary until an environment variable says otherwise. That is
// also what makes an A/B honest: one binary, one model load, one machine.
#include <cstddef>
#include <cstdio>

#if defined(RECURLOCAL_WITH_CUDA) || defined(RECURLLOCAL_WITH_CUDA)
#include <cuda_runtime_api.h>

namespace recurlocal {
namespace sparkinfer {

// The two mutable recurrent states a Qwen3.5/3.6/3.8 hybrid layer carries, described once
// per decode step. Both are single allocations indexed by absolute layer, so a layer's
// slice is base + layer * stride.
struct GdnStateLayout {
    void* lin_state = nullptr;              // fp32 Gated-DeltaNet matrix state
    std::size_t lin_state_stride = 0;       // bytes per layer
    void* lin_conv_state = nullptr;         // bf16 causal-convolution window
    std::size_t lin_conv_stride = 0;        // bytes per layer
    int n_layers = 0;                       // layers the allocations are indexed by
    int full_attn_interval = 0;             // layer L is recurrent iff (L+1) % interval != 0
    cudaStream_t compute = nullptr;         // the stream the decode step runs on
};

// Concurrent decode. SparkInfer batches sequences through decode_packed(), where each
// sequence owns its own pair of state allocations and the runtime gathers their base
// pointers into device-side arrays to drive its batched kernels. That is the regime the
// locality question actually lives in: model weights are read once per step however many
// sequences are in flight, while recurrent state scales with concurrency, so the recurrent
// share of decode traffic grows from ~1.6% at batch 1 to tens of percent at batch 32.
struct GdnPackedLayout {
    // DEVICE arrays of `rows` base pointers, one per sequence. Used for the pre-touch.
    const void* const* device_lin_state = nullptr;
    const void* const* device_lin_conv = nullptr;
    // The same pointers host-side, for the persisting window - a window needs an address
    // the host can name. Only row 0 is required; nullptr means no window is placed.
    void* host_lin_state_row0 = nullptr;
    void* host_lin_conv_row0 = nullptr;
    int rows = 0;
    std::size_t lin_state_stride = 0;   // bytes per layer, per sequence
    std::size_t lin_conv_stride = 0;
    int n_layers = 0;
    int full_attn_interval = 0;
    cudaStream_t compute = nullptr;
};

// As begin_token(), for the packed path. Must likewise be called outside graph capture.
bool begin_token_packed(const GdnPackedLayout& layout) noexcept;

// True once RECURLOCAL names a mode other than off. Cheap after the first call; safe to
// call before attach().
bool enabled() noexcept;

// The mode name as configured, for provenance in a result file. "off" when disabled.
const char* mode_name() noexcept;

// Start a decode step. Must be called OUTSIDE CUDA Graph capture: the first call
// initialises the controller, which reserves the L2 set-aside and allocates the pre-touch
// scratch, and neither is legal inside a capture. Cheap on every later call.
// Returns false when the adapter is disabled or could not initialise, in which case the
// per-layer calls below are all no-ops.
bool begin_token(const GdnStateLayout& layout) noexcept;

// Bracket one recurrent layer. `layer` is the absolute layer index, the same one that
// indexes the state allocations.
void before_layer(int layer) noexcept;

// Which state kernel just launched. Under graph capture the persisting window is attached
// to the node this launch recorded, so it has to be the launch that reads the state the
// window was aimed at - the convolution kernel for the conv state, the recurrence for the
// matrix state. Calling both is correct and costs one capture query.
enum class StateKernel : int { Conv, Gdn };
void after_launch(StateKernel which) noexcept;

void after_layer() noexcept;

// Close the step's layer walk. Under PrefetchJoin::TokenEnd this emits the single join,
// which under capture is mandatory: an unjoined fork ends the capture invalid.
void end_token() noexcept;

// One JSON object describing the configuration and what the controller actually did.
// Written to `out`; a runtime with no JSON dependency can print it verbatim.
void write_stats_json(std::FILE* out) noexcept;

// Drop the window, release the set-aside, destroy the prefetch stream.
void shutdown() noexcept;

} // namespace sparkinfer
} // namespace recurlocal

#else   // no CUDA: the hook compiles away entirely

// cudaStream_t is `struct CUstream_st*`. Declaring `compute` as void* here instead gave the
// two branches of this header two DIFFERENT types with the same name, so a CUDA-built and a
// non-CUDA-built translation unit linked into one binary were an ODR violation that nothing
// would diagnose. Forward-declaring the tag makes both branches declare the identical type
// without this branch including a CUDA header.
struct CUstream_st;

namespace recurlocal { namespace sparkinfer {
struct GdnStateLayout { void* lin_state; std::size_t lin_state_stride;
                        void* lin_conv_state; std::size_t lin_conv_stride;
                        int n_layers; int full_attn_interval; ::CUstream_st* compute; };
inline bool enabled() noexcept { return false; }
inline const char* mode_name() noexcept { return "off"; }
struct GdnPackedLayout { const void* const* device_lin_state; const void* const* device_lin_conv;
                         void* host_lin_state_row0; void* host_lin_conv_row0; int rows;
                         std::size_t lin_state_stride; std::size_t lin_conv_stride;
                         int n_layers; int full_attn_interval; ::CUstream_st* compute; };
inline bool begin_token(const GdnStateLayout&) noexcept { return false; }
inline bool begin_token_packed(const GdnPackedLayout&) noexcept { return false; }
inline void before_layer(int) noexcept {}
enum class StateKernel : int { Conv, Gdn };
inline void after_launch(StateKernel) noexcept {}
inline void after_layer() noexcept {}
inline void end_token() noexcept {}
inline void write_stats_json(std::FILE*) noexcept {}
inline void shutdown() noexcept {}
}}
#endif
