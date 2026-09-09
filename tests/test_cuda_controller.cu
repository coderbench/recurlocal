// Device-side tests for the CUDA locality controller.
//
// Until this file existed, every line of CUDA in the repository was untested: the planner had
// 400 lines of coverage and the controller, the pre-touch kernels and the graph-capture state
// machine had none. That is the wrong way round — the planner is pure arithmetic that a reader
// can check, while the controller's contract is with a driver whose behaviour under graph
// capture is the entire reason this project is hard.
//
// Every test below either pins a documented promise or is a regression test for a defect that
// was actually shipped. The ones marked REGRESSION each correspond to a real bug.
//
// Requires a GPU. Registered with ctest only when RECURLOCAL_BUILD_CUDA=ON; skips cleanly
// (exit 0) on a machine with no device, so a CUDA build on a device-less CI box still passes.
#include "recurlocal/cuda_api.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace recurlocal;

static int g_failures = 0;
static int g_checks = 0;
#define CHECK(cond) do { ++g_checks; if (!(cond)) { \
    std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++g_failures; } } while (0)
#define CHECK_CUDA(expr) do { ++g_checks; cudaError_t _e = (expr); if (_e != cudaSuccess) { \
    std::printf("FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(_e)); \
    ++g_failures; } } while (0)

namespace {

constexpr std::size_t kLayers = 8;
constexpr std::size_t kSliceBytes = 256 * 1024;   // small: these are contract tests, not benchmarks

PlannerConfig base_config(LocalityMode mode) {
    PlannerConfig cfg;
    cfg.mode = mode;
    cfg.persisting_budget_fraction = 0.25;
    cfg.hit_ratio = 0.6;
    cfg.prefetch_distance = 1;
    return cfg;
}

struct Fixture {
    float* state = nullptr;
    cudaStream_t compute{}, prefetch{};
    std::size_t total_bytes = kLayers * kSliceBytes;

    bool init() {
        if (cudaMalloc(&state, total_bytes) != cudaSuccess) return false;
        if (cudaMemset(state, 0x3f, total_bytes) != cudaSuccess) return false;
        if (cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking) != cudaSuccess) return false;
        if (cudaStreamCreateWithFlags(&prefetch, cudaStreamNonBlocking) != cudaSuccess) return false;
        return true;
    }
    ~Fixture() {
        if (state) cudaFree(state);
        if (compute) cudaStreamDestroy(compute);
        if (prefetch) cudaStreamDestroy(prefetch);
    }
    StateSegment slice(int layer, StateKind kind = StateKind::Matrix) const {
        StateSegment s;
        s.ptr = reinterpret_cast<const unsigned char*>(state) + layer * kSliceBytes;
        s.bytes = kSliceBytes;
        s.base = state;
        s.base_bytes = total_bytes;
        s.kind = kind;
        return s;
    }
};

RecurrentGeometry geometry(int sequences = 1) {
    RecurrentGeometry g;
    g.recurrent_layers = (int)kLayers;
    g.bytes_per_layer = kSliceBytes;
    g.sequences = sequences;
    return g;
}

// REGRESSION: configure_persisting_l2() called cudaSetDevice() and never restored it, so
// merely constructing a controller silently repointed the calling thread's GPU. On a
// multi-GPU runtime the next unqualified launch lands on the wrong device.
void test_initialize_does_not_steal_the_current_device() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1) return;
    if (devices < 2) {
        // Honest coverage note rather than a green tick: with one GPU the current device
        // already IS the target, so configure_persisting_l2() never switches and the restore
        // path is not exercised. Verified by reverting the fix - this test still passed.
        // The assertions below are kept because they must hold on any device count, but a
        // real regression here can only be caught on a multi-GPU box.
        std::printf("[partial] single GPU: the device-restore path is NOT exercised\n");
    }
    int before = -1;
    CHECK_CUDA(cudaGetDevice(&before));
    {
        CudaLocalityController c;
        c.initialize(0, base_config(LocalityMode::Combined));
        int during = -1;
        CHECK_CUDA(cudaGetDevice(&during));
        CHECK(during == before);
    }
    int after = -1;
    CHECK_CUDA(cudaGetDevice(&after));
    CHECK(after == before);
}

// REGRESSION: the device-wide persisting-L2 set-aside was never returned. reset()'s header
// promised to release it and only evicted persisting lines, and the destructor cleared a
// bookkeeping member without telling the driver. A process that ever built a controller ran
// the rest of its life with a smaller L2 for every other kernel.
void test_l2_set_aside_is_returned() {
    // Ask for the whole persisting budget, so the limit demonstrably MOVES. An earlier
    // version of this test used the 0.25 default, which on an RTX 5090 rounds to the
    // driver's pre-existing 18 MiB default - the limit never changed and the test passed
    // even with the restore deleted. A regression test that cannot fail is not a test.
    auto cfg = base_config(LocalityMode::Persist);
    cfg.persisting_budget_fraction = 1.0;

    std::size_t before = 0, during = 0, after = 0;
    CHECK_CUDA(cudaDeviceGetLimit(&before, cudaLimitPersistingL2CacheSize));
    {
        CudaLocalityController c;
        if (c.initialize(0, cfg) != cudaSuccess) return;
        if (c.l2_set_aside_bytes() == 0) return;    // device cannot reserve one
        CHECK_CUDA(cudaDeviceGetLimit(&during, cudaLimitPersistingL2CacheSize));
        CHECK(during != before);                    // the reservation really happened
        CHECK(c.l2_set_aside_bytes() == during);    // and we report what the driver granted,
        CHECK(c.planner().granted_l2_set_aside() == during);  // ...and budget against it
    }
    CHECK_CUDA(cudaDeviceGetLimit(&after, cudaLimitPersistingL2CacheSize));
    CHECK(after == before);                         // destruction gives the device back

    CudaLocalityController c2;
    if (c2.initialize(0, cfg) != cudaSuccess) return;
    if (c2.l2_set_aside_bytes() == 0) return;
    std::size_t held = 0;
    CHECK_CUDA(cudaDeviceGetLimit(&held, cudaLimitPersistingL2CacheSize));
    CHECK(held != before);
    CHECK_CUDA(c2.reset());                         // reset() promises the same release
    std::size_t after_reset = 0;
    CHECK_CUDA(cudaDeviceGetLimit(&after_reset, cudaLimitPersistingL2CacheSize));
    CHECK(after_reset == before);
}

void test_bind_streams_rejects_one_stream_used_twice() {
    Fixture f;
    if (!f.init()) return;
    CudaLocalityController c;
    if (c.initialize(0, base_config(LocalityMode::Combined)) != cudaSuccess) return;
    // A pre-touch issued on the compute stream is not a prefetch, it is the same work on the
    // critical path — the API is documented to refuse it.
    CHECK(c.bind_streams(f.compute, f.compute) == cudaErrorInvalidValue);
    CHECK(c.bind_streams(f.compute, f.prefetch) == cudaSuccess);
}

void test_window_is_scoped_to_the_layer_outside_capture() {
    Fixture f;
    if (!f.init()) return;
    CudaLocalityController c;
    if (c.initialize(0, base_config(LocalityMode::Persist)) != cudaSuccess) return;
    if (c.bind_streams(f.compute, f.prefetch) != cudaSuccess) return;
    if (c.l2_set_aside_bytes() == 0) return;

    const auto cur = f.slice(0), nxt = f.slice(1);
    LayerActions a{};
    CHECK_CUDA(c.before_layer(&cur, 1, &nxt, 1, true, geometry(), &a));
    CHECK(a.window_applied_to_stream);            // no capture -> applied directly
    CHECK(!a.window_requires_launch_attribute);
    CHECK_CUDA(c.after_layer());
    CHECK(c.stats().windows_applied == 1);
    CHECK(c.stats().windows_cleared >= 1);        // the window must not outlive the layer
}

// REGRESSION: the legacy five-argument before_layer() armed graph-node mutation regardless of
// WindowAttach, so a consumer that asked for the documented-safe default still got a graph
// mutated mid-capture. The segment overload was fixed; this one was not.
void test_window_attach_default_is_honoured_on_both_overloads() {
    for (int overload = 0; overload < 2; ++overload) {
        Fixture f;
        if (!f.init()) return;
        auto cfg = base_config(LocalityMode::Persist);
        cfg.window_attach = WindowAttach::Stream;      // the safe default
        CudaLocalityController c;
        if (c.initialize(0, cfg) != cudaSuccess) return;
        if (c.bind_streams(f.compute, f.prefetch) != cudaSuccess) return;
        if (c.l2_set_aside_bytes() == 0) return;

        float* scratch = nullptr;
        if (cudaMalloc(&scratch, 4096 * sizeof(float)) != cudaSuccess) return;

        CHECK_CUDA(cudaStreamBeginCapture(f.compute, cudaStreamCaptureModeThreadLocal));
        c.begin_sequence();
        LayerActions a{};
        if (overload == 0) {
            const auto cur = f.slice(0), nxt = f.slice(1);
            c.before_layer(&cur, 1, &nxt, 1, true, geometry(), &a);
        } else {
            c.before_layer(f.state, kSliceBytes,
                           reinterpret_cast<const float*>(f.slice(1).ptr),
                           kSliceBytes / sizeof(float), true, 0, &a);
        }
        // Record a real kernel so that a node EXISTS to attach to. Without this the attach
        // fails for lack of a node and the "not attached" assertion below holds for the
        // wrong reason - which is exactly how the first version of this test passed while
        // the guard it was meant to protect was missing.
        CHECK_CUDA(pre_touch_bytes_async(PreTouchStrategy::Vec4, f.slice(2).ptr, kSliceBytes,
                                         scratch, 4096, f.compute));
        c.attach_window_to_captured_node();
        c.end_sequence();
        cudaGraph_t g = nullptr;
        CHECK_CUDA(cudaStreamEndCapture(f.compute, &g));
        if (g) cudaGraphDestroy(g);

        cudaFree(scratch);
        CHECK(a.window_requires_launch_attribute);          // handed back to the caller...
        CHECK(c.stats().windows_deferred_to_caller == 1);
        CHECK(c.stats().windows_attached_to_node == 0);     // ...and NOT mutated behind its back
    }
}

void test_capture_node_attach_is_opt_in_and_works() {
    Fixture f;
    if (!f.init()) return;
    auto cfg = base_config(LocalityMode::Persist);
    cfg.window_attach = WindowAttach::CaptureNode;
    CudaLocalityController c;
    if (c.initialize(0, cfg) != cudaSuccess) return;
    if (c.bind_streams(f.compute, f.prefetch) != cudaSuccess) return;
    if (c.l2_set_aside_bytes() == 0) return;

    // Allocated BEFORE capture begins: cudaMalloc is forbidden while a stream is capturing.
    float* scratch = nullptr;
    if (cudaMalloc(&scratch, 4096 * sizeof(float)) != cudaSuccess) return;

    CHECK_CUDA(cudaStreamBeginCapture(f.compute, cudaStreamCaptureModeThreadLocal));
    c.begin_sequence();
    const auto cur = f.slice(0), nxt = f.slice(1);
    LayerActions a{};
    c.before_layer(&cur, 1, &nxt, 1, true, geometry(), &a);
    // attach_window_to_captured_node() sets the attribute on whatever kernel node the
    // capture last recorded, so a real kernel has to be on the compute stream first --
    // exactly as a runtime would have just launched its recurrent kernel there.
    CHECK_CUDA(pre_touch_bytes_async(PreTouchStrategy::Vec4, f.slice(2).ptr, kSliceBytes,
                                     scratch, 4096, f.compute));
    CHECK_CUDA(c.attach_window_to_captured_node());
    c.end_sequence();
    cudaGraph_t g = nullptr;
    CHECK_CUDA(cudaStreamEndCapture(f.compute, &g));
    CHECK(g != nullptr);
    if (g) {
        cudaGraphExec_t exec = nullptr;
        // The point of the whole mechanism: the graph must still instantiate after we
        // mutated one of its nodes. If this fails, capture_node is unusable, not merely risky.
        CHECK_CUDA(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));
        if (exec) cudaGraphExecDestroy(exec);
        cudaGraphDestroy(g);
    }
    cudaFree(scratch);
    CHECK(c.stats().windows_deferred_to_caller == 1);
    CHECK(c.stats().windows_attached_to_node == 1);   // opt-in path DOES attach
    CHECK(c.stats().capture_invalidations == 0);
}

// The capture must remain instantiable with the pre-touch fork/join inside it, under BOTH
// join policies. An unjoined fork makes cudaStreamEndCapture fail, which is how a locality
// layer silently drops a host runtime onto a slower path.
void test_capture_stays_valid_under_both_join_policies() {
    for (auto join : {PrefetchJoin::PerLayer, PrefetchJoin::TokenEnd}) {
        Fixture f;
        if (!f.init()) return;
        auto cfg = base_config(LocalityMode::Prefetch);
        cfg.prefetch_join = join;
        CudaLocalityController c;
        if (c.initialize(0, cfg) != cudaSuccess) return;
        if (c.bind_streams(f.compute, f.prefetch) != cudaSuccess) return;

        CHECK_CUDA(cudaStreamBeginCapture(f.compute, cudaStreamCaptureModeThreadLocal));
        c.begin_sequence();
        for (int l = 0; l + 1 < (int)kLayers; ++l) {
            const auto cur = f.slice(l), nxt = f.slice(l + 1);
            c.before_layer(&cur, 1, &nxt, 1, true, geometry(), nullptr);
            c.after_layer();
        }
        c.end_sequence();
        cudaGraph_t g = nullptr;
        CHECK_CUDA(cudaStreamEndCapture(f.compute, &g));
        CHECK(g != nullptr);
        if (g) {
            cudaGraphExec_t exec = nullptr;
            CHECK_CUDA(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));   // must be replayable
            if (exec) { CHECK_CUDA(cudaGraphLaunch(exec, f.compute));
                        CHECK_CUDA(cudaStreamSynchronize(f.compute));
                        cudaGraphExecDestroy(exec); }
            cudaGraphDestroy(g);
        }
        CHECK(c.stats().pre_touch_launches > 0);
    }
}

// The exact-locality contract: pre-touch is read-only. Every strategy, over the same buffer,
// must leave it bit-identical — this is the guarantee that a locality change cannot disturb
// model state.
void test_every_pre_touch_strategy_is_read_only() {
    Fixture f;
    if (!f.init()) return;
    std::vector<unsigned char> before(f.total_bytes), after(f.total_bytes);
    CHECK_CUDA(cudaMemcpy(before.data(), f.state, f.total_bytes, cudaMemcpyDeviceToHost));

    float* scratch = nullptr;
    if (cudaMalloc(&scratch, 4096 * sizeof(float)) != cudaSuccess) return;
    const PreTouchStrategy all[] = {PreTouchStrategy::Scalar, PreTouchStrategy::Vec4,
                                    PreTouchStrategy::Vec4Ldcg, PreTouchStrategy::PtxL2,
                                    PreTouchStrategy::WarpTile, PreTouchStrategy::Partial};
    for (auto s : all) {
        CHECK_CUDA(pre_touch_bytes_async(s, f.state, f.total_bytes, scratch, 4096, f.prefetch));
        CHECK_CUDA(cudaStreamSynchronize(f.prefetch));
        CHECK_CUDA(cudaMemcpy(after.data(), f.state, f.total_bytes, cudaMemcpyDeviceToHost));
        CHECK(std::memcmp(before.data(), after.data(), f.total_bytes) == 0);
    }
    cudaFree(scratch);
}

// The row-major form exists so the launch count does not scale with concurrency. It reads
// base pointers out of DEVICE memory, which is the part that cannot be checked by inspection.
void test_row_major_pre_touch_reads_device_pointer_array() {
    Fixture f;
    if (!f.init()) return;
    const int rows = 4;
    const void* host_bases[rows];
    for (int i = 0; i < rows; ++i) host_bases[i] = f.state;
    const void** dev_bases = nullptr;
    if (cudaMalloc(&dev_bases, rows * sizeof(void*)) != cudaSuccess) return;
    CHECK_CUDA(cudaMemcpy(dev_bases, host_bases, rows * sizeof(void*), cudaMemcpyHostToDevice));
    float* scratch = nullptr;
    if (cudaMalloc(&scratch, 4096 * sizeof(float)) != cudaSuccess) { cudaFree(dev_bases); return; }

    std::vector<unsigned char> before(f.total_bytes), after(f.total_bytes);
    CHECK_CUDA(cudaMemcpy(before.data(), f.state, f.total_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(pre_touch_rows_async(PreTouchStrategy::Vec4, dev_bases, rows,
                                    /*byte_offset=*/kSliceBytes, kSliceBytes,
                                    scratch, 4096, f.prefetch));
    CHECK_CUDA(cudaStreamSynchronize(f.prefetch));
    CHECK_CUDA(cudaMemcpy(after.data(), f.state, f.total_bytes, cudaMemcpyDeviceToHost));
    CHECK(std::memcmp(before.data(), after.data(), f.total_bytes) == 0);   // still read-only

    CHECK(pre_touch_rows_async(PreTouchStrategy::Vec4, nullptr, rows, 0, kSliceBytes,
                               scratch, 4096, f.prefetch) == cudaErrorInvalidValue);
    CHECK(pre_touch_rows_async(PreTouchStrategy::Vec4, dev_bases, 0, 0, kSliceBytes,
                               scratch, 4096, f.prefetch) == cudaErrorInvalidValue);
    cudaFree(scratch); cudaFree(dev_bases);
}

// REGRESSION: stats().hot_set_oversubscribed was incremented from hit_ratio_reduced, which
// HotSetPolicy::Fixed never sets — so the counter read zero under precisely the policy that
// ignores oversubscription hardest, and a badly oversubscribed run looked healthy.
void test_oversubscription_is_counted_under_every_policy() {
    for (auto policy : {HotSetPolicy::Fixed, HotSetPolicy::Proportional,
                        HotSetPolicy::Sqrt, HotSetPolicy::Cliff}) {
        Fixture f;
        if (!f.init()) return;
        auto cfg = base_config(LocalityMode::Persist);
        cfg.hot_set_policy = policy;
        cfg.hot_set_model = HotSetModel::TokenFootprint;
        CudaLocalityController c;
        if (c.initialize(0, cfg) != cudaSuccess) return;
        if (c.bind_streams(f.compute, f.prefetch) != cudaSuccess) return;
        if (c.l2_set_aside_bytes() == 0) return;

        // 4096 sequences of 8 layers x 256 KiB is far past any set-aside.
        const auto cur = f.slice(0), nxt = f.slice(1);
        c.before_layer(&cur, 1, &nxt, 1, true, geometry(4096), nullptr);
        c.after_layer();
        CHECK(c.stats().hot_set_oversubscribed == 1);
        if (policy == HotSetPolicy::Fixed) CHECK(c.stats().hit_ratio_reduced == 0);
        else                               CHECK(c.stats().hit_ratio_reduced == 1);
    }
}

// A controller destroyed while its stream is still capturing must not poison the caller's
// context. Found by compute-sanitizer: this produced 14 cudaErrorStreamCaptureUnsupported
// errors, because cudaMalloc/cudaFree/cudaDeviceSetLimit are all illegal during capture.
// REGRESSION: the fork that pulls the prefetch stream into a capture was emitted before the
// flag saying a join is owed. An error between the two - including a STALE error picked up by
// pre_touch's bare cudaGetLastError() - left the capture unjoined with nothing recording it,
// so end_sequence() emitted no join and the host runtime's cudaStreamEndCapture failed.
void test_capture_survives_a_stale_error_at_the_fork() {
    // This test deliberately provokes a failing cudaMalloc, which compute-sanitizer reports
    // as an error because it reports every error-returning API call. RECURLOCAL_TEST_NO_
    // DELIBERATE_ERRORS lets scripts/sanitize.sh skip it and say so, rather than the script
    // carrying a magic "2 expected errors" baseline that would mask a third, real one.
    if (std::getenv("RECURLOCAL_TEST_NO_DELIBERATE_ERRORS")) {
        std::printf("[skip] stale-error-at-fork test (deliberate CUDA errors suppressed)\n");
        return;
    }
    Fixture f;
    if (!f.init()) return;
    auto cfg = base_config(LocalityMode::Prefetch);
    cfg.prefetch_join = PrefetchJoin::TokenEnd;
    CudaLocalityController c;
    if (c.initialize(0, cfg) != cudaSuccess) return;
    if (c.bind_streams(f.compute, f.prefetch) != cudaSuccess) return;

    // Latch an unrelated error on this thread, exactly as a host runtime that probes a
    // capability and handles the failure locally would leave behind. SparkInfer's own cu()
    // helper logs without clearing, so this is not hypothetical.
    void* doomed = nullptr;
    const auto latched = cudaMalloc(&doomed, static_cast<std::size_t>(-1));
    CHECK(latched != cudaSuccess);
    // Deliberately NOT calling cudaGetLastError() here: the whole point is that the error is
    // still pending when pre_touch's own bare cudaGetLastError() picks it up and reports it
    // as if the launch had failed.

    CHECK_CUDA(cudaStreamBeginCapture(f.compute, cudaStreamCaptureModeThreadLocal));
    c.begin_sequence();
    const auto cur = f.slice(0), nxt = f.slice(1);
    c.before_layer(&cur, 1, &nxt, 1, true, geometry(), nullptr);   // return code deliberately ignored
    c.after_layer();
    c.end_sequence();                                             // must emit the owed join
    cudaGraph_t g = nullptr;
    const auto end = cudaStreamEndCapture(f.compute, &g);
    CHECK(end == cudaSuccess);          // NOT cudaErrorStreamCaptureUnjoined
    CHECK(g != nullptr);                // and NOT a null graph the runtime would replay
    if (g) cudaGraphDestroy(g);
}

void test_release_during_active_capture_does_not_poison_the_context() {
    Fixture f;
    if (!f.init()) return;
    CHECK_CUDA(cudaGetLastError());          // start from a clean context
    {
        CudaLocalityController c;
        if (c.initialize(0, base_config(LocalityMode::Combined)) != cudaSuccess) return;
        if (c.bind_streams(f.compute, f.prefetch) != cudaSuccess) return;
        CHECK_CUDA(cudaStreamBeginCapture(f.compute, cudaStreamCaptureModeThreadLocal));
        // ...and now it goes out of scope mid-capture, as it would on an exception unwind.
    }
    cudaGraph_t g = nullptr;
    cudaStreamEndCapture(f.compute, &g);
    if (g) cudaGraphDestroy(g);
    CHECK(cudaGetLastError() == cudaSuccess);   // the caller's context is still usable
}

void test_uninitialised_controller_is_inert_not_crashy() {
    CudaLocalityController c;
    CHECK(c.status() != cudaSuccess);
    CHECK(c.l2_set_aside_bytes() == 0);
    const StateSegment* none = nullptr;   // nullptr alone is ambiguous across the overloads
    CHECK(c.before_layer(none, 0, none, 0, false, geometry(), nullptr) != cudaSuccess);
    CHECK(c.after_layer() != cudaSuccess);
    CHECK(c.end_sequence() != cudaSuccess);
    CHECK(c.attach_window_to_captured_node() != cudaSuccess);
    c.begin_sequence();          // must not crash
    CHECK(!c.graph_capture_active());
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("[SKIP] no CUDA device; controller tests need one\n");
        return 0;
    }
    test_initialize_does_not_steal_the_current_device();
    test_l2_set_aside_is_returned();
    test_bind_streams_rejects_one_stream_used_twice();
    test_window_is_scoped_to_the_layer_outside_capture();
    test_window_attach_default_is_honoured_on_both_overloads();
    test_capture_node_attach_is_opt_in_and_works();
    test_capture_stays_valid_under_both_join_policies();
    test_every_pre_touch_strategy_is_read_only();
    test_row_major_pre_touch_reads_device_pointer_array();
    test_oversubscription_is_counted_under_every_policy();
    test_capture_survives_a_stale_error_at_the_fork();
    test_release_during_active_capture_does_not_poison_the_context();
    test_uninitialised_controller_is_inert_not_crashy();

    if (g_failures) { std::printf("%d/%d controller check(s) failed\n", g_failures, g_checks); return 1; }
    std::printf("controller tests passed (%d checks)\n", g_checks);
    return 0;
}
