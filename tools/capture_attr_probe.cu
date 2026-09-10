// Does an access-policy window set on a node of a graph that is STILL BEING CAPTURED
// survive into the instantiated graph and take effect on replay?
//
// RecurLocal's whole persist result comes from that path (WindowAttach::CaptureNode,
// src/cuda/cache_control.cu:407). CUDA does not document mutating a mid-capture node, and
// nothing in the repository has ever checked that the attribute reaches the REPLAY rather
// than merely being accepted by the setter. The runtime API has no cudaGraphExecGetNodes,
// so the exec graph cannot be read back -- which is why the second half of this probe is
// behavioural: the same capture is attached with a PERSISTING window and with a STREAMING
// window over the same buffer, and the two replays are timed. A policy that never reached
// the replay cannot make them differ.
//
// nvcc -O3 -arch=sm_120 -o capture_attr_probe capture_attr_probe.cu
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
    std::exit(2); } } while (0)

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("FAIL: "); std::printf(__VA_ARGS__); \
    std::printf("\n"); ++g_fail; } else { std::printf("ok:   "); std::printf(__VA_ARGS__); \
    std::printf("\n"); } } while (0)

// Re-reads a hot region many times while streaming a cold region past it. With the hot
// region held by a persisting window the re-reads hit L2; with a streaming policy over the
// same region they do not.
__global__ void mix_kernel(const float* __restrict__ hot, size_t hotN,
                           const float* __restrict__ cold, size_t coldN,
                           float* __restrict__ out, int iters) {
    size_t tid = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    float acc = 0.f;
    for (int it = 0; it < iters; ++it) {
        for (size_t i = tid; i < hotN; i += stride) acc += hot[i];
        for (size_t i = tid; i < coldN; i += stride) acc += cold[i] * 1e-9f;
    }
    if (tid == 0) out[0] = acc;
}

struct Capture {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    std::vector<cudaGraphNode_t> touched;   // nodes we set the attribute on, mid-capture
    int set_ok = 0, set_fail = 0;
    bool invalidated = false;
};

// Capture `nodes` launches, setting `attr` on each recorded kernel node WHILE THE CAPTURE
// IS STILL OPEN -- exactly what CudaLocalityController::attach_window_to_captured_node does.
static Capture capture_with_attr(cudaStream_t s, int nodes, const float* hot, size_t hotN,
                                 const float* cold, size_t coldN, float* out, int iters,
                                 const cudaAccessPolicyWindow* win) {
    Capture c;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (int n = 0; n < nodes; ++n) {
        mix_kernel<<<256, 256, 0, s>>>(hot, hotN, cold, coldN, out, iters);
        if (!win) continue;

        cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
        unsigned long long id = 0;
        const cudaGraphNode_t* deps = nullptr;
        size_t dep_count = 0;
        cudaGraph_t g = nullptr;
#if CUDART_VERSION >= 13000
        const cudaGraphEdgeData* edges = nullptr;
        cudaError_t e = cudaStreamGetCaptureInfo(s, &st, &id, &g, &deps, &edges, &dep_count);
#else
        cudaError_t e = cudaStreamGetCaptureInfo_v2(s, &st, &id, &g, &deps, &dep_count);
#endif
        if (e != cudaSuccess || st != cudaStreamCaptureStatusActive || !deps || !dep_count) {
            cudaGetLastError(); ++c.set_fail; continue;
        }
        cudaKernelNodeAttrValue av{};
        av.accessPolicyWindow = *win;
        for (size_t i = 0; i < dep_count; ++i) {
            cudaGraphNodeType t{};
            if (cudaGraphNodeGetType(deps[i], &t) != cudaSuccess) { cudaGetLastError(); continue; }
            if (t != cudaGraphNodeTypeKernel) continue;
            if (cudaGraphKernelNodeSetAttribute(deps[i], cudaKernelNodeAttributeAccessPolicyWindow,
                                                &av) != cudaSuccess) {
                cudaGetLastError(); ++c.set_fail; continue;
            }
            ++c.set_ok;
            c.touched.push_back(deps[i]);
        }
        cudaStreamCaptureStatus after = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(s, &after) == cudaSuccess &&
            after == cudaStreamCaptureStatusInvalidated) c.invalidated = true;
    }
    CK(cudaStreamEndCapture(s, &c.graph));
    CK(cudaGraphInstantiate(&c.exec, c.graph, 0));
    return c;
}

static float time_replay(cudaGraphExec_t exec, cudaStream_t s, int reps) {
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 3; ++i) CK(cudaGraphLaunch(exec, s));   // warm
    CK(cudaStreamSynchronize(s));
    CK(cudaEventRecord(a, s));
    for (int i = 0; i < reps; ++i) CK(cudaGraphLaunch(exec, s));
    CK(cudaEventRecord(b, s));
    CK(cudaStreamSynchronize(s));
    float ms = 0.f; CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return ms / reps;
}

int main(int argc, char** argv) {
    const int NODES = (argc > 1) ? atoi(argv[1]) : 48;   // SparkInfer's batch-1 decode capture
    const int ITERS = (argc > 2) ? atoi(argv[2]) : 8;
    int dev = 0; CK(cudaSetDevice(dev));
    cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, dev));
    std::printf("device: %s  sm_%d%d  L2 %zu MiB  persistingL2max %zu MiB  maxWindow %zu MiB\n",
                prop.name, prop.major, prop.minor, (size_t)prop.l2CacheSize >> 20,
                (size_t)prop.persistingL2CacheMaxSize >> 20,
                (size_t)prop.accessPolicyMaxWindowSize >> 20);

    size_t setaside = prop.persistingL2CacheMaxSize;
    CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, setaside));
    size_t granted = 0; CK(cudaDeviceGetLimit(&granted, cudaLimitPersistingL2CacheSize));
    std::printf("set-aside requested %zu MiB, granted %zu MiB\n", setaside >> 20, granted >> 20);

    // Hot region sized to sit inside the set-aside; cold region large enough to evict it.
    size_t hotN  = (granted ? granted : (32u << 20)) / sizeof(float) / 2;
    // Per-NODE cold bytes, so the whole GRAPH streams a constant ~512 MiB whatever the node
    // count. Sizing it per node instead made the cold stream 48x larger at 48 nodes than at 1,
    // which buried the hot region's re-read under traffic that has nothing to do with the
    // policy -- the probe then reported no difference and it was the benchmark, not CUDA.
    size_t coldN = ((512u << 20) / (unsigned)NODES) / sizeof(float);
    if (coldN < (1u << 20) / sizeof(float)) coldN = (1u << 20) / sizeof(float);
    float *hot = nullptr, *cold = nullptr, *out = nullptr;
    CK(cudaMalloc(&hot, hotN * sizeof(float)));
    CK(cudaMalloc(&cold, coldN * sizeof(float)));
    CK(cudaMalloc(&out, sizeof(float)));
    CK(cudaMemset(hot, 0, hotN * sizeof(float)));
    CK(cudaMemset(cold, 0, coldN * sizeof(float)));
    std::printf("hot %zu MiB, cold %zu MiB, %d nodes, %d iters/kernel\n\n",
                hotN * sizeof(float) >> 20, coldN * sizeof(float) >> 20, NODES, ITERS);

    cudaStream_t s; CK(cudaStreamCreate(&s));

    cudaAccessPolicyWindow persist{};
    persist.base_ptr = (void*)hot;
    persist.num_bytes = hotN * sizeof(float);
    if (persist.num_bytes > (size_t)prop.accessPolicyMaxWindowSize)
        persist.num_bytes = prop.accessPolicyMaxWindowSize;
    persist.hitRatio = 1.0f;
    persist.hitProp = cudaAccessPropertyPersisting;
    persist.missProp = cudaAccessPropertyStreaming;

    cudaAccessPolicyWindow stream_win = persist;
    stream_win.hitProp = cudaAccessPropertyStreaming;   // same region, opposite intent

    // ---- 1. the mid-capture set is accepted and does not invalidate the capture ----
    Capture cp = capture_with_attr(s, NODES, hot, hotN, cold, coldN, out, ITERS, &persist);
    CHECK(cp.set_ok == NODES, "mid-capture attribute set on %d/%d kernel nodes (%d failures)",
          cp.set_ok, NODES, cp.set_fail);
    CHECK(!cp.invalidated, "capture never reported invalidated after a mid-capture attribute set");

    // ---- 2. the attribute is present on the CAPTURED graph after EndCapture ----
    size_t ngraph = 0; CK(cudaGraphGetNodes(cp.graph, nullptr, &ngraph));
    std::vector<cudaGraphNode_t> gnodes(ngraph);
    CK(cudaGraphGetNodes(cp.graph, gnodes.data(), &ngraph));
    int present = 0, correct = 0, kernels = 0;
    for (size_t i = 0; i < ngraph; ++i) {
        cudaGraphNodeType t{};
        if (cudaGraphNodeGetType(gnodes[i], &t) != cudaSuccess || t != cudaGraphNodeTypeKernel) continue;
        ++kernels;
        cudaKernelNodeAttrValue got{};
        if (cudaGraphKernelNodeGetAttribute(gnodes[i], cudaKernelNodeAttributeAccessPolicyWindow,
                                            &got) != cudaSuccess) { cudaGetLastError(); continue; }
        if (got.accessPolicyWindow.num_bytes == 0) continue;
        ++present;
        if (got.accessPolicyWindow.base_ptr == persist.base_ptr &&
            got.accessPolicyWindow.num_bytes == persist.num_bytes &&
            got.accessPolicyWindow.hitProp == persist.hitProp) ++correct;
    }
    CHECK(kernels == NODES, "captured graph holds %d kernel nodes (expected %d)", kernels, NODES);
    CHECK(present == NODES, "window present on %d/%d kernel nodes of the finished graph", present, NODES);
    CHECK(correct == NODES, "window matches what was set on %d/%d nodes", correct, NODES);

    // ---- 3. a CLONE carries it too (the clone is what an update path would diff against) ----
    cudaGraph_t clone = nullptr;
    if (cudaGraphClone(&clone, cp.graph) == cudaSuccess) {
        size_t nc = 0; CK(cudaGraphGetNodes(clone, nullptr, &nc));
        std::vector<cudaGraphNode_t> cn(nc); CK(cudaGraphGetNodes(clone, cn.data(), &nc));
        int cpresent = 0;
        for (size_t i = 0; i < nc; ++i) {
            cudaGraphNodeType t{};
            if (cudaGraphNodeGetType(cn[i], &t) != cudaSuccess || t != cudaGraphNodeTypeKernel) continue;
            cudaKernelNodeAttrValue got{};
            if (cudaGraphKernelNodeGetAttribute(cn[i], cudaKernelNodeAttributeAccessPolicyWindow,
                                                &got) == cudaSuccess &&
                got.accessPolicyWindow.num_bytes == persist.num_bytes) ++cpresent;
            else cudaGetLastError();
        }
        CHECK(cpresent == NODES, "clone carries the window on %d/%d nodes", cpresent, NODES);
        cudaGraphDestroy(clone);
    } else { cudaGetLastError(); std::printf("note: cudaGraphClone unavailable\n"); }

    // ---- 4. BEHAVIOURAL: does it reach the REPLAY? ----
    // There is no cudaGraphExecGetNodes, so the exec graph cannot be inspected. Instead run
    // three captures that differ ONLY in the mid-capture attribute and time their replays.
    // If the attribute never reached the replay all three would time the same.
    Capture cs = capture_with_attr(s, NODES, hot, hotN, cold, coldN, out, ITERS, &stream_win);
    Capture cn = capture_with_attr(s, NODES, hot, hotN, cold, coldN, out, ITERS, nullptr);

    const int REPS = 20;
    float t_persist = time_replay(cp.exec, s, REPS);
    float t_stream  = time_replay(cs.exec, s, REPS);
    float t_none    = time_replay(cn.exec, s, REPS);
    // interleave once more, same order reversed, to blunt drift
    t_none    = 0.5f * (t_none    + time_replay(cn.exec, s, REPS));
    t_stream  = 0.5f * (t_stream  + time_replay(cs.exec, s, REPS));
    t_persist = 0.5f * (t_persist + time_replay(cp.exec, s, REPS));

    std::printf("\nreplay ms/graph:  no-window %.4f   persisting %.4f   streaming %.4f\n",
                t_none, t_persist, t_stream);
    const float rel = (t_stream - t_persist) / t_none * 100.f;
    std::printf("persisting vs streaming over the same region: %+.2f%% of the no-window time\n", rel);
    std::printf("hot re-read share of the graph's traffic: %.1f%%\n",
                100.0 * (double)(hotN * sizeof(float) * (size_t)ITERS * NODES) /
                (double)((hotN + coldN) * sizeof(float) * (size_t)ITERS * NODES));
    CHECK(t_stream > t_persist,
          "a persisting window replays faster than a streaming one over the same region "
          "(%.4f vs %.4f ms) -- the attribute set mid-capture IS live in the replay", t_persist, t_stream);

    // ---- 5. re-instantiating the SAME graph keeps it ----
    cudaGraphExec_t exec2 = nullptr;
    CK(cudaGraphInstantiate(&exec2, cp.graph, 0));
    float t2 = time_replay(exec2, s, REPS);
    std::printf("second instantiation of the same graph: %.4f ms (first %.4f)\n", t2, t_persist);
    // The same graph instantiated twice must behave the same. Compared against the streaming
    // arm rather than against an absolute tolerance, so this asserts the POLICY survived
    // re-instantiation rather than asserting a timing reproducibility the box cannot promise
    // (its clocks cannot be pinned).
    CHECK(t2 < t_stream, "a second instantiation of the same graph still carries the policy");

    std::printf("\n%s (%d failure(s))\n", g_fail ? "PROBE FAILED" : "PROBE PASSED", g_fail);
    return g_fail ? 1 : 0;
}
