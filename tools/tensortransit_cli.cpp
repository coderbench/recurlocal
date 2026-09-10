// tensortransit -- inspect a trace, build a plan from it, compare planners, report.
//
// The point of this tool is that the interesting part of this project can be worked on
// without a GPU. A trace is a file; a plan is a function of that file, a device profile and
// a config; and two plans can be diffed. So a contributor can write a planner, see exactly
// what it would do to a real recorded workload, and compare it against the four other arms
// -- all before anyone allocates hardware to measure whether the model was right.
//
// It is deliberately NOT required by the runtime library (spec section 47): a runtime embeds
// the core and never needs this binary.
//
// Nothing this tool prints is a measurement. Every figure derived from the cost model is
// labelled, because the one thing this repository has been repeatedly bitten by is a
// confident number whose provenance was not on the page.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tensortransit/ceiling.h"
#include "tensortransit/planner.h"
#include "tensortransit/runtime.h"
#include "tensortransit/trace.h"

using namespace tensortransit;

namespace {

int usage(const char* program) {
    std::fprintf(stderr,
        "usage: %s <command> [options]\n"
        "\n"
        "commands:\n"
        "  inspect <trace.json>          summarise a trace: roles, reuse, the ceiling\n"
        "  plan <trace.json>             build a plan and print it\n"
        "  compare <trace.json>          run the five policy arms over one trace\n"
        "  replay <plan.json>            feed a serialized plan back to an executor\n"
        "  devices                       list the device profiles this build knows\n"
        "  planners                      list the registered planners\n"
        "\n"
        "options:\n"
        "  --planner NAME                baseline|recurrent_v0|greedy|budgeted|concurrency\n"
        "  --device NAME                 device profile to plan against (default rtx5090)\n"
        "  --admission RULE              density|quota|proportional|reuse_order|role_floor|survival\n"
        "  --reuse-metric METRIC         ordinal|bytes|time\n"
        "  --window-binding BINDING      per_consumer|sticky\n"
        "  --budget-fraction F           share of persisting capacity to ask for\n"
        "  --hit-ratio F                 requested hit ratio\n"
        "  --cost-model M                linear | residency (default residency)\n"
        "                                linear is the CONTROL: under it greedy-on-density is\n"
        "                                provably optimal, so NO admission rule can beat it\n"
        "  --residency-beta F            survival exponent; fitted 0.1100. 0 = linear\n"
        "  --reservation-cost F          what the set-aside costs the traffic it displaces\n"
        "  --stream-relief F             share of a Stream hint's bytes that stop interfering\n"
        "  --window-preference P         densest | widest | narrowest | soonest_reuse\n"
        "  --max-windows-per-kernel N    default 1, which is what a kernel node carries and\n"
        "                                what the adapter sets; 0 = unbounded, which is what\n"
        "                                this tool used to assume and the hardware does not\n"
        "  --role-floor-share F          role_floor only: share reserved per role\n"
        "  --max-reuse-budgets F         decline reuse further than F budgets away\n"
        "  --requests N                  active requests to plan for\n"
        "  --prefetch                    enable prefetch actions\n"
        "  --json                        machine-readable output on stdout\n",
        program);
    return 2;
}

struct Options {
    std::string planner = "budgeted";
    std::string device = "rtx5090";
    TransitPlannerConfig config{};
    int requests = 0;  // 0 = take it from the trace
    bool json = false;
};

bool parse_options(int argc, char** argv, int start, Options* out) {
    // Defaults that make the tool useful on a mixed trace out of the box: coordinate the two
    // roles a hybrid model actually has, and let the weight stream be told to get out of the
    // way. A tool whose default plan is empty teaches nothing.
    out->config.persist_roles =
        RoleMask::of(TensorRole::RecurrentState, TensorRole::KVCache);
    out->config.stream_roles = RoleMask::of(TensorRole::ModelWeight, TensorRole::ExpertWeight);
    out->config.prefetch_roles = RoleMask::of(TensorRole::RecurrentState);
    // ONE window per kernel, which is what the SparkInfer adapter sets and what a kernel node
    // actually carries. The library's own default is 0 (unbounded) and stays that way -- a
    // library should not assume a device -- but this tool exists to tell a contributor what
    // their plan will do ON HARDWARE, and until 0.2.1 it answered a different question.
    //
    // It is not a cosmetic difference. On the recorded 64-kernel graph the `global` arm's
    // predicted gain goes from +0.161% unbounded to +0.006% capped, because its two mechanisms
    // -- persist the state, tell the weight stream to get out of the way -- compete for the one
    // access-policy window each kernel node has. Unbounded, `global` beats the best independent
    // arm on all five golden traces. Capped, it loses on both traces recorded from the runtime
    // and wins only on the hand-written ones, which model 176 kernels where the runtime has 64.
    // Pass `--max-windows-per-kernel 0` to see what an unconstrained planner would do.
    out->config.max_windows_per_kernel = 1;

    for (int i = start; i < argc; ++i) {
        const char* arg = argv[i];
        const auto next = [&](const char** value) {
            if (i + 1 >= argc) return false;
            *value = argv[++i];
            return true;
        };
        const char* value = nullptr;
        if (std::strcmp(arg, "--planner") == 0 && next(&value)) out->planner = value;
        else if (std::strcmp(arg, "--device") == 0 && next(&value)) out->device = value;
        else if (std::strcmp(arg, "--admission") == 0 && next(&value)) {
            if (!parse_admission_rule(value, &out->config.admission)) {
                std::fprintf(stderr, "unknown admission rule: %s\n", value);
                return false;
            }
        } else if (std::strcmp(arg, "--reuse-metric") == 0 && next(&value)) {
            if (!parse_reuse_metric(value, &out->config.reuse_metric)) {
                std::fprintf(stderr, "unknown reuse metric: %s\n", value);
                return false;
            }
        } else if (std::strcmp(arg, "--window-binding") == 0 && next(&value)) {
            if (!parse_window_binding(value, &out->config.window_binding)) {
                std::fprintf(stderr, "unknown window binding: %s\n", value);
                return false;
            }
        } else if (std::strcmp(arg, "--cost-model") == 0 && next(&value)) {
            if (!parse_cost_model(value, &out->config.cost_model)) {
                std::fprintf(stderr, "unknown cost model: %s\n", value);
                return false;
            }
        } else if (std::strcmp(arg, "--window-preference") == 0 && next(&value)) {
            if (!parse_window_preference(value, &out->config.window_preference)) {
                std::fprintf(stderr, "unknown window preference: %s\n", value);
                return false;
            }
        } else if (std::strcmp(arg, "--max-windows-per-kernel") == 0 && next(&value))
            out->config.max_windows_per_kernel = std::atoi(value);
        else if (std::strcmp(arg, "--residency-beta") == 0 && next(&value))
            out->config.residency_beta = std::atof(value);
        else if (std::strcmp(arg, "--reservation-cost") == 0 && next(&value))
            out->config.reservation_cost = std::atof(value);
        else if (std::strcmp(arg, "--stream-relief") == 0 && next(&value))
            out->config.stream_relief = std::atof(value);
        else if (std::strcmp(arg, "--budget-fraction") == 0 && next(&value))
            out->config.budget_fraction = std::atof(value);
        else if (std::strcmp(arg, "--hit-ratio") == 0 && next(&value))
            out->config.hit_ratio = std::atof(value);
        else if (std::strcmp(arg, "--role-floor-share") == 0 && next(&value))
            out->config.role_floor_share = std::atof(value);
        else if (std::strcmp(arg, "--max-reuse-budgets") == 0 && next(&value))
            out->config.max_reuse_distance_budgets = std::atof(value);
        else if (std::strcmp(arg, "--requests") == 0 && next(&value))
            out->requests = std::atoi(value);
        else if (std::strcmp(arg, "--prefetch") == 0) out->config.prefetch_enabled = true;
        else if (std::strcmp(arg, "--json") == 0) out->json = true;
        else {
            std::fprintf(stderr, "unknown option: %s\n", arg);
            return false;
        }
    }
    if (const char* problem = validate(out->config)) {
        std::fprintf(stderr, "bad configuration: %s\n", problem);
        return false;
    }
    return true;
}

bool load(const char* path, TensorRegistry* registry, TransitGraph* graph, TraceMetadata* meta) {
    std::string error;
    if (!read_trace_file(path, registry, graph, meta, &error)) {
        std::fprintf(stderr, "%s: %s\n", path, error.c_str());
        return false;
    }
    if (const char* problem = graph->validate()) {
        std::fprintf(stderr, "%s: %s\n", path, problem);
        return false;
    }
    if (graph->unresolved_uses())
        std::fprintf(stderr,
                     "warning: %llu uses named tensors the trace does not declare\n",
                     static_cast<unsigned long long>(graph->unresolved_uses()));
    return true;
}

double percent(double ratio) { return (ratio - 1.0) * 100.0; }

int cmd_inspect(int argc, char** argv) {
    if (argc < 3) return usage(argv[0]);
    TensorRegistry registry;
    TransitGraph graph;
    TraceMetadata meta;
    if (!load(argv[2], &registry, &graph, &meta)) return 1;

    Options options;
    if (!parse_options(argc, argv, 3, &options)) return 2;
    DeviceProfile device{};
    device_profile_by_name(options.device.c_str(), &device);

    std::printf("trace: %s\n", argv[2]);
    std::printf("  model    %s%s%s\n", meta.model.empty() ? "(unnamed)" : meta.model.c_str(),
                meta.runtime.empty() ? "" : " on ", meta.runtime.c_str());
    std::printf("  phase    %s, %d active request(s), %s\n",
                meta.phase.empty() ? "unknown" : meta.phase.c_str(), meta.active_requests,
                graph.cyclic() ? "cyclic (one loop iteration)" : "acyclic");
    std::printf("  kernels  %zu, uses %zu, tensors %zu\n", graph.kernels().size(),
                graph.uses().size(), registry.size());
    std::printf("  traffic  %zu B per iteration\n", graph.step_traffic_bytes());
    std::printf("  peak live set %zu B", graph.peak_live_bytes());
    if (device.persisting_l2_max_bytes) {
        const double residency =
            graph.peak_live_bytes()
                ? static_cast<double>(device.persisting_l2_max_bytes) /
                      static_cast<double>(graph.peak_live_bytes())
                : 0.0;
        std::printf("  (%.1fx the %s persisting capacity of %zu B; residency %.1f%%)",
                    graph.peak_live_bytes()
                        ? static_cast<double>(graph.peak_live_bytes()) /
                              static_cast<double>(device.persisting_l2_max_bytes)
                        : 0.0,
                    options.device.c_str(), device.persisting_l2_max_bytes,
                    (residency > 1.0 ? 1.0 : residency) * 100.0);
    }
    std::printf("\n");

    std::printf("\nper role:\n");
    std::printf("  %-18s %10s %8s %14s %14s\n", "role", "tensors", "uses", "bytes", "removable");
    for (int r = 0; r <= static_cast<int>(TensorRole::MultimodalFeature); ++r) {
        const auto role = static_cast<TensorRole>(r);
        std::size_t tensors = 0, uses = 0, bytes = 0;
        for (const TensorProfile& profile : graph.profiles()) {
            if (profile.role != role) continue;
            ++tensors;
            uses += profile.uses;
            bytes += profile.bytes;
        }
        if (!tensors) continue;
        std::printf("  %-18s %10zu %8zu %14zu %14zu\n", to_string(role), tensors, uses, bytes,
                    graph.removable_bytes_in_role(role));
    }

    // The number that decides whether to spend a week here, per role and overall. A
    // density-greedy fill of the device's budget, not a proportional scaling of the
    // unlimited-cache figure: those differ whenever the candidates differ in density.
    std::printf("\nceilings [MODEL, not measurements]:\n");
    std::printf("  %-18s %12s %14s %12s %10s\n", "policy scope", "unlimited", "on this device",
                "resident", "held B");

    struct Scope { const char* label; RoleMask roles; };
    const Scope scopes[] = {
        {"recurrent_state", RoleMask::of(TensorRole::RecurrentState)},
        {"kv_cache", RoleMask::of(TensorRole::KVCache)},
        {"recurrent+kv", RoleMask::of(TensorRole::RecurrentState, TensorRole::KVCache)},
        {"every role", RoleMask::all()},
    };
    for (const Scope& scope : scopes) {
        const CeilingReport report = compute_ceiling(graph, device, scope.roles);
        if (!report.removable_bytes) continue;
        char unlimited[24];
        if (report.unbounded_saturated)
            std::snprintf(unlimited, sizeof(unlimited), "%s", "unbounded");
        else
            std::snprintf(unlimited, sizeof(unlimited), "%+.3f%%", percent(report.removable_ratio()));
        std::printf("  %-18s %12s %+13.3f%% %11.1f%% %10zu\n", scope.label, unlimited,
                    percent(report.bounded_ratio()), report.resident_fraction() * 100.0,
                    report.bounded_resident_bytes);
    }

    {
        const CeilingReport all = compute_ceiling(graph, device, RoleMask::all());
        if (all.unbounded_saturated)
            std::printf(
                "\n  \"unbounded\" is not a large number, it is a broken question: over a cyclic\n"
                "  window every tensor is re-read next iteration, so a cache of unlimited size\n"
                "  removes the whole step. The column that binds is the device one.\n");
    }

    std::printf("\nAssumptions, stated because a ceiling nobody can audit is a claim: every\n"
                "resident byte hits, replacement within the budget is perfect, and a saved\n"
                "byte converts to time at the step's average rate. A real policy reaches\n"
                "some fraction of one of these. docs/evaluation.md.\n");
    return 0;
}

int cmd_plan(int argc, char** argv) {
    if (argc < 3) return usage(argv[0]);
    TensorRegistry registry;
    TransitGraph graph;
    TraceMetadata meta;
    if (!load(argv[2], &registry, &graph, &meta)) return 1;

    Options options;
    if (!parse_options(argc, argv, 3, &options)) return 2;
    DeviceProfile device{};
    if (!device_profile_by_name(options.device.c_str(), &device)) {
        std::fprintf(stderr, "unknown device profile: %s (try `devices`)\n",
                     options.device.c_str());
        return 2;
    }

    auto planner = make_planner(options.planner.c_str(), options.config);
    if (!planner) {
        std::fprintf(stderr, "unknown planner: %s (try `planners`)\n", options.planner.c_str());
        return 2;
    }

    RuntimeState state{};
    state.active_requests = options.requests ? options.requests : meta.active_requests;
    parse_runtime_phase(meta.phase.c_str(), &state.phase);

    PlanInput input{};
    input.graph = &graph;
    input.registry = &registry;
    input.device = device;
    input.runtime = state;

    TransitPlan plan = planner->build_plan(input);
    plan.finalize();
    if (const char* problem = plan.validate()) {
        std::fprintf(stderr, "the planner produced an invalid plan: %s\n", problem);
        return 1;
    }
    std::printf("%s", options.json ? plan.to_json().c_str() : plan.to_text().c_str());
    if (options.json) std::printf("\n");
    return 0;
}

int cmd_compare(int argc, char** argv) {
    if (argc < 3) return usage(argv[0]);
    TensorRegistry registry;
    TransitGraph graph;
    TraceMetadata meta;
    if (!load(argv[2], &registry, &graph, &meta)) return 1;

    Options options;
    if (!parse_options(argc, argv, 3, &options)) return 2;
    DeviceProfile device{};
    if (!device_profile_by_name(options.device.c_str(), &device)) {
        std::fprintf(stderr, "unknown device profile: %s\n", options.device.c_str());
        return 2;
    }

    RuntimeState state{};
    state.active_requests = options.requests ? options.requests : meta.active_requests;
    parse_runtime_phase(meta.phase.c_str(), &state.phase);
    PlanInput input{};
    input.graph = &graph;
    input.registry = &registry;
    input.device = device;
    input.runtime = state;

    // The five arms of spec section 38, from one base config, so no arm differs from
    // another in more than the one thing under test.
    const PolicyPreset presets[] = {PolicyPreset::Baseline, PolicyPreset::RecurrentOnly,
                                    PolicyPreset::KVOnly, PolicyPreset::NaiveBothPersistent,
                                    PolicyPreset::Global};
    if (!options.json) {
        std::printf("policy comparison on %s, device %s, %d request(s)\n", argv[2],
                    options.device.c_str(), state.active_requests);
        std::printf("  %-16s %9s %9s %14s %12s %9s\n", "arm", "actions", "declines",
                    "committed B", "predicted", "digest");
    } else {
        std::printf("{\"comparison_schema_version\":1,\"basis\":\"model\",\"arms\":[");
    }

    bool first = true;
    for (const PolicyPreset preset : presets) {
        const TransitPlannerConfig config = preset_config(preset, options.config);
        auto planner = make_planner(preset_planner(preset), config);
        if (!planner) continue;
        TransitPlan plan = planner->build_plan(input);
        plan.finalize();
        const char* problem = plan.validate();

        if (options.json) {
            if (!first) std::printf(",");
            first = false;
            std::printf("{\"arm\":\"%s\",\"planner\":\"%s\",\"valid\":%s,\"plan\":%s}",
                        to_string(preset), preset_planner(preset), problem ? "false" : "true",
                        plan.to_json().c_str());
        } else {
            std::printf("  %-16s %9zu %9zu %14zu %+11.3f%% %9llx%s\n", to_string(preset),
                        plan.actions().size(), plan.declines().size(),
                        plan.cost().committed_bytes,
                        percent(plan.cost().predicted_throughput_ratio()),
                        static_cast<unsigned long long>(plan.digest() & 0xffffffffull),
                        problem ? "  INVALID" : "");
        }
    }

    if (options.json) {
        std::printf("]}\n");
    } else {
        std::printf(
            "\n`predicted` is a COST-MODEL output, not a measurement -- every artifact\n"
            "carrying one declares \"basis\": \"model\", and eval/test_schemas.py fails if it\n"
            "does not. Under `residency` (the default) saving goes as resident^(1+beta) with\n"
            "beta fitted to the paired hardware arms in results/, and the reservation is\n"
            "charged what it costs the traffic it displaces; under `linear` saving is\n"
            "proportional to residency, which makes greedy-on-density provably optimal and\n"
            "the whole admission axis unmeasurable. Compare the two with --cost-model.\n"
            "Whether the global arm beats the independent ones is settled by MEASURING it:\n"
            "TENSORTRANSIT_PRESET on the adapter, eval/real_eval.py, tools/tt-frontier.\n");
    }
    return 0;
}

int cmd_devices() {
    for (const char* const* name = known_device_names(); *name; ++name) {
        DeviceProfile profile{};
        device_profile_by_name(*name, &profile);
        std::printf("%s\n", *name);
        std::printf("  L2 %zu B, persisting capacity %zu B, max window %zu B\n",
                    profile.l2_bytes, profile.persisting_l2_max_bytes,
                    profile.access_policy_max_window_bytes);
        std::printf("  %d SMs, sm_%d%d, %zu B of global memory, %.0f GB/s peak\n",
                    profile.sm_count, profile.major, profile.minor, profile.global_memory_bytes,
                    static_cast<double>(profile.peak_bandwidth_bytes_per_s) / 1e9);
    }
    return 0;
}

// Replay a serialized plan against a trace: the last third of the offline planning loop.
//
// trace -> planner -> plan.json -> REPLAY. Before this the first two thirds existed and the
// third did not, which meant a plan could be dumped and read by a human but never fed back
// to an executor -- so "optimize without touching the runtime" was two thirds true.
//
// The replay runs on RecordingExecutor, not on CUDA: a plan read from a file carries no
// pointers, and rebinding it against a trace's registry gives it SYNTHETIC ones, which the
// CUDA executor refuses by design. What this checks is everything that does not need a
// device -- which kernel each action fires on, that every Persist is cleared before the step
// ends, and that every fork is joined.
int cmd_replay(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: tensortransit replay <plan.json> [--trace <trace.json>] [--json]\n"
            "  --trace T   rebind the plan's regions against T's tensor table, and walk T's\n"
            "              kernels in order. Without it the plan is replayed on the kernels\n"
            "              it names, which checks placement but not coverage.\n");
        return 2;
    }
    const char* plan_path = argv[2];
    const char* trace_path = nullptr;
    bool json = false;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--trace") == 0 && i + 1 < argc) trace_path = argv[++i];
        else if (std::strcmp(argv[i], "--json") == 0) json = true;
        else { std::fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
    }

    TransitPlan plan;
    std::string error;
    if (!read_plan_file(plan_path, &plan, &error)) {
        std::fprintf(stderr, "cannot read %s: %s\n", plan_path, error.c_str());
        return 2;
    }

    TensorRegistry registry;
    TransitGraph graph;
    TraceMetadata meta;
    std::vector<KernelId> kernels;
    if (trace_path) {
        if (!read_trace_file(trace_path, &registry, &graph, &meta, &error)) {
            std::fprintf(stderr, "cannot read %s: %s\n", trace_path, error.c_str());
            return 2;
        }
        if (!plan.rebind(registry, &error)) {
            std::fprintf(stderr, "cannot rebind the plan onto %s: %s\n", trace_path,
                         error.c_str());
            return 2;
        }
        for (const KernelEvent& kernel : graph.kernels()) kernels.push_back(kernel.id);
    } else {
        for (const TransitAction& action : plan.actions()) {
            for (const KernelId id : {action.before_kernel, action.after_kernel}) {
                if (id == kInvalidKernelId) continue;
                if (std::find(kernels.begin(), kernels.end(), id) == kernels.end())
                    kernels.push_back(id);
            }
        }
        std::sort(kernels.begin(), kernels.end());
    }

    RecordingExecutor executor;
    executor.set_plan(&plan);
    executor.begin_step();
    for (const KernelId kernel : kernels) {
        executor.before_kernel(kernel);
        executor.after_kernel(kernel);
    }
    executor.end_step();

    const ExecutorStats& stats = executor.stats();
    const bool balanced = executor.policies_balanced();
    const bool covered = executor.records().size() == plan.actions().size();

    if (json) {
        std::printf("{\"plan\":\"%s\",\"planner\":\"%s\",\"digest\":\"%016llx\","
                    "\"trace\":%s%s%s,\"rebound\":%s,"
                    "\"kernels_walked\":%zu,\"actions\":%zu,\"actions_fired\":%zu,"
                    "\"persist\":%llu,\"clear\":%llu,\"stream\":%llu,\"prefetch\":%llu,"
                    "\"events_recorded\":%llu,\"events_waited\":%llu,"
                    "\"policies_balanced\":%s,\"every_action_fired\":%s}\n",
                    plan_path, plan.planner_name().c_str(),
                    static_cast<unsigned long long>(plan.digest()),
                    trace_path ? "\"" : "null", trace_path ? trace_path : "",
                    trace_path ? "\"" : "", trace_path ? "true" : "false",
                    kernels.size(), plan.actions().size(), executor.records().size(),
                    (unsigned long long)stats.persist_applied,
                    (unsigned long long)stats.clear_applied,
                    (unsigned long long)stats.stream_applied,
                    (unsigned long long)stats.prefetch_applied,
                    (unsigned long long)stats.events_recorded,
                    (unsigned long long)stats.events_waited,
                    balanced ? "true" : "false", covered ? "true" : "false");
    } else {
        std::printf("replay %s  planner=%s  digest=%016llx\n", plan_path,
                    plan.planner_name().c_str(),
                    static_cast<unsigned long long>(plan.digest()));
        if (trace_path)
            std::printf("  rebound onto %s: %zu tensors, %zu kernels\n", trace_path,
                        registry.size(), kernels.size());
        std::printf("  walked %zu kernels, fired %zu of %zu actions\n",
                    kernels.size(), executor.records().size(), plan.actions().size());
        std::printf("  persist %llu  clear %llu  stream %llu  prefetch %llu  fork %llu  join %llu\n",
                    (unsigned long long)stats.persist_applied,
                    (unsigned long long)stats.clear_applied,
                    (unsigned long long)stats.stream_applied,
                    (unsigned long long)stats.prefetch_applied,
                    (unsigned long long)stats.events_recorded,
                    (unsigned long long)stats.events_waited);
        std::printf("  policies balanced: %s\n", balanced ? "yes" : "NO");
        if (!covered)
            std::printf("  !! %zu action(s) never fired: they name kernels this walk did not "
                        "visit, so the plan does not cover the trace it was replayed on\n",
                        plan.actions().size() - executor.records().size());
    }
    return (balanced && covered) ? 0 : 1;
}

int cmd_planners() {
    for (const char* const* name = planner_names(); *name; ++name) std::printf("%s\n", *name);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return usage(argv[0]);
    const char* command = argv[1];
    if (std::strcmp(command, "inspect") == 0) return cmd_inspect(argc, argv);
    if (std::strcmp(command, "plan") == 0) return cmd_plan(argc, argv);
    if (std::strcmp(command, "compare") == 0) return cmd_compare(argc, argv);
    if (std::strcmp(command, "replay") == 0) return cmd_replay(argc, argv);
    if (std::strcmp(command, "devices") == 0) return cmd_devices();
    if (std::strcmp(command, "planners") == 0) return cmd_planners();
    if (std::strcmp(command, "--version") == 0 || std::strcmp(command, "version") == 0) {
        std::printf("tensortransit %s\n", version_string());
        return 0;
    }
    return usage(argv[0]);
}
