#!/usr/bin/env python3
"""Produce the one measurement that decides this project: real end-to-end decode.

Everything else RecurLocal measures is explanation. Sections 17, 21 and 28 say the score is
real serving throughput on a pinned model and runtime, and until this script existed there
was nothing that could produce a `real-result.json` for `decide.py --real` to read.

Shape of the experiment, and why:

  one binary          The hook is inert unless RECURLOCAL names a mode, so control and
                      candidate are the same executable with a different environment. Two
                      separately linked binaries cannot separate the policy from the link.

  interleaved         Graphics clocks cannot be pinned on the measurement box, so absolute
                      numbers drift with temperature over minutes. control/candidate pairs
                      run back to back and are compared within the pair; the medians of the
                      per-pair ratios are what get reported.

  token-exact gate    The exact-locality track forbids changing model output. The gate is a
                      greedy replay: identical prompt, identical token ids, compared as a
                      list and not as a score. A candidate that fails it is not scored.

  noise floor         Control repeats are compared against each other to get the run-to-run
                      spread of the measurement itself. A gain inside that spread is
                      reported as unresolved rather than as a win.

Usage:
    eval/real_eval.py --binary .../qwen3_gguf_bench --generate .../qwen3_gguf_generate \\
                      --model /path/to/checkpoint --output real-result.json \\
                      --candidate RECURLOCAL=combined RECURLOCAL_PREFETCH_DISTANCE=1
"""
import argparse, fcntl, json, os, re, statistics, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1

# Section 44's matrix. Batch 1 is the only arm a single-sequence decode benchmark can fill;
# the concurrency arms need a runtime path that decodes several sequences at once, and the
# runner refuses to invent them (see --concurrency).
DEFAULT_WEIGHTS = {"batch1": 0.40, "concurrency4": 0.20,
                   "concurrency16": 0.20, "concurrency32": 0.20}


# The model is 21 GB of VRAM and the driver does not always have it back by the time the
# next process asks for it, so a back-to-back arm can fail to load for a reason that has
# nothing to do with what is being measured. Settle, then retry once; a second failure is
# real and is reported.
SETTLE_SECONDS = 0.0 if os.environ.get('RECURLOCAL_EVAL_FAST') else 3.0
# Two eval processes on one GPU do not merely go slower: the second fails to load 21 GB of
# weights, the harness retries into the same contention, and what comes out is a number for a
# run that never happened. That is not hypothetical - it happened twice while this harness was
# being built, and both times the failure looked like a result. An advisory lock is cheap and
# turns a corrupt measurement into a wait.
GPU_LOCK_PATH = os.environ.get("RECURLOCAL_EVAL_LOCK", "/tmp/recurlocal-eval.lock")


class GpuLock:
    def __init__(self, path=GPU_LOCK_PATH, verbose=True):
        self.path, self.verbose, self.fh = path, verbose, None

    def __enter__(self):
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            if self.verbose:
                print(f">> another eval holds {self.path}; waiting rather than racing it for VRAM",
                      flush=True)
            fcntl.flock(self.fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        if self.fh:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()
        return False
LOAD_FAILURE_MARKERS = ("[FAIL] load", "out of memory", "cudaErrorMemoryAllocation")


# Every RECURLOCAL_* name the adapter reads. The control arm must have all of them scrubbed
# from the inherited environment, not merely left unset by the caller: an operator with
# `export RECURLOCAL=combined` in their shell would otherwise run a hooked "control", and the
# harness would report ~0% for a comparison of the candidate against itself.
ADAPTER_ENV_PREFIX = "RECURLOCAL"


def run(cmd, env_extra, timeout=1800, scrub_adapter_env=False):
    env = dict(os.environ)
    if scrub_adapter_env:
        for k in [k for k in env if k.startswith(ADAPTER_ENV_PREFIX)]:
            del env[k]
    env.update(env_extra)
    t0 = time.time()
    time.sleep(SETTLE_SECONDS)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    out = p.stdout + p.stderr
    if p.returncode != 0 and any(m in out for m in LOAD_FAILURE_MARKERS):
        time.sleep(SETTLE_SECONDS * 4)
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
        out = p.stdout + p.stderr
    return p.returncode, out, time.time() - t0


def parse_sweep(text):
    """The bench prints one SWEEP_JSON line per process: {ctx: {decode_tps, prefill_pp}}."""
    m = re.search(r"SWEEP_JSON (\{.*\})", text)
    if not m:
        return None
    return json.loads(m.group(1))


def parse_adapter_stats(text):
    """The hook's own JSON, printed to stderr. Absent for the control, which is the point:
    it is how a result file records that the control really was unhooked."""
    m = re.search(r"RECURLOCAL_STATS (\{.*\})", text)
    return json.loads(m.group(1)) if m else None


def is_control(env_extra):
    return env_extra.get("RECURLOCAL", "off") in ("off", "0")


def policy_applied(stats):
    """Did this run actually DO anything, or merely load?

    A window that is computed and handed back is not a window that reached the kernel. Under
    CUDA-Graph decode with the safe WindowAttach::Stream, `persist` defers every window to a
    runtime that does not attach it: windows_applied 0, windows_attached_to_node 0,
    pre_touch_launches 0, and 192 windows_deferred_to_caller. That configuration is a null
    candidate — it measures the hook's overhead against the control and nothing else — and
    the first version of this harness scored exactly that as a go/no-go verdict because the
    guard only checked that the hook had initialised.
    """
    st = stats.get("stats", {}) if stats else {}
    return (st.get("windows_applied", 0) + st.get("windows_attached_to_node", 0)
            + st.get("pre_touch_launches", 0)) > 0


def hook_ran(stats):
    """Did the adapter ever initialise in this process?

    ever_initialised, not initialised: the stats snapshot is printed at process exit, by
    which time the runtime's model destructor has already called shutdown() and cleared the
    live flag. Older adapters do not emit the field, so fall back to the live one.

    One definition, used by both the guard that refuses an unhooked candidate and the
    correctness record. They disagreed once: the record read the live flag and so said
    candidate_hook_active=false on every run that worked, which reads as "the token-exact
    gate compared the control against itself" - the opposite of what the gate proved.
    """
    if not stats:
        return False
    return bool(stats.get("ever_initialised", stats.get("initialised")))


def require_hook_engaged(text, env_extra, label):
    """A candidate run that produced no telemetry ran the control.

    The adapter turns itself off on a configuration it cannot parse, which is the right
    behaviour for a runtime and the wrong one for a measurement: the run would complete,
    the number would look fine, and it would be a number for the unhooked binary carrying
    the candidate's label. Every arm that asked for a mode has to prove it got one."""
    if is_control(env_extra):
        return None
    stats = parse_adapter_stats(text)
    if stats is None:
        raise SystemExit(f"{label}: candidate asked for RECURLOCAL={env_extra['RECURLOCAL']} but the "
                         "run emitted no RECURLOCAL_STATS line. The binary is unhooked, or the "
                         "adapter refused the configuration. Refusing to report it as a candidate.\n"
                         + text[-2000:])
    if not hook_ran(stats) or stats.get("broken"):
        raise SystemExit(f"{label}: the hook did not initialise (ever_initialised="
                         f"{stats.get('ever_initialised')}, initialised={stats.get('initialised')}, "
                         f"broken={stats.get('broken')}, "
                         f"config_error={stats.get('config_error')}).\n" + text[-2000:])
    if stats.get("stats", {}).get("layers", 0) == 0:
        raise SystemExit(f"{label}: the hook initialised but bracketed no recurrent layer. "
                         "Either the model is not hybrid or the hook site was not reached.")
    if env_extra.get("RECURLOCAL") != "baseline" and not policy_applied(stats):
        st = stats.get("stats", {})
        raise SystemExit(
            f"{label}: NULL CANDIDATE. The hook loaded and bracketed "
            f"{st.get('layers')} layers but applied no policy at all — "
            f"windows_applied={st.get('windows_applied')}, "
            f"windows_attached_to_node={st.get('windows_attached_to_node')}, "
            f"pre_touch_launches={st.get('pre_touch_launches')}, "
            f"windows_deferred_to_caller={st.get('windows_deferred_to_caller')}.\n"
            "Deferred windows are computed and handed back; unless the runtime attaches them "
            "they never reach a kernel. Scoring this would report the hook's overhead as a "
            "locality result. Use RECURLOCAL=baseline if measuring overhead is the intent.")
    return stats


# Below this share of a concurrency arm's tokens going through the runtime's packed decode
# path, the run did not measure concurrent decode at all -- it measured the single-sequence
# path executed once per row. A tail chunk of one row always falls through, so the normal
# figure is high but not 1.0 (the pinned integration measures 133 of 142 at concurrency 8);
# a collapse is near zero, not a few percent short.
PACKED_SHARE_MIN = 0.50


def packed_path_used(stats, concurrency):
    """Did the runtime actually batch, or did it fall back to decoding one row at a time?

    This is the 32-sequence cliff, made visible. SparkInfer declines a packed forward for
    reasons that have nothing to do with locality -- a row set that moved, a tail chunk of one
    row, an unsupported shape -- and when it does, aggregate throughput drops by about a third
    while every other number in the run looks normal. A median over repeats then turns one
    collapsed run into a plausible-looking 'result' for whatever configuration happened to be
    running.

    The adapter already counts what settles it: `tokens_packed` of `tokens`, and
    `max_rows_seen`. Returns a record for the artifact, and whether the arm is usable.

    Limitation, stated rather than hidden: the control arm is unhooked by construction, so it
    emits no telemetry and a collapse THERE is still invisible. What this catches is a
    collapse in the candidate, which is where both observed ones were.
    """
    if not stats or concurrency < 2:
        return None
    st = stats.get("stats", {})
    tokens = st.get("tokens", 0) or 0
    packed = st.get("tokens_packed", 0) or 0
    share = (packed / tokens) if tokens else 0.0
    return {"tokens": tokens, "tokens_packed": packed, "packed_share": share,
            "max_rows_seen": st.get("max_rows_seen", 0),
            "layers_packed": st.get("layers_packed", 0),
            "concurrency_asked": concurrency,
            "used_packed_path": share >= PACKED_SHARE_MIN}


def require_packed_path(stats, concurrency, label):
    """A concurrency measurement that ran the per-row path is not a concurrency measurement."""
    rec = packed_path_used(stats, concurrency)
    if rec is None or rec["used_packed_path"]:
        return rec
    raise SystemExit(
        f"{label}: RUNTIME FELL OFF THE BATCHED DECODE PATH. Asked for {concurrency} "
        f"concurrent sequences; the runtime packed {rec['tokens_packed']} of {rec['tokens']} "
        f"tokens ({rec['packed_share'] * 100:.1f}%), max_rows_seen={rec['max_rows_seen']}. "
        "Aggregate throughput from a run that decoded one row at a time is not this "
        "workload's number, and averaging it with runs that did batch produces a gain for a "
        "measurement that never happened. Re-run the arm.")


def parse_output_ids(text):
    m = re.search(r"OUTPUT_IDS:((?: -?\d+)+)", text)
    return [int(x) for x in m.group(1).split()] if m else None


def median(xs):
    return statistics.median(xs) if xs else 0.0


def geomean(xs):
    if not xs:
        return 1.0
    prod = 1.0
    for x in xs:
        prod *= x
    return prod ** (1.0 / len(xs))


def batch1_arm(per_context, ctxs):
    """Combine the per-context results into the single batch-1 workload decide.py scores.

    One estimator, everywhere. This used to be a ratio *of medians*
    (median(candidate)/median(baseline)) while every other arm — and this module's own
    documented method — used the median *of paired ratios*. On tight data the two agree; on
    the drifting clocks this harness is designed for they do not, and the arm that disagreed
    carried the highest weight in the verdict. Controls of [100, 110, 120] against candidates
    of [105, 110, 132] are +5.0% paired and +0.0% as a ratio of medians: same measurements,
    opposite go/no-go.

    Pairing is the whole point of interleaving. Discarding it throws away the only thing that
    makes a same-box comparison valid.

    Lives out here, not inline in main(), so it can be tested without a GPU and a 21 GB
    checkpoint — which is why the inconsistency survived as long as it did.
    """
    ratios, refs = [], []
    for c in ctxs:
        arm = per_context[str(c)]
        if arm["paired_ratios"]:
            ratios.append(median(arm["paired_ratios"]))
        if arm["baseline_tps"] > 0:
            refs.append(arm["baseline_tps"])
    ratio = geomean(ratios) if ratios else 1.0
    baseline_ref = geomean(refs) if refs else 0.0
    return ratio, baseline_ref


def resolution(gain_pct, *spreads):
    """Is this workload's difference bigger than the run-to-run spread that produced it?

    A median hides an outlier instead of reporting one. At 32 concurrent sequences a no-op
    configuration measured -14.8% here, from two repeats that differed by 32% -- an effect a
    policy doing nothing cannot have. The number was not wrong, it was unresolved, and the
    only thing that distinguishes those is publishing the spread beside the gain."""
    floor = max([s for s in spreads if s is not None], default=None)
    if floor is None:
        return {"resolved": None, "floor_pct": None,
                "note": "fewer than two repeats; no spread to compare against"}
    return {"resolved": abs(gain_pct) > floor, "floor_pct": floor,
            "note": "resolved=false means the difference is inside the run-to-run spread of "
                    "the runs that produced it, whichever way it points"}


def rel_spread_pct(xs):
    """Peak-to-peak spread as a fraction of the median: the noise floor of this measurement
    on this box, measured rather than assumed."""
    if len(xs) < 2:
        return None
    mid = median(xs)
    return (max(xs) - min(xs)) / mid * 100.0 if mid else None


def measure(binary, model, tokens, ctxs, env_extra, label, verbose):
    env = dict(env_extra)
    env["SPARKINFER_BENCH_SWEEP_CTXS"] = ",".join(str(c) for c in ctxs)
    code, out, secs = run([binary, model, str(tokens), "sweep"], env,
                          scrub_adapter_env=is_control(env_extra))
    if code != 0:
        raise SystemExit(f"{label}: benchmark exited {code}\n{out[-4000:]}")
    sweep = parse_sweep(out)
    if not sweep:
        raise SystemExit(f"{label}: no SWEEP_JSON in output\n{out[-4000:]}")
    if verbose:
        print(f"    {label}: " + "  ".join(f"ctx{c}={sweep[str(c)]['decode_tps']:.2f}" for c in ctxs)
              + f"   ({secs:.0f}s)", flush=True)
    return sweep, require_hook_engaged(out, env_extra, label), out


def measure_concurrent(binary, model, concurrency, prompt_len, max_new, long_prefill,
                       env_extra, label, verbose):
    """Aggregate decode throughput under continuous batching.

    A different code path from the single-sequence bench, and the one the locality question
    is actually about: model weights are read once per step however many sequences are in
    flight, so the recurrent share of decode traffic grows with concurrency while the weight
    share does not."""
    code, out, secs = run([binary, model, str(concurrency), str(prompt_len), str(max_new),
                           str(long_prefill)], env_extra,
                          scrub_adapter_env=is_control(env_extra))
    if code != 0:
        raise SystemExit(f"{label}: cb bench exited {code}\n{out[-4000:]}")
    m = re.search(r"agg_tok_s=([0-9.]+)", out)
    itl = re.search(r"mean_itl_ms=([0-9.]+)", out)
    if not m:
        raise SystemExit(f"{label}: no agg_tok_s in output\n{out[-4000:]}")
    tps = float(m.group(1))
    stats = require_hook_engaged(out, env_extra, label)
    packing = require_packed_path(stats, concurrency, label)
    if verbose:
        pk = f"  packed={packing['packed_share'] * 100:.0f}% rows={packing['max_rows_seen']}" if packing else ""
        print(f"    {label}: agg={tps:.1f} tok/s  itl={itl.group(1) if itl else '?'} ms{pk}   ({secs:.0f}s)",
              flush=True)
    return tps, stats, out


def greedy_replay(generate, model, prompt_ids, max_new, env_extra, label):
    code, out, _ = run([generate, model, str(max_new), *[str(t) for t in prompt_ids]], env_extra,
                       scrub_adapter_env=is_control(env_extra))
    if code != 0:
        raise SystemExit(f"{label}: generate exited {code}\n{out[-4000:]}")
    ids = parse_output_ids(out)
    if ids is None:
        raise SystemExit(f"{label}: no OUTPUT_IDS in output\n{out[-4000:]}")
    return ids, require_hook_engaged(out, env_extra, label)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", required=True, help="qwen3_gguf_bench from the patched build")
    ap.add_argument("--generate", help="qwen3_gguf_generate, for the token-exact gate")
    ap.add_argument("--model", required=True, help="checkpoint directory or GGUF")
    ap.add_argument("--output", type=Path, default=Path("real-result.json"))
    ap.add_argument("--tokens", type=int, default=128, help="decode tokens timed per context")
    ap.add_argument("--contexts", default="128,4096,16384",
                    help="prefill depths to time decode at; the batch-1 arm is their geometric mean")
    ap.add_argument("--repeats", type=int, default=3, help="interleaved control/candidate pairs")
    ap.add_argument("--gate-tokens", type=int, default=64, help="tokens compared by the exactness gate")
    ap.add_argument("--gate-prompt", default="9707,3837,1879,13,25001,752,911",
                    help="prompt token ids for the greedy replay gate")
    ap.add_argument("--candidate", nargs="+", required=True, metavar="KEY=VALUE",
                    help="environment that turns the hook on, e.g. RECURLOCAL=combined")
    ap.add_argument("--control", nargs="*", default=[], metavar="KEY=VALUE",
                    help="environment for the control arm; empty means the unhooked runtime")
    ap.add_argument("--cb-binary", help="qwen3_gguf_cb_bench, for the concurrency arms")
    ap.add_argument("--concurrency", default="",
                    help="comma-separated concurrencies to measure, e.g. 4,16,32; needs --cb-binary")
    ap.add_argument("--cb-prompt-len", type=int, default=128)
    ap.add_argument("--cb-max-new", type=int, default=64)
    ap.add_argument("--cb-long-prefill", type=int, default=4096)
    ap.add_argument("--label", default="", help="short name for this candidate, recorded in the result")
    ap.add_argument("--note", default="", help="free text recorded with the result")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    def as_env(pairs):
        env = {}
        for item in pairs:
            if "=" not in item:
                raise SystemExit(f"--candidate/--control entries must be KEY=VALUE, got {item!r}")
            k, v = item.split("=", 1)
            env[k] = v
        return env

    cand_env = as_env(a.candidate)
    ctrl_env = as_env(a.control)
    # The control must not carry the hook. Otherwise the comparison is between two
    # RecurLocal configurations and the result file would still call one of them "baseline".
    if ctrl_env.get("RECURLOCAL", "off") not in ("off", "0"):
        raise SystemExit("the control arm must have RECURLOCAL off; it is the unhooked runtime")
    cand_env.setdefault("RECURLOCAL_STATS", "1")
    ctxs = [int(c) for c in a.contexts.split(",") if c.strip()]
    verbose = not a.quiet

    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": None, "driver": None,
        "model": a.model,
        "binary": str(Path(a.binary).resolve()),
        "candidate_env": cand_env,
        "control_env": ctrl_env,
        "tokens_per_measurement": a.tokens,
        "contexts": ctxs,
        "repeats": a.repeats,
        "label": a.label,
        "note": a.note,
    }
    pin = Path(__file__).resolve().parent.parent / "integrations" / "sparkinfer" / "pin.json"
    if pin.exists():
        doc = json.loads(pin.read_text())
        provenance["runtime_repository"] = doc.get("repository")
        provenance["runtime_commit"] = doc.get("commit")
        provenance["model_pin"] = doc.get("model", {}).get("repository")
    try:
        provenance["recurlocal_commit"] = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip() or None
    except Exception:
        pass
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        if smi:
            provenance["gpu"], provenance["driver"] = [x.strip() for x in smi.split(",")[:2]]
    except Exception:
        pass

    lock = GpuLock(verbose=verbose)
    lock.__enter__()
    # ---- correctness gate first: a candidate that changes the output is not scored, and
    # there is no reason to spend an hour timing it.
    correctness = {"output_identical": None, "method": "greedy replay, token-exact"}
    if a.generate:
        prompt = [int(x) for x in a.gate_prompt.replace(" ", "").split(",") if x]
        if verbose:
            print(">> token-exact greedy replay gate", flush=True)
        ctrl_ids, _ = greedy_replay(a.generate, a.model, prompt, a.gate_tokens, ctrl_env, "control")
        cand_ids, gate_stats = greedy_replay(a.generate, a.model, prompt, a.gate_tokens, cand_env, "candidate")
        identical = ctrl_ids == cand_ids
        correctness.update(output_identical=identical,
                           tokens_compared=len(ctrl_ids),
                           prompt_ids=prompt,
                           first_divergence=None if identical else next(
                               (i for i, (x, y) in enumerate(zip(ctrl_ids, cand_ids)) if x != y),
                               min(len(ctrl_ids), len(cand_ids))))
        if gate_stats:
            correctness["candidate_hook_active"] = hook_ran(gate_stats)
        if verbose:
            print(f"    {'identical' if identical else 'DIVERGED'} over {len(ctrl_ids)} tokens", flush=True)
    else:
        correctness["method"] = "not run (--generate not given)"

    # ---- timing, interleaved
    concurrencies = [int(c) for c in a.concurrency.split(",") if c.strip()]
    if concurrencies and not a.cb_binary:
        raise SystemExit("--concurrency needs --cb-binary (qwen3_gguf_cb_bench)")

    ctrl_runs, cand_runs = [], []
    cb_ctrl = {n: [] for n in concurrencies}
    cb_cand = {n: [] for n in concurrencies}
    adapter_stats = None
    cb_adapter_stats = {}
    for i in range(a.repeats):
        if verbose:
            print(f">> pair {i + 1}/{a.repeats}", flush=True)
        c, _, _ = measure(a.binary, a.model, a.tokens, ctxs, ctrl_env, "control", verbose)
        d, stats, _ = measure(a.binary, a.model, a.tokens, ctxs, cand_env, "candidate", verbose)
        ctrl_runs.append(c)
        cand_runs.append(d)
        if stats:
            adapter_stats = stats
        for n in concurrencies:
            # The adapter counts sequences from the runtime's own row count, so the candidate
            # environment does not have to be told the concurrency it is running at.
            b_tps, _, _ = measure_concurrent(a.cb_binary, a.model, n, a.cb_prompt_len,
                                             a.cb_max_new, a.cb_long_prefill, ctrl_env,
                                             f"control  c={n}", verbose)
            c_tps, cstats, _ = measure_concurrent(a.cb_binary, a.model, n, a.cb_prompt_len,
                                                  a.cb_max_new, a.cb_long_prefill, cand_env,
                                                  f"candidate c={n}", verbose)
            cb_ctrl[n].append(b_tps)
            cb_cand[n].append(c_tps)
            if cstats:
                cb_adapter_stats[str(n)] = cstats

    per_context = {}
    for ctx in ctxs:
        key = str(ctx)
        base = [r[key]["decode_tps"] for r in ctrl_runs]
        cand = [r[key]["decode_tps"] for r in cand_runs]
        # Ratio per pair, then the median of the ratios: a pair is the only comparison the
        # box supports, because both halves of it saw the same clock state.
        ratios = [d / b for b, d in zip(base, cand) if b > 0]
        gain = (median(ratios) - 1.0) * 100.0 if ratios else 0.0
        per_context[key] = {
            "baseline_tps": median(base), "candidate_tps": median(cand),
            "baseline_runs": base, "candidate_runs": cand,
            "paired_ratios": ratios,
            "gain_pct": gain,
            "baseline_spread_pct": rel_spread_pct(base),
            "candidate_spread_pct": rel_spread_pct(cand),
            "resolution": resolution(gain, rel_spread_pct(base), rel_spread_pct(cand),
                                     rel_spread_pct(ratios)),
        }

    # The batch-1 workload decide.py scores is the whole context sweep, combined as a
    # geometric mean so no single context can carry the arm — using the same paired-ratio
    # estimator as every other arm. See batch1_arm().
    batch1_ratio, baseline_ref = batch1_arm(per_context, ctxs)

    # The measurement's own noise floor: the worst per-context control spread. A gain under
    # it is not resolved by this experiment, whatever its sign.
    floors = [per_context[str(c)]["baseline_spread_pct"] for c in ctxs
              if per_context[str(c)]["baseline_spread_pct"] is not None]
    noise_floor = max(floors) if floors else None
    gain_pct = (batch1_ratio - 1.0) * 100.0

    workloads = {"batch1": {"weight": DEFAULT_WEIGHTS["batch1"] if concurrencies else 1.0,
                           "baseline_tps": baseline_ref,
                           "candidate_tps": baseline_ref * batch1_ratio}}
    concurrency_detail = {}
    for n in concurrencies:
        base, cand = cb_ctrl[n], cb_cand[n]
        ratios = [d / b for b, d in zip(base, cand) if b > 0]
        name = f"concurrency{n}"
        gain = (median(ratios) - 1.0) * 100.0 if ratios else 0.0
        stats_n = cb_adapter_stats.get(str(n), {})
        concurrency_detail[name] = {
            "baseline_runs": base, "candidate_runs": cand, "paired_ratios": ratios,
            "gain_pct": gain,
            "baseline_spread_pct": rel_spread_pct(base),
            "candidate_spread_pct": rel_spread_pct(cand),
            "resolution": resolution(gain, rel_spread_pct(base), rel_spread_pct(cand),
                                     rel_spread_pct(ratios)),
            # Proof the concurrency arm measured the concurrent code path. A runtime that
            # declined to batch would decode row by row through the single-sequence path and
            # produce a perfectly good number for the wrong workload.
            "sequences_seen_by_adapter": stats_n.get("geometry", {}).get("sequences"),
            "tokens_packed": stats_n.get("stats", {}).get("tokens_packed"),
            "tokens_total": stats_n.get("stats", {}).get("tokens"),
            "max_rows_seen": stats_n.get("stats", {}).get("max_rows_seen"),
        }
        if name in DEFAULT_WEIGHTS:
            workloads[name] = {"weight": DEFAULT_WEIGHTS[name],
                               "baseline_tps": median(base),
                               "candidate_tps": median(base) * median(ratios) if ratios else median(cand)}
    measured = ["batch1"] + [f"concurrency{n}" for n in concurrencies if f"concurrency{n}" in DEFAULT_WEIGHTS]

    doc = {
        "schema_version": SCHEMA_VERSION,
        "what_this_is": "Real end-to-end decode on a pinned SparkInfer commit and checkpoint. "
                        "This is the metric sections 17, 21 and 28 make authoritative.",
        "provenance": provenance,
        "correctness": correctness,
        "workloads": workloads,
        "workload_coverage": {
            "measured": measured,
            "unmeasured": [k for k in DEFAULT_WEIGHTS if k not in measured],
            "why": "Section 44 weights batch 1 at 0.40 and concurrency at 0.60. An arm that was "
                   "not run is left out of the matrix rather than filled in, and the weights "
                   "that remain are renormalised by decide.py - so a partial run is reported as "
                   "a partial verdict instead of a full one with invented halves.",
        },
        "per_context": per_context,
        "per_concurrency": concurrency_detail,
        "measurement": {
            "noise_floor_pct": noise_floor,
            "gain_pct": gain_pct,
            "resolved": None if noise_floor is None else abs(gain_pct) > noise_floor,
            "note": "resolved=false means the difference is inside the control arm's own "
                    "run-to-run spread on this box; the axis is open, not settled.",
            # Named explicitly so a reader does not have to check every arm by hand. An
            # unresolved arm still carries its weight in the verdict below -- it is a
            # measurement, not a discard -- but it is not evidence of anything.
            "unresolved_workloads": sorted(
                [f"batch1/ctx{c}" for c in ctxs
                 if per_context[str(c)]["resolution"]["resolved"] is False]
                + [k for k, v in concurrency_detail.items()
                   if v["resolution"]["resolved"] is False]),
        },
        "adapter": adapter_stats,
    }
    lock.__exit__()
    a.output.write_text(json.dumps(doc, indent=2) + "\n")
    if verbose:
        print(f"\nbatch-1 decode: {gain_pct:+.2f}%  (noise floor "
              f"{'unknown' if noise_floor is None else f'{noise_floor:.2f}%'})")
        print(f"wrote {a.output}")
        print(f"score it: eval/decide.py --real {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
