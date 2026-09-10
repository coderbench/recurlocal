#!/usr/bin/env python3
"""Tests for the go/no-go decision.

The bands decide whether the project continues, so a silent drift in a boundary changes
what every past and future verdict meant. These pin them exactly as the overview states
them.
"""
import json, math, subprocess, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_eval
import inspect
import real_sweep
import traffic_budget
import decide as label  # noqa: E402

DECIDE_PY = Path(__file__).resolve().parent / "decide.py"


# A document that reports NO resolution evidence is INCONCLUSIVE whatever it measured -- absent
# is not resolved -- so the default here carries an empty `unresolved_workloads`, which is what
# `real_eval.py` writes for a run where every arm cleared its own spread. Tests that are about
# the absence itself build their document without it; see
# `test_a_result_with_no_resolution_evidence_is_inconclusive`.
def real_doc(workloads, identical=True, measurement=("default",)):
    doc = {"correctness": {"output_identical": identical, "method": "greedy replay"},
           "workloads": workloads}
    if measurement == ("default",):
        measurement = {"unresolved_workloads": []}
    if measurement is not None:
        doc["measurement"] = measurement
    return doc


def flat(ratio, weights=None):
    """Every workload improved by the same ratio, so the geometric mean is that ratio."""
    weights = weights or label.DEFAULT_WEIGHTS
    return {name: {"weight": w, "baseline_tps": 100.0, "candidate_tps": 100.0 * ratio}
            for name, w in weights.items()}


class Statuses(unittest.TestCase):
    def test_there_is_no_impact_band_table(self):
        # The bands are gone and their absence is load-bearing, so it is asserted rather than
        # left to a reader noticing. The lowest paying step was 2% weighted throughput gain
        # and the physical ceiling for the whole shipped policy family is 0.52% on the scored
        # model -- a submission could remove every recoverable byte and score `none`. A band
        # structure whose lowest step sits above what the hardware can deliver is a broken
        # instrument, and re-adding one would silently re-break it.
        self.assertFalse(hasattr(label, "IMPACT"),
                         "decide.IMPACT is back; see frontier/README.md for why it went away")
        source = Path(label.__file__).read_text()
        for banned in ('"XL"', '"XS"', 'impact = band('):
            self.assertNotIn(banned, source, f"{banned} is back in decide.py")

    def test_the_status_vocabulary_is_the_ledgers(self):
        # One language for a single-axis A/B and for a full frontier evaluation. Two
        # vocabularies for the same outcomes is how a verdict comes to mean two things.
        self.assertEqual(set(label.STATUSES),
                         {"FRONTIER_GAIN", "NO_FRONTIER_GAIN", "INCONCLUSIVE",
                          "CORRECTNESS_FAIL", "REGRESSION_GUARD_FAIL", "BUILD_FAIL",
                          "EVAL_ERROR"})

    def test_status_follows_from_the_numbers(self):
        # Every one of these carries resolution evidence, because a document without it is
        # INCONCLUSIVE whatever it measured -- see the test below.
        resolved = {"unresolved_workloads": []}
        clear = label.score_real(real_doc(flat(1.10), measurement=resolved))
        self.assertEqual(clear["status"], "FRONTIER_GAIN")
        self.assertTrue(clear["significant"])

        flatline = label.score_real(real_doc(flat(1.0), measurement=resolved))
        self.assertEqual(flatline["status"], "NO_FRONTIER_GAIN")

        # 0.1%, resolved, on a complete matrix. The physical ceiling for the persist family on
        # the scored model is 0.52%, so a scorer that called this "not a gain" would be the
        # impact-band table again under another name -- and that table was removed for exactly
        # this reason. `verdict` still says `reject`, because the project's go/no-go bar is a
        # different question from whether a submission moved the number.
        tiny = label.score_real(real_doc(flat(1.001), measurement=resolved))
        self.assertEqual(tiny["status"], "FRONTIER_GAIN")
        self.assertEqual(tiny["verdict"], "reject")
        self.assertAlmostEqual(tiny["weighted_gain_pct"], 0.1, places=6,
                               msg="and the continuous figure is still reported as itself")

    def test_a_result_with_no_resolution_evidence_is_inconclusive(self):
        # Absent is not resolved. Until the 2% floor stopped gating `significant`, a document
        # that never reported a noise floor was saved by the floor rather than caught by it.
        out = label.score_real(real_doc(flat(1.10), measurement=None))
        self.assertEqual(out["status"], "INCONCLUSIVE")
        self.assertFalse(out["significant"])
        self.assertEqual(out["resolution_evidence"], "absent")
        self.assertIn("absent is not", out["reason"])

    def test_go_no_go_boundaries(self):
        # Overview section 21.
        for gain, decision in [(-5.0, "reject"), (1.99, "reject"), (2.0, "weak"), (3.99, "weak"),
                               (4.0, "promising"), (6.99, "promising"), (7.0, "strong"),
                               (9.99, "strong"), (10.0, "expand"), (25.0, "expand")]:
            with self.subTest(gain=gain):
                self.assertEqual(label.band(gain, label.GO_NO_GO)[1], decision)

    def test_section_37_minimum_is_promising(self):
        # "at minimum >= 4% real end-to-end would justify continued serious development".
        # GO_NO_GO survives the band removal on purpose: it answers whether this research
        # direction should continue, which is the project's decision about ITSELF, not a
        # label attached to somebody's submission.
        self.assertEqual(label.band(4.0, label.GO_NO_GO)[1], "promising")


class RealScoring(unittest.TestCase):
    def test_uniform_gain_maps_to_that_gain(self):
        out = label.score_real(real_doc(flat(1.10)))
        self.assertTrue(out["scored"])
        self.assertAlmostEqual(out["weighted_gain_pct"], 10.0, places=9)
        self.assertEqual(out["status"], "FRONTIER_GAIN")
        self.assertEqual(out["verdict"], "expand")

    def test_weighted_geometric_mean(self):
        ratios = {"batch1": 1.10, "concurrency4": 1.05, "concurrency16": 1.02, "concurrency32": 0.99}
        doc = real_doc({n: {"weight": label.DEFAULT_WEIGHTS[n], "baseline_tps": 100.0,
                            "candidate_tps": 100.0 * r} for n, r in ratios.items()})
        out = label.score_real(doc)
        expected = math.exp(sum(label.DEFAULT_WEIGHTS[n] * math.log(r) for n, r in ratios.items()))
        self.assertAlmostEqual(out["weighted_ratio"], expected, places=12)
        self.assertEqual(out["status"], "FRONTIER_GAIN")
        self.assertEqual(out["verdict"], "promising")

    def test_partial_matrix_renormalises_rather_than_inventing_arms(self):
        # real_eval.py leaves out a workload it did not run, so the weights that remain do
        # not sum to 1. They must be renormalised, not treated as if the missing arms scored
        # zero gain -- which would silently dilute every result by 60% of section 44's
        # weight and make a partial run look like a worse full one.
        doc = real_doc({"batch1": {"weight": 0.40, "baseline_tps": 100.0, "candidate_tps": 110.0},
                        "concurrency4": {"weight": 0.20, "baseline_tps": 100.0, "candidate_tps": 110.0}})
        out = label.score_real(doc)
        self.assertAlmostEqual(out["weighted_gain_pct"], 10.0, places=9)
        # And the single-arm case, which is what a batch-1-only run produces.
        only = real_doc({"batch1": {"weight": 1.0, "baseline_tps": 100.0, "candidate_tps": 104.0}})
        self.assertAlmostEqual(label.score_real(only)["weighted_gain_pct"], 4.0, places=9)

    def test_geometric_mean_does_not_let_one_win_mask_a_loss(self):
        # Arithmetic mean of 2.0 and 0.5 is 1.25; geometric is 1.0.
        doc = real_doc({"a": {"weight": 0.5, "baseline_tps": 100.0, "candidate_tps": 200.0},
                        "b": {"weight": 0.5, "baseline_tps": 100.0, "candidate_tps": 50.0}})
        out = label.score_real(doc, allow_regression=True)
        self.assertAlmostEqual(out["weighted_ratio"], 1.0, places=12)
        self.assertEqual(out["status"], "NO_FRONTIER_GAIN")

    def test_regression_guard_blocks(self):
        doc = real_doc({"batch1": {"weight": 0.5, "baseline_tps": 100.0, "candidate_tps": 130.0},
                        "concurrency32": {"weight": 0.5, "baseline_tps": 100.0, "candidate_tps": 97.0}})
        out = label.score_real(doc)
        self.assertFalse(out["scored"])
        self.assertEqual(out["verdict"], "REGRESSION")
        self.assertEqual(out["regressed_workloads"], ["concurrency32"])
        self.assertEqual(out["status"], "REGRESSION_GUARD_FAIL")

    def test_exactly_two_percent_regression_is_allowed(self):
        doc = real_doc({"batch1": {"weight": 0.5, "baseline_tps": 100.0, "candidate_tps": 130.0},
                        "concurrency32": {"weight": 0.5, "baseline_tps": 100.0, "candidate_tps": 98.0}})
        self.assertTrue(label.score_real(doc)["scored"])

    def test_regression_waiver_scores_but_records(self):
        doc = real_doc({"batch1": {"weight": 0.5, "baseline_tps": 100.0, "candidate_tps": 130.0},
                        "concurrency32": {"weight": 0.5, "baseline_tps": 100.0, "candidate_tps": 90.0}})
        out = label.score_real(doc, allow_regression=True)
        self.assertTrue(out["scored"])
        self.assertTrue(out["regression_waived"])
        self.assertEqual(out["regressed_workloads"], ["concurrency32"])

    def test_changed_output_is_rejected_however_fast(self):
        out = label.score_real(real_doc(flat(2.0), identical=False))
        self.assertFalse(out["scored"])
        self.assertEqual(out["verdict"], "REJECT")
        self.assertEqual(out["status"], "CORRECTNESS_FAIL")

    def test_missing_correctness_is_rejected(self):
        out = label.score_real({"workloads": flat(1.10)})
        self.assertEqual(out["verdict"], "REJECT")

    def test_the_go_no_go_threshold_does_not_gate_a_submission(self):
        # 1.5%: below the project's 2% go/no-go bar and above the 0.52% physical ceiling for
        # the persist family. `verdict` rejects the direction; `status` reports what the
        # submission did, and the two are different questions.
        out = label.score_real(real_doc(flat(1.015), measurement={"unresolved_workloads": []}))
        self.assertTrue(out["scored"])
        self.assertEqual(out["verdict"], "reject")
        self.assertTrue(out["significant"])
        self.assertEqual(out["status"], "FRONTIER_GAIN")
        self.assertIn("go/no-go threshold", out["reason"])
        self.assertIn("physical ceiling", out["reason"])

    def test_default_weights_apply_by_name(self):
        doc = real_doc({n: {"baseline_tps": 100.0, "candidate_tps": 110.0} for n in label.DEFAULT_WEIGHTS})
        out = label.score_real(doc)
        self.assertAlmostEqual(out["weighted_gain_pct"], 10.0, places=9)

    def test_bad_input_is_rejected_loudly(self):
        for bad in ({"correctness": {"output_identical": True}, "workloads": {}},
                    real_doc({"x": {"weight": 1.0, "baseline_tps": 0.0, "candidate_tps": 1.0}}),
                    real_doc({"x": {"weight": 1.0, "baseline_tps": 1.0, "candidate_tps": -1.0}}),
                    real_doc({"unknown_name": {"baseline_tps": 1.0, "candidate_tps": 2.0}}),
                    real_doc({"x": {"weight": 0.0, "baseline_tps": 1.0, "candidate_tps": 2.0}})):
            with self.subTest(bad=bad), self.assertRaises(SystemExit):
                label.score_real(bad)


class SyntheticRefusal(unittest.TestCase):
    BASE = {"correctness": "pass", "best_mode": "combined", "best_synthetic_gain_pct": 35.0,
            "stability": {"verdict": "stable", "max_rel_spread_pct": 0.4, "threshold_pct": 2.0}}

    def test_large_synthetic_gain_is_never_tiered(self):
        out = label.score_synthetic(dict(self.BASE))
        self.assertFalse(out["scored"])
        self.assertEqual(out["status"], "EVAL_ERROR")
        self.assertEqual(out["verdict"], "SYNTHETIC-ONLY")
        self.assertEqual(out["best_synthetic_gain_pct"], 35.0)

    def test_unstable_run_is_not_a_result(self):
        doc = dict(self.BASE, stability={"verdict": "unstable", "max_rel_spread_pct": 19.0,
                                         "threshold_pct": 2.0})
        self.assertEqual(label.score_synthetic(doc)["verdict"], "UNSTABLE")

    def test_failed_correctness_is_rejected(self):
        self.assertEqual(label.score_synthetic(dict(self.BASE, correctness="fail"))["verdict"], "REJECT")


class CommandLine(unittest.TestCase):
    def run_label(self, args, doc):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "in.json"
            p.write_text(json.dumps(doc))
            r = subprocess.run([sys.executable, str(DECIDE_PY), *args, str(p)],
                               capture_output=True, text=True)
            return r.returncode, json.loads(r.stdout)

    def test_scored_improvement_exits_zero(self):
        code, out = self.run_label(["--real"], real_doc(flat(1.10)))
        self.assertEqual(code, 0)
        self.assertEqual(out["status"], "FRONTIER_GAIN")

    def test_synthetic_exits_nonzero_so_it_cannot_be_mistaken_for_a_pass(self):
        code, out = self.run_label(["--synthetic"], SyntheticRefusal.BASE)
        self.assertEqual(code, 1)
        self.assertEqual(out["status"], "EVAL_ERROR")

    def test_regression_exits_nonzero(self):
        code, _ = self.run_label(["--real"], real_doc(
            {"batch1": {"weight": 1.0, "baseline_tps": 100.0, "candidate_tps": 90.0}}))
        self.assertEqual(code, 1)



class RealEvalEstimator(unittest.TestCase):
    """The batch-1 arm carries weight 0.40, so the estimator behind it decides verdicts.

    These tests exist because that arm used a different estimator from every other arm and
    from the module's own documented method, and nothing could catch it: the computation was
    inline in main(), which needs a GPU and a 21 GB checkpoint to reach.
    """

    @staticmethod
    def _arm(base, cand):
        return {"baseline_tps": real_eval.median(base), "candidate_tps": real_eval.median(cand),
                "paired_ratios": [c / b for b, c in zip(base, cand)]}

    def test_pairing_is_not_discarded(self):
        # Drifting clocks: the pairing is the only thing that makes this comparison valid.
        per = {"128": self._arm([100.0, 110.0, 120.0], [105.0, 110.0, 132.0])}
        ratio, _ = real_eval.batch1_arm(per, [128])
        self.assertAlmostEqual(ratio, 1.05, places=9)          # median of [1.05, 1.00, 1.10]
        # The ratio-of-medians estimator this replaced would have returned 110/110 = 1.0 and
        # turned a 'promising' verdict into 'reject' on the same measurements.
        self.assertNotAlmostEqual(ratio, 1.0, places=3)

    def test_matches_the_per_context_number_it_is_built_from(self):
        per = {"128": self._arm([100.0, 100.0, 100.0], [101.0, 102.0, 103.0])}
        ratio, base = real_eval.batch1_arm(per, [128])
        self.assertAlmostEqual((ratio - 1.0) * 100.0, 2.0, places=9)   # the median pair, +2%
        self.assertAlmostEqual(base, 100.0, places=9)

    def test_contexts_combine_geometrically_so_none_can_carry_the_arm(self):
        per = {"128": self._arm([100.0], [200.0]), "4096": self._arm([100.0], [50.0])}
        ratio, _ = real_eval.batch1_arm(per, [128, 4096])
        self.assertAlmostEqual(ratio, 1.0, places=9)   # 2x and 0.5x cancel, as they must

    def test_degenerate_inputs_do_not_fabricate_a_gain(self):
        self.assertEqual(real_eval.batch1_arm({}, []), (1.0, 0.0))
        per = {"128": {"baseline_tps": 0.0, "candidate_tps": 0.0, "paired_ratios": []}}
        self.assertEqual(real_eval.batch1_arm(per, [128]), (1.0, 0.0))


class HarnessIntegrity(unittest.TestCase):
    """The harness is meant to arbitrate. These cover the ways it could quietly not."""

    def test_a_run_whose_arms_did_not_resolve_is_not_significant(self):
        # A complete matrix, so that resolution is the only thing under test here.
        doc = real_doc(flat(1.08))
        self.assertTrue(label.score_real(doc)["significant"])          # resolved: a real result
        doc["measurement"] = {"unresolved_workloads": ["batch1/ctx128"]}
        out = label.score_real(doc)
        self.assertFalse(out["significant"])                           # same number, no evidence
        self.assertEqual(out["unresolved_workloads"], ["batch1/ctx128"])
        self.assertIn("did not resolve", out["reason"])

    def test_a_candidate_that_applied_no_policy_is_a_null_candidate(self):
        # Deferred windows are computed and handed back; unless the runtime attaches them they
        # never reach a kernel. Scoring that reports hook overhead as a locality result.
        null = {"stats": {"windows_applied": 0, "windows_attached_to_node": 0,
                          "pre_touch_launches": 0, "windows_deferred_to_caller": 192}}
        self.assertFalse(real_eval.policy_applied(null))
        for live in ({"windows_applied": 1}, {"windows_attached_to_node": 1},
                     {"pre_touch_launches": 1}):
            self.assertTrue(real_eval.policy_applied({"stats": live}))

    def test_control_is_identified_by_asking_for_no_mode(self):
        self.assertTrue(real_eval.is_control({}))
        self.assertTrue(real_eval.is_control({"RECURLOCAL": "off"}))
        self.assertTrue(real_eval.is_control({"RECURLOCAL": "0"}))
        self.assertFalse(real_eval.is_control({"RECURLOCAL": "persist"}))
        self.assertFalse(real_eval.is_control({"RECURLOCAL": "baseline"}))

    def test_every_eval_entry_point_can_print_its_own_help(self):
        """`--help` is the first thing anyone types, and `decide.py --help` crashed.

        argparse `%`-expands help strings, so a literal "2%" in `--allow-regression`'s help
        raised `ValueError: unsupported format character` from inside `print_help`. Nothing
        else exercises the help text, so it had been broken since the flag was added -- and it
        is the one command a contributor runs before any of the ones that are tested.
        """
        import subprocess
        root = Path(__file__).resolve().parent.parent
        entries = sorted(p for p in (root / "eval").glob("*.py")
                         if not p.name.startswith("test_")) + [root / "tools" / "tt-frontier"]
        for entry in entries:
            with self.subTest(entry=entry.name):
                done = subprocess.run([sys.executable, str(entry), "--help"],
                                      capture_output=True, text=True, timeout=60)
                self.assertEqual(done.returncode, 0,
                                 f"{entry.name} --help exited {done.returncode}\n"
                                 f"{done.stderr[-800:]}")
                self.assertIn("usage:", done.stdout)

    def test_the_baseline_arm_is_recognised_under_both_spellings(self):
        # The incident: `TENSORTRANSIT=baseline` -- the hook with no window, one of the five
        # arms the specification names -- was refused as a NULL CANDIDATE. `is_control` read
        # both spellings of the mode; the null-candidate guard read only RECURLOCAL, so an arm
        # that spelled it the 0.2 way looked like a policy arm that had applied no policy.
        for env in ({"TENSORTRANSIT": "baseline"}, {"RECURLOCAL": "baseline"},
                    {"TENSORTRANSIT": "persist", "TENSORTRANSIT_PRESET": "baseline"},
                    {"TENSORTRANSIT": "persist", "RECURLOCAL_PLANNER": "baseline"}):
            self.assertTrue(real_eval.is_baseline_arm(env), env)
            self.assertFalse(real_eval.is_control(env), env)
        for env in ({}, {"TENSORTRANSIT": "off"}, {"TENSORTRANSIT": "persist"},
                    {"RECURLOCAL": "combined"}):
            self.assertFalse(real_eval.is_baseline_arm(env), env)

    def test_the_null_candidate_guard_lets_the_baseline_arm_through(self):
        # End to end through the guard itself, not only through the predicate: a baseline arm
        # that loaded, bracketed layers and applied nothing is doing its job.
        stats = ('RECURLOCAL_STATS {"ever_initialised":true,"initialised":true,"broken":false,'
                 '"stats":{"layers":48,"windows_applied":0,"windows_attached_to_node":0,'
                 '"pre_touch_launches":0,"windows_deferred_to_caller":0}}')
        for env in ({"TENSORTRANSIT": "baseline"}, {"RECURLOCAL": "baseline"}):
            self.assertIsNotNone(real_eval.require_hook_engaged(stats, env, "baseline arm"))
        with self.assertRaises(SystemExit) as caught:
            real_eval.require_hook_engaged(stats, {"TENSORTRANSIT": "persist"}, "policy arm")
        self.assertIn("NULL CANDIDATE", str(caught.exception))

    def test_an_unhooked_run_names_the_mode_it_asked_for(self):
        # The message used to index env_extra['RECURLOCAL'] directly, so an arm that spelled
        # the mode TENSORTRANSIT= raised KeyError from inside the error path -- the guard
        # crashing instead of reporting.
        with self.assertRaises(SystemExit) as caught:
            real_eval.require_hook_engaged("no telemetry here",
                                           {"TENSORTRANSIT": "persist"}, "arm")
        self.assertIn("persist", str(caught.exception))

    def test_an_empty_plan_is_told_apart_from_a_plumbing_failure(self):
        """Both read `windows_applied 0`, and they are opposite facts about a run.

        `naive_both` -- persist both tensor classes, no arbitration -- declines every one of
        512 candidates as `below_min_hit_ratio` at concurrency 4, because a budget shared among
        that many gives each less than the configured floor. That is the arm doing exactly what
        the specification says it should do badly. Refusing it as a NULL CANDIDATE would have
        taken the other four arms of the five-arm run down with it.

        The incident the NULL CANDIDATE guard exists for is different and keeps its guard: a
        window that was computed and handed back to a runtime that never attached it.
        """
        def stats(deferred, declines, actions=None, committed=None):
            body = {"ever_initialised": True, "initialised": True, "broken": False,
                    "stats": {"layers": 48, "windows_applied": 0,
                              "windows_attached_to_node": 0, "pre_touch_launches": 0,
                              "windows_deferred_to_caller": deferred}}
            transit = {}
            if declines is not None:
                transit["declines"] = declines
            if actions is not None:
                transit["plan_actions"] = actions
            if committed is not None:
                transit["committed_bytes"] = committed
            if transit:
                body["transit"] = transit
            return "RECURLOCAL_STATS " + json.dumps(body)

        env = {"TENSORTRANSIT": "persist"}
        # A planner that declined everything for a stated reason is an arm, not a failure.
        self.assertIsNotNone(real_eval.require_hook_engaged(
            stats(0, {"below_min_hit_ratio": 448, "role_excluded": 64}, actions=0),
            env, "naive_both"))

        # A non-empty census is NOT enough, and the five-arm run proved it: `global` declined
        # 192 candidates and still ADMITTED 14 persists and 50 stream hints, none of which
        # reached a kernel. That is the plumbing failure wearing a census.
        with self.assertRaises(SystemExit) as caught:
            real_eval.require_hook_engaged(
                stats(0, {"budget_exhausted": 99, "not_supported": 29, "role_excluded": 64},
                      actions=64), env, "global")
        self.assertIn("NULL CANDIDATE", str(caught.exception))

        # A build older than that counter falls back to committed bytes, which sees a plan that
        # reserved a partition and not a stream-only one -- which is why the counter exists.
        with self.assertRaises(SystemExit):
            real_eval.require_hook_engaged(
                stats(0, {"budget_exhausted": 99}, committed=47185920), env, "old build")
        self.assertIsNotNone(real_eval.require_hook_engaged(
            stats(0, {"budget_exhausted": 192}, committed=0), env, "old build, empty"))

        # Deferred windows are the plumbing failure and still abort, census or no census.
        for declines in (None, {"below_min_hit_ratio": 448}):
            with self.assertRaises(SystemExit) as caught:
                real_eval.require_hook_engaged(stats(192, declines, actions=0), env, "arm")
            self.assertIn("NULL CANDIDATE", str(caught.exception))

        # And an engine that keeps no census gets no benefit of the doubt.
        with self.assertRaises(SystemExit) as caught:
            real_eval.require_hook_engaged(stats(0, None, actions=0), env, "v0 arm")
        self.assertIn("NULL CANDIDATE", str(caught.exception))
        with self.assertRaises(SystemExit):
            real_eval.require_hook_engaged(stats(0, {"none": 512}, actions=0), env,
                                           "no reason given")

    def test_the_packed_path_guard_counts_decode_steps_not_prefill_chunks(self):
        """The measured numbers that corrected it, from the reference box.

        `tokens` counts every step the hook brackets and at ctx4096 most of them are prefill
        chunks: a c=16 run brackets 133 steps of which 63 are decode. A run whose EVERY decode
        step batched sixteen rows therefore reported a 47.4% "packed share" and was refused,
        and the first full TTF-1 matrix lost two cells to that. `max_new` is how many decode
        steps a concurrency arm should produce -- one step advances every live row -- and it is
        immune to how long the prefill was.
        """
        def rec(tokens, packed, rows, concurrency, max_new):
            return real_eval.packed_path_used(
                {"stats": {"tokens": tokens, "tokens_packed": packed, "max_rows_seen": rows}},
                concurrency, max_new)

        # Every decode step batched, at short context and at long, and the OLD denominator
        # separates them for a reason that has nothing to do with decode.
        short = rec(78, 63, 16, 16, 64)
        long_ctx = rec(133, 63, 16, 16, 64)
        self.assertTrue(short["used_packed_path"])
        self.assertTrue(long_ctx["used_packed_path"])
        self.assertAlmostEqual(short["packed_share"], long_ctx["packed_share"], places=6)
        self.assertLess(long_ctx["packed_share_of_all_steps"], real_eval.PACKED_SHARE_MIN)
        self.assertGreater(short["packed_share_of_all_steps"], real_eval.PACKED_SHARE_MIN)

        # A longer decode changes neither answer.
        self.assertTrue(rec(517, 255, 32, 32, 256)["used_packed_path"])

        # Not one batched decode step, at any decode length. This is the unambiguous case and
        # it is checked before any share.
        for record in (rec(266, 0, 0, 4, 64), rec(1034, 0, 0, 4, 256)):
            self.assertFalse(record["batched_at_all"])
            self.assertFalse(record["used_packed_path"])

        # And the incident the guard was built for: above eight rows the pinned runtime stops
        # batching and decodes one row at a time, at 5.4x the cost. Still caught.
        cliff = rec(142, 4, 8, 16, 64)
        self.assertTrue(cliff["batched_at_all"])
        self.assertFalse(cliff["used_packed_path"])
        with self.assertRaises(SystemExit) as caught:
            real_eval.require_packed_path({"stats": {"tokens": 142, "tokens_packed": 4,
                                                     "max_rows_seen": 8}}, 16, "arm", 64)
        self.assertIn("FELL OFF THE BATCHED DECODE PATH", str(caught.exception))
        self.assertIn("decode steps this arm asked for", str(caught.exception))

        # A run that never batched says so in different words, because the two failures are
        # different and a message that named a share would be describing the wrong one.
        with self.assertRaises(SystemExit) as caught:
            real_eval.require_packed_path({"stats": {"tokens": 266, "tokens_packed": 0,
                                                     "max_rows_seen": 0}}, 4, "arm", 64)
        self.assertIn("not one decode step batched", str(caught.exception))

    def test_an_arm_scoped_to_a_family_the_registry_never_saw_is_unmeasurable(self):
        # Through 0.2.0 no adapter exposed KV to the registry, so a KV-scoped arm would have
        # reported the recurrent policy's number under a KV label. "Weak" and "absent" are
        # indistinguishable from outside a run unless the run says which tensors it knew about.
        empty = {"registry": {"kv_tensors": 0, "recurrent_tensors": 96, "weight_tensors": 0}}
        for preset in ("kv_only", "naive_both", "global"):
            with self.assertRaises(SystemExit) as caught:
                real_eval.require_registered_families(empty, {"TENSORTRANSIT_PRESET": preset},
                                                      "arm")
            self.assertIn("UNMEASURABLE ARM", str(caught.exception))
        # recurrent_only is about the family that IS there, so it runs.
        self.assertIsNotNone(real_eval.require_registered_families(
            empty, {"TENSORTRANSIT_PRESET": "recurrent_only"}, "arm"))

    def test_a_global_arm_without_weights_says_stream_was_untested(self):
        # The 0.2.1 arms sweep measured stream_applied=0 on every arm because no ModelWeight
        # tensor was registered, and the result was read as evidence about coordination until
        # the counters said otherwise. Not a failure -- but it must travel with the result.
        both = {"registry": {"kv_tensors": 24, "recurrent_tensors": 96, "weight_tensors": 0}}
        record = real_eval.require_registered_families(both, {"TENSORTRANSIT_PRESET": "global"},
                                                       "arm")
        self.assertEqual(record["untested_mechanisms"], ["stream"])
        self.assertIn("STREAMED_BYTES_PER_TOKEN", record["note"])
        with_weights = {"registry": dict(both["registry"], weight_tensors=48)}
        self.assertEqual(real_eval.require_registered_families(
            with_weights, {"TENSORTRANSIT_PRESET": "global"}, "arm")["untested_mechanisms"], [])

    def test_a_build_without_the_registry_counter_is_not_accused(self):
        # eval/run_from_base.sh runs the evaluator from the BASE commit against a candidate
        # build, and a base evaluator that refused every build older than its own newest
        # counter would be a false accusation aimed at the contributor.
        self.assertIsNone(real_eval.require_registered_families(
            {"stats": {}}, {"TENSORTRANSIT_PRESET": "kv_only"}, "arm"))

    def test_control_arm_scrubs_an_ambient_adapter_env(self):
        # An operator with RECURLOCAL exported in their shell would otherwise run a hooked
        # "control" and the harness would report ~0% for the candidate against itself.
        import os
        os.environ["RECURLOCAL"] = "combined"
        os.environ["RECURLOCAL_PREFETCH_DISTANCE"] = "6"
        try:
            code, out, _ = real_eval.run([sys.executable, "-c",
                                          "import os;print(os.environ.get('RECURLOCAL','<unset>'),"
                                          "os.environ.get('RECURLOCAL_PREFETCH_DISTANCE','<unset>'))"],
                                         {}, scrub_adapter_env=True)
            self.assertIn("<unset> <unset>", out)
        finally:
            os.environ.pop("RECURLOCAL", None)
            os.environ.pop("RECURLOCAL_PREFETCH_DISTANCE", None)



class ShippedArtifactIsScorable(unittest.TestCase):
    """The repository publishes results/rtx5090-real.json and the README tells you to score
    it. That command used to fail: the matrix is nested under "scored_result" and score_real
    only looked at the top level."""

    def test_the_committed_result_scores(self):
        path = Path(__file__).resolve().parent.parent / "results" / "rtx5090-real.json"
        if not path.exists():
            self.skipTest("no committed result in this checkout")
        out = label.score_real(json.loads(path.read_text()))
        self.assertIn(out["verdict"], {"reject", "weak", "promising", "strong", "expand",
                                       "REGRESSION", "REJECT"})
        self.assertIn("workloads", out)

    def test_both_shapes_score_identically(self):
        bare = real_doc({"batch1": {"weight": 1.0, "baseline_tps": 100.0, "candidate_tps": 105.0}})
        wrapped = {"what_this_is": "bundle", "batch1_axes": {}, "scored_result": bare}
        self.assertEqual(label.score_real(bare)["weighted_gain_pct"],
                         label.score_real(wrapped)["weighted_gain_pct"])

    def test_a_bare_doc_missing_workloads_still_errors(self):
        with self.assertRaises(SystemExit):
            label.score_real({"correctness": {"output_identical": True}})


class WorkloadCoverage(unittest.TestCase):
    """An omitted workload is not a neutral omission.

    Missing arms are renormalised away, so leaving one out removes it from the mean
    instead of averaging it in. docs/MINING.md points the competition at concurrency 32,
    which makes that the arm a submission profits most from not running. The verdict has
    to say what was not measured and refuse to call the result significant.
    """

    def _partial(self, ratio=1.10):
        return real_doc({name: {"weight": w, "baseline_tps": 100.0,
                                "candidate_tps": 100.0 * ratio}
                         for name, w in label.DEFAULT_WEIGHTS.items()
                         if name != "concurrency32"})

    def test_a_missing_arm_is_named_with_the_weight_it_took_with_it(self):
        out = label.score_real(self._partial())
        self.assertTrue(out["partial"])
        self.assertEqual(out["workload_coverage"]["missing"], ["concurrency32"])
        self.assertAlmostEqual(out["workload_coverage"]["missing_weight_share"], 0.20)

    def test_an_incomplete_matrix_cannot_be_significant_however_large_the_gain(self):
        out = label.score_real(self._partial(ratio=1.10))
        self.assertAlmostEqual(out["weighted_gain_pct"], 10.0, places=9)
        self.assertEqual(out["status"], "INCONCLUSIVE")   # a hole in the matrix is not a gain
        self.assertFalse(out["significant"])
        self.assertIn("concurrency32", out["reason"])

    def test_allow_partial_is_the_only_way_through_and_it_is_recorded(self):
        out = label.score_real(self._partial(), allow_partial=True)
        self.assertTrue(out["significant"])
        self.assertTrue(out["partial"])
        self.assertTrue(out["partial_waived"])

    def test_a_complete_matrix_is_not_flagged(self):
        out = label.score_real(real_doc(flat(1.10)))
        self.assertFalse(out["partial"])
        self.assertNotIn("workload_coverage", out)
        self.assertTrue(out["significant"])

    def test_coverage_is_derived_not_taken_from_the_document(self):
        # A submission that declares full coverage while omitting the arm is still partial:
        # the scorer counts the workloads it actually scored.
        doc = self._partial()
        doc["workload_coverage"] = {"measured": sorted(label.DEFAULT_WEIGHTS), "missing": []}
        out = label.score_real(doc)
        self.assertEqual(out["workload_coverage"]["missing"], ["concurrency32"])
        self.assertFalse(out["significant"])

    def test_a_custom_matrix_is_not_forced_onto_section_44_names(self):
        # A result that uses none of the section 44 names is a different matrix, not an
        # incomplete one, and must not be flagged as missing every arm.
        doc = real_doc({"my_workload": {"weight": 1.0, "baseline_tps": 100.0,
                                        "candidate_tps": 110.0}})
        out = label.score_real(doc)
        self.assertFalse(out["partial"])
        self.assertTrue(out["significant"])

    def test_the_cli_refuses_to_exit_zero_on_a_partial_matrix(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "real.json"
            f.write_text(json.dumps(self._partial()))
            blocked = subprocess.run([sys.executable, str(DECIDE_PY), "--real", str(f)],
                                     capture_output=True, text=True)
            self.assertNotEqual(blocked.returncode, 0)
            waived = subprocess.run([sys.executable, str(DECIDE_PY), "--real", str(f),
                                     "--allow-partial"], capture_output=True, text=True)
            self.assertEqual(waived.returncode, 0)


class Summary(unittest.TestCase):
    """The human summary is what anyone actually reads. It must not be able to read cleaner
    than the JSON it summarizes, and it must survive every verdict shape."""

    def _text(self, verdict):
        import io
        buf = io.StringIO()
        label.summarize(verdict, stream=buf)
        return buf.getvalue()

    def test_every_qualifier_on_the_verdict_reaches_the_summary(self):
        doc = real_doc({name: {"weight": w, "baseline_tps": 100.0, "candidate_tps": 110.0}
                        for name, w in label.DEFAULT_WEIGHTS.items() if name != "concurrency32"})
        doc["measurement"] = {"unresolved_workloads": ["batch1/ctx128"]}
        text = self._text(label.score_real(doc))
        self.assertIn("NOT MEASURED", text)
        self.assertIn("concurrency32", text)
        self.assertIn("unresolved", text)
        self.assertIn("significant false", text)

    def test_it_survives_the_verdicts_that_carry_no_workloads(self):
        # A rejected-correctness verdict has no workloads and no gain; a synthetic one has
        # neither and a different track. Both must print rather than raise.
        for v in (label.score_real(real_doc(flat(1.10), identical=False)),
                  label.score_synthetic({"correctness": "pass"})):
            self.assertIn("verdict:", self._text(v))

    def test_a_regression_is_named(self):
        w = dict(flat(1.10))
        w["concurrency32"] = {"weight": 0.20, "baseline_tps": 100.0, "candidate_tps": 90.0}
        text = self._text(label.score_real(real_doc(w)))
        self.assertIn("regressed: concurrency32", text)


class HookRan(unittest.TestCase):
    """The adapter clears its live flag in shutdown(), and the stats line is printed after
    that. Anything asking "did the hook run" has to read ever_initialised or it reads false
    on every run that worked."""

    def test_a_shut_down_adapter_still_counts_as_having_run(self):
        # Exactly the snapshot a successful run emits: shutdown() has already happened.
        self.assertTrue(real_eval.hook_ran({"initialised": False, "ever_initialised": True}))

    def test_an_adapter_that_never_started_does_not(self):
        self.assertFalse(real_eval.hook_ran({"initialised": False, "ever_initialised": False}))
        self.assertFalse(real_eval.hook_ran(None))
        self.assertFalse(real_eval.hook_ran({}))

    def test_an_older_adapter_without_the_field_falls_back(self):
        self.assertTrue(real_eval.hook_ran({"initialised": True}))
        self.assertFalse(real_eval.hook_ran({"initialised": False}))

    def test_the_guard_and_the_correctness_record_cannot_drift_apart(self):
        # They did: the guard read ever_initialised and the record read initialised, so a run
        # the guard accepted was recorded as having no hook active. One function now, and this
        # test fails if a second definition reappears.
        import inspect
        src = inspect.getsource(real_eval)
        self.assertEqual(src.count('get("ever_initialised"'), 1,
                         "ever_initialised must be read in exactly one place: hook_ran()")


class MatrixCeiling(unittest.TestCase):
    """What is the most this repository can ever pay?

    Each arm has a physical ceiling, and the verdict is a weighted mean over all of them, so
    the number that decides whether the project is worth competing on is what a submission
    scores if it hits every ceiling at once. That is only meaningful if it is computed with
    the same weights and bands the scorer uses -- hence the agreement test below.
    """

    PIN = json.loads((Path(__file__).resolve().parent.parent /
                      "adapters" / "sparkinfer" / "pin.json").read_text())["model"]

    def _spec(self, names=("batch1", "concurrency4", "concurrency16", "concurrency32")):
        base = {"batch1": {"sequences": 1, "ms_per_token": 10.41, "state_bytes_scale": 1.0},
                "concurrency4": {"sequences": 4, "aggregate_tps": 329.43, "state_bytes_scale": 0.5},
                "concurrency16": {"sequences": 16, "aggregate_tps": 770.1, "state_bytes_scale": 0.5},
                "concurrency32": {"sequences": 32, "aggregate_tps": 928.0, "state_bytes_scale": 0.5}}
        return {"arms": {k: v for k, v in base.items() if k in names}}

    def _run(self, spec, persisting=None):
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            traffic_budget.matrix_ceiling(self.PIN, spec, 1792.0, None, persisting)
        return json.loads(buf.getvalue())

    def test_the_ceiling_is_weighted_the_way_the_scorer_weights(self):
        # Feed each arm's ceiling to decide.py as a candidate that achieved exactly it. The
        # scorer's weighted gain must equal the ceiling tool's. If the two ever disagree,
        # the repository is advertising a ceiling in one currency and paying in another.
        out = self._run(self._spec())
        doc = real_doc({name: {"baseline_tps": 100.0,
                               "candidate_tps": 100.0 * (1 + arm["ceiling_pct"] / 100.0)}
                        for name, arm in out["arms"].items() if arm.get("measured")})
        self.assertAlmostEqual(label.score_real(doc, allow_partial=True)["weighted_gain_pct"],
                               out["best_possible_weighted_gain_pct"], places=9)

    def test_an_arm_with_no_measured_rate_is_excluded_and_named(self):
        out = self._run(self._spec(("batch1", "concurrency4", "concurrency16")))
        self.assertEqual(out["arms_without_a_measured_rate"], ["concurrency32"])
        self.assertAlmostEqual(out["weights_covered"], 0.80)
        self.assertFalse(out["arms"]["concurrency32"]["measured"])

    def test_aggregate_tps_is_converted_per_step_not_per_token(self):
        # One step advances every sequence, so 4 sequences at 329.43 aggregate tok/s is a
        # 12.14 ms step, not 3.04 ms. Getting this backwards inflates the ceiling 4x.
        out = self._run(self._spec(("concurrency4",)))
        self.assertAlmostEqual(out["arms"]["concurrency4"]["measured_ms_per_token"],
                               4 / 329.43 * 1000.0, places=6)

    def test_the_ceiling_is_continuous_and_uses_the_scorers_go_no_go(self):
        out = self._run(self._spec())
        best = out["best_possible_weighted_gain_pct"]
        self.assertAlmostEqual(out["best_possible_gain_pct"], best, places=12,
                               msg="the ceiling is reported as a continuous figure, not a band")
        self.assertEqual(out["best_possible_verdict"], label.band(best, label.GO_NO_GO)[1])

    def test_the_ceiling_is_in_throughput_terms_not_traffic_share(self):
        # A step carrying f less traffic runs in (1-f) of the time, so tok/s rise by
        # f/(1-f). The scorer measures candidate_tps/baseline_tps, so the ceiling has to be
        # quoted in that currency; quoting the raw share understates it, and a submission
        # that beat a published "ceiling" would discredit every other number here.
        out = self._run(self._spec())
        for name, arm in out["arms"].items():
            if not arm.get("measured"):
                continue
            with self.subTest(arm=name):
                f = arm["recurrent_share_of_traffic_pct"] / 100.0
                self.assertAlmostEqual(arm["ceiling_pct"], 100.0 * f / (1 - f), places=9)
                self.assertGreater(arm["ceiling_pct"], arm["recurrent_share_of_traffic_pct"])

    def test_removing_that_traffic_really_does_return_the_ceiling(self):
        # End to end, in the units that decide the verdict: take the measured step time,
        # subtract exactly the recurrent-state bytes, and the resulting tok/s must be the
        # ceiling the tool published.
        out = self._run(self._spec(("concurrency16",)))
        arm = out["arms"]["concurrency16"]
        total = arm["implied_total_bytes_per_token"]
        freed = arm["state_traffic_bytes_per_token"]
        gain = (total / (total - freed) - 1.0) * 100.0
        self.assertAlmostEqual(arm["ceiling_pct"], gain, places=9)

    def test_the_persist_family_ceiling_never_exceeds_the_traffic_ceiling(self):
        out = self._run(self._spec(), persisting=62914560)
        for name, arm in out["arms"].items():
            if not arm.get("measured"):
                continue
            with self.subTest(arm=name):
                self.assertLessEqual(arm["persist_family"]["ceiling_pct"], arm["ceiling_pct"])

    def test_the_persist_ceiling_falls_with_concurrency_while_the_room_grows(self):
        # The finding this whole bound exists to state. More sequences means more recurrent
        # traffic to recover -- but the footprint that has to stay resident to recover ANY of
        # it grows just as fast, and the cache does not. So the persist family's ceiling moves
        # the opposite way from the opportunity, which is why it measures nothing at 16
        # sequences even though the traffic ceiling has quadrupled.
        out = self._run(self._spec(), persisting=62914560)
        order = ["batch1", "concurrency4", "concurrency16", "concurrency32"]
        traffic = [out["arms"][n]["ceiling_pct"] for n in order]
        persist = [out["arms"][n]["persist_family"]["ceiling_pct"] for n in order]
        self.assertEqual(traffic, sorted(traffic), "the traffic ceiling must rise")
        self.assertEqual(persist, sorted(persist, reverse=True), "the persist ceiling must fall")

    def test_a_cache_big_enough_to_hold_it_all_reaches_the_traffic_ceiling(self):
        # The bound has to be tight at the easy end, or it is not measuring residency.
        out = self._run(self._spec(("batch1",)), persisting=10 ** 12)
        arm = out["arms"]["batch1"]
        self.assertAlmostEqual(arm["persist_family"]["resident_fraction_of_state"], 1.0)
        self.assertAlmostEqual(arm["persist_family"]["ceiling_pct"], arm["ceiling_pct"], places=9)

    def test_the_resident_fraction_is_capacity_over_footprint(self):
        cap = 62914560
        out = self._run(self._spec(("concurrency16",)), persisting=cap)
        pf = out["arms"]["concurrency16"]["persist_family"]
        self.assertAlmostEqual(pf["resident_fraction_of_state"],
                               cap / pf["per_token_state_footprint_bytes"])
        self.assertAlmostEqual(pf["footprint_over_capacity"],
                               pf["per_token_state_footprint_bytes"] / cap)

    def test_the_footprint_is_every_layer_not_one(self):
        # State is reused a whole model pass later, so a per-layer footprint would understate
        # the residency requirement 48-fold and make the persist family look viable.
        out = self._run(self._spec(("concurrency16",)), persisting=62914560)
        arm = out["arms"]["concurrency16"]
        self.assertAlmostEqual(arm["persist_family"]["per_token_state_footprint_bytes"],
                               arm["state_bytes_per_layer_per_sequence"] * 48 * 16)

    def test_a_matrix_with_nothing_measured_is_an_error_not_a_zero_ceiling(self):
        with self.assertRaises(SystemExit):
            self._run({"arms": {}})

    def test_an_arm_without_a_rate_at_all_is_an_error(self):
        with self.assertRaises(SystemExit):
            self._run({"arms": {"batch1": {"sequences": 1}}})


class WithinLayerCeiling(unittest.TestCase):
    """Surface 2 in docs/MINING.md: reuse at a distance the cache can actually serve.

    Every shipped policy targets reuse across a token, which is a whole model pass and far
    larger than any cache. Reuse inside one layer is microseconds away and L2 serves it for
    free -- so the question is not whether the cache could serve it but whether there are any
    bytes at that distance. These pin the answer so nobody re-derives it by building it.
    """

    PIN = MatrixCeiling.PIN

    def _arm(self, m=None, sequences=1, ms=10.344706303443338):
        return traffic_budget.arm_ceiling(m or self.PIN, sequences, ms, 1.0, 1792.0)

    def test_the_matrix_state_contributes_nothing(self):
        # The pinned runtime's GDN kernel keeps each state column in registers across both
        # passes: one global read, one global write. 98% of the recurrent bytes therefore
        # have no within-layer reuse at all, and a policy aimed at them has nothing to catch.
        self.assertEqual(self._arm()["within_layer_family"]["matrix_state_reuse_bytes"], 0)

    def test_only_the_conv_windows_shift_reread_is_reusable(self):
        m = self.PIN
        k = m["linear_conv_kernel_dim"]
        expected = m["lin_conv_state_bytes_per_layer"] * (k - 2) / (k - 1) * m["recurrent_layers"]
        self.assertAlmostEqual(self._arm()["within_layer_family"]["reusable_bytes_per_token"],
                               expected)

    def test_a_two_tap_conv_has_no_window_to_shift(self):
        m = dict(self.PIN, linear_conv_kernel_dim=2)
        self.assertEqual(self._arm(m)["within_layer_family"]["reusable_bytes_per_token"], 0)
        self.assertEqual(self._arm(m)["within_layer_family"]["ceiling_pct"], 0.0)

    def test_it_is_orders_below_the_significance_floor_on_the_pinned_model(self):
        # The finding: this surface is closed by arithmetic, not by effort. If a future
        # geometry ever pushes it near the floor this test is where that shows up.
        c = self._arm()["within_layer_family"]["ceiling_pct"]
        self.assertLess(c, label.SIGNIFICANCE_PCT / 100.0)

    def test_it_never_exceeds_the_traffic_ceiling_it_is_a_subset_of(self):
        for seqs in (1, 4, 16, 32):
            with self.subTest(sequences=seqs):
                a = self._arm(sequences=seqs)
                self.assertLessEqual(a["within_layer_family"]["ceiling_pct"], a["ceiling_pct"])


class ControlReproducibility(unittest.TestCase):
    """The gate must not blame the candidate for a runtime that disagrees with itself.

    Found by using the harness: on a sparse-MoE checkpoint two UNHOOKED control runs diverge at
    token 2, because a few ULP in the prefill flip a discrete top-k expert choice and that moves
    the argmax. The gate compared one control to one candidate, saw a difference, and reported
    "the exact-locality track requires bit-identical model output" — an accusation against a
    candidate that had changed nothing. On the dense checkpoint the same binary is bit-identical
    across control, control and candidate.
    """

    def _doc(self, **correctness):
        base = {"method": "greedy replay, token-exact"}
        base.update(correctness)
        return {"correctness": base, "workloads": flat(1.05)}

    def test_a_candidate_that_really_diverged_is_still_blamed(self):
        out = label.score_real(self._doc(output_identical=False, runtime_reproducible=True,
                                        first_divergence=7))
        self.assertEqual(out["verdict"], "REJECT")
        self.assertIn("the candidate changed model output", out["reason"])
        self.assertIn("token 7", out["reason"])

    def test_an_irreproducible_control_is_inconclusive_not_an_accusation(self):
        out = label.score_real(self._doc(output_identical=None, runtime_reproducible=False,
                                         control_first_divergence=2))
        self.assertEqual(out["verdict"], "REJECT")
        self.assertFalse(out["scored"])
        self.assertIn("INCONCLUSIVE", out["reason"])
        self.assertIn("not the candidate's fault", out["reason"])
        self.assertNotIn("the candidate changed model output", out["reason"])

    def test_neither_is_scorable(self):
        # Refusing to score is the same in both cases; only the attribution differs.
        for c in ({"output_identical": False, "runtime_reproducible": True},
                  {"output_identical": None, "runtime_reproducible": False}):
            with self.subTest(**c):
                self.assertFalse(label.score_real(self._doc(**c))["scored"])

    def test_a_reproducible_and_identical_run_scores(self):
        out = label.score_real(self._doc(output_identical=True, runtime_reproducible=True))
        self.assertTrue(out["scored"])

    def test_the_runner_compares_the_control_against_itself_before_the_candidate(self):
        src = inspect.getsource(real_eval)
        i = src.index('f"control (reproducibility {r + 1})"')
        j = src.index('"candidate")', i)
        self.assertLess(i, j, "the control replays must precede the candidate's")
        self.assertIn("runtime_reproducible", src)

    def test_one_agreeing_pair_is_not_enough_to_certify_a_runtime(self):
        # Nondeterminism seeded by a few ULP does not fork every replay -- it forks the ones
        # where some argmax along the way happens to be close. Measured on the sparse-MoE
        # checkpoint at a 256-token prompt, five unhooked single-token replays returned
        # 8894, 8894, 25001, 25001, 8894: a single pair drawn from that agrees more often than
        # not. A gate that asks once certifies a runtime that is not reproducible at all.
        src = inspect.getsource(real_eval)
        self.assertIn("gate_control_replays", src)
        self.assertIn("all(ids == ctrl_ids for ids in ctrl_replays[1:])", src)

    def test_the_replay_count_has_a_floor_of_two(self):
        # --gate-control-replays 1 or 0 would silently disable the self-check, which is the
        # single thing standing between "this candidate changed the output" and "this runtime
        # disagrees with itself". The floor is not negotiable from the command line.
        src = inspect.getsource(real_eval)
        self.assertIn("max(2, a.gate_control_replays)", src)

    def test_a_divergent_replay_is_reported_with_which_replay_found_it(self):
        # A 2 means the runtime forked immediately; a 5 means four replays agreed before one
        # did not -- exactly the case a single-pair check would have certified. The number is
        # the difference between "not reproducible" and "not reproducibly reproducible", and a
        # reader cannot tell them apart without it.
        src = inspect.getsource(real_eval)
        self.assertIn('correctness["control_first_divergent_replay"] = len(ctrl_replays)', src)

    def test_the_verdict_names_how_many_replays_disagreed(self):
        doc = self._doc(output_identical=None, runtime_reproducible=False)
        doc["correctness"].update(control_replays=4, control_first_divergent_replay=4,
                                  control_first_divergence=11)
        reason = label.score_real(doc)["reason"]
        self.assertIn("4 of 4 unhooked control replays", reason)
        self.assertIn("token 11", reason)

    def test_an_older_result_without_the_replay_counts_still_reads(self):
        # Results written before the replay count existed must still produce a sentence, not a
        # None in the middle of one.
        doc = self._doc(output_identical=None, runtime_reproducible=False)
        doc["correctness"]["control_first_divergence"] = 2
        reason = label.score_real(doc)["reason"]
        self.assertIn("two control runs", reason)
        self.assertNotIn("None of None", reason)

    def test_the_runner_leaves_output_identical_none_when_the_control_is_unstable(self):
        src = inspect.getsource(real_eval)
        self.assertIn('correctness["output_identical"] = identical if runtime_reproducible else None',
                      src)


class CompletedRequestsGuard(unittest.TestCase):
    """The 32-sequence collapse, identified and turned into a refusal.

    Two of six identical isolated runs of the dense checkpoint at 32 sequences printed
    `[qwen35] malloc: out of memory` for most of their requests, decoded 320 and 384 tokens
    instead of 2056, and reported 142.9 and 164.8 tok/s against a healthy 925 -- with mean
    inter-token latency UNCHANGED at 20.77 and 24.26 ms against 19.29. Nothing about the
    decode path was slow. The arm lost four fifths of its requests, and aggregate tok/s is
    tokens over wall time, so a failure was reported as a slowdown. Every existing guard
    passed it: the hook ran, the packed path was used, agg_tok_s parsed.
    """

    def _out(self, tokens, oom=0, tps=925.0):
        warn = "\n".join(["[qwen35] malloc: out of memory",
                           "[warn] request error: device out of memory "
                           "(not a capacity/queue condition -- requires operator attention)"]
                          * oom)
        return (f"{warn}\nwall_s=2.22 decode_tokens={tokens} agg_tok_s={tps} "
                f"mean_itl_ms=19.29 max_itl_ms=660.78\n")

    def test_a_healthy_arm_passes_and_is_recorded(self):
        rec = real_eval.require_requests_completed(self._out(2056), 32, 64, "control c=32")
        self.assertEqual(rec["decode_tokens"], 2056)
        self.assertEqual(rec["expected_tokens"], 2048)
        self.assertEqual(rec["out_of_memory_warnings"], 0)

    def test_an_arm_that_lost_its_requests_is_refused_by_name(self):
        with self.assertRaises(SystemExit) as e:
            real_eval.require_requests_completed(self._out(320, oom=20, tps=142.9),
                                                 32, 64, "control c=32")
        msg = str(e.exception)
        self.assertIn("REQUESTS DID NOT COMPLETE", msg)
        self.assertIn("320 tokens", msg)
        self.assertIn("out-of-memory", msg)
        self.assertIn("failure reported as a slowdown", msg)

    def test_a_few_tokens_short_is_not_a_failure(self):
        # The harness cannot know the long-prefill request's exact contribution, and a
        # legitimate run has come in slightly under. The guard must not fire on that.
        self.assertIsNotNone(
            real_eval.require_requests_completed(self._out(2000), 32, 64, "control c=32"))

    def test_an_older_bench_without_the_counter_is_not_refused(self):
        # A build that does not print decode_tokens leaves the question unanswerable, and an
        # unanswerable question is not an accusation.
        self.assertIsNone(real_eval.require_requests_completed(
            "wall_s=2.2 agg_tok_s=925.0\n", 32, 64, "control c=32"))

    def test_the_guard_runs_before_the_number_is_used(self):
        src = inspect.getsource(real_eval.measure_concurrent)
        self.assertLess(src.index("require_requests_completed"),
                        src.index("require_packed_path"),
                        "a lost-request arm must be refused before anything else reads it")


class PackedPathGuard(unittest.TestCase):
    """The 32-sequence cliff, turned from an invisible number into a named refusal.

    Something occasionally drops the runtime onto its per-row decode path at 32 sequences and
    costs about a third of aggregate throughput. It is not a locality policy -- it has hit
    `baseline`, which installs no window and issues no pre-touch. The cause is still
    unidentified, but the adapter has always counted enough to tell that it happened, and
    nothing read those counters. Now they gate the arm.
    """

    def _stats(self, tokens, packed, rows):
        return {"stats": {"tokens": tokens, "tokens_packed": packed, "max_rows_seen": rows,
                          "layers_packed": packed * 30}}

    def test_a_healthy_batched_run_passes_and_is_recorded(self):
        rec = real_eval.require_packed_path(self._stats(142, 133, 9), 8, "candidate c=8",
                                            max_new=134)
        self.assertTrue(rec["used_packed_path"])
        self.assertAlmostEqual(rec["packed_share"], 133 / 134)
        # The 0.1 figure, kept under a name that says what it is: contaminated by prefill.
        self.assertAlmostEqual(rec["packed_share_of_all_steps"], 133 / 142)
        self.assertEqual(rec["max_rows_seen"], 9)

    def test_without_max_new_the_guard_falls_back_to_did_it_batch_at_all(self):
        # `packed_path_used` is called from places that do not know the decode length, and a
        # guard that guessed one would refuse honest runs. Without it the only question asked
        # is the unambiguous one.
        rec = real_eval.require_packed_path(self._stats(133, 63, 16), 16, "c=16")
        self.assertTrue(rec["used_packed_path"])
        self.assertIsNone(rec["packed_share"])
        with self.assertRaises(SystemExit):
            real_eval.require_packed_path(self._stats(266, 0, 0), 4, "c=4")

    def test_a_run_that_fell_to_the_per_row_path_is_refused_by_name(self):
        with self.assertRaises(SystemExit) as e:
            real_eval.require_packed_path(self._stats(4096, 0, 1), 32, "candidate c=32")
        self.assertIn("FELL OFF THE BATCHED DECODE PATH", str(e.exception))
        self.assertIn("max_rows_seen=1", str(e.exception))

    def test_the_tail_chunk_that_always_falls_through_is_not_a_collapse(self):
        # One row of an odd batch is never packed, by design. Refusing that would refuse
        # every honest run.
        rec = real_eval.require_packed_path(self._stats(1000, 969, 32), 32, "c=32",
                                            max_new=1000)
        self.assertTrue(rec["used_packed_path"])

    def test_batch_one_is_not_subject_to_it(self):
        # The single-sequence path is the correct path at concurrency 1; there is nothing to
        # fall off.
        self.assertIsNone(real_eval.require_packed_path(self._stats(128, 0, 0), 1, "b1"))

    def test_an_unhooked_control_arm_cannot_be_checked_and_is_not_failed(self):
        # The control is unhooked by construction, so it emits no telemetry. The guard must
        # pass it through rather than refuse every control run -- and the docstring says this
        # is the hole that remains.
        self.assertIsNone(real_eval.require_packed_path(None, 32, "control c=32"))

    def test_a_zero_token_run_does_not_divide_by_zero(self):
        with self.assertRaises(SystemExit):
            real_eval.require_packed_path(self._stats(0, 0, 0), 32, "c=32")


class WarmupDiscard(unittest.TestCase):
    """The opening measurement of a run can be slow, and it lands entirely in pair 1.

    Measured on the sparse-MoE checkpoint: the first concurrency-4 control came in at 841.9
    aggregate tok/s against 910.9 and 917.6 for the two after it. Every candidate in that pair
    was compared against the slow control, so pair 1's ratios all read about 8% high, the
    control's own spread became 8.3%, and an arm whose real differences are a fraction of a
    percent could not resolve. Discarding leading runs is the fix; defaulting it to zero is
    what keeps every result published before it reproducible.
    """

    def test_both_runners_take_the_flag_and_default_to_discarding_nothing(self):
        for mod in (real_eval, real_sweep):
            with self.subTest(runner=mod.__name__):
                src = inspect.getsource(mod)
                self.assertIn("--warmup-runs", src)
                self.assertIn('ap.add_argument("--warmup-runs", type=int, default=0', src)

    def test_the_count_is_recorded_in_the_artifact(self):
        # A result whose noise floor was helped by discarding runs has to say so, or the floor
        # is not comparable with one measured without it.
        for mod in (real_eval, real_sweep):
            with self.subTest(runner=mod.__name__):
                self.assertIn("warmup_runs_discarded", inspect.getsource(mod))

    def test_warm_up_runs_are_not_counted_as_repeats(self):
        # The discarded runs must not reach control_runs/ctrl_runs, or they would be exactly
        # the outlier the flag exists to remove.
        src = inspect.getsource(real_sweep)
        i = src.index("for w in range(a.warmup_runs):")
        j = src.index("for rep in range(a.repeats):", i)
        self.assertNotIn("append", src[i:j])


class BreakEvenScreen(unittest.TestCase):
    """The number to screen a candidate model with before integrating it.

    The persist bound is 2*min(capacity, footprint) / step_traffic. The capacity is the
    device's and cannot be raised, so on any model whose footprint already exceeds the cache
    the ONLY lever is the denominator. Inverting the bound at the significance floor turns
    'try a sparser model' from advice into a threshold.
    """

    PIN = MatrixCeiling.PIN

    def _pf(self, ms, cap=62914560, m=None):
        return traffic_budget.arm_ceiling(m or self.PIN, 1, ms, 1.0, 1792.0,
                                          cap)["persist_family"]

    def test_a_step_at_the_break_even_lands_exactly_on_the_floor(self):
        pf = self._pf(10.0)
        at = pf["break_even_step_traffic_bytes"]
        # Re-run with a step time that moves exactly that many bytes.
        pf2 = self._pf(at / (1792.0 * 1e9) * 1000.0)
        self.assertAlmostEqual(pf2["ceiling_pct"], label.SIGNIFICANCE_PCT, places=6)

    def test_the_flag_agrees_with_the_ceiling_it_summarises(self):
        for ms in (0.5, 1.0, 2.0, 3.5, 6.0, 10.34, 40.0):
            with self.subTest(ms=ms):
                pf = self._pf(ms)
                self.assertEqual(pf["clears_significance_floor"],
                                 pf["ceiling_pct"] >= label.SIGNIFICANCE_PCT)

    def test_the_pinned_dense_model_does_not_clear_it(self):
        pf = self._pf(10.344706303443338)
        self.assertFalse(pf["clears_significance_floor"])
        self.assertGreater(pf["step_traffic_bytes"], pf["break_even_step_traffic_bytes"])

    def test_the_break_even_does_not_depend_on_the_step_it_is_measured_against(self):
        # It is a property of the footprint and the cache, not of the workload -- otherwise it
        # could not be used to screen a model whose decode rate is not yet known.
        self.assertAlmostEqual(self._pf(2.0)["break_even_step_traffic_bytes"],
                               self._pf(40.0)["break_even_step_traffic_bytes"])

    def test_a_footprint_smaller_than_the_cache_lowers_the_break_even(self):
        # min(capacity, footprint): a model whose whole recurrent state fits has less to save,
        # so it needs an even shorter step to be worth a window. Screening must not reward a
        # tiny state by pretending the cache is full of it.
        small = dict(self.PIN, recurrent_layers=4)
        self.assertLess(self._pf(10.0, m=small)["break_even_step_traffic_bytes"],
                        self._pf(10.0)["break_even_step_traffic_bytes"])


class SecondModelGeometry(unittest.TestCase):
    """A second model measured on the same runtime must not inherit the first one's shape.

    The matrix spec carries decode rates. If the geometry silently came from the pinned
    integration while the rates came from another model, the tool would report a confident
    ceiling for a state shape that model does not have, and nothing in the output would say
    so. This is the same failure the coverage guard exists for: a wrong number that looks
    exactly like a right one.
    """

    PIN = MatrixCeiling.PIN
    OTHER = {"label": "test hybrid", "recurrent_layers": 30,
             "lin_state_bytes_per_layer": 2097152, "lin_conv_state_bytes_per_layer": 49152,
             "linear_conv_kernel_dim": 4}

    def _run(self, spec):
        import contextlib, io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            traffic_budget.matrix_ceiling(self.PIN, spec, 1792.0, None, 62914560)
        return json.loads(buf.getvalue())

    def _spec(self, model=None):
        spec = {"arms": {"batch1": {"sequences": 1, "ms_per_token": 2.0,
                                    "state_bytes_scale": 1.0}}}
        if model:
            spec["model"] = model
        return spec

    def test_the_spec_model_is_used_and_recorded(self):
        out = self._run(self._spec(self.OTHER))
        self.assertEqual(out["model_geometry_source"], "matrix spec")
        self.assertEqual(out["model"], "test hybrid")
        self.assertEqual(out["arms"]["batch1"]["recurrent_layers"], 30)

    def test_the_pin_is_used_and_recorded_when_the_spec_carries_none(self):
        out = self._run(self._spec())
        self.assertEqual(out["model_geometry_source"], "pin")
        self.assertEqual(out["arms"]["batch1"]["recurrent_layers"],
                         self.PIN["recurrent_layers"])

    def test_the_two_geometries_do_not_produce_the_same_ceiling(self):
        # If they did, the override would be untested by everything above.
        self.assertNotAlmostEqual(
            self._run(self._spec(self.OTHER))["arms"]["batch1"]["ceiling_pct"],
            self._run(self._spec())["arms"]["batch1"]["ceiling_pct"], places=3)


class BandwidthBoundCheck(unittest.TestCase):
    """Is the ceiling tight, or merely true?

    `implied_total_bytes_per_token` is measured time times peak bandwidth: what the step could
    have moved, not what it did. On a dense model streaming every weight those are nearly the
    same and the ceiling is close to achievable. On a sparse MoE the step reads a tenth of its
    weights and spends much of its time on launch latency and small-GEMV occupancy, so the
    same arithmetic produces a ceiling no policy could approach. Publishing the second kind
    without saying which it is would be the ceiling-in-the-wrong-currency mistake again.
    """

    def _check(self, active, total):
        return traffic_budget.bandwidth_bound_check(active, total)["bandwidth_bound_check"]

    def test_a_saturated_step_is_tight(self):
        c = self._check(18.0e9, 18.5e9)
        self.assertEqual(c["bound"], "tight")
        self.assertAlmostEqual(c["bandwidth_utilisation"], 18.0 / 18.5)

    def test_a_half_idle_memory_system_is_loose(self):
        c = self._check(1.7e9, 3.5e9)
        self.assertEqual(c["bound"], "loose")
        self.assertIn("upper bound", c["note"])

    def test_the_threshold_is_the_named_constant_and_is_inclusive(self):
        total = 10.0e9
        self.assertEqual(self._check(traffic_budget.BANDWIDTH_BOUND_MIN * total, total)["bound"],
                         "tight")
        self.assertEqual(self._check(traffic_budget.BANDWIDTH_BOUND_MIN * total * 0.999,
                                     total)["bound"], "loose")

    def test_the_state_traffic_is_added_to_the_declared_weight_bytes(self):
        # A config states one number, the weights the checkpoint says the step reads. The
        # recurrent traffic is already computed from the geometry, so composing them here is
        # what keeps a concurrency arm from having to hand-total bytes that scale with rows.
        arm = traffic_budget.arm_ceiling(MatrixCeiling.PIN, 4, 12.0, 0.5, 1792.0,
                                         active_weight_bytes_per_token=int(2.0e9))
        c = arm["bandwidth_bound_check"]
        self.assertEqual(c["active_weight_bytes_per_token"], int(2.0e9))
        self.assertAlmostEqual(c["active_bytes_per_token"],
                               2.0e9 + arm["state_traffic_bytes_per_token"])

    def test_it_is_absent_unless_the_arm_declares_active_bytes(self):
        # It is an optional diagnostic, so an arm that cannot supply the number must still
        # produce a ceiling rather than a crash or a fabricated utilisation.
        arm = traffic_budget.arm_ceiling(MatrixCeiling.PIN, 1, 10.34, 1.0, 1792.0)
        self.assertNotIn("bandwidth_bound_check", arm)


if __name__ == "__main__":
    unittest.main(verbosity=2)
