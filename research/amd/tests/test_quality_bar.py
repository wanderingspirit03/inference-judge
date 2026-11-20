"""The fixed quality bar (lib/quality.py) over HTTP with the stand-ins, as in
test_protocol.py: measure A from the walk, measure B from stand-in tasks
(controller/standin/standin_tasks.py), and the verdict of answers.decide.

The limits here are stand-in limits for the toy model only; the real config's
bar (set on 2026-11-18) is checked by test_real_config_has_the_bar_the_user_set.  The honest quantised
stand-in (fake_engine --quantised: a little noise in the logits) fails the
agreement limit and must pass this bar; every cheater must fail it, and fail
each of its limits on its own.  Stdlib only."""
import copy
import json
import sys
import unittest

from test_protocol import AMD, CFG, ITEMS, STOPS, Sess, serve, server_at

sys.path.insert(0, str(AMD / "examples" / "cheaters"))
import answers  # noqa: E402
import cheat  # noqa: E402
import fake_engine  # noqa: E402
import quality  # noqa: E402
import standin_tasks  # noqa: E402
import tokmap  # noqa: E402
import toylm  # noqa: E402

MORE = [{"id": f"m{i}", "ids": toylm.encode(t), "max_tokens": 32} for i, t in enumerate([
    "Once upon a time in a quiet harbour", "SELECT name FROM users WHERE", "The recipe calls for two cups of",
    "Dear committee, I write to", "A list of rivers: Nile, Amazon,"])]
JUDGED = ITEMS + MORE
TASKS = standin_tasks.tasks(48, seed=7)
TM = tokmap.load(CFG["model"]["dir"])

# stand-in limits, far from both sides: quantised top-5 1.0, gap 0.06, tasks 47/48;
# damage 3: top-5 0.40, gap 2.1, tasks about 9/48
STANDIN_BAR = {"min_topk_share": {"1": None, "2": None, "5": 0.9}, "max_mean_gap": 0.5, "min_task_delta_lb": -0.15}
OPEN = {"min_topk_share": {"5": 0.0}, "max_mean_gap": 1e9, "min_task_delta_lb": -1.0}


def cfg_with(bar=None, replaces=True):
    c = copy.deepcopy(CFG)
    c["quality"] = {"topk": [1, 2, 5], "bar_replaces_agreement": replaces,
                    "bar": bar or {"min_topk_share": {"1": None, "2": None, "5": None}, "max_mean_gap": None,
                                   "min_task_delta_lb": None}}
    return c


def noisy(sigma, ids_only=False):
    class H(fake_engine.H):
        def do_POST(self):
            fake_engine.DAMAGE, fake_engine.IDS_ONLY = sigma, ids_only
            try:
                super().do_POST()
            finally:
                fake_engine.DAMAGE, fake_engine.IDS_ONLY = 0.0, False
    return H


def cheater(m):
    class H(cheat.H):
        def do_POST(self):
            cheat.MODE = m
            super().do_POST()
    return H


RUNS = {}


def run(handler, key=None):
    """The engine answers the judged prompts and the tasks; the reference
    walks the answers and answers the tasks.  Returns the outputs and the
    measures (cached per key: the verdicts under different bars reuse them)."""
    if key in RUNS:
        return RUNS[key]
    srv, port = serve(handler)
    try:
        e = server_at(port)
        outs = answers.engine_answers(CFG, e, JUDGED, lambda m: None)
        titems = [{"id": t["id"], "ids": t["prompt_token_ids"], "max_tokens": t["max_tokens"], "task": t} for t in TASKS]
        etexts = answers.engine_task_texts(CFG, e, titems, TM, lambda m: None)
    finally:
        srv.shutdown()
        srv.server_close()
    s = Sess()
    rows = answers.walk(CFG, s, JUDGED, outs, answers.limits(CFG), lambda m: None)
    qb = quality.measure_b(TASKS, etexts, answers.reference_task_texts(s, titems, TM))
    got = {"outs": outs, "items": [r["item"] for r in rows], "qb": qb}
    if key:
        RUNS[key] = got
    return got


def verdict(r, cfg):
    return answers.decide(cfg, r["items"], answers.limits(cfg), r["qb"])


class MeasuresTest(unittest.TestCase):
    def test_honest_engine_measures(self):
        r = run(fake_engine.H, "honest")
        ok, why, agg, q = verdict(r, cfg_with())
        self.assertTrue(ok, why)
        self.assertEqual(q["verdict"], quality.NOT_SET)
        self.assertEqual(q["a"]["agreement_strict"], 1.0)
        self.assertEqual(q["a"]["topk_share"], {"1": 1.0, "2": 1.0, "5": 1.0})
        self.assertEqual(q["a"]["mean_gap"], 0.0)
        self.assertEqual(q["a"]["walked"], q["a"]["positions"])
        self.assertEqual((q["b"]["engine_correct"], q["b"]["reference_correct"], q["b"]["tasks"]), (48, 48, 48))
        self.assertEqual(q["b"]["delta_lb95"], 0.0)

    def test_reference_answers_every_standin_task(self):
        r = run(fake_engine.H, "honest")
        self.assertTrue(all(t["reference_ok"] for t in r["qb"]["per_task"]))

    def test_a_needs_no_engine_logprobs(self):
        """The same engine sending token ids only gets the same measure A."""
        a = verdict(run(noisy(0.3), "quantised"), cfg_with())[3]["a"]
        b = verdict(run(noisy(0.3, ids_only=True), "quantised-ids"), cfg_with())[3]["a"]
        self.assertEqual(a, b)
        self.assertTrue(all(o.get("lps") and all(x is None for x in o["lps"])
                            for o in run(noisy(0.3, ids_only=True), "quantised-ids")["outs"]))

    def test_rank_and_gap_of_a_parting(self):
        from compare import _rank
        top = {1: -0.1, 2: -0.5, 3: -0.5, 4: -2.0}
        self.assertEqual(_rank(top, 1), (1, 0.0, False))
        self.assertEqual(_rank(top, 3)[0], 2)               # a tie shares the better rank
        self.assertAlmostEqual(_rank(top, 4)[1], 1.9)
        self.assertEqual(_rank(top, 9)[0], None)            # outside the top: the smallest gap it can have
        self.assertAlmostEqual(_rank(top, 9)[1], 1.9)

    def test_paired_bound(self):
        self.assertEqual(quality.paired_lower_bound([0] * 50, 1), 0.0)
        lb = quality.paired_lower_bound([-1] * 10 + [0] * 40, 1)
        self.assertLess(lb, -0.2)
        self.assertGreater(lb, -0.4)
        self.assertEqual(lb, quality.paired_lower_bound([-1] * 10 + [0] * 40, 1))

    def test_grading_follows_quality2(self):
        self.assertTrue(quality.correct({"kind": "math", "answer": 1234}, "so 1,000 + 234\nANSWER: 1,234"))
        self.assertFalse(quality.correct({"kind": "math", "answer": 12}, "ANSWER: 12.5"))
        self.assertTrue(quality.correct({"kind": "mc", "answer": "C"}, "The answer is (C)."))
        self.assertTrue(quality.correct({"kind": "mc", "answer": "B"}, "B"))
        self.assertFalse(quality.correct({"kind": "mc", "answer": "B"}, "no letter here"))


class BarNotSetTest(unittest.TestCase):
    def test_quantised_fails_the_agreement_limit_and_quality_is_only_recorded(self):
        r = run(noisy(0.3), "quantised")
        ok, why, agg, q = verdict(r, cfg_with())
        self.assertFalse(ok)
        self.assertIn("agreement with stock", why[-1])
        self.assertEqual(q["verdict"], quality.NOT_SET)
        self.assertEqual(sorted(q["not_set"]), ["max_mean_gap", "min_task_delta_lb", "min_topk_share"])
        self.assertLess(q["a"]["agreement_strict"], 0.95)
        self.assertTrue(any(quality.NOT_SET in t for t in quality.lines(q)))

    def test_partly_set_is_not_set(self):
        bar = dict(STANDIN_BAR, max_mean_gap=None)
        ok, why, agg, q = verdict(run(noisy(0.3), "quantised"), cfg_with(bar))
        self.assertEqual(q["verdict"], quality.NOT_SET)
        self.assertFalse(ok)                                    # the agreement limit still applies

    def test_a_null_bar_is_not_set(self):
        real = json.loads((AMD / "config.json").read_text())
        real["quality"]["bar"] = {"min_topk_share": {"1": None, "2": None, "5": None}, "max_mean_gap": None,
                                  "min_task_delta_lb": None}
        self.assertFalse(quality.is_set(real))
        self.assertFalse(quality.replaces_agreement(real))
        self.assertEqual(sorted(quality.missing(real)), ["max_mean_gap", "min_task_delta_lb", "min_topk_share"])

    def test_real_config_has_the_bar_the_user_set(self):
        # set by the user on 2026-11-18 from the card numbers (measurements/2026-11-17/NOTES.md)
        real = json.loads((AMD / "config.json").read_text())
        self.assertTrue(quality.is_set(real))
        self.assertTrue(quality.replaces_agreement(real))
        self.assertEqual(quality.topk_limits(real), {"1": None, "2": None, "5": 0.98})
        b = quality.bar(real)
        self.assertEqual((b["max_mean_gap"], b["min_task_delta_lb"]), (0.08, -0.15))
        self.assertEqual(real["limits"]["min_agreement"], 0.95)     # unchanged; reported, not applied
        self.assertTrue(real["limits"]["compare_stop"])
        # the stand-in GPU machine's config (the e2e) judges by the same bar
        standin = json.loads((AMD / "controller" / "standin" / "amd-config.standin.json").read_text())
        self.assertEqual(quality.bar(standin)["min_topk_share"], b["min_topk_share"])
        self.assertEqual([quality.bar(standin)[k] for k in ("max_mean_gap", "min_task_delta_lb")], [0.08, -0.15])
        self.assertTrue(quality.replaces_agreement(standin))


class BarSetTest(unittest.TestCase):
    def test_quantised_passes_the_standin_bar(self):
        r = run(noisy(0.3), "quantised")
        ok, why, agg, q = verdict(r, cfg_with(STANDIN_BAR))
        self.assertTrue(ok, why)
        self.assertEqual(q["verdict"], "pass")
        self.assertLess(agg["agreement"], 0.95)
        self.assertLess(q["a"]["agreement_strict"], 0.95)

    def test_token_ids_only_quantised_passes(self):
        ok, why, *_ = verdict(run(noisy(0.3, ids_only=True), "quantised-ids"), cfg_with(STANDIN_BAR))
        self.assertTrue(ok, why)

    def test_without_replacing_the_agreement_limit_both_apply(self):
        ok, why, agg, q = verdict(run(noisy(0.3), "quantised"), cfg_with(STANDIN_BAR, replaces=False))
        self.assertFalse(ok)
        self.assertEqual(q["verdict"], "pass")
        self.assertTrue(any("agreement with stock" in w for w in why))

    def test_honest_passes(self):
        ok, why, *_ = verdict(run(fake_engine.H, "honest"), cfg_with(STANDIN_BAR))
        self.assertTrue(ok, why)

    ALONE = {"top-5": (dict(OPEN, min_topk_share={"5": 0.9}), "top-5 share"),
             "gap": (dict(OPEN, max_mean_gap=0.5), "mean gap"),
             "tasks": (dict(OPEN, min_task_delta_lb=-0.15), "task accuracy")}

    def _fails(self, r, label, limits=("top-5", "gap", "tasks")):
        ok, why, agg, q = verdict(r, cfg_with(STANDIN_BAR))
        self.assertFalse(ok, label)
        self.assertEqual(q["verdict"], "fail", label)
        # each limit alone, the others wide open: that limit's own reason is there
        for name in limits:
            bar, text = self.ALONE[name]
            ok, why, agg, q = verdict(r, cfg_with(bar))
            self.assertFalse(ok, f"{label}: {name} alone")
            self.assertTrue(any(text in w for w in q["reasons"]), f"{label}: {name} alone: {q['reasons']}")
        return why

    def test_damaged_engine_fails_each_limit(self):
        self._fails(run(noisy(3.0), "damaged"), "damaged")

    def test_fabricator_fails_measure_a(self):
        """It answers short requests honestly (a task is one token), so
        measure B cannot see it; measure A does, and so does the walk."""
        r = run(cheater("fabricate"), "fabricate")
        self._fails(r, "fabricate", ("top-5", "gap"))
        self.assertEqual(r["qb"]["delta"], 0.0)

    def test_replay_on_fresh_prompts_fails_each_limit(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            cheat.STORE = Path(d) / "replay"
            self._fails(run(cheater("replay"), None), "replay")

    def test_proxy_with_the_reference_down_fails(self):
        cheat.REF = "http://127.0.0.1:9"
        try:
            r = run(cheater("proxy-ref"), None)
        finally:
            cheat.REF = "http://127.0.0.1:8001"
        ok, why, agg, q = verdict(r, cfg_with(STANDIN_BAR))
        self.assertFalse(ok)
        self.assertTrue(any("request failed" in w for w in why))
        self.assertEqual(q["verdict"], "fail")

    def test_no_tasks_fails_once_set(self):
        r = dict(run(noisy(0.3), "quantised"), qb={"tasks": 0, "note": "none"})
        ok, why, agg, q = verdict(r, cfg_with(STANDIN_BAR))
        self.assertFalse(ok)
        self.assertTrue(any("measure B was not taken" in w for w in why))


if __name__ == "__main__":
    unittest.main()
