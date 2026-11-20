"""The speed breakdown of ./check and ./submit (answers.breakdown) and
./check's order (answers.start_then_items): a serve that cannot start fails
before the reference is started."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(__file__))
import test_protocol as TP  # noqa: E402  (stand-in reference, engine and items)

answers = TP.answers
E = TP.E


def honest_breakdown(repeats):
    seen = []

    def record(body, doc):
        seen.append(body)
        return doc
    srv, port = TP.serve(TP.rewrite(record))
    try:
        e = TP.server_at(port)
        outs = answers.engine_answers(TP.CFG, e, TP.ITEMS, lambda m: None)
        n_judged = len(seen)
        b = answers.breakdown(e, TP.ITEMS, outs, TP.STOPS, repeats, lambda m: None)
        return b, outs, seen[:n_judged], seen[n_judged:]
    finally:
        srv.shutdown()
        srv.server_close()


class BreakdownTest(unittest.TestCase):
    def test_parts_come_from_separate_requests(self):
        b, outs, judged, extra = honest_breakdown(2)
        self.assertEqual(len(extra), 2 * 2 * len(TP.ITEMS))            # per repeat: a max_tokens=1 and a whole request
        self.assertEqual(sum(1 for x in extra if x["max_tokens"] == 1), 2 * len(TP.ITEMS))
        judged_prompts = {tuple(x["prompt"]) for x in judged}
        self.assertFalse(any(tuple(x["prompt"]) in judged_prompts for x in extra))
        self.assertEqual({tuple(sorted(x)) for x in judged + extra}.__len__(), 1)   # one request shape throughout
        self.assertEqual(len(b["requests"]), 2 * len(TP.ITEMS))
        prompts = [tuple(x["prompt"]) for x in judged + extra]
        self.assertEqual(len(set(prompts)), len(prompts))              # no request repeats an earlier prompt (no cache hit)

    def test_min_max_over_passes_and_the_judges_formula(self):
        b, outs, _, _ = honest_breakdown(3)
        self.assertEqual([s["pass"] for s in b["samples"]], ["judged", "repeat 1", "repeat 2", "repeat 3"])
        for key in ("whole_tps", "decode_tps", "prompt_s"):
            self.assertLessEqual(b[key]["min"], b[key]["max"], key)
        sp = answers.speed(TP.ITEMS, outs, TP.STOPS)
        self.assertEqual(b["samples"][0]["whole_tps"], sp["tokens_per_s"])
        for r in (1, 2, 3):
            per = [q for q in b["requests"] if q["repeat"] == r]
            whole = round(sum(q["tokens"] for q in per) / sum(q["seconds"] for q in per), 2)
            self.assertEqual(b["samples"][r]["whole_tps"], whole)
        self.assertIn("prompt time included", b["formula"])
        line = answers.breakdown_line(b)
        self.assertIn("whole request", line)
        self.assertIn("decode", line)
        self.assertIn("prompt time", line)

    def test_set_rates_by_hand(self):
        per = [{"tokens": 10, "seconds": 1.0, "prompt_s": 0.1}, {"tokens": 30, "seconds": 3.0, "prompt_s": 0.5}]
        r = answers._set_rates(per)
        self.assertEqual(r["whole_tps"], 10.0)                         # 40 tokens / 4 s
        self.assertEqual(r["prompt_s"], 0.6)
        self.assertEqual(r["decode_tps"], round(38 / 3.4, 2))          # (9 + 29) tokens / (0.9 + 2.5) s


class FakeSess:
    def __init__(self, cached):
        self.cached, self.tokenized, self.closed = cached, 0, 0

    def ids_cached(self, p):
        return self.cached

    def ids_for(self, p):
        self.tokenized += 1
        return [1, 2, 3]

    def close(self):
        self.closed += 1


class FakeServer:
    def __init__(self, fail):
        self.fail, self.starts, self.stops = fail, 0, 0

    def start(self):
        self.starts += 1
        if self.fail:
            raise E.EngineFailed("./serve exited with 1 before it answered")
        return 1.0

    def stop(self):
        self.stops += 1


class CheckOrderTest(unittest.TestCase):
    P = [{"id": "a", "prompt": "x"}, {"id": "b", "prompt": "y"}]

    def test_serve_that_cannot_start_fails_before_the_reference(self):
        s, srv = FakeSess(cached=False), FakeServer(fail=True)
        with self.assertRaises(E.EngineFailed):
            answers.start_then_items(s, srv, self.P, 8, {}, lambda m: None)
        self.assertEqual(s.tokenized, 0)                               # the reference was never asked

    def test_uncached_prompts_start_serve_first_then_tokenize(self):
        s, srv, times = FakeSess(cached=False), FakeServer(fail=False), {}
        items = answers.start_then_items(s, srv, self.P, 8, times, lambda m: None)
        self.assertEqual((srv.starts, srv.stops, s.tokenized, s.closed), (2, 1, 2, 1))
        self.assertEqual([i["id"] for i in items], ["a", "b"])
        self.assertIn("engine_first_start_s", times)

    def test_cached_prompts_start_serve_once(self):
        s, srv, times = FakeSess(cached=True), FakeServer(fail=False), {}
        answers.start_then_items(s, srv, self.P, 8, times, lambda m: None)
        self.assertEqual((srv.starts, srv.stops), (1, 0))
        self.assertNotIn("engine_first_start_s", times)


if __name__ == "__main__":
    unittest.main()
