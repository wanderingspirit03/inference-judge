"""The walk of protocol 2 (compare.walk_engine): the engine answers once, the
reference walks along it.  Stdlib only; the model is the stand-ins' toylm.
  python3 -m unittest discover -s research/amd/tests -p 'test_*.py'"""
import sys
import unittest
from pathlib import Path

AMD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AMD / "lib"))
sys.path.insert(0, str(AMD / "controller" / "standin"))
import compare as C  # noqa: E402
import toylm  # noqa: E402

LIM = C.Limits(min_agreement=0.95, tie_eps=0.125, max_partings_per_item=40)
STOPS = {toylm.EOS}


def ref_answer(ids, n, top=20) -> C.RefSeg:
    """The reference: toylm greedy, stops at EOS; the stop entry kept apart."""
    gen = toylm.greedy(list(ids), n, ignore_eos=False)
    seg = C.RefSeg([], [], "length", None)
    for g, lp in gen:
        t = {j: lp[j] for j in toylm.top(lp, top)}
        if g == toylm.EOS:
            seg.finish, seg.stop = "stop", {"id": g, "top": t}
            break
        seg.tokens.append(g)
        seg.top.append(t)
    return seg


class Ref:
    def __init__(self, ids, max_tokens):
        self.ids, self.n, self.calls = list(ids), max_tokens, []

    def first(self):
        return ref_answer(self.ids, self.n)

    def resumer(self, eng):
        def resume(pos):
            self.calls.append(pos)
            return ref_answer(self.ids + list(eng[:pos + 1]), self.n - pos - 1)
        return resume


def engine(ids, n, damage=0.0):
    """An engine on the same model: (tokens, lps, finish), the stop token included as vLLM gives it."""
    gen = toylm.greedy(list(ids), n, damage=damage, ignore_eos=False)
    toks = [g for g, _ in gen]
    return toks, [lp[g] for g, lp in gen], ("stop" if toks and toks[-1] == toylm.EOS else "length")


def walk(ids, n, eng, lim=LIM):
    r = Ref(ids, n)
    toks, lps, fin = eng
    return C.walk_engine("p", toks, lps, fin, n, r.first(), r.resumer(toks), lim, STOPS), r


PROMPT = toylm.encode("The lighthouse keeper had not seen a ship in eleven days.")


class WalkTest(unittest.TestCase):
    def test_same_model_agrees_everywhere(self):
        it, r = walk(PROMPT, 64, engine(PROMPT, 64))
        self.assertTrue(it.ok)
        self.assertEqual((it.positions, it.compared, it.partings, r.calls), (64, 64, [], []))

    def test_damaged_engine_fails(self):
        items = []
        for k in range(4):
            ids = PROMPT + toylm.encode(f" {k}")
            items.append(walk(ids, 64, engine(ids, 64, damage=8.0))[0])
        ok, why, agg = C.judge(items, LIM)
        self.assertFalse(ok)
        self.assertLess(agg["agreement"], 0.9)

    def test_old_walk_passes_a_garbage_first_answer_new_walk_does_not(self):
        """Protocol 1 resumed the engine from the reference's prefix, so an
        engine that answers the first request badly and the resumes well lost
        one position per answer."""
        n = 64
        ref = ref_answer(PROMPT, n)
        top = ref.top
        garbage = ([ord("#")] * n, [0.0] * n)

        def honest_resume(pos):
            t, lp, _ = engine(PROMPT + ref.tokens[:pos + 1], n - pos - 1)
            return t, lp
        old = C.compare_item("p", ref.tokens, top, garbage, honest_resume, LIM)
        ok_old, _, agg_old = C.judge([old], LIM)
        self.assertTrue(ok_old, agg_old)
        self.assertAlmostEqual(agg_old["agreement"], (n - 1) / n, places=4)
        new, _ = walk(PROMPT, n, (garbage[0], garbage[1], "length"))
        ok_new, _, agg_new = C.judge([new], LIM)
        self.assertFalse(ok_new)
        self.assertLess(agg_new["agreement"], 0.1)

    def test_partings_resume_the_reference_from_the_engines_prefix(self):
        toks, lps, fin = engine(PROMPT, 32)
        toks = list(toks)
        toks[5] = ord("#") if toks[5] != ord("#") else ord("%")
        # after its odd token the engine goes on as the model would
        rest = engine(PROMPT + toks[:6], 26)
        toks[6:], lps[6:] = rest[0], rest[1]
        it, r = walk(PROMPT, 32, (toks, lps, "length"))
        self.assertEqual(r.calls, [5])
        self.assertEqual([p for p, _ in it.partings], [5])
        self.assertEqual(it.compared, 31)
        self.assertEqual(it.notes[0]["eng"], toks[5])

    def test_engine_stops_where_the_reference_goes_on(self):
        n = 32
        toks, lps, _ = engine(PROMPT, 10)
        it, _ = walk(PROMPT, n, (toks + [toylm.EOS], lps + [-0.1], "stop"))
        self.assertTrue(it.ok)
        self.assertEqual(it.positions, 11)
        self.assertEqual(it.compared, 10)
        self.assertEqual(it.notes[0]["eng"], "stop")
        self.assertIsNone(it.partings[0][1])          # EOS is far outside the reference's top here
        off, _ = walk(PROMPT, n, (toks + [toylm.EOS], lps + [-0.1], "stop"),
                      C.Limits(max_partings_per_item=40, compare_stop=False))
        self.assertEqual((off.positions, off.compared, off.partings), (10, 10, []))

    def test_both_stop_at_the_same_place(self):
        ids = toylm.encode("hello ~")
        eng = engine(ids, 16)
        self.assertEqual(eng[2], "stop")
        it, _ = walk(ids, 16, eng)
        self.assertEqual((it.positions, it.compared, it.partings), (len(eng[0]), len(eng[0]), []))
        only = toylm.encode("hello ~~")
        eng = engine(only, 16)
        self.assertEqual(eng[0], [toylm.EOS])          # an answer that is only the stop token
        it, _ = walk(only, 16, eng)
        self.assertEqual((it.ok, it.positions, it.compared), (True, 1, 1))

    def test_reference_stops_where_the_engine_goes_on(self):
        ids = toylm.encode("hello ~")
        ref = ref_answer(ids, 16)
        self.assertEqual(ref.finish, "stop")
        k = len(ref.tokens)
        toks = ref.tokens + [ord("x")] * (16 - k)
        it, r = walk(ids, 16, (toks, [0.0] * 16, "length"))
        self.assertEqual(it.notes[0]["ref"], "stop")
        self.assertEqual(it.notes[0]["pos"], k)
        self.assertEqual(r.calls[0], k)                # the reference goes on from the engine's prefix
        self.assertEqual(it.positions, 16)

    def test_short_answer_without_stop_fails(self):
        toks, lps, _ = engine(PROMPT, 10)
        it, _ = walk(PROMPT, 32, (toks, lps, "length"))
        self.assertFalse(it.ok)
        self.assertIn("did not stop", it.reason)

    def test_stop_token_inside_the_answer_fails(self):
        toks, lps, _ = engine(PROMPT, 10)
        toks[4] = toylm.EOS
        it, _ = walk(PROMPT, 10, (toks, lps, "length"))
        self.assertFalse(it.ok)

    def test_more_tokens_than_asked_fails(self):
        toks, lps, _ = engine(PROMPT, 12)
        it, _ = walk(PROMPT, 10, (toks, lps, "length"))
        self.assertFalse(it.ok)

    def test_close_call_counts_as_agreeing(self):
        first = C.RefSeg([1, 2, 3], [{1: -0.1}, {2: -0.5, 9: -0.55}, {3: -0.1}], "length")

        def resume(pos):
            return C.RefSeg([3], [{3: -0.1}], "length")
        it = C.walk_engine("p", [1, 9, 3], [0, 0, 0], "length", 3, first, resume, LIM, STOPS)
        self.assertEqual((it.compared, it.close, it.positions), (2, 1, 3))
        ok, _, agg = C.judge([it], LIM)
        self.assertEqual((agg["agreement"], round(agg["agreement_strict"], 3)), (1.0, 0.667))

    def test_cut_after_max_partings(self):
        n = 64
        toks = [ord("#")] * n
        it, r = walk(PROMPT, n, (toks, [0.0] * n, "length"), C.Limits(max_partings_per_item=5))
        self.assertTrue(it.cut)
        self.assertEqual(len(r.calls), 4)
        self.assertEqual(it.positions, n)


if __name__ == "__main__":
    unittest.main()
