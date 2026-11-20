"""The cheating stand-in engines of examples/cheaters, judged in process by
protocol 2 (engine answers, then the reference walks them), as in
test_protocol.py.  The account and pool defenses (cand, fresh draws) are
tested on the controller by controller/tests/e2e-standin.sh; here the
controls show each attack would work without them.  Stdlib only.
  python3 -m unittest discover -s research/amd/tests -p 'test_*.py'"""
import sys
import tempfile
import unittest
from pathlib import Path

from test_protocol import ITEMS, judge, AMD, CFG

sys.path.insert(0, str(AMD / "examples" / "cheaters"))
import cheat  # noqa: E402
import toylm  # noqa: E402


def mode(m):
    class H(cheat.H):
        def do_POST(self):
            cheat.MODE = m
            super().do_POST()
    return H


def items(texts):
    return [{"id": f"q{i}", "ids": toylm.encode(t), "max_tokens": 32} for i, t in enumerate(texts)]


class CheatersTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cheat.STORE = Path(self.tmp.name) / "replay"

    def tearDown(self):
        self.tmp.cleanup()

    def test_fabricator_fails(self):
        ok, why, agg, sp, outs, rows = judge(mode("fabricate"))
        self.assertFalse(ok)
        self.assertTrue(all("error" not in o for o in outs))
        self.assertIn("agreement with stock", why[-1])

    def test_replay_fails_on_fresh_prompts(self):
        first = items(["Once upon a time in a quiet harbour", "SELECT name FROM users WHERE"])
        second = items(["The recipe calls for two cups of", "import numpy as np\nx ="])
        self.assertFalse(judge(mode("replay"), first)[0])
        ok, why, *_ = judge(mode("replay"), second)
        self.assertFalse(ok)
        self.assertIn("agreement with stock", why[-1])

    def test_replay_control_passes_when_prompts_come_back(self):
        """Why the pool never sends a prompt twice: the same prompts again
        are answered from the record, correctly and at once."""
        first = items(["Once upon a time in a quiet harbour", "SELECT name FROM users WHERE"])
        self.assertFalse(judge(mode("replay"), first)[0])
        ok, why, agg, sp, *_ = judge(mode("replay"), first)
        self.assertTrue(ok, why)

    def test_proxy_fails_when_the_reference_is_down(self):
        cheat.REF = "http://127.0.0.1:9"            # nothing listens: as during the engine's turn in protocol 2
        try:
            ok, why, agg, sp, outs, rows = judge(mode("proxy-ref"))
        finally:
            cheat.REF = "http://127.0.0.1:8001"
        self.assertFalse(ok)
        self.assertTrue(all("error" in o for o in outs))

    def test_proxy_control_passes_when_the_reference_is_up(self):
        """Why the reference never runs while the engine's account has a
        process: a proxy to it passes."""
        cheat.REF = CFG["ref"]["url"]
        try:
            ok, why, *_ = judge(mode("proxy-ref"))
        finally:
            cheat.REF = "http://127.0.0.1:8001"
        self.assertTrue(ok, why)

    def test_read_judge_exits_when_it_reads_nothing(self):
        if cheat.try_read_judge():
            self.skipTest("this machine has readable judge paths")
        self.assertEqual(cheat.main(["read-judge", "0"]), 3)


if __name__ == "__main__":
    unittest.main()
