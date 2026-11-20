"""ref.generate against a fake ATOM (bench/tests/fake_atom.py) that carries the
reference image's `token_logprobs` patch: a hidden marker in the middle of the
answer and an answer that is only a stop token.

Needs numpy and tokenizers (bench's environment):
  cd bench && uv run --frozen python -m unittest discover -s ../research/amd/tests -p 'test_ref_*.py'
"""
import sys
import tempfile
import unittest
from pathlib import Path

AMD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AMD / "lib"))
try:
    import numpy as np  # noqa: F401
    import tokenizers  # noqa: F401
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("needs numpy and tokenizers (run it in bench's uv environment)")
sys.path.append(str(AMD.parents[1] / "bench" / "tests"))
import fake_atom  # noqa: E402
import ref  # noqa: E402

TMP = tempfile.TemporaryDirectory()
TOK = Path(TMP.name) / "tokenizer.json"
fake_atom.train_tokenizer(str(TOK))


def cfg_for(url: str) -> dict:
    return {"model": {"dir": TMP.name}, "ref": {"url": url, "served_model": "fake", "image": "fake",
                                                "gpu_memory_utilization": 0.5}}


class Fake:
    def __init__(self, **opts):
        self.fake = fake_atom.FakeAtom(str(TOK), **opts)
        self.srv, self.url = self.fake.serve()
        self.cfg = cfg_for(self.url)

    def greedy(self, ids, n):
        out, finish = self.fake.run(list(ids), n, ignore_eos=False)
        return [g for g, _ in out], finish

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def prompt(f: Fake) -> list[int]:
    return f.fake.prompt_ids("the quick brown fox")


class TokenLogprobsTest(unittest.TestCase):
    def test_hidden_marker_mid_answer(self):
        probe = Fake(token_logprobs=True)
        ids = prompt(probe)
        plain, _ = probe.greedy(ids, 6)
        probe.close()
        # the marker follows the answer's third token
        f = Fake(token_logprobs=True)
        f.fake.lm.triggers = {plain[2]: f.fake.think}
        try:
            want, finish = f.greedy(ids, 8)
            self.assertIn(f.fake.think, want[:-1], "the marker must sit inside the answer")
            got = ref.generate(f.cfg, ids, 8, 5)
            self.assertEqual(got["tokens"], want)
            self.assertEqual(len(got["logprobs"]), len(want))
            k = want.index(f.fake.think)
            lp = f.fake.lm.logprobs(ids + want[:k])
            self.assertAlmostEqual(got["logprobs"][k], float(lp[f.fake.think]), places=5)
            self.assertIn(str(f.fake.think), got["top_logprobs"][k])
            self.assertIsNone(got["stop"])
            # without the patch the marker has no entry: refused, not misplaced
            g = Fake(token_logprobs=False)
            g.fake.lm.triggers = dict(f.fake.lm.triggers)
            try:
                with self.assertRaises(ref.RefError):
                    ref.generate(g.cfg, ids, 8, 5)
            finally:
                g.close()
        finally:
            f.close()

    def test_stop_token_only(self):
        f = Fake(token_logprobs=True)
        ids = prompt(f)
        f.fake.lm.triggers = {ids[-1]: f.fake.eos}
        try:
            want, finish = f.greedy(ids, 8)
            self.assertEqual((want, finish), ([f.fake.eos], "stop"))
            got = ref.generate(f.cfg, ids, 8, 5)
            self.assertEqual(got["tokens"], [])
            self.assertEqual(got["finish_reason"], "stop")
            self.assertEqual(got["stop"]["id"], f.fake.eos)
            self.assertIn(str(f.fake.eos), got["stop"]["top"])
            self.assertAlmostEqual(got["stop"]["logprob"], float(f.fake.lm.logprobs(ids)[f.fake.eos]), places=5)
        finally:
            f.close()

    def test_stop_after_answer(self):
        probe = Fake(token_logprobs=True)
        ids = prompt(probe)
        plain, _ = probe.greedy(ids, 6)
        probe.close()
        f = Fake(token_logprobs=True)
        f.fake.lm.triggers = {plain[3]: f.fake.eos}
        try:
            want, finish = f.greedy(ids, 16)
            self.assertEqual(finish, "stop")
            got = ref.generate(f.cfg, ids, 16, 5)
            self.assertEqual(got["tokens"], want[:-1])
            self.assertEqual(got["stop"]["id"], f.fake.eos)
        finally:
            f.close()

    def test_entry_count_mismatch_refused(self):
        f = Fake(token_logprobs=True)
        ids = prompt(f)
        real_chat = f.fake.chat

        def short(body):
            st, doc = real_chat(body)
            doc["choices"][0]["token_logprobs"] = doc["choices"][0]["token_logprobs"][:-1]
            return st, doc
        f.fake.chat = short
        try:
            with self.assertRaises(ref.RefError):
                ref.generate(f.cfg, ids, 6, 5)
        finally:
            f.close()


if __name__ == "__main__":
    unittest.main()
