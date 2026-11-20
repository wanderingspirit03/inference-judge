"""Protocol 2 end to end over HTTP, without a GPU: the stand-in engine
(controller/standin/fake_engine.py) answers through engine.Server.ask, the
stand-in reference (fake_ref.py) walks along the answers through the real
ref.generate, and the speed comes from the same requests.  Also engines that
cheat in the ways the reviews named.  Stdlib only.
  python3 -m unittest discover -s research/amd/tests -p 'test_*.py'"""
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

AMD = Path(__file__).resolve().parents[1]
STANDIN = AMD / "controller" / "standin"
sys.path.insert(0, str(AMD / "lib"))
sys.path.insert(0, str(STANDIN))
import answers  # noqa: E402
import engine as E  # noqa: E402
import fake_engine  # noqa: E402
import fake_ref  # noqa: E402
import ref as R  # noqa: E402
import tokmap  # noqa: E402
import toylm  # noqa: E402

TMP = tempfile.TemporaryDirectory()
fake_ref.write_tokenizer(TMP.name)


def serve(handler) -> tuple[ThreadingHTTPServer, int]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


REF_SRV, REF_PORT = serve(fake_ref.H)
CFG = {"model": {"dir": TMP.name}, "engine": {"port": 0, "request_timeout_s": 30, "health_path": "/v1/models"},
       "check": {"engine_top": 5},
       "ref": {"url": f"http://127.0.0.1:{REF_PORT}", "served_model": "toy", "image": "fake", "top_logprobs": 20,
               "gpu_memory_utilization": 0.5},
       "limits": {"min_agreement": 0.95, "tie_eps": 0.125, "margin": 0.03, "max_partings_per_item": 40,
                  "compare_stop": True}}
STOPS = tokmap.load(TMP.name).special_ids


class Sess:
    """ref.Session without the start: the stand-in reference is already up."""
    def __init__(self):
        self.calls = 0

    def answer(self, ids, mt):
        self.calls += 1
        return R.generate(CFG, ids, mt, 20)


def server_at(port: int) -> E.Server:
    s = E.Server({**CFG, "engine": {**CFG["engine"], "port": port}}, None, Path("."), Path("."), lambda m: None)
    s.model = "toy"
    return s


ITEMS = [{"id": f"p{i}", "ids": toylm.encode(t), "max_tokens": 32} for i, t in enumerate([
    "The lighthouse keeper had not seen a ship in eleven days.",
    "Write a haiku about rain ~",                      # its answer stops at once ("~~" then stop)
    "count: 1 2 3 4 5 6 7",
    "def add(a, b):\n    return"])]


def judge(handler, items=ITEMS):
    srv, port = serve(handler)
    try:
        e = server_at(port)
        outs = answers.engine_answers(CFG, e, items, lambda m: None)
        rows = answers.walk(CFG, Sess(), items, outs, answers.limits(CFG), lambda m: None)
        ok, why, agg = answers.compare.judge([r["item"] for r in rows], answers.limits(CFG))
        return ok, why, agg, answers.speed(items, outs, STOPS) if ok else None, outs, rows
    finally:
        srv.shutdown()
        srv.server_close()


class Honest(fake_engine.H):
    pass


class Damaged(fake_engine.H):
    def do_POST(self):
        fake_engine.DAMAGE = 3.0
        try:
            super().do_POST()
        finally:
            fake_engine.DAMAGE = 0.0


def rewrite(fn):
    """An engine whose answers fake_engine computes and fn(body, doc) changes."""
    class H(fake_engine.H):
        def _json(self, code, obj):
            if code == 200 and "choices" in obj:
                obj = fn(self._body, obj)
            super()._json(code, obj)

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            self._body = json.loads(raw)
            import io
            self.rfile = io.BytesIO(raw)
            super().do_POST()
    return H


class ProtocolTest(unittest.TestCase):
    def test_honest_engine_passes_and_its_speed_counts_checked_tokens(self):
        ok, why, agg, sp, outs, rows = judge(Honest)
        self.assertTrue(ok, why)
        self.assertEqual(agg["agreement"], 1.0)
        self.assertEqual(outs[1]["finish"], "stop")
        self.assertEqual(sp["requests"], len(ITEMS))
        self.assertEqual(sp["tokens"], sum(len(o["tokens"]) - (o["finish"] == "stop") for o in outs))
        self.assertEqual(sp["tokens"], agg["positions"] - 1)       # the stop position is checked, not counted as a token
        self.assertEqual(len(sp["per_request"]), len(ITEMS))

    def test_damaged_engine_fails(self):
        ok, why, agg, *_ = judge(Damaged)
        self.assertFalse(ok)
        self.assertIn("agreement with stock", why[-1])

    def test_the_walk_stops_once_fail_is_decided(self):
        items = ITEMS * 3                       # 12 answers, as a draw of 12
        lim = answers.limits(CFG)
        for handler, stops in ((Honest, False), (Damaged, True)):
            srv, port = serve(handler)
            try:
                outs = answers.engine_answers(CFG, server_at(port), items, lambda m: None)
            finally:
                srv.shutdown()
                srv.server_close()
            full = answers.walk(CFG, Sess(), items, outs, lim, lambda m: None)
            stopped = []
            rows = answers.walk(CFG, Sess(), items, outs, lim, lambda m: None, stopped=stopped)
            ok_full = answers.compare.judge([r["item"] for r in full], lim)[0]
            ok_part = answers.compare.judge([r["item"] for r in rows], lim)[0]
            self.assertEqual(ok_full, ok_part)                 # the same result either way
            self.assertEqual(bool(stopped), stops, stopped)
            if stops:
                self.assertFalse(ok_part)
                self.assertLess(len(rows), len(items))
                self.assertIn("under the limit", stopped[0])
            else:
                self.assertEqual(len(rows), len(items))
        # the agreement bound is not used while the quality bar replaces the limit; a failed answer still stops
        q = {**CFG, "quality": {"bar": {"min_topk_share": {"1": 0.9}, "max_mean_gap": 1.0, "min_task_delta_lb": -0.1}}}
        bad = {"item": answers.compare.ItemResult("p0", positions=33), "id": "p0"}
        self.assertIsNone(answers.decided_fail(q, [bad], items[1:], lim))
        self.assertIsNotNone(answers.decided_fail(CFG, [bad], items[1:], lim))
        failed = {"item": answers.compare.ItemResult("p0", ok=False, reason="x"), "id": "p0"}
        self.assertIn("p0 failed", answers.decided_fail(q, [failed], items[1:], lim))

    def test_with_the_bar_set_the_walk_stops_once_a_topk_share_is_out_of_reach(self):
        items = ITEMS * 3
        lim = answers.limits(CFG)
        q = {**CFG, "quality": {"bar": {"min_topk_share": {"1": 0.98, "5": None}, "max_mean_gap": 10.0,
                                        "min_task_delta_lb": -1.0}, "bar_replaces_agreement": True, "topk": [1, 5]}}
        qb = {"tasks": 4, "engine_correct": 4, "reference_correct": 4, "delta": 0.0, "delta_lb95": 0.0}
        for handler, stops in ((Honest, False), (Damaged, True)):
            srv, port = serve(handler)
            try:
                outs = answers.engine_answers(CFG, server_at(port), items, lambda m: None)
            finally:
                srv.shutdown()
                srv.server_close()
            full = answers.walk(q, Sess(), items, outs, lim, lambda m: None)
            stopped = []
            rows = answers.walk(q, Sess(), items, outs, lim, lambda m: None, stopped=stopped)
            ok_full = answers.decide(q, [r["item"] for r in full], lim, qb)[0]
            self.assertEqual(bool(stopped), stops, stopped)
            self.assertEqual(ok_full, not stops)              # the full walk gives the same result
            if stops:
                self.assertLess(len(rows), len(items))
                self.assertIn("top-1 share", stopped[0])
                self.assertIn("under the bar's limit 0.98", stopped[0])
            else:
                self.assertEqual(len(rows), len(items))
        # the bound itself: 30 of 33 in the top 1, one answer of 32 tokens left -> at most 63/66 = 0.9545
        it = answers.compare.ItemResult("p0", positions=33)
        it.ranks = [(1, 0.0, False)] * 30 + [(3, 1.0, False)] * 3
        row = {"item": it, "id": "p0"}
        self.assertIn("at most 0.9545", answers.decided_fail(q, [row], ITEMS[:1], lim))
        q5 = {**q, "quality": {**q["quality"], "bar": {**q["quality"]["bar"], "min_topk_share": {"5": 0.95}}}}
        self.assertIsNone(answers.decided_fail(q5, [row], ITEMS[:1], lim))      # all 33 in the top 5
        self.assertIsNone(answers.decided_fail(q, [row], [], lim))             # nothing left: decide() judges
        it.cut = True
        self.assertIn("cut at the parting limit", answers.decided_fail(q5, [row], ITEMS[:1], lim))

    def test_tokenize_in_threads_keeps_the_order(self):
        import contextlib
        import importlib.machinery
        import importlib.util
        loader = importlib.machinery.SourceFileLoader("ij_submit_tok", str(AMD / "submit"))
        sub = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
        loader.exec_module(sub)
        d = Path(tempfile.mkdtemp())
        rows = [{"id": f"t{i}", "messages": [{"role": "user", "content": f"task number {i} " * (1 + i % 5)}],
                 "answer": str(i), "kind": "math"} for i in range(40)]
        (d / "in.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        cfg = CFG

        class S(R.Session):
            def _ensure(self):
                self.started = True              # the stand-in reference is up already

            def close(self):
                pass
        with mock.patch.object(sub.common, "hold", lambda *a, **k: contextlib.nullcontext(None)), \
                mock.patch.object(sub.R, "Session", lambda c, f, log: S(cfg, f, log)), \
                mock.patch.object(sub.R, "cache_dir", lambda c: d / "cache"):
            (d / "cache").mkdir()
            self.assertEqual(sub.run_tokenize(cfg, d, str(d / "in.jsonl"), str(d / "out.jsonl")), 0)
        out = [json.loads(x) for x in (d / "out.jsonl").read_text().splitlines()]
        self.assertEqual([o["id"] for o in out], [r["id"] for r in rows])
        for r, o in zip(rows, out):
            self.assertEqual(o["prompt_token_ids"], R.tokenize(cfg, messages=r["messages"]))
            self.assertEqual(o["answer"], r["answer"])

    def test_fabricated_token_count_fails(self):
        """Claims max_tokens tokens but computes 8 and pads with a repeat."""
        def fab(body, doc):
            lp = doc["choices"][0]["logprobs"]
            n = body["max_tokens"]
            if len(lp["tokens"]) < 8:
                return doc
            for k in ("tokens", "token_logprobs", "top_logprobs"):
                lp[k] = (lp[k][:8] + [lp[k][7]] * n)[:n]
            doc["choices"][0]["finish_reason"] = "length"
            doc["usage"]["completion_tokens"] = n
            return doc
        ok, why, agg, sp, outs, rows = judge(rewrite(fab))
        self.assertFalse(ok)
        self.assertTrue(all("error" not in o for o in outs))
        self.assertIn("agreement with stock", why[-1])
        self.assertLess(agg["agreement"], 0.9)

    def test_usage_alone_is_not_believed(self):
        """Few tokens returned, usage claims many: the request is refused."""
        def lie(body, doc):
            doc["usage"]["completion_tokens"] = body["max_tokens"] * 10
            return doc
        ok, why, agg, sp, outs, rows = judge(rewrite(lie))
        self.assertFalse(ok)
        self.assertIn("usage says", why[0])

    def test_short_answer_without_stop_fails(self):
        def cut(body, doc):
            lp = doc["choices"][0]["logprobs"]
            for k in ("tokens", "token_logprobs", "top_logprobs"):
                lp[k] = lp[k][:4]
            doc["choices"][0]["finish_reason"] = "length"
            doc["usage"]["completion_tokens"] = 4
            return doc
        ok, why, *_ = judge(rewrite(cut))
        self.assertFalse(ok)
        self.assertIn("did not stop", " ".join(why))

    def test_early_stop_is_a_parting(self):
        def stop_early(body, doc):
            lp = doc["choices"][0]["logprobs"]
            for k in ("tokens", "token_logprobs", "top_logprobs"):
                lp[k] = lp[k][:10]
            lp["tokens"][-1] = f"token_id:{toylm.EOS}"
            doc["choices"][0]["finish_reason"] = "stop"
            doc["usage"]["completion_tokens"] = 10
            return doc
        ok, why, agg, *_ = judge(rewrite(stop_early))
        self.assertFalse(ok)
        self.assertEqual(agg["partings"], 3)                # the three that do not stop on their own

    def test_requests_have_one_shape(self):
        seen = []

        def record(body, doc):
            seen.append(sorted(body))
            return doc
        srv, port = serve(rewrite(record))
        try:
            e = server_at(port)
            answers.warm_up(e, ITEMS, [toylm.encode("warm-up text, never judged. " * 4)], lambda m: None)
            answers.engine_answers(CFG, e, ITEMS, lambda m: None)
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertEqual(len({tuple(s) for s in seen}), 1, seen)
        self.assertEqual(len(seen), 2 * len({len(i["ids"]) for i in ITEMS}) + len(ITEMS))

    def test_warm_up_never_sends_a_judged_prompt(self):
        shapes = answers.warm_shapes(ITEMS, [toylm.encode("x" * 7)])
        self.assertEqual(sorted(len(s) for s in shapes), sorted({len(i["ids"]) for i in ITEMS}))
        judged = {tuple(i["ids"]) for i in ITEMS}
        self.assertFalse(any(tuple(s) in judged for s in shapes))


if __name__ == "__main__":
    unittest.main()
