"""quality2: the v2 quality guard against a fixed reference R.

Two adapters sit at the CandidateClient seam: FakeCandidate (in memory) for
the gate's behaviour, and quality2.HttpCandidate against a tiny HTTP server
in this file for the wire format. Both are backed by ToyModel, a
deterministic next-token distribution keyed by the whole context, so the
in-memory and HTTP paths see the same model.
"""
from __future__ import annotations

import hashlib
import json
import math
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

import quality2
from quality2 import LogprobsUnavailable, Reference

VOCAB = [f"w{i}" for i in range(40)]
MAX_TOKENS = 48
PROMPTS = quality2.MOCK_PROMPTS[:12]
TASKS = quality2.MOCK_TASKS


# ------------------------------------------------------------- the model ---
def _rng(*parts: str) -> np.random.Generator:
    h = hashlib.sha256("|".join(parts).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


class ToyModel:
    """logits = base(context) + eps * noise(context, salt). A small eps with a
    per-launch salt is launch-to-launch nondeterminism; a large eps is a lossy
    quantisation."""

    def __init__(self, salt: str = "r0", eps: float = 0.0) -> None:
        self.salt, self.eps = salt, eps

    def top(self, context: list[str], k: int = 20) -> dict[str, float]:
        ctx = " ".join(context)
        logits = _rng("base", ctx).normal(0.0, 2.0, len(VOCAB))
        if self.eps:
            logits = logits + self.eps * _rng(self.salt, ctx).normal(0, 1, len(VOCAB))
        lp = logits - (logits.max() + math.log(np.exp(logits - logits.max()).sum()))
        order = np.argsort(-lp)[:k]
        return {VOCAB[i]: float(lp[i]) for i in order}

    def greedy(self, prompt: str, n: int) -> list[str]:
        ctx, out = prompt.split(), []
        for _ in range(n):
            top = self.top(ctx + out)
            out.append(max(top, key=top.get))
        return out


def honest_reply(task: dict) -> dict:
    if task["kind"] == "tool":
        return {"text": "", "tool_calls": [dict(task["answer"])]}
    return {"text": f"The answer is {task['answer']}.", "tool_calls": []}


def wrong_reply(task: dict) -> dict:
    if task["kind"] == "tool":
        return {"text": "", "tool_calls": [{"name": "cancel_booking",
                                            "arguments": {}}]}
    return {"text": "The answer is 999." if task["kind"] == "math"
            else "The answer is D.", "tool_calls": []}


class FakeCandidate:
    """In-memory adapter for quality2.CandidateClient."""

    def __init__(self, model: ToyModel | None = None, *, behaviour: str = "honest",
                 logprobs: bool = True, wrong: tuple[str, ...] = ()) -> None:
        self.model = model or ToyModel()
        self.behaviour, self.logprobs, self.wrong = behaviour, logprobs, set(wrong)

    def greedy(self, prompt: str, max_tokens: int):
        if self.behaviour == "junk":
            toks = ["a"] * max_tokens
        else:
            toks = self.model.greedy(prompt, max_tokens)
            if self.behaviour == "truncate":
                toks = toks[:max_tokens // 2]
        return toks, " ".join(toks)

    def teacher_force(self, prompt, text, tokens, k):
        if not self.logprobs:
            raise LogprobsUnavailable("fake: no logprobs")
        ctx = prompt.split()
        return [self.model.top(ctx + tokens[:j], k) for j in range(len(tokens))]

    def answer(self, task):
        return wrong_reply(task) if task["id"] in self.wrong else honest_reply(task)


# ------------------------------------------------------------- reference ---
def build_reference(tmp_path, name="ref", launch_eps=0.02, n=3,
                    r0_wrong=()) -> Reference:
    l0 = quality2.collect(FakeCandidate(ToyModel("r0", launch_eps),
                                        wrong=r0_wrong),
                          PROMPTS, TASKS, max_tokens=MAX_TOKENS)
    rest = [quality2.collect(FakeCandidate(ToyModel(f"r{k}", launch_eps)),
                             PROMPTS, TASKS, anchor=l0, max_tokens=MAX_TOKENS)
            for k in range(1, n)]
    return quality2.make_reference([l0] + rest, tmp_path / name,
                                   prompts=PROMPTS, tasks=TASKS,
                                   target_id="cpu-mock")


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    return build_reference(tmp_path_factory.mktemp("q2"))


def test_reference_artefacts_are_complete_and_read_only(ref):
    names = {p.name for p in ref.dir.iterdir()}
    assert names == set(quality2.REFERENCE_FILES) | {"manifest.json"}
    for p in ref.dir.iterdir():
        assert not p.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    m = ref.manifest
    assert m["target_id"] == "cpu-mock" and m["n_launches"] == 3
    assert m["source"] == "launch" and m["tokenizer"] == "ws"
    assert len(ref.greedy[PROMPTS[0]["id"]]) == 3
    assert len(ref.logprobs[PROMPTS[0]["id"]]) == MAX_TOKENS
    for key in ("kl_mean", "kl_p99", "top1", "divergence_median",
                "task_acc_mean", "task_acc_sd"):
        assert key in ref.floors
    assert ref.floors["task_draws"] == 3
    assert len(ref.hash) == 64


def test_a_reference_is_written_once(ref):
    one = quality2.collect(FakeCandidate(), PROMPTS[:1], TASKS[:1], max_tokens=4)
    with pytest.raises(FileExistsError):
        quality2.make_reference([one], ref.dir, prompts=PROMPTS[:1],
                                tasks=TASKS[:1], target_id="x")


def test_a_tampered_reference_is_refused(tmp_path):
    r = build_reference(tmp_path, n=2)
    floors = r.dir / "floors.json"
    floors.chmod(0o644)
    data = json.loads(floors.read_text())
    data["kl_mean"] = 10.0                          # loosen every threshold
    floors.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="floors.json"):
        Reference.load(r.dir)


def test_launches_must_be_anchored_to_launch_zero(tmp_path):
    a = quality2.collect(FakeCandidate(), PROMPTS[:2], TASKS[:1], max_tokens=4)
    b = quality2.collect(FakeCandidate(), PROMPTS[:2], TASKS[:1], max_tokens=4)
    with pytest.raises(ValueError, match="anchored"):
        quality2.make_reference([a, b], tmp_path / "r", prompts=PROMPTS[:2],
                                tasks=TASKS[:1], target_id="x")


# ------------------------------------------------------------- thresholds --
def test_identical_launches_give_the_starting_thresholds(tmp_path):
    r = build_reference(tmp_path, launch_eps=0.0)
    th = r.thresholds()
    assert th["kl_mean_max"] == quality2.KL_MEAN_START
    assert th["kl_p99_max"] == quality2.KL_P99_START
    assert th["top1_min"] == quality2.TOP1_START
    assert th["divergence_min"] == pytest.approx(0.8 * MAX_TOKENS)
    assert th["task_lb_min"] == -0.015


def test_noisy_launches_raise_thresholds_to_three_times_the_floor(tmp_path):
    r = build_reference(tmp_path, launch_eps=0.6)
    f, th = r.floors, r.thresholds()
    assert f["kl_mean"] * 3 > quality2.KL_MEAN_START
    assert th["kl_mean_max"] == pytest.approx(3 * f["kl_mean"])
    assert th["top1_min"] == pytest.approx(min(0.97, 1 - 3 * (1 - f["top1"])))
    assert th["divergence_min"] == pytest.approx(0.8 * f["divergence_median"])
    assert th["divergence_min"] < 0.8 * MAX_TOKENS
    assert r.thresholds({"kl_mean_max": 0.5})["kl_mean_max"] == 0.5


# ------------------------------------------------------------------ gate ---
def test_an_honest_candidate_passes(ref):
    res = quality2.gate(FakeCandidate(ToyModel("c1", 0.02)), ref).to_dict()
    assert res["passed"] is True, res["reasons"]
    assert set(res["checks"]) == {"kl", "top1", "divergence", "tasks", "sanity"}
    assert all(c["status"] == "pass" for c in res["checks"].values())
    assert res["checks"]["kl"]["mode"] == "teacher_forced"
    assert res["reasons"] == []
    assert res["reference_hash"] == ref.hash
    assert "kl_mean_max" in res["thresholds"]


class _SlowConcurrentCandidate(FakeCandidate):
    """FakeCandidate that asks for concurrent calls and records the peak."""
    concurrency = 8

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.lock, self.active, self.peak = threading.Lock(), 0, 0

    def _enter(self):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.01)
        with self.lock:
            self.active -= 1

    def greedy(self, prompt, max_tokens):
        self._enter()
        return super().greedy(prompt, max_tokens)

    def teacher_force(self, prompt, text, tokens, k):
        self._enter()
        return super().teacher_force(prompt, text, tokens, k)


def test_concurrent_requests_change_the_wall_time_not_the_result(ref, tmp_path):
    seq = quality2.collect(FakeCandidate(ToyModel("c1", 0.02)), PROMPTS, TASKS,
                           max_tokens=MAX_TOKENS)
    fast = _SlowConcurrentCandidate(ToyModel("c1", 0.02))
    par = quality2.collect(fast, PROMPTS, TASKS, max_tokens=MAX_TOKENS)
    assert fast.peak > 1
    assert (par.greedy, par.logprobs, par.task_answers) == (
        seq.greedy, seq.logprobs, seq.task_answers)
    assert list(par.greedy) == [p["id"] for p in PROMPTS]
    g_seq = quality2.gate(FakeCandidate(ToyModel("c1", 0.02)), ref).to_dict()
    g_par = quality2.gate(_SlowConcurrentCandidate(ToyModel("c1", 0.02)),
                          ref).to_dict()
    assert g_par == g_seq


def test_a_lossy_quantisation_fails_kl_top1_and_divergence(ref):
    res = quality2.gate(FakeCandidate(ToyModel("q", 0.6)), ref)
    assert res.passed is False
    for name in ("kl", "top1", "divergence"):
        assert res.checks[name]["status"] == "fail", (name, res.checks[name])
    assert any(r.startswith("kl:") for r in res.reasons)


def test_fast_junk_fails_sanity_and_divergence(ref):
    res = quality2.gate(FakeCandidate(behaviour="junk"), ref)
    assert res.passed is False
    assert res.checks["sanity"]["status"] == "fail"
    assert res.checks["sanity"]["frac_outside"] == 1.0
    assert res.checks["divergence"]["status"] == "fail"


def test_sanity_does_not_flag_text_that_is_merely_different(ref):
    """A small R makes the 1st-99th band nearly a point; ordinary text that
    differs from R's must not fail sanity (other checks judge closeness)."""
    res = quality2.gate(FakeCandidate(ToyModel("other", 0.15)), ref)
    assert res.checks["sanity"]["status"] == "pass", res.checks["sanity"]


def test_a_truncating_candidate_fails_divergence(ref):
    res = quality2.gate(FakeCandidate(ToyModel("c1", 0.02), behaviour="truncate"),
                        ref)
    assert res.checks["divergence"]["status"] == "fail"
    assert res.passed is False


def test_tasks_fail_when_the_candidate_gets_items_wrong(ref):
    res = quality2.gate(FakeCandidate(ToyModel("c1", 0.02),
                                      wrong=("m-math-1", "m-tool-3")), ref)
    t = res.checks["tasks"]
    assert t["status"] == "fail"
    assert t["delta"] == pytest.approx(-2 / len(TASKS))
    assert set(t["wrong"]) == {"m-math-1", "m-tool-3"}
    assert res.passed is False


def test_tasks_are_paired_against_r_per_item_mean(tmp_path):
    # R itself got m-mc-2 wrong on its launch-0 draw: R's item mean is 2/3.
    r = build_reference(tmp_path, r0_wrong=("m-mc-2",))
    assert r.r_item_accuracy()["m-mc-2"] == pytest.approx(2 / 3)
    same_miss = quality2.gate(FakeCandidate(ToyModel("c", 0.02),
                                            wrong=("m-mc-2",)), r)
    # one item, 1/3 below R's mean: lower bound is -1/3 / 12 < -1.5 points
    assert same_miss.checks["tasks"]["lower_bound"] < -0.015
    assert quality2.gate(FakeCandidate(ToyModel("c", 0.02)), r
                         ).checks["tasks"]["status"] == "pass"


# ------------------------------------------------- no logprobs: fail closed --
def test_no_logprobs_fails_closed_by_default(ref):
    res = quality2.gate(FakeCandidate(ToyModel("c1", 0.02), logprobs=False), ref)
    assert res.checks["top1"]["status"] == "unavailable"
    assert res.checks["kl"]["mode"] == "fallback_r_scored"
    assert res.checks["kl"]["status"] == "pass"
    assert res.passed is False
    assert any("unavailable" in r and "failing closed" in r for r in res.reasons)


def test_no_logprobs_passes_when_the_target_allows_it(ref):
    res = quality2.gate(FakeCandidate(ToyModel("c1", 0.02), logprobs=False), ref,
                        allow_unavailable=True)
    assert res.passed is True, res.reasons


def test_no_logprobs_fallback_still_catches_a_lossy_candidate(ref):
    res = quality2.gate(FakeCandidate(ToyModel("q", 0.6), logprobs=False), ref,
                        allow_unavailable=True)
    assert res.passed is False
    assert res.checks["kl"]["status"] == "fail"


def test_a_candidate_that_errors_fails_closed(ref):
    class Broken(FakeCandidate):
        def greedy(self, prompt, max_tokens):
            raise ConnectionError("down")

        def answer(self, task):
            raise ConnectionError("down")
    res = quality2.gate(Broken(), ref)
    assert res.passed is False
    assert res.checks["divergence"]["status"] == "fail"
    assert res.checks["tasks"]["status"] == "fail"


# ---------------------------------------------------------- answer parsing --
@pytest.mark.parametrize("kind,gold,text,ok", [
    ("math", 1200, "That comes to 1,200 in total.", True),
    ("math", 7.5, "It is 7.5", True),
    ("math", 42, "Maybe 41, no wait, 43.", False),
    ("math", 42, "I do not know.", False),
    ("mc", "B", "The answer is B.", True),
    ("mc", "B", "B", True),
    ("mc", "B", "A good question! The answer is (C).", False),
    ("mc", "C", "C) 29 is prime.", True),
])
def test_answer_extraction(kind, gold, text, ok):
    task = {"id": "t", "kind": kind, "answer": gold, "prompt": ""}
    ans = quality2.extract_answer(task, {"text": text, "tool_calls": []})
    assert quality2.is_correct(task, ans) is ok


def test_tool_answers_need_exactly_one_matching_call():
    task = TASKS[10]
    good = {"text": "", "tool_calls": [{"name": "create_reservation",
                                        "arguments": {"name": "okafor",
                                                      "party_size": "4",
                                                      "time": "19:30"}}]}
    two = {"text": "", "tool_calls": good["tool_calls"] * 2}
    assert quality2.is_correct(task, quality2.extract_answer(task, good))
    assert not quality2.is_correct(task, quality2.extract_answer(task, two))


def test_mock_task_set_shape():
    kinds = {t["kind"] for t in quality2.MOCK_TASKS}
    assert kinds == {"math", "mc", "tool"}
    assert any(t["lang"] != "en" for t in quality2.MOCK_TASKS)
    for t in quality2.MOCK_TASKS:
        assert quality2.is_correct(t, quality2.extract_answer(t, honest_reply(t)))
        if t["kind"] == "tool":           # arguments reachable from the prompt
            for v in t["answer"]["arguments"].values():
                assert str(v) in t["prompt"]
    assert len(quality2.MOCK_PROMPTS) == 32


# --------------------------------------------------- the HTTP adapter -------
class _Handler(BaseHTTPRequestHandler):
    model: ToyModel = ToyModel("r0", 0.0)
    shape = "openai"          # openai | vllm | misaligned | no_logprobs

    def log_message(self, *a):            # quiet
        pass

    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):                    # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls = type(self)
        if cls.compile_s:                 # a shape compiled on first use
            wait, cls.compile_s = cls.compile_s, 0.0
            time.sleep(wait)
        if self.path.endswith("/chat/completions"):
            task = next(t for t in TASKS
                        if t["prompt"] == body["messages"][-1]["content"])
            reply = honest_reply(task)
            msg = {"role": "assistant", "content": reply["text"] or None}
            if reply["tool_calls"] and body.get("tools"):
                msg["tool_calls"] = [{"id": "c1", "type": "function", "function": {
                    "name": c["name"], "arguments": json.dumps(c["arguments"])}}
                    for c in reply["tool_calls"]]
            return self._send(200, {"choices": [{"message": msg}]})
        prompt = body["prompt"]
        if not body.get("echo"):
            toks = cls.model.greedy(prompt, body["max_tokens"])
            return self._send(200, {"choices": [{"text": " " + " ".join(toks),
                                                 "finish_reason": "length"}]})
        if cls.shape == "no_logprobs":
            return self._send(400, {"error": {"message": "echo not supported"}})
        words = prompt.split()
        tops = [None] + [cls.model.top(words[:i]) for i in range(1, len(words))]
        gen = cls.model.greedy(prompt, 1)[0]
        if cls.shape == "vllm":
            plp = [None] + [{str(n): {"logprob": lp, "decoded_token": t}
                             for n, (t, lp) in enumerate(top.items())}
                            for top in tops[1:]]
            return self._send(200, {"choices": [{"text": gen,
                                                 "prompt_logprobs": plp}]})
        offsets, pos = [], 0
        for w in words:
            pos = prompt.index(w, pos)
            offsets.append(pos)
            pos += len(w)
        toks = list(words)
        if cls.shape == "misaligned":
            toks[-3] = "w999"
        return self._send(200, {"choices": [{"text": prompt + " " + gen,
            "logprobs": {"tokens": toks + [gen],
                         "top_logprobs": tops + [cls.model.top(words)],
                         "text_offset": offsets + [len(prompt) + 1]}}]})


@pytest.fixture()
def http_server():
    handler = type("H", (_Handler,), {"model": ToyModel("r0", 0.0),
                                      "shape": "openai", "compile_s": 0.0})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield handler, f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_http_adapter_round_trip_through_the_cli(http_server, tmp_path):
    handler, url = http_server
    ref_dir, out = tmp_path / "ref", tmp_path / "gate.json"
    assert quality2.main(["make-reference", "--base-url", url, "--out",
                          str(ref_dir), "--replicate", "2",
                          "--max-tokens", "16"]) == 0
    r = Reference.load(ref_dir)
    assert r.manifest["source"] == "mock-replicated"
    assert r.floors["kl_mean"] == pytest.approx(0.0, abs=1e-12)
    assert quality2.main(["gate", "--base-url", url, "--reference",
                          str(ref_dir), "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["passed"] is True, res["reasons"]
    assert res["checks"]["tasks"]["c_acc"] == 1.0
    assert res["checks"]["divergence"]["median"] == 16

    handler.model = ToyModel("q", 0.6)                 # now quantised
    assert quality2.main(["gate", "--base-url", url, "--reference",
                          str(ref_dir), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["passed"] is False


def _small_ref(tmp_path) -> Reference:
    l0 = quality2.collect(FakeCandidate(ToyModel("r0", 0.0)), PROMPTS[:4],
                          TASKS[:3], max_tokens=12)
    return quality2.make_reference([l0], tmp_path / "small", prompts=PROMPTS[:4],
                                   tasks=TASKS[:3], target_id="cpu-mock",
                                   source="mock-replicated")


def test_http_adapter_reads_vllm_prompt_logprobs(http_server, tmp_path):
    handler, url = http_server
    handler.shape = "vllm"
    res = quality2.gate(quality2.HttpCandidate(url), _small_ref(tmp_path))
    assert res.checks["kl"]["status"] == "pass", res.checks["kl"]
    assert res.checks["kl"]["mean"] == pytest.approx(0.0, abs=1e-9)


def test_http_adapter_fails_closed_on_misaligned_echo(http_server, tmp_path):
    handler, url = http_server
    handler.shape = "misaligned"
    res = quality2.gate(quality2.HttpCandidate(url), _small_ref(tmp_path))
    assert res.checks["kl"]["status"] == "fail"
    assert "alignment" in res.checks["kl"]["error"]
    assert res.passed is False


def test_http_adapter_without_logprobs_falls_back(http_server, tmp_path):
    handler, url = http_server
    handler.shape = "no_logprobs"
    res = quality2.gate(quality2.HttpCandidate(url), _small_ref(tmp_path))
    assert res.checks["kl"]["mode"] == "fallback_r_scored"
    assert res.checks["top1"]["status"] == "unavailable"
    assert res.passed is False


def test_the_mock_task_set_is_refused_for_a_real_engine(tmp_path):
    target = tmp_path / "t.yaml"
    target.write_text(
        "id: real-tpu\n"
        "accelerator: {kind: tpu, chips: 8, hosts: 1}\n"
        "model: {served_name: glm, revision: '0', local_dir: '', tokenizer: ws}\n"
        "engine: {name: sglang-jax, image: '', docker_flags: []}\n"
        "metric: {kind: tps, isl: 2048, osl: 512, concurrency: [16], "
        "slo: {p99_tpot_ms: 50, p95_ttft_ms: 2000}, ramp_s: 30, window_s: 180}\n"
        "budgets: {build_s: 1200, compile_s: 600, health_s: 120, run_s: 2400}\n"
        "reference: {dir: reference/real-tpu}\n")
    with pytest.raises(SystemExit, match="non-mock target"):
        quality2.main(["collect", "--base-url", "http://127.0.0.1:9/v1",
                       "--target", str(target), "--out", str(tmp_path / "l0")])


REAL_TARGET = (
    "id: real-tpu\n"
    "accelerator: {kind: tpu, chips: 8, hosts: 1}\n"
    "model: {served_name: glm, revision: '0', local_dir: '', tokenizer: ws}\n"
    "engine: {name: sglang-jax, image: '', docker_flags: []}\n"
    "metric: {kind: tps, isl: 2048, osl: 512, concurrency: [16], "
    "slo: {p99_tpot_ms: 50, p95_ttft_ms: 2000}, ramp_s: 30, window_s: 180}\n"
    "budgets: {build_s: 1200, compile_s: 600, health_s: 120, run_s: 2400}\n"
    "reference: {dir: reference/real-tpu}\n")


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_a_request_waits_as_long_as_the_cli_is_told(http_server, tmp_path):
    """A quality prompt can land on a shape the server has not compiled: the
    request then takes as long as the compile, and the caller says how long
    that may be."""
    import httpx
    handler, url = http_server
    few = ["--prompts", _jsonl(tmp_path / "p.jsonl", PROMPTS[:2]),
           "--tasks", _jsonl(tmp_path / "t.jsonl", TASKS[:2]), "--max-tokens", "8"]
    handler.compile_s = 1.0
    with pytest.raises(httpx.TimeoutException):
        quality2.main(["collect", "--base-url", url, "--request-timeout", "0.2",
                       "--out", str(tmp_path / "short"), *few])
    handler.compile_s = 1.0
    assert quality2.main(["collect", "--base-url", url, "--request-timeout", "10",
                          "--out", str(tmp_path / "long"), *few]) == 0


def test_a_reference_from_cold_launches_through_the_cli(http_server, tmp_path):
    """The reference of a real target: launch 0 collected alone, every later
    launch anchored to it, assembled without a server."""
    handler, url = http_server
    dirs = [str(tmp_path / f"l{k}") for k in range(3)]
    for k, d in enumerate(dirs):
        anchor = ["--anchor", dirs[0]] if k else []
        assert quality2.main(["collect", "--base-url", url, "--out", d,
                              "--max-tokens", "16", *anchor]) == 0
    ref_dir = tmp_path / "ref"
    assert quality2.main(["make-reference", "--launches-from", ",".join(dirs),
                          "--out", str(ref_dir)]) == 0
    r = Reference.load(ref_dir)
    assert r.manifest["source"] == "launch" and r.manifest["n_launches"] == 3
    assert r.max_tokens == 16
    assert r.floors["task_draws"] == 3                  # one draw per launch
    out = tmp_path / "gate.json"
    assert quality2.main(["gate", "--base-url", url, "--reference",
                          str(ref_dir), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["passed"] is True


def test_one_server_measured_again_is_no_reference_for_a_real_engine(tmp_path):
    """Its floors hold no launch-to-launch noise (acceptable for the mock):
    refused before a request is sent."""
    target = tmp_path / "t.yaml"
    target.write_text(REAL_TARGET)
    with pytest.raises(SystemExit, match="--launches-from"):
        quality2.main(["make-reference", "--base-url", "http://127.0.0.1:9/v1",
                       "--target", str(target), "--out", str(tmp_path / "ref"),
                       "--prompts", _jsonl(tmp_path / "p.jsonl", PROMPTS[:2]),
                       "--tasks", _jsonl(tmp_path / "t.jsonl", TASKS[:2])])
    assert not (tmp_path / "ref").exists()
