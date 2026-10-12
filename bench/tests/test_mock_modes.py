"""The mock server's --behaviour modes and its toy-LM logprobs (DESIGN 2.3).

These talk to the mock over HTTP, the way the harness and quality2 do."""
from __future__ import annotations

import json
import math
import socket
import time

import httpx
import pytest

from conftest import MockServer

OSL = 24


@pytest.fixture(scope="module")
def servers():
    """One mock per behaviour, started on first use."""
    started: dict[str, MockServer] = {}

    def get(behaviour: str, **flags: object) -> MockServer:
        key = behaviour + json.dumps(flags, sort_keys=True)
        if key not in started:
            started[key] = MockServer(behaviour=behaviour, ttft_ms=5,
                                      tpot_ms=1, **flags)
        return started[key]

    yield get
    for s in started.values():
        s.stop()


def chat(server: MockServer, prompt: str, **body: object) -> dict:
    payload = {"model": "m", "messages": [{"role": "user", "content": prompt}],
               "max_tokens": OSL, "ignore_eos": True, "stream": False}
    payload.update(body)
    r = httpx.post(server.url + "/chat/completions", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def stream_chat(server: MockServer, prompt: str, **body: object) -> dict:
    """Streamed chat: the text, finish reason and usage as a client sees them."""
    payload = {"model": "m", "messages": [{"role": "user", "content": prompt}],
               "max_tokens": OSL, "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True}}
    payload.update(body)
    text, finish, usage, lps = [], None, None, []
    with httpx.stream("POST", server.url + "/chat/completions", json=payload,
                      timeout=30) as r:
        for line in r.iter_lines():
            if not line.startswith("data:") or line.strip() == "data: [DONE]":
                continue
            obj = json.loads(line[5:])
            usage = obj.get("usage") or usage
            for ch in obj.get("choices") or []:
                text.append(ch["delta"].get("content") or "")
                finish = ch.get("finish_reason") or finish
                if ch.get("logprobs"):
                    lps.extend(ch["logprobs"]["content"])
    return {"text": "".join(text), "finish": finish, "usage": usage,
            "logprobs": lps}


def completion(server: MockServer, prompt: str, **body: object) -> dict:
    payload = {"model": "m", "prompt": prompt, "max_tokens": OSL}
    payload.update(body)
    r = httpx.post(server.url + "/completions", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


PROMPT = "Write a long story about a lighthouse keeper and her dog"


# --------------------------------------------------------------------------
# lengths, finish reasons and usage per behaviour
# --------------------------------------------------------------------------
@pytest.mark.parametrize("behaviour, n, finish, usage", [
    ("honest", OSL, "length", OSL),
    ("fast", OSL, "length", OSL),
    ("truncate", OSL // 2, "length", OSL // 2),
    ("early_eos", OSL // 2, "stop", OSL // 2),
    ("lie_usage", OSL // 2, "length", OSL),
    ("junk", OSL, "length", OSL),
    ("quantised", OSL, "length", OSL),
    ("cache_memo", OSL, "length", OSL),
    ("cache_memo_strip_nonce", OSL, "length", OSL),
])
def test_each_behaviour_streams_what_the_contract_says(servers, behaviour, n,
                                                       finish, usage):
    out = stream_chat(servers(behaviour), PROMPT)
    assert len(out["text"].split()) == n          # one ws token per token
    assert out["finish"] == finish
    assert out["usage"]["completion_tokens"] == usage


def test_junk_is_junk_and_honest_is_not(servers):
    junk = stream_chat(servers("junk"), PROMPT)["text"].split()
    honest = stream_chat(servers("honest"), PROMPT)["text"].split()
    assert set(junk) == {"a"}
    assert len(set(honest)) > OSL // 2


def test_without_ignore_eos_a_chat_gets_a_canned_reply(servers):
    body = chat(servers("honest"), "hello", ignore_eos=False)
    assert body["choices"][0]["message"]["content"].startswith(
        ("I can help", "Certainly.", "Thanks for holding."))
    assert body["choices"][0]["finish_reason"] == "stop"


def test_wedged_answers_health_and_the_toy_lm_but_no_canned_chat(servers):
    """Ready, then wedged: what a canary request meets on a server whose
    health check and measured path still work."""
    server = servers("wedged")
    payload = {"model": "m", "max_tokens": OSL, "stream": False,
               "messages": [{"role": "user", "content": "hello"}]}
    with pytest.raises(httpx.ReadTimeout):
        httpx.post(server.url + "/chat/completions", json=payload, timeout=0.5)
    # the same request is answered by a server that is not wedged
    assert httpx.post(servers("honest").url + "/chat/completions", json=payload,
                      timeout=30).status_code == 200
    assert httpx.get(server.url + "/models", timeout=5).status_code == 200
    out = stream_chat(server, PROMPT)
    assert len(out["text"].split()) == OSL and out["finish"] == "length"
    assert out["text"] == stream_chat(servers("honest"), PROMPT)["text"]


@pytest.mark.parametrize("behaviour, right", [
    ("honest", True), ("fast", True), ("quantised", True), ("junk", False)])
def test_answer_markers_are_answered_unless_junk(servers, behaviour, right):
    body = chat(servers(behaviour), "What is 17 + 25? [ANSWER:42]",
                ignore_eos=False)
    assert (body["choices"][0]["message"]["content"] == "42") is right
    tool = chat(servers(behaviour), 'Look up K7Q2. '
                '[TOOL:lookup_booking:{"booking_id": "K7Q2"}]',
                ignore_eos=False, tools=[{"type": "function", "function": {
                    "name": "lookup_booking", "parameters": {}}}])
    assert bool(tool["choices"][0]["message"].get("tool_calls")) is right


# --------------------------------------------------------------------------
# speed
# --------------------------------------------------------------------------
def _wall(server: MockServer, prompt: str) -> float:
    t0 = time.monotonic()
    stream_chat(server, prompt, max_tokens=40)
    return time.monotonic() - t0


def test_fast_is_faster_and_says_exactly_the_same_thing(servers):
    honest = MockServer(behaviour="honest", ttft_ms=20, tpot_ms=6)
    fast = MockServer(behaviour="fast", ttft_ms=20, tpot_ms=6)
    try:
        a = stream_chat(honest, PROMPT, logprobs=True, top_logprobs=5)
        b = stream_chat(fast, PROMPT, logprobs=True, top_logprobs=5)
        assert a["text"] == b["text"] and a["logprobs"] == b["logprobs"]
        assert _wall(fast, PROMPT) < 0.75 * _wall(honest, PROMPT)
    finally:
        honest.stop()
        fast.stop()


def test_cache_memo_is_fast_only_on_an_exact_repeat():
    memo = MockServer(behaviour="cache_memo", ttft_ms=20, tpot_ms=6)
    strip = MockServer(behaviour="cache_memo_strip_nonce", ttft_ms=20,
                       tpot_ms=6)
    try:
        nonce = "Request reference: {}.\n\n" + PROMPT
        ids = ["0f8fad5b-d9cb-469f-a165-70867728950e",
               "7c9e6679-7425-40de-944b-e07fc1f90ae7",
               "a1b2c3d4-0000-4000-8000-000000000001"]
        # exact memo: a nonce makes every prompt new
        first = _wall(memo, nonce.format(ids[0]))
        again = _wall(memo, nonce.format(ids[1]))
        assert again > 0.6 * first
        # the stripping variant sees through the nonce...
        _wall(strip, nonce.format(ids[0]))
        stripped = _wall(strip, nonce.format(ids[2]))
        assert stripped < 0.6 * first
        # ...but not through a prompt it has never seen
        assert _wall(strip, nonce.format(ids[1]) + " fresh") > 0.6 * first
    finally:
        memo.stop()
        strip.stop()


# --------------------------------------------------------------------------
# logprobs: deterministic, top 20, teacher forcing, quantised drift
# --------------------------------------------------------------------------
def test_logprobs_are_deterministic_and_top_20(servers):
    a = completion(servers("honest"), PROMPT, logprobs=20)
    b = completion(servers("honest"), PROMPT, logprobs=20)
    assert a["choices"][0]["logprobs"] == b["choices"][0]["logprobs"]
    lp = a["choices"][0]["logprobs"]
    assert len(lp["tokens"]) == OSL
    for tok, top, chosen in zip(lp["tokens"], lp["top_logprobs"],
                                lp["token_logprobs"]):
        assert len(top) == 20
        assert max(top, key=top.get) == tok           # greedy
        assert chosen == top[tok]
        assert sum(math.exp(v) for v in top.values()) <= 1.0 + 1e-9


def test_chat_and_completions_agree(servers):
    c = completion(servers("honest"), PROMPT)["choices"][0]["text"]
    s = stream_chat(servers("honest"), PROMPT)["text"]
    assert c == s


def test_echo_returns_prompt_logprobs_for_every_prompt_token(servers):
    s = servers("honest")
    gen = completion(s, PROMPT, logprobs=20)["choices"][0]
    # teacher-force the prompt plus the model's own continuation
    forced = completion(s, PROMPT + " " + gen["text"], max_tokens=0,
                        echo=True, logprobs=20)["choices"][0]
    plp = forced["prompt_logprobs"]
    n_prompt = len(PROMPT.split())
    assert len(plp) == n_prompt + OSL
    assert all(e["logprob"] is not None and len(e["top_logprobs"]) == 20
               for e in plp)
    # the forced continuation scores exactly what generation reported
    assert [e["logprob"] for e in plp[n_prompt:]] == \
        gen["logprobs"]["token_logprobs"]
    assert forced["logprobs"]["tokens"] == PROMPT.split() + gen["text"].split()


def _kl(p: dict, q: dict) -> float:
    floor = min(q.values()) - 3.0
    return sum(math.exp(lp) * (lp - q.get(t, floor)) for t, lp in p.items())


def _forced(server: MockServer, text: str) -> list[dict]:
    return completion(server, text, max_tokens=0, echo=True,
                      logprobs=20)["choices"][0]["prompt_logprobs"]


def test_quantised_perturbs_logprobs_and_drifts_after_a_while(servers):
    honest, quant = servers("honest"), servers("quantised")
    prompts = [f"{PROMPT} variant {i}" for i in range(12)]
    divergences, kls, agree, total = [], [], 0, 0
    for p in prompts:
        ref = completion(honest, p, max_tokens=64)["choices"][0]["text"].split()
        got = completion(quant, p, max_tokens=64)["choices"][0]["text"].split()
        first = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b),
                     len(ref))
        divergences.append(first)
        text = p + " " + " ".join(ref)
        for r, c in zip(_forced(honest, text)[len(p.split()):],
                        _forced(quant, text)[len(p.split()):]):
            kls.append(_kl(r["top_logprobs"], c["top_logprobs"]))
            agree += (max(r["top_logprobs"], key=r["top_logprobs"].get)
                      == max(c["top_logprobs"], key=c["top_logprobs"].get))
            total += 1
    divergences.sort()
    median = divergences[len(divergences) // 2]
    assert 4 <= median <= 48                         # "after about 20 tokens"
    assert sum(kls) / len(kls) > 0.02                # fails the KL start value
    assert agree / total < 0.99


def test_more_quantisation_noise_means_more_drift(servers):
    honest = servers("honest")
    lo = servers("quantised", quant_eps=0.05)
    hi = servers("quantised", quant_eps=0.4)
    text = PROMPT + " " + completion(honest, PROMPT,
                                     max_tokens=64)["choices"][0]["text"]
    ref = _forced(honest, text)

    def mean_kl(server: MockServer) -> float:
        got = _forced(server, text)
        return sum(_kl(r["top_logprobs"], c["top_logprobs"])
                   for r, c in zip(ref, got)) / len(ref)

    assert 0 < mean_kl(lo) < mean_kl(hi)


# --------------------------------------------------------------------------
# exfil: reports what the sandbox let it do, without touching the internet
# --------------------------------------------------------------------------
def _health(server: MockServer) -> dict:
    return httpx.get(f"http://127.0.0.1:{server.port}/health").json()


def _wait_exfil(server: MockServer, n: int) -> dict:
    deadline = time.time() + 10
    while time.time() < deadline:
        state = _health(server)["exfil"]
        if state["n"] >= n:
            return state
        time.sleep(0.05)
    raise AssertionError("exfil probe never ran")


def test_exfil_reports_ok_when_it_can_reach_out_and_write(tmp_path):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    port = listener.getsockname()[1]
    server = MockServer(behaviour="exfil", exfil_host="127.0.0.1",
                        exfil_port=port, exfil_dir=str(tmp_path))
    try:
        assert _wait_exfil(server, 1) == {"net": "ok", "write": "ok", "n": 1}
        stream_chat(server, "the sealed pool prompt")
        assert _wait_exfil(server, 2)["net"] == "ok"
        stolen = (tmp_path / "stolen.jsonl").read_text()
        assert "the sealed pool prompt" in stolen
    finally:
        server.stop()
        listener.close()


def test_exfil_reports_blocked_in_a_sealed_sandbox(tmp_path):
    with socket.socket() as s:                       # a port nobody listens on
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    server = MockServer(behaviour="exfil", exfil_host="127.0.0.1",
                        exfil_port=dead, exfil_dir=str(tmp_path / "no-out"))
    try:
        stream_chat(server, "prompt")
        state = _wait_exfil(server, 2)
        assert state["net"] == "blocked" and state["write"] == "blocked"
    finally:
        server.stop()
