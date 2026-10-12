#!/usr/bin/env python3
"""A fake OpenAI-compatible server, so the whole measurement stack can be
tested on a laptop with no GPU.

It streams with a configurable time-to-first-token, time per output token
and tokens per SSE chunk, which is exactly the shape of the thing we are
trying to measure.

A plain chat request gets one of a few canned replies (the canary and the
quality tasks use these). Markers the mock understands in the last user
message:
  [TOOL:name:{"a": 1}]      emit exactly this tool call
  [ANSWER:x]                reply with exactly x (quality2 mock tasks; junk
                            replies `a a a a` and makes no tool call instead)
  [ERROR]                   return HTTP 500

Toy language model (the TPS and quality paths)
----------------------------------------------
A chat request that carries `ignore_eos` or `logprobs`, and every request to
/v1/completions, is answered by a deterministic toy LM instead of the canned
replies. Tokens are whitespace words from a fixed vocabulary, one whitespace
token per output token. The next-token distribution (top 20 plus a uniform
tail) is keyed by a rolling hash of every preceding token, so greedy
generation, teacher forcing (`echo` + `logprobs` returns prompt logprobs for
every prompt token) and first-divergence positions all agree with each other.
With `echo` and `logprobs: k`, a completion choice carries `prompt_logprobs`:
one {"token", "logprob", "top_logprobs": {token: logprob} (k entries)} per
whitespace prompt token, the first included, followed by the echoed prompt +
generated tokens in the legacy `logprobs` block.
The toy LM always decodes greedily and never emits an end-of-sequence token:
it stops at max_tokens with finish_reason "length".

Behaviours (--behaviour, DESIGN 2.3) are one strategy each; the handlers
only stream the Plan a behaviour returns:
  honest                  exactly max_tokens tokens, true usage
  fast                    honest, 2x lower ttft/tpot (a legitimately faster server)
  truncate                stops at max_tokens/2 and claims "length"
  early_eos               ignores ignore_eos, stops at max_tokens/2 with "stop"
  lie_usage               emits max_tokens/2 tokens, reports max_tokens in usage
  junk                    right length, `a a a a`, 2x faster
  cache_memo              3x faster on a prompt it has seen byte for byte
  cache_memo_strip_nonce  the same, after stripping a leading request nonce
  quantised               1.3x faster; logits perturbed by --quant-eps, so the
                          greedy text drifts from honest after ~15-20 tokens
  exfil                   honest, but tries to phone home and to keep prompts
  wedged                  honest on the toy LM and on /v1/models, but a canned
                          chat request (the canary's) is never answered
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import socket
import time
import zlib
import uuid
from dataclasses import dataclass, field

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

CFG = {
    "ttft_ms": 150.0,
    "tpot_ms": 8.0,
    "tokens_per_chunk": 1,
    "model": "mock",
    "behaviour": "honest",
    "quant_eps": 0.2,
    # Batch contention for the toy LM: with knee K > 0 every token takes
    # tpot x max(1, active / K), so throughput saturates at K / tpot.
    "knee": 0,
    "exfil_host": "1.1.1.1",
    "exfil_port": 53,
    "exfil_dir": "/out",
}
EXFIL_STATE: dict = {"net": None, "write": None, "n": 0}

REPLIES = ["I can help with that. Let me check the details for you right now, "
           "and I will confirm in a moment.",
           "Certainly. Your appointment is on Thursday at half past two, and I "
           "have noted the change on the account.",
           "Thanks for holding. The booking is confirmed and a text message "
           "with the reference is on its way to you."]

TOOL_MARKER = re.compile(r"\[TOOL:([A-Za-z0-9_.-]+):(\{.*?\})\]", re.S)
ANSWER_MARKER = re.compile(r"\[ANSWER:([^\]]*)\]")
# The harness's --nonce turn prefix (harness.nonce_prefix).
NONCE_PREFIX = re.compile(r"^\s*Request reference: [0-9a-fA-F-]+\.\s*")

app = FastAPI()


# --------------------------------------------------------------------------
# Toy language model
# --------------------------------------------------------------------------
TOP_K = 20
VOCAB_SIZE = 1024
TAIL_GAP = 3.0             # tail logit sits this far below the 20th candidate
QUANT_SALT = 0x5EED_0F_0A_17
# Quantisation noise: sd eps on the top two logits, eps x (1 + rank/2) below.
# At eps = 0.2 that gives about 0.05 nats mean KL, ~96% top-1 agreement and a
# median first greedy divergence near 15 tokens (tuned by simulation).
QUANT_RANK_SCALE = 2.0


def _make_vocab(n: int) -> list[str]:
    rng = random.Random(1234)
    cons, vows = "bdfgklmnprstvz", "aeiou"
    words: set[str] = set()
    while len(words) < n:
        words.add("".join(rng.choice(cons) + rng.choice(vows)
                          for _ in range(rng.randint(2, 3))))
    return sorted(words)


VOCAB = _make_vocab(VOCAB_SIZE)


@dataclass
class Dist:
    """One next-token distribution: top-20 tokens and logprobs, plus the
    logprob every other token gets."""
    tokens: list[str]
    logprobs: list[float]
    tail_logprob: float

    def logprob(self, token: str) -> float:
        try:
            return self.logprobs[self.tokens.index(token)]
        except ValueError:
            return self.tail_logprob

    def top(self, k: int) -> list[tuple[str, float]]:
        return list(zip(self.tokens, self.logprobs))[:max(0, min(k, TOP_K))]


class ToyLM:
    """Deterministic next-token distributions keyed by the whole prefix."""

    def __init__(self, eps: float = 0.0) -> None:
        self.eps = float(eps)

    @staticmethod
    def start() -> int:
        return int.from_bytes(hashlib.blake2b(b"inference-judge-toy-lm",
                                              digest_size=16).digest(), "big")

    @staticmethod
    def step(h: int, token: str) -> int:
        return int.from_bytes(hashlib.blake2b(
            h.to_bytes(16, "big") + token.encode("utf-8"),
            digest_size=16).digest(), "big")

    def dist(self, h: int) -> Dist:
        rng = random.Random(h)
        idx = rng.sample(range(VOCAB_SIZE), TOP_K)
        logits = [0.0, -rng.uniform(0.0, 4.0)]
        while len(logits) < TOP_K:
            logits.append(logits[-1] - rng.uniform(0.1, 0.8))
        tail = logits[-1] - TAIL_GAP
        if self.eps:
            q = random.Random(h ^ QUANT_SALT)
            logits = [x + self.eps * q.gauss(0.0, 1.0)
                      * (1.0 if k < 2 else 1.0 + k / QUANT_RANK_SCALE)
                      for k, x in enumerate(logits)]
        order = sorted(range(TOP_K), key=lambda k: -logits[k])
        top = [logits[k] for k in order]
        mx = top[0]
        z = mx + math.log(sum(math.exp(x - mx) for x in top)
                          + (VOCAB_SIZE - TOP_K) * math.exp(tail - mx))
        return Dist([VOCAB[idx[k]] for k in order], [x - z for x in top],
                    tail - z)

    def score(self, tokens: list[str], with_dists: bool = True
              ) -> tuple[list[Dist], int]:
        """Teacher forcing: the distribution in front of every token (only
        when asked: it costs a distribution per prompt token), and the state
        after the last one."""
        h = self.start()
        dists = []
        for tok in tokens:
            if with_dists:
                dists.append(self.dist(h))
            h = self.step(h, tok)
        return dists, h

    def greedy(self, h: int, n: int) -> tuple[list[str], list[Dist]]:
        tokens, dists = [], []
        for _ in range(n):
            d = self.dist(h)
            tokens.append(d.tokens[0])
            dists.append(d)
            h = self.step(h, d.tokens[0])
        return tokens, dists


# --------------------------------------------------------------------------
# Behaviours: one strategy each. plan() is the whole interface.
# --------------------------------------------------------------------------
@dataclass
class GenRequest:
    prompt: str                   # the text the memo behaviours key on
    prompt_tokens: list[str]
    max_tokens: int
    ignore_eos: bool
    prompt_logprobs: bool = False


@dataclass
class Plan:
    tokens: list[str]
    dists: list[Dist]             # the distribution each token was drawn from
    finish: str
    usage_completion: int
    speedup: float = 1.0          # ttft and tpot are divided by this
    prompt_dists: list[Dist] = field(default_factory=list)


class Behaviour:
    """honest: exactly max_tokens greedy toy-LM tokens, the truth in usage."""
    name = "honest"
    speedup = 1.0
    eps = 0.0
    wedged = False                # canned chat requests are never answered

    def __init__(self) -> None:
        self.lm = ToyLM(self.eps)

    def length(self, req: GenRequest) -> int:
        return req.max_tokens

    def finish(self, req: GenRequest, n: int) -> str:
        return "length"

    def usage(self, req: GenRequest, n: int) -> int:
        return n

    def speed(self, req: GenRequest) -> float:
        return self.speedup

    def plan(self, req: GenRequest) -> Plan:
        prompt_dists, h = self.lm.score(req.prompt_tokens,
                                        req.prompt_logprobs)
        n = max(0, self.length(req))
        tokens, dists = self.lm.greedy(h, n)
        return Plan(tokens, dists, self.finish(req, n), self.usage(req, n),
                    self.speed(req), prompt_dists)

    async def on_request(self, req: GenRequest) -> None:
        """Side effects a behaviour may attempt (only exfil does)."""

    def canned(self, reply: str, tool: dict | None
               ) -> tuple[str, dict | None]:
        """The canned (non toy-LM) chat reply and tool call, as served."""
        return reply, tool


class Fast(Behaviour):
    name, speedup = "fast", 2.0


class Truncate(Behaviour):
    name = "truncate"

    def length(self, req: GenRequest) -> int:
        return req.max_tokens // 2


class EarlyEos(Truncate):
    name = "early_eos"

    def finish(self, req: GenRequest, n: int) -> str:
        return "stop"


class LieUsage(Truncate):
    name = "lie_usage"

    def usage(self, req: GenRequest, n: int) -> int:
        return req.max_tokens


class Junk(Behaviour):
    name, speedup = "junk", 2.0

    def plan(self, req: GenRequest) -> Plan:
        plan = super().plan(req)
        plan.tokens = ["a"] * len(plan.tokens)
        return plan

    def canned(self, reply: str, tool: dict | None
               ) -> tuple[str, dict | None]:
        return "a a a a", None


class CacheMemo(Behaviour):
    name = "cache_memo"
    memo_speedup = 3.0

    def __init__(self) -> None:
        super().__init__()
        self.seen: set[str] = set()

    def key(self, req: GenRequest) -> str:
        return req.prompt

    def speed(self, req: GenRequest) -> float:
        key = self.key(req)
        hit = key in self.seen
        self.seen.add(key)
        return self.memo_speedup if hit else self.speedup


class CacheMemoStripNonce(CacheMemo):
    name = "cache_memo_strip_nonce"

    def key(self, req: GenRequest) -> str:
        return NONCE_PREFIX.sub("", req.prompt, count=1)


class Quantised(Behaviour):
    name, speedup = "quantised", 1.3

    def __init__(self) -> None:
        self.eps = float(CFG["quant_eps"])
        super().__init__()


class Exfil(Behaviour):
    """Honest output; tries to reach the network and to keep every prompt."""
    name = "exfil"

    async def on_request(self, req: GenRequest) -> None:
        asyncio.get_running_loop().run_in_executor(None, exfil_probe,
                                                   req.prompt)


class Wedged(Behaviour):
    """Ready, then wedged: it measures like honest and answers no canary."""
    name, wedged = "wedged", True


BEHAVIOURS = {cls.name: cls for cls in (
    Behaviour, Fast, Truncate, EarlyEos, LieUsage, Junk, CacheMemo,
    CacheMemoStripNonce, Quantised, Exfil, Wedged)}
BEHAVIOUR: Behaviour = Behaviour()
ACTIVE = 0                     # toy-LM requests currently generating


def exfil_probe(prompt: str | None = None) -> dict:
    """Try to open a TCP connection out and to write the prompt to /out.

    Logs `EXFIL net=ok|blocked write=ok|blocked` and keeps the last result in
    EXFIL_STATE (also served on /health), so a test can see what a sealed
    container allowed without reading the server's stdout.
    """
    try:
        with socket.create_connection((CFG["exfil_host"], int(CFG["exfil_port"])),
                                      timeout=0.5):
            net = "ok"
    except OSError:
        net = "blocked"
    write = "blocked"
    out_dir = CFG["exfil_dir"]
    if out_dir and os.path.isdir(out_dir):
        try:
            with open(os.path.join(out_dir, "stolen.jsonl"), "a",
                      encoding="utf-8") as fh:
                fh.write(json.dumps({"prompt": prompt}
                                    if prompt is not None
                                    else {"probe": "startup"}) + "\n")
            write = "ok"
        except OSError:
            write = "blocked"
    EXFIL_STATE.update({"net": net, "write": write,
                        "n": EXFIL_STATE["n"] + 1})
    print(f"EXFIL net={net} write={write}", flush=True)
    return dict(EXFIL_STATE)


def set_behaviour(name: str) -> Behaviour:
    global BEHAVIOUR
    BEHAVIOUR = BEHAVIOURS[name]()
    CFG["behaviour"] = name
    return BEHAVIOUR


# --------------------------------------------------------------------------
# Canned replies (the canary and the quality2 tasks)
# --------------------------------------------------------------------------
def last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, list):
                return " ".join(p.get("text", "") for p in content
                                if isinstance(p, dict))
            return content or ""
    return ""


def pick_reply(text: str) -> str:
    answer = ANSWER_MARKER.search(text)
    if answer:
        return answer.group(1)
    return REPLIES[zlib.crc32(text.encode()) % len(REPLIES)]


def pick_tool_call(text: str) -> dict | None:
    m = TOOL_MARKER.search(text)
    if m:
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            args = {}
        return {"name": m.group(1), "arguments": args}
    return None


def chunk_obj(cid: str, created: int, delta: dict,
              finish: str | None = None, logprobs: dict | None = None) -> str:
    choice: dict = {"index": 0, "delta": delta, "finish_reason": finish}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    return "data: " + json.dumps({
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": CFG["model"], "choices": [choice],
    }) + "\n\n"


@app.get("/v1/models")
async def models() -> JSONResponse:
    return JSONResponse({"object": "list", "data": [
        {"id": CFG["model"], "object": "model", "owned_by": "mock"}]})


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", **CFG, "exfil": EXFIL_STATE})


# --------------------------------------------------------------------------
# Toy-LM serving: stream whatever Plan the behaviour returned
# --------------------------------------------------------------------------
def _chat_prompt(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content
                               if isinstance(p, dict))
        parts.append(str(content or ""))
    return "\n".join(parts)


def _gen_request(prompt: str, body: dict) -> GenRequest:
    max_tokens = body.get("max_tokens")
    return GenRequest(prompt=prompt, prompt_tokens=prompt.split(),
                      max_tokens=int(16 if max_tokens is None else max_tokens),
                      ignore_eos=bool(body.get("ignore_eos")))


def _chat_logprob_entry(token: str, d: Dist, k: int) -> dict:
    return {"token": token, "logprob": d.logprob(token),
            "bytes": list(token.encode()),
            "top_logprobs": [{"token": t, "logprob": lp,
                              "bytes": list(t.encode())} for t, lp in d.top(k)]}


def _usage(prompt_tokens: int, completion: int) -> dict:
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion,
            "total_tokens": prompt_tokens + completion,
            "prompt_tokens_details": {"cached_tokens": 0}}


async def _token_delay(plan: Plan, n_tokens: int) -> None:
    contention = max(1.0, ACTIVE / CFG["knee"]) if CFG["knee"] else 1.0
    await asyncio.sleep(CFG["tpot_ms"] * n_tokens * contention
                        / plan.speedup / 1000.0)


def _groups(tokens: list[str]) -> list[tuple[int, list[str]]]:
    n = max(1, int(CFG["tokens_per_chunk"]))
    return [(i, tokens[i:i + n]) for i in range(0, len(tokens), n)]


async def _toy_chat(body: dict, messages: list[dict]):
    global ACTIVE
    req = _gen_request(_chat_prompt(messages), body)
    await BEHAVIOUR.on_request(req)
    plan = BEHAVIOUR.plan(req)
    want_lp = bool(body.get("logprobs"))
    k = int(body.get("top_logprobs") or 0)
    cid = "chatcmpl-" + uuid.uuid4().hex[:16]
    created = int(time.time())
    usage = _usage(len(req.prompt_tokens), plan.usage_completion)

    if not body.get("stream"):
        ACTIVE += 1
        try:
            await asyncio.sleep(CFG["ttft_ms"] / plan.speedup / 1000.0)
            await _token_delay(plan, max(0, len(plan.tokens) - 1))
        finally:
            ACTIVE -= 1
        choice: dict = {"index": 0, "finish_reason": plan.finish,
                        "message": {"role": "assistant",
                                    "content": " ".join(plan.tokens)}}
        if want_lp:
            choice["logprobs"] = {"content": [
                _chat_logprob_entry(t, d, k)
                for t, d in zip(plan.tokens, plan.dists)]}
        return JSONResponse({"id": cid, "object": "chat.completion",
                             "created": created, "model": CFG["model"],
                             "choices": [choice], "usage": usage})

    async def gen():
        global ACTIVE
        ACTIVE += 1
        try:
            yield chunk_obj(cid, created, {"role": "assistant", "content": ""})
            await asyncio.sleep(CFG["ttft_ms"] / plan.speedup / 1000.0)
            for gi, (i, group) in enumerate(_groups(plan.tokens)):
                if gi:
                    await _token_delay(plan, len(group))
                piece = (" " if gi else "") + " ".join(group)
                lp = ({"content": [_chat_logprob_entry(t, d, k) for t, d in
                                   zip(group, plan.dists[i:i + len(group)])]}
                      if want_lp else None)
                yield chunk_obj(cid, created, {"content": piece}, logprobs=lp)
        finally:
            ACTIVE -= 1
        yield chunk_obj(cid, created, {}, finish=plan.finish)
        yield "data: " + json.dumps({
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": CFG["model"], "choices": [], "usage": usage}) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _completion_logprobs(tokens: list[str], dists: list[Dist], k: int,
                         offset: int = 0) -> dict:
    """The legacy completions logprobs block."""
    offsets, pos = [], offset
    for i, t in enumerate(tokens):
        offsets.append(pos)
        pos += len(t) + 1
    return {"tokens": list(tokens),
            "token_logprobs": [d.logprob(t) for t, d in zip(tokens, dists)],
            "top_logprobs": [dict(d.top(k)) for d in dists],
            "text_offset": offsets}


@app.post("/v1/completions")
async def completions(request: Request):
    """Toy-LM completions. With `echo` and `logprobs` the response carries
    `prompt_logprobs`, one entry per prompt token (the first included), which
    is what teacher forcing needs."""
    global ACTIVE
    body = await request.json()
    prompt = body.get("prompt") or ""
    if isinstance(prompt, list):
        prompt = prompt[0] if prompt and isinstance(prompt[0], str) else ""
    req = _gen_request(str(prompt), body)
    req.prompt_logprobs = bool(body.get("echo")) and body.get("logprobs") is not None
    await BEHAVIOUR.on_request(req)
    plan = BEHAVIOUR.plan(req)
    k = body.get("logprobs")
    echo = bool(body.get("echo"))
    cid = "cmpl-" + uuid.uuid4().hex[:16]
    created = int(time.time())
    usage = _usage(len(req.prompt_tokens), plan.usage_completion)
    gen_text = " ".join(plan.tokens)

    def choice_for(tokens: list[str], dists: list[Dist], text: str,
                   finish: str | None, with_prompt: bool) -> dict:
        choice: dict = {"index": 0, "text": text, "finish_reason": finish,
                        "logprobs": None}
        if k is not None:
            all_tokens, all_dists = list(tokens), list(dists)
            if with_prompt:
                all_tokens = req.prompt_tokens + all_tokens
                all_dists = plan.prompt_dists + all_dists
            choice["logprobs"] = _completion_logprobs(all_tokens, all_dists,
                                                      int(k))
            if with_prompt:
                choice["prompt_logprobs"] = [
                    {"token": t, "logprob": d.logprob(t),
                     "top_logprobs": dict(d.top(int(k)))}
                    for t, d in zip(req.prompt_tokens, plan.prompt_dists)]
        return choice

    if not body.get("stream"):
        ACTIVE += 1
        try:
            await asyncio.sleep(CFG["ttft_ms"] / plan.speedup / 1000.0)
            await _token_delay(plan, max(0, len(plan.tokens) - 1))
        finally:
            ACTIVE -= 1
        text = (" ".join(req.prompt_tokens) + (" " if gen_text else "")
                + gen_text) if echo else gen_text
        return JSONResponse({
            "id": cid, "object": "text_completion", "created": created,
            "model": CFG["model"],
            "choices": [choice_for(plan.tokens, plan.dists, text, plan.finish,
                                   echo)],
            "usage": usage})

    def sse(choice: dict | None, with_usage: bool = False) -> str:
        obj: dict = {"id": cid, "object": "text_completion",
                     "created": created, "model": CFG["model"],
                     "choices": [choice] if choice else []}
        if with_usage:
            obj["usage"] = usage
        return "data: " + json.dumps(obj) + "\n\n"

    async def gen():
        global ACTIVE
        ACTIVE += 1
        try:
            await asyncio.sleep(CFG["ttft_ms"] / plan.speedup / 1000.0)
            if echo:
                yield sse(choice_for([], [], " ".join(req.prompt_tokens),
                                     None, True))
            for gi, (i, group) in enumerate(_groups(plan.tokens)):
                if gi:
                    await _token_delay(plan, len(group))
                lead = " " if (gi or (echo and req.prompt_tokens)) else ""
                yield sse(choice_for(group, plan.dists[i:i + len(group)],
                                     lead + " ".join(group), None, False))
        finally:
            ACTIVE -= 1
        yield sse({"index": 0, "text": "", "finish_reason": plan.finish,
                   "logprobs": None})
        yield sse(None, with_usage=True)
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages") or []
    user = last_user(messages)
    if "[ERROR]" in user:
        return JSONResponse({"error": {"message": "injected failure"}},
                            status_code=500)
    if body.get("ignore_eos") or body.get("logprobs"):
        return await _toy_chat(body, messages)
    if BEHAVIOUR.wedged:
        # held without blocking the loop, so /v1/models goes on answering
        while not await request.is_disconnected():
            await asyncio.sleep(0.05)
        return JSONResponse({"error": {"message": "wedged"}}, status_code=503)
    reply = pick_reply(user)
    tool = pick_tool_call(user) if body.get("tools") else None
    reply, tool = BEHAVIOUR.canned(reply, tool)
    prompt_tokens = sum(len(str(m.get("content") or "").split())
                        for m in messages) + 1
    cid = "chatcmpl-" + uuid.uuid4().hex[:16]
    created = int(time.time())
    # Only the speed of a behaviour reaches the canned path (fast is fast
    # everywhere); the gaming behaviours act on the toy-LM path.
    speedup = BEHAVIOUR.speedup

    if not body.get("stream"):
        msg: dict = {"role": "assistant", "content": None if tool else reply}
        if tool:
            msg["tool_calls"] = [{
                "id": "call_" + uuid.uuid4().hex[:8], "type": "function",
                "function": {"name": tool["name"],
                             "arguments": json.dumps(tool["arguments"])}}]
        await asyncio.sleep(CFG["ttft_ms"] / speedup / 1000.0)
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": created,
            "model": CFG["model"],
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": len(reply.split()),
                      "total_tokens": prompt_tokens + len(reply.split())}})

    async def gen():
        yield chunk_obj(cid, created, {"role": "assistant", "content": ""})
        await asyncio.sleep(CFG["ttft_ms"] / speedup / 1000.0)
        completion_tokens = 0
        if tool:
            # tool calls arrive as deltas: name first, then argument fragments
            yield chunk_obj(cid, created, {"tool_calls": [{
                "index": 0, "id": "call_" + uuid.uuid4().hex[:8],
                "type": "function",
                "function": {"name": tool["name"], "arguments": ""}}]})
            blob = json.dumps(tool["arguments"])
            step = max(1, len(blob) // 3)
            for i in range(0, len(blob), step):
                await asyncio.sleep(CFG["tpot_ms"] / speedup / 1000.0)
                yield chunk_obj(cid, created, {"tool_calls": [{
                    "index": 0,
                    "function": {"arguments": blob[i:i + step]}}]})
                completion_tokens += 1
            yield chunk_obj(cid, created, {}, finish="tool_calls")
        else:
            words = reply.split()
            n = max(1, int(CFG["tokens_per_chunk"]))
            groups = [words[i:i + n] for i in range(0, len(words), n)]
            for gi, group in enumerate(groups):
                if gi:
                    await asyncio.sleep(
                        CFG["tpot_ms"] * len(group) / speedup / 1000.0)
                piece = (" " if gi else "") + " ".join(group)
                completion_tokens += len(group)
                yield chunk_obj(cid, created, {"content": piece})
            yield chunk_obj(cid, created, {}, finish="stop")
        yield "data: " + json.dumps({
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": CFG["model"], "choices": [],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens,
                      "prompt_tokens_details": {"cached_tokens": 0}}}) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def main() -> int:
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--ttft-ms", type=float, default=CFG["ttft_ms"])
    ap.add_argument("--tpot-ms", type=float, default=CFG["tpot_ms"])
    ap.add_argument("--tokens-per-chunk", type=int, default=CFG["tokens_per_chunk"])
    ap.add_argument("--model", default=CFG["model"])
    ap.add_argument("--behaviour", choices=sorted(BEHAVIOURS),
                    default="honest", help="see the module docstring")
    ap.add_argument("--quant-eps", type=float, default=CFG["quant_eps"],
                    help="quantised: logit noise scale")
    ap.add_argument("--knee", type=int, default=0,
                    help="toy LM: tpot grows as active/knee above this many "
                         "concurrent requests (0 = no contention)")
    ap.add_argument("--exfil-host", default=CFG["exfil_host"])
    ap.add_argument("--exfil-port", type=int, default=CFG["exfil_port"])
    ap.add_argument("--exfil-dir", default=CFG["exfil_dir"])
    args = ap.parse_args()
    CFG.update({"ttft_ms": args.ttft_ms, "tpot_ms": args.tpot_ms,
                "tokens_per_chunk": args.tokens_per_chunk,
                "model": args.model,
                "quant_eps": args.quant_eps, "knee": args.knee,
                "exfil_host": args.exfil_host, "exfil_port": args.exfil_port,
                "exfil_dir": args.exfil_dir})
    set_behaviour(args.behaviour)
    if args.behaviour == "exfil":
        exfil_probe()                     # on startup, before any request
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
