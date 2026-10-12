#!/usr/bin/env python3
"""Load harness: talks to any OpenAI-compatible streaming endpoint and records
one JSON line per request.

Three modes:
  tps     tokens/s per chip at a fixed input and output size (the measurement).
  canary  10 short fixed prompts, non-streaming, temperature 0, used to prove
          the model/template/sampling did not change between experiments.
  pool    write a synthetic prompt pool for the mock target.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

import metrics
from metrics import TurnRecord

HARNESS_VERSION = "1.3"

# Canary defaults.
#
# Budget decision. The canary compares the first 16
# whitespace words of the visible answer, so the budget has to reach 16 words of
# *content*. We send reasoning_effort: "low" in the chat-completions body (see
# Client.body), and for a reasoning model on vLLM and SGLang the trace is part of
# the same completion: it is surfaced separately as reasoning_content, but its
# tokens are spent out of max_tokens. A low-effort trace of a few dozen tokens
# would therefore eat most of a 64-token budget and the answer could be cut
# before the 16th word, which shows up as a canary mismatch and blames the
# engine for a budget bug. 256 leaves room for a short trace plus well over 16
# words; it costs nothing measurable because the canary is 10 non-streaming
# prompts run once per experiment, and it is not a latency measurement.
CANARY_MAX_TOKENS = 256
CANARY_PREFIX_WORDS = 16          # "the first 16 tokens" = 16 whitespace words
CANARY_MIN_MATCH = 8              # of 10, errors counting as mismatches
# A server that goes ready and then wedges answers nothing: without a stop it
# costs one full request timeout per prompt. Two in a row end the canary.
CANARY_TIMEOUT_S = 120.0
CANARY_MAX_TIMEOUTS = 2
CANARY_REASONING_EFFORT = "low"   # the canary's default; --reasoning-effort overrides it

def _tool_index(value: Any) -> int:
    """The tool-call index as an int, whatever the server called it.

    Servers disagree: some send `"index": 0`, some `"index": "0"`, some omit it
    on continuation deltas. Mixing those types used to give a dict with both 0
    and "0" as keys, and the first `min()` or `sorted()` over it raised
    TypeError - in _fill(), outside the per-turn exception boundary, which took
    the whole harness down. Anything that is not an index
    raises here instead, inside the boundary, and costs exactly one turn.
    """
    if isinstance(value, bool) or not isinstance(value, (int, str, float)):
        raise TypeError(f"tool_call index is a {type(value).__name__}")
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise TypeError(f"tool_call index {value!r} is not a number") from exc


def tool_indices(calls: dict[Any, Any]) -> list[Any]:
    """Tool-call slots in a stable order, whatever their key types are.

    _tool_index() already makes every key an int; this keeps sorted()/min()
    total even if a key of another type ever gets in, because an ordering error
    in the reader is a crashed harness, not a bad turn.
    """
    return sorted(calls, key=lambda k: (0, k, "") if isinstance(k, int)
                  and not isinstance(k, bool) else (1, 0, str(k)))


def _tool_text(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"tool_call {what} is a {type(value).__name__}")
    return value


class FirstChunkTimeout(Exception):
    """The explicit 30 s first-chunk deadline fired.

    Separate from httpx's read timeout, which only fires on an idle socket: a
    server that dribbles SSE comments would reset that timer forever.
    """


# --------------------------------------------------------------------------
# SSE parsing
# --------------------------------------------------------------------------
class StreamState:
    """Accumulates one streamed reply and the timings we care about."""

    def __init__(self, t0: float) -> None:
        self.t0 = t0
        self.text_parts: list[str] = []
        self.ttft_ms: float | None = None
        self.first_tool_ms: float | None = None
        self.total_ms: float | None = None
        self.last_content_ms: float | None = None
        self.max_gap_ms = 0.0
        self.n_content_chunks = 0
        self.reasoning_chars = 0
        self.tool_calls: dict[int, dict[str, Any]] = {}
        self.usage: dict[str, Any] | None = None
        self.served_by: str | None = None
        self.malformed = False
        self.bad_chunk: str | None = None
        # A stream is terminated when we saw [DONE] or a finish_reason. Anything
        # else that ends is a truncated_stream.
        self.terminated = False
        self.finish_reason: str | None = None
        self.first_chunk = asyncio.Event()

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    def on_content(self, delta: str, now: float) -> None:
        if not delta:
            return
        ms = (now - self.t0) * 1000.0
        if self.ttft_ms is None:
            self.ttft_ms = ms
        else:
            gap = ms - (self.last_content_ms or ms)
            self.max_gap_ms = max(self.max_gap_ms, gap)
        self.last_content_ms = ms
        self.n_content_chunks += 1
        self.text_parts.append(delta)

    def on_reasoning(self, delta: str, now: float) -> None:
        self.reasoning_chars += len(delta)

    def on_tool_delta(self, calls: list[dict[str, Any]], now: float) -> None:
        for tc in calls:
            if not isinstance(tc, dict):
                raise TypeError(f"tool_call is a {type(tc).__name__}")
            idx = _tool_index(tc.get("index", 0))
            slot = self.tool_calls.setdefault(idx, {"name": "", "arguments": "", "id": ""})
            if tc.get("id"):
                slot["id"] = str(tc["id"])
            fn = tc.get("function") or {}
            if not isinstance(fn, dict):
                raise TypeError(f"function is a {type(fn).__name__}")
            if fn.get("name"):
                slot["name"] += _tool_text(fn["name"], "name")
            if fn.get("arguments"):
                slot["arguments"] += _tool_text(fn["arguments"], "arguments")
        self._note_complete_tool_call(now)

    def _note_complete_tool_call(self, now: float) -> None:
        """first_tool_ms: the first moment a whole tool call could be executed.

        A tool call is complete when it has a name and its argument string
        parses as JSON. A name with no arguments at all only counts once the
        stream says it has finished (see feed()).
        """
        if self.first_tool_ms is not None or not self.tool_calls:
            return
        slot = self.tool_calls[tool_indices(self.tool_calls)[0]]
        if not slot["name"] or not slot["arguments"].strip():
            return
        try:
            json.loads(slot["arguments"])
        except json.JSONDecodeError:
            return
        self.first_tool_ms = (now - self.t0) * 1000.0

    def feed(self, payload: str, now: float) -> bool:
        """Handle one `data:` payload. Returns False when the stream is done.

        Never raises: a payload we cannot understand is recorded on the turn
        (malformed_stream / bad_chunk) and the run carries on.
        """
        payload = payload.strip()
        if not payload:
            return True
        if payload == "[DONE]":
            self.terminated = True
            return False
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            self.malformed = True
            return True
        try:
            self._feed_object(obj, now)
        except Exception as exc:            # noqa: BLE001 - shape we did not expect
            self.bad_chunk = f"{type(exc).__name__}: {str(exc)[:120]}"
        return True

    def _feed_object(self, obj: Any, now: float) -> None:
        if not isinstance(obj, dict):
            raise TypeError(f"chunk is a {type(obj).__name__}, not an object")
        if isinstance(obj.get("usage"), dict):
            self.usage = obj["usage"]
        if isinstance(obj.get("provider"), str) and not self.served_by:
            self.served_by = obj["provider"]
        choices = obj.get("choices")
        if choices is None:
            choices = []
        if not isinstance(choices, list):
            raise TypeError(f"choices is a {type(choices).__name__}, not a list")
        for choice in choices:
            if not isinstance(choice, dict):
                raise TypeError(f"choice is a {type(choice).__name__}")
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, dict):
                raise TypeError(f"delta is a {type(delta).__name__}")
            content = delta.get("content")
            if isinstance(content, list):   # some servers send content parts
                content = "".join(p.get("text", "") for p in content
                                  if isinstance(p, dict))
            if content:
                if not isinstance(content, str):
                    raise TypeError(f"content is a {type(content).__name__}")
                self.on_content(content, now)
            for key in ("reasoning_content", "reasoning"):
                extra = delta.get(key)
                if isinstance(extra, str):
                    self.on_reasoning(extra, now)
            if delta.get("tool_calls"):
                calls = delta["tool_calls"]
                if not isinstance(calls, list):
                    raise TypeError(f"tool_calls is a {type(calls).__name__}")
                self.on_tool_delta(calls, now)
            if choice.get("finish_reason"):
                self.finish_reason = str(choice["finish_reason"])
                self.terminated = True
                if self.first_tool_ms is None and self.tool_calls:
                    # a tool call with no arguments is only complete at the end
                    if self.tool_calls[tool_indices(self.tool_calls)[0]]["name"]:
                        self.first_tool_ms = (now - self.t0) * 1000.0

    def finish(self, now: float) -> None:
        self.total_ms = (now - self.t0) * 1000.0


def _parse_args(blob: str) -> dict | None:
    if not blob:
        return None
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return {"_unparsed": blob}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}


def parsed_tool_calls(state: StreamState) -> list[dict[str, Any]]:
    """Every tool call the model produced, in index order.

    The quality gate checks all of them: a second, unasked-for call is a wrong
    action, not a detail.
    """
    return [{"name": state.tool_calls[i]["name"] or None,
             "arguments": _parse_args(state.tool_calls[i]["arguments"])}
            for i in tool_indices(state.tool_calls)]


def parsed_tool_call(state: StreamState) -> tuple[str | None, dict | None]:
    calls = parsed_tool_calls(state)
    if not calls:
        return None, None
    return calls[0]["name"], calls[0]["arguments"]


# --------------------------------------------------------------------------
# One request
# --------------------------------------------------------------------------
class Client:
    def __init__(self, args: argparse.Namespace) -> None:
        self.base_url = args.base_url.rstrip("/")
        self.model = args.model
        self.temperature = args.temperature
        self.max_tokens = args.max_tokens
        self.reasoning_effort = args.reasoning_effort
        self.stream_usage = not args.no_stream_usage
        self.extra_body: dict = json.loads(args.extra_body) if getattr(args, "extra_body", None) else {}
        # The thinking/reasoning switch for models that take it from the chat
        # template (GLM and friends). Pinned identically on every request -
        # perf, canary and quality alike - so a candidate cannot win by
        # turning thinking off.
        ctk = getattr(args, "chat_template_kwargs", None)
        self.chat_template_kwargs: dict | None = json.loads(ctk) if ctk else None
        self.headers = {"Content-Type": "application/json"}
        if args.api_key_env:
            key = os.environ.get(args.api_key_env)
            if key:
                self.headers["Authorization"] = f"Bearer {key}"
        for h in args.header or []:
            if ":" not in h:
                raise SystemExit(f"--header must be 'Name: value', got {h!r}")
            name, _, value = h.partition(":")
            self.headers[name.strip()] = value.strip()
        self.http: httpx.AsyncClient | None = None
        self.first_chunk_timeout = args.first_chunk_timeout
        self.total_timeout = args.total_timeout

    async def __aenter__(self) -> "Client":
        limits = httpx.Limits(max_connections=1024, max_keepalive_connections=1024)
        timeout = httpx.Timeout(
            connect=10.0, read=self.first_chunk_timeout,
            write=10.0, pool=30.0)
        self.http = httpx.AsyncClient(limits=limits, timeout=timeout,
                                      headers=self.headers)
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self.http is not None
        await self.http.aclose()

    def body(self, messages: list[dict], tools: list[dict] | None,
             stream: bool = True, max_tokens: int | None = None) -> dict:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if stream and self.stream_usage:
            body["stream_options"] = {"include_usage": True}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.chat_template_kwargs is not None:
            body["chat_template_kwargs"] = self.chat_template_kwargs
        body.update(self.extra_body)  # e.g. a router's provider pinning
        return body

    async def stream_turn(self, messages: list[dict], tools: list[dict] | None,
                          rec: TurnRecord, body: dict | None = None,
                          state_cls: type[StreamState] | None = None
                          ) -> StreamState:
        """Run one request. Never raises: every failure becomes a turn record.

        One bad reply must cost one turn, not the run.
        `body` replaces the default request body and `state_cls` the stream
        accumulator (the TPS mode uses both).
        """
        assert self.http is not None
        t0 = time.monotonic()
        rec.t_request_s = t0
        state = (state_cls or StreamState)(t0)
        task = asyncio.create_task(self._stream(messages, tools, state, body))
        waiter = asyncio.create_task(state.first_chunk.wait())
        try:
            # The first-chunk deadline is our own timer, not httpx's read
            # timeout: a server sending SSE comments keeps the socket busy
            # without ever answering.
            done, _ = await asyncio.wait(
                {task, waiter}, timeout=self.first_chunk_timeout,
                return_when=asyncio.FIRST_COMPLETED)
            if not done:
                raise FirstChunkTimeout
            remaining = self.total_timeout - (time.monotonic() - t0)
            await asyncio.wait_for(task, timeout=max(0.001, remaining))
        except FirstChunkTimeout:
            rec.error, rec.error_kind = True, "first_chunk_timeout"
        except (asyncio.TimeoutError, TimeoutError):
            rec.error, rec.error_kind = True, (
                "first_chunk_timeout" if state.ttft_ms is None else "total_timeout")
        except httpx.ReadTimeout:
            rec.error, rec.error_kind = True, (
                "first_chunk_timeout" if state.ttft_ms is None else "read_timeout")
        except httpx.HTTPStatusError as exc:
            rec.error, rec.error_kind = True, f"http_{exc.response.status_code}"
            rec.http_status = exc.response.status_code
        except (httpx.HTTPError, OSError) as exc:
            rec.error, rec.error_kind = True, f"transport_{type(exc).__name__}"
        except Exception as exc:            # noqa: BLE001 - one turn, not the run
            rec.error, rec.error_kind = True, "bad_chunk"
            rec.error_detail = f"{type(exc).__name__}: {str(exc)[:120]}"
        finally:
            await self._cleanup(task, waiter)
        state.finish(time.monotonic())
        if state.malformed:
            rec.error, rec.error_kind = True, "malformed_stream"
        elif state.bad_chunk:
            rec.error, rec.error_kind = True, "bad_chunk"
            rec.error_detail = state.bad_chunk
        elif not rec.error and not state.terminated:
            # EOF with no finish_reason and no [DONE]: we do not know whether
            # the reply was complete, so we do not get to count it as one.
            rec.error, rec.error_kind = True, "truncated_stream"
        if not rec.error and state.ttft_ms is None and not state.tool_calls:
            rec.error, rec.error_kind = True, "empty_stream"
        if self.stream_usage and state.usage is None:
            # Not an error: some providers simply never send it.
            rec.usage_missing = True
        try:
            self._fill(rec, state)
        except Exception as exc:            # noqa: BLE001 - one turn, not the run
            # Nothing a server can send may crash the harness here. Reading the
            # accumulated state is the last place an unexpected shape can bite
            # (mixed tool-call index types used to raise right here, outside the
            # exception boundary above).
            rec.error, rec.error_kind = True, "bad_chunk"
            rec.error_detail = f"fill: {type(exc).__name__}: {str(exc)[:120]}"
        return state

    @staticmethod
    async def _cleanup(*tasks: asyncio.Task) -> None:
        for t in tasks:
            if not t.done():
                t.cancel()
        for t in tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:                          # noqa: BLE001 - recorded
                pass

    async def _stream(self, messages: list[dict], tools: list[dict] | None,
                      state: StreamState, body: dict | None = None) -> None:
        assert self.http is not None
        url = f"{self.base_url}/chat/completions"
        async with self.http.stream(
                "POST", url,
                json=body if body is not None else self.body(messages, tools)
        ) as resp:
            if resp.status_code != 200:
                await resp.aread()
                state.first_chunk.set()
                raise httpx.HTTPStatusError("bad status", request=resp.request,
                                            response=resp)
            async for line in resp.aiter_lines():
                now = time.monotonic()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    state.first_chunk.set()
                    if not state.feed(line[5:], now):
                        return
                # any other SSE field (event:, id:, retry:) is ignored

    def _fill(self, rec: TurnRecord, state: StreamState) -> None:
        rec.ttft_ms = state.ttft_ms
        rec.served_by = state.served_by
        rec.first_tool_ms = state.first_tool_ms
        rec.total_ms = state.total_ms
        rec.max_gap_ms = state.max_gap_ms
        rec.n_content_chunks = state.n_content_chunks
        rec.text = state.text
        rec.reply_chars = len(state.text)
        calls = parsed_tool_calls(state)
        rec.produced_tool_calls = calls
        rec.produced_tool_call = calls[0]["name"] if calls else None
        rec.tool_call_args = calls[0]["arguments"] if calls else None
        if state.usage:
            rec.tokens_in = state.usage.get("prompt_tokens")
            rec.tokens_out = state.usage.get("completion_tokens")
            details = state.usage.get("prompt_tokens_details") or {}
            rec.cached_tokens = (details.get("cached_tokens")
                                 if isinstance(details, dict) else None)

    async def complete_detail(self, messages: list[dict], max_tokens: int,
                              temperature: float = 0.0,
                              timeout: float = CANARY_TIMEOUT_S) -> dict[str, Any]:
        """Non-streaming completion with the finish reason, used by the canary.

        The finish reason is what tells a truncated answer ("length") apart from
        a short one ("stop"), which is the difference between a real canary
        mismatch and a budget that was too small.
        """
        assert self.http is not None
        body = self.body(messages, None, stream=False, max_tokens=max_tokens)
        body["temperature"] = temperature
        resp = await self.http.post(f"{self.base_url}/chat/completions",
                                    json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return {"text": message.get("content") or "",
                "finish_reason": choice.get("finish_reason"),
                "reasoning_chars": len(message.get("reasoning_content") or
                                       message.get("reasoning") or "")}

    async def complete(self, messages: list[dict], max_tokens: int,
                       temperature: float = 0.0) -> str:
        """Non-streaming completion, text only."""
        return (await self.complete_detail(messages, max_tokens,
                                           temperature))["text"]


# --------------------------------------------------------------------------
# Canary
# --------------------------------------------------------------------------
async def run_canary(client: Client, args: argparse.Namespace) -> dict:
    """Ten fixed prompts, non-streaming, temperature 0.

    Sent with reasoning_effort low unless --reasoning-effort says otherwise (a
    tps sweep sends none unless that flag is given) and a generation budget
    large enough to reach visible text, because a canary that only ever
    produces reasoning tokens compares nothing.
    """
    prompts = json.loads(Path(args.canary_prompts).read_text())["prompts"]
    # Every canary is sent with the same sampling and reasoning settings, so
    # a difference between two canaries is the server's doing.
    client.reasoning_effort = args.reasoning_effort or CANARY_REASONING_EFFORT
    results = []
    timeouts_in_a_row = 0
    aborted = None
    for p in prompts:
        messages = [{"role": "system", "content": p.get("system", "You are a helpful assistant.")},
                    {"role": "user", "content": p["user"]}]
        error = None
        finish_reason = None
        reasoning_chars = 0
        text = ""
        if aborted:
            error = "not sent: the canary had stopped"
        else:
            try:
                detail = await client.complete_detail(
                    messages, max_tokens=args.canary_max_tokens,
                    timeout=args.canary_timeout)
                text = detail["text"]
                finish_reason = detail["finish_reason"]
                reasoning_chars = detail["reasoning_chars"]
                timeouts_in_a_row = 0
            except Exception as exc:                  # noqa: BLE001 - report, do not crash
                error = f"{type(exc).__name__}: {str(exc)[:160]}"
                print(f"canary {p['id']} failed: {error}", file=sys.stderr)
                timed_out = isinstance(exc, (httpx.TimeoutException, asyncio.TimeoutError))
                timeouts_in_a_row = timeouts_in_a_row + 1 if timed_out else 0
                if 0 < args.canary_max_timeouts <= timeouts_in_a_row:
                    aborted = (f"{timeouts_in_a_row} consecutive request timeouts "
                               f"({args.canary_timeout:g} s each), the last at "
                               f"{p['id']}: the server stopped answering, the "
                               f"remaining prompts were not sent")
                    print(f"canary stopped: {aborted}", file=sys.stderr)
        prefix = prefix_words(text, args.canary_prefix_words)
        # Cut short before the compared prefix was even produced: the budget is
        # too small for this model, not the engine's fault.
        truncated = bool(finish_reason == "length"
                         and len(prefix.split()) < args.canary_prefix_words)
        results.append({"id": p["id"], "text": text, "error": error,
                        "finish_reason": finish_reason,
                        "reasoning_chars": reasoning_chars,
                        "truncated_before_prefix": truncated,
                        "prefix": prefix})
    out = {"model": args.model,
           "reasoning_effort": client.reasoning_effort,
           "max_tokens": args.canary_max_tokens,
           # "the first 16 tokens" is implemented as the first 16
           # whitespace-separated words of the visible content. We do not run the
           # model's tokenizer here; words are what both sides can reproduce.
           "prefix_words": args.canary_prefix_words,
           "prefix_tokens": args.canary_prefix_words,   # old name, same number
           "truncated_before_prefix": sum(
               1 for r in results if r.get("truncated_before_prefix")),
           "results": results}
    if aborted:
        out["aborted"] = aborted
    if args.canary_baseline and Path(args.canary_baseline).exists():
        base = json.loads(Path(args.canary_baseline).read_text())
        by_id = {r["id"]: r["prefix"] for r in base["results"]}
        # An errored or empty reply cannot match anything: it is a mismatch,
        # never a free pass.
        matched = sum(1 for r in results
                      if not r["error"] and r["prefix"]
                      and by_id.get(r["id"]) == r["prefix"])
        out["matched"] = matched
        out["total"] = len(results)
        out["min_match"] = args.canary_min_match
        out["passed"] = matched >= args.canary_min_match
        out["baseline"] = args.canary_baseline
    else:
        out.update(check_canary_baseline(results))
    return out


def check_canary_baseline(results: list[dict]) -> dict:
    """A baseline canary is only usable if all 10 replies are real replies.

    "Real" also means not cut off before the compared prefix: a baseline whose
    answers stop at the token budget freezes a truncation into the file every
    later candidate is measured against.
    """
    bad = [r["id"] for r in results
           if r["error"] or not r["prefix"].strip()
           or r.get("truncated_before_prefix")]
    return {"baseline_valid": not bad, "invalid_ids": bad,
            "total": len(results)}


def prefix_words(text: str, n: int) -> str:
    """The first n whitespace-separated words of the visible content."""
    return " ".join(text.split()[:n])


# Backwards-compatible alias; the unit is words, which is what the name now says.
prefix_tokens = prefix_words


# --------------------------------------------------------------------------
# TPS mode (DESIGN 2.2): tokens/s per chip at fixed ISL/OSL
# --------------------------------------------------------------------------
# The whole measurement is one call: measure_tps(client, spec, tokenizer).
# Scheduling (offline / closed / open), prompt sampling, the nonce, the
# validity rules, client-side token counting and the ramp/window/drain split
# all live behind it; the CLI only builds the spec and writes the result.
TPS_LOADS = ("offline", "closed", "open")
# The client checks itself: an event loop that falls this far behind its own
# timer was the bottleneck, not the server.
LOOP_LAG_WARN_MS = 50.0
LOOP_LAG_TICK_S = 0.01


@dataclass
class TpsSpec:
    """What to measure. `levels` are request counts (offline), concurrencies
    (closed) or request rates per second (open), run in order, one steady
    window each.

    `isl` is one input size, or a list: then every stream of a closed loop
    walks the sizes in the order given, round and round, so two runs of the
    same spec send the same sizes in the same order, and the levels are
    scored on whole-answer speed with every size reported (metrics.py).
    `warm_cycles` full passes over the sizes are sent by every stream before
    the ramp starts: a server that compiles a shape on first use pays for it
    there, however long it takes, and not inside the window."""
    load: str
    levels: list[float]
    isl: int | list[int]
    osl: int
    prompts: list[dict]               # the pool: {"id", "prompt"}
    ramp_s: float = 2.0
    window_s: float = 10.0
    seed: int = 7
    nonce: bool = True                # a fresh reference at the START
    ignore_eos: bool = True
    chips: int = 1
    slo: dict = field(default_factory=lambda: dict(metrics.TPS_DEFAULT_SLO))
    arrival: str = "poisson"          # open loop: poisson | uniform
    length_tol: float = 0.0
    exclude_ids: frozenset = frozenset()
    warm_cycles: int = 0


@dataclass
class TpsResult:
    records: list[dict]
    levels: list[dict]
    headline: float
    self_check: dict


class TpsStreamState(StreamState):
    """A stream accumulator that times every output token.

    Reasoning deltas are output tokens too: they count, and they start the
    TTFT clock. Gaps between token-bearing chunks are kept as itl_ms.
    """

    def __init__(self, t0: float) -> None:
        super().__init__(t0)
        self.reasoning_parts: list[str] = []
        self.t_first: float | None = None
        self.t_last: float | None = None
        self.itl_ms: list[float] = []

    def _token(self, now: float) -> None:
        if self.t_first is None:
            self.t_first = now
            if self.ttft_ms is None:
                self.ttft_ms = (now - self.t0) * 1000.0
        else:
            self.itl_ms.append(round((now - (self.t_last or now)) * 1000.0, 3))
        self.t_last = now

    def on_content(self, delta: str, now: float) -> None:
        if delta:
            self._token(now)
            self.n_content_chunks += 1
            self.text_parts.append(delta)

    def on_reasoning(self, delta: str, now: float) -> None:
        if delta:
            self._token(now)
            self.reasoning_chars += len(delta)
            self.reasoning_parts.append(delta)

    def finish(self, now: float) -> None:
        self.total_ms = (now - self.t0) * 1000.0


class _PromptSampler:
    """Prompts in a seeded random order, each used once per invocation.

    A prompt the server has seen before can be memoised, so a repeat is not a
    fresh measurement: when the pool runs dry the sampler starts again and
    flags every repeat, and the record is invalid (prompt_reused).
    """

    def __init__(self, pool: list[dict], seed: int,
                 exclude: frozenset) -> None:
        self.items = [p for p in pool if str(p["id"]) not in exclude]
        if not self.items:
            raise SystemExit("tps: the prompt pool is empty after exclusions")
        self.rng = random.Random(seed)
        self.order: list[dict] = []
        self.passes = 0
        self.n_reused = 0

    def next(self) -> tuple[dict, bool]:
        if not self.order:
            self.order = list(self.items)
            self.rng.shuffle(self.order)
            self.passes += 1
        reused = self.passes > 1
        self.n_reused += reused
        return self.order.pop(), reused


class _LoopLag:
    """How late the event loop wakes a 10 ms timer: the client self-check."""

    def __init__(self) -> None:
        self.samples: list[float] = []
        self.task: asyncio.Task | None = None

    async def _run(self) -> None:
        while True:
            t = time.monotonic()
            await asyncio.sleep(LOOP_LAG_TICK_S)
            self.samples.append(max(0.0, (time.monotonic() - t
                                          - LOOP_LAG_TICK_S) * 1000.0))

    def __enter__(self) -> "_LoopLag":
        self.task = asyncio.get_running_loop().create_task(self._run())
        return self

    def __exit__(self, *exc: object) -> None:
        if self.task:
            self.task.cancel()

    def stats(self) -> dict:
        return {"loop_lag_max_ms": max(self.samples, default=0.0),
                "loop_lag_p99_ms": metrics.percentile(self.samples, 99) or 0.0,
                "loop_lag_samples": len(self.samples)}


class _TpsRun:
    """One measure_tps invocation: shared clock, sampler and request path."""

    def __init__(self, client: "Client", spec: TpsSpec, tok: Any) -> None:
        self.client, self.spec, self.tok = client, spec, tok
        self.t0 = time.monotonic()
        self.sampler = _PromptSampler(spec.prompts, spec.seed,
                                      spec.exclude_ids)
        self.mixed = isinstance(spec.isl, (list, tuple))
        self.sizes = [int(x) for x in spec.isl] if self.mixed else [int(spec.isl)]

    def size_at(self, k: int) -> int:
        """The input size of a stream's k-th request."""
        return self.sizes[k % len(self.sizes)]

    def rel(self, t: float | None) -> float | None:
        return None if t is None else round(t - self.t0, 6)

    def prompt(self, isl: int) -> tuple[str, str, bool]:
        item, reused = self.sampler.next()
        text = str(item["prompt"])
        if not self.spec.nonce:
            return str(item["id"]), self.tok.fit(text, isl), reused
        head = nonce_prefix()
        body = self.tok.fit(text, max(1, isl - self.tok.count(head)))
        return str(item["id"]), head + body, reused

    def body(self, content: str) -> dict:
        messages = [{"role": "user", "content": content}]
        body = self.client.body(messages, None)
        # extra_body is merged inside client.body(); the measured fields are
        # set after it so nothing can quietly override them.
        body.update({"model": self.client.model, "messages": messages,
                     "stream": True, "max_tokens": self.spec.osl,
                     "stream_options": {"include_usage": True}})
        if self.spec.ignore_eos:
            body["ignore_eos"] = True
        if self.client.chat_template_kwargs is not None:
            body["chat_template_kwargs"] = self.client.chat_template_kwargs
        return body

    async def one(self, load: str, level: float,
                  t_sched: float | None = None, k: int = 0) -> dict:
        isl = self.size_at(k)
        prompt_id, content, reused = self.prompt(isl)
        rec = TurnRecord()
        state = await self.client.stream_turn(
            [], None, rec, body=self.body(content), state_cls=TpsStreamState)
        assert isinstance(state, TpsStreamState)
        text = state.text
        reasoning = "".join(state.reasoning_parts)
        out_text = reasoning + ("\n" if reasoning and text else "") + text
        tokens = self.tok.count(text) + self.tok.count(reasoning)
        usage = state.usage or {}
        d4, cr = metrics.text_sanity(out_text)
        row = {
            "req_id": uuid.uuid4().hex[:16], "prompt_id": prompt_id,
            "load": load, "level": level,
            "t_sched": self.rel(t_sched), "t_send": self.rel(state.t0),
            "t_first": self.rel(state.t_first), "t_last": self.rel(state.t_last),
            "tokens_client": tokens,
            "tokens_usage": usage.get("completion_tokens"),
            # Recorded, not judged: with a nonce at the start only the chat
            # template's own header can be a cache hit.
            "cached_tokens": ((usage.get("prompt_tokens_details") or {})
                              .get("cached_tokens")
                              if isinstance(usage.get("prompt_tokens_details"),
                                            dict) else None),
            "finish_reason": state.finish_reason,
            "itl_ms": state.itl_ms,
            "isl": isl, "isl_client": self.tok.count(content),
            "error_kind": rec.error_kind if rec.error else None,
            "prompt_reused": reused,
            "distinct4": round(d4, 4), "compression": round(cr, 4),
            "text_sha1": hashlib.sha1(out_text.encode("utf-8")).hexdigest(),
        }
        reason = metrics.tps_invalid_reason(row, self.spec.osl,
                                            self.spec.length_tol)
        row["ok"], row["invalid_reason"] = reason is None, reason
        return row

    # ---- loads -----------------------------------------------------------
    async def offline(self, n: int) -> tuple[list[dict], tuple[float, float]]:
        rows = await asyncio.gather(*[self.one("offline", n, k=k)
                                      for k in range(int(n))])
        start = min(r["t_send"] for r in rows)
        end = max((r["t_last"] if r["t_last"] is not None else r["t_send"])
                  for r in rows)
        return list(rows), (start, end)

    async def steady(self, load: str, level: float
                     ) -> tuple[list[dict], tuple[float, float], dict]:
        """Closed or open loop: ramp, steady window, drain.

        Requests sent before the window closes are scored; while any of them
        is still running the load keeps going (drain fillers, sent after the
        window and never scored), so the last scored requests see the same
        concurrency as the first. Fillers are not cancelled: they finish.
        The spec's warm cycles come before all of it and are never scored.
        """
        rows: list[dict] = []
        warm = self.spec.warm_cycles * len(self.sizes) if load == "closed" else 0

        async def warm_stream(w: int) -> None:
            for k in range(warm):
                rows.append(await self.one(load, level, None, w + k))
        await asyncio.gather(*[warm_stream(w) for w in range(int(level))
                               if warm])
        start = time.monotonic()
        w0, w1 = start + self.spec.ramp_s, start + self.spec.ramp_s + \
            self.spec.window_s
        open_scored = 0
        extra: dict = {}

        async def tracked(t_sched: float | None, k: int) -> None:
            nonlocal open_scored
            scored = time.monotonic() < w1
            open_scored += scored
            try:
                rows.append(await self.one(load, level, t_sched, k))
            finally:
                open_scored -= scored

        def keep_going() -> bool:
            return time.monotonic() < w1 or open_scored > 0

        if load == "closed":
            async def worker(w: int) -> None:
                k = w                  # a multiple of the sizes past the warm cycles
                while keep_going():
                    await tracked(None, k)
                    k += 1
            await asyncio.gather(*[worker(w) for w in range(int(level))])
        else:
            rng = random.Random(self.spec.seed * 7919 + int(level * 1000))
            tasks: set[asyncio.Task] = set()
            k, t_next = 0, start
            lags: list[float] = []
            while keep_going():
                # Absolute schedule: send k is due at start + sum of gaps,
                # whatever the previous sends cost (03 bug 3: sleeping a fixed
                # interval after each send drifted below the nominal rate).
                delay = t_next - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                lags.append(max(0.0, (time.monotonic() - t_next) * 1000.0))
                task = asyncio.create_task(tracked(t_next, k))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
                k += 1
                gap = (rng.expovariate(level) if self.spec.arrival == "poisson"
                       else 1.0 / level)
                t_next += gap
            if tasks:
                await asyncio.gather(*list(tasks))
            extra = {"send_lag_p99_ms": metrics.percentile(lags, 99),
                     "send_lag_max_ms": max(lags, default=0.0)}
        window = (w0 - self.t0, w1 - self.t0)
        sent = [r for r in rows if window[0] <= r["t_send"] <= window[1]]
        if load == "open":
            extra["achieved_rate"] = len(sent) / self.spec.window_s
        return rows, window, extra


async def measure_tps(client: "Client", spec: TpsSpec, tokenizer: Any
                      ) -> TpsResult:
    """Run every level of `spec` against `client` and score it.

    Returns every request record and one metrics.tps_level_summary per level,
    plus the headline (metrics.tps_headline) and the client self-check. Never
    reads the server's usage as a count: tokens are `tokenizer.count` of the
    text received.
    """
    if spec.load not in TPS_LOADS:
        raise ValueError(f"tps load must be one of {TPS_LOADS}")
    run = _TpsRun(client, spec, tokenizer)
    if run.mixed and spec.load != "closed":
        raise ValueError("a list of input sizes is measured closed loop only: "
                         "its score is the streams' own speed")
    records: list[dict] = []
    levels: list[dict] = []
    warnings: list[str] = []
    for level in spec.levels:
        with _LoopLag() as lag:
            if spec.load == "offline":
                rows, window = await run.offline(int(level))
                extra: dict = {}
            else:
                rows, window, extra = await run.steady(spec.load, level)
        summary = metrics.tps_level_summary(rows, window, spec.chips, spec.slo,
                                            run.sizes if run.mixed else None)
        summary.update({"load": spec.load, "level": level, **extra,
                        **lag.stats()})
        if summary["loop_lag_max_ms"] > LOOP_LAG_WARN_MS:
            warnings.append(
                f"level {level}: client event loop lagged "
                f"{summary['loop_lag_max_ms']:.0f} ms (> {LOOP_LAG_WARN_MS:.0f}); "
                f"the client may be the bottleneck")
        levels.append(summary)
        records.extend(sorted(rows, key=lambda r: r["t_send"]))
    if run.sampler.n_reused:
        warnings.append(f"prompt pool exhausted: {run.sampler.n_reused} "
                        f"requests reused a prompt and are invalid")
    self_check = {
        "loop_lag_max_ms": max((s["loop_lag_max_ms"] for s in levels),
                               default=0.0),
        "loop_lag_warn_ms": LOOP_LAG_WARN_MS,
        "prompt_reused": run.sampler.n_reused,
        "warnings": warnings,
    }
    return TpsResult(records, levels, metrics.tps_headline(levels, spec.slo),
                     self_check)


# --------------------------------------------------------------------------
# Prompt pool
# --------------------------------------------------------------------------
_POOL_TOPICS = ["a lighthouse keeper", "a city under the sea", "the first "
                "train across a desert", "a market at dawn", "a lost robot",
                "two rival bakers", "a village choir", "a winter expedition",
                "an old clockmaker", "a river that changed course"]
_POOL_WORDS = ("time year people way day man thing woman life child world "
               "school state family student group country problem hand part "
               "place case week company system program question work night "
               "point home water room mother area money story fact month lot "
               "right study book eye job word business issue side kind head "
               "house service friend father power hour game line end member "
               "law car city name team minute idea kid body back parent face "
               "others level office door health person art war history party "
               "result change morning reason research girl guy moment air "
               "teacher force education foot boy age policy music market "
               "sense nation plan college interest death experience effect "
               "class control care field development role effort rate heart "
               "drug show leader light voice wife police mind price report").split()


def make_prompt_pool(n: int, words: int, seed: int = 20261009) -> list[dict]:
    """A synthetic long-form prompt pool for the mock target.

    Each prompt asks for a long continuation and then carries `words` random
    whitespace words, so it can be fitted to any ISL up to that length. Real
    targets use a bench-only pool of natural prompts; this one only has to be
    large and distinct.
    """
    rng = random.Random(seed)
    out = []
    for i in range(int(n)):
        head = (f"Write a long, detailed story about "
                f"{_POOL_TOPICS[i % len(_POOL_TOPICS)]}. Notes:")
        filler = " ".join(rng.choice(_POOL_WORDS) for _ in range(int(words)))
        out.append({"id": f"mock-{i:06d}", "prompt": f"{head} {filler}"})
    return out


def _excluded_ids(paths: list[str]) -> frozenset:
    """prompt_ids already used by earlier invocations (e.g. the warmup)."""
    ids: set[str] = set()
    for path in paths or []:
        for row in metrics.read_jsonl(path):
            if row.get("prompt_id") is not None:
                ids.add(str(row["prompt_id"]))
    return frozenset(ids)


def _levels(value: str | None) -> list[float]:
    if value is None or value == "":
        return []
    vals = [float(v) for v in str(value).split(",") if v.strip()]
    return [int(v) if v.is_integer() else v for v in vals]


def _sizes(value: str | int) -> int | list[int]:
    """--isl: `2048` is one size, `128,1024,8192` a mix."""
    parts = [int(v) for v in str(value).split(",") if v.strip()]
    if not parts or min(parts) < 1:
        raise SystemExit(f"--isl {value!r}: a positive size, or sizes "
                         "comma-separated")
    return parts if "," in str(value) else parts[0]


def tps_spec_from_args(args: argparse.Namespace) -> TpsSpec:
    """The CLI's flags as a TpsSpec (the only place that reads them)."""
    load = args.tps_load
    levels = _levels({"offline": args.requests, "closed": args.concurrency,
                      "open": args.rate}[load])
    if not levels:
        raise SystemExit(f"--tps-load {load} needs "
                         + {"offline": "--requests", "closed": "--concurrency",
                            "open": "--rate"}[load])
    if not args.prompt_pool:
        raise SystemExit("--mode tps needs --prompt-pool")
    slo = json.loads(args.slo) if args.slo else dict(metrics.TPS_DEFAULT_SLO)
    return TpsSpec(
        load=load, levels=levels, isl=_sizes(args.isl), osl=args.osl,
        prompts=metrics.read_jsonl(args.prompt_pool),
        ramp_s=args.ramp_s, window_s=args.window_s, seed=args.seed,
        nonce=args.nonce != "none", ignore_eos=args.ignore_eos,
        chips=args.chips, slo=slo, arrival=args.arrival,
        length_tol=args.length_tol,
        exclude_ids=_excluded_ids(args.pool_exclude),
        warm_cycles=args.warm_cycles)


def _sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


async def run_tps_cli(args: argparse.Namespace, out_dir: Path) -> int:
    """--mode tps: build the spec, measure, write tps_records.jsonl and
    summary.json."""
    from tokenizer import load_tokenizer
    started = time.time()
    spec = tps_spec_from_args(args)
    tok = load_tokenizer(args.tokenizer)
    async with Client(args) as client:
        try:
            result = await measure_tps(client, spec, tok)
        except ValueError as exc:
            raise SystemExit(f"--mode tps: {exc}") from exc
    best = metrics.tps_headline_level(result.levels, spec.slo) or {}
    metrics.write_jsonl(str(out_dir / "tps_records.jsonl"), result.records)
    summary = {
        "harness_version": HARNESS_VERSION,
        "metrics_version": metrics.METRICS_VERSION,
        "run_id": args.run_id, "label": args.label, "phase": args.phase,
        "mode": "tps", "base_url": args.base_url, "model": args.model,
        "load": spec.load, "levels_requested": spec.levels,
        "isl": spec.isl, "osl": spec.osl, "ramp_s": spec.ramp_s,
        "warm_cycles": spec.warm_cycles,
        "window_s": spec.window_s, "chips": spec.chips, "seed": spec.seed,
        "arrival": spec.arrival if spec.load == "open" else None,
        "nonce": "turn" if spec.nonce else "none",
        "ignore_eos": spec.ignore_eos, "length_tol": spec.length_tol,
        "temperature": args.temperature,
        "reasoning_effort": args.reasoning_effort,
        "chat_template_kwargs": (json.loads(args.chat_template_kwargs)
                                 if args.chat_template_kwargs else None),
        "tokenizer": {"name": tok.name, "sha256": tok.fingerprint()},
        "prompt_pool": {"path": args.prompt_pool,
                        "sha256": _sha256_file(args.prompt_pool),
                        "n": len(spec.prompts),
                        "n_excluded": len(spec.exclude_ids)},
        "slo": spec.slo,
        # beside the per-chip headline, one stream's own speed at the level
        # the headline came from: the whole answer (what a mix of sizes is
        # scored on), and decode only
        "metric": {"kind": "tps", "headline": result.headline,
                   "decode_tps": best.get("decode_tps"),
                   "stream_tps": best.get("stream_tps"),
                   **({"by_isl": best["by_isl"]} if "by_isl" in best else {}),
                   "levels": result.levels},
        "self_check": result.self_check,
        "started": started, "finished": time.time(),
        "wall_s": time.time() - started,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    for w in result.self_check["warnings"]:
        print(f"[harness] WARNING: {w}", file=sys.stderr)
    for s in result.levels:
        print(f"[harness] tps {spec.load} level={s['level']:g} "
              f"goodput/chip={s['goodput_tps_per_chip']:.1f} "
              f"tps/chip={s['tps_per_chip']:.1f} "
              f"p99_tpot={_fmt(s['p99_tpot_ms'])} "
              f"p95_ttft={_fmt(s['p95_ttft_ms'])} ms "
              f"invalid={s['n_invalid']}/{s['n']} slo_met={s['slo_met']}")
        for isl, v in (s.get("by_isl") or {}).items():
            print(f"[harness] tps   isl={isl} n={v['n']} "
                  f"decode={_fmt1(v['decode_tps'])} "
                  f"stream={_fmt1(v['stream_tps'])} tok/s "
                  f"p50_ttft={_fmt(v['p50_ttft_ms'])} ms")
    print(f"[harness] tps headline={result.headline:.1f} tok/s/chip "
          f"(one stream: {_fmt1(best.get('stream_tps'))} tok/s for the whole "
          f"answer, {_fmt1(best.get('decode_tps'))} decoding) -> {out_dir}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def nonce_prefix() -> str:
    """A fresh reference at the very start of a request, so nothing in it can
    be a prompt-cache hit."""
    return f"Request reference: {uuid.uuid4()}.\n\n"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Load harness for OpenAI-compatible streaming endpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                    help="OpenAI-compatible base URL, including /v1")
    ap.add_argument("--model", default="mock",
                    help="the name the server serves the model under")
    ap.add_argument("--mode", choices=["tps", "canary", "pool"], default="tps",
                    help="tps: tokens/s per chip at fixed ISL/OSL (see the "
                         "TPS flags); canary: the ten fixed prompts; pool: "
                         "write a synthetic prompt pool "
                         "(<out>/prompt_pool.jsonl) for the mock target")
    ap.add_argument("--out", default="results/local",
                    help="output directory")
    ap.add_argument("--api-key-env", default=None,
                    help="env var holding the bearer token")
    ap.add_argument("--extra-body", default="",
                    help="JSON object merged into every request body "
                         "(e.g. a router's provider pinning)")
    ap.add_argument("--header", action="append", default=[],
                    help="extra header, 'Name: value'; repeatable")
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["low", "medium", "high"],
                    help="sent as reasoning_effort in the request body")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--first-chunk-timeout", type=float,
                    default=metrics.FIRST_CHUNK_TIMEOUT_S)
    ap.add_argument("--total-timeout", type=float,
                    default=metrics.TOTAL_TIMEOUT_S)
    ap.add_argument("--no-stream-usage", action="store_true",
                    help="do not ask for usage in the final chunk")
    ap.add_argument("--nonce", choices=["none", "turn"], default="turn",
                    help="turn: a unique prefix on every tps request (nothing "
                         "cacheable); none: send the prompt as it is")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--phase", default="",
                    help="label for summary.json (the launch name)")
    ap.add_argument("--run-id", default="")
    ap.add_argument("--label", default="", help="free-text label for summary.json")
    ap.add_argument("--canary-prompts", default="canary/prompts.json")
    ap.add_argument("--canary-baseline", default=None,
                    help="compare against this canary file instead of writing one")
    ap.add_argument("--canary-max-tokens", type=int, default=CANARY_MAX_TOKENS,
                    help="generation budget per canary prompt")
    ap.add_argument("--canary-prefix-words", "--canary-prefix-tokens",
                    dest="canary_prefix_words", type=int,
                    default=CANARY_PREFIX_WORDS,
                    help="how many leading whitespace words are compared")
    ap.add_argument("--canary-timeout", type=float, default=CANARY_TIMEOUT_S,
                    help="seconds one canary request may take")
    ap.add_argument("--canary-max-timeouts", type=int, default=CANARY_MAX_TIMEOUTS,
                    help="stop the canary after this many request timeouts "
                         "in a row (0 = never)")
    ap.add_argument("--canary-min-match", type=int, default=CANARY_MIN_MATCH,
                    help="a candidate needs at least this many of 10 to match")
    ap.add_argument("--chat-template-kwargs", default="",
                    help="JSON object sent as chat_template_kwargs on every "
                         "request (e.g. the thinking switch), and recorded")
    tps = ap.add_argument_group(
        "tps mode", "One invocation runs every level in order (a sweep) and "
        "writes <out>/tps_records.jsonl and <out>/summary.json.")
    tps.add_argument("--tps-load", choices=list(TPS_LOADS), default="closed")
    tps.add_argument("--rate", default="8.0",
                     help="open loop: request rates per second, "
                          "comma-separated, one level each")
    tps.add_argument("--concurrency", default="",
                     help="closed loop: concurrencies, comma-separated (4,8,16)")
    tps.add_argument("--requests", default="",
                     help="offline: requests submitted at once, comma-separated")
    tps.add_argument("--arrival", choices=["poisson", "uniform"],
                     default="poisson", help="open loop inter-arrival times")
    tps.add_argument("--isl", default="256",
                     help="prompt tokens per request, nonce included; several "
                          "sizes comma-separated are a balanced mix "
                          "(closed loop, scored on whole-answer speed)")
    tps.add_argument("--warm-cycles", type=int, default=0,
                     help="closed loop: passes over the sizes every stream "
                          "sends before the ramp starts, unscored")
    tps.add_argument("--osl", type=int, default=64,
                     help="output tokens per request (max_tokens)")
    tps.add_argument("--ramp-s", type=float, default=2.0)
    tps.add_argument("--window-s", type=float, default=10.0)
    tps.add_argument("--prompt-pool", default="",
                     help='JSONL of {"id","prompt"}; sampled per run with --seed')
    tps.add_argument("--pool-exclude", action="append", default=[],
                     help="tps_records.jsonl whose prompt_ids must not be "
                          "reused (e.g. the warmup's); repeatable")
    tps.add_argument("--pool-size", type=int, default=4096,
                     help="--mode pool: prompts to write")
    tps.add_argument("--tokenizer", default="ws",
                     help="ws, or the path to a tokenizer.json")
    tps.add_argument("--ignore-eos", action=argparse.BooleanOptionalAction,
                     default=True, help="send ignore_eos: true")
    tps.add_argument("--chips", type=int, default=1,
                     help="chips the server runs on (from the target, never "
                          "from the candidate)")
    tps.add_argument("--slo", default="",
                     help="JSON: p99_tpot_ms, p95_ttft_ms, and optionally the "
                          "sanity band min_distinct4, max_compression")
    tps.add_argument("--length-tol", type=float, default=0.0,
                     help="allowed |tokens_client - osl| / osl (0 for ws)")
    return ap


async def amain(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "tps":
        return await run_tps_cli(args, out_dir)
    if args.mode == "pool":
        metrics.write_jsonl(str(out_dir / "prompt_pool.jsonl"),
                            make_prompt_pool(
                                args.pool_size,
                                2 * max(int(v) for v in str(args.isl).split(",")),
                                args.seed))
        print(f"[harness] {args.pool_size} prompts -> "
              f"{out_dir / 'prompt_pool.jsonl'}")
        return 0
    async with Client(args) as client:
        result = await run_canary(client, args)
    (out_dir / "canary.json").write_text(json.dumps(result, indent=2))
    if "matched" in result:
        print(f"canary matched {result['matched']}/{result['total']} "
              f"(need {result['min_match']})")
        return 0
    if not result["baseline_valid"]:
        # Refuse to become the thing every later run is compared with.
        print(f"canary baseline REFUSED: {len(result['invalid_ids'])} of "
              f"{result['total']} prompts returned an error or no visible "
              f"content ({', '.join(result['invalid_ids'])}). Fix the "
              f"server, the chat template or --canary-max-tokens and run "
              f"the baseline again; a canary baseline with a blank answer "
              f"in it makes every later experiment fail for the wrong "
              f"reason. See {out_dir / 'canary.json'}.", file=sys.stderr)
        return 3
    print(f"canary baseline written to {out_dir / 'canary.json'} "
          f"({result['total']}/{result['total']} prompts answered)")
    return 0


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0f}"


def _fmt1(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.run_id:
        args.run_id = uuid.uuid4().hex[:12]
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
