"""The HTTP side of ./check: one streamed request to an OpenAI-compatible
server, and a closed loop of such requests at a fixed concurrency.  Stdlib
only, so the installed check needs nothing the builder could change.

Every request asks for greedy decoding, the logprob of each chosen token
(plus a few top alternatives) and token ids in place of token text
(`return_tokens_as_token_ids`).  CHECK.md lists what the engine must accept.
"""
from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.parse

from compare import parse_token

ENGINE_TOP = 5          # top logprobs asked of the engine at each position


class EngineError(Exception):
    pass


def _conn(base: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    u = urllib.parse.urlsplit(base)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    return cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout), u.path.rstrip("/")


def get_json(base: str, path: str, key: str | None = None, timeout: float = 10.0):
    c, prefix = _conn(base, timeout)
    try:
        h = {"Authorization": "Bearer " + key} if key else {}
        c.request("GET", prefix + path, headers=h)
        r = c.getresponse()
        body = r.read()
        if r.status != 200:
            raise EngineError(f"GET {path}: HTTP {r.status}")
        return json.loads(body)
    finally:
        c.close()


def build_body(model: str, endpoint: str, payload, max_tokens: int, top: int = ENGINE_TOP,
               stream: bool = True) -> dict:
    """endpoint 'chat': payload is the messages list.  'completions': payload is
    a prompt string or a list of token ids."""
    body = {"model": model, "max_tokens": int(max_tokens), "temperature": 0.0,
            "return_tokens_as_token_ids": True, "stream": stream}
    if stream:
        body["stream_options"] = {"include_usage": True}
    if endpoint == "chat":
        body["messages"] = payload
        body["logprobs"] = True
        body["top_logprobs"] = top
    else:
        body["prompt"] = payload
        body["logprobs"] = top
    return body


def _entries(endpoint: str, lp) -> list[tuple[int, float | None, dict]]:
    """(token id, its logprob, {id: logprob} of the top alternatives) for each
    token in one logprobs object of either endpoint."""
    out = []
    if not lp:
        return out
    if endpoint == "chat":
        for e in lp.get("content") or []:
            top = {parse_token(t["token"]): float(t["logprob"]) for t in (e.get("top_logprobs") or [])}
            out.append((parse_token(e["token"]), None if e.get("logprob") is None else float(e["logprob"]), top))
    else:
        toks = lp.get("tokens") or []
        lps = lp.get("token_logprobs") or []
        tops = lp.get("top_logprobs") or []
        for i, t in enumerate(toks):
            top = {parse_token(k): float(v) for k, v in ((tops[i] or {}).items() if i < len(tops) else [])}
            out.append((parse_token(t), None if i >= len(lps) or lps[i] is None else float(lps[i]), top))
    return out


def request(base: str, endpoint: str, body: dict, key: str | None = None,
            timeout: float = 1800.0) -> dict:
    """Send one request; return {tokens, lps, tops, finish, usage, t0, ttft, t_end,
    prompt_ids}.  Raises EngineError on any HTTP or format problem."""
    path = "/v1/chat/completions" if endpoint == "chat" else "/v1/completions"
    c, prefix = _conn(base, timeout)
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    toks, lps, tops = [], [], []
    finish, usage, ttft, prompt_ids = None, None, None, None
    t0 = time.monotonic()
    try:
        c.request("POST", prefix + path, body=json.dumps(body).encode(), headers=h)
        r = c.getresponse()
        if r.status != 200:
            raise EngineError(f"HTTP {r.status}: {r.read()[:300].decode('utf-8', 'replace')}")

        def take(doc):
            nonlocal finish, usage, ttft, prompt_ids
            if doc.get("usage"):
                usage = doc["usage"]
            if doc.get("prompt_token_ids"):
                prompt_ids = doc["prompt_token_ids"]
            for ch in doc.get("choices") or []:
                got = _entries(endpoint, ch.get("logprobs"))
                if not got and isinstance(ch.get("token_ids"), list):
                    # token ids without logprobs (the judge uses the engine's logprobs for no verdict)
                    got = [(int(t), None, {}) for t in ch["token_ids"]]
                if got and ttft is None:
                    ttft = time.monotonic() - t0
                for t, l, tp in got:
                    toks.append(t); lps.append(l); tops.append(tp)
                if ch.get("prompt_token_ids"):
                    prompt_ids = ch["prompt_token_ids"]
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]

        if body.get("stream"):
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                doc = json.loads(data)
                if isinstance(doc, dict) and doc.get("error"):
                    raise EngineError(f"error in stream: {str(doc['error'])[:300]}")
                take(doc)
        else:
            take(json.loads(r.read()))
    except (OSError, http.client.HTTPException, ValueError) as e:
        raise EngineError(f"{type(e).__name__}: {e}"[:400]) from None
    finally:
        c.close()
    t_end = time.monotonic()
    if not toks and finish != "stop":
        raise EngineError("the answer carried no tokens (send logprobs.tokens as token_id:<n>, "
                          "or choices[0].token_ids)")
    # usage is not trusted for anything; a count that disagrees with the
    # tokens is a sign the logprobs leave tokens out (a stop token may be
    # counted without an entry).
    n = usage.get("completion_tokens") if usage else None
    if n is not None and not (n == len(toks) or (finish == "stop" and n == len(toks) + 1)):
        raise EngineError(f"usage says {n} tokens, logprobs cover {len(toks)}")
    return {"tokens": toks, "lps": lps, "tops": tops, "finish": finish, "usage": usage,
            "t0": t0, "ttft": ttft, "t_end": t_end, "prompt_ids": prompt_ids}


def closed_loop(jobs: list, concurrency: int, fn) -> list:
    """Run fn(job) for every job with `concurrency` in flight at all times
    (a new one starts as soon as one finishes).  Returns the results in job
    order; a failed job's result is the exception."""
    out: list = [None] * len(jobs)
    nxt = [0]
    lock = threading.Lock()

    def worker():
        while True:
            with lock:
                i = nxt[0]
                nxt[0] += 1
            if i >= len(jobs):
                return
            try:
                out[i] = fn(jobs[i])
            except Exception as e:  # noqa: BLE001 - recorded per job
                out[i] = e

    ts = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, concurrency))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return out
