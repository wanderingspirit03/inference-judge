#!/usr/bin/env python3
"""Probe a live OpenAI-compatible server for what the ruler needs from it.

Run it in the first minutes of a paid session, against the engine that will be
measured. Standard library only.

    tools/tpu_probe_logprobs.py --base-url http://127.0.0.1:8000 --model NAME

One line per check, PASS / FAIL / WARN / INFO, the raw response after every
FAIL, and a non-zero exit if any check FAILed (2 if the server is unreachable).

  echo_logprobs     /v1/completions accepts echo + logprobs and returns a
                    logprob entry for every prompt token. This is the request
                    quality2.py teacher-forces with: {"prompt": prompt +
                    continuation, "max_tokens": 1, "temperature": 0.0,
                    "echo": true, "logprobs": 20}. quality2 reads
                    choices[0].prompt_logprobs (vLLM) if present, else
                    choices[0].logprobs {tokens, top_logprobs, text_offset}.
  top_logprobs      how many alternatives each position carries (needs >= 20).
                    INFO lines say whether a larger logprobs value is accepted.
  chat_stream_usage /v1/chat/completions streams, and the last chunk carries
                    usage (stream_options.include_usage), as harness.py needs.
  exact_length      max_tokens + ignore_eos gives exactly that many tokens and
                    finish_reason "length", on a prompt whose natural answer is
                    short. Checked on chat (the speed test) and on completions
                    (quality2's greedy). The probe has no tokenizer, so the
                    count it trusts is the server's usage.completion_tokens;
                    the ruler recounts with the real tokenizer.
  determinism       two identical temperature-0 requests, one after the other,
                    return byte-identical text (completions and chat).
  determinism_concurrent  the same two requests sent at once. WARN, not FAIL:
                    it tells you whether batching changes the answer.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field

TOP_K = 20                    # quality2.TOP_K: what the ruler asks for
RAW_LIMIT = 3000              # characters of raw response to print on FAIL
PROMPT = "The three primary colours of light are red, green and"
SHORT_ANSWER = "Reply with the single word: yes."
STORY = "Write a long, detailed story about a lighthouse keeper and a storm."


class Unreachable(Exception):
    pass


@dataclass
class Result:
    name: str
    status: str                       # PASS | FAIL | WARN | INFO
    detail: str
    raw: str = field(default="", repr=False)


class Server:
    """The few HTTP calls the probe makes. `base_url` may end in /v1 or not."""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0,
                 api_key: str | None = None, chat_template_kwargs: dict | None = None) -> None:
        root = base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        self.root, self.model, self.timeout = root, model, timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.chat_template_kwargs = chat_template_kwargs

    def _open(self, req: urllib.request.Request):
        try:
            return urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            return exc                          # has .code and .read()
        except (urllib.error.URLError, OSError) as exc:
            raise Unreachable(f"{req.full_url}: {exc}") from exc

    def get(self, path: str) -> tuple[int, str]:
        resp = self._open(urllib.request.Request(self.root + path, headers=self.headers))
        return resp.getcode(), resp.read().decode("utf-8", "replace")

    def post(self, path: str, body: dict) -> tuple[int, dict | None, str]:
        data = json.dumps({"model": self.model, **body}).encode()
        resp = self._open(urllib.request.Request(self.root + path, data=data, headers=self.headers,
                                                 method="POST"))
        text = resp.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        return resp.getcode(), parsed, text

    def chat_body(self, content: str, **extra) -> dict:
        body = {"messages": [{"role": "user", "content": content}], "temperature": 0.0, **extra}
        if self.chat_template_kwargs is not None:
            body["chat_template_kwargs"] = self.chat_template_kwargs
        return body

    def stream_chat(self, body: dict) -> tuple[int, list[str], list[dict]]:
        """(status, raw SSE lines, parsed data chunks)."""
        data = json.dumps({"model": self.model, **body}).encode()
        resp = self._open(urllib.request.Request(self.root + "/v1/chat/completions", data=data,
                                                 headers=self.headers, method="POST"))
        lines, chunks = [], []
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not line:
                continue
            lines.append(line)
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                try:
                    chunks.append(json.loads(line[5:]))
                except json.JSONDecodeError:
                    pass
        return resp.getcode(), lines, chunks


def _clip(text: str) -> str:
    return text if len(text) <= RAW_LIMIT else text[:RAW_LIMIT] + f"... [{len(text) - RAW_LIMIT} more chars]"


def _finite_nonpositive(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x) and x <= 1e-3


# ---------------------------------------------------------------------------
# reading the echo response the way quality2.HttpCandidate does
# ---------------------------------------------------------------------------
def _entry_top(entry) -> dict | None:
    """{token: logprob} of one position, in the OpenAI, vLLM or bench-mock shape."""
    if not entry:
        return None
    if isinstance(entry.get("top_logprobs"), dict):
        out = {str(t): v for t, v in entry["top_logprobs"].items()}
        if "token" in entry and entry.get("logprob") is not None:
            out.setdefault(str(entry["token"]), entry["logprob"])
        return out
    out = {}
    for key, val in entry.items():
        out[str(val.get("decoded_token", key)) if isinstance(val, dict) else str(key)] = \
            val["logprob"] if isinstance(val, dict) else val
    return out


def read_echo(choice: dict, full: str, start: int, text: str, prompt_tokens: int | None) -> dict:
    """What a teacher-forcing client can take from `choice`: the shape, the
    entries for the continuation, and what is wrong with them."""
    problems: list[str] = []
    if choice.get("prompt_logprobs"):
        shape, entries = "prompt_logprobs", choice["prompt_logprobs"]
        tops = [_entry_top(e) for e in entries]
        if prompt_tokens is not None and len(entries) != prompt_tokens:
            problems.append(f"{len(entries)} prompt_logprobs entries for {prompt_tokens} prompt tokens")
        cont = tops[1:]                                  # entry 0 has nothing before it
    else:
        lp = choice.get("logprobs") or {}
        tokens, top_list = lp.get("tokens"), lp.get("top_logprobs")
        if not tokens or not top_list:
            return {"shape": None, "problems": ["response has neither prompt_logprobs nor "
                                                "logprobs.tokens/top_logprobs"], "tops": [], "n": 0}
        shape = "logprobs"
        offsets = lp.get("text_offset")
        if prompt_tokens is not None and len(tokens) < prompt_tokens:
            problems.append(f"{len(tokens)} echoed tokens for {prompt_tokens} prompt tokens")
        if offsets:
            # a token that carries the joining space starts before `start` but ends after it
            idx = [i for i, off in enumerate(offsets) if start < off + len(tokens[i]) and off < len(full)]
            joined = "".join(tokens[i] for i in idx)
            if not idx or not joined.replace(" ", "").endswith(text.strip().replace(" ", "")):
                problems.append(f"tokens at text_offset >= {start} are {joined!r}, expected the continuation "
                                f"{text!r}")
        else:
            problems.append("no text_offset: the continuation cannot be located (quality2 falls back to "
                            "counting from the end)")
        cont = [_entry_top(top_list[i]) if i < len(top_list) else None
                for i in (idx if offsets else range(1, len(tokens)))]
        tops = [_entry_top(e) for e in top_list]
    empty = sum(1 for t in cont if not t)
    if empty:
        problems.append(f"{empty} continuation positions have no top_logprobs")
    sizes = [len(t) for t in cont if t]
    bad = [v for t in cont if t for v in t.values() if not _finite_nonpositive(v)]
    if bad:
        problems.append(f"{len(bad)} logprob values are not finite and <= 0 (first: {bad[0]!r})")
    return {"shape": shape, "problems": problems, "tops": tops, "n": len(tops),
            "sizes": sizes, "n_cont": len(cont)}


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------
def _usage(parsed: dict | None) -> dict:
    return (parsed or {}).get("usage") or {}


def _completion_text(parsed: dict | None) -> str | None:
    try:
        return parsed["choices"][0]["text"]
    except (TypeError, KeyError, IndexError):
        return None


def _join(prompt: str, text: str) -> tuple[str, int]:
    """quality2.join_continuation."""
    sep = "" if (not prompt or not text or prompt[-1].isspace() or text[0].isspace()) else " "
    return prompt + sep + text, len(prompt) + len(sep)


def check_echo_logprobs(srv: Server, k: int = TOP_K) -> list[Result]:
    code, parsed, raw = srv.post("/v1/completions", {
        "prompt": PROMPT, "max_tokens": 16, "temperature": 0.0, "ignore_eos": True})
    text = _completion_text(parsed)
    if code != 200 or text is None:
        return [Result("echo_logprobs", "FAIL", f"could not get a greedy continuation first: HTTP {code}", raw)]
    full, start = _join(PROMPT, text)
    code, parsed, raw = srv.post("/v1/completions", {
        "prompt": full, "max_tokens": 1, "temperature": 0.0, "echo": True, "logprobs": k})
    if code != 200 or not isinstance(parsed, dict) or not parsed.get("choices"):
        return [Result("echo_logprobs", "FAIL", f"HTTP {code} for echo + logprobs={k} (quality2 reads a 4xx "
                       f"as 'logprobs unavailable' and the gate then fails closed)", raw),
                Result("top_logprobs", "FAIL", "no response to count", raw)]
    info = read_echo(parsed["choices"][0], full, start, text, _usage(parsed).get("prompt_tokens"))
    out: list[Result] = []
    if info["shape"] is None or info["problems"]:
        out.append(Result("echo_logprobs", "FAIL", "; ".join(info["problems"]), raw))
    else:
        out.append(Result("echo_logprobs", "PASS",
                          f"{info['shape']} shape, {info['n']} entries for the prompt, "
                          f"{info['n_cont']} scored positions after the first token"))
    sizes = info.get("sizes") or []
    if not sizes:
        out.append(Result("top_logprobs", "FAIL", "no top_logprobs to count", raw))
    else:
        lo, med = min(sizes), sorted(sizes)[len(sizes) // 2]
        status = "PASS" if lo >= k else "FAIL"
        out.append(Result("top_logprobs", status, f"{lo} alternatives at the thinnest position, median {med} "
                          f"(asked for {k}; the ruler needs >= {k})", "" if status == "PASS" else raw))
    for bigger in (k + 30,):
        code, parsed2, raw2 = srv.post("/v1/completions", {
            "prompt": full, "max_tokens": 1, "temperature": 0.0, "echo": True, "logprobs": bigger})
        if code == 200 and parsed2 and parsed2.get("choices"):
            sz = read_echo(parsed2["choices"][0], full, start, text, None).get("sizes") or [0]
            out.append(Result("top_logprobs_max", "INFO", f"logprobs={bigger} accepted, {min(sz)} alternatives"))
        else:
            out.append(Result("top_logprobs_max", "INFO", f"logprobs={bigger} refused with HTTP {code}: "
                              f"the limit is below {bigger}"))
    return out


def check_chat_stream(srv: Server, osl: int) -> list[Result]:
    body = srv.chat_body(SHORT_ANSWER, stream=True, max_tokens=osl, ignore_eos=True,
                         stream_options={"include_usage": True})
    code, lines, chunks = srv.stream_chat(body)
    raw = "\n".join(lines)
    if code != 200 or not chunks:
        return [Result("chat_stream_usage", "FAIL", f"HTTP {code}, {len(chunks)} data chunks", raw),
                Result("exact_length", "FAIL", "no stream to measure", raw)]
    n_token_chunks, finish, usage = 0, None, None
    for ch in chunks:
        usage = ch.get("usage") or usage
        for c in ch.get("choices") or []:
            d = c.get("delta") or {}
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                n_token_chunks += 1
            finish = c.get("finish_reason") or finish
    last = chunks[-1]
    out = []
    problems = []
    if not n_token_chunks:
        problems.append("no content chunks")
    if not isinstance(usage, dict) or "completion_tokens" not in usage:
        problems.append("no usage with completion_tokens in any chunk")
    elif not last.get("usage"):
        problems.append("usage is not in the final chunk")
    if not any(ln.strip() == "data: [DONE]" for ln in lines):
        problems.append("no data: [DONE]")
    out.append(Result("chat_stream_usage", "FAIL" if problems else "PASS",
                      "; ".join(problems) or f"{n_token_chunks} token chunks, usage in the final chunk: {usage}",
                      raw if problems else ""))
    n = (usage or {}).get("completion_tokens")
    bad = []
    if finish != "length":
        bad.append(f"finish_reason {finish!r}, expected 'length'")
    if n != osl:
        bad.append(f"usage.completion_tokens {n}, expected {osl}")
    if bad:
        # name the alternative mechanism, in case ignore_eos is what failed
        alt_body = srv.chat_body(SHORT_ANSWER, stream=False, max_tokens=osl, min_tokens=osl)
        c2, p2, _ = srv.post("/v1/chat/completions", alt_body)
        alt = _usage(p2).get("completion_tokens") if c2 == 200 else None
        bad.append(f"min_tokens={osl} instead gives {alt} tokens (HTTP {c2})")
    out.append(Result("exact_length", "FAIL" if bad else "PASS",
                      "; ".join(bad) or f"chat: {n} tokens, finish_reason length, in {n_token_chunks} chunks "
                                        f"(usage trusted: the probe has no tokenizer)",
                      raw if bad else ""))
    code, parsed, raw_c = srv.post("/v1/completions", {
        "prompt": "Answer with one word. Is water wet? Answer:", "max_tokens": osl, "temperature": 0.0,
        "ignore_eos": True})
    u = _usage(parsed).get("completion_tokens")
    fr = ((parsed or {}).get("choices") or [{}])[0].get("finish_reason") if parsed else None
    if code == 200 and u == osl and fr == "length":
        out.append(Result("exact_length_completions", "PASS", f"completions: {u} tokens, finish_reason length"))
    else:
        out.append(Result("exact_length_completions", "FAIL",
                          f"HTTP {code}, completion_tokens {u}, finish_reason {fr!r}, expected {osl} / 'length'",
                          raw_c))
    return out


def check_determinism(srv: Server, n_tokens: int = 48) -> list[Result]:
    def comp() -> str | None:
        return _completion_text(srv.post("/v1/completions", {
            "prompt": STORY, "max_tokens": n_tokens, "temperature": 0.0, "ignore_eos": True})[1])

    def chat() -> str | None:
        try:
            p = srv.post("/v1/chat/completions", srv.chat_body(STORY, max_tokens=n_tokens, ignore_eos=True))[1]
            return p["choices"][0]["message"]["content"]
        except (TypeError, KeyError, IndexError):
            return None

    out = []
    pairs = {}
    for name, fn in (("completions", comp), ("chat", chat)):
        a, b = fn(), fn()
        pairs[name] = (a, b)
    bad = [f"{n}: {'no text' if a is None or b is None else 'texts differ'}" for n, (a, b) in pairs.items()
           if a is None or b is None or a != b]
    raw = json.dumps(pairs, ensure_ascii=False)
    out.append(Result("determinism", "FAIL" if bad else "PASS",
                      "; ".join(bad) or f"two sequential temperature-0 requests are byte-identical "
                                        f"(completions and chat, {n_tokens} tokens)", raw if bad else ""))
    got: list[str | None] = [None, None]

    def worker(i: int) -> None:
        got[i] = comp()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    same = got[0] is not None and got[0] == got[1] and got[0] == pairs["completions"][0]
    out.append(Result("determinism_concurrent", "PASS" if same else "WARN",
                      "two simultaneous requests match the sequential answer" if same else
                      "simultaneous requests differ from each other or from the sequential answer: batching "
                      "changes the output, so expect R-vs-R noise", "" if same else json.dumps(got)))
    return out


def run_probe(srv: Server, osl: int = 64) -> list[Result]:
    code, raw = srv.get("/v1/models")
    results = [Result("reachable", "PASS" if code == 200 else "FAIL", f"GET /v1/models -> HTTP {code}",
                      "" if code == 200 else raw)]
    if code != 200:
        return results
    for check in (lambda: check_echo_logprobs(srv), lambda: check_chat_stream(srv, osl),
                  lambda: check_determinism(srv)):
        results += check()
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8000 (a trailing /v1 is fine)")
    ap.add_argument("--model", required=True, help="the served model name")
    ap.add_argument("--osl", type=int, default=64, help="output length for the length check (the speed test "
                    "uses 512; pass --osl 512 to check that too)")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--api-key-env", default=None, help="name of an env var holding a bearer token")
    ap.add_argument("--chat-template-kwargs", default=None, help="JSON sent as chat_template_kwargs on chat "
                    "requests, as the harness's --chat-template-kwargs does")
    a = ap.parse_args(argv)
    srv = Server(a.base_url, a.model, a.timeout, os.environ.get(a.api_key_env) if a.api_key_env else None,
                 json.loads(a.chat_template_kwargs) if a.chat_template_kwargs else None)
    try:
        results = run_probe(srv, a.osl)
    except Unreachable as exc:
        print(f"FAIL reachable: {exc}")
        return 2
    for r in results:
        print(f"{r.status} {r.name}: {r.detail}")
        if r.status == "FAIL" and r.raw:
            print("  raw response:\n    " + _clip(r.raw).replace("\n", "\n    "))
    n_fail = sum(r.status == "FAIL" for r in results)
    print(f"{'FAIL' if n_fail else 'PASS'}: {n_fail} failed, "
          f"{sum(r.status == 'PASS' for r in results)} passed, {sum(r.status == 'WARN' for r in results)} warnings")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
