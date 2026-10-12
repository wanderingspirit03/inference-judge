#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "numpy>=1.26", "tokenizers>=0.23"]
# ///
"""Measure how fast an OpenAI-style endpoint answers. No target file, no
Docker, no agent: one command, one table.

    measure.py URL --model NAME
               [--tokenizer auto|ws|server|FILE|hf:REPO]
               [--input 256,2048] [--output 256] [--users 1] [--seconds 15]
               [--api-key-env VAR] [--json FILE]

It sends the same load the judge sends (harness.measure_tps: fresh prompts,
a nonce at the start of each, streaming) and reads the result with the same
arithmetic (metrics.tps_level_summary). Two things differ from a judged run,
and the output says which applies:

  who counts   The judge counts tokens itself and never trusts the server's
               `usage`. So does this, when it has the model's tokenizer:
               a tokenizer.json (FILE), one fetched from the Hugging Face Hub
               (hf:REPO), or `auto`, which tries hf:<model name>. Without one
               it falls back to the server's own count and labels it.
  answer length  The judge asks for answers of a fixed length (`ignore_eos`),
               so every request does the same work. A server that does not
               honour that (most hosted APIs) is measured at whatever length
               it sends, and the table says "free length".

A number from this command is a measurement, not a verdict: nothing was
sealed and nothing was compared. Exit status: 0 measured, 1 no valid answer
at some input size, 2 the endpoint could not be measured at all.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness  # noqa: E402
import metrics  # noqa: E402
from tokenizer import load_tokenizer  # noqa: E402

LENGTH_TOL = 0.05          # a real tokenizer re-counting decoded text may be a little off
POOL_MAX = 20000           # prompts; past this the pool costs more memory than it is worth
MIN_ANSWERS = 4            # per input size: a slow server is measured for longer to get them
HINT = ("pass --tokenizer hf:<org>/<model> or the path of a tokenizer.json to count "
        "the tokens yourself")
ERRORS = {"transport": "could not reach {url}", "http_401": "the server refused the key (401)",
          "http_403": "the server refused the key (403)",
          "http_404": "no such path or model at {url} (404)"}


# ------------------------------------------------------------ the tokenizer --
def fetch_hf(repo: str, dest: Path) -> None:
    """tokenizer.json of a Hub repo into `dest` (HF_TOKEN, when set, for a gated one)."""
    import httpx
    token = os.environ.get("HF_TOKEN")
    r = httpx.get(f"https://huggingface.co/{repo}/resolve/main/tokenizer.json",
                  headers={"Authorization": f"Bearer {token}"} if token else {},
                  follow_redirects=True, timeout=30)
    if r.status_code != 200:
        raise OSError(f"HTTP {r.status_code} from huggingface.co; a private or gated repo "
                      "needs HF_TOKEN")
    dest.write_bytes(r.content)


def _hf(repo: str, cache: Path | None, fetch: Callable[[str, Path], None]):
    cache = cache or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") \
        / "inference-judge" / "tokenizers"
    dest = cache / repo.replace("/", "--") / "tokenizer.json"
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        fetch(repo, tmp)
        tmp.replace(dest)
    return load_tokenizer(str(dest))


def resolve_tokenizer(spec: str, model: str, cache: Path | None = None,
                      fetch: Callable[[str, Path], None] = fetch_hf) -> tuple[Any, str, str]:
    """-> (tokenizer, who counts the answer: "client" | "server", a note to print).

    When the server counts, the tokenizer returned only sizes the prompts
    (whitespace words). An explicit FILE or hf:REPO that cannot be loaded is
    an error; `auto` falls back to the server's count and says so.
    """
    ws = load_tokenizer("ws")
    if spec == "ws":
        return ws, "client", ""
    if spec == "server":
        return ws, "server", ""
    if spec == "auto":
        if "/" not in model:
            return ws, "server", HINT
        try:
            return _hf(model, cache, fetch), "client", ""
        except (Exception, SystemExit) as exc:
            return ws, "server", f"no tokenizer fetched for {model} ({exc}); {HINT}"
    if spec.startswith("hf:"):
        try:
            return _hf(spec[3:], cache, fetch), "client", ""
        except Exception as exc:
            raise SystemExit(f"--tokenizer {spec}: could not fetch the tokenizer of "
                             f"{spec[3:]} ({exc})") from exc
    return load_tokenizer(spec), "client", ""


# --------------------------------------------------------------- one answer --
def invalid_reason(row: dict, osl: int, counted_by: str, fixed: bool, tol: float) -> str | None:
    """Why this answer does not count, or None. Sets row["tokens_client"] to
    the count the table uses (the server's, when the server counts).

    Fixed length with the client's count is the judge's own rule. At free
    length an answer may stop early, but one that claims it hit the limit
    must be that long.
    """
    if counted_by == "server":
        row["tokens_client"] = row.get("tokens_usage")
    elif fixed:
        return metrics.tps_invalid_reason(row, osl, tol)
    tokens = row.get("tokens_client")
    if row.get("error_kind"):
        return f"error:{row['error_kind']}"
    if row.get("prompt_reused"):
        return "prompt_reused"
    if tokens is None:
        return "no_usage:the server sent no token count"
    hit_limit = row.get("finish_reason") == "length"
    if fixed and not hit_limit:
        return f"finish_reason:{row.get('finish_reason')}"
    if hit_limit and abs(tokens - osl) > tol * osl:
        return f"length:{tokens}!={osl}"
    if tokens < 2 or tokens > osl * (1 + tol):
        return f"length:{tokens}"
    return None


def _mark(rows: list[dict], osl: int, counted_by: str, fixed: bool, tol: float) -> None:
    for r in rows:
        reason = invalid_reason(r, osl, counted_by, fixed, tol)
        r["ok"], r["invalid_reason"] = reason is None, reason


# ------------------------------------------------------------ the measuring --
def _client_args(url: str, model: str, api_key_env: str | None) -> argparse.Namespace:
    argv = ["--base-url", url, "--model", model, "--mode", "tps"]
    if api_key_env:
        argv += ["--api-key-env", api_key_env]
    return harness.build_parser().parse_args(argv)


def _spec(pool: list[dict], sizes: list[int], osl: int, fixed: bool, **kw) -> harness.TpsSpec:
    return harness.TpsSpec(isl=sizes if len(sizes) > 1 else sizes[0], osl=osl, prompts=pool,
                           ignore_eos=fixed, chips=1, **kw)


async def _probe(client, tok, sizes: list[int], osl: int, counted_by: str, tol: float,
                 url: str) -> tuple[bool, float]:
    """One request with ignore_eos, and one without if that does not give a
    fixed-length answer. -> (fixed length?, seconds one answer took).

    When neither answer is valid the load still runs, so the table can say
    what is wrong with them: at free length if the server refused ignore_eos
    outright, by the judge's fixed-length rule otherwise.
    """
    pool = harness.make_prompt_pool(4, sizes[0], seed=1)
    errors, took = [], 1.0
    for fixed in (True, False):
        res = await harness.measure_tps(
            client, _spec(pool, sizes[:1], osl, fixed, load="offline", levels=[1.0],
                          seed=2 + fixed), tok)
        row = res.records[0]
        kind = str(row.get("error_kind") or "")
        for key, text in ERRORS.items():
            if kind.startswith(key):
                raise SystemExit(f"measure: {text.format(url=url)}")
        if not kind:
            took = max(1e-3, float(row["t_last"]) - float(row["t_send"]))
        if invalid_reason(row, osl, counted_by, fixed, tol) is None:
            return fixed, took
        errors.append(kind)
    if all(errors):
        raise SystemExit(f"measure: the server answered with an error ({errors[-1]})")
    return not errors[0], took


def _row(users: int, size: int, rates: dict, level_rows: list[dict], window: list[float],
         total: float) -> dict:
    """One table row. It counts the requests sent inside the measured window,
    the ones the rates are computed from."""
    mine = [r for r in level_rows if r.get("isl") == size and r.get("t_send") is not None
            and window[0] <= float(r["t_send"]) <= window[1]]
    ok = [r["tokens_client"] for r in mine if r.get("ok")]
    reasons: dict[str, int] = {}
    for r in mine:
        if not r.get("ok"):
            kind = str(r.get("invalid_reason")).split(":", 1)[0]
            reasons[kind] = reasons.get(kind, 0) + 1
    bad = next((str(r.get("invalid_reason")) for r in mine if not r.get("ok")), None)
    return {"users": users, "input": size,
            "tokens_per_s": rates.get("stream_tps"),
            "decode_tokens_per_s": rates.get("decode_tps"),
            "first_token_ms": rates.get("p50_ttft_ms"),
            "answer_tokens": int(statistics.median(ok)) if ok else None,
            "requests_ok": len(ok), "requests_invalid": len(mine) - len(ok),
            "invalid_reasons": reasons, "invalid_example": bad,
            "total_tokens_per_s": total}


async def measure(url: str, model: str, tok: Any, counted_by: str, sizes: list[int],
                  osl: int, users: list[int], seconds: float,
                  api_key_env: str | None = None) -> dict:
    """Probe the endpoint, load it at each number of users, return the rows."""
    tol = 0.0 if tok.name == "ws" and counted_by == "client" else LENGTH_TOL
    ramp = min(2.0, max(0.5, seconds / 4))
    async with harness.Client(_client_args(url, model, api_key_env)) as client:
        fixed, took = await _probe(client, tok, sizes, osl, counted_by, tol, url)
        seconds = max(seconds, MIN_ANSWERS * len(sizes) * took)
        print(f"measure: {'fixed' if fixed else 'free'}-length answers, about "
              f"{(ramp + seconds) * len(users):.0f} s to go", file=sys.stderr)
        per_user = (ramp + seconds) / took * 3 + 8 * len(sizes)
        n = min(POOL_MAX, int(math.ceil(per_user * sum(users))))
        pool = harness.make_prompt_pool(n, max(sizes), seed=int(time.time()))
        res = await harness.measure_tps(
            client, _spec(pool, sizes, osl, fixed, load="closed",
                          levels=[float(u) for u in users], ramp_s=ramp, window_s=seconds), tok)
    rows = []
    for level in res.levels:
        level_rows = [r for r in res.records if r.get("level") == level["level"]]
        _mark(level_rows, osl, counted_by, fixed, tol)
        s = metrics.tps_level_summary(level_rows, level["window"], 1, None,
                                      sizes if len(sizes) > 1 else None)
        by = s.get("by_isl") or {str(sizes[0]): s}
        rows += [_row(int(level["level"]), size, by[str(size)], level_rows, s["window"],
                      s["goodput_tps_per_chip"]) for size in sizes]
    return {"url": url, "model": model, "counted_by": counted_by, "tokenizer": tok.name,
            "answers": "fixed" if fixed else "free", "output": osl, "seconds": seconds,
            "rows": rows, "warnings": res.self_check["warnings"]}


# ------------------------------------------------------------------ the CLI --
def _num(v: float | None, fmt: str = "{:.1f}") -> str:
    return "-" if v is None else fmt.format(v)


def table(result: dict, note: str = "") -> str:
    by_server = result["counted_by"] == "server"
    out = [f"{result['url']}   model {result['model']}",
           "tokens counted by: " + ("the server (not checked)" if by_server
                                    else f"this tool ({result['tokenizer']})"),
           "answers: " + (f"fixed at {result['output']} tokens" if result["answers"] == "fixed"
                          else f"free length, up to {result['output']} tokens (the server "
                               "does not keep to a fixed length, so runs differ in work done)")]
    if by_server:
        out.append("input sizes are in words here, not tokens")
    if note:
        out.append("note: " + note)
    head = ("users", "input", "tokens/s", "after first token", "first token", "answers")
    body = []
    for r in result["rows"]:
        bad = ", ".join(f"{n} invalid ({k})" for k, n in r["invalid_reasons"].items())
        if len(r["invalid_reasons"]) == 1:
            bad = f"{r['requests_invalid']} invalid ({r['invalid_example']})"
        length = f" of {r['answer_tokens']} tokens" if r["answer_tokens"] is not None else ""
        body.append((str(r["users"]), str(r["input"]), _num(r["tokens_per_s"]),
                     _num(r["decode_tokens_per_s"]), _num(r["first_token_ms"], "{:.0f} ms"),
                     f"{r['requests_ok']} ok{length}" + (f", {bad}" if bad else "")))
    widths = [max(len(row[i]) for row in [head, *body]) for i in range(len(head))]
    out.append("")
    for row in [head, *body]:
        out.append("  ".join(c.rjust(w) if i < 5 else c
                             for i, (c, w) in enumerate(zip(row, widths))).rstrip())
    out.append("")
    for users in dict.fromkeys(r["users"] for r in result["rows"] if r["users"] > 1):
        total = next(r["total_tokens_per_s"] for r in result["rows"] if r["users"] == users)
        out.append(f"all {users} users together: {_num(total)} tokens/s")
    out.append("tokens/s is one user's whole answer, the wait for the first token included. "
               f"Measured for {result['seconds']:.3g} s.")
    out += ["warning: " + w for w in result["warnings"]]
    return "\n".join(out)


def _ints(value: str) -> list[int]:
    out = [int(x) for x in value.split(",") if x.strip()]
    if not out or min(out) < 1 or len(set(out)) != len(out):
        raise argparse.ArgumentTypeError(f"{value!r}: distinct positive numbers, e.g. 256,2048")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="measure", description="Measure how fast an OpenAI-style endpoint answers.")
    ap.add_argument("url", help="the endpoint's base URL, e.g. http://localhost:8000/v1")
    ap.add_argument("--model", required=True, help="the model name the server expects")
    ap.add_argument("--tokenizer", default="auto",
                    help="auto (default) | hf:<org>/<model> | a tokenizer.json | "
                         "server (take the server's count) | ws (the mock server)")
    ap.add_argument("--input", type=_ints, default=[256, 2048],
                    help="input sizes in tokens (default 256,2048)")
    ap.add_argument("--output", type=int, default=256, help="tokens per answer (default 256)")
    ap.add_argument("--users", type=_ints, default=[1],
                    help="parallel users; several values are measured one after another")
    ap.add_argument("--seconds", type=float, default=15.0,
                    help="measured time for each number of users (default 15; longer when "
                         "answers are slow, so every input size gets a few)")
    ap.add_argument("--api-key-env", default=None,
                    help="name of the environment variable that holds the API key")
    ap.add_argument("--json", default=None, help="also write the result to this file")
    args = ap.parse_args(argv)
    try:
        tok, counted_by, note = resolve_tokenizer(args.tokenizer, args.model)
        result = asyncio.run(measure(args.url, args.model, tok, counted_by, args.input,
                                     args.output, args.users, args.seconds, args.api_key_env))
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 2
    print(table(result, note))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
    return 0 if all(r["requests_ok"] for r in result["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
