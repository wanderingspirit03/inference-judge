#!/usr/bin/env python3
"""Quality guard v2: judge a candidate against a fixed reference R.

R is a dataset, not a server. It is written once from n
launches of the reference config and read-only afterwards; every quality rule
compares against it, never against the incumbent, so degradation cannot
accumulate down a chain of promotions.

Interface:
  Reference.load(dir) -> Reference       verified artefacts, floors, .hash,
                                         .thresholds(overrides)
  gate(candidate, reference, ...) -> GateResult
  collect(candidate, prompts, tasks, anchor=...) -> LaunchDump
  make_reference(launches, out_dir, ...) -> Reference
  CandidateClient                        the port; HttpCandidate is the
                                         production adapter (OpenAI-compatible
                                         /v1/completions and /v1/chat/completions)

The checks (all against R; thresholds are max(start, 3 x R-vs-R floor)):
  kl          teacher-forced KL(R||C) per position over the union of both
              top-k sets, a token missing from one side's top-k taking that
              side's k-th logprob (the most it can have). Mean and p99.
  top1        share of positions where C's argmax equals R's.
  divergence  greedy first-divergence position of C against R's launch 0,
              median over prompts, must be >= 0.8 x R's own launch-vs-launch
              median. Catches lossy spec-decode that teacher forcing cannot see.
  tasks       exact-match accuracy (math number, multiple-choice letter, tool
              call with arguments reachable from the prompt). The one-sided
              95% paired-bootstrap lower bound of mean(C_i - R_i) over items,
              R_i being R's per-item mean over its draws, must be >= -1.5 pts.
  sanity      per-text distinct-4-gram ratio and compression ratio; at most
              10% of C's texts may fall outside R's 1st-99th percentile band
              (edges widened 10%, so a small R does not make it a point).

When the candidate cannot return logprobs the kl check falls back to scoring
C's greedy tokens under R's stored logprobs: the mean excess NLL along R's
launch-0 continuation up to and including the first divergent token (beyond
it the prefix differs and R's stored logprobs no longer apply, so the value
shrinks as divergence comes later). top1 is then `unavailable`, and an
unavailable check fails the gate unless the caller allows it (target opt-in).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import numpy as np

QUALITY2_VERSION = "2.0"

# Starting thresholds (DESIGN 2.4); final = max(start, 3 x floor).
KL_MEAN_START = 0.02          # nats
KL_P99_START = 0.30           # nats
TOP1_START = 0.97
DIVERGENCE_FRACTION = 0.8     # C's median >= 0.8 x R's
TASK_LB_MIN = -0.015          # -1.5 points
NLL_EXCESS_START = 0.02       # nats/token, the no-logprobs fallback
SANITY_MAX_OUTSIDE = 0.10
SANITY_SLACK = 0.10           # band edges widened 10% (relative): with few R
                              # texts the 1st-99th band is nearly a point
FLOOR_MULTIPLIER = 3.0

TOP_K = 20
DEFAULT_MAX_TOKENS = 64       # the mock; real targets use 128
DEFAULT_TASK_DRAWS = 3
MAX_IN_FLIGHT = 8             # concurrent requests to one server (HttpCandidate)
REQUEST_TIMEOUT_S = 120.0     # one request; the ruler adds the compile budget
BOOTSTRAP_RESAMPLES = 5000

REFERENCE_FILES = ("prompts.jsonl", "greedy.jsonl", "logprobs.jsonl",
                   "tasks.jsonl", "task_answers.jsonl", "floors.json")
CHECKS = ("kl", "top1", "divergence", "tasks", "sanity")

Top = dict[str, float]


# --------------------------------------------------------------------------
# The port
# --------------------------------------------------------------------------
class LogprobsUnavailable(Exception):
    """The candidate cannot return teacher-forced logprobs."""


class AlignmentError(Exception):
    """The candidate's echoed tokens are not the continuation we sent."""


class CandidateClient(Protocol):
    """What the guard needs from a server. Two adapters: HttpCandidate here,
    and the in-memory fake in tests/test_quality2.py."""

    def greedy(self, prompt: str, max_tokens: int) -> tuple[list[str], str]:
        """Temperature-0 continuation of exactly max_tokens tokens (ignore_eos):
        (tokens, text)."""

    def teacher_force(self, prompt: str, text: str, tokens: list[str],
                      k: int) -> list[Top]:
        """Top-k logprobs at each continuation position when `prompt + text`
        is fed in; position j is the distribution over tokens[j]. Raises
        LogprobsUnavailable or AlignmentError."""

    def answer(self, task: dict) -> dict:
        """Temperature-0 chat reply: {"text": str, "tool_calls": [{"name",
        "arguments"}]}."""


# --------------------------------------------------------------------------
# Production adapter
# --------------------------------------------------------------------------
def join_continuation(prompt: str, text: str) -> tuple[str, int]:
    """prompt + text, with a space between when neither side has one so a
    whitespace tokenizer does not merge the boundary words. Returns the full
    string and the offset where the continuation starts."""
    sep = "" if (not prompt or not text or prompt[-1].isspace()
                 or text[0].isspace()) else " "
    return prompt + sep + text, len(prompt) + len(sep)


def _same_token(a: str, b: str) -> bool:
    return a == b or a.strip() == b.strip()


def _norm_top(entry: Any) -> Top | None:
    """One position's top-k, from the OpenAI ({tok: lp}), vLLM
    ({id: {"logprob", "decoded_token"}}) or bench mock ({"token", "logprob",
    "top_logprobs": {tok: lp}}) shape."""
    if not entry:
        return None
    if isinstance(entry.get("top_logprobs"), dict):          # bench mock
        out = {str(t): float(v) for t, v in entry["top_logprobs"].items()}
        if "token" in entry and entry.get("logprob") is not None:
            out.setdefault(str(entry["token"]), float(entry["logprob"]))
        return out
    out: Top = {}
    for key, val in entry.items():
        if isinstance(val, dict):
            out[str(val.get("decoded_token", key))] = float(val["logprob"])
        else:
            out[str(key)] = float(val)
    return out


class HttpCandidate:
    """An OpenAI-compatible server (vLLM, SGLang, the mock).

    Teacher forcing sends POST /v1/completions with the continuation appended
    to the prompt, echo=true, logprobs=k, max_tokens=1, temperature 0, and
    reads either choices[0].prompt_logprobs (vLLM) or choices[0].logprobs
    {tokens, top_logprobs, text_offset} (OpenAI legacy echo). The positions
    whose text_offset falls inside the continuation are kept, and each echoed
    token must equal the token we sent, else AlignmentError.
    """

    def __init__(self, base_url: str, model: str = "mock",
                 tokenize: Callable[[str], list[str]] = str.split,
                 timeout: float = REQUEST_TIMEOUT_S,
                 api_key: str | None = None) -> None:
        import httpx                                  # bench dependency
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(base_url=base_url.rstrip("/"),
                                  timeout=timeout, headers=headers)
        self.model = model
        self.tokenize = tokenize
        # Requests in flight at once. The answers do not depend on it (temperature
        # 0, results kept in input order); only the wall time does.
        self.concurrency = MAX_IN_FLIGHT

    def close(self) -> None:
        self._http.close()

    def _post(self, path: str, body: dict) -> dict:
        r = self._http.post(path, json={"model": self.model, **body})
        r.raise_for_status()
        return r.json()

    def greedy(self, prompt: str, max_tokens: int) -> tuple[list[str], str]:
        data = self._post("/completions", {
            "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
            "ignore_eos": True})
        text = data["choices"][0].get("text") or ""
        return self.tokenize(text), text

    def teacher_force(self, prompt: str, text: str, tokens: list[str],
                      k: int) -> list[Top]:
        import httpx
        full, start = join_continuation(prompt, text)
        try:
            data = self._post("/completions", {
                "prompt": full, "max_tokens": 1, "temperature": 0.0,
                "echo": True, "logprobs": k})
        except httpx.HTTPStatusError as exc:
            if 400 <= exc.response.status_code < 500:
                raise LogprobsUnavailable(
                    f"HTTP {exc.response.status_code} on echo+logprobs") from exc
            raise
        choice = (data.get("choices") or [{}])[0]
        if choice.get("prompt_logprobs"):
            entries = choice["prompt_logprobs"][-len(tokens):] if tokens else []
            if len(entries) != len(tokens):
                raise AlignmentError(f"{len(entries)} prompt_logprobs entries "
                                     f"for {len(tokens)} continuation tokens")
            tops = []
            for j, (e, tok) in enumerate(zip(entries, tokens)):
                top = _norm_top(e)
                echoed = e.get("token") if isinstance(e, dict) else None
                if top is None or (
                        not _same_token(echoed, tok) if isinstance(echoed, str)
                        else not any(_same_token(t, tok) for t in top)):
                    raise AlignmentError(f"position {j}: {tok!r} does not "
                                         f"match prompt_logprobs")
                tops.append(top)
            return tops
        lp = choice.get("logprobs") or {}
        echoed, top_list = lp.get("tokens"), lp.get("top_logprobs")
        if not echoed or not top_list:
            raise LogprobsUnavailable("response has no echoed logprobs")
        offsets = lp.get("text_offset")
        if offsets:
            idx = [i for i, off in enumerate(offsets) if start <= off < len(full)]
        else:                       # no offsets: the last token is generated
            idx = list(range(len(echoed) - 1 - len(tokens), len(echoed) - 1))
        if len(idx) != len(tokens):
            raise AlignmentError(f"{len(idx)} echoed continuation tokens, "
                                 f"sent {len(tokens)}")
        out = []
        for j, i in enumerate(idx):
            if not _same_token(echoed[i], tokens[j]):
                raise AlignmentError(f"position {j}: echoed {echoed[i]!r}, "
                                     f"sent {tokens[j]!r}")
            top = _norm_top(top_list[i])
            if top is None:
                raise LogprobsUnavailable(f"no top_logprobs at position {j}")
            out.append(top)
        return out

    def answer(self, task: dict) -> dict:
        body: dict = {"messages": [{"role": "user", "content": task["prompt"]}],
                      "temperature": 0.0, "max_tokens": 256, "stream": False}
        if task.get("tools"):
            body["tools"] = task["tools"]
        msg = self._post("/chat/completions", body)["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {"_unparsed": args}
            calls.append({"name": fn.get("name"), "arguments": args})
        return {"text": msg.get("content") or "", "tool_calls": calls}


# --------------------------------------------------------------------------
# Measures (private)
# --------------------------------------------------------------------------
def _logsumexp(x: np.ndarray) -> float:
    m = float(x.max())
    return m + math.log(float(np.exp(x - m).sum()))


def _kl(p: Top, q: Top) -> float:
    """KL(p||q) over the union of the two top-k sets."""
    union = list(set(p) | set(q))
    pf, qf = min(p.values()), min(q.values())
    lp = np.array([p.get(t, pf) for t in union])
    lq = np.array([q.get(t, qf) for t in union])
    lp, lq = lp - _logsumexp(lp), lq - _logsumexp(lq)
    return max(0.0, float(np.sum(np.exp(lp) * (lp - lq))))


def _argmax(top: Top) -> str:
    return max(top, key=lambda t: top[t])


def _divergence(a: Sequence[str], b: Sequence[str]) -> int:
    """Index of the first differing token; the shorter length if one is a
    prefix of the other."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _excess_nll(r_tokens: list[str], r_tops: list[Top],
                c_tokens: list[str]) -> float:
    """Mean over the scorable positions (up to and including the first
    divergent token) of logp_R(R's token) - logp_R(C's token)."""
    d = _divergence(r_tokens, c_tokens)
    if d >= len(r_tops) or d >= len(c_tokens) or d >= len(r_tokens):
        return 0.0
    top = r_tops[d]
    floor = min(top.values())
    gap = top.get(r_tokens[d], floor) - top.get(c_tokens[d], floor)
    return max(0.0, gap) / (d + 1)


def _sanity_stats(text: str) -> tuple[float, float]:
    toks = text.split()
    grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
    distinct = len(set(grams)) / len(grams) if grams else 1.0
    raw = text.encode()
    compression = len(zlib.compress(raw)) / len(raw) if raw else 1.0
    return distinct, compression


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_MC_PATTERNS = (re.compile(r"answer\s*(?:is)?\s*[:\-]?\s*\(?([A-E])\b", re.I),
                re.compile(r"^\s*\(?([A-E])\)?\s*[.:)]?\s*$"),
                re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])"))


def extract_answer(task: dict, reply: dict) -> Any:
    """The comparable answer in a reply: a float, a letter, or a tool call."""
    kind = task["kind"]
    text = reply.get("text") or ""
    if kind == "math":
        nums = _NUM_RE.findall(text)
        return float(nums[-1].replace(",", "")) if nums else None
    if kind == "mc":
        for pat in _MC_PATTERNS:
            m = pat.search(text)
            if m:
                return m.group(1).upper()
        return None
    if kind == "tool":
        calls = reply.get("tool_calls") or []
        return calls[0] if len(calls) == 1 else {"n_calls": len(calls)}
    raise ValueError(f"unknown task kind {kind!r}")


def _normalise_scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _num(float(value))
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip().lower()
    text = text.replace("’", "'")
    stripped = text.replace(",", "") if re.fullmatch(r"-?[\d,]*\.?\d+", text) else text
    try:
        return _num(float(stripped))
    except ValueError:
        return text


def _num(f: float) -> Any:
    return int(f) if f == int(f) else round(f, 6)


def _normalise_args(args: dict | None) -> dict:
    if not isinstance(args, dict):
        return {}
    out = {}
    for k, v in args.items():
        key = re.sub(r"\s+", "", str(k)).lower()
        if v is None or v == "":
            continue          # an omitted optional argument is not a mismatch
        if isinstance(v, (list, tuple)):
            out[key] = tuple(sorted(str(_normalise_scalar(x)) for x in v))
        elif isinstance(v, dict):
            out[key] = tuple(sorted((kk.lower(), str(_normalise_scalar(vv)))
                                    for kk, vv in v.items()))
        else:
            out[key] = _normalise_scalar(v)
    return out


def _tool_call_matches(expected: dict, produced_name: str | None,
                       produced_args: dict | None) -> bool:
    if not produced_name:
        return False
    if produced_name.strip().lower() != str(expected.get("name", "")).strip().lower():
        return False
    return _normalise_args(produced_args) == _normalise_args(expected.get("arguments"))


def is_correct(task: dict, answer: Any) -> bool:
    gold = task["answer"]
    if answer is None:
        return False
    if task["kind"] == "math":
        return abs(float(answer) - float(gold)) <= 1e-6 * max(1.0, abs(float(gold)))
    if task["kind"] == "mc":
        return str(answer).upper() == str(gold).upper()
    return (isinstance(answer, dict) and "name" in answer
            and _tool_call_matches(gold, answer.get("name"),
                                          answer.get("arguments")))


def _paired_lower_bound(diffs: np.ndarray, seed: int) -> float:
    """One-sided 95% percentile-bootstrap lower bound of mean(diffs)."""
    if diffs.size == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, (BOOTSTRAP_RESAMPLES, diffs.size))
    return float(np.quantile(diffs[idx].mean(axis=1), 0.05))


# --------------------------------------------------------------------------
# Per-launch collection
# --------------------------------------------------------------------------
def _pmap(candidate: CandidateClient, fn: Callable[[Any], Any],
          items: Sequence) -> list:
    """fn over items, in input order, with up to candidate.concurrency calls in
    flight (1 when the client does not say: the test fakes stay sequential).
    The first failure in input order is raised, as a plain loop would."""
    n = int(getattr(candidate, "concurrency", 1) or 1)
    if n <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(n, len(items))) as pool:
        futures = [pool.submit(fn, x) for x in items]
        return [f.result() for f in futures]


@dataclass
class LaunchDump:
    """What one launch of R produced. Launch 0 teacher-forces its own greedy
    continuations; launch k > 0 teacher-forces launch 0's (its `anchor`)."""
    max_tokens: int
    greedy: dict[str, dict]                 # id -> {"tokens", "text"}
    logprobs: dict[str, list[Top]]          # id -> tops along the anchor
    task_answers: list[dict]                # {"id", "draw", "answer", "correct"}
    anchored: bool = False

    def save(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        if out.exists():
            raise FileExistsError(f"{out} exists")
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
        try:
            _write_jsonl(tmp / "greedy.jsonl", (
                {"id": i, **g} for i, g in self.greedy.items()))
            _write_jsonl(tmp / "logprobs.jsonl", (
                {"id": i, "pos": p, "top": t}
                for i, tops in self.logprobs.items() for p, t in enumerate(tops)))
            _write_jsonl(tmp / "task_answers.jsonl", self.task_answers)
            _write_atomic(tmp / "launch.json", json.dumps(
                {"max_tokens": self.max_tokens, "anchored": self.anchored}))
            tmp.chmod(0o755)
            os.rename(tmp, out)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    @classmethod
    def load(cls, in_dir: str | Path) -> "LaunchDump":
        d = Path(in_dir)
        meta = json.loads((d / "launch.json").read_text())
        greedy = {r["id"]: {"tokens": r["tokens"], "text": r["text"]}
                  for r in _read_jsonl(d / "greedy.jsonl")}
        return cls(max_tokens=meta["max_tokens"], greedy=greedy,
                   logprobs=_group_tops(_read_jsonl(d / "logprobs.jsonl")),
                   task_answers=_read_jsonl(d / "task_answers.jsonl"),
                   anchored=meta.get("anchored", False))


def _answer_tasks(candidate: CandidateClient, tasks: list[dict],
                  draws: int) -> list[dict]:
    jobs = [(draw, task) for draw in range(draws) for task in tasks]
    replies = _pmap(candidate, lambda j: candidate.answer(j[1]), jobs)
    out = []
    for (draw, task), reply in zip(jobs, replies):
        ans = extract_answer(task, reply)
        out.append({"id": task["id"], "draw": draw, "answer": ans,
                    "correct": is_correct(task, ans)})
    return out


def collect(candidate: CandidateClient, prompts: list[dict], tasks: list[dict],
            *, anchor: LaunchDump | None = None,
            max_tokens: int = DEFAULT_MAX_TOKENS,
            task_draws: int = 1, k: int = TOP_K) -> LaunchDump:
    """One launch of R. Pass launch 0's dump as `anchor` for launches k > 0."""
    def one(p: dict) -> tuple[dict, list[Top]]:
        toks, text = candidate.greedy(p["prompt"], max_tokens)
        mine = {"tokens": toks, "text": text}
        src = anchor.greedy[p["id"]] if anchor else mine
        return mine, candidate.teacher_force(p["prompt"], src["text"],
                                             src["tokens"], k)

    greedy, logprobs = {}, {}
    for p, (mine, tops) in zip(prompts, _pmap(candidate, one, prompts)):
        greedy[p["id"]] = mine
        logprobs[p["id"]] = tops
    return LaunchDump(max_tokens=max_tokens, greedy=greedy, logprobs=logprobs,
                      task_answers=_answer_tasks(candidate, tasks, task_draws),
                      anchored=anchor is not None)


# --------------------------------------------------------------------------
# The reference
# --------------------------------------------------------------------------
def _write_atomic(path: Path, text: str) -> None:
    """A reader sees the old file or the whole new one, never a torn write."""
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.chmod(tmp, 0o644)              # mkstemp makes it 0600
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _write_jsonl(path: Path, rows) -> None:
    _write_atomic(path, "".join(
        json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def _group_tops(rows: list[dict]) -> dict[str, list[Top]]:
    out: dict[str, list[Top]] = {}
    for r in sorted(rows, key=lambda r: (r["id"], r["pos"])):
        out.setdefault(r["id"], []).append(r["top"])
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _floors(launches: list[LaunchDump], prompts: list[dict]) -> dict:
    """R-vs-R noise: every launch k > 0 against launch 0."""
    r0 = launches[0]
    kls, top1, divs, nll = [], [], [], []
    for lk in launches[1:]:
        for p in prompts:
            pid = p["id"]
            for a, b in zip(r0.logprobs[pid], lk.logprobs.get(pid, [])):
                kls.append(_kl(a, b))
                top1.append(_argmax(a) == _argmax(b))
            divs.append(_divergence(r0.greedy[pid]["tokens"],
                                    lk.greedy[pid]["tokens"]))
            nll.append(_excess_nll(r0.greedy[pid]["tokens"], r0.logprobs[pid],
                                   lk.greedy[pid]["tokens"]))
    by_draw: dict[tuple[int, int], list[bool]] = {}
    for k, lk in enumerate(launches):
        for a in lk.task_answers:
            by_draw.setdefault((k, a["draw"]), []).append(bool(a["correct"]))
    accs = [sum(v) / len(v) for v in by_draw.values()]
    texts = [g["text"] for lk in launches for g in lk.greedy.values()]
    distinct, compression = zip(*(_sanity_stats(t) for t in texts)) if texts \
        else ((1.0,), (1.0,))
    return {
        "kl_mean": float(np.mean(kls)) if kls else 0.0,
        "kl_p99": float(np.percentile(kls, 99)) if kls else 0.0,
        "top1": float(np.mean(top1)) if top1 else 1.0,
        "divergence_median": float(statistics.median(divs)) if divs
        else float(r0.max_tokens),
        "nll_excess": float(np.mean(nll)) if nll else 0.0,
        "task_acc_mean": float(np.mean(accs)) if accs else 1.0,
        "task_acc_sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "task_draws": len(accs),
        "sanity_band": {
            "distinct4": [float(np.percentile(distinct, 1)),
                          float(np.percentile(distinct, 99))],
            "compression": [float(np.percentile(compression, 1)),
                            float(np.percentile(compression, 99))]},
        "n_launches": len(launches),
    }


@dataclass(frozen=True)
class Reference:
    dir: Path
    manifest: dict
    prompts: list[dict]
    greedy: dict[str, list[dict]]           # id -> per-launch {"tokens","text"}
    logprobs: dict[str, list[Top]]          # id -> launch-0 teacher-forced
    tasks: list[dict]
    task_answers: list[dict]
    floors: dict

    @classmethod
    def load(cls, ref_dir: str | Path) -> "Reference":
        """Load and verify R. Any file whose sha256 differs from the manifest
        is refused: R is written once and never edited."""
        d = Path(ref_dir)
        manifest = json.loads((d / "manifest.json").read_text())
        for name in REFERENCE_FILES:
            want = manifest.get("sha256", {}).get(name)
            if want is None or _sha256(d / name) != want:
                raise ValueError(f"reference {d}: {name} does not match its "
                                 f"manifest sha256")
        greedy: dict[str, list[dict]] = {}
        for r in sorted(_read_jsonl(d / "greedy.jsonl"),
                        key=lambda r: (r["id"], r["launch"])):
            greedy.setdefault(r["id"], []).append(
                {"tokens": r["tokens"], "text": r["text"]})
        return cls(dir=d, manifest=manifest,
                   prompts=_read_jsonl(d / "prompts.jsonl"), greedy=greedy,
                   logprobs=_group_tops(_read_jsonl(d / "logprobs.jsonl")),
                   tasks=_read_jsonl(d / "tasks.jsonl"),
                   task_answers=_read_jsonl(d / "task_answers.jsonl"),
                   floors=json.loads((d / "floors.json").read_text()))

    @property
    def hash(self) -> str:
        """sha256 of manifest.json: stamp it as results.json reference_hash."""
        return _sha256(self.dir / "manifest.json")

    @property
    def max_tokens(self) -> int:
        return int(self.manifest["max_tokens"])

    def thresholds(self, overrides: dict | None = None) -> dict:
        """max(start, 3 x floor) for every check; overrides win."""
        f, m = self.floors, FLOOR_MULTIPLIER
        out = {
            "kl_mean_max": max(KL_MEAN_START, m * f["kl_mean"]),
            "kl_p99_max": max(KL_P99_START, m * f["kl_p99"]),
            "top1_min": min(TOP1_START, 1.0 - m * (1.0 - f["top1"])),
            "divergence_min": DIVERGENCE_FRACTION * f["divergence_median"],
            "task_lb_min": TASK_LB_MIN,
            "nll_excess_max": max(NLL_EXCESS_START, m * f["nll_excess"]),
            "sanity_max_outside": SANITY_MAX_OUTSIDE,
        }
        out.update(overrides or {})
        return out

    def r_item_accuracy(self) -> dict[str, float]:
        per: dict[str, list[bool]] = {}
        for a in self.task_answers:
            per.setdefault(a["id"], []).append(bool(a["correct"]))
        return {i: sum(v) / len(v) for i, v in per.items()}


def make_reference(launches: list[LaunchDump], out_dir: str | Path, *,
                   prompts: list[dict], tasks: list[dict], target_id: str,
                   tokenizer: str = "ws", source: str = "launch") -> Reference:
    """Write R atomically (temp dir, then rename) and return it loaded.
    Refuses an existing out_dir. Files are left read-only."""
    if not launches:
        raise ValueError("make_reference needs at least one launch")
    if launches[0].anchored or any(not lk.anchored for lk in launches[1:]):
        raise ValueError("launch 0 must be unanchored and launches 1.. anchored "
                         "to it")
    if source not in ("launch", "mock-replicated"):
        raise ValueError(f"unknown source {source!r}")
    out = Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists: a reference is written once")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        r0 = launches[0]
        _write_jsonl(tmp / "prompts.jsonl", prompts)
        _write_jsonl(tmp / "greedy.jsonl", (
            {"id": p["id"], "launch": k, **lk.greedy[p["id"]]}
            for k, lk in enumerate(launches) for p in prompts))
        _write_jsonl(tmp / "logprobs.jsonl", (
            {"id": p["id"], "pos": pos, "top": top}
            for p in prompts for pos, top in enumerate(r0.logprobs[p["id"]])))
        _write_jsonl(tmp / "tasks.jsonl", tasks)
        _write_jsonl(tmp / "task_answers.jsonl", (
            {"launch": k, **a} for k, lk in enumerate(launches)
            for a in lk.task_answers))
        (tmp / "floors.json").write_text(json.dumps(
            _floors(launches, prompts), indent=2, sort_keys=True))
        manifest = {
            "target_id": target_id,
            "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "source": source, "n_launches": len(launches),
            "tokenizer": tokenizer, "max_tokens": r0.max_tokens,
            "quality2_version": QUALITY2_VERSION,
            "sha256": {n: _sha256(tmp / n) for n in REFERENCE_FILES}}
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                      sort_keys=True))
        for f in tmp.iterdir():
            f.chmod(0o444)
        tmp.chmod(0o755)
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return Reference.load(out)


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------
@dataclass
class GateResult:
    passed: bool
    checks: dict[str, dict]
    thresholds: dict
    reasons: list[str]
    reference_hash: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": self.checks,
                "thresholds": self.thresholds, "reasons": self.reasons,
                "reference_hash": self.reference_hash,
                "quality2_version": QUALITY2_VERSION, **self.extra}


def _check(ok: bool, **values: Any) -> dict:
    return {"status": "pass" if ok else "fail", **values}


def _distribution_checks(candidate: CandidateClient, ref: Reference,
                         th: dict, c_greedy: dict[str, list[str]]
                         ) -> tuple[dict, dict]:
    kls, agree = [], []
    try:
        def force(p: dict) -> list[Top]:
            r0 = ref.greedy[p["id"]][0]
            return candidate.teacher_force(p["prompt"], r0["text"],
                                           r0["tokens"], TOP_K)

        for p, tops in zip(ref.prompts, _pmap(candidate, force, ref.prompts)):
            r_tops = ref.logprobs[p["id"]]
            if len(tops) != len(r_tops):
                raise AlignmentError(f"{p['id']}: {len(tops)} positions back, "
                                     f"R has {len(r_tops)}")
            for a, b in zip(r_tops, tops):
                kls.append(_kl(a, b))
                agree.append(_argmax(a) == _argmax(b))
    except AlignmentError as exc:
        fail = {"status": "fail", "error": f"alignment: {exc}"}
        return dict(fail, mode="teacher_forced"), dict(fail)
    except LogprobsUnavailable as exc:
        excess = [_excess_nll(ref.greedy[p["id"]][0]["tokens"],
                              ref.logprobs[p["id"]], c_greedy[p["id"]])
                  for p in ref.prompts if p["id"] in c_greedy]
        value = float(np.mean(excess)) if excess else None
        kl = {"mode": "fallback_r_scored", "unavailable_reason": str(exc),
              "nll_excess": value, "threshold": th["nll_excess_max"]}
        if value is None:
            kl["status"] = "unavailable"
        else:
            kl["status"] = "pass" if value <= th["nll_excess_max"] else "fail"
        return kl, {"status": "unavailable", "unavailable_reason": str(exc)}
    kl_mean = float(np.mean(kls)) if kls else 0.0
    kl_p99 = float(np.percentile(kls, 99)) if kls else 0.0
    top1 = float(np.mean(agree)) if agree else 1.0
    kl = _check(kl_mean <= th["kl_mean_max"] and kl_p99 <= th["kl_p99_max"],
                mode="teacher_forced", mean=kl_mean, p99=kl_p99,
                n_positions=len(kls), threshold_mean=th["kl_mean_max"],
                threshold_p99=th["kl_p99_max"])
    return kl, _check(top1 >= th["top1_min"], value=top1,
                      threshold=th["top1_min"])


def _task_check(candidate: CandidateClient, ref: Reference, th: dict,
                draws: int, seed: int) -> dict:
    r_acc = ref.r_item_accuracy()
    c_rows = _answer_tasks(candidate, ref.tasks, draws)
    per: dict[str, list[bool]] = {}
    for a in c_rows:
        per.setdefault(a["id"], []).append(a["correct"])
    ids = [t["id"] for t in ref.tasks if t["id"] in r_acc]
    c = np.array([sum(per[i]) / len(per[i]) for i in ids])
    r = np.array([r_acc[i] for i in ids])
    lb = _paired_lower_bound(c - r, seed)
    return _check(lb >= th["task_lb_min"], c_acc=float(c.mean()) if ids else None,
                  r_acc=float(r.mean()) if ids else None,
                  delta=float((c - r).mean()) if ids else None, lower_bound=lb,
                  threshold=th["task_lb_min"], n_items=len(ids), draws=draws,
                  wrong=[i for i, v in zip(ids, c) if v < 1.0])


def _sanity_check(texts: list[str], ref: Reference, th: dict) -> dict:
    band = ref.floors["sanity_band"]
    lo, hi = 1.0 - SANITY_SLACK, 1.0 + SANITY_SLACK
    (d_lo, d_hi), (c_lo, c_hi) = band["distinct4"], band["compression"]
    d_lo, d_hi, c_lo, c_hi = d_lo * lo, d_hi * hi, c_lo * lo, c_hi * hi
    outside = 0
    for t in texts:
        d, c = _sanity_stats(t)
        if not (d_lo <= d <= d_hi and c_lo <= c <= c_hi):
            outside += 1
    frac = outside / len(texts) if texts else 1.0
    return _check(frac <= th["sanity_max_outside"], frac_outside=frac,
                  threshold=th["sanity_max_outside"], band=band, n=len(texts))


def gate(candidate: CandidateClient, reference: Reference,
         thresholds: dict | None = None, *, allow_unavailable: bool = False,
         task_draws: int = 1, seed: int = 20261009) -> GateResult:
    """Run every check of the candidate against R.

    thresholds: overrides for reference.thresholds(). allow_unavailable: the
    target accepts a candidate that cannot return logprobs (the fallback kl
    still has to pass). An error talking to the candidate fails its check.
    """
    th = reference.thresholds(thresholds)
    checks: dict[str, dict] = {}
    c_greedy: dict[str, list[str]] = {}
    c_texts: list[str] = []
    try:
        outs = _pmap(candidate, lambda p: candidate.greedy(
            p["prompt"], reference.max_tokens), reference.prompts)
        for p, (toks, text) in zip(reference.prompts, outs):
            c_greedy[p["id"]] = toks
            c_texts.append(text)
        divs = [_divergence(reference.greedy[i][0]["tokens"], t)
                for i, t in c_greedy.items()]
        med = float(statistics.median(divs)) if divs else 0.0
        checks["divergence"] = _check(
            med >= th["divergence_min"], median=med,
            r_median=reference.floors["divergence_median"],
            threshold=th["divergence_min"])
        checks["sanity"] = _sanity_check(c_texts, reference, th)
    except Exception as exc:                          # noqa: BLE001
        err = {"status": "fail", "error": f"greedy: {exc!r}"}
        checks["divergence"], checks["sanity"] = dict(err), dict(err)
    try:
        checks["kl"], checks["top1"] = _distribution_checks(
            candidate, reference, th, c_greedy)
    except Exception as exc:                          # noqa: BLE001
        err = {"status": "fail", "error": f"teacher_force: {exc!r}"}
        checks["kl"], checks["top1"] = dict(err), dict(err)
    try:
        checks["tasks"] = _task_check(candidate, reference, th, task_draws, seed)
    except Exception as exc:                          # noqa: BLE001
        checks["tasks"] = {"status": "fail", "error": f"tasks: {exc!r}"}

    reasons = []
    for name in CHECKS:
        c = checks[name]
        if c["status"] == "fail":
            reasons.append(f"{name}: " + (c.get("error") or _why(name, c)))
        elif c["status"] == "unavailable" and not allow_unavailable:
            reasons.append(f"{name}: unavailable ({c.get('unavailable_reason')}); "
                           f"failing closed, the target does not allow it")
    return GateResult(passed=not reasons, checks=checks, thresholds=th,
                      reasons=reasons, reference_hash=reference.hash,
                      extra={"allow_unavailable": allow_unavailable})


def _why(name: str, c: dict) -> str:
    if name == "kl" and c.get("mode") == "fallback_r_scored":
        return f"excess NLL under R {c['nll_excess']:.4f} > {c['threshold']:.4f}"
    if name == "kl":
        return (f"mean {c['mean']:.4f} (max {c['threshold_mean']:.4f}), "
                f"p99 {c['p99']:.4f} (max {c['threshold_p99']:.4f})")
    if name == "top1":
        return f"{c['value']:.4f} < {c['threshold']:.4f}"
    if name == "divergence":
        return f"median {c['median']:.1f} < {c['threshold']:.1f}"
    if name == "tasks":
        return (f"paired lower bound {c['lower_bound']:+.4f} < "
                f"{c['threshold']:+.4f}; wrong: {c['wrong']}")
    return f"{c['frac_outside']:.2f} of texts outside R's band"


# --------------------------------------------------------------------------
# The mock reference set (the real ~600-item set and 256 prompts are deferred)
# --------------------------------------------------------------------------
# Math and multiple-choice items carry an [ANSWER:x] marker for the mock
# server to echo; tool items use the mock's existing [TOOL:name:{args}] marker
# and every argument value is in the prompt text.
_LOOKUP = {"type": "function", "function": {
    "name": "lookup_booking", "parameters": {
        "type": "object", "properties": {"booking_id": {"type": "string"}},
        "required": ["booking_id"]}}}
_RESERVE = {"type": "function", "function": {
    "name": "create_reservation", "parameters": {
        "type": "object", "properties": {
            "name": {"type": "string"}, "party_size": {"type": "integer"},
            "time": {"type": "string"}},
        "required": ["name", "party_size", "time"]}}}

MOCK_TASKS: list[dict] = [
    {"id": "m-math-1", "kind": "math", "lang": "en", "answer": 42,
     "prompt": "What is 17 + 25? Reply with the number only. [ANSWER:42]"},
    {"id": "m-math-2", "kind": "math", "lang": "en", "answer": 144,
     "prompt": "What is 12 squared? Reply with the number only. [ANSWER:144]"},
    {"id": "m-math-3", "kind": "math", "lang": "en", "answer": 7.5,
     "prompt": "Half of 15 is what? Reply with the number only. [ANSWER:7.5]"},
    {"id": "m-math-4", "kind": "math", "lang": "de", "answer": 36,
     "prompt": "Wie viel ist 9 mal 4? Nur die Zahl, bitte. [ANSWER:36]"},
    {"id": "m-math-5", "kind": "math", "lang": "en", "answer": 1200,
     "prompt": "A room costs 400 a night. What do 3 nights cost? Number "
               "only. [ANSWER:1200]"},
    {"id": "m-mc-1", "kind": "mc", "lang": "en", "answer": "B",
     "prompt": "Which planet is closest to the sun? A) Venus B) Mercury "
               "C) Mars D) Earth. Answer with the letter. [ANSWER:B]"},
    {"id": "m-mc-2", "kind": "mc", "lang": "en", "answer": "C",
     "prompt": "Which of these is a prime number? A) 21 B) 27 C) 29 D) 33. "
               "Answer with the letter. [ANSWER:C]"},
    {"id": "m-mc-3", "kind": "mc", "lang": "fr", "answer": "A",
     "prompt": "Quelle est la capitale de la France ? A) Paris B) Lyon "
               "C) Nice D) Lille. Repondez par la lettre. [ANSWER:A]"},
    {"id": "m-tool-1", "kind": "tool", "lang": "en", "tools": [_LOOKUP],
     "answer": {"name": "lookup_booking", "arguments": {"booking_id": "K7Q2"}},
     "prompt": 'Please look up booking K7Q2. '
               '[TOOL:lookup_booking:{"booking_id": "K7Q2"}]'},
    {"id": "m-tool-2", "kind": "tool", "lang": "en", "tools": [_LOOKUP],
     "answer": {"name": "lookup_booking", "arguments": {"booking_id": "ZX90"}},
     "prompt": 'Can you find my booking, the reference is ZX90? '
               '[TOOL:lookup_booking:{"booking_id": "ZX90"}]'},
    {"id": "m-tool-3", "kind": "tool", "lang": "en", "tools": [_RESERVE],
     "answer": {"name": "create_reservation", "arguments": {
         "name": "Okafor", "party_size": 4, "time": "19:30"}},
     "prompt": 'Book a table for 4 at 19:30 under the name Okafor. '
               '[TOOL:create_reservation:{"name": "Okafor", "party_size": 4, '
               '"time": "19:30"}]'},
    {"id": "m-tool-4", "kind": "tool", "lang": "es", "tools": [_RESERVE],
     "answer": {"name": "create_reservation", "arguments": {
         "name": "Ruiz", "party_size": 2, "time": "21:00"}},
     "prompt": 'Reserve una mesa para 2 a las 21:00 a nombre de Ruiz. '
               '[TOOL:create_reservation:{"name": "Ruiz", "party_size": 2, '
               '"time": "21:00"}]'},
]

_TOPICS = ("the history of the printing press", "how tides work",
           "why the sky is blue", "the rules of chess", "how bread rises",
           "the water cycle", "how a bicycle stays upright", "the life of bees")
_FORMS = ("Write a short essay about", "Explain to a child", "Describe in detail",
          "Give a long answer about")
MOCK_PROMPTS: list[dict] = [
    {"id": f"m-p{i:02d}", "prompt": f"{_FORMS[i % 4]} {_TOPICS[i % 8]}"
                                    f" (variant {i // 8})."}
    for i in range(32)]


# --------------------------------------------------------------------------
# CLI (thin adapter)
# --------------------------------------------------------------------------
def _load_target(path: str | None) -> dict:
    if not path:
        return {"id": "cpu-mock", "engine": {"name": "mock"},
                "model": {"served_name": "mock", "tokenizer": "ws"}}
    try:
        import target as _target                      # F's loader, if present
        return _target.load_target(path)
    except ImportError:
        import yaml
        return yaml.safe_load(Path(path).read_text())


def _tokenize_fn(spec: str) -> Callable[[str], list[str]]:
    if spec in ("", "ws"):
        return str.split
    import tokenizer as _tok                          # B's module
    tok = _tok.load_tokenizer(spec)
    if not hasattr(tok, "tokens"):
        raise SystemExit(f"tokenizer {spec}: needs a tokens(text) -> list[str] "
                         f"method for divergence checks")
    return tok.tokens


def _client(args, tgt: dict, tokenizer: str | None = None) -> HttpCandidate:
    model = (tgt.get("model") or {})
    spec = tokenizer if tokenizer is not None else model.get("tokenizer", "ws")
    return HttpCandidate(args.base_url, model=args.model or model.get(
        "served_name", "mock"), tokenize=_tokenize_fn(spec),
        timeout=args.request_timeout)


def _is_mock(tgt: dict) -> bool:
    return (tgt.get("engine") or {}).get("name", "mock") == "mock"


def _inputs(args, tgt: dict) -> tuple[list[dict], list[dict]]:
    """--prompts/--tasks, or the built-in mock set. The mock tasks carry their
    answers in [ANSWER:x] markers, so they are refused for a real engine."""
    if not (args.prompts and args.tasks) and not _is_mock(tgt):
        raise SystemExit("quality2: a non-mock target needs --prompts and "
                         "--tasks (the built-in set is for the mock only)")
    prompts = _read_jsonl(Path(args.prompts)) if args.prompts else MOCK_PROMPTS
    tasks = _read_jsonl(Path(args.tasks)) if args.tasks else MOCK_TASKS
    return prompts, tasks


def cmd_collect(args) -> int:
    tgt = _load_target(args.target)
    prompts, tasks = _inputs(args, tgt)
    anchor = LaunchDump.load(args.anchor) if args.anchor else None
    client = _client(args, tgt)
    try:
        dump = collect(client, prompts, tasks, anchor=anchor,
                       max_tokens=args.max_tokens, task_draws=args.task_draws)
    finally:
        client.close()
    dump.save(args.out)
    print(f"[quality2] collected launch -> {args.out}")
    return 0


def cmd_make_reference(args) -> int:
    tgt = _load_target(args.target)
    prompts, tasks = _inputs(args, tgt)
    tokenizer = (tgt.get("model") or {}).get("tokenizer", "ws")
    if args.launches_from:
        launches = [LaunchDump.load(d) for d in args.launches_from.split(",")]
        source = "launch"
    elif args.base_url:
        # One server process measured n times: no cold-launch noise, so the
        # floors are a lower bound. Acceptable for the mock only.
        if not _is_mock(tgt):
            raise SystemExit(
                "make-reference --base-url measures one server several times: "
                "for the mock only. A real target needs --launches-from, one "
                "`collect` per cold launch")
        client = _client(args, tgt)
        try:
            draws = max(1, DEFAULT_TASK_DRAWS // args.replicate)
            l0 = collect(client, prompts, tasks, max_tokens=args.max_tokens,
                         task_draws=draws)
            launches = [l0] + [
                collect(client, prompts, tasks, anchor=l0,
                        max_tokens=args.max_tokens, task_draws=draws)
                for _ in range(args.replicate - 1)]
        finally:
            client.close()
        source = "mock-replicated"
    else:
        raise SystemExit("make-reference needs --launches-from or --base-url")
    ref = make_reference(launches, args.out, prompts=prompts, tasks=tasks,
                         target_id=tgt.get("id", "unknown"), tokenizer=tokenizer,
                         source=source)
    print(f"[quality2] reference {ref.dir} ({source}, {len(launches)} launches) "
          f"hash {ref.hash[:12]}")
    return 0


def cmd_gate(args) -> int:
    tgt = _load_target(args.target)
    ref = Reference.load(args.reference)
    allow = args.allow_unavailable or bool(
        (tgt.get("quality") or {}).get("allow_missing_logprobs"))
    # R's tokenizer, not the target's: divergence compares token streams.
    client = _client(args, tgt, tokenizer=ref.manifest.get("tokenizer", "ws"))
    try:
        result = gate(client, ref, allow_unavailable=allow,
                      task_draws=args.task_draws)
    finally:
        client.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(out, json.dumps(result.to_dict(), indent=2))
    status = " ".join(f"{k}={v['status']}" for k, v in result.checks.items())
    print(f"[quality2] passed={result.passed} {status} -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Quality guard v2 against a fixed "
                                             "reference R.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, need_url: bool) -> None:
        p.add_argument("--base-url", required=need_url,
                       help="OpenAI-compatible base, e.g. http://127.0.0.1:8000/v1")
        p.add_argument("--target", default=None, help="targets/<id>.yaml")
        p.add_argument("--model", default=None, help="served name override")
        p.add_argument("--request-timeout", type=float, default=REQUEST_TIMEOUT_S,
                       help="seconds one request may take (a prompt on a shape "
                            "the server has not compiled waits for the compile)")

    c = sub.add_parser("collect", help="one launch of R -> a launch dir")
    common(c, True)
    c.add_argument("--out", required=True)
    c.add_argument("--anchor", default=None, help="launch 0's dir (launches 1..)")
    c.add_argument("--prompts", default=None)
    c.add_argument("--tasks", default=None)
    c.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    c.add_argument("--task-draws", type=int, default=1)
    c.set_defaults(func=cmd_collect)

    m = sub.add_parser("make-reference", help="assemble R from launches")
    common(m, False)
    m.add_argument("--out", required=True)
    m.add_argument("--launches-from", default=None,
                   help="comma-separated launch dirs from `collect`, launch 0 first")
    m.add_argument("--replicate", type=int, default=3,
                   help="with --base-url: measure the one server this many times")
    m.add_argument("--prompts", default=None)
    m.add_argument("--tasks", default=None)
    m.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    m.set_defaults(func=cmd_make_reference)

    g = sub.add_parser("gate", help="judge a candidate against R")
    common(g, True)
    g.add_argument("--reference", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--allow-unavailable", action="store_true",
                   help="accept a candidate that cannot return logprobs")
    g.add_argument("--task-draws", type=int, default=1)
    g.set_defaults(func=cmd_gate)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
