"""bench/pools: the speed pools, the quality prompts and the quality tasks.

The data files are checked in; build_pools.py is the one interface that makes
them. These tests hold the files to what the harness and the ruler need of
them: counts, unique ids, disjoint main and confirm pools, lines longer than
the largest input size, a prompt length spread, task answers that reproduce
independently, and the probe tool working against the mock server.

Token lengths are checked with the real Gemma tokenizer when GEMMA_TOKENIZER_JSON
points at its tokenizer.json (and the `tokenizers` package is installed); without
it, a conservative characters-per-token bound stands in.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

import metrics
import quality2
from tokenizer import WhitespaceTokenizer

BENCH = Path(__file__).resolve().parent.parent
POOLS = BENCH / "pools"
sys.path.insert(0, str(POOLS))

import build_pools            # noqa: E402
import pool_tasks             # noqa: E402

LARGEST_ISL = 8192
NONCE_AND_HEADROOM = 400                    # tokens: "Request reference: <uuid>.", chat noise
MAX_PROMPT_TOKENS = 7800                    # + 128 forced tokens must fit in 8192
# characters per Gemma token, highest seen on the checked-in files (4.85 / 3.25 / 4.00)
CPT_BOUND = {"prose": 4.9, "code": 3.3, "mixed": 4.05}

TOKENIZER_JSON = os.environ.get("GEMMA_TOKENIZER_JSON", "")


def _load(name: str) -> list[dict]:
    return metrics.read_jsonl(str(POOLS / name))


@pytest.fixture(scope="module")
def main_pool():
    return _load(build_pools.FILES["main"])


@pytest.fixture(scope="module")
def confirm_pool():
    return _load(build_pools.FILES["confirm"])


@pytest.fixture(scope="module")
def prompts():
    return _load(build_pools.FILES["prompts"])


@pytest.fixture(scope="module")
def tasks():
    return _load(build_pools.FILES["tasks"])


@pytest.fixture(scope="module")
def tokenizer():
    if not TOKENIZER_JSON:
        pytest.skip("GEMMA_TOKENIZER_JSON not set")
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER_JSON)
    return lambda text: len(tok.encode(text, add_special_tokens=False).ids)


def _kind(row: dict) -> str:
    return row["id"].rsplit("-", 1)[1]


# ------------------------------------------------------------ the files ---
def test_counts_and_ids(main_pool, confirm_pool, prompts, tasks):
    assert (len(main_pool), len(confirm_pool), len(prompts), len(tasks)) == (384, 320, 256, 200)
    for rows in (main_pool, confirm_pool, prompts, tasks):
        ids = [r["id"] for r in rows]
        assert len(set(ids)) == len(ids)
        assert all(isinstance(r["prompt"], str) and r["prompt"].strip() for r in rows)
    assert all(r["id"].startswith("main-") for r in main_pool)
    assert all(r["id"].startswith("confirm-") for r in confirm_pool)


def test_files_are_a_fresh_build():
    """The checked-in bytes are what the generator makes; nothing was hand-edited."""
    assert build_pools.check(POOLS) == []


def test_build_is_deterministic_and_parameterised():
    small = dict(main_lines=6, confirm_lines=5, min_tokens=1200, n_prompts=40, n_tasks=30)
    a, b = build_pools.build(**small), build_pools.build(**small)
    assert a == b
    counts = {name: len(text.splitlines()) for name, text in a.items()}
    assert counts == {build_pools.FILES["main"]: 6, build_pools.FILES["confirm"]: 5,
                      build_pools.FILES["prompts"]: 40, build_pools.FILES["tasks"]: 30}
    shorter = build_pools.build(**{**small, "min_tokens": 600})
    assert (sum(len(json.loads(l)["prompt"]) for l in a[build_pools.FILES["main"]].splitlines())
            > sum(len(json.loads(l)["prompt"]) for l in shorter[build_pools.FILES["main"]].splitlines()))


def test_cli_check_reports_a_changed_file(tmp_path):
    small = dict(main_lines=3, confirm_lines=3, min_tokens=600, n_prompts=36, n_tasks=20)
    assert build_pools.write(tmp_path, **small)
    assert build_pools.check(tmp_path, **small) == []
    target = tmp_path / build_pools.FILES["tasks"]
    target.write_text(target.read_text().replace("ANSWER", "ANSVER", 1))
    assert build_pools.check(tmp_path, **small) == [build_pools.FILES["tasks"]]


# ------------------------------------------------------- the speed pools ---
def test_main_and_confirm_are_disjoint(main_pool, confirm_pool):
    assert not {r["id"] for r in main_pool} & {r["id"] for r in confirm_pool}

    def paragraphs(rows):
        return {p.strip() for r in rows for p in r["prompt"].split("\n\n")[1:] if len(p.strip()) > 80}

    # the code generator has a dozen shapes, so a few short boilerplate blocks
    # (an __init__, a loop) recur across pools; no passage of any size does
    shared = paragraphs(main_pool) & paragraphs(confirm_pool)
    assert all(len(p) < 300 for p in shared), max(shared, key=len)[:200]
    prose = [r for r in main_pool + confirm_pool if _kind(r) == "prose"]
    assert not paragraphs([r for r in prose if r["id"].startswith("main")]) & \
        paragraphs([r for r in prose if r["id"].startswith("confirm")])
    # no shared long prefix, in the pool or across pools
    heads = Counter(r["prompt"][:400] for r in main_pool + confirm_pool)
    assert max(heads.values()) == 1
    # body prefixes too (the instruction line is shared by design)
    bodies = Counter(r["prompt"].split("\n\n", 1)[1][:300] for r in main_pool + confirm_pool)
    assert max(bodies.values()) == 1


def test_pool_lines_are_long_enough_by_characters(main_pool, confirm_pool):
    for r in main_pool + confirm_pool:
        need = (LARGEST_ISL + NONCE_AND_HEADROOM) * CPT_BOUND[_kind(r)]
        assert len(r["prompt"]) >= need, (r["id"], len(r["prompt"]), need)


def test_pool_kinds_are_mixed(main_pool, confirm_pool):
    for rows in (main_pool, confirm_pool):
        kinds = Counter(_kind(r) for r in rows)
        assert set(kinds) == {"prose", "code", "mixed"}
        assert kinds["prose"] > kinds["code"] > 0 and kinds["mixed"] > 0


def test_pool_lines_ask_for_a_long_output(main_pool, confirm_pool):
    """The metrics sanity band reads the output text, so every line opens with an
    instruction that makes the model write a lot rather than one word."""
    heads = set(build_pools.PROSE_HEADS) | set(build_pools.CODE_HEADS)
    for r in main_pool + confirm_pool:
        assert r["prompt"].split("\n\n", 1)[0] in heads, r["id"]
    for h in heads:
        assert re.search(r"\b(long|extensive|detailed|thorough|full|every|exhaustive|at length|"
                         r"as long as you can|complete)\b", h), h


def test_python_in_pool_lines_parses(main_pool, confirm_pool):
    """The Python files in code lines are well-formed: a model asked to review
    them is not being asked to read noise."""
    checked = 0
    for r in main_pool + confirm_pool:
        for m in re.finditer(r"^# file: \S+\.py\n(.*?)(?=^(?:#|//|--|/\*) file: |\Z|^```)", r["prompt"], re.S | re.M):
            ast.parse(m.group(1))
            checked += 1
    assert checked >= 50


def test_pools_load_through_the_harness(main_pool, confirm_pool):
    import harness
    ws = WhitespaceTokenizer()
    for pool, seed in ((main_pool, 1), (confirm_pool, 2)):
        sampler = harness._PromptSampler(pool, seed, frozenset())
        seen = set()
        for _ in range(len(pool)):
            row, reused = sampler.next()
            assert not reused and row["id"] not in seen
            seen.add(row["id"])
            assert len(ws.fit(row["prompt"], 128).split()) == 128
        assert sampler.n_reused == 0
    # a sweep excludes the previous sweep's ids and still has a pool left
    spent = frozenset(r["id"] for r in main_pool[:100])
    assert len(harness._PromptSampler(main_pool, 3, spent).items) == len(main_pool) - 100


# -------------------------------------------------------- quality prompts ---
def test_prompt_categories_and_lengths_by_characters(prompts):
    kinds = Counter(_kind(r) for r in prompts)
    assert set(kinds) == {"factual", "arith", "code", "summary", "translate", "json", "retrieve"}
    assert all(n >= 30 for n in kinds.values())
    chars = sorted(len(r["prompt"]) for r in prompts)
    assert chars[0] < 400                                    # short prompts exist
    assert sum(c > 4096 * 2.9 for c in chars) >= 20          # >= 20 above ~4096 tokens at the lowest chars/token
    assert chars[-1] < MAX_PROMPT_TOKENS * 6                 # nothing absurd


def test_prompts_are_distinct(prompts):
    assert len({r["prompt"] for r in prompts}) == len(prompts)
    assert len({r["prompt"][:300] for r in prompts if len(r["prompt"]) > 4000}) == \
        sum(len(r["prompt"]) > 4000 for r in prompts)


def test_prompts_work_as_ruler_prompts(prompts):
    """The ruler reads {"id", "prompt"} and sorts them deterministically."""
    assert all(set(r) >= {"id", "prompt"} for r in prompts)
    assert quality2.join_continuation(prompts[0]["prompt"], "x")[0].startswith(prompts[0]["prompt"])


# --------------------------------------------------------- quality tasks ---
def test_task_answers_reproduce(tasks):
    """Every gold answer is re-derived from the prompt text by code that shares
    nothing with the generator's arithmetic."""
    for t in tasks:
        got = pool_tasks.verify(t)
        if t["kind"] == "math":
            assert abs(float(got) - float(t["answer"])) < 1e-9, t["id"]
        else:
            assert got == t["answer"], t["id"]


def test_task_shape_and_balance(tasks):
    assert Counter(t["kind"] for t in tasks) == {"math": 100, "mc": 100}
    assert all(t["lang"] in {"en", "de", "fr", "es"} for t in tasks)
    assert {t["lang"] for t in tasks} == {"en", "de", "fr", "es"}
    assert sum(t["lang"] != "en" for t in tasks) >= 10
    assert len({t["prompt"] for t in tasks}) == len(tasks)
    letters = Counter(t["answer"] for t in tasks if t["kind"] == "mc")
    assert set(letters) <= set("ABCD") and min(letters.values()) >= 12, letters
    for t in tasks:
        if t["kind"] == "mc":
            opts = re.findall(r"^([A-D])[).:] ", t["prompt"], re.M)
            assert opts == list("ABCD"), t["id"]
        assert 40 < len(t["prompt"]) < 1500


def test_gold_answers_score_through_the_ruler(tasks):
    """A reply in the format the prompt asks for is read back as the gold answer."""
    for t in tasks:
        reply = {"text": f"Working.\nANSWER: {t['answer']}"} if t["kind"] == "math" \
            else {"text": f"{t['answer']}"}
        assert quality2.is_correct(t, quality2.extract_answer(t, reply)), t["id"]
    wrong = {"text": "ANSWER: 123456789"}
    math_task = next(t for t in tasks if t["kind"] == "math")
    assert not quality2.is_correct(math_task, quality2.extract_answer(math_task, wrong))


def test_every_task_family_appears(tasks):
    fams = Counter((t["kind"], t["id"].split("-")[2]) for t in tasks)
    assert len(fams) >= 25 and min(fams.values()) >= 3, fams


def test_task_generator_is_seeded():
    a = pool_tasks.make_tasks(random.Random(5), 40)
    assert a == pool_tasks.make_tasks(random.Random(5), 40)
    assert a != pool_tasks.make_tasks(random.Random(6), 40)


# ------------------------------------------------ the real tokenizer ---
def test_pool_lines_exceed_the_largest_input(main_pool, confirm_pool, tokenizer):
    counts = [tokenizer(r["prompt"]) for r in main_pool + confirm_pool]
    assert min(counts) >= LARGEST_ISL + NONCE_AND_HEADROOM, min(counts)


def test_quality_prompt_length_spread(prompts, tokenizer):
    counts = sorted(tokenizer(r["prompt"]) for r in prompts)
    assert counts[0] <= 40
    assert counts[-1] <= MAX_PROMPT_TOKENS
    assert counts[-1] >= 7000
    assert sum(c > 4096 for c in counts) >= 20
    assert sum(c <= 128 for c in counts) >= 30
    assert sum(c <= 2048 for c in counts) >= 150


# ------------------------------------------------------------- the probe ---
def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "tpu_probe_logprobs", BENCH.parent / "tools" / "tpu_probe_logprobs.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _statuses(results) -> dict:
    return {r.name: r.status for r in results}


def test_probe_passes_against_the_honest_mock(mock_server):
    probe = _load_probe()
    for url in (mock_server.url, mock_server.url[:-3]):          # with and without /v1
        results = probe.run_probe(probe.Server(url, "mock"), osl=40)
        status = _statuses(results)
        assert "FAIL" not in status.values(), [r for r in results if r.status == "FAIL"]
        assert {"echo_logprobs", "top_logprobs", "chat_stream_usage", "exact_length",
                "determinism"} <= set(status)


def test_probe_fails_a_server_that_ignores_ignore_eos():
    from conftest import MockServer
    server = MockServer(behaviour="early_eos")
    try:
        probe = _load_probe()
        status = _statuses(probe.run_probe(probe.Server(server.url, "mock"), osl=40))
    finally:
        server.stop()
    assert status["exact_length"] == "FAIL"


def test_probe_exit_codes(mock_server, capsys):
    probe = _load_probe()
    assert probe.main(["--base-url", mock_server.url, "--model", "mock", "--osl", "32"]) == 0
    assert "PASS echo_logprobs" in capsys.readouterr().out
    assert probe.main(["--base-url", "http://127.0.0.1:9", "--model", "mock", "--timeout", "2"]) == 2


def test_probe_reads_both_logprob_shapes():
    probe = _load_probe()
    full, start = probe._join("one two", "three four")
    assert (full, start) == ("one two three four", 8)
    legacy = {"logprobs": {
        "tokens": ["one", " two", " three", " four"], "text_offset": [0, 3, 7, 13],
        "top_logprobs": [None] + [{f"t{i}": -1.0 - i / 100 for i in range(20)}] * 3}}
    info = probe.read_echo(legacy, full, 8, "three four", 4)
    assert info["shape"] == "logprobs" and info["problems"] == [], info
    broken = {"logprobs": {"tokens": ["one"], "top_logprobs": [None]}}
    assert probe.read_echo(broken, full, 8, "three four", 4)["problems"]
    vllm = {"prompt_logprobs": [None] + [{"11": {"logprob": -0.5, "rank": 1, "decoded_token": "a"},
                                           "12": {"logprob": -1.5, "rank": 2, "decoded_token": "b"}}] * 3}
    info = probe.read_echo(vllm, full, 8, "three four", 4)
    assert info["shape"] == "prompt_logprobs" and info["problems"] == [], info
    assert probe.read_echo({}, full, 8, "x", 4)["shape"] is None

