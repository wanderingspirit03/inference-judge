"""bench/pools/real: the speed pools of whole requests.

One file per request under pools/real/<pool>/<size>/; real_pool.py copies them
into the two .jsonl files the target names. These tests hold the files to
what a run needs: a fresh build, enough requests of every size, nothing shared
between the main and the confirm pool, and (with the real tokenizer, when
GEMMA_TOKENIZER_JSON points at its tokenizer.json) every request inside its
size's band and accepted by the harness as it is.
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

import harness
from tokenizer import load_tokenizer

BENCH = Path(__file__).resolve().parent.parent
POOLS = BENCH / "pools"
sys.path.insert(0, str(POOLS))

import real_pool              # noqa: E402

TOKENIZER_JSON = os.environ.get("GEMMA_TOKENIZER_JSON", "")


def pool(name: str) -> list[dict]:
    return [json.loads(l) for l in (POOLS / real_pool.FILES[name]).open(encoding="utf-8")]


@pytest.fixture(scope="module")
def main_pool():
    return pool("main")


@pytest.fixture(scope="module")
def confirm_pool():
    return pool("confirm")


@pytest.fixture(scope="module")
def tokenizer():
    if not TOKENIZER_JSON:
        pytest.skip("GEMMA_TOKENIZER_JSON not set")
    pytest.importorskip("tokenizers")
    return load_tokenizer(TOKENIZER_JSON)


def test_files_are_a_fresh_build():
    assert real_pool.check(POOLS) == []


def test_every_size_has_the_requests_a_run_needs(main_pool, confirm_pool):
    for name, rows in (("main", main_pool), ("confirm", confirm_pool)):
        per_size = Counter(r["isl"] for r in rows)
        assert set(per_size) == set(real_pool.SIZES)
        assert min(per_size.values()) >= real_pool.MIN_PER_SIZE[name], per_size
        assert len({r["id"] for r in rows}) == len(rows)
        assert all(set(r) == {"id", "isl", "prompt"} for r in rows)


def test_a_pool_short_of_a_size_is_not_built(tmp_path):
    for name in real_pool.FILES:
        for size in real_pool.SIZES:
            d = tmp_path / name / str(size)
            d.mkdir(parents=True)
            for i in range(real_pool.MIN_PER_SIZE[name]):
                (d / f"{i:02d}-ask.txt").write_text(f"request {name} {size} {i}\n")
    built = real_pool.build(tmp_path)
    assert {n: t.count("\n") for n, t in built.items()} == {
        real_pool.FILES[n]: len(real_pool.SIZES) * real_pool.MIN_PER_SIZE[n]
        for n in real_pool.FILES}
    (tmp_path / "confirm" / "2000" / "00-ask.txt").unlink()
    with pytest.raises(SystemExit, match=r"real/confirm/2000 holds \d+ requests"):
        real_pool.build(tmp_path)


def _passages(text: str) -> set[str]:
    """The lines long enough to be content and not layout."""
    return {" ".join(l.split()) for l in text.splitlines()
            if len(l.strip()) >= 60 and any(c.isalpha() for c in l)}


def test_no_request_is_there_twice_and_the_pools_share_nothing(main_pool, confirm_pool):
    rows = main_pool + confirm_pool
    assert len({r["prompt"] for r in rows}) == len(rows)
    main = set().union(*(_passages(r["prompt"]) for r in main_pool))
    confirm = set().union(*(_passages(r["prompt"]) for r in confirm_pool))
    assert main & confirm == set()


def test_requests_are_whole_text(main_pool, confirm_pool):
    for r in main_pool + confirm_pool:
        text = r["prompt"]
        assert text == text.strip() + "\n", r["id"]
        assert text.count("```") % 2 == 0, r["id"]          # no fence left open
        for marker in ("lorem ipsum", "[...]", "(continued)", "TODO: fill"):
            assert marker not in text.lower(), (r["id"], marker)
        # an address at a named host (user@10.0.0.4 in a config is not one)
        for host in re.findall(r"[\w.+-]+@([\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,})\b", text):
            assert host.split(".")[-2] == "example", (r["id"], host)


def test_pasted_python_parses(main_pool, confirm_pool):
    seen = 0
    for r in main_pool + confirm_pool:
        if not r["id"].endswith("-python"):
            continue
        for block in re.findall(r"```python\n(.*?)```", r["prompt"], flags=re.S):
            ast.parse(block)
            seen += 1
    assert seen >= 8


def test_every_request_is_inside_its_band(main_pool, confirm_pool, tokenizer):
    for r in main_pool + confirm_pool:
        lo, hi = real_pool.bounds(r["isl"])
        assert lo <= tokenizer.count(r["prompt"]) <= hi, r["id"]


def test_the_reference_line_is_as_long_as_the_band_assumes(tokenizer):
    """The builder allows for a line at least as short and as long as any the
    harness checks a request against, so what it builds the harness takes."""
    for n in (4096, 200_000):               # the harness's own set, and a far larger one
        shortest, longest = harness.nonce_bounds(tokenizer, n)
        assert real_pool.NONCE_TOKENS[0] <= shortest and longest <= real_pool.NONCE_TOKENS[1]


@pytest.mark.parametrize("target", ["v6e4-gemma4-26b-tps", "gpu-gemma4-26b-tps"])
def test_the_targets_take_what_the_builder_allows(target):
    """A target that names these pools is no stricter than the builder was."""
    import yaml
    metric = yaml.safe_load((BENCH.parent / "targets" / f"{target}.yaml").read_text())["metric"]
    assert Path(metric["prompt_pool"]).name == real_pool.FILES["main"]
    assert Path(metric["confirm_pool"]).name == real_pool.FILES["confirm"]
    assert metric["isl_tol"] >= real_pool.BAND
    assert set(metric["isl"]) <= set(real_pool.SIZES)


@pytest.mark.parametrize("name", list(real_pool.FILES))
def test_the_harness_takes_every_request_as_it_is(name, tokenizer):
    rows = pool(name)
    spec = harness.TpsSpec(load="closed", levels=[1], isl=list(real_pool.SIZES), osl=512,
                           prompts=rows, isl_tol=real_pool.BAND)
    run = harness._TpsRun(None, spec, tokenizer)
    by_id = {r["id"]: r for r in rows}
    for size in real_pool.SIZES:
        prompt_id, content, reused = run.prompt(size)
        assert by_id[prompt_id]["isl"] == size and not reused
        assert content.endswith(by_id[prompt_id]["prompt"])
        assert size * (1 - real_pool.BAND) <= tokenizer.count(content) <= size
