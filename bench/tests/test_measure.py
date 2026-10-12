"""measure.py: one command against any OpenAI-style endpoint, no target file,
no Docker, no agent. It prints a speed table and can write the same as JSON.

Interface under test (command line only):
    measure.py URL --model NAME [--tokenizer auto|ws|server|FILE|hf:REPO]
               [--input N,N] [--output N] [--users N,N] [--seconds S] [--json FILE]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import BENCH, MockServer, free_port

import measure

FAST = dict(ttft_ms=20, tpot_ms=2, model="mock")


def run(url: str, *args: str, out: Path | None = None) -> tuple[subprocess.CompletedProcess, dict]:
    argv = [sys.executable, "-I", str(BENCH / "measure.py"), url, "--model", "mock",
            "--output", "16", "--seconds", "1.5", *args]
    if out is not None:
        argv += ["--json", str(out)]
    p = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    data = json.loads(out.read_text()) if out is not None and out.exists() else {}
    return p, data


@pytest.fixture(scope="module")
def honest():
    server = MockServer(**FAST)
    yield server
    server.stop()


def test_one_row_per_input_size_counted_by_the_client(honest, tmp_path):
    p, data = run(honest.url, "--tokenizer", "ws", "--input", "32,128", out=tmp_path / "m.json")
    assert p.returncode == 0, p.stderr
    assert data["counted_by"] == "client" and data["answers"] == "fixed"
    rows = data["rows"]
    assert [(r["users"], r["input"]) for r in rows] == [(1, 32), (1, 128)]
    for r in rows:
        assert r["requests_ok"] > 0 and r["requests_invalid"] == 0
        assert r["answer_tokens"] == 16
        # the mock decodes at 2 ms a token: 500 tokens/s at best, and the
        # whole answer (first token included) is slower than the decode alone
        assert 50 < r["tokens_per_s"] <= r["decode_tokens_per_s"] < 600
        assert r["first_token_ms"] >= 20
    # the table a person reads carries the same rows
    assert "tokens/s" in p.stdout
    for size in ("32", "128"):
        assert any(line.split()[:2] == ["1", size] for line in p.stdout.splitlines())


def test_several_user_counts_give_a_row_each_and_a_total(honest, tmp_path):
    p, data = run(honest.url, "--tokenizer", "ws", "--input", "32", "--users", "1,4",
                  out=tmp_path / "m.json")
    assert p.returncode == 0, p.stderr
    rows = data["rows"]
    assert [r["users"] for r in rows] == [1, 4]
    assert rows[1]["total_tokens_per_s"] > 2 * rows[0]["total_tokens_per_s"]


def test_a_slow_server_is_measured_for_longer_so_every_size_gets_answers(tmp_path):
    """16 tokens at 25 ms each: an answer takes 0.4 s, so half a second would
    hold one answer for two input sizes."""
    server = MockServer(ttft_ms=20, tpot_ms=25, model="mock")
    try:
        p, data = run(server.url, "--tokenizer", "ws", "--input", "32,128", "--seconds", "0.5",
                      out=tmp_path / "m.json")
    finally:
        server.stop()
    assert p.returncode == 0, p.stderr
    assert data["seconds"] > 2
    assert all(r["requests_ok"] >= 2 for r in data["rows"])


def test_a_server_that_ends_answers_early_is_measured_at_free_length(tmp_path):
    """A hosted API does not honour ignore_eos. The answers are then as long
    as the server makes them, and the output says so. (Without ignore_eos the
    mock sends a canned reply of a few words and stops.)"""
    server = MockServer(behaviour="early_eos", **FAST)
    try:
        p, data = run(server.url, "--tokenizer", "ws", "--input", "32", "--output", "64",
                      out=tmp_path / "m.json")
    finally:
        server.stop()
    assert p.returncode == 0, p.stderr
    assert data["answers"] == "free"
    assert data["rows"][0]["requests_ok"] > 0 and data["rows"][0]["requests_invalid"] == 0
    assert 2 <= data["rows"][0]["answer_tokens"] < 64
    assert "free length" in p.stdout


def test_the_servers_own_count_is_used_only_when_asked_and_is_labelled(tmp_path):
    """lie_usage sends half the tokens and reports all of them: counting on
    the client sees 8 where 16 were asked for and accepts none of them,
    taking the server's word gives 16."""
    server = MockServer(behaviour="lie_usage", **FAST)
    try:
        _, mine = run(server.url, "--tokenizer", "ws", "--input", "32", out=tmp_path / "a.json")
        p, theirs = run(server.url, "--tokenizer", "server", "--input", "32",
                        out=tmp_path / "b.json")
    finally:
        server.stop()
    # the mock's canned reply (what it sends without ignore_eos) is longer than
    # the 16 asked for, so free length is no way out either: the fixed rule stands
    assert mine["counted_by"] == "client" and mine["answers"] == "fixed"
    assert mine["rows"][0]["requests_ok"] == 0
    assert mine["rows"][0]["invalid_example"] == "length:8!=16"
    assert theirs["counted_by"] == "server" and theirs["rows"][0]["answer_tokens"] == 16
    assert "not checked" in p.stdout


def test_auto_without_a_tokenizer_to_fetch_takes_the_servers_count(honest, tmp_path):
    p, data = run(honest.url, "--input", "32", out=tmp_path / "m.json")   # --tokenizer auto
    assert p.returncode == 0, p.stderr
    assert data["counted_by"] == "server"
    assert "--tokenizer" in p.stdout                      # says how to count them yourself


def test_an_unreachable_server_is_a_plain_error(tmp_path):
    p, data = run(f"http://127.0.0.1:{free_port()}/v1", "--tokenizer", "ws", "--input", "32",
                  out=tmp_path / "m.json")
    assert p.returncode == 2
    assert "could not reach" in p.stderr and "Traceback" not in p.stderr
    assert data == {}


def test_a_server_whose_every_answer_is_invalid_exits_1(tmp_path):
    """truncate claims finish_reason length at half the length: with the
    client's own count that is neither a fixed-length answer nor an honest
    early stop."""
    server = MockServer(behaviour="truncate", **FAST)
    try:
        p, data = run(server.url, "--tokenizer", "ws", "--input", "32", out=tmp_path / "m.json")
    finally:
        server.stop()
    assert p.returncode == 1
    assert data["rows"][0]["requests_ok"] == 0 and data["rows"][0]["requests_invalid"] > 0
    assert "length" in p.stdout


# ------------------------------------------------------------ the tokenizer --
def test_hf_spec_fetches_once_into_the_cache(tmp_path):
    calls = []

    def fetch(repo: str, dest: Path) -> None:
        calls.append(repo)
        dest.write_text(json.dumps({"version": "1.0", "model": {"type": "WordLevel",
                        "vocab": {"a": 0, "[UNK]": 1}, "unk_token": "[UNK]"},
                        "pre_tokenizer": {"type": "Whitespace"}}))

    for _ in range(2):
        tok, counted_by, note = measure.resolve_tokenizer("hf:org/model", "served-name",
                                                          cache=tmp_path, fetch=fetch)
        assert counted_by == "client" and tok.count("a a a") == 3
    assert calls == ["org/model"]


def test_auto_falls_back_to_the_server_when_the_fetch_fails(tmp_path):
    def fetch(repo: str, dest: Path) -> None:
        raise OSError("no network")

    tok, counted_by, note = measure.resolve_tokenizer("auto", "org/model", cache=tmp_path,
                                                      fetch=fetch)
    assert counted_by == "server" and "org/model" in note


def test_an_explicit_hf_spec_that_cannot_be_fetched_is_an_error(tmp_path):
    def fetch(repo: str, dest: Path) -> None:
        raise OSError("no network")

    with pytest.raises(SystemExit) as exc:
        measure.resolve_tokenizer("hf:org/model", "x", cache=tmp_path, fetch=fetch)
    assert "org/model" in str(exc.value)
