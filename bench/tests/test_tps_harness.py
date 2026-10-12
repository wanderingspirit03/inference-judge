"""--mode tps end to end: load shapes, validity, and the gaming behaviours of
the mock server (DESIGN 2.2, 2.3). Each gaming mode must either be marked
invalid by the client or lose on goodput to honest; `fast` must win."""
from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import harness
import metrics
from conftest import MockServer

ISL, OSL = 48, 16
FAST_WINDOW = ["--ramp-s", "0.3", "--window-s", "1.5"]


@pytest.fixture(scope="module")
def pool(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("pool") / "pool.jsonl"
    metrics.write_jsonl(str(path), harness.make_prompt_pool(3000, 2 * ISL))
    return path


@pytest.fixture(scope="module")
def mocks():
    started: dict[str, MockServer] = {}

    def get(behaviour: str = "honest", **flags: object) -> MockServer:
        flags = {"ttft_ms": 20, "tpot_ms": 8, **flags}
        key = behaviour + json.dumps(flags, sort_keys=True)
        if key not in started:
            started[key] = MockServer(behaviour=behaviour, **flags)
        return started[key]

    yield get
    for s in started.values():
        s.stop()


def tps(server_url: str, out: Path, pool: Path, *extra: str) -> dict:
    argv = ["--base-url", server_url, "--model", "m", "--mode", "tps",
            "--prompt-pool", str(pool), "--isl", str(ISL), "--osl", str(OSL),
            "--out", str(out), *FAST_WINDOW]
    if "--tps-load" not in extra:
        argv += ["--tps-load", "closed", "--concurrency", "4"]
    assert harness.main(argv + list(extra)) == 0
    summary = json.loads((out / "summary.json").read_text())
    summary["records"] = metrics.read_jsonl(str(out / "tps_records.jsonl"))
    return summary


def headline(summary: dict) -> float:
    return summary["metric"]["headline"]


def level(summary: dict, i: int = 0) -> dict:
    return summary["metric"]["levels"][i]


@pytest.fixture(scope="module")
def honest(mocks, pool, tmp_path_factory) -> dict:
    return tps(mocks("honest").url, tmp_path_factory.mktemp("honest"), pool)


# --------------------------------------------------------------------------
# the honest baseline and the record format
# --------------------------------------------------------------------------
def test_honest_closed_loop_is_valid_and_scored(honest):
    s = level(honest)
    assert s["n_invalid"] == 0 and s["slo_met"] is True
    assert s["goodput_tps_per_chip"] > 0
    assert headline(honest) == s["goodput_tps_per_chip"]
    rec = honest["records"][0]
    for key in ("req_id", "prompt_id", "load", "level", "t_send", "t_first",
                "t_last", "tokens_client", "tokens_usage", "finish_reason",
                "itl_ms", "ok", "invalid_reason", "text_sha1"):
        assert key in rec
    assert all(r["tokens_client"] == OSL and r["finish_reason"] == "length"
               and r["isl_client"] == ISL for r in honest["records"])
    assert honest["nonce"] == "turn" and honest["ignore_eos"] is True
    assert honest["tokenizer"]["name"] == "ws"
    assert honest["self_check"]["loop_lag_max_ms"] >= 0


def test_closed_loop_holds_its_concurrency_through_the_window(honest):
    """Zero think time and immediate replacement: from the window start until
    the last scored request finishes, `level` requests are in flight - the
    drain fillers keep the last scored requests company. Measured as the exact
    time-average (sampling instants alias with the lockstep replacement gaps,
    which are client bookkeeping, sub-millisecond on an idle machine)."""
    s = level(honest)
    w0, w1 = s["window"]
    rows = honest["records"]
    end = max(r["t_last"] for r in rows if w0 <= r["t_send"] <= w1)
    busy = sum(max(0.0, min(r["t_last"], end) - max(r["t_send"], w0))
               for r in rows)
    assert 3.6 < busy / (end - w0) <= 4.0
    # never more than `level` at once
    for t in sorted(r["t_send"] + 1e-4 for r in rows):
        assert sum(1 for r in rows if r["t_send"] <= t <= r["t_last"]) <= 4


def test_every_prompt_is_fresh_and_starts_with_a_nonce(honest):
    ids = [r["prompt_id"] for r in honest["records"]]
    assert len(ids) == len(set(ids))
    assert honest["self_check"]["prompt_reused"] == 0


def test_a_sweep_runs_every_level_in_one_invocation(mocks, pool, tmp_path):
    out = tps(mocks().url, tmp_path, pool, "--tps-load", "closed",
              "--concurrency", "2,6")
    assert [s["level"] for s in out["metric"]["levels"]] == [2, 6]
    assert {r["level"] for r in out["records"]} == {2, 6}
    a, b = out["metric"]["levels"]
    assert b["goodput_tps_per_chip"] > 1.8 * a["goodput_tps_per_chip"]
    # prompts are not reused across levels either
    ids = [r["prompt_id"] for r in out["records"]]
    assert len(ids) == len(set(ids))


def test_offline_scores_the_makespan(mocks, pool, tmp_path):
    out = tps(mocks().url, tmp_path, pool, "--tps-load", "offline",
              "--requests", "12")
    s = level(out)
    assert s["n"] == 12 and s["n_window"] == 12 and s["n_invalid"] == 0
    assert s["window_s"] > 0
    assert s["tps_per_chip"] == pytest.approx(12 * OSL / s["window_s"])


def test_chips_divide_the_throughput(mocks, pool, tmp_path):
    out = tps(mocks().url, tmp_path, pool, "--tps-load", "offline",
              "--requests", "8", "--chips", "4")
    s = level(out)
    assert s["tps_per_chip"] == pytest.approx(8 * OSL / s["window_s"] / 4)


# --------------------------------------------------------------------------
# gaming behaviours: invalid, or losing, never winning
# --------------------------------------------------------------------------
@pytest.mark.parametrize("behaviour, reason", [
    ("truncate", "length"),
    ("early_eos", "finish_reason"),
    ("lie_usage", "length"),
])
def test_short_replies_are_invalid_and_score_zero(mocks, pool, tmp_path,
                                                  behaviour, reason):
    out = tps(mocks(behaviour).url, tmp_path, pool)
    s = level(out)
    assert s["n_invalid"] == s["n"] > 0
    assert set(s["invalid_reasons"]) == {reason}
    assert s["goodput_tps_per_chip"] == 0.0
    assert headline(out) == 0.0


def test_lie_usage_is_caught_by_counting_the_text_not_the_usage(mocks, pool,
                                                                tmp_path):
    out = tps(mocks("lie_usage").url, tmp_path, pool)
    r = out["records"][0]
    assert r["tokens_usage"] == OSL            # what the server claimed
    assert r["tokens_client"] == OSL // 2      # what it sent
    # raw tokens/s is the client's count too: half of what usage would say
    s = level(out)
    assert s["tps_per_chip"] == pytest.approx(
        sum(x["tokens_client"] for x in out["records"]
            if s["window"][0] <= x["t_last"] <= s["window"][1])
        / s["window_s"])


def test_fast_legitimately_wins(mocks, pool, tmp_path, honest):
    out = tps(mocks("fast").url, tmp_path, pool)
    assert level(out)["n_invalid"] == 0
    assert headline(out) > 1.4 * headline(honest)


def test_junk_at_speed_loses_once_the_sanity_band_is_applied(mocks, pool,
                                                             tmp_path, honest):
    """Junk has the right length and a true usage, so validity alone lets it
    through at 2x. The n-gram sanity band (R's, from quality2) is what stops
    it; here the band is taken from the honest run."""
    raw = tps(mocks("junk").url, tmp_path / "raw", pool)
    assert headline(raw) > headline(honest)           # the exploit, unguarded
    h = level(honest)
    band = {**metrics.TPS_DEFAULT_SLO,
            "min_distinct4": 0.8 * h["distinct4"],
            "max_compression": 1.25 * h["compression"]}
    assert level(raw)["distinct4"] < 0.2 * h["distinct4"]
    guarded = tps(mocks("junk").url, tmp_path / "guarded", pool,
                  "--slo", json.dumps(band))
    assert level(guarded)["sanity_ok"] is False
    assert headline(guarded) == 0.0
    honest_banded = tps(mocks("honest").url, tmp_path / "honest", pool,
                        "--slo", json.dumps(band))
    assert headline(honest_banded) > 0


def test_cache_memo_gains_nothing_from_fresh_nonced_prompts(mocks, pool,
                                                            tmp_path, honest):
    for behaviour in ("cache_memo", "cache_memo_strip_nonce"):
        out = tps(mocks(behaviour).url, tmp_path / behaviour, pool)
        assert level(out)["n_invalid"] == 0
        assert headline(out) < 1.25 * headline(honest), behaviour


def test_a_pool_too_small_to_stay_fresh_invalidates_the_level(mocks, tmp_path):
    """With a tiny pool the stripping memo would win on repeats: every
    repeat is flagged and invalid, so the level scores nothing."""
    small = tmp_path / "small.jsonl"
    metrics.write_jsonl(str(small), harness.make_prompt_pool(6, 2 * ISL))
    out = tps(mocks("cache_memo_strip_nonce").url, tmp_path / "o", small)
    s = level(out)
    assert out["self_check"]["prompt_reused"] > 0
    assert s["invalid_reasons"].get("prompt_reused", 0) > 0
    assert headline(out) == 0.0


def test_pool_exclude_keeps_warmup_prompts_out(mocks, pool, tmp_path, honest):
    warm = tmp_path / "warm"
    warm_out = tps(mocks().url, warm, pool)
    meas = tps(mocks().url, tmp_path / "meas", pool, "--pool-exclude",
               str(warm / "tps_records.jsonl"))
    used = {r["prompt_id"] for r in warm_out["records"]}
    assert used and not used & {r["prompt_id"] for r in meas["records"]}


def test_quantised_passes_the_load_ruler_and_is_left_to_quality(mocks, pool,
                                                                tmp_path,
                                                                honest):
    """Quantisation is a valid stream at the right length; only the quality
    gate against R (quality2) can see it. Recorded so nobody expects the TPS
    client to catch it."""
    out = tps(mocks("quantised").url, tmp_path, pool)
    assert level(out)["n_invalid"] == 0
    assert headline(out) > headline(honest)


# --------------------------------------------------------------------------
# open loop on an absolute schedule
# --------------------------------------------------------------------------
def test_open_loop_sends_stay_on_schedule_under_a_slow_server(mocks, pool,
                                                              tmp_path):
    """Sends are due at t0 + k/rate whatever the server does: a slow server
    (each request ~0.6 s) must not slow the sender."""
    slow = mocks("honest", ttft_ms=300, tpot_ms=20)
    out = tps(slow.url, tmp_path, pool, "--tps-load", "open", "--rate", "25",
              "--arrival", "uniform")
    s = level(out)
    assert s["achieved_rate"] == pytest.approx(25, rel=0.08)
    assert s["send_lag_max_ms"] < 30
    rows = [r for r in out["records"] if r["t_sched"] is not None]
    drift = [r["t_send"] - r["t_sched"] for r in rows]
    assert max(drift) < 0.03
    # the k-th send is due at k / rate from the first: no cumulative drift
    rows.sort(key=lambda r: r["t_sched"])
    gaps = [b["t_sched"] - a["t_sched"] for a, b in zip(rows, rows[1:])]
    # (records keep 6 decimals, so two rounded stamps differ by <= 2e-6)
    assert all(g == pytest.approx(1 / 25, abs=5e-6) for g in gaps)


def test_open_loop_poisson_is_seeded(mocks, pool, tmp_path):
    a = tps(mocks().url, tmp_path / "a", pool, "--tps-load", "open",
            "--rate", "20", "--seed", "3")
    b = tps(mocks().url, tmp_path / "b", pool, "--tps-load", "open",
            "--rate", "20", "--seed", "3")

    def gaps(out: dict) -> list[float]:
        t = sorted(r["t_sched"] for r in out["records"])
        return [round(y - x, 6) for x, y in zip(t, t[1:])]
    assert gaps(a)[:10] == pytest.approx(gaps(b)[:10], abs=1e-5)
    assert len(set(gaps(a))) > 5                     # not a fixed interval


# --------------------------------------------------------------------------
# what goes over the wire
# --------------------------------------------------------------------------
@contextlib.contextmanager
def capture_server(n_tokens: int, usage: int | None, finish: str = "length"):
    """Answers every chat request with n_tokens ws tokens and records the
    request bodies."""
    bodies: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_POST(self) -> None:          # noqa: N802
            body = json.loads(self.rfile.read(
                int(self.headers.get("content-length") or 0)))
            bodies.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            lines = []
            for i in range(n_tokens):
                lines.append({"choices": [{"index": 0, "delta": {
                    "content": ("" if i == 0 else " ") + f"w{i}"},
                    "finish_reason": None}]})
            lines.append({"choices": [{"index": 0, "delta": {},
                                       "finish_reason": finish}]})
            if usage is not None:
                lines.append({"choices": [], "usage": {
                    "prompt_tokens": 1, "completion_tokens": usage}})
            for obj in lines:
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", bodies
    finally:
        server.shutdown()
        server.server_close()


def test_every_request_pins_length_eos_nonce_and_thinking(pool, tmp_path):
    with capture_server(OSL, OSL) as (url, bodies):
        tps(url, tmp_path, pool, "--tps-load", "offline", "--requests", "6",
            "--chat-template-kwargs", '{"enable_thinking": false}',
            "--extra-body", '{"max_tokens": 9999, "ignore_eos": false}')
    assert len(bodies) == 6
    for b in bodies:
        assert b["max_tokens"] == OSL and b["ignore_eos"] is True
        assert b["stream"] is True
        assert b["chat_template_kwargs"] == {"enable_thinking": False}
        content = b["messages"][0]["content"]
        assert content.startswith("Request reference: ")
        assert len(content.split()) == ISL
    heads = {b["messages"][0]["content"].split("\n")[0] for b in bodies}
    assert len(heads) == 6                            # a fresh nonce each
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["chat_template_kwargs"] == {"enable_thinking": False}


def test_usage_is_recorded_but_never_trusted(pool, tmp_path):
    # right text length, wrong usage: invalid on the usage cross-check
    with capture_server(OSL, OSL + 3) as (url, _):
        out = tps(url, tmp_path / "a", pool, "--tps-load", "offline",
                  "--requests", "3")
    assert set(level(out)["invalid_reasons"]) == {"usage_mismatch"}
    # short text, usage claiming the full length: invalid on the count
    with capture_server(OSL - 2, OSL) as (url, _):
        out = tps(url, tmp_path / "b", pool, "--tps-load", "offline",
                  "--requests", "3")
    assert set(level(out)["invalid_reasons"]) == {"length"}
    assert level(out)["tps_per_chip"] == pytest.approx(
        3 * (OSL - 2) / level(out)["window_s"])
    # no usage at all: the client's count stands
    with capture_server(OSL, None) as (url, _):
        out = tps(url, tmp_path / "c", pool, "--tps-load", "offline",
                  "--requests", "3")
    assert level(out)["n_invalid"] == 0


def test_reasoning_tokens_count_as_output(pool, tmp_path):
    class Reasoning(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_POST(self) -> None:          # noqa: N802
            self.rfile.read(int(self.headers.get("content-length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            half = OSL // 2
            for i in range(OSL):
                key = "reasoning_content" if i < half else "content"
                text = ("" if i in (0, half) else " ") + f"t{i}"
                self.wfile.write(("data: " + json.dumps({"choices": [{
                    "index": 0, "delta": {key: text},
                    "finish_reason": None}]}) + "\n\n").encode())
            self.wfile.write(("data: " + json.dumps({"choices": [{
                "index": 0, "delta": {}, "finish_reason": "length"}]})
                + "\n\ndata: [DONE]\n\n").encode())

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Reasoning)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = tps(f"http://127.0.0.1:{server.server_port}/v1", tmp_path, pool,
                  "--tps-load", "offline", "--requests", "2")
    finally:
        server.shutdown()
        server.server_close()
    assert all(r["tokens_client"] == OSL and r["ok"] for r in out["records"])


# --------------------------------------------------------------------------
# the client checks itself
# --------------------------------------------------------------------------
def test_a_blocked_event_loop_is_reported(mocks, pool):
    """measure_tps records how late the loop wakes; a client that blocks
    its own loop for 120 ms gets a warning in the self-check."""
    args = harness.build_parser().parse_args(
        ["--base-url", mocks().url, "--model", "m"])
    spec = harness.TpsSpec(load="closed", levels=[2], isl=ISL, osl=OSL,
                           prompts=harness.make_prompt_pool(200, 2 * ISL),
                           ramp_s=0.2, window_s=0.8)
    from tokenizer import load_tokenizer

    async def go() -> harness.TpsResult:
        async def hog() -> None:
            await asyncio.sleep(0.4)
            time.sleep(0.12)                      # blocks the event loop
        async with harness.Client(args) as client:
            result, _ = await asyncio.gather(
                harness.measure_tps(client, spec, load_tokenizer("ws")), hog())
        return result

    result = asyncio.run(go())
    assert result.levels[0]["loop_lag_max_ms"] >= 100
    assert any("event loop" in w for w in result.self_check["warnings"])


def test_pool_mode_writes_a_distinct_pool(tmp_path):
    assert harness.main(["--mode", "pool", "--pool-size", "50", "--isl", "32",
                         "--out", str(tmp_path)]) == 0
    rows = metrics.read_jsonl(str(tmp_path / "prompt_pool.jsonl"))
    assert len(rows) == 50
    assert len({r["id"] for r in rows}) == 50
    assert len({r["prompt"] for r in rows}) == 50
    assert all(len(r["prompt"].split()) >= 64 for r in rows)


def test_a_saturating_server_sweep_picks_the_best_level_inside_the_slo(
        mocks, pool, tmp_path):
    """With batch contention (--knee 2) tpot grows past the knee: the high
    level moves more tokens but breaks the TPOT SLO, so the headline is the
    best level that kept it."""
    sat = mocks("honest", knee=2)
    slo = json.dumps({"p99_tpot_ms": 20, "p95_ttft_ms": 2000})
    out = tps(sat.url, tmp_path, pool, "--concurrency", "2,8", "--slo", slo)
    lo, hi = out["metric"]["levels"]
    assert lo["slo_met"] and not hi["slo_met"]
    assert hi["p99_tpot_ms"] > 2.5 * lo["p99_tpot_ms"]
    assert headline(out) == lo["goodput_tps_per_chip"]


# --------------------------------------------------------------------------
# a mix of input sizes (metric.isl as a list)
# --------------------------------------------------------------------------
MIX = "16,48,96"


@pytest.fixture(scope="module")
def mixed(mocks, pool, tmp_path_factory) -> dict:
    return tps(mocks("honest").url, tmp_path_factory.mktemp("mixed"), pool,
               "--isl", MIX, "--concurrency", "1", "--warm-cycles", "1")


def test_a_mix_serves_the_sizes_in_a_fixed_balanced_order(mixed):
    recs = mixed["records"]
    sizes = [r["isl"] for r in recs]
    assert sizes[:6] == [16, 48, 96, 16, 48, 96]          # round robin, from the top
    assert all(r["isl_client"] == r["isl"] for r in recs)  # fitted to its own size
    assert mixed["isl"] == [16, 48, 96] and mixed["warm_cycles"] == 1


def test_the_warm_cycle_touches_every_size_before_the_clock_starts(mixed):
    """A lazily compiling server pays for a shape on first use; with
    --warm-cycles every stream has sent every size before the ramp begins."""
    recs = mixed["records"]
    window = level(mixed)["window"]
    before = [r["isl"] for r in recs if r["t_last"] < window[0] - mixed["ramp_s"]]
    assert set(before) == {16, 48, 96}


def test_a_mix_is_scored_on_whole_answer_speed_and_reports_every_size(mixed):
    s = level(mixed)
    assert set(s["by_isl"]) == {"16", "48", "96"}
    assert all(v["n"] >= 1 and v["decode_tps"] > 0 for v in s["by_isl"].values())
    assert headline(mixed) == pytest.approx(s["stream_tps_per_chip"])
    assert headline(mixed) == pytest.approx(mixed["metric"]["stream_tps"])   # one chip
    m = mixed["metric"]
    assert m["decode_tps"] == pytest.approx(s["decode_tps"])
    assert m["stream_tps"] < m["decode_tps"]              # the prefill costs time
    assert m["by_isl"] == s["by_isl"]
    assert 100 < m["decode_tps"] < 130                    # the mock's 8 ms a token


def test_one_size_is_measured_exactly_as_before(honest):
    assert honest["isl"] == ISL and "by_isl" not in level(honest)
    assert {r["isl"] for r in honest["records"]} == {ISL}
    assert honest["metric"]["stream_tps"] > 0


def test_a_mix_needs_the_closed_loop(mocks, pool, tmp_path):
    with pytest.raises((SystemExit, ValueError)):
        tps(mocks("honest").url, tmp_path, pool, "--isl", MIX,
            "--tps-load", "open", "--rate", "5")
