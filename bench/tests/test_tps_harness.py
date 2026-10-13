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


def test_an_invalid_request_keeps_the_answer_it_got(mocks, honest, pool, tmp_path):
    """Whoever asks why a request was refused needs the answer itself, not its
    hash. A valid one keeps the hash only: the file stays small."""
    import hashlib
    out = tps(mocks("truncate").url, tmp_path, pool)
    assert out["records"] and not any(r["ok"] for r in out["records"])
    for r in out["records"]:
        assert r["text"] and hashlib.sha1(r["text"].encode("utf-8")).hexdigest() == r["text_sha1"]
    assert any(r["ok"] for r in honest["records"])
    assert not any("text" in r for r in honest["records"] if r["ok"])


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
def capture_server(n_tokens: int, usage: int | None, finish: str = "length",
                   delay_s: float = 0.0):
    """Answers every chat request with n_tokens ws tokens, after delay_s, and
    records the request bodies."""
    bodies: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_POST(self) -> None:          # noqa: N802
            body = json.loads(self.rfile.read(
                int(self.headers.get("content-length") or 0)))
            bodies.append(body)
            time.sleep(delay_s)
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


def test_extra_body_fields_reach_every_request_and_the_summary(pool, tmp_path):
    with capture_server(OSL, OSL) as (url, bodies):
        tps(url, tmp_path, pool, "--tps-load", "offline", "--requests", "3",
            "--extra-body", '{"min_tokens": 7}')
    assert len(bodies) == 3 and all(b["min_tokens"] == 7 for b in bodies)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["extra_body"] == {"min_tokens": 7}


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


# --------------------------------------------------------------------------
# Whole requests: a pool row that names its input size is sent as written
# --------------------------------------------------------------------------
WHOLE_SIZES = (16, 48, 96)


def whole_rows(sizes=WHOLE_SIZES, per_size: int = 40) -> list[dict]:
    """Rows {"id", "isl", "prompt"}. With the harness's 3-word reference line
    in front, a row is its size or up to four words under it, never over."""
    return [{"id": f"w-{size}-{i}", "isl": size,
             "prompt": " ".join(f"s{size}r{i}w{j}" for j in range(size - 3 - i % 2))}
            for size in sizes for i in range(per_size)]


def whole_pool(path: Path, rows: list[dict] | None = None) -> Path:
    metrics.write_jsonl(str(path), whole_rows() if rows is None else rows)
    return path


def whole_run(rows: list[dict], isl=list(WHOLE_SIZES), **spec) -> "harness._TpsRun":
    from tokenizer import load_tokenizer
    return harness._TpsRun(None, harness.TpsSpec(
        load="closed", levels=[1], isl=isl, osl=OSL, prompts=rows, **spec),
        load_tokenizer("ws"))


def test_a_whole_request_goes_out_as_written_behind_the_reference_line():
    rows = whole_rows()
    by_id = {r["id"]: r for r in rows}
    run = whole_run(rows)
    for size in (16, 48, 96, 16, 48, 96):
        prompt_id, content, reused = run.prompt(size)
        head, _, body = content.partition("\n\n")
        assert head.startswith("Request reference: ") and not reused
        assert by_id[prompt_id]["isl"] == size        # from its own size's rows
        assert body == by_id[prompt_id]["prompt"]     # not cut, not padded
    assert whole_run(rows, nonce=False).prompt(48)[1] in {r["prompt"] for r in rows}


def test_whole_requests_are_measured_under_their_own_size(mocks, tmp_path):
    out = tps(mocks("honest").url, tmp_path / "o", whole_pool(tmp_path / "p.jsonl"),
              "--isl", "16,48,96", "--concurrency", "1", "--warm-cycles", "1")
    recs = out["records"]
    assert [r["isl"] for r in recs[:6]] == [16, 48, 96, 16, 48, 96]
    assert all(r["prompt_id"].startswith(f"w-{r['isl']}-") for r in recs)
    assert all(r["isl"] - 4 <= r["isl_client"] <= r["isl"] for r in recs)
    assert {r["isl_client"] for r in recs} - set(WHOLE_SIZES)   # some are under
    s = level(out)
    assert s["n_invalid"] == 0 and set(s["by_isl"]) == {"16", "48", "96"}
    assert out["prompts"] == "whole" and out["isl_tol"] == 0.3


def test_a_cut_pool_says_so_in_the_summary(honest):
    assert honest["prompts"] == "cut"


@pytest.mark.parametrize("words, what", [(14, "too long"), (7, "too short")])
def test_a_whole_request_that_does_not_fit_its_size_is_refused(words, what):
    rows = whole_rows()
    rows[3] = {"id": "w-16-odd", "isl": 16,
               "prompt": " ".join(f"x{j}" for j in range(words))}
    with pytest.raises(SystemExit, match=f"w-16-odd.*{what}"):
        whole_run(rows)
    whole_run(rows, isl=[48, 96])        # a size that is not measured is not checked


def test_isl_tol_sets_how_far_under_its_size_a_whole_request_may_be():
    rows = whole_rows(sizes=(96,))
    rows[0]["prompt"] = " ".join(f"x{j}" for j in range(60))   # 63 of 96 with the line
    with pytest.raises(SystemExit, match="w-96-0.*too short"):
        whole_run(rows, isl=96)
    whole_run(rows, isl=96, isl_tol=0.4)


class _ZerosCount:
    """A word is a token and so is every 0: the reference line varies in length."""

    def count(self, text: str) -> int:
        return len(text.split()) + text.count("0")


def test_the_reference_line_is_sized_the_same_way_every_time():
    from tokenizer import load_tokenizer
    assert harness.nonce_bounds(load_tokenizer("ws")) == (3, 3)
    lo, hi = harness.nonce_bounds(_ZerosCount())
    assert 3 <= lo < hi
    assert all(harness.nonce_bounds(_ZerosCount()) == (lo, hi) for _ in range(3))


def test_a_whole_request_is_judged_the_same_on_every_load():
    """The check does not depend on which reference lines a load happens to draw."""
    lo, hi = harness.nonce_bounds(_ZerosCount())

    def load(words: int) -> None:
        rows = [{"id": "w-96-0", "isl": 96, "prompt": " ".join("x" for _ in range(words))}]
        harness._TpsRun(None, harness.TpsSpec(load="closed", levels=[1], isl=96, osl=OSL,
                                              prompts=rows), _ZerosCount())

    for _ in range(5):
        load(96 - hi)                       # the longest line still fits
        with pytest.raises(SystemExit, match="too long"):
            load(96 - hi + 1)
        with pytest.raises(SystemExit, match="too short"):
            load(67 - lo)                   # one under 0.7 of 96 with the shortest


def test_a_size_with_no_whole_request_is_refused():
    with pytest.raises(SystemExit, match="no request of size 96"):
        whole_run(whole_rows(sizes=(16, 48)))
    rows = whole_rows()
    spent = frozenset(r["id"] for r in rows if r["isl"] == 48)
    with pytest.raises(SystemExit, match="no request of size 48"):
        whole_run(rows, exclude_ids=spent)


def test_a_pool_may_not_mix_whole_and_cut_rows():
    rows = whole_rows() + [{"id": "plain", "prompt": "a b c d e f g h"}]
    with pytest.raises(SystemExit, match="mixes"):
        whole_run(rows)


def test_a_size_that_runs_out_of_whole_requests_flags_its_repeats():
    run = whole_run(whole_rows(per_size=2))
    seen = [run.prompt(16) for _ in range(5)]
    assert [reused for _, _, reused in seen] == [False, False, True, True, True]
    assert not run.prompt(48)[2]          # another size still has fresh rows
    assert run.sampler.n_reused == 3


def test_special_tokens_in_an_answer_are_counted_but_not_read_as_junk(tmp_path):
    """A server asked to show its special tokens (skip_special_tokens: false)
    fills the rest of a forced-length answer with them once the model is done.
    They are tokens the engine produced, so they count; they are not text, so
    the junk check reads the answer without them."""
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers
    vocab = {"[UNK]": 0, **{f"w{i}": i + 1 for i in range(OSL)}}
    hf = tokenizers.Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    hf.pre_tokenizer = pre_tokenizers.Whitespace()
    hf.add_special_tokens(["<pad>"])
    tok_path = tmp_path / "tokenizer.json"
    hf.save(str(tok_path))
    words = [f"w{i}" for i in range(OSL // 2)]

    class Padded(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_POST(self) -> None:          # noqa: N802
            self.rfile.read(int(self.headers.get("content-length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            parts = [("" if i == 0 else " ") + w for i, w in enumerate(words)]
            parts += ["<pad>"] * (OSL - len(words))
            for part in parts:
                self.wfile.write(("data: " + json.dumps({"choices": [{
                    "index": 0, "delta": {"content": part},
                    "finish_reason": None}]}) + "\n\n").encode())
            self.wfile.write(("data: " + json.dumps({"choices": [{
                "index": 0, "delta": {}, "finish_reason": "length"}]})
                + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args: object) -> None:
            pass

    pool = tmp_path / "pool.jsonl"
    metrics.write_jsonl(str(pool), [{"id": f"p{i}", "prompt": "w1 w2 w3"}
                                    for i in range(8)])
    server = ThreadingHTTPServer(("127.0.0.1", 0), Padded)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = tps(f"http://127.0.0.1:{server.server_port}/v1", tmp_path / "o",
                  pool, "--tps-load", "offline", "--requests", "3",
                  "--tokenizer", str(tok_path))
    finally:
        server.shutdown()
        server.server_close()
    assert level(out)["n_invalid"] == 0
    d4, cr = metrics.text_sanity(" ".join(words))
    for r in out["records"]:
        assert r["tokens_client"] == OSL
        assert r["distinct4"] == round(d4, 4) and r["compression"] == round(cr, 4)
