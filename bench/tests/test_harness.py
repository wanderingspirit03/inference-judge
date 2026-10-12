"""SSE parsing, the streaming client against broken servers, and the canary."""
from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import harness
from harness import StreamState
from metrics import TurnRecord


def chunk(delta: dict, finish: str | None = None) -> str:
    return json.dumps({"choices": [{"index": 0, "delta": delta,
                                    "finish_reason": finish}]})


# --------------------------------------------------------------------------
# A server that sends exactly the bytes a test asks for, so we can reproduce
# the broken streams real providers produce.
# --------------------------------------------------------------------------
@contextlib.contextmanager
def raw_sse_server(lines: list[str], delay_s: float = 0.0,
                   keep_open_s: float = 0.0):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"        # close-delimited body

        def do_POST(self) -> None:           # noqa: N802 - stdlib name
            self.rfile.read(int(self.headers.get("content-length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                for line in lines:
                    if delay_s:
                        time.sleep(delay_s)
                    self.wfile.write(line.encode())
                    self.wfile.flush()
                deadline = time.time() + keep_open_s
                while time.time() < deadline:
                    self.wfile.write(b": keepalive\n\n")   # not a data line
                    self.wfile.flush()
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()


def stream_turns(url: str, n: int = 1, user: str = "hello",
                 tools: list[dict] | None = None, flags: tuple = ()) -> list[dict]:
    """n requests through Client.stream_turn, one after another: one record
    each, as a dict."""
    args = harness.build_parser().parse_args(["--base-url", url, *flags])

    async def go() -> list[dict]:
        rows = []
        async with harness.Client(args) as client:
            for i in range(n):
                rec = TurnRecord()
                await client.stream_turn(
                    [{"role": "user", "content": f"{user} {i}"}], tools, rec)
                rows.append(asdict(rec))
        return rows
    return asyncio.run(go())


def test_multi_token_chunks_are_one_arrival():
    st = StreamState(t0=0.0)
    st.feed(chunk({"content": "Let me check that "}), 0.10)
    st.feed(chunk({"content": "for you now."}), 0.20)
    assert st.text == "Let me check that for you now."
    assert st.n_content_chunks == 2
    assert st.ttft_ms == pytest.approx(100.0)


def test_the_longest_gap_between_content_chunks_is_kept():
    st = StreamState(t0=0.0)
    st.feed(chunk({"content": "one"}), 0.10)
    st.feed(chunk({"content": " two"}), 0.20)      # 100 ms gap
    st.feed(chunk({"content": " three"}), 0.60)    # 400 ms gap
    assert st.max_gap_ms == pytest.approx(400.0)


def test_tool_call_deltas_are_reassembled():
    st = StreamState(t0=0.0)
    st.feed(chunk({"tool_calls": [{"index": 0, "id": "call_1", "function":
                                   {"name": "create_", "arguments": ""}}]}), 0.1)
    st.feed(chunk({"tool_calls": [{"index": 0, "function":
                                   {"name": "reservation", "arguments": '{"party'}}]}), 0.2)
    st.feed(chunk({"tool_calls": [{"index": 0, "function":
                                   {"arguments": '_size": 4}'}}]}), 0.3)
    name, args = harness.parsed_tool_call(st)
    assert name == "create_reservation"
    assert args == {"party_size": 4}


def test_unparsable_tool_arguments_are_kept_not_crashed():
    st = StreamState(t0=0.0)
    st.feed(chunk({"tool_calls": [{"index": 0, "function":
                                   {"name": "x", "arguments": "{oops"}}]}), 0.1)
    name, args = harness.parsed_tool_call(st)
    assert name == "x" and "_unparsed" in args


def test_usage_in_the_final_chunk():
    st = StreamState(t0=0.0)
    st.feed(json.dumps({"choices": [], "usage": {
        "prompt_tokens": 1800, "completion_tokens": 42,
        "prompt_tokens_details": {"cached_tokens": 1024}}}), 0.5)
    assert st.usage["prompt_tokens"] == 1800


def test_done_sentinel_ends_the_stream():
    st = StreamState(t0=0.0)
    assert st.feed("[DONE]", 0.1) is False


def test_malformed_json_is_flagged_not_raised():
    st = StreamState(t0=0.0)
    st.feed("{not json", 0.1)
    assert st.malformed is True


def test_reasoning_deltas_do_not_count_as_content():
    st = StreamState(t0=0.0)
    st.feed(chunk({"reasoning_content": "thinking about the booking policy"}), 0.1)
    assert st.ttft_ms is None
    st.feed(chunk({"content": "Right, here is the answer for you."}), 0.2)
    assert st.ttft_ms == pytest.approx(200.0)


def test_reasoning_effort_goes_into_the_body():
    args = harness.build_parser().parse_args(["--reasoning-effort", "low"])
    client = harness.Client(args)
    body = client.body([{"role": "user", "content": "hi"}], None)
    assert body["reasoning_effort"] == "low"
    assert body["stream_options"] == {"include_usage": True}


def test_no_reasoning_effort_by_default():
    args = harness.build_parser().parse_args([])
    assert "reasoning_effort" not in harness.Client(args).body([], None)


def test_extra_headers_and_api_key(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "sk-abc")
    args = harness.build_parser().parse_args(
        ["--api-key-env", "TEST_KEY", "--header", "X-Trace: 42"])
    client = harness.Client(args)
    assert client.headers["Authorization"] == "Bearer sk-abc"
    assert client.headers["X-Trace"] == "42"


# --------------------------------------------------------------- end to end --
def test_errors_are_recorded_not_raised(mock_server):
    rows = stream_turns(mock_server.url, user="please fail [ERROR]")
    assert rows[0]["error"] is True
    assert rows[0]["error_kind"] == "http_500"
    assert rows[0]["http_status"] == 500


def test_usage_comes_back_in_the_final_chunk(mock_server):
    rows = stream_turns(mock_server.url, n=3)
    assert all(not r["error"] for r in rows)
    assert all(r["tokens_in"] and r["tokens_out"] for r in rows)
    assert all(r["ttft_ms"] <= r["total_ms"] for r in rows)


def test_canary_writes_then_matches(mock_server, tmp_path):
    base_dir = tmp_path / "base"
    harness.main(["--base-url", mock_server.url, "--mode", "canary",
                  "--canary-prompts", "canary/prompts.json",
                  "--out", str(base_dir)])
    written = json.loads((base_dir / "canary.json").read_text())
    assert len(written["results"]) == 10
    assert all(r["prefix"] for r in written["results"])

    again = tmp_path / "again"
    harness.main(["--base-url", mock_server.url, "--mode", "canary",
                  "--canary-prompts", "canary/prompts.json",
                  "--canary-baseline", str(base_dir / "canary.json"),
                  "--out", str(again)])
    assert json.loads((again / "canary.json").read_text())["matched"] == 10


# ------------------------------------------ SSE safety --
def test_a_stream_that_just_stops_is_a_truncated_stream():
    """EOF with no finish_reason and no [DONE] is not a successful reply."""
    lines = ["data: " + chunk({"content": "I can help with that, one moment"}) + "\n\n"]
    with raw_sse_server(lines) as url:
        rows = stream_turns(url)
    assert len(rows) == 1
    assert rows[0]["error"] is True
    assert rows[0]["error_kind"] == "truncated_stream"


def test_a_finish_reason_without_done_is_accepted():
    lines = ["data: " + chunk({"content": "Yes of course, I can do that now."}) + "\n\n",
             "data: " + chunk({}, finish="stop") + "\n\n"]
    with raw_sse_server(lines) as url:
        rows = stream_turns(url)
    assert rows[0]["error"] is False
    assert rows[0]["text"] == "Yes of course, I can do that now."
    # usage was asked for and never sent: recorded, not an error
    assert rows[0]["usage_missing"] is True


def test_a_malformed_chunk_costs_one_turn_and_the_run_continues():
    """The whole point: a bad chunk must not kill the run."""
    lines = ["data: " + chunk({"content": "starting"}) + "\n\n",
             "data: {not json at all\n\n",
             "data: " + chunk({}, finish="stop") + "\n\n",
             "data: [DONE]\n\n"]
    with raw_sse_server(lines) as url:
        rows = stream_turns(url, n=3)
    assert len(rows) == 3, "the run continued after the bad chunk"
    assert all(r["error"] for r in rows)
    assert all(r["error_kind"] == "malformed_stream" for r in rows)


def test_valid_json_of_the_wrong_shape_is_a_bad_chunk():
    """Valid JSON that is not a chat completion chunk used to raise."""
    lines = ['data: {"choices": "not-a-list"}\n\n',
             "data: [DONE]\n\n"]
    with raw_sse_server(lines) as url:
        rows = stream_turns(url, n=2)
    assert len(rows) == 2
    assert all(r["error_kind"] == "bad_chunk" for r in rows)
    assert rows[0]["error_detail"]


def test_feed_never_raises_on_junk():
    st = StreamState(t0=0.0)
    for payload in ('{"choices": [1, 2]}', '{"choices": [{"delta": 7}]}',
                    '["a", "b"]', '"just a string"', '{"choices": {"a": 1}}'):
        assert st.feed(payload, 0.1) is True
    assert st.bad_chunk is not None


def test_the_first_chunk_deadline_is_its_own_timer():
    """SSE comments keep the socket busy; httpx's read timeout would not fire."""
    with raw_sse_server([], keep_open_s=10.0) as url:
        started = time.monotonic()
        rows = stream_turns(url, flags=("--first-chunk-timeout", "1",
                                        "--total-timeout", "30"))
        elapsed = time.monotonic() - started
    assert rows[0]["error_kind"] == "first_chunk_timeout"
    assert elapsed < 8, f"the deadline did not fire ({elapsed:.1f}s)"


# --------------------------------- Tool-call timing -----
def test_first_tool_ms_is_the_first_complete_tool_call():
    st = StreamState(t0=0.0)
    st.feed(chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {
        "name": "create_reservation", "arguments": '{"party'}}]}), 0.1)
    assert st.first_tool_ms is None            # arguments do not parse yet
    st.feed(chunk({"tool_calls": [{"index": 0, "function": {
        "arguments": '_size": 4}'}}]}), 0.3)
    assert st.first_tool_ms == pytest.approx(300.0)
    st.feed(chunk({}, finish="tool_calls"), 0.4)
    assert st.first_tool_ms == pytest.approx(300.0)   # not overwritten


def test_a_tool_call_with_no_arguments_completes_at_the_finish_reason():
    st = StreamState(t0=0.0)
    st.feed(chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {
        "name": "list_slots", "arguments": ""}}]}), 0.1)
    assert st.first_tool_ms is None
    st.feed(chunk({}, finish="tool_calls"), 0.2)
    assert st.first_tool_ms == pytest.approx(200.0)


def test_every_produced_tool_call_is_recorded():
    st = StreamState(t0=0.0)
    st.feed(chunk({"tool_calls": [
        {"index": 0, "function": {"name": "book", "arguments": '{"a": 1}'}},
        {"index": 1, "function": {"name": "cancel", "arguments": '{"b": 2}'}}]}), 0.1)
    calls = harness.parsed_tool_calls(st)
    assert [c["name"] for c in calls] == ["book", "cancel"]
    assert calls[1]["arguments"] == {"b": 2}


def test_tool_turns_are_measured_end_to_end(mock_server):
    tools = [{"type": "function", "function": {
        "name": "create_reservation", "description": "book a table",
        "parameters": {"type": "object", "properties": {
            "restaurant_name": {"type": "string"},
            "party_size": {"type": "integer"}}}}}]
    rows = stream_turns(
        mock_server.url, n=2, tools=tools,
        user='book it [TOOL:create_reservation:{"restaurant_name": "Marlowe", '
             '"party_size": 4}]')
    assert all(not r["error"] for r in rows)
    assert all(r["produced_tool_call"] == "create_reservation" for r in rows)
    assert all(r["tool_call_args"] == {"restaurant_name": "Marlowe",
                                       "party_size": 4} for r in rows)
    assert all(r["first_tool_ms"] is not None for r in rows)


# ------------------------------------------- Canary -----
def test_canary_sends_low_reasoning_effort_and_256_tokens(mock_server, tmp_path):
    out = tmp_path / "canary"
    harness.main(["--base-url", mock_server.url, "--mode", "canary",
                  "--canary-prompts", "canary/prompts.json", "--out", str(out)])
    written = json.loads((out / "canary.json").read_text())
    assert written["reasoning_effort"] == "low"
    assert written["max_tokens"] == 256
    assert written["prefix_words"] == 16
    assert written["baseline_valid"] is True


def test_canary_baseline_refuses_to_be_written_when_a_prompt_fails(tmp_path):
    """An unusable baseline fails every later candidate: refuse it loudly."""
    prompts = {"prefix_tokens": 16, "prompts": [
        {"id": f"c{i:02d}", "system": "s", "user": "hello"} for i in range(10)]}
    pfile = tmp_path / "prompts.json"
    pfile.write_text(json.dumps(prompts))
    with raw_sse_server([]) as url:          # answers nothing useful
        rc = harness.main(["--base-url", url, "--mode", "canary",
                           "--canary-prompts", str(pfile),
                           "--out", str(tmp_path / "c")])
    assert rc != 0
    written = json.loads((tmp_path / "c" / "canary.json").read_text())
    assert written["baseline_valid"] is False
    assert len(written["invalid_ids"]) == 10


def test_canary_counts_errors_as_mismatches(tmp_path):
    baseline = {"results": [{"id": f"c{i:02d}", "prefix": "Paris is the capital"}
                            for i in range(10)]}
    bfile = tmp_path / "baseline.json"
    bfile.write_text(json.dumps(baseline))
    prompts = {"prompts": [{"id": f"c{i:02d}", "system": "s", "user": "hi"}
                           for i in range(10)]}
    pfile = tmp_path / "prompts.json"
    pfile.write_text(json.dumps(prompts))
    with raw_sse_server([]) as url:
        harness.main(["--base-url", url, "--mode", "canary",
                      "--canary-prompts", str(pfile),
                      "--canary-baseline", str(bfile),
                      "--out", str(tmp_path / "c")])
    written = json.loads((tmp_path / "c" / "canary.json").read_text())
    assert written["matched"] == 0
    assert written["passed"] is False


def test_canary_prefix_is_sixteen_whitespace_words():
    text = " ".join(f"w{i}" for i in range(30))
    assert harness.prefix_words(text, 16).split() == [f"w{i}" for i in range(16)]
    assert harness.prefix_words("  spaced   out  ", 16) == "spaced out"


# ------------------ Malformed events cost one turn ------
def test_mixed_tool_call_index_types_do_not_break_the_reader():
    """Some servers send 0, some "0". Both used to land in the same dict and
    the next sorted() raised TypeError inside _fill(), killing the harness."""
    st = StreamState(t0=0.0)
    st.feed(chunk({"tool_calls": [{"index": 0, "id": "c1", "function": {
        "name": "book", "arguments": '{"a":'}}]}), 0.1)
    st.feed(chunk({"tool_calls": [{"index": "0", "function": {
        "arguments": ' 1}'}}]}), 0.2)
    assert list(st.tool_calls) == [0]
    assert st.bad_chunk is None
    name, args = harness.parsed_tool_call(st)
    assert name == "book" and args == {"a": 1}


def test_an_unusable_tool_call_index_is_a_bad_chunk_not_a_crash():
    st = StreamState(t0=0.0)
    for payload in (chunk({"tool_calls": [{"index": {"a": 1}, "function": {
                        "name": "x", "arguments": "{}"}}]}),
                    chunk({"tool_calls": [{"index": "abc", "function": {
                        "name": "x", "arguments": "{}"}}]}),
                    chunk({"tool_calls": [{"index": 0, "function": 7}]}),
                    chunk({"tool_calls": [{"index": 0, "function": {
                        "name": ["x"], "arguments": "{}"}}]}),
                    chunk({"tool_calls": ["not-an-object"]})):
        assert st.feed(payload, 0.1) is True
    assert st.bad_chunk is not None
    harness.parsed_tool_calls(st)               # must not raise


def test_tool_indices_orders_mixed_key_types():
    assert harness.tool_indices({1: {}, 0: {}, "late": {}}) == [0, 1, "late"]


def test_a_mixed_index_stream_costs_one_turn_and_the_run_continues():
    """End to end: a malformed tool-call event is an errored turn, never an
    uncaught exception out of the harness."""
    lines = ["data: " + chunk({"tool_calls": [{"index": 0, "function": {
                 "name": "book", "arguments": "{}"}}]}) + "\n\n",
             "data: " + chunk({"tool_calls": [{"index": "not-a-number",
                 "function": {"arguments": "{}"}}]}) + "\n\n",
             "data: " + chunk({}, finish="tool_calls") + "\n\n",
             "data: [DONE]\n\n"]
    with raw_sse_server(lines) as url:
        rows = stream_turns(url, n=3)
    assert len(rows) == 3, "the run continued after the malformed event"
    assert all(r["error"] and r["error_kind"] == "bad_chunk" for r in rows)
    assert all(r["error_detail"] for r in rows)


# ----------------------------- Canary budget ------------
def test_a_canary_baseline_truncated_before_the_prefix_is_refused(tmp_path):
    """finish_reason "length" before 16 words means the budget was too small;
    freezing that into the baseline fails every later candidate."""
    results = [{"id": "c01", "text": "Paris is the", "error": None,
                "prefix": "Paris is the", "finish_reason": "length",
                "truncated_before_prefix": True}]
    assert harness.check_canary_baseline(results)["baseline_valid"] is False
    ok = [{"id": "c01", "text": "Paris.", "error": None, "prefix": "Paris.",
           "finish_reason": "stop", "truncated_before_prefix": False}]
    assert harness.check_canary_baseline(ok)["baseline_valid"] is True


def test_the_canary_records_the_finish_reason(mock_server, tmp_path):
    out = tmp_path / "canary"
    harness.main(["--base-url", mock_server.url, "--mode", "canary",
                  "--canary-prompts", "canary/prompts.json", "--out", str(out)])
    written = json.loads((out / "canary.json").read_text())
    assert written["truncated_before_prefix"] == 0
    assert all("finish_reason" in r for r in written["results"])


# ------------------------------------------- a server that wedges mid-canary --
@contextlib.contextmanager
def wedging_server(hang):
    """Answers a chat completion unless `hang(n)` says the n-th one (from 0)
    is never answered: ready, then wedged."""
    stop, seen = threading.Event(), []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            seen.append(self.path)
            if hang(len(seen) - 1):
                stop.wait(60)
                return
            body = json.dumps({"choices": [{"finish_reason": "stop", "message": {
                "content": "Paris is the capital of France."}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", seen
    finally:
        stop.set()
        server.shutdown()
        server.server_close()


def _canary(url, tmp_path, *extra):
    harness.main(["--base-url", url, "--mode", "canary",
                  "--canary-prompts", "canary/prompts.json",
                  "--canary-timeout", "0.5", "--out", str(tmp_path / "c"), *extra])
    return json.loads((tmp_path / "c" / "canary.json").read_text())


def test_the_canary_stops_after_two_timeouts_in_a_row(tmp_path):
    """A server that went ready and then wedged used to cost ten full request
    timeouts; two in a row are enough to know."""
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"results": [
        {"id": f"c{i:02d}", "prefix": "x"} for i in range(1, 11)]}))
    started = time.time()
    with wedging_server(lambda n: n >= 1) as (url, seen):
        written = _canary(url, tmp_path, "--canary-baseline", str(base))
    assert time.time() - started < 10
    assert len(seen) == 3                      # one answer, two timeouts, stop
    assert "2 consecutive request timeouts" in written["aborted"]
    assert written["passed"] is False and written["total"] == 10
    assert sum("not sent" in (r["error"] or "") for r in written["results"]) == 7


def test_canary_timeouts_that_are_not_in_a_row_do_not_stop_it(tmp_path):
    with wedging_server(lambda n: n in (1, 3, 5)) as (url, seen):
        written = _canary(url, tmp_path)
    assert len(seen) == 10 and not written.get("aborted")
    assert sum(1 for r in written["results"] if r["error"]) == 3
