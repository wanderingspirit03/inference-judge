#!/bin/bash
# Run under ./try: does ATOM's /v1/completions give what bench/quality2.py
# asks for (echo=true with logprobs, max_tokens=1)?  Prints the answers.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
export IJ_REF_CONTAINER=ij-echo-probe IJ_REF_PORT=8301
"$HERE/reference/atom-serve.sh" start >/dev/null
for i in $(seq 1 300); do curl -fsS -m 2 http://127.0.0.1:8301/v1/models >/dev/null 2>&1 && break; sleep 1; done
q() { echo "-- $1"; curl -sS -m 300 http://127.0.0.1:8301/v1/completions -H 'Content-Type: application/json' -d "$2" | head -c 900; echo; }
q "greedy, logprobs 5" '{"model":"qwen3.8-27b","prompt":"The capital of France is","max_tokens":4,"temperature":0,"logprobs":5}'
q "echo + logprobs 5, max_tokens 1 (quality2)" '{"model":"qwen3.8-27b","prompt":"The capital of France is Paris","max_tokens":1,"temperature":0,"echo":true,"logprobs":5}'
"$HERE/reference/atom-serve.sh" stop >/dev/null
