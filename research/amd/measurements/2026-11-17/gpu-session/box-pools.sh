#!/bin/bash
# box-pools.sh: build the judge's hidden trace pool on the GPU machine, as root (run-session.sh step 2).
#   - a venv with pyarrow and tokenizers (/root/gpu-session/poolvenv);
#   - the dataset nebius/SWE-rebench-openhands-trajectories at revision 35455389ab51..., downloaded here only,
#     checked against the sha256 of 2026-11-17; no login: a gated or refused download stops the script (exit 4), any other dataset problem
#     (no network, another HTTP error, a changed file list, a different sha256) with exit 6;
#   - bench/pools/judge_traces.py (copied to /root/gpu-session/bench by run-session.sh) over the sessions no
#     public pool holds, into /root/gpu-session/pools (mode 700), in chunks.
# Args: SIZES PER_SIZE (default 8000,12000 and 2000; 2000 and 4000 give no rows, 2026-11-18).
# Log: /root/gpu-session/pools.log.
set -uo pipefail
G=/root/gpu-session
SIZES="${1:-8000,12000}"
PER="${2:-2000}"
REPO=nebius/SWE-rebench-openhands-trajectories
REV=35455389ab51bf5e2306bfd436ef72d0f98bf882
SHA=14048dd1fcd22ce094b6e85f8a38f223a9ef1327031aaaad052804870212efa1
MAN=/opt/ij-amd/amd/measurements/2026-11-17/qwen38-agentic-manifest.json
TOK=/srv/models/Qwen3.8-27B/tokenizer.json
exec >> "$G/pools.log" 2>&1
echo "== $(date -u +%FT%TZ) box-pools.sh sizes $SIZES, $PER per size"
set -e
install -d -m 700 "$G/data" "$G/pools"
[ -x "$G/poolvenv/bin/python" ] || { python3 -m venv "$G/poolvenv"; "$G/poolvenv/bin/pip" install -q pyarrow tokenizers; }
P="$G/data/trajectories.parquet"
if ! echo "$SHA  $P" | sha256sum -c --quiet 2>/dev/null; then
  # the file's path at that revision, from the public API; no token is sent or held
  # exit 4: gated or refused (HTTP 401 or 403); exit 6: anything else that stops the pool build
  http_stop() { case "$1" in 401|403) echo "STOP: $2 gave HTTP $1: gated or refused; no login is used"; exit 4 ;;
                *) echo "STOP: $2 gave HTTP $1"; exit 6 ;; esac; }
  # `|| true`: curl's own failure (no network: HTTP code 000) goes through http_stop, not set -e
  code=$(curl -s -o "$G/data/tree.json" -w '%{http_code}' "https://huggingface.co/api/datasets/$REPO/tree/$REV?recursive=1" || true)
  [ "$code" = 200 ] || http_stop "$code" "the dataset's file list"
  F=$(python3 -c "import json,sys; t=json.load(open(sys.argv[1])); p=[x['path'] for x in t if x['path'].endswith('.parquet')]; print(p[0] if len(p)==1 else '')" "$G/data/tree.json")
  [ -n "$F" ] || { echo "STOP: not exactly one parquet file at $REV"; exit 6; }
  code=$(curl -sL -o "$P" -w '%{http_code}' "https://huggingface.co/datasets/$REPO/resolve/$REV/$F" || true)
  [ "$code" = 200 ] || { rm -f "$P"; http_stop "$code" "the download"; }
  echo "$SHA  $P" | sha256sum -c || { echo "STOP: the file's sha256 differs from 2026-11-17's"; exit 6; }
fi
rm -f "$G"/pools/judge-traces-*.jsonl
t0=$(date +%s)
"$G/poolvenv/bin/python" "$G/bench/pools/judge_traces.py" --parquet "$P" --tokenizer "$TOK" --manifest "$MAN" \
  --sizes "$SIZES" --per-size "$PER" --out "$G/pools"
echo "trace pool built in $(( $(date +%s) - t0 )) s: $(ls "$G"/pools/judge-traces-*.jsonl | wc -l) chunks"
chmod 600 "$G"/pools/*
