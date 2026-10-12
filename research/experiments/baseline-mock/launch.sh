#!/usr/bin/env bash
# baseline-mock - the cpu-mock target's baseline: bench/mock_server.py, honest.
#
# run.sh mounts the mock server read-only at /opt/inference-judge/mock_server.py in a
# sealed launch (--local passes IJ_MOCK_SERVER and IJ_PY instead). A
# copy of this directory with a one-word `behaviour` file next to it serves
# one of the mock's other behaviours (fast, truncate, quantised, ...): that is
# how the end-to-end tests build their candidates.
set -euo pipefail
here="$(cd -- "$(dirname -- "$0")" && pwd -P)"
behaviour=honest
if [[ -f "$here/behaviour" ]]; then
  behaviour="$(head -n 1 "$here/behaviour" | tr -d '[:space:]')"
fi
exec "${IJ_PY:-python3}" -I "${IJ_MOCK_SERVER:-/opt/inference-judge/mock_server.py}" \
  --behaviour "$behaviour" --port "${PORT:-8000}" \
  --host "${IJ_HOST:-0.0.0.0}" --model "${IJ_SERVED_NAME:-mock}"
