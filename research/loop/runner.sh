#!/usr/bin/env bash
# The driver-side runner (DESIGN 2.7): reads queue/<id>.json, pre-flights,
# and calls the bench serially, independent of any agent session.  All logic
# is in runner.py; this wrapper only pins the interpreter and isolation.
#
#     loop/runner.sh           run until the run finishes (inference-judge-runner.service)
#     loop/runner.sh --once    at most one queue item, then exit
#
# Environment: IJ_STATE_DIR, IJ_RESEARCH_DIR, IJ_QUEUE_DIR,
# IJ_RESULTS_DIR, IJ_BENCH_CMD (default "sudo -n -u bench
# /opt/bench/run.sh"), IJ_PREFLIGHT_CMD, IJ_RESEARCHER_SUDO,
# IJ_RUNNER_POLL_S, IJ_PY.
set -uo pipefail
LOOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${IJ_PY:-python3}" -I "$LOOP_DIR/runner.py" "$@"
