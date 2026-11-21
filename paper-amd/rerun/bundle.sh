#!/bin/bash
# bundle.sh: make rerun-bundle.tar.gz, everything the machine needs: the judge's code as committed
# (research/amd at HEAD), the data, the runner, the rules.  Unpack on the machine as /root/rerun.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
mkdir -p "$T/rerun/amd"
git -C "$REPO" archive HEAD:research/amd | tar -x -C "$T/rerun/amd"
git -C "$REPO" rev-parse HEAD > "$T/rerun/JUDGE_COMMIT"
cp -R "$HERE/data" "$T/rerun/data"
cp "$HERE/rerun.py" "$HERE/RULES.md" "$HERE/prepare_box.sh" "$T/rerun/"
COPYFILE_DISABLE=1 tar -czf "$HERE/rerun-bundle.tar.gz" -C "$T" rerun
shasum -a 256 "$HERE/rerun-bundle.tar.gz"; ls -la "$HERE/rerun-bundle.tar.gz"
