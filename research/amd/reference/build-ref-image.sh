#!/bin/bash
# Build the reference image: the pinned ATOM image plus atom_token_logprobs.py (one logprob entry per generated
# token in the chat answer; see that file). Needs no GPU; about a minute. Prints the tag it built.
set -euo pipefail
HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
BASE="rocm/atom-dev@sha256:ad256a94209d2f58bfa6b0a69c555e0f8aab4fd8b76673c3532258803599099c"
TAG="${1:-ij-atom-ref:ad256a94-lp1}"
CTX=$(mktemp -d)
trap 'rm -rf "$CTX"' EXIT
cp "$HERE/atom_token_logprobs.py" "$CTX/"
cat > "$CTX/Dockerfile" <<EOF
FROM $BASE
COPY atom_token_logprobs.py /opt/ij/atom_token_logprobs.py
RUN python3 /opt/ij/atom_token_logprobs.py && python3 -c "import atom.entrypoints.openai.serving_chat as s; assert s._ij_token_entry"
EOF
docker build -q -t "$TAG" "$CTX" >/dev/null
echo "$TAG"
