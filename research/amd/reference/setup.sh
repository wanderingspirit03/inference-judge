#!/bin/bash
# setup.sh - the steps that made the reference on the MI350X box on 2026-11-16,
# as run there (as root).  Everything large goes onto the box, nothing to the
# controller.  The model is not gated: no login, no token.
#
#   image:  rocm/atom-dev:latest = rocm/atom-dev@sha256:ad256a94209d2f58bfa6b0a69c555e0f8aab4fd8b76673c3532258803599099c
#           (ATOM 0.1.10.dev141+g273c78de5, torch 2.10.0+rocm7.2.4)
#   model:  Qwen/Qwen3.8-27B, BF16, 18 safetensors files, 51 GB; Hub commit of
#           the files as fetched: see .cache/huggingface/download/*.metadata
#           (first line), 37561816e66b5b9153b1b64c10968c18c1e6f4a0 for config.json
#   draft:  incoai/Qwen3.8-27B-DFlash2 (the tuned engine's draft model), into $MODELS/Qwen3.8-27B-DFlash2
#
# controller/box/setup-box.sh runs it on every new GPU machine.  Safe to run
# again: the venv is kept, the download resumes, the pull is a no-op when the
# image is there.  IJ_REF_IMAGE pins another image (for example the digest
# above); the default is the digest above (the tag moves; the reference must not).
set -euo pipefail
MODELS=${IJ_MODELS:-/srv/models}
IMAGE=${IJ_REF_IMAGE:-rocm/atom-dev@sha256:ad256a94209d2f58bfa6b0a69c555e0f8aab4fd8b76673c3532258803599099c}

docker pull "$IMAGE"
# The reference runs the pinned image plus one patch: a logprob entry for every generated token
# (reference/atom_token_logprobs.py), built here as ij-atom-ref:ad256a94-lp1.
bash "$(dirname "$(readlink -f "$0")")/build-ref-image.sh"

[ -x /opt/hfvenv/bin/python ] || python3 -m venv /opt/hfvenv
/opt/hfvenv/bin/pip -q install "huggingface_hub[hf_transfer]"
mkdir -p "$MODELS"
/opt/hfvenv/bin/python -c "from huggingface_hub import snapshot_download; \
print(snapshot_download('Qwen/Qwen3.8-27B', local_dir='$MODELS/Qwen3.8-27B', max_workers=16))"
# The draft model of AMD's tuned engine (examples/amd-tuned-engine: --draft-model /models/Qwen3.8-27B-DFlash2),
# fetched by hand on 2026-11-18; not gated, no login.  Engines read it from $MODELS like the main model.
/opt/hfvenv/bin/python -c "from huggingface_hub import snapshot_download; \
print(snapshot_download('incoai/Qwen3.8-27B-DFlash2', local_dir='$MODELS/Qwen3.8-27B-DFlash2', max_workers=16))"
chmod -R a+rX "$MODELS"

mkdir -p /var/cache/ij-ref
# Then: reference/atom-serve.sh start (the commands start it themselves).
