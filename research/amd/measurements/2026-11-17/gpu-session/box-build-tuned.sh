#!/bin/bash
# box-build-tuned.sh: build AMD's tuned engine image ij-atom-tuned:dev973 on the GPU machine, as root, by the
# recipe of NOTES.md ("Rebuild the GPU machine", step 3), in one go:
#   1. proaditya/ATOM, branch qwen38-megakernels, commit 352229d6, FLYDSL_VERSION 0.4.0.dev973 in
#      blog_benchmark/build_and_repro.sh (the pinned dev963 wheel is no longer published);
#   2. blog_benchmark/Dockerfile without its setup step (the RUN SETUP_ONLY=1 ... line), as ij-atom-tuned:base;
#   3. in a container of that image: the setup step, then the fix (the wheel's own flydsl frontend in place of
#      the checkout's), the m8 megakernel and the drafter export; committed as ij-atom-tuned:dev973.
# No GPU is needed.  Log: $LOG (default /root/gpu-session/build-tuned.log).  Safe to run again: an existing
# ij-atom-tuned:dev973 is kept unless REBUILD=1.
set -uo pipefail
LOG="${LOG:-/root/gpu-session/build-tuned.log}"
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
echo "== $(date -u +%FT%TZ) box-build-tuned.sh"
if docker image inspect ij-atom-tuned:dev973 > /dev/null 2>&1 && [ "${REBUILD:-0}" != 1 ]; then
  echo "ij-atom-tuned:dev973 is there already"; exit 0
fi
set -e
SRC=/root/gpu-session/atom-src
rm -rf "$SRC"
git clone -q --branch qwen38-megakernels https://github.com/proaditya/ATOM "$SRC"
git -C "$SRC" checkout -q 352229d6
sed -i 's/0\.4\.0\.dev963/0.4.0.dev973/g' "$SRC/blog_benchmark/build_and_repro.sh"
grep -n 'FLYDSL_VERSION' "$SRC/blog_benchmark/build_and_repro.sh" | head -3
n=$(grep -c '^RUN SETUP_ONLY=1' "$SRC/blog_benchmark/Dockerfile" || true)
[ "$n" = 1 ] || { echo "expected one 'RUN SETUP_ONLY=1' line in the Dockerfile, found $n: the recipe changed, stop"; exit 3; }
# cut from the setup step (two lines, continued) to the end: the state of 2026-11-17's image (step 13), whose ENV
# lines the commit below adds back; the pip check step after it was not run that day either
sed -i '/^RUN SETUP_ONLY=1/,$d' "$SRC/blog_benchmark/Dockerfile"
tail -4 "$SRC/blog_benchmark/Dockerfile"
t0=$(date +%s)
docker build -t ij-atom-tuned:base -f "$SRC/blog_benchmark/Dockerfile" "$SRC"
echo "base image built in $(( $(date +%s) - t0 )) s"
# FlyDSL at b4071f00 (commit count 973) and the 0.4.0.dev973 wheel, given to the setup step as tuned-setup.sh did
# on 2026-11-17 (build_and_repro.sh reads FLYDSL_SRC and FLYDSL_WHEEL; its own default is github.com/ROCm/FlyDSL).
FLY=/root/gpu-session/flydsl-973 WHL=/root/gpu-session/fwheel
rm -rf "$FLY" "$WHL"; mkdir -p "$WHL"
git clone -q "${FLYDSL_REPO:-https://github.com/ROCm/FlyDSL}" "$FLY"
git -C "$FLY" checkout -q b4071f00
docker run --rm -v "$WHL:/w" ij-atom-tuned:base python3 -m pip download --no-deps -d /w flydsl==0.4.0.dev973
W=$(basename "$(ls "$WHL"/flydsl-0.4.0.dev973-*.whl)")
docker rm -f ij-tuned-setup > /dev/null 2>&1 || true
docker run --name ij-tuned-setup -w /opt/atom \
  -v "$FLY:/flysrc:ro" -v "$WHL:/fwheel:ro" -e FLYDSL_SRC=/flysrc -e FLYDSL_WHEEL="/fwheel/$W" \
  -e FLYDSL_DIR=/opt/qwen38/flydsl -e DRAFTER_OBJECTS=/opt/qwen38/drafter-objects \
  -e ATOM_QWEN38_MK_CACHE=/opt/qwen38/kernel-cache \
  ij-atom-tuned:base bash -c '
set -e
SETUP_ONLY=1 WORK=/tmp/setup PY=python3 bash blog_benchmark/build_and_repro.sh || echo "setup step failed (expected at the megakernel); the fix follows"
F=$FLYDSL_DIR
ls $F/wheel/x
rm -rf $F/python/flydsl && cp -r $F/wheel/x/flydsl $F/python/flydsl
export ATOM_QWEN38_MK_FLYDSL=$F/python
python3 -c "from atom.models.qwen38_27b.mono.kernels.build import build; print(\"megakernel\", build(\"m8\"))" 2>&1 | tail -5
PYTHONPATH=$F/python FLY_KERNELS_ROOT=$F/src PROTO_ROOT=/opt/atom/atom/models/qwen38_27b/mono/drafter/kernels_src \
  python3 atom/models/qwen38_27b/mono/drafter/aot_runtime.py export --out /opt/qwen38/drafter-objects --P 40968 2>&1 | tail -5
ls /opt/qwen38/drafter-objects
rm -rf /tmp/setup "$F/wheel/x"'
docker commit -c 'ENV WORK=/work HF_HOME=/work/hf-cache FLYDSL_DIR=/opt/qwen38/flydsl DRAFTER_OBJECTS=/opt/qwen38/drafter-objects ATOM_QWEN38_MK_CACHE=/opt/qwen38/kernel-cache' \
  -c 'WORKDIR /opt/atom' ij-tuned-setup ij-atom-tuned:dev973
docker rm -f ij-tuned-setup > /dev/null
echo "built ij-atom-tuned:dev973 in $(( $(date +%s) - t0 )) s"
