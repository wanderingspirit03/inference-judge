#!/bin/bash
# ij, job 4, last attempt: the flydsl 0.4.0.dev973 wheel's own Python frontend (not the checkout's) with its _mlir,
# the kernels from the FlyDSL checkout at b4071f00; then the megakernel build and the drafter export (no GPU needed).
set -x
docker rm -f tuned-setup3 > /dev/null 2>&1
docker run --name tuned-setup3 --cpuset-cpus 16-23 -w /opt/atom \
  -e FLYDSL_DIR=/opt/qwen38/flydsl -e ATOM_QWEN38_MK_CACHE=/opt/qwen38/kernel-cache \
  ij-atom-tuned:fly973 bash -c '
set -e
F=/opt/qwen38/flydsl
ls $F/wheel/x
rm -rf $F/python/flydsl && cp -r $F/wheel/x/flydsl $F/python/flydsl
export ATOM_QWEN38_MK_FLYDSL=$F/python
nice -n 10 python3 -c "from atom.models.qwen38_27b.mono.kernels.build import build; print(\"megakernel\", build(\"m8\"))" 2>&1 | tail -5
PYTHONPATH=$F/python FLY_KERNELS_ROOT=$F/src PROTO_ROOT=/opt/atom/atom/models/qwen38_27b/mono/drafter/kernels_src \
  nice -n 10 python3 atom/models/qwen38_27b/mono/drafter/aot_runtime.py export --out /opt/qwen38/drafter-objects --P 40968 2>&1 | tail -5
ls /opt/qwen38/drafter-objects'
echo "rc $?"
