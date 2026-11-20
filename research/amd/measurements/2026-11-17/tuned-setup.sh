#!/bin/bash
# ij: steps 3-4 of AMD's build_and_repro.sh in the partial image (layer a008fb455282, Dockerfile step 13 done),
# with FlyDSL source at b4071f00 (dev973, commit count 973) and the flydsl 0.4.0.dev973 wheel, because
# the pinned 0.4.0.dev963 wheel is no longer published (2026-11-17).
set -e
docker rm -f tuned-setup >/dev/null 2>&1 || true
docker run --name tuned-setup --network host --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt seccomp=unconfined --shm-size 16g --cpuset-cpus 12-23 \
  -v /root/flydsl-973:/flysrc:ro -v /root/fwheel:/fwheel:ro \
  -e FLYDSL_DIR=/opt/qwen38/flydsl -e DRAFTER_OBJECTS=/opt/qwen38/drafter-objects \
  -e ATOM_QWEN38_MK_CACHE=/opt/qwen38/kernel-cache \
  -e FLYDSL_SRC=/flysrc -e FLYDSL_WHEEL=/fwheel/flydsl-0.4.0.dev973-cp312-cp312-manylinux_2_27_x86_64.whl \
  -w /opt/atom a008fb455282 \
  bash -c 'SETUP_ONLY=1 WORK=/tmp/setup PY=python3 bash blog_benchmark/build_and_repro.sh && rm -rf /tmp/setup "$FLYDSL_DIR/wheel/x"'
docker commit -c 'ENV WORK=/work HF_HOME=/work/hf-cache FLYDSL_DIR=/opt/qwen38/flydsl DRAFTER_OBJECTS=/opt/qwen38/drafter-objects ATOM_QWEN38_MK_CACHE=/opt/qwen38/kernel-cache' \
  -c 'WORKDIR /opt/atom' tuned-setup ij-atom-tuned:dev973
echo built
