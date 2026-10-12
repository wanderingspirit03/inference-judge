# shellcheck shell=bash
# The NVIDIA GPU box, for the setup scripts (bootstrap.sh, check.sh).  Sourced, not run.
# Everything here reads the T_* variables of setup/target.sh; nothing is named for
# one cloud, one card or one model.  Each function is one bootstrap step: it is
# idempotent, prints what it did, and returns non-zero with a message that says
# what to do.
#
# UNPROVEN ON REAL HARDWARE.  These steps have only run against fakes
# (setup/tests/test_gpu.py); no GPU has been through them.
#
# What the box must have before bootstrap: Ubuntu, an NVIDIA driver (nvidia-smi
# works), and Docker with the NVIDIA container toolkit (docker run --gpus all
# works).  Bootstrap checks all three and installs none of them.
#
#   gpu_devices      the host shows exactly accelerator.chips GPUs (the ruler launches
#                    with --gpus all, so more is a different machine than the target
#                    describes); they are accelerator.type and have
#                    accelerator.hbm_gb_per_chip of memory when the target says so;
#                    the driver is at least gpu_min_driver, when there is a minimum
#   gpu_passthrough  docker reaches the GPU.  An engine built here is probed with the
#                    CUDA base it is built on; a prebuilt image is its own probe (gpu_gate)
#   gpu_gate         THE gate for a prebuilt image: in that image, with the target's own
#                    docker_flags and env, as the ruler starts it, nvidia-smi lists
#                    accelerator.chips GPUs.  The image carries its own CUDA and its own
#                    minimum driver, and the container toolkit refuses to start it on an
#                    older one; this is the test that says so
#   gpu_not_runnable not a step: what bootstrap prints when it stops after the gate
#                    because the target is runnable: false
#   gpu_nsight       Nsight Systems on the host (optional)
# For an engine built here from setup/Dockerfile.base (T_ENGINE_CLASS built):
#   require_engine_pins  the operator named a release tag for every engine it builds
#   gpu_channel      the torch wheel channel and the tags, recorded
#   gpu_host_venv    the researcher's host venv with the same wheels
#   gpu_base_image   inference-judge-base
#
# Callers define as_researcher (bootstrap.sh) and SCRIPT_DIR.  Test seams are environment
# variables with the production values as defaults: IJ_GPU_DRIVER_FILE,
# IJ_GPU_CHANNEL_FILE, IJ_CUDA_PROBE_IMAGE.

gpu_say()  { printf '   %s\n' "$*"; }
gpu_warn() { printf '   WARN %s\n' "$*" >&2; }

GPU_DRIVER_FILE="${IJ_GPU_DRIVER_FILE:-/run/inference-judge-driver-version}"
GPU_CHANNEL_FILE="${IJ_GPU_CHANNEL_FILE:-/run/inference-judge-channel}"

# The engines setup/Dockerfile.base can build, and what its wheels need.  Every
# release of both that this repo has built pins torch with CUDA 13 wheels (channel
# cu130), and CUDA 13 needs driver r580 or newer.  A prebuilt image is not held to
# this: it states its own minimum and gpu_gate finds out.
GPU_BUILDABLE="vllm sglang"
GPU_BUILT_CHANNEL=cu130
GPU_BUILT_MIN_DRIVER=580

# gpu_probe_image: the small image the passthrough probe runs, the same CUDA release
# setup/Dockerfile.base builds on (its ARG CUDA_IMAGE, -base- instead of -devel-).
gpu_probe_image() {
  local img="${IJ_CUDA_PROBE_IMAGE:-}"
  if [ -z "$img" ]; then
    img=$(sed -n 's/^ARG CUDA_IMAGE=//p' "$SCRIPT_DIR/Dockerfile.base" 2>/dev/null | head -1)
    img="${img/-devel-/-base-}"
  fi
  printf '%s' "$img"
}
GPU_PROBE_IMAGE="$(gpu_probe_image)"

# gpu_count: how many GPUs nvidia-smi lists on the host.
gpu_count() { nvidia-smi -L 2>/dev/null | grep -c '^GPU [0-9]' || true; }

# gpu_min_driver: the lowest driver major this target accepts ("" = any), and
# gpu_min_driver_why: where that number comes from.
gpu_min_driver() {
  local m
  m=$(tget_opt provision.min_driver_major "")
  if [ -z "$m" ] && [ "$T_ENGINE_CLASS" = built ]; then m=$GPU_BUILT_MIN_DRIVER; fi
  printf '%s' "${m%%.*}"
}
gpu_min_driver_why() {
  if [ -n "$(tget_opt provision.min_driver_major "")" ]; then printf 'provision.min_driver_major in %s' "$T_ID"
  else printf 'the engines built here use CUDA 13 wheels (torch channel %s)' "$GPU_BUILT_CHANNEL"; fi
}

# The four things gpu_devices checks, one function each so check.sh can ask them
# one at a time.  Each prints one line and returns non-zero when the answer is no.
gpu_check_count() {
  local n
  n=$(gpu_count)
  if [ "${n:-0}" -eq "$T_CHIPS" ]; then echo "$n GPU(s) on the host, as accelerator.chips says"; return 0; fi
  echo "nvidia-smi shows ${n:-0} GPU(s), target $T_ID wants $T_CHIPS (accelerator.chips)" >&2
  echo "  the ruler launches with --gpus all, so the engine would use every GPU on the host:" >&2
  echo "  use a machine with exactly $T_CHIPS, or a target written for this one" >&2
  return 1
}

gpu_check_type() {
  local want names lower_want
  want=$(tget_opt accelerator.type "")
  names=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort -u | tr '\n' ';' | sed 's/;$//')
  if [ -z "$want" ]; then echo "$names (the target names no accelerator.type)"; return 0; fi
  lower_want=$(printf '%s' "$want" | tr '[:upper:]' '[:lower:]')
  if [ -n "$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '[:upper:]' '[:lower:]' | grep -vF -- "$lower_want" || true)" ]; then
    echo "accelerator.type is '$want' in $T_ID, but the host has: $names" >&2
    echo "  numbers measured on another card are not this target's: use the card it names, or another target" >&2
    return 1
  fi
  echo "$names (accelerator.type $want)"
}

# A card reports a little less than its nominal size (an "80 GB" card 81559 MiB, a
# "48 GB" one 46068 MiB), so the floor is 95% of the nominal GB counted in MiB.
gpu_check_memory() {
  local want floor least
  want=$(tget_opt accelerator.hbm_gb_per_chip "")
  least=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tr -dc '0-9\n' | sort -n | head -1)
  if [ -z "$want" ]; then echo "${least:-?} MiB per GPU (the target names no accelerator.hbm_gb_per_chip)"; return 0; fi
  floor=$(( ${want%%.*} * 950 ))
  if [ "${least:-0}" -lt "$floor" ]; then
    echo "accelerator.hbm_gb_per_chip is $want in $T_ID, but a GPU here has ${least:-0} MiB" >&2
    echo "  the target's model and flags were sized for a ${want} GB card" >&2
    return 1
  fi
  echo "${least} MiB per GPU (accelerator.hbm_gb_per_chip $want)"
}

gpu_check_driver() {
  local drv min
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  min=$(gpu_min_driver)
  if [ -z "$min" ]; then echo "driver r$drv (no minimum: the engine image states its own, see the gpu gate)"; return 0; fi
  case "${drv%%.*}" in
    ""|*[!0-9]*) echo "nvidia-smi gave no driver version ('$drv'), and $T_ID needs r$min or newer" >&2; return 1 ;;
  esac
  if [ "${drv%%.*}" -lt "$min" ]; then
    echo "driver r$drv is older than r$min, which $T_ID needs: $(gpu_min_driver_why)" >&2
    echo "  install NVIDIA driver r$min or newer on the host (or start from a machine image that has it)," >&2
    echo "  then run this again; retrying on this driver cannot work" >&2
    return 1
  fi
  echo "driver r$drv (r$min or newer: $(gpu_min_driver_why))"
}

# gpu_devices: the host has the GPUs the target describes.  Failing here costs two
# minutes; failing at the baseline costs the whole install and the GPU time to find out.
gpu_devices() {
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi missing: a gpu target needs an NVIDIA driver on the host" >&2; return 1; }
  nvidia-smi -L
  local c out
  for c in gpu_check_count gpu_check_type gpu_check_memory gpu_check_driver; do
    out=$("$c") || return 1
    gpu_say "$out"
  done
  nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 > "$GPU_DRIVER_FILE"
}

# gpu_passthrough: docker can start a container that sees the GPU.
gpu_passthrough() {
  if [ "$T_ENGINE_CLASS" = pulled ]; then
    # The proof needs the engine image, which is not here yet: gpu_gate runs it
    # after the pull, in that image, so no second image is pulled just to probe.
    gpu_say "gpu: passthrough is proved by the gpu gate, in the engine image, after it is pulled"
    return 0
  fi
  docker run --rm --gpus all "$GPU_PROBE_IMAGE" nvidia-smi -L \
    || { echo "docker cannot start a container with the GPU: the NVIDIA container toolkit must be installed and configured for docker (nvidia-ctk runtime configure --runtime=docker)" >&2; return 1; }
}

# gpu_gate_args: GPU_GATE_ARGS = the docker flags the ruler launches this target's
# engine with (run.sh seal_args: a network seal unless the flags name a network,
# every GPU, shm size, docker_flags, env).
gpu_gate_args() {
  local f has_net=0 e
  GPU_GATE_ARGS=()
  for f in ${T_DOCKER_FLAGS[@]+"${T_DOCKER_FLAGS[@]}"}; do
    case "$f" in --network*|--net=*|--net) has_net=1 ;; esac
  done
  [ "$has_net" = 1 ] || GPU_GATE_ARGS+=(--network none)
  GPU_GATE_ARGS+=(--gpus all)
  if [ -n "$T_SHM" ]; then GPU_GATE_ARGS+=(--shm-size "$T_SHM"); fi
  GPU_GATE_ARGS+=(${T_DOCKER_FLAGS[@]+"${T_DOCKER_FLAGS[@]}"})
  for e in ${T_ENV[@]+"${T_ENV[@]}"}; do GPU_GATE_ARGS+=(-e "$e"); done
}

# gpu_gate_run: one `bash -c 'nvidia-smi -L'` container, started the way the ruler
# starts a launch (`<image> bash -c ...`), so an image entrypoint the target's flags
# do not clear fails here and not at the baseline.  Returns docker's exit code;
# leaves the output in $GPU_GATE_OUT and the number of GPUs listed in $GPU_GATE_SEEN.
gpu_gate_run() {
  local rc=0
  gpu_gate_args
  GPU_GATE_SEEN=0
  GPU_GATE_OUT=$(docker run --rm ${GPU_GATE_ARGS[@]+"${GPU_GATE_ARGS[@]}"} "$T_IMAGE" bash -c 'nvidia-smi -L' 2>&1) || rc=$?
  [ "$rc" -eq 0 ] && GPU_GATE_SEEN=$(printf '%s\n' "$GPU_GATE_OUT" | grep -c '^GPU [0-9]' || true)
  return "$rc"
}

gpu_gate() {
  local rc=0
  gpu_gate_args
  gpu_say "gpu gate: nvidia-smi must list $T_CHIPS GPU(s) (docker run ${GPU_GATE_ARGS[*]} $T_IMAGE)"
  gpu_gate_run || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "gpu gate FAILED: $T_IMAGE did not start with the GPU (exit $rc)" >&2
    printf '%s\n' "$GPU_GATE_OUT" | tail -3 | sed 's/^/  /' >&2
    case "$GPU_GATE_OUT" in
      *"requirement error"*|*"unsatisfied condition"*)
        echo "  the host's NVIDIA driver is too old for this image (the image states the CUDA it needs," >&2
        echo "  and the container toolkit refuses an older driver): install a newer driver on the host" >&2 ;;
      *)
        echo "  the usual causes, most likely first:" >&2
        echo "  - the NVIDIA container toolkit is not installed, or not configured for docker" >&2
        echo "    (nvidia-ctk runtime configure --runtime=docker, then restart docker)" >&2
        echo "  - the image has an entrypoint that engine.docker_flags in $T_ID does not clear" >&2
        echo "    (the ruler starts '<image> bash -c ...'; --entrypoint= clears one)" >&2
        echo "  - the image has no bash or no nvidia-smi" >&2 ;;
    esac
    return 1
  fi
  if [ "$GPU_GATE_SEEN" -ne "$T_CHIPS" ]; then
    echo "gpu gate FAILED: the engine image sees $GPU_GATE_SEEN GPU(s), target $T_ID wants $T_CHIPS" >&2
    echo "  the host showed $T_CHIPS, so the container flags (engine.docker_flags, engine.env) hide them" >&2
    return 1
  fi
  gpu_say "gpu gate passed: $GPU_GATE_SEEN GPU(s) in the engine image"
}

# gpu_not_runnable: where bootstrap stops for a target that is runnable: false, and
# what the operator does about it.  Nothing the ruler would lock has happened.  It
# only prints: bootstrap's run_plan does the stopping.
gpu_not_runnable() {
  cat <<EOF
bootstrap stopped here on purpose: target $T_ID is runnable: false.
  Done: every step up to the gpu gate (the checklist above): the host has the GPUs the
  target describes and the engine image starts on them with the target's own flags.
  Not done: every step after the gate.  The run clock is not started and the ruler is
  not locked.
  The target file is  $T_FILE
  Next, in this order:
  1. Read the comment at the top of the target file: it lists what nobody has verified
     (the image, the serve flags, whether the weights fit).  The gate proves none of that:
     it started the image, it did not load a model.
  2. Finish the file's edits now (engine.docker_flags, engine.env, the budgets): the
     ruler locks its bytes when the clock starts.  The serve command is
     research/experiments/$T_BASELINE/launch.sh; a model that needs its own flags gets
     them in a baseline the target names (provision.baseline).
  3. Set  runnable: true  in the target file (the comment above that line says what
     it claims).
  4. Run bootstrap again; it repeats the gate with the file as it then is and carries
     on to the end.  With --no-clock it installs everything and leaves the clock for a
     later run, which is the place to see the first real launch of $T_BASELINE (the
     reference run, docs/RUNBOOK.md) before any hour is counted:
       sudo bash setup/bootstrap.sh --target $T_FILE --no-clock
EOF
}

# gpu_nsight: nsys on the host, from NVIDIA's devtools repo for this Ubuntu and CPU.
gpu_nsight() {
  if command -v nsys >/dev/null; then gpu_say "nsys already installed: $(nsys --version | head -1)"; return 0; fi
  local repo
  # shellcheck disable=SC1091
  repo="https://developer.download.nvidia.com/devtools/repos/ubuntu$(. /etc/os-release && printf '%s' "${VERSION_ID//./}")/$(dpkg --print-architecture)"
  curl -fsSL "$repo/nvidia.pub" | gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg
  echo "deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] $repo/ /" > /etc/apt/sources.list.d/nvidia-devtools.list
  DEBIAN_FRONTEND=noninteractive apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y nsight-systems-cli
  nsys --version | head -1
}

# ------------------------------------------------ an engine built on the box
# gpu_built_engines: the engines setup/Dockerfile.base builds for this target.  vllm
# always (it is the image's first venv, the one `python` and deps.txt mean); sglang
# only when the target serves it or lists its tree.
gpu_built_engines() {
  local n out=vllm
  for n in "$T_ENGINE" ${T_SOURCES[@]+"${T_SOURCES[@]}"}; do
    [ "$n" = sglang ] && out="vllm sglang"
  done
  printf '%s' "$out"
}

# gpu_engine_tag NAME: the release tag the operator pinned for a built engine.
gpu_engine_tag() {
  case "$1" in
    vllm)   printf '%s' "${VLLM_TAG:-${IJ_VLLM_TAG:-}}" ;;
    sglang) printf '%s' "${SGLANG_TAG:-${IJ_SGLANG_TAG:-}}" ;;
  esac
}

# An engine tag must be spelled out as a release tag (v0.29.0, v0.5.19, v0.5.20rc1);
# "latest", "main" and branch names are refused.
GPU_RELEASE_TAG_RE='^v[0-9]+\.[0-9]+(\.[0-9]+)?((rc|\.post)[0-9]+)?$'
require_engine_pins() {
  local name val bad=0 e n
  for n in ${T_SOURCES[@]+"${T_SOURCES[@]}"}; do
    case " $GPU_BUILDABLE " in
      *" $n "*) ;;
      *) echo "engine.sources names '$n', which setup/Dockerfile.base does not build (it builds: $GPU_BUILDABLE)." >&2
         echo "  A tree from anywhere else goes with a prebuilt engine.image and provision.sources.$n (url, ref)." >&2
         bad=1 ;;
    esac
  done
  case " ${T_SOURCES[*]-} " in
    *" $T_ENGINE "*) ;;
    *) echo "an engine built on the box is built from its tree: add engine.sources: [$T_ENGINE] to $T_ID" >&2
       bad=1 ;;
  esac
  for e in $(gpu_built_engines); do
    name="IJ_$(printf '%s' "$e" | tr '[:lower:]' '[:upper:]')_TAG"
    val="${!name:-}"
    if [ -z "$val" ]; then
      echo "$name is required: pin the engine release explicitly, e.g. IJ_VLLM_TAG=v0.29.0 IJ_SGLANG_TAG=v0.5.19" >&2
      bad=1
    elif ! [[ "$val" =~ $GPU_RELEASE_TAG_RE ]]; then
      echo "$name=$val is not a release tag like v0.29.0 (refusing latest, main or a branch)" >&2
      bad=1
    fi
  done
  [ "$bad" = 0 ] || exit 2
}

# gpu_channel: which release tags and which torch wheel channel the box is built
# from.  There is exactly one channel (see GPU_BUILT_CHANNEL above); the tags are the
# operator's (checked up front by require_engine_pins; never "latest").  An engine
# the target does not name has an empty tag and is not built.
gpu_channel() {
  local drv major
  drv=$(cat "$GPU_DRIVER_FILE" 2>/dev/null || echo 0)
  major=${drv%%.*}
  TORCH_BACKEND="${IJ_TORCH_BACKEND:-$GPU_BUILT_CHANNEL}"
  if [ "$TORCH_BACKEND" != "$GPU_BUILT_CHANNEL" ]; then
    echo "IJ_TORCH_BACKEND=$TORCH_BACKEND is not supported." >&2
    echo "$GPU_BUILT_CHANNEL is the only channel setup/Dockerfile.base builds: the engine" >&2
    echo "releases it installs pin torch with CUDA 13 wheels." >&2
    return 1
  fi
  if [ "${major:-0}" -lt "$(gpu_min_driver)" ]; then
    echo "driver r$drv cannot run $GPU_BUILT_CHANNEL wheels (need r$(gpu_min_driver) or newer): install a newer NVIDIA driver on the host" >&2
    return 1
  fi
  VLLM_TAG="${IJ_VLLM_TAG:?IJ_VLLM_TAG is required (e.g. v0.29.0)}"
  SGLANG_TAG=""
  case " $(gpu_built_engines) " in
    *" sglang "*) SGLANG_TAG="${IJ_SGLANG_TAG:?IJ_SGLANG_TAG is required (e.g. v0.5.19)}" ;;
  esac
  gpu_say "driver $drv -> torch channel $TORCH_BACKEND, vllm $VLLM_TAG, sglang ${SGLANG_TAG:-not built}"
  printf 'TORCH_BACKEND=%s\nVLLM_TAG=%s\nSGLANG_TAG=%s\n' \
    "$TORCH_BACKEND" "$VLLM_TAG" "$SGLANG_TAG" > "$GPU_CHANNEL_FILE"
  # /run is gone after a reboot; keep a durable record of what this box was built from.
  printf '# written by setup/bootstrap.sh %s\nVLLM_TAG=%s\nSGLANG_TAG=%s\nTORCH_BACKEND=%s\n' \
    "$(date -u +%FT%TZ)" "$VLLM_TAG" "$SGLANG_TAG" "$TORCH_BACKEND" > "$ENGINE_PINS_FILE"
  chmod 0644 "$ENGINE_PINS_FILE"
}

# The host venv for user `researcher`, for a built engine.  IT IS NOT WHERE ANYTHING
# IS SERVED: every experiment runs in a container (setup/Dockerfile.base).  It
# installs the SAME PyPI wheels the image installs.
# setup/harden_research.sh later retires it from the researcher (torch on the
# host is accelerator access by another name) unless --keep-gpu-venv.
gpu_host_venv() {
  local vllm_ver="${VLLM_TAG#v}" sglang_ver="${SGLANG_TAG#v}"
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    as_researcher "uv venv --python 3.12 '$VENV_DIR'"
  fi
  as_researcher "uv pip install --python '$VENV_DIR/bin/python' --torch-backend=$TORCH_BACKEND torch"
  as_researcher "uv pip install --python '$VENV_DIR/bin/python' triton httpx numpy pyyaml setuptools wheel ninja"
  as_researcher "uv pip install --python '$VENV_DIR/bin/python' --torch-backend=$TORCH_BACKEND 'vllm==$vllm_ver'"
  if [ -n "$sglang_ver" ]; then
    as_researcher "uv pip install --python '$VENV_DIR/bin/python' --no-deps 'sglang==$sglang_ver'" \
      || gpu_warn "sglang $sglang_ver would not install in the host venv; the container's own venv is what matters"
  fi
  as_researcher "'$VENV_DIR/bin/python' -c \"import torch,vllm;print('torch',torch.__version__,'cuda',torch.version.cuda,'vllm',vllm.__version__)\""
  chmod 0755 "$VENV_DIR"
}

# inference-judge-base, for a built engine (vLLM, and SGLang when the target names it, from
# the pinned tags).
gpu_base_image() {
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    local built
    built=$(docker image inspect -f '{{index .Config.Labels "inference-judge.tags"}}' "$IMAGE" 2>/dev/null || true)
    if [ "$built" = "$VLLM_TAG/$SGLANG_TAG/$TORCH_BACKEND" ]; then
      gpu_say "$IMAGE already built for $built"; return 0
    fi
    gpu_say "$IMAGE exists but was built for '${built:-unknown}'; rebuilding"
  fi
  DOCKER_BUILDKIT=1 docker build \
    -f "$SCRIPT_DIR/Dockerfile.base" \
    -t "$IMAGE" \
    --build-arg "VLLM_TAG=$VLLM_TAG" \
    --build-arg "SGLANG_TAG=$SGLANG_TAG" \
    --build-arg "TORCH_BACKEND=$TORCH_BACKEND" \
    --label "inference-judge.tags=$VLLM_TAG/$SGLANG_TAG/$TORCH_BACKEND" \
    "$SCRIPT_DIR"
  if [ -n "$SGLANG_TAG" ] && docker run --rm "$IMAGE" test -f /opt/venv/sglang/.install-failed; then
    gpu_warn "the image's SGLang venv failed to install; the SGLang track is unavailable"
    gpu_warn "retry with IJ_SGLANG_TAG=<older tag> if you need it"
  fi
  docker image inspect -f 'built {{.Id}} size {{.Size}}' "$IMAGE"
}
