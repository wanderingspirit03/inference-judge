# shellcheck shell=bash disable=SC2034  # the T_* variables are for the callers
# The target, for the setup scripts (bootstrap.sh, check.sh, rehearse.sh).
# Sourced, not run.  The target file is read ONLY through bench/target.py's CLI
# (DESIGN 2.1); nothing here parses YAML.
#
#     target_load FILE       sets the T_* variables below
#     target_summary         one line for logs
#     tget KEY / tget_opt KEY [DEFAULT]
#
# The variables every caller branches on:
#   T_FILE         the target file
#   T_ID           target id
#   T_KIND         accelerator.kind: none | gpu | tpu
#   T_CHIPS        accelerator.chips
#   T_ENGINE       engine.name (mock, vllm, sglang, sglang-jax, ...)
#   T_IMAGE        engine.image ("" if none)
#   T_ENGINE_CLASS built  - vLLM/SGLang built here from setup/Dockerfile.base (pins needed)
#                  mock   - the CPU mock; its image is built from targets/<id>.Dockerfile
#                  pulled - a prebuilt image named by digest/tag, pulled
#                  none   - no engine image at all
#   T_MODEL_ID     model.hf_id ("" = nothing to download)
#   T_MODEL_REV    model.revision
#   T_MODEL_DIR    model.local_dir
#   T_METRIC       metric.kind: tps
#   T_RUNNABLE     runnable: true | false
#   T_SOURCES      (array) engine.sources: the engine source trees the researcher edits,
#                  each at /home/researcher/<name>-src; empty when the target lists none
#   T_DOCKER_FLAGS (array) engine.docker_flags    T_ENV (array of K=V) engine.env
#   T_SHM          engine.shm_size
#   T_DISK_LABEL   provision.cache_disk.label ("" = the target has no cache disk); with
#   T_DISK_MOUNT, T_DISK_IMAGE_TAR, T_DISK_HF, T_DISK_COMPILE (paths on that disk)
#   T_COMPILE_SEED engine.compile_cache.seed ("" = none)
#   T_BASELINE     the baseline experiment every candidate is measured against:
#                  provision.baseline if the target names one, else baseline-tpu for a
#                  tpu target, else baseline-<engine.name>
#
# Fields another agent adds to the target are read with tget_opt, so a target
# that lacks them still loads.
#
# Seams (tests): IJ_TARGET_PY (default: the repo's bench/target.py, else
# /opt/bench/target.py), IJ_TARGET_PYTHON (default python3).

target_py() {
  local py="${IJ_TARGET_PY:-}"
  if [ -z "$py" ]; then
    if [ -n "${IJ_REPO:-}" ] && [ -f "$IJ_REPO/bench/target.py" ]; then py="$IJ_REPO/bench/target.py"
    else py=/opt/bench/target.py; fi
  fi
  "${IJ_TARGET_PYTHON:-python3}" -I "$py" "$@"
}

# tget KEY: the value, or a non-zero exit naming the missing key.
tget() { target_py get "$T_FILE" "$1"; }
# tget_opt KEY [DEFAULT]: the value, or DEFAULT when the key is absent.
tget_opt() {
  local v
  if v=$(target_py get "$T_FILE" "$1" 2>/dev/null); then printf '%s' "$v"; else printf '%s' "${2:-}"; fi
}

# json_lines JSON: a JSON list as one item per line, an object as KEY=VALUE lines.
json_lines() {
  "${IJ_TARGET_PYTHON:-python3}" -I -c 'import json, sys
v = json.loads(sys.argv[1])
for x in (v.items() if isinstance(v, dict) else v):
    print("%s=%s" % x if isinstance(v, dict) else x)' "$1"
}

# tget_array NAME KEY [DEFAULT-WORDS]: bash array NAME from the list/object at KEY.
# (Plain read loop: macOS ships bash 3.2, which has no mapfile.)
tget_array() {
  local name="$1" key="$2" json item
  eval "$name=()"
  if json=$(target_py get "$T_FILE" "$key" 2>/dev/null) && [ -n "$json" ]; then
    while IFS= read -r item; do [ -n "$item" ] && eval "$name+=(\"\$item\")"; done < <(json_lines "$json")
  else
    for item in ${3:-}; do eval "$name+=(\"\$item\")"; done
  fi
}

target_load() {
  T_FILE="${1:-}"
  T_SOURCES=() T_DOCKER_FLAGS=() T_ENV=() T_SHM="" T_BASELINE=""
  T_DISK_LABEL="" T_DISK_MOUNT="" T_DISK_IMAGE_TAR="" T_DISK_HF="" T_DISK_COMPILE="" T_COMPILE_SEED=""
  [ -f "$T_FILE" ] || { echo "target file $T_FILE not found" >&2; return 1; }
  target_py validate "$T_FILE" >/dev/null || { echo "target $T_FILE does not validate (bench/target.py validate)" >&2; return 1; }
  T_ID=$(tget id) || return 1
  T_KIND=$(tget accelerator.kind) || return 1
  T_CHIPS=$(tget_opt accelerator.chips 1)
  T_ENGINE=$(tget engine.name) || return 1
  T_IMAGE=$(tget_opt engine.image "")
  T_MODEL_ID=$(tget_opt model.hf_id "")
  T_MODEL_REV=$(tget_opt model.revision "")
  T_MODEL_DIR=$(tget_opt model.local_dir "")
  T_METRIC=$(tget metric.kind) || return 1
  T_RUNNABLE=$(tget_opt runnable true)
  tget_array T_SOURCES engine.sources ""
  tget_array T_DOCKER_FLAGS engine.docker_flags ""
  tget_array T_ENV engine.env ""
  T_SHM=$(tget_opt engine.shm_size "")
  T_DISK_LABEL=$(tget_opt provision.cache_disk.label "")
  T_DISK_MOUNT=$(tget_opt provision.cache_disk.mount "")
  T_DISK_IMAGE_TAR=$(tget_opt provision.cache_disk.image_tar "")
  T_DISK_HF=$(tget_opt provision.cache_disk.hf_cache "")
  T_DISK_COMPILE=$(tget_opt provision.cache_disk.compile_cache "")
  T_COMPILE_SEED=$(tget_opt engine.compile_cache.seed "")
  T_BASELINE=$(tget_opt provision.baseline "")
  if [ -z "$T_BASELINE" ]; then
    case "$T_KIND" in tpu) T_BASELINE=baseline-tpu ;; *) T_BASELINE="baseline-$T_ENGINE" ;; esac
  fi
  case "$T_ENGINE" in
    mock) T_ENGINE_CLASS=mock ;;
    vllm|sglang) case "$T_IMAGE" in inference-judge-base:*) T_ENGINE_CLASS=built ;; "") T_ENGINE_CLASS=none ;; *) T_ENGINE_CLASS=pulled ;; esac ;;
    *) if [ -n "$T_IMAGE" ]; then T_ENGINE_CLASS=pulled; else T_ENGINE_CLASS=none; fi ;;
  esac
}

target_summary() {
  printf 'target %s (%s): accelerator %s x%s, engine %s [%s] image %s, model %s@%s, metric %s\n' \
    "$T_ID" "$T_FILE" "$T_KIND" "$T_CHIPS" "$T_ENGINE" "$T_ENGINE_CLASS" \
    "${T_IMAGE:-none}" "${T_MODEL_ID:-none}" "${T_MODEL_REV:-?}" "$T_METRIC"
}
