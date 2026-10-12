# shellcheck shell=bash
# The TPU VM, for the setup scripts (bootstrap.sh, check.sh).  Sourced, not run.
# Everything here reads the T_* variables of setup/target.sh; nothing is named for
# one model.  Each function is one bootstrap step: it is idempotent, prints what
# it did, and returns non-zero with a message that says what to do.
#
#   tpu_devices        the chips are on the host: probes.present counts accelerator.chips
#   tpu_mount_disk     the persistent cache disk (provision.cache_disk.label) is mounted,
#                      and mounted again after a reboot (fstab, nofail)
#   tpu_engine_image   engine.image is in the local docker store at its pinned digest:
#                      loaded from the disk's tar, else pulled
#   tpu_gate           THE chip gate: in that image, with the target's own docker_flags
#                      and env, jax sees every chip.  A node can report healthy and still
#                      be unable to start a session; this is the only test that says so.
#   tpu_not_runnable   not a step: what bootstrap prints when it stops after the gate
#                      because the target is runnable: false (probe, edit, re-run)
#   tpu_models         model.local_dir: a flat directory of real files (hardlinks into
#                      the Hugging Face cache on the disk), checked against the hashes
#                      the cache's own blob names carry
#   tpu_compile_seed   engine.compile_cache.seed, bench-owned, filled from the disk
#   tpu_engine_sources the engine source trees (engine.sources), copied out of the image
#                      so they match it, each a git repo with one baseline commit
#
# Callers define as_researcher and, for tpu_models, verify_model_manifest (bootstrap.sh).
# Test seams are environment variables with the production values as defaults:
# IJ_FSTAB, IJ_DEV_DIR, IJ_TPU_GATE_TIMEOUT_S, IJ_TPU_GATE_VERDICT,
# IJ_TPU_SRC_ROOT.

tpu_say()  { printf '   %s\n' "$*"; }
tpu_warn() { printf '   WARN %s\n' "$*" >&2; }

# The directory on the disk that holds the flat model dir, and where it is mounted
# for the ruler: the parent of model.local_dir ( /opt/models ).
tpu_models_backing() { printf '%s/models' "$T_DISK_MOUNT"; }

# tpu_chip_nodes: how many TPU chips the kernel exposes, by the two layouts there
# are: /dev/accel<N> (older runtimes) and /dev/vfio/<N> (v6e).
tpu_chip_nodes() {
  local dev="${IJ_DEV_DIR:-/dev}" a=0 v=0 f
  for f in "$dev"/accel[0-9]*; do [ -e "$f" ] && a=$((a + 1)); done
  for f in "$dev"/vfio/[0-9]*; do [ -e "$f" ] && v=$((v + 1)); done
  if [ "$a" -ge "$v" ]; then echo "$a"; else echo "$v"; fi
}

# tpu_devices: the host sees every chip.  The count comes from the target's own probe
# (probes.present prints a number).  A probe that counts the wrong device path (the
# first draft of a target names /dev/accel*, which a v6e VM does not have) is a bug in
# the target, not in the node: it is reported, and the real device nodes decide.
tpu_devices() {
  local probe n real
  probe=$(tget_opt probes.present "")
  n=0
  if [ -n "$probe" ]; then n=$(bash -c "$probe" 2>/dev/null | tail -1 | tr -dc '0-9'); fi
  if [ "${n:-0}" -eq "$T_CHIPS" ]; then tpu_say "$n chip(s) visible to the host (probes.present)"; return 0; fi
  real=$(tpu_chip_nodes)
  if [ "${real:-0}" -eq "$T_CHIPS" ]; then
    tpu_warn "probes.present counts ${n:-0} but the host has $real device nodes: fix the probe in $T_ID (a v6e lists /dev/vfio/0..N)"
    return 0
  fi
  echo "the host shows ${real:-0} chip(s) (probes.present: ${n:-0}), target $T_ID wants $T_CHIPS" >&2
  echo "  wrong accelerator type for this VM, or the node is broken: delete and recreate it" >&2
  return 1
}

# tpu_mount_disk: mount the disk by its ext4 label, once and for every later boot.
# The disk outlives the VM, so it is also where the spend guard (tools/tpu_deadman.sh)
# copies the run's results every hour and before it deletes the VM: the mount point
# goes into the one file the guard reads, <its state dir>/save-to.
tpu_mount_disk() {
  local fstab="${IJ_FSTAB:-/etc/fstab}" dev guard="${IJ_DEADMAN_DIR:-/var/lib/inference-judge-deadman}"
  if [ -z "$T_DISK_LABEL" ]; then tpu_say "target names no cache disk"; return 0; fi
  [ -n "$T_DISK_MOUNT" ] || { echo "provision.cache_disk.mount is missing in $T_ID" >&2; return 1; }
  if mountpoint -q "$T_DISK_MOUNT"; then
    tpu_say "$T_DISK_MOUNT already mounted"
  else
    if ! dev=$(findfs "LABEL=$T_DISK_LABEL" 2>/dev/null) || [ -z "$dev" ]; then
      echo "no disk with the ext4 label '$T_DISK_LABEL' is attached to this VM" >&2
      echo "  from your laptop: gcloud alpha compute tpus tpu-vm attach-disk <node> --disk <disk> \\" >&2
      echo "                      --mode read-write --zone <zone> --project <project>" >&2
      return 1
    fi
    mkdir -p "$T_DISK_MOUNT"
    mount -o noatime "$dev" "$T_DISK_MOUNT"
    tpu_say "mounted $dev ($T_DISK_LABEL) at $T_DISK_MOUNT"
  fi
  # nofail: a reboot without the disk must not stop the VM from booting.
  if ! grep -qs "^LABEL=${T_DISK_LABEL}[[:space:]]" "$fstab"; then
    printf 'LABEL=%s %s ext4 defaults,nofail,noatime 0 2\n' "$T_DISK_LABEL" "$T_DISK_MOUNT" >> "$fstab"
    tpu_say "fstab: $T_DISK_LABEL -> $T_DISK_MOUNT (nofail)"
  fi
  mkdir -p "$guard"
  printf '%s\n' "$T_DISK_MOUNT" > "$guard/save-to.tmp" && mv -f "$guard/save-to.tmp" "$guard/save-to"
  tpu_say "results will be saved under $T_DISK_MOUNT/inference-judge-saved (hourly and before the VM is deleted)"
  [ -s "$guard/deadline" ] \
    || tpu_warn "the spend guard is not armed: nothing deletes this VM or saves its results.  sudo bash tools/tpu_deadman.sh arm <hours>"
}

# tpu_image_present: engine.image is in the docker store AT its pinned digest.  The
# digest is read from the image's own record, not from the name we asked for.
tpu_image_present() {
  local digest="${T_IMAGE##*@}" have
  have=$(docker image inspect -f '{{json .RepoDigests}} {{.Id}}' "$T_IMAGE" 2>/dev/null) || return 1
  case "$have" in *"$digest"*) return 0 ;; esac
  return 1
}

tpu_sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

# tpu_engine_image: loaded from the disk's tar, falling back to a pull (the tar's
# RepoDigests may not survive `docker load` on a host that never pulled the image).
tpu_engine_image() {
  case "$T_IMAGE" in
    *@sha256:*) ;;
    *) echo "engine.image '$T_IMAGE' is not pinned by digest (name@sha256:...): the compile cache and the results are only valid for one image" >&2; return 1 ;;
  esac
  if tpu_image_present; then tpu_say "$T_IMAGE present"; return 0; fi
  local tar="" want
  if [ -n "$T_DISK_LABEL" ] && [ -n "$T_DISK_IMAGE_TAR" ]; then tar="$T_DISK_MOUNT/$T_DISK_IMAGE_TAR"; fi
  if [ -n "$tar" ] && [ -f "$tar" ]; then
    want=$(tget_opt provision.cache_disk.image_tar_sha256 "")
    if [ -n "$want" ] && [ "$(tpu_sha256 "$tar")" != "$want" ]; then
      echo "$tar does not match provision.cache_disk.image_tar_sha256 ($want): the copy on the disk is damaged" >&2
      return 1
    fi
    tpu_say "docker load < $tar"
    docker load --input "$tar"
  fi
  if ! tpu_image_present; then
    tpu_warn "$T_IMAGE is not in the docker store with that digest after the load; pulling it"
    docker pull "$T_IMAGE"
  fi
  tpu_image_present || { echo "$T_IMAGE is not at the pinned digest after load and pull" >&2; return 1; }
  tpu_say "$T_IMAGE verified by digest"
}

# tpu_timeout SECONDS CMD...: GNU timeout (always there on the VM) or gtimeout.
tpu_timeout() {
  local s="$1"; shift
  if command -v timeout >/dev/null 2>&1; then timeout -k 10 "$s" "$@"
  elif command -v gtimeout >/dev/null 2>&1; then gtimeout -k 10 "$s" "$@"
  else "$@"; fi
}

# tpu_gate_run DOCKER-ARGS...: one `python -c "import jax; ..."` container, started
# through the image's entrypoint as a real launch and tools/tpu_seal_probe.sh are.
# Returns docker's exit code (124 = ran out of time); leaves its stderr in $TPU_GATE_ERR.
tpu_gate_run() {
  local secs="${IJ_TPU_GATE_TIMEOUT_S:-300}" rc=0 name=inference-judge-chip-gate
  docker rm -f "$name" >/dev/null 2>&1 || true
  TPU_GATE_ERR=$(tpu_timeout "$secs" docker run --rm --name "$name" "$@" "$T_IMAGE" \
    python -c "import jax; assert len(jax.devices()) == $T_CHIPS, jax.devices()" 2>&1 >/dev/null) || rc=$?
  docker rm -f "$name" >/dev/null 2>&1 || true
  return "$rc"
}

# tpu_gate_args: TPU_GATE_ARGS = the docker flags the ruler launches this target's engine
# with (run.sh seal_args: a network seal unless the flags name a network, shm size,
# docker_flags, env).
tpu_gate_args() {
  local f has_net=0 e
  TPU_GATE_ARGS=()
  for f in ${T_DOCKER_FLAGS[@]+"${T_DOCKER_FLAGS[@]}"}; do
    case "$f" in --network*|--net=*|--net) has_net=1 ;; esac
  done
  [ "$has_net" = 1 ] || TPU_GATE_ARGS+=(--network none)
  if [ -n "$T_SHM" ]; then TPU_GATE_ARGS+=(--shm-size "$T_SHM"); fi
  TPU_GATE_ARGS+=(${T_DOCKER_FLAGS[@]+"${T_DOCKER_FLAGS[@]}"})
  for e in ${T_ENV[@]+"${T_ENV[@]}"}; do TPU_GATE_ARGS+=(-e "$e"); done
}

# tpu_seal_probe_cmd: the one command that finds launch flags that reach the chips.
# The gate only says whether the target's own flags do; it never suggests others.
tpu_seal_probe_cmd() { printf "sudo bash tools/tpu_seal_probe.sh '%s' %s" "$T_IMAGE" "$T_CHIPS"; }

TPU_GATE_VERDICT="${IJ_TPU_GATE_VERDICT:-/run/inference-judge-chip-gate}"

# tpu_gate: stage 1 launches exactly as the ruler will.  If that fails, stage 2
# repeats it with the flags known to work, which says whether the NODE or the
# TARGET's flags are at fault -- recreating a healthy node would not help.
# A broken node always fails the step.  Flags that cannot reach the chips fail it
# only on a runnable target: on a sketch (runnable: false) they are what the
# operator is here to find out, so the step passes and leaves its verdict
# (flags | node) in $TPU_GATE_VERDICT for tpu_not_runnable.
tpu_gate() {
  local rc=0
  rm -f "$TPU_GATE_VERDICT"
  tpu_gate_args
  local args=(${TPU_GATE_ARGS[@]+"${TPU_GATE_ARGS[@]}"})
  tpu_say "chip gate: jax.devices() must list $T_CHIPS chips (docker run ${args[*]})"
  tpu_gate_run ${args[@]+"${args[@]}"} || rc=$?
  if [ "$rc" -eq 0 ]; then
    echo flags > "$TPU_GATE_VERDICT"
    tpu_say "chip gate passed"; return 0
  fi
  local first_err="$TPU_GATE_ERR" rc2=0
  tpu_gate_run --privileged --network host || rc2=$?
  if [ "$rc2" -eq 0 ]; then
    cat >&2 <<EOF
chip gate: the node is healthy but the target's launch flags cannot reach the chips.
  docker run ${args[*]}   (exit $rc)
  $(printf '%s' "$first_err" | tail -3 | tr '\n' ' ')
  the same image with --privileged --network host sees all $T_CHIPS.  Do NOT recreate the node:
  fix engine.docker_flags / engine.env in $T_ID (the ruler launches exactly like this).
  The flags that work on this node, strongest seal first, come from
    $(tpu_seal_probe_cmd)
EOF
    [ "$T_RUNNABLE" = true ] && return 1
    echo node > "$TPU_GATE_VERDICT"
    tpu_say "chip gate: $T_ID is runnable: false, so unproven flags do not fail this step"
    return 0
  fi
  cat >&2 <<EOF
chip gate FAILED: the TPU node cannot start a session with ANY container flags
  (exit $rc$( [ "$rc" -eq 124 ] && printf ', timed out after %ss' "${IJ_TPU_GATE_TIMEOUT_S:-300}"))
  $(printf '%s' "$TPU_GATE_ERR" | tail -3 | tr '\n' ' ')
  A node can report READY and still be broken.  Nothing on the node fixes it:
  DELETE AND RECREATE THE NODE, then re-run this step.
    gcloud compute tpus tpu-vm delete <node> --zone <zone> --project <project> --quiet
EOF
  return 1
}

# tpu_not_runnable: where bootstrap stops for a target that is runnable: false, and
# what the operator does about it.  Everything the seal probe needs is in place by
# now (docker, the disk, the image, the gate's verdict); nothing the ruler would
# lock has happened.  It only prints: bootstrap's run_plan does the stopping.
tpu_not_runnable() {
  local verdict n=1
  verdict=$(cat "$TPU_GATE_VERDICT" 2>/dev/null || true)
  cat <<EOF
bootstrap stopped here on purpose: target $T_ID is runnable: false.
  Done: every step up to the chip gate (the checklist above).  Not done: every step
  after the gate.  The run clock is not started and the ruler is not locked.
  The target file is  $T_FILE
  Next, in this order:
EOF
  if [ "$verdict" = flags ]; then
    cat <<EOF
  $n. Nothing to probe: the chip gate reached all $T_CHIPS chips with the target's
     own engine.docker_flags and engine.env.  Leave them as they are.
EOF
  else
    cat <<EOF
  $n. Find the launch flags that reach the chips on this node:
       $(tpu_seal_probe_cmd)
     It tries the strongest seal first and ends with a line  engine.docker_flags: ...
     ("This node is faulty" instead means delete the node and create a new one.)
EOF
    n=$((n + 1))
    cat <<EOF
  $n. Put that line in the target file:
       engine.docker_flags   the words of the line as a YAML list, without --network none
                             (the ruler adds it) and without any -e NAME=VALUE pair
       engine.env            every -e NAME=VALUE of the line as NAME: VALUE; delete
                             VBAR_CONTROL_SERVICE_URL from it if the line has none
EOF
  fi
  n=$((n + 1))
  cat <<EOF
  $n. Set  runnable: true  in the target file (the comment above that line says what
     it claims).  Finish the file's other edits now: the ruler locks its bytes when
     the clock starts.
EOF
  n=$((n + 1))
  cat <<EOF
  $n. Run bootstrap again; it repeats the chip gate with the file as it then is and
     carries on to the end (add --no-clock to install everything and start the clock
     with a later run):
       sudo bash setup/bootstrap.sh --target $T_FILE
EOF
}

# tpu_models: model.local_dir ( /opt/models/<name> ) as a flat directory of real
# files.  The ruler mounts the parent read-only into a container, where a symlink
# into the Hugging Face cache would dangle; a hardlink is a real file.  Hardlinks need
# the same filesystem, so the parent is a bind mount of <disk>/models.
# They also need the same MOUNT: link(2) from the disk into the bind mount is EXDEV
# although both are one filesystem.  So every link is made at the disk's own path
# ($real); the bind mount shows it at $dir, where it is then read and checked the way
# the ruler will see it.
tpu_models() {
  local fstab="${IJ_FSTAB:-/etc/fstab}" dir="$T_MODEL_DIR" parent backing real hub snap f rel src dst blob n=0 manifest
  if [ -z "$T_DISK_LABEL" ] || [ -z "$T_DISK_HF" ] || [ -z "$dir" ]; then
    tpu_say "no cache-disk model for $T_ID"; return 0
  fi
  [[ "$T_MODEL_REV" =~ ^[0-9a-f]{40}$ ]] || { echo "model.revision '$T_MODEL_REV' is not a 40-hex commit" >&2; return 1; }
  hub="$T_DISK_MOUNT/$T_DISK_HF/models--${T_MODEL_ID//\//--}"
  snap="$hub/snapshots/$T_MODEL_REV"
  [ -d "$snap" ] || { echo "the disk's Hugging Face cache has no $snap (model.revision $T_MODEL_REV)" >&2; return 1; }
  parent=$(dirname "$dir"); backing=$(tpu_models_backing)
  mkdir -p "$backing" "$parent"
  if ! mountpoint -q "$parent"; then
    mount --bind "$backing" "$parent"
    tpu_say "bind mount $backing -> $parent"
  fi
  grep -qs "^${backing}[[:space:]]" "$fstab" \
    || printf '%s %s none bind,nofail 0 0\n' "$backing" "$parent" >> "$fstab"
  real="$backing/$(basename "$dir")"
  mkdir -p "$real"
  chmod -R u+w "$real"
  [ "$real" -ef "$dir" ] || { echo "$parent does not show $backing: the bind mount is not in place" >&2; return 1; }
  manifest="${dir%/}.sha256"
  : > "$manifest.new"
  while IFS= read -r f; do
    rel="${f#"$snap"/}"
    src=$(readlink -f "$f")
    case "$src" in "$hub"/blobs/*) ;; *) echo "$rel resolves to $src, outside $hub/blobs: refusing to link it" >&2; return 1 ;; esac
    [ -f "$src" ] || { echo "$rel: blob $src is missing from the disk" >&2; return 1; }
    dst="$real/$rel"
    mkdir -p "$(dirname "$dst")"
    [ "$src" -ef "$dst" ] || ln -f "$src" "$dst"
    # The blob's name IS its hash: sha256 for LFS files, the git blob sha1 for small ones.
    blob=$(basename "$src")
    if   [[ "$blob" =~ ^[0-9a-f]{64}$ ]]; then printf 'sha256 %s %s\n' "$blob" "$rel" >> "$manifest.new"
    elif [[ "$blob" =~ ^[0-9a-f]{40}$ ]]; then printf 'gitsha1 %s %s\n' "$blob" "$rel" >> "$manifest.new"
    else tpu_warn "$rel: blob name $blob is not a hash; not verified"; fi
    n=$((n + 1))
  done < <(find "$snap" -mindepth 1 \( -type f -o -type l \) | sort)
  [ "$n" -gt 0 ] || { echo "$snap holds no files" >&2; return 1; }
  # HF blob names make the manifest; once written for this revision it is the
  # record the quick check on later runs compares against.
  if [ -s "$manifest" ] && cmp -s "$manifest" "$manifest.new" && [ "$(head -1 "$manifest.revision" 2>/dev/null)" = "$T_MODEL_REV" ] \
     && [ "${IJ_VERIFY_MODEL:-0}" != 1 ]; then
    rm -f "$manifest.new"
    verify_model_manifest "$dir" "$manifest" quick
  else
    chmod u+w "$manifest" "$manifest.revision" 2>/dev/null || true
    mv -f "$manifest.new" "$manifest"
    tpu_say "hashing $n files against the cache's blob names"
    verify_model_manifest "$dir" "$manifest" full
    printf '%s\n' "$T_MODEL_REV" > "$manifest.revision"
    chmod 0444 "$manifest" "$manifest.revision"
  fi
  # No symlink may survive: the container sees /opt/models and nothing else.
  if [ -n "$(find "$dir" -type l 2>/dev/null | head -1)" ]; then
    echo "symlinks in $dir: the sealed container could not follow them" >&2; return 1
  fi
  chown -R root:root "$dir"
  find "$dir" -type d -exec chmod 0555 {} +
  find "$dir" -type f -exec chmod 0444 {} +
  tpu_say "model dir $dir: $n real files, root-owned, read-only"
}

# tpu_compile_seed: engine.compile_cache.seed is the cache a sealed launch starts from.
# bench owns it and the researcher cannot write it.  It matches one image digest, one
# slice size and one set of flags (the disk's cache was made by exactly those).
# The seed only ever grows: the ruler adds what each kept launch compiled (run.sh,
# rsync --ignore-existing), and so does this step.  A re-run of bootstrap copies in
# what the disk has and the seed lacks; it never deletes or replaces a file, so
# hours of compiling on this VM cannot be lost to it.
tpu_compile_seed() {
  local src had
  if [ -z "$T_COMPILE_SEED" ]; then tpu_say "target names no compile-cache seed"; return 0; fi
  install -d -o bench -g bench -m 0750 "$(dirname "$T_COMPILE_SEED")"      # first run: /opt/bench/cache itself
  if [ -z "$T_DISK_LABEL" ] || [ -z "$T_DISK_COMPILE" ]; then
    install -d -o bench -g bench -m 0750 "$T_COMPILE_SEED"
    tpu_say "seed $T_COMPILE_SEED created empty (no cache on a disk): the first launch compiles"; return 0
  fi
  src="$T_DISK_MOUNT/$T_DISK_COMPILE"
  [ -d "$src" ] || { echo "$src is not a directory: the disk has no compile cache" >&2; return 1; }
  install -d -o bench -g bench -m 0750 "$T_COMPILE_SEED"
  had=$(find "$T_COMPILE_SEED" -type f | wc -l | tr -d ' ')
  rsync -a --ignore-existing "$src/" "$T_COMPILE_SEED/"
  chown -R bench:bench "$T_COMPILE_SEED"
  chmod -R u+rwX,go-rwx "$T_COMPILE_SEED"
  tpu_say "seed $T_COMPILE_SEED <- $src ($(du -sh "$T_COMPILE_SEED" | cut -f1); $had file(s) already there were kept)"
}

# tpu_engine_sources: /home/researcher/<name>-src for every name in engine.sources.  The
# trees come out of the image itself (docker create + docker cp), so they are the
# code the image runs.  The ruler snapshots `git diff HEAD`, so each tree is a git
# repo; image trees built with `COPY . .` have no .git, and get one baseline commit,
# tagged inference-judge-baseline.  A tree that exists is left alone (the researcher edits
# it) unless --reinstall-engines.
tpu_engine_sources() {
  local root="${IJ_TPU_SRC_ROOT:-$(tget_opt provision.image_src_root /workspace)}"
  local name dest cid="" todo=() fresh=" "
  for name in ${T_SOURCES[@]+"${T_SOURCES[@]}"}; do
    dest="$RESEARCHER_HOME/$name-src"
    if [ -d "$dest" ] && [ "${IJ_REINSTALL_ENGINES:-0}" != 1 ]; then
      tpu_say "$dest exists; left alone (--reinstall-engines re-extracts it)"
    else
      todo+=("$name"); fresh="$fresh$name "
    fi
  done
  if [ "${#todo[@]}" -gt 0 ]; then
    cid=$(docker create --entrypoint true "$T_IMAGE")
    for name in "${todo[@]}"; do
      dest="$RESEARCHER_HOME/$name-src"
      rm -rf "$dest.new"; mkdir -p "$dest.new"
      if ! docker cp "$cid:$root/$name/." "$dest.new"; then
        docker rm -f "$cid" >/dev/null 2>&1 || true
        rm -rf "$dest.new"
        echo "the image has no $root/$name (engine.sources lists $name): fix engine.sources or provision.image_src_root" >&2
        return 1
      fi
      rm -rf "$dest"; mv "$dest.new" "$dest"
      tpu_say "$dest <- $T_IMAGE:$root/$name"
    done
    docker rm -f "$cid" >/dev/null 2>&1 || true
  fi
  for name in ${T_SOURCES[@]+"${T_SOURCES[@]}"}; do
    dest="$RESEARCHER_HOME/$name-src"
    chown -R researcher:researcher "$dest"
    chmod 0755 "$dest"
    case "$fresh" in *" $name "*) tpu_git_baseline "$name" "$dest" fresh ;; *) tpu_git_baseline "$name" "$dest" ;; esac
  done
}

# tpu_git_baseline NAME DIR [fresh]: DIR is a git repo with a commit, and for a tree that
# was just copied out of the image (fresh) a clean working tree: whatever the image build
# left uncommitted (a patched checkout) is committed as the baseline, or `git diff HEAD`
# would later show the image's own patches as the researcher's work.  A tree that is
# not fresh holds the researcher's edits and is never committed here.
tpu_git_baseline() {
  local name="$1" dir="$2" fresh="${3:-}" has_head=0 clean=0
  if as_researcher "cd '$dir' && git rev-parse -q --verify HEAD >/dev/null 2>&1"; then has_head=1; fi
  if [ "$has_head" = 1 ] && [ "$fresh" != fresh ]; then
    tpu_say "$name-src: git HEAD $(as_researcher "cd '$dir' && git rev-parse --short HEAD")"
    return 0
  fi
  if [ "$has_head" = 1 ] && as_researcher "cd '$dir' && [ -z \"\$(git status --porcelain 2>/dev/null)\" ]"; then clean=1; fi
  if [ "$clean" = 1 ]; then
    as_researcher "cd '$dir' && git tag -f inference-judge-baseline >/dev/null"
    tpu_say "$name-src: git HEAD $(as_researcher "cd '$dir' && git rev-parse --short HEAD"), clean"
    return 0
  fi
  # bytecode would show up in every diff; keep it out of the baseline and out of `git status`
  as_researcher "cd '$dir' && { [ -d .git ] || git init -q; } \
    && mkdir -p .git/info && printf '__pycache__/\n*.pyc\n' >> .git/info/exclude \
    && git add -A \
    && git -c user.name='inference-judge setup' -c user.email='setup@inference-judge.local' commit -qm 'baseline: $name as shipped in the engine image' \
    && git tag -f inference-judge-baseline >/dev/null"
  tpu_say "$name-src: baseline commit $(as_researcher "cd '$dir' && git rev-parse --short HEAD")"
}
