# setup/ — turning a fresh box into the autoresearch box for one target

Everything that depends on the hardware, the engine, the model or the metric
comes from one target file, `targets/<id>.yaml` (DESIGN 2.1). The scripts
here read it through one seam, `setup/target.sh`, which calls `bench/target.py`'s
command line and nothing else; none of them parses YAML or assumes a GPU.

## What has been run where

- **CPU mock** (`targets/cpu-mock.yaml`): the judge and the loop run end to end on a laptop,
  including the whole procedure in `local_box/`.
- **TPU (Google Cloud v6e)**: the model server of `targets/v6e4-gemma4-26b-tps.yaml` was brought up by
  hand on a v6e-4. `tpu.sh`, bootstrap and a judged run have not yet been run on real TPU hardware.
  That target is still `runnable: false`, and its sealed launch is marked "NOT PROVEN" in the file.
- **GPU (NVIDIA)**: built and tested on a laptop with fakes. It has never run on real hardware (see
  "The GPU path").
- **Coding agents.** Three adapters ship in `research/adapters/`: `claude` (Claude Code), `codex`
  (Codex CLI) and `pi`. Claude Code is the only agent that has run a real loop, and that was before
  the loop called it through `session.sh`; through the adapter it has been tested with a stand-in
  program only (a real session on a laptop did not get past an expired login). The Codex and
  Pi adapters were written from those tools' documentation and tested with stand-in programs. Each
  has since run one short real session on a laptop through its adapter. Neither has run a real loop.
  Unseen for both: a loop on a hardened machine, and the wording of their quota and login errors.
  Unseen for Codex: that its hooks are enforced from `/etc/codex/requirements.toml`. See
  `research/adapters/README.md`.

| file | what it is |
|---|---|
| `bootstrap.sh` | run as root on a fresh Ubuntu 24.04 box. **Install-or-resume**: re-running fixes what is missing and touches nothing that already works. `--agent NAME` (or `IJ_AGENT=NAME`; default `claude`) picks the coding agent. `--plan` prints the steps a target would run or skip, with an `agent:` line, as any user, and changes nothing |
| `target.sh` | the target seam, sourced: `target_load FILE` validates the file and sets `T_*` for the caller; `tget key` reads one more field |
| `gpu.sh` | the NVIDIA GPU steps, sourced by bootstrap and check: GPU count, type and memory against the target, the driver minimum, docker's GPU passthrough, the gpu gate, and the steps for an engine built on the box. **Unproven on real hardware** (see below) |
| `tpu.sh` | the TPU VM steps, sourced the same way |
| `harden_research.sh` | called by bootstrap, with `AGENT=<name>`: the driver user `driver`, the root-owned `/opt/inference-judge/{loop,guard,adapters}`, the agent's managed settings file (when its adapter has one), `/etc/inference-judge/guard.env`, the sudoers rules, the systemd units, `/etc/inference-judge/loop.env` |
| `check.sh` | verifies every precondition for the installed target, including the agent's version and login (a missing login is a FAIL where the adapter can tell; Pi's can only when `IJ_PRIMARY_MODEL` names a model). Changes nothing, prints PASS/WARN/FAIL |
| `rehearse.sh` | sends deliberately broken experiments through the real path (queue, installed runner, ruler) before the units are enabled, so a funnel bug costs an hour and not a day |
| `Dockerfile.base` | the `inference-judge-base:latest` image, for targets whose engine is built on the box |
| `systemd/` | `inference-judge-driver`, `inference-judge-runner`, `inference-judge-heartbeat` units and `loop.env.example` |
| `local_box/` | the whole procedure on a laptop, in a container, with a stand-in for Claude Code's command reached through the Claude adapter: `bash setup/local_box/run_local_box.sh` (its own README) |
| `tests/` | `bench/.venv/bin/python -m pytest setup/tests`. The local box test is marked `slow` and runs only with `IJ_LOCALBOX=1` (see `local_box/README.md`) |

## Operator steps, exactly

`docs/RUNBOOK.md` is the full procedure with the reasons. In short, as root:

```bash
# 0. on the laptop: what would this target do?  (needs PyYAML: use the bench venv)
PATH="$PWD/bench/.venv/bin:$PATH" bash setup/bootstrap.sh --target targets/<id>.yaml --plan

# 1. get the repo onto the box
rsync -az --exclude .git --exclude '*/.venv' <checkout>/ root@<box>:/root/inference-judge/

# 2. on the box.  Add --agent NAME (or IJ_AGENT=NAME) for an agent other than claude.
#    A target whose engine is built here needs a release tag for every
#    engine it names (vLLM always, SGLang if the target names it); nothing is "latest".
#    A target with a prebuilt engine image needs neither variable.
ssh root@<box>
cd /root/inference-judge && mkdir -p /var/log/inference-judge
IJ_VLLM_TAG=<tag> IJ_SGLANG_TAG=<tag> \
  bash setup/bootstrap.sh --target targets/<id>.yaml 2>&1 | tee /var/log/inference-judge/bootstrap.log

# 3. log the agent in once, interactively, as the researcher.  The line to run is the
#    adapter's own: bash research/adapters/<agent>/setup.sh login
ssh -t root@<box> "su - researcher -c claude"     # Claude Code: follow the browser link, then /exit
#   codex: su - researcher, then  codex login   (codex login --device-auth with no browser)
#   pi:    su - researcher, then  pi, type /login, pick a provider, then /quit

# 4. verify: every line PASS (WARNs are explained below; a missing login is a FAIL where the adapter can tell)
bash setup/check.sh

# 5. the reference run and the calibration, by hand
#    (see docs/RUNBOOK.md)

# 6. rehearse the funnel; on PASS it restarts the run clock (not with --no-clock, or once a session has run)
bash setup/rehearse.sh

# 7. start the loop
systemctl enable --now inference-judge-runner inference-judge-driver inference-judge-heartbeat.timer
```

Do not copy credentials or tokens from another machine to do the login.

Why `su - researcher` and not `sudo -iu researcher`: macOS ssh clients forward `LC_*`,
and sudo's audit plugin on some Ubuntu images aborts on a locale it cannot load.
The scripts pin `LC_ALL=C.UTF-8` for the same reason: every `sudo -u bench`
inside bootstrap, check and rehearse would otherwise fail for a reason that has
nothing to do with the install.

Useful flags while iterating on the setup itself (each with `--target ...`):

```bash
bash setup/bootstrap.sh --plan             # the steps, and why each skipped one is skipped
bash setup/bootstrap.sh --agent codex --plan   # the same, for another agent: the `agent:` line and the agent step change
bash setup/bootstrap.sh --skip-model       # everything except the weights
bash setup/bootstrap.sh --no-clock         # install everything, do not start the clock
IJ_VERIFY_MODEL=1 bash setup/bootstrap.sh   # re-hash every weight file against the Hub manifest
```

and the two that destroy work, which is why they are flags and not the default:

```bash
bash setup/bootstrap.sh --reinstall-engines  # git checkout -f the pinned engine refs,
                                             # throwing away uncommitted patches
bash setup/bootstrap.sh --reset-clock        # a new run: restarts the clock and re-locks
                                             # the ruler; the hours already run are lost
```

## What bootstrap.sh does

`--plan` is the answer for a given target: it prints the same list `main` runs
(`plan_steps` is the one place that decides what runs for which target). What
varies:

| target says | bootstrap does |
|---|---|
| `accelerator.kind: none` | no NVIDIA driver check, no `nvidia-smi`, no nsys, no GPU passthrough test; Docker only |
| `accelerator.kind: gpu` | the steps in `setup/gpu.sh` (see "The GPU path" below): `nvidia-smi` shows exactly `accelerator.chips` GPUs, of `accelerator.type` and with `accelerator.hbm_gb_per_chip` of memory when the target says so; `docker run --gpus all`; nsys (optional). A prebuilt engine image is then started on the GPUs with the target's own flags (the gpu gate), and a target that is not `runnable` stops there (exit 3) and prints what to do |
| `accelerator.kind: tpu` | the steps in `setup/tpu.sh`: chip count, cache disk, engine image from the disk, chip gate, model directory, compile-cache seed, engine source trees. A target that is not `runnable` stops after the chip gate (exit 3) and prints what to do |
| engine built here (vLLM, SGLang on `inference-judge-base`) | requires `IJ_VLLM_TAG`, and `IJ_SGLANG_TAG` only when the target serves sglang or lists its tree (release tags; refused otherwise, before any network use); needs driver r580 or newer (the wheels are CUDA 13); clones the trees in `engine.sources` and no others at those tags, builds the host venv and `inference-judge-base`, records the pins in `/etc/inference-judge-engine-pins` |
| engine image with a registry digest | `docker pull` by digest; no build, no engine tags, no driver series asked for. The trees in `engine.sources` are fetched at the refs the target pins (`provision.sources.<name>.ref`, a 40-hex commit or a release tag); a tree with no pinned ref is refused. On a TPU the trees come out of the image instead |
| engine `mock` | builds the image from `targets/<id>.Dockerfile` |
| `model.hf_id` + a 40-hex `model.revision` | downloads exactly that revision to `model.local_dir`, checks every file against the hashes the Hub publishes for that revision (a re-run only checks that the files are there, unless `IJ_VERIFY_MODEL=1`), makes it root-owned and read-only. No `hf_id`: the step is skipped and the weights must already be there |

Bootstrap's own knobs live in the target's `provision:` section, which the
ruler does not read. All are optional:

| key | meaning | without it |
|---|---|---|
| `provision.min_free_gb` | free space on `/` the install needs: weights + engine image + room for experiment images and results | 5 (`kind: none`), 40 (`tpu`), 100 (`gpu`, a guess). A re-run that already has the weights and the engine image needs 40 |
| `provision.min_driver_major` | the lowest NVIDIA driver series the target accepts | 580 for an engine built here; none for a prebuilt image, which carries its own CUDA and is refused by the container toolkit on a driver too old for it |
| `provision.sources.<name>.ref`, `.url` | where an `engine.sources` tree comes from, for a prebuilt image. `url` is known for `vllm` and `sglang` | the ref is required; an unknown tree also needs the url |
| `provision.baseline` | the baseline experiment, `research/experiments/<name>/launch.sh` | `baseline-tpu` on a TPU, else `baseline-<engine.name>`. Preflight and `check.sh` require the launcher to exist |
| `provision.model_ignore` | globs of files in the Hub repo that are not downloaded or hashed | everything is downloaded |

(A TPU target's section also carries `provision.cache_disk`, read by
`setup/tpu.sh`, and what it takes to create the VM. Its `data_disk_gb` is the
size of the disk to create, which is never all free; that is why the disk gate
reads `min_free_gb` instead.)

### The GPU path

**Unproven on real hardware.** `setup/gpu.sh`, the GPU branch of `check.sh`,
`targets/gpu-gemma4-26b-tps.yaml` and `research/experiments/baseline-vllm` have
only been exercised on a laptop: `--plan`, and unit tests in which
`nvidia-smi`, `docker` and `git` are fakes (`setup/tests/test_gpu.py`). No GPU
has been through any of it. Expect the first real run to find something.

What the box must already have, whatever cloud or desk it is on: Ubuntu
(24.04 is what the scripts are written for), an NVIDIA driver so that
`nvidia-smi` works, and Docker with the NVIDIA container toolkit so that
`docker run --gpus all` works. Bootstrap checks these and installs none of them.

- **GPU count is exact.** The ruler launches with `--gpus all`, so a host with
  more GPUs than `accelerator.chips` would serve on hardware the target does
  not describe. More or fewer is a failure, not a warning.
- **Type and memory only when the target names them.** `accelerator.type` is
  matched against the name `nvidia-smi` reports, ignoring case; a target that
  leaves it out takes any card. `accelerator.hbm_gb_per_chip` is checked
  against the card's memory (95% of the nominal size, since cards report a
  little less).
- **The gpu gate** (prebuilt image only) starts `<image> bash -c 'nvidia-smi -L'`
  with the target's `docker_flags`, `env` and `shm_size`, `--gpus all` and no
  network, which is how the ruler starts a launch. It fails on a missing
  container toolkit, a driver older than the image's CUDA needs, an image
  entrypoint the flags do not clear, or flags that hide a GPU. It does not load
  a model: whether the engine serves, and whether the weights fit, is first
  seen at the baseline.
- **A sketch stops after the gate.** For a `runnable: false` target bootstrap
  exits 3 with the next steps, as it does for a TPU sketch.

What never varies: preflight; packages; the users `bench` and
`researcher`; `/opt/bench` installed from the repo with its venv; the target
installed three ways (`/opt/bench/targets/<id>.yaml` and the one-line
`/opt/bench/TARGET` for the ruler, `/etc/inference-judge/target.yaml`
world-readable for the researcher); `/home/researcher/research`, with
`adapters/<agent>/project/` copied to its top (`CLAUDE.md` and `.claude/` for
Claude Code, `.codex/` for Codex); the agent, installed through
`adapters/<agent>/setup.sh install` unless its command is already on the
`PATH` (an adapter with no `setup.sh` is not an error: you install the
command); `harden_research.sh` (the `driver` user, the frozen loop, guard and
adapters, the agent's managed settings, sudoers, units);
`/etc/inference-judge/loop.env`; the ruler image
`inference-judge-ruler:latest`; and last, the run clock
(`loopstate.py start-clock`, as `driver`) together with
`/opt/bench/RULER_LOCK`, written by asking the ruler to lock itself through
the driver's own sudo route. A re-run leaves a started clock and an existing
lock alone.

The reference run and the calibration are not bootstrap steps: the operator
runs them by hand (see `docs/RUNBOOK.md`).

Every log goes to `/var/log/inference-judge/`.

### The three users

| user | owns | sudo |
|---|---|---|
| `bench` | `/opt/bench` (ruler, prompt pools, results, leaderboard, reference, calibration); in the `docker` group; the accelerator device group | none installed by setup |
| `researcher` | `/home/researcher/research` except the root-owned `loop/`, `guard/`, `adapters/` and the agent's settings folder (`.claude/` or `.codex/`); the engine checkouts | none after `harden_research.sh`, which renames any sudoers file that lets `researcher` run the judge to `*.disabled-by-harden` (unless `--keep-direct-bench`). The researcher asks for a measurement by writing `queue/<id>.json` |
| `driver` | `/home/driver/research-state` (events.jsonl and everything derived from it); runs the units from `/opt/inference-judge/loop` | `(bench) NOPASSWD: /opt/bench/run.sh` and `(researcher) NOPASSWD: ALL`, in `/etc/sudoers.d/inference-judge-driver` |

`check.sh` asserts each of these on the installed box, and the local box
probes them from the researcher's side (21 things `researcher` must not be able to
do, 6 it must).

### The agent in `loop.env`

`/etc/inference-judge/loop.env` is the units' environment file, and
`systemd/loop.env.example` is its template. These are the settings that
concern the agent:

| variable | meaning |
|---|---|
| `IJ_AGENT` | the adapter in `research/adapters/` that does the research (default `claude`). `check.sh` and `rehearse.sh` read it from here too |
| `IJ_AGENT_BIN` | the agent's command, when it is not in `/usr/local/bin` or `/usr/bin` (`sudo` replaces `PATH` with a `secure_path`). Bootstrap adds it when it finds the command elsewhere. It replaces `IJ_CLAUDE_BIN`, which no longer exists |
| `IJ_RESEARCHER_USER`, `IJ_RESEARCHER_SUDO` | the researcher's Unix account (`researcher`) and the `sudo` prefix the driver uses to act as it |
| `IJ_FROZEN_PATHS` | the paths the driver hashes every pass: `/opt/inference-judge`, and in the research repo `loop/`, `guard/`, `adapters/` and the agent's settings folder |
| `IJ_PRIMARY_MODEL`, `IJ_FALLBACK_MODEL` | by default the words `primary` and `fallback`, which each adapter maps to a model. Any other value is passed to the agent as a model name |

When `loop.env` already exists, `harden_research.sh` keeps your settings and
rewrites only `IJ_AGENT` and `IJ_FROZEN_PATHS`. When it changes them, it also
drops a leftover `IJ_CLAUDE_BIN` line, and an `IJ_AGENT_BIN` line if the agent
changed.

## Decisions

Choices that the scripts do not explain on their own.

1. **CUDA 13 only, for an engine built here.** `setup/Dockerfile.base` installs
   the engines' PyPI wheels, and the torch channel for them is `cu130`. CUDA 13
   needs driver r580 or newer. There is no cu128 path: `gpu_channel` accepts
   `cu130` and nothing else, and a driver below r580 fails the gpu step, with
   the reason, rather than building something that cannot run. None of this
   binds a target that names a prebuilt engine image: that image carries its
   own CUDA, and the gpu gate is where a driver too old for it shows.
   `CUDA_IMAGE` is `nvidia/cuda:13.0.3-devel-ubuntu24.04`.
   The engine tags have no default: the operator names them (`IJ_VLLM_TAG`,
   and `IJ_SGLANG_TAG` when the target names sglang) and bootstrap records
   them.
2. **Wheels, not `VLLM_USE_PRECOMPILED` editable installs.** An editable
   install with `VLLM_USE_PRECOMPILED=1` on a detached tag does not give
   matching prebuilt kernels: vLLM's `setup.py` asks git for
   the current *branch* to find a merge base, a detached checkout has none, and
   it falls back to a **nightly** wheel whose kernels do not match the source.
   The image therefore does `uv pip install vllm==<version>`, keeps a read-only
   source checkout at the same tag for patching, and records the exact wheel URL
   it installed in **`/opt/vllm-wheel-url`**. An experiment that patched python
   under `/src/vllm` re-installs against that URL:

   ```bash
   VLLM_PRECOMPILED_WHEEL_LOCATION="$(cat /opt/vllm-wheel-url)" \
     uv pip install --python /opt/venv/vllm/bin/python -e /src/vllm --no-deps
   ```

   `experiments/TEMPLATE/launch.sh` documents this where the researcher reads it.
3. **Two venvs in the image, one on the host.** vLLM and SGLang pin conflicting
   torch/flashinfer/tokenizers versions. The image has `/opt/venv/vllm` (first on
   PATH) and `/opt/venv/sglang`; a failed SGLang install leaves
   `/opt/venv/sglang/.install-failed` instead of failing the build, and so does
   a build for a target that names no sglang (nothing is installed there). The host venv
   installs the same wheels and is for reading source, editing it, and quick
   scripts — nothing is served from it.
4. **Install-or-resume is the default.** Re-running bootstrap after a failure
   must not force-checkout engine source the researcher patched, must not chmod
   existing results back to private, and must not restart the clock. The two
   destructive behaviours are `--reinstall-engines` and `--reset-clock`.
5. **`/home/researcher` is 0751.** Ubuntu 24.04 creates home directories 0750, which
   stops `bench` from traversing into `research/experiments` — and `run.sh` runs
   as `bench`. `harden_research.sh` sets 0750 too, so bootstrap sets 0751 again
   after it, and both bootstrap and check.sh verify it with an actual
   `sudo -u bench test -r`, not by reading the mode.
6. **`/opt/bench` is 0751, not 0750.** `/opt/bench/results/<id>` must be
   world-readable so that the researcher can read `results.json`, and that needs
   execute permission on `/opt/bench`. The researcher still cannot list
   `/opt/bench` or read the harness or the prompt pools. `check.sh` accepts 0751
   or 0750 for the directory.
7. **`bench` is in the docker group**, because `run.sh` builds and runs the
   experiment containers. `researcher` is not, and check.sh fails if it ever is.
8. **The clock and the ruler lock are one step.** DESIGN 2.6 has `RULER_LOCK`
   written at clock start. Bootstrap's clock step does it, by asking the ruler
   to lock itself as the driver would
   (`sudo -u driver sudo -n -u bench /opt/bench/run.sh --lock-ruler`), and
   `--reset-clock` removes the lock first. The rsync into `/opt/bench` never
   touches `TARGET`, `targets/`, `RULER_LOCK`, the references or the
   calibration.
9. **The rehearsal goes through the queue.** `rehearse.sh` writes experiments
   and queue files as `researcher` and runs the installed `runner.sh --once` as
   `driver` against a scratch state directory with its own clock, so it tests
   the route the day uses and none of it reaches the researcher's tallies. Its
   results directories are moved to `/var/log/inference-judge/rehearse-<utc>/` because
   the driver counts every directory under `/opt/bench/results`.

## Expected WARNs from check.sh

| WARN | meaning |
|---|---|
| `not running as root` | you ran `check.sh` as another user, so the permission checks may be incomplete |
| `ubuntu 24.04` | the machine runs another release; the scripts are written for 24.04 |
| `docker GPU passthrough` | `docker run --gpus all ... nvidia-smi -L` failed (gpu targets that build their image) |
| `host venv vllm importable` | the host venv cannot import `vllm` (targets that build their engine) |
| `cache disk in fstab (survives a reboot)` | the cache disk would not come back after a reboot; re-run bootstrap |
| `nsys` | `nsys --version` failed (gpu targets only) |
| `image sglang venv` | the SGLang install failed at image build, so the sglang track is unavailable |
| `stored reference` | only before the operator's reference run |
| `metric.prompt_pool`, `metric.confirm_pool` | the named pool file is missing or empty |
| `compile-cache seed filled` | the seed directory is empty, so the first launch compiles inside its budget |
| `ruler locked (DESIGN 2.6)`, `clock started` | bootstrap ran with `--no-clock` |
| `results` | no run yet |
| `gpu idle` | something is still holding the GPU — check before starting the loop |
| `gpu gate`, `chip gate` | skipped because a container is running; it would have started a second one on the same accelerators |

## If bootstrap fails

Every step is resumable: fix the cause and re-run the same command. A failed
critical step aborts immediately, so the clock is never started on top of a
broken install. Common causes and fixes are in `docs/RUNBOOK.md`.
