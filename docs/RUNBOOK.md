# Runbook: from an empty machine to a running loop

Use this page to turn a machine into an inference-judge box, start the loop, watch it, stop it and take the results away. It assumes you can use a shell and `ssh`; it does not assume you know anything about inference serving.

The page follows the order in which you do things:

1. [Pick a path](#pick-a-path-and-what-has-been-proven): laptop, TPU or NVIDIA GPU.
2. [Choose or write a target](#choose-or-write-a-target) and [choose the agent](#choose-the-agent).
3. [Read the plan](#read-the-plan) that bootstrap would run.
4. [Bootstrap the machine](#bootstrap-the-machine).
5. [Log the agent in](#log-the-agent-in), then [check the box](#check-the-box).
6. [Measure the reference and the calibration](#reference-and-calibration).
7. [Rehearse](#rehearse), then [start the loop](#start-the-loop).
8. [Watch it](#watch-the-loop), [stop it](#stop-the-loop), [take the results away](#take-the-results-away) and [read the summary](#read-the-summary).

When a step fails, go to [If bootstrap fails](#if-bootstrap-fails). The [limits](#known-limits) you should know about are at the end.

## Pick a path and what has been proven

Every path uses the same scripts. Only the target file and a few hardware steps differ. The table says how far each path has been taken.

| path | where it starts | what has been run |
|---|---|---|
| CPU mock, on a laptop | [Try it on a laptop first](#try-it-on-a-laptop-first) | The judge and the loop run end to end on a laptop. Part of that run uses a stand-in judge; the section says which. |
| Google Cloud TPU v6e | [A TPU VM](#a-tpu-vm) | The model server was brought up by hand on a v6e-4. `setup/tpu.sh`, bootstrap and a judged run have not yet been run on TPU hardware. The scripts are tested with fakes. |
| NVIDIA GPU | [An NVIDIA GPU machine](#an-nvidia-gpu-machine) | The scripts are built and tested on a laptop with fakes. Nothing has run on a real GPU. |

Expect the first run on real hardware to find something. If you only want a speed table for a server you already have, you do not need this page: `uv run bench/measure.py URL --model NAME` needs no setup (see `README.md`).

A word on names. The code calls the judge "the ruler". This page says "the judge", except in file and variable names such as `RULER_LOCK`.

## Try it on a laptop first

You can run the whole procedure below on a laptop, with no accelerator, no model and no cloud account. It needs Docker and [uv](https://docs.astral.sh/uv/).

```bash
./demo.sh                                    # the cheat suite, then the loop rehearsal
bash setup/local_box/run_local_box.sh        # only the loop rehearsal, exit 0 iff every check passed
LOCALBOX_TIER=2 bash setup/local_box/run_local_box.sh
```

What each one proves:

- **`demo.sh` part 1, the cheat suite**, runs the real judge against mock servers that fake a speed-up. It exercises the judge itself.
- **`demo.sh` part 2 and `run_local_box.sh`** install a box in a container with the real `setup/bootstrap.sh` for the `cpu-mock` target, then run the real driver, runner and heartbeat units. A scripted stand-in plays the agent through the Claude adapter, and a stand-in judge plays the judge. This tests the install, the three Unix users, the queue, the driver and crash recovery. It says nothing about the judge's statistics.
- **`LOCALBOX_TIER=2`** replaces the stand-in judge with the real `bench/run.sh` measuring the sealed mock server. It then runs the reference run, a short calibration (two launches, not ten), and the rehearsal with every case, as described below. It takes much longer. It never reaches a `PROMOTED` run.

The rehearsal's quick tier (`setup/rehearse.sh --quick`) is the one the laptop box runs in its default tier. It is judged by the stand-in. `setup/local_box/README.md` lists what the laptop box cannot show: any accelerator, real build and compile times, a real model, and any real coding agent. It does not rehearse the Codex or Pi adapters.

## Choose or write a target

Everything about the machine, the engine, the model and the metric comes from one file, `targets/<id>.yaml`. Bootstrap, the checks and the judge all read it through `bench/target.py`.

| file | machine | state |
|---|---|---|
| `targets/cpu-mock.yaml` | none: a mock server, any laptop | runs end to end |
| `targets/v6e4-gemma4-26b-tps.yaml` | Google Cloud TPU v6e-4 | `runnable: false`; the sealed launch is marked NOT PROVEN in the file |
| `targets/gpu-gemma4-26b-tps.yaml` | one NVIDIA GPU | `runnable: false`; never run on a GPU |
| `targets/sketch-tpu-8chip.yaml` | an eight-chip TPU | a sketch: it validates and shows a plan, and nothing more |

To use your own machine, write a new file. [`your-first-target.md`](your-first-target.md) walks through it section by section, and every example in it validates.

Validate the file before you copy anything to the machine:

```bash
make -C bench venv                                           # once; makes bench/.venv
bench/.venv/bin/python bench/target.py validate targets/<id>.yaml
```

A file that does not validate prints the field at fault. Bootstrap refuses the same file with the same message.

Finish every edit to the target file now. When bootstrap starts the clock it also locks the judge, and the lock covers the target file (see [The clock and the lock](#the-clock-and-the-lock)).

## Choose the agent

The agent is the coding program that does the research. The loop reaches it through an adapter, a folder in `research/adapters/<name>/`. Three adapters ship: `claude` (Claude Code), `codex` (Codex CLI) and `pi` (the Pi coding agent). `research/adapters/TEMPLATE/` is a starting point for your own. [`research/adapters/README.md`](../research/adapters/README.md) describes what the loop asks of an adapter.

Pick the agent when you bootstrap, with `--agent NAME` or `IJ_AGENT=NAME`. The flag wins over the variable, and the default is `claude`. Bootstrap refuses a name that has no `research/adapters/NAME/session.sh` and lists the adapters it found. It writes your choice to `IJ_AGENT` in `/etc/inference-judge/loop.env`, and the driver, `setup/check.sh` and `setup/rehearse.sh` read it from there.

`--plan` shows the choice before you touch a machine. For Codex, on a machine that does not have `codex` yet, it prints these lines among the others:

```text
agent: codex (research/adapters/codex)
RUN   node 22 + npm
RUN   agent codex (adapters/codex/setup.sh install)
```

With no `--agent` the first line reads `agent: claude (research/adapters/claude)`. Where the agent's command is already on the `PATH`, the two `RUN` lines become `SKIP` lines.

**What has been proven for each agent.**

- **Claude Code** is the only agent that has run a real loop, and that was before the loop called it through `session.sh`. Through the adapter it has been tested with a stand-in program only. A real session through the adapter, on a laptop, did not get past an expired login.
- **Codex** and **Pi**: the adapters were written from those tools' documentation and tested with stand-in programs that print the documented output. Each has then run one short real session on a laptop through its adapter, a five-step wiring check and not research. The adapter read the agent's output, the guard refused an edit of `loop/run.sh` and a call to `nvidia-smi`, `audit.log` got a line for each call that ran, and the agent had `AGENTS.md` in its context. For Pi, `guard.ts` loaded. Neither has run a real loop. Nobody has yet seen, for either, the driver starting sessions on a hardened machine as `researcher` with the managed settings file in place, or the wording of their quota and login errors. For Codex, nobody has seen the hooks enforced from `/etc/codex/requirements.toml`. `research/adapters/README.md` has the detail.
- For Claude Code, nobody has checked that a session without a person loads `AGENTS.md` through the `@AGENTS.md` line in `CLAUDE.md`. The driver's prompt names `AGENTS.md` too, so the agent is told to read it either way.

Treat a first loop with Codex or Pi as a test of the adapter. Read `research/adapters/codex/session.sh` or `research/adapters/pi/session.sh` first, and try one session by hand (step 5 of "Write your own" in the adapters README) before you trust it with a long run.

## Read the plan

Print what bootstrap would do for your target before you touch a machine. This works as any user, on your laptop, and changes nothing. The bench environment goes first on `PATH` because the target loader needs PyYAML.

```bash
PATH="$PWD/bench/.venv/bin:$PATH" bash setup/bootstrap.sh --target targets/<id>.yaml --plan
PATH="$PWD/bench/.venv/bin:$PATH" bash setup/bootstrap.sh --target targets/<id>.yaml --agent codex --plan
```

The second command shows the plan for Codex instead of the default agent. In either plan the first line is a summary of the target: its id, accelerator, engine, model and metric. The second names the agent, as in `agent: claude (research/adapters/claude)`. Each later line is one step:

| line starts with | meaning |
|---|---|
| `RUN` | Bootstrap runs this step. A step marked `(optional)` records a WARN and carries on if it fails; every other step aborts bootstrap if it fails. |
| `SKIP  <step>  -- <reason>` | Bootstrap does not run this step for this target. The reason is on the line. |
| `STOP` | The target has `runnable: false`. Bootstrap runs the steps above this line, prints what to do next and exits with code 3. The lines below it run after you set `runnable: true` and run bootstrap again. |

Read these lines against what you meant to ask for:

- **The accelerator line.** `gpu: nvidia-smi works and shows N GPU(s)` or `tpu: the host shows all N chips` means the target asks for that hardware. `SKIP  accelerator driver` means the target has `accelerator.kind: none`.
- **The engine lines.** Bootstrap builds the engine on the box (`build inference-judge-base`), builds a mock image from `targets/<id>.Dockerfile`, or pulls a prebuilt image. If it builds the engine, you need to name release tags before the real run (see [Bootstrap the machine](#bootstrap-the-machine)).
- **The model line.** A line that starts `RUN   model <org>/<name>@<40-hex revision>` means bootstrap downloads exactly that revision. `SKIP  model download  -- target names no model.hf_id` means the weights must already be at the target's `model.local_dir`. On a TPU target with a cache disk, the line says the model comes from the disk's Hub cache.
- **The agent lines.** `agent: <name> (research/adapters/<name>)` is the agent you chose (see [Choose the agent](#choose-the-agent)). `RUN   agent <name> (adapters/<name>/setup.sh install)` means bootstrap installs it, and `SKIP  agent <name>  -- <name> already on PATH: <path>` means it will not. An adapter with no `setup.sh` shows a `RUN   agent <name> (no adapters/<name>/setup.sh: ...)` line that tells you to install the command yourself. Bootstrap then installs nothing, and you install it before the loop starts.
- **The reference line.** `SKIP  reference + calibration  -- run by hand` appears in every plan. Bootstrap does not measure the reference. You do, in [Reference and calibration](#reference-and-calibration).
- **The clock line.** `RUN   research clock ... and RULER_LOCK` is the last step. It becomes a `SKIP` line when you pass `--no-clock`.

`--plan` does not need the engine tags, and it does not need root.

## Bootstrap the machine

You need an Ubuntu machine (the scripts are written for 24.04 and warn on any other version), root, and the hardware the target describes. Bootstrap checks the accelerator and Docker. It does not install an NVIDIA driver or the NVIDIA container toolkit.

Copy the checkout to the machine. This example puts it at `/root/inference-judge`:

```bash
rsync -az --exclude .git --exclude '*/.venv' <checkout>/ root@<box>:/root/inference-judge/
ssh root@<box>
```

Then run bootstrap as root. Keep its output: every log also lands in `/var/log/inference-judge/`.

```bash
mkdir -p /var/log/inference-judge
cd /root/inference-judge
IJ_VLLM_TAG=<tag> IJ_SGLANG_TAG=<tag> \
  bash setup/bootstrap.sh --target targets/<id>.yaml 2>&1 | tee /var/log/inference-judge/bootstrap.log
```

Add `--agent NAME` (or set `IJ_AGENT=NAME`) to use an agent other than the default `claude`; see [Choose the agent](#choose-the-agent).

Which variables you set depends on how the target gets its engine:

- **A built engine** (vLLM or SGLang on `inference-judge-base`) needs `IJ_VLLM_TAG`, and `IJ_SGLANG_TAG` when the target serves SGLang or lists its source tree. Each must be a release tag such as `v0.29.0`. Bootstrap refuses `latest`, a branch name or a missing tag before it uses the network, and records the tags in `/etc/inference-judge-engine-pins` and in the image label. The wheels these builds install need CUDA 13, so the NVIDIA driver must be r580 or newer.
- **A prebuilt image** needs neither variable. Name it by digest (`name@sha256:...`). A TPU target refuses a tag, and any other target gets a warning for one.
- **A gated model** needs `HF_TOKEN` in the environment of the bootstrap command. You supply your own token; nothing else uses it.

The other options, all in `bash setup/bootstrap.sh --help`:

| option | effect |
|---|---|
| `--agent NAME` | The coding agent that does the research: a folder in `research/adapters/` (`claude`, `codex`, `pi` or your own). Also `IJ_AGENT=NAME`; the flag wins. Default `claude`. |
| `--skip-model` | Skip the weights, for a dry install. |
| `--no-clock` | Install everything but do not start the clock or lock the judge. |
| `--reset-clock` | Restart the clock even if it is running. **Destructive:** the hours already run are lost. |
| `--reinstall-engines` | Force the engine checkouts back to the pinned refs, or copy them out of the image again on a TPU. **Destructive:** it discards uncommitted engine patches. |
| `IJ_VERIFY_MODEL=1` | Re-hash every weight file against the Hub manifest (slow). |
| `IJ_ACCEL_HOURS`, `IJ_WALL_HOURS` | The run's budget in accelerator hours, and its wall-clock deadline in hours from the clock start. The deadline defaults to the accelerator-hour budget, and that defaults to 24. |

Bootstrap is install-or-resume. After a failure, fix the cause and run the same command again: it fixes what is missing and touches nothing that works. It never restarts a running clock. Its exit code is 0 when every critical step passed, 1 when a critical step failed, 2 for a usage error or an invalid target, and 3 for the deliberate stop on a `runnable: false` target. It ends with a checklist that marks each step PASS, WARN, FAIL, SKIP or STOP.

### What bootstrap installs

For every target, it installs the following, whatever the hardware:

- the packages Docker and the tools need, and the users `bench` and `researcher`;
- `/opt/bench`, the judge, with its own virtual environment, and the ruler image `inference-judge-ruler:latest`;
- the target, as `/opt/bench/targets/<id>.yaml` for the judge and as `/etc/inference-judge/target.yaml` (world-readable) for the researcher;
- `/home/researcher/research`, the researcher's repository, with the files from `research/adapters/<agent>/project/` copied to its top (for Claude Code, `CLAUDE.md` and `.claude/`; for Codex, `.codex/`);
- the agent, through `adapters/<agent>/setup.sh install`, unless its command is already on the `PATH`. An adapter with no `setup.sh` is not an error: you install the command yourself;
- the `driver` user, the root-owned copies of the loop, the guard and the adapters in `/opt/inference-judge`, the agent's managed settings file when its adapter has one (Claude Code: `/etc/claude-code/managed-settings.json`; Codex: `/etc/codex/requirements.toml`; Pi has none), `/etc/inference-judge/guard.env`, the sudoers rules and the systemd units, through `setup/harden_research.sh`;
- `/etc/inference-judge/loop.env`, with `IJ_AGENT` set to your agent. If the agent's command is outside the `PATH` that `sudo` allows, bootstrap also adds `IJ_AGENT_BIN` (the command's full path) to it;
- last, the run clock and `/opt/bench/RULER_LOCK`.

### The three users

Nobody on the box can both write an experiment and measure it. That separation is the reason to believe the numbers.

| user | role | owns | may not |
|---|---|---|---|
| `bench` | the judge | `/opt/bench`: `run.sh`, the prompt pools, the results, the leaderboard, the reference and the calibration. It is in the `docker` group and owns the accelerator device group. | be reached by the researcher |
| `researcher` | the researcher (the agent's Unix account) | `/home/researcher/research`, except the root-owned `loop/`, `guard/`, `adapters/` and the agent's settings folder (`.claude/` or `.codex/`), and the engine checkouts | run the judge, read `/opt/bench` apart from `results/`, use Docker or the accelerator, use `sudo`, or edit the loop, the guard or the adapters |
| `driver` | the loop | `/home/driver/research-state`; it runs the units from the root-owned `/opt/inference-judge/loop` | do anything as root |

The researcher asks for a measurement by writing `queue/<id>.json`. The runner (`inference-judge-runner`) takes the oldest request, runs a pre-flight check, then calls `sudo -n -u bench /opt/bench/run.sh <id>`. The judge measures one experiment at a time, and a second caller gets the verdict `BUSY`. The driver (`inference-judge-driver`) starts the agent's sessions through the adapter. `inference-judge-heartbeat.timer` writes a status file every minute. You are root, outside all three.

The sudoers rules sit in `/etc/sudoers.d/inference-judge-driver`: `driver` may run `/opt/bench/run.sh` as `bench`, and anything as `researcher`.

### The clock and the lock

The last bootstrap step starts the run clock and locks the judge.

- **The clock** is an event in the driver's log, `events.jsonl`. It sets the budget in accelerator hours and a wall-clock deadline, and the run ends at whichever comes first. The researcher cannot move it.
- **`RULER_LOCK`** is a hash of the judge's files (`harness.py`, `metrics.py`, `compare.py`, `quality2.py`, `stats.py`, `tokenizer.py`, `target.py`, `run.sh`) and of the target file. From then on `run.sh` refuses to measure, with exit code 70, if any of them differs from what was locked. One judge scores a run from start to end.

Two consequences:

- Do not edit `bench/` or the target file after the clock starts. If you must, the only way forward is a new run: `bash setup/bootstrap.sh --target ... --reset-clock`. That discards the hours already run, and you measure the reference and the calibration again.
- The units are installed but not enabled, so nothing spends the clock yet. The rehearsal restarts the clock when it passes, provided no research session has run. Time you spend on the reference, the calibration and the rehearsal is therefore not charged to the run, and results that finished before the clock restarted are not counted by the driver.

A target with `runnable: false` gets neither the clock nor the lock. `run.sh` refuses to measure on it.

### A TPU VM

**What has been proven.** The model server of `targets/v6e4-gemma4-26b-tps.yaml` was brought up by hand on a v6e-4, with the flags in `research/experiments/baseline-tpu/launch.sh`. `setup/tpu.sh` is tested with fakes (`setup/tests/test_tpu.py`). `setup/tpu.sh`, bootstrap and a judged run have not yet completed on a TPU. The target ships as `runnable: false`, and its sealed launch is marked NOT PROVEN in the file's header. Read that header first.

What the TPU steps do (`setup/tpu.sh`):

1. Count the chips, using the target's `probes.present` and falling back to the device nodes under `/dev/vfio` and `/dev/accel`.
2. Mount the disk whose ext4 label is `provision.cache_disk.label` at `provision.cache_disk.mount`, and add it to `/etc/fstab` with `nofail`.
3. Load the engine image from the disk's tar when it is not in the Docker store at its pinned digest, falling back to `docker pull` by digest, and check the digest.
4. Run the **chip gate**: start the engine image the way a real launch starts it, with the target's own `docker_flags` and `env`, and require `jax.devices()` to list every chip.
5. Stop there, with exit 3, if the target is `runnable: false`.
6. Copy the engine source trees in `engine.sources` out of the image into `/home/researcher/<name>-src`, each as a git repository with one baseline commit tagged `inference-judge-baseline`.
7. Make the model a flat directory of hardlinks into the disk's Hugging Face cache, and check every file against the hash in its blob name.
8. Add the disk's compile cache to the seed named by `engine.compile_cache.seed`, owned by `bench`. This only adds files and never replaces one.

The order that works:

1. **Create the VM** with your cloud tools, matching the target's `accelerator` section, and attach the cache disk named in its `provision.cache_disk`. No script creates the VM for you; the `provision` fields `zone`, `runtime_version` and `mode` are notes for the person who does. Create it as a plain TPU VM and not through a queued resource, because the spend guard below is not written for a queued-resource node.
2. **Copy the checkout** to the VM and become root (see [Bootstrap the machine](#bootstrap-the-machine)).
3. **Arm the spend guard first.** `tools/tpu_deadman.sh` is a systemd timer on the VM that deletes it at a deadline, whether or not anyone is connected:

   ```bash
   sudo bash tools/tpu_deadman.sh arm 8          # delete this VM in 8 hours
   bash tools/tpu_deadman.sh status              # deadline, time left, where results were last saved
   sudo bash tools/tpu_deadman.sh extend 2       # add 2 hours; `disarm` cancels
   sudo bash tools/tpu_deadman.sh save           # copy the results to the cache disk now
   ```

   `arm` checks that it can authenticate, and exits 3 with the timer still armed when it cannot confirm that deletion would work. It cannot test the delete permission without deleting, so treat exit 3 as "not proven". While armed, the guard copies the results, leaderboard, calibration, reference, the researcher's repository and state, the logs and the compile seed to the cache disk every hour, and once more before the delete, under `inference-judge-saved/` on the disk. At the deadline it tries `gcloud compute tpus tpu-vm delete` five times. If all five fail it halts the VM, which may not stop the billing. Check the VM list from your laptop after any deadline. Size the guard to cover the reference, the calibration, the rehearsal and the loop together.
4. **Bootstrap, first pass.** It stops after the chip gate with exit 3 and prints what to do next.
5. **If the gate says the target's flags cannot reach the chips**, find flags that can. Bootstrap prints this exact command:

   ```bash
   sudo bash tools/tpu_seal_probe.sh '<engine.image from the target>' <chips>
   ```

   The probe tries the strongest container seal first and ends with a line `engine.docker_flags: ...`. If it reports no working set, the node is faulty: delete it and create another.
6. **Edit the target once, completely.** Put the probe's flags into `engine.docker_flags` as a YAML list, without `--network none` (the judge adds it) and without the `-e NAME=VALUE` pairs. Put each pair into `engine.env` as `NAME: VALUE`. Set `runnable: true`.
7. **Run bootstrap again**, to the end. Every step is safe to repeat. Then continue from [Log the agent in](#log-the-agent-in).
8. **Copy the edited target back to your laptop** and commit it, or the next fresh VM stops at the same gate.

A launch whose container flags are weaker than a full seal is recorded as such: `results.json` names what the container was given in its `isolation` record, and a privileged launch says so. A full seal means no device, no host socket and no privilege.

`tools/tpu_probe_logprobs.py` checks, against a server you started by hand, the behaviours the quality gate needs from the engine. Run it before you rely on a new engine image.

### An NVIDIA GPU machine

**What has been proven.** The GPU path is built and tested on a laptop with fakes for `nvidia-smi`, `docker` and `git` (`setup/tests/test_gpu.py`). No real GPU has run it. `targets/gpu-gemma4-26b-tps.yaml` ships as `runnable: false` for that reason.

The machine must already have an NVIDIA driver (so that `nvidia-smi` works) and Docker with the NVIDIA container toolkit (so that `docker run --gpus all` works). Bootstrap checks both and installs neither. The GPU steps (`setup/gpu.sh`) check:

- **The GPU count.** `nvidia-smi` must show exactly `accelerator.chips` GPUs. More or fewer is a failure, not a warning, because the judge starts every launch with `--gpus all`. A host with extra GPUs would serve on hardware the target does not describe.
- **The type and the memory, when the target names them.** `accelerator.type` is matched against the name `nvidia-smi` reports, ignoring case, as a substring. `accelerator.hbm_gb_per_chip` is checked against the card's memory at 95% of the nominal size, because cards report a little less.
- **The driver, for a built engine.** It must be r580 or newer. A prebuilt image carries its own CUDA, and the gate shows whether the driver is new enough for it.
- **The gpu gate, for a prebuilt image.** Bootstrap starts `<image> bash -c 'nvidia-smi -L'` with the target's `docker_flags`, `env` and `shm_size`, `--gpus all` and no network, which is how the judge starts a launch. It fails on a missing container toolkit, a driver older than the image's CUDA needs, an image entrypoint that the flags do not clear, or flags that hide a GPU. It does not load a model. Whether the engine serves and whether the weights fit is first seen at the reference run.

For a `runnable: false` target, bootstrap stops after the gate with exit 3 and prints these steps:

1. Read the comment at the top of the target file. It lists what nobody has verified.
2. Finish the file's edits now, because the judge locks its bytes when the clock starts.
3. Set `runnable: true`.
4. Run bootstrap again. With `--no-clock`, it installs everything and leaves the clock for a later run. That is the place to see the first real launch of the baseline, before any hour is counted. Run bootstrap once more without `--no-clock` before the loop, so that the clock starts and the judge is locked.

A `gpu` target gets every GPU on the machine, so give the judge a machine with no spare GPUs. `deps.txt` in an experiment installs into `/opt/venv/vllm` when the image has it, and into the system Python otherwise.

## Log the agent in

The agent runs as the Unix user `researcher`, so the login has to happen as `researcher`, and you do it yourself. Each adapter knows its own login line. Print it as root, from the checkout on the box, then open a shell as `researcher` and run that line:

```bash
bash /root/inference-judge/research/adapters/<agent>/setup.sh login     # prints the line to run
ssh -t root@<box> "su - researcher"
```

Bootstrap prints the same line at the end of its output, and so does `check.sh` when the login is missing. In short, the three adapters say this:

| agent | what to run as `researcher` |
|---|---|
| `claude` | `claude`, then log in when it asks, then `/exit`. In one step: `ssh -t root@<box> "su - researcher -c claude"` |
| `codex` | `codex login`, or `codex login --device-auth` on a machine with no browser |
| `pi` | `pi`, then `/login`, pick a provider, then `/quit` |

Follow the login flow the agent shows. The login persists, so this is the only interactive step. Do not copy credentials or tokens from another machine onto the box.

Use `su - researcher`, not `sudo -iu researcher`. macOS `ssh` clients forward `LC_*` locale variables, and the audit plugin of `sudo` on some Ubuntu images aborts on a locale it cannot load. The setup scripts pin `LC_ALL=C.UTF-8` for the same reason, and so should you: start every root shell on the box with `export LC_ALL=C.UTF-8 LANG=C.UTF-8`.

The account needs enough quota for unattended sessions for the whole run. The driver treats a usage limit as a reason to wait, not as a crash. It treats a failed login as a pause with an alert, and probes again every 1800 s.

## Check the box

```bash
bash /root/inference-judge/setup/check.sh
```

`check.sh` verifies every precondition for the installed target, changes nothing and exits non-zero if anything failed. It prints PASS, WARN or FAIL per item, so you can run it at any time, including while the loop runs. Every line should be PASS, apart from WARNs such as these.

| WARN | meaning |
|---|---|
| `stored reference` | You have not yet run the reference. Expected until the next section. |
| `ruler locked (DESIGN 2.6)`, `clock started` | Bootstrap ran with `--no-clock`. |
| `results` | No run yet. |
| `nsys` | The NVIDIA developer-tools repository was unreachable. Profiling in the container still works if the image has `nsys`. |
| `image sglang venv` | SGLang could not resolve on this CUDA channel. The run goes on with vLLM. |
| `gpu idle` | Something still holds the GPU. Find out what before you start the loop. |
| `gpu gate`, `chip gate` | Skipped because a container is running. |

For Claude Code and Codex, a missing login is a FAIL, not a WARN. The agent section at the end of the output reads `agent <name> is logged in`. On a FAIL it prints the `ssh` and login line from [Log the agent in](#log-the-agent-in). Codex's adapter asks Codex itself. Claude Code's adapter asks `claude auth status`, so a current Claude Code gives `agent claude is logged in`, and no login is a FAIL. A Claude Code too old to have that command falls back to checking whether anything is stored in the researcher's home folder: nothing stored is a FAIL, and anything else is a PASS that reads `agent claude has a login` with the note "log-in state not inspected". Pi checks a login per model. When `IJ_PRIMARY_MODEL` in `loop.env` names a model, `check.sh` asks Pi about that model and a missing login is a FAIL. With the default word `primary`, Pi's adapter cannot tell, so the line reads `agent pi has a login` whether or not you have logged in. For Pi, a login that has expired shows only when a session runs. It has not been seen whether `claude auth status` reports an expired Claude Code login. The `agent <name> (its adapter's version, as researcher)` line shows that the agent's command runs as the researcher.

## Reference and calibration

Bootstrap does not do this step. The plan said `SKIP  reference + calibration  -- run by hand`. You run it once, as root, after the checks and before the rehearsal.

A candidate is judged against the current best configuration, the incumbent, started again in the same run. Before the first candidate there must be a first incumbent, a stored set of reference answers to hold candidates to, and a measured noise level. Each command below ends by printing `VERDICT <word>`. If you get a different word from the one named, stop and read the log.

**1. The reference run.** The reference is a plain, documented way to serve the model for this target, with no tuning. It lives in `research/experiments/<name>/launch.sh`. The target picks its name with `provision.baseline`. Without that key the name is `baseline-tpu` on a TPU and `baseline-<engine.name>` otherwise. Bootstrap's first step prints it as `baseline experiment: <name>`, and refuses to start if the launcher is missing. For a new engine, write one from `research/experiments/TEMPLATE/` before you bootstrap, and do not tune it afterwards: every promotion is measured against it.

```bash
install -o bench -g bench -m 0640 /dev/null /opt/bench/.allow-baseline
sudo -u bench /opt/bench/run.sh <name> 2>&1 | tee /var/log/inference-judge/<name>.log
rm -f /opt/bench/.allow-baseline
```

It must end with `VERDICT BASELINE`. Ids that start with `baseline-` are reserved for you. The judge accepts one only while `/opt/bench/.allow-baseline` exists, which is why the researcher can never replace the reference. The run stores:

- the canary reference, `/opt/bench/canary/<target id>.json`: the answers to ten fixed prompts, which show later that the model, template and sampling did not change;
- the reference answers every candidate's quality is compared with, in the directory named by the target's `reference.dir`;
- the first incumbent, in `/opt/bench/leaderboard/<target id>.json`.

On the mock it is one launch of the server. If the target sets `reference.launches`, as a real engine does, the reference answers come from that many cold launches, and the run's time limit grows to match. If a baseline run fails, `run.sh` refuses to reuse its id (exit 2, "pick a new id"), because `/opt/bench/results/<name>` already exists. Move that directory out of `/opt/bench/results` before you run it again. `/var/log/inference-judge/` is a good place.

**2. The calibration.** This is ten launches of the same incumbent, to measure how much two identical launches differ:

```bash
env -u SUDO_USER runuser -u bench -- /opt/bench/run.sh --calibrate calibrate-1 \
  2>&1 | tee /var/log/inference-judge/calibrate-1.log
```

It must end with `VERDICT CALIBRATED`. It writes `/opt/bench/calibration/<target id>.json`, which holds `sigma_L`, the launch-to-launch noise, and the minimum detectable effect, the smallest gain the judge will call real. Without this file every candidate ends `INCONCLUSIVE`. `calibrate-1` is just an id, and a second calibration needs a new one.

The command looks odd for a reason. `--calibrate` and `--confirm` are operator modes, and `run.sh` refuses every flag when it is entered through `sudo`, because that is the door the researcher's runner uses. `runuser` enters as `bench` without `sudo`, and `env -u SUDO_USER` covers the case where your root shell itself came from `sudo`.

**How long.** One launch is the server's start-up, bounded by the target's `health.ready_timeout_s`, or by `budgets.compile_s` plus `budgets.health_s` when that is not set, followed by one warm sweep and `metric.blocks` scored sweeps. The calibration is ten launches, and `run.sh` scales its time limit to match. Work out the time from your target file, run both commands inside `tmux`, and extend the TPU spend guard to cover them.

Then run `check.sh` again. The `stored reference` line should now be PASS.

**Running the judge by hand later.** The judge measures one experiment at a time, and a second caller gets `BUSY`. If you start a measurement while the runner is working, one of the two is wasted, and it may be the researcher's. Every operator run during a live loop therefore follows the same three lines:

```bash
pgrep -u bench -f /opt/bench/run.sh || systemctl stop inference-judge-runner   # only while the judge is idle
#   ... your run.sh command ...
systemctl start inference-judge-runner
```

If `pgrep` prints a number, a measurement is running. Wait for it and try again.

To measure a promoted run again on prompts it has not seen, against the previous best, use `env -u SUDO_USER runuser -u bench -- /opt/bench/run.sh --confirm <id>`. The leaderboard then keeps the confirmed effect. `--confirm` refuses a `baseline-*` id.

## Rehearse

```bash
bash /root/inference-judge/setup/rehearse.sh
```

The rehearsal sends deliberately broken experiments through the real path before the units are enabled, so a funnel bug is found before the run and not during it. Every case takes the route the unattended run uses: `researcher` writes the experiment and queues it, the installed runner picks it up as `driver`, and the judge measures it as `bench`. Nobody calls `run.sh` by hand.

It needs root and `jq`. It refuses to run while the runner or the driver is active, and it refuses a target that has no incumbent yet. That is why it comes after the reference.

| case | what it is | what must happen |
|---|---|---|
| `driver` | the driver's dry run, as `driver` | The agent's `version` command runs as `researcher`. The dry run prints a session command that calls `adapters/<agent>/session.sh run` with `-u researcher`, `--model primary` and `--effort high` (or the values `IJ_PRIMARY_MODEL` and `IJ_EFFORT` name in `loop.env`). The driver's classifier for limits and crashes passes its self-test. It starts no session, takes seconds and uses no quota. |
| `preflight` | a `launch.sh` with a syntax error | `PREFLIGHT_FAILED` in seconds, and no judge slot is spent |
| `buildfail` | a `launch.sh` that exits 1 | `BUILD_FAILED` |
| `hang` | a server that reports ready and then never answers | the judge ends it by itself as a failed run (`TIMEOUT`, `HUNG`, `CANARY_FAILED`, `INVALID` or `BUILD_FAILED`) and removes the container. This case waits on the judge's own deadline, at most `budgets.run_s`. |
| `same` | the incumbent again under a new id | `REJECTED`, `REJECTED_SCREEN` or `INCONCLUSIVE`, and never `PROMOTED`. A funnel that promoted a copy of the reference would fill the leaderboard with noise. |

Options: `--quick` leaves out `same`, a full measured run. `--only hang,same` runs only the cases you name. `--no-clock` leaves the clock alone. `--keep` leaves the rehearsal's experiment directories in place. The script's own outer bound for the judge in each of `buildfail`, `hang` and `same` is the target's `budgets.build_s`, `health_s` and `run_s` added together plus 600 s, so allow for that in the worst case.

**If anything prints FAIL, do not start the loop.** A funnel that cannot end a broken experiment will spend the run sitting behind one.

The rehearsal cannot disturb the real run. Its verdicts go to a scratch state directory, not to the researcher's tallies, and its results move out of `/opt/bench/results` to `/var/log/inference-judge/rehearse-<utc>/`. On PASS it restarts the run clock with the same budget, so nothing before this point is charged. It leaves the clock alone if a research session has already run.

## Start the loop

```bash
systemctl enable --now inference-judge-runner inference-judge-driver inference-judge-heartbeat.timer
```

The three units now run unattended. They restart after a crash or a reboot, resume from the event log, and stop by themselves when the budget is spent. You can close the `ssh` session.

Two optional things to do before you start:

- **Send alerts to your phone.** In `/etc/inference-judge/loop.env`, set `IJ_ALERT` to an `http(s)` URL, for example `https://ntfy.sh/<a topic only you know>`. The driver posts an ntfy-style message there when it pauses or moves to the fallback model. Any other value is a file to append to; the default is `/home/driver/research-state/ALERTS.push`. Every alert is also appended to `ALERTS.md` in the state directory. After you edit `loop.env` on a running loop, restart the units.
- **See what the driver would do, spending nothing.** This prints the next action, the status page and the exact prompt it would send, on a copy of the state, and writes nothing real:

  ```bash
  sudo -u driver -H bash -c 'set -a; . /etc/inference-judge/loop.env; set +a; exec /opt/inference-judge/loop/run.sh --dry-run'
  ```

`research/README.md` describes the loop's policy in full. In short, the driver starts an agent session when something happened that the researcher should see: a verdict arrived, the queue ran low, an hour passed, or the run is about to end. A session is capped at `IJ_ITER_TIMEOUT_S` (3600 s by default). The runner keeps measuring while the researcher thinks, and it starts no run with less than 2700 s of budget left. In the last 1800 s the researcher gets a wrap-up session to write `SUMMARY.md`.

## Watch the loop

Two minutes, a few times a day, from a root shell on the box:

```bash
# one page: the clock, the budget left, the queue, the latest verdicts, the tallies
cat /home/driver/research-state/BRIEF.md

# are the units alive, and does anything need you?
systemctl is-active inference-judge-driver inference-judge-runner inference-judge-heartbeat.timer
cat /home/driver/research-state/heartbeat.json
cat /home/driver/research-state/ALERTS.md 2>/dev/null     # exists only if something needed you

# the driver's own log: one line per decision
tail -f /home/driver/research-state/driver.log
journalctl -u inference-judge-runner -n 30 --no-pager

# what the researcher thinks it is doing
tail -40 /home/researcher/research/LAB_NOTEBOOK.md
cat /home/researcher/research/PLAN.md

# the judge's verdicts, independent of the notebook
for f in /opt/bench/results/*/results.json; do jq -r '[.id, .verdict] | @tsv' "$f"; done
```

`BRIEF.md` is the page the driver also hands the researcher at the start of every session, so you and the researcher read the same status.

Two checks are worth doing by eye once or twice. First, the notebook's predictions are written before the results: read one entry. Second, the numbers the researcher quotes match `/opt/bench/results/*/results.json`. The results files are the judge's originals, and the notebook is the researcher's copy.

Many rejections are normal and are the point of the gate. If the researcher reports nothing but promotions, suspect the gate or the reference, not the researcher.

**If `ALERTS.md` says the driver is paused.** The driver never exits on trouble. After 3 crashed sessions in a row it moves from the primary to the fallback model for 3 sessions, and if those also fail it pauses and raises an alert. With Claude Code the two are `opus` and `sonnet`. With Codex and Pi both words mean the model the tool is configured for (see [Known limits](#known-limits)). The runner keeps measuring what is already queued. The driver also pauses, together with the runner, if a file under one of the frozen paths changes. Those are `/opt/inference-judge` and, in the research repo, `loop/`, `guard/`, `adapters/` and the agent's settings folder (`.claude/` or `.codex/`); `IJ_FROZEN_PATHS` in `loop.env` lists them. Read the alert and fix the cause. Then resume with:

```bash
sudo -u driver touch /home/driver/research-state/RESUME
```

## Stop the loop

- **Normal end.** The driver stops itself, with exit status 10, when the accelerator-hour budget or the wall-clock deadline is reached, whichever comes first. Systemd does not restart it on 10. If the researcher wrote no `SUMMARY.md`, the driver leaves a stub, `/home/driver/research-state/SUMMARY.driver.md`.
- **Early stop.**

  ```bash
  systemctl disable --now inference-judge-driver inference-judge-runner inference-judge-heartbeat.timer
  ```

  Systemd waits up to 90 s for the driver and up to 300 s for the runner before it kills them. If `docker ps` still shows a measurement container afterwards, `docker rm -f <name>` ends it. Use `systemctl stop` instead of `disable --now` if you want the units back at the next boot.

An early stop spends the clock as time passes, not as work is done. The clock does not pause while the units are stopped. Stopping early and resuming later does not give back the hours in between.

## Take the results away

Do this before you delete the machine. A TPU VM's boot disk is deleted with the VM, and the other paths lose their disk with the machine too. Nothing here is automatic apart from the TPU spend guard's save to the cache disk.

Copy these paths. The directories under `/opt/bench` hold the evidence:

| path | what it holds |
|---|---|
| `/opt/bench/results/` | one directory per measured id: `results.json` (the verdict, the reasons, the metric, the stamps and the isolation record), `summary.txt`, `run.log`, `launches/`, and `snapshot/` |
| `/opt/bench/leaderboard/`, `calibration/`, `canary/`, `reference/` | the current best and its history, the noise estimate, and the stored references |
| `/home/researcher/research/` | the researcher's repository with its git history: experiments, `LAB_NOTEBOOK.md`, `PLAN.md`, `HYPOTHESES.jsonl` and `SUMMARY.md` |
| `/home/driver/research-state/` | `events.jsonl`, `runs.jsonl`, `BRIEF.md`, `driver.log`, `iterations/<n>.json` (the raw result of each session), `runs/<id>.log` and `ALERTS.md` |
| `/var/log/inference-judge/` | the logs of bootstrap, the baseline, the calibration and the rehearsals |

Run this on your laptop. It keeps each path's full name under the destination:

```bash
dest=~/inference-judge-run && mkdir -p "$dest"
for d in /opt/bench/results /opt/bench/leaderboard /opt/bench/calibration /opt/bench/canary \
         /opt/bench/reference /home/researcher/research /home/driver/research-state \
         /var/log/inference-judge; do
  rsync -azR "root@<box>:$d/" "$dest/"
done
```

On a TPU VM, `sudo bash tools/tpu_deadman.sh save` also copies the run to the cache disk, which outlives the VM. The copy lands in `<mount>/inference-judge-saved/<node>-<time of first arm>/`, readable by root only. To get it back after the VM is gone, attach the disk to any VM, mount it and copy that directory off. The guard's default sources also include `/opt/bench/cache`, and it leaves out `*.safetensors` files.

To reproduce a winner somewhere else, you need `results/<id>/snapshot/`, which holds the experiment directory and, for each engine source tree, its commit, its diff against that commit and its list of untracked files, and `results/<id>/results.json`, which records the engine image digest and the stamps of the judge, target and model.

Then delete the machine with the tools of your cloud. That is what stops the billing.

## Read the summary

When the run ends you should find `/home/researcher/research/SUMMARY.md`. The researcher's standing instructions ask it to say:

- the best configuration against the reference, with confidence intervals;
- what it tried;
- which new ideas it did not reach;
- what it would do with more time.

Treat it as the researcher's account, and check it against the judge's records:

1. **Read the verdicts, not the prose.** A claim is real when `results/<id>/results.json` says `PROMOTED`, and the leaderboard in `/opt/bench/leaderboard/<target id>.json` names it. A gain the judge cannot tell from noise is `REJECTED`, and the best configuration stays as it was.
2. **Look for the interval.** If a quoted interval crosses zero, the honest reading is "no proven win", however good the point estimate looks.
3. **Check the numbers** the summary quotes against `results.json`.
4. **Check the reference.** Every comparison is against the reference you measured. If you changed the reference's launcher, say so in your own notes: the number is only meaningful alongside the flags that produced it.

If there is no `SUMMARY.md` and only `SUMMARY.driver.md`, the run ended before the researcher's wrap-up. Nothing is lost: `/home/driver/research-state/` and `/opt/bench/results/` still hold everything.

The verdicts you may see in `results.json`:

| verdict | meaning |
|---|---|
| `PROMOTED` | A real gain over the incumbent. The candidate becomes the new best. |
| `REJECTED` | The gain is below the minimum detectable effect, or the run failed the quality gate or a guard. |
| `REJECTED_SCREEN` | The first, single launch already showed no gain, so the run stopped early. |
| `INCONCLUSIVE` | No calibration or incumbent exists yet. The best stays. |
| `VOID_DRIFT` | The machine drifted during the run, so there is no judgement. |
| `INVALID` | The measurement was not usable. For example the judge's stamps differ from the calibration's or the incumbent's (`ruler_mismatch`), or the server returned invalid requests, text outside the sanity band, or stalled requests. |
| `CANARY_FAILED` | The server's answers to the ten fixed prompts changed. |
| `BUILD_FAILED`, `TIMEOUT`, `HUNG` | The build failed, the run's time limit was reached, or a sweep outlived its bound. The container was killed. |
| `PREFLIGHT_FAILED` | The runner refused the experiment before the judge. It cost no slot. |
| `BUSY` | A second measurement started while one was running. |
| `INSTRUMENT` | An unscored profiling run. It never reaches the leaderboard. |
| `BASELINE`, `CALIBRATED` | Your own operator runs. |
| `REJECTED_RESERVED_ID` | A `baseline-*` id without `.allow-baseline`. |
| `INFRASTRUCTURE_FAILURE`, `NO_RESULT` | The measurement itself broke. `NO_RESULT` is the runner's note that `run.sh` left no `results.json`. |

`run.sh` exits 0 for every verdict. Exit code 2 is a usage error, and 70 is an infrastructure fault, including a judge that no longer matches `RULER_LOCK`. The reasons for a verdict are in `results.json` under `reasons`.

## If bootstrap fails

Fix the cause and run the same command again, with the same `--target` and the same engine tags. Every step is resumable. A failed critical step aborts at once, so the clock never starts on top of a broken install. Find the failing step in the checklist, or the message above it, and use the table. Exit code 3 is a deliberate stop on a `runnable: false` target, not a failure.

| message or step | what it means | what to do |
|---|---|---|
| `--target is required` | No target was named. | Add `--target targets/<id>.yaml`. The message lists the targets in the checkout. |
| `bench/target.py needs PyYAML` | The machine's `python3` has no YAML module. | `apt-get install -y python3-yaml`, then run again. On your laptop, put `bench/.venv/bin` first on `PATH` as in [Read the plan](#read-the-plan). |
| `target ... does not validate` or a message about a field | The target file is missing a field or has a value the judge does not know. | `bench/.venv/bin/python bench/target.py validate targets/<id>.yaml` prints the same message. Fix the file. |
| `IJ_VLLM_TAG is required`, or `... is not a release tag` | The engine is built on the box and its release tag is missing or is not a release tag. | Name the tags on the command line: `IJ_VLLM_TAG=<tag> IJ_SGLANG_TAG=<tag> bash setup/bootstrap.sh --target ...`. Use release tags only, spelled exactly, with the leading `v`. |
| `must run as root` | You ran it as another user. | Run it as root. (`--plan` is the only mode that works as any user.) |
| `need >=NG free on /` | Not enough disk for this target. | Free space or use a bigger disk. The figure comes from `provision.min_free_gb`, or from a default for the accelerator kind. |
| `IJ_REPO=... does not look like the repo` | Bootstrap was run from outside a full checkout. | Run it from the checkout, or set `IJ_REPO`. Check that `bench/` and `research/` were copied. |
| `target ... has no baseline: research/experiments/<name>/launch.sh is not in the repo` | The reference launcher is missing. | Add the launcher, or name another with `provision.baseline`. |
| `docker missing`, `docker daemon not usable` | Docker is not installed or not running. | Install it, run `systemctl restart docker` and retry. |
| `gpu: nvidia-smi works ...` or `nvidia-smi missing` | The driver is not loaded, or the GPU count, type or memory does not match the target. | The message says which. Reboot if the driver is not loaded. If the count is wrong, use a machine with exactly `accelerator.chips` GPUs, or a target written for this one. |
| a driver older than r580, for a built engine | The engine wheels need CUDA 13. | Install driver r580 or newer. Retrying on the old driver cannot work. |
| `docker cannot start a container with the GPU` | The NVIDIA container toolkit is missing or not configured. | `nvidia-ctk runtime configure --runtime=docker`, then `systemctl restart docker`. |
| `gpu gate FAILED` | The engine image did not start with the GPU, or sees fewer GPUs than the target wants. | Read the lines under it. The usual causes are a container toolkit that is not configured, a driver too old for the image, an image entrypoint that `engine.docker_flags` does not clear (`--entrypoint=` clears one), or flags that hide a GPU. |
| `the host shows N chip(s) ... target wants M` (TPU) | Wrong accelerator type for this VM, or a broken node. | Delete the node and create it again. |
| `no disk with the ext4 label '...' is attached to this VM` | The cache disk is not attached. | Attach it with your cloud tools. The label must be the one in `provision.cache_disk.label`. The message prints a command you can adapt. |
| `... is not at the pinned digest after load and pull` | The disk's image tar is missing or damaged, and the pull did not give the pinned digest either. | Check that `engine.image` and the disk's tar agree. |
| `chip gate: the node is healthy but the target's launch flags cannot reach the chips` | The node works, and the target's flags do not. | Do not recreate the node. Run the `tools/tpu_seal_probe.sh` command the message prints, then fix `engine.docker_flags` and `engine.env`. |
| `chip gate FAILED ... DELETE AND RECREATE THE NODE` | The node cannot start a TPU session with any flags. | Delete the node, create another, attach the disk and run again. |
| `model.revision ... is not a 40-hex commit` | The target does not pin the model revision. | Put the 40-hex commit in `model.revision`. The message tells you how to find it. |
| a weight file fails the hash check, or `blob ... is missing` | The download is damaged, or the disk's Hub cache is. | Run again; the download resumes. On a TPU, replace the damaged file in the disk's cache from a working machine. |
| `engine.sources names ...` or `provision.sources.<name>.ref is missing` | A source tree has no pinned `ref` or `url`. | Add them to the target. A `ref` is a 40-hex commit or a release tag, never a branch. The message names the field. |
| `the image has no <root>/<name>` (TPU) | A name in `engine.sources` is not in the image. | Fix `engine.sources`, or `provision.image_src_root`. `--reinstall-engines` copies the trees out again and discards edits. |
| `could not fetch <ref> from <url>` | GitHub rate limit, or the ref is not a commit of that repository. | Run again. Check the ref. |
| `bench cannot read ...` | The `bench` user cannot traverse `/home/researcher`. | Re-run after the harden step, or read the message for the path. `check.sh` tests this with a real `sudo -u bench test -r`. |
| the harden step: driver user, frozen loop, sudoers, units | A half-installed user, or a rejected sudoers file. | Run again once. If it repeats, `visudo -c` names the bad file. Do not start the loop on a box where this step failed: the separation of the three users is what it installs. |
| `the bench venv cannot import what the ruler needs`, or `inference-judge-ruler:latest cannot import what the ruler needs` | A dependency is missing from the judge's environment. | The message names the file to add it to (`bench/pyproject.toml` for the venv, the ruler image's Dockerfile for the image). Copy the checkout again and re-run. |
| the clock and `RULER_LOCK` step | The driver's user could not start the clock or lock the judge. | `bash setup/check.sh` shows which of the three users or sudo rules is wrong. |
| every later run ends at once with exit code 70 and "ruler" in the message | `bench/` or the target changed after the clock started. | Put the original files back and run bootstrap again, or accept a new run with `--reset-clock`, which loses the hours already run. |
| apt, Node or network errors in a package step | A mirror was unreachable for a moment. | Run again. A failure in `nsight-systems` is only a WARN. |

If you have to stop and think, delete or stop the machine first and start again later. A failed install on a rented accelerator still costs money by the hour.

## Known limits

- **A `gpu` target gets every GPU.** The judge starts every launch with `--gpus all`. Bootstrap refuses a machine whose GPU count differs from `accelerator.chips` for that reason.
- **`deps.txt` goes to `/opt/venv/vllm` first.** An experiment's `deps.txt` is installed there when the image has it, and into the system Python otherwise. It is installed at build time, because the launch has no network.
- **The canary sends `reasoning_effort: low` by default** (`CANARY_REASONING_EFFORT` in `bench/harness.py`). A model that ignores that field is compared at its own default.
- **Codex and Pi have each run one short real session on a laptop, and neither has run a real loop.** See [Choose the agent](#choose-the-agent) for what was seen and what is unseen. Claude Code has run a real loop only from before the loop called it through `session.sh`; a real session through its adapter did not get past an expired login.
- **The Bash guard assumes the judge is installed at `/opt/bench`.**
- **The laptop rehearsal runs the Claude adapter only, with a stand-in agent.**
- **With Codex or Pi, the driver's fallback step repeats the same model.** The adapters map `primary` and `fallback` to whatever model the tool is configured for. The step runs that model again for `IJ_FALLBACK_SESSIONS` sessions (default 3) before the driver pauses. Set `IJ_FALLBACK_MODEL` to another model name for a real second model, or `IJ_FALLBACK_SESSIONS=0` to skip the step. `research/adapters/README.md` explains this.
- **The laptop rehearsal's quick tier uses a stand-in judge.** Only `LOCALBOX_TIER=2` runs the real one.
- **The TPU and GPU paths have not completed a run on real hardware.** See [A TPU VM](#a-tpu-vm) and [An NVIDIA GPU machine](#an-nvidia-gpu-machine) for what exactly has and has not been run.
- **Some target keys are accepted and not read.** `your-first-target.md` lists them.
- **The guard is a tripwire.** The scripts in `research/guard/` catch an agent that drifts and explain the rule; they do not stop a process the agent starts. The boundary is the Unix ownership that `setup/harden_research.sh` sets. `setup/check.sh` confirms it on an installed box.
