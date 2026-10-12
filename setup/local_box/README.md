# The local box

A laptop-sized rehearsal of a whole inference-judge box. `setup/bootstrap.sh` runs for `targets/cpu-mock.yaml` inside a fresh Ubuntu 24.04 container. The real driver, runner and heartbeat units then run under a systemd stand-in. A scripted stand-in for Claude Code's command, `claude`, plays the researcher: the driver reaches it through the Claude adapter (`research/adapters/claude/session.sh`), as the `researcher` user. No cloud, no keys, no model.

```bash
bash setup/local_box/run_local_box.sh                   # tier 1; exit 0 iff every check passed
LOCALBOX_TIER=2 bash setup/local_box/run_local_box.sh   # tier 2, the real judge
(cd bench && IJ_LOCALBOX=1 .venv/bin/python -m pytest -m slow ../setup/tests/test_local_box.py)
```

It needs Docker. Tier 2 takes much longer than tier 1, because it launches the mock server for real.

Flags of `run_local_box.sh`: `--keep` (leave the container up), `--clean` (also delete the image and the `inference-judge-localbox-docker` volume) and `--out DIR` (where the logs go; the default is `/tmp/inference-judge-localbox-<utc>`).

## Which tier uses which judge

The code calls the judge "the ruler". The two tiers differ in what stands in for it.

| tier | how to run it | the judge | what it covers |
|---|---|---|---|
| 1 (default) | `run_local_box.sh` | a stand-in: `stub_bench.sh` replaces `/opt/bench/run.sh` after bootstrap has installed and locked the real one. It sleeps a few seconds and returns a verdict chosen from the text of the experiment's `launch.sh`. | the install, the permissions, the queue, the driver, the runner, crash recovery |
| 2 | `LOCALBOX_TIER=2 run_local_box.sh` | the real `bench/run.sh`, with the sealed mock server, in the box's own Docker | the operator's steps, and the rehearsal judged by the real judge |

Tier 1 therefore proves the plumbing around the judge and says nothing about the judge's statistics. In particular, `setup/rehearse.sh --quick`, which tier 1 runs, is judged by the stand-in. Tier 2 runs `setup/rehearse.sh` with every case and sends the cases through the real `run.sh`.

## What tier 1 proves

Each item is one PASS line in the output.

- Bootstrap is target-driven: for `kind: none` there is no GPU, `nsys`, engine build or model step. Running it again resumes and keeps the clock.
- The three users exist and the driver has its sudoers rule. `/opt/inference-judge` (`loop`, `guard`, `adapters`) is frozen, and so are `loop/`, `guard/`, `adapters/` and `.claude/` in the research repo. `/etc/inference-judge/guard.env` is root's. A planted world-readable ruler file is locked to `bench`, and the device goes to group `bench`.
- The target is installed for the judge and for the researcher, and `RULER_LOCK` is written at clock start through the driver's sudo route.
- `setup/check.sh` passes. That includes the agent's version and login: the scenario gives the stand-in a login marker, `~/.claude.json` in the researcher's home, which the Claude adapter reads as "has logged in once".
- `setup/rehearse.sh --quick` passes through the queue and the installed runner, including the `driver` case (the dry run calls `adapters/claude/session.sh run` with `-u researcher`, `--model primary` and `--effort high`), leaves nothing in the real run's state, and restarts the clock. The scenario plants a stub `leaderboard/cpu-mock.json` beside the stand-in judge, because the rehearsal refuses a throughput target with no incumbent.
- The queue flow works: `PREFLIGHT_FAILED` takes no judge slot, runs go first in, first out, and each gets one verdict. `BRIEF.md` is rewritten between sessions, and the tallies equal the number of distinct run ids.
- 21 researcher-boundary probes fail and 6 legitimate actions succeed. The stand-in's break-in attempts aim at the judge's files, the installed and the repo copies of `loop/`, `guard/` and `adapters/`, the agent's settings, `guard.env`, the driver's state and `loop.env`, `sudo`, Docker and the accelerator device.
- Killing the driver with `SIGKILL` mid-session and mid-run, it resumes from `events.jsonl` without double counting.
- Crashes back off, fall back from `opus` to `sonnet` (the Claude adapter's names for `primary` and `fallback`), then pause with an alert, and the driver never exits. The runner keeps draining the queue meanwhile, by design: a session pause is not a judge pause.

## What tier 2 proves

Tier 2 does the same install, then the operator's part of the runbook against the real `bench/run.sh` and the sealed mock server:

- the reference run under a `baseline-*` id (`BASELINE`, which becomes the incumbent on the leaderboard); the scenario creates `/opt/bench/.allow-baseline` for that run, because `run.sh` accepts a `baseline-*` id only while that file exists;
- `--calibrate` refused through `sudo`, then accepted as `env -u SUDO_USER runuser -u bench` (`CALIBRATED`, with two launches instead of the usual ten, to keep the box short);
- `--dry-run --confirm` on the baseline, whose exit status is reported but not asserted;
- the whole rehearsal judged by the real judge: the build failure (`BUILD_FAILED`), a server that becomes ready and then never answers a request (ended by `run.sh` with one of the verdicts `setup/rehearse.sh` accepts for a wedged server: `TIMEOUT`, `HUNG`, `CANARY_FAILED`, `INVALID` or `BUILD_FAILED`, with its container removed), and a copy of the incumbent's `launch.sh` (not `PROMOTED`).

Tier 2 stops there. The driver phases (a session in flight, a driver kill, repeated crashes) are timed against a judge that answers in seconds, so they run in tier 1 only.

## The files

| file | role |
|---|---|
| `Dockerfile`, `box_init.sh` | the box: `ubuntu:24.04`, apt packages, `uv`, and its own `dockerd`, started by `box_init.sh` as PID 1 |
| `supervise.py` | runs the installed unit files (`User=`, `EnvironmentFile=`, `Restart=`, timers) |
| `stub_claude.py` | a stand-in for Claude Code's command `claude -p`, installed as `/usr/local/bin/claude` and reached through the Claude adapter; reads the mode file `/etc/inference-judge/localbox/mode` (`research`, `slow`, `quiet`, `crash`) |
| `stub_bench.sh` | tier 1 only: the stand-in judge described above |
| `scenario.py` | the stages and every check; `summary.json` lands in the log directory |
| `run_local_box.sh` | builds the image, starts the box, runs `scenario.py` inside it and copies the logs out |

## Why Docker-in-Docker

`bench/run.sh` bind-mounts box paths into the engine container and talks to it on the box's `127.0.0.1`. With the host's Docker socket, both would resolve inside the Docker Desktop virtual machine instead.

## What it cannot prove

- Any accelerator, driver, CUDA or TPU path.
- Real compile and build times.
- A real model, tokenizer or quality gate.
- Archiving to cloud storage.
- Any real coding agent. The stand-in is not Claude Code, so no agent calls the guard scripts in `research/guard/` as hooks here. `setup/check.sh` runs them itself, with sample tool calls. The probes are filesystem and `sudo` checks. The guard has its own tests: `bash research/guard/test-hooks.sh`.
- Codex and Pi. The rehearsal runs the Claude adapter only, and there is no option to rehearse another.
- The judge's statistics in tier 1, whose verdicts come from the stand-in. Tier 2 does not reach a `PROMOTED` run, and it covers `--confirm` only as a dry run (reported, not asserted); the tests in `bench/tests` cover those with `--local`.
