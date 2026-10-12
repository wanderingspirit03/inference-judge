# research/: the researcher's side of the loop

This directory is the repository that the researcher (a coding agent, Claude Code by default, running as the Unix user `researcher`) works in. On a box it is installed at `/home/researcher/research`. Everything in it that is not frozen is the researcher's to change. Nothing in it can measure anything: measurement belongs to the judge in `bench/`, which runs as another user. (The code calls the judge "the ruler"; this page says "the judge" except where it names code.)

Read this page to learn how the loop is built, what the researcher may and may not do, and how to run the loop's tests. To install a box and start a run, use [`docs/RUNBOOK.md`](../docs/RUNBOOK.md). For the reasoning behind the design, see [`docs/DESIGN.md`](../docs/DESIGN.md); code comments cite its sections as "DESIGN 2.1" to "DESIGN 2.7".

## How the loop works

Three Unix users take part:

| user | what it runs | what it owns |
|---|---|---|
| `researcher` | the researcher: the agent's sessions, started through `adapters/<agent>/session.sh` | the research repo (this directory) and the engine source checkouts |
| `driver` | `loop/run.sh` (the driver), `loop/runner.sh` (the runner) and `loop/heartbeat.sh` | the state directory (`~/research-state` by default, so `/home/driver/research-state` on a box) |
| `bench` | `bench/run.sh` (the judge) | `/opt/bench`, the accelerators and the results |

```
 researcher (the agent)            driver                              bench (the judge)
 -----------------------            ------                              -----------------
 reads BRIEF.md           <------   loop/run.sh: decides when to
 writes experiments/<id>/            start an agent session, writes
 writes HYPOTHESES.jsonl             BRIEF.md and the prompt
 runs loop/preflight.sh <id>
 writes queue/<id>.json   ------>   loop/runner.sh: reads the queue,
                                     pre-flights, then calls ...   --->  /opt/bench/run.sh [--instrument] <id>
                                                                                       |
 reads the verdict        <---------  results/<id>/results.json  <---------------------+
 in results.json
```

The key property: the researcher never calls the judge. It writes a request into `queue/`, and the runner, running as `driver`, calls the judge. The runner is a separate process from the driver's sessions, so the judge keeps measuring while the researcher thinks.

The driver and the runner share one append-only log, `events.jsonl`, in the state directory. A single module, `loop/loopstate.py`, folds that log into the current state (tallies, the budget clock, crash recovery, the queue view and `BRIEF.md`). Nothing else is state.

### The queue

To ask for a measurement, the researcher writes `queue/<id>.json`:

```json
{"id": "<id>", "kind": "candidate"}
```

`kind` is `candidate` (a scored comparison against the current best) or `instrument` (a profiling run that is never scored). The runner reads only those two fields; anything else in the file is ignored. `loop/loopstate.py` (`read_queue`) refuses a file when:

- the file name is not `<id>.json`, or `id` is missing, differs from the file name, or does not match `^[a-z0-9][a-z0-9-]{2,40}$`;
- the id starts with `baseline-` (reserved for the operator's baseline runs);
- the file is a symlink, is not a regular file, is larger than 16 KiB, or is not a JSON object;
- `kind` is neither `candidate` nor `instrument`;
- a results directory with that id already exists;
- the kind is `candidate` and `HYPOTHESES.jsonl` has no line that names the experiment and carries a number (see below).

Accepted files run first in, first out, one at a time. An id that has already run is skipped.

**Pre-registration.** A candidate needs a line in `HYPOTHESES.jsonl` before it can run:

```json
{"experiment": "<id>", "hypothesis": "...", "mechanism": "...", "prediction": 4.0, "unit": "..."}
```

`experiment` may also be a list under `experiments`, and the number may sit in `prediction` or `predicted`. A prediction that is not a number is refused. `HYPOTHESES.jsonl` is the machine-read record that the researcher writes at run time; it is not shipped in this directory. `HYPOTHESES.md` is the separate, human-readable backlog.

### Pre-flight

`loop/preflight.sh <id>` checks an experiment on the CPU in seconds, so a mistake does not cost a judge slot. It checks that:

- the id is well formed, is not reserved and has not been used (no results directory, no verdict in the driver's ledger);
- `experiments/<id>/launch.sh` exists and passes `bash -n` and, when `shellcheck` is installed, `shellcheck -S error`;
- every `experiments/<id>/patches/*.diff` or `*.patch` applies (`git apply --check`) to at least one engine checkout (the trees listed in `IJ_PATCH_TREES`, which by default come from `/etc/inference-judge/engine-sources`);
- `experiments/<id>/deps.txt`, if present, parses as pip requirements.

The last line it prints is `PREFLIGHT_OK <id>` or `PREFLIGHT_FAILED <id>: <reasons>`. The exit status is 0, 1 or 2 (bad usage). The runner runs pre-flight again before it calls the judge. A failure is logged as a `PREFLIGHT_FAILED` verdict, uses no judge slot, and is not retried until the researcher edits the queue file.

### The runner

`loop/runner.sh` runs `loop/runner.py` under `python3 -I`. For each accepted queue item it:

1. pre-flights the experiment, as the researcher;
2. calls the judge as an argument list (never a shell string): `$IJ_BENCH_CMD <id>` for a candidate, `$IJ_BENCH_CMD --instrument <id>` for an instrument run. `IJ_BENCH_CMD` defaults to `sudo -n -u bench /opt/bench/run.sh`;
3. reads the verdict from `results/<id>/results.json` and appends it to `events.jsonl`.

It starts no run with less than 2700 s (`run_min_s`) of the budget left. `loop/runner.sh --once` handles at most one item and exits.

### The driver

`loop/run.sh` owns the clock and the recovery policy. On each pass it asks `loopstate.py` what to do next. The answer is one of: start a session, sleep, or finish. It then runs one agent session through the agent's adapter (see [The agent adapter](#the-agent-adapter)) and classifies how the session ended: ok, timeout, quota limit, authentication failure or crash.

A session starts when something happened that the researcher should see. The reason is recorded and shown in the prompt:

| reason | when |
|---|---|
| `first` | no session has run yet |
| `wrap_up` | the last 1800 s (`summary_lead_s`) of the run, if `SUMMARY.md` does not exist |
| `verdict` | a verdict arrived since the last `BRIEF.md` |
| `queue_low` | fewer than 2 experiments (`queue_target`) are queued; the gap between such sessions grows while the researcher leaves the queue low |
| `timer` | 3600 s (`timer_s`) since the last session |
| `retry_after_crash` | a crash back-off ended |
| `auth_probe` | the driver is paused on authentication and retries every 1800 s |

Before every session the driver writes `BRIEF.md` (the clock, the budget left, the queue, the latest verdicts and the tallies) and `prompt.txt` into the state directory. The researcher reads `BRIEF.md` first. A session is capped at 3600 s (`iter_timeout_s`) or the time left in the run, whichever is shorter, and at least 60 s apart from the previous one (`min_gap_s`).

Recovery, with the default policy:

- **Crash.** The driver never exits on a crash. It backs off (60 s, then 120 s, 240 s and so on, up to 1800 s). After 3 consecutive crashes (`max_crashes`) it switches from the primary model to the fallback model for 3 sessions (`fallback_sessions`). The two models are the words `primary` and `fallback`; the agent's adapter decides which model each word means. If the fallback also fails, it pauses and raises an alert while the runner keeps measuring.
- **Quota limit.** A limit is not a crash. The driver sleeps until the reset time that the result advertises, or 10, 20, then 40 minutes, never more than an hour. It raises one alert after 6 limits in a row.
- **Authentication failure.** The driver pauses at once, raises an alert, and probes again every 1800 s.
- **Frozen files.** If a file under `loop/`, `guard/`, `adapters/` or the agent's settings folder (`.claude/` or `.codex/`) changes during the run, the driver raises an alert and pauses sessions until `RESUME`. The runner starts no run during that pause.
- **Resume.** `./loop/run.sh --resume`, or `touch RESUME` in the state directory, clears a pause.

The budget is accelerator-hours (charged while the box is up) plus a wall-clock deadline. The run ends at whichever comes first. `./loop/run.sh --start-clock` starts the clock; its defaults are `--accel-hours 24` and `--wall-hours 24`. When the budget is spent the driver reaps the researcher's detached jobs, writes `SUMMARY.driver.md` into the state directory if the researcher wrote no `SUMMARY.md`, and exits with status 10 (the systemd unit does not restart on 10).

Alerts are appended to `ALERTS.md` in the state directory. If `IJ_ALERT` is set to an `http(s)` URL, each alert is also sent as an ntfy-style POST; any other value is treated as a file to append to.

The driver's other state files are `events.jsonl` (the log), `runs.jsonl` and `process.jsonl` (ledgers derived from it), `heartbeat.json`, `driver.log`, `iterations/<n>.json` (the adapter's result object for session `n`) and `runs/<id>.log` (the judge's output for each run).

`loop/heartbeat.sh` writes `heartbeat.json` once (systemd runs it every minute from `inference-judge-heartbeat.timer`): whether the driver is alive, the runner's state, accelerator hours used, the end of the run, disk use and the last verdict. It also charges accelerator time while the driver is down, and alerts once if the driver looks wedged.

Run the driver in the foreground with:

```bash
./loop/run.sh --dry-run     # on a copy of the state directory: print the next action, BRIEF.md and the prompt, and self-test the classifier; starts no agent
./loop/run.sh               # the real thing
```

On a box, `setup/harden_research.sh` installs the systemd units (`inference-judge-driver.service`, `inference-judge-runner.service`, `inference-judge-heartbeat.service` and `.timer`) and `/etc/inference-judge/loop.env`; the units are installed but not started. Start them with `systemctl enable --now inference-judge-runner inference-judge-driver inference-judge-heartbeat.timer`. The full environment-variable list is in `./loop/run.sh --help`.

### The agent adapter

The loop does not know which coding agent does the research. Choose it with `IJ_AGENT` (default `claude`), or with `--agent <name>` when you run `setup/bootstrap.sh`. Three adapters ship: `claude` (Claude Code), `codex` (Codex CLI) and `pi` (Pi). `adapters/TEMPLATE/` is a starting point for your own.

The driver runs `adapters/$IJ_AGENT/session.sh run --model M --effort E` once per session, with the prompt on stdin, as the researcher's Unix user, in the research repo. The adapter prints one JSON object, which the driver reads. `IJ_AGENT_BIN` names the agent's command when it is not on the PATH that `sudo` allows. `loop/run.sh` stops at once, `--dry-run` included, when `adapters/$IJ_AGENT/session.sh` is missing or not executable. When a session is due, `./loop/run.sh --dry-run` prints the command line it would run.

`M` is `primary` or `fallback`, and each adapter decides which model each word means. Claude's adapter maps them to `opus` and `sonnet`. The Codex and Pi adapters use the model the tool is configured for, for both words. The adapters' README, [`adapters/README.md`](adapters/README.md), has what `session.sh` must do, how the guard reaches each agent, and how to write your own.

What is proven where: Claude Code is the only agent that has run a real loop, and that was before the loop called it through `session.sh`. Through the adapter it has been tested only with a stand-in program; a real session on a laptop did not get past an expired login. The Codex and Pi adapters were written from those tools' documentation and tested with stand-in programs. Each has since run one short real session on a laptop through its adapter. Neither has run a real loop. [`adapters/README.md`](adapters/README.md) says what that session showed and what is still unseen.

### The guard

`guard/` holds three scripts. Each reads the tool call as JSON on stdin; exit status 2 blocks the call and shows the message on stderr to the researcher. Claude Code and Codex call them as hooks. Pi calls them from the extension `adapters/pi/guard.ts`. [`adapters/README.md`](adapters/README.md#the-guard) says how each agent is wired to them.

- `pretooluse-edit.sh` (before a file edit: Edit, Write, MultiEdit and NotebookEdit in Claude Code, `apply_patch` in Codex, `write` and `edit` in Pi): allows writes only inside the research repo and the engine source checkouts the target lists, and refuses the frozen paths (`loop/`, `guard/`, `adapters/`, the agent's settings folder and the driver's state directory). Reading is not restricted.
- `pretooluse-bash.sh` (before a shell command): refuses `nvidia-smi` and `docker` as whole words, and passes every command through `_guard.py`, which refuses accelerator access (`nvidia-smi` or `tpu-info` under any spelling, `/dev/nvidia*`, `/dev/accel*`, `/dev/vfio`, and host Python importing `torch`, `jax`, `libtpu`, `vllm` or `sglang`) and write-shaped commands on frozen paths. For a command that mentions `/opt/bench`, it allows only read-only commands on `/opt/bench/results/`, so the researcher can read its own verdicts, and refuses writes and every other path. It splits the command with a small `python3` pass that treats `|`, `;`, `&`, `<`, `>` and `$(` as operators only when unquoted. If `python3` is missing, the script refuses any command that mentions `/opt/bench`.
- `posttooluse-audit.sh` (after any tool call): appends one line to `audit.log` for each tool call that ran; a call the guard refused leaves no line. It never blocks.

The scripts find the research repo and the frozen paths in `/etc/inference-judge/guard.env`, which `setup/harden_research.sh` writes. A variable already in the environment (`IJ_RESEARCH_DIR`, `IJ_PROTECTED_ROOTS`) wins over the file. With neither, the research repo is `CLAUDE_PROJECT_DIR` when Claude Code sets it and otherwise the folder that contains `guard/`, and `loop/`, `guard/`, `adapters/`, `.claude/`, `.codex/` and the driver's state directory are frozen.

**The guard is a tripwire, not the boundary.** It reads a command line; it does not see what a script does once it runs, and any process the researcher starts can get around it. It exists to turn a mistake into a clear message instead of a confusing permission error. The boundary is Unix permissions, set by `setup/harden_research.sh`:

- `loop/`, `guard/`, `adapters/` and the agent's settings folder (`.claude/` or `.codex/`) in the research repo become root-owned and read-only. Root-owned copies of `loop/`, `guard/` and `adapters/` are installed in `/opt/inference-judge`;
- the agent's managed settings are written when its adapter has any (Claude Code: `/etc/claude-code/managed-settings.json`; Codex: `/etc/codex/requirements.toml`; Pi has none), and they load only the root-owned copy of the guard (`allowManagedHooksOnly` in Claude Code, `allow_managed_hooks_only` in Codex);
- the state directory belongs to `driver` (mode 0755: the researcher can read `BRIEF.md` and the ledgers but cannot write there);
- the accelerator devices belong to group `bench`;
- the researcher gets a CPU-only virtual environment;
- any `sudoers` rule that would let `researcher` call `/opt/bench/run.sh` is disabled (unless `--keep-direct-bench` is passed), so measurement goes through the queue only.

`pretooluse-bash.sh` still has a pattern that allows the single command `sudo -u bench /opt/bench/run.sh <id>` (also under `timeout`, or piped to `head`, `tail` or `grep`). After hardening, `researcher` has no `sudoers` right to run it, and `setup/check.sh` fails if one remains.

### The helper agents (Claude Code only)

`adapters/claude/project/.claude/agents/` holds three subagent prompts that Claude Code can call: `scout.md`, `hypothesizer.md` and `sceptic.md`. Setup copies them to `.claude/agents/` in the research repo, which is frozen for the run. Each runs at `effort: high`. They are a worked example of a design with several agents. The Codex and Pi adapters ship none on purpose: an adapter is the place to build your own design with several agents, because the loop reads only the one result object and the files the session leaves.

- `scout` (model `sonnet`): reads server logs, `results.json`, profile summaries, engine source, documentation and web pages, and returns a short factual summary with `file:line` references. It proposes nothing and ranks nothing. The researcher uses it so that its own context goes to decisions.
- `hypothesizer` (model `sonnet`): proposes changes that could make the target faster, each with a mechanism and a numeric prediction. It is a template: copy it and narrow it to one angle (for example scheduling or kernels) to get generators that disagree with each other.
- `sceptic` (model `opus`): takes a list of hypotheses and returns them re-ranked with the strongest objection to each. The researcher answers that objection in the notebook before the experiment runs.

The models named here are set in the helper agents' own files. The driver's sessions ask only for `primary` or `fallback`, at the effort set by `IJ_EFFORT` (`high` by default); `IJ_PRIMARY_MODEL` and `IJ_FALLBACK_MODEL` send a model name instead.

## What the researcher may and may not do

The standing instructions, the same for every agent, are in [`AGENTS.md`](AGENTS.md). Claude Code's `CLAUDE.md` is in `adapters/claude/project/` and imports that file with `@AGENTS.md`; the driver's prompt names `AGENTS.md` too. Neither file is frozen.

The researcher may:

- change anything in this repository that is not frozen, any engine checkout in its home directory, any dependency, and read anything online;
- write `experiments/<id>/`, the notebook, the backlog, the plan and `queue/<id>.json`;
- read `BRIEF.md` and the state directory, and read `/opt/bench/results/**`;
- start long CPU-only jobs that must outlive a session with `loop/bg.sh <name> <command...>`. The job runs in its own process group, with a log under `$IJ_JOBS_SPOOL` (default `~/.inference-judge-jobs`). The driver registers it and kills it at the end of the run.

The researcher may not:

- call the judge, or read, list or write `/opt/bench`, other than reading `/opt/bench/results/**`;
- reach an accelerator by any route, or run `docker`;
- edit `loop/`, `guard/`, `adapters/`, the agent's settings folder (`.claude/` or `.codex/`) or the state directory;
- special-case the judge's prompts, change outputs, or work around the judge. A faster wrong answer is a rejection.

The researcher's working loop, in order: read `BRIEF.md`; write the hypothesis (a `HYPOTHESES.jsonl` line and a notebook entry) before the run; build `experiments/<id>/launch.sh`; run `loop/preflight.sh <id>`; write `queue/<id>.json`; record the verdict in the notebook when it lands; keep about two experiments queued; and leave `NEXT.md` (three lines: what it queued, what it is waiting for, what it would do next). Near the end it confirms its best configuration and writes `SUMMARY.md`.

## Files

| path | what it is |
|---|---|
| `AGENTS.md` | the standing instructions for the researcher, whichever agent it is. Claude Code's `CLAUDE.md` is in `adapters/claude/project/`. |
| `LAB_NOTEBOOK.md` | one entry per experiment: hypothesis, mechanism and prediction, written before the run; then the measured result and verdict. Ships with one entry marked EXAMPLE. |
| `LEADERBOARD.md` | the best promoted configurations. Numbers come only from `results.json`. Ships with one row marked EXAMPLE. |
| `HYPOTHESES.md` | the human-readable backlog, appended to by the researcher, the literature pass and the subagents. Ships with one entry marked EXAMPLE. |
| `PLAN.md` | the researcher's intent against the clock, and what changed. |
| `NEXT.md` | written by the researcher at the end of each session. Not shipped. |
| `SUMMARY.md` | written by the researcher near the end of the run. Not shipped. |
| `HYPOTHESES.jsonl` | the pre-registration lines that the queue checks. Written by the researcher. Not shipped. |
| `experiments/TEMPLATE/launch.sh` | the commented starting point for a new experiment (next section). |
| `experiments/baseline-vllm/`, `baseline-sglang/`, `baseline-tpu/`, `baseline-mock/` | the plain reference launchers for each kind of target. Do not tune them: every promotion is measured against the baseline. New ideas go in a new directory. |
| `experiments/profile-baseline/` | `baseline-vllm` under `nsys`, to be queued as an `instrument` run. Profiling perturbs the numbers, so its verdict is `INSTRUMENT`, not a measurement. |
| `profiles/` | profiler output. Empty until someone runs `profile-baseline`; `profiles/README.md` says what lands there. |
| `guard/` | the guard: `pretooluse-bash.sh`, `pretooluse-edit.sh`, `posttooluse-audit.sh`, their helpers and their tests (`test-hooks.sh`). |
| `adapters/` | one folder per agent (`claude/`, `codex/`, `pi/`), `TEMPLATE/` and `common.sh`. See [`adapters/README.md`](adapters/README.md). |
| `loop/` | the driver, runner, pre-flight, heartbeat, `bg.sh`, `literature.sh` and `loopstate.py`, plus the loop's tests in `loop/tests/`. |
| `loop/literature.sh` | the literature pass: a session of the agent, started through its adapter, that reads, runs no experiments and appends hypotheses to `HYPOTHESES.md`. `./loop/run.sh --install-cron` adds it to the crontab of the user it runs as (daily at 00:00 UTC). |
| `loop/state.json` | not shipped. If this file exists and holds a `clock_start_epoch`, the driver takes the clock from it once. Bootstrap starts the clock with `loopstate.py start-clock` instead. The driver's own state is `events.jsonl` in the state directory. |
| `queue/` | the queue directory. Created at run time. |
| `audit.log` | every tool call that ran, appended by `guard/posttooluse-audit.sh`. |
| `.gitignore` | keeps machine output (`loop/*.log`, `audit.log`, profiler binaries, `queue/`) out of the researcher's commits. |

## The experiment folder

An experiment is a directory, `experiments/<id>/`, with `launch.sh` as its only required file. Copy `experiments/TEMPLATE/` and edit it; its header comments are the full reference. In short:

- **Around your script.** The judge snapshots `experiments/<id>/` and mounts it read-only. It builds `Dockerfile` if you wrote one, otherwise it uses the target's engine image. It bakes `deps.txt` into a derived image before the launch. It starts the container sealed (no network, every mount read-only except a private copy of the compile cache when the target has one), then copies the experiment to a writable `/exp` and the engine trees to `/src/<name>`, and runs `bash /exp/launch.sh`.
- **Your script must** serve OpenAI-compatible HTTP on `0.0.0.0:$PORT` under the model name `$IJ_SERVED_NAME`, with weights from `$IJ_MODEL_DIR` (read-only). `GET /v1/models` must return 200 when the server is ready. `/v1/completions` must support `echo` with `logprobs` (the quality gate teacher-forces), and `/v1/chat/completions` must stream with `usage` in the final chunk. The last line must be `exec <server>`, so the container ends when the server ends.
- **Optional files:** `Dockerfile`, `deps.txt` (pip requirements, installed into `/opt/venv/vllm` when the image has it and otherwise into the system Python; wheels must be reachable at build time because the launch has no network), `patches/*.diff` (the judge does not apply them; `launch.sh` applies them to `/src/<name>`) and `kernels/`.
- **Limits.** Nothing can be downloaded, and nothing written leaves the container in a scored run. Profiles go to `/out`, which exists only under `--instrument`. The target's `health.ready_timeout_s` (or `compile_s + health_s`) bounds start-up, and its `run_s` is the hard timeout.
- **An accelerator target gives the container every accelerator on the machine.** A `gpu` target is started with `--gpus all`.

## Running the loop's tests

The loop's tests use only the standard library and `pytest`, and never call a real agent or the real `/opt/bench`: they put stand-in `claude`, `codex` and `pi` commands first on the `PATH` and a stub judge behind `IJ_BENCH_CMD`. From the repository root, with the bench environment (see [`bench/README.md`](../bench/README.md)):

```bash
bench/.venv/bin/python -m pytest research/loop/tests
```

A few tests skip themselves when `shellcheck`, or Docker with an `ubuntu:24.04` image, is not available.

The guard tests need no box and no agent. They feed JSON payloads to the three guard scripts and check the exit status of each:

```bash
bash research/guard/test-hooks.sh
```

Run them after you change a guard script. `setup/check.sh` runs the same file on a box, as the `researcher` user.

## Known limits

- Codex and Pi have each run one short real session on a laptop, and neither has run a real loop. Unseen for both: a loop on a hardened machine, as the `researcher` user, with the managed settings file in place; and the wording of their quota and login errors. Unseen for Codex: that its hooks are enforced from `/etc/codex/requirements.toml`. Claude Code has run a real loop only from before the loop called it through `session.sh`; a real session through its adapter did not get past an expired login. It has not been checked that a Claude Code session without a person loads `AGENTS.md` through the `@AGENTS.md` import in `CLAUDE.md`; the driver's prompt names `AGENTS.md` as well.
- The laptop rehearsal (`setup/local_box/`) runs the Claude adapter only, with a stand-in agent.
- The guard compares paths as text, after tidying them. If `IJ_RESEARCH_DIR`, or the path the guard works out for the research repo, reaches the repo through a symbolic link, the agent reports the real path, the two do not match, and every edit is refused. This happened on a Mac under `/tmp`, which is a link to `/private/tmp`. Give the real path. The standard layout (`/home/researcher/research`) has no link.
- With Codex or Pi the driver's fallback step runs the same model again, unless `IJ_FALLBACK_MODEL` names another model or `IJ_FALLBACK_SESSIONS=0`. [`adapters/README.md`](adapters/README.md) explains why.
- The guard is a tripwire. The guarantee comes from the file ownership that `setup/harden_research.sh` sets; run `setup/check.sh` after installing a box to confirm it.
- The Bash guard assumes the judge is installed at `/opt/bench`.
- The literature pass (`loop/literature.sh`) has not been brought up to date with a hardened machine. It works on the repo it sits in and starts the agent as the user who runs it, so run it by hand as the researcher, from the research repo. After `setup/harden_research.sh` the researcher cannot write its default log in `loop/`; set `IJ_LITERATURE_LOG` to a file the researcher can write. `loop/run.sh --install-cron` has not been tried on a hardened machine.
- On a `gpu` target the judge gives the container every GPU on the machine (`--gpus all`).
