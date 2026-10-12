# Design

inference-judge lets a coding agent try to make a model server faster, and
decides with a sealed judge whether each speed-up is real. This document
describes how the judge decides, what it trusts, and what each interface
looks like. It describes the code as it is. Where a number is a default, the
file that sets it is named, so you can check it.

Read section 1 for the trust model. Read section 2 when you write a target,
read a result or change the loop. Read section 3 to see which cheats the
judge is tested against, and section 4 for what it does not cover.

The code calls the judge "the ruler". Names such as `RULER_LOCK` and
`ruler_hash` keep that word, and this document keeps them as they are.

## 1. What this is and what it trusts

### The idea

A researcher (a coding agent) proposes one change to how a model server is
built or started. The judge measures the changed server against the current
best server, and says `PROMOTED` only when the speed-up is larger than the
measured noise and the output has not drifted from a fixed reference.

The only metric is `tps`: output tokens per second at fixed input and output
lengths, per chip. The judge does not measure anything else, and a target
file cannot ask it to.

The researcher cannot edit the judge, cannot read the prompts it will send,
and cannot write its results. The separation is made of Unix users, not of
promises in a prompt.

### The three Unix users

| User | Role | What it owns or may do |
|---|---|---|
| `bench` | the judge | owns `/opt/bench` (`bench/` installed), the results, the leaderboard, the reference and the canary reference. The accelerator devices are in its group. Runs Docker. |
| `driver` | runs the loop | runs the driver, the runner and the heartbeat. Owns the state directory. May run `/opt/bench/run.sh` as `bench` and may run anything as `researcher`. |
| `researcher` | the researcher | writes `~/research` (`/home/researcher/research`): experiments, patches, the queue, `HYPOTHESES.jsonl`, the notebook. Runs the coding agent's sessions (Claude Code, Codex or Pi; see section 2.7). |

`setup/harden_research.sh` creates the `driver` user and its sudoers file
`/etc/sudoers.d/inference-judge-driver`. It contains two rules:

```
driver ALL=(bench) NOPASSWD: /opt/bench/run.sh
driver ALL=(researcher) NOPASSWD: ALL
```

The researcher has no sudo route to `run.sh`. It asks for a measurement by
writing a queue file; the runner, which runs as `driver`, calls the judge.
`setup/check.sh` verifies this layout without changing anything.

### What the researcher can and cannot touch

The researcher can:

- write `~/research/experiments/<id>/` (a `launch.sh`, optional `patches/`,
  `deps.txt` and `Dockerfile`) and edit the engine source trees the target
  lists in `engine.sources`;
- append to `HYPOTHESES.jsonl`, write `queue/<id>.json`, and keep the
  notebook;
- read the target file (`/etc/inference-judge/target.yaml`), `BRIEF.md` and
  the other ledgers in the driver's state directory, and every
  `/opt/bench/results/<id>/` (these are world-readable by design).

The researcher cannot:

- list `/opt/bench`, or read the harness, the prompt pools, the reference or
  the canary reference (the directories are `0750` and owned by `bench`);
- write the state directory (it belongs to `driver`), or the frozen code in
  the research repo: `loop/`, `guard/`, `adapters/` and the agent's own
  settings folder, `.claude/` for Claude Code or `.codex/` for Codex (Pi has
  none). These are root-owned and read-only. `setup/harden_research.sh` also
  installs root-owned copies of `loop`, `guard` and `adapters` in
  `/opt/inference-judge`, and an installed machine runs the guard from those
  copies;
- start the model server on the accelerator itself (the devices belong to
  group `bench`);
- change the judge's files or the target file after the clock starts (see
  `RULER_LOCK` in section 2.6).

The guard in `research/guard/` refuses commands and edits that touch the
accelerator or the frozen files. It has three scripts:
`pretooluse-bash.sh` (before a shell command), `pretooluse-edit.sh` (before a
file edit) and `posttooluse-audit.sh` (one line per tool call that
ran in `audit.log`). Each reads one JSON object on stdin. Exit status `2` blocks
the call, and stderr tells the agent why and what to do instead. The guard
is a tripwire, not the boundary. A process that the agent starts can bypass
it; it cannot bypass Unix permissions.

The guard reads `/etc/inference-judge/guard.env`, which
`setup/harden_research.sh` writes. That file names the research repo and the
frozen paths: the repo's `loop/`, `guard/`, `adapters/` and the agent's
settings folder, the driver's state directory and `/opt/inference-judge`.

Each agent reaches the guard in its own way:

- **Claude Code** and **Codex** call the three scripts as hooks. The hook
  files are `.claude/settings.json` and `.codex/hooks.json`; they sit in
  `research/adapters/claude/project/` and `research/adapters/codex/project/`,
  and setup copies them to the top of the research repo. On a hardened
  machine, root-owned managed settings
  (`/etc/claude-code/managed-settings.json`,
  `/etc/codex/requirements.toml`) name the root-owned copies in
  `/opt/inference-judge/guard` and allow only those hooks.
- **Pi** has no hook file. `research/adapters/pi/session.sh` loads the
  extension `research/adapters/pi/guard.ts`, which calls the same scripts:
  `pretooluse-bash.sh` before a `bash` call, `pretooluse-edit.sh` before
  every other call except Pi's own `read`, `grep`, `find` and `ls`, and
  `posttooluse-audit.sh` after every call that ran. It sends each script the tool's
  name, its input and the working directory, and it blocks a call when a
  script is missing.

The standing instructions for every agent are in `research/AGENTS.md`;
Claude Code is pointed at them by `CLAUDE.md`, which imports `AGENTS.md`;
setup copies it from the adapter folder to the top of the research repo.
The driver's prompt names `AGENTS.md` too. The researcher can write both
files at the top of the repo. They are guidance, not part of the boundary.

### What "sealed" means for a launch

A scored launch is the judge starting the model server in a container and
measuring it. The container is sealed:

- `--network none`, and no published port;
- every mount is read-only: the judge copies `/ro/exp` (the experiment) and
  `/ro/src/<name>` (each engine tree) into the container's own filesystem
  before the server starts;
- the one writable mount is a private copy of the compile cache, when the
  target has `engine.compile_cache`; `--instrument` runs alone also get
  `/out`;
- the judge measures from a sidecar container that joins the server's network
  namespace, so no port needs to be published.

Before the launches, a probe container with the same flags records what the
seal actually holds. The result is written to `results.json` under
`isolation`: `network`, `writable_mounts`, `exfil_probe` (`blocked`,
`leaked`, `not_proven` or `unknown`), `seal` (`full` or `weaker: ...`) and
`host_sockets`.

Some targets need a hole in the seal. The seal reads `weaker: ...` when a
launch has any of these:

- `--privileged`;
- `network=host`;
- a `--device` flag;
- a unix socket under a target mount;
- a writable mount other than the private compile cache;
- a cache mount that writes through to the seed.

The exceptions come only from the target's `engine.docker_flags`, and they
are reported in the result. A reader can see how strong the seal was for
every verdict. `target.py` refuses some flags outright: `-p`, writable
`-v`/`--mount`, networks other than `none` or `host`, `--gpus`, and
`--device` paths outside `/dev`. For `accelerator.kind: gpu` the judge adds
`--gpus all` itself (see section 4).

### The life of one experiment

1. The researcher writes `experiments/<id>/launch.sh` (and anything it
   needs), appends a line to `HYPOTHESES.jsonl` with a numeric prediction, and
   writes `queue/<id>.json`.
2. The runner (as `driver`) picks the oldest valid queue file, runs
   `preflight.sh <id>` as the researcher, and, if that passes, calls
   `sudo -n -u bench /opt/bench/run.sh <id>`.
3. `run.sh` scrubs its environment, checks the id, refuses to run if the
   judge's files differ from `RULER_LOCK`, and takes the exclusive bench lock.
4. It copies the experiment and the engine sources to a private place,
   snapshots them, and builds the image (baking in `deps.txt` if present).
5. It runs the launches in ABBA order (incumbent, candidate, candidate,
   incumbent), or ICI. Each launch waits for health, runs one unscored warm
   sweep, then `metric.blocks` scored sweeps. The first candidate launch also
   runs the canary. Every candidate launch is run through the quality gate.
6. `compare.py` turns the launches into a verdict, and `stats.py` supplies
   the rule. A `PROMOTED` candidate becomes the incumbent: its engine trees
   are kept and its image tagged.
7. `run.sh` removes the containers, writes `results.json` (atomically) and
   `summary.txt`, and exits `0`.
8. The runner reads `results.json`, appends a `verdict` event to the
   driver's log, and the next `BRIEF.md` shows it to the researcher.

### Verdict names

The judge writes one of these into `results.json`:

| Verdict | Meaning |
|---|---|
| `PROMOTED` | Faster by more than the noise threshold, quality and guards held. Becomes the incumbent. |
| `REJECTED` | Slower, not faster enough, failed the quality gate, broke a guard, or met the SLO at no level. `reasons` is never empty. |
| `REJECTED_SCREEN` | The screen (section 2.5) measured the candidate more than `metric.screen_tol` below the incumbent's stored value. |
| `INCONCLUSIVE` | There is no calibration, no incumbent or no stored snapshot of the incumbent yet. The code also gives it when the estimated gain reaches the threshold without being significant, which the default settings cannot produce (section 2.5, step 8 of the decision order). |
| `INVALID` | The measurement cannot be trusted or compared: stamps from a different judge (`ruler_mismatch`), an incumbent launch that is not a measurement or never became healthy (the box needs looking at, not the candidate), a candidate launch that is not a measurement, or, in a confirmation, a missing snapshot of the best. |
| `VOID_DRIFT` | The two incumbent launches differ by more than the noise allows, so the pair is thrown away. |
| `BUILD_FAILED` | The image build failed or timed out, `deps.txt` did not install in time, or the candidate server was not healthy within the health budget. |
| `CANARY_FAILED` | Fewer than 8 of 10 fixed prompts matched the stored canary reference. |
| `HUNG` | A sweep or a quality gate outlived its bound. |
| `TIMEOUT` | The run's budget (`budgets.run_s`) ran out. |
| `BUSY` | Another run holds the bench lock. |
| `INSTRUMENT` | An unscored `--instrument` run. |
| `BASELINE` | The operator's reference run. |
| `CALIBRATED` | The operator's calibration run. |
| `INFRASTRUCTURE_FAILURE` | The judge could not measure for a reason that is not the candidate's. |
| `REJECTED_RESERVED_ID` | A `baseline-*` id without the operator's flag file. |

The loop adds two verdicts of its own to the driver's ledger:
`PREFLIGHT_FAILED` (pre-flight failed; no judge slot was used) and
`NO_RESULT` (the judge exited without a readable `results.json`).

## 2. Interfaces

### 2.1 The target file

A target describes one machine, one model, one engine and one metric. It is
a YAML file, `targets/<id>.yaml`. `bench/target.py` validates it, and
everything that depends on hardware (the judge, `setup/bootstrap.sh`, the
shell scripts) reads it through `target.py` or through `setup/target.sh`.
Nothing else parses the YAML.

Sections and fields:

| Section | Required | Optional (default) |
|---|---|---|
| top level | `id` | `runnable` (`true`) |
| `accelerator` | `kind`, `chips`, `hosts` | `type`, `topology`, `hbm_gb_per_chip` |
| `model` | `served_name`, `revision`, `local_dir`, `tokenizer` | `chat_template` (`""`), `hf_id`, `gcs` |
| `engine` | `name`, `image`, `docker_flags` | `env` (`{}`), `launch`, `port` (`8000`), `shm_size` (`""`), `sources` (`[]`), `compile_cache` (`{seed, mount}`) |
| `metric` | `kind` (`tps`), `isl`, `osl`, `concurrency`, `slo`, `ramp_s`, `window_s` | `screen_window_s`, `design` (`abba`), `blocks` (`2`), `prompt_pool`, `confirm_pool`, `pool_size` (`2048`), `length_tol` (`0.0`), `guard_tol` (`0.03`), `screen_tol` (`0.02`), `stall_ms` (`1000`) |
| `budgets` | `build_s`, `compile_s`, `health_s`, `run_s` | none |
| `health` | none | `url`, `ready_timeout_s` |
| `reference` | `dir` | `prompts`, `tasks`, `max_tokens`, `launches` |
| `quality` | none | `allow_missing_logprobs` (`false`) |

The rules:

- An unknown key inside a known section is an error. An unknown top-level
  section (for example `provision`, which `setup/bootstrap.sh` reads) is kept
  and ignored by the judge.
- `accelerator.kind` is the target's own word for its hardware. The judge
  acts only on `gpu` (it adds `--gpus all`); `none` and `tpu` need no extra
  flag.
- `metric.isl` is one input size or a list. A list is served as a balanced
  mix, in a fixed order that candidate and incumbent share.
- `engine.sources` lists engine source trees by name. The researcher edits
  `~/<name>-src`; a launch sees a private copy at `/src/<name>`.
- `engine.compile_cache` has a `seed` directory that the judge owns and a
  `mount` path inside the container. A launch never writes to the seed: each
  incumbent launch gets its own copy, and a candidate's launches share one.
- `reference.launches` and `reference.max_tokens` set how the reference is
  recorded (section 2.4). Left out, they are 3 launches and 128 tokens for a
  real engine, and one server measured three times and 64 tokens for the
  mock.
- A launch waits for health up to `health.ready_timeout_s`, or, if that is
  not set, `budgets.compile_s + budgets.health_s`.
- `engine.compile_cache.mount` may not be, or lie under, `/ro`, `/exp`,
  `/src`, `/out`, `/opt/models` or `/opt/inference-judge`; the judge uses
  those paths itself.
- A target with `runnable: false` is a sketch. The judge refuses to measure
  it, and `setup/bootstrap.sh` stops before the clock starts.

The CLI of `target.py` has seven commands: `get`, `validate`, `hash`,
`shell`, `ruler-hash`, `lock` and `check-lock`. The last three are the
ruler lock (section 2.6).

On a box, the judge reads the target named in `$BENCH_ROOT/TARGET`. A public
copy of the same file sits at `/etc/inference-judge/target.yaml`, readable by
the researcher; `setup/check.sh` verifies that the two are identical.

The judge publishes the path of its lock in `$BENCH_ROOT/LOCK_PATH`, a file
that does not depend on the target. The driver reads it to tell whether the
bench is busy, without knowing whether the lock is a `flock` file or a
`.lock.d` directory.

`targets/cpu-mock.yaml` is the smallest working target. It runs the mock
server (`bench/mock_server.py`) as the engine and needs no accelerator.

### 2.2 Load generation for tokens per second

`bench/harness.py --mode tps` sends the load; `bench/metrics.py` turns the
per-request records into a level summary. `--mode` has
two other values: `canary` (section 2.4) and `pool`, which writes a
synthetic prompt pool.

**Loads.** `--tps-load` is `offline`, `closed` (the default) or `open`. A
sweep runs one level after another. A closed load takes its levels from
`--concurrency` and keeps that many requests in flight. An open load takes
`--rate` values (requests per second) and spaces arrivals with
`--arrival poisson` (the default) or `uniform`. An offline load takes
`--requests` and submits that many at once. A single `--isl` is one input
size. A comma-separated `--isl` is a size mix: it runs closed-loop only and
is scored on whole-answer speed (`stream_tps`).

**What every request carries.** `max_tokens` is `osl`, `ignore_eos` is
`true` (`--ignore-eos` is on by default), and `stream_options.include_usage`
is set. The prompt starts with a one-off `Request reference: <uuid>.` line
(`--nonce turn`, the default), so a server cannot answer from a cache by
matching earlier prompts. The nonce is at the start of the prompt, so a
prefix cache cannot match across requests either.

**What the harness counts.** The harness counts a request's output tokens by
re-tokenising the text it received, content and reasoning deltas together.
The server's `usage` field is only a cross-check. `--tokenizer` is `ws` (a
whitespace tokeniser, used with the mock) or the path to a Hugging Face
`tokenizer.json`, which needs the `tokenizers` package.

**Per-request record.** Each row in `tps_records.jsonl` has: `req_id`,
`prompt_id`, `load`, `level`, `t_sched`, `t_send`, `t_first`, `t_last`,
`tokens_client`, `tokens_usage`, `cached_tokens`, `finish_reason`, `itl_ms`,
`isl`, `isl_client`, `error_kind`, `prompt_reused`, `distinct4`,
`compression`, `text_sha1`, `ok` and `invalid_reason`.

A request is invalid, with `invalid_reason` set, when it:

- failed (`error:<kind>`);
- repeated a prompt that the invocation had already sent (`prompt_reused`);
- ended with a `finish_reason` other than `length` (`finish_reason:<x>`);
- returned a different length from `osl` (`length:<n>!=<osl>`, tolerance
  `--length-tol` as a fraction of `osl`, default `0.0`);
- reported a `usage` count that differs from the client count by more than
  1% of `osl` (`usage_mismatch:<n>`, `TPS_USAGE_TOLERANCE`). A missing
  `usage` is not a reason, because `usage` is only a cross-check.

The prompt sampler uses each prompt once per invocation. When a pool runs
dry it starts again and flags every repeat.

**Level summary.** Each concurrency level gets `goodput_tps_per_chip`,
`tps_per_chip`, p50, p95 and p99 of `ttft`, p50 and p99 of `tpot`, `n`,
`n_window`, `n_latency`, `n_invalid`, `invalid_reasons`, `distinct4`,
`compression`, `decode_tps`, `stream_tps`, `slo_met` and `sanity_ok`, and
for a size mix `by_isl` and `stream_tps_per_chip`. Throughput counts
requests that complete inside the measurement window; latency percentiles
use valid requests sent inside it. `--chips` sets the divisor for the
per-chip numbers.

**The headline.** The headline is the highest goodput per chip among the
levels that meet all three conditions: `slo_met`, zero invalid requests, and
text inside the sanity band when the SLO sets one. For a size mix the value
is whole-answer speed per chip. The service-level objective is `--slo`, a
JSON object with `p99_tpot_ms` and `p95_ttft_ms` (defaults 50 and 2000), and
optionally `min_distinct4` and `max_compression`.

**Self-check.** The harness measures its own event-loop lag and warns when
it exceeds 50 ms, because a slow client would look like a slow server.

**Windows and warm-up.** Each level has a `--ramp-s` ramp (default 2) and a
`--window-s` window (default 10). In a closed loop, `--warm-cycles` makes
every stream send that many unscored passes over the input sizes before the
ramp starts. `--prompt-pool` names a JSONL pool of `{"id", "prompt"}` rows,
sampled with `--seed`. `--pool-exclude` names `tps_records.jsonl` files
whose prompts must not be sent again. `run.sh` uses it to retire prompts: a
candidate keeps its writable cache between its launches and could carry an
answer from one to the next, so the prompts of the first candidate launch are
excluded from the second. Each incumbent launch still draws the same
prompts as the candidate launch next to it. In an ABBA run, `I1` and `I2`
therefore answer different prompts.

### 2.3 Mock server behaviours

`bench/mock_server.py` is a small OpenAI-style server with a toy language
model. It is the engine for the CPU target and the opponent in the cheat
suite. `--behaviour` picks one strategy; the default is `honest`.

Other flags: `--ttft-ms`, `--tpot-ms`, `--tokens-per-chunk`, `--knee`
(where throughput stops growing with load) and `--quant-eps` (default
`0.2`). The server supports logprobs (top 20) and `/v1/completions` with
`echo` and `logprobs`, which the quality gate needs.

| Behaviour | What it does | Why the judge does not credit it |
|---|---|---|
| `honest` | Streams the requested number of tokens at the configured speed. | The baseline for the suite. |
| `fast` | Honest, with time to first token and time per token halved. | It is credited: the control for a real speed-up. |
| `truncate` | Stops at `max_tokens / 2` and reports `finish_reason: length`. | The request is invalid: `length:<n>!=<osl>`. |
| `early_eos` | Ignores `ignore_eos` and stops at half length with `stop`. | The request is invalid: `finish_reason:stop`. |
| `lie_usage` | Sends half the tokens and reports `max_tokens` in `usage`. | The request is invalid: the client count disagrees with `usage`, and the length is wrong. |
| `junk` | Streams `a a a a` filler of the right length, twice as fast. | The quality gate fails it: the text is outside R's sanity band. |
| `cache_memo` | Answers a byte-identical prompt it has seen three times faster. | Every prompt starts with a one-off nonce, so no prompt repeats. |
| `cache_memo_strip_nonce` | The same, but it strips the leading nonce from its cache key. | The sampler sends each prompt once, and a repeat is invalid (`prompt_reused`). |
| `quantised` | 1.3 times faster; logits are perturbed by `--quant-eps`, so greedy text diverges after about 15 to 20 tokens. | It fails the quality gate. |
| `exfil` | Tries to connect to `1.1.1.1:53` and to write prompts to `/out/stolen.jsonl`; logs `EXFIL net=ok\|blocked write=ok\|blocked`. | Under the seal both attempts are blocked. |
| `wedged` | Answers `/v1/models` and the toy-model requests, so it becomes healthy and measures like `honest`, but never answers a plain chat request that carries neither `ignore_eos` nor `logprobs`. That is the canary. | The canary times out and the run ends as `CANARY_FAILED`. |

The mock emits one whitespace-separated word per output token, so the `ws`
tokeniser counts it exactly.

### 2.4 The reference and the quality gate

Speed is only credited if the output is still the model's output. The
quality gate (`bench/quality2.py`, `QUALITY2_VERSION` `2.0`) checks every
candidate launch against a fixed reference called R. It never compares a
candidate with the incumbent, so quality cannot drift by small steps.

**R.** R is a directory (`reference.dir` in the target) with
`manifest.json`, `prompts.jsonl`, `greedy.jsonl`, `logprobs.jsonl`,
`tasks.jsonl`, `task_answers.jsonl` and `floors.json`. The operator's
baseline run writes it once, from `reference.launches` cold launches of the
stock engine. It is atomic, refuses to overwrite an existing directory, and
leaves the files read-only. `manifest.json` records the sha256 of every
other file, and the gate refuses to load R if any file differs. The
`reference_hash` stamp is the sha256 of `manifest.json`.

The first launch of R produces the greedy continuations and teacher-forces
its own text for the logprobs. Later launches teacher-force the first
launch's text, so the only difference between launches is the engine's own
noise. That noise is R's floor: the spread of KL, top-1 agreement,
divergence and task accuracy between R's launches.

**Thresholds.** Each check has a start value. For `kl` and `top1` the final
threshold is the looser of the start value and three times R's floor. The
other three are fixed.

| Check | Passes when | Start value |
|---|---|---|
| `kl` | mean KL and p99 KL against R, teacher-forced, are both at or below the limits | mean 0.02, p99 0.30 |
| `top1` | the candidate's top token agrees with R's at least this often | 0.97 |
| `divergence` | the median position where greedy text first differs from R is at least this fraction of R's own median | 0.8 |
| `tasks` | the paired-bootstrap one-sided 95% lower bound of (candidate accuracy - R accuracy) is at least this (5000 resamples) | -0.015 |
| `sanity` | at most this fraction of texts lie outside R's 1st to 99th percentile band of `distinct4` and compression, widened by 10% | 0.10 |

If the candidate cannot return logprobs, `kl` falls back to the excess
negative log-likelihood of its text under R (limit 0.02 nats per token, or
three times R's floor), and `top1` is `unavailable`. An unavailable check
fails the gate, unless the target sets `quality.allow_missing_logprobs`.

Any error talking to the candidate also fails the affected check. The gate
fails closed.

**Tasks.** `tasks.jsonl` holds items of three kinds: `math`, `mc` (multiple
choice) and `tool` (a tool call whose name and arguments must match). Each
item is answered `task_draws` times per launch (default 1). The checkout
ships a small task and prompt set for the mock only; a real target supplies
its own with `reference.prompts` and `reference.tasks`.

**CLI.** `quality2.py` has three commands. `collect` runs one launch of R
into a directory. `make-reference` assembles R from the launch directories
(`--launches-from`), or, for the mock only, from one server measured several
times (`--base-url`, source `mock-replicated`). `gate` judges a server
against R and writes a JSON result.

**The canary.** The canary is a separate, cheaper check in
`harness.py --mode canary`. It sends 10 fixed prompts at temperature 0,
non-streaming, with `max_tokens` 256, and compares the first 16
whitespace-separated words of each answer with a stored reference. At least
8 of 10 must match (`CANARY_MIN_MATCH`), or the verdict is `CANARY_FAILED`.
It sends `reasoning_effort: low` by default; the measured traffic does not
set it. In a scored run each request may take up to `120 + budgets.compile_s`
seconds, because a canary prompt can land on a shape the warm sweep did not
compile. One timeout then ends the canary, because it means the server is
dead. The harness's own defaults are 120 seconds and two timeouts in a
row. The reference is stored per target (`canary/<target>.json`). Through
the sudo route only the operator's baseline writes it.

### 2.5 Statistics

All numbers are log tokens per second, higher is better, so a difference is
a relative change. `bench/stats.py` holds the rules; `bench/compare.py`
applies them to a run's launches.

**Calibration.** The operator runs `run.sh --calibrate <id>`. It launches the
incumbent `CALIBRATE_LAUNCHES` times (default 10) and measures the spread of
the launch means, `sigma_L`. Everything else is judged against it.
`compare.py` floors `sigma_L` at `0.005`.

The minimum detectable effect is `mde = max(0.02, 3.2 * sigma_L)`. The 3.2 is
the sum of two normal quantiles (2.33 for a 1% false-promotion rate and 0.84
for 80% power). No threshold ever sits below 2%.

**Screen.** If the target sets `metric.screen_window_s`, the candidate is
first launched alone for one window of that length. If it measures more than
`metric.screen_tol` (default `0.02`) below the incumbent's stored value, the
verdict is `REJECTED_SCREEN` after that one launch. Without
`screen_window_s` the same test is applied to the first candidate launch.

**ABBA.** The design `abba` runs incumbent, candidate, candidate, incumbent.

```
delta = 0.5 * [(C1 - I1) + (C2 - I2)]      SE = sigma_L      alpha = 0.01
```

The pair is void (`VOID_DRIFT`) when `|I2 - I1| > 3 * sqrt(2) * sigma_L`.
Otherwise it passes when `delta - z * SE > 0` and `delta >= mde`.

**ICI.** The design `ici` runs incumbent, candidate, incumbent.

```
delta = C - (I1 + I2) / 2                  SE = sigma_L * sqrt(1.5)
```

Void and pass rules are the same.

**The decision order.** `compare.py` decides in this order and stops at the
first that applies:

1. The stamps of the launches must be compatible, or `INVALID`.
2. Any launch that is not a measurement is `INVALID`. A candidate launch
   where no level met the SLO is `REJECTED`.
3. No calibration gives `INCONCLUSIVE`.
4. The drift test gives `VOID_DRIFT`.
5. The quality gate must have run and passed on every candidate launch.
6. Every level must stay within `metric.guard_tol` (default `0.03`) of the
   incumbent and of the stock reference. A size mix has per-size guards too.
   A failure in 5 or 6 gives `REJECTED`.
7. If the test passes, `PROMOTED`.
8. If it does not pass but `delta >= mde`, `INCONCLUSIVE`. With the default
   alpha `0.01` this step is never reached: `mde` is at least `3.2 * sigma_L`,
   and a gain is already significant at `2.33 * sigma_L` (ABBA) or
   `2.85 * sigma_L` (ICI), below `mde`.
9. Otherwise `REJECTED`.

**Launch status.** A launch is `ok` or one of `invalid_requests`,
`insane_text`, `compile_stalls`, `no_blocks` or `no_goodput`. A block with an
inter-token gap over `metric.stall_ms` (default `1000`) is dropped, so one
stall does not decide a verdict.

**Confirmation.** The operator runs `run.sh --confirm <id>` for the current
leaderboard best. It reruns the pair as ICI on a prompt pool the promotion
never saw (`metric.confirm_pool`, or a disjoint draw), at alpha `0.05`, and
requires `delta' >= mde / 2`. The leaderboard keeps the confirmed `delta'`,
not the first one, because the first estimate of a winner is biased high. A
failed confirmation demotes the best.

**Leaderboard.** `leaderboard/<target>.json` holds `best`, `history`,
`reference` (the baseline entry, with the stock levels) and `demoted`. It
is written atomically.

**False-promotion rate.** `stats.aa_false_promotion_rate` simulates
identical candidates under a noise model. The cheat suite runs its own
simulation through the same rules (section 3, rows 3a and 3b).

### 2.6 Results

Each run writes `$BENCH_ROOT/results/<id>/results.json` (schema 2,
atomically) and `summary.txt`. The fields:

| Field | Meaning |
|---|---|
| `schema` | `2`. |
| `id`, `run_id` | The experiment id and a unique id for this run. |
| `kind` | `candidate`, `baseline`, `instrument`, `calibration` or `confirm`. |
| `verdict`, `reasons` | One of the verdict names in section 1. `reasons` is never empty for a `REJECTED*` verdict. |
| `target` | `id`, `file`, `hash`, `metric`. |
| `stamps` | See below. |
| `ruler_lock` | `state` (`absent` or `matched`) and `path`. |
| `attestation` | `weights_sha256` (`not computed`), `tokenizer_sha256`, `chat_template_sha256` (`not computed`), `engine_flags`, `engine_image_digest`. |
| `metric` | The per-level values, the headline and the guards. |
| `stats` | `delta`, `se`, `z`, `alpha`, `mde`, `sigma_L`, `drift_gap`, `passed`, `void_drift`. |
| `isolation` | The seal (section 1). |
| `started`, `finished`, `duration_s` | Timing. |
| `engine_image`, `engine_image_digest`, `base_url` | What was run. |
| `deps_installed`, `cleanup_ok`, `snapshot` | Bookkeeping about the build and the snapshot. |
| `canary` | `matched` and `total`. |
| `quality` | The gate results per candidate launch. |
| `launches`, `design`, `screen` | The launches, the design (`abba`, `ici` or `ici-confirm`) and the screen result. |

The attestation records two hashes as `not computed`: the judge does not
hash the weights or the chat template. The model is pinned by
`model.revision` and by a read-only mount.

**Stamps.** Every result carries `ruler_hash`, `compare_version` (`2.0`),
`quality_version` (`2.0`), `target_id`, `target_hash`, `image_digest`,
`model_revision` and `reference_hash`. `compare.py` calls two measurements
comparable only if `ruler_hash`, `compare_version` and `quality_version` all
agree. A mismatch ends the run `INVALID` with reason `ruler_mismatch`. A
calibration or incumbent that carries no stamps at all is not checked.
This keeps the incumbent, measured by an older judge, from being compared
with the candidate.

**The ruler lock.** `ruler_hash` is the sha256 over these files, sorted by
name, plus the target file: `harness.py`, `metrics.py`, `compare.py`,
`quality2.py`, `stats.py`, `tokenizer.py`, `target.py` and `run.sh`. For
each file the hash takes `name`, a NUL byte, the file's bytes and a NUL
byte.

At clock start `setup/bootstrap.sh` runs `run.sh --lock-ruler` as `driver`,
through its sudo route.
That writes `$BENCH_ROOT/RULER_LOCK` once (`O_EXCL`; it refuses to
overwrite). The file holds the hash and a `target NAME` line. From then on,
`run.sh` calls `target.py check-lock` at the start of every run and exits
`70` if the live judge or the target differs from the lock. A lock without
a `target` line covers the judge's files only. The lock covers code and the
target file, not the pools, the reference or the canary reference. Those are
protected by Unix permissions and by being written once.

**The seal.** The seal of a scored launch is a rule of the judge, not a
setting of the target: `--network none`, no published port, read-only mounts
and one private writable cache. The target can only weaken it, through
`engine.docker_flags`, and `target.py` refuses the flags that would break it
outright (section 1). Every result records the outcome in `isolation`:
`seal` is `full` or `weaker: ...`, and `exfil_probe` says whether a probe
container saw its connection blocked. `run.sh --dry-run` prints every
command a run would execute, and row 5 of the cheat suite replays those
flags to test them.

**Operator steps.** Under `sudo`, `run.sh` accepts only a bare id,
`--instrument <id>` and `--lock-ruler`, and it ignores environment
overrides. It refuses `--calibrate` and `--confirm` there, so the operator
runs those two as `bench` directly, for example with
`env -u SUDO_USER runuser -u bench`. The operator's runs are:

- baseline: `run.sh baseline-<name>`, allowed only while
  `$BENCH_ROOT/.allow-baseline` exists. It writes the canary reference, R
  and the first leaderboard entry and stock reference.
- `run.sh --calibrate <id>`.
- `run.sh --confirm <id>`.
- `run.sh --lock-ruler`.

**Snapshot.** For every candidate and baseline run, the judge keeps the
experiment directory, the engine diffs, the untracked-file list and the
`HEAD`s, collected by git inside a throwaway container, never as `bench`.
When a run is promoted, or is the baseline, the engine trees (without
`.git`) go to `results/<id>/snapshot/src` and the image is
tagged `inference-judge-inc:<target>-<id>`. Old `inference-judge-exp:*`
images are pruned, keeping the five newest.

**Dependencies.** An experiment may bring a `deps.txt`. The judge bakes it
into a derived image, `inference-judge-exp:<id>-deps`, at build time, with a
10-minute limit (`DEPS_TIMEOUT_S`). The build has network access. The scored
launch never does.

**Locks and limits.** One run holds the exclusive bench lock (`flock`, or a
`.lock.d` directory where `flock` does not exist); a second run gets `BUSY`.
A watchdog writes `TIMEOUT` when `budgets.run_s` is used up (scaled up for
baseline and calibration). A sweep longer than
`2 * n * (ramp + window) + 120` seconds, where `n` is the number of levels,
is `HUNG` (the warm sweep also gets `budgets.compile_s`). The judge ignores
a `SIGTERM` or `SIGHUP` from its caller. Exit codes: `0` for every verdict,
`2` for bad usage, `70` for infrastructure faults.

### 2.7 The loop side

The loop moves experiments from the researcher to the judge and the
verdicts back. It never lets a queue file reach the judge's command line
except as a validated id and kind.

**Queue.** The researcher writes `~/research/queue/<id>.json`, for example
`{"id": "try-flag", "kind": "candidate"}`. The file name is `<id>.json`.
`kind` is `candidate` or `instrument`. The id matches
`^[a-z0-9][a-z0-9-]{2,40}$`; `baseline-*` is reserved. `loopstate.read_queue`
is the only code that opens a queue file, and it returns only the id and the
kind. A file is refused if it is a symlink or not a regular file, is larger
than 16 KiB, is not a JSON object, has an id that does not match its file
name, has an unknown kind, already has a results directory, or (for
`candidate`) has no entry in `HYPOTHESES.jsonl` with `experiment` set to the
id and a numeric `prediction` or `predicted`. Every refusal is logged as a
`queue_reject` event. Instruments are exempt from the prediction rule.

**Runner.** `research/loop/runner.sh` runs `runner.py` as `driver`, one
experiment at a time, independent of any agent session. For each pass it:

1. stops if the run has finished, or idles if the clock has not started, if
   the loop is paused because frozen files changed, or if less than
   `run_min_s` (2700 s) of the budget remains;
2. takes the oldest accepted queue file (by modification time, then name);
3. runs `preflight.sh <id>` as the researcher. A failure logs a
   `PREFLIGHT_FAILED` verdict that costs no judge slot, and the same file is
   not retried until the researcher edits it;
4. runs `$IJ_BENCH_CMD <id>` (or `--instrument <id>`). The default
   `IJ_BENCH_CMD` is `sudo -n -u bench /opt/bench/run.sh`. The command is an
   argument list, never a shell string;
5. reads `results/<id>/results.json` and logs a `verdict` event, or
   `NO_RESULT` if there is none.

A `runner.lock` makes sure only one runner exists. If a runner died between
`run_start` and `run_end`, the next one closes the run as interrupted; the
driver's scan of the results directory still picks up the result, and
verdicts are de-duplicated by `run_id`.

**Pre-flight.** `preflight.sh <id>` runs on the CPU in seconds. It checks
that the id is well formed, is not `baseline-*`, and is not used (no results
directory, no verdict in `runs.jsonl`); that `experiments/<id>/launch.sh`
exists, passes `bash -n` and passes `shellcheck -S error` (skipped with a
warning if `shellcheck` is not installed); that every
`patches/*.diff` or `*.patch` applies (`git apply --check`) to at least one
tree in `IJ_PATCH_TREES`; and that `deps.txt`, if present, parses as pip
requirements. Its last line is the verdict, `PREFLIGHT_OK <id>` or
`PREFLIGHT_FAILED <id>: <reasons>`. The exit code is `0`, `1` or `2` (bad
usage). The researcher runs it before queueing, and the runner runs it again.

**Events and ledgers.** Everything the driver and runner learn is appended
to one log, `events.jsonl`, in the state directory (default
`~/research-state/`). Nothing else is state: tallies, the budget clock,
crash recovery, the queue view and `BRIEF.md` are all computed by replaying
that log. Each event gets a `seq` and a `ts`. The event types are
`run_config`, `driver_start`, `tick`, `session_start`, `session_end`,
`brief`, `verdict`, `run_start`, `run_end`, `queue_reject`, `job`,
`job_reaped`, `frozen_hash`, `pause`, `resume`, `alert` and `finish`.
Verdicts are mirrored to `runs.jsonl` and session ends to `process.jsonl`.
The directory also holds `BRIEF.md`, `prompt.txt`, `heartbeat.json`,
`ALERTS.md`, `driver.log`, `iterations/<n>.json` and `runs/<id>.log`.

**BRIEF.md.** The driver rewrites `BRIEF.md` before each session. It holds
the clock, the accelerator-hour budget left, the queue, recent verdicts and
tallies. The researcher reads it first and cannot write it.

**When a session starts.** `loopstate.py` decides, from the replayed log,
whether to start an agent session, sleep or finish. A session starts
for one of these reasons: `first`, `verdict` (a new verdict landed),
`queue_low` (fewer than `queue_target` experiments queued, with back-off),
`timer`, `wrap_up` (within `summary_lead_s` of the end, if `SUMMARY.md` does
not exist), `retry_after_crash` or `auth_probe`. A session ends in one of
five classes: `ok`, `timeout`, `limit`, `auth` or `crash`. Measurement
never waits for a session: the runner keeps draining the queue while the
researcher thinks.

**How a session starts.** `research/loop/run.sh` reaches the coding agent
through an adapter, one script per agent: `adapters/$IJ_AGENT/session.sh`.
`IJ_AGENT` defaults to `claude`; three adapters ship (`claude`, `codex` and
`pi`), plus a `TEMPLATE` to copy. If the script is missing, `run.sh` stops at
once, `--dry-run` included. `IJ_AGENT_BIN` names the agent's command when it
is not on the PATH that `sudo` allows. The adapter script takes three verbs.
For each session the driver calls `run`, in the research repo and as the
researcher's Unix user:

```
adapters/$IJ_AGENT/session.sh run --model M --effort E    # prompt on stdin
adapters/$IJ_AGENT/session.sh version                     # print the agent's version
adapters/$IJ_AGENT/session.sh auth                        # exit 0 logged in, 1 not, 2 cannot tell
```

- `run` reads the prompt from stdin, runs one session that needs no person,
  and prints exactly one JSON object on stdout:
  `{"type": "result", "is_error": false, "result": "..."}`. Nothing else goes
  to stdout. `./loop/run.sh --dry-run` prints the command line without
  starting an agent.
- The exit status is the agent's, and it is never 0 after a failure. The
  driver takes a session that exits 0 as a success unless the result names a
  quota limit or a login problem. When `is_error` is `true`, `result`
  carries the agent's own error text, because the driver reads that text to
  tell a quota limit and an expired login from a crash. The driver ends a
  session that runs past its cap (`iter_timeout_s`, shortened near the end
  of the run) with SIGTERM.
- `M` is the word `primary` or `fallback`, or a model name when
  `IJ_PRIMARY_MODEL` or `IJ_FALLBACK_MODEL` is set (see the policy table
  below). `E` is the reasoning effort (`IJ_EFFORT`, default `high`).
- `version` and `auth` are read by `setup/check.sh`.

`research/adapters/README.md` has the rest: the optional fields of the
result, the helper functions in `adapters/common.sh`, and how to write your
own adapter. The loop reads only the one result object and the files the
session leaves in the research repo, so an adapter is also the place to
build a design with several agents. The `claude/` folder ships three helper
agents (scout, hypothesizer, sceptic) as one worked example; the `codex/`
and `pi/` adapters ship none.

**Recovery.** The defaults are in `DEFAULT_POLICY` in
`research/loop/loopstate.py`, and each can be set with an `IJ_<NAME>`
environment variable:

| Policy | Default | Meaning |
|---|---|---|
| `primary_model`, `fallback_model` | `primary`, `fallback` | Passed as `--model`. The agent's adapter maps the two words to models: Claude's maps them to `opus` and `sonnet`, Codex's and Pi's use the model the tool is configured for, for both. `IJ_PRIMARY_MODEL` and `IJ_FALLBACK_MODEL` send a model name instead. |
| `max_crashes` | 3 | Consecutive crashes before the next step. |
| `fallback_sessions` | 3 | Sessions on the fallback model before trying the primary again. |
| `backoff_base_s`, `backoff_max_s` | 60, 1800 | Crash back-off doubles from 60 s to the cap. |
| `iter_timeout_s` | 3600 | Cap on one session. |
| `queue_target` | 2 | Experiments to keep queued. |
| `timer_s`, `min_gap_s`, `poll_s` | 3600, 60, 30 | Pacing. |
| `auth_probe_s` | 1800 | Retry interval while paused on authentication. |
| `limit_alert_streak` | 6 | Quota limits in a row before an alert. |
| `summary_lead_s` | 1800 | Wrap-up phase before the end. |
| `run_min_s` | 2700 | The runner starts no run with less time left. |

A crash backs off, then moves to the fallback model, then pauses and raises
an alert. A quota limit sleeps and is never a crash. An authentication
failure pauses, retries on a timer, and is never a crash. A pause stops
sessions, not measurement: the runner keeps running. The run ends at the
earlier of the wall deadline and the accelerator-hour budget (both default
to 24 hours); the driver then exits `10`. The driver also hashes the frozen
files (`loop/`, `guard/`, `adapters/`, the agent's settings folder and, on an
installed machine, `/opt/inference-judge`) and pauses if they change.

With Codex or Pi, `primary` and `fallback` name the same model, so the move
to the fallback model runs that model again, and the pause comes after
`max_crashes` more crashes in a row. Set `IJ_FALLBACK_MODEL` to a model name
to get a real second model, or `IJ_FALLBACK_SESSIONS=0` to skip the step.

Alerts go to `ALERTS.md` in the state directory and to `IJ_ALERT`, which is
an `http(s)` URL (an ntfy-style `POST`) or a file path.

## 3. The cheat suite

`scorecard/run_scorecard.py` runs the real judge against mock servers and
fixtures that fake a speed-up, or break something around the judge, and
prints one row per attempt. It exits `1` if any row is `FAIL` or `ERROR`.
The evidence for every row goes to `scorecard/results/latest.json`
(`partial.json` with `--only` or `--quick`), or to the `--out` file. Every
row that checks a refusal also runs a control, so a judge that refuses
everything would not pass.

| Row | What is tried | What the judge must do | How the row is obtained |
|---|---|---|---|
| 1a | A server that really is twice as fast (the control). | Credit it. | end-to-end |
| 1b | Answers cut to half length, reported as full length (`truncate`). | Not credit it. | end-to-end |
| 1c | Stopping early although the request says to ignore end-of-text (`early_eos`). | Not credit it. | end-to-end |
| 1d | Half the tokens, full count reported (`lie_usage`). | Not credit it. | end-to-end |
| 1e | Filler text of the right length, twice as fast (`junk`). | Not credit it. | end-to-end |
| 1f | A cache that answers a repeated prompt three times faster (`cache_memo`). | Not credit it. | end-to-end |
| 1g | The same cache with the nonce stripped from its key (`cache_memo_strip_nonce`). | Not credit it. | end-to-end |
| 2a | A server 1.3 times faster because its output drifted (`quantised`). | Fail it at the quality gate. | end-to-end |
| 2b | Four candidates in a row, each 10% faster at one load level and 2.9% slower at another than the one before, just inside the 3% step guard. | Stop the chain at step 2, because every candidate is also held against the first baseline. | simulated |
| 3a | A candidate identical to the incumbent, on a noisy drifting machine. | Promote it at most once in 1000 tries. | simulated |
| 3b | A candidate that really is 12% faster, under the same noise. | Promote it at least 8 times in 10. | simulated |
| 4a | An incumbent measured by a different version of the judge. | End as `INVALID` before launching anything. | end-to-end |
| 4b | A faster candidate whose quality-gate result is missing. | Not promote it: a missing result is not a pass. | simulated |
| 4c | A judge file edited after the lock. | Refuse to run. | end-to-end |
| 5 | Code in a scored container that opens an internet connection and writes a file to the host. | Block both. | end-to-end |
| 6a | The agent running commands that touch the accelerator or its settings. | Deny all of them. | end-to-end |
| 6b | The agent running ordinary commands (the control). | Allow all of them. | end-to-end |
| 7a | The agent crashing at every start, at the beginning of a 24-hour run. | Keep the run alive: back off, or pause and alert. | end-to-end |
| 7b | A queued experiment whose patch does not apply to the engine source. | Reject it before the judge starts, so no run is spent. | end-to-end |

What "end-to-end" and "simulated" mean:

- **End-to-end:** the real code ran from start to finish on the machine
  against the mock server or a small fixture. Rows 1 and 2a start the mock
  server and run the real harness and quality gate. Rows 4a and 4c run the
  real `bench/run.sh` on a private copy of the judge. Row 5 takes the
  `docker run` flags from `run.sh --dry-run` and replays them on a plain
  Python image, dropping `--gpus`, `-p`, `--publish`, `--name`, `--label`,
  `--device`, `--runtime` and `--entrypoint`, and replacing every mount
  source with an empty temporary directory. Row 7a runs the real driver and
  the Claude adapter with a stub in place of the agent's command; row 7b
  runs the real runner with a stub judge that only records that it was
  called.
- **Simulated:** the real decision code ran on inputs the row made up. Rows
  2b and 4b write launch summaries by hand and give them to
  `compare.py tps-verdict` and `tps-promote`. Row 3 simulates thousands of
  comparisons and gives them to the calibration, ABBA and confirmation rules
  in `stats.py`; a candidate counts as promoted when it passes both tests.

Row 3 uses four illustrative noise numbers written at the top of
`scenarios/s3_statistics.py`, not a measured machine. On a real target, the
noise is whatever `run.sh --calibrate` measures. A real gain right at the
threshold is promoted about half the time; the suite prints how often
smaller gains are missed. A missed gain can be queued again. A false
promotion becomes the incumbent that all later experiments are compared
with, which is why the judge is strict.

`judge.py` is the only way a scenario reaches the judge. It starts the
judge's code in a child process the way `bench/run.sh` does, and
`probes/probe.py` refuses any module loaded from outside the tree, so the
suite tests what a scored run executes. Row 5 needs Docker and the
`python:3.12-slim` image; without them the row reads `SKIP`, which is not a
pass.

## 4. Known limits

**What has been run where.**

- TPU (Google Cloud v6e): the model server of
  `targets/v6e4-gemma4-26b-tps.yaml` was brought up by hand on a v6e-4.
  `setup/tpu.sh`, bootstrap and a judged run have not been run on real TPU
  hardware, and the target is still `runnable: false`.
- GPU (NVIDIA): built and tested on a laptop with fakes. It has not been run
  on real hardware. Expect to find problems.
- CPU mock: runs on any laptop. The cheat suite and the rehearsal use it.
- `bench/measure.py` has been tested against the mock server and run once
  against a real server (`mlx_lm.server` with a 135M model, on a laptop). It
  has not been run against a hosted API, or against a server on a GPU or TPU.

**The cheat suite is evidence, not proof.**

- It runs on a mock server. It shows that the judge's rules catch these
  behaviours; it does not show that a real engine cannot misbehave in a way
  the mock does not imitate, and it cannot show there is no other cheat.
- Row 5 checks container flags, not the host. It does not test the firewall,
  the accounts or the file permissions that `setup/` creates on a real box.
- Row 6 feeds five commands of each kind to the guard's Bash script
  (`research/guard/pretooluse-bash.sh`). It is a sample, not a proof that the
  guard cannot be bypassed. It does not run any agent, and it does not test
  `pretooluse-edit.sh`.
- Timing rows measure a few seconds on a shared laptop and can wobble by a
  few percent.

**The laptop rehearsal uses a stand-in judge.** Part 1 of `demo.sh` (the
cheat suite) exercises the real judge. Part 2 (the loop rehearsal in
`setup/local_box/`) exercises the queue, the driver, the walls and crash
recovery with a scripted stand-in for the agent and, in its default tier, a
stand-in judge (`stub_bench.sh`). It tests what is around the judge, not the
judge. Tier 2 (`LOCALBOX_TIER=2`) uses the real `bench/run.sh` and the
sealed mock, and it stops before the driver phases.

**Seal and hardware.**

- A `gpu` target is given every GPU on the machine (`--gpus all`). There is
  no per-GPU selection.
- A target whose engine needs a device or a socket weakens the seal, and
  its results say so (`seal: weaker: ...`, and `exfil_probe` `not_proven` or
  `unknown`). The weaker seal is reported, not hidden, but it is weaker.
- `--local` runs `launch.sh` on the host with no seal. It exists for tests
  and is allowed only for the mock engine.

**Dependencies.** `deps.txt` installs with `uv` into `/opt/venv/vllm` if
that venv exists (`CONTAINER_VENV`), else with `uv pip install --system`,
else, where the image has no `uv`, with `pip`.
It is not a clean environment per experiment. The base image decides which
interpreter an experiment's dependencies end up in.

**Canary.** The canary sends `reasoning_effort: low` by default. The
measured traffic does not set it. A model for which `low` changes its
behaviour is compared at that setting only. It compares 16 words of 10
prompts; it catches gross output changes, not subtle ones. The quality gate
does the fine work.

**Which agents have run.**

- Claude Code is the only agent that has run a real loop, and that run was
  before the loop called it through `session.sh`. Through the adapter it has
  been tested with a stand-in program only. A real session through the
  adapter, on a laptop, did not get past an expired login.
- Codex and Pi: the adapters were written from those tools' documentation
  and tested with stand-in programs that print the documented output. Each
  has then run one short real session on a laptop through its adapter, a
  wiring check and not research. The adapter read the agent's output, the
  guard refused an edit of a frozen file and a call to `nvidia-smi`,
  `audit.log` got a line for each call that ran, and the agent had
  `AGENTS.md` in its context. For Pi, `guard.ts` loaded. Neither has run a
  real loop. Unseen for both: the driver starting sessions on a hardened
  machine as the `researcher` user, with the managed settings file in place,
  and the wording of their quota and login errors. Unseen for Codex: that
  its hooks are enforced from `/etc/codex/requirements.toml`.
- Not checked for Claude Code: that a session without a person loads
  `AGENTS.md` through the `@AGENTS.md` import in `CLAUDE.md`. The driver's
  prompt names `AGENTS.md` too, so the agent is told to read it either way.
- With Codex or Pi, the driver's fallback step repeats the same model unless
  `IJ_FALLBACK_MODEL` names another or `IJ_FALLBACK_SESSIONS=0` (section 2.7).
- The laptop rehearsal runs the Claude adapter only, with a stand-in agent.
  There is no option to rehearse Codex or Pi.
- Another agent needs its own adapter (section 2.7).

**Statistics.**

- The decision depends on `sigma_L`, which the operator measures once, at
  the start. If the machine gets noisier later, the judge does not know.
- `stats.py` assumes the confirmation runs at least 6 hours after the
  promotion. `run.sh --confirm` does not enforce that gap.
- A real gain below `mde` is not promoted, by design. A noisier machine
  needs a bigger gain.

**The lock and the guard.**

- `RULER_LOCK` covers the judge's code and the target file. It does not
  hash the prompt pools, the reference or the canary reference; Unix
  permissions and write-once files protect those.
- The attestation does not hash the weights or the chat template.
- The guard (`research/guard/`) is a tripwire, however the agent calls it.
  The boundary is the Unix permissions that `setup/harden_research.sh` sets,
  and that script runs on Linux only.
- The guard's Bash script assumes the judge is installed at `/opt/bench`.
- Some tools can bypass an agent's hooks. `research/adapters/README.md` lists
  the ones it knows of for Codex and Pi.

**Pre-flight.** If `shellcheck` is not installed, pre-flight skips the lint
and prints a warning.

**Tokenisers.** The harness counts tokens by re-tokenising the response with
the target's tokenizer. A target that sets `ws` is counted by whitespace,
which suits the mock and nothing else.
