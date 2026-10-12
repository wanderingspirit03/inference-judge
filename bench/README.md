# bench/ - the measuring half of the loop

This directory answers one question, over and over, without ever flattering us:

> **Does this change make the server produce tokens faster, without changing
> what it says?**

The researcher half of the loop proposes changes; `bench/` decides whether they
were real. On the box the researcher cannot edit anything in this directory: it
belongs to a different Unix user (`bench`). In the hardened layout the researcher
asks for a measurement by writing a queue file, and the loop's runner, running as
`driver`, calls `sudo -n -u bench /opt/bench/run.sh <experiment-id>`
(see [`research/README.md`](../research/README.md)).

This page calls `bench/` "the judge". The code calls it "the ruler" (`RULER_LOCK`,
`ruler_hash`, `--lock-ruler`). The only metric is `tps`, output tokens per second.

---

## The number

**Output tokens per second per chip** (`tps`), at a fixed input size, a fixed output
size and a fixed load, from a server that meets a latency limit.

* The harness counts the tokens itself by re-tokenising the text it received.
  The server's `usage` is only a cross-check.
* Every request must stop on the length limit with exactly the output size
  asked for. A reply that ends early, comes back short, or disagrees with its
  own `usage` does not count.
* Every prompt starts with a unique line, so nothing can be answered from a
  cache, and no prompt is sent twice to the same candidate.
* A load level only counts when it meets the target's limits: p99 time per
  output token and p95 time to first token.
* When the target lists several input sizes, one stream is served a balanced
  mix of them and the score is whole-answer speed: output tokens over the time
  from sending the request to its last token, harmonic mean over the sizes.

Speed is half of it. A candidate is only promoted when its answers still match
a fixed reference (`quality2.py`), so a faster server that says something else
does not win.

---

## What each file is

| file | in one sentence |
|---|---|
| `run.sh` | The whole experiment: lock, copy, build, sealed launches, sweeps, canary, quality gate, verdict. The only program that measures. |
| `harness.py` | The load generator. It talks to an OpenAI-compatible server over streaming HTTP and records when every token arrived. Also runs the canary and writes the mock's prompt pool. |
| `metrics.py` | The arithmetic on those records: which requests count, the per-level summary, the headline. |
| `tokenizer.py` | The token counter the judge uses: `ws` (whitespace, for the mock) or a Hugging Face `tokenizer.json`. |
| `stats.py` | The promotion statistics: launch noise, minimum detectable effect, the screen, the ABBA and ICI decisions, the confirmation. |
| `compare.py` | The judge: turns measured launches into PROMOTED / INCONCLUSIVE / REJECTED / INVALID and keeps the leaderboard. |
| `quality2.py` | The quality guard: compares a candidate with a fixed reference R (token distributions, greedy divergence, task answers, text sanity). |
| `target.py` | Reads and checks `targets/<id>.yaml`, computes the ruler hash, writes and checks `RULER_LOCK`. |
| `measure.py` | A stand-alone speed table for any OpenAI-style endpoint: no target file, no Docker, no verdict (see below). |
| `mock_server.py` | A fake inference server with adjustable speed and a set of cheating behaviours, so all of the above can be tested with no accelerator. |
| `canary/prompts.json` | Ten fixed prompts that prove the model, template and sampling did not change between runs. |
| `pools/` | Prompt pools and quality prompts for a real target, with the generator that wrote them (`pools/README.md`). |
| `../targets/` | One file per target: the hardware, the model, the engine image, the metric and the budgets. `cpu-mock` runs the whole judge on a laptop. |
| `tests/` | The test suite. |

---

## On a laptop (no accelerator)

```bash
cd bench
make venv          # uv venv --python 3.12 .venv && uv pip install --python .venv -e '.[test]'
make test          # the whole suite (it includes end-to-end runs, so it is not quick)
make mock          # a fake server on :8001 you can curl
```

A single test file runs on its own:
`.venv/bin/python -m pytest tests/test_tps_metrics.py`.

To see what a run would do, without running anything:

```bash
BENCH_TARGET=cpu-mock ./run.sh --dry-run my-experiment
```

It prints every command, the seal flags and the state of the `RULER_LOCK` check, and
ends with a `dry run:` line instead of a `VERDICT` line.

A real run against the mock server, with no Docker:

```bash
BENCH_TARGET=cpu-mock RESEARCH_ROOT=$PWD/../research IJ_SRC_ROOT=$PWD/.. \
  ./run.sh --local baseline-mock
```

`--local` runs the experiment's `launch.sh` as a host process and is allowed
for the mock engine only. Its results say `isolation.exfil_probe: not_sealed`.
`baseline-*` ids are reserved for the operator: create the empty file
`.allow-baseline` next to `run.sh` (`touch .allow-baseline`) before the command
above, and remove it afterwards. For a sealed
laptop run, build the image once
(`docker build -t inference-judge-mock:latest -t inference-judge-ruler:latest -f ../targets/cpu-mock.Dockerfile ../targets`)
and drop `--local`.

### Measuring one endpoint

`measure.py` sends the judge's load to any OpenAI-compatible endpoint and prints
a speed table. It is a uv script (its dependencies are declared in the file):

```bash
uv run measure.py URL --model NAME     # from bench/; from the repository root: uv run bench/measure.py
```

Its options are `--tokenizer`, `--input`, `--output`, `--users`, `--seconds`,
`--api-key-env` and `--json`; see `measure.py --help`. It counts tokens itself
when it has the model's tokenizer and otherwise falls back to the server's count,
and it says which. A server that ignores the fixed answer length is measured at
whatever length it sends, and the table says "free length". The result is a
measurement, not a verdict: nothing is sealed or compared. It is tested against
`mock_server.py` (`tests/test_measure.py`) and has been run once against a real
server (`mlx_lm.server` with a 135M model, on a laptop), not against a hosted API.

### Measuring one endpoint with the harness directly

`harness.py` alone measures any OpenAI-compatible streaming endpoint. It needs
a prompt pool (JSONL of `{"id", "prompt"}`) and a tokenizer for the model:

```bash
.venv/bin/python harness.py --mode pool --pool-size 512 --isl 256 --seed 1 --out /tmp/pool
.venv/bin/python harness.py --mode tps --base-url http://127.0.0.1:8001/v1 --model mock \
  --tokenizer ws --prompt-pool /tmp/pool/prompt_pool.jsonl \
  --isl 256 --osl 64 --tps-load closed --concurrency 1 \
  --ramp-s 2 --window-s 10 --chips 1 --out /tmp/run
```

It prints the headline and writes `/tmp/run/summary.json` and
`/tmp/run/tps_records.jsonl` (one line per request). For a real model pass the
path of its `tokenizer.json` to `--tokenizer`, use a pool of real text longer
than the input size (the `--mode pool` text is made for the `ws` tokenizer),
and allow `--length-tol` a few percent. The server has to honour
`ignore_eos: true` and stop with `finish_reason: length`; without that every
request is marked invalid. This is a measurement, not a verdict: no seal, no
incumbent, no quality gate.

---

## On the box

Operator commands, in the order they are used:

```bash
sudo -n -u bench /opt/bench/run.sh --lock-ruler          # freeze the judge at clock start
sudo -u bench /opt/bench/run.sh baseline-<name>          # needs /opt/bench/.allow-baseline
env -u SUDO_USER runuser -u bench -- /opt/bench/run.sh --calibrate cal-1  # as root: the launch noise
sudo -u bench /opt/bench/run.sh <id>                     # a candidate (the loop's runner issues this)
env -u SUDO_USER runuser -u bench -- /opt/bench/run.sh --confirm <id>     # as root: re-measure a promoted candidate
sudo -u bench /opt/bench/run.sh --instrument <id>        # one unscored launch with /out mounted
```

Through `sudo -u bench` run.sh accepts only a bare id, `--instrument <id>` and
`--lock-ruler`, and ignores the environment. `--calibrate` and `--confirm` are
the operator's: they are entered from a root shell with `runuser` and no
`SUDO_USER`, a door the researcher cannot open. `--confirm` refuses a
`baseline-*` id: a baseline is the reference, not a promotion.

The target is named by one line in `/opt/bench/TARGET` (an id under
`targets/`). A target with `runnable: false` is refused.

### What a run does

1. Checks the id and refuses to overwrite an old result. Refuses to run when a
   judge file changed since `RULER_LOCK` was written.
2. Takes the exclusive lock (a second run gets the verdict `BUSY`) and starts
   the watchdog: when the run's time limit is gone (`budgets.run_s` for a
   candidate, more for a calibration or a baseline with further reference
   launches) the run is killed and written up as `TIMEOUT`.
3. Copies the experiment and the engine checkouts the target names somewhere
   private, never following a link out of the researcher's home, and stores a
   snapshot (the experiment, the engine diffs, the untracked file list, the
   HEADs) under `results/<id>/snapshot/`. The git that reads the checkouts
   runs in a throwaway container with no network.
4. Builds the image when the experiment brought a Dockerfile, and bakes its
   `deps.txt` into a derived image (into `/opt/venv/vllm` when the image has it, otherwise into the system Python). The build has the network; the launch
   never does. A failed build is `BUILD_FAILED`.
5. Checks that the run can be judged at all (`preflight.json`): stamps that
   differ from the calibration's or the incumbent's are `INVALID`, a missing
   incumbent or calibration is `INCONCLUSIVE`, before anything is launched.
6. Launches. Every launch is a sealed container: no network, no published
   port, read-only mounts (plus a private copy of the compile cache when the
   target has one). A `gpu` target is given every GPU on the machine (`--gpus all`). The judge measures from a sidecar container in the
   server's network namespace. Each launch waits for `/v1/models`, runs one
   unscored warm sweep and then `metric.blocks` scored sweeps.
7. With `metric.screen_window_s` the candidate is first launched alone
   (launch `S`) and screened against the incumbent's stored number, so a
   candidate that is clearly not faster costs one launch.
8. Then the incumbent is launched from its stored snapshot next to the
   candidate, in the order `I C C I` (`design: abba`) or `I C I`
   (`design: ici`), so drift of the machine cancels. The first candidate
   launch also answers the canary; every candidate launch answers the quality
   prompts.
9. `compare.py tps-verdict` decides, `results.json` and `summary.txt` are
   written, the containers are removed, and only then is a `PROMOTED` run put
   on `leaderboard/<target>.json`.

`run.sh` exits 0 for every verdict: a rejection is information, not an error.
Exit 2 means the call itself was wrong (a bad id, a flag passed through sudo,
an id that already has results) and exit 70 means the measurement broke.

### The baseline, the calibration, the confirmation

* **Baseline** (`baseline-<name>`, the operator's): one measured launch `B`.
  It becomes the first entry of the leaderboard and writes the canary
  reference (`canary/<target>.json`) and the quality reference R
  (`reference.dir`). For a real engine R comes from `reference.launches` cold
  launches; the mock's R is one server measured three times.
* **Calibration** (`--calibrate`): `CALIBRATE_LAUNCHES` (10) launches of the
  incumbent. Their spread is `sigma_L`, the launch-to-launch noise, and the
  minimum detectable effect is `max(2%, 3.2 x sigma_L)`. Without a calibration
  every verdict is `INCONCLUSIVE`.
* **Confirmation** (`--confirm`): a promoted run is measured again, `I C I`
  against the previous best, on prompts it has not seen. The leaderboard keeps
  the confirmed effect.

---

## How a change gets promoted

A candidate is **PROMOTED** only when all of these hold:

* both sides were measured by the same judge (`ruler_hash`, `compare_version`,
  `quality_version` agree with the calibration and the incumbent);
* the canary matched: at least 8 of the 10 replies start with the same 16
  words as the baseline's. The canary sends `reasoning_effort: low` by default
  (`CANARY_REASONING_EFFORT` in `harness.py`), so a model that ignores that field
  is compared at its own default;
* every candidate launch is a measurement: no invalid requests, text inside
  the sanity band, and a block left that was not dropped for a compile stall
  (a gap between tokens over `metric.stall_ms`);
* the two incumbent launches agree with each other within the calibrated
  noise, otherwise the verdict is `VOID_DRIFT`;
* the candidate's gain over the incumbent is at least the minimum detectable
  effect and significant at 1% against `sigma_L`;
* no concurrency level lost more than `guard_tol` (3% by default) of its speed
  against the incumbent or against the first baseline (the stock reference),
  and with several input sizes no single size lost more than that either;
* the quality gate passed against R.

A gain that reaches the minimum detectable effect but is not significant is
**INCONCLUSIVE**, and so is any run with no calibration; the current best
stays. A candidate whose gain is below the minimum detectable effect, or that
fails the quality gate or a guard, is **REJECTED** (`REJECTED_SCREEN` when the
screen already said so). A run whose measurement cannot be trusted is **INVALID**.
Other endings: `CANARY_FAILED`, `BUILD_FAILED`, `HUNG` (a sweep outlived its
bound), `TIMEOUT`, `BUSY`, `REJECTED_RESERVED_ID`, `INFRASTRUCTURE_FAILURE`.
The reasons are in `results.json` under `reasons`.

---

## Why the measurements are trustworthy

* **The judge is frozen.** `--lock-ruler` records a hash of the judge's files
  and the target; a run refuses to start when any of them changed. Every
  result carries the stamps, and results under different stamps are never
  compared.
* **The candidate is sealed.** A scored container has no network and nothing
  to write to. `results.json` records under `isolation` what a probe container
  with the same flags could reach. `seal` reads `full` only when the target's
  flags bring no host device, no unix socket, no `--privileged` and no host
  network, and the probe found no shared mount it could write. With a device,
  a socket or `--privileged`, a clean probe is reported as `not_proven`.
* **The incumbent is measured again, beside the candidate.** Speed is never
  compared with a stored number, only with the incumbent started from its own
  snapshot in the same run.
* **The noise is measured, not assumed.** The verdict is held against the
  calibrated launch-to-launch spread, not against the spread inside one run.
* **The canary.** Ten fixed prompts, temperature 0: different weights, a
  different chat template or different sampling defaults change the first
  words of the replies. The check is on whitespace words, so anybody can
  repeat it by eye.
* **Quality is compared with a fixed reference.** R is written once by the
  baseline and is read-only afterwards, so small losses cannot add up over a
  chain of promotions.
* **The server's own account is not believed.** Tokens are counted by the
  client, and the mock server's cheating behaviours (`truncate`, `early_eos`,
  `lie_usage`, `junk`, a response cache and others) are run through the judge
  in the tests and none of them is promoted.
* **One run at a time, under a wall clock.** The lock and the watchdog belong
  to the bench user, and every Docker call the watchdog makes is bounded.

---

## The target file

Everything about the hardware, the engine and the model comes from
`targets/<id>.yaml`. The mock's, in full:

```yaml
id: cpu-mock
runnable: true
accelerator: {kind: none, chips: 1, hosts: 1}
model: {served_name: mock, revision: "0", local_dir: "", tokenizer: "ws"}
engine: {name: mock, image: "inference-judge-mock:latest", docker_flags: [], port: 8000}
metric:
  kind: tps
  isl: 256              # input tokens; a list is a balanced mix of sizes
  osl: 64               # output tokens
  concurrency: [4, 8, 16]
  slo: {p99_tpot_ms: 50, p95_ttft_ms: 2000}
  ramp_s: 2
  window_s: 10
  design: abba          # abba: I C C I, ici: I C I
  blocks: 2             # measured sweeps per launch, after one unscored warm sweep
budgets: {build_s: 600, compile_s: 120, health_s: 120, run_s: 2400}
reference: {dir: "reference/cpu-mock"}
quality: {allow_missing_logprobs: false}
```

`target.py` checks every field and says what is wrong with a file it refuses.
[`docs/DESIGN.md`](../docs/DESIGN.md) is the full description of the fields, of
`results.json` and of the rules above.

---

## Where the results go

```
results/<id>/results.json     the verdict, the reasons, the metric, the stamps, the isolation record
results/<id>/summary.txt      the same for a person
results/<id>/run.log          everything run.sh printed
results/<id>/launches/<L>/    per launch: server.log, launch.json, and one directory per sweep
                              with summary.json and tps_records.jsonl
results/<id>/snapshot/        the experiment and the engine trees as they were measured
leaderboard/<target>.json     the current best and the history
calibration/<target>.json     sigma_L and the minimum detectable effect
```
