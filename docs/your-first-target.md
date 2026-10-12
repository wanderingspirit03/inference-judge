# Your first target

A target file describes one machine, one model, one engine and one metric. This page shows you how to write `targets/<id>.yaml` for your own hardware, check it, and see what `setup/bootstrap.sh` would do with it before you rent or touch anything.

The judge (the code calls it "the ruler") reads this file and nothing else to learn what to start and how to measure it. No chip, engine flag or model path is written anywhere else in `bench/`.

Everything on this page was checked against `SCHEMA` in `bench/target.py` and the files in `targets/`. When this page and the code disagree, the code is right: `bench/target.py validate` is the test.

## The short version

1. Copy the shipped target closest to your machine (see the table below).
2. Change `id`, the accelerator, the model, the engine and the metric to match your machine.
3. Run `bench/.venv/bin/python bench/target.py validate targets/<id>.yaml` until it says `ok`.
4. Run `setup/bootstrap.sh --target targets/<id>.yaml --plan` to see the install steps.
5. Keep the file `runnable: false` until a real launch of the baseline has answered on your machine.

| file | what it is | how far it has been run |
|---|---|---|
| `targets/cpu-mock.yaml` | the mock server, for a laptop or CI | runs on any laptop |
| `targets/v6e4-gemma4-26b-tps.yaml` | one user's speed on a Google Cloud TPU v6e-4, with vLLM | the model server was brought up by hand on a v6e-4; bootstrap and a judged run have not been run on a TPU, so the file ships as `runnable: false` and its sealed launch is marked NOT PROVEN |
| `targets/gpu-gemma4-26b-tps.yaml` | one user's speed on one NVIDIA GPU, with vLLM | built and tested on a laptop with fakes only; no GPU has run it |
| `targets/sketch-tpu-8chip.yaml` | an example of a target that is not ready to run; every value marked `verify` is a placeholder | not meant to run |

## The smallest valid file

This file validates. Every key shown is required. The `quality` and `health` sections may be left out, and so may every other section not listed here.

```yaml
id: minimal
accelerator: {kind: none, chips: 1, hosts: 1}
model: {served_name: mock, revision: "", local_dir: "", tokenizer: ws}
engine: {name: mock, image: "inference-judge-mock:latest", docker_flags: []}
metric:
  kind: tps
  isl: 256
  osl: 64
  concurrency: [4]
  slo: {p99_tpot_ms: 50, p95_ttft_ms: 2000}
  ramp_s: 2
  window_s: 10
budgets: {build_s: 600, compile_s: 120, health_s: 120, run_s: 2400}
reference: {dir: reference/minimal}
```

Three rules apply to the whole file.

- **The file name must be `<id>.yaml`.** Bootstrap installs the target as `/opt/bench/targets/<id>.yaml`, and the judge finds it by the id in `/opt/bench/TARGET`. The id uses only `a-z`, `0-9` and `-`, starts with a letter or digit, and is 2 to 41 characters long.
- **A typo in a known section is an error.** The judge refuses unknown keys inside `accelerator`, `model`, `engine`, `metric`, `budgets`, `health`, `reference` and `quality`, so a misspelt `window_s` cannot fall back to a default without telling you.
- **Unknown top-level sections pass through.** `provision`, `probes`, `metrics` and anything else you add are kept for the tools that own them (mostly bootstrap). The judge does not check them.

If you are writing a mock target, it also needs `targets/<id>.Dockerfile`, because bootstrap builds the mock engine's image from that file. `targets/cpu-mock.Dockerfile` is the one to copy.

## Validate it

```bash
bench/.venv/bin/python bench/target.py validate targets/<id>.yaml
```

A good file prints `targets/<id>.yaml: ok (<id>, tps, runnable)`, or `NOT runnable` when it says `runnable: false`. A bad file prints the field and the reason, and exits with 1:

```text
target.py: metric.windows_s: unknown key (known: kind, isl, osl, concurrency, slo, ramp_s, window_s, screen_window_s, design, blocks, prompt_pool, confirm_pool, pool_size, length_tol, guard_tol, screen_tol, stall_ms)
```

Create `bench/.venv` with `make -C bench venv` if you do not have it yet. `bench/target.py` needs PyYAML, and the venv has it.

You can also read one value as the setup scripts do:

```bash
bench/.venv/bin/python bench/target.py get targets/<id>.yaml metric.isl
```

## See the plan

`--plan` prints the steps bootstrap would run for your target, changes nothing, and needs no root. Put the venv's `bin` first on `PATH` so that `python3` can import PyYAML:

```bash
PATH="$PWD/bench/.venv/bin:$PATH" bash setup/bootstrap.sh --target targets/<id>.yaml --plan
```

Each line starts with `RUN`, `SKIP` or `STOP`. `docs/RUNBOOK.md` explains them under "Read the plan". In short, the plan shows whether your target is asking for the steps you meant to ask for: the GPU or TPU check, the engine image, the model download, the engine source trees. A step you expected and cannot see usually means a key is missing from the target.

## The sections, one at a time

### id and runnable

```yaml
id: my-gpu-box
runnable: false
```

`runnable` defaults to `true`. With `runnable: false`, `bench/target.py` still validates the file and `bootstrap.sh --plan` still lists its steps, but `bench/run.sh` refuses to measure on it. See "Sketches" below.

### accelerator

```yaml
accelerator: {kind: gpu, chips: 1, hosts: 1, type: "H100", hbm_gb_per_chip: 80}
```

| key | required | what happens to it |
|---|---|---|
| `kind` | yes | The target's own label for its hardware: any word made of `a-z`, `0-9` and `-` validates. The judge acts on one value only. With `gpu`, it starts every launch with `--gpus all`. With any other label, it adds no accelerator flag, and your `engine.docker_flags` must bring the devices. Bootstrap also branches on `tpu` (the steps in `setup/tpu.sh`) and on `none`, as `setup/README.md` explains. |
| `chips` | yes | The number of chips. The harness divides throughput by it to give a per-chip number. For `gpu` and `tpu` targets, bootstrap also checks that the host has this many. |
| `hosts` | yes | The number of hosts. Today every shipped target has one. |
| `type` | no | A name to match. For `gpu`, bootstrap requires every card's `nvidia-smi` name to contain this text, ignoring case, and refuses any other card. Leave it out to accept any card. |
| `topology` | no | A label for the record, for example `2x2`. |
| `hbm_gb_per_chip` | no | Memory per chip in GB. For `gpu`, bootstrap checks each card has at least 95% of this. |

A `gpu` target gets every GPU on the machine, because the judge passes `--gpus all`. Bootstrap therefore requires exactly `chips` GPUs on the host and fails on more or fewer. Use a machine with exactly the count the target names, or write the target for the machine you have.

### model

```yaml
model:
  hf_id: example-org/example-model
  revision: "0123456789abcdef0123456789abcdef01234567"
  local_dir: /opt/models/example-model
  served_name: example-model
  tokenizer: /opt/models/example-model/tokenizer.json
```

| key | required | what happens to it |
|---|---|---|
| `served_name` | yes | The model name the judge sends in each request. Your launch script must serve under this name (`IJ_SERVED_NAME` in the launch environment). |
| `revision` | yes | The model's commit. Bootstrap needs a 40-character commit hash when `hf_id` is set, and refuses `main`. An empty string is valid for a target with no download. |
| `local_dir` | yes | Where the weights are on the box. The judge mounts the host's `/opt/models` into each launch at the same path, read-only, and tells the launch script this value as `IJ_MODEL_DIR`, so the weights must sit under `/opt/models`. An empty string means no model mount. |
| `tokenizer` | yes | The tokenizer the harness uses to build prompts of the right length. The mock target uses the word `ws`. |
| `hf_id` | no | The Hugging Face repository. With `hf_id` and a 40-character `revision`, bootstrap downloads exactly that revision to `local_dir`, checks every file against the Hub's sha256 list, and makes the result read-only. Without `hf_id`, the weights must already be at `local_dir`. |
| `chat_template`, `gcs` | no | Accepted by the schema. Nothing in `bench/` reads them today. |

### engine

```yaml
engine:
  name: vllm
  image: "vllm/vllm-openai@sha256:..."
  sources: [vllm]
  docker_flags: ["--entrypoint="]
  shm_size: 8g
  env:
    HF_HUB_OFFLINE: "1"
  port: 8000
```

| key | required | what happens to it |
|---|---|---|
| `name` | yes | A label for the engine (`mock`, `vllm`, `sglang`, `vllm-tpu`, anything). It also picks the default baseline experiment, `baseline-<name>`, or `baseline-tpu` when `accelerator.kind` is `tpu`. |
| `image` | yes | The Docker image every launch runs in. See "Where the engine comes from" below. |
| `docker_flags` | yes | Extra `docker run` flags for each launch, as a list of words. `[]` is valid. The next list says which are refused. |
| `env` | no | Environment variables for each launch, as a mapping of strings. |
| `port` | no | The port the server listens on. Default 8000. |
| `shm_size` | no | Shared memory size, passed as `--shm-size`. |
| `sources` | no | The engine source trees the researcher may edit. Default: none. |
| `compile_cache` | no | A compile cache the judge owns. See below. |
| `launch` | no | Accepted by the schema. Nothing in `bench/` reads it. The serve command is the `launch.sh` of each experiment. |

**The judge refuses flags that would undo its seal.** It refuses `-p`, `--publish`, `-P` and `--publish-all` (published ports), `-v`, `--volume` and `--mount` without `:ro` or `readonly` (writable mounts; a `tmpfs` mount is allowed), `--network` or `--net` with any value except `none` or `host`, and `--gpus` (which comes from `accelerator.kind`). It accepts `--privileged`, `--net=host` and `--device /dev/...`, because a TPU host may need them. Each of those three is recorded in the run's `results.json` as a weaker seal, so a reader of the result can see it.

**`engine.sources` lists the engine source trees the researcher may edit, and it defaults to none.** Each name is a tree in the researcher's home as `/home/researcher/<name>-src` (for example `vllm` gives `/home/researcher/vllm-src`). A launch sees a private copy of the tree at `/src/<name>`. An experiment can then carry a patch to an engine and have it measured. With `sources` left out, the researcher can change how the server is started and built, but not the engine's code. A name starts with a lowercase letter, continues with lowercase letters, digits or `_`, and may not repeat.

Where those trees come from depends on the engine:

- An engine built on the box (image `inference-judge-base:*`, engine `vllm` or `sglang`): bootstrap clones the trees at the release tags in `IJ_VLLM_TAG` and `IJ_SGLANG_TAG`.
- A prebuilt image on a GPU box: add `provision.sources.<name>` with a `url` and a `ref`, where `ref` is the 40-character commit the image was built from. Bootstrap refuses a tree with no pinned ref.
- A prebuilt image on a TPU: bootstrap copies the trees out of the image.

**`engine.compile_cache`** has two keys, both absolute paths with no spaces: `seed`, a directory on the host that only the judge writes, and `mount`, where a launch sees its own private copy. A candidate keeps its copy from its first launch to its last and then discards it. Only incumbent and baseline launches add to the seed. `mount` may not sit under `/ro`, `/exp`, `/src`, `/out`, `/opt/models` or `/opt/inference-judge`.

#### Where the engine comes from

Bootstrap sorts engines into four classes from `engine.name` and `engine.image`. `--plan` names the class in its first line.

| class | how it is recognised | what bootstrap does |
|---|---|---|
| built | `name` is `vllm` or `sglang`, and the image starts `inference-judge-base:` | Builds the engine from a release tag. You must set `IJ_VLLM_TAG` (and `IJ_SGLANG_TAG` if the target serves or lists sglang) to release tags. The wheels need CUDA 13, so the host driver must be r580 or newer. |
| mock | `name` is `mock` | Builds the image from `targets/<id>.Dockerfile`. |
| pulled | any other image | Pulls the image. On a TPU, loads it from the cache disk's tar if it is there. Pin it by digest (`name@sha256:...`) so results mean one image only; the TPU path requires this. |
| none | no image | No engine image step. |

### metric

```yaml
metric:
  kind: tps
  isl: [100, 1000, 4000]
  osl: 512
  concurrency: [1]
  slo: {p99_tpot_ms: 50, p95_ttft_ms: 2000}
  ramp_s: 5
  window_s: 60
  screen_window_s: 45
  design: abba
  blocks: 2
```

`kind` must be `tps`, the only metric: output tokens per second at fixed input and output lengths.

| key | required | what happens to it |
|---|---|---|
| `isl` | yes | Input length in tokens. One number, or a list of distinct sizes. With a list, each stream is served a balanced mix of the sizes in a fixed order, the same for candidate and incumbent, and the headline is the whole-answer speed over the mix. |
| `osl` | yes | Output length in tokens. The harness asks for exactly this many and marks a request invalid if the server returns another count. |
| `concurrency` | yes | A list of concurrent streams to test, such as `[4, 8, 16]`. With the single level `[1]`, the headline is one user's speed. A higher level makes it a throughput number. |
| `slo` | yes | Ceilings a level must meet to count. It needs both `p99_tpot_ms` and `p95_ttft_ms`, as positive numbers. The headline is the best level, per chip, among the levels that meet the ceilings and have no invalid request. |
| `ramp_s` | yes | Seconds at the start of each level that are sent but not scored. |
| `window_s` | yes | Seconds of each level that are scored. |
| `screen_window_s` | no | If set, a candidate is first launched alone for one window of this length and screened against the incumbent's record. A clear loser costs one launch and never starts the incumbent. |
| `design` | no | `abba` (incumbent, candidate, candidate, incumbent: four launches) or `ici` (incumbent, candidate, incumbent: three launches). Default `abba`. Use `ici` when each launch is dear, for example when it compiles. |
| `blocks` | no | Measured sweeps per launch, after one unscored warm sweep. Default 2. |
| `prompt_pool`, `confirm_pool` | no | Files of prompts, in the judge's own area. `confirm_pool` is used by the held-out `--confirm` run and must not overlap `prompt_pool`. Without `prompt_pool`, the judge draws a fresh pool for each run. |
| `pool_size` | no | The size of that fresh pool. Default 2048. |
| `length_tol` | no | How far a request's token count may differ from `osl`, as a fraction of `osl`. Default 0, which suits the mock's tokenizer. A real tokenizer usually needs a small value such as 0.01. |
| `guard_tol` | no | How far a candidate may fall below the incumbent (and the stock baseline) at any level before it fails the guard, as a fraction. Default 0.03. |
| `screen_tol` | no | The tolerance the screening launch is held to against the incumbent's stored value. Default 0.02. |
| `stall_ms` | no | The longest silence allowed between two tokens of one answer. A block with a stall is dropped. A launch whose blocks all stall is `INVALID`. Default 1000. |

Choose `ramp_s` and `window_s` so that a window holds several whole requests at your speed. The judge counts requests that finish inside the window.

### budgets

```yaml
budgets: {build_s: 1200, compile_s: 1800, health_s: 1800, run_s: 5400}
```

All four are in seconds, and all are required.

| key | what the judge does with it |
|---|---|
| `build_s` | The time limit for building an experiment's image. `0` is valid. |
| `compile_s` | Extra time for a server that compiles on first use. It is added to the warm sweep's limit and to the canary's and the quality gate's request limits. `0` is valid. |
| `health_s` | With `compile_s`, the default time for a server to answer the model-list request. |
| `run_s` | The time limit for one candidate run. A calibration gets the same share for each of its launches. The verdict is `TIMEOUT` when a run exceeds it. |

### health

```yaml
health: {ready_timeout_s: 900}
```

Optional. `ready_timeout_s` is how long a launch may take to answer the model-list request. Left out, it is `compile_s + health_s`. Set it for a server that answers first and compiles during its first requests, so a hung start does not also cost the compile budget. A server that exits is given up on at once whatever this says. The schema also accepts `health.url`, but the judge always sends the model-list request to `127.0.0.1` on the target's port.

### reference

```yaml
reference:
  dir: reference/my-gpu-box
  prompts: /opt/bench/pools/quality-prompts.jsonl
  tasks: /opt/bench/pools/quality-tasks.jsonl
  max_tokens: 128
  launches: 3
```

The reference is the set of answers from the baseline that every later candidate must stay close to. The operator records it once, with the baseline run (`docs/RUNBOOK.md`, "Reference and calibration").

| key | required | what happens to it |
|---|---|---|
| `dir` | yes | Where the reference is stored. A relative path is under `/opt/bench`. |
| `prompts`, `tasks` | no | The quality prompts and tasks. Left out, the judge uses its own. |
| `max_tokens` | no | Tokens per answer in the reference. Default 128, or 64 for the `mock` engine. |
| `launches` | no | How many cold launches of the baseline the reference comes from. Default 3 for a real engine. The mock engine is one server measured three times. |

### quality

```yaml
quality: {allow_missing_logprobs: false}
```

Optional. The quality gate compares a candidate's answers with the reference. Some of its checks need log-probabilities from the server. With `allow_missing_logprobs: false` (the default), a check that is unavailable counts as a failure, so a server that cannot return log-probabilities cannot pass. With `true`, the gate accepts such a server, and a weaker fallback check still has to pass. `tools/tpu_probe_logprobs.py` checks a running server for what the gate needs.

### The optional sections bootstrap owns

None of these is read by the judge, and a typo in one is not caught by `validate`.

- **`provision:`** holds bootstrap's own settings: `min_free_gb`, `min_driver_major`, `sources.<name>.url` and `.ref`, `baseline`, `model_ignore`, and for a TPU the `cache_disk` block. `setup/README.md` has the table. `provision.baseline` names the baseline experiment when the default (`baseline-<engine.name>`, or `baseline-tpu`) does not fit.
- **`probes:`** On a TPU, `probes.present` is a shell command that prints how many chips the host has. A `gpu` target ignores it, because bootstrap counts GPUs with `nvidia-smi`.
- **`metrics:`** is a note. Nothing reads it.

## Sketches: `runnable: false`

A sketch is a target you have written but not yet proved on the machine. Set `runnable: false` and the file does three things:

- `bench/target.py validate` and `bootstrap.sh --plan` work as usual.
- `bootstrap.sh` runs up to the hardware gate (the GPU gate for `gpu`, the chip gate for `tpu`), then stops with exit code 3 and prints what to do next. The run clock does not start and the ruler is not locked.
- `bench/run.sh` refuses to measure on it.

The reason to stop there is that the judge fingerprints the target file when the clock starts. After that, an edit makes every measurement fail with exit code 70, and fixing it means a new run. A sketch lets you do all the editing first. The workflow is:

1. Write the target with `runnable: false` and put a comment at the top saying what is unverified.
2. Run bootstrap. It stops at the gate and prints the next steps.
3. Make every remaining edit now: `docker_flags`, `env`, budgets.
4. Set `runnable: true` and run bootstrap again.

Here is a whole sketch for a single GPU. It validates, and `--plan` lists its steps.

```yaml
id: my-gpu-box
runnable: false
accelerator: {kind: gpu, chips: 1, hosts: 1, type: "H100", hbm_gb_per_chip: 80}
provision:
  min_free_gb: 120
  sources:
    vllm: {url: "https://github.com/vllm-project/vllm.git", ref: "0123456789abcdef0123456789abcdef01234567"}
model:
  hf_id: example-org/example-model
  revision: "0123456789abcdef0123456789abcdef01234567"
  local_dir: /opt/models/example-model
  served_name: example-model
  tokenizer: /opt/models/example-model/tokenizer.json
engine:
  name: vllm
  image: "vllm/vllm-openai@sha256:0000000000000000000000000000000000000000000000000000000000000000"
  sources: [vllm]
  docker_flags: ["--entrypoint="]
  shm_size: 8g
  env:
    HF_HUB_OFFLINE: "1"
  port: 8000
metric:
  kind: tps
  isl: [100, 1000, 4000]
  osl: 512
  concurrency: [1]
  slo: {p99_tpot_ms: 50, p95_ttft_ms: 2000}
  ramp_s: 5
  window_s: 60
  screen_window_s: 45
  design: abba
  blocks: 2
  prompt_pool: /opt/bench/pools/main.jsonl
  confirm_pool: /opt/bench/pools/confirm.jsonl
health: {ready_timeout_s: 900}
budgets: {build_s: 1200, compile_s: 1800, health_s: 1800, run_s: 5400}
reference:
  dir: reference/my-gpu-box
  prompts: /opt/bench/pools/quality-prompts.jsonl
  tasks: /opt/bench/pools/quality-tasks.jsonl
  max_tokens: 128
  launches: 3
quality: {allow_missing_logprobs: false}
```

The image digest above is a placeholder, so this file would fail at the image pull on a real box. The image `ENTRYPOINT` is cleared with `--entrypoint=` because the judge starts each launch as `<image> bash -c ...`. The GPU path as a whole has not been run on real hardware, so read `targets/gpu-gemma4-26b-tps.yaml` for what its author marked unverified before you copy from it.

## The baseline

Every target needs a baseline experiment: `research/experiments/<name>/launch.sh`. It is the plain, documented way to start the server, and every promotion is measured against it. Bootstrap checks that the file exists. The name is `baseline-<engine.name>`, or `baseline-tpu` on a `tpu` target, unless `provision.baseline` names another. The checkout ships `baseline-mock`, `baseline-vllm`, `baseline-sglang` and `baseline-tpu`. For an engine with another name, write `research/experiments/baseline-<name>/launch.sh`, starting from `research/experiments/TEMPLATE/launch.sh`.

Do not tune the baseline. If its default start fails on your machine (for example, the model does not fit at the engine's default context length), give the target its own baseline with the extra flag and name it in `provision.baseline`, so the shipped one stays plain.

## Before you start the clock

- Finish every edit to the file. The ruler lock covers it.
- `validate` says `ok` and `--plan` shows the steps you expect.
- Keep `runnable: false` until the baseline has been launched on the machine and answered.
- Then follow `docs/RUNBOOK.md`.
