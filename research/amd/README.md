# AMD MI350X Judge and Agent Harness (`research/amd`)

Isolated evaluation judge, container sandbox, single-GPU queue, and controller harness for autonomous inference-engine optimization of `Qwen/Qwen3.8-27B` on a single AMD Instinct MI350X (CDNA 4, 288 GB HBM).

This harness graded the 32.3-hour autonomous run that produced [`wanderingspirit03/qwen38-mi350x-agent-engine`](https://github.com/wanderingspirit03/qwen38-mi350x-agent-engine) (commit `657edee`), beating AMD's tuned `proaditya/ATOM` megakernel path by 1.51x across 24 unseen coding-agent prompts under the same fixed quality bar.

## Why this exists

An optimizing agent that controls the server can game weak benchmarks by truncating replies, trading away output quality, replaying cached answers, or cherry-picking lucky draws. This harness separates the agents from the judge at the OS and container level and grades quality directly on the exact timed tokens the candidate produced.

## Security and isolation model

The setup uses two machines (a controller VM and a DigitalOcean MI350X spot box) and three separate Unix accounts:

- **`agents`**: runs `prime-agent` and writes engine code. Cannot read the judge's pools, hidden sets, ledger, or judge internals (`controller/code-access.sh`).
- **`cand`**: builds and runs candidate Docker images during `./submit`. Docker builds run offline; runtime containers launch only through an allowlisted sudo wrapper (`controller/box/ij-docker`) with `/srv/models` mounted read-only, no other host mounts, private IPC, dropped capabilities, `no-new-privileges`, and UID-scoped firewall rules (`controller/box/ij-fw`) restricting traffic to loopback.
- **`judge`**: holds an exclusive lock on the MI350X during grading, times the candidate, stops it, loads the unmodified BF16 reference on the same GPU to grade quality, and cleans up staged files.

## Agent commands

From the engine repository, agents interact with the GPU and judge through eight thin wrappers (`controller/bin/ijc`):

| Command | Role | Uses GPU |
|---|---|---|
| `./gpu` | Shows who holds the GPU, queue state, and VRAM usage | no |
| `./box <cmd>` | Runs a CPU-only command on the GPU machine (compiling, inspection) | no |
| `./try <cmd>` | Runs a command on the GPU in a serialized FIFO queue | yes |
| `./profile <cmd>` | Runs `./try` under `rocprofv3` and `amd-smi` sampling | yes |
| `./ref <prompt>` | Queries the BF16 reference engine (cached answers) | on cache miss |
| `./check` | Runs a local answer and speed check on public prompts | yes |
| `./submit` | Archives `HEAD`, acquires the GPU exclusively, and runs the full judge | yes, alone |
| `./score [N]` | Prints the submission ledger and baseline reference speeds | no |

## Scoring protocol (Protocol 2) and fixed quality bar

Every `./submit` draws 12 fresh coding-agent prompts without reusing earlier sessions (6 from the `8000`-token bucket and 6 from the `12000`-token bucket), warms up every distinct prompt length twice from `warm.jsonl`, and sends the 12 scored prompts serially at temperature zero with raw token IDs and a 128-token output cap.

- **Official score:** total generated answer tokens (excluding any terminal stop token) divided by total wall-clock seconds across all 12 requests, prompt processing included.
- **Diagnostic breakdown:** two post-score passes on left-rotated prompts estimate one-token prompt time (`prompt_s`) and decode speed (`decode_tps`); these are logged for diagnosis only and do not affect the score.
- **Quality bar (all three must hold, with zero failed requests and under 40 divergences per answer):**
  1. **Teacher-forced top-5 reference share >= 0.98** on the candidate's original timed tokens, replayed through the BF16 reference (`lib/quality.py`).
  2. **Teacher-forced mean log-probability gap <= 0.08 nats** per position against the BF16 reference's top-20 log-probabilities (`reference/atom_token_logprobs.py`).
  3. **Paired task accuracy lower bound >= -0.15** (`delta_lb95` from 2,000 bootstrap resamples over 64 fresh math and multiple-choice tasks answered by both candidate and reference).

## Pre-run cheat validation

Before the autonomous run, four deliberate cheat engines (`examples/cheaters/`, `tests/test_cheaters.py`) and repeated baseline draws were tested on an identical MI350X calibration droplet:

| Check | Outcome | Evidence |
|---|---|---|
| `fabricate` (made-up answers) | Rejected | 0.000 agreement with reference walk |
| `replay` (answers cached from earlier prompts) | Rejected | 0.007 and 0.010 agreement on fresh draws |
| `read-judge` (reads judge files from `cand`) | Blocked | Could read none of the judge's files and exited |
| `proxy-ref` (forwards requests to reference) | Blocked | Connection refused; warm-up returned HTTP 502 |
| Honest ATOM through `./submit` | Passed | Agreement 1.0, 64.05 tok/s |

## Baselines

- **BF16 Reference (`reference/`):** pinned `rocm/atom-dev` image (`sha256:ad256a94...`) serving BF16 weights and BF16 KV cache, patched only with `atom_token_logprobs.py`.
- **AMD tuned baseline (`examples/amd-tuned-engine/` + `measurements/2026-11-17/gpu-session/box-build-tuned.sh`):** `proaditya/ATOM` (`qwen38-megakernels`, commit `352229d6`, FlyDSL `0.4.0.dev973`, `incoai/Qwen3.8-27B-DFlash2`), served through `examples/atom-engine/shim.py` and `patch_token_ids.py`. Verified on an MI350X against AMD's single-request benchmark: 1,036 / 852 / 773 tok/s on GSM8K / HumanEval / MT-Bench vs. AMD's published 998 / 844 / 752.

## Layout

```text
research/amd/
├── lib/           Core judge modules (answers, quality bar, compare walk, ledger, ref, engine)
├── controller/    Two-machine controller, GPU queue, ij-docker, ij-fw, ij-judge, stand-in tests
├── reference/     BF16 reference server script, setup script, and top-20 logprob patch
├── examples/      AMD tuned baseline, ATOM shim, and cheat-suite engines
├── prompts/       Public prompts used by ./check
├── tests/         Unit tests, cheat suite, containment checks, and HIP copy probe
├── measurements/  Pre-run MI350X calibration notes and AMD tuned build scripts
├── config.json    Protocol and quality-bar configuration
└── submit, check, score, gpu, try, profile, ref
```

## Running the unit tests

```bash
python3 -m unittest discover -s research/amd/tests -p 'test_*.py'
```

Some containment and firewall tests require Linux utilities (`iptables`, `sudo`, or Docker); the protocol, quality-bar, pool, and scoring tests run directly with Python 3.
