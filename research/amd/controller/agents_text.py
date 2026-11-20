#!/usr/bin/env python3
"""The two texts the agents read, from docs/amd-loop-prompt-draft.md (draft 2
of the goal and of the reference speeds, accepted by the author on
2026-11-17), with the facts filled in from the AMD config.

  agents_text.py agents-md <amd config.json>   AGENTS.md for the engine repo
  agents_text.py goal                          the goal of the main agent

Changes to the draft's AGENTS.md, all facts: the model directory comes from
the config; the command times are those measured on 2026-11-17 (NOTES.md
there); ./box has a row; one line says how the working tree reaches the GPU
machine; the score's formula, the fresh prompts and the `cand` account (batch
A of 2026-11-17); smaller weights are allowed and the quality bar is
described, with its limits only once config quality.bar holds them (batch C
of 2026-11-17); the published decode speeds are marked as not comparable;
the bandwidth is the measured copy rate from the config.  From 2026-11-18:
the reference figures point to ./score, with the session's values; the draft
model's folder; what a judged container sees (the weights boundary, NOTES)
and serve's time limit; the measured ./submit time; the two models and the
reasoning levels (the user's decision, a fact about the machine).  The part
"What you may use" is left out (the author's decision of 2026-11-16 dropped
it; the draft still holds it only as a placeholder)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "lib"))
import quality  # noqa: E402

DRAFT = "Qwen3.8-27B-DFlash2"     # reference/setup.sh downloads it next to the model (examples/amd-tuned-engine)
# Prime Agent's selectors of the two models, as in its model catalog (provider anthropic, 2026-11-18); the same
# two are agent.settings.allowedModels in controller.installed.json (a test keeps them equal).
MODELS = ("anthropic/claude-opus-5-5", "anthropic/claude-sonnet-5-5")

GOAL = """\
Write a new inference engine in this repository and make it faster than
every existing engine for this model on this GPU, as measured by ./score.
The engine must be your own design and your own code: you may study any
existing engine, but a copy or a patched version of one does not count.
AGENTS.md describes the machine and the commands. The run has no end point:
do not mark this goal complete.
"""

AGENTS_MD = """\
# This repository

This repository starts empty. It will hold an inference engine for one model
on one GPU. A separate program, the judge, measures the engine.

## The machine
- GPU: one AMD Instinct MI350X, 288 GB, architecture {arch}, as a virtual
  function inside a virtual machine. ROCm 10.0. 24 CPUs, 251 GB RAM.
- The model: {model}. Weights are at `{model_dir}`,
  read-only. `{models}` also holds `{draft}`,
  the draft model AMD's tuned engine (below) uses.
- The GPU machine is rented at spot price and can be taken away with about
  two hours of notice. This repository and `notes/` are stored off the
  machine and survive. Everything else on the machine is lost.

## The rules of the measurement
- The engine is this repository with an executable `serve`. `serve <port>`
  must answer `GET /v1/models` when ready and `POST /v1/completions` with a
  prompt given as token ids. It returns its tokens as ids, either in
  `logprobs` or as `choices[0].token_ids`; the judge does not need the
  engine's own log-probabilities. If an executable `build` exists, it runs
  first.
- Smaller weights (4-bit and the like) are allowed. The reference engine
  runs the model's 16-bit weights, and AMD's tuned engine (below) is held to
  the same quality bar as yours.
{quality}
- The score is for one request at a time: the sum of the output tokens over
  the sum of the whole-request seconds, prompt time included. The timed
  requests are the same requests whose answers are checked. `./check`,
  `./submit` and `./score` also show the decode speed and the prompt time
  apart; those come from extra requests and do not count.
- The judge draws fresh prompts from a hidden pool for each submission. A
  drawn prompt is never sent to an engine in a later submission.
- The judge keeps the best engine that passed. A worse later submission
  does not replace it.
- The judge runs under another account. During `./submit`, `build` and
  `serve` run as the account `cand`, which cannot read the judge's files.
  `cand` reaches only this machine's loopback, not the network.
- Docker: if the repository has a `Dockerfile`, `./check` and `./submit`
  build it first, with no network during the build, and `serve` finds the
  image's name in `$IJ_ENGINE_IMAGE`. Containers start only through
  `sudo /opt/ij-amd/amd/controller/box/ij-docker run` (`--help` lists the
  options it takes): your own account's user id, the host's network, the
  GPU devices, the model read-only at `/models`, no other host directory
  and no added privileges. `/opt/ij-amd/amd/examples/amd-tuned-engine` on
  the GPU machine is an example.
- In its container a judged engine sees `{models}` read-only at `/models`
  and no other folder of the machine, and no network beyond this machine's
  loopback. Smaller weights are therefore made at start-up from those, or
  come with the commit or with the image named in `FROM`. `serve` must
  answer `GET /v1/models` within {ready_min} minutes of its start.

## Commands
| Command | What it does | Time | Uses the GPU |
|---|---|---|---|
| `./gpu` | Shows who holds the GPU and who waits. | about 2 s | no |
| `./ref <prompt>` | Returns the reference engine's answer and token probabilities. | about 2 s if cached; 88 s measured when the reference had to start | on a cache miss |
| `./try <cmd>` | Runs your command on the GPU when it is free. | yours | yes |
| `./profile <cmd>` | Like `./try`, under rocprofv3. Prints the command's wall time, the kernel span, the kernels with the most time and amd-smi samples; the command's output goes to a log file. Hardware counters only with `--pmc`; whether this virtual GPU gives them is not confirmed. Processes inside Docker are not traced. | yours, plus a few seconds | yes |
| `./check` | Checks answers on a small public prompt set. Prints each failure and the speed in parts. | 3 min 12 s measured, before the speed parts were added (they add two more passes over the prompts) | yes |
| `./box <cmd>` | Runs your command on the GPU machine without taking the GPU (compiling, looking at the machine). | yours | no |
| `./submit` | Sends the current commit to the judge. Uses prompts drawn fresh from a hidden pool. | 9 min 03 s measured on 2026-11-18 for plain ATOM; longer when the engine is slower or the machine's reference speed is measured first | yes, alone |
| `./score` | Shows every submission: passed or failed, speed, and the two reference speeds below. `./score N` lists the requests of submission N. | under 1 s | no |

- There is one GPU. Commands that use it wait in one queue.
- `./submit` takes the GPU alone. All other GPU work waits until it ends.
- `./try`, `./profile`, `./check` and `./box` first copy your working tree
  to the GPU machine (not `.git`, not files that `.gitignore` excludes;
  such files made on the GPU machine stay there). Each working tree has its
  own copy there.

## Reference speeds (shown by `./score`)
- The reference engine on this machine: AMD's ATOM from its official image,
  pinned by digest, with one patch that only adds a logprob entry for every
  generated token (marker tokens such as `<think>` included).
- AMD's tuned engine for this model, measured on this machine. It is public:
  repository `proaditya/ATOM`, branch `qwen38-megakernels`. It is the
  fastest known engine for this model on this GPU.
- `./score` shows both speeds as the judge measured them on each GPU
  machine, by the same formula as the score. On the machine of 2026-11-18
  they were about 65 tokens/s for the reference engine and about 260 for
  the tuned engine.
- Published speeds of 700 to 1,000 tokens/s for the tuned engine are decode
  speeds. They leave out the prompt time, so they cannot be compared with
  the score, which includes it.
- The memory-bound speed of named configurations on this GPU: a measured
  copy rate divided by the size of the weights, for BF16 weights and for
  4-bit weights, each read once per token, one request, no speculative
  decoding. The copy rate is {bw:,} GB/s, read plus write, from a plain copy
  kernel on this virtual GPU; it is a measurement, not the GPU's limit.
  These speeds are facts about those configurations, not limits on every
  engine: an engine that reads fewer bytes per token (smaller weights,
  speculative decoding) can be faster.

## Models
Two models are available: `{opus}` and `{sonnet}`.
`rlm.spawn(prompt, name=, model=, thinking=)` sets both per child. A child
spawned without them gets its parent's model and level, which is the first
model at `high`.
- `{opus}` is the stronger model: design decisions, new kernels, hard
  debugging, anything where a wrong answer costs hours.
- `{sonnet}` is faster and cheaper: reading the web and research (papers,
  documentation, source code), running and watching tests, parameter sweeps,
  profiling, reading logs, routine edits, writing notes.

The reasoning levels, from least to most effort: `low`, `medium`, `high`,
`xhigh`, `max`. Lower levels answer faster and cost less; higher levels
think longer before each step. `low` or `medium` suits mechanical work with
a clear procedure; `high` suits most engineering; `xhigh` and `max` are for
problems that resisted a first attempt. (`off` and `minimal` exist, but
Prime Agent's model catalog gives these two models no setting of their own
for them.)

All agents share one usage allowance. Work done on the cheaper model or a
lower level leaves more of it for the hard parts.

## The end of the run
There is no finish line. The run ends when the operator stops it.
"""


QUALITY = """\
- Quality is judged against a fixed bar, with two measures. A: the
  reference engine reads along each of your answers; the share of your
  tokens that are among its top choices, and how much less likely it found
  your tokens than its own first choice. B: your engine and the reference
  answer the same fresh tasks with checkable answers (arithmetic, multiple
  choice), graded by rule; your accuracy is compared with the reference's.
  The judge measures speed only after the answers pass."""

QUALITY_NOT_SET = QUALITY + """
- The limits of the bar are not set yet; they will be set from measurements
  of the reference and of AMD's tuned engine on this machine. Until then a
  submission passes only if its answers match the reference's within a
  fixed tolerance (the same check as before), and `./check` and `./submit`
  show both measures with "quality bar not set"."""


def quality_text(c: dict) -> str:
    if not quality.is_set(c):
        return QUALITY_NOT_SET
    b = quality.bar(c)
    tk = ", ".join(f"top-{k} share at least {v}" for k, v in sorted(quality.topk_limits(c).items()) if v is not None)
    if quality.replaces_agreement(c):
        rule = """
- A submission passes when every judged request is answered and all three
  limits hold. The bar replaces the earlier rule that the answers match the
  reference's within a fixed tolerance; that agreement is still shown, not
  applied."""
    else:
        rule = """
- A submission passes when every judged request is answered, all three
  limits hold, and the answers also match the reference's within a fixed
  tolerance (the earlier rule, still applied)."""
    return QUALITY + f"""
- The limits: {tk}; mean gap at most {b['max_mean_gap']} nats per token;
  for B, the one-sided 95% lower bound of (your accuracy minus the
  reference's) at least {b['min_task_delta_lb']}.""" + rule


def main(argv) -> int:
    if argv[:1] == ["goal"]:
        sys.stdout.write(GOAL)
        return 0
    if argv[:1] == ["agents-md"] and len(argv) == 2:
        c = json.load(open(argv[1]))
        sys.stdout.write(AGENTS_MD.format(arch=c["gpu_device"]["arch"], model=c["model"]["name"],
                                          model_dir=c["model"]["dir"], bw=int(c["gpu_device"]["hbm_bandwidth_gb_s"]),
                                          models=os.path.dirname(c["model"]["dir"].rstrip("/")), draft=DRAFT,
                                          ready_min=round(c["engine"]["health_timeout_s"] / 60),
                                          quality=quality_text(c), opus=MODELS[0], sonnet=MODELS[1]))
        return 0
    print(__doc__.strip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
