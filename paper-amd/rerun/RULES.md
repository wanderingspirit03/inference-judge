# Rerun of the paper's measurements: the rules, fixed before the machine is rented

Written 2026-11-21, before any run. `rerun.py` in this folder implements these rules. `manifest.json` in the
results records the checksums of this file, of `rerun.py` and of every data file.

## What is measured

Three configurations, all on one AMD Instinct MI350X, one request at a time, greedy decoding:

| Name | What it is |
|---|---|
| `amd_tuned` | AMD's tuned engine (ATOM `352229d6`, FlyDSL 0.4.0.dev973, DFlash draft model, 7 speculative tokens, `--max-num-seqs 4`), behind the same shim as in the paper |
| `agent_tau0` | The agents' engine at commit `657edee` with `ENG_LOSSY_TAU=0` (exact verification) |
| `agent_tau030` | The same engine with `ENG_LOSSY_TAU=0.3` (the paper's setting) |

The reference is the BF16 model served by stock ATOM, as in the paper.

Prompts: the three unseen sets of 12 prompts each (36 prompts, 6,324 to 12,017 prompt tokens), 128 answer
tokens. Tasks: one draw of 512 from the judge's task pool (256 arithmetic, 256 multiple choice, seed 20261120),
the same 512 for every configuration and for the reference.

## How it is run

1. **Three rounds.** Each round launches every configuration once, fresh. The order of the configurations in a
   round comes from a shuffle with seed 20261121 + round number.
2. **All three prompt sets in one launch.** Round 1 times the sets in the order 0, 1, 2; round 2 in the order
   1, 2, 0; round 3 in the order 2, 0, 1. Each set gets its own warm-up right before its timed pass (one
   warm-up prompt for each prompt length in the set, sent twice, cut from rows that are never judged).
   So every configuration has three timed answers for each of the 36 prompts, each from a fresh launch.
3. **Prompt time and decode rate.** After all three timed passes of a launch, every prompt is sent again
   rotated by one token with one answer token, and rotated by two tokens whole. These extra requests never
   enter the speed.
4. **Tasks.** In round 1, after step 3, the engine answers the 512 tasks. They are not timed.
5. **Reference.** After round 1 the reference is started once. It is timed on the three sets the way an engine
   is (warm-up, then each prompt once, logprobs on, as in the paper). It is timed again without logprobs on
   prompts rotated by three tokens; this second figure is reported next to the first and does not replace it.
   Then the reference walks along every answer of round 1 (with 20 alternatives per token, as the judge does)
   and answers the 512 tasks.
6. **Rounds 2 and 3** are for speed. Their answers are walked too, in a second reference start at the end.
   An answer equal to round 1's costs no new reference request.
7. **Extra arms**, after the core part of a launch so that they cannot change it:
   - round 3, every configuration: 512 answer tokens on set 1 and 2,048 on set 2 (prompts rotated, so that no
     earlier request is a prefix); 2 and 4 requests in flight on set 0. Speed only. These answers are not
     checked against the reference.
   - round 3, `amd_tuned`: one-token requests through the shim and straight to ATOM, alternating, to measure
     what the shim adds.
   - one more launch of AMD's engine with `--max-num-seqs 1`, sets in round 1's order, speed and walk.

## The short trial before the run

Before round 1, the script is run once in a short form on the same machine to find faults in the set-up: 2
prompts of each set, 16 answer tokens, 8 tasks, round 1 with the extra arms, the reference block. It writes to
its own folder. Nothing from it is a result, whatever it shows, and the run's plan does not change because of
its numbers. If it shows a fault in the script, the fault is fixed and the fix is listed with the results.

## What is reported

- **Speed** of a configuration: answer tokens divided by whole-request seconds (prompt time included), summed
  over the 36 prompts and the three rounds. A final stop token is not counted.
- **Main comparison:** the ratio of that speed, `agent_tau030` over `amd_tuned`, and `agent_tau0` over
  `amd_tuned`. Interval: 95% percentile bootstrap over the 36 prompts (10,000 resamples, the same resampled
  prompts on both sides, seed derived from 20261121). Each round's ratio is reported too.
- **Against the reference:** round 1 of each configuration over the reference's timed pass with logprobs.
- **Quality, measure A** (from the walk): share of answer positions whose token is in the reference's top 1, 2
  and 5, and the mean gap in nats, over the 36 answers of round 1, and per set.
- **Quality, measure B** (tasks): accuracy of each configuration and of the reference on the 512 tasks; paired
  difference with a 95% bootstrap interval, the one-sided 95% lower bound the judge uses, and McNemar's exact
  test; for all tasks and for each kind.
- **The paper's bar**, unchanged: top-5 share at least 0.98, mean gap at most 0.08 nats, lower bound of the
  task difference at least -0.15. Reported as pass or fail for every configuration.
- Start-up time of every launch, GPU temperature, power and clock around every timed pass.

## Rules for things that can go wrong

- A launch that fails is recorded in `failures.json` and tried once more after the other launches of its step.
  A launch that fails twice is reported as failed. No launch is dropped silently and none is run a third time.
- A request that fails inside a timed pass fails the launch.
- If rounds differ in their answers, the count of differing answers is reported, and every round's answers are
  walked. The number of round-1 answers that differ between `agent_tau0` and `agent_tau030` is reported too.
- No engine is started while the reference is loaded or while the GPU still holds memory from the one before;
  the launch fails instead.
- If the machine is taken away, the script is started again on a new machine and continues with the steps that
  have no result file. Results of different machines are marked as such.
- Nothing is rerun because of the number it gave.

## Where this differs from the paper's runs, on purpose

1. **Three sets per launch**, not one launch per set. The start-up of an engine costs about five minutes and is
   not part of the speed; rotating the set order covers position effects.
2. **Reference requests four at a time** during the walks and the tasks, to save GPU time. Before that, set 0
   is answered again four at a time and compared with the serial answers. If any token differs, all later
   reference work is serial. The largest logprob difference is reported either way. The reference's own speed
   is always measured serially.
3. **Tasks and the full quality measures on round 1 only.**
4. **How the engines are started:** as root with Docker directly (`--ipc host`, `seccomp=unconfined`), not
   through the judge's launcher under a separate user. There is no second machine: the script runs on the GPU
   machine itself. This checks the numbers, not the isolation.
5. **A new machine**, so the GPU is not the card of the paper's runs.

## Not in this rerun

- Conversations with several turns (needs new prompts and new judge code).
- Other models, other GPUs.
