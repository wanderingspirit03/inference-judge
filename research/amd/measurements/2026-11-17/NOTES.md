# MI350X checks, 2026-11-17

One DigitalOcean `gpu-amd-base` droplet, `ij-amd-mi350x-checks`. It had one MI350X VF (gfx950, 288 GB), 24 CPUs,
251 GB RAM and Ubuntu 24.04.5, and cost $3/hour from 12:54 UTC. The files in this folder are the logs and scripts
of that session. The box is gone. This note says how to rebuild the same state on a fresh machine.

## Rebuild the GPU machine

1. **Reference.** Copy `research/amd` to `/opt/ij-amd/amd` and the controller's public keys to
   `/opt/ij-amd/keys/{agents,judge}.pub`. Then run `controller/box/setup-box.sh`.
   - `gpu-amd-base` has ROCm but no Docker and no pip. `setup-box.sh` now installs `docker.io`, `python3-venv` and
     `rsync` first.
   - It took 2 m 33 s (`setup-box.log`). It pulls the pinned image
     `rocm/atom-dev@sha256:ad256a94209d2f58bfa6b0a69c555e0f8aab4fd8b76673c3532258803599099c` (70.7 GB; ATOM
     0.1.10.dev141+g273c78de5, torch 2.10.0+rocm7.2.4) and the weights (`/srv/models/Qwen3.8-27B`, 52 GB).
   - DFlash2 draft for the tuned engine: `incoai/Qwen3.8-27B-DFlash2` in `/srv/models/Qwen3.8-27B-DFlash2`.
2. **Full judge (`bench/run.sh`).** Run `ij-judge-install.sh` from a clone at `/root/inference-judge`. It uses
   setup/bootstrap.sh's library functions. There is no agent, harden, clock or RULER_LOCK step.
   - Stand-in pools: the target names `/opt/bench/pools/qwen38-*.jsonl`, which were never built. The run used
     copies of the gemma4 pools:
     - `qwen38-quality-prompts.jsonl` and `qwen38-quality-tasks.jsonl` are byte copies of `gemma4-quality-*`
       (256 prompts, 200 tasks).
     - `qwen38-confirm.jsonl` is a byte copy of `gemma4-real-confirm.jsonl` (96 rows).
     - `qwen38-main.jsonl` is `gemma4-real-main.jsonl` without the 9 rows that fall outside the size bands in
       Qwen tokens (247 rows).
   - Run it after the reference is stopped, because run.sh refuses any card with more than 1 GiB of VRAM in use:
     `install -o bench -g bench -m 0640 /dev/null /opt/bench/.allow-baseline; sudo -u bench /opt/bench/run.sh baseline-atom-mi350x`
3. **AMD's tuned engine** (`proaditya/ATOM`, branch `qwen38-megakernels`, commit 352229d6). Follow
   `blog_benchmark/Dockerfile` with two deviations, because the pinned `flydsl==0.4.0.dev963` wheel is no longer
   published:
   - Set `FLYDSL_VERSION=0.4.0.dev973` in `build_and_repro.sh`. The build then fails at the megakernel with
     "address space ... does not lower", because the checkout's Python frontend does not match the wheel's
     `_mlir`.
   - Fix it with `tuned-setup3.sh`: replace `$FLYDSL_DIR/python/flydsl` with the wheel's own `flydsl` package
     (frontend plus `_mlir`), keep `kernels/` from the FlyDSL checkout, then build `m8` and export the drafter.
     It needs no GPU and took about 1 minute.
   - The result is the image `ij-atom-tuned:dev973`. `tuned-engine-serve` serves it as an engine for
     `./check` and `./submit`.
4. **Agentic pools.** Use `nebius/SWE-rebench-openhands-trajectories` at revision
   35455389ab51bf5e2306bfd436ef72d0f98bf882 (`trajectories.parquet`, sha256 14048dd1...12efa1). Download it onto
   the GPU machine only. Run
   `bench/pools/agentic_pool.py --tokenizer <Qwen3.8-27B tokenizer.json> --prefix qwen38-agentic --sizes 8000,16000,32000`.
   It takes 508 s on one core. Manifest: `qwen38-agentic-manifest.json`. Hashes:
   - main: fdf47bd5c12416574594018a559dc760bf28c1046edafbfe7297d78516a6d882
   - confirm: bcc6936772999fe60722b72efb4f9575ae2b4cddcbfc59b7e0198e02d3587bd8
   - manifest: c5a6f1a3623088c520a988e80c2c95478d7893ff6bbb1fa3b36b03ba33f81368

## Variation of the reference against itself (job 3)

There were three launches of plain ATOM with the pinned image, BF16 and greedy decoding. The overlap noted below
did not show in the numbers.

- **Speed** (`./submit --reference`, the 8 public prompts, 256 answer tokens): 77.58, 77.56 and 77.81 tokens/s.
  The spread is 0.3 %.
  - Each run took 132 to 209 s; the first included the image's first start.
  - Run 1 overlapped a CPU-only FlyDSL build that had the GPU devices attached.
- **Answers** (`./check` of examples/atom-engine, which is plain ATOM again, against the reference): 43 prompts,
  5,087 positions per run.
  - The 43 prompts are the 8 public prompts plus 35 of 48 quality rows. 13 rows were dropped because the
    unpatched reference could not place their answers (see the logprob fix below, which restores them).
  - Strict agreement was 0.99744, 0.99764 and 0.99646. Agreement with ties counted was 1.0 in every run.
  - Partings were 13, 12 and 18, none outside the top 20. The largest parting gap was 0.125 in every run, and the
    most partings in one item was 2.
  - Absolute difference in logprob: mean 0.0034 to 0.0040, p99 0.063, max 0.113.
  - Each check took 178 to 182 s (66 s engine start, 112 s compare).
- **Proposed limits.** These are proposals only. `config.json` is unchanged. They rest on three launches and 43
  prompts, so they give a range, not a distribution.
  - `floor`: 1.0, the reference's measured agreement with itself.
  - `margin`: 0.01 instead of 0.03, so PASS needs 0.99. Today PASS needs `min_agreement` 0.95.
  - `tie_eps`: keep 0.125. compare.py describes 0.125 as the grid of the logits, and every parting gap seen was
    exactly 0 or 0.125, a tie on that grid. `is_close` already uses <= with 1e-9 of slack, so 0.125 counts as a tie.
  - `max_partings_per_item`: 8 instead of 40. The most seen in a 128-token answer was 2.
  - Speed noise: about 0.3 % between launches. Treat a difference under 1 % as no difference.
  - After the logprob fix below, one check on all 56 prompts saw one parting with a gap of 0.25 (q-q-0136-json,
    position 7), two steps of the grid. So `tie_eps` 0.125 is not a bound on what the reference does against
    itself; a fourth launch and the 13 restored rows would be needed before setting it.

## ATOM's missing logprob entries: root cause and fix

- **Cause, shown with one prompt** (`lpdiag.py`, `lpdiag.log`). For `q-q-0001-translate` the chat answer had 127
  logprob entries for 128 tokens. Stepping the greedy answer one token at a time, the token without an entry is
  at position 3: id 248068, `<think>`, bytes `3c 74 68 69 6e 6b 3e`. It is valid UTF-8 (plain ASCII), not a piece
  of a multi-byte character. It is an added token with `special: false`. The model writes it right after
  "German:\n\n" in a raw-text prompt. The same token is missing in `q-q-0007-translate` (position 3) and
  `q-q-0232-json` (position 117).
  - Why ATOM drops it: ATOM attaches logprobs only to visible text (`atom/entrypoints/openai/logprobs.py`,
    `LogprobsProcessor.take_ranges`). Its reasoning filter (`reasoning.py`, `ReasoningFilter`) watches for the
    think markers even when the channel is closed, and removes the marker from the text. No text, no entry.
    Tokens of special ids are skipped outright (`feed`: `if int(token_id) in self.codec.special: continue`).
  - The UTF-8 guess is refuted. ATOM tracks token bytes across partial characters, and all 11 translation
    failures were the `<think>` marker.
  - `q-q-0154-json` is a different case: the reference's greedy answer is the stop token alone (one token,
    finish stop), so the answer is empty.
- **Fix: one patch to ATOM's chat answer** (`reference/atom_token_logprobs.py`, built by
  `reference/build-ref-image.sh` into `ij-atom-ref:ad256a94-lp1` on top of the pinned digest, about 30 s).
  - The patch adds a field `token_logprobs` to each choice of the non-streamed chat answer: one entry per
    generated token id, in order, taken straight from the engine's records. Each entry has `id`, `token`,
    `bytes`, `logprob` and `top_logprobs`, each alternative with its own `id`.
  - Nothing else changes: the text, `logprobs`, sampling and kernels are ATOM's. The patch refuses to apply if
    the code it edits is not found exactly once. It raises ATOM's own `LogprobError` if the records and ids
    differ in length.
  - `lib/ref.py`, `examples/atom-engine/shim.py` and the judge's `atom_echo_proxy.py` read `token_logprobs`
    when it is there. They fail closed: the entries must cover exactly `completion_tokens`, and only a final
    special stop token is dropped. Without the field they keep the old rules.
  - The reference's identity includes the image, so the cache starts fresh.
- **Rerun of the 56 prompts** (`fix/check56.log`, all 13 dropped rows included, 6 m 16 s): 55 of 56 compared.
  - Strict agreement was 0.98673 over 6,707 positions. Agreement with ties counted was 0.98733 (the failed
    item counts against it). There were 5 partings, none outside the top 20, with a maximum gap of 0.25.
  - Every translation row agrees 1.000.
  - It failed only on `q-q-0154-json`: the client refuses an empty answer ("the answer carried no logprobs").
    Comparing the stop decision would need the final stop token to be compared. Today it is never compared, by
    design.
- **Rerun of the full judge baseline on the patched image** (`fix/baseline/`, 15:23:01 to 15:33:40, 639 s).
  - quality2 collect passed: "collected launch -> staged/reference-launches/0", about 4 minutes, no 502 from the
    proxy. The logprob blocker for the judge is gone.
  - Canary 10/10. Speed: one stream 77.5 tok/s for the whole answer, 79.5 decoding; isl 100/1000/2000/4000 at
    78.7/77.5/77.4/76.4 tok/s, TTFT 114/128/177/256 ms.
  - The verdict is still INVALID, for a different reason: block b2 had 1 of 10 requests invalid,
    `length:505!=512` (`real-main-4000-40-python`). The server counted 512 tokens; the client's re-count of the
    streamed text was 505. The first baseline run (`run1-1341`) failed on the same prompt in its warm-up,
    `length:499!=512`; its blocks did not draw it. Both records are in `fix/baseline/invalid_records.jsonl`.
  - Cause, shown (`fix/lenprobe.py`, `fix/lenprobe.log`, `fix/lenprobe-nononce.log`, 15:45 to 15:50): the same
    prompt sent non-streamed to the patched reference, greedy, `ignore_eos`, 512 tokens, thinking off, with 8
    fresh "Request reference" lines as the harness adds them. In 8 of 8 answers the model ends its turn and
    keeps going: it writes `<|im_end|>` (248046) and `<|im_start|>` (248045), which are special tokens, 5 of them
    in 7 answers and 42 in one. ATOM leaves special tokens out of the text, so the client's re-count of the text
    was 504, 505, 505, 505, 505, 504, 505 and 458 against 512 from the server. The target's `length_tol` is 0.01
    (5 tokens), so this prompt fails every time a block draws it. Without the reference line the answer stayed
    in its turn (512 = 512). This has nothing to do with logprobs.
  - Not changed (limits are not mine to change). `count_by: server` alone would not help: `metrics.py` still
    holds the client's re-count to `length_tol` as a cross-check. What would: a larger `length_tol` for this
    target, or a request option that makes ATOM keep special tokens in the streamed text (not looked for, not
    checked). Neither was tried.
  - A first try of the tuned engine's text comparison started at 15:25 beside this run: my trigger matched the
    line "job 5b done". The tuned engine failed to load (the GPU was held) and exited at 15:25:29, during the
    baseline's warm-up. The warm-up and block speeds match the first run, so I read the baseline as unaffected.

## AMD's tuned engine (job 4)

- **Build.** It builds with the FlyDSL fix above. The pinned dev963 wheel is gone, and dev973 works only with its
  own Python frontend.
- **AMD's own repro**, megakernel arm only (cut for time), 777 s:

  | set       | here, MI350X | AMD's README, MI350X | AMD, MI355X |
  |-----------|-------------:|---------------------:|------------:|
  | GSM8K     | 1036         | 998                  | 1091        |
  | HumanEval | 852          | 844                  | 913         |
  | MT-Bench  | 773          | 752                  | 824         |

  The figures are decode tokens/s per request, mean of 128/128/80 prompts at 2048 max tokens. The draft acceptance
  over the run was 47.4 % of draft tokens, 4.32 tokens per forward pass. The regular arm (MXFP4 plus DFlash) was
  not run.
- **Our answer check: it cannot be judged.** In judge mode, `./submit` failed every prompt: "the answer carried no
  logprobs". The megakernel path returns no logprobs, so it fails our check by rule.
  - Why it has none: the megakernel path (`atom/models/qwen38_27b/mono/`) verifies drafts with a row argmax
    over the target's logits (`verdict.py`) and emits token ids only. No code there computes a log-softmax.
  - It also reads **MXFP4 weights** (`mono/mxfp4.py`: E2M1 values with one E8M0 scale per 32, quantized from the
    BF16 checkpoint). It is a 4-bit engine, so its answers should not be expected to match the BF16 reference
    token for token.
  - Three ways to check it, none run:
    1. Patch the verify step to also return the log-softmax top-k of each emitted row. The logits are there,
       but this is AMD's kernel code and could cost speed.
    2. Teacher-force its greedy token ids through the patched reference, which the echo proxy already does
       for quality2. This needs nothing from the tuned engine and measures how often its tokens are the
       reference's greedy choice or a tie. This is the cheapest real check.
    3. Compare the text, as below. This is the weakest.
  - `tuned_textcmp.py` compares the tuned engine's greedy text on the same 43 prompts with the reference's
    cached answers (`tuned/textcmp.log`).
    - The first run stopped at prompt 5 with `TypeError: 'NoneType' object is not subscriptable`. My script
      looked up the cached chat token ids without the prompt's `chat_template_kwargs`, so the cache key missed
      and `ref.cached` returned None.
    - The first four rows differed at tokens 2, 13, 108 and 1. The script then re-encoded the tuned text,
      which can split differently at token boundaries.
    - The fixed script (decode the reference instead, key with the kwargs) was rerun after the fix on all 56
      prompts against the patched reference's cache (`tuned/textcmp2.log`, `tuned/textcmp.json`, 369 s with
      the engine start). Identical: 1 of 56. First difference: median token 2; 14 rows at token 0 and 11 at
      token 1.
    - Of the 30 rows that part by token 3, I sorted them by reading the two texts:
      - 8 part at the `<think>` marker: the reference's decoded tokens keep `\n\n<think>\n`, the tuned engine's
        text drops it (`German:\n\n<think>\nThe user ...` against `\nThe user ...`).
      - 5 more part because the tuned engine starts reasoning in plain text (`\nThe user wants me to ...`) where
        the reference answers at once (`q-q-0039`, `0060`, `0165` translate, `p7-json`, `q-q-0215-retrieve`).
      - 1 is whitespace only (`p5-chat-explain`, a leading `\n\n`).
      - 16 are real word choices from the first tokens (`p1-prose` "expecting it" against "counting the days",
        summaries opening "Based on the provided text", `q-q-0150-arith` answering in Spanish against English).
    - So a text comparison mixes formatting with real differences, and even the real ones are expected from a
      4-bit engine. Option 2 above (teacher-forcing its token ids) is the real check.
  - AMD says the megakernel is not bit-reproducible run to run.
- **Our speed method.** These are streamed chat requests, one at a time, `ignore_eos`, 220 tokens, one untimed
  warm-up per pool (`agentic_measure.py`). The same client was used for both engines. Speed is 220 / median request
  seconds; mean figures are in `atom.json` and `tuned.json`.

  | prompts            | median prompt tokens | plain ATOM whole request | tuned whole request | ATOM decode | tuned decode |
  |--------------------|---------------------:|-------------------------:|--------------------:|------------:|-------------:|
  | 8 public (7 timed) | 73                   | 76.7                     | 570.6               | 79.6        | 713.6        |
  | agentic 8K         | 7,503                | 73.5                     | 471.5               | 78.5        | 813.5        |
  | agentic 16K        | 14,836               | 63.0                     | 227.6               | 77.8        | 739.0        |
  | agentic 32K        | 31,715               | 45.4                     | 96.5                | 76.4        | 700.1        |

  - Time to first token is the same for both engines: 0.1 / 0.2 / 0.7 / 2.0 s. Prefill is not sped up, so on long
    inputs the tuned engine's lead shrinks from about 7x to 2x.
  - Outliers, left out of the median figures: the first timed public request took about 12 s to its first token
    on ATOM and 10.5 s on the tuned engine (a first compile). One 16K request on the tuned engine took 58 s to its
    first token.
  - The cause is a one-time compile, not draft misses. That request's prefill ended in a chunk of exactly 1 new
    token (prefix cache hit, then 6,880 tokens, then 1). It was the first prefill tail of length 1, and torch
    dynamo and the FLA kernel warned for the new shape at 14:32:09. The engine said nothing more until 14:33:07.
  - Its decode took 0.34 s, as the others did. The mean whole-request speed at 16K in `chain.log` (27.0 tok/s)
    includes those 58 s; the median figure above (227.6) does not.
  - A `./submit` on agentic prompts would hit such first-shape compiles unless the engine warms them up or the
    warm-up covers them.
  - Draft acceptance on this traffic: the engine prints it every 1000 forward passes, and only one line fell in the
    run. It covers the public prompts, 8K and part of 16K: 46.3 % of draft tokens, 4.24 tokens per forward pass.
  - The ./submit-style reference speed (256 tokens, no `ignore_eos`) was 77.6 tokens/s, against 76.7 here.

## Proposal (not applied): ./submit drawing from the traces

- **Speed.** Use whole requests from `qwen38-agentic-main` at 8K, 16K and 32K: greedy, `ignore_eos`, 220 answer
  tokens, streamed, one request at a time. Run one untimed warm-up, then 4 timed requests per size, and report
  tokens/s per size and their mean.
  - Plain ATOM takes 3.0, 3.4 and 4.8 s per request at those sizes, so about 15 x 3.7 s, roughly 60 s per
    `./submit`. A faster engine takes less.
  - Today's speed run (8 short prompts, 256 tokens, no `ignore_eos`) takes about 30 s.
  - The current speed run cannot simply be pointed at these pools. Without `ignore_eos` the model answers a raw
    agentic prompt in about 2 tokens: the 8K run gave 34 tokens over 16 requests, which is meaningless. The speed
    run would need `ignore_eos` and chat requests.
- **Hidden answer-check prompts.** Draw them from sessions that `agentic_pool.py` skips: it keeps one session in
  `SAMPLE_MOD` = 4, so the other three quarters are in no public pool. Also exclude every repository in main or
  confirm. Use 8K requests, 128 answer tokens, about 8 prompts.
  - The cost is about 2 x 8 x (0.2 s prefill + 1.7 s decode), roughly 30 s, plus the reference's start when its
    answers are not cached. The reference's answers can be cached once per hidden set.
  - Risk: agentic answers contain tool calls and think markers (`<tool_call>`, `<think>` and `</think>` are
    added tokens, not marked special). With the patched reference image (`token_logprobs`, see the logprob fix
    above), every generated token has its entry whatever the parsers do with the text, so no prompt needs to be
    dropped. This is untested on agentic answers.
- Keep the eight short public prompts for `./check`.

## Rebuild the controller's link to it

Only public keys moved. The GPU machine got the controller users' public keys from `/var/lib/ij-ctl/pubkeys/`. The
controller got the GPU machine's public host key, written as `ij-box <key>` in `/var/lib/ij-ctl/box_known_hosts`.
Without the driver, `box.json` is written by hand with `state ready`, the IP, port 22 and a made-up id. The old
file is kept as `box.json.before-checks`. The driver unit stays disabled.

## The controller path (job 5)

These ran as `agents` from two working trees with the ATOM test engine and `tests/hip_probe.cpp` committed
(`job5.sh`, `job5b.sh`; logs in `job5/` and `job5b/`).

| step | result | time |
|------|--------|------|
| `./gpu` | GPU free, machine ready | 2 s |
| `./box` (hostname, ls, hipcc --version) | ok | 3 s |
| `./try` (hipcc + hip_probe) | copy 4,703 GB/s read+write, fma 34.9 TFLOP/s fp32 | 5 s |
| `./profile` (the same) | 221 dispatches of 3 kernels, rocprofv3 CSV and amd-smi copied back | 7 s |
| `./ref --chat ... --max-tokens 32`, first run | FAILED: "logprobs cover 31 of its 32 tokens" (the `<think>` gap) | 88 s |
| `./ref`, after the fix | ok | 88 s (66 s reference start) |
| `./ref`, again (cached) | ok | 2 s |
| `./check`, first run | FAILED: `serve` named `../../reference/atom-serve.sh`, which a copied tree does not have | - |
| `./check`, after the fix | PASS, agreement 1.0 over 792 positions of 8 prompts | 3 m 15 s |
| two `./submit` started 5 s apart | both PASSED, agreement 1.0 on 48 hidden rows, 78.3 tokens/s | 617 s and 872 s from the start |
| `./score` | lists both, reference 78.2 tokens/s on this machine | 0 s |

- The first `./submit` measured the reference speed for the machine: 78.2 tokens/s, 3 minutes. It then checked
  and timed the engine, 7 m 06 s in all. The second waited in the queue and ran in 4 m 07 s once the GPU was
  free. The two did not mix.
- The hidden set was the 48 quality rows of the variation prompts, the 13 previously dropped rows included. It
  was installed as `/var/lib/ij-judge/hidden.jsonl` for the test and removed afterwards. The submit log shows
  hidden rows by number only.
- The fixes this needed:
  - the reference image patch above;
  - `examples/atom-engine/serve` and `shim.py` fall back to the GPU machine's `/opt/ij-amd/amd` copy when they
    run from an agent's tree;
  - the shim now fails closed when ATOM's entries do not cover every token.
- `hbm_bandwidth_gb_s` in `config.json` is now the measured 4,700 GB/s (a plain copy kernel, a lower bound), not
  the spec sheet's 8,000. The memory-bound lines become 92.2 tokens/s for BF16 and 345.6 for 4-bit. Plain ATOM
  decodes at 79.6, 86 % of the BF16 line.
- `./score` on the controller still shows the old label and the one-line memory limit until the judge writes
  its score list again. The updated `ij-judge` is installed.
- `q-q-0154-json` (the reference's answer is the stop token alone) failed `./check` as an empty answer. As
  hidden row it showed agreement 1.000 in `./submit`. Which code path treats the empty answer differently was
  not looked into.

## Judge soundness (batch A): what needs the real card, and what is not decided

Protocol 2 is tested only on stand-ins: the unit tests in `tests/` and `controller/tests/e2e-standin.sh` on the
controller VM. Nothing below ran on an MI350X.

### What changed

- **One request shape.** The engine answers each judged prompt once. Those same answers are timed and checked. The
  reference walks the engine's tokens after the engine is stopped. Speed is the sum of the answer tokens over the
  sum of whole-request seconds. The final stop token is not counted. `usage` is not trusted.
- **Fresh prompts.** Each judged run draws `judge.draw` rows per bucket from the pool in `ij-judge`. Each row
  comes from a session that no run has drawn before. A judged run is a submission, a rerun of the best engine, or
  a reference measurement. A drawn session is never drawn again. The run is refused when a bucket is used up, and
  also when there are no warm-up rows.
- **The `cand` account.** `build` and `serve` run as `cand`. It is in the render and video groups. It is not in
  docker or ijgpu. Its home is 700, and it has no cron or at. `cand` cannot read `/home/judge` or
  `/var/lib/ij-amd/judge` (the drawn prompts, the judge's copy of the source, the reference's cache).
  - After `serve` stops, every `cand` process is killed.
  - The run fails if a `cand` process survives, or if port 8100 or 8001 still answers.
  - It also fails if a `cand` process appears while the reference runs.
- **Speed results from protocol 1** (78.2 reference, 78.3 for #1 and #2) are shown in `./score` with a "p1" mark.
  They are not compared with protocol 2. A machine whose reference was measured under protocol 1 is measured again
  before its next submission.

### To do on the next GPU machine

1. **Build the hidden pool.**
   - `ij-judge prepare-pool FILE` tokenizes the rows with the reference on the machine. `--warm` does the same
     for the warm-up rows.
   - The rows must be ones agents cannot see. If agents can get a pool's rows, an engine can carry the
     reference's answers to every row and pass any fresh draw from it.
     - `bench/pools/qwen38-main` and `qwen38-confirm` are not in the agents' repository or on the controller
       today. Whether they are published anywhere agents can reach (for example a public dataset) is not known
       here. That is a question for the coordinator before they are used.
   - Another source is trace sessions that no pool holds, for example the three quarters that `agentic_pool.py`
     skips (`SAMPLE_MOD` = 4). Leave out every repository that appears in main or confirm.
   - **Speed on those rows is not measured.** Under protocol 2 the judged answers are also the speed sample, and
     `ignore_eos` is not used.
     - The section above found that a raw agentic prompt is answered in about 2 tokens (34 tokens over 16
       requests at 8K). Such a sample measures prefill, not decode.
     - The isl-bucket rows of `qwen38-main` are plain prompts. Their answer lengths under the reference are not
       measured either.
     - Before a pool is installed, run the reference over a sample of it and keep only rows whose answers reach a
       useful length. Another option is to decide that speed requests use `ignore_eos`; that is the user's call,
       and it changes what `compare_stop` sees.
   - The buckets must fit the reference: `--max-model-len 16384`. A 4000-token bucket plus 128 answer tokens fits.
     8K and 16K agentic prompts also fit, and 32K does not.
2. **Re-measure the reference speed under protocol 2.** `ij-judge remeasure` does it; the first submission does
   it too.
3. **Run an honest engine under `cand` on the real card.** The Docker-based `examples/atom-engine` cannot run as
   `cand`, which has no Docker access (see the Docker design below).
4. **Check that `judge` sees `cand`'s GPU processes.** `gpuqueue.kfd_users` reads `/proc/<pid>/fd`, and `judge`
   cannot read that for another uid. So `wait_alone` and the queue may not see `cand`'s, or `builder`'s, open
   `/dev/kfd`. `E.user_pids` reads only the owner of `/proc/<pid>`, which works. This has not been checked on the
   card.
5. **Run `tests/box-queue.sh` and `tests/box-submit.sh` on the card.** They need it and were not run for this
   batch. `box-submit.sh` uses the local `./submit`, without the judge's pool.

### Cost per submit

This is an estimate from the 2026-11-17 measurements, not a measured figure.

**Inputs:**
- N = 16 judged prompts (4 per bucket);
- M = 128 answer tokens;
- L = the number of distinct prompt lengths, up to 16;
- v = about 78 tokens/s decode for plain ATOM.

**Formula:** T ≈ engine start + (2 L + N) x (prefill + M / v) + reference start + N x (prefill + M / v) +
partings x (prefill + M / v).

**Plain ATOM:**
- engine start: about 60 to 120 s;
- engine side: 48 x about 1.8 s, about 85 s;
- reference start: 66 s;
- walk: 16 x about 1.8 s, about 30 s;
- total: about 4 to 5 minutes.
- Batch B adds the speed parts: `submit.speed_repeats` (2) x (16 x about 1.8 s + 16 short requests), about
  1 minute more on plain ATOM. Total about 5 to 6 minutes. `speed_repeats` = 0 takes it back to 4 to 5.

**Other costs:**
- A faster engine shortens only the engine side.
- Each new machine adds one reference measurement, about 48 x 1.8 s. Its start is shared: either the driver's
  `ij-judge remeasure` at idle time (66 s start of its own), or, when a submission comes first, inside that
  submission's reference start (no second start).
- It also adds one rerun of the best engine.
- Fresh prompts mean the reference cache never hits for judged rows.

### Pool size

Each run uses k sessions per bucket, with k = 4 now.

**Sessions per bucket per week** = k x (submits per week + machines per week). Each machine costs one rerun of
the best engine; its reference measurement gives its draw back (see below).

**Example:**
- 20 submits a day and one machine a day;
- 4 x (140 + 7) = 588 sessions per bucket per week;
- 2,352 per bucket for four weeks;
- about 9,400 rows in all.

**What is visible today:**
- `qwen38-main` has about 60 rows per bucket, about 15 runs.
- `ij-judge status` prints what is left.

**What does not count against the pool:**
- A reference measurement gives its draw back, because no engine under test saw those prompts.
- A submission gives its draw back when no judged prompt reached the engine: `build`, `serve` or the warm-up
  failed (`times.judged_sent` = 0 in the result).
- A submission that reached the engine and then failed still uses its 16 sessions.

### Draw noise

The reference and each submission are timed on different draws, so the score now varies with the draw. A
resubmit of the same engine gets a new draw. The best of several resubmits rises with no change to the engine.

- On the card, measure the reference over several draws (`ij-judge remeasure` a few times, or `./submit
  --reference` with several drawn files).
- The spread across draws is the smallest gain that means anything.
- Stratifying by bucket (4 per bucket) narrows the spread; how much is not measured.

### Backstops not exercised

- The checks "port 8001 answers after the engine stopped" and "a `cand` process exists while the reference
  runs" never fired in any test. The proxy cheater fails earlier, in the warm-up, and its squatter is killed with
  the rest of `cand`. These are untested backstops.

### Not decided (left as they are, for the user)

- **`limits.compare_stop` (true).** It compares where the answers end: an engine that stops early or late parts
  from the reference there.
  - It is an assumption the user has not confirmed. `false` in `config.json` switches it off; the positions are
    then not counted.
- **The speed run's `max_tokens`.** `submit.speed_max_tokens` (256) and `speed_prompts` are gone, because speed
  now comes from the judged answers. Those answers use `submit.max_tokens` = 128, unchanged.
  - Whether 128 is the right length for the speed measure is the user's call.
- **16 prompts per run** (`judge.draw`, 4 per bucket) is a proposal.
- **No pass limit was changed,** and nothing was decided about what "correct" means for 4-bit weights.
- **Docker for builder on the GPU machine.** Built in batch C2 (see "Containment (C2, C3)" below); the text here
  is what was written before it. `builder` is still in the docker group, which is root on the machine
  in effect, so `builder` can read the judge's files. The `cand` isolation therefore holds against the engine
  under test, not against the agents' own `./box` runs.
  - Design, not built: a root-owned launcher (`ij-docker`, through sudo) that runs only images built from the
    engine's tree, with `--network none` (or loopback only), `--user` set to the caller's uid, no host mounts
    besides the engine directory and the model (read only), `/dev/kfd` and `/dev/dri`, and a label carrying the
    GPU job token.
  - `cand` would use the same launcher, so a Docker-based engine can be judged.
  - Stand-in test: the launcher refuses `-v /:/host`, `--privileged` and `--network host`.
- **Network egress for `cand`.** Built in batch C3 (see "Containment (C2, C3)" below); the text here is what was
  written before it. `cand` can still open connections out. An engine could send each prompt to a
  faster machine elsewhere and pass the answers back as its own; fresh prompts do not stop that.
  - The fix is a firewall rule (`iptables -m owner --uid-owner cand` allowing loopback only). That is a security
    setting, so it was not applied.
- **perf_event_paranoid** and other security settings were not touched.
- **The full judge's length check in `bench/`** and **`./ref --dump`** were not touched.

## Useful commands (batch B): what needs the real card

Nothing here was run on the card; all of it was tested with the stand-ins and made-up rocprofv3 files.

### What changed

- **`./check`** prints the speed in parts after the PASS or FAIL line: whole-request tokens/s by the judge's
  formula, decode tokens/s and the prompt time, each as min-max over the passes.
  - The parts come from extra requests after the checked answers: each prompt rotated by a few tokens (same length,
    not a cached prefix), once with `max_tokens=1` (the prompt time) and once whole, `check.speed_repeats` (2)
    times. The checked answers are one more sample of the whole-request rate.
  - Decode = sum of (tokens - 1) / sum of (request seconds - prompt seconds).
  - The report keeps every request (`report.json`, key `speed`).
- **`./check` fails fast.** When a prompt's token ids are not cached, `serve` is started once and stopped before
  the reference tokenizes; a `serve` that cannot start fails before the reference starts. This costs one more
  engine start on the first run only (the ids are cached after).
- **`./submit`** does the same parts after the judged answers (`submit.speed_repeats`, 2; 0 switches them off).
  They never change the score. The result keeps them under `speed.parts`, and the per-request list stays under
  `speed.per_request`.
  - `./submit` refuses prompts without token ids (given or cached), before it takes a submission number, so the reference starts once per run. Under
    the judge every drawn row has ids, so this changes nothing there.
- **`./score`** (and the judge's public list) prints, for each passed protocol-2 submission, the score line and
  the parts. `./score N` prints submission N's requests one by one.
- **`./profile`** writes the command's output and rocprofv3's to `run.log`. The summary shows the command wall
  time, the kernel span and GPU busy (as a share of the span) apart; counters per pass and per kernel with units;
  GB/s only from `FETCH_SIZE` and `WRITE_SIZE` with the same dispatch's duration; percentages as mean, min and
  max, never summed. The help says a persistent kernel is one row and that Docker processes are not traced.
- **AGENTS.md** (`controller/agents_text.py`): measured times, the score's formula, `cand`, the old-method
  figures 78 and 570 marked, published decode speeds marked as not comparable, and the bandwidth as a measured
  copy rate. install.sh (install-vm.sh) rewrote `/home/agents/engine/AGENTS.md` on the controller and committed it there, as it does on every install.

### To do on the next GPU machine

1. **Cost of the parts.** Each repeat is about one more pass of the answers (plus 16 short requests). With 2
   repeats, `./check` and `./submit` take about 2 x 16 x 1.8 s = about 1 minute more on plain ATOM. Measure it.
2. **Rotated prompts.** Check that the rotated prompts do not hit the engine's prefix cache in a way that makes
   the parts faster than the judged answers (compare the "judged" sample with the repeats).
3. **Counters on the virtual GPU.** Run `./profile --pmc "FETCH_SIZE WRITE_SIZE GRBM_COUNT MemUnitBusy" ./hip_probe`
   and check: does the VF give the counters at all; are `FETCH_SIZE` and `WRITE_SIZE` in KiB on gfx950; does
   `counter_collection.csv` carry `Start_Timestamp` and `End_Timestamp`, or does the summary fall back to the
   kernel trace by dispatch id; does a multi-pass run write one directory per pass. Compare the GB/s with the
   4,700 GB/s copy figure.
4. **`./check` time.** 3 min 12 s was measured before the parts; measure it again, and the fast-fail path with a
   broken `serve` and an empty token-id cache.

### The reference starts once per submission

- On 2026-11-17 (job5b, submission #1) the reference started twice in one `./submit`: once for the machine's
  reference speed (`ij-judge` ran `./submit --reference` first) and once for the walk, 66 s each.
- Now, while a machine has no reference speed under protocol 2, `ij-judge` gives the submission a second draw
  (`--ref-hidden`). After the walk, with the reference still up, `./submit` times the reference on that draw the
  way an engine is timed (warm-up rows, then each prompt once) and returns it as `reference_speed`. The judge
  stores it for the machine and gives that draw back to the pool.
- If the engine never reaches the walk, the reference is not started and a later run measures it.
- `ij-judge remeasure` (the driver, after a new machine) still measures it in a run of its own when none is
  stored, at idle time.
- Tested end to end with the stand-ins (`reference_starts` = 1). On the card, check that timing the reference
  right after its walk does not differ from a fresh start (the walk warms it; the warm-up rows run anyway).

## Quality bar (batch C1): the mechanism, with no limit set

The user chose a fixed quality bar: engines may use smaller weights (4-bit and the like), and AMD's tuned engine
must pass the same bar. This batch builds the measures and the gate. **No limit is set**; `config.json`
`quality.bar` holds `null` for each. Nothing here ran on the card.

### What changed

- **Measure A, from the walk** (`lib/compare.py` `measure_a`). At each position of an engine answer the reference
  already scores its top 20. Recorded per position: the rank of the engine's token in the reference's top, and the
  gap (the reference's best log-probability minus that of the engine's token). A token outside the top 20 gets
  the smallest gap it can have (best minus the 20th) and is counted as outside.
  - `agreement_strict`: share of positions where the engine's token is the reference's first choice.
  - `topk_share` for k = 1, 2, 5 (`quality.topk`): share of positions whose token is in the reference's top k.
    The denominator is every judged position; positions never walked (an answer cut at the parting limit, or a
    failed request) count as outside.
  - `mean_gap`: nats per walked position.
- **Measure B, task accuracy** (`lib/quality.py`). After all timed requests, the engine answers the run's tasks
  (one request each, untimed, not in the speed). After the walk, the reference answers the same tasks. Both are
  graded by the rules of `bench/quality2.py` (math: the last number; mc: the letter), ported and checked against
  quality2 by `bench/tests/test_judge_tasks.py`. The result is the paired difference (engine minus reference)
  with a one-sided 95% bootstrap lower bound (2,000 resamples, seeded from the task ids).
- **Tasks.** `bench/pools/judge_tasks.py N` makes task rows from `pool_tasks.make_tasks` (half math, half
  multiple choice, four languages, thinking off). Without `--seed` the seed is random and not kept. It stays in
  `bench/` because `research/amd` is readable on the agents' machines and `pool_tasks.verify` solves every
  family. One seed gives about 10,000 distinct tasks (20,000 fails: the `percent` family runs out); a second seed
  may repeat a question, and `prepare-pool --tasks` skips a row whose token ids are already in the pool.
  - The judge keeps the task pool in `/var/lib/ij-judge/tasks.jsonl` (`ij-judge prepare-pool FILE --tasks`).
    Each run draws `judge.task_draw` (64) tasks never drawn before, with the same run id as its prompts, so a
    released run frees both. The gold answers go only to the judge's run directory on the machine.
  - While the bar is not set, a run with too few tasks goes on without measure B. Once it is set, the run is
    refused, like an empty prompt bucket.
  - `./check` uses the 16 public tasks in `prompts/public-tasks.jsonl` (`judge_tasks.py 16 --seed 20261117
    --prefix public`).
- **The gate** (`answers.decide`, `quality.verdict`). The bar is set only when `max_mean_gap`,
  `min_task_delta_lb` and at least one `min_topk_share` are numbers.
  - Not set: `./submit` and `./check` record both measures, print them, and print "quality bar not set". PASS
    or FAIL is the agreement limit's, as before (`limits` unchanged: 0.95, tie 0.125, `compare_stop` true).
  - Set: the run fails on any limit missed, on a cut or failed answer, or when measure B was not taken. With
    `quality.bar_replaces_agreement` true (default) the agreement limit is then reported, not applied; false
    applies both. Request failures always fail. The stop comparison stays in the walk and so in measure A.
- **The engine contract** can drop logprobs: an answer with `choices[0].token_ids` and no logprobs is accepted
  (measure A and the walk use only the reference's numbers). The request shape is unchanged (it still asks for
  `logprobs` 5), so an engine that returns logprobs works as before.
- **AGENTS.md** says smaller weights are allowed, describes both measures, and says the limits are not set yet
  and that the agreement check applies until then. It names limits only once `quality.bar` holds them.
- **The public score** shows quality per run (per-task rows stay in the judge's own result file), and the
  measured speed of runs that did not pass, marked "not ranked".

### Stand-in figures (toy model, not the card)

| engine | old agreement | top-5 share | mean gap | tasks right (of 48; reference 48) |
|---|---|---|---|---|
| honest | 1.0 | 1.0 | 0.0 | 48 |
| quantised (`fake_engine --quantised`, noise 0.3) | about 0.85 | 1.0 | about 0.06 | about 47 |
| damaged (noise 3) | about 0.17 | about 0.40 | about 2.0 | about 9 |
| fabricate | | about 0.47 | about 1.6 | 48 (its short answers are honest) |
| replay on fresh prompts | | about 0.23 | about 2.5 | lower bound about -0.98 |

The stand-in limits in `tests/test_quality_bar.py` (top-5 0.9, gap 0.5, lower bound -0.15) are for the toy model
only.

### Setting the bar (the user's decision; the exact procedure)

1. On a GPU machine with the task pool loaded (the GPU session below), run each of these through the judge's
   path, on at least 5 separate draws each (each run draws fresh prompts and 64 fresh tasks):
   - the reference against itself: `examples/atom-engine` (plain ATOM, the reference's image, as an engine);
   - AMD's tuned engine (`ij-atom-tuned:dev973`, the wheel's own frontend).
2. Read from each result: `quality.a.topk_share`, `quality.a.mean_gap`, `quality.a.answers_cut`,
   `quality.b.delta` and `quality.b.delta_lb95` (`ij-judge quality` prints them per submission).
3. Points to weigh (the numbers are the user's call):
   - The reference against itself shows the noise floor: its top-k share is below 1 and its gap above 0 only
     through nondeterminism, and its `delta_lb95` shows how low the bound goes by chance with 64 tasks.
   - Every tuned-engine draw should pass; a limit that fails one of them fails AMD's own engine.
   - Any margin past the tuned engine's worst draw is room for smaller weights. Measure B's bound at 64 tasks is
     coarse (one task is 0.016); more tasks per run (`judge.task_draw`) narrows it and costs time.
4. Write the numbers into `research/amd/config.json` `quality.bar` (all three kinds), keep or change
   `bar_replaces_agreement`, commit, and run `controller/install.sh`. The install rewrites AGENTS.md with the
   numbers. From then on every judged run needs `task_draw` unused tasks: check `ij-judge status`.
5. Earlier passed submissions were judged by agreement only; whether to rerun the best one under the bar is the
   user's call.

### Public task sets (not used; not downloaded here)

Sizes as published, not checked here: GSM8K test 1,319 (train 7,473), MATH 12,500, MMLU test 14,042, MMLU-Pro
12,032. Two days at 64 tasks a run and up to about 500 runs need about 32,000 tasks: only MMLU and MMLU-Pro
together come close. Public sets have one more problem: their answers are public, so an engine could carry an
answer table and pass measure B whatever its weights. The generated tasks have neither problem.

## Containment (C2, C3): security settings the user was told about

Both are on by default in `controller/box/setup-box.sh`. They were tested on the stand-in machine only
(`controller/tests/e2e-standin.sh`, section 5c, 96 passed on the controller VM), not on the card.

### The Docker launcher (C2)

- **What it is.** `controller/box/ij-docker`, owned by root, run through sudo. `builder`, `judge` and `cand` may
  run it. `builder` leaves the docker group (a member of it is root on the machine in effect); `cand` was never in
  it. `setup-box.sh --no-docker-launcher` keeps the old way (`builder` in docker, no launcher).
- **Builds.** `builder` builds `ij-builder/engine` from its own tree; `judge` builds `ij-cand/engine` from its
  copy of a submission (`build --for cand`); `cand` cannot build. The context is packed by tar under the caller's
  uid, and the build runs with `--network none`. `FROM` images are pulled as usual, and a local image (for example
  `ij-atom-tuned:dev973`) is used as it is.
- **Runs.** Only the caller's own image, with `--pull never`. Every container gets the caller's uid and gid, the
  render and video groups, `/dev/kfd` and `/dev/dri`, `--cap-drop ALL`, `no-new-privileges`, its own IPC
  namespace, a pid limit, the models directory read-only at `/models`, `HOME=/tmp`, and the labels of the GPU
  job token and the owner. The options a caller may add are `-d`, `--rm`, `--name`, `-e`, `--shm-size` (64g at
  most), `--workdir` and `--entrypoint`. Anything else is refused before Docker is called: `-v`, `--mount`,
  `--privileged`, `--network`, `--pid`, `--ipc`, `--device`, `--cap-add`, `--security-opt`, `--user` and the
  rest. Names become `ij-<caller>-<name>`.
- **The reference profile.** `ij-docker ref start|stop|status|logs` runs the reference image with its fixed
  flags, as root in the container, as `reference/atom-serve.sh` always did. The caller chooses only the name
  (`ref` or `engine`), the port and the memory share. `IJ_REF_EXTRA` is refused. This is how
  `examples/atom-engine` runs as `cand`.
- **Departure from the brief: host network, not loopback-only.** Containers use the host's network, so they are
  under the same uid rule as the caller's own processes (a bridge network would bypass the owner match). The
  result: a `cand` container reaches loopback only (C3), and a `builder` container has `builder`'s network, as
  `builder`'s own processes do.
- **No engine directory mount.** The image carries the engine (it is built from the tree); only `/models` is
  mounted.
- **`ij-reap --machine`** (the judge, before a judged run) now kills every `builder` process: SIGTERM to the
  `ij-box` run groups, 5 s, then a child that becomes `builder` and calls `kill(-1, SIGKILL)` until none is left.

### cand's firewall rule (C3)

- **The rule.** `controller/box/ij-fw apply` makes a chain `IJ-CAND` in iptables and ip6tables: loopback port 53
  (udp and tcp) is refused, the rest of loopback is allowed, everything else is refused. It is hooked first in
  `OUTPUT` with `-m owner --uid-owner <cand>`. A systemd unit (`ij-fw.service`) applies it again at boot.
- **The check.** `./submit` runs `sudo -n ij-fw check` (judge may run only `check`) before every judged run
  whenever `submit.engine_user` is set. It fails closed: when the rule is absent or changed, the run ends with
  status error, no judged prompt is sent, and the draw goes back to the pool.
- **Launcher containers.** They run on the host network as the caller's uid, so the same rule covers `cand`'s
  containers. No separate rule is needed.

### Holes left open (the user's call)

- **`builder` during a judged run.** `ij-reap --machine` clears `builder` before the run, but `builder` can log in
  again or start processes while the run goes on. A `builder` helper could relay for `cand` through `/tmp` files,
  Unix sockets or an ssh session; no packet rule sees that. Legitimate `builder` runs (`./box`) wait in the queue's
  shared lock while the judge holds the machine, so a guard that kills `builder` processes would kill those too.
  - Options: (1) a guard that kills every new `builder` process during the run; agents' commands then fail instead
    of waiting. (2) Expire the `builder` account for the run (`usermod -e`), at the risk of a lockout if the judge
    dies before it puts it back. Also close `builder`'s cron, at and systemd linger, which no option above covers.
- **The reference profile is reachable by `cand`.** `examples/atom-engine` as a submission runs the reference
  image as root on the host network, so `cand` reaches it on loopback. Card check: can a request make ATOM's
  server fetch a URL (a model path or a chat-template URL)? If it can, the server is a way out for `cand`.
- **Name resolution without packets.** Lookups through systemd-resolved's varlink or D-Bus socket do not pass the
  packet filter. `box-card-checks.sh` prints what `getent hosts` gives `cand` on the card. A lookup leaks the name
  asked for, a few bytes at a time, not a prompt stream; closing it means a different nsswitch for `cand`.

## GPU session (C4): the order, the sizes, and what needs the card

Everything here is prepared and tested against stand-ins only. The scripts are in `gpu-session/`.

### The order

0. `controller/install.sh` from the Mac. The driver creates the machine and runs `setup-box.sh` (launcher and
   firewall on by default). Set `judge.draw` in `/etc/ij-amd/controller.json` to the trace pool's buckets, for
   example `{"2000": 4, "4000": 4, "8000": 4}` (**a proposal**, see "Pool and cost" below; used on 2026-11-18:
   `{"8000": 6, "12000": 6}`, now the installed default). Then
   `gpu-session/start.sh` from the Mac. It ships five `bench/` files to the controller (root only) and starts
   `run-session.sh` there, detached. `run-session.sh 3 4` runs chosen steps again.
1. **Preflight**: the machine is ready, `ij-fw check` is ok, the launcher is in sudoers, `builder` is not in
   docker, the reference image is there, and `judge.draw`'s buckets equal the trace pool's sizes. Any failure
   stops the chain before anything is judged.
2. **Pools** (`box-pools.sh` on the machine, then `ij-judge prepare-pool` on the controller):
   - the dataset (revision `35455389...`, sha256 `14048dd1...`) is downloaded on the machine only, without a
     login; HTTP 401 or 403 stops the chain (exit 4), and so does any other dataset problem (no network,
     another HTTP error, another file list, another sha256: exit 6);
   - `bench/pools/judge_traces.py` cuts requests from the sessions no public pool holds (the three quarters
     `agentic_pool.py` skips, minus every session and repository in `qwen38-agentic-manifest.json`), one row per
     session, sent as one chat user turn, sizes 2000, 4000 and 8000 (8000 plus 128 answer tokens fits the
     reference's 16,384; on 2026-11-18 2000 and 4000 gave no rows, and the sizes are now 8000 and 12000, cut
     in one run of `judge_traces.py`, one row per session across both; `--exclude` keeps a later, separate cut
     off the sessions of earlier files, as the hand-made 12000 cut of that day did), written in chunks of 500 and copied into the judge's own directory; a cut that ends
     before the session's task (inside a long system prompt or tool list) is not a row, and is counted as
     `cuts_without_the_task` in the summary;
   - the first 12 rows become the warm-up rows; chunk 0 is loaded; the task pool is generated on the controller
     as `judge` (4 runs of `judge_tasks.py 10000`, random seeds not kept) and loaded with `--tasks`;
   - `ij-judge reference-draws 3` on chunk 0; the log lists every request's answer tokens and finish reason
     (kept in `/var/lib/ij-judge/refdraws/`), then the other chunks are loaded.
3. **Plain ATOM**: `ij-judge reference-draws 5` (the draw noise), then `ij-judge measure atom
   examples/atom-engine` 5 times (the reference against itself: measures A and B, and the speed).
4. **AMD's tuned engine**: `box-build-tuned.sh` builds `ij-atom-tuned:dev973` by the recipe of "Rebuild the GPU
   machine", step 3, in one script. Then `ij-judge measure tuned examples/amd-tuned-engine` 5 times, through the
   same path as a submission (judged build, `cand`, the launcher).
5. **Measures A and B**: `ij-judge quality` for every run, saved as `quality.txt`. The numbers for the bar are the
   user's.
6. **One real `./submit`** by `agents` from a throw-away clone with `examples/atom-engine`, timed. It is a real
   submission: the public list shows it with the note "operator: timed ./submit, plain ATOM". Unlike a
   measurement run it is ranked, it can become the best (plain ATOM's speed, the floor any agent must beat), and
   it is rerun on new machines like any submission. Skip step 6 if the user does not want that.
7. **On the card**: `box-card-checks.sh` (the firewall for `cand` and its containers, IPv6, the launcher's
   refusals through real sudo, `ij-reap --machine`, the fail-closed check, what `judge` sees of `cand`'s
   `/dev/kfd`, name lookups), then the four cheaters (`fabricate`, `replay` twice, `read-judge`, `proxy-ref`)
   through `./submit` as `agents`. Each must fail; they show in the public list with "operator: cheater" notes.

Logs: `/var/lib/ij-ctl/gpu-session/<stamp>/session.log` on the controller, copied after every step to
`/root/gpu-session/<stamp>/` on the machine. A failed step is logged and the next runs. The chain stops at once
on an ssh host key failure (never bypassed), a gated or different dataset, or a failed preflight. At the end it
removes the throw-away clones `/home/agents/gpu-session-*` and the controller's copy of the bench files.

**Measurement runs are not ranked.** `ij-judge measure` queues a run with the kind `atom` or `tuned`. It goes
through a submission's path (fresh draw, tasks, `cand`, the walk), is marked "[operator's measurement: ...]" in
the public list with an "m" after its speed, and is never the best or rerun on new machines. Its draw is used up
like a submission's. For `tuned`, the machine record keeps the median measured speed of its runs (protocol 2).

### Pool and cost (one to two days)

- **Per submission, with measure B.** The earlier figure was 5 to 6 minutes on plain ATOM (16 prompts, 128
  tokens, speed parts). Measure B adds 64 tasks on each side, up to 512 tokens each, one at a time:
  - plain ATOM at about 78 tokens/s: 64 x 512 / 78 = 420 s per side at most; if answers average 150 tokens,
    64 x 150 / 78 = 123 s per side;
  - so plain ATOM takes about 9 to 20 minutes per submission (5 to 6, plus 4 to 14);
  - a fast engine shortens only its own side: at 600 tokens/s, 64 x 512 / 600 = 55 s, and the reference's side
    stays 2 to 7 minutes;
  - longer prompts (8000 tokens) add about 0.7 s of prefill each on both sides, about 10 s a run.
- **How many runs.** At 10 to 20 minutes a run, one machine judges at most 72 to 144 runs a day, 150 to 290 in two
  days. 500 runs is a safe upper bound.
- **The trace pool.** 500 runs x 4 rows per bucket = 2,000 sessions per bucket, 6,000 in all with three buckets
  (12 prompts a run, against 16 before). Five per bucket would need 2,500. The dataset's number of eligible
  sessions at each size is printed by `judge_traces.py` (`judge-traces-summary.json`); a short size is written
  short, not refused. About 6,000 rows of up to 8,000 tokens as token ids is roughly 200 MB in
  `/var/lib/ij-judge/pool.jsonl`, which the judge reads at every draw (a few seconds).
- **The task pool.** 500 runs x 64 tasks = 32,000 tasks. One seed gives about 10,000 distinct tasks, so four
  seeds (40,000 rows, repeats skipped by `prepare-pool --tasks`).
- **Card time.** Step 2 about 30 to 60 minutes (2 GB download, the cut, tokenizing 46,000 rows); step 3 about
  1 to 2 hours; step 4 about 30 to 60 minutes of build (the base Dockerfile took 11 minutes on 2026-11-17 before
  its failing step) and 5 runs of 8 to 12 minutes; steps 5 to 7 about 1 hour. In all about 4 to 6 hours.

### Proposed, not decided

- **`judge.draw` = 4 rows per bucket at 2000, 4000 and 8000.** The old buckets (100, 1000, 2000, 4000) came from
  plain prompts; an agent session cannot be cut to 100 or 1,000 tokens, because its system prompt and tools are
  longer. Whether 2,000 can be cut is printed by the pool build. (Settled on 2026-11-18: 2,000 and 4,000 give
  no rows; the operator chose 6 rows per bucket at 8000 and 12000.)
- **Answer lengths on trace rows.** Under protocol 2 the judged answers are also the speed sample, and the speed
  run does not use `ignore_eos`. A raw agentic prompt was answered in about 2 tokens on 2026-11-17. The rows are
  now chat turns, so the chat template applies and the model may think first; step 2 logs the reference's answer
  lengths on chunk 0 before the rest is loaded. Whether short answers need a filter or `ignore_eos` is the user's
  call; the chain does neither.

### What needs the card (not prepared without it)

- Whether the tuned engine runs under the launcher at all: on 2026-11-17 it ran as root with `--ipc host` and
  `seccomp=unconfined`, which the launcher refuses. It also writes JIT and kernel caches; the example's
  Dockerfile opens `/opt/qwen38`, `/opt/atom`, `/opt/aiter` and `/work` to any uid (that copies those trees into
  a new layer: disk and build time to be seen).
- The generated token ids, on the card. AMD's server (commit 352229d6, read from its source) does not return
  them in a non-streamed chat answer: `return_token_ids` echoes only the prompt's ids. So the example's
  Dockerfile applies `patch_token_ids.py`, which adds `choices[0].token_ids` from the engine's own output ids
  (refuses unless its anchor occurs once; tested on that file: it applies once and compiles). Read in the
  source: `choices` is a list of dicts, `model_copy` keeps them, and the engine's `token_ids` are the answer's
  ids only (`num_tokens_output` is their count); the field is set only when `return_token_ids` is asked. The shim
  (`SHIM_IDS_ONLY=1`) asks for `return_token_ids` and reads those ids. Not yet seen running on the card; if it
  fails, every tuned run fails with "the engine's answer carried no token ids".
- The tuned image recipe as one script (`box-build-tuned.sh`). Read from AMD's repository: `build_and_repro.sh`
  takes `FLYDSL_SRC` and `FLYDSL_WHEEL`, and its default source is `https://github.com/ROCm/FlyDSL`; the
  Dockerfile's setup step spans two lines and is cut to the end of the file (the ENV lines after it are added
  back by the commit). Still to be seen: that the setup step fails only at the megakernel, as on 2026-11-17.
- The answer lengths on trace rows, the pool's per-size counts, and the reference's spread over draws.
- Measures A and B for the reference against itself and for the tuned engine, the inputs to the bar.
- The time of a real `./submit` with measure B.
- On the machine: the firewall for `cand` and IPv6, a `cand` container under the rule, the launcher's refusals
  through real sudo, `judge`'s view of `cand`'s `/dev/kfd`, name lookups through systemd-resolved, whether
  ATOM's server can be made to fetch a URL, and the four cheaters.

## GPU session of 2026-11-18: results

One MI350X machine, created by the driver, destroyed after the session. Logs on the controller, in
`/var/lib/ij-ctl/gpu-session/<stamp>/session.log`: `20261118-005055` (steps 1 and 2), `20261118-031640` (step 3,
one run), `20261118-033210` and `20261118-033511` (step 4, one run each) and `20261118-034742` (steps 3 to 7);
also `stage1b.log`, and the operator's hand-made scripts and the machine's logs in `box-files/`. The numbers
below are the logs' own lines. (The session headers still print the old default "sizes 2000,4000,8000 x 2000";
the pool actually used is the one below.)

### Pool

- The first pool build (`20261118-005055`, sizes 2000, 4000 and 8000) gave rows at 8000 only: an agent
  session's system prompt plus tool list is longer than 2,000 and 4,000 tokens, so every cut there ends before
  the task. The operator set the draw to 8000 and 12000, 6 rows each (12 prompts a run). The 12000 rows were cut
  by a hand-patched copy of `judge_traces.py` (`MAX_SIZE = 12000`, skipping the sessions of the 8000 file); that
  is now `judge_traces.py --max-size` and `--exclude`, and 8000,12000 are the defaults of `box-pools.sh`,
  `run-session.sh` and the installed `judge.draw`. `box-pools.sh` now cuts both sizes in one run (one row per
  session across them); that single cut has not yet run on the real dataset.
- After the load (`stage1b.log`): "unused sessions per bucket {'12000': 1994, '8000': 1988}; draw per run
  {'8000': 6, '12000': 6}; warm-up rows 18; unused tasks 37837 (draw per run 64)". That is about 330 runs.
- The reference's answers on chunk 0 of the 8000 rows: "36 requests; tokens min 99, median 128.0, max 128;
  finish {'stop': 5, 'length': 31}".
- Tokenizing the tasks through the reference took about 20 minutes per 10,000 rows, one request at a time.
  `./submit --tokenize` now sends 16 at a time; not yet timed on the card.

### Speeds and measures (final draw, `20261118-034742`)

| run | tokens/s | agreement | strict | top-1 | top-2 | top-5 | mean gap | tasks engine/ref | task delta lb95 |
|---|---|---|---|---|---|---|---|---|---|
| reference draws 1-5 | 64.36, 64.37, 64.53, 64.11, 63.54 | | | | | | | | |
| plain ATOM #6 | 64.30 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 52/52 | +0.000 |
| plain ATOM #7 | 63.90 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 | 48/49 | -0.062 |
| plain ATOM #8 | 64.11 | 1.0 | 0.99935 | 1.0 | 1.0 | 1.0 | 0.0 | 51/51 | +0.000 |
| plain ATOM #9 | 64.44 | 1.0 | 0.99869 | 0.99935 | 1.0 | 1.0 | 8e-05 | 53/53 | +0.000 |
| plain ATOM #10 | 63.15 | 1.0 | 0.99934 | 0.99934 | 1.0 | 1.0 | 8e-05 | 45/45 | +0.000 |
| AMD tuned dev973 #11 | 267.61 | 0.96615 | 0.95638 | 0.96159 | 0.99609 | 1.0 | 0.03752 | 52/49 | +0.016 |
| AMD tuned dev973 #12 | 265.09 | 0.96076 | 0.94768 | 0.95095 | 0.99019 | 1.0 | 0.03573 | 51/53 | -0.078 |
| AMD tuned dev973 #13 | 255.94 | 0.96484 | 0.94141 | 0.94727 | 0.99414 | 1.0 | 0.02132 | 52/52 | -0.031 |
| AMD tuned dev973 #14 | 258.40 | 0.96354 | 0.94792 | 0.95117 | 0.99089 | 1.0 | 0.02743 | 49/49 | -0.047 |
| AMD tuned dev973 #15 | 256.55 | 0.96789 | 0.94954 | 0.95478 | 0.99476 | 1.0 | 0.0335 | 55/55 | -0.031 |

- Medians: the reference 64.36, plain ATOM as a submission 64.11, the tuned engine 258.40 tokens/s (about 4.0
  times the reference). Plain ATOM's own task lower bound reached -0.0625 (`quality.txt`; the log rounds it to
  -0.062). Every walk covered every position: no answer was cut, and no token was outside the reference's top.
- Speed parts: plain ATOM whole request 62.6-64.4, decode 76.5-76.8 tokens/s, prompt time 4.16-4.60 s per pass
  of 12 prompts; tuned whole request 151.9-267.6, decode 637.4-793.5 tokens/s, prompt time 7.17-7.54 s per pass.
  The tuned engine was ready after 164-166 s, plain ATOM after 64-86 s.
- A measurement run took about 8 to 9 minutes for plain ATOM and 11 to 12 minutes for the tuned engine.
- Earlier tuned runs: `20261118-033210` failed in 58 s because `serve` lacked a space between two `docker -e`
  options ("ij-docker: refused: image 'GPU_MAX_HW_QUEUES=1' is not one this launcher built for cand"; fixed in
  commit c0b0bdd). `20261118-033511` (#5) passed at 262.05 tokens/s, agreement 0.95833.
- **Not examined: the no-network path for an agent's own weights.** The tuned image (`ij-atom-tuned:dev973`)
  and its draft model `incoai/Qwen3.8-27B-DFlash2` (in `/srv/models/Qwen3.8-27B-DFlash2`) were placed on the
  machine by root, with network. `reference/setup.sh` now downloads the draft model too. How an agent would bring
  its own weights to a judged build, which has no network, is not settled.

### The timed `./submit` and the cheaters

- `./submit` of plain ATOM by `agents` (#16): "PASSED, agreement 1.0 (limit 0.95), 64.05 tokens/s"; "./submit as
  agents: rc 0 in 543 s".
- All four cheaters failed, as they must:
  - `fabricate` (#17): "FAILED, agreement 0.0 (limit 0.95)", took 14m34s;
  - `replay` (#18): "FAILED, agreement 0.00716 (limit 0.95)", took 14m40s; again (#21): agreement 0.00977, 14m46s;
  - `read-judge` (#19): "./serve exited with 3 before it answered", its last line "read-judge: could read none
    of the judge files", 2 s;
  - `proxy-ref` (#20): "the engine failed in the warm-up: HTTP 502" from its own proxy, whose connection to the
    reference was refused, 2 s.
- A hopeless submission took about 15 minutes: its walk ran about 13 minutes (every answer went to 40 partings,
  about a minute each), then the reference answered the 64 tasks (73 to 91 s). `./submit` now stops the walk
  once the FAIL is decided: a failed answer, or the agreement limit out of reach even if every answer left agreed
  fully. The second rule applies only while the agreement limit applies, not once the quality bar replaces it.
  The reference's side of measure B is then skipped. For these runs the walk stops after the first answer, so a
  run takes about 3 minutes instead of 15. The result stays FAIL; measure A and the agreement then cover the
  walked answers only. Not yet run on the card. With the bar set (below), see "Early stop under the bar".

### The quality bar the user set (2026-11-18)

- Set by the user on 2026-11-18 from the card numbers above (commit 036f454, `config.json` `quality.bar`, and the
  same three values in `/opt/ij-amd/amd/config.json` on the controller):
  - top-5 share at least 0.98 (top-1 and top-2: no limit);
  - mean gap at most 0.08 nats per position;
  - task delta, one-sided 95% lower bound, at least -0.15.
- `quality.bar_replaces_agreement` is true, so the agreement limit (0.95) is reported, not applied. A submission
  passes when every judged request is answered, no answer is cut at the parting limit, and the three limits hold.
  The engine repository's AGENTS.md says so (`agents_text.py`).
- Against the session's numbers: the tuned engine's top-5 share was 1.0 in all five runs, its mean gap
  0.021-0.038 and its task lower bound -0.078 to +0.016; plain ATOM's top-5 1.0, gap at most 8e-05, task lower
  bound down to -0.0625. All of those pass the bar. The tuned engine's agreement (0.961-0.968) and strict
  agreement (0.941-0.956) would have failed a stricter agreement limit; under the bar they do not count.

### Early stop under the bar

- With the bar set, the agreement rule of the early stop is off (it would fail engines the bar passes). In its
  place `answers.decided_fail` stops the walk on what `quality.verdict` fails whatever the answers left hold:
  - a failed answer (as before);
  - an answer cut at the parting limit (the verdict fails any cut answer);
  - a top-k share with a limit that cannot be reached even if every position of the answers left, at most
    `max_tokens + 1` each, were in the reference's top k. The share's denominator is every judged position,
    walked or not, so the bound is exact, not a guess.
- The mean gap and measure B do not stop the walk: the mean gap is over walked positions only and can fall
  again, and measure B comes after the walk.
- For `fabricate` and `replay` (top-5 share near 0) this stops after the first answer, as before. An engine whose
  answers are mostly right but a few percent outside the top 5 is walked further, until the bound is under 0.98.
- Test: `test_protocol.ProtocolTest.test_with_the_bar_set_the_walk_stops_once_a_topk_share_is_out_of_reach`
  (stand-in model; the damaged engine stops early and the full walk gives the same FAIL; the honest one walks
  every answer and passes). Not yet run on the card.

### On the card (`box-card-checks.sh`)

- 21 passed, 3 failed. The three failures were the script's own fault: it ran `cand`'s containers while nobody
  held the GPU queue, and `ij-docker` refuses that. Run by hand under a hold of `judge` (`box-files/hand3.sh`),
  all three passed: `cand`'s container got ConnectionRefusedError for 1.1.1.1:443, ran as uid 1002 (`cand`), and
  saw the GPU devices. The script now takes that hold itself.
- `gpuqueue.kfd_users()` run as `judge` returned `[]` while `cand` held `/dev/kfd`, because `/proc/<pid>/fd` of
  another account is readable by root only. What asks "who else uses the card" as `judge`:
  - the reaping (`ij-reap --keep`, `ij-reap --machine`) already runs as root through sudo and sees everyone;
  - `submit`'s wait for the GPU to be alone, and the `gpu` status tool, called `kfd_users()` as `judge`. They saw
    only `judge`'s own processes and relied on the VRAM check (under 4 GiB in use) alone.

  Fixed: in a judged run (`IJ_AMD_REAP_SUDO`, which `ij-box` sets) `kfd_users()` asks `sudo ij-reap --kfd-users`
  (root, `judge` only). That lists the processes of `builder`, `judge` and `cand` that have `/dev/kfd` open; a
  refusal falls back to the own view. `box-card-checks.sh` now checks that `judge` sees `cand`'s process. Not
  yet run on the card.
- The rest passed: the firewall rule for `cand` and its boot unit, `cand` refused at 1.1.1.1:443 and on IPv6,
  its own loopback reachable, `builder` and `cand` not in the docker group, the nine `ij-docker run` refusals
  (rc 2), `ij-reap --machine` killing `builder`'s detached process, and the firewall check failing closed and
  coming back with `ij-fw apply`. `cand`'s `getent hosts example.com` gave nothing (rc 2).

## Smaller weights: what a judged engine can reach (2026-11-18)

The question: how does an agent's engine bring its own smaller weights (a 4-bit copy, its own draft model) to a
judged `./submit`? Nothing was changed for this; the user decides. What the code does:

### Engine with a Dockerfile (the container path)

- The judge builds `ij-cand/engine` from its own copy of the commit (`ij-docker build --for cand`): the context is
  the commit only, `RUN` steps have `--network none`.
- `FROM` is resolved by the Docker daemon, which pulls with its own network. Checked on the controller's Docker
  29.7: a `--network none` build `FROM` an image that exists only locally is built from that local image.
- The container (`ij-docker` `forced()`; `test_ij_docker` checks it): `/srv/models` read-only at `/models`, no other
  host path, `HOME=/tmp`, the host network under `cand`'s firewall rule (loopback only), no capabilities.
- No size limit on the container's own files: no `--tmpfs`, `--read-only` or `--storage-opt`, so `/tmp` is the
  container's writable layer on the Docker data disk. No `--memory` limit (251 GB RAM). `/dev/shm` follows
  `--shm-size`, default 64 MB, at most 64g.
- `serve` must answer `GET /v1/models` within `engine.health_timeout_s`, 1800 s (30 minutes), not 20:
  `ref.ready_timeout_s` (1200 s) is the reference's. The image build has `engine.build_timeout_s`, 3600 s.

### Engine without a Dockerfile (the host path)

- `build` and `serve` run as `cand` directly on the machine, with `HOME=/home/cand` and `TMPDIR=/home/cand/tmp`
  (`run/` and `tmp/` are emptied for each run; the rest of `/home/cand` stays from run to run, which the replay
  cheater uses).
- `cand` can read whatever the host lets any account read, not only the models. Measured by the stand-in e2e
  (section 5a): a file that `builder` made readable in `/var/lib/ij-amd/builder` (mode 755, owner builder) and in
  `/tmp` was readable by the judged engine; `/home/builder` (700) was not. `/srv/models` was not writable.

### The stand-in test (e2e section 5a)

- `controller/standin/convert_and_serve.py`: at start-up it reads every file of the models folder, writes a
  derived "converted weights" file under its own `$TMPDIR`, and serves the toy model from that file
  (`fake_engine.py --weights`). Committed as `serve`, it passed a judged `./submit` as `cand`: "converts: rc 0:
  passed". Its report: models `/srv/models/standin` read, `models_writable` false, weights in
  `/home/cand/tmp/converted/weights.json`.
- The stand-in machine has no Docker, so this run tests the host path. The container's boundary is tested by
  `test_ij_docker.test_cand_run_carries_the_forced_flags` (argv level) and was seen on the card in
  `box-card-checks.sh` (uid, network, devices).

### The paths that exist

1. Conversion at start-up from `/models`, in memory or under the container's `/tmp` (or `cand`'s `$TMPDIR`). AMD's
   tuned engine starts this way. Bound by the 30-minute limit and the disk.
2. Small files in the commit (copied into the image or the run). The commit is copied to the controller and the
   machine; nothing limits its size in the code, but it is a git repository.
3. An image named in `FROM`, pulled by the daemon. Inflow: anything an anonymous pull can reach, including an
   image an agent pushed to a public registry that accepts anonymous pushes, if `builder` has outbound network
   (only `cand` is firewalled). Outflow: the pull tells the registry the image name and the machine's address, at
   build time. The image name is fixed in the commit and the hidden prompts are not in the build context (the
   engine gets them only after the build, under `cand`, behind its firewall rule), so a pull cannot carry them
   out.
4. Not in the coordinator's list: `FROM ij-builder/engine:latest`. That image is the one `builder` built last
   (`ij-docker build`), and its context can hold any file `builder` made on the machine, of any size. A judged
   build would take it as its base. Not tried in a judged build; the local-image behaviour was checked on the
   controller's Docker only.
5. Not in the coordinator's list: on the host path, any file `builder` leaves world-readable in
   `/var/lib/ij-amd/builder` or `/tmp`; and anything an earlier judged run of `cand` left in `/home/cand` outside
   `run/` and `tmp/`.

So a large file an agent makes on the machine can reach a judged run today, by path 4 or 5. Both depend on
machine state that is not in the commit, so the same commit can be judged differently on another machine.

### Options (the user's decision; none is done)

- Close paths 4 and 5: refuse `FROM ij-*` in a judged build (or build with `--pull always`, which refuses local
  base images), make `/var/lib/ij-amd/builder` 700, give `cand` a private `/tmp` (a judged run under `systemd-run
  -p PrivateTmp=yes`, or always run judged engines in a container), and empty `/home/cand` before each run.
- Or allow large artifacts on purpose, through one named path: for example a folder `builder` fills, copied by
  the judge into the run read-only, with a size cap and its hash in the ledger, so a submission records what it
  used.
- Either way, write it in AGENTS.md. Today AGENTS.md states only what a container sees (paths 1-3).

## The pool's run names after the numbers start again (2026-11-18)

- `pool-used.jsonl` frees a draw by run name: `release("sub-N")` frees every session drawn under `sub-N`. With the
  submission numbers starting again at #1, a new `sub-19` that was judged would have its draw freed by the old
  `sub-19`'s release (read-judge), so its prompts could be drawn again; and a new release would free an old run's
  kept draw.
- `ij-judge` now names a submission's run `sub-N-<8 hex>` (`run_name`) and its reference draw `ref-sub-N-<hex>`.
  Old names stay as they are in `pool-used.jsonl`. Test: `test_judge_pool.test_run_names_never_repeat_after_the_numbers_start_again`.

## The public list cleared (2026-11-18, the user: "Yes, clear them")

- Moved, not deleted, to `/var/lib/ij-judge/archive-20261118-gpu-session/` (judge, 700): `ledger.jsonl` (21
  submissions), `subs/`, `refdraws/`, `incoming/` (the pool's source files: tasks-1..4, traces, traces12k,
  warm, warm12k), `seq`, a copy of `machines.json`, `engine-before-clear.bundle` (every ref of the engine
  repository before: main, try-a, try-b, their worktrees) and `agents-ij-profiles/` (`./profile` output of the
  tests, from `/home/agents/ij-profiles`).
- `seq` removed, so the next submission is #1. `machines.json` keeps machine 620476947 (reference 65.05,
  tuned 260.23, the ten reference draws); the protocol-1 entry `manual-checks-20261117` (78.2) was taken out.
  `ij-judge score` rebuilt `score.json`: "submissions: 0", "best passing submission: none".
- Not touched: `pool.jsonl`, `warm.jsonl`, `tasks.jsonl`, `pool-used.jsonl` (no used prompt is drawn again; the
  run names above keep that true with the numbers starting again).
- `/home/agents/engine`: worktrees `wt-a`, `wt-b` and branches `try-a`, `try-b` removed; `notes/` held only
  `.keep`; `main` held 8 installer commits and was squashed to one, old objects pruned.
- Nothing brings them back: the installer does not write judge records; the e2e sets aside the records it finds
  and puts back exactly those, and now also removes its own `./profile` output.
