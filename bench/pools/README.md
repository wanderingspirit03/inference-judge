# Pools for the Gemma 4 26B-A4B targets

Five data files. `real_pool.py` copies the two speed pools from text files in
`real/`. `build_pools.py` generates the three quality files from a seed. No
file holds real user data, and nothing here needs an API or a network.

| file | lines | on disk | used by |
|---|---|---|---|
| `gemma4-real-main.jsonl` | 256 | 1.4 MB | speed test, `--prompt-pool` for the measured sweeps of both Gemma targets |
| `gemma4-real-confirm.jsonl` | 96 | 0.5 MB | speed test, the confirm sweeps of both Gemma targets (no request or passage shared with main) |
| `gemma4-quality-prompts.jsonl` | 256 | 1.8 MB | `quality2` prompts (greedy + teacher forcing), the full set |
| `gemma4-quality-prompts-6144.jsonl` | 240 | 1.3 MB | the same, without the 16 prompts a 6144 context cannot hold; both Gemma targets name this one |
| `gemma4-quality-tasks.jsonl` | 200 | 54 KB | `quality2` tasks: 100 math, 100 multiple choice |

Rows are `{"id", "prompt"}`; the speed pools' rows also carry `"isl"` (see
below); tasks add `"answer"`, `"kind"` (`math` | `mc`), `"lang"`.

## Speed pools

Both Gemma targets (`targets/v6e4-gemma4-26b-tps.yaml` and
`targets/gpu-gemma4-26b-tps.yaml`) use these two files, so the two are asked
the same question.

Whole requests, one per text file, in `real/<pool>/<size>/<NN-kind>.txt`.
There are 352 files. The sizes are 100, 1000, 2000 and 4000 input tokens:
64 requests per size in main, 24 per size in confirm. `real_pool.py` copies
each file into a row `{"id", "isl", "prompt"}`, for example id
`real-main-1000-04-emails`.

**What a request is.** The whole user message as a person would send it: the
ask first, then the document, email thread, notes or code they paste. The
size 100 files (kind `ask`) are the ask alone. The larger sizes come in 16
kinds: report, emails, minutes, tickets, policy, paper, design, spec, story,
lecture, data, config, python, typescript, go and sql. Every ask needs a long
answer, because the speed test measures a 512-token answer. The requests were
written for this project: invented people, companies and places, addresses at
example hosts, nothing copied.

**How the harness sends it.** A row with `isl` goes out as written behind the
reference line ("Request reference: <uuid>."). It is never cut and never
padded, and it is sent only under its own size. Each record keeps `isl` (the
size class) and `isl_client` (the tokens actually sent, reference line
included). The harness also accepts a pool whose rows have no `isl`: it cuts
or pads each prompt to the exact size (`Tokenizer.fit`). No pool here is of
that kind, and the harness refuses a pool that mixes the two.

**The fit check.** At load, `_TpsRun._check_whole_requests` requires that,
with the reference line in front, a request is at most its size and at least
`(1 - isl_tol)` of it. Both targets set `isl_tol: 0.3`. The reason: the
TPU server pads a prompt up to the next power of two, so every request of a
size becomes the same amount of work for it. The harness sizes the reference
line with `harness.nonce_bounds`, over a fixed set of 4096 references, so the
check gives the same answer on every load. With the Gemma tokenizer that line
is 32 to 42 tokens there; over 200,000 references it is 29 to 42, and the
builder allows 29 to 48.

| size | server pads to | tokens allowed per file | held in main | held in confirm |
|---|---|---|---|---|
| 100 | 128 | 41 to 52 | 41 to 52 | 41 to 52 |
| 1000 | 1024 | 671 to 952 | 719 to 951 | 741 to 945 |
| 2000 | 2048 | 1371 to 1952 | 1473 to 1944 | 1514 to 1852 |
| 4000 | 4096 | 2771 to 3952 | 2987 to 3849 | 2994 to 3635 |

The allowed range is `real_pool.bounds(size)`. The last two columns were
measured with the Gemma 4 tokenizer on 2026-10-13.

**Why 64 and 24.** A request goes out under its own size only, and `run.sh`
never sends a candidate the same request twice across its launches (the
screen, the first draw and the second), so a size can run dry. On the first
v6e-4 session a sweep sent 15 to 19 requests, which over four sizes is 4 or 5
of each and puts one candidate run at about 32 per size. Main holds 64 so
that a candidate twice as fast still fits, and confirm needs one draw of
about 10. The comment above
`MIN_PER_SIZE` in `real_pool.py` has the sum. The builder refuses to build a
pool with fewer. If a size does run dry mid-run, the harness stops with "the
prompt pool has no request of size N".

**What the tests check.** `tests/test_real_pool.py` checks that the files are
a fresh build, that each size has enough requests, that main and confirm
share no request and no line of 60 characters or more (a divider line
without letters does not count), that every request is
whole text (fences closed, no placeholder markers, addresses only at example
hosts), and that pasted Python parses. With the real tokenizer it also
checks that every request is inside its band and that the harness accepts
every request. Without `GEMMA_TOKENIZER_JSON`, the tests that need the
tokenizer are skipped.

Run these from `bench/`:

    GEMMA_TOKENIZER_JSON=/path/to/tokenizer.json .venv/bin/python pools/real_pool.py count pools/real/main/1000/*.txt
    .venv/bin/python pools/real_pool.py build --out pools            # rewrite the two files
    .venv/bin/python pools/real_pool.py build --out pools --check    # do the files match a fresh build?
    GEMMA_TOKENIZER_JSON=/path/to/tokenizer.json .venv/bin/python -m pytest tests/test_real_pool.py

To add or edit a request:

1. Write or edit `real/<pool>/<size>/<NN-kind>.txt`. A new request takes the
   next free number in its directory.
2. Run `count` on it. It prints the token count and `ok`, `TOO SHORT` or
   `TOO LONG` against the allowed range.
3. Run `build --out pools` to rewrite the two `.jsonl` files.
4. Run the tests. Keep main and confirm apart: no request and no long line
   in both.

## Quality prompts

256 prompts: 36 factual, 36 arithmetic word problems, 36 code, 36 summarise,
30 translate, 30 JSON extraction, 52 retrieval-from-context. Length spread, in
Gemma tokens: min 16, median 745, max 7744; 42 above 4096, 53 at or below
64, 28 in 256 to 512, 41 in 1024 to 2048. The ceiling is 7800 so that prompt +
128 forced tokens fits in 8192.

The v6e-4 target serves a 6144 context, and its server refuses a longer
prompt (HTTP 400). `gemma4-quality-prompts-6144.jsonl` is the full file
without the 16 prompts over 6100 tokens: 240 rows, the same rows in the same
order, the longest 5873 tokens. Both Gemma targets name it in
`reference.prompts`, so the two are checked on the same prompts.
`build_pools.py` writes it: the rows to leave out are listed by id in
`OVER_6144`, because the builder has no tokenizer. With the real tokenizer,
`tests/test_pools.py` checks that the list is exactly the prompts over 6100
tokens, so update the list when the quality prompts change.

## Quality tasks

200 tasks over 25 families (arithmetic expressions, shopping totals, speeds,
units, dates, modular powers, Python loop output, sequences, gcd, primes,
percent, letter counts, ages; and multiple choice on largest number, primality,
Python output, weekday, ordering, alphabet, divisor counts, logic, shifted
letters, nth item). About 8% are in German, French or Spanish.
Math replies end with `ANSWER: <number>` (the ruler reads the last number);
multiple choice asks for the letter only.
Every task has two independent derivations: the generator's, and
`pool_tasks.verify(task)`, which re-derives the answer from the prompt text
alone. `tests/test_pools.py` checks that they agree for all 200.

## Rebuild

    cd bench/pools
    ../.venv/bin/python build_pools.py --check          # do the files match a fresh build?
    ../.venv/bin/python build_pools.py                  # rewrite the three quality files here
    ../.venv/bin/python build_pools.py --out /tmp/x --n-prompts 256 --n-tasks 600

Output is byte-for-byte reproducible (seed 20261011, only `random.Random`).
The generator aims each prompt at a token length from its characters, using
typical characters per Gemma token (prose 4.65, code 3.1), so rebuilding
needs no tokenizer. After changing the text banks, verify with the real
tokenizer:

    GEMMA_TOKENIZER_JSON=/path/to/tokenizer.json PYTHONPATH=<site with tokenizers> \
        ../.venv/bin/python -m pytest tests/test_pools.py

Measured with `tokenizer.json` sha256 `cc8d3a0c...fe0f` (the Gemma 4 26B-A4B-it
tokenizer, 2026-10-11; the target pins model revision 4d7ae498...). Observed
characters per token: prose 4.43 to 4.85, code 2.93 to 3.25, mixed 3.68 to 4.00.

Files: `build_pools.py` (the interface: `build`, `write`, `check`),
`pool_text.py` (seeded prose and code), `pool_tasks.py` (task families and
their verifiers).

## Known limits

- The code generator has about a dozen Python shapes with varying names and
  constants, so long code prompts repeat shapes: the quality check's code
  coverage is narrow.
- Prompts are raw text, not wrapped in the chat template. The ruler's greedy
  and teacher-forcing calls use `/v1/completions` with the raw prompt.
- No tool-call tasks: the target launches without tool-call parser flags.
- The real pool has no size above 4000. A target that asks for a larger
  size, such as 8000, stops at load with "the prompt pool has no request of
  size 8000". Such a target needs new request files of that size, or a pool
  of its own without `isl`.
- A real request is under its nominal size. With the reference line in front
  it sits between 70% of the size and the size, so compare `isl_client`, not
  `isl`, when you need the tokens actually sent.
- Editing a pool or a quality file changes the ruler hash
  (`target.ruler_hash` covers the data files a target names), so a result
  measured before the edit is not comparable with one measured after it.
  Measure the baseline and the calibration again.
