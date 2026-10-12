# Pools for the Gemma 4 26B-A4B TPU target

Four data files, one generator. Everything here is synthetic and seeded: no
real user data, no API, no network.

| file | lines | on disk | used by |
|---|---|---|---|
| `gemma4-main.jsonl` | 384 | 16.6 MB | speed test, `--prompt-pool` for the measured sweeps |
| `gemma4-confirm.jsonl` | 320 | 13.7 MB | speed test, the confirm sweeps (no prose shared with main) |
| `gemma4-quality-prompts.jsonl` | 256 | 1.8 MB | `quality2` prompts (greedy + teacher forcing) |
| `gemma4-quality-tasks.jsonl` | 200 | 54 KB | `quality2` tasks: 100 math, 100 multiple choice |

Rows are `{"id", "prompt"}`; tasks add `"answer"`, `"kind"` (`math` | `mc`), `"lang"`.

## Speed pools

`harness._TpsRun.prompt()` prepends a nonce line ("Request reference: <uuid>.")
and calls `Tokenizer.fit(text, isl - count(nonce))`. `fit` cuts a long text and
pads a short one by repeating it, so a line shorter than the input size would
come back as a repeated loop. Every line is therefore longer than the largest
input size (8192): **min 9369 Gemma tokens, median about 10 000, max 11 599**
(real tokenizer, below). Sizes 128 to 8192 all cut the same lines.

`_PromptSampler` uses each id once per invocation and flags a repeat as
`prompt_reused`. A sweep uses (sizes x requests) ids, and the next sweep
excludes only the previous one (`--pool-exclude`), so 384 main lines serve a
sweep of up to 384 requests and 320 confirm lines a confirm sweep of up to 320.
Raise `--main-lines` / `--confirm-lines` if a sweep needs more.

Each line starts with an instruction that asks for a long answer (the metrics
sanity band reads the OUTPUT text, so one-word replies would look degenerate),
then the body. Kinds: about 62% prose documents (articles, minutes, email
threads, FAQs, stories, reports), 23% code (Python, JS, SQL, C, shell files),
15% a document followed by a code block. Main and confirm draw on disjoint
halves of the topic and name banks, so they share no prose. They do share code
boilerplate: 140 of the 384 main lines contain a code passage of 200 characters
or more that also appears in confirm (longest 513, measured 2026-10-11). Every
request gets a fresh nonce prefix, so this cannot turn into a cache hit.
Distinct lines share no long prefix (first 300 characters of every body differ).

## Quality prompts

256 prompts: 36 factual, 36 arithmetic word problems, 36 code, 36 summarise,
30 translate, 30 JSON extraction, 52 retrieval-from-context. Length spread, in
Gemma tokens: min 16, median 745, max 7744; 42 above 4096, 53 at or below
64, 28 in 256 to 512, 41 in 1024 to 2048. The ceiling is 7800 so that prompt +
128 forced tokens fits in 8192.

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
    ../.venv/bin/python build_pools.py                  # rewrite the four files here
    ../.venv/bin/python build_pools.py --out /tmp/x --main-lines 1024 --confirm-lines 1024 \
        --min-tokens 20000 --n-prompts 256 --n-tasks 600

Output is byte-for-byte reproducible (seed 20261011, only `random.Random`).
The generator estimates length from characters, using per-kind characters per
Gemma token measured on the real tokenizer (prose 5.0, code 3.4 as ceilings),
so rebuilding needs no tokenizer. After changing `--min-tokens` or the text
banks, verify with the real tokenizer:

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
  constants. Long code prompts and code lines repeat shapes. The nonce at the
  start of every speed request defeats prefix caching, so this does not
  matter for speed; for quality it means code coverage is narrow.
- Prompts are raw text, not wrapped in the chat template. The ruler's greedy
  and teacher-forcing calls use `/v1/completions` with the raw prompt.
- No tool-call tasks: the target launches without tool-call parser flags.
