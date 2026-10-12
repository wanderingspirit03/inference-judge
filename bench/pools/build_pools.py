#!/usr/bin/env python3
"""Build the Gemma 4 speed pools, quality prompts and quality tasks.

The files in this directory are the output of this script; rebuild them with

    python build_pools.py --out .            # writes the five files
    python build_pools.py --out . --check    # exit 1 if the files on disk differ

and the bytes are the same on every machine (one seed, `random.Random` only,
no tokenizer, no network). Nothing here is scraped or copied: the text comes
from the banks and templates in pool_text.py and the tasks from pool_tasks.py.

Interface:
  build(main_lines, confirm_lines, min_tokens, n_prompts, n_tasks) -> {filename: text}
  write(out_dir, **sizes)  /  check(out_dir, **sizes) -> list of differing files

How the harness uses a speed pool line (harness.py, _TpsRun.prompt): the line's
`prompt` is cut to exactly `isl - nonce` tokens with Tokenizer.fit, which also
PADS a short text by repeating it. So every line must be longer than the
largest input size; a line is built from at least `min_tokens` Gemma tokens,
which is sized from per-kind chars-per-token ceilings (CEIL) so that the real
token count is above min_tokens whatever the text. The cut keeps the start of
the line, so the instruction sits at the start and the tail is what gets lost.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pool_tasks  # noqa: E402
import pool_text  # noqa: E402

SEED = 20261011
FILES = {"main": "gemma4-main.jsonl", "confirm": "gemma4-confirm.jsonl",
         "prompts": "gemma4-quality-prompts.jsonl", "tasks": "gemma4-quality-tasks.jsonl"}

# Defaults. A single-stream sweep sends about (ramp + window) / 4.4 s = 75 requests
# at today's 126 tok/s; each prompt is used once per sweep and a sweep also
# avoids the previous one's prompts, so a pool must hold twice a sweep.
MAIN_LINES = 384
CONFIRM_LINES = 320
MIN_TOKENS = 9000            # longest speed input is 8192, plus the nonce and headroom
N_PROMPTS = 256
N_TASKS = 200
MAX_PROMPT_TOKENS = 7800     # 8192 context - 128 forced tokens - margin

# Characters per Gemma token. CEIL is the most characters one token covered in
# any text measured (so chars / CEIL is a lower bound on tokens); MID is the
# typical value, used to aim quality prompts at a token length.
CEIL = {"prose": 5.0, "code": 3.4}
MID = {"prose": 4.65, "code": 3.1}

PROSE_HEADS = [
    "Read the following document carefully, then write a long, detailed essay of at least 1500 words about it.",
    "Below are some notes. Write an extensive commentary that goes through them one by one. Do not stop early.",
    "Continue the following text for as long as you can, in the same voice, adding sections, details and examples.",
    "Summarise the text below section by section, then write a long FAQ with thirty questions and full answers.",
    "Here is a long document. Write a thorough critique of it, quoting passages, and finish with recommendations.",
    "Rewrite the text below as a detailed training handout with headings, examples and a quiz at the end.",
    "Study the material below and write a long story set in the same world, with characters and dialogue.",
    "Read these papers and write a detailed briefing for a new colleague, covering every section in turn.",
]
CODE_HEADS = [
    "Review the source code below in detail. Explain every function, list bugs and edge cases, then write exhaustive unit tests.",
    "Write complete documentation for the following code: an overview, one section per function, and worked examples.",
    "Refactor the code below. Explain each change at length and show the full new version of every file.",
    "You are a senior engineer. Go through the files below line by line and write a long code review.",
]

FACTUAL = [
    "Why does ice float on water?", "How do vaccines train the immune system?", "What causes the seasons on Earth?",
    "Why is the sea salty?", "How does a refrigerator keep food cold?", "What is the difference between weather and climate?",
    "How do bees make honey?", "Why do we have leap years?", "How does a bill become a law in a parliamentary system?",
    "What is inflation and why does it matter?", "How do airplanes stay in the air?", "What is photosynthesis?",
    "Why do some metals rust and others do not?", "How does GPS work?", "What is the greenhouse effect?",
    "Why is the night sky dark if there are so many stars?", "How do batteries store energy?",
    "What happens inside a volcano before it erupts?", "How does the internet route a message from one computer to another?",
    "Why do onions make people cry?", "What is the difference between a virus and a bacterium?",
    "How do tides work?", "Why is the sky blue and the sunset red?", "What is a prime number and why do primes matter?",
    "How does compound interest work?", "What is the difference between speed and velocity?",
    "How are rainbows formed?", "Why do cats purr?", "How does a mechanical clock keep time?",
    "What is the water cycle?", "How do noise-cancelling headphones work?", "Why do we dream?",
    "What is supply and demand?", "How does a microwave oven heat food?", "What causes earthquakes?",
    "How do plants know which way is up?", "What is the difference between baking soda and baking powder?",
    "How does the human eye focus?", "Why did the Roman Empire fall?", "How do submarines dive and surface?",
    "What is entropy?", "How is paper made?", "Why do leaves change colour in autumn?", "How does a vote count work in a ranked-choice election?",
    "What is the Pythagorean theorem and how is it proved?", "Why is the Moon always showing the same face to Earth?",
]
FACTUAL_FRAMES = [
    "Question: {q}\nGive a thorough answer with background and examples.\n\nAnswer:",
    "{q}\nExplain step by step, as if to a curious student, and mention common misconceptions.\n\nExplanation:",
    "Q: {q}\nA detailed answer follows.\n\nA:",
]
LANGS = ["French", "German", "Spanish", "Italian", "Portuguese", "Dutch"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _jsonl(rows: list[dict]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)


def _trim(text: str, n: int) -> str:
    """text cut to about n characters, at a paragraph or sentence end."""
    if len(text) <= n:
        return text
    cut = text[:n]
    for sep in ("\n\n", "\n", ". "):
        i = cut.rfind(sep)
        if i > 0.6 * n:
            return cut[:i + len(sep)].rstrip() + ("." if sep == ". " else "")
    return cut


def _logspace(rng: random.Random, lo: float, hi: float, n: int) -> list[int]:
    """n values spread log-uniformly over [lo, hi], one per equal log slice."""
    a, b = math.log(lo), math.log(hi)
    return [int(math.exp(a + (b - a) * (i + rng.random()) / n)) for i in range(n)]


# ---------------------------------------------------------------------------
# speed pools
# ---------------------------------------------------------------------------
def _pool_line(w: pool_text.Writer, rng: random.Random, min_tokens: int) -> tuple[str, str]:
    r = rng.random()
    if r < 0.62:
        kind, body = "prose", w.document(int(min_tokens * CEIL["prose"]))
        head = rng.choice(PROSE_HEADS)
    elif r < 0.85:
        kind, body = "code", w.code(int(min_tokens * CEIL["code"]))
        head = rng.choice(CODE_HEADS)
    else:
        kind = "mixed"
        half = min_tokens // 2 + 1
        body = w.document(int(half * CEIL["prose"])) + "\n\n```\n" + w.code(int(half * CEIL["code"])) + "\n```\n"
        head = rng.choice(PROSE_HEADS)
    return kind, f"{head}\n\n{body}"


def build_pool(name: str, domains: list[int], names: tuple[int, int], seed: int, n_lines: int,
               min_tokens: int, seen: set[str]) -> list[dict]:
    rng = random.Random(seed)
    w = pool_text.Writer(rng, domains, names)
    rows = []
    for i in range(n_lines):
        while True:
            kind, text = _pool_line(w, rng, min_tokens)
            key = text.split("\n\n", 2)[1][:300]         # the body's first paragraph
            if key not in seen:
                seen.add(key)
                break
        rows.append({"id": f"{name}-{i + 1:04d}-{kind}", "prompt": text})
    return rows


# ---------------------------------------------------------------------------
# quality prompts
# ---------------------------------------------------------------------------
def _p_factual(w, rng, qs: list[str], i: int, tokens: int) -> str:
    q = qs[i % len(qs)]
    frame = FACTUAL_FRAMES[(i // len(qs) + i) % len(FACTUAL_FRAMES)].format(q=q)
    if tokens > 90:                       # some carry background notes to fill a bigger bucket
        notes = _trim(w.document(int(tokens * MID["prose"])), int(tokens * MID["prose"]))
        return f"Background notes (they may or may not be relevant):\n{notes}\n\n{frame}"
    return frame


def _p_arith(w, rng, stem: str) -> str:
    return f"{stem}\n\nSolve this problem. Show every step of your working in full sentences.\n\nSolution:"


def _p_code(w, rng, tokens: int) -> str:
    return w.python_completion(int(tokens * MID["code"]))


def _p_summary(w, rng, tokens: int) -> str:
    body = _trim(w.document(int(tokens * MID["prose"])), int(tokens * MID["prose"]))
    return f"{body}\n\nSummarise the text above in a few paragraphs, covering every section.\n\nSummary:"


def _p_translate(w, rng, tokens: int) -> str:
    lang = rng.choice(LANGS)
    body = _trim(w.document(int(tokens * MID["prose"])), int(tokens * MID["prose"]))
    return f"Translate the following text from English into {lang}. Keep the formatting.\n\nEnglish:\n{body}\n\n{lang}:\n"


def _p_json(w, rng, tokens: int) -> str:
    recs = []
    total = 0
    while total < tokens * MID["prose"]:
        text, _ = w.record()
        recs.append(text)
        total += len(text) + 12
    lines = "\n".join(f"Record {i + 1}: {t}" for i, t in enumerate(recs))
    return ("Convert every record below into a JSON object with lower_snake_case keys, and output one JSON array "
            f"holding all of them. Output only JSON.\n\n{lines}\n\nJSON:\n")


def _p_retrieve(w, rng, tokens: int) -> str:
    n_needles = 1 if rng.random() < 0.6 else 3
    towns = rng.sample(w.towns, n_needles)
    codes = [f"{rng.choice(pool_text.WORDS)}-{rng.randrange(1000, 9999)}" for _ in towns]
    target = int(tokens * MID["prose"])
    paras, size = [], 0
    while size < target:
        p = w.paragraph()
        paras.append(p)
        size += len(p) + 2
    for town, code in zip(towns, codes):
        pos = rng.randrange(len(paras) + 1)
        paras.insert(pos, f"Note from the office: the access code for the {town} depot is {code}.")
    doc = "\n\n".join(paras)
    if n_needles == 1:
        q = (f"What is the access code for the {towns[0]} depot? Give the code, then quote the sentence it comes "
             f"from and explain where in the notes you found it.")
    else:
        q = ("List the access code of every depot named in the notes (" + ", ".join(towns) +
             "), one per line, each with the sentence it comes from.")
    return f"Below is a long collection of notes. Read all of it, then answer the question at the end.\n\n{doc}\n\nQuestion: {q}\n\nAnswer:"


def build_prompts(n: int) -> list[dict]:
    """n quality prompts: a fixed mix of kinds with token lengths spread from
    about 30 to MAX_PROMPT_TOKENS and at least 20 above 4096."""
    rng = random.Random(SEED + 3)
    w = pool_text.Writer(rng, list(range(len(pool_text.DOMAINS))), (0, 1))
    scale = n / 256
    counts = {k: max(1, round(v * scale)) for k, v in
              dict(factual=36, arith=36, code=36, summary=36, translate=30, json=30, retrieve=52).items()}
    counts["retrieve"] += n - sum(counts.values())
    out: list[tuple[str, str]] = []

    qs = FACTUAL[:]
    rng.shuffle(qs)
    n_big = min(12, counts["factual"] // 3)
    f_tokens = [rng.randrange(25, 70) for _ in range(counts["factual"] - n_big)] + \
        [rng.randrange(110, 700) for _ in range(n_big)]
    out += [("factual", _p_factual(w, rng, qs, i, t)) for i, t in enumerate(f_tokens[:counts["factual"]])]
    out += [("arith", _p_arith(w, rng, s)) for s in pool_tasks.math_stems(rng, counts["arith"])]
    c = counts["code"]
    code_t = _logspace(rng, 150, 1500, c // 3) + _logspace(rng, 1500, 4000, c // 3) + \
        _logspace(rng, 4200, MAX_PROMPT_TOKENS, c - 2 * (c // 3))
    out += [("code", _p_code(w, rng, t)) for t in code_t]
    s = counts["summary"]
    n_long = min(8, s // 2)
    sum_t = _logspace(rng, 180, 2500, s - n_long) + _logspace(rng, 4300, MAX_PROMPT_TOKENS, n_long)
    out += [("summary", _p_summary(w, rng, t)) for t in sum_t]
    out += [("translate", _p_translate(w, rng, t)) for t in _logspace(rng, 60, 1600, counts["translate"])]
    out += [("json", _p_json(w, rng, t)) for t in _logspace(rng, 150, 3200, counts["json"])]
    r = counts["retrieve"]
    n_long = min(20, r // 2)
    ret_t = _logspace(rng, 700, 4000, r - n_long) + _logspace(rng, 4300, MAX_PROMPT_TOKENS, n_long)
    out += [("retrieve", _p_retrieve(w, rng, t)) for t in ret_t]
    rng.shuffle(out)
    return [{"id": f"q-{i + 1:04d}-{kind}", "prompt": text} for i, (kind, text) in enumerate(out)]


# ---------------------------------------------------------------------------
# interface
# ---------------------------------------------------------------------------
def build(main_lines: int = MAIN_LINES, confirm_lines: int = CONFIRM_LINES, min_tokens: int = MIN_TOKENS,
          n_prompts: int = N_PROMPTS, n_tasks: int = N_TASKS) -> dict[str, str]:
    """{filename: file text} for the four data files. Main and confirm draw on
    disjoint halves of the topic and name banks, so they cannot share a passage."""
    n_dom = len(pool_text.DOMAINS)
    seen: set[str] = set()
    main = build_pool("main", list(range(0, n_dom, 2)), (0, 1), SEED + 1, main_lines, min_tokens, seen)
    confirm = build_pool("confirm", list(range(1, n_dom, 2)), (1, 0), SEED + 2, confirm_lines, min_tokens, seen)
    tasks = pool_tasks.make_tasks(random.Random(SEED + 4), n_tasks)
    return {FILES["main"]: _jsonl(main), FILES["confirm"]: _jsonl(confirm),
            FILES["prompts"]: _jsonl(build_prompts(n_prompts)), FILES["tasks"]: _jsonl(tasks)}


def write(out_dir: str | Path, **sizes) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = build(**sizes)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
    return sorted(files)


def check(out_dir: str | Path, **sizes) -> list[str]:
    """Names of files whose bytes on disk differ from a fresh build."""
    out = Path(out_dir)
    return sorted(n for n, text in build(**sizes).items()
                  if not (out / n).is_file() or (out / n).read_bytes() != text.encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(HERE))
    ap.add_argument("--check", action="store_true", help="compare with the files on disk, write nothing")
    ap.add_argument("--main-lines", type=int, default=MAIN_LINES)
    ap.add_argument("--confirm-lines", type=int, default=CONFIRM_LINES)
    ap.add_argument("--min-tokens", type=int, default=MIN_TOKENS, help="Gemma tokens every pool line must exceed")
    ap.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    ap.add_argument("--n-tasks", type=int, default=N_TASKS)
    a = ap.parse_args(argv)
    sizes = dict(main_lines=a.main_lines, confirm_lines=a.confirm_lines, min_tokens=a.min_tokens,
                 n_prompts=a.n_prompts, n_tasks=a.n_tasks)
    if a.check:
        bad = check(a.out, **sizes)
        print("differ: " + ", ".join(bad) if bad else "ok: files match a fresh build")
        return 1 if bad else 0
    for name in write(a.out, **sizes):
        print(f"wrote {Path(a.out) / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
