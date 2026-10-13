#!/usr/bin/env python3
"""Build the Gemma 4 quality prompts and quality tasks.

The files in this directory are the output of this script; rebuild them with

    python build_pools.py --out .            # writes the three files
    python build_pools.py --out . --check    # exit 1 if the files on disk differ

and the bytes are the same on every machine (one seed, `random.Random` only,
no tokenizer, no network). Nothing here is scraped or copied: the text comes
from the banks and templates in pool_text.py and the tasks from pool_tasks.py.

The speed pools are not made here: real_pool.py copies them from the request
files in real/.

Interface:
  build(n_prompts, n_tasks) -> {filename: text}
  write(out_dir, **sizes)  /  check(out_dir, **sizes) -> list of differing files
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
FILES = {"prompts": "gemma4-quality-prompts.jsonl",
         "prompts_6144": "gemma4-quality-prompts-6144.jsonl",
         "tasks": "gemma4-quality-tasks.jsonl"}

N_PROMPTS = 256
N_TASKS = 200
MAX_PROMPT_TOKENS = 7800     # 8192 context - 128 forced tokens - margin
# The prompts a 6144 context cannot hold: over 6100 Gemma tokens (6108 to 7744;
# the longest that fits is 5873). The builder has no tokenizer, so they are
# listed by id; tests/test_pools.py checks the list with the real one.
OVER_6144 = frozenset({
    "q-0037-retrieve", "q-0040-summary", "q-0041-retrieve", "q-0056-retrieve",
    "q-0063-code", "q-0075-code", "q-0078-retrieve", "q-0086-code",
    "q-0095-summary", "q-0133-retrieve", "q-0188-code", "q-0210-retrieve",
    "q-0216-code", "q-0230-retrieve", "q-0234-summary", "q-0241-retrieve"})

# Typical characters per Gemma token, used to aim a quality prompt at a token
# length.
MID = {"prose": 4.65, "code": 3.1}

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
def build(n_prompts: int = N_PROMPTS, n_tasks: int = N_TASKS) -> dict[str, str]:
    """{filename: file text} for the three data files. The 6144 file is the
    prompts file without the rows named in OVER_6144, in the same order."""
    prompts = build_prompts(n_prompts)
    tasks = pool_tasks.make_tasks(random.Random(SEED + 4), n_tasks)
    return {FILES["prompts"]: _jsonl(prompts),
            FILES["prompts_6144"]: _jsonl([r for r in prompts if r["id"] not in OVER_6144]),
            FILES["tasks"]: _jsonl(tasks)}


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
    ap.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    ap.add_argument("--n-tasks", type=int, default=N_TASKS)
    a = ap.parse_args(argv)
    sizes = dict(n_prompts=a.n_prompts, n_tasks=a.n_tasks)
    if a.check:
        bad = check(a.out, **sizes)
        print("differ: " + ", ".join(bad) if bad else "ok: files match a fresh build")
        return 1 if bad else 0
    for name in write(a.out, **sizes):
        print(f"wrote {Path(a.out) / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
