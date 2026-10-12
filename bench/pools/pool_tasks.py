"""Mechanically checkable tasks for the quality gate (internal to build_pools).

Every task is a question whose gold answer is computed in code, never typed in
by hand, and every family has a second path, `verify`, that re-derives the
answer from the prompt text alone (it parses the numbers, options, code or
constraints back out of the prompt). A bug that makes the prompt and the gold
answer disagree therefore shows up in tests.

Interface:
  make_tasks(rng, n) -> list[dict]       quality2 task rows
                                         {"id","prompt","answer","kind","lang"}
  verify(task) -> answer                 the answer re-derived from task["prompt"]
  math_stems(rng, n) -> list[str]        word problems without the answer-format
                                         suffix (for the quality prompts)

kind is "math" (the reply's last number is compared) or "mc" (a letter A-D).
The prompt always ends with an instruction that makes the answer the last
number, or the only letter, in a short reply.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import io
import itertools
import math
import random
import re

# ---------------------------------------------------------------------------
# answer-format suffixes (the reply is read by quality2.extract_answer)
# ---------------------------------------------------------------------------
MATH_SUFFIX = {
    "en": "Work it out in a few short lines, then end with the line ANSWER: <number> and write nothing after it.",
    "de": "Rechne es in wenigen kurzen Zeilen aus und beende die Antwort mit der Zeile ANSWER: <Zahl>, danach nichts mehr.",
    "fr": "Calcule en quelques lignes courtes, puis termine par la ligne ANSWER: <nombre> sans rien ecrire apres.",
    "es": "Resuelvelo en pocas lineas cortas y termina con la linea ANSWER: <numero> sin escribir nada despues.",
}
MC_SUFFIX = {
    "en": "Reply with only the letter of the correct option.",
    "de": "Antworte nur mit dem Buchstaben der richtigen Option.",
    "fr": "Reponds uniquement par la lettre de la bonne option.",
    "es": "Responde solo con la letra de la opcion correcta.",
}
LETTERS = "ABCD"
SAFE = {"__builtins__": {"len": len, "sorted": sorted, "min": min, "max": max, "sum": sum,
                         "abs": abs, "str": str, "int": int, "range": range, "list": list}}


def _eval(expr: str):
    return eval(expr, dict(SAFE), {})            # noqa: S307 - our own expressions


def _run(code: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, {"__builtins__": __builtins__})   # noqa: S102 - our own snippets
    return buf.getvalue().strip()


def _nums(text: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", text)]


def _options(prompt: str) -> dict[str, str]:
    return {m.group(1): m.group(2).strip()
            for m in re.finditer(r"^([A-D])\) (.*)$", prompt, re.M)}


def _letter_for(options: dict[str, str], text: str) -> str:
    hits = [k for k, v in options.items() if v == text]
    assert len(hits) == 1, (options, text)
    return hits[0]


def _with_options(rng: random.Random, correct: str, wrong: list[str]) -> tuple[str, str]:
    """Option block and the gold letter; the correct one lands in a random slot."""
    opts = list(wrong[:3])
    slot = rng.randrange(4)
    opts.insert(slot, correct)
    assert len(set(opts)) == 4, opts
    return "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(opts)), LETTERS[slot]


PEOPLE = ["Maren", "Idris", "Collette", "Tomasz", "Ngozi", "Halvard", "Priya", "Joaquin", "Linnea", "Saoirse",
          "Dmitri", "Amara", "Bastien", "Yusuf", "Elodie", "Keiko", "Rafael", "Thandi"]
ITEMS = {"en": ["notebooks", "lamps", "chairs", "mugs", "scarves", "plants", "toolkits", "posters"],
         "de": ["Notizbuecher", "Lampen", "Stuehle", "Tassen", "Schals", "Pflanzen", "Werkzeugkoffer", "Poster"],
         "fr": ["carnets", "lampes", "chaises", "tasses", "echarpes", "plantes", "trousses a outils", "affiches"],
         "es": ["cuadernos", "lamparas", "sillas", "tazas", "bufandas", "plantas", "cajas de herramientas", "carteles"]}
WEEKDAYS = {"en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            "de": ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"],
            "fr": ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"],
            "es": ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]}
WORDS = ["amber", "birch", "cobalt", "dune", "ember", "fjord", "garnet", "harbour", "indigo", "juniper", "kelp",
         "lotus", "mesa", "nectar", "onyx", "pine", "quartz", "reef", "sage", "tundra", "umber", "velvet",
         "willow", "yarrow", "zephyr", "anchor", "bridge", "cedar", "delta", "meadow", "orchard", "pebble"]


def _lang(rng: random.Random, p_other: float = 0.3) -> str:
    return rng.choice(["de", "fr", "es"]) if rng.random() < p_other else "en"


# ---------------------------------------------------------------------------
# math families: gen(rng) -> (stem, answer, lang); verify(stem) -> answer
# ---------------------------------------------------------------------------
def g_expr(rng):
    a, b, c, d, e = (rng.randrange(3, 60) for _ in range(5))
    expr = rng.choice([f"{a} * {b} - {c} * ({d} + {e})", f"({a} + {b}) * ({c} - {d}) + {e}",
                       f"{a} * {b} + {c} * {d} - {e} ** 2", f"{a} ** 2 - {b} * {c} + {d} * {e}"])
    return f"Compute {expr}.", _eval(expr), "en"


def v_expr(stem):
    return _eval(re.fullmatch(r"Compute (.+)\.", stem).group(1))


def g_shop(rng):
    lang = _lang(rng)
    while True:
        price, qty = rng.randrange(4, 60), rng.randrange(3, 25)
        pct, ship = rng.choice([10, 20, 25, 40, 50]), rng.randrange(2, 30)
        if (price * qty * pct) % 100 == 0:
            break
    name, item = rng.choice(PEOPLE), rng.choice(ITEMS[lang])
    stem = {
        "en": f"{name} buys {qty} {item} at {price} euros each. A coupon takes {pct} percent off the total, and "
              f"delivery adds a flat {ship} euros. How many euros does {name} pay in all?",
        "de": f"{name} kauft {qty} {item} zu je {price} Euro. Ein Gutschein nimmt {pct} Prozent vom Gesamtpreis ab, "
              f"die Lieferung kostet pauschal {ship} Euro. Wie viele Euro zahlt {name} insgesamt?",
        "fr": f"{name} achete {qty} {item} a {price} euros piece. Un bon de reduction retire {pct} pour cent du total "
              f"et la livraison coute {ship} euros. Combien d'euros {name} paie-t-il en tout ?",
        "es": f"{name} compra {qty} {item} a {price} euros cada uno. Un cupon descuenta el {pct} por ciento del "
              f"total y el envio cuesta {ship} euros fijos. Cuantos euros paga {name} en total?"}[lang]
    return stem, price * qty * (100 - pct) // 100 + ship, lang


def v_shop(stem):
    price, qty, pct, ship = _nums(stem)[:4]
    return price * qty * (100 - pct) // 100 + ship


def g_speed(rng):
    lang = _lang(rng)
    v, h, m, v2, h2 = rng.randrange(40, 120), rng.randrange(1, 6), rng.randrange(10, 50), rng.randrange(30, 110), rng.randrange(1, 5)
    stem = {
        "en": f"A train travels at {v} km/h for {h} hours, waits {m} minutes at a station, then continues at {v2} km/h "
              f"for {h2} hours. How many kilometres does it travel in total?",
        "de": f"Ein Zug faehrt {h} Stunden lang mit {v} km/h, wartet {m} Minuten im Bahnhof und faehrt dann {h2} Stunden "
              f"mit {v2} km/h weiter. Wie viele Kilometer legt er insgesamt zurueck?",
        "fr": f"Un train roule a {v} km/h pendant {h} heures, attend {m} minutes en gare, puis repart a {v2} km/h "
              f"pendant {h2} heures. Combien de kilometres parcourt-il au total ?",
        "es": f"Un tren viaja a {v} km/h durante {h} horas, espera {m} minutos en una estacion y sigue a {v2} km/h "
              f"durante {h2} horas. Cuantos kilometros recorre en total?"}[lang]
    return stem, v * h + v2 * h2, lang


def v_speed(stem):
    n = _nums(stem)
    if stem.startswith("Ein Zug"):                 # de: hours first
        h, v, _, h2, v2 = n[:5]
        return v * h + v2 * h2
    if stem.startswith("Un train") or stem.startswith("Un tren") or stem.startswith("A train"):
        v, h, _, v2, h2 = n[:5]
        return v * h + v2 * h2
    raise AssertionError(stem)


def g_units(rng):
    lang = _lang(rng)
    h, m = rng.randrange(2, 9), rng.randrange(1, 60)
    stem = {"en": f"How many seconds are there in {h} hours and {m} minutes?",
            "de": f"Wie viele Sekunden sind {h} Stunden und {m} Minuten?",
            "fr": f"Combien de secondes y a-t-il dans {h} heures et {m} minutes ?",
            "es": f"Cuantos segundos hay en {h} horas y {m} minutos?"}[lang]
    return stem, h * 3600 + m * 60, lang


def v_units(stem):
    h, m = _nums(stem)[:2]
    return h * 3600 + m * 60


def g_dates(rng):
    d1 = dt.date(2020, 1, 1) + dt.timedelta(days=rng.randrange(0, 1800))
    d2 = d1 + dt.timedelta(days=rng.randrange(20, 700))
    return f"How many days are there from {d1.isoformat()} to {d2.isoformat()}?", (d2 - d1).days, "en"


def v_dates(stem):
    d1, d2 = re.findall(r"\d{4}-\d\d-\d\d", stem)
    return (dt.date.fromisoformat(d2) - dt.date.fromisoformat(d1)).days


def g_modpow(rng):
    m = rng.choice([7, 11, 13, 17, 19, 23, 29, 31])
    a, e = rng.choice([x for x in range(2, 30) if x % m]), rng.randrange(5, 60)
    return f"What is the remainder when {a}^{e} is divided by {m}?", pow(a, e, m), "en"


def v_modpow(stem):
    a, e, m = _nums(stem)[:3]
    return pow(a, e, m)


def g_pyloop(rng):
    a = rng.randrange(0, 20)
    b = a + rng.randrange(30, 120)
    s, m = rng.randrange(1, 4), rng.randrange(2, 7)
    r, k = rng.randrange(0, m), rng.randrange(2, 9)
    code = (f"total = 0\nfor i in range({a}, {b}, {s}):\n    if i % {m} == {r}:\n        total += i * {k}\nprint(total)")
    return f"What number does this Python program print?\n```\n{code}\n```", int(_run(code)), "en"


def v_pyloop(stem):
    return int(_run(re.search(r"```\n(.*?)\n```", stem, re.S).group(1)))


def g_seq(rng):
    a, d, n = rng.randrange(1, 40), rng.randrange(2, 15), rng.randrange(8, 40)
    return (f"The first term of an arithmetic sequence is {a} and each term is {d} more than the one before. "
            f"What is the sum of the first {n} terms?"), n * (2 * a + (n - 1) * d) // 2, "en"


def v_seq(stem):
    a, d, n = _nums(stem)[:3]
    return sum(a + i * d for i in range(n))


def g_gcd(rng):
    g = rng.randrange(3, 25)
    a, b = g * rng.randrange(2, 30), g * rng.randrange(2, 30)
    if rng.random() < 0.5:
        return f"What is the greatest common divisor of {a} and {b}?", math.gcd(a, b), "en"
    return f"What is the least common multiple of {a} and {b}?", math.lcm(a, b), "en"


def v_gcd(stem):
    a, b = _nums(stem)[:2]
    return math.gcd(a, b) if "greatest common" in stem else math.lcm(a, b)


def _is_prime(n: int) -> bool:
    return n > 1 and all(n % p for p in range(2, int(n ** 0.5) + 1))


def g_countprimes(rng):
    a = rng.randrange(2, 100)
    b = a + rng.randrange(40, 200)
    return (f"How many prime numbers are there from {a} to {b}, both ends included?",
            sum(_is_prime(n) for n in range(a, b + 1)), "en")


def v_countprimes(stem):
    a, b = _nums(stem)[:2]
    return sum(_is_prime(n) for n in range(a, b + 1))


def g_percent(rng):
    while True:
        p = 400 * rng.randrange(1, 30)
        x, y = rng.choice([10, 20, 25, 50]), rng.choice([10, 20, 25, 40])
        if (p * (100 + x) * (100 - y)) % 10000 == 0:
            break
    return (f"A jacket costs {p} euros. The price goes up by {x} percent, and a month later it goes down by {y} "
            f"percent of the new price. What does it cost now, in euros?"), p * (100 + x) * (100 - y) // 10000, "en"


def v_percent(stem):
    p, x, y = _nums(stem)[:3]
    return p * (100 + x) * (100 - y) // 10000


def g_letters(rng):
    ws = [rng.choice(WORDS) for _ in range(rng.randrange(3, 6))]
    sentence = " ".join(ws)
    c = rng.choice("aeiorst")
    return f"How many times does the letter '{c}' appear in the text \"{sentence}\"?", sentence.count(c), "en"


def v_letters(stem):
    c = re.search(r"letter '(.)'", stem).group(1)
    return re.search(r'text "(.*)"\?', stem).group(1).count(c)


def g_ages(rng):
    b = rng.randrange(5, 40)
    a, y = rng.randrange(2, 20), rng.randrange(2, 15)
    s = b + (b + a) + 2 * y
    n1, n2 = rng.sample(PEOPLE, 2)
    return (f"{n1} is {a} years older than {n2}. In {y} years, the sum of their ages will be {s}. "
            f"How old is {n2} now?"), b, "en"


def v_ages(stem):
    a, y, s = _nums(stem)[:3]
    return (s - a - 2 * y) // 2


MATH_FAMILIES = [("expr", g_expr, v_expr), ("shop", g_shop, v_shop), ("speed", g_speed, v_speed),
                 ("units", g_units, v_units), ("dates", g_dates, v_dates), ("modpow", g_modpow, v_modpow),
                 ("pyloop", g_pyloop, v_pyloop), ("seq", g_seq, v_seq), ("gcd", g_gcd, v_gcd),
                 ("primes", g_countprimes, v_countprimes), ("percent", g_percent, v_percent),
                 ("letters", g_letters, v_letters), ("ages", g_ages, v_ages)]


# ---------------------------------------------------------------------------
# multiple-choice families: gen(rng) -> (stem_with_options, letter, lang)
# ---------------------------------------------------------------------------
def _distinct_ints(rng, correct: int, spread: int = 12) -> list[int]:
    out: list[int] = []
    while len(out) < 3:
        x = correct + rng.choice([-1, 1]) * rng.randrange(1, spread)
        if x != correct and x not in out:
            out.append(x)
    return out


def g_largest(rng):
    while True:
        exprs = [f"{rng.randrange(3, 40)} * {rng.randrange(3, 40)} - {rng.randrange(1, 90)}" for _ in range(4)]
        vals = [_eval(e) for e in exprs]
        if len(set(vals)) == 4:
            break
    big = rng.random() < 0.5
    want = max(vals) if big else min(vals)
    opts = "\n".join(f"{LETTERS[i]}) {e}" for i, e in enumerate(exprs))
    word = "greatest" if big else "smallest"
    return f"Which of these expressions has the {word} value?\n{opts}", LETTERS[vals.index(want)], "en"


def v_largest(stem):
    o = _options(stem)
    vals = {k: _eval(v) for k, v in o.items()}
    pick = max if "greatest" in stem else min
    return pick(vals, key=vals.get)


def g_prime(rng):
    lang = _lang(rng)
    while True:
        p = rng.randrange(20, 400)
        if _is_prime(p):
            break
    comps = []
    while len(comps) < 3:
        c = rng.randrange(2, 20) * rng.randrange(2, 20)
        if not _is_prime(c) and c not in comps and c != p and c > 3:
            comps.append(c)
    block, letter = _with_options(rng, str(p), [str(c) for c in comps])
    q = {"en": "Which of these numbers is prime?", "de": "Welche dieser Zahlen ist eine Primzahl?",
         "fr": "Lequel de ces nombres est premier ?", "es": "Cual de estos numeros es primo?"}[lang]
    return f"{q}\n{block}", letter, lang


def v_prime(stem):
    return next(k for k, v in _options(stem).items() if _is_prime(int(v)))


def g_pyout(rng):
    n, k, m = rng.randrange(4, 12), rng.randrange(2, 6), rng.randrange(3, 9)
    code = rng.choice([
        f"x = {n}\nfor _ in range({k}):\n    x = (x * {m}) % {n + 7}\nprint(x)",
        f"words = {[rng.choice(WORDS) for _ in range(n % 4 + 3)]!r}\nprint(len(''.join(words)) + {k})",
        f"def f(a, b={k}):\n    return a * b + {m}\nprint(f({n}) + f({m}, {n}))",
        f"total = 0\nfor i in range({n}):\n    total += i % {k}\nprint(total)"])
    out = int(_run(code))
    block, letter = _with_options(rng, str(out), [str(x) for x in _distinct_ints(rng, out)])
    return f"What does this Python program print?\n```\n{code}\n```\n{block}", letter, "en"


def v_pyout(stem):
    out = _run(re.search(r"```\n(.*?)\n```", stem, re.S).group(1))
    return _letter_for(_options(stem), out)


def g_weekday(rng):
    lang = _lang(rng)
    d = dt.date(2019, 1, 1) + dt.timedelta(days=rng.randrange(0, 2800))
    names = WEEKDAYS[lang]
    right = names[d.weekday()]
    wrong = [n for n in rng.sample(names, 7) if n != right][:3]
    block, letter = _with_options(rng, right, wrong)
    q = {"en": f"Which day of the week was {d.isoformat()}?", "de": f"Welcher Wochentag war der {d.isoformat()}?",
         "fr": f"Quel jour de la semaine etait le {d.isoformat()} ?",
         "es": f"Que dia de la semana fue el {d.isoformat()}?"}[lang]
    return f"{q}\n{block}", letter, lang


def v_weekday(stem):
    d = dt.date.fromisoformat(re.search(r"\d{4}-\d\d-\d\d", stem).group(0))
    o = _options(stem)
    for names in WEEKDAYS.values():
        if all(v in names for v in o.values()):
            return _letter_for(o, names[d.weekday()])
    raise AssertionError(stem)


def g_order(rng):
    people = rng.sample(PEOPLE, 4)
    order = people[:]                                 # tallest first
    rng.shuffle(order)
    chain = [f"{order[i]} is taller than {order[i + 1]}." for i in range(3)]
    extra = [f"{order[0]} is taller than {order[2]}.", f"{order[1]} is taller than {order[3]}."]
    cons = chain + ([rng.choice(extra)] if rng.random() < 0.5 else [])   # the chain alone fixes the order
    rng.shuffle(cons)
    rank = rng.choice([1, 2, 3, 4])
    word = {1: "tallest", 2: "second tallest", 3: "second shortest", 4: "shortest"}[rank]
    block, letter = _with_options(rng, order[rank - 1], [p for p in people if p != order[rank - 1]])
    return f"{' '.join(cons)} Who is the {word}?\n{block}", letter, "en"


def _solutions(cons: list[str], people: list[str]) -> list[tuple]:
    pairs = [re.fullmatch(r"(\w+) is taller than (\w+)\.", c).groups() for c in cons]
    sols = []
    for perm in itertools.permutations(people):          # tallest first
        idx = {p: i for i, p in enumerate(perm)}
        if all(idx[a] < idx[b] for a, b in pairs):
            sols.append(perm)
    return sols


def v_order(stem):
    body = stem.split("\n")[0]
    cons = re.findall(r"\w+ is taller than \w+\.", body)
    o = _options(stem)
    sols = _solutions(cons, list(o.values()))
    assert len(sols) == 1, stem
    rank = {"tallest": 1, "second tallest": 2, "second shortest": 3, "shortest": 4}[
        re.search(r"Who is the (.*)\?", body).group(1)]
    return _letter_for(o, sols[0][rank - 1])


def g_alpha(rng):
    ws = rng.sample(WORDS, 4)
    last = rng.random() < 0.5
    right = max(ws) if last else min(ws)
    block, letter = _with_options(rng, right, [w for w in ws if w != right])
    return (f"Which of these words comes {'last' if last else 'first'} in alphabetical order?\n{block}"), letter, "en"


def v_alpha(stem):
    o = _options(stem)
    pick = max if "comes last" in stem else min
    return pick(o, key=lambda k: o[k])


def g_divcount(rng):
    k = rng.choice([3, 4, 6, 7, 9])
    xs = [rng.randrange(10, 300) for _ in range(rng.randrange(6, 10))]
    right = sum(x % k == 0 for x in xs)
    wrong = [str(x) for x in range(0, 10) if x != right]
    rng.shuffle(wrong)
    block, letter = _with_options(rng, str(right), wrong[:3])
    return (f"How many of the following numbers are divisible by {k}? {', '.join(map(str, xs))}\n{block}"), letter, "en"


def v_divcount(stem):
    m = re.match(r"How many of the following numbers are divisible by (\d+)\? (.*)\n", stem)
    k, xs = int(m.group(1)), [int(x) for x in m.group(2).split(", ")]
    return _letter_for(_options(stem), str(sum(x % k == 0 for x in xs)))


def g_pyexpr(rng):
    w = rng.choice(WORDS)
    n, k = rng.randrange(2, 9), rng.randrange(1, 30)
    xs = [rng.randrange(1, 90) for _ in range(5)]
    expr, kind = rng.choice([(f"len('{w}') * {n} - {k}", "int"), (f"'{w}'[::-1][1:4]", "str"),
                             (f"sorted({xs})[2] + {k}", "int"), (f"max({xs}) - min({xs})", "int")])
    val = _eval(expr)
    if kind == "int":
        wrong = [str(x) for x in _distinct_ints(rng, val)]
        right = str(val)
    else:
        right = val
        wrong = []
        for x in [w[:3], w[1:4], w[::-1][:3], w[::-1][2:5], w[2:5], w[::-1][:2]]:
            if x != right and x not in wrong:
                wrong.append(x)
    block, letter = _with_options(rng, right, wrong)
    return f"What does the Python expression `{expr}` evaluate to?\n{block}", letter, "en"


def v_pyexpr(stem):
    expr = re.search(r"`(.*)`", stem).group(1)
    return _letter_for(_options(stem), str(_eval(expr)))


UNIT_M = {"m": 1.0, "km": 1000.0, "cm": 0.01, "miles": 1609.344, "ft": 0.3048}


def g_length(rng):
    while True:
        picks = rng.sample(list(UNIT_M), 4)
        qs = [(rng.randrange(2, 400) / 10 if u in ("km", "miles") else rng.randrange(20, 9000), u) for u in picks]
        meters = [q * UNIT_M[u] for q, u in qs]
        s = sorted(meters)
        if all(s[i + 1] > s[i] * 1.08 for i in range(3)):
            break
    longest = rng.random() < 0.5
    want = max(meters) if longest else min(meters)
    texts = [f"{q:g} {u}" for q, u in qs]
    opts = "\n".join(f"{LETTERS[i]}) {t}" for i, t in enumerate(texts))
    return f"Which of these is the {'longest' if longest else 'shortest'} distance?\n{opts}", \
        LETTERS[meters.index(want)], "en"


def v_length(stem):
    o = _options(stem)
    m = {k: float(v.split()[0]) * UNIT_M[v.split()[1]] for k, v in o.items()}
    return (max if "longest" in stem else min)(m, key=m.get)


def g_logic(rng):
    a, b, c = (rng.choice([True, False]) for _ in range(3))
    while True:
        exprs = [rng.choice(["a and b", "a or c", "not a and c", "b and not c", "not (a or b)", "(a and b) or c",
                             "a and (b or c)", "not b or c", "not (b and c)", "a != b", "b == c", "(a or b) and not c"])
                 for _ in range(4)]
        if len(set(exprs)) == 4:
            vals = [_eval_bool(e, a, b, c) for e in exprs]
            if sum(vals) == 1:
                break
        a, b, c = (rng.choice([True, False]) for _ in range(3))
    opts = "\n".join(f"{LETTERS[i]}) {e}" for i, e in enumerate(exprs))
    return (f"Suppose a = {a}, b = {b} and c = {c}. Which one of these Python expressions is True?\n{opts}"), \
        LETTERS[vals.index(True)], "en"


def _eval_bool(e, a, b, c):
    return bool(eval(e, {"__builtins__": {}}, {"a": a, "b": b, "c": c}))   # noqa: S307


def v_logic(stem):
    a, b, c = (x == "True" for x in re.search(r"a = (\w+), b = (\w+) and c = (\w+)", stem).groups())
    o = _options(stem)
    hits = [k for k, e in o.items() if _eval_bool(e, a, b, c)]
    assert len(hits) == 1, stem
    return hits[0]


def g_shift(rng):
    lang = _lang(rng)
    names = WEEKDAYS[lang]
    i, n = rng.randrange(7), rng.randrange(8, 400)
    right = names[(i + n) % 7]
    wrong = [x for x in rng.sample(names, 7) if x != right][:3]
    block, letter = _with_options(rng, right, wrong)
    q = {"en": f"Today is {names[i]}. What day of the week will it be in {n} days?",
         "de": f"Heute ist {names[i]}. Welcher Wochentag ist in {n} Tagen?",
         "fr": f"Aujourd'hui, c'est {names[i]}. Quel jour sera-t-on dans {n} jours ?",
         "es": f"Hoy es {names[i]}. Que dia de la semana sera dentro de {n} dias?"}[lang]
    return f"{q}\n{block}", letter, lang


def v_shift(stem):
    o = _options(stem)
    n = int(re.search(r"(\d+) (?:days|Tagen|jours|dias)", stem).group(1))
    for names in WEEKDAYS.values():
        if all(v in names for v in o.values()):
            today = next(x for x in names if x in stem.split("\n")[0])
            return _letter_for(o, names[(names.index(today) + n) % 7])
    raise AssertionError(stem)


def g_nth(rng):
    xs = rng.sample(range(10, 500), rng.randrange(6, 10))
    k = rng.randrange(2, 5)
    right = sorted(xs)[k - 1]
    wrong = [str(x) for x in xs if x != right][:3]
    block, letter = _with_options(rng, str(right), wrong)
    return (f"Sort these numbers from smallest to largest: {', '.join(map(str, xs))}. "
            f"Which number is in position {k}?\n{block}"), letter, "en"


def v_nth(stem):
    m = re.match(r"Sort these numbers from smallest to largest: (.*)\. Which number is in position (\d+)\?", stem)
    xs = sorted(int(x) for x in m.group(1).split(", "))
    return _letter_for(_options(stem), str(xs[int(m.group(2)) - 1]))


MC_FAMILIES = [("largest", g_largest, v_largest), ("prime", g_prime, v_prime), ("pyout", g_pyout, v_pyout),
               ("weekday", g_weekday, v_weekday), ("order", g_order, v_order), ("alpha", g_alpha, v_alpha),
               ("divcount", g_divcount, v_divcount), ("pyexpr", g_pyexpr, v_pyexpr),
               ("length", g_length, v_length), ("logic", g_logic, v_logic), ("shift", g_shift, v_shift),
               ("nth", g_nth, v_nth)]

_VERIFY = {name: v for name, _, v in MATH_FAMILIES + MC_FAMILIES}


# ---------------------------------------------------------------------------
# interface
# ---------------------------------------------------------------------------
def make_tasks(rng: random.Random, n: int) -> list[dict]:
    """n tasks, half math and half multiple-choice, families in rotation. A
    repeated prompt is redrawn, so no two tasks are the same question."""
    n_math = n // 2
    plan = ([("math", MATH_FAMILIES[i % len(MATH_FAMILIES)]) for i in range(n_math)]
            + [("mc", MC_FAMILIES[i % len(MC_FAMILIES)]) for i in range(n - n_math)])
    seen, out = set(), []
    counters: dict[str, int] = {}
    for kind, (fam, gen, _) in plan:
        for _try in range(100):
            stem, answer, lang = gen(rng)
            suffix = (MATH_SUFFIX if kind == "math" else MC_SUFFIX)[lang]
            prompt = f"{stem}\n\n{suffix}"
            if prompt not in seen:
                break
        else:
            raise RuntimeError(f"family {fam} cannot produce enough distinct tasks")
        seen.add(prompt)
        i = counters[kind] = counters.get(kind, 0) + 1
        out.append({"id": f"t-{kind}-{fam}-{i:03d}", "prompt": prompt, "answer": answer, "kind": kind,
                    "lang": lang})
    return out


def verify(task: dict):
    """The answer re-derived from the prompt text alone."""
    fam = task["id"].split("-")[2]
    stem = task["prompt"].rsplit("\n\n", 1)[0]
    return _VERIFY[fam](stem)


def math_stems(rng: random.Random, n: int) -> list[str]:
    """n distinct word problems (no answer-format suffix), families in rotation."""
    out: list[str] = []
    for i in range(n):
        _, gen, _ = MATH_FAMILIES[i % len(MATH_FAMILIES)]
        for _try in range(100):
            stem = gen(rng)[0]
            if stem not in out:
                break
        out.append(stem)
    return out
