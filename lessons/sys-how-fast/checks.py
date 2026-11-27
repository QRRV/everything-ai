# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson sys-how-fast: how fast a computer calculates, and how fast a model writes. FLOP/s from the
measured times of matrix products, tokens per second from a clock around a writer, and on the learner's own computer:
where the time of one character goes, and how many characters per second many texts at once give.

"The course's computer" is the computer the course was made on, with 4 cores: its times are the fastest of 9 runs of
each matrix product, and of a few runs of the bigram writer, in float32. Every computer gives other times; the checks
look at what a function makes of times, not at the times of the learner's computer."""
import contextlib
import html
import io
import math
import time

import numpy as np
import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import chart, check, paths, yours
from everything_ai.check import FunctionBox, Mistake, Timeout

LESSON = "sys-how-fast"
LANGS = ("en", "nl")
DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md

# The matrix sizes of the measurements, and the time of one n x n product on the course's computer, in microseconds.
SIZES = [8, 16, 32, 64, 128, 256, 512, 1024]
COURSE_MICROS = [1.1, 1.3, 1.5, 7.4, 17.0, 83.0, 650.0, 4800.0]

# One character of the bigram writer on the course's computer, in microseconds per part, and per step in all,
# with the loop around the parts.
PARTS = ("onehot", "product", "softmax", "draw")
COURSE_PARTS = {"onehot": 2.5, "product": 1.9, "softmax": 1.3, "draw": 14.5}
COURSE_STEP = 20.5
# Many texts at once on the course's computer: the microseconds of one step, which gives every text one character.
TEXTS = (1, 10, 100, 1000)
COURSE_TEXTS = {1: 20.5, 10: 45.0, 100: 180.0, 1000: 1300.0}
STEPS = 2000          # the characters the demo writes for the parts of one step
TEXT_STEPS = 200      # the steps the demo takes for each number of texts
PRODUCT_FLOPS = 2 * 65 * 65   # one one-hot row times the 65 x 65 table: 8,450 FLOPs


# --- the functions of the exercises, as the course writes them ----------------------------------------------------------

def flop_per_second(sizes, micros):
    rates = []
    for i in range(len(sizes)):
        n = sizes[i]
        seconds = micros[i] / 10 ** 6
        rates.append(2 * n ** 3 / seconds)
    return rates


def tokens_per_second(write, n):
    start = time.perf_counter()
    write(n)
    seconds = time.perf_counter() - start
    return n / seconds


def course_rates() -> list:
    return flop_per_second(SIZES, COURSE_MICROS)


# --- the course's bigram writer -----------------------------------------------------------------------------------------

def _table():
    """The 65 characters of the plays, and the bigram table counted on all of them, with 1 added to every count, as
    logits: softmax of a row gives that row's probabilities back."""
    text = (paths.ROOT / DATA).read_text(encoding="utf-8")
    chars = sorted(set(text))
    index = {c: i for i, c in enumerate(chars)}
    ids = np.fromiter((index[c] for c in text), dtype=np.int64, count=len(text))
    counts = np.ones((len(chars), len(chars)))
    np.add.at(counts, (ids[:-1], ids[1:]), 1)
    P = counts / counts.sum(axis=1, keepdims=True)
    return chars, torch.log(torch.tensor(P, dtype=torch.float32))


def table():
    return check.kept((LESSON, "table"), _table)


def write(n: int, texts: int = 1, seed: int = 1, spent: dict | None = None) -> list:
    """`texts` texts of n characters each, all starting with "T", written by the course's bigram table one character
    per step: the one-hot rows of the last characters times the table, softmax, and a draw of the next character.
    With `spent`, a dict of lists, the seconds of each part of every step, and of the whole step, are added to it."""
    chars, W = table()
    eye = torch.eye(len(chars))
    g = torch.Generator().manual_seed(seed)
    x = torch.full((texts,), chars.index("T"), dtype=torch.long)
    out = [x]
    clock = time.perf_counter
    for _ in range(n - 1):
        if spent is None:
            p = torch.softmax(eye[x] @ W, dim=1)
            x = torch.multinomial(p, 1, generator=g)[:, 0]
        else:
            t = clock()
            X = eye[x]
            t2 = clock()
            logits = X @ W
            t3 = clock()
            p = torch.softmax(logits, dim=1)
            t4 = clock()
            x = torch.multinomial(p, 1, generator=g)[:, 0]
            t5 = clock()
            spent["onehot"].append(t2 - t)
            spent["product"].append(t3 - t2)
            spent["softmax"].append(t4 - t3)
            spent["draw"].append(t5 - t4)
            spent["step"].append(t5 - t)
        out.append(x)
    codes = torch.stack(out, dim=1).tolist()
    return ["".join(chars[i] for i in row) for row in codes]


class slow_writer:
    """A stand-in for `write` in the checks: it takes `seconds` seconds, whatever n, and writes nothing."""

    def __init__(self, seconds: float):
        self.seconds = seconds

    def __call__(self, n):
        time.sleep(self.seconds)
        return ""

    def __repr__(self):
        return f"slow_writer({self.seconds})"


# --- measuring on the learner's computer --------------------------------------------------------------------------------

def _fastest(make, repeats: int) -> float:
    best = None
    for _ in range(repeats):
        start = time.perf_counter()
        make()
        seconds = time.perf_counter() - start
        if best is None or seconds < best:
            best = seconds
    return best


def measure_micros(repeats: int = 5) -> list:
    """The microseconds of one n x n product for every size of SIZES on this computer, the fastest of `repeats`."""
    out = []
    for n in SIZES:
        a, b = torch.ones((n, n)), torch.ones((n, n))
        a @ b   # the first product of a size also prepares it
        out.append(_fastest(lambda: a @ b, repeats if n < 1024 else 3) * 10 ** 6)
    return out


def my_micros() -> list:
    """This computer's times, measured once when a picture first needs them, so a picture that redraws keeps them."""
    return check.kept((LESSON, "micros"), measure_micros)


def measure_parts(steps: int = STEPS) -> dict:
    """The microseconds of each part of one character of one text, and of the whole step, on this computer: the
    median of all steps, so that a step in which the computer did something else for a moment does not count."""
    table()
    write(50)   # PyTorch prepares each operation the first time
    spent = {k: [] for k in PARTS + ("step",)}
    write(steps + 1, spent=spent)
    return {k: float(np.median(v)) * 10 ** 6 for k, v in spent.items()}


def measure_texts(steps: int = TEXT_STEPS, counts=TEXTS) -> dict:
    """For each number of texts at once, the microseconds of one step on this computer, the fastest of 3 runs."""
    table()
    out = {}
    for k in counts:
        write(20, texts=k)
        out[k] = _fastest(lambda: write(steps + 1, texts=k), 3) / steps * 10 ** 6
    return out


def demo() -> dict:
    """What the demo measures: the parts of one character, and steps with many texts at once."""
    return {"parts": measure_parts(), "texts": measure_texts()}


# --- numbers and words -------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-2);margin-top:0.3rem;max-width:var(--ea-measure)"
_LOSS_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-loss-text);margin-top:0.3rem;max-width:var(--ea-measure)"


def number(x: float, digits: int, lang: str) -> str:
    """A number for the lesson's text: a decimal comma and a thousands dot in Dutch."""
    out = f"{x:,.{digits}f}"
    return out.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else out


def big(x: float, lang: str) -> str:
    """A large rate in words, with two or three digits: 6.3 billion, 413 billion, 49,000; in Dutch 6,3 miljard."""
    words = {"en": ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")),
             "nl": ((1e12, "biljoen"), (1e9, "miljard"), (1e6, "miljoen"))}[lang]
    for size, word in words:
        if x >= size:
            v = x / size
            return f"{number(v, 1 if v < 10 else 0, lang)} {word}"
    return number(round(x, -3) if x >= 10_000 else round(x), 0, lang)


def _marked(text_: str) -> str:
    """Escaped text in which `name` is set as code and *word* in italics."""
    out = []
    for k, part in enumerate(html.escape(text_).split("`")):
        if k % 2:
            out.append(f"<code>{part}</code>")
        else:
            out.append("".join(f"<i>{p}</i>" if j % 2 else p for j, p in enumerate(part.split("*"))))
    return "".join(out)


def _note(text_: str, style: str = _UI) -> str:
    return f'<div style="{style}">{_marked(text_)}</div>'


def _figure(svgs: list, alt: str, width: int) -> str:
    return (f'<figure class="ea-figure" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.5rem;'
            f'max-width:min(100%,{width}px);display:flex;flex-wrap:wrap;gap:0.6rem 1rem;align-items:flex-start">'
            f'{"".join(svgs)}</figure>')


def _text_block(text_: str) -> str:
    """What the model wrote, in a box that wraps its lines, so that a long line stays readable on a phone."""
    return (f'<pre style="white-space:pre-wrap;word-break:break-word;margin:0.25rem 0 0.75rem;padding:0.6rem 0.8rem;'
            f'border-radius:6px;background:var(--ea-inset);color:var(--ea-ink);font-family:var(--ea-font-code);'
            f'font-size:0.875rem;line-height:1.45;max-width:var(--ea-measure)">{html.escape(text_)}</pre>')


# --- drawing ------------------------------------------------------------------------------------------------------------

_RATE_TEXT = {
    "en": {"x": "n, for two n × n matrices", "y": "FLOP/s", "course": "the course's computer", "mine": "your computer",
           "idea_title": "How fast a matrix product goes",
           "idea_alt": "FLOP/s of an n × n matrix product on the course's computer, for n from 8 to 1,024, on log scales: about 0.9 billion at 8, 6.3 billion at 16, 44 billion at 32, then rising to about 250 billion at 128 and 450 billion at 1,024."},
    "nl": {"x": "n, voor twee n × n-matrices", "y": "FLOP/s", "course": "de computer van de course", "mine": "jouw computer",
           "idea_title": "Hoe snel een matrixproduct gaat",
           "idea_alt": "FLOP/s van een n × n-matrixproduct op de computer van de course, voor n van 8 tot en met 1.024, op logaritmische assen: ongeveer 0,9 miljard bij 8, 6,3 miljard bij 16, 44 miljard bij 32, en dan oplopend tot ongeveer 250 miljard bij 128 en 450 miljard bij 1.024."},
}


def _bounds(values: list) -> tuple[float, float]:
    """Powers of 10 around the positive values, at least two apart."""
    good = [v for v in values if v > 0 and math.isfinite(v)]
    lo = 10 ** math.floor(math.log10(min(good))) if good else 1e8
    hi = 10 ** math.ceil(math.log10(max(good))) if good else 1e12
    return lo, max(hi, lo * 100)


def rates_svg(lines: list, title: str, lang: str, alt: str) -> str:
    """A chart of FLOP/s against n on log scales: lines as (rates, role, dashed, label). The label of the line that
    ends highest goes above its end, the other one below its own end, so that two labels never cover each other."""
    w = _RATE_TEXT[lang]
    lo, hi = _bounds([v for rates, _, _, _ in lines for v in rates])
    p = chart.Panel(x=(6, 1400), y=(lo, hi), title=title, xlabel=w["x"], ylabel=w["y"], xlog=True, ylog=True,
                    xticks=[8, 64, 512], xtick_labels=["8", "64", "512"])
    top = max(range(len(lines)), key=lambda i: lines[i][0][-1])
    for i, (rates, role, dashed, label) in enumerate(lines):
        p.line(SIZES, rates, role=role, dashed=dashed)
        if not dashed:
            p.points(SIZES, rates, role=role, r=3.5)
        last = rates[-1]
        if last > 0 and math.isfinite(last):
            y = min(max(last * (1.8 if i == top else 0.45), lo * 1.5), hi / 1.25)
            p.text(1300, y, label, role=role if role != "ink-3" else "ink-2", anchor="end")
    return chart.figure([p], alt=alt, width=330, height=260)


def idea_picture(lang: str) -> str:
    w = _RATE_TEXT[lang]
    return rates_svg([(course_rates(), "data", False, w["course"])], w["idea_title"], lang, w["idea_alt"])


def bars_svg(rows: list, title: str, unit: str, lang: str, alt: str, top: float | None = None) -> str:
    """One bar per row, (label, value, role), from 0 to its value, with its label above it and its value at its end."""
    BAR, ROW = 14, 40
    W = 322 - 7.4 * max(len(value_text(v, unit, lang)) for _, v, _ in rows)   # room for the longest value after its bar
    top = top or max(v for _, v, _ in rows) or 1
    parts, y = [], 0
    if title:
        parts.append(f'<text x="0" y="15" style="font-size:14px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
        y = 28
    for label, value, role in rows:
        width = max(2.0, W * value / top)
        color = "var(--ea-ink)" if role == "ink" else f"var(--ea-{role})"
        parts.append(f'<text x="0" y="{y + 12}" style="font-size:12.5px;fill:var(--ea-ink-2)">{html.escape(label)}</text>'
                     f'<rect x="0" y="{y + 17}" width="{width:.1f}" height="{BAR}" rx="2" style="fill:{color}"/>'
                     f'<text x="{width + 6:.1f}" y="{y + 29}" style="font-size:12.5px;font-weight:600;fill:var(--ea-ink)">'
                     f'{html.escape(value_text(value, unit, lang))}</text>')
        y += ROW
    return (f'<svg viewBox="0 0 330 {y + 4}" role="img" aria-label="{html.escape(alt)}" style="width:100%;max-width:330px;'
            f'height:auto;font-family:var(--ea-font-ui);font-variant-numeric:tabular-nums">{"".join(parts)}</svg>')


def value_text(value: float, unit: str, lang: str) -> str:
    if unit == "µs":
        return f"{number(value, 1, lang)} µs"
    return f"{big(value, lang)} {unit}"


# --- the pictures of the learner's work -----------------------------------------------------------------------------------

_STOPPED = object()   # what _call gives for a function that stopped with an error
_SLOW = object()      # what _call gives for a function that ran out of time
PICTURE_SECONDS = 3   # what the learner's function gets for all the calls of one picture together


def _fn(seen):
    run_ = seen.mine
    if run_ is None:
        return None
    fn = run_.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(make_, seconds: float = PICTURE_SECONDS):
    """make_() within a time limit, with what it prints thrown away; _SLOW when the time runs out, _STOPPED when it
    stops with an error."""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return check.with_time_limit(make_, max(seconds, 0.05))
    except Timeout:
        return _SLOW
    except Exception:   # noqa: BLE001   the check above says what went wrong
        return _STOPPED


def _numbers(v, n: int) -> list | None:
    """v as a list of n finite floats, or None: never from text or True and False."""
    try:
        items = list(v)
        if len(items) != n or any(isinstance(x, (bool, str, bytes)) for x in items):
            return None
        out = [float(x) for x in items]
    except (TypeError, ValueError):
        return None
    return out if all(math.isfinite(x) for x in out) else None


_YOURS_TEXT = {
    "en": {"rates": {"title": "Your flop_per_second",
                     "intro": "Your `flop_per_second` on the times of an n × n product measured just now on your computer, and dashed on the times of the course's computer.",
                     "chart": "Your flop_per_second",
                     "alt": "FLOP/s for n from 8 to 1,024 on log scales, worked out by your function: on your computer, and dashed on the course's computer.",
                     "say": "On your computer: {lo} FLOP/s at 8 × 8, and {hi} at 1,024 × 1,024.",
                     "none": "Your `flop_per_second` stops with an error, takes too long, or does not give one number for each size, so nothing can be drawn yet."},
           "tokens": {"title": "Your tokens_per_second",
                      "intro": "Your `tokens_per_second` around the course's bigram writer, on your computer: one text of {n} characters, starting with \"T\", with seed 1.",
                      "chart": "Characters per second",
                      "alt": "Two bars: the characters per second your function measures on your computer, and the number measured on the course's computer.",
                      "mine": "your computer, by your function", "course": "the course's computer", "unit": "per second",
                      "wrote": "What the writer wrote:",
                      "none": "Your `tokens_per_second` stops with an error, takes too long, or does not give a number above 0, so nothing can be drawn yet."}},
    "nl": {"rates": {"title": "Je flop_per_second",
                     "intro": "Je `flop_per_second` op de net op je computer gemeten tijden van een n × n-product, en gestreept op de tijden van de computer van de course.",
                     "chart": "Je flop_per_second",
                     "alt": "FLOP/s voor n van 8 tot en met 1.024 op logaritmische assen, uitgerekend door je functie: op je computer, en gestreept op de computer van de course.",
                     "say": "Op je computer: {lo} FLOP/s bij 8 × 8, en {hi} bij 1.024 × 1.024.",
                     "none": "Je `flop_per_second` stopt met een fout, duurt te lang, of geeft niet één getal per grootte, dus is er nog geen afbeelding."},
           "tokens": {"title": "Je tokens_per_second",
                      "intro": "Je `tokens_per_second` om de bigram-schrijver van de course, op je computer: één tekst van {n} tekens, beginnend met \"T\", met seed 1.",
                      "chart": "Tekens per seconde",
                      "alt": "Twee balken: de tekens per seconde die je functie op je computer meet, en het aantal dat op de computer van de course gemeten is.",
                      "mine": "jouw computer, met je functie", "course": "de computer van de course", "unit": "per seconde",
                      "wrote": "Wat de schrijver schreef:",
                      "none": "Je `tokens_per_second` stopt met een fout, duurt te lang, of geeft geen getal boven 0, dus is er nog geen afbeelding."}},
}
PICTURE_CHARS = 400   # the characters the writer writes for the picture of tokens_per_second


def rates_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w, r = _YOURS_TEXT[lang]["rates"], _RATE_TEXT[lang]
    fn = _fn(seen)
    micros = my_micros()   # measured outside the time the learner's function gets, with all of PyTorch's threads
    mine = _call(lambda: (fn(list(SIZES), list(micros)), fn(list(SIZES), list(COURSE_MICROS)))) if fn is not None else None
    got = (_numbers(mine[0], len(SIZES)), _numbers(mine[1], len(SIZES))) if isinstance(mine, tuple) else (None, None)
    if got[0] is None or got[1] is None:
        return yours.figure(w["title"], _note(w["intro"]) + _note(w["none"], _LOSS_UI), alt=w["title"])
    svg = rates_svg([(got[1], "ink-3", True, r["course"]), (got[0], "data", False, r["mine"])], w["chart"], lang, w["alt"])
    say = w["say"].format(lo=big(got[0][0], lang), hi=big(got[0][-1], lang))
    return yours.figure(w["title"], _note(w["intro"]) + svg + _note(say), alt=w["title"])


def tokens_picture(seen) -> str:
    lang = seen.lang if seen.lang in LANGS else "en"
    w = _YOURS_TEXT[lang]["tokens"]
    intro = w["intro"].format(n=number(PICTURE_CHARS, 0, lang))
    fn = _fn(seen)
    written = []

    def writer(n):
        written[:] = write(n)

    table()
    write(20)   # PyTorch prepares each operation the first time: outside the time the learner's function gets
    got = _call(lambda: fn(writer, PICTURE_CHARS)) if fn is not None else None
    ok = isinstance(got, (int, float)) and not isinstance(got, bool) and math.isfinite(got) and got > 0 and written
    if not ok:
        return yours.figure(w["title"], _note(intro) + _note(w["none"], _LOSS_UI), alt=w["title"])
    course = 10 ** 6 / COURSE_STEP
    svg = bars_svg([(w["mine"], float(got), "ink"), (w["course"], course, "ink-3")], w["chart"], w["unit"], lang, w["alt"])
    return yours.figure(w["title"], _note(intro) + _figure([svg], w["alt"], 330) + _note(w["wrote"]) + _text_block(written[0]),
                        alt=w["title"])


# --- the demo: where the time of a character goes, and many texts at once -----------------------------------------------------

DEMO_TEXT = {
    "en": {"parts_title": "One character: where the time goes",
           "parts_alt": "Four bars, the microseconds of each part of one character on your computer: the one-hot row, the product with the table, softmax and the draw.",
           "onehot": "the one-hot row", "product": "the product with the table", "softmax": "softmax", "draw": "the draw",
           "parts_say": "One character takes {step} µs on your computer: {rate} characters per second. The product, all {flops} FLOPs of it, takes {share}% of that time.",
           "texts_title": "Many texts at once", "x": "texts at once", "y": "characters per second",
           "texts_alt": "Characters per second of all texts together, for 1, 10, 100 and 1,000 texts at once on your computer, on log scales.",
           "texts_say": "With {k} at once, a step takes {step} µs and gives every text one character: {rate} characters per second together, {times} times as many as with one text.",
           "texts_say_one": "With one text, a step takes {step} µs: {rate} characters per second.",
           "one": "1 text", "many": "{k} texts"},
    "nl": {"parts_title": "Eén teken: waar de tijd heen gaat",
           "parts_alt": "Vier balken, de microseconden van elk deel van één teken op je computer: de one-hot-rij, het product met de tabel, softmax en het trekken.",
           "onehot": "de one-hot-rij", "product": "het product met de tabel", "softmax": "softmax", "draw": "het trekken",
           "parts_say": "Eén teken kost {step} µs op je computer: {rate} tekens per seconde. Het product, met alle {flops} FLOPs, neemt {share}% van deze tijd.",
           "texts_title": "Veel teksten tegelijk", "x": "teksten tegelijk", "y": "tekens per seconde",
           "texts_alt": "Tekens per seconde van alle teksten samen, voor 1, 10, 100 en 1.000 teksten tegelijk op je computer, op logaritmische assen.",
           "texts_say": "Met {k} tegelijk kost een stap {step} µs en krijgt elke tekst één teken: samen {rate} tekens per seconde, {times} keer zoveel als met één tekst.",
           "texts_say_one": "Met één tekst kost een stap {step} µs: {rate} tekens per seconde.",
           "one": "1 tekst", "many": "{k} teksten"},
}


def parts_picture(parts: dict, lang: str) -> str:
    w = DEMO_TEXT[lang]
    rows = [(w[k], parts[k], "ink" if k == "product" else "ink-3") for k in PARTS]   # ink stands out in light and dark
    return bars_svg(rows, w["parts_title"], "µs", lang, w["parts_alt"])


def parts_say(parts: dict, lang: str) -> str:
    w = DEMO_TEXT[lang]
    return w["parts_say"].format(step=number(parts["step"], 1, lang), rate=big(10 ** 6 / parts["step"], lang),
                                 flops=number(PRODUCT_FLOPS, 0, lang), share=round(100 * parts["product"] / parts["step"]))


def texts_frame(steps: dict, k: int, lang: str) -> tuple[str, str]:
    """The chart of characters per second against the number of texts, with k marked, and the sentence under it."""
    w = DEMO_TEXT[lang]
    counts = sorted(steps)
    rates = [c * 10 ** 6 / steps[c] for c in counts]
    lo, hi = _bounds(rates)
    p = chart.Panel(x=(0.7, 1500), y=(lo, hi), title=w["texts_title"], xlabel=w["x"], ylabel=w["y"], xlog=True, ylog=True)
    p.line(counts, rates, role="data")
    p.points(counts, rates, role="ink-3", r=3.5)
    p.points([k], [k * 10 ** 6 / steps[k]], role="data", r=6)
    svg = chart.figure([p], alt=w["texts_alt"], width=330, height=250)
    rate = k * 10 ** 6 / steps[k]
    if k == 1:
        return svg, w["texts_say_one"].format(step=number(steps[1], 1, lang), rate=big(rate, lang))
    many = w["many"].format(k=number(k, 0, lang))
    return svg, w["texts_say"].format(k=many, step=number(steps[k], 1, lang), rate=big(rate, lang),
                                      times=number(rate / (10 ** 6 / steps[1]), 1, lang))


# --- the exercises --------------------------------------------------------------------------------------------------------

def _dots_left(got, text_) -> bool:
    """A `...` of the start code left in the box: an Ellipsis in the code, not three dots inside a string."""
    import ast
    try:
        tree = ast.parse(text_)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.Constant) and node.value is Ellipsis for node in ast.walk(tree))


_DOTS = Mistake("dots-left", matches=_dots_left,
                en="There is still a `...` in your code. Can you replace it with your own code?",
                nl="Er staat nog een `...` in je code. Kun je de `...` vervangen door je eigen code?")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "All the code in the box is gone. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then begins with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def show_rates(v) -> dict:
    """What a flop_per_second gave back, with its numbers rounded to whole numbers, never in e-notation."""
    try:
        shown = "[" + ", ".join(f"{float(x):.0f}" if abs(float(x)) >= 1 else f"{float(x):.2g}" for x in v) + "]"
    except (TypeError, ValueError):
        shown = repr(v)
    if len(shown) > 80:
        shown = shown[:77] + "..."
    return {"en": f"`{shown}`", "nl": f"`{shown}`"}


def show_rate(v) -> dict:
    """What a tokens_per_second gave back: a whole number, with its decimals only when it is below 10."""
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
        shown = f"{v:.0f}" if abs(v) >= 10 else f"{v:.3g}"
    else:
        shown = repr(v)
    return {"en": f"`{shown}`", "nl": f"`{shown}`"}


def same_rate(a, b) -> bool:
    """Two measured rates within 30% of each other: the times of the stand-in writer vary a little."""
    if isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a):
        return False
    return abs(a - b) <= 0.3 * abs(b)


# exercise 1: flop_per_second
def _f(line: str = "        rates.append(2 * n ** 3 / seconds)\n") -> str:
    return ("def flop_per_second(sizes, micros):\n"
            "    rates = []\n"
            "    for i in range(len(sizes)):\n"
            "        n = sizes[i]\n"
            "        seconds = micros[i] / 10 ** 6\n"
            f"{line}"
            "    return rates\n")


_F_START = _f("        ...\n")
_F = _f()
_F_NO_TWO = _f("        rates.append(n ** 3 / seconds)\n")
_F_SQUARE = _f("        rates.append(2 * n ** 2 / seconds)\n")
_F_UPSIDE = _f("        rates.append(seconds / (2 * n ** 3))\n")
_F_MICROS = _f("        rates.append(2 * n ** 3 / micros[i])\n")

# exercise 2: tokens_per_second
_T_HEAD = "import time\n\n\ndef tokens_per_second(write, n):\n"
_T_BODY = "    start = time.perf_counter()\n    write(n)\n    seconds = time.perf_counter() - start\n"
_T_END = "    return n / seconds\n"
_T_START = _T_HEAD + "    ...\n" + _T_END
_T = _T_HEAD + _T_BODY + _T_END

EXERCISES = {
    "flop-rate": FunctionBox(
        name="flop-rate",
        function="flop_per_second",
        start=_F_START,
        solution=_F,
        show=show_rates,
        visible=[([16, 512], [1.3, 650.0])],
        hidden=[(SIZES, COURSE_MICROS), ([2], [1.0]), ([1000], [2000.0]), ([], [])],
        first=[_DOTS],
        picture=rates_picture,
        mistakes=[
            Mistake("no-two", wrong=_F_NO_TWO,
                    en="`{call}` gives {got}, and it should give {want}: half of each. Each number of the result takes n multiplications and n additions: how many FLOPs is that for each of the n · n numbers?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: de helft van elk. Elk getal van het resultaat kost n vermenigvuldigingen en n optellingen: hoeveel FLOPs zijn dat voor elk van de n · n getallen?"),
            Mistake("square", wrong=_F_SQUARE,
                    en="`{call}` gives {got}, and it should give {want}: n times too little. The result has n · n numbers, and each of them takes 2n FLOPs. How many FLOPs is that for the whole product?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: n keer te weinig. Het resultaat heeft n · n getallen, en elk daarvan kost 2n FLOPs. Hoeveel FLOPs zijn dat voor het hele product?"),
            Mistake("upside-down", wrong=_F_UPSIDE,
                    en="`{call}` gives tiny numbers: your function divides the seconds by the FLOPs, which gives the seconds per FLOP. A rate is work per second: which of the two goes on top?",
                    nl="`{call}` geeft piepkleine getallen: je functie deelt de seconden door de FLOPs, en dat geeft de seconden per FLOP. Een snelheid is werk per seconde: welke van de twee staat boven de deelstreep?"),
            Mistake("micros", wrong=_F_MICROS,
                    en="`{call}` gives {got}, and it should give {want}: a million times too little. Your function divides by `micros[i]`, the time in microseconds. Which name holds that time in seconds?",
                    nl="`{call}` geeft {got}, en het moet {want} zijn: een miljoen keer te weinig. Je functie deelt door `micros[i]`, de tijd in microseconden. Welke naam bevat deze tijd in seconden?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with a line that adds the FLOP/s of size `n` to `rates`: the FLOPs of an n × n product divided by `seconds`.",
                      "nl": "Vervang de `...` door een regel die de FLOP/s van grootte `n` aan `rates` toevoegt: de FLOPs van een n × n-product gedeeld door `seconds`."},
            "arguments": {"en": "The check calls `{call}`, with a list of sizes and a list of times, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(sizes, micros)`?",
                          "nl": "De check roept `{call}` aan, met een lijst met groottes en een lijst met tijden, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(sizes, micros)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Does every round of the loop add one rate to `rates`: the FLOPs of an n × n product divided by `seconds`?",
                      "nl": "`{call}` geeft {got}, en het moet {want} zijn. Voegt elke ronde van de loop één snelheid aan `rates` toe: de FLOPs van een n × n-product gedeeld door `seconds`?"},
            "almost": {"en": "Right for the sizes 16 and 512. But `{call}` gives {got}, and it should give {want}. Does your function work out the rate of every size from its own `n` and its own time?",
                       "nl": "Goed voor de groottes 16 en 512. Maar `{call}` geeft {got}, en het moet {want} zijn. Rekent je functie de snelheid van elke grootte uit met de eigen `n` en de eigen tijd?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return rates`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return rates`?"},
            "correct": {"en": "Right in all {n} cases: 6.3 billion FLOP/s for 16 × 16 and 413 billion for 512 × 512, 66 times as fast. In the picture, how many times as fast is 1,024 × 1,024 as 8 × 8 on your computer?",
                        "nl": "Goed in alle {n} gevallen: 6,3 miljard FLOP/s voor 16 × 16 en 413 miljard voor 512 × 512, 66 keer zo snel. Hoeveel keer zo snel is 1.024 × 1.024 in de afbeelding als 8 × 8 op je computer?"},
        },
    ),
    "tokens-rate": FunctionBox(
        name="tokens-rate",
        function="tokens_per_second",
        start=_T_START,
        solution=_T,
        compare=same_rate,
        show=show_rate,
        visible=[(slow_writer(0.05), 1000)],
        hidden=[(slow_writer(0.1), 500), (slow_writer(0.05), 30)],
        first=[_DOTS],
        picture=tokens_picture,
        mistakes=[
            Mistake("negative", matches=lambda got, text_: isinstance(got, (int, float)) and not isinstance(got, bool) and got < 0,
                    en="`{call}` gives a number below 0: your function subtracts the later reading of the clock from the earlier one. Which reading is the larger of the two?",
                    nl="`{call}` geeft een getal onder 0: je functie trekt de latere stand van de klok af van de eerdere. Welke stand is de grootste van de twee?"),
            Mistake("nothing-timed", matches=lambda got, text_: isinstance(got, (int, float)) and not isinstance(got, bool) and got > 10 ** 6,
                    en="`{call}` gives {got}: far more tokens per second than the stand-in can write, as if no time passed between the two readings of the clock. Does `write(n)` come between them?",
                    nl="`{call}` geeft {got}: veel meer tokens per seconde dan de stand-in kan schrijven, alsof er geen tijd verstreek tussen de twee standen van de klok. Staat `write(n)` ertussen?"),
            # on a clock that ticks coarsely, as on Windows, two readings right after each other can be the same
            Mistake("zero-time", error=lambda e: isinstance(e, ZeroDivisionError),
                    en="With `{call}`, your function divides by a time of 0 seconds, as if no time passed between the two readings of the clock. Does `write(n)` come between them?",
                    nl="Met `{call}` deelt je functie door een tijd van 0 seconden, alsof er geen tijd verstreek tussen de twee standen van de klok. Staat `write(n)` ertussen?"),
            Mistake("upside-down", matches=lambda got, text_: isinstance(got, (int, float)) and not isinstance(got, bool) and 0 <= got < 1,
                    en="`{call}` gives less than 1 token per second: your function gives seconds, or seconds per token. Tokens per second is the tokens divided by the seconds: which goes on top?",
                    nl="`{call}` geeft minder dan 1 token per seconde: je functie geeft seconden, of seconden per token. Tokens per seconde is het aantal tokens gedeeld door de seconden: wat staat er boven de deelstreep?"),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the clock around `write(n)`: read `time.perf_counter()` once before it and once after, and make `seconds` the difference.",
                      "nl": "Vervang de `...` door de klok om `write(n)`: lees `time.perf_counter()` één keer voor `write(n)` en één keer erna, en maak `seconds` het verschil."},
            "arguments": {"en": "The check calls `{call}`, with a writer and a number of tokens, and the `def` line of your function has a different number of names between its parentheses. Can you keep `(write, n)`?",
                          "nl": "De check roept `{call}` aan, met een schrijver en een aantal tokens, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Kun je `(write, n)` houden?"},
            "wrong": {"en": "`{call}` gives {got}, and it should give about {want}: the stand-in takes as many seconds as it says, for `n` tokens. Does the clock run around exactly one `write(n)`?",
                      "nl": "`{call}` geeft {got}, en het moet ongeveer {want} zijn: de stand-in doet er zoveel seconden over als er staat, voor `n` tokens. Loopt de klok om precies één `write(n)`?"},
            "almost": {"en": "Right for 1,000 tokens. But `{call}` gives {got}, and it should give about {want}. Does your function use `n`, the number of tokens it gets, and not a fixed number?",
                       "nl": "Goed voor 1.000 tokens. Maar `{call}` geeft {got}, en het moet ongeveer {want} zijn. Gebruikt je functie `n`, het aantal tokens dat de functie krijgt, en geen vast getal?"},
            "no_return": {"en": "With `{call}`, your function gives nothing back. Does it still end with `return n / seconds`?",
                          "nl": "Met `{call}` geeft je functie niets terug. Eindigt je functie nog met `return n / seconds`?"},
            "correct": {"en": "Right in all {n} cases: 1,000 tokens in 0.05 seconds is 20,000 tokens per second. In the picture, how many characters per second does the bigram table write on your computer?",
                        "nl": "Goed in alle {n} gevallen: 1.000 tokens in 0,05 seconde is 20.000 tokens per seconde. Hoeveel tekens per seconde schrijft de bigram-tabel in de afbeelding op je computer?"},
        },
    ),
}

# The lesson is done when these are correct. "part-faster" and "many-texts" are the choice questions after the demo.
REQUIRED = ["flop-rate", "tokens-rate", "part-faster", "many-texts"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("flop-rate", _F_START, "empty", None),
    ("flop-rate", _F, "correct", None),
    ("flop-rate", _f("        rates.append(... / seconds)\n"), "wrong", "dots-left"),
    ("flop-rate", _F_NO_TWO, "wrong", "no-two"),
    ("flop-rate", _F_SQUARE, "wrong", "square"),
    ("flop-rate", _F_UPSIDE, "wrong", "upside-down"),
    ("flop-rate", _F_MICROS, "wrong", "micros"),
    ("flop-rate", _F.replace("    return rates\n", ""), "wrong", "no-return"),
    # other right answers: the same rate written another way
    ("flop-rate", _f("        flops = 2 * n * n * n\n        rates.append(flops / seconds)\n"), "correct", None),
    ("flop-rate", _f("        rates.append(2 * n ** 3 * 10 ** 6 / micros[i])\n"), "correct", None),
    # the two answers of the visible case written out: right there, wrong for every other size
    ("flop-rate", "def flop_per_second(sizes, micros):\n    return [6301538461.5, 412977624615.4]\n", "almost", None),
    ("tokens-rate", _T_START, "empty", None),
    ("tokens-rate", _T, "correct", None),
    ("tokens-rate", _T_HEAD + "    start = time.perf_counter()\n    write(n)\n    seconds = ...\n" + _T_END, "wrong", "dots-left"),
    ("tokens-rate", _T_HEAD + "    start = time.perf_counter()\n    write(n)\n    seconds = start - time.perf_counter()\n" + _T_END, "wrong", "negative"),
    ("tokens-rate", _T_HEAD + "    write(n)\n    start = time.perf_counter()\n    seconds = time.perf_counter() - start\n" + _T_END, "wrong", "nothing-timed"),
    ("tokens-rate", _T_HEAD + _T_BODY + "    return seconds / n\n", "wrong", "upside-down"),
    ("tokens-rate", _T_HEAD + "    write(n)\n    seconds = 0\n" + _T_END, "error", "zero-time"),
    ("tokens-rate", _T_HEAD + _T_BODY + "    return seconds\n", "wrong", "upside-down"),
    ("tokens-rate", _T_HEAD + "    start = time.perf_counter()\n    write(n)\n    write(n)\n    seconds = time.perf_counter() - start\n" + _T_END, "wrong", None),
    ("tokens-rate", _T_HEAD + _T_BODY, "wrong", "no-return"),
    # other right answers: the clock written another way
    ("tokens-rate", _T_HEAD + "    start = time.perf_counter()\n    write(n)\n    end = time.perf_counter()\n    seconds = end - start\n" + _T_END, "correct", None),
    ("tokens-rate", "import time\n\n\ndef tokens_per_second(write, n):\n    start = time.perf_counter()\n    write(n)\n    return n / (time.perf_counter() - start)\n", "correct", None),
    # a fixed 1,000 tokens: right for the visible case, wrong for the others
    ("tokens-rate", _T_HEAD + _T_BODY + "    return 1000 / seconds\n", "almost", None),
]
