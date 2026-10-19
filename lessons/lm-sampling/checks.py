# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson lm-sampling: greedy, which writes by always taking the most likely next character, and
generate, which draws every next character from the row of the last one; each saved in the learner's project
(work/projects/llm/) when right. And a picture of five steps of writing, and a demo that lets the learner's model write
with the table of the plays."""
import html
import re
from functools import cache

import numpy as np

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit, Timeout

# The table comes from the learner's count_bigrams, bigram_matrix and to_probs, or the course's versions.
projects.builds_on(("llm", "bigrams", "lm-bigram-counts", "count-bigrams"),
                   ("llm", "bigram_matrix", "lm-bigram-matrix", "bigram-matrix"),
                   ("llm", "bigram_rows", "lm-bigram-matrix", "to-probs"))

DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md


@cache
def text() -> str:
    return (paths.ROOT / DATA).read_text(encoding="utf-8")


@cache
def chars() -> tuple[str, ...]:
    """The 65 characters of the plays, in sorted order: the order of the rows and the columns."""
    return tuple(sorted(set(text())))


@cache
def course_probs() -> np.ndarray:
    """The course's own matrix of probabilities: row i is the distribution of the next character after chars()[i]."""
    t, cs = text(), chars()
    counts = {}
    for i in range(len(t) - 1):
        counts[t[i:i + 2]] = counts.get(t[i:i + 2], 0) + 1
    n = np.array([[counts.get(a + b, 0) for b in cs] for a in cs], dtype=float)
    return n / n.sum(axis=1, keepdims=True)


def _write(P, cs, start: str, n: int, seed: int | None) -> str:
    """The course's way of writing: greedy without a seed, sampling with one."""
    rng = np.random.default_rng(seed) if seed is not None else None
    row_of = {c: k for k, c in enumerate(cs)}
    out = start
    while len(out) < n:
        i = row_of[out[-1]]
        j = int(np.argmax(P[i])) if rng is None else int(rng.choice(len(cs), p=P[i]))
        out = out + cs[j]
    return out


def greedy_text(start: str, n: int) -> str:
    """What greedy decoding writes with the course's table, for the reveal."""
    return _write(course_probs(), chars(), start, n, None)


def text_block(s: str) -> str:
    """A text the model wrote, in a box that wraps its lines, so that a long line stays readable on a phone."""
    return (f'<pre style="white-space:pre-wrap;word-break:break-word;margin:0.25rem 0 0.75rem;padding:0.6rem 0.8rem;'
            f'border-radius:6px;background:var(--ea-inset);color:var(--ea-ink);font-family:var(--ea-font-code);'
            f'font-size:0.875rem;line-height:1.45">{html.escape(s)}</pre>')


def shown(ch: str) -> str:
    return {" ": "␣", "\n": "↵"}.get(ch, ch)


# --- the picture: five steps of writing, each drawn from the row of the last character ---------------------------------

def steps_picture(start: str, seed: int, k: int, labels: dict[str, str]) -> str:
    """k steps of sampling with the course's table: per step the text so far, the four most likely characters in the
    row of its last character, and the one drawn, which is outlined."""
    P, cs = course_probs(), chars()
    row_of = {c: i for i, c in enumerate(cs)}
    rng = np.random.default_rng(seed)
    rows, t = [], start
    for _ in range(k):
        r = P[row_of[t[-1]]]
        j = int(rng.choice(len(cs), p=r))
        rows.append((t, [(int(b), float(r[b])) for b in np.argsort(-r)[:4]], j))
        t = t + cs[j]
    W, top, ROW = 520, 56, 36
    x_cells, cw = 132, 70
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>',
         f'<text x="104" y="44" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:end">{html.escape(labels["text"])}</text>',
         f'<text x="{x_cells}" y="44" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(labels["row"])}</text>',
         f'<text x="{W - 4}" y="44" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:end">{html.escape(labels["drawn"])}</text>']
    for s, (so_far, best, j) in enumerate(rows):
        y = top + s * ROW
        head, last = so_far[:-1], so_far[-1]
        p.append(f'<text x="104" y="{y + 18}" style="font-size:15px;font-family:var(--ea-font-code);fill:var(--ea-ink);text-anchor:end">'
                 f'<tspan style="fill:var(--ea-ink-2)">{html.escape(head.replace(" ", "␣"))}</tspan>'
                 f'<tspan style="font-weight:700">{html.escape(shown(last))}</tspan></text>')
        for c, (b, pr) in enumerate(best):
            x = x_cells + c * (cw + 4)
            mark = ';stroke:var(--ea-pred-text);stroke-width:2.5' if b == j else ''
            p.append(f'<rect x="{x}" y="{y}" width="{cw}" height="26" rx="4" style="fill:var(--ea-pred);fill-opacity:{0.12 + 0.6 * pr:.2f}{mark}"/>'
                     f'<text x="{x + cw / 2}" y="{y + 18}" style="font-size:13px;fill:var(--ea-ink);text-anchor:middle">'
                     f'<tspan style="font-weight:700">{html.escape(shown(cs[b]))}</tspan> {round(pr * 100)}%</text>')
        p.append(f'<text x="{W - 4}" y="{y + 18}" style="font-size:15px;font-weight:700;font-family:var(--ea-font-code);'
                 f'fill:var(--ea-pred-text);text-anchor:end">→ {html.escape(shown(cs[j]))}</text>')
    H = top + k * ROW + 4
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


# The lesson is done when these are correct. "draw-line" is the choice question in the lesson.
REQUIRED = ["draw-line", "greedy", "generate"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


class _Matrix(np.ndarray):
    """A matrix of probabilities that a check message writes the way the lesson's code does: np.array([[...]]).
    The learner's function gets a plain copy (the check deep-copies every argument), so `P[i].max()` there gives a
    plain number, as it does in the lesson."""

    def __deepcopy__(self, memo):
        return np.array(self)

    def __repr__(self):
        a = np.asarray(self)
        if a.ndim != 2:
            return repr(a)
        rows = [[int(x) if x == int(x) else float(x) for x in row] for row in a.tolist()]
        return f"np.array({rows!r})"


def _matrix(rows) -> np.ndarray:
    return np.array(rows, dtype=float).view(_Matrix)


_SAVED = {
    "correct": {"en": "Your function is right in all {n} cases. The check saved it in your project, as `{saved}`: the demo and later lessons import it from there.",
                "nl": "Je functie klopt in alle {n} gevallen. De check heeft hem in je project bewaard, als `{saved}`: de demo en latere lessen halen hem daar binnen."},
    "kept": {"en": "Your function is right in all {n} cases. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
             "nl": "Je functie klopt in alle {n} gevallen. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren? Verwijder het bestand en laat opnieuw nakijken."},
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

_ROWS_OF = "    row_of = {}\n    for k in range(len(chars)):\n        row_of[chars[k]] = k\n    text = start\n"

# The small tables the checks try. No column adds up to 1, so a column given to rng.choice stops with an error.
_P3 = _matrix([[0.1, 0.6, 0.3], [0.4, 0, 0.6], [0.8, 0.2, 0]])
_C3 = ["a", "b", "c"]
_P4 = _matrix([[0, 0.5, 0.3, 0.2], [0.6, 0, 0.1, 0.3], [0.2, 0.7, 0, 0.1], [0.5, 0.1, 0.4, 0]])
_C4 = [" ", "e", "h", "t"]

# --- exercise 2: greedy -------------------------------------------------------------------------------------------------

_G_HEAD = "import numpy as np\n\n\ndef greedy(P, chars, start, n):\n" + _ROWS_OF + "    while len(text) < n:\n        i = row_of[text[-1]]\n"
_G_END = "        text = text + chars[j]\n    return text\n"
_G_START = _G_HEAD + "        j = ...\n" + _G_END


def _g(pick: str = "np.argmax(P[i])") -> str:
    return _G_HEAD + f"        j = {pick}\n" + _G_END


_G = _g()

# --- exercise 3: generate -----------------------------------------------------------------------------------------------

_S_HEAD = "import numpy as np\n\n\ndef generate(P, chars, start, n, seed):\n    rng = np.random.default_rng(seed)\n" + _ROWS_OF
_S_END = "    return text\n"
_S_START = _S_HEAD + "    ...\n" + _S_END


def _s(loop: str = "    while len(text) < n:\n", row: str = "text[-1]", pick: str = "rng.choice(len(chars), p=P[i])",
       add: str = "chars[j]") -> str:
    return _S_HEAD + loop + f"        i = row_of[{row}]\n        j = {pick}\n        text = text + {add}\n" + _S_END


_S = _s()


def _text(v) -> str:
    """A text a function gave back, the way code writes it, so that a space and a newline can be seen."""
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'
    return repr(v)


EXERCISES = {
    "greedy": FunctionBox(
        name="greedy",
        function="greedy",
        start=_G_START,
        solution=_G,
        visible=[(_P3, _C3, "a", 6)],
        hidden=[(_P3, _C3, "c", 4), (_P3, _C3, "b", 1), (_P4, _C4, "t", 9), (_P4, _C4, "e", 6)],
        show=_text,
        save_as=("llm", "greedy"),
        first=[_DOTS],
        mistakes=[
            Mistake("column", wrong=_g("np.argmax(P[:, i])"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you look in column `i`, not in row `i`. What comes after a character is in its row: `P[i]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je kijkt in kolom `i`, niet in rij `i`. Wat er na een teken komt, staat in zijn rij: `P[i]`."),
            Mistake("value", error=lambda e: isinstance(e, TypeError) and "indices" in str(e),
                    en="With `{call}`, Python stops at `chars[j]`: your `j` is not a whole number that says a place. `max` gives the largest probability itself; `np.argmax` gives its place, the number of its column.",
                    nl="Met `{call}` stopt Python bij `chars[j]`: je `j` is geen heel getal dat een plaats aangeeft. `max` geeft de grootste kans zelf; `np.argmax` geeft haar plaats, het nummer van haar kolom."),
        ],
        texts={
            **_SAVED,
            "empty": {"en": "Replace the `...` with the number of the most likely next character in row `i`.",
                      "nl": "Vervang de `...` door het nummer van het waarschijnlijkste volgende teken in rij `i`."},
            "arguments": {"en": "The check calls `{call}`, with a table, the characters, a start and a length, and the `def` line of your function has another number of names between its parentheses. Keep `(P, chars, start, n)`.",
                          "nl": "De check roept `{call}` aan, met een tabel, de tekens, een begin en een lengte, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(P, chars, start, n)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Is `j` the place of the largest number in row `i`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Is `j` de plaats van het grootste getal in rij `i`?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every table and every start?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke tabel en elk begin?"},
        },
    ),
    "generate": FunctionBox(
        name="generate",
        function="generate",
        start=_S_START,
        solution=_S,
        visible=[(_P3, _C3, "a", 8, 1)],
        hidden=[(_P3, _C3, "c", 10, 7), (_P3, _C3, "b", 1, 3), (_P4, _C4, "t", 12, 2), (_P4, _C4, " ", 12, 5), (_P4, _C4, "the", 9, 3)],
        show=_text,
        save_as=("llm", "sampling"),
        first=[_DOTS],
        mistakes=[
            Mistake("greedy", wrong=_s(pick="np.argmax(P[i])"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function always takes the most likely character, as `greedy` does. Draw it instead, with `rng.choice(len(chars), p=P[i])`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie neemt altijd het waarschijnlijkste teken, zoals `greedy`. Trek het in plaats daarvan, met `rng.choice(len(chars), p=P[i])`."),
            Mistake("even", wrong=_s(pick="rng.choice(len(chars))"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your function gives every character the same chance, whatever the row of the last character says. Draw with that row: `rng.choice(len(chars), p=P[i])`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je functie geeft elk teken dezelfde kans, wat de rij van het laatste teken ook zegt. Trek met die rij: `rng.choice(len(chars), p=P[i])`."),
            Mistake("first-row", wrong=_s(row="text[0]"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: every character is drawn from the row of the first one. The next character depends on the last one: `text[-1]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: elk teken wordt getrokken uit de rij van het eerste. Het volgende teken hangt af van het laatste: `text[-1]`."),
            Mistake("one-more", wrong=_s(loop="    while len(text) <= n:\n"),
                    en="`{call}` gives `{got}`, one character more than it should: `{want}`. The whole text, the start included, is `n` characters long: repeat while `len(text) < n`.",
                    nl="`{call}` geeft `{got}`, één teken meer dan de bedoeling is: `{want}`. De hele tekst, met het begin erbij, is `n` tekens lang: herhaal zolang `len(text) < n`."),
            Mistake("n-more", wrong=_s(loop="    for k in range(n):\n"),
                    en="`{call}` gives `{got}`, longer than `{want}`: your loop adds `n` characters to the start. The whole text, the start included, is `n` characters long: repeat while `len(text) < n`.",
                    nl="`{call}` geeft `{got}`, langer dan `{want}`: je loop zet `n` tekens achter het begin. De hele tekst, met het begin erbij, is `n` tekens lang: herhaal zolang `len(text) < n`."),
            Mistake("column", error=lambda e: isinstance(e, ValueError) and "sum to 1" in str(e),
                    en="With `{call}`, `rng.choice` stops: the probabilities you give it do not add up to 1. A row of `P` does: `p=P[i]`.",
                    nl="Met `{call}` stopt `rng.choice`: de kansen die je meegeeft, tellen niet op tot 1. Een rij van `P` doet dat wel: `p=P[i]`."),
            Mistake("number", error=lambda e: isinstance(e, TypeError) and "concatenate" in str(e),
                    en="With `{call}`, Python stops when you add to the text: you add something that is not text. `rng.choice(len(chars), ...)` gives the number of a column; the character itself is `chars[j]`.",
                    nl="Met `{call}` stopt Python als je iets aan de tekst toevoegt: wat je toevoegt, is geen tekst. `rng.choice(len(chars), ...)` geeft het nummer van een kolom; het teken zelf is `chars[j]`."),
        ],
        texts={
            **_SAVED,
            "empty": {"en": "Replace the `...` with step 4 of the recipe: a `while` loop that adds a drawn character until the text is `n` characters long.",
                      "nl": "Vervang de `...` door stap 4 van het recept: een `while`-loop die een getrokken teken toevoegt tot de tekst `n` tekens lang is."},
            "arguments": {"en": "The check calls `{call}`, with a table, the characters, a start, a length and a seed, and the `def` line of your function has another number of names between its parentheses. Keep `(P, chars, start, n, seed)`.",
                          "nl": "De check roept `{call}` aan, met een tabel, de tekens, een begin, een lengte en een seed, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(P, chars, start, n, seed)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your loop with step 4 of the recipe: do you draw with the `rng` made from the seed, once for each new character?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je loop met stap 4 van het recept: trek je met de `rng` die van de seed is gemaakt, één keer voor elk nieuw teken?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every table, every start and every length?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke tabel, elk begin en elke lengte?"},
        },
    ),
}


# --- the demo: the learner's model writes, with the table of the plays ---------------------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "llm" / f"{module}.py").exists()


def demo(seeds=(1, 2), n: int = 200):
    """For the demo: None when greedy.py or sampling.py is not in the project yet; "greedy" or "sampling" when that
    saved module no longer gives what the course's gives on a small case, or stops with an error; "table" when the
    table of the plays cannot be made; "slow" when it takes more than 30 seconds; else a dict with the greedy text
    and one sampled text per seed."""
    if not (_saved("greedy") and _saved("sampling")):
        return None
    small = (_P3, _C3, "a", 8)
    run = run_code("from llm.greedy import greedy")
    if run.error is not None:
        return "greedy"
    greedy = run.names["greedy"]
    run = run_code("from llm.sampling import generate")
    if run.error is not None:
        return "sampling"
    generate = run.names["generate"]
    try:
        if with_time_limit(lambda: greedy(*small), 10) != _write(*small, None):
            return "greedy"
    except Exception:
        return "greedy"
    try:
        if with_time_limit(lambda: generate(*small, 1), 10) != _write(*small, 1):
            return "sampling"
    except Exception:
        return "sampling"
    # the table: the learner's, or the course's when theirs is missing or broken
    run = run_code("from llm.bigrams import count_bigrams\nfrom llm.bigram_matrix import bigram_matrix\nfrom llm.bigram_rows import to_probs")
    if run.error is not None:
        return "table"
    count, matrix, probs = run.names["count_bigrams"], run.names["bigram_matrix"], run.names["to_probs"]
    cs = list(chars())
    try:
        P = np.asarray(with_time_limit(lambda: probs(matrix(count(text()), cs)), 30), dtype=float)
        if P.shape != (len(cs), len(cs)) or not np.allclose(P.sum(axis=1), 1):
            return "table"
        out = {"greedy": with_time_limit(lambda: greedy(P, cs, "T", 60), 30)}
        for seed in seeds:
            out[seed] = with_time_limit(lambda: generate(P, cs, "T", n, seed), 30)
        return out
    except Timeout:
        return "slow"
    except Exception:
        return "table"


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("greedy", _G_START, "empty", None),
    ("greedy", _g("int(np.argmax(P[i]))"), "correct", None),
    ("greedy", _g("np.argmax(P[:, i])"), "wrong", "column"),
    ("greedy", _g("max(P[i])"), "error", "value"),
    ("generate", _S_START, "empty", None),
    ("generate", _s(pick="rng.choice(len(chars), p=P[i])"), "correct", None),
    ("generate", _s(loop="    while not len(text) == n:\n"), "correct", None),
    ("generate", _s(pick="np.argmax(P[i])"), "wrong", "greedy"),
    ("generate", _s(pick="rng.choice(len(chars))"), "wrong", "even"),
    ("generate", _s(row="text[0]"), "wrong", "first-row"),
    ("generate", _s(loop="    while len(text) <= n:\n"), "wrong", "one-more"),
    ("generate", _s(loop="    for k in range(n):\n"), "wrong", "n-more"),
    ("generate", _s(loop="    for k in range(n - len(start)):\n"), "correct", None),
    ("generate", _s(loop="    for k in range(n - 1):\n"), "almost", None),
    ("generate", _s(pick="rng.integers(0, len(chars))"), "wrong", "even"),
    ("greedy", _g("P[i].max()"), "error", "value"),
    ("generate", _s(pick="rng.choice(len(chars), p=P[:, i])"), "error", "column"),
    ("generate", _s(add="j"), "error", "number"),
]
