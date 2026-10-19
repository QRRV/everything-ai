# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the project lesson lm-bigram-matrix: bigram_matrix, the counts of every pair of characters as a matrix,
and to_probs, every row divided by its sum; each saved in the learner's project (work/projects/llm/) when right. And
the heatmap of the probabilities, and a demo that builds the whole table from the plays with the learner's code."""
import html
import math
import re
from functools import cache

import numpy as np

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, run_code, with_time_limit, Timeout

# The count of every pair comes from the learner's count_bigrams of lm-bigram-counts, or the course's version.
projects.builds_on(("llm", "bigrams", "lm-bigram-counts", "count-bigrams"))

DATA = "data/tiny-shakespeare.txt"   # about 1.1 MB of plays by Shakespeare: see data/README.md


@cache
def text() -> str:
    return (paths.ROOT / DATA).read_text(encoding="utf-8")


@cache
def chars() -> tuple[str, ...]:
    """The 65 characters of the plays, in sorted order: the order of the rows and the columns."""
    return tuple(sorted(set(text())))


@cache
def _course_counts() -> dict:
    t = text()
    counts = {}
    for i in range(len(t) - 1):
        counts[t[i:i + 2]] = counts.get(t[i:i + 2], 0) + 1
    return counts


@cache
def course_probs() -> np.ndarray:
    """The course's own matrix of probabilities: row i is the distribution of the next character after chars()[i]."""
    cs = chars()
    counts = _course_counts()
    n = np.array([[counts.get(a + b, 0) for b in cs] for a in cs], dtype=float)
    return n / n.sum(axis=1, keepdims=True)


def shown(ch: str) -> str:
    return {" ": "␣", "\n": "↵"}.get(ch, ch)


# --- the picture: a heatmap of the rows and columns of the space and the small letters ----------------------------------

_LEVELS = [(0.0, 0.08), (0.02, 0.18), (0.08, 0.38), (0.2, 0.6), (0.4, 0.8), (0.7, 1.0)]


def _opacity(p: float) -> float:
    out = 0.0
    for lo, op in _LEVELS:
        if p > lo:
            out = op
    return out


def heatmap(probs: np.ndarray, labels: dict[str, str]) -> str:
    """The probabilities after the space and the 26 small letters, for the space and the 26 small letters: a square
    per cell, darker the more likely. The rest of the 65 rows and columns is left out, so that the letters can be read."""
    cs = chars()
    keep = [" "] + [c for c in "abcdefghijklmnopqrstuvwxyz"]
    idx = [cs.index(c) for c in keep]
    cell, left, top = 14, 30, 58
    n = len(keep)
    W, H = left + n * cell + 4, top + n * cell + 34
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>',
         f'<text x="{left + n * cell / 2:.0f}" y="36" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(labels["next"])}</text>']
    for k, c in enumerate(keep):
        p.append(f'<text x="{left + k * cell + cell / 2:.1f}" y="{top - 6}" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(shown(c))}</text>')
        p.append(f'<text x="{left - 5}" y="{top + k * cell + cell * 0.75:.1f}" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:end">{html.escape(shown(c))}</text>')
    for r, i in enumerate(idx):
        for k, j in enumerate(idx):
            op = _opacity(float(probs[i, j]))
            if op:
                p.append(f'<rect x="{left + k * cell}" y="{top + r * cell}" width="{cell - 1}" height="{cell - 1}" '
                         f'style="fill:var(--ea-pred);fill-opacity:{op}"/>')
    p.append(f'<rect x="{left - 0.5}" y="{top - 0.5}" width="{n * cell}" height="{n * cell}" style="fill:none;stroke:var(--ea-line)"/>')
    p.append(f'<text x="{left}" y="{H - 10}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(labels["rows"])}</text>')
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


# The lesson is done when these are correct. "space-column" is the choice question in the lesson.
REQUIRED = ["space-column", "bigram-matrix", "to-probs"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _rows(v) -> str:
    """A small matrix for the messages, as a list of rows, with numbers rounded to 4 decimals."""
    try:
        raw = np.asarray(v)
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return repr(v)
    if raw.dtype == object:   # such as None in a row: shown as it is, not as nan
        return repr(raw.tolist())
    a = np.round(a, 4) + 0.0
    # nan and inf stay as they are: int() of them fails, and a function that divides by 0 gives them
    return repr([[int(x) if math.isfinite(x) and x == int(x) else float(x) for x in row] for row in a.tolist()]) if a.ndim == 2 else repr(a.tolist())


class _Matrix(np.ndarray):
    """A matrix that a check message writes the way the lesson's code does: np.array([[...]])."""

    def __repr__(self):
        a = np.asarray(self)
        if a.ndim != 2:
            return repr(a)
        return f"np.array({[[int(x) if x == int(x) else float(x) for x in row] for row in a.tolist()]!r})"


class _Counts(dict):
    """Counts that a check message writes the way the lesson's code does, with double quotes."""

    def __repr__(self):
        return "{" + ", ".join(f'"{k}": {v}' for k, v in self.items()) + "}"


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

# --- exercise 2: bigram_matrix ------------------------------------------------------------------------------------------

_M_HEAD = ("import numpy as np\n\n\ndef bigram_matrix(counts, chars):\n    rows = []\n    for a in chars:\n        row = []\n"
           "        for b in chars:\n")
_M_END = "        rows.append(row)\n    return np.array(rows)\n"
_M_START = _M_HEAD + "            ...\n" + _M_END


def _m(inner: str = "            if a + b in counts:\n                row.append(counts[a + b])\n            else:\n                row.append(0)\n") -> str:
    return _M_HEAD + inner + _M_END


_M = _m()

# --- exercise 3: to_probs -----------------------------------------------------------------------------------------------

_T_HEAD = "def to_probs(N):\n"
_T_START = _T_HEAD + "    sums = ...\n    return N / sums\n"


def _t(sums: str = "N.sum(axis=1, keepdims=True)", ret: str = "N / sums") -> str:
    return _T_HEAD + f"    sums = {sums}\n    return {ret}\n"


_T = _t()
_N1 = np.array([[3, 1], [0, 2]]).view(_Matrix)
_N2 = np.array([[1, 1, 2], [0, 5, 0], [2, 2, 0]]).view(_Matrix)
_N3 = np.array([[1, 3], [4, 4]]).view(_Matrix)

EXERCISES = {
    "bigram-matrix": FunctionBox(
        name="bigram-matrix",
        function="bigram_matrix",
        start=_M_START,
        solution=_M,
        visible=[(_Counts({"ab": 3, "ac": 1, "ba": 2}), ["a", "b", "c"])],
        hidden=[(_Counts({"ab": 3, "ac": 1, "ba": 2}), ["c", "b", "a"]), (_Counts({"aa": 2, "ab": 1}), ["a", "b"]), (_Counts(), ["x", "y"]),
                (_Counts({"qu": 5, " q": 2, "u ": 3}), [" ", "q", "u"])],
        show=_rows,
        save_as=("llm", "bigram_matrix"),
        first=[_DOTS],
        mistakes=[
            Mistake("swapped", wrong=_m("            if b + a in counts:\n                row.append(counts[b + a])\n            else:\n                row.append(0)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your rows and columns are swapped. Row `a` holds what comes after `a`, so the pair is `a + b`, with `a` first.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je rijen en kolommen zijn verwisseld. Rij `a` bevat wat er na `a` komt, dus het paar is `a + b`, met `a` voorop."),
            Mistake("ones", wrong=_m("            if a + b in counts:\n                row.append(1)\n            else:\n                row.append(0)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your table says only whether a pair occurs. Put in how often: `counts[a + b]`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je tabel zegt alleen of een paar voorkomt. Zet erin hoe vaak: `counts[a + b]`."),
            Mistake("missing-pairs", error=lambda e: isinstance(e, KeyError),
                    en="With `{call}`, Python stops at `counts[...]`: most pairs never occur, so they are not in `counts`. Look first with `if a + b in counts:`, and put a 0 in the row when it is not there.",
                    nl="Met `{call}` stopt Python bij `counts[...]`: de meeste paren komen nooit voor, dus ze staan niet in `counts`. Kijk eerst met `if a + b in counts:`, en zet een 0 in de rij als het er niet in staat."),
        ],
        texts={
            **_SAVED,
            "empty": {"en": "Replace the `...` with what goes into the row for the pair `a + b`: its count, or 0 when it never occurs.",
                      "nl": "Vervang de `...` door wat er voor het paar `a + b` in de rij komt: zijn telling, of 0 als het nooit voorkomt."},
            "arguments": {"en": "The check calls `{call}`, with the counts and the characters, and the `def` line of your function has another number of names between its parentheses. Keep `(counts, chars)`.",
                          "nl": "De check roept `{call}` aan, met de tellingen en de tekens, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(counts, chars)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare the inner loop with step 1 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk de binnenste loop met stap 1 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function follow the order of `chars`, and work when no pair occurs at all?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Volgt je functie de volgorde van `chars`, en werkt hij ook als er helemaal geen paar voorkomt?"},
        },
    ),
    "to-probs": FunctionBox(
        name="to-probs",
        function="to_probs",
        start=_T_START,
        solution=_T,
        visible=[(_N1,)],
        hidden=[(_N2,), (_N3,), (np.array([[5]]).view(_Matrix),)],
        show=_rows,
        save_as=("llm", "bigram_rows"),
        first=[_DOTS],
        mistakes=[
            Mistake("columns", wrong=_t(sums="N.sum(axis=0, keepdims=True)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: now every column adds up to 1, not every row. A row runs along axis 1: sum with `axis=1`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: nu telt elke kolom op tot 1, niet elke rij. Een rij loopt langs axis 1: tel op met `axis=1`."),
            Mistake("no-keepdims", wrong=_t(sums="N.sum(axis=1)"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Your sums have the shape (rows,), so broadcasting lines them up with the columns: each column is divided by the sum of a row. Keep the axis with `keepdims=True`, so that the sums have the shape (rows, 1).",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je sommen hebben de shape (rijen,), dus broadcasting koppelt ze aan de kolommen: elke kolom wordt gedeeld door de som van een rij. Houd de axis met `keepdims=True`, zodat de sommen de shape (rijen, 1) hebben."),
            Mistake("whole-sum", wrong=_t(sums="N.sum()"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you divide by the sum of the whole table, so all numbers together add up to 1, not each row. Sum each row with `axis=1`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je deelt door de som van de hele tabel, dus alle getallen samen tellen op tot 1, niet elke rij. Tel elke rij op met `axis=1`."),
        ],
        texts={
            **_SAVED,
            "empty": {"en": "Replace the `...` with the sum of each row, in a shape that broadcasting lines up with the rows.",
                      "nl": "Vervang de `...` door de som van elke rij, in een shape waarmee broadcasting elke rij door haar eigen som deelt."},
            "arguments": {"en": "The check calls `{call}`, with one matrix, and the `def` line of your function has another number of names between its parentheses. Keep `(N)`.",
                          "nl": "De check roept `{call}` aan, met één matrix, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(N)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Does every row of your result add up to 1?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Telt elke rij van je uitkomst op tot 1?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for a matrix of any size?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor een matrix van elke grootte?"},
        },
    ),
}


# --- the demo: the whole table of the plays, with the learner's own three functions ------------------------------------

def _saved(module: str) -> bool:
    return (projects.folder() / "llm" / f"{module}.py").exists()


def demo():
    """For the demo: None when bigram_matrix.py or bigram_rows.py is not in the project yet; "matrix" or "rows" when that
    saved module no longer gives what the course's gives on a small case, or stops with an error; "counts" when
    count_bigrams stops with an error or does not count the plays the way the course's does (the contract test of a
    function exercise cannot catch that); "slow" when counting or building the table takes more than 30 seconds; else
    (the matrix of probabilities of the plays, whether count_bigrams was the learner's own)."""
    if not (_saved("bigram_matrix") and _saved("bigram_rows")):
        return None
    run = run_code("from llm.bigram_matrix import bigram_matrix\nbigram_matrix({'ab': 3, 'ba': 1}, ['a', 'b'])")
    if run.error is not None or not np.array_equal(np.asarray(run.value), [[0, 3], [1, 0]]):
        return "matrix"
    matrix = run.names["bigram_matrix"]
    run = run_code("from llm.bigram_rows import to_probs\nimport numpy as np\nto_probs(np.array([[3, 1], [0, 2]]))")
    if run.error is not None or not np.allclose(np.asarray(run.value, dtype=float), [[0.75, 0.25], [0, 1]]):
        return "rows"
    probs = run.names["to_probs"]
    run = run_code("from llm.bigrams import count_bigrams")   # the learner's, or the course's when theirs is missing or broken
    if run.error is not None:
        return "counts"
    count = run.names["count_bigrams"]
    yours = all(module != "bigrams" for _, module, _, _ in projects.stand_ins())
    try:
        counts = with_time_limit(lambda: count(text()), 30)
    except Timeout:
        return "slow"
    except Exception:
        return "counts"
    if counts != _course_counts():
        return "counts"
    try:   # np.asarray: the check also counts a list of rows as right
        table = np.asarray(with_time_limit(lambda: probs(np.asarray(matrix(counts, list(chars())))), 30), dtype=float)
    except Timeout:
        return "slow"
    except Exception:
        return "matrix"
    if table.shape != (len(chars()), len(chars())) or not np.allclose(table.sum(axis=1), 1):
        return "matrix"
    return table, yours


def top_after(probs: np.ndarray, ch: str, n: int = 5) -> list[tuple[str, float]]:
    """The n most likely characters after ch, from a matrix of probabilities, for a picture."""
    cs = chars()
    row = probs[cs.index(ch)]
    best = np.argsort(-row)[:n]
    return [(shown(cs[j]), float(row[j])) for j in best]


# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("bigram-matrix", _M_START, "empty", None),
    ("bigram-matrix", _m("            row.append(counts.get(a + b, 0))\n"), "correct", None),
    ("bigram-matrix", _m("            if b + a in counts:\n                row.append(counts[b + a])\n            else:\n                row.append(0)\n"), "wrong", "swapped"),
    ("bigram-matrix", _m("            if a + b in counts:\n                row.append(1)\n            else:\n                row.append(0)\n"), "wrong", "ones"),
    ("bigram-matrix", _m("            row.append(counts[a + b])\n"), "error", "missing-pairs"),
    ("to-probs", _T_START, "empty", None),
    ("to-probs", _T_HEAD + "    return N / N.sum(axis=1, keepdims=True)\n", "correct", None),
    ("to-probs", _t(sums="N.sum(axis=0, keepdims=True)"), "wrong", "columns"),
    ("to-probs", _t(sums="N.sum(axis=1)"), "wrong", "no-keepdims"),
    ("to-probs", _t(sums="N.sum()"), "wrong", "whole-sum"),
]
