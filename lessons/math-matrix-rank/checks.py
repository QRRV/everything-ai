# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson math-matrix-rank: thin_numbers, how many numbers two thin matrices hold instead of one big
one, and from_pieces, a low-rank matrix built as a sum of rank-1 pieces, column times row. And the picture: a tall
thin matrix times a wide thin one gives a square matrix of rank at most r."""
import html
import math
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake


class _Matrix(np.ndarray):
    """An array that a check message writes the way the lesson's code does: np.array([...]). The learner's code gets
    a plain copy, so that what it computes is an ordinary array."""

    def __repr__(self):
        return f"np.array({_plain(np.asarray(self).tolist())!r})"

    def __deepcopy__(self, memo):
        return np.array(self)


class _Pieces(list):
    """A list of arrays that a check message writes as code, [np.array(...), ...]; the learner's code gets plain
    arrays."""

    def __repr__(self):
        return "[" + ", ".join(repr(x) for x in self) + "]"

    def __deepcopy__(self, memo):
        return [np.array(x) for x in self]


def _plain(v):
    if isinstance(v, list):
        return [_plain(x) for x in v]
    return int(v) if float(v) == int(v) else float(v)


def _m(rows) -> np.ndarray:
    return np.array(rows, dtype=float).view(_Matrix)


def _num(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    text = f"{round(x, 3) + 0.0:.3f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _shown(v) -> str:
    if isinstance(v, (list, tuple)):   # such as a list of pieces, each shown the same way
        return "[" + ", ".join(_shown(x) for x in v) + "]"
    if isinstance(v, np.ndarray):
        def rows(a):
            return "[" + ", ".join(rows(x) for x in a) + "]" if a.ndim else _num(float(a))
        return f"np.array({rows(np.asarray(v, dtype=float))})"
    try:
        return _num(float(v))
    except (TypeError, ValueError):
        return repr(v)


def _close(got, want) -> bool:
    try:
        g = np.asarray(got, dtype=float)
    except (TypeError, ValueError):
        return False
    w = np.asarray(want, dtype=float)
    return g.shape == w.shape and bool(np.allclose(g, w, rtol=1e-9, atol=1e-9))


def grouped(n: int, lang: str) -> str:
    s = f"{n:,}"
    return s.replace(",", ".") if lang == "nl" else s


# --- the picture: B (d × r) @ C (r × d) = A (d × d), drawn to their shapes -----------------------------------------------

def thin_picture(labels: dict, d: int = 8, r: int = 2) -> str:
    U = 13
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    top = 34

    def box(x, rows, cols, name, count):
        w, h = cols * U, rows * U
        p.append(f'<rect x="{x}" y="{top}" width="{w}" height="{h}" rx="3" style="fill:var(--ea-param-wash);stroke:var(--ea-param);stroke-width:1.5"/>')
        p.append(f'<text x="{x + w / 2}" y="{top + h / 2 + 5}" style="font-size:14px;font-weight:600;fill:var(--ea-ink);text-anchor:middle">{html.escape(name)}</text>')
        p.append(f'<text x="{x + w / 2}" y="{top + d * U + 16}" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:middle">({rows}, {cols})</text>')
        p.append(f'<text x="{x + w / 2}" y="{top + d * U + 31}" style="font-size:12px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(count)}</text>')
        return x + w

    def sign(x, s):
        p.append(f'<text x="{x}" y="{top + d * U / 2 + 6}" style="font-size:18px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(s)}</text>')

    x = box(26, d, r, "B", labels["numbers"].format(n=d * r))
    sign(x + 16, "@")
    x = box(x + 32, r, d, "C", labels["numbers"].format(n=r * d))
    sign(x + 16, "=")
    x = box(x + 32, d, d, "A", labels["numbers"].format(n=d * d))
    W, H = max(x + 30, 380), top + d * U + 40
    for line in labels["note"].format(r=r).split("\n"):   # the note, on as many lines as the label has
        H += 18
        p.append(f'<text x="0" y="{H}" style="font-size:13px;fill:var(--ea-ink)">{html.escape(line)}</text>')
    H += 8
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


# The lesson is done when these are correct. "rank-3" is the choice question in the lesson.
REQUIRED = ["rank-3", "thin-numbers", "from-pieces"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# --- exercise 2: how many numbers two thin matrices hold -----------------------------------------------------------------

_N_HEAD = "def thin_numbers(d_out, d_in, r):\n    # B has the shape (d_out, r), C the shape (r, d_in)\n"
_N_START = _N_HEAD + "    return ...\n"


def _n(body: str = "d_out * r + r * d_in") -> str:
    return _N_HEAD + f"    return {body}\n"


# --- exercise 3: a low-rank matrix as a sum of rank-1 pieces --------------------------------------------------------------

_P_HEAD = ("import numpy as np\n\n\ndef from_pieces(cols, rows):\n"
           "    # cols: r arrays of shape (d, 1); rows: r arrays of shape (1, d)\n"
           "    A = cols[0] @ rows[0]\n    for i in range(1, len(cols)):\n")
_P_END = "    return A\n"
_P_START = ("import numpy as np\n\n\ndef from_pieces(cols, rows):\n"
            "    # cols: r arrays of shape (d, 1); rows: r arrays of shape (1, d)\n    ...\n")


def _p(inner: str = "        A = A + cols[i] @ rows[i]\n") -> str:
    return _P_HEAD + inner + _P_END


_P_LIST = _P_START.replace("    ...\n", "    pieces = []\n    for i in range(len(cols)):\n        pieces.append(cols[i] @ rows[i])\n    return pieces\n")
_P_TWICE = _P_HEAD.replace("range(1, len(cols))", "range(len(cols))") + "        A = A + cols[i] @ rows[i]\n" + _P_END


def _pieces(cols, rows):
    return (_Pieces(_m(c) for c in cols), _Pieces(_m(r) for r in rows))


_C1, _R1 = _pieces([[[1], [0], [2]], [[0], [1], [1]]], [[[1, 2, 0]], [[0, 1, 3]]])
_C2, _R2 = _pieces([[[1], [1]], [[2], [-1]], [[0], [3]]], [[[1, 0]], [[0, 1]], [[1, 1]]])
_C3, _R3 = _pieces([[[2], [1], [0], [1]]], [[[1, -1, 0, 2]]])

EXERCISES = {
    "thin-numbers": FunctionBox(
        name="thin-numbers",
        function="thin_numbers",
        start=_N_START,
        solution=_n(),
        visible=[(5, 4, 2)],
        hidden=[(10, 3, 1), (4096, 4096, 8), (768, 3072, 16), (5, 5, 5)],
        first=[_DOTS],
        mistakes=[
            Mistake("square", wrong=_n("2 * d_out * r"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `2 * d * r` is only right for a square matrix. Count each thin matrix by its own shape: `B` is (d_out, r), `C` is (r, d_in).",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `2 * d * r` klopt alleen voor een vierkante matrix. Tel elke dunne matrix met zijn eigen shape: `B` is (d_out, r), `C` is (r, d_in)."),
            Mistake("full", wrong=_n("d_out * d_in"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the number of numbers in the full matrix. The question is how many the two thin matrices hold together.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is het aantal getallen in de volle matrix. De vraag is hoeveel de twee dunne matrices samen bevatten."),
            Mistake("times", wrong=_n("d_out * r * r * d_in"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you multiply the two counts. The two matrices are stored side by side, so their numbers add up.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je vermenigvuldigt de twee aantallen. De twee matrices worden naast elkaar bewaard, dus hun getallen tel je op."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the number of numbers in `B` and `C` together.",
                      "nl": "Vervang de `...` door het aantal getallen in `B` en `C` samen."},
            "arguments": {"en": "The check calls `{call}`, with three whole numbers, and the `def` line of your function has another number of names between its parentheses. Keep `(d_out, d_in, r)`.",
                          "nl": "De check roept `{call}` aan, met drie gehele getallen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(d_out, d_in, r)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. How many numbers does a matrix of shape (n, m) hold, and what are the shapes of `B` and `C`?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Hoeveel getallen bevat een matrix van shape (n, m), en wat zijn de shapes van `B` en `C`?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every shape and every `r`?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke shape en elke `r`?"},
            "correct": {"en": "Your function is right in all {n} cases: the numbers of the two thin matrices, added.",
                        "nl": "Je functie klopt in alle {n} gevallen: de getallen van de twee dunne matrices, opgeteld."},
        },
    ),
    "from-pieces": FunctionBox(
        name="from-pieces",
        function="from_pieces",
        start=_P_START,
        solution=_p(),
        visible=[(_C1, _R1)],
        hidden=[(_C2, _R2), (_C3, _R3)],
        show=_shown,
        compare=_close,
        first=[_DOTS],
        mistakes=[
            Mistake("overwrite", wrong=_p("        A = cols[i] @ rows[i]\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is the last piece alone, as if each piece replaced the one before. Add the pieces up.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is alleen het laatste stuk, alsof elk stuk het vorige vervangt. Tel de stukken op."),
            Mistake("row-first", wrong=_p("        A = A + rows[i] @ cols[i]\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: `rows[i] @ cols[i]` is (1, d) @ (d, 1), a single number, which NumPy then adds to every entry. A rank-1 piece is the column times the row: (d, 1) @ (1, d).",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: `rows[i] @ cols[i]` is (1, d) @ (d, 1), één getal, dat NumPy dan bij elk getal optelt. Een stuk van rang 1 is de kolom keer de rij: (d, 1) @ (1, d)."),
            Mistake("times", wrong=_p("        A = A * (cols[i] @ rows[i])\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you multiply the pieces number by number. The pieces are added: each adds at most one direction.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je vermenigvuldigt de stukken getal voor getal. De stukken tel je op: elk voegt hooguit één richting toe."),
            Mistake("list", wrong=_P_LIST,
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is a list with each piece on its own. Add the pieces up into one matrix.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is een lijst met elk stuk apart. Tel de stukken op tot één matrix."),
            Mistake("first-twice", wrong=_P_TWICE,
                    en="`{call}` gives `{got}`, and it should give `{want}`: the first piece counts twice. Add each piece once: if `A` starts with the first piece, the loop starts at the second.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: het eerste stuk telt twee keer mee. Tel elk stuk één keer: begint `A` met het eerste stuk, dan begint de loop bij het tweede."),
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the body of the function: every piece, column times row, added up.",
                      "nl": "Vervang de `...` door de body van de functie: elk stuk, kolom keer rij, opgeteld."},
            "arguments": {"en": "The check calls `{call}`, with a list of columns and a list of rows, and the `def` line of your function has another number of names between its parentheses. Keep `(cols, rows)`.",
                          "nl": "De check roept `{call}` aan, met een lijst kolommen en een lijst rijen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(cols, rows)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Is every piece the column times the row, and are the pieces added?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Is elk stuk de kolom keer de rij, en tel je de stukken op?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for any number of pieces, also one?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elk aantal stukken, ook één?"},
            "correct": {"en": "Your function is right in all {n} cases: r pieces of rank 1 added, a matrix of rank at most r.",
                        "nl": "Je functie klopt in alle {n} gevallen: r stukken van rang 1 opgeteld, een matrix van rang hooguit r."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("thin-numbers", _N_START, "empty", None),
    ("thin-numbers", _n(), "correct", None),
    ("thin-numbers", _n("r * (d_out + d_in)"), "correct", None),
    ("thin-numbers", _n("2 * d_out * r"), "wrong", "square"),
    ("thin-numbers", _n("d_out * d_in"), "wrong", "full"),
    ("thin-numbers", _n("d_out * r * r * d_in"), "wrong", "times"),
    ("from-pieces", _P_START, "empty", None),
    ("from-pieces", _p(), "correct", None),
    ("from-pieces", _p("        piece = cols[i] @ rows[i]\n        A = A + piece\n"), "correct", None),
    ("from-pieces", _p("        A = cols[i] @ rows[i]\n"), "wrong", "overwrite"),
    ("from-pieces", _p("        A = A + rows[i] @ cols[i]\n"), "wrong", "row-first"),
    ("from-pieces", _p("        A = A * (cols[i] @ rows[i])\n"), "wrong", "times"),
    ("from-pieces", _P_LIST, "wrong", "list"),
    ("from-pieces", _P_TWICE, "wrong", "first-twice"),
    ("from-pieces", _P_START.replace("    ...\n", "    return sum(cols[i] @ rows[i] for i in range(len(cols)))\n"), "correct", None),
    ("thin-numbers", _n("2 * d_out + 2 * d_in"), "almost", None),
]
