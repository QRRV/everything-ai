# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson gpt-attention-maps: previous_score, how much of its weight a head puts on the character just
before, and share_on, how much it puts on some characters, such as spaces. The attention weights come from the
learner's trained GPT of gpt-train-chars, or the course's when they have not trained one: worked out from its saved
parameters, so the learner's code of earlier steps is not needed."""
import html
import json
import math
import re
from functools import cache

import torch   # imported once, when the lesson opens: the first import takes longer than a check may

from everything_ai import paths, projects
from everything_ai.check import FunctionBox, Mistake, run_code

COURSE_GPT = "data/course-gpt-chars"   # .pt and .json: the course's GPT of gpt-train-chars, see data/README.md
TEXT = "JULIET:\nWhat's in a name?"       # the text of the pictures
DEMO_TEXT = "ROMEO:\nBut, soft! what light through yonder window breaks?\nIt is the east, and Juliet is the sun."
MARKS = "\n"                             # new lines: where a line starts


def shown(ch: str) -> str:
    return {" ": "␣", "\n": "↵"}.get(ch, ch)


def trained() -> dict:
    """The trained GPT whose attention the lesson shows: the learner's of gpt-train-chars ("yours"), else the course's
    ("course"); with its sizes, its characters and its parameters. A checkpoint that does not fit is skipped."""
    folder = projects.folder() / "llm"
    for source, base in (("yours", folder / "gpt_chars"), ("course", paths.ROOT / COURSE_GPT)):
        pt, info = base.with_suffix(".pt"), base.with_suffix(".json")
        if not (pt.exists() and info.exists()):
            continue
        try:
            meta = json.loads(info.read_text(encoding="utf-8"))
            state = {k: v.float() for k, v in torch.load(pt, map_location="cpu", weights_only=True).items()}   # the course's is in float16
            d, h, n_layers, block = meta["sizes"]
            weights(state, (d, h, n_layers, block), meta["chars"], "ab")   # does it fit?
        except Exception:
            continue
        return {"source": source, "sizes": (d, h, n_layers, block), "chars": meta["chars"], "state": state}
    return {"source": None}


@torch.no_grad()
def weights(state: dict, sizes: tuple, chars: str, text: str) -> list:
    """The attention weights of every head of every block for `text`: a list of blocks, each a list of heads, each an
    (n, n) tensor whose row t holds the weights position t gives to positions 0 to t. Worked out from the parameters
    with the course's GPT of gpt-assemble: pre-norm blocks, causal attention, an MLP of 4 d with ReLU."""
    F = torch.nn.functional
    d, h, n_layers, block = sizes
    ids = torch.tensor([chars.index(c) for c in text[:block]])
    n, dh = len(ids), d // h
    x = state["tok.weight"][ids] + state["pos.weight"][:n]
    future = torch.triu(torch.ones(n, n, dtype=torch.bool), 1)
    out = []
    for layer in range(n_layers):
        p = f"blocks.{layer}."
        a = F.layer_norm(x, (d,), state[p + "norm1.weight"], state[p + "norm1.bias"])
        q, k, v = (a @ state[p + f"attention.{m}.weight"].T for m in ("query", "key", "value"))
        heads, mixed = [], []
        for i in range(h):
            s = q[:, i * dh:(i + 1) * dh] @ k[:, i * dh:(i + 1) * dh].T / math.sqrt(dh)
            w = torch.softmax(s.masked_fill(future, float("-inf")), dim=1)
            heads.append(w)
            mixed.append(w @ v[:, i * dh:(i + 1) * dh])
        out.append(heads)
        x = x + torch.cat(mixed, dim=1) @ state[p + "attention.out.weight"].T + state[p + "attention.out.bias"]
        b = F.layer_norm(x, (d,), state[p + "norm2.weight"], state[p + "norm2.bias"])
        x = x + torch.relu(b @ state[p + "lin1.weight"].T + state[p + "lin1.bias"]) @ state[p + "lin2.weight"].T + state[p + "lin2.bias"]
    return out


@cache
def course_maps(text: str = TEXT) -> list:
    """The attention weights of the course's GPT for `text`, as for `weights`."""
    meta = json.loads((paths.ROOT / (COURSE_GPT + ".json")).read_text(encoding="utf-8"))
    state = {k: v.float() for k, v in torch.load(paths.ROOT / (COURSE_GPT + ".pt"), map_location="cpu", weights_only=True).items()}
    return weights(state, tuple(meta["sizes"]), meta["chars"], text)


# --- the picture: heatmaps of attention weights --------------------------------------------------------------------------------

_LEVELS = [(0.0, 0.06), (0.05, 0.18), (0.15, 0.38), (0.3, 0.6), (0.5, 0.8), (0.75, 1.0)]


def _opacity(p: float) -> float:
    out = 0.0
    for lo, op in _LEVELS:
        if p > lo:
            out = op
    return out


def heatmap(w, text: str, title: str, alt: str, labels: dict) -> str:
    """One head's attention weights as a heatmap: row t is the position that looks, column j the position it looks at;
    a square per weight, darker the larger. Attention weights are worked out, not learned: the hidden colour."""
    n = len(text)
    cell = min(13, 270 // n)
    left, top = 22, 56
    W, H = left + n * cell + 6, top + n * cell + 30
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>',
         f'<text x="{left}" y="36" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(labels["cols"])}</text>']
    for k, c in enumerate(text):
        p.append(f'<text x="{left + k * cell + cell / 2:.1f}" y="{top - 5}" style="font-size:11px;fill:var(--ea-ink-2);text-anchor:middle">{html.escape(shown(c))}</text>')
        p.append(f'<text x="{left - 4}" y="{top + k * cell + cell * 0.78:.1f}" style="font-size:11px;fill:var(--ea-ink-2);text-anchor:end">{html.escape(shown(c))}</text>')
    for t in range(n):
        for j in range(t + 1):
            op = _opacity(float(w[t][j]))
            if op:
                p.append(f'<rect x="{left + j * cell}" y="{top + t * cell}" width="{cell - 1}" height="{cell - 1}" '
                         f'style="fill:var(--ea-hidden);fill-opacity:{op}"/>')
    p.append(f'<path d="M{left - 0.5} {top - 0.5} H{left + n * cell - 0.5} V{top + n * cell - 0.5} H{left - 0.5} Z" style="fill:none;stroke:var(--ea-line)"/>')
    p.append(f'<text x="{left}" y="{H - 8}" style="font-size:12px;fill:var(--ea-ink-2)">{html.escape(labels["rows"])}</text>')
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(alt)}</title>{"".join(p)}</svg></figure>')


# --- the course's own measures, which stand in for the learner's until theirs are right ----------------------------------

def previous_score(W):
    n = len(W)
    return sum([W[t][t - 1] for t in range(1, n)]) / (n - 1)


def share_on(W, text, targets):
    n = len(W)
    total = 0.0
    for t in range(n):
        for j in range(t + 1):
            if text[j] in targets:
                total = total + W[t][j]
    return total / n


# --- exercise 1: previous_score ------------------------------------------------------------------------------------------------

_P_HEAD = "def previous_score(W):\n    n = len(W)\n"
_P_START = _P_HEAD + "    # the weight of every row t on the column just before it, t - 1, averaged over the rows that have one\n    return ...\n"


def _p(body: str = "    return sum([W[t][t - 1] for t in range(1, n)]) / (n - 1)\n") -> str:
    return _P_HEAD + body


_P_LOOP = ("    total = 0\n    for t in range(1, n):\n        total = total + W[t][t - 1]\n    return total / (n - 1)\n")

_P_WRONG = {
    "self": _p("    return sum([W[t][t] for t in range(n)]) / n\n"),
    "all-rows": _p("    return sum([W[t][t - 1] for t in range(1, n)]) / n\n"),
    "above": _p("    return sum([W[t - 1][t] for t in range(1, n)]) / (n - 1)\n"),
}

# Attention weights of 3 or 4 positions: lower triangles whose rows add up to 1, as a causal head gives.
_W3 = [[1.0, 0.0, 0.0], [0.8, 0.2, 0.0], [0.1, 0.6, 0.3]]
_W4 = [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.2, 0.7, 0.1, 0.0], [0.1, 0.1, 0.6, 0.2]]
_W4B = [[1.0, 0.0, 0.0, 0.0], [0.3, 0.7, 0.0, 0.0], [0.4, 0.4, 0.2, 0.0], [0.25, 0.25, 0.25, 0.25]]
_W2 = [[1.0, 0.0], [0.9, 0.1]]
_W4P = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]   # only the character before
_P_CASES = [(_W3,), (_W4,), (_W4B,), (_W4P,), (_W2,)]
_S_CASES = [(_W4, "a b,", " ,"), (_W3, "ab ", " "), (_W4B, "x.y.", "."), (_W4, "abcd", " "), (_W2, "  ", " ")]

# --- exercise 2: share_on ------------------------------------------------------------------------------------------------------

_S_HEAD = "def share_on(W, text, targets):\n    n = len(W)\n    total = 0.0\n    for t in range(n):\n        for j in range(t + 1):\n"
_S_START = (_S_HEAD + "            # the weight row t gives to position j, when the character at j is one of the targets\n            ...\n"
            "    # averaged over the rows\n    return ...\n")


def _s(inner: str = "            if text[j] in targets:\n                total = total + W[t][j]\n", end: str = "    return total / n\n") -> str:
    return _S_HEAD + inner + end


_S_WRONG = {
    "looker": _s(inner="            if text[t] in targets:\n                total = total + W[t][j]\n"),
    "sum": _s(end="    return total\n"),
    "column": _s(inner="            if text[j] in targets:\n                total = total + W[j][t]\n"),
}

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

def _rounded(got) -> bool:
    """`got` is a right answer of some case, rounded to 0 to 4 decimals: what `round` makes of a right answer."""
    try:
        g = float(got)
    except (TypeError, ValueError):
        return False
    wants = [previous_score(*c) for c in _P_CASES] + [share_on(*c) for c in _S_CASES]
    return any(math.isclose(g, round(w, k), abs_tol=1e-9) for w in wants for k in range(5))


_ROUNDED = Mistake("rounded", matches=lambda got, text: re.search(r"\bround\s*\(", text) is not None and _rounded(got),
                   en="`{call}` gives {got}, and it should give {want}: you round with `round`, and the check compares more digits than it shows. Give back the number as it comes out, without rounding.",
                   nl="`{call}` geeft {got}, en dat moet {want} zijn: je rondt af met `round`, en de check vergelijkt meer cijfers dan hij laat zien. Geef het getal terug zoals het uitkomt, zonder af te ronden.")

_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _close(got, want) -> bool:
    try:
        return not isinstance(got, (str, bytes, bool)) and math.isclose(float(got), float(want), rel_tol=1e-5, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def _show(v):
    """A number rounded to 4 decimals, without trailing zeros; a decimal comma in Dutch."""
    try:
        if isinstance(v, (str, bytes, bool)) or v is None:
            return repr(v)
        x = float(v)
    except (TypeError, ValueError):
        return repr(v)
    if x != x or math.isinf(x):
        return str(x)
    text = f"{round(x, 4) + 0.0:.4f}".rstrip("0").rstrip(".")
    return {"en": text, "nl": text.replace(".", ",")}


EXERCISES = {
    "previous-score": FunctionBox(
        name="previous-score",
        function="previous_score",
        start=_P_START,
        solution=_p(),
        visible=_P_CASES[:1],
        hidden=_P_CASES[1:],
        compare=_close,
        show=_show,
        first=[_DOTS],
        mistakes=[
            Mistake("self", wrong=_P_WRONG["self"],
                    en="`{call}` gives {got}, and it should give {want}: you take the weight every position gives to itself, the main diagonal. The character just before position t is at column t - 1: `W[t][t - 1]`, for t from 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je neemt de weight die elke positie aan zichzelf geeft, de hoofddiagonaal. Het teken vlak ervoor staat voor positie t in kolom t - 1: `W[t][t - 1]`, voor t vanaf 1."),
            Mistake("all-rows", wrong=_P_WRONG["all-rows"],
                    en="`{call}` gives {got}, and it should give {want}: you divide by all n rows, but row 0 has no character before it. Average over the n - 1 rows that have one.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je deelt door alle n rijen, maar rij 0 heeft geen teken ervoor. Neem het gemiddelde over de n - 1 rijen die er een hebben."),
            Mistake("above", wrong=_P_WRONG["above"],
                    en="`{call}` gives {got}, and it should give {want}: you take weights above the diagonal, what a position gives to a position after it, and the mask makes those 0. The weight of row t on the character before is in column t - 1: `W[t][t - 1]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je neemt weights boven de diagonaal, wat een positie aan een positie erna geeft, en de mask maakt die 0. De weight van rij t op het teken ervoor staat in kolom t - 1: `W[t][t - 1]`."),
            Mistake("comma-index", error=lambda e: isinstance(e, TypeError) and "not tuple" in str(e),
                    en="With `{call}`, your function stops with an error: `W` is a list of rows, and a list takes one index at a time. First the row, then the column: `W[t][t - 1]`. `W[t, t - 1]`, with a comma, works on a NumPy array or a tensor.",
                    nl="Met `{call}` stopt je functie met een fout: `W` is een lijst van rijen, en een lijst neemt één index tegelijk. Eerst de rij, dan de kolom: `W[t][t - 1]`. `W[t, t - 1]`, met een komma, werkt op een NumPy-array of een tensor."),
            _ROUNDED,
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the `...` with the average weight on the character just before, over the rows that have one.",
                      "nl": "Vervang de `...` door de gemiddelde weight op het teken vlak ervoor, over de rijen die er een hebben."},
            "arguments": {"en": "The check calls `{call}`, with the weights of one head, and the `def` line of your function has another number of names between its parentheses. Keep `(W)`.",
                          "nl": "De check roept `{call}` aan, met de weights van één head, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(W)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Which entry of row t is the weight on the character just before, and over how many rows do you average?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Welk getal van rij t is de weight op het teken vlak ervoor, en over hoeveel rijen neem je het gemiddelde?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does it work for 4 positions, and for 2?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt het ook voor 4 posities, en voor 2?"},
            "correct": {"en": "Your score is right in all {n} cases: 1 for a head that only looks at the character before, and far less for one that spreads its weight evenly, as row t then gives it only 1/(t + 1).",
                        "nl": "Je score klopt in alle {n} gevallen: 1 voor een head die alleen naar het teken ervoor kijkt, en veel minder voor een die zijn weight gelijk verdeelt, want rij t geeft het dan maar 1/(t + 1)."},
        },
    ),
    "share-on": FunctionBox(
        name="share-on",
        function="share_on",
        start=_S_START,
        solution=_s(),
        visible=_S_CASES[:1],
        hidden=_S_CASES[1:],
        compare=_close,
        show=_show,
        first=[_DOTS],
        mistakes=[
            Mistake("looker", wrong=_S_WRONG["looker"],
                    en="`{call}` gives {got}, and it should give {want}: you look at the character of the row, `text[t]`, the position that looks. What counts is where the weight goes: the character at column j, `text[j]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je kijkt naar het teken van de rij, `text[t]`, de positie die kijkt. Het gaat erom waar de weight heen gaat: het teken in kolom j, `text[j]`."),
            Mistake("sum", wrong=_S_WRONG["sum"],
                    en="`{call}` gives {got}, and it should give {want}: the total over all rows, so a longer text gives a larger number. Divide by the number of rows, n, for the share of one row on average.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: het totaal over alle rijen, dus een langere tekst geeft een groter getal. Deel door het aantal rijen, n, voor het gemiddelde deel van één rij."),
            Mistake("column", wrong=_S_WRONG["column"],
                    en="`{call}` gives {got}, and it should give {want}: `W[j][t]` is what position j gives to t, and with j up to t that is mostly above the diagonal, where the mask puts 0. Row t gives its weight to column j: `W[t][j]`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: `W[j][t]` is wat positie j aan t geeft, en met j tot en met t ligt dat vooral boven de diagonaal, waar de mask 0 zet. Rij t geeft zijn weight aan kolom j: `W[t][j]`."),
            Mistake("comma-index", error=lambda e: isinstance(e, TypeError) and "not tuple" in str(e),
                    en="With `{call}`, your function stops with an error: `W` is a list of rows, and a list takes one index at a time. First the row, then the column: `W[t][j]`. `W[t, j]`, with a comma, works on a NumPy array or a tensor.",
                    nl="Met `{call}` stopt je functie met een fout: `W` is een lijst van rijen, en een lijst neemt één index tegelijk. Eerst de rij, dan de kolom: `W[t][j]`. `W[t, j]`, met een komma, werkt op een NumPy-array of een tensor."),
            _ROUNDED,
        ],
        texts={
            **_TEXTS,
            "empty": {"en": "Replace the two `...`: add the weight of row t on position j when the character at j is a target, and give back the average over the rows.",
                      "nl": "Vervang de twee `...`: tel de weight van rij t op positie j op als het teken op j een doel is, en geef het gemiddelde over de rijen terug."},
            "arguments": {"en": "The check calls `{call}`, with the weights of one head, the text and the characters to look for, and the `def` line of your function has another number of names between its parentheses. Keep `(W, text, targets)`.",
                          "nl": "De check roept `{call}` aan, met de weights van één head, de tekst en de tekens om naar te zoeken, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(W, text, targets)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. For every row t: the weights on the positions j whose character is a target; then the average over the rows.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Voor elke rij t: de weights op de posities j waarvan het teken een doel is; dan het gemiddelde over de rijen."},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does it work for other texts and targets, and when no character is a target?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt het ook voor andere teksten en doelen, en als geen enkel teken een doel is?"},
            "correct": {"en": "Your share is right in all {n} cases: how much of its weight a head gives, on average, to the characters you ask about.",
                        "nl": "Je deel klopt in alle {n} gevallen: hoeveel van zijn weight een head gemiddeld geeft aan de tekens waar je naar vraagt."},
        },
    ),
}

# The lesson is done when these are correct. "not-explanation" is the choice question.
REQUIRED = ["previous-score", "share-on", "not-explanation"]

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("previous-score", _P_START, "empty", None),
    ("previous-score", _p(), "correct", None),
    ("previous-score", _p(_P_LOOP), "correct", None),
    ("previous-score", _p("    return round(sum([W[t][t - 1] for t in range(1, n)]) / (n - 1), 2)\n"), "wrong", "rounded"),
    ("previous-score", _p("    return round(sum([W[t][t] for t in range(n)]) / n, 2)\n"), "wrong", None),   # rounds, but the diagonal is the mistake
    ("previous-score", _p("    return sum([W[t][t - 1] for t in range(1, n)]) // (n - 1)\n"), "wrong", None),
    ("previous-score", _p("    return sum([W[t][t + 1] for t in range(n - 1)]) / (n - 1)\n"), "wrong", "above"),
    ("previous-score", _p("    return sum([W[t, t - 1] for t in range(1, n)]) / (n - 1)\n"), "error", "comma-index"),
] + [("previous-score", code, "wrong", key) for key, code in _P_WRONG.items()] + [
    ("share-on", _S_START, "empty", None),
    ("share-on", _s(), "correct", None),
    ("share-on", _s(inner="            total = total + W[t][j] * (text[j] in targets)\n"), "correct", None),
    ("share-on", _s(end="    return round(total / n, 1)\n"), "wrong", "rounded"),
    ("share-on", _s(inner="            if text[j] in targets:\n                total = total + W[t, j]\n"), "error", "comma-index"),
] + [("share-on", code, "wrong", key) for key, code in _S_WRONG.items()]


def _functions(prev_text, share_text):
    """The learner's two functions where they are right, else the course's: (previous_score, share_on, yours, yours)."""
    prev = share = None
    if prev_text:
        ran = run_code(prev_text)
        prev = ran.names.get("previous_score") if ran.error is None else None
    if share_text:
        ran = run_code(share_text)
        share = ran.names.get("share_on") if ran.error is None else None
    return (prev or previous_score, share or share_on, prev is not None, share is not None)


def demo(prev_text, share_text) -> dict:
    """For every head of the trained GPT, on DEMO_TEXT: its score on the character before and its share on new
    lines, with the learner's functions where they are right. The heads with the highest of each."""
    which = trained()
    if which["source"] is None:
        return {"source": None}
    prev, share, p_yours, s_yours = _functions(prev_text, share_text)
    maps = weights(which["state"], which["sizes"], which["chars"], DEMO_TEXT)
    rows = []
    for layer, heads in enumerate(maps):
        for i, w in enumerate(heads):
            W = w.tolist()
            try:
                rows.append((layer, i, float(prev(W)), float(share(W, DEMO_TEXT, MARKS))))
            except Exception:
                rows.append((layer, i, float(previous_score(W)), float(share_on(W, DEMO_TEXT, MARKS))))
    uniform = share_on([[1 / (t + 1)] * (t + 1) + [0.0] * (len(DEMO_TEXT) - t - 1) for t in range(len(DEMO_TEXT))], DEMO_TEXT, MARKS)
    return {"source": which["source"], "rows": rows, "prev": max(rows, key=lambda r: r[2]), "marks": max(rows, key=lambda r: r[3]),
            "uniform": uniform, "yours": (p_yours, s_yours), "maps": maps}
