# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rnn-memory-idea: rnn, a recurrent network that reads a sequence one step at a time and carries
its hidden state from step to step. And the picture: the hidden states of the worked example's network while it reads
"abbb"."""
import html
import re

import numpy as np

from everything_ai.check import FunctionBox, Mistake

# The network of the worked example: the first number of the hidden state remembers the a's, the second the b's.
VEC = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
W_X = np.array([[1.0, 0.0], [0.0, 1.0]])
W_H = np.array([[0.5, 0.0], [0.0, 0.5]])
B = np.array([0.0, 0.0])


def rnn(xs, W_x, W_h, b):
    """The course's own recurrent network: the last hidden state after reading every input of xs, in order."""
    W_x, W_h, b = np.asarray(W_x, dtype=float), np.asarray(W_h, dtype=float), np.asarray(b, dtype=float)
    h = np.zeros(len(b))
    for x in xs:
        h = np.tanh(W_x @ np.asarray(x, dtype=float) + W_h @ h + b)
    return h


def states(text: str) -> list:
    """The hidden state of the worked example's network after each character of text."""
    h, out = np.zeros(2), []
    for ch in text:
        h = np.tanh(W_X @ VEC[ch] + W_H @ h + B)
        out.append(h)
    return out


def trace(n: int) -> float:
    """What is left of the a in the hidden state after an "a" and then n b's."""
    return float(states("a" + "b" * n)[-1][0])


# --- the picture: the hidden state as two bars after every step, the inputs below, the weights on the arrows --------

def picture(text: str, labels: dict[str, str], lang: str) -> str:
    """The network of the worked example reading `text`: the hidden state after every step as two bars, one for the
    a's and one for the b's, with an arrow W_h from step to step and an arrow W_x from every character up."""
    hs = [np.zeros(2)] + states(text)
    box_w, box_h, step, top, bar_h = 60, 100, 94, 34, 70
    width = 8 + step * (len(hs) - 1) + box_w + 8
    char_y = top + box_h + 62
    height = char_y + 56
    num = lambda v: "0" if abs(v) < 0.005 else (f"{v:.2f}".replace(".", ",") if lang == "nl" else f"{v:.2f}")
    p = ['<defs><marker id="rnn-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
         '<path d="M0 0L10 5L0 10z" style="fill:var(--ea-ink-2)"/></marker></defs>',
         f'<text x="0" y="18" style="font-size:17px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    for k, h in enumerate(hs):
        x = 8 + k * step
        p.append(f'<rect x="{x}" y="{top}" width="{box_w}" height="{box_h}" rx="6" '
                 f'style="fill:var(--ea-hidden-wash);stroke:var(--ea-hidden);stroke-width:1.5"/>')
        base = top + box_h - 20
        for j, (v, name) in enumerate(zip(h, ("a", "b"))):
            bx = x + 8 + j * 24
            hgt = max(0.0, float(v)) * bar_h * 0.95
            p.append(f'<rect x="{bx}" y="{base - hgt:.1f}" width="18" height="{hgt:.1f}" style="fill:var(--ea-hidden)"/>')
            p.append(f'<text x="{bx + 9}" y="{base - hgt - 4:.1f}" style="font-size:14px;fill:var(--ea-ink-2);text-anchor:middle">{num(float(v))}</text>')
            p.append(f'<text x="{bx + 9}" y="{base + 15}" style="font-size:14px;fill:var(--ea-ink-2);text-anchor:middle">{name}</text>')
        p.append(f'<line x1="{x + 4}" y1="{base}" x2="{x + box_w - 4}" y2="{base}" style="stroke:var(--ea-line);stroke-width:1"/>')
        p.append(f'<text x="{x + box_w / 2}" y="{top + box_h + 19}" style="font-size:16px;font-style:italic;fill:var(--ea-hidden-text);'
                 f'text-anchor:middle">h<tspan dy="4" style="font-size:12px">{k}</tspan></text>')
        if k > 0:
            ax = x - step + box_w
            p.append(f'<line x1="{ax + 3}" y1="{top + box_h / 2}" x2="{x - 3}" y2="{top + box_h / 2}" '
                     f'style="stroke:var(--ea-ink-2);stroke-width:1.5" marker-end="url(#rnn-arrow)"/>')
            p.append(f'<text x="{(ax + x) / 2}" y="{top + box_h / 2 - 12}" style="font-size:15px;font-style:italic;'
                     f'fill:var(--ea-param-text);text-anchor:middle">W<tspan dy="4" style="font-size:11px">h</tspan></text>')
            cx = x + box_w / 2
            p.append(f'<line x1="{cx}" y1="{char_y - 4}" x2="{cx}" y2="{top + box_h + 28}" '
                     f'style="stroke:var(--ea-ink-2);stroke-width:1.5" marker-end="url(#rnn-arrow)"/>')
            p.append(f'<text x="{cx + 6}" y="{char_y - 18}" style="font-size:15px;font-style:italic;fill:var(--ea-param-text)">'
                     f'W<tspan dy="4" style="font-size:11px">x</tspan></text>')
            ch = text[k - 1]
            p.append(f'<rect x="{cx - 17}" y="{char_y}" width="34" height="30" rx="4" '
                     f'style="fill:var(--ea-data-wash);stroke:var(--ea-data);stroke-width:1.5"/>')
            p.append(f'<text x="{cx}" y="{char_y + 21}" style="font-size:17px;font-family:var(--ea-font-code);fill:var(--ea-data-text);'
                     f'text-anchor:middle">{html.escape(ch)}</text>')
            p.append(f'<text x="{cx}" y="{char_y + 49}" style="font-size:14px;fill:var(--ea-ink-2);text-anchor:middle">'
                     f'{html.escape(labels["step"])} {k}</text>')
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:min(40rem,{width}px)">'
            f'<svg viewBox="0 0 {width} {height}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


# The lesson is done when these are correct. "size" and "no-memory" are choice questions in the lesson.
REQUIRED = ["size", "rnn", "no-memory"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _one_line(v):
    """A vector as a list on one line, rounded to 4 decimals, for the check's messages."""
    try:
        a = np.round(np.asarray(v, dtype=float), 4)
    except (TypeError, ValueError):
        return repr(v)
    return repr((a + 0.0).tolist())


# --- exercise 2: rnn ----------------------------------------------------------------------------------------------------

_R_HEAD = "import numpy as np\n\n\ndef rnn(xs, W_x, W_h, b):\n    W_x, W_h, b = np.array(W_x), np.array(W_h), np.array(b)\n"
_R_START = _R_HEAD + "    h = ...\n    for x in xs:\n        h = ...\n    return h\n"


def _r(start: str = "    h = np.zeros(len(b))\n", step: str = "        h = np.tanh(W_x @ x + W_h @ h + b)\n") -> str:
    return _R_HEAD + start + "    for x in xs:\n" + step + "    return h\n"


_R = _r()

EXERCISES = {
    "rnn": FunctionBox(
        name="rnn",
        function="rnn",
        start=_R_START,
        solution=_R,
        visible=[([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[0.5, 0], [0, 0.5]], [0, 0])],
        hidden=[([[1, 0], [0, 1], [1, 1]], [[1, -1], [0.5, 0], [0, 2]], [[0.2, 0, -0.3], [0.1, 0.4, 0], [0, -0.5, 0.3]], [0, 0.1, -0.2]),
                ([[2, -1]], [[1, 1]], [[0.9]], [0.5]),
                ([[1], [-1], [2], [0], [1]], [[0.8], [-0.4]], [[0.5, -0.6], [0.3, 0.2]], [0.1, 0]),
                ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[0, 0], [0, 0]], [0, 0])],
        show=lambda v: _one_line(v),
        first=[_DOTS],
        mistakes=[
            Mistake("no-memory", wrong=_r(step="        h = np.tanh(W_x @ x + b)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: only the last input counts, as if the network forgets everything at every step. Does every step add `W_h @ h`, with `h` the hidden state of the step before, not zeros?",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: alleen de laatste input telt, alsof het netwerk bij elke stap alles vergeet. Telt elke stap `W_h @ h` op, met `h` de hidden state van de stap ervoor, en niet nullen?"),
            Mistake("no-tanh", wrong=_r(step="        h = W_x @ x + W_h @ h + b\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your step adds up the right parts, but does not put the sum through `np.tanh`, which keeps every number between −1 and 1.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je stap telt de goede delen op, maar stopt de som niet in `np.tanh`, die elk getal tussen −1 en 1 houdt."),
            Mistake("swapped", wrong=_r(step="        h = np.tanh(W_h @ x + W_x @ h + b)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. `W_x` weighs the input `x`, and `W_h` the hidden state `h` of the step before: in your step they are the other way round.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. `W_x` weegt de input `x`, en `W_h` de hidden state `h` van de stap ervoor: in jouw stap zijn ze omgedraaid."),
            Mistake("return-in-loop", wrong=_R.replace("\n    return h\n", "\n        return h\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: only the first input counts. Your `return` is indented under the loop, so the function stops after the first step. Put `return h` after the loop, as far in as `for`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: alleen de eerste input telt. Je `return` is ingesprongen onder de loop, dus stopt de functie na de eerste stap. Zet `return h` na de loop, even ver naar binnen als `for`."),
            Mistake("adds-up", wrong=_r(step="        h = h + np.tanh(W_x @ x + W_h @ h + b)\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your hidden state piles up step after step. The new hidden state replaces the old one: `h = np.tanh(...)`, without adding the old `h` to it.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je hidden state stapelt zich stap na stap op. De nieuwe hidden state vervangt de oude: `h = np.tanh(...)`, zonder de oude `h` erbij op te tellen."),
            Mistake("no-bias", state="almost", wrong=_r(step="        h = np.tanh(W_x @ x + W_h @ h)\n"),
                    en="Right for the example, where `b` is all zeros. But `{call}` gives `{got}`, and it should give `{want}`: your step leaves out the bias terms `b`. Every step adds them too, inside `np.tanh`.",
                    nl="Goed voor het voorbeeld, waar `b` vol nullen staat. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn: je stap laat de bias terms `b` weg. Elke stap telt ze ook op, binnen `np.tanh`."),
            Mistake("bias-outside", state="almost", wrong=_r(step="        h = np.tanh(W_x @ x + W_h @ h) + b\n"),
                    en="Right for the example, where `b` is all zeros. But `{call}` gives `{got}`, and it should give `{want}`: your step adds `b` after `np.tanh`. The bias terms go inside, with the other two parts, as in step 3 of the recipe.",
                    nl="Goed voor het voorbeeld, waar `b` vol nullen staat. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn: je stap telt `b` op na `np.tanh`. De bias terms horen erbinnen, bij de andere twee delen, zoals in stap 3 van het recept."),
            Mistake("shapes", error=lambda e: isinstance(e, ValueError) and ("mismatch" in str(e) or "broadcast" in str(e)),
                    en="With `{call}`, Python stops at a `@` or a `+`: the shapes do not fit. Write the shape next to every part, as in *The forward pass in NumPy, with shapes*: `W_x` is (hidden, inputs), `x` is (inputs,), `W_h` is (hidden, hidden), and `h` and `b` are (hidden,).",
                    nl="Met `{call}` stopt Python bij een `@` of een `+`: de shapes passen niet. Schrijf de shape naast elk deel, zoals in *De forward pass in NumPy, met shapes*: `W_x` is (hidden, inputs), `x` is (inputs,), `W_h` is (hidden, hidden), en `h` en `b` zijn (hidden,)."),
        ],
        digits=4,
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the two `...`: the hidden state to start with, and one step of the network for every input `x`.",
                      "nl": "Vervang de twee `...`: de hidden state om mee te beginnen, en één stap van het netwerk voor elke input `x`."},
            "arguments": {"en": "The check calls `{call}`, with the inputs, the two weight matrices and the bias terms, and the `def` line of your function has another number of names between its parentheses. Keep `(xs, W_x, W_h, b)`.",
                          "nl": "De check roept `{call}` aan, met de inputs, de twee weight-matrices en de bias terms, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(xs, W_x, W_h, b)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your two lines with steps 2 and 3 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je twee regels met stap 2 en 3 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Compare every part of your step with step 3 of the recipe: does your function use the weights, sizes and inputs it gets, not those of the example?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk elk deel van je stap met stap 3 van het recept: gebruikt je functie de weights, groottes en inputs die ze krijgt, en niet die van het voorbeeld?"},
            "correct": {"en": "Your `rnn` is right for the example and for networks with a hidden state of 1 to 3 numbers, reading 1 to 5 inputs: it reads a sequence of any length.",
                        "nl": "Je `rnn` klopt voor het voorbeeld en voor netwerken met een hidden state van 1 tot 3 getallen, die 1 tot 5 inputs lezen: hij leest een rij van elke lengte."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("rnn", _R.replace("\n    return h\n", "\n        return h\n"), "wrong", "return-in-loop"),
    ("rnn", _R_START, "empty", None),
    ("rnn", _r(start="    h = np.zeros(len(W_h))\n", step="        z = W_x @ x + W_h @ h + b\n        h = np.tanh(z)\n"), "correct", None),
    ("rnn", _r(start="    h = np.zeros_like(b)\n", step="        h = np.tanh(np.dot(W_x, x) + np.dot(W_h, h) + b)\n"), "correct", None),
    ("rnn", _r(step="        h = np.tanh(W_x @ x + b)\n"), "wrong", "no-memory"),
    ("rnn", _r(step="        h = np.zeros(len(b))\n        h = np.tanh(W_x @ x + W_h @ h + b)\n"), "wrong", "no-memory"),
    ("rnn", _r(step="        h = W_x @ x + W_h @ h + b\n"), "wrong", "no-tanh"),
    ("rnn", _r(step="        h = np.tanh(W_h @ x + W_x @ h + b)\n"), "wrong", "swapped"),
    ("rnn", _r(step="        h += np.tanh(W_x @ x + W_h @ h + b)\n"), "wrong", "adds-up"),
    ("rnn", _r(step="        h = np.tanh(W_x @ x + W_h @ h)\n"), "almost", "no-bias"),
    ("rnn", _r(step="        h = np.tanh(W_x @ x + W_h @ h) + b\n"), "almost", "bias-outside"),
    ("rnn", _r(step="        h = np.tanh(W_x @ x + W_h.T @ h + b)\n"), "almost", None),
    ("rnn", _r(start="    h = np.zeros(2)\n"), "error", "shapes"),
    ("rnn", _r(start="    h = np.zeros((len(b), len(b)))\n"), "wrong", None),
]
