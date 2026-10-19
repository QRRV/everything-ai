# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson lm-temperature: with_temperature, which divides the logits of a row by a temperature before
softmax. And the pictures: a row of the table of the plays at the temperature of a slider, and what the model then
writes, drawn once for every stop of the slider (L.explore)."""
import html
import re
from functools import cache

import numpy as np

from everything_ai import paths
from everything_ai.check import FunctionBox, Mistake

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


def with_temp(p, tau: float) -> np.ndarray:
    """The course's way: the logits np.log(p), divided by tau, through softmax. A probability of 0 stays 0."""
    with np.errstate(divide="ignore"):
        z = np.log(np.asarray(p, dtype=float)) / tau
    e = np.exp(z - z.max())
    return e / e.sum()


def shown(ch: str) -> str:
    return {" ": "␣", "\n": "↵"}.get(ch, ch)


def after(ch: str, tau: float, n: int = 6) -> list[tuple[str, float]]:
    """The n most likely characters after ch at temperature tau, for a picture."""
    cs = chars()
    q = with_temp(course_probs()[cs.index(ch)], tau)
    return [(shown(cs[j]), float(q[j])) for j in np.argsort(-q, kind="stable")[:n]]


def prob_after(ch: str, nxt: str, tau: float) -> float:
    cs = chars()
    return float(with_temp(course_probs()[cs.index(ch)], tau)[cs.index(nxt)])


def bars(ch: str, tau: float, labels: dict, n: int = 6) -> str:
    """The n most likely characters after ch at temperature tau, as bars on a fixed scale: a full bar is probability 1,
    so that a bar grows and shrinks when the slider moves. The bars are the model's probabilities, so they are pred."""
    W, LW, ROW, BW, TOP = 520, 40, 28, 380, 30
    p = [f'<text x="0" y="16" style="font-size:15px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    for i, (c, share) in enumerate(after(ch, tau, n)):
        y = TOP + i * ROW
        w = max(BW * share, 2)
        p.append(f'<text x="{LW - 10}" y="{y + 15}" style="font-size:15px;font-weight:600;fill:var(--ea-ink);text-anchor:end">{html.escape(c)}</text>'
                 f'<rect x="{LW}" y="{y + 2}" width="{BW}" height="18" rx="3" style="fill:var(--ea-ink-3);fill-opacity:0.12"/>'
                 f'<rect x="{LW}" y="{y + 2}" width="{w:.1f}" height="18" rx="3" style="fill:var(--ea-pred)"/>'
                 f'<text x="{LW + w + 8:.1f}" y="{y + 16}" style="font-size:13px;fill:var(--ea-pred-text)">{round(share * 100)}%</text>')
    H = TOP + n * ROW + 4
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {H}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


def text_block(s: str) -> str:
    """A text the model wrote, in a box that wraps its lines, so that a long line stays readable on a phone."""
    return (f'<pre style="white-space:pre-wrap;word-break:break-word;margin:0.25rem 0 0.75rem;padding:0.6rem 0.8rem;'
            f'border-radius:6px;background:var(--ea-inset);color:var(--ea-ink);font-family:var(--ea-font-code);'
            f'font-size:0.875rem;line-height:1.45">{html.escape(s)}</pre>')


def possible_after(ch: str) -> int:
    """How many characters ever came after ch: the ones with a probability above 0, at every temperature."""
    cs = chars()
    return int((course_probs()[cs.index(ch)] > 0).sum())


def sample_text(tau: float, seed: int = 1, n: int = 160, start: str = "T") -> str:
    """What the model of the plays writes at temperature tau: sampling, with the table's rows at that temperature."""
    P, cs = course_probs(), chars()
    row_of = {c: k for k, c in enumerate(cs)}
    rng = np.random.default_rng(seed)
    out = start
    while len(out) < n:
        out = out + cs[int(rng.choice(len(cs), p=with_temp(P[row_of[out[-1]]], tau)))]
    return out


def most_likely(s: str) -> int:
    """How many characters of s after the first are the most likely one after the character before them, in the
    table of the plays: a tie for the most likely counts too."""
    P, cs = course_probs(), chars()
    row_of = {c: k for k, c in enumerate(cs)}
    return sum(int(P[row_of[a]][row_of[b]] == P[row_of[a]].max()) for a, b in zip(s, s[1:]))


# The lesson is done when these are correct. "pick-tau" and "never-seen" are the choice questions in the lesson.
REQUIRED = ["pick-tau", "with-temperature", "never-seen"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


class _Vector(np.ndarray):
    """Probabilities that a check message writes the way the lesson's code does: np.array([...])."""

    def __repr__(self):
        a = np.asarray(self)
        if a.ndim != 1:
            return repr(a)
        return f"np.array({[int(x) if x == int(x) else float(x) for x in a.tolist()]!r})"

    def __deepcopy__(self, memo):
        return np.array(self)


def _vec(values) -> np.ndarray:
    return np.array(values, dtype=float).view(_Vector)


def _rounded(v) -> str:
    """Probabilities for the messages, rounded to 3 decimals, so that no float noise shows."""
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return repr(v)
    if a.ndim == 0:
        return repr(round(float(a), 3) + 0.0)
    if a.ndim != 1:
        return repr(v)
    return repr([round(float(x), 3) + 0.0 for x in a])


_HEAD = "import numpy as np\n\n\ndef with_temperature(p, tau):\n    z = np.log(p)\n"
_END = "    return e / e.sum()\n"
_START = _HEAD + "    ...\n" + _END


def _w(body: str = "    z = z / tau\n    e = np.exp(z - z.max())\n") -> str:
    return _HEAD + body + _END


_W = _w()


def _quiet_compare(got, want) -> bool:
    try:
        return np.allclose(np.asarray(got, dtype=float), np.asarray(want, dtype=float), rtol=1e-6, atol=1e-9)
    except (TypeError, ValueError):
        return False


EXERCISES = {
    "with-temperature": FunctionBox(
        name="with-temperature",
        function="with_temperature",
        start=_START,
        solution=_W,
        visible=[(_vec([0.6, 0.3, 0.1]), 0.5)],
        hidden=[(_vec([0.5, 0.3, 0.2]), 2), (_vec([0.5, 0.3, 0.2]), 0.5), (_vec([0.6, 0.4]), 1), (_vec([0.7, 0.2, 0.1, 0]), 0.5), (_vec([0.25, 0.25, 0.25, 0.25]), 0.3)],
        show=_rounded,
        compare=_quiet_compare,
        first=[_DOTS],
        mistakes=[
            Mistake("times", wrong=_w("    z = z * tau\n    e = np.exp(z - z.max())\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you multiply the logits by the temperature. A temperature below 1 should make the likely characters more likely: divide by `tau`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je vermenigvuldigt de logits met de temperature. Een temperature onder 1 moet de waarschijnlijke tekens waarschijnlijker maken: deel door `tau`."),
            Mistake("unchanged", wrong=_w("    e = np.exp(z - z.max())\n"),
                    en="`{call}` gives `{got}`: the same probabilities as `p`, so the temperature did nothing. Divide the logits `z` by `tau`, before `np.exp`.",
                    nl="`{call}` geeft `{got}`: dezelfde kansen als `p`, dus de temperature deed niets. Deel de logits `z` door `tau`, vóór `np.exp`."),
            Mistake("probs", wrong=_w("    z = p / tau\n    e = np.exp(z - z.max())\n"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: you divide the probabilities `p`, not the logits. The logits are `z = np.log(p)`: divide those by `tau`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je deelt de kansen `p`, niet de logits. De logits zijn `z = np.log(p)`: deel die door `tau`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the `...` with steps 2 and 3 of the recipe: divide the logits by `tau`, and take `e` to the power of each, with the largest taken off first.",
                      "nl": "Vervang de `...` door stap 2 en 3 van het recept: deel de logits door `tau`, en neem `e` tot de macht van elk, met de grootste er eerst af."},
            "arguments": {"en": "The check calls `{call}`, with probabilities and a temperature, and the `def` line of your function has another number of names between its parentheses. Keep `(p, tau)`.",
                          "nl": "De check roept `{call}` aan, met kansen en een temperature, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(p, tau)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your two lines with steps 2 and 3 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je twee regels met stap 2 en 3 van het recept."},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does your function work for every temperature, and for a probability of 0?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt je functie voor elke temperature, en voor een kans van 0?"},
            "correct": {"en": "Your function is right in all {n} cases: below 1 it makes the likely characters more likely, and above 1 it evens the probabilities out.",
                        "nl": "Je functie klopt in alle {n} gevallen: onder 1 maakt hij de waarschijnlijke tekens waarschijnlijker, en boven 1 trekt hij de kansen gelijker."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("with-temperature", _START, "empty", None),
    ("with-temperature", _w("    z = z / tau\n    e = np.exp(z)\n"), "correct", None),
    ("with-temperature", _w("    e = np.exp(z / tau - (z / tau).max())\n"), "correct", None),
    ("with-temperature", _w("    z = z * tau\n    e = np.exp(z - z.max())\n"), "wrong", "times"),
    ("with-temperature", _w("    e = np.exp(z - z.max())\n"), "wrong", "unchanged"),
    ("with-temperature", _w("    e = p / tau\n"), "wrong", "unchanged"),
    ("with-temperature", _w("    z = p / tau\n    e = np.exp(z - z.max())\n"), "wrong", "probs"),
]
