# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson prob-bayes: bayes, P(A | B) from the prior P(A) and the two conditional probabilities of the
evidence B, with and without A. And the picture: the positive tests of 100,000 people, split into ill and healthy."""
import html
import math
import re

from everything_ai.check import FunctionBox, Mistake

# The rare illness of the guess: 1 in 1,000 people has it; the test is positive for 99% of the ill and 1% of the healthy.
PRIOR, HIT, FALSE_ALARM = 0.001, 0.99, 0.01


def bayes(p_a: float, p_b_given_a: float, p_b_given_not_a: float) -> float:
    """The course's own Bayes' rule, with P(B) from the law of total probability."""
    p_b = p_b_given_a * p_a + p_b_given_not_a * (1 - p_a)
    return p_b_given_a * p_a / p_b


def tested(people: int = 100_000) -> tuple[int, int, int, int]:
    """Of `people` people: how many are ill, how many of them test positive, how many are healthy, and how many of
    those test positive, all rounded to whole people."""
    ill = round(people * PRIOR)
    healthy = people - ill
    return ill, round(ill * HIT), healthy, round(healthy * FALSE_ALARM)


def count(n: int, lang: str) -> str:
    """A whole number for the text and the picture, with the thousands separator of the language: 1,098 or 1.098."""
    return f"{n:,}".replace(",", ".") if lang == "nl" else f"{n:,}"


# --- the picture: the positive tests as one bar, split into ill and healthy ---------------------------------------------

def picture(labels: dict[str, str], lang: str) -> str:
    """The positive tests of 100,000 people as one bar of 100%: the share from ill people (filled) and from healthy
    people (open). Two groups, so no colour: filled and open in data grey (style guide 1.7)."""
    _, true_pos, _, false_pos = tested()
    total = true_pos + false_pos
    W, x0, y0, bar_h = 600, 0, 40, 46
    w_ill = W * true_pos / total
    num = lambda n: count(n, lang)
    pct = lambda n: f"{round(100 * n / total)}%"
    p = [f'<text x="0" y="18" style="font-size:16px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>',
         f'<rect x="{x0}" y="{y0}" width="{w_ill:.1f}" height="{bar_h}" style="fill:var(--ea-data)"/>',
         f'<rect x="{x0 + w_ill + 0.75:.2f}" y="{y0 + 0.75}" width="{W - w_ill - 1.5:.2f}" height="{bar_h - 1.5}" '
         f'style="fill:none;stroke:var(--ea-data);stroke-width:1.5"/>',
         f'<text x="{x0}" y="{y0 + bar_h + 24}" style="font-size:14px;fill:var(--ea-ink)">'
         f'{html.escape(labels["ill"])}: {num(true_pos)} ({pct(true_pos)})</text>',
         f'<text x="{W}" y="{y0 + bar_h + 24}" style="font-size:14px;fill:var(--ea-ink);text-anchor:end">'
         f'{html.escape(labels["healthy"])}: {num(false_pos)} ({pct(false_pos)})</text>',
         f'<text x="{W / 2}" y="{y0 + bar_h / 2 + 5}" style="font-size:14px;fill:var(--ea-ink);text-anchor:middle;font-weight:600">'
         f'{html.escape(labels["total"].format(n=num(total)))}</text>']
    height = y0 + bar_h + 34
    return (f'<figure class="ea-figure" style="margin:0.25rem 0 0.75rem;max-width:{W}px">'
            f'<svg viewBox="0 0 {W} {height}" role="img" style="width:100%;height:auto;font-family:var(--ea-font-ui)">'
            f'<title>{html.escape(labels["alt"])}</title>{"".join(p)}</svg></figure>')


# The lesson is done when these are correct. "no-news" and "second-test" are choice questions in the lesson.
REQUIRED = ["no-news", "bayes", "second-test"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _short(v):
    """A probability for the messages: four significant digits, without float noise or e-notation, so that 0.00099
    does not show as 0.001 and 0.0902 not as the 0.0902 it should be; whole numbers without .0."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return repr(v)
    if x == 0 or not math.isfinite(x):
        return "0" if x == 0 else repr(x)
    text = f"{x:.{max(0, 3 - math.floor(math.log10(abs(x))))}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


# --- exercise 2: bayes --------------------------------------------------------------------------------------------------

_B_HEAD = "def bayes(p_a, p_b_given_a, p_b_given_not_a):\n"
_B_START = _B_HEAD + "    p_not_a = 1 - p_a\n    p_b = ...\n    return ...\n"


def _b(p_b: str = "p_b_given_a * p_a + p_b_given_not_a * p_not_a", ret: str = "p_b_given_a * p_a / p_b") -> str:
    return _B_HEAD + f"    p_not_a = 1 - p_a\n    p_b = {p_b}\n    return {ret}\n"


_B = _b()

EXERCISES = {
    "bayes": FunctionBox(
        name="bayes",
        function="bayes",
        start=_B_START,
        solution=_B,
        visible=[(0.001, 0.99, 0.01)],
        hidden=[(0.2, 0.6, 0.05), (0.5, 0.8, 0.2), (0.3, 1, 0), (0.1, 0.5, 0.5), (0.9, 0.2, 0.6)],
        show=_short,
        first=[_DOTS],
        mistakes=[
            Mistake("no-prior", wrong=_b(ret="p_b_given_a / p_b"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the prior is missing on top. Bayes' rule multiplies P(B | A) by P(A) before it divides by P(B).",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de prior ontbreekt bovenaan. De regel van Bayes vermenigvuldigt P(B | A) met P(A) voordat hij door P(B) deelt."),
            Mistake("not-divided", wrong=_b(ret="p_b_given_a * p_a"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: that is P(A and B), how often A and B happen together. Divide it by `p_b`, so that you count among the cases where B happened.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: dat is P(A en B), hoe vaak A en B samen gebeuren. Deel het door `p_b`, zodat je telt onder de gevallen waarin B gebeurde."),
            # Also what code gives that leaves the prior out everywhere: p_b_given_a / (p_b_given_a + p_b_given_not_a).
            Mistake("prior-drops-out", wrong=_b(p_b="p_b_given_a * p_a + p_b_given_not_a * p_a"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: the prior drops out of your answer, as if A happened just as often as not A. That is the base-rate fallacy. Each way B can happen counts with how often its case happens: `p_b_given_a * p_a`, plus `p_b_given_not_a * p_not_a`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: de prior valt uit je antwoord weg, alsof A even vaak gebeurt als niet A. Dat is de base-rate fallacy. Elke weg waarlangs B kan gebeuren, telt mee met hoe vaak zijn geval voorkomt: `p_b_given_a * p_a`, plus `p_b_given_not_a * p_not_a`."),
            Mistake("plain-sum", wrong=_b(p_b="p_b_given_a + p_b_given_not_a"),
                    en="`{call}` gives `{got}`, and it should give `{want}`. Your `p_b` adds the two conditional probabilities as they are. Weigh each way B can happen by how often its case happens: `p_b_given_a * p_a`, plus `p_b_given_not_a * p_not_a`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn. Je `p_b` telt de twee voorwaardelijke kansen op zoals ze zijn. Weeg elke weg waarlangs B kan gebeuren met hoe vaak zijn geval voorkomt: `p_b_given_a * p_a`, plus `p_b_given_not_a * p_not_a`."),
            Mistake("only-a", wrong=_b(p_b="p_b_given_a * p_a"),
                    en="`{call}` gives `{got}`, and it should give `{want}`: your `p_b` only counts B together with A, so you divide P(A and B) by itself. B also happens without A: add `p_b_given_not_a * p_not_a`.",
                    nl="`{call}` geeft `{got}`, en dat moet `{want}` zijn: je `p_b` telt alleen B samen met A, dus deel je P(A en B) door zichzelf. B gebeurt ook zonder A: tel er `p_b_given_not_a * p_not_a` bij op."),
        ],
        digits=4,
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "empty": {"en": "Replace the two `...`: `p_b` with the law of total probability, and then Bayes' rule.",
                      "nl": "Vervang de twee `...`: `p_b` met de wet van de totale kans, en dan de regel van Bayes."},
            "arguments": {"en": "The check calls `{call}`, with three probabilities, and the `def` line of your function has another number of names between its parentheses. Keep `(p_a, p_b_given_a, p_b_given_not_a)`.",
                          "nl": "De check roept `{call}` aan, met drie kansen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(p_a, p_b_given_a, p_b_given_not_a)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Compare your two lines with steps 3 and 4 of the recipe.",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Vergelijk je twee regels met stap 3 en 4 van het recept."},
            "almost": {"en": "Right for the test of the guess. But `{call}` gives `{got}`, and it should give `{want}`. Does your function use its three arguments, and not the numbers of the test?",
                       "nl": "Goed voor de test uit de gok. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Gebruikt je functie haar drie argumenten, en niet de getallen van de test?"},
            "correct": {"en": "Your `bayes` is right in all {n} cases: about 0.09 for the rare illness of the guess, and 0.75 for the spam of the worked example.",
                        "nl": "Je `bayes` klopt in alle {n} gevallen: ongeveer 0,09 voor de zeldzame ziekte uit de gok, en 0,75 voor de spam uit het uitgewerkte voorbeeld."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("bayes", _B_START, "empty", None),
    ("bayes", _B_HEAD + "    p_b = p_b_given_a * p_a + p_b_given_not_a * (1 - p_a)\n    return p_a * p_b_given_a / p_b\n", "correct", None),
    ("bayes", _b(ret="p_b_given_a / p_b"), "wrong", "no-prior"),
    ("bayes", _b(ret="p_b_given_a * p_a"), "wrong", "not-divided"),
    ("bayes", _b(p_b="p_b_given_a * p_a + p_b_given_not_a * p_a"), "wrong", "prior-drops-out"),
    ("bayes", _B_HEAD + "    p_b = p_b_given_a + p_b_given_not_a\n    return p_b_given_a / p_b\n", "wrong", "prior-drops-out"),
    ("bayes", _b(p_b="p_b_given_a + p_b_given_not_a"), "wrong", "plain-sum"),
    ("bayes", _b(p_b="p_b_given_a * p_a"), "wrong", "only-a"),
    ("bayes", _b(ret="p_b_given_not_a * p_not_a / p_b"), "wrong", None),
    ("bayes", _B_HEAD + "    return 0.0902\n", "wrong", None),
]
