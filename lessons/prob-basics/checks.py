# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson prob-basics: a distribution from counts, as a function, and a probability estimated from
500 throws of a die. And the picture: the share of each face after more and more throws."""
import random
import re

from everything_ai.check import CodeBox, FunctionBox, Mistake, same_value

# 2000 throws of a fair die, always the same ones: the picture shows the first n of them.
_rng = random.Random(41)
THROWS = [_rng.randint(1, 6) for _ in range(2000)]


def shares(n: int) -> list[float]:
    """The share of each face, 1 to 6, in the first n throws."""
    first = THROWS[:n]
    return [first.count(face) / n for face in range(1, 7)]


# The smallest and the largest share after more and more throws: the second chart of the picture. Every 10 throws up to
# 200, where they still jump, then every 40, which is as smooth at this size and keeps the frames of L.explore small.
_STOPS = list(range(10, 200, 10)) + list(range(200, 2001, 40))
_LOWEST = [min(shares(n)) for n in _STOPS]
_HIGHEST = [max(shares(n)) for n in _STOPS]


def picture(n: int, labels: dict[str, str]) -> str:
    """A bar per face, as high as its share in the first n throws, and a dashed line at 1/6. Beside it, the smallest
    and the largest of the six shares for every number of throws, with a mark at n: the two lines close in on 1/6, so
    you see what more throws do without moving the slider. Drawn as SVG text, for every stop of the slider once
    (L.explore); dragging along the second chart moves the slider."""
    from everything_ai import chart
    now = shares(n)
    bars = chart.Panel((0.4, 6.6), (0, 0.4), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"], xticks=[1, 2, 3, 4, 5, 6])
    for face, share in zip(range(1, 7), now):
        bars.rect(face - 0.32, 0, face + 0.32, share, "data")
    bars.line([0.4, 6.6], [1 / 6, 1 / 6], "ink-3", 1.4, dashed=True)
    spread = chart.Panel((0, 2000), (0, 0.4), title=labels["title_spread"], xlabel=labels["x_spread"], ylabel=labels["y"])
    spread.line([0, 2000], [1 / 6, 1 / 6], "ink-3", 1.4, dashed=True)
    spread.line(_STOPS, _HIGHEST, "data", 1.6)
    spread.line(_STOPS, _LOWEST, "data", 1.6)
    spread.line([n, n], [0, 0.4], "ink-3", 1)
    spread.points([n, n], [min(now), max(now)], "data")
    return chart.figure([bars, spread], alt=labels["alt"], width=330, height=260, drag=1)   # two side by side: 330 px each


# The lesson is done when these are correct. "marbles" is the choice question in the lesson.
REQUIRED = ["marbles", "distribution", "estimate"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_D_START = "def distribution(counts):\n    ...\n"
_D = ("def distribution(counts):\n    total = 0\n    for outcome in counts:\n        total = total + counts[outcome]\n"
      "    probs = {}\n    for outcome in counts:\n        probs[outcome] = counts[outcome] / total\n    return probs")


def _d(new: str) -> str:
    return "def distribution(counts):\n" + new


_LOOP_TOTAL = "    total = 0\n    for outcome in counts:\n        total = total + counts[outcome]\n    probs = {}\n    for outcome in counts:\n"

_E_HEAD = ("import random\n\nseed = 8\nrandom.seed(seed)\nthrows = []\nfor i in range(500):\n"
           "    throws.append(random.randint(1, 6))\n\n")
_E_START = _E_HEAD + "p_six = ...\np_six\n"
_E = _E_HEAD + "sixes = 0\nfor t in throws:\n    if t == 6:\n        sixes = sixes + 1\np_six = sixes / len(throws)\np_six"
_SIXES = "sixes = 0\nfor t in throws:\n    if t == 6:\n        sixes = sixes + 1\n"


def _e(new: str) -> str:
    return _E_HEAD + new + "\np_six"


def _same_distribution(got, want) -> bool:
    """The same outcomes with the same probabilities, within the usual margin: 3 * (1 / 10) is 0.30000000000000004."""
    return (isinstance(got, dict) and isinstance(want, dict) and got.keys() == want.keys()
            and all(same_value(got[k], want[k]) for k in want))


def _near(got, want: float, margin: float = 1e-9) -> bool:
    try:
        return abs(float(got) - want) < margin
    except (TypeError, ValueError):
        return False


EXERCISES = {
    "distribution": FunctionBox(
        name="distribution",
        function="distribution",
        start=_D_START,
        solution=_D,
        compare=_same_distribution,
        visible=[({"heads": 7, "tails": 3},)],
        hidden=[({"rain": 2, "dry": 6},), ({"a": 5},), ({"x": 1, "y": 1, "z": 2},), ({"red": 3, "blue": 5, "green": 2},)],
        first=[_DOTS],
        mistakes=[
            Mistake("unchanged", wrong=_d("    return counts"),
                    en="`{call}` gives `{got}`: the counts themselves. A probability is a share: divide each count by the total number of tries.",
                    nl="`{call}` geeft `{got}`: de tellingen zelf. Een kans is een deel: deel elke telling door het totale aantal pogingen."),
            Mistake("by-outcomes", wrong=_d("    probs = {}\n    for outcome in counts:\n        probs[outcome] = counts[outcome] / len(counts)\n    return probs"),
                    en="`{call}` gives `{got}`: you divide by the number of outcomes, `len(counts)`. Divide by the number of tries: all the counts added up.",
                    nl="`{call}` geeft `{got}`: je deelt door het aantal uitkomsten, `len(counts)`. Deel door het aantal pogingen: alle tellingen opgeteld."),
            Mistake("percent", wrong=_d(_LOOP_TOTAL + "        probs[outcome] = counts[outcome] / total * 100\n    return probs"),
                    en="`{call}` gives `{got}`: percentages. A probability is a number from 0 to 1: leave out the `* 100`.",
                    nl="`{call}` geeft `{got}`: procenten. Een kans is een getal van 0 tot en met 1: laat de `* 100` weg."),
            Mistake("by-max", wrong=_d("    biggest = 0\n    for outcome in counts:\n        if counts[outcome] > biggest:\n            biggest = counts[outcome]\n"
                                       "    probs = {}\n    for outcome in counts:\n        probs[outcome] = counts[outcome] / biggest\n    return probs"),
                    en="`{call}` gives `{got}`: you divide by the biggest count, so the probabilities add up to more than 1. Divide by all the counts added up.",
                    nl="`{call}` geeft `{got}`: je deelt door de grootste telling, dus de kansen zijn samen meer dan 1. Deel door alle tellingen opgeteld."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a dictionary with the probability of each outcome, given back with `return`.",
                      "nl": "Vervang de `...` door een dictionary met de kans op elke uitkomst, teruggegeven met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with one dictionary, and the `def` line of your function has a different number of names between its parentheses. Keep `(counts)`.",
                          "nl": "De check roept `{call}` aan, met één dictionary, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(counts)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you add up all the counts first, and then divide each count by that total?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Tel je eerst alle tellingen op, en deel je dan elke telling door dat totaal?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Do you work it out from `counts` itself, for any outcomes?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Reken je het uit met `counts` zelf, voor elke uitkomst?"},
            "correct": {"en": "Your function is right in all {n} cases. 7 heads and 3 tails are 10 tries, so P(heads) = 0.7 and P(tails) = 0.3: together 1, as in every distribution.",
                        "nl": "Je functie klopt in alle {n} gevallen. 7 keer kop en 3 keer munt zijn 10 pogingen, dus P(kop) = 0,7 en P(munt) = 0,3: samen 1, zoals bij elke kansverdeling."},
        },
    ),
    "estimate": CodeBox(
        name="estimate",
        start=_E_START,
        solution=_E,
        keep=["seed = 8", "random.seed(seed)", "for i in range(500):", "    throws.append(random.randint(1, 6))"],
        # The check also runs the box with another seed: other throws, so a number typed in by hand shows up.
        others=[{"seed": 2}],
        first=[_DOTS],
        mistakes=[
            Mistake("theory", matches=lambda got, text: _near(got, 1 / 6, 0.0006),
                    en="Your last line gives `{got}`: 1/6, the probability of a fair die. The task asks for the share of sixes in these 500 throws: count them.",
                    nl="Je laatste regel geeft `{got}`: 1/6, de kans bij een eerlijke dobbelsteen. De opdracht vraagt het deel zessen in deze 500 worpen: tel ze."),
            Mistake("count-only", wrong=_e(_SIXES + "p_six = sixes"),
                    en="Your last line gives `{got}`: the number of sixes. A probability is a share: divide by the number of throws.",
                    nl="Je laatste regel geeft `{got}`: het aantal zessen. Een kans is een deel: deel door het aantal worpen."),
            Mistake("by-faces", wrong=_e(_SIXES + "p_six = sixes / 6"),
                    en="Your last line gives `{got}`: you divide by 6, the number of faces. Divide by the number of tries: `len(throws)`.",
                    nl="Je laatste regel geeft `{got}`: je deelt door 6, het aantal kanten. Deel door het aantal pogingen: `len(throws)`."),
            Mistake("percent", wrong=_e(_SIXES + "p_six = sixes / len(throws) * 100"),
                    en="Your last line gives `{got}`: a percentage. A probability is a number from 0 to 1: leave out the `* 100`.",
                    nl="Je laatste regel geeft `{got}`: een percentage. Een kans is een getal van 0 tot en met 1: laat de `* 100` weg."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the share of sixes in `throws`: count them, and divide by the number of throws.",
                      "nl": "Vervang de `...` door het deel zessen in `throws`: tel ze, en deel door het aantal worpen."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another seed by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere seed."},
            "no_value": {"en": "The last line of the box shows nothing. Put `p_six` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `p_six` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Do you count the throws that are a 6, and divide by the number of throws?",
                      "nl": "Je laatste regel geeft `{got}`. Tel je de worpen die een 6 zijn, en deel je door het aantal worpen?"},
            "other": {"en": "Right for these throws. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you count in `throws` itself?",
                      "nl": "Goed voor deze worpen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Tel je in `throws` zelf?"},
            "other_error": {"en": "Right for these throws. But with {where}, your code stops with an error. Do you count in `throws` itself?",
                            "nl": "Goed voor deze worpen. Maar met {where} stopt je code met een fout. Tel je in `throws` zelf?"},
            "correct": {"en": "80 sixes out of 500: 0.16, close to 1/6, about 0.167. With other throws, seed 2, it was 0.182. An estimate from tries wobbles; with more throws it wobbles less.",
                        "nl": "80 zessen van de 500: 0,16, dicht bij 1/6, ongeveer 0,167. Met andere worpen, seed 2, was het 0,182. Een schatting uit pogingen schommelt; met meer worpen schommelt hij minder."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("distribution", _D_START, "empty", None),
    ("distribution", _D, "correct", None),
    ("distribution", _d("    total = 0\n    for k in counts:\n        total += counts[k]\n    return {k: counts[k] / total for k in counts}"), "correct", None),
    ("distribution", _d("    total = sum(counts.values())\n    return {k: v / total for k, v in counts.items()}"), "correct", None),
    ("distribution", _d("    total = 0\n    for k in counts:\n        total = total + counts[k]\n    return {k: counts[k] * (1 / total) for k in counts}"), "correct", None),
    ("distribution", _d("    return counts"), "wrong", "unchanged"),
    ("distribution", _d("    probs = {}\n    for outcome in counts:\n        probs[outcome] = counts[outcome] / len(counts)\n    return probs"), "wrong", "by-outcomes"),
    ("distribution", _d(_LOOP_TOTAL + "        probs[outcome] = counts[outcome] / total * 100\n    return probs"), "wrong", "percent"),
    ("distribution", _d("    return {\"heads\": 0.7, \"tails\": 0.3}"), "almost", None),
    ("estimate", _E_START, "empty", None),
    ("estimate", _E, "correct", None),
    ("estimate", _e("p_six = throws.count(6) / len(throws)"), "correct", None),
    ("estimate", _e(_SIXES + "p_six = sixes / 500"), "correct", None),
    ("estimate", _e("p_six = 1 / 6"), "wrong", "theory"),
    ("estimate", _e(_SIXES + "p_six = sixes"), "wrong", "count-only"),
    ("estimate", _e(_SIXES + "p_six = sixes / 6"), "wrong", "by-faces"),
    ("estimate", _e(_SIXES + "p_six = sixes / len(throws) * 100"), "wrong", "percent"),
    ("estimate", _e("p_six = 0.16"), "almost", "other-value"),
    ("estimate", _E.replace("seed = 8", "seed = 3"), "wrong", "changed-line"),
    ("estimate", _E_HEAD + _SIXES + "p_six = sixes / len(throws)\nprint(p_six)", "wrong", "print-last"),
]
