# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-why-discrimination-is-wrong: how far apart two people with the same test score end up
when an employer pulls each estimate toward the average of their group, sorting claims by the account of
discrimination they belong to, and what philosophers say about a correct average; with the pictures of the
estimates."""
import html
import re

from everything_ai.check import CodeBox, Mistake, Sort
from everything_ai import yours

# Made-up numbers, as in Phelps's model of statistical discrimination (in the form of Aigner and Cain, 1977): the
# estimate of a person is (1 - trust) times the mean of their group plus trust times their own test score.
SCORE = 70
MEANS = {"A": 60, "B": 50}
EXAMPLE_TRUST = 0.5
TRUST = 0.2


def estimates(score: float, means: dict, trust: float) -> dict:
    """The estimate for a person of each group with this score."""
    found = {}
    for group in means:
        found[group] = (1 - trust) * means[group] + trust * score
    return found


def gap(score: float, means: dict, trust: float) -> float:
    e = estimates(score, means, trust)
    return e["A"] - e["B"]


def _number(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _line_picture(title: str, rows: list, labels: dict, alt: str, decimal: str) -> str:
    """Two rows on a scale from 40 to 80: the mean of each group as an open circle, the shared score as a tick, and
    the estimate as a filled dot with its value. `rows` holds (group, mean, estimate or None)."""
    from everything_ai import chart
    p = chart.Panel(x=(38, 82), y=(0, 3.2), xlabel=labels["xlabel"], yticks=[],
                    xticks=[40, 50, 60, 70, 80], xtick_labels=["40", "50", "60", "70", "80"])
    for i, (group, mean, estimate) in enumerate(rows):
        y = 2.3 - 1.2 * i
        p.text(38.6, y + 0.42, f'{labels["group"]} {group}', "ink-2", anchor="start")
        p.line([40, 80], [y, y], role="ink-3", width=1)
        p.points([mean], [y], role="ink", r=5, hollow=True)
        p.text(mean, y - 0.38, labels["mean"], "ink-3")
        p.line([SCORE, SCORE], [y - 0.16, y + 0.16], role="ink", width=2)
        p.text(SCORE, y - 0.38, labels["score"], "ink-3")
        if estimate is not None and 38 <= estimate <= 82:
            p.points([estimate], [y], role="ink", r=5)
            p.text(estimate, y + 0.32, f"{estimate:.1f}".replace(".", decimal), "ink")
    svg = chart.figure([p], alt=alt, width=520, height=250)
    return yours.figure(title, chart.with_decimal_comma(svg) if decimal == "," else svg)


def picture(labels: dict) -> str:
    """The estimates of the worked example. `labels` gives "title", "group", "mean", "score", "xlabel", "decimal"
    and "alt"."""
    e = estimates(SCORE, MEANS, EXAMPLE_TRUST)
    rows = [(g, MEANS[g], e[g]) for g in MEANS]
    return _line_picture(labels["title"], rows, labels, labels["alt"], labels["decimal"])


def _gap_picture(seen) -> str:
    """The learner's estimates per group on the same scale, with the gap they work out."""
    mine = seen.mine
    if mine is None or not mine.has_value or _number(mine.value) is None:
        return ""
    found = mine.names.get("estimates") if isinstance(mine.names, dict) else None
    rows = []
    for g in MEANS:
        v = _number(found.get(g)) if isinstance(found, dict) else None
        rows.append((g, MEANS[g], v))
    words = {"en": {"group": "group", "mean": "group mean", "score": "score", "xlabel": "estimate of the employer",
                    "title": "Your estimates, with the same score"},
             "nl": {"group": "groep", "mean": "groepsgemiddelde", "score": "score", "xlabel": "schatting van de werkgever",
                    "title": "Jouw schattingen, bij dezelfde score"}}[seen.lang]
    decimal = "," if seen.lang == "nl" else "."
    shown = f"gap = {_number(mine.value):.1f}"   # as Python writes it, with a point
    title = f'{words["title"]}: {shown}'
    return _line_picture(title, rows, words, title, decimal)


# The lesson is done when these are correct. "average" is the choice question in the lesson.
REQUIRED = ["gap", "accounts", "average"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _line(name: str, values) -> str:
    return f"{name} = {values!r}".replace("'", '"')


_SCORE_LINE = f"score = {SCORE}"
_MEANS_LINE = _line("means", MEANS)
_TRUST_LINE = f"trust = {TRUST}"
_HEAD = ("# made-up numbers: one test score, the mean of each group, and how much the employer trusts the test\n"
         + _SCORE_LINE + "\n" + _MEANS_LINE + "\n" + _TRUST_LINE + "\n\n")
_START = _HEAD + "...\ngap\n"
_BODY = ("estimates = {}\nfor group in means:\n    estimates[group] = (1 - trust) * means[group] + trust * score\n"
         'gap = estimates["A"] - estimates["B"]\n')
_SOLUTION = _HEAD + _BODY + "gap"


def _s(body: str) -> str:
    return _HEAD + body + "gap"


def _swap(old: str, new: str) -> str:
    assert _BODY.count(old) == 1
    return _s(_BODY.replace(old, new))


def _is(value: float):
    return lambda got, text: isinstance(got, (int, float)) and not isinstance(got, bool) and abs(got - value) < 1e-6


EXERCISES = {
    "gap": CodeBox(
        name="gap",
        picture=_gap_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_SCORE_LINE, _MEANS_LINE, _TRUST_LINE],
        # The check also runs the box with a test the employer trusts much more, and with other group means.
        others=[{"trust": 0.9}, {"means": {"A": 70, "B": 40}}],
        first=[_DOTS],
        digits=2,
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_means`. In the box the names have none: `score`, `means`, `trust`, `estimates`, `group`, `gap`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_means`. In het vak hebben de namen dat niet: `score`, `means`, `trust`, `estimates`, `group`, `gap`."),
            Mistake("key", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks a dictionary for a key it does not have. The groups are `\"A\"` and `\"B\"`, and `estimates` gets them only in the loop.",
                    nl="Je code vraagt een dictionary om een sleutel die hij niet heeft. De groepen zijn `\"A\"` en `\"B\"`, en `estimates` krijgt ze pas in de loop."),
            Mistake("same", matches=_is(0.0),
                    en="Your last line gives `0.0`: both people get the same estimate. In the model, the mean of the person's group counts too, with `(1 - trust)`.",
                    nl="Je laatste regel geeft `0.0`: beide mensen krijgen dezelfde schatting. In het model telt het gemiddelde van de groep van de persoon ook mee, met `(1 - trust)`."),
            Mistake("swapped", matches=_is(2.0),
                    en="Your last line gives `2.0`. You gave the weight `trust` to the group mean and `(1 - trust)` to the score: the other way round. The group mean gets `(1 - trust)`.",
                    nl="Je laatste regel geeft `2.0`. Je gaf het gewicht `trust` aan het groepsgemiddelde en `(1 - trust)` aan de score: andersom. Het groepsgemiddelde krijgt `(1 - trust)`."),
            Mistake("means", matches=_is(10.0),
                    en="Your last line gives `10.0`: the difference between the two group means. `gap` is the difference between the two estimates, where the score counts too.",
                    nl="Je laatste regel geeft `10.0`: het verschil tussen de twee groepsgemiddelden. `gap` is het verschil tussen de twee schattingen, waarin de score ook meetelt."),
            Mistake("order", matches=_is(-8.0),
                    en="Your last line gives `-8.0`: the right size, but the wrong way round. Take the estimate of `\"A\"` minus that of `\"B\"`.",
                    nl="Je laatste regel geeft `-8.0`: de goede grootte, maar andersom. Neem de schatting van `\"A\"` min die van `\"B\"`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the loop of the worked example, and then work out `gap`: the estimate of `\"A\"` minus that of `\"B\"`.",
                      "nl": "Vervang de `...` door de loop van het uitgewerkte voorbeeld, en reken daarna `gap` uit: de schatting van `\"A\"` min die van `\"B\"`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the lines of `score`, `means` and `trust` exactly as they were: put them back. The check tries other numbers by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `score`, `means` en `trust` nodig zoals ze waren: zet ze terug. De check probeert zelf andere getallen."},
            "no_value": {"en": "The last line of the box shows nothing. Put `gap` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `gap` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Is each estimate `(1 - trust)` times the group mean plus `trust` times the score, and is `gap` the estimate of `\"A\"` minus that of `\"B\"`?",
                      "nl": "Je laatste regel geeft `{got}`. Is elke schatting `(1 - trust)` keer het groepsgemiddelde plus `trust` keer de score, en is `gap` de schatting van `\"A\"` min die van `\"B\"`?"},
            "other": {"en": "Right for these numbers. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you use `score`, `means` and `trust` themselves, and not their numbers?",
                      "nl": "Goed voor deze getallen. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruik je `score`, `means` en `trust` zelf, en niet hun getallen?"},
            "other_error": {"en": "Right for these numbers. But with {where}, your code stops with an error. Do you use `score`, `means` and `trust` themselves?",
                            "nl": "Goed voor deze getallen. Maar met {where} stopt je code met een fout. Gebruik je `score`, `means` en `trust` zelf?"},
            "correct": {"en": "`8.0`: with a test it trusts little, the employer gives two people with the same score 62 and 54, only because of their group. With `trust = 0.5` it was 5 points; the less the test is trusted, the more the group decides.",
                        "nl": "`8.0`: met een test die de werkgever weinig vertrouwt, geeft die twee mensen met dezelfde score 62 en 54, alleen vanwege hun groep. Met `trust = 0.5` was het 5 punten; hoe minder de test wordt vertrouwd, hoe meer de groep beslist."},
        },
    ),
    "accounts": Sort(
        name="accounts",
        groups={
            "harm": {"en": "It harms (Lippert-Rasmussen)", "nl": "Het schaadt (Lippert-Rasmussen)"},
            "demean": {"en": "It demeans (Hellman)", "nl": "Het vernedert (Hellman)"},
            "respect": {"en": "It disrespects a person (Eidelson)", "nl": "Het respecteert de persoon niet (Eidelson)"},
        },
        items={
            "cost": {"en": "**The cost.** A sexist firing is wrong because of what it costs the person who is fired.",
                     "nl": "**De kosten.** Een seksistisch ontslag is fout vanwege wat het de ontslagen persoon kost."},
            "worse": {"en": "**The worse off.** Harm to those who are already worse off weighs more.",
                      "nl": "**Wie slechter af is.** Schade aan wie al slechter af is, weegt zwaarder."},
            "makeup": {"en": "**The makeup.** Requiring women at work to wear makeup says that their bodies are there to be looked at.",
                       "nl": "**De make-up.** Vrouwen op het werk verplichten om make-up te dragen, zegt dat hun lichaam er is om naar te kijken."},
            "power": {"en": "**The power.** It is wrong when someone with power over others expresses that a group has a lower status.",
                      "nl": "**De macht.** Het is fout als iemand met macht over anderen uitdrukt dat een groep een lagere status heeft."},
            "fire": {"en": "**The fire department.** Turning down every woman without a fitness test ignores what each applicant has made of their own life.",
                     "nl": "**De brandweer.** Elke vrouw afwijzen zonder conditietest negeert wat elke sollicitant van het eigen leven heeft gemaakt."},
            "author": {"en": "**The author of a life.** People are of equal worth and shape their own lives; a decision must respect both.",
                       "nl": "**Schrijver van het eigen leven.** Mensen zijn even veel waard en geven hun eigen leven vorm; een beslissing moet allebei respecteren."},
        },
        solution={"cost": "harm", "worse": "harm", "makeup": "demean", "power": "demean", "fire": "respect", "author": "respect"},
        nudge={
            "cost": {"en": "Is this about what the person loses, what the act says, or the person's own life?",
                     "nl": "Gaat dit over wat de persoon kwijtraakt, wat de daad zegt, of het eigen leven van de persoon?"},
            "worse": {"en": "Whose view weighs the worse off more heavily?", "nl": "Wiens opvatting laat wie slechter af is zwaarder wegen?"},
            "makeup": {"en": "Is this about a loss, or about what the rule says about women?", "nl": "Gaat dit over een verlies, of over wat de regel over vrouwen zegt?"},
            "power": {"en": "Who needs power for an act to be wrong in this view?", "nl": "Voor wie is macht nodig om een daad fout te maken in deze opvatting?"},
            "fire": {"en": "Is this about what the rule expresses, or about looking at each applicant?", "nl": "Gaat dit over wat de regel uitdrukt, of over kijken naar elke sollicitant?"},
            "author": {"en": "Equal worth and shaping your own life: whose two parts are these?", "nl": "Even veel waard en je eigen leven vormgeven: van wie zijn deze twee delen?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the claims again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de beweringen opnieuw uit."},
            "empty": {"en": "Sort each claim by the answer it belongs to.",
                      "nl": "Deel elke bewering in bij het antwoord waar ze bij hoort."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The claim *{item}* belongs somewhere else.",
                      "nl": "De bewering *{item}* hoort ergens anders."},
            "correct": {"en": "All {n} are right. The first answer looks at outcomes, as consequentialism does; the other two look at what an act says or how it treats a person, closer to deontology.",
                        "nl": "Alle {n} kloppen. Het eerste antwoord kijkt naar uitkomsten, zoals het consequentialisme; de andere twee kijken naar wat een daad zegt of hoe die een persoon behandelt, dichter bij de deontologie."},
        },
    ),
}

_SORTED = {"cost": "harm", "worse": "harm", "makeup": "demean", "power": "demean", "fire": "respect", "author": "respect"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("gap", _START, "empty", None),
    ("gap", _SOLUTION, "correct", None),
    ("gap", _s('a = (1 - trust) * means["A"] + trust * score\nb = (1 - trust) * means["B"] + trust * score\n'
               "gap = a - b\n"), "correct", None),
    ("gap", _s(_BODY.replace("in means", "in _means")), "error", "underscore"),
    ("gap", _swap('estimates["A"] - estimates["B"]', 'estimates["a"] - estimates["b"]'), "error", "key"),
    ("gap", _swap("(1 - trust) * means[group] + trust * score", "score"), "wrong", "same"),
    ("gap", _swap("(1 - trust) * means[group] + trust * score", "trust * means[group] + (1 - trust) * score"),
     "wrong", "swapped"),
    ('gap', _s('gap = means["A"] - means["B"]\n'), "wrong", "means"),
    ("gap", _swap('estimates["A"] - estimates["B"]', 'estimates["B"] - estimates["A"]'), "wrong", "order"),
    ("gap", _swap("(1 - trust) * means[group] + trust * score", "0.5 * means[group] + 0.5 * score"), "wrong", None),
    ("gap", _s("gap = 8.0\n"), "almost", "other-value"),
    ("gap", _swap("(1 - trust) * means[group] + trust * score", "0.8 * means[group] + 0.2 * score"),
     "almost", "other-value"),
    ("gap", _SOLUTION.replace("trust = 0.2", "trust = 0.5"), "wrong", "changed-line"),
    ("gap", _s("...\ngap = 8.0\n"), "wrong", "dots-left"),
    ("accounts", {}, "empty", None),
    ("accounts", _SORTED, "correct", None),
    ("accounts", {"cost": "harm", "fire": "respect"}, "wrong", "unfinished"),
    ("accounts", {**_SORTED, "power": "respect"}, "wrong", "wrong:power"),
    ("accounts", {**_SORTED, "fire": "demean"}, "wrong", "wrong:fire"),
    ("accounts", {**_SORTED, "worse": "respect"}, "wrong", "wrong:worse"),
]
