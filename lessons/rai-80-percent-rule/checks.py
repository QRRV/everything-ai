# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-80-percent-rule: the impact ratios of a neutral rule about working hours on the UCI Adult
table, which groups the four-fifths rule flags in the agencies' own example, and what passing the rule tells you;
with the pictures of the selection rates."""
import functools
import html
import re

from everything_ai.check import CodeBox, Mistake
from everything_ai import paths, yours

DATA = paths.ROOT / "data" / "adult.csv.gz"
HOURS = 45            # the made-up rule of the exercise: yes for at least this many hours a week
OTHER_HOURS = 40      # the other value the check tries
# The example of the questions and answers of the US agencies (1979), number 12: applicants and hires per group.
APPLICANTS = {"White": 80, "Black": 40}
HIRED = {"White": 48, "Black": 12}
# Number 11 of the same: hiring rates in percent.
EXAMPLE_RATES = {"White": 60, "American Indian": 45, "Hispanic": 48, "Black": 51}


def impact_ratios(rates: dict) -> dict:
    """Each selection rate divided by the highest one."""
    highest = max(rates.values())
    return {group: rate / highest for group, rate in rates.items()}


def flagged(rates: dict) -> list:
    """The groups whose impact ratio is below four-fifths: the signal of the rule."""
    return [group for group, ratio in impact_ratios(rates).items() if ratio < 0.8 - 1e-12]


@functools.cache
def _people():
    import pandas as pd
    return pd.read_csv(DATA)


def rates_by_sex(condition: str) -> dict:
    """The selection rate per sex on the UCI Adult table: for "label" a yes for earning more than 50,000 dollars, for
    "hours" a yes for working at least HOURS hours a week."""
    people = _people()
    picked = people[people["income"] == ">50K"] if condition == "label" else people[people["hours_per_week"] >= HOURS]
    rates = picked.groupby("sex").size() / people.groupby("sex").size()
    return {k: float(v) for k, v in rates.items()}


_UI = "font-family:var(--ea-font-ui)"


def picture(labels: dict) -> str:
    """The selection rates per sex for the labels of UCI Adult, as bars, with a dashed line at four-fifths of the highest.
    `labels` gives "title", "groups" (words for Female and Male), "ylabel", "line", "decimal" and "alt"."""
    from everything_ai import chart
    rates = rates_by_sex("label")
    groups = ["Female", "Male"]
    highest = max(rates.values())
    p = chart.Panel(x=(-0.7, 1.7), y=(0, 0.4), ylabel=labels["ylabel"], yticks=[0, 0.1, 0.2, 0.3, 0.4],
                    xticks=[0, 1], xtick_labels=[labels["groups"][g] for g in groups])
    for i, g in enumerate(groups):
        p.rect(i - 0.28, 0, i + 0.28, rates[g], role="data")
        p.text(i, rates[g] + 0.015, f"{rates[g]:.3f}".replace(".", labels["decimal"]), "ink")
    p.line([-0.6, 1.6], [0.8 * highest] * 2, role="ink", width=1.5, dashed=True)
    p.text(-0.6, 0.8 * highest + 0.03, labels["line"], "ink-2", anchor="start", halo=True)
    svg = chart.figure([p], alt=labels["alt"], width=460, height=300)
    return yours.figure(labels["title"], chart.with_decimal_comma(svg) if labels["decimal"] == "," else svg)


def _number(v) -> float | None:
    if isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _ratios_picture(seen) -> str:
    """The learner's `ratios` as bars, one per group, with a dashed line at 0.8."""
    from everything_ai import chart
    import pandas as pd
    mine = seen.mine
    if mine is None or not mine.has_value or not isinstance(mine.value, pd.Series) or len(mine.value) > 8:
        return ""
    values = [_number(v) for v in mine.value.tolist()]
    if not values or None in values:
        return ""
    names = [str(k) for k in mine.value.index]
    top = max(1.2, min(3.0, max(values) + 0.2))
    p = chart.Panel(x=(-0.7, len(values) - 0.3), y=(0, top), ylabel=seen.word({"en": "impact ratio", "nl": "verhouding"}),
                    yticks=[0, 0.4, 0.8, 1.2] + ([2.0] if top > 2 else []), xticks=list(range(len(values))),
                    xtick_labels=names)
    for i, v in enumerate(values):
        shown = max(0.0, min(top, v))
        if shown > 0:
            p.rect(i - 0.28, 0, i + 0.28, shown, role="data")
        p.text(i, shown + top * 0.04, f"{v:.2f}".replace(".", "," if seen.lang == "nl" else "."), "ink")
    p.line([-0.6, len(values) - 0.4], [0.8, 0.8], role="ink", width=1.5, dashed=True)
    p.text(-0.6, 0.8 + top * 0.07, seen.word({"en": "four-fifths", "nl": "vier vijfde"}), "ink-2", anchor="start",
           halo=True)
    title = seen.word({"en": "Your impact ratios per group", "nl": "Jouw verhoudingen per groep"})
    svg = chart.figure([p], alt=title, width=460, height=280)
    return yours.figure(title, chart.with_decimal_comma(svg) if seen.lang == "nl" else svg)


# The lesson is done when these are correct. "flag" and "meaning" are the choice questions in the lesson.
REQUIRED = ["ratios", "flag", "meaning"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_READ_LINE = 'people = pd.read_csv("data/adult.csv.gz")'
_HOURS_LINE = f"hours = {HOURS}"
_HEAD = ("import pandas as pd\n\n# 48,842 people from a US survey of 1994 (UCI Adult)\n" + _READ_LINE + "\n"
         + "# a made-up rule: yes for whoever works at least this many hours a week\n" + _HOURS_LINE + "\n\n")
_START = _HEAD + "...\nratios\n"
_BODY = ('picked = people[people["hours_per_week"] >= hours]\n'
         'rates = picked.groupby("sex").size() / people.groupby("sex").size()\n'
         "ratios = rates / rates.max()\n")
_SOLUTION = _HEAD + _BODY + "ratios"


def _s(body: str) -> str:
    return _HEAD + body + "ratios"


def _swap(old: str, new: str) -> str:
    assert _BODY.count(old) == 1
    return _s(_BODY.replace(old, new))


def _close(got, want: dict) -> bool:
    """Whether `got` is a Series with these values per group."""
    import pandas as pd
    if not isinstance(got, pd.Series) or len(got) != len(want):
        return False
    try:
        return all(abs(float(got[k]) - v) < 1e-6 for k, v in want.items())
    except (KeyError, TypeError, ValueError):
        return False


def _of(condition: str, kind: str) -> dict:
    """What a learner gets with the label or the hours rule: "rates", "ratios", "upside" (the highest divided by each
    rate) or "lowest" (each rate divided by the lowest)."""
    if condition == "strict":
        people = _people()
        picked = people[people["hours_per_week"] > HOURS]
        r = {k: float(v) for k, v in (picked.groupby("sex").size() / people.groupby("sex").size()).items()}
    else:
        r = rates_by_sex(condition)
    if kind == "rates":
        return r
    if kind == "upside":
        return {k: max(r.values()) / v for k, v in r.items()}
    if kind == "lowest":
        return {k: v / min(r.values()) for k, v in r.items()}
    return impact_ratios(r)


EXERCISES = {
    "ratios": CodeBox(
        name="ratios",
        picture=_ratios_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_READ_LINE, _HOURS_LINE],
        # The check also runs the box with another number of hours.
        others=[{"hours": OTHER_HOURS}],
        first=[_DOTS],
        digits=3,
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_people`. In the box the names have none: `people`, `hours`, `picked`, `rates`, `ratios`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_people`. In het vak hebben de namen dat niet: `people`, `hours`, `picked`, `rates`, `ratios`."),
            Mistake("column", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks the table for a column it does not have. The columns are written with an underscore, such as `hours_per_week`, and the groups are in `sex`.",
                    nl="Je code vraagt de tabel om een kolom die hij niet heeft. De kolommen zijn geschreven met een liggend streepje, zoals `hours_per_week`, en de groepen staan in `sex`."),
            Mistake("label", matches=lambda got, text: _close(got, _of("label", "ratios")),
                    en="Your last line gives the impact ratios of the labels of UCI Adult, as in the worked example. Here the yes comes from the rule about hours: pick the rows where `hours_per_week` is at least `hours`.",
                    nl="Je laatste regel geeft de verhoudingen van de labels uit UCI Adult, zoals in het uitgewerkte voorbeeld. Hier komt het ja van de regel over uren: kies de rijen waarin `hours_per_week` minstens `hours` is."),
            Mistake("strict", matches=lambda got, text: _close(got, _of("strict", "ratios")),
                    en="Close. But your rule leaves out the people who work exactly `hours` hours. At least is `>=`.",
                    nl="Bijna. Maar je regel laat de mensen weg die precies `hours` uur werken. Minstens is `>=`."),
            Mistake("rates", matches=lambda got, text: _close(got, _of("hours", "rates")),
                    en="Your last line gives the selection rates themselves. The impact ratio is each rate divided by the highest rate: `rates / rates.max()`.",
                    nl="Je laatste regel geeft de selectiepercentages zelf. De verhouding is elk percentage gedeeld door het hoogste: `rates / rates.max()`."),
            Mistake("upside", matches=lambda got, text: _close(got, _of("hours", "upside")),
                    en="Your ratios are upside down: you divided the highest rate by each rate. Divide each rate by the highest: `rates / rates.max()`.",
                    nl="Je verhoudingen staan op hun kop: je deelde het hoogste percentage door elk percentage. Deel elk percentage door het hoogste: `rates / rates.max()`."),
            Mistake("lowest", matches=lambda got, text: _close(got, _of("hours", "lowest")),
                    en="You divided by the lowest rate. The rule compares each group with the group that gets a yes most often: divide by `rates.max()`.",
                    nl="Je deelde door het laagste percentage. De regel vergelijkt elke groep met de groep die het vaakst een ja krijgt: deel door `rates.max()`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the three lines of the worked example, with the rule about hours instead of the label.",
                      "nl": "Vervang de `...` door de drie regels van het uitgewerkte voorbeeld, met de regel over uren in plaats van het label."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the lines of `people` and `hours` exactly as they were: put them back. The check tries another number of hours by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `people` en `hours` nodig zoals ze waren: zet ze terug. De check probeert zelf een ander aantal uren."},
            "no_value": {"en": "The last line of the box shows nothing. Put `ratios` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `ratios` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does `picked` hold the rows with at least `hours` hours, are `rates` the picked people per sex divided by all people per sex, and are `ratios` the rates divided by the highest?",
                      "nl": "Je laatste regel geeft `{got}`. Bevat `picked` de rijen met minstens `hours` uur, zijn `rates` de gekozen mensen per geslacht gedeeld door alle mensen per geslacht, en zijn `ratios` de percentages gedeeld door het hoogste?"},
            "other": {"en": "Right for 45 hours. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does your rule use `hours` itself, and not the number 45?",
                      "nl": "Goed voor 45 uur. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruikt je regel `hours` zelf, en niet het getal 45?"},
            "other_error": {"en": "Right for 45 hours. But with {where}, your code stops with an error. Do you use `hours` and `people` themselves?",
                            "nl": "Goed voor 45 uur. Maar met {where} stopt je code met een fout. Gebruik je `hours` en `people` zelf?"},
            "correct": {"en": "About `0.433` for women: the rule names no sex, but women get a yes less than half as often as men, far below four-fifths. That is a signal of disparate impact; whether the rule is allowed depends on what it is for.",
                        "nl": "Ongeveer `0.433` voor vrouwen: de regel noemt geen geslacht, maar vrouwen krijgen minder dan half zo vaak een ja als mannen, ver onder vier vijfde. Dat is een signaal van disparate impact; of de regel mag, hangt af van waar hij voor is."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("ratios", _START, "empty", None),
    ("ratios", _SOLUTION, "correct", None),
    ("ratios", _s('picked = people[people["hours_per_week"] >= hours]\nratios = (picked.groupby("sex").size()\n'
                  '          / people.groupby("sex").size())\nratios = ratios / ratios.max()\n'), "correct", None),
    ("ratios", _s(_BODY.replace("people[", "_people[")), "error", "underscore"),
    ("ratios", _swap('people["hours_per_week"]', 'people["hours-per-week"]'), "error", "column"),
    ("ratios", _swap('people["hours_per_week"] >= hours', 'people["income"] == ">50K"'), "wrong", "label"),
    ("ratios", _swap(">= hours", "> hours"), "wrong", "strict"),
    ("ratios", _swap("ratios = rates / rates.max()\n", "ratios = rates\n"), "wrong", "rates"),
    ("ratios", _swap("rates / rates.max()", "rates.max() / rates"), "wrong", "upside"),
    ("ratios", _swap("rates / rates.max()", "rates / rates.min()"), "wrong", "lowest"),
    ("ratios", _s(_BODY.replace('"sex"', '"race"')), "wrong", None),
    ("ratios", _swap(">= hours", ">= 45"), "almost", "other-value"),
    ("ratios", _SOLUTION.replace("hours = 45", "hours = 40"), "wrong", "changed-line"),
    ("ratios", _s("...\nratios = 0.43\n"), "wrong", "dots-left"),
]
