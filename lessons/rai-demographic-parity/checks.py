# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-demographic-parity: the difference and the ratio of the yes rates per value of a column of
the table UCI Adult for a made-up rule, sorting ways of handing out a yes by what demographic parity sees in them, and
where equal yes rates fit as a goal; with the picture of the yes rates and the precision of a model with one limit, a
limit per group and the women drawn at random, and the picture of the learner's yes rates."""
import math
import re

from everything_ai.check import CodeBox, Mistake, Sort
from everything_ai import paths, yours

DATA = paths.ROOT / "data" / "adult.csv.gz"
NUMBERS = ["age", "education_num", "hours_per_week"]
WORDS = ["workclass", "marital_status", "occupation", "relationship", "race"]   # the word columns, without sex
COLUMN = "race"
OTHER_COLUMN = "sex"
SCHOOL = 13   # the made-up rule of the exercise: a yes for an education_num of at least this (a bachelor's degree)

# What the demo works out on the test part of the table, per way of choosing (tests/test_rai_demographic_parity.py
# checks it): the yes rates of women and men, the precision for women and men, and how often the model is right.
WAYS = {"one": ((0.0705, 0.2479), (0.6728, 0.6891), 0.8344),
        "per-group": ((0.1889, 0.1889), (0.4316, 0.7348), 0.8143),
        "random": ((0.2453, 0.2479), (0.1113, 0.6891), 0.7628)}
LIMITS = (0.2048, 0.5979)   # the limits of women and men in the second way

_TABLE = {}


def people():
    """The table UCI Adult of the course, read once."""
    if "people" not in _TABLE:
        import pandas as pd
        _TABLE["people"] = pd.read_csv(DATA)
    return _TABLE["people"]


def ways() -> dict:
    """What the demo does: the model without the column sex, and per way of choosing its yes for each person of the
    test part, with "limits" the two limits of the second way."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    table = people()
    train, test = table["split"] == "train", table["split"] == "test"
    tested = table[test]
    women, men = tested["sex"] == "Female", tested["sex"] == "Male"
    X = pd.get_dummies(table[NUMBERS + WORDS], columns=WORDS, dtype=float)
    X = X / X.max()
    with threadpool_limits(1, user_api="openmp"):
        model = LogisticRegression(max_iter=1000)
        model.fit(X[train], table["income"][train] == ">50K")
        score = model.predict_proba(X[test])[:, 1]
    one = score > 0.5
    rate = one.mean()
    limit_women, limit_men = np.quantile(score[women], 1 - rate), np.quantile(score[men], 1 - rate)
    chance = np.random.default_rng(0).random(len(tested)) < one[men].mean()
    return {"one": one, "per-group": score > limit_women * women + limit_men * men,
            "random": chance * women + one * men, "limits": (float(limit_women), float(limit_men))}


def summary(yes) -> tuple:
    """For a yes per person of the test part: the yes rates of women and men, their precision, and how often the yes
    is right, as the demo prints them."""
    tested = people()[people()["split"] == "test"]
    picked = tested[yes]
    rates = picked.groupby("sex").size() / tested.groupby("sex").size()
    rich = picked[picked["income"] == ">50K"]
    precision = rich.groupby("sex").size() / picked.groupby("sex").size()
    right = float((yes == (tested["income"] == ">50K")).mean())
    return ((float(rates["Female"]), float(rates["Male"])), (float(precision["Female"]), float(precision["Male"])),
            right)


def rates(column: str, rule: str = "school") -> dict:
    """Per value of `column`, the share of the people with that value who get a yes: from the rule of the exercise
    ("school"), with > instead of >= ("strict"), or from a perfect model ("rich"). With of="yes" on `shares`, the
    share of all the yes instead."""
    table = people()
    if rule == "school":
        yes = table[table["education_num"] >= SCHOOL]
    elif rule == "strict":
        yes = table[table["education_num"] > SCHOOL]
    else:
        yes = table[table["income"] == ">50K"]
    s = yes.groupby(column).size() / table.groupby(column).size()
    return {k: float(v) for k, v in s.items()}


def shares_of_yes(column: str) -> dict:
    """Per value of `column`, its share of all the people the rule gives a yes."""
    table = people()
    yes = table[table["education_num"] >= SCHOOL]
    s = yes.groupby(column).size() / len(yes)
    return {k: float(v) for k, v in s.items()}


def parity(r: dict) -> tuple:
    """The difference (highest minus lowest) and the ratio (lowest divided by highest) of yes rates per group."""
    return max(r.values()) - min(r.values()), min(r.values()) / max(r.values())


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"


def picture(labels: dict) -> str:
    """Two panels with a row per way of choosing, in the order of WAYS: on the left the yes rates of women (open bars)
    and men (filled bars), on the right their precision. `labels` gives "title", "rows" (three names), "xlabels" (two),
    "key", "decimal" and "alt"."""
    from everything_ai import chart
    keys = list(WAYS)
    top = 1.6 * len(keys) + 0.1
    panels = []
    for k, (xmax, ticks) in enumerate(((0.35, [0, 0.1, 0.2, 0.3]), (1, [0, 0.25, 0.5, 0.75, 1]))):
        p = chart.Panel(x=(0, xmax), y=(0, top), xlabel=labels["xlabels"][k], yticks=[], xticks=ticks,
                        xtick_labels=[f"{t:g}" for t in ticks])
        for i, (key, name) in enumerate(zip(keys, labels["rows"])):
            women, men = WAYS[key][k]
            y = top - 1.6 * (i + 1)
            p.text(0.002, y + 1.2, name, "ink", anchor="start")
            p.rect(0, y + 0.55, women, y + 0.85, role="data")
            p.rect(0, y + 0.15, men, y + 0.45, role="data", solid=True)
            gap = 0.012 * xmax
            p.text(women + gap, y + 0.7, f"{women:.2f}".replace(".", labels["decimal"]), "ink-2", anchor="start")
            p.text(men + gap, y + 0.3, f"{men:.2f}".replace(".", labels["decimal"]), "ink-2", anchor="start")
        panels.append(p)
    svg = chart.figure(panels, alt=labels["alt"], width=330, height=300)
    if labels["decimal"] == ",":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(labels["title"], svg + f'<div style="{_UI}">{labels["key"]}</div>')


def _rates(v) -> dict | None:
    """The learner's Series of a yes rate per value, when it is one: 1 to 12 values, each from 0 to 1."""
    import pandas as pd
    if not isinstance(v, pd.Series) or not 1 <= len(v) <= 12:
        return None
    out = {}
    for k, x in v.items():
        try:
            x = float(x)
        except (TypeError, ValueError):
            return None
        if not isinstance(k, str) or math.isnan(x) or not 0 <= x <= 1:
            return None
        out[k] = x
    return out


def _number(x) -> float | None:
    """A real number that is not True or False, as a float; else None."""
    import numbers
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return None
    return float(x)


def _pair(v) -> tuple | None:
    """The learner's two numbers, when the last line gives two."""
    if not isinstance(v, (tuple, list)) or len(v) != 2:
        return None
    a, b = _number(v[0]), _number(v[1])
    return None if a is None or b is None else (a, b)


_WORDS = {
    "en": {"title": "Your yes rates per value of {column}", "xlabel": "yes rate",
           "pair": "difference {d}, ratio {r}",
           "alt": "One bar per value of {column}, as long as its yes rate: {shown}. Dashed lines at the lowest and the highest rate."},
    "nl": {"title": "Jouw deel met een ja per waarde van {column}", "xlabel": "deel met een ja",
           "pair": "verschil {d}, verhouding {r}",
           "alt": "Eén balk per waarde van {column}, zo lang als het deel met een ja: {shown}. Gestippelde lijnen bij het kleinste en het grootste deel."},
}


def _rates_picture(seen) -> str:
    """Per value of the column a filled bar as long as the learner's yes rate, with dashed lines at the lowest and the
    highest rate, and above them the learner's difference and ratio when the last line gives two numbers."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    value = _rates(names.get("rates"))
    column = names.get("column", COLUMN)
    if value is None or not isinstance(column, str):
        return ""
    w = _WORDS[seen.lang]
    sign = "," if seen.lang == "nl" else "."

    def num(x):
        return f"{x:.2f}".replace(".", sign)

    keys = list(value)
    lo, hi = min(value.values()), max(value.values())
    top = 1.2 * len(keys) + 0.6
    p = chart.Panel(x=(0, 1), y=(0, top), xlabel=w["xlabel"], yticks=[],
                    xticks=[0, 0.25, 0.5, 0.75, 1], xtick_labels=["0", "0.25", "0.5", "0.75", "1"])
    for i, k in enumerate(keys):
        y = top - 0.6 - 1.2 * (i + 1)
        p.text(0.005, y + 0.72, k + ": " + num(value[k]), "ink", anchor="start", halo=True)   # Panel.text escapes it
        p.rect(0, y + 0.15, value[k], y + 0.45, role="data", solid=True)
    for x in (lo, hi):
        p.line([x, x], [0, top - 0.6], role="ink-3", width=1, dashed=True)
    pair = _pair(mine.value)
    if pair is not None and all(math.isfinite(x) for x in pair):
        p.text(0.005, top - 0.3, w["pair"].format(d=num(pair[0]), r=num(pair[1])), "ink-2", anchor="start", halo=True)
    shown = "; ".join(f"{k} {num(value[k])}" for k in keys)
    svg = chart.figure([p], alt=w["alt"].format(column=column, shown=shown), width=520,
                       height=max(180, 44 * len(keys) + 100))
    return yours.figure(w["title"].format(column=column), chart.with_decimal_comma(svg) if seen.lang == "nl" else svg)


def _close(got, want) -> bool:
    """Two numbers, each the same to two decimals: rounded or not."""
    a, b = _pair(got), _pair(want)
    if a is None or b is None:
        return False
    return all(math.isfinite(x) and abs(x - y) < 0.006 for x, y in zip(a, b))


def _like(make):
    """A mistake that matches the two numbers `make()` gives for the column race."""
    return lambda got, text: _close(got, make())


def _swapped(got, text) -> bool:
    p = _pair(got)
    return p is not None and _close((p[1], p[0]), parity(rates(COLUMN)))


def _upside(got, text) -> bool:
    p = _pair(got)
    d, r = parity(rates(COLUMN))
    return p is not None and abs(p[0] - d) < 0.006 and abs(p[1] * r - 1) < 0.01


def _negative(got, text) -> bool:
    p = _pair(got)
    d, r = parity(rates(COLUMN))
    return p is not None and abs(p[0] + d) < 0.006


def _counts(got, text) -> bool:
    p = _pair(got)
    return p is not None and p[0] > 1


def _missing(got, text) -> bool:
    p = _pair(got)
    return p is not None and (math.isnan(p[0]) or math.isnan(p[1]))


def _series(got, text) -> bool:
    import pandas as pd
    return isinstance(got, pd.Series)


def _one_number(got, text) -> bool:
    return _number(got) is not None


# The lesson is done when these are correct. "fit" is the choice question in the lesson.
REQUIRED = ["parity", "care", "fit"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_READ_LINE = 'people = pd.read_csv("data/adult.csv.gz")'
_COLUMN_LINE = f'column = "{COLUMN}"'
_HEAD = ("import pandas as pd\n\n# 48,842 people from a US survey of 1994 (UCI Adult)\n" + _READ_LINE + "\n"
         + _COLUMN_LINE + "\n\n")
_START = _HEAD + "...\ndifference, ratio\n"
_BODY = (f'yes = people[people["education_num"] >= {SCHOOL}]\n'
         "rates = yes.groupby(column).size() / people.groupby(column).size()\n"
         "difference = rates.max() - rates.min()\n"
         "ratio = rates.min() / rates.max()\n")
_SOLUTION = _HEAD + _BODY + "difference, ratio"


def _s(body: str, last: str = "difference, ratio") -> str:
    return _HEAD + body + last


def _swap(old: str, new: str) -> str:
    assert _BODY.count(old) == 1
    return _s(_BODY.replace(old, new))


EXERCISES = {
    "parity": CodeBox(
        name="parity",
        picture=_rates_picture,
        start=_START,
        solution=_SOLUTION,
        keep=[_READ_LINE, _COLUMN_LINE],
        compare=_close,
        digits=2,
        # The check also runs the box with another column.
        others=[{"column": OTHER_COLUMN}],
        first=[_DOTS],
        mistakes=[
            Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                    en="Your code uses a name with an underscore, such as `_rates`. In the box the names have none: `people`, `column`, `yes`, `rates`, `difference`, `ratio`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_rates`. In het vak hebben de namen dat niet: `people`, `column`, `yes`, `rates`, `difference`, `ratio`."),
            Mistake("column", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks the table for a column it does not have. Group by the name `column` itself, without quotes: it holds the text `\"race\"`. The years of schooling are in the column `education_num`.",
                    nl="Je code vraagt de tabel om een kolom die hij niet heeft. Groepeer op de naam `column` zelf, zonder aanhalingstekens: daarin staat de tekst `\"race\"`. Het aantal jaren school staat in de kolom `education_num`."),
            Mistake("text", error=lambda e: isinstance(e, TypeError),
                    en="Your code compares text with a number. Is the column `education_num`, the number of years, and not `education`, the words? And is `13` written without quotes?",
                    nl="Je code vergelijkt tekst met een getal. Is de kolom `education_num`, het aantal jaren, en niet `education`, de woorden? En staat `13` er zonder aanhalingstekens?"),
            Mistake("series", matches=_series,
                    en="Your last line gives the yes rates per group. Keep them in `rates`, and put the two numbers on the last line: `difference, ratio`.",
                    nl="Je laatste regel geeft het deel met een ja per groep. Bewaar dat in `rates`, en zet de twee getallen op de laatste regel: `difference, ratio`."),
            Mistake("one-number", matches=_one_number,
                    en="Your last line gives one number. The check wants two, with a comma between them: `difference, ratio`.",
                    nl="Je laatste regel geeft één getal. De check wil er twee, met een komma ertussen: `difference, ratio`."),
            Mistake("missing", matches=_missing,
                    en="Your numbers are `nan`: `yes` has no rows. Is the rule `people[\"education_num\"] >= 13`, with the column name spelled as in the table?",
                    nl="Je getallen zijn `nan`: `yes` heeft geen rijen. Is de regel `people[\"education_num\"] >= 13`, met de kolomnaam gespeld zoals in de tabel?"),
            Mistake("rich", matches=_like(lambda: parity(rates(COLUMN, "rich"))),
                    en="Your numbers are those of the perfect model of the worked example: you picked the people who earn more than 50,000 dollars. The rule picks the people with an `education_num` of 13 or more.",
                    nl="Je getallen zijn die van het perfecte model uit het uitgewerkte voorbeeld: je koos de mensen die meer dan 50.000 dollar verdienen. De regel kiest de mensen met een `education_num` van 13 of meer."),
            Mistake("strict", matches=_like(lambda: parity(rates(COLUMN, "strict"))),
                    en="Your rule leaves out the people with exactly 13, the bachelor's degree itself. Use `>=`, not `>`.",
                    nl="Je regel laat de mensen met precies 13 weg, de bachelor zelf. Gebruik `>=`, niet `>`."),
            Mistake("swapped", matches=_swapped,
                    en="Your two numbers are in the wrong order. The last line is `difference, ratio`: first the difference, then the ratio.",
                    nl="Je twee getallen staan in de verkeerde volgorde. De laatste regel is `difference, ratio`: eerst het verschil, dan de verhouding."),
            Mistake("upside", matches=_upside,
                    en="Your ratio is above 1: you divided the highest rate by the lowest. Divide the lowest by the highest: `rates.min() / rates.max()`.",
                    nl="Je verhouding is groter dan 1: je deelde het grootste deel door het kleinste. Deel het kleinste door het grootste: `rates.min() / rates.max()`."),
            Mistake("negative", matches=_negative,
                    en="Your difference is negative: you took the lowest rate minus the highest. Take the highest minus the lowest: `rates.max() - rates.min()`.",
                    nl="Je verschil is negatief: je nam het kleinste deel min het grootste. Neem het grootste min het kleinste: `rates.max() - rates.min()`."),
            Mistake("counts", matches=_counts,
                    en="Your difference is larger than 1, so `rates` holds counts, not shares. Divide the yes per group by all people per group: `people.groupby(column).size()`.",
                    nl="Je verschil is groter dan 1, dus `rates` bevat aantallen, geen delen. Deel de ja's per groep door alle mensen per groep: `people.groupby(column).size()`."),
            Mistake("of-yes", matches=_like(lambda: parity(shares_of_yes(COLUMN))),
                    en="Your rates add up to 1 over the groups: you divided by all the people with a yes. Divide by the people per group: `people.groupby(column).size()`.",
                    nl="Je delen tellen over de groepen op tot 1: je deelde door alle mensen met een ja. Deel door de mensen per groep: `people.groupby(column).size()`."),
            Mistake("by-sex", matches=_like(lambda: parity(rates(OTHER_COLUMN))),
                    en="Your numbers are those per sex, because you grouped by `sex`, as in the worked example. Group by the column you are checking: `groupby(column)`.",
                    nl="Je getallen zijn die per geslacht, omdat je op `sex` groepeerde, zoals in het uitgewerkte voorbeeld. Groepeer op de kolom die je controleert: `groupby(column)`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the four lines of the worked example, with the rule instead of the people who earn more than 50,000 dollars.",
                      "nl": "Vervang de `...` door de vier regels van het uitgewerkte voorbeeld, met de regel in plaats van de mensen die meer dan 50.000 dollar verdienen."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the lines of `people` and `column` exactly as they were: put them back. The check tries another column by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `people` en `column` nodig zoals ze waren: zet ze terug. De check probeert zelf een andere kolom."},
            "no_value": {"en": "The last line of the box shows nothing. Put `difference, ratio` on the last line, so the check can see both.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `difference, ratio` op de laatste regel, zodat de check ze allebei ziet."},
            "wrong": {"en": "Your last line gives `{got}`. Does `yes` hold the rows with an `education_num` of 13 or more, is `rates` the yes per group divided by all people per group, and are `difference` and `ratio` the highest minus the lowest and the lowest divided by the highest?",
                      "nl": "Je laatste regel geeft `{got}`. Bevat `yes` de rijen met een `education_num` van 13 of meer, is `rates` de ja's per groep gedeeld door alle mensen per groep, en zijn `difference` en `ratio` het grootste min het kleinste en het kleinste gedeeld door het grootste?"},
            "other": {"en": "Right for race. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you group by `column` itself, and not by the text `\"race\"`?",
                      "nl": "Goed voor race. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Groepeer je op `column` zelf, en niet op de tekst `\"race\"`?"},
            "other_error": {"en": "Right for race. But with {where}, your code stops with an error. Do you use `column` and `people` themselves?",
                            "nl": "Goed voor race. Maar met {where} stopt je code met een fout. Gebruik je `column` en `people` zelf?"},
            "correct": {"en": "A difference of `0.33` and a ratio of `0.23`: the rule gives a yes to 0.43 of the group Asian-Pac-Islander and to 0.10 of the group Amer-Indian-Eskimo, far from demographic parity. Per sex, its ratio is 0.84, above four-fifths. Whether a rule meets demographic parity depends on the groups you compare.",
                        "nl": "Een verschil van `0.33` en een verhouding van `0.23`: de regel geeft 0,43 van de groep Asian-Pac-Islander een ja en 0,10 van de groep Amer-Indian-Eskimo, ver van demographic parity. Per geslacht is de verhouding 0,84, boven de 80%. Of een regel aan demographic parity voldoet, hangt af van de groepen die je vergelijkt."},
        },
    ),
    "care": Sort(
        name="care",
        groups={
            "same": {"en": "Equal yes rates, the same care in each group", "nl": "Evenveel ja per groep, met dezelfde zorg in elke groep"},
            "careless": {"en": "Equal yes rates, careless in one group", "nl": "Evenveel ja per groep, maar slordig in één groep"},
            "unequal": {"en": "Unequal yes rates", "nl": "Niet evenveel ja per groep"},
        },
        items={
            "savings": {"en": "**Savings.** A bank lends to the 30 percent with the most savings in each group.",
                        "nl": "**Spaargeld.** Een bank leent aan de 30 procent met het meeste spaargeld in elke groep."},
            "chance": {"en": "**Savings and chance.** A bank lends to the 30 percent with the most savings in one group, and to a random 30 percent in the other.",
                       "nl": "**Spaargeld en toeval.** Een bank leent aan de 30 procent met het meeste spaargeld in de ene groep, en aan een willekeurige 30 procent in de andere."},
            "school": {"en": "**One in five.** A school admits the best-scoring one in five applicants from each neighborhood.",
                       "nl": "**Een op de vijf.** Een school laat uit elke wijk het beste vijfde deel van de aanmelders toe."},
            "first": {"en": "**First come.** A firm hires 10 percent of each group: in one group after interviews, in the other the first ones who applied.",
                      "nl": "**Wie het eerst komt.** Een bedrijf neemt 10 procent van elke groep aan: in de ene groep na gesprekken, in de andere wie het eerst solliciteerde."},
            "perfect": {"en": "**A perfect model.** On UCI Adult, a model says yes exactly to the people who earn more than 50,000 dollars.",
                        "nl": "**Een perfect model.** Op UCI Adult zegt een model precies ja tegen de mensen die meer dan 50.000 dollar verdienen."},
            "unaware": {"en": "**Without the column.** A model without the column sex gives a yes to 0.07 of the women and 0.25 of the men.",
                        "nl": "**Zonder de kolom.** Een model zonder de kolom sex geeft 0,07 van de vrouwen en 0,25 van de mannen een ja."},
        },
        solution={"savings": "same", "chance": "careless", "school": "same", "first": "careless", "perfect": "unequal",
                  "unaware": "unequal"},
        nudge={
            "savings": {"en": "Is the share the same in each group? And does each group get the same kind of choice?",
                        "nl": "Is het deel in elke groep hetzelfde? En krijgt elke groep dezelfde soort keuze?"},
            "chance": {"en": "The share is the same. But how are the people with a yes picked in the second group?",
                       "nl": "Het deel is hetzelfde. Maar hoe worden de mensen met een ja in de tweede groep gekozen?"},
            "school": {"en": "One in five from each neighborhood, and in each the best scores: what changes from one neighborhood to the next?",
                       "nl": "Een op de vijf uit elke wijk, en in elke wijk de beste scores: wat verschilt er van wijk tot wijk?"},
            "first": {"en": "Ten percent of each group, but does the firm look at the applicants in both?",
                      "nl": "Tien procent van elke groep, maar kijkt het bedrijf in allebei naar de sollicitanten?"},
            "perfect": {"en": "In the data, 0.11 of the women and 0.30 of the men earn more than 50,000 dollars. What does a model that is always right then give each group?",
                        "nl": "In de data verdient 0,11 van de vrouwen en 0,30 van de mannen meer dan 50.000 dollar. Wat geeft een model dat altijd gelijk heeft dan elke groep?"},
            "unaware": {"en": "Compare the two shares: 0.07 and 0.25.",
                        "nl": "Vergelijk de twee delen: 0,07 en 0,25."},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the ways of choosing again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de manieren van kiezen opnieuw uit."},
            "empty": {"en": "Sort each way of handing out a yes by what demographic parity sees in it.",
                      "nl": "Deel elke manier om een ja te geven in bij wat demographic parity erin ziet."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The way *{item}* belongs somewhere else.",
                      "nl": "De manier *{item}* hoort ergens anders."},
            "correct": {"en": "All {n} are right. All four with equal rates meet demographic parity, the careless ones too: it only counts the yes. And the perfect model does not meet it, because the labels themselves differ per group.",
                        "nl": "Alle {n} kloppen. Alle vier met evenveel ja voldoen aan demographic parity, ook de slordige: het telt alleen de ja's. En het perfecte model voldoet er niet aan, omdat de labels zelf per groep verschillen."},
        },
    ),
}

_SORTED = {"savings": "same", "chance": "careless", "school": "same", "first": "careless", "perfect": "unequal",
           "unaware": "unequal"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("parity", _START, "empty", None),
    ("parity", _SOLUTION, "correct", None),
    ("parity", _s(_BODY.replace("rates.max() - rates.min()", "round(rates.max() - rates.min(), 2)")
                  .replace("rates.min() / rates.max()", "round(rates.min() / rates.max(), 2)")), "correct", None),
    ("parity", _s('yes = people[people["education_num"] > 12]\nper_group = people.groupby(column).size()\n'
                  "rates = yes.groupby(column).size() / per_group\nhighest = rates.max()\nlowest = rates.min()\n"
                  "difference = highest - lowest\nratio = lowest / highest\n"), "correct", None),
    ("parity", _s(_BODY.replace("ratio = rates.min()", "ratio = _rates.min()")), "error", "underscore"),
    ("parity", _swap("yes.groupby(column)", 'yes.groupby("column")'), "error", "column"),
    ("parity", _swap('>= 13', '>= "13"'), "error", "text"),
    ("parity", _s(_BODY, "rates"), "wrong", "series"),
    ("parity", _s(_BODY, "difference"), "wrong", "one-number"),
    ("parity", _swap('people["education_num"]', 'people["educaton_num"]'), "error", "column"),
    ("parity", _swap('>= 13', '>= 17'), "wrong", "missing"),
    ("parity", _swap('people["education_num"] >= 13', 'people["income"] == ">50K"'), "wrong", "rich"),
    ("parity", _swap('>= 13', '> 13'), "wrong", "strict"),
    ("parity", _s(_BODY, "ratio, difference"), "wrong", "swapped"),
    ("parity", _swap("rates.min() / rates.max()", "rates.max() / rates.min()"), "wrong", "upside"),
    ("parity", _swap("rates.max() - rates.min()", "rates.min() - rates.max()"), "wrong", "negative"),
    ("parity", _swap(" / people.groupby(column).size()", ""), "wrong", "counts"),
    ("parity", _swap("people.groupby(column).size()", "len(yes)"), "wrong", "of-yes"),
    ("parity", _s(_BODY.replace("groupby(column)", 'groupby("sex")')), "wrong", "by-sex"),
    ("parity", _s(_BODY.replace("groupby(column)", 'groupby("relationship")')), "wrong", None),
    ("parity", _s(_BODY.replace("groupby(column)", 'groupby("race")')), "almost", "other-value"),
    ("parity", _SOLUTION.replace(_COLUMN_LINE, 'column = "sex"'), "wrong", "changed-line"),
    ("parity", _s("...\ndifference = 0.33\nratio = 0.23\n"), "wrong", "dots-left"),
    ("care", {}, "empty", None),
    ("care", _SORTED, "correct", None),
    ("care", {"savings": "same", "perfect": "unequal"}, "wrong", "unfinished"),
    ("care", {**_SORTED, "chance": "same"}, "wrong", "wrong:chance"),
    ("care", {**_SORTED, "perfect": "same"}, "wrong", "wrong:perfect"),
    ("care", {**_SORTED, "first": "unequal"}, "wrong", "wrong:first"),
]
