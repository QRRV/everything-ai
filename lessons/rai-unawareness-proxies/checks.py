# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-unawareness-proxies: the share of men per value of a column of the table UCI Adult, sorting
pieces of data by the discriminatiegrond they can stand in for, and why a team keeps the sex column to check; with the
picture of the yes rates of a model with and without the sex column, and the picture of the learner's shares."""
import re

from everything_ai.check import CodeBox, Mistake, Sort
from everything_ai import paths, yours

DATA = paths.ROOT / "data" / "adult.csv.gz"
NUMBERS = ["age", "education_num", "hours_per_week"]
WORDS = ["workclass", "marital_status", "occupation", "relationship", "race"]   # the word columns, without sex
LESS = ["workclass", "occupation", "race"]   # also without relationship and marital status
COLUMN = "relationship"
OTHER_COLUMN = "marital_status"

# The yes rates of women and men on the test part of the table, as the demo works them out (tests/test_rai_unawareness_
# proxies.py checks them): the labels themselves, the model with the sex column, without it, and also without
# relationship and marital status.
RATES = {"data": (0.1088, 0.2998), "with": (0.0677, 0.2535), "without": (0.0705, 0.2479), "less": (0.0828, 0.1725)}

_TABLE = {}


def people():
    """The table UCI Adult of the course, read once."""
    if "people" not in _TABLE:
        import pandas as pd
        _TABLE["people"] = pd.read_csv(DATA)
    return _TABLE["people"]


def features(words: list):
    """The columns of numbers, and each word column as one-hot columns, every column scaled to run from 0 to 1."""
    import pandas as pd
    table = people()
    X = pd.get_dummies(table[NUMBERS + words], columns=words, dtype=float)
    return X / X.max()


def model_says(words: list, target: str = "income") -> dict:
    """What the demo works out for a logistic regression on these word columns: the yes rates of women and men on the
    test part, and how often it is right there. With target "sex" it guesses who is a man instead of the income."""
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    table = people()
    train, test = table["split"] == "train", table["split"] == "test"
    truth = table["income"] == ">50K" if target == "income" else table["sex"] == "Male"
    X = features(words)
    with threadpool_limits(1, user_api="openmp"):
        model = LogisticRegression(max_iter=1000)
        model.fit(X[train], truth[train])
        yes = model.predict(X[test])
    tested = table[test]
    rates = tested[yes].groupby("sex").size() / tested.groupby("sex").size()
    return {"women": float(rates["Female"]), "men": float(rates["Male"]), "right": float((yes == truth[test]).mean())}


def share(column: str, of: str = "men") -> dict:
    """Per value of `column`, the share of the people with that value who are men (or, with of="rich", who earn more
    than 50,000 dollars)."""
    table = people()
    picked = table[table["sex"] == "Male"] if of == "men" else table[table["income"] == ">50K"]
    s = picked.groupby(column).size() / table.groupby(column).size()
    return {k: float(v) for k, v in s.items()}


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"


def picture(labels: dict) -> str:
    """The yes rates of women (open bars) and men (filled bars) in four rows: the labels, the model with the sex column,
    without it, and also without relationship and marital status. `labels` gives "title", "rows" (four names, in the
    order of RATES), "xlabel", "key", "decimal" and "alt"."""
    from everything_ai import chart
    rows = list(RATES)
    top = 1.6 * len(rows) + 0.1
    p = chart.Panel(x=(0, 0.4), y=(0, top), xlabel=labels["xlabel"], yticks=[],
                    xticks=[0, 0.1, 0.2, 0.3, 0.4], xtick_labels=["0", "0.1", "0.2", "0.3", "0.4"])
    for i, (key, name) in enumerate(zip(rows, labels["rows"])):
        women, men = RATES[key]
        y = top - 1.6 * (i + 1)
        p.text(0.002, y + 1.2, name, "ink", anchor="start")
        p.rect(0, y + 0.55, women, y + 0.85, role="data")
        p.rect(0, y + 0.15, men, y + 0.45, role="data", solid=True)
        p.text(women + 0.004, y + 0.7, f"{women:.2f}".replace(".", labels["decimal"]), "ink-2", anchor="start")
        p.text(men + 0.004, y + 0.3, f"{men:.2f}".replace(".", labels["decimal"]), "ink-2", anchor="start")
    svg = chart.figure([p], alt=labels["alt"], width=520, height=340)
    if labels["decimal"] == ",":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(labels["title"], svg + f'<div style="{_UI}">{labels["key"]}</div>')


def _shares(v) -> dict | None:
    """The learner's Series of a share per value, when it is one: at most 12 values, each from 0 to 1."""
    import math
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


_WORDS = {
    "en": {"title": "Your share of men per value of {column}", "xlabel": "share of men",
           "all": "all people: {men}", "alt": "One bar per value of {column}, as long as the share of men with that value: {shown}. A dashed line at {men}, the share of men among all people."},
    "nl": {"title": "Jouw aandeel mannen per waarde van {column}", "xlabel": "aandeel mannen",
           "all": "alle mensen: {men}", "alt": "Eén balk per waarde van {column}, zo lang als het aandeel mannen met die waarde: {shown}. Een gestippelde lijn op {men}, het aandeel mannen onder alle mensen."},
}


def _share_picture(seen) -> str:
    """Per value of the column a filled bar as long as the learner's share of men, with a dashed line at the share of
    men among all people."""
    import pandas as pd
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    value = _shares(mine.value)
    if value is None:
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    column = names.get("column", COLUMN)
    table = names.get("people")
    if not isinstance(column, str) or not isinstance(table, pd.DataFrame) or "sex" not in table.columns or len(table) == 0:
        return ""
    men = float((table["sex"] == "Male").mean())
    w = _WORDS[seen.lang]
    sign = "," if seen.lang == "nl" else "."

    def num(x):
        return f"{x:.2f}".replace(".", sign)

    keys = list(value)
    top = 1.2 * len(keys) + 0.6
    p = chart.Panel(x=(0, 1), y=(0, top), xlabel=w["xlabel"], yticks=[],
                    xticks=[0, 0.25, 0.5, 0.75, 1], xtick_labels=["0", "0.25", "0.5", "0.75", "1"])
    for i, k in enumerate(keys):
        y = top - 0.6 - 1.2 * (i + 1)
        p.text(0.005, y + 0.72, k + ": " + num(value[k]), "ink", anchor="start")   # Panel.text escapes it
        p.rect(0, y + 0.15, value[k], y + 0.45, role="data", solid=True)
    p.line([men, men], [0, top], role="ink-3", width=1, dashed=True)
    p.text(min(men + 0.01, 0.99), top - 0.3, w["all"].format(men=num(men)), "ink-3", anchor="start", halo=True)
    shown = "; ".join(f"{k} {num(value[k])}" for k in keys)
    alt = w["alt"].format(column=column, shown=shown, men=num(men))
    svg = chart.figure([p], alt=alt, width=520, height=max(180, 44 * len(keys) + 90))
    return yours.figure(w["title"].format(column=column), chart.with_decimal_comma(svg) if seen.lang == "nl" else svg)


def _close(got, want) -> bool:
    """The same values, and per value the same share to two decimals: rounded or not."""
    a, b = _shares(got), _shares(want)
    if a is None or b is None or a.keys() != b.keys():
        return False
    return all(abs(a[k] - b[k]) < 0.006 for k in a)


def _like(want: dict):
    """A mistake that matches a Series with these shares per value of relationship."""
    import pandas as pd
    return lambda got, text: _close(got, pd.Series(want))


def _counts(got, text) -> bool:
    import pandas as pd
    if not isinstance(got, pd.Series) or len(got) == 0 or not pd.api.types.is_numeric_dtype(got):
        return False
    return bool((got > 1).all()) and bool((got == got.round()).all())


def _all_missing(got, text) -> bool:
    import pandas as pd
    return isinstance(got, pd.Series) and len(got) > 0 and bool(got.isna().all())


def _by_sex(got, text) -> bool:
    import pandas as pd
    return isinstance(got, pd.Series) and sorted(str(k) for k in got.index) == ["Female", "Male"]


# The lesson is done when these are correct. "check" is the choice question in the lesson.
REQUIRED = ["proxy", "proxies", "check"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_READ_LINE = 'people = pd.read_csv("data/adult.csv.gz")'
_COLUMN_LINE = f'column = "{COLUMN}"'
_HEAD = ("import pandas as pd\n\n# 48,842 people from a US survey of 1994 (UCI Adult)\n" + _READ_LINE + "\n"
         + _COLUMN_LINE + "\n\n")
_START = _HEAD + "...\nshare\n"
_BODY = ('men = people[people["sex"] == "Male"]\n'
         "share = men.groupby(column).size() / people.groupby(column).size()\n")
_SOLUTION = _HEAD + _BODY + "share"


def _s(body: str) -> str:
    return _HEAD + body + "share"


def _swap(old: str, new: str) -> str:
    assert _BODY.count(old) == 1
    return _s(_BODY.replace(old, new))


# What a learner gets with the known mistakes, per value of relationship (tests/test_rai_unawareness_proxies.py
# checks these against the table).
_RICH = {"Husband": 0.4486, "Not-in-family": 0.1021, "Other-relative": 0.0352, "Own-child": 0.0146,
         "Unmarried": 0.0602, "Wife": 0.4689}
_WOMEN = {"Husband": 0.0001, "Not-in-family": 0.4665, "Other-relative": 0.4575, "Own-child": 0.4453,
          "Unmarried": 0.7664, "Wife": 0.9987}
_TOTAL = {"Husband": 0.4036, "Not-in-family": 0.1374, "Other-relative": 0.0167, "Own-child": 0.0861,
          "Unmarried": 0.0245, "Wife": 0.0001}

EXERCISES = {
    "proxy": CodeBox(
        name="proxy",
        picture=_share_picture,
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
                    en="Your code uses a name with an underscore, such as `_people`. In the box the names have none: `people`, `column`, `men`, `share`.",
                    nl="Je code gebruikt een naam met een liggend streepje, zoals `_people`. In het vak hebben de namen dat niet: `people`, `column`, `men`, `share`."),
            Mistake("column", error=lambda e: isinstance(e, KeyError),
                    en="Your code asks the table for a column it does not have. Group by the name `column` itself, without quotes: it holds the text `\"relationship\"`. The sex of each person is in the column `sex`, and `men` must be rows of the table: `people[people[\"sex\"] == \"Male\"]`.",
                    nl="Je code vraagt de tabel om een kolom die hij niet heeft. Groepeer op de naam `column` zelf, zonder aanhalingstekens: daarin staat de tekst `\"relationship\"`. Het geslacht van elke persoon staat in de kolom `sex`, en `men` moet rijen van de tabel bevatten: `people[people[\"sex\"] == \"Male\"]`."),
            Mistake("rich", matches=_like(_RICH),
                    en="Your last line gives the share that earns more than 50,000 dollars, as in the worked example. Here you want the share of men: pick the rows where `sex` is `\"Male\"`.",
                    nl="Je laatste regel geeft het deel dat meer dan 50.000 dollar verdient, zoals in het uitgewerkte voorbeeld. Hier wil je het deel mannen: kies de rijen waarin `sex` gelijk is aan `\"Male\"`."),
            Mistake("women", matches=_like(_WOMEN),
                    en="Your last line gives the share of women per value. The question is the share of men: pick the rows where `sex` is `\"Male\"`.",
                    nl="Je laatste regel geeft het deel vrouwen per waarde. De vraag is het deel mannen: kies de rijen waarin `sex` gelijk is aan `\"Male\"`."),
            Mistake("total", matches=_like(_TOTAL),
                    en="Your shares add up to 0.67, the share of men in the whole table: you divided by all 48,842 people. Divide by the people with that value: `people.groupby(column).size()`.",
                    nl="Je delen tellen op tot 0,67, het deel mannen in de hele tabel: je deelde door alle 48.842 mensen. Deel door de mensen met die waarde: `people.groupby(column).size()`."),
            Mistake("counts", matches=_counts,
                    en="Your last line gives how many people (or men) have each value, not a share. Divide by how many people have that value: `people.groupby(column).size()`.",
                    nl="Je laatste regel geeft hoeveel mensen (of mannen) elke waarde hebben, niet een deel. Deel door hoeveel mensen die waarde hebben: `people.groupby(column).size()`."),
            Mistake("upside", matches=lambda got, text: _upside(got),
                    en="Your shares are upside down: you divided all people by the men. Divide the men by all people: `men.groupby(column).size() / people.groupby(column).size()`.",
                    nl="Je delen staan op hun kop: je deelde alle mensen door de mannen. Deel de mannen door alle mensen: `men.groupby(column).size() / people.groupby(column).size()`."),
            Mistake("by-sex", matches=_by_sex,
                    en="Your last line has a value per sex, because you grouped by `sex`. Group by the column you are checking: `groupby(column)`.",
                    nl="Je laatste regel heeft een waarde per geslacht, omdat je op `sex` groepeerde. Groepeer op de kolom die je controleert: `groupby(column)`."),
            Mistake("lowercase", matches=_all_missing,
                    en="Every share is `NaN`: `men` has no rows. The values in `sex` start with a capital: `\"Male\"`.",
                    nl="Elk deel is `NaN`: `men` heeft geen rijen. De waarden in `sex` beginnen met een hoofdletter: `\"Male\"`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the two lines of the worked example, with the men instead of the people who earn more than 50,000 dollars.",
                      "nl": "Vervang de `...` door de twee regels van het uitgewerkte voorbeeld, met de mannen in plaats van de mensen die meer dan 50.000 dollar verdienen."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the lines of `people` and `column` exactly as they were: put them back. The check tries another column by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `people` en `column` nodig zoals ze waren: zet ze terug. De check probeert zelf een andere kolom."},
            "no_value": {"en": "The last line of the box shows nothing. Put `share` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `share` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Does `men` hold the rows where `sex` is `\"Male\"`, and is `share` the men per value divided by all people per value?",
                      "nl": "Je laatste regel geeft `{got}`. Bevat `men` de rijen waarin `sex` gelijk is aan `\"Male\"`, en is `share` de mannen per waarde gedeeld door alle mensen per waarde?"},
            "other": {"en": "Right for relationship. But with {where}, your last line gives `{got}`, and it should give `{want}`. Do you group by `column` itself, and not by the text `\"relationship\"`?",
                      "nl": "Goed voor relationship. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Groepeer je op `column` zelf, en niet op de tekst `\"relationship\"`?"},
            "other_error": {"en": "Right for relationship. But with {where}, your code stops with an error. Do you use `column` and `people` themselves?",
                            "nl": "Goed voor relationship. Maar met {where} stopt je code met een fout. Gebruik je `column` en `people` zelf?"},
            "correct": {"en": "Husband about `1.0` and Wife about `0.0`: of the 19,716 husbands all but one are men, and of the 2,331 wives all but three are women. So for 22,043 of the 48,842 people, the column relationship gives away their sex, while the model never sees the column sex.",
                        "nl": "Husband ongeveer `1.0` en Wife ongeveer `0.0`: van de 19.716 echtgenoten zijn er op één na allemaal man, en van de 2.331 echtgenotes op drie na allemaal vrouw. Dus voor 22.043 van de 48.842 mensen verraadt de kolom relationship hun geslacht, terwijl het model de kolom sex nooit ziet."},
        },
    ),
    "proxies": Sort(
        name="proxies",
        groups={
            "ethnicity": {"en": "Can stand in for ethnicity", "nl": "Kan staan voor afkomst"},
            "sex": {"en": "Can stand in for sex", "nl": "Kan staan voor geslacht"},
        },
        items={
            "postcode": {"en": "**Postcode.** The postcode of where an applicant lives.",
                         "nl": "**Postcode.** De postcode van het adres van de sollicitant."},
            "surname": {"en": "**Surname.** The last name on an application.",
                        "nl": "**Achternaam.** De achternaam op een sollicitatie."},
            "height": {"en": "**Height.** Only applicants taller than 1.70 meters get through, as everyone on the current staff is.",
                       "nl": "**Lengte.** Alleen wie langer is dan 1,70 meter, komt door, omdat iedereen in het huidige personeel dat is."},
            "club": {"en": "**A club on a resume.** \"Captain of the women's chess club\".",
                     "nl": "**Een club op een cv.** Aanvoerder van de schaakclub voor vrouwen."},
            "gap": {"en": "**A gap in the resume.** A few years without work between two jobs.",
                    "nl": "**Een gat in het cv.** Een paar jaar zonder werk tussen twee banen."},
            "shoes": {"en": "**Shoe size.** The size of the work shoes someone ordered.",
                      "nl": "**Schoenmaat.** De maat van de werkschoenen die iemand bestelde."},
        },
        solution={"postcode": "ethnicity", "surname": "ethnicity", "height": "sex", "club": "sex", "gap": "sex",
                  "shoes": "sex"},
        nudge={
            "postcode": {"en": "Where people live is often not mixed. What can a neighborhood say about who lives there?",
                         "nl": "Waar mensen wonen, is vaak niet gemengd. Wat kan een wijk zeggen over wie er woont?"},
            "surname": {"en": "What does a last name often say about where a family comes from?",
                        "nl": "Wat zegt een achternaam vaak over waar een familie vandaan komt?"},
            "height": {"en": "Who is on average shorter?",
                       "nl": "Wie is gemiddeld kleiner?"},
            "club": {"en": "Read the name of the club again.",
                     "nl": "Lees de naam van de club nog eens."},
            "gap": {"en": "Who more often stops working for a while when a child is born?",
                    "nl": "Wie stopt vaker een tijdje met werken als er een kind komt?"},
            "shoes": {"en": "Who on average has bigger feet?",
                      "nl": "Wie heeft gemiddeld grotere voeten?"},
        },
        texts={
            "missing": {"en": "I cannot see your choices. Run the cell with the pieces of data again.",
                        "nl": "Ik zie je keuzes niet. Voer de cel met de gegevens opnieuw uit."},
            "empty": {"en": "Sort each piece of data by what it can give away.",
                      "nl": "Deel elk gegeven in bij wat het kan verraden."},
            "unfinished": {"en": "Everything you sorted so far is right. Sort the rest too ({n} to go).",
                           "nl": "Alles wat je tot nu toe indeelde, klopt. Deel ook de rest in (nog {n})."},
            "wrong": {"en": "The piece of data *{item}* belongs somewhere else.",
                      "nl": "Het gegeven *{item}* hoort ergens anders."},
            "correct": {"en": "All {n} are right. Postcode and surname can stand in for ethnicity; height, the club, the gap and shoe size for sex. None of them names a protected ground, and still a model can learn one from them.",
                        "nl": "Alle {n} kloppen. Postcode en achternaam kunnen staan voor afkomst; lengte, de club, het gat en de schoenmaat voor geslacht. Geen ervan noemt een discriminatiegrond, en toch kan een model er een uit leren."},
        },
    ),
}


def _upside(got) -> bool:
    """Whether 1 divided by each value of the learner's Series gives the shares of men per value of relationship:
    the division turned around."""
    import pandas as pd
    if not isinstance(got, pd.Series) or len(got) == 0:
        return False
    try:
        return _close(1 / got.astype(float), pd.Series(share(COLUMN)))
    except (TypeError, ValueError, ZeroDivisionError):
        return False


_SORTED = {"postcode": "ethnicity", "surname": "ethnicity", "height": "sex", "club": "sex", "gap": "sex",
           "shoes": "sex"}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("proxy", _START, "empty", None),
    ("proxy", _SOLUTION, "correct", None),
    ("proxy", _s('men = people[people["sex"] == "Male"]\n'
                 "share = (men.groupby(column).size() / people.groupby(column).size()).round(2)\n"), "correct", None),
    ("proxy", _s('men = people[people["sex"] != "Female"]\nper_value = people.groupby(column).size()\n'
                 "share = men.groupby(column).size() / per_value\n"), "correct", None),
    ("proxy", _s(_BODY.replace("men.groupby", "_men.groupby")), "error", "underscore"),
    ("proxy", _swap("men.groupby(column)", 'men.groupby("column")'), "error", "column"),
    ("proxy", _swap('people[people["sex"] == "Male"]', 'people["sex"] == "Male"'), "error", "column"),
    ("proxy", _s('men = people[people["sex"] == "Male"]\nshare = men[column]\n'), "wrong", None),
    ("proxy", _swap('people["sex"] == "Male"', 'people["income"] == ">50K"'), "wrong", "rich"),
    ("proxy", _swap('"Male"', '"Female"'), "wrong", "women"),
    ("proxy", _swap("people.groupby(column).size()", "len(people)"), "wrong", "total"),
    ("proxy", _swap(" / people.groupby(column).size()", ""), "wrong", "counts"),
    ("proxy", _swap("men.groupby(column).size() / people.groupby(column).size()",
                    "people.groupby(column).size() / men.groupby(column).size()"), "wrong", "upside"),
    ("proxy", _s(_BODY.replace("groupby(column)", 'groupby("sex")')), "wrong", "by-sex"),
    ("proxy", _swap('"Male"', '"male"'), "wrong", "lowercase"),
    ("proxy", _s(_BODY.replace("groupby(column)", 'groupby("race")')), "wrong", None),
    ("proxy", _s(_BODY.replace("groupby(column)", 'groupby("relationship")')), "almost", "other-value"),
    ("proxy", _SOLUTION.replace(_COLUMN_LINE, 'column = "occupation"'), "wrong", "changed-line"),
    ("proxy", _s("...\nshare = 1\n"), "wrong", "dots-left"),
    ("proxies", {}, "empty", None),
    ("proxies", _SORTED, "correct", None),
    ("proxies", {"postcode": "ethnicity", "height": "sex"}, "wrong", "unfinished"),
    ("proxies", {**_SORTED, "surname": "sex"}, "wrong", "wrong:surname"),
    ("proxies", {**_SORTED, "gap": "ethnicity"}, "wrong", "wrong:gap"),
    ("proxies", {**_SORTED, "postcode": "sex"}, "wrong", "wrong:postcode"),
]
