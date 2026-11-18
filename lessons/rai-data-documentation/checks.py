# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-data-documentation: two numbers for a data card of UCI Adult, worked out by the learner,
who is in it (the share of each group) and how often the occupation is unknown per group; with the pictures of both."""
import functools
import html
import re

from everything_ai.check import CodeBox, Mistake
from everything_ai import paths, yours

DATA = paths.ROOT / "data" / "adult.csv.gz"
COLUMNS_WITH_UNKNOWNS = ["workclass", "occupation", "native_country"]


@functools.cache
def _people():
    import pandas as pd
    return pd.read_csv(DATA)


def shares(column: str) -> dict:
    """The share of all rows of UCI Adult that each group of `column` has."""
    people = _people()
    return {str(k): float(v) for k, v in (people[column].value_counts() / len(people)).items()}


def unknown(column: str, where: str = "occupation") -> dict:
    """Per group of `column`, the share of its people whose `where` is unknown."""
    people = _people()
    missing = people[people[where].isna()]
    out = missing[column].value_counts() / people[column].value_counts()
    return {str(k): float(v) for k, v in out.items()}


def _number(x) -> float | None:
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def _per_group(got) -> dict | None:
    """A value per group, from a Series or a dict with numbers under names, as {name: number}; None otherwise."""
    import pandas as pd
    if isinstance(got, pd.Series):
        items = list(got.items())
    elif type(got) is dict:
        items = list(got.items())
    else:
        return None
    out = {}
    for k, v in items:
        n = _number(v)
        if n is None:
            return None
        out[str(k)] = n
    return out


def _same_groups(got, want) -> bool:
    """The same number under each name, whatever the order of the groups; `want` a Series or a dict."""
    mine, want = _per_group(got), _per_group(want)
    return (mine is not None and want is not None and mine.keys() == want.keys()
            and all(abs(mine[k] - want[k]) < 1e-6 for k in want))


def _is(make):
    """A mistake that matches the value per group that `make(column)` gives for the box's own column."""
    def match(got, text):
        m = re.search(r'^column = "(\w+)"$', text, re.MULTILINE)
        return m is not None and m.group(1) in ("race", "sex") and _same_groups(got, make(m.group(1)))
    return match


def _show(x: float, lang: str, digits: int = 3) -> str:
    text = f"{x:.{digits}f}"
    return text.replace(".", ",") if lang == "nl" else text


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"


def _bars(seen, words: dict, line: tuple | None) -> str:
    """The learner's value per group as horizontal bars, the name and the value above each; with `line`, a dashed line
    at (value, words) for everyone together."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None or not mine.has_value:
        return ""
    values = _per_group(mine.value)
    if not values or len(values) > 8 or not all(0 <= v <= 1 for v in values.values()):
        return ""
    names = mine.names if isinstance(mine.names, dict) else {}
    column = names.get("column") if isinstance(names.get("column"), str) else "?"
    w = words[seen.lang]
    rows = sorted(values.items(), key=lambda kv: -kv[1])
    top = max([v for _, v in rows] + ([line[0]] if line else []))
    right = 1.0 if top > 0.5 else (0.5 if top > 0.2 else 0.2 if top > 0.1 else 0.1)
    ticks = [right * i / 4 for i in range(5)]
    height = 1.1 * len(rows) + 0.5
    p = chart.Panel(x=(0, right * 1.05), y=(0, height), xlabel=w["xlabel"], yticks=[], xticks=ticks,
                    xtick_labels=[_show(t, seen.lang, 2 if right >= 0.2 else 3) for t in ticks])
    for i, (name, v) in enumerate(rows):
        y = height - 0.5 - 1.1 * (i + 1)
        p.text(right * 0.004, y + 0.68, f"{name}: {_show(v, seen.lang)}", "ink", anchor="start", halo=True)
        p.rect(0, y + 0.12, max(v, right * 0.003), y + 0.42, role="data", solid=True)
    if line:
        p.line([line[0], line[0]], [0, height - 0.3], role="ink", width=1.5, dashed=True)
        p.text(line[0], height - 0.15, line[1], "ink-2", halo=True)
    shown = "; ".join(f"{name} {_show(v, seen.lang)}" for name, v in rows)
    svg = chart.figure([p], alt=w["alt"].format(column=column, shown=shown), width=520, height=46 * len(rows) + 100)
    svg = chart.with_decimal_comma(svg) if seen.lang == "nl" else svg
    title = w["title"].format(column=column)
    return yours.figure(title, svg + f'<div style="{_UI}">{html.escape(w["key"])}</div>')


_SHARE_WORDS = {
    "en": {"title": "Your shares per group of {column}", "xlabel": "share of all 48,842 people",
           "alt": "Your shares per group of {column}, as bars, largest first: {shown}.",
           "key": "a bar per group, as long as its share of all rows; together they make 1"},
    "nl": {"title": "Jouw aandelen per groep van {column}", "xlabel": "aandeel van alle 48.842 mensen",
           "alt": "Jouw aandelen per groep van {column}, als staven, de grootste eerst: {shown}.",
           "key": "een staaf per groep, zo lang als het aandeel van die groep in alle rijen; samen is dat 1"},
}

_UNKNOWN_WORDS = {
    "en": {"title": "Your share with an unknown occupation, per group of {column}",
           "xlabel": "share of the group's people whose occupation is unknown",
           "alt": "Your share with an unknown occupation per group of {column}, as bars, largest first: {shown}. A dashed "
                  "line marks everyone together.",
           "key": "a bar per group; the dashed line is the share for all 48,842 people together", "line": "everyone: {v}"},
    "nl": {"title": "Jouw aandeel met een onbekend beroep, per groep van {column}",
           "xlabel": "aandeel van de mensen in de groep van wie het beroep onbekend is",
           "alt": "Jouw aandeel met een onbekend beroep per groep van {column}, als staven, de grootste eerst: {shown}. "
                  "Een stippellijn geeft iedereen samen.",
           "key": "een staaf per groep; de stippellijn is het aandeel voor alle 48.842 mensen samen", "line": "iedereen: {v}"},
}


def _shares_picture(seen) -> str:
    return _bars(seen, _SHARE_WORDS, None)


def _unknown_picture(seen) -> str:
    people = _people()
    everyone = float(people["occupation"].isna().mean())
    return _bars(seen, _UNKNOWN_WORDS, (everyone, _UNKNOWN_WORDS[seen.lang]["line"].format(v=_show(everyone, seen.lang))))


# The lesson is done when these are correct. "fair" and "write" are the choice questions in the lesson.
REQUIRED = ["shares", "fair", "unknown", "write"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_UNDERSCORE = Mistake("underscore", error=lambda e: isinstance(e, NameError) and "'_" in str(e),
                      en="Your code uses a name with an underscore, such as `_people`. In the box the names have none: `people`, `column`.",
                      nl="Je code gebruikt een naam met een liggend streepje, zoals `_people`. In het vak hebben de namen dat niet: `people`, `column`.")
_COLUMN = Mistake("column", error=lambda e: isinstance(e, KeyError),
                  en="Your code asks the table for a column it does not have. Write the names as the table does, such as `occupation`, and use `column` for the groups: `people[column]`.",
                  nl="Je code vraagt de tabel om een kolom die hij niet heeft. Schrijf de namen zoals de tabel ze schrijft, zoals `occupation`, en gebruik `column` voor de groepen: `people[column]`.")
_MISSING = {"en": "I cannot see the box. Run the cell with the box again.",
            "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."}
_CLEARED = {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
            "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."}

_READ_LINE = 'people = pd.read_csv("data/adult.csv.gz")'


def _head(column: str, what: str) -> str:
    return ("import pandas as pd\n\n# 48,842 people from a US survey of 1994 (UCI Adult)\n" + _READ_LINE + "\n"
            + f"# {what}\n" + f'column = "{column}"\n\n')


def _swap(head: str, body: str, last: str):
    def swap(old: str, new: str) -> str:
        assert body.count(old) == 1
        return head + body.replace(old, new) + last
    return swap


def _counts(column: str) -> dict:
    people = _people()
    return {str(k): float(v) for k, v in people[column].value_counts().items()}


# --- exercise 1: who is in it ---------------------------------------------------------------------------------------
_SH_LINE = 'column = "race"'
_SH_HEAD = _head("race", "the column with the groups")
_SH_START = _SH_HEAD + "...\nshares\n"
_SH_BODY = "counts = people[column].value_counts()\nshares = counts / len(people)\n"
_SH_SOLUTION = _SH_HEAD + _SH_BODY + "shares"
_sh = _swap(_SH_HEAD, _SH_BODY, "shares")

_PERCENT = lambda column: {k: 100 * v for k, v in shares(column).items()}                       # noqa: E731
_OF_GROUPS = lambda column: {k: v / len(_counts(column)) for k, v in _counts(column).items()}  # noqa: E731
_UPSIDE = lambda column: {k: 1 / v for k, v in shares(column).items()}                        # noqa: E731
_TO_LARGEST = lambda column: {k: v / max(_counts(column).values()) for k, v in _counts(column).items()}  # noqa: E731

# --- exercise 3: what is unknown ------------------------------------------------------------------------------------
_UN_LINE = 'column = "sex"'
_UN_HEAD = _head("sex", "the column with the groups")
_UN_START = _UN_HEAD + "...\nunknown\n"
_UN_BODY = ('missing = people[people["occupation"].isna()]\n'
            "unknown = missing[column].value_counts() / people[column].value_counts()\n")
_UN_SOLUTION = _UN_HEAD + _UN_BODY + "unknown"
_un = _swap(_UN_HEAD, _UN_BODY, "unknown")


def _missing_rows(column: str) -> dict:
    people = _people()
    return {str(k): float(v) for k, v in people[people["occupation"].isna()][column].value_counts().items()}


_UN_COUNTS = _missing_rows
_UN_OF_ALL = lambda column: {k: v / len(_people()) for k, v in _missing_rows(column).items()}           # noqa: E731
_UN_OF_MISSING = lambda column: {k: v / sum(_missing_rows(column).values())                              # noqa: E731
                                 for k, v in _missing_rows(column).items()}
_UN_KNOWN = lambda column: {k: 1 - v for k, v in unknown(column).items()}                               # noqa: E731
_UN_PERCENT = lambda column: {k: 100 * v for k, v in unknown(column).items()}                           # noqa: E731
_UN_UPSIDE = lambda column: {k: 1 / v for k, v in unknown(column).items()}                              # noqa: E731


def _other_column(got, text) -> bool:
    m = re.search(r'^column = "(\w+)"$', text, re.MULTILINE)
    if m is None or m.group(1) not in ("race", "sex"):
        return False
    return any(_same_groups(got, unknown(m.group(1), where)) for where in COLUMNS_WITH_UNKNOWNS if where != "occupation")


EXERCISES = {
    "shares": CodeBox(
        name="shares",
        picture=_shares_picture,
        start=_SH_START,
        solution=_SH_SOLUTION,
        keep=[_READ_LINE, _SH_LINE],
        compare=_same_groups,
        # The check also runs the box with the groups of another column.
        others=[{"column": "sex"}],
        first=[_DOTS],
        digits=3,
        mistakes=[
            _UNDERSCORE,
            _COLUMN,
            Mistake("counts", matches=_is(_counts),
                    en="Your last line gives the number of people in each group. A share is that number divided by the number of all rows: `len(people)`.",
                    nl="Je laatste regel geeft het aantal mensen in elke groep. Een aandeel is dat aantal gedeeld door het aantal van alle rijen: `len(people)`."),
            Mistake("percent", matches=_is(_PERCENT),
                    en="Your shares are 100 times too large: percentages. A share is a number between 0 and 1; leave out the `* 100`.",
                    nl="Je aandelen zijn 100 keer te groot: percentages. Een aandeel is een getal tussen 0 en 1; laat de `* 100` weg."),
            Mistake("of-groups", matches=_is(_OF_GROUPS),
                    en="You divided by the number of groups. Divide by the number of people: `len(people)` counts the rows of the table.",
                    nl="Je deelde door het aantal groepen. Deel door het aantal mensen: `len(people)` telt de rijen van de tabel."),
            Mistake("upside", matches=_is(_UPSIDE),
                    en="Your shares are upside down: all the people divided by each group. Divide the count of each group by the number of all rows.",
                    nl="Je aandelen staan op hun kop: alle mensen gedeeld door elke groep. Deel het aantal van elke groep door het aantal van alle rijen."),
            Mistake("to-largest", matches=_is(_TO_LARGEST),
                    en="You divided by the largest group, so it gets 1. A share of all the people divides by the number of all rows: `len(people)`.",
                    nl="Je deelde door de grootste groep, zodat die 1 krijgt. Een aandeel van alle mensen deel je door het aantal van alle rijen: `len(people)`."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with your code: count the people per group of `column`, and divide by the number of all rows.",
                      "nl": "Vervang de `...` door je code: tel de mensen per groep van `column`, en deel door het aantal van alle rijen."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the lines of `people` and `column` exactly as they were: put them back. The check tries another column by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `people` en `column` nodig zoals ze waren: zet ze terug. De check probeert zelf een andere kolom."},
            "no_value": {"en": "The last line of the box shows nothing. Put `shares` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `shares` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Is it, for each group of `column`, the number of its people divided by the number of all rows?",
                      "nl": "Je laatste regel geeft `{got}`. Is dat voor elke groep van `column` het aantal mensen gedeeld door het aantal van alle rijen?"},
            "other": {"en": "Right for `column = \"race\"`. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does your code use `column` itself, and not `\"race\"`?",
                      "nl": "Goed voor `column = \"race\"`. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruikt je code `column` zelf, en niet `\"race\"`?"},
            "other_error": {"en": "Right for `column = \"race\"`. But with {where}, your code stops with an error. Does your code use `column` itself?",
                            "nl": "Goed voor `column = \"race\"`. Maar met {where} stopt je code met een fout. Gebruikt je code `column` zelf?"},
            "correct": {"en": "White `0.855`, Black `0.096`, Asian-Pac-Islander `0.031`, Amer-Indian-Eskimo `0.010`, Other `0.008`; and with `column = \"sex\"`, men `0.668` and women `0.332`. On a data card these shares go under who is in the data, with the names the data uses: Fabris and colleagues call the class \"Asian Pacific Islander\" outdated, and the card can say so.",
                        "nl": "White `0.855`, Black `0.096`, Asian-Pac-Islander `0.031`, Amer-Indian-Eskimo `0.010`, Other `0.008`; en met `column = \"sex\"` mannen `0.668` en vrouwen `0.332`. Op een data card staan deze aandelen onder wie er in de data zit, met de namen die de data gebruikt: Fabris en collega's noemen de groep Asian Pacific Islander verouderd, en de card kan dat zeggen."},
        },
    ),
    "unknown": CodeBox(
        name="unknown",
        picture=_unknown_picture,
        start=_UN_START,
        solution=_UN_SOLUTION,
        keep=[_READ_LINE, _UN_LINE],
        compare=_same_groups,
        # The check also runs the box with the groups of another column.
        others=[{"column": "race"}],
        first=[_DOTS],
        digits=3,
        mistakes=[
            _UNDERSCORE,
            _COLUMN,
            Mistake("counts", matches=_is(_UN_COUNTS),
                    en="Your last line gives the number of people with an unknown occupation in each group. Divide by the number of people in each group: `people[column].value_counts()`.",
                    nl="Je laatste regel geeft het aantal mensen met een onbekend beroep in elke groep. Deel door het aantal mensen in elke groep: `people[column].value_counts()`."),
            Mistake("of-all", matches=_is(_UN_OF_ALL),
                    en="You divided by the number of all people. Then a large group gets a large number, even if its occupation is unknown less often. Divide by the people in each group: `people[column].value_counts()`.",
                    nl="Je deelde door het aantal van alle mensen. Dan krijgt een grote groep een groot getal, ook als het beroep er minder vaak onbekend is. Deel door de mensen in elke groep: `people[column].value_counts()`."),
            Mistake("of-missing", matches=_is(_UN_OF_MISSING),
                    en="Your shares add up to 1: they say how the people with an unknown occupation are divided over the groups. The exercise asks, per group, which share of its own people has an unknown occupation: divide by `people[column].value_counts()`.",
                    nl="Je aandelen tellen op tot 1: ze zeggen hoe de mensen met een onbekend beroep over de groepen verdeeld zijn. De oefening vraagt per groep welk deel van de eigen mensen een onbekend beroep heeft: deel door `people[column].value_counts()`."),
            Mistake("known", matches=_is(_UN_KNOWN),
                    en="Your shares are those of the people whose occupation is known. Take the rows where it is unknown: `people[\"occupation\"].isna()`.",
                    nl="Je aandelen zijn die van de mensen van wie het beroep bekend is. Neem de rijen waarin het onbekend is: `people[\"occupation\"].isna()`."),
            Mistake("other-column", matches=_other_column,
                    en="Your shares are of an unknown value in another column than `occupation`. Take the rows where `people[\"occupation\"].isna()`.",
                    nl="Je aandelen gaan over een onbekende waarde in een andere kolom dan `occupation`. Neem de rijen waarin `people[\"occupation\"].isna()`."),
            Mistake("percent", matches=_is(_UN_PERCENT),
                    en="Your shares are 100 times too large: percentages. A share is a number between 0 and 1; leave out the `* 100`.",
                    nl="Je aandelen zijn 100 keer te groot: percentages. Een aandeel is een getal tussen 0 en 1; laat de `* 100` weg."),
            Mistake("upside", matches=_is(_UN_UPSIDE),
                    en="Your shares are upside down: the people of each group divided by those of them with an unknown occupation. Divide the other way round.",
                    nl="Je aandelen staan op hun kop: de mensen van elke groep gedeeld door die met een onbekend beroep. Deel andersom."),
        ],
        texts={
            "missing": _MISSING,
            "empty": {"en": "Replace the `...` with your code: take the rows where `occupation` is unknown, count them per group of `column`, and divide by the number of people in each group.",
                      "nl": "Vervang de `...` door je code: neem de rijen waarin `occupation` onbekend is, tel ze per groep van `column`, en deel door het aantal mensen in elke groep."},
            "cleared": _CLEARED,
            "keep": {"en": "Your code may be right, but the check needs the lines of `people` and `column` exactly as they were: put them back. The check tries another column by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regels van `people` en `column` nodig zoals ze waren: zet ze terug. De check probeert zelf een andere kolom."},
            "no_value": {"en": "The last line of the box shows nothing. Put `unknown` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `unknown` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives `{got}`. Is it, for each group of `column`, the number of its people with an unknown `occupation` divided by the number of its people?",
                      "nl": "Je laatste regel geeft `{got}`. Is dat voor elke groep van `column` het aantal mensen met een onbekende `occupation` gedeeld door het aantal mensen in die groep?"},
            "other": {"en": "Right for `column = \"sex\"`. But with {where}, your last line gives `{got}`, and it should give `{want}`. Does your code use `column` itself, and not `\"sex\"`?",
                      "nl": "Goed voor `column = \"sex\"`. Maar met {where} geeft je laatste regel `{got}`, en dat moet `{want}` zijn. Gebruikt je code `column` zelf, en niet `\"sex\"`?"},
            "other_error": {"en": "Right for `column = \"sex\"`. But with {where}, your code stops with an error. Does your code use `column` itself?",
                            "nl": "Goed voor `column = \"sex\"`. Maar met {where} stopt je code met een fout. Gebruikt je code `column` zelf?"},
            "correct": {"en": "Women `0.079`, men `0.047`: the occupation is unknown for a larger share of the women. Divided by all people instead, the order turns around (`0.026` against `0.031`), because there are twice as many men. A team that drops every row with an unknown value, as the example in AIF360's documentation does with 3,620 rows, drops 9.2 percent of the women and 6.5 percent of the men. That belongs on the data card.",
                        "nl": "Vrouwen `0.079`, mannen `0.047`: het beroep is bij een groter deel van de vrouwen onbekend. Gedeeld door alle mensen draait de volgorde om (`0.026` tegen `0.031`), omdat er twee keer zoveel mannen zijn. Een team dat elke rij met een onbekende waarde weglaat, zoals het voorbeeld in de documentatie van AIF360 doet met 3.620 rijen, laat 9,2 procent van de vrouwen weg en 6,5 procent van de mannen. Dat hoort op de data card."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_LOOP = ("counts = {}\nfor value in people[column]:\n    if value in counts:\n        counts[value] = counts[value] + 1\n"
         "    else:\n        counts[value] = 1\nshares = {}\nfor group, count in counts.items():\n"
         "    shares[group] = count / len(people)\n")
CASES = [
    ("shares", _SH_START, "empty", None),
    ("shares", _SH_SOLUTION, "correct", None),
    ("shares", _SH_HEAD + "shares = people[column].value_counts(normalize=True)\nshares", "correct", None),
    ("shares", _SH_HEAD + _LOOP + "shares", "correct", None),
    ("shares", _sh("people[column]", "_people[column]"), "error", "underscore"),
    ("shares", _sh("people[column]", "people[\"races\"]"), "error", "column"),
    ("shares", _sh("counts / len(people)", "counts"), "wrong", "counts"),
    ("shares", _sh("counts / len(people)", "100 * counts / len(people)"), "wrong", "percent"),
    ("shares", _sh("counts / len(people)", "counts / len(counts)"), "wrong", "of-groups"),
    ("shares", _sh("counts / len(people)", "len(people) / counts"), "wrong", "upside"),
    ("shares", _sh("counts / len(people)", "counts / counts.max()"), "wrong", "to-largest"),
    ("shares", _sh("counts / len(people)", "counts / (len(people) + 1)"), "wrong", None),
    ("shares", _sh("people[column]", "people[\"race\"]"), "almost", "other-value"),
    ("shares", _SH_SOLUTION.replace('column = "race"', 'column = "sex"'), "wrong", "changed-line"),
    ("shares", _SH_HEAD + "...\nshares = 0.5\nshares", "wrong", "dots-left"),
    ("unknown", _UN_START, "empty", None),
    ("unknown", _UN_SOLUTION, "correct", None),
    ("unknown", _UN_HEAD + 'unknown = (people[people["occupation"].isna()][column].value_counts()\n'
                           "           / people[column].value_counts())\nunknown", "correct", None),
    ("unknown", _un("people[people", "_people[people"), "error", "underscore"),
    ("unknown", _un('people["occupation"]', 'people["Occupation"]'), "error", "column"),
    ("unknown", _un(" / people[column].value_counts()", ""), "wrong", "counts"),
    ("unknown", _un("people[column].value_counts()\n", "len(people)\n"), "wrong", "of-all"),
    ("unknown", _un("people[column].value_counts()\n", "len(missing)\n"), "wrong", "of-missing"),
    ("unknown", _un(".isna()", ".notna()"), "wrong", "known"),
    ("unknown", _un('"occupation"', '"workclass"'), "wrong", "other-column"),
    ("unknown", _un("unknown = missing", "unknown = 100 * missing"), "wrong", "percent"),
    ("unknown", _un("missing[column].value_counts() / people[column].value_counts()",
                    "people[column].value_counts() / missing[column].value_counts()"), "wrong", "upside"),
    ("unknown", _un("people[column].value_counts()\n", "(people[column].value_counts() + 1)\n"), "wrong", None),
    ("unknown", _un("missing[column]", 'missing["sex"]'), "almost", "other-value"),
    ("unknown", _UN_SOLUTION.replace('column = "sex"', 'column = "race"'), "wrong", "changed-line"),
    ("unknown", _UN_HEAD + "...\nunknown = 0.5\nunknown", "wrong", "dots-left"),
]
