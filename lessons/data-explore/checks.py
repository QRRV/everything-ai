# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson data-explore: the fraction of missing values per column of a month of measurements from a made-up
weather station, and a function that counts how often each value of a column appears. And the picture."""
import re

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake, same_value

STATION = "lessons/data-explore/station.csv"
STATION_2 = "lessons/data-explore/station-2.csv"


def picture(labels: dict[str, str]) -> str:
    """The month of the station as two line charts, one above the other: the temperature, with gaps where it is
    missing, and the wind, which dives to -999 on one day."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from everything_ai import plots
    df = pd.read_csv(STATION)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(5.6, 4.2), sharex=True)
    top.plot(df["day"], df["temp"])
    top.set_ylabel(labels["temp"])
    top.set_title(labels["title"])
    bottom.plot(df["day"], df["wind"])
    bottom.set_ylabel(labels["wind"])
    bottom.set_xlabel(labels["day"])
    fig.align_ylabels()
    fig.tight_layout()
    html = plots.render(fig, alt=labels["alt"])
    plt.close(fig)
    return html


# The lesson is done when these are correct. "error-code" is the choice question in the lesson.
REQUIRED = ["error-code", "missing", "counts"]

_READ = f'df = pd.read_csv("{STATION}")'
_MISSING_START = "import pandas as pd\n\n" + _READ + "\nmissing = ...\nmissing\n"
_MISSING = "import pandas as pd\n\n" + _READ + "\nmissing = df.isna().mean()\nmissing"

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_NO_PARENTHESES = Mistake("no-parentheses", error=lambda e: isinstance(e, AttributeError) and "'function' object has no attribute" in str(e)
                          or isinstance(e, AttributeError) and "'method' object has no attribute" in str(e),
                          en="`isna` needs its parentheses, also with nothing between them: `df.isna()`, not `df.isna`. Without them, you get `isna` itself, not what it gives.",
                          nl="`isna` heeft zijn haakjes nodig, ook als er niets tussen staat: `df.isna()`, niet `df.isna`. Zonder haakjes krijg je `isna` zelf, niet wat het geeft.")
_NO_COLUMN = Mistake("no-column", error=lambda e: isinstance(e, KeyError) and bool(e.args) and isinstance(e.args[0], str) and e.args[0] != "column",
                     en="The table has no column with that name. Its columns are `day`, `temp`, `rain`, `wind`, `dir` and `sky`, in small letters, each between quotes.",
                     nl="De tabel heeft geen kolom met die naam. De kolommen zijn `day`, `temp`, `rain`, `wind`, `dir` en `sky`, in kleine letters, elk tussen aanhalingstekens.")
_ITSELF = Mistake("itself", matches=lambda got, text: callable(got),
                  en="Your code gives back something that still needs its parentheses: such as `.mean` instead of `.mean()`. Without them, you get the thing itself, not what it gives.",
                  nl="Je code geeft iets terug dat nog zijn haakjes nodig heeft: zoals `.mean` in plaats van `.mean()`. Zonder haakjes krijg je het ding zelf, niet wat het geeft.")


def _missing(new: str) -> str:
    return _MISSING.replace("missing = df.isna().mean()", new)


def _number(got) -> bool:
    return type(got).__name__ in ("int", "int64", "float", "float64") and not isinstance(got, bool)


def _as_counts(value):
    """How often each value appears, as a dictionary, from a Series of numbers (such as value_counts gives) or a
    dictionary. Anything else, such as a column of text, stays as it is."""
    if type(value).__name__ == "Series":
        if value.dtype.kind not in "iuf":
            return value
        value = value.to_dict()
    if isinstance(value, dict):
        try:
            return {str(k): int(v) if float(v).is_integer() else float(v) for k, v in value.items()}
        except (TypeError, ValueError):
            return value
    return value


def _same_counts(a, b) -> bool:
    """Counts are the same with the same values and numbers, in any order; anything else is compared as usual."""
    a, b = _as_counts(a), _as_counts(b)
    if isinstance(a, dict) or isinstance(b, dict):
        return isinstance(a, dict) and isinstance(b, dict) and a == b
    a, b = (v.tolist() if hasattr(v, "tolist") else v for v in (a, b))   # such as the values of a column, unique
    return same_value(a, b)


def _shown_counts(value) -> str:
    counts = _as_counts(value)
    if isinstance(counts, dict):
        return "{" + ", ".join(f'"{k}": {v}' for k, v in counts.items()) + "}"
    if hasattr(value, "tolist"):   # an array of values, or a NumPy number
        value = value.tolist()
    if isinstance(value, list):
        return "[" + ", ".join(f'"{x}"' if isinstance(x, str) else str(x) for x in value) + "]"
    return repr(value)


_COUNTS_START = "import pandas as pd\n\n\ndef counts(path, column):\n    df = pd.read_csv(path)\n    ...\n"
_COUNTS = "import pandas as pd\n\n\ndef counts(path, column):\n    df = pd.read_csv(path)\n    return df[column].value_counts()"


def _counts(new: str) -> str:
    return _COUNTS.replace("    return df[column].value_counts()", new)


EXERCISES = {
    "missing": CodeBox(
        name="missing",
        start=_MISSING_START,
        solution=_MISSING,
        keep=["import pandas as pd", _READ],
        # The check also runs the box with another month, with other values missing.
        others=[{"df": Code(f'pd.read_csv("{STATION_2}")')}],
        first=[_DOTS],
        mistakes=[
            _NO_PARENTHESES, _NO_COLUMN, _ITSELF,
            Mistake("counts", wrong=_missing("missing = df.isna().sum()"),
                    en="Your last line gives the number of missing values in each column, as in the worked example. The task asks for the fraction, from 0 to 1: what in the table of new words gives the fraction of `True` values?",
                    nl="Je laatste regel geeft het aantal ontbrekende waarden in elke kolom, zoals in het uitgewerkte voorbeeld. De opdracht vraagt om het deel, van 0 tot 1: wat in de tabel met nieuwe woorden geeft het deel van de `True`-waarden?"),
            Mistake("percent", wrong=_missing("missing = df.isna().mean() * 100"),
                    en="Your last line gives a percentage, from 0 to 100. The task asks for a fraction, from 0 to 1: leave out `* 100`.",
                    nl="Je laatste regel geeft een percentage, van 0 tot 100. De opdracht vraagt om een deel van 0 tot 1: laat `* 100` weg."),
            Mistake("present", wrong=_missing("missing = df.notna().mean()"),
                    en="Your last line gives the fraction of values that are there, per column. The task asks for the ones that are missing: `df.isna()` is `True` for those.",
                    nl="Je laatste regel geeft het deel van de waarden dat er wel is, per kolom. De opdracht vraagt om de ontbrekende: `df.isna()` is `True` voor die."),
            Mistake("mask-table", matches=lambda got, text: type(got).__name__ == "DataFrame" and all(t.kind == "b" for t in got.dtypes),
                    en="That is a whole table of `True` and `False`, as big as `df`: `True` where a value is missing. Turn each column into one number: the fraction of its `True` values.",
                    nl="Dat is een hele tabel van `True` en `False`, zo groot als `df`: `True` waar een waarde ontbreekt. Maak van elke kolom één getal: het deel van zijn `True`-waarden."),
            Mistake("one-number", matches=lambda got, text: _number(got),
                    en="Your last line gives one number, `{got}`. The task asks for a number for each of the six columns. Does your line take only one column, or put all columns together?",
                    nl="Je laatste regel geeft één getal, `{got}`. De opdracht vraagt om een getal voor elk van de zes kolommen. Neemt je regel maar één kolom, of neemt hij alle kolommen samen?"),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the fraction of missing values in each column of `df`.",
                      "nl": "Vervang de `...` door het deel ontbrekende waarden in elke kolom van `df`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another month by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere maand."},
            "no_value": {"en": "The last line of the box shows nothing. Put `missing` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `missing` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives:\n\n```\n{got}\n```\n\nand it should be the fraction of missing values in each of the six columns, from 0 to 1: 2 of 30 for `temp`, and 0 for the others. Do you take, per column, the fraction of `True` values in `df.isna()`?",
                      "nl": "Je laatste regel geeft:\n\n```\n{got}\n```\n\nen dat moet het deel ontbrekende waarden in elk van de zes kolommen zijn, van 0 tot 1: 2 van de 30 voor `temp`, en 0 voor de andere. Neem je per kolom het deel `True`-waarden in `df.isna()`?"},
            "other": {"en": "Right for this month. But with {where}, your last line gives:\n\n```\n{got}\n```\n\nand it should give:\n\n```\n{want}\n```\n\nDo you work it out from `df` itself?",
                      "nl": "Goed voor deze maand. Maar met {where} geeft je laatste regel:\n\n```\n{got}\n```\n\nen dat moet dit zijn:\n\n```\n{want}\n```\n\nReken je het uit met `df` zelf?"},
            "other_error": {"en": "Right for this month. But with {where}, your code stops with an error. Do you work it out from `df` itself?",
                            "nl": "Goed voor deze maand. Maar met {where} stopt je code met een fout. Reken je het uit met `df` zelf?"},
            "correct": {"en": "2 of the 30 temperatures are missing, a fraction of 0.067, and nothing else. The check also tried another month, with three temperatures and one rain measurement missing, and your code gave those fractions too.",
                        "nl": "2 van de 30 temperaturen ontbreken, een deel van 0,067, en verder niets. De check probeerde ook een andere maand, waarin drie temperaturen en één regenmeting ontbreken, en je code gaf ook daar de goede delen."},
        },
    ),
    "counts": FunctionBox(
        name="counts",
        function="counts",
        start=_COUNTS_START,
        solution=_COUNTS,
        visible=[(STATION, "sky")],
        hidden=[(STATION_2, "sky"), (STATION, "dir")],
        compare=_same_counts,
        show=_shown_counts,
        first=[_DOTS],
        mistakes=[
            _NO_PARENTHESES, _NO_COLUMN,
            Mistake("name-in-quotes", error=lambda e: isinstance(e, KeyError) and "column" in str(e),
                    en="`df[\"column\"]` looks for a column that is called `column`. Use the name `column` itself, without quotes: `df[column]`; it holds the name the check gives, such as `\"sky\"`.",
                    nl="`df[\"column\"]` zoekt een kolom die `column` heet. Gebruik de naam `column` zelf, zonder aanhalingstekens: `df[column]`; daarin zit de naam die de check geeft, zoals `\"sky\"`."),
            Mistake("method-itself", matches=lambda got, text: callable(got),
                    en="`{call}` gives `value_counts` itself, not what it counts. `value_counts` needs its parentheses, also with nothing between them: `.value_counts()`.",
                    nl="`{call}` geeft `value_counts` zelf, niet wat het telt. `value_counts` heeft zijn haakjes nodig, ook als er niets tussen staat: `.value_counts()`."),
            Mistake("whole-table", matches=lambda got, text: type(got).__name__ == "DataFrame",
                    en="`{call}` gives back a whole table. Take the column `column` first, and then count its values.",
                    nl="`{call}` geeft een hele tabel terug. Neem eerst de kolom `column`, en tel dan zijn waarden."),
            Mistake("column-itself", wrong=_counts("    return df[column]"),
                    en="`{call}` gives the column itself, every value of it in turn. Count how often each value appears: the table of new words says how.",
                    nl="`{call}` geeft de kolom zelf, elke waarde ervan na elkaar. Tel hoe vaak elke waarde voorkomt: de tabel met nieuwe woorden zegt hoe."),
            Mistake("unique", wrong=_counts("    return df[column].unique()"),
                    en="`{call}` gives `{got}`: each value once, but not how often it appears. `.value_counts()` gives both.",
                    nl="`{call}` geeft `{got}`: elke waarde één keer, maar niet hoe vaak hij voorkomt. `.value_counts()` geeft allebei."),
            Mistake("how-many-kinds", wrong=_counts("    return df[column].nunique()"),
                    en="`{call}` gives `{got}`: how many different values there are. The task asks how often each value appears: `.value_counts()`.",
                    nl="`{call}` geeft `{got}`: hoeveel verschillende waarden er zijn. De opdracht vraagt hoe vaak elke waarde voorkomt: `.value_counts()`."),
            Mistake("all-rows", wrong=_counts("    return df[column].count()"),
                    en="`{call}` gives `{got}`: how many values the column has in all. The task asks how often each value appears: `.value_counts()`.",
                    nl="`{call}` geeft `{got}`: hoeveel waarden de kolom in totaal heeft. De opdracht vraagt hoe vaak elke waarde voorkomt: `.value_counts()`."),
            Mistake("fractions", wrong=_counts("    return df[column].value_counts(normalize=True)"),
                    en="`{call}` gives `{got}`: the fraction of each value, not how often it appears. Leave out `normalize=True`.",
                    nl="`{call}` geeft `{got}`: het aandeel van elke waarde, niet hoe vaak hij voorkomt. Laat `normalize=True` weg."),
            Mistake("whole-rows", wrong=_counts("    return df.value_counts()"),
                    en="`{call}` counts whole rows, and every row of the month is different. Take the column `column` first.",
                    nl="`{call}` telt hele rijen, en elke rij van de maand is anders. Neem eerst de kolom `column`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with how often each value of the column `column` appears, given back with `return`.",
                      "nl": "Vervang de `...` door hoe vaak elke waarde van de kolom `column` voorkomt, teruggegeven met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with a path and a column name, and the `def` line of your function has a different number of names between its parentheses. Keep `(path, column)`.",
                          "nl": "De check roept `{call}` aan, met een pad en een kolomnaam, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(path, column)`."},
            "wrong": {"en": "`{call}` gives `{got}`, and it should give `{want}`. Do you take the column `column` of `df`, and count how often each of its values appears?",
                      "nl": "`{call}` geeft `{got}`, en dat moet `{want}` zijn. Neem je de kolom `column` van `df`, en tel je hoe vaak elk van zijn waarden voorkomt?"},
            "almost": {"en": "Right for the example. But `{call}` gives `{got}`, and it should give `{want}`. Does it work for every file and every column?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft `{got}`, en dat moet `{want}` zijn. Werkt hij voor elk bestand en elke kolom?"},
            "correct": {"en": "Your function is right in all {n} cases. In the column `sky` of the first month, `Sunny` with a capital is counted apart from `sunny`: one value, written in two ways.",
                        "nl": "Je functie klopt in alle {n} gevallen. In de kolom `sky` van de eerste maand wordt `Sunny` met een hoofdletter apart geteld van `sunny`: één waarde, op twee manieren geschreven."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_M = "import pandas as pd\n\n" + _READ + "\n"
_C = "import pandas as pd\n\n\ndef counts(path, column):\n    df = pd.read_csv(path)\n"
CASES = [
    ("missing", _MISSING_START, "empty", None),
    ("missing", _MISSING, "correct", None),
    ("missing", _M + "missing = df.isnull().mean()\nmissing", "correct", None),
    ("missing", _M + "missing = df.isna().sum() / len(df)\nmissing", "correct", None),
    ("missing", _M + "missing = df.isna().sum()\nmissing", "wrong", "counts"),
    ("missing", _M + "missing = df.isna().mean() * 100\nmissing", "wrong", "percent"),
    ("missing", _M + "missing = df.notna().mean()\nmissing", "wrong", "present"),
    ("missing", _M + "missing = df.isna()\nmissing", "wrong", "mask-table"),
    ("missing", _M + "missing = df.describe()\nmissing", "wrong", None),
    ("missing", _M + "missing = df.isna().mean().mean()\nmissing", "wrong", "one-number"),
    ("missing", _M + 'missing = df["temp"].isna().mean()\nmissing', "wrong", "one-number"),
    ("missing", _M + 'missing = df["Temp"].isna().mean()\nmissing', "error", "no-column"),
    ("missing", _M + "missing = df.isna.mean()\nmissing", "error", "no-parentheses"),
    ("missing", _M + "missing = df.isna().mean\nmissing", "wrong", "itself"),
    ("missing", _M + 'missing = pd.Series({"day": 0, "temp": 2 / 30, "rain": 0, "wind": 0, "dir": 0, "sky": 0})\nmissing', "almost", "other-value"),
    ("missing", _M + "missing = df.isna().mean()\nprint(missing)", "wrong", "print-last"),
    ("counts", _COUNTS_START, "empty", None),
    ("counts", _COUNTS, "correct", None),
    ("counts", _C + "    return dict(df[column].value_counts())", "correct", None),
    ("counts", _C + "    seen = {}\n    for v in df[column]:\n        if v in seen:\n            seen[v] = seen[v] + 1\n        else:\n            seen[v] = 1\n    return seen", "correct", None),
    ("counts", _C + '    return df["column"].value_counts()', "error", "name-in-quotes"),
    ("counts", _C + "    return df[column].unique()", "wrong", "unique"),
    ("counts", _C + "    return df[column].nunique()", "wrong", "how-many-kinds"),
    ("counts", _C + "    return df[column].count()", "wrong", "all-rows"),
    ("counts", _C + "    return df[column].value_counts(normalize=True)", "wrong", "fractions"),
    ("counts", _C + "    return df.value_counts()", "wrong", "whole-rows"),
    ("counts", _C + "    return df[column].value_counts", "wrong", "method-itself"),
    ("counts", _C + "    return df[column]", "wrong", "column-itself"),
    ("counts", _C + "    return df[column].describe()", "wrong", None),
    ("counts", _C + "    return df", "wrong", "whole-table"),
    ("counts", _C + '    return df["Sky"].value_counts()', "error", "no-column"),
    ("counts", _C + '    return df["sky"].value_counts()', "almost", None),
]
