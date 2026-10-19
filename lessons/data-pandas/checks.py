# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson data-pandas: the windy days of a made-up week of weather, selected from a DataFrame,
and the names of the warm days, read from a CSV file in a function. And the table in the picture."""
import csv
import re
from pathlib import Path

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake, _as_code

HERE = Path(__file__).parent
WEEK = "lessons/data-pandas/weather.csv"
WEEK_2 = "lessons/data-pandas/weather-2.csv"


def _table(name: str) -> tuple[list[str], list[list[str]]]:
    with open(HERE / name, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def picture(labels: list[str]) -> list[tuple]:
    """Two panels of the week's weather: one column taken, and the rows of the rainy days kept."""
    header, rows = _table("weather.csv")
    rain = header.index("rain")
    column = {(i, rain) for i in range(len(rows))}
    rainy = {(i, j) for i, row in enumerate(rows) if float(row[rain]) > 0 for j in range(len(header))}
    return [
        (labels[0], header, rows, column, 'df["rain"]'),
        (labels[1], header, rows, rainy, 'df[df["rain"] > 0]'),
    ]


# The lesson is done when these are correct. "two-columns" is the choice question in the lesson.
REQUIRED = ["two-columns", "windy", "warm-days"]

_READ = f'df = pd.read_csv("{WEEK}")'
_WINDY_START = "import pandas as pd\n\n" + _READ + "\nwindy = ...\nwindy\n"
_WINDY = "import pandas as pd\n\n" + _READ + '\nwindy = df[df["wind"] > 20]\nwindy'

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_NAME_COMPARED = Mistake("name-compared", error=lambda e: isinstance(e, TypeError) and "'str' and 'int'" in str(e),
                         en="Your code compares a column name, a text, with a number. Compare the column itself: first take it out of `df` with its name, then `> 20`.",
                         nl="Je code vergelijkt een kolomnaam, een tekst, met een getal. Vergelijk de kolom zelf: haal hem eerst met zijn naam uit `df`, dan `> 20`.")
_TEXT_NUMBER = Mistake("text-number", error=lambda e: isinstance(e, TypeError) and "Invalid comparison" in str(e),
                       en="Your code compares a column of numbers with text: a number between quotes, such as `\"20\"`, is text. Write the number without quotes.",
                       nl="Je code vergelijkt een kolom met getallen met tekst: een getal tussen aanhalingstekens, zoals `\"20\"`, is tekst. Schrijf het getal zonder aanhalingstekens.")
_BRACKETS = Mistake("brackets", error=lambda e: isinstance(e, KeyError) and "None of" in str(e),
                    en="A square bracket closes too early: `df[df[\"wind\"]]` asks `df` for columns named after the wind numbers. The comparison goes inside: `df[df[\"wind\"] > 20]`.",
                    nl="Een vierkante haak sluit te vroeg: `df[df[\"wind\"]]` vraagt `df` om kolommen met de windgetallen als naam. De vergelijking hoort erbinnen: `df[df[\"wind\"] > 20]`.")
_NO_COLUMN = Mistake("no-column", error=lambda e: isinstance(e, KeyError) and bool(e.args) and isinstance(e.args[0], (str, tuple)),
                     en="The table has no column with that name. Its columns are `day`, `temp`, `rain` and `wind`, in small letters, each between quotes.",
                     nl="De tabel heeft geen kolom met die naam. De kolommen zijn `day`, `temp`, `rain` en `wind`, in kleine letters, elk tussen aanhalingstekens.")


def _is_mask(got) -> bool:
    return type(got).__name__ == "Series" and str(got.dtype) == "bool"


def _shown(value):
    """What a function gave back, in one line: a column as the list of its values, in code; a table by its size,
    in words, in the lesson's language."""
    kind = type(value).__name__
    if kind == "Series":
        return f"`{_as_code(value.tolist())}`"
    if kind == "DataFrame":
        rows, cols = value.shape
        return {"en": f"a table of {rows} row{'' if rows == 1 else 's'} and {cols} column{'' if cols == 1 else 's'}",
                "nl": f"een tabel van {rows} {'rij' if rows == 1 else 'rijen'} en {cols} {'kolom' if cols == 1 else 'kolommen'}"}
    return f"`{_as_code(value) if isinstance(value, (str, list)) else repr(value)}`"


_WARM_START = "import pandas as pd\n\n\ndef warm_days(path, limit):\n    df = pd.read_csv(path)\n    ...\n"
_WARM = ('import pandas as pd\n\n\ndef warm_days(path, limit):\n    df = pd.read_csv(path)\n'
         '    warm = df[df["temp"] > limit]\n    return warm["day"]')

EXERCISES = {
    "windy": CodeBox(
        name="windy",
        start=_WINDY_START,
        solution=_WINDY,
        keep=["import pandas as pd", _READ],
        # The check also runs the box with another week: rows typed in by hand do not change with it.
        others=[{"df": Code(f'pd.read_csv("{WEEK_2}")')}],
        first=[_DOTS],
        mistakes=[
            _NAME_COMPARED, _TEXT_NUMBER, _BRACKETS, _NO_COLUMN,
            Mistake("mask-itself", matches=lambda got, text: _is_mask(got),
                    en="That is the mask: `True` or `False` for every day. Put it between the square brackets of `df`, to keep the rows where it is `True`.",
                    nl="Dat is het mask: `True` of `False` voor elke dag. Zet het tussen de vierkante haken van `df`, om de rijen te houden waar het `True` is."),
            Mistake("at-least", wrong=_WINDY.replace("> 20", ">= 20"),
                    en="Your table also has Thursday, with a wind of exactly 20. The task asks for more than 20: use `>`.",
                    nl="Je tabel heeft ook donderdag, met een wind van precies 20. De opdracht vraagt om meer dan 20: gebruik `>`."),
            Mistake("one-column", matches=lambda got, text: type(got).__name__ == "Series" and not _is_mask(got),
                    en="That is one column, a Series. The task asks for whole rows, with all four columns: put the mask straight between the square brackets of `df`, without a column name.",
                    nl="Dat is één kolom, een Series. De opdracht vraagt om hele rijen, met alle vier de kolommen: zet het mask direct tussen de vierkante haken van `df`, zonder kolomnaam."),
            Mistake("some-columns", matches=lambda got, text: type(got).__name__ == "DataFrame" and 0 < got.shape[1] < 4,
                    en="Those are the right days, perhaps, but not all four columns. The task asks for the whole rows: put the mask straight between the square brackets of `df`.",
                    nl="Dat zijn misschien de goede dagen, maar niet alle vier de kolommen. De opdracht vraagt om de hele rijen: zet het mask direct tussen de vierkante haken van `df`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the rows of `df` where the wind is more than 20.",
                      "nl": "Vervang de `...` door de rijen van `df` waar de wind meer dan 20 is."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
            "no_value": {"en": "The last line of the box shows nothing. Put `windy` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `windy` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives:\n\n```\n{got}\n```\n\nand it should be the rows of the days with more wind than 20, with all four columns: Tuesday, Wednesday and Sunday. Which rows does your mask keep, and which columns do you take?",
                      "nl": "Je laatste regel geeft:\n\n```\n{got}\n```\n\nen dat moeten de rijen zijn van de dagen met meer wind dan 20, met alle vier de kolommen: dinsdag, woensdag en zondag. Welke rijen houdt je mask over, en welke kolommen neem je?"},
            "other": {"en": "Right for this week. But with {where}, your last line gives:\n\n```\n{got}\n```\n\nand it should give:\n\n```\n{want}\n```\n\nDo you pick the rows from `df` itself, with a mask?",
                      "nl": "Goed voor deze week. Maar met {where} geeft je laatste regel:\n\n```\n{got}\n```\n\nen dat moet dit zijn:\n\n```\n{want}\n```\n\nKies je de rijen uit `df` zelf, met een mask?"},
            "other_error": {"en": "Right for this week. But with {where}, your code stops with an error. Do you select the rows from `df` itself, with a mask on the column `wind`?",
                            "nl": "Goed voor deze week. Maar met {where} stopt je code met een fout. Selecteer je de rijen uit `df` zelf, met een mask op de kolom `wind`?"},
            "correct": {"en": "Three windy days: Tuesday, Wednesday and Sunday, with all four columns. The check also tried another week, and your code picked its windy days too.",
                        "nl": "Drie winderige dagen: dinsdag, woensdag en zondag, met alle vier de kolommen. De check probeerde ook een andere week, en je code koos ook daar de winderige dagen."},
        },
    ),
    "warm-days": FunctionBox(
        name="warm-days",
        function="warm_days",
        start=_WARM_START,
        solution=_WARM,
        visible=[(WEEK, 13)],
        hidden=[(WEEK_2, 12), (WEEK, 20), (WEEK_2, 0)],
        show=_shown,
        first=[_DOTS],
        mistakes=[
            _TEXT_NUMBER, _BRACKETS, _NO_COLUMN,
            Mistake("fixed-path", matches=lambda got, text: re.search(r"read_csv\(\s*[\"']", text) is not None,
                    en="Right for the example, perhaps, but `{call}` gives {got}, and it should give {want}: your function reads a file of its own choice. Read the file the function gets: `pd.read_csv(path)`.",
                    nl="Misschien goed voor het voorbeeld, maar `{call}` geeft {got}, en dat moet {want} zijn: je functie leest een bestand naar eigen keuze. Lees het bestand dat de functie krijgt: `pd.read_csv(path)`.",
                    state="almost"),
            Mistake("fixed-limit", state="almost", wrong=_WARM.replace("> limit", "> 13"),
                    en="Right for the example. But `{call}` gives {got}, and it should give {want}: your mask compares with `13`, not with `limit`. Use the name `limit`, so the function works for every limit.",
                    nl="Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn: je mask vergelijkt met `13`, niet met `limit`. Gebruik de naam `limit`, dan werkt de functie voor elke grens."),
            Mistake("double-brackets", wrong=_WARM.replace('return warm["day"]', 'return warm[["day"]]'),
                    en="`{call}` gives {got}: a table with one column, because of the two pairs of brackets. For one column on its own, use one pair: `warm[\"day\"]`.",
                    nl="`{call}` geeft {got}: een tabel met één kolom, door de twee paar haken. Voor één kolom op zichzelf gebruik je één paar: `warm[\"day\"]`."),
            Mistake("at-least", wrong=_WARM.replace("> limit", ">= limit"),
                    en="`{call}` gives {got}, and it should give {want}: your mask also keeps the days that are exactly as warm as `limit`. The task asks for warmer: use `>`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: je mask houdt ook de dagen die precies zo warm zijn als `limit`. De opdracht vraagt om warmer: gebruik `>`."),
            Mistake("whole-rows", wrong=_WARM.replace('return warm["day"]', "return warm"),
                    en="`{call}` gives {got}: the whole rows. The task asks only for the names of the days: the column `day` of those rows.",
                    nl="`{call}` geeft {got}: de hele rijen. De opdracht vraagt alleen om de namen van de dagen: de kolom `day` van die rijen."),
            Mistake("mask-itself", wrong=_WARM.replace('    warm = df[df["temp"] > limit]\n    return warm["day"]', '    return df["temp"] > limit'),
                    en="`{call}` gives the mask itself: `True` or `False` for every day. Put it between the square brackets of `df` to keep those rows, and then take their column `day`.",
                    nl="`{call}` geeft het mask zelf: `True` of `False` voor elke dag. Zet het tussen de vierkante haken van `df` om die rijen te houden, en neem dan hun kolom `day`."),
            Mistake("all-days", wrong=_WARM.replace('    warm = df[df["temp"] > limit]\n    return warm["day"]', '    return df["day"]'),
                    en="`{call}` gives {got}: every day of the week. Keep only the warm rows first, with a mask on `temp`, and then take their column `day`.",
                    nl="`{call}` geeft {got}: elke dag van de week. Houd eerst alleen de warme rijen, met een mask op `temp`, en neem dan hun kolom `day`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the names of the days that are warmer than `limit`, given back with `return`.",
                      "nl": "Vervang de `...` door de namen van de dagen die warmer zijn dan `limit`, teruggegeven met `return`."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "arguments": {"en": "The check calls `{call}`, with a path and a limit, and the `def` line of your function has a different number of names between its parentheses. Keep `(path, limit)`.",
                          "nl": "De check roept `{call}` aan, met een pad en een grens, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(path, limit)`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Do you keep the rows warmer than `limit`, and give back their column `day`?",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Houd je de rijen die warmer zijn dan `limit`, en geef je hun kolom `day` terug?"},
            "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does it work for every file and every limit, also when no day is warm enough?",
                       "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt hij voor elk bestand en elke grens, ook als geen dag warm genoeg is?"},
            "correct": {"en": "Your function is right in all {n} cases. It reads the file each time, so it works for any week in this format.",
                        "nl": "Je functie klopt in alle {n} gevallen. Hij leest het bestand elke keer, dus werkt hij voor elke week in deze vorm."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_W = "import pandas as pd\n\n" + _READ + "\n"
_F = "import pandas as pd\n\n\ndef warm_days(path, limit):\n    df = pd.read_csv(path)\n"
CASES = [
    ("windy", _WINDY_START, "empty", None),
    ("windy", _WINDY, "correct", None),
    ("windy", _W + 'strong = df["wind"] > 20\nwindy = df[strong]\nwindy', "correct", None),
    ("windy", _W + 'windy = df[df["wind"] > 20].reset_index(drop=True)\nwindy', "correct", None),
    ("windy", _W + 'windy = df["wind"] > 20\nwindy', "wrong", "mask-itself"),
    ("windy", _W + 'windy = df[df["wind"] >= 20]\nwindy', "wrong", "at-least"),
    ("windy", _W + 'windy = df[df["wind"] > 20]["wind"]\nwindy', "wrong", "one-column"),
    ("windy", _W + 'windy = df[df["Wind"] > 20]\nwindy', "error", "no-column"),
    ("windy", _W + 'windy = df["wind" > 20]\nwindy', "error", "name-compared"),
    ("windy", _W + 'windy = df[df["wind"] > 20]\nprint(windy)', "wrong", "print-last"),
    ("windy", _W + 'windy = df.iloc[[1, 2, 6]]\nwindy', "almost", "other-value"),
    ("windy", _W + 'windy = df[["day", "wind"]][df["wind"] > 20]\nwindy', "wrong", "some-columns"),
    ("windy", _W + 'windy = df[df["wind"] > "20"]\nwindy', "error", "text-number"),
    ("windy", _W + 'windy = df[df["wind"]] > 20\nwindy', "error", "brackets"),
    ("windy", _W + 'windy = df["wind"]\nwindy', "wrong", "one-column"),
    ("windy", _W + 'windy = df[df["wind"] > ...]\nwindy', "wrong", "dots-left"),
    ("warm-days", _WARM_START, "empty", None),
    ("warm-days", _WARM, "correct", None),
    ("warm-days", _F + '    return df[df["temp"] > limit]["day"]', "correct", None),
    ("warm-days", _F + '    return list(df[df["temp"] > limit]["day"])', "correct", None),
    ("warm-days", _WARM.replace("> limit", ">= limit"), "wrong", "at-least"),
    ("warm-days", _WARM.replace('return warm["day"]', "return warm"), "wrong", "whole-rows"),
    ("warm-days", _F + '    return df["temp"] > limit', "wrong", "mask-itself"),
    ("warm-days", _F + '    return df["day"]', "wrong", "all-days"),
    ("warm-days", _F + '    return df[df["temp"] > 13]["day"]', "almost", "fixed-limit"),
    ("warm-days", _WARM.replace('return warm["day"]', 'return warm[["day"]]'), "wrong", "double-brackets"),
    ("warm-days", _F.replace("pd.read_csv(path)", 'pd.read_csv("lessons/data-pandas/weather.csv")') + '    return df[df["temp"] > limit]["day"]', "almost", "fixed-path"),
    ("warm-days", _F + '    return df[df["Temp"] > limit]["day"]', "error", "no-column"),
]
