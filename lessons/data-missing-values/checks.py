# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson data-missing-values: the wind of the month of the made-up weather station of data-explore,
with its code -999 read as missing and filled in with the median, and a function that marks the gaps of a column in a
missing-indicator column before it fills them in. And the play block: the month's temperatures with more and more
days missing, dropped or filled in with the mean."""
import re
from functools import cache

import numpy as np
import pandas as pd   # here, so the first import of pandas never counts against the time limit of a check

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake, same_value

STATION = "lessons/data-explore/station.csv"
STATION_2 = "lessons/data-explore/station-2.csv"
WEATHER = "lessons/data-pandas/weather.csv"
SALES = "lessons/data-merge/sales.csv"     # the stand was closed on Monday
PENGUINS = "data/penguins.csv"
LOST = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]   # the stops of the slider: days of the month without a temperature


def month(path: str = STATION) -> pd.DataFrame:
    """A month of the station, with the code -999 read as missing, as the worked example reads it."""
    return pd.read_csv(path, na_values=[-999])


# --- the play block: more and more days missing --------------------------------------------------------------------------

@cache
def _order() -> tuple:
    """The measured days, in the fixed order in which the thermometer "fails" when the slider asks for more missing
    days: the 28 days with a temperature, shuffled once with seed 0."""
    measured = np.flatnonzero(month()["temp"].notna().to_numpy())
    return tuple(int(i) for i in np.random.default_rng(0).permutation(measured))


def gaps(k: int) -> pd.Series:
    """The month's temperatures with k days missing: the 2 that really are, and k - 2 more in the fixed order."""
    temp = month()["temp"].copy()
    temp.iloc[list(_order()[:k - 2])] = np.nan
    return temp


def numbers(k: int) -> dict:
    """What the sentence under the play block says for k missing days: the mean of the measured days, and the
    standard deviation (as pandas and describe work it out) when you drop the missing days, and when you fill them in
    with that mean. Rounded to two decimals, as the sentence shows them."""
    temp = gaps(k)
    mean = temp.mean()
    return {"left": int(temp.count()), "mean": round(mean, 2),
            "drop": round(temp.dropna().std(), 2), "fill": round(temp.fillna(mean).std(), 2)}


def lost_picture(k: int, labels: dict) -> str:
    """The month as a dot per day at its temperature: a filled dot for a measured day, an open dot on the dashed line
    of the mean for a missing day filled in with that mean (style guide 1.7: two groups, filled and open, no colour).
    Ink and grey only: nothing here is learned or predicted. Drawn once for every stop of the slider (L.explore); the
    lesson writes the alt text, with the numbers in its own language."""
    from everything_ai import chart
    temp = gaps(k)
    mean = temp.mean()
    days = month()["day"].to_numpy()
    measured = temp.notna().to_numpy()
    p = chart.Panel((0, 31), (4, 13.5), title=labels["title"], xlabel=labels["day"], ylabel=labels["temp"],
                    xticks=[1, 5, 10, 15, 20, 25, 30], yticks=[5, 7, 9, 11, 13])
    p.line([0.5, 30.5], [mean, mean], "ink-3", 1.4, dashed=True)
    p.points(days[measured], temp.to_numpy()[measured], "data", r=4)
    p.points(days[~measured], [mean] * int((~measured).sum()), "data", r=4, hollow=True)
    return chart.figure([p], alt=labels["alt"], width=460, height=280)


# The lesson is done when these are correct. "discount" is the choice question in the lesson.
REQUIRED = ["filled", "fill-flag", "discount"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _kind(got) -> str:
    return type(got).__name__


def _number(got) -> bool:
    return _kind(got) in ("int", "int64", "float", "float64") and not isinstance(got, bool)


def _gaps_left(got) -> bool:
    return _kind(got) == "Series" and bool(got.isna().any())


def _text_in(got) -> bool:
    """A column with a text in it, such as the word "median" filled in instead of the median itself."""
    return _kind(got) == "Series" and any(isinstance(v, str) for v in got.tolist())


def _method_in(got) -> str:
    """The name of a method put in a column or table instead of what it gives, such as `median` itself put in the
    gaps instead of the median, or `isna` itself as the indicator; "" when there is none."""
    if _kind(got) == "DataFrame":
        return next((m for m in (_method_in(got[c]) for c in got.columns) if m), "")
    if _kind(got) == "Series" and got.dtype.kind == "O":
        return next((getattr(v, "__name__", "?") for v in got.tolist() if callable(v)), "")
    return ""


def _callable_in(got) -> bool:
    """A column or table with a method in it, such as `median` itself put in the gaps instead of the median."""
    return bool(_method_in(got))


# `inplace=True` on one column, `df["wind"].fillna(..., inplace=True)`: with Copy-on-Write that changes a copy, not
# `df`. On the whole table, `df.fillna({...}, inplace=True)`, it works, so that is not this mistake.
_INPLACE = re.compile(r"\]\s*\.\s*fillna\([^\n]*inplace\s*=\s*True")


# --- exercise 1: the wind, filled in with its median -----------------------------------------------------------------------

_READ = f'df = pd.read_csv("{STATION}", na_values=[-999])'
_FILLED_START = "import pandas as pd\n\n" + _READ + "\nfilled = ...\nfilled\n"
_FILLED = "import pandas as pd\n\n" + _READ + '\nfilled = df["wind"].fillna(df["wind"].median())\nfilled'


def _filled(new: str) -> str:
    return _FILLED.replace('filled = df["wind"].fillna(df["wind"].median())', new)


_NO_COLUMN = Mistake("no-column", error=lambda e: isinstance(e, KeyError),
                     en="The table has no column with that name. Its columns are `day`, `temp`, `rain`, `wind`, `dir` and `sky`, in small letters, each between quotes.",
                     nl="De tabel heeft geen kolom met die naam. De kolommen zijn `day`, `temp`, `rain`, `wind`, `dir` en `sky`, in kleine letters, elk tussen aanhalingstekens.")
_NO_PARENTHESES_MEAN = Mistake("no-parentheses-mean", matches=lambda got, text: _method_in(got) == "mean",
                               en="`fillna` gets `mean` itself, not the mean, and puts that on day 17. A method needs its parentheses, also with nothing between them; and the task asks for the median: `df[\"wind\"].median()`.",
                               nl="`fillna` krijgt `mean` zelf, niet het gemiddelde, en zet dat op dag 17. Een methode heeft zijn haakjes nodig, ook als er niets tussen staat; en de opdracht vraagt om de mediaan: `df[\"wind\"].median()`.")
_NO_PARENTHESES = Mistake("no-parentheses", matches=lambda got, text: _callable_in(got),
                          detail=lambda got, text: {"method": _method_in(got)},
                          en="`fillna` gets `{method}` itself, not what it works out, and puts that on day 17. A method needs its parentheses, also with nothing between them: `df[\"wind\"].median()`.",
                          nl="`fillna` krijgt `{method}` zelf, niet wat die uitrekent, en zet dat op dag 17. Een methode heeft zijn haakjes nodig, ook als er niets tussen staat: `df[\"wind\"].median()`.")
# `df.median()` works on every column, and the station's `dir` and `sky` hold text: pandas stops with a TypeError.
_TABLE_MEDIAN = r"\bdf\.(median|mean)\(\s*\)"

EXERCISES = {
    "filled": CodeBox(
        name="filled",
        start=_FILLED_START,
        solution=_FILLED,
        keep=["import pandas as pd", _READ],
        # The check also runs the box with the other month, where the station wrote -999 on another day.
        others=[{"df": Code(f'pd.read_csv("{STATION_2}", na_values=[-999])')}],
        first=[_DOTS],
        digits=2,   # one number in a message, such as the mean wind: 21.9, not 21.896551724137932
        mistakes=[
            _NO_COLUMN, _NO_PARENTHESES_MEAN, _NO_PARENTHESES,
            Mistake("table-median", syntax=_TABLE_MEDIAN,
                    en="Python stops on the line `{line}`: `df.median()` and `df.mean()` work on every column of the table, and `dir` and `sky` hold text. Take the median of the wind alone: `df[\"wind\"].median()`.",
                    nl="Python stopt op de regel `{line}`: `df.median()` en `df.mean()` werken op elke kolom van de tabel, en in `dir` en `sky` staat tekst. Neem de mediaan van alleen de wind: `df[\"wind\"].median()`."),
            Mistake("whole-table", matches=lambda got, text: _kind(got) == "DataFrame",
                    en="Your last line gives a whole table. The task asks for one column, the wind: start from `df[\"wind\"]`.",
                    nl="Je laatste regel geeft een hele tabel. De opdracht vraagt om één kolom, de wind: begin bij `df[\"wind\"]`."),
            Mistake("one-number", matches=lambda got, text: _number(got),
                    en="Your last line gives one number, `{got}`, such as the median of the column. The task asks for the column of 30 wind speeds, with the median in place of the gap: `fillna` puts it there.",
                    nl="Je laatste regel geeft één getal, `{got}`, zoals de mediaan van de kolom. De opdracht vraagt om de kolom van 30 windsnelheden, met de mediaan op de plek van het gat: `fillna` zet hem daar."),
            Mistake("mask", matches=lambda got, text: _kind(got) == "Series" and got.dtype.kind == "b",
                    en="Your last line gives `True` and `False` for every day: where the wind is missing. That is the mask of a missing-indicator column. The task asks for the wind speeds themselves, with the gap filled in: `fillna`.",
                    nl="Je laatste regel geeft `True` en `False` voor elke dag: waar de wind ontbreekt. Dat is het masker van een missing-indicatorkolom. De opdracht vraagt om de windsnelheden zelf, met het gat ingevuld: `fillna`."),
            Mistake("text", matches=lambda got, text: _text_in(got),
                    en="Day 17 now holds a text, such as the word `\"median\"`, not a number. Give `fillna` the median itself: `df[\"wind\"].median()`, without quotes.",
                    nl="Op dag 17 staat nu een tekst, zoals het woord `\"median\"`, geen getal. Geef `fillna` de mediaan zelf: `df[\"wind\"].median()`, zonder aanhalingstekens."),
            Mistake("mean", wrong=_filled('filled = df["wind"].fillna(df["wind"].mean())'),
                    en="Your last line fills the gap in with the mean wind. The task asks for the median: `df[\"wind\"].median()`.",
                    nl="Je laatste regel vult het gat in met de gemiddelde wind. De opdracht vraagt om de mediaan: `df[\"wind\"].median()`."),
            Mistake("zero", wrong=_filled('filled = df["wind"].fillna(0)'),
                    en="Your last line fills the gap in with 0: a day without any wind, which nobody measured. You do not know the wind of day 17, so the task asks for a guess from the column itself: its median.",
                    nl="Je laatste regel vult het gat in met 0: een dag zonder wind, die niemand heeft gemeten. Je weet de wind van dag 17 niet, dus de opdracht vraagt om een gok uit de kolom zelf: de mediaan."),
            Mistake("dropped", wrong=_filled('filled = df["wind"].dropna()'),
                    en="Your last line gives 29 wind speeds: `dropna` leaves out day 17. The task keeps all 30 days and fills the gap in: `fillna`.",
                    nl="Je laatste regel geeft 29 windsnelheden: `dropna` laat dag 17 weg. De opdracht houdt alle 30 dagen en vult het gat in: `fillna`."),
            Mistake("previous-day", wrong=_filled('filled = df["wind"].ffill()'),
                    en="Your last line fills the gap in with the wind of the day before. The task asks for the median of the column: `df[\"wind\"].fillna(df[\"wind\"].median())`.",
                    nl="Je laatste regel vult het gat in met de wind van de dag ervoor. De opdracht vraagt om de mediaan van de kolom: `df[\"wind\"].fillna(df[\"wind\"].median())`."),
            Mistake("inplace", matches=lambda got, text: _INPLACE.search(text) is not None and _gaps_left(got),
                    en="Your last line still has the gap of day 17. With `inplace=True` on `df[\"wind\"]`, this version of pandas does not change `df`. Leave out `inplace=True`, and give `filled` what `fillna` gives back: `filled = df[\"wind\"].fillna(...)`.",
                    nl="Je laatste regel heeft nog het gat van dag 17. Met `inplace=True` op `df[\"wind\"]` verandert deze versie van pandas `df` niet. Laat `inplace=True` weg, en geef `filled` wat `fillna` teruggeeft: `filled = df[\"wind\"].fillna(...)`."),
            Mistake("gap-left", matches=lambda got, text: _gaps_left(got),
                    en="Your last line still has a gap: `NaN` on day 17. `fillna` puts a value in its place; give it the median of the wind.",
                    nl="Je laatste regel heeft nog een gat: `NaN` op dag 17. `fillna` zet er een waarde in; geef hem de mediaan van de wind."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the wind of `df`, its gap filled in with the median wind.",
                      "nl": "Vervang de `...` door de wind van `df`, met het gat ingevuld met de mediaan van de wind."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another month by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere maand."},
            "no_value": {"en": "The last line of the box shows nothing. Put `filled` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `filled` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives something other than the 30 wind speeds of `df` with the gap of day 17 filled in by the median wind. Do you take `df[\"wind\"]`, and give `fillna` the median of that column?",
                      "nl": "Je laatste regel geeft iets anders dan de 30 windsnelheden van `df` met het gat van dag 17 ingevuld met de mediaan van de wind. Neem je `df[\"wind\"]`, en geef je `fillna` de mediaan van die kolom?"},
            "other": {"en": "Right for this month. But with {where}, your last line does not give its 30 wind speeds with the gap filled in by their median, 16. Do you work the median out from `df` itself?",
                      "nl": "Goed voor deze maand. Maar met {where} geeft je laatste regel niet de 30 windsnelheden van die maand met het gat ingevuld met hun mediaan, 16. Reken je de mediaan uit met `df` zelf?"},
            "other_error": {"en": "Right for this month. But with {where}, your code stops with an error. Do you work it out from `df` itself?",
                            "nl": "Goed voor deze maand. Maar met {where} stopt je code met een fout. Reken je het uit met `df` zelf?"},
            "correct": {"en": "Day 17, where the station wrote -999, now has the median wind, 21 kilometers per hour: 30 wind speeds and no gap. The check also tried the other month, and your code filled its gap with its own median, 16.",
                        "nl": "Dag 17, waar het station -999 schreef, heeft nu de mediaan van de wind, 21 kilometer per uur: 30 windsnelheden en geen gat. De check probeerde ook de andere maand, en je code vulde daar het gat met de mediaan van die maand, 16."},
        },
    ),
}


# --- exercise 2: mark the gaps, then fill them in ----------------------------------------------------------------------------

def _flag(column: str) -> str:
    return f"{column}_missing"


def _as_numbers(df: pd.DataFrame) -> pd.DataFrame:
    """The same table with every column of True and False as 1 and 0, so that an indicator of 0 and 1 counts the same
    as one of True and False."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind == "b":
            out[c] = out[c].astype(int)
    return out


def _same_flagged(a, b) -> bool:
    """The same table: the same columns, in any order, and the same values, row by row; an indicator may be True and
    False or 1 and 0. Anything else is compared as usual."""
    if _kind(a) == "DataFrame" and _kind(b) == "DataFrame":
        if set(map(str, a.columns)) != set(map(str, b.columns)) or a.shape != b.shape:
            return False
        a, b = _as_numbers(a), _as_numbers(b[list(a.columns)])
    return same_value(a, b)


def _held(value: pd.DataFrame) -> dict:
    """What a column holds where its indicator is True, such as 10.75 in `temp` where `temp_missing` is True: so that
    two tables that differ only in the value filled in, 0 or the median, are not shown in the same words."""
    en, nl = [], []
    for flag in value.columns:
        column = str(flag)[:-len("_missing")]
        if not str(flag).endswith("_missing") or column not in value.columns or value[flag].dtype.kind not in "bi":
            continue
        held = sorted({float(v) for v in value.loc[value[flag].astype(bool), column].tolist()
                       if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v})
        if 0 < len(held) <= 3:
            shown = [f"{v:g}" for v in held]
            en.append(f"; `{column}` holds " + " and ".join(shown) + f" where `{flag}` is `True`")
            nl.append(f"; in `{column}` staat " + " en ".join(s.replace(".", ",") for s in shown) + f" waar `{flag}` `True` is")
    return {"en": "".join(en), "nl": "".join(nl)}


def _shown_table(value):
    """A table in words, in both languages: its size, its columns, what is still missing and what was filled in where
    an indicator is True, instead of all its rows."""
    if _kind(value) == "DataFrame":
        cols = [f"`{c}`" for c in value.columns]
        gaps_ = {c: int(n) for c, n in value.isna().sum().items() if n}
        en_cols, nl_cols = ", ".join(cols[:-1]) + " and " + cols[-1], ", ".join(cols[:-1]) + " en " + cols[-1]
        if gaps_:
            en_gaps = "; still missing: " + ", ".join(f"{n} in `{c}`" for c, n in gaps_.items())
            nl_gaps = "; nog ontbrekend: " + ", ".join(f"{n} in `{c}`" for c, n in gaps_.items())
        else:
            en_gaps, nl_gaps = "; nothing missing", "; niets ontbrekend"
        held = _held(value)
        return {"en": f"a table of {len(value)} rows with the columns {en_cols}{en_gaps}{held['en']}",
                "nl": f"een tabel van {len(value)} rijen met de kolommen {nl_cols}{nl_gaps}{held['nl']}"}
    if _kind(value) == "Series":
        return {"en": f"one column of {len(value)} values, not a table", "nl": f"één kolom van {len(value)} waarden, geen tabel"}
    return f"`{value}`"


_FLAG_START = ("import pandas as pd\n\n\ndef fill_flag(path, column):\n    df = pd.read_csv(path, na_values=[-999])\n"
               "    ...\n    return df\n")
_FLAG = ('import pandas as pd\n\n\ndef fill_flag(path, column):\n    df = pd.read_csv(path, na_values=[-999])\n'
         '    df[f"{column}_missing"] = df[column].isna()\n    df[column] = df[column].fillna(df[column].median())\n    return df')
_FLAG_LINES = '    df[f"{column}_missing"] = df[column].isna()\n    df[column] = df[column].fillna(df[column].median())\n'


def _flagged(new: str) -> str:
    return _FLAG.replace(_FLAG_LINES, new)


def _has_flag(got, column: str) -> bool:
    return _kind(got) == "DataFrame" and _flag(column) in got.columns


def _flag_after(got, text: str) -> bool:
    """An indicator that is False on every row, and an `isna` that comes after the first `fillna` in the code: the
    gaps were marked after they were filled in. An indicator that is all False for another reason is not this."""
    if _kind(got) != "DataFrame":
        return False
    flags = [c for c in got.columns if str(c).endswith("_missing")]
    fill = re.search(r"\.\s*(fillna|ffill)\(", text)
    marks = [m.start() for m in re.finditer(r"\.\s*(isna|isnull)\(", text)]
    return (bool(flags) and not any(got[c].astype(bool).any() for c in flags)
            and fill is not None and any(i > fill.start() for i in marks))


def _new_bool_column(got) -> bool:
    """A table with a column of True and False that the files do not have: an indicator, under another name."""
    return _kind(got) == "DataFrame" and any(got[c].dtype.kind == "b" and not str(c).endswith("_missing") for c in got.columns)


EXERCISES["fill-flag"] = FunctionBox(
    name="fill-flag",
    function="fill_flag",
    start=_FLAG_START,
    solution=_FLAG,
    visible=[(STATION, "temp")],
    hidden=[(STATION_2, "temp"), (STATION_2, "rain"), (STATION, "wind"), (PENGUINS, "body_mass_g")],
    compare=_same_flagged,
    show=_shown_table,
    first=[_DOTS],
    mistakes=[
        Mistake("name-in-quotes", error=lambda e: isinstance(e, KeyError) and bool(e.args) and e.args[0] == "column",
                en="`df[\"column\"]` looks for a column that is called `column`. Use the name `column` itself, without quotes: `df[column]`; it holds the name the check gives, such as `\"temp\"`.",
                nl="`df[\"column\"]` zoekt een kolom die `column` heet. Gebruik de naam `column` zelf, zonder aanhalingstekens: `df[column]`; daarin zit de naam die de check geeft, zoals `\"temp\"`."),
        Mistake("no-parentheses-isna", matches=lambda got, text: _method_in(got) in ("isna", "isnull", "notna", "notnull"),
                en="`{call}` puts `isna` itself in the missing-indicator column, not `True` and `False`: `isna` needs its parentheses, also with nothing between them, `df[column].isna()`.",
                nl="`{call}` zet `isna` zelf in de missing-indicatorkolom, niet `True` en `False`: `isna` heeft zijn haakjes nodig, ook als er niets tussen staat, `df[column].isna()`."),
        Mistake("no-parentheses-mean", matches=lambda got, text: _method_in(got) == "mean",
                en="`{call}` puts `mean` itself in the gaps, not the mean: a method needs its parentheses, also with nothing between them. And the task asks for the median: `df[column].median()`.",
                nl="`{call}` zet `mean` zelf in de gaten, niet het gemiddelde: een methode heeft zijn haakjes nodig, ook als er niets tussen staat. En de opdracht vraagt om de mediaan: `df[column].median()`."),
        Mistake("no-parentheses", matches=lambda got, text: _callable_in(got),
                en="`{call}` puts a method itself in the gaps, not what it works out: a method needs its parentheses, also with nothing between them, as in `df[column].median()`.",
                nl="`{call}` zet een methode zelf in de gaten, niet wat die uitrekent: een methode heeft zijn haakjes nodig, ook als er niets tussen staat, zoals in `df[column].median()`."),
        Mistake("table-median", syntax=_TABLE_MEDIAN,
                en="Python stops on the line `{line}`: `df.median()` and `df.mean()` work on every column of the table, and some columns hold text, such as `sky`. Take the median of the one column: `df[column].median()`.",
                nl="Python stopt op de regel `{line}`: `df.median()` en `df.mean()` werken op elke kolom van de tabel, en in sommige kolommen staat tekst, zoals in `sky`. Neem de mediaan van alleen die ene kolom: `df[column].median()`."),
        Mistake("flag-after", matches=_flag_after,
                en="`{call}` gives an indicator that is `False` on every row: your function marks the gaps after it filled them in, and by then nothing is missing. Mark first, then fill in: swap the two lines.",
                nl="`{call}` geeft een indicator die op elke rij `False` is: je functie markeert de gaten nadat hij ze heeft ingevuld, en dan ontbreekt er niets meer. Eerst markeren, dan invullen: wissel de twee regels om."),
        Mistake("inplace", matches=lambda got, text: _INPLACE.search(text) is not None,
                en="`{call}` gives {got}. With `inplace=True` on `df[column]`, this version of pandas does not change `df`: the gaps stay. Leave out `inplace=True`, and put what `fillna` gives back in the column: `df[column] = df[column].fillna(...)`.",
                nl="`{call}` geeft {got}. Met `inplace=True` op `df[column]` verandert deze versie van pandas `df` niet: de gaten blijven. Laat `inplace=True` weg, en zet wat `fillna` teruggeeft in de kolom: `df[column] = df[column].fillna(...)`."),
        Mistake("not-kept", wrong=_flagged('    df[f"{column}_missing"] = df[column].isna()\n    df[column].fillna(df[column].median())\n'),
                en="`{call}` gives {got}: the gaps are still there. `fillna` gives back a new column and leaves `df` as it was. Put what it gives back in the column: `df[column] = df[column].fillna(...)`.",
                nl="`{call}` geeft {got}: de gaten zitten er nog. `fillna` geeft een nieuwe kolom terug en laat `df` zoals hij was. Zet wat hij teruggeeft in de kolom: `df[column] = df[column].fillna(...)`."),
        Mistake("inverted", wrong=_flagged('    df[f"{column}_missing"] = df[column].notna()\n    df[column] = df[column].fillna(df[column].median())\n'),
                en="`{call}` gives an indicator that is `True` where a value was there, and `False` where it was missing: the other way round. `df[column].isna()` is `True` for the gaps.",
                nl="`{call}` geeft een indicator die `True` is waar een waarde er was, en `False` waar hij ontbrak: precies andersom. `df[column].isna()` is `True` voor de gaten."),
        Mistake("mean", wrong=_flagged('    df[f"{column}_missing"] = df[column].isna()\n    df[column] = df[column].fillna(df[column].mean())\n'),
                en="`{call}` fills the gaps in with the mean of the column. The task asks for the median: `df[column].median()`.",
                nl="`{call}` vult de gaten in met het gemiddelde van de kolom. De opdracht vraagt om de mediaan: `df[column].median()`."),
        Mistake("whole-table", wrong=_flagged('    df[f"{column}_missing"] = df[column].isna()\n    df = df.fillna(df[column].median())\n'),
                en="`{call}` gives {got}. Your function fills every gap of the table with the median of `column`, also the gaps of other columns, such as the wind of day 17 with a temperature. Fill in only the column itself: `df[column] = df[column].fillna(...)`.",
                nl="`{call}` geeft {got}. Je functie vult elk gat van de tabel met de mediaan van `column`, ook de gaten van andere kolommen, zoals de wind van dag 17 met een temperatuur. Vul alleen de kolom zelf in: `df[column] = df[column].fillna(...)`."),
        Mistake("dropped", matches=lambda got, text: _kind(got) == "DataFrame" and "dropna" in text,
                en="`{call}` gives {got}: `dropna` leaves out the rows with a gap. The task keeps every row, marks the gaps, and fills them in.",
                nl="`{call}` geeft {got}: `dropna` laat de rijen met een gat weg. De opdracht houdt elke rij, markeert de gaten en vult ze in."),
        Mistake("one-column", matches=lambda got, text: _kind(got) == "Series",
                en="`{call}` gives {got}. The task asks for the whole table, `df`, with the indicator as a new column: give back `df`.",
                nl="`{call}` geeft {got}. De opdracht vraagt om de hele tabel, `df`, met de indicator als nieuwe kolom: geef `df` terug."),
        Mistake("no-f", matches=lambda got, text: _kind(got) == "DataFrame" and any(str(c).startswith("{") for c in got.columns),
                en="`{call}` gives a table in which the name of your indicator still has the braces in it. Without the `f` in front of the quotes, the text is not an f-string, and Python does not fill in the name from `column`.",
                nl="`{call}` geeft een tabel waarin de naam van je indicator de accolades nog heeft. Zonder de `f` voor de aanhalingstekens is de tekst geen f-string, en vult Python de naam uit `column` niet in."),
        Mistake("flag-name", matches=lambda got, text: _new_bool_column(got),
                en="`{call}` gives {got}. Your indicator has a different name from the one the task asks for: the name of the column with `_missing` after it, such as `temp_missing`. An f-string makes it from `column`.",
                nl="`{call}` geeft {got}. Je indicator heeft een andere naam dan de opdracht vraagt: de naam van de kolom met `_missing` erachter, zoals `temp_missing`. Een f-string maakt hem uit `column`."),
    ],
    texts={
        "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                    "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
        "empty": {"en": "Replace the `...` with two lines: the missing-indicator column, then the column filled in with its median.",
                  "nl": "Vervang de `...` door twee regels: de missing-indicatorkolom, en dan de kolom ingevuld met zijn mediaan."},
        "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                    "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
        "arguments": {"en": "The check calls `{call}`, with a path and a column name, and the `def` line of your function has a different number of names between its parentheses. Keep `(path, column)`.",
                      "nl": "De check roept `{call}` aan, met een pad en een kolomnaam, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(path, column)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Do you first make the indicator with `isna`, under the name of the column with `_missing` after it, then fill in `column` with its median, and give back `df`?",
                  "nl": "`{call}` geeft {got}. Het moet zijn: {want}. Maak je eerst de indicator met `isna`, onder de naam van de kolom met `_missing` erachter, vul je dan `column` in met zijn mediaan, en geef je `df` terug?"},
        "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does it work for every file and every column?",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}. Het moet zijn: {want}. Werkt hij voor elk bestand en elke kolom?"},
        "correct": {"en": "Your function is right in all {n} cases. In the month, `temp_missing` is `True` on days 9 and 22, and both now hold the median, 10.75 degrees: a guess you can still tell from a measurement.",
                    "nl": "Je functie klopt in alle {n} gevallen. In de maand is `temp_missing` `True` op dag 9 en 22, en op allebei staat nu de mediaan, 10,75 graden: een gok die je nog altijd van een meting kunt onderscheiden."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_W = "import pandas as pd\n\n" + _READ + "\n"
_F = "import pandas as pd\n\n\ndef fill_flag(path, column):\n    df = pd.read_csv(path, na_values=[-999])\n"
CASES = [
    ("filled", _FILLED_START, "empty", None),
    ("filled", _FILLED, "correct", None),
    ("filled", _W + 'median = df["wind"].median()\nfilled = df["wind"].fillna(median)\nfilled', "correct", None),
    ("filled", _W + 'filled = df.fillna({"wind": df["wind"].median()})["wind"]\nfilled', "correct", None),
    ("filled", _W + 'filled = df["wind"].fillna(df["wind"].quantile(0.5))\nfilled', "correct", None),
    ("filled", _W + 'filled = df["wind"].fillna(df["wind"].mean())\nfilled', "wrong", "mean"),
    ("filled", _W + 'filled = df["wind"].fillna(0)\nfilled', "wrong", "zero"),
    ("filled", _W + 'filled = df["wind"].dropna()\nfilled', "wrong", "dropped"),
    ("filled", _W + 'filled = df["wind"].ffill()\nfilled', "wrong", "previous-day"),
    ("filled", _W + 'filled = df["wind"]\nfilled', "wrong", "gap-left"),
    ("filled", _W + 'filled = df["wind"].fillna("median")\nfilled', "wrong", "text"),
    ("filled", _W + 'filled = df["wind"].median()\nfilled', "wrong", "one-number"),
    ("filled", _W + 'filled = df["wind"].mean()\nfilled', "wrong", "one-number"),
    ("filled", _W + 'filled = df["wind"].isna()\nfilled', "wrong", "mask"),
    ("filled", _W + 'filled = df.fillna(df["wind"].median())\nfilled', "wrong", "whole-table"),
    ("filled", _W + 'filled = df["wind"].fillna(df["wind"].median(), inplace=True)\nfilled', "correct", None),
    ("filled", _W + 'df["wind"].fillna(df["wind"].median(), inplace=True)\nfilled = df["wind"]\nfilled', "wrong", "inplace"),
    ("filled", _W + 'filled = df["wind"].fillna(df["wind"].median)\nfilled', "wrong", "no-parentheses"),
    ("filled", _W + 'filled = df["wind"].fillna(df["wind"].mean)\nfilled', "wrong", "no-parentheses-mean"),
    ("filled", _W + 'filled = df["wind"].fillna(df["wind"].max)\nfilled', "wrong", "no-parentheses"),
    ("filled", _W + 'filled = df["wind"].fillna(df.median())\nfilled', "error", "table-median"),
    ("filled", _W + 'filled = df["Wind"].fillna(df["Wind"].median())\nfilled', "error", "no-column"),
    ("filled", _W + 'filled = df["temp"].fillna(df["temp"].median())\nfilled', "wrong", None),
    ("filled", _W + 'filled = df["wind"].fillna(21)\nfilled', "almost", "other-value"),
    ("filled", _W + 'filled = df["wind"].fillna(df["wind"].median())\nprint(filled)', "wrong", "print-last"),
    ("filled", _W + 'filled = df["wind"].fillna(...)\nfilled', "wrong", "dots-left"),
    ("fill-flag", _FLAG_START, "empty", None),
    ("fill-flag", _FLAG, "correct", None),
    ("fill-flag", _F + '    df[column + "_missing"] = df[column].isna()\n    df[column] = df[column].fillna(df[column].median())\n    return df', "correct", None),
    ("fill-flag", _F + '    df[f"{column}_missing"] = df[column].isna().astype(int)\n    df[column] = df[column].fillna(df[column].median())\n    return df', "correct", None),
    ("fill-flag", _F + '    missing = df[column].isna()\n    df[column] = df[column].fillna(df[column].median())\n    df[f"{column}_missing"] = missing\n    return df', "correct", None),
    ("fill-flag", _F + '    df[column] = df[column].fillna(df[column].median())\n    df[f"{column}_missing"] = df[column].isna()\n    return df', "wrong", "flag-after"),
    ("fill-flag", _F + '    df[f"{column}_missing"] = df[column].isna()\n    df[column].fillna(df[column].median())\n    return df', "wrong", "not-kept"),
    ("fill-flag", _F + '    df[f"{column}_missing"] = df[column].isna()\n    df[column].fillna(df[column].median(), inplace=True)\n    return df', "wrong", "inplace"),
    ("fill-flag", _F + '    df[f"{column}_missing"] = df[column].notna()\n    df[column] = df[column].fillna(df[column].median())\n    return df', "wrong", "inverted"),
    ("fill-flag", _F + '    df[f"{column}_missing"] = df[column].isna()\n    df[column] = df[column].fillna(df[column].mean())\n    return df', "wrong", "mean"),
    ("fill-flag", _F + '    df[f"{column}_missing"] = df[column].isna()\n    df = df.fillna(df[column].median())\n    return df', "wrong", "whole-table"),
    ("fill-flag", _F + '    df[f"{column}_missing"] = df[column].isna()\n    df = df.dropna()\n    return df', "wrong", "dropped"),
    ("fill-flag", _F + '    df[f"{column}_missing"] = df[column].isna()\n    return df[column].fillna(df[column].median())', "wrong", "one-column"),
    ("fill-flag", _F + '    df["missing"] = df[column].isna()\n    df[column] = df[column].fillna(df[column].median())\n    return df', "wrong", "flag-name"),
    ("fill-flag", _F + '    df[column] = df[column].fillna(df[column].median())\n    return df', "wrong", None),
    ("fill-flag", _F + '    df["column_missing"] = df["column"].isna()\n    df[column] = df[column].fillna(df[column].median())\n    return df', "error", "name-in-quotes"),
    ("fill-flag", _F + '    df["temp_missing"] = df["temp"].isna()\n    df["temp"] = df["temp"].fillna(df["temp"].median())\n    return df', "almost", None),
    ("fill-flag", _F + '    df[f"{column}_missing"] = df[column].isna()\n    df[column] = df[column].fillna(df[column].median)\n    return df', "wrong", "no-parentheses"),
    ("fill-flag", _F + '    df[f"{column}_missing"] = df[column].isna\n    df[column] = df[column].fillna(df[column].median())\n    return df', "wrong", "no-parentheses-isna"),
    ("fill-flag", _F + '    df[f"{column}_missing"] = df[column].isna()\n    df[column] = df[column].fillna(df[column].mean)\n    return df', "wrong", "no-parentheses-mean"),
    ("fill-flag", _F + '    df[f"{column}_missing"] = df[column].isna()\n    df = df.fillna(df.median())\n    return df', "error", "table-median"),
    ("fill-flag", _F + '    df[f"{column}_missing"] = df[column] == None\n    df[column] = df[column].fillna(df[column].median())\n    return df', "wrong", None),
    ("fill-flag", _F + '    df.fillna({column: df[column].median()}, inplace=True)\n    df[f"{column}_missing"] = df[column].isnull()\n    return df', "wrong", "flag-after"),
    ("fill-flag", _F + '    df["{column}_missing"] = df[column].isna()\n    df[column] = df[column].fillna(df[column].median())\n    return df', "wrong", "no-f"),
    ("fill-flag", _F + '    df[f"{column}_missing"] = df[column].isna()\n    df[column] = df[column].fillna(0)\n    return df', "wrong", None),
    ("fill-flag", _F + '    df[f"{column}_missing"] = df[column].isna()\n    df.fillna({column: df[column].median()}, inplace=True)\n    return df', "correct", None),
    ("fill-flag", _F + '    df[f"{column}_missing"] = df[column].isna()\n    df.fillna({column: df[column].mean()}, inplace=True)\n    return df', "wrong", "mean"),
]
