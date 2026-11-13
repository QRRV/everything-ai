# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson ml-project-table-explore: the traffic table of the project and what is wrong with it, the
calendar of the play block, and the three rules the learner writes and keeps in the project: drop_impossible,
one_per_hour and clean, which builds on the other two."""
import re

import pandas as pd

from everything_ai import draw, paths, projects, yours
from everything_ai.check import FunctionBox, Mistake

# clean builds on the two rules of exercises 1 and 2: where yours is missing or broken, the course's stands in
projects.builds_on(("tabular", "impossible", "ml-project-table-explore", "impossible"),
                   ("tabular", "hours", "ml-project-table-explore", "hours"))

FILE = "data/metro-interstate-traffic.csv.gz"
SHOWN = ["date_time", "temp", "rain_1h", "weather_main", "traffic_volume"]   # the columns the pictures under the checks show


class Table:
    """A table the check calls by the name the lesson gives it, such as TRAFFIC, so that a check message stays short;
    the learner's code gets its own copy of the DataFrame."""

    def __init__(self, df: pd.DataFrame, name: str):
        self.df, self.name = df, name

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return self.df.copy()


def read() -> pd.DataFrame:
    return pd.read_csv(paths.ROOT / FILE)


RAW = read()
TRAFFIC = Table(RAW, "TRAFFIC")
COLUMNS = list(RAW.columns)


def _table(rows: list[tuple], name: str) -> Table:
    """A small table of the same columns: each row is (date_time, temp, rain_1h, weather_main, traffic_volume)."""
    out = pd.DataFrame([{"holiday": None, "temp": t, "rain_1h": r, "snow_1h": 0.0, "clouds_all": 40, "weather_main": w,
                         "weather_description": w.lower(), "date_time": d, "traffic_volume": v} for d, t, r, w, v in rows],
                       columns=COLUMNS)
    return Table(out, name)


# a storm that never was, an hour at 0 K, and an hour with two kinds of weather
STORM = _table([("2016-07-11 16:00:00", 301.2, 0.0, "Clouds", 5310), ("2016-07-11 17:00:00", 302.11, 9831.3, "Rain", 5535),
                ("2016-07-11 18:00:00", 300.5, 2.54, "Rain", 4802)], "STORM")
COLD = _table([("2014-01-31 02:00:00", 252.1, 0.0, "Clear", 402), ("2014-01-31 03:00:00", 0.0, 0.0, "Clear", 361),
               ("2014-01-31 04:00:00", 0.0, 0.0, "Clear", 734)], "COLD")
TWICE = _table([("2013-04-18 07:00:00", 276.4, 0.25, "Rain", 6512), ("2013-04-18 07:00:00", 276.4, 0.25, "Mist", 6512),
                ("2013-04-18 08:00:00", 277.0, 0.0, "Mist", 6020), ("2013-04-18 09:00:00", 277.9, 0.0, "Clouds", 5205),
                ("2013-04-18 09:00:00", 277.9, 0.0, "Haze", 5205)], "TWICE")
FINE = _table([("2017-09-02 12:00:00", 293.15, 0.0, "Clear", 4100), ("2017-09-02 13:00:00", 294.0, 0.0, "Clear", 4230)], "FINE")
# an hour with more rain than ever fell anywhere, but not the 9831.3 mm of the table: a limit copied from that one row
# keeps it
FLOOD = _table([("2015-08-20 15:00:00", 295.0, 0.0, "Clouds", 5020), ("2015-08-20 16:00:00", 294.1, 512.0, "Thunderstorm", 4710),
                ("2015-08-20 17:00:00", 293.0, 12.7, "Rain", 4890)], "FLOOD")


# --- the course's own rules, as the lesson writes them --------------------------------------------------------------

def drop_impossible(df):
    df = df[df["temp"] > 0]
    df = df[df["rain_1h"] <= 100]
    return df


def one_per_hour(df):
    return df.drop_duplicates(subset="date_time")


def clean(df):
    df = drop_impossible(df)
    df = one_per_hour(df)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df["temp"] = df["temp"] - 273.15
    return df


CLEAN = clean(RAW.copy())
TIMES = pd.to_datetime(RAW["date_time"])
HOURS = TIMES.drop_duplicates().sort_values()
SPAN = int((HOURS.iloc[-1] - HOURS.iloc[0]) / pd.Timedelta("1h")) + 1   # hours from the first to the last
_GAPS = HOURS.diff().reset_index(drop=True)
_AFTER = int(_GAPS.idxmax())
LONGEST = (HOURS.iloc[_AFTER - 1], HOURS.iloc[_AFTER])   # the hours on both sides of the longest gap

FACTS = {
    "rows": len(RAW),
    "zero_k": int((RAW["temp"] == 0).sum()),
    "storm": int((RAW["rain_1h"] > 100).sum()),
    "rain_max": float(RAW["rain_1h"].max()),
    "rain_next": float(RAW["rain_1h"][RAW["rain_1h"] <= 100].max()),
    "coldest": float(RAW["temp"][RAW["temp"] > 0].min()),
    "repeats": int(RAW["date_time"].duplicated().sum()),
    "same_rows": int(RAW.duplicated().sum()),
    "hours": len(HOURS),
    "span": SPAN,
    "missing": SPAN - len(HOURS),
    "gaps": int((_GAPS > pd.Timedelta("1h")).sum()),
    "longest_days": int((LONGEST[1] - LONGEST[0]) / pd.Timedelta("1D")),
    "holiday_rows": int(RAW["holiday"].notna().sum()),
    "clean_rows": len(CLEAN),
}


def grouped(n: int, lang: str) -> str:
    """A whole number with its thousands marked, as the language writes it: 48,204 or 48.204."""
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def number(x: float, digits: int, lang: str) -> str:
    text = f"{x:,.{digits}f}"
    return text.replace(",", "_").replace(".", ",").replace("_", ".") if lang == "nl" else text


# --- the play block: one year of the table as a calendar ---------------------------------------------------------

YEARS = list(range(HOURS.iloc[0].year, HOURS.iloc[-1].year + 1))


def year_facts(year: int) -> dict:
    """For one year: every day within the data, with the share of its hours in the table; the days on which an hour
    has more than one row; and the counts the sentence under the calendar gives."""
    first, last = HOURS.iloc[0], HOURS.iloc[-1]
    start, end = max(pd.Timestamp(year=year, month=1, day=1), first), min(pd.Timestamp(year=year, month=12, day=31, hour=23), last)
    every = pd.date_range(start, end, freq="h")
    have = HOURS[(HOURS >= start) & (HOURS <= end)]
    per_day_all = pd.Series(1, index=every).groupby(every.normalize()).sum()
    per_day = pd.Series(1, index=have.values).groupby(have.dt.normalize().values).sum()
    cells = {(d.month, d.day): float(per_day.get(d, 0)) / float(n) for d, n in per_day_all.items()}
    in_year = (TIMES >= start) & (TIMES <= end)
    twice = TIMES[in_year & RAW["date_time"].duplicated(keep=False)]
    marked = {(t.month, t.day) for t in twice}
    months_gone = [m for m in range(1, 13) if any(k[0] == m for k in cells) and all(v == 0 for k, v in cells.items() if k[0] == m)]
    return {"cells": cells, "marked": marked, "hours": len(every), "have": len(have),
            "repeats": int((in_year & RAW["date_time"].duplicated()).sum()), "months_gone": months_gone}


def calendar(year: int, months: list[str], legend: list[str], title: str, alt: str):
    f = year_facts(year)
    return draw.calendar(f["cells"], months=months, title=title, alt=alt, marked=f["marked"], legend=legend)


# --- how the learner's tables are judged ---------------------------------------------------------------------------

def show_table(v):
    """A table in words, for the check's messages: its number of rows, which is what the rules change."""
    if isinstance(v, pd.DataFrame):
        if len(v) == 1:
            return {"en": "a table of 1 row", "nl": "een tabel van 1 rij"}
        return {"en": f"a table of {grouped(len(v), 'en')} rows", "nl": f"een tabel van {grouped(len(v), 'nl')} rijen"}
    return {"en": f"`{v!r}`"[:60], "nl": f"`{v!r}`"[:60]}


def _left(got, column: str, test) -> bool:
    return isinstance(got, pd.DataFrame) and column in got.columns and len(got) > 0 and bool(test(got[column]).any())


def _only_impossible(got) -> bool:
    return (isinstance(got, pd.DataFrame) and len(got) > 0 and {"temp", "rain_1h"} <= set(got.columns)
            and bool(((got["temp"] == 0) | (got["rain_1h"] > 100)).all()))


def _text_dates(got) -> bool:
    return (isinstance(got, pd.DataFrame) and "date_time" in got.columns and len(got) > 0
            and not pd.api.types.is_datetime64_any_dtype(got["date_time"]))


def _temp_between(got, lo: float, hi: float) -> bool:
    return (isinstance(got, pd.DataFrame) and "temp" in got.columns and len(got) > 0
            and pd.api.types.is_numeric_dtype(got["temp"]) and lo < float(got["temp"].median()) < hi)


# The lesson is done when these are correct. "gap" is the choice question in the lesson.
REQUIRED = ["impossible", "hours", "clean", "gap"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
    "arguments": {"en": "The check calls `{call}`, with one table, and the `def` line of your function has another number of names between its parentheses. Keep `(df)`.",
                  "nl": "De check roept `{call}` aan, met één tabel, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(df)`."},
}

# --- exercise 1: drop_impossible ------------------------------------------------------------------------------------


def _i(temp: str = 'df = df[df["temp"] > 0]', rain: str = 'df = df[df["rain_1h"] <= 100]') -> str:
    return ("def drop_impossible(df):\n"
            f"    {temp}\n"
            f"    {rain}\n"
            "    return df\n")


_I_START = _i("...", "...")
_I = _i()

# --- exercise 2: one_per_hour ---------------------------------------------------------------------------------------


def _h(body: str = 'return df.drop_duplicates(subset="date_time")') -> str:
    return "def one_per_hour(df):\n" f"    {body}\n"


_H_START = _h("return ...")
_H = _h()

# --- exercise 3: clean ----------------------------------------------------------------------------------------------


def _c(dates: str = 'pd.to_datetime(df["date_time"])', temp: str = 'df["temp"] - 273.15') -> str:
    return ("import pandas as pd\n"
            "from tabular.impossible import drop_impossible\n"
            "from tabular.hours import one_per_hour\n\n\n"
            "def clean(df):\n"
            "    df = drop_impossible(df)\n"
            "    df = one_per_hour(df)\n"
            f'    df["date_time"] = {dates}\n'
            f'    df["temp"] = {temp}\n'
            "    return df\n")


_C_START = _c("...", "...")
_C = _c()

EXERCISES = {
    "impossible": FunctionBox(
        name="impossible",
        picture=yours.dropped(0, columns=SHOWN),
        function="drop_impossible",
        start=_I_START,
        solution=_I,
        show=show_table,
        save_as=("tabular", "impossible"),
        visible=[(TRAFFIC,)],
        hidden=[(STORM,), (COLD,), (FLOOD,), (FINE,)],
        first=[_DOTS],
        mistakes=[
            Mistake("none-left", matches=lambda got, text: isinstance(got, pd.DataFrame) and len(got) == 0,
                    en="`{call}` gives a table without rows: your rules leave out every hour. A mask in `df[...]` picks the rows to keep, so keep the hours above 0 K and with at most 100 mm of rain.",
                    nl="`{call}` geeft een tabel zonder rijen: je regels laten elk uur weg. Een mask in `df[...]` kiest de rijen die blijven, dus houd de uren boven 0 K en met hoogstens 100 mm regen."),
            Mistake("only-impossible", matches=lambda got, text: _only_impossible(got),
                    en="`{call}` gives {got}, and every one of them is impossible: your masks pick the rows you want to drop. A mask in `df[...]` picks the rows to keep, so keep the hours above 0 K and with at most 100 mm of rain.",
                    nl="`{call}` geeft {got}, en elke rij daarvan is onmogelijk: je masks kiezen de rijen die weg moeten. Een mask in `df[...]` kiest de rijen die blijven, dus houd de uren boven 0 K en met hoogstens 100 mm regen."),
            Mistake("zero-left", matches=lambda got, text: _left(got, "temp", lambda c: c == 0),
                    en="`{call}` gives {got}, and it should give {want}: some hours still have a temperature of 0 K. Keep only the rows where `temp` is above 0.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: sommige uren hebben nog een temperatuur van 0 K. Houd alleen de rijen waarin `temp` boven 0 ligt."),
            Mistake("storm-left", matches=lambda got, text: _left(got, "rain_1h", lambda c: c > 100),
                    en="`{call}` gives {got}, and it should give {want}: an hour with more than 100 mm of rain is still there. Keep only the rows where `rain_1h` is at most 100.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn: een uur met meer dan 100 mm regen staat er nog in. Houd alleen de rijen waarin `rain_1h` hoogstens 100 is."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...` with the two rules: one line that keeps the hours above 0 K, and one that keeps the hours with at most 100 mm of rain.",
                      "nl": "Vervang de twee `...` door de twee regels: één regel die de uren boven 0 K houdt, en één die de uren met hoogstens 100 mm regen houdt."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Your rules leave out other rows than the impossible ones: the picture under this message shows which.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Je regels laten andere rijen weg dan de onmogelijke: het plaatje onder dit bericht laat zien welke."},
            "almost": {"en": "Right for the whole table. But `{call}` gives {got}, and it should give {want}. Does your function also work for other tables? A rule should catch every impossible value, not only the ones in this table.",
                       "nl": "Goed voor de hele tabel. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook voor andere tabellen? Een regel moet elke onmogelijke waarde vangen, niet alleen die in deze tabel."},
            "correct": {"en": "Your rules are right in all {n} cases: they leave out the 11 impossible hours, and nothing else. Your `drop_impossible` is now in your project, in `{saved}`.",
                        "nl": "Je regels kloppen in alle {n} gevallen: ze laten de 11 onmogelijke uren weg, en niets anders. Je `drop_impossible` staat nu in je project, in `{saved}`."},
            "kept": {"en": "Your `drop_impossible` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `drop_impossible` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "hours": FunctionBox(
        name="hours",
        picture=yours.dropped(0, columns=SHOWN),
        function="one_per_hour",
        start=_H_START,
        solution=_H,
        show=show_table,
        save_as=("tabular", "hours"),
        visible=[(TRAFFIC,)],
        hidden=[(TWICE,), (FINE,)],
        first=[_DOTS],
        mistakes=[
            Mistake("whole-rows", wrong=_h("return df.drop_duplicates()"),
                    en="`{call}` gives {got}, and it should give {want}. `drop_duplicates()` without a column only leaves out rows that are the same in every column. An hour with two kinds of weather differs in `weather_main`, so it stays twice. Name the column that must not repeat: `subset=\"date_time\"`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. `drop_duplicates()` zonder kolom laat alleen rijen weg die in elke kolom gelijk zijn. Een uur met twee soorten weer verschilt in `weather_main`, dus het blijft twee keer staan. Noem de kolom die niet dubbel mag: `subset=\"date_time\"`."),
            Mistake("kept-last", wrong=_h('return df.drop_duplicates(subset="date_time", keep="last")'),
                    en="`{call}` keeps one row per hour, but the last of each hour, and the rule keeps the first. Leave out `keep=\"last\"`: keeping the first is what `drop_duplicates` does by itself.",
                    nl="`{call}` houdt één rij per uur, maar de laatste van elk uur, en de regel houdt de eerste. Laat `keep=\"last\"` weg: de eerste houden doet `drop_duplicates` vanzelf."),
            Mistake("only-repeats", wrong=_h('return df[df.duplicated(subset="date_time")]'),
                    en="`{call}` gives {got}: only the rows that repeat an hour, and none of the first rows. Leave the repeats out instead, with `df.drop_duplicates(subset=\"date_time\")`.",
                    nl="`{call}` geeft {got}: alleen de rijen die een uur herhalen, en geen van de eerste rijen. Laat de herhalingen juist weg, met `df.drop_duplicates(subset=\"date_time\")`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with the rule: give back the table with one row for every hour, the first one.",
                      "nl": "Vervang de `...` door de regel: geef de tabel terug met één rij voor elk uur, de eerste."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}: one row for each of the hours in the table. The picture under this message shows which rows your code leaves out.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn: één rij voor elk uur in de tabel. Het plaatje onder dit bericht laat zien welke rijen jouw code weglaat."},
            "almost": {"en": "Right for the whole table. But `{call}` gives {got}, and it should give {want}. Does your function also work for a table where no hour repeats?",
                       "nl": "Goed voor de hele tabel. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie ook voor een tabel waarin geen uur dubbel staat?"},
            "correct": {"en": "Your rule is right in all {n} cases: one row for every hour, the first. Your `one_per_hour` is now in your project, in `{saved}`.",
                        "nl": "Je regel klopt in alle {n} gevallen: één rij voor elk uur, de eerste. Je `one_per_hour` staat nu in je project, in `{saved}`."},
            "kept": {"en": "Your `one_per_hour` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `one_per_hour` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
    "clean": FunctionBox(
        name="clean",
        picture=yours.table(0, columns=SHOWN),
        function="clean",
        start=_C_START,
        solution=_C,
        show=show_table,
        save_as=("tabular", "clean"),
        visible=[(TRAFFIC,)],
        hidden=[(STORM,), (COLD,), (TWICE,), (FINE,)],
        first=[_DOTS],
        mistakes=[
            Mistake("whole-table", error=lambda e: isinstance(e, (ValueError, TypeError)) and re.search(r"assemble|to_datetime|arg must be", str(e)) is not None,
                    en="With `{call}`, your function stops with an error: `pd.to_datetime` gets the whole table, and it needs one column. Give it `df[\"date_time\"]`.",
                    nl="Met `{call}` stopt je functie met een fout: `pd.to_datetime` krijgt de hele tabel, en hij heeft één kolom nodig. Geef hem `df[\"date_time\"]`."),
            Mistake("text-dates", matches=lambda got, text: _text_dates(got),
                    en="`{call}` gives a table whose `date_time` is still text, such as `\"2012-10-02 09:00:00\"`. Make it a column of moments in time with `pd.to_datetime(df[\"date_time\"])`.",
                    nl="`{call}` geeft een tabel waarin `date_time` nog tekst is, zoals `\"2012-10-02 09:00:00\"`. Maak er een kolom momenten in de tijd van met `pd.to_datetime(df[\"date_time\"])`."),
            Mistake("still-kelvin", matches=lambda got, text: _temp_between(got, 200, 350),
                    en="`{call}` gives a table whose `temp` is still in kelvin, around 280. Degrees Celsius are kelvin minus 273.15.",
                    nl="`{call}` geeft een tabel waarin `temp` nog in kelvin staat, rond 280. Graden Celsius zijn kelvin min 273,15."),
            Mistake("plus", matches=lambda got, text: _temp_between(got, 350, 700),
                    en="`{call}` gives a table whose `temp` is around 550: you added 273.15. Degrees Celsius are kelvin *minus* 273.15.",
                    nl="`{call}` geeft een tabel waarin `temp` rond 550 ligt: je telde 273,15 op. Graden Celsius zijn kelvin *min* 273,15."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: make `date_time` a column of moments in time, and the temperature degrees Celsius.",
                      "nl": "Vervang de twee `...`: maak van `date_time` een kolom momenten in de tijd, en van de temperatuur graden Celsius."},
            "wrong": {"en": "`{call}` does not give the right table. The picture under this message shows the first rows of yours beside the right ones, with their number of rows and columns, and the cells that differ marked.",
                      "nl": "`{call}` geeft niet de goede tabel. Het plaatje onder dit bericht laat de eerste rijen van de jouwe naast de goede zien, met hun aantal rijen en kolommen, en de cellen die verschillen gemarkeerd."},
            "almost": {"en": "Right for the whole table. But `{call}` does not give the right table. Does your function work for any table: your two rules first, then the two changes to the columns?",
                       "nl": "Goed voor de hele tabel. Maar `{call}` geeft niet de goede tabel. Werkt je functie voor elke tabel: eerst je twee regels, dan de twee veranderingen aan de kolommen?"},
            "correct": {"en": "Your `clean` is right in all {n} cases, with your two rules from exercises 1 and 2. It is now in your project, in `{saved}`: the next step of the project starts from it.",
                        "nl": "Je `clean` klopt in alle {n} gevallen, met je twee regels uit oefening 1 en 2. Hij staat nu in je project, in `{saved}`: de volgende stap van het project begint ermee."},
            "kept": {"en": "Your `clean` is right. You changed `{saved}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
                     "nl": "Je `clean` klopt. Je hebt `{saved}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."},
        },
    ),
}

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
CASES = [
    ("impossible", _I_START, "empty", None),
    ("impossible", _i(rain="..."), "wrong", "dots-left"),
    ("impossible", _i('df = df[df["temp"] == 0]', 'df = df[df["rain_1h"] > 100]'), "wrong", "none-left"),
    ("impossible", _i('df = df[df["temp"] == 0]', "df = df"), "wrong", "only-impossible"),
    ("impossible", _i("df = df"), "wrong", "zero-left"),
    ("impossible", _i(rain="df = df"), "wrong", "storm-left"),
    ("impossible", _i(rain='df = df[df["rain_1h"] == 0]'), "wrong", None),
    ("impossible", _i(rain='df = df[df["rain_1h"] < 9831.3]'), "wrong", "storm-left"),
    ("impossible", _I, "correct", None),
    ("impossible", _i('df = df[df["temp"] != 0]', 'df = df[df["rain_1h"] < 100]'), "correct", None),
    ("impossible", _i('df = df[df["rain_1h"] <= 100]', 'df = df[df["temp"] > 0]'), "correct", None),
    ("hours", _H_START, "empty", None),
    ("hours", _h("return df.drop_duplicates()"), "wrong", "whole-rows"),
    ("hours", _h('return df.drop_duplicates(subset="date_time", keep="last")'), "wrong", "kept-last"),
    ("hours", _h('return df[df.duplicated(subset="date_time")]'), "wrong", "only-repeats"),
    ("hours", _H, "correct", None),
    ("hours", _h('return df.drop_duplicates("date_time")'), "correct", None),
    ("hours", _h('return df.drop_duplicates(subset=["date_time"], keep="first")'), "correct", None),
    ("clean", _C_START, "empty", None),
    ("clean", _c(temp="..."), "wrong", "dots-left"),
    ("clean", _c('df["date_time"]'), "wrong", "text-dates"),
    ("clean", _c("pd.to_datetime(df)"), "error", "whole-table"),
    ("clean", _c(temp='df["temp"]'), "wrong", "still-kelvin"),
    ("clean", _c(temp='df["temp"] + 273.15'), "wrong", "plus"),
    ("clean", _c(temp='df["temp"] - 273'), "wrong", None),
    ("clean", _C, "correct", None),
    ("clean", _c(temp='df["temp"] + -273.15'), "correct", None),
]


# --- what your project can do now: your clean on the whole table ---------------------------------------------------

_DEMO_RUNS: dict = {}


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file)."""
    texts = []
    for module in ("impossible", "hours", "clean"):
        path = projects.folder() / "tabular" / f"{module}.py"
        texts.append(path.read_text(encoding="utf-8") if path.exists() else None)
    return tuple(texts)


def _demo() -> dict | str | None:
    from everything_ai.check import run_code, with_time_limit
    if not (projects.folder() / "tabular" / "clean.py").exists():
        return None
    run = run_code("from tabular.clean import clean\nfrom tabular.impossible import drop_impossible")
    if run.error is not None:
        return "clean"
    stand_ins = [f"{package}/{module}.py" for package, module, _, _ in projects.stand_ins()]
    try:
        first = with_time_limit(lambda: run.names["drop_impossible"](RAW.copy()), 30)
        out = with_time_limit(lambda: run.names["clean"](RAW.copy()), 30)
    except Exception:   # also Timeout
        return "clean"
    if not (isinstance(out, pd.DataFrame) and {"date_time", "temp", "traffic_volume"} <= set(out.columns)
            and pd.api.types.is_datetime64_any_dtype(out["date_time"])):
        return "clean"
    t = out["date_time"]
    weekend = t.dt.dayofweek >= 5
    by_hour = {"work": out["traffic_volume"][~weekend].groupby(t[~weekend].dt.hour).mean(),
               "weekend": out["traffic_volume"][weekend].groupby(t[weekend].dt.hour).mean()}
    hours = t.drop_duplicates().sort_values()
    gaps = hours.diff()
    return {"rows": len(RAW), "first": len(first) if isinstance(first, pd.DataFrame) else None, "clean": len(out),
            "repeats": int(t.duplicated().sum()), "gaps": int((gaps > pd.Timedelta("1h")).sum()),
            "temp": (float(out["temp"].min()), float(out["temp"].max())), "by_hour": by_hour, "stand_ins": stand_ins}


def demo() -> dict | str | None:
    """Your `clean` from your project on the whole table, worked out once for the project files as they are: the
    rows after your first rule and after all of clean, the hours that still repeat, the gaps that stay, the range of
    the temperature, and the mean traffic for every hour of the day on working days and at weekends. None while
    clean.py is not in your project; "clean" when your clean stops with an error or does not give back the table
    with a date_time of moments in time. The course's version stands in for a module of yours that clean imports
    and that is missing or broken."""
    key = _project_texts()
    if key not in _DEMO_RUNS:
        _DEMO_RUNS.clear()
        _DEMO_RUNS[key] = _demo()
    return _DEMO_RUNS[key]


def hours_chart(by_hour: dict, labels: dict, lang: str) -> str:
    """The mean traffic for every hour of the day: on working days solid and at weekends dashed, each named at its
    peak, the working days above their line and the weekends under theirs, where the other line is not."""
    from everything_ai import chart
    top = max(float(s.max()) for s in by_hour.values() if len(s)) if any(len(s) for s in by_hour.values()) else 1.0
    p = chart.Panel((0, 23), (0, top * 1.18), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=[0, 6, 12, 18, 23])
    for key, dashed in (("work", False), ("weekend", True)):
        s = by_hour[key]
        if len(s):
            p.line(list(s.index), list(s.values), "data", 2.2, dashed=dashed)
            peak = int(s.idxmax())
            p.text(peak, float(s.max()) + (-top * 0.1 if dashed else top * 0.07), labels[key], "data")
    svg = chart.figure([p], alt=labels["alt"], width=400, height=260)
    return chart.with_decimal_comma(svg) if lang == "nl" else svg
