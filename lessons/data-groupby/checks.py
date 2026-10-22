# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson data-groupby: the mean temperature per kind of sky in the month of the made-up weather station
of data-explore, and a function that gives the mean and the count of a column per group. And the pictures: the body
mass of the Palmer penguins by species, each species with its mean, and by island, with a slider for how many of
Biscoe's Gentoos are counted."""
import re

import numpy as np
import pandas as pd   # here, so the first import of pandas never counts against the time limit of a check

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake

PENGUINS = "data/penguins.csv"
STATION = "lessons/data-explore/station.csv"
STATION_2 = "lessons/data-explore/station-2.csv"
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]   # sorted by name, as groupby sorts them; group colour 1, 2 and 3
ISLANDS = ["Biscoe", "Dream", "Torgersen"]
GENTOOS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 123]   # the stops of the slider: Gentoos counted on Biscoe


def penguins():
    return pd.read_csv(PENGUINS)


def means(by: str) -> dict:
    """The mean body mass in each group, as the worked example prints it, rounded to grams."""
    return {k: round(v) for k, v in penguins().groupby(by)["body_mass_g"].mean().items()}


def _weighed() -> pd.DataFrame:
    """The 342 penguins with a body mass, each with a fixed place a little beside the middle of its group, so the dots
    do not hide each other and stay put when the slider moves."""
    df = penguins()
    df = df[df["body_mass_g"].isna() == False].reset_index(drop=True)
    df["jitter"] = np.random.default_rng(3).uniform(-0.24, 0.24, len(df))
    return df


def counted(k: int) -> pd.DataFrame:
    """The penguins with a body mass, with only the first k of Biscoe's 123 Gentoos, in the order of the table."""
    df = _weighed()
    gentoo = (df["species"] == "Gentoo").to_numpy()   # all Gentoos live on Biscoe
    keep = ~gentoo | (np.cumsum(gentoo) <= k)
    return df[keep]


def island_means(k: int) -> dict:
    """The mean body mass of each island, rounded to grams, with k of Biscoe's Gentoos counted."""
    return {name: round(v) for name, v in counted(k).groupby("island")["body_mass_g"].mean().items()}


def _panel(df, by: str, names: list, labels: dict):
    """One chart of the body masses grouped by `by`: every penguin a dot in the colour and shape of its species
    (style guide 1.7: three groups), and a short ink line at the mean of each group, the one number groupby gives."""
    from everything_ai import chart
    p = chart.Panel((-0.6, 2.6), (2500, 6900), title=labels["title"], xlabel=labels["x"], ylabel=labels["mass"],
                    xticks=[], yticks=[3000, 4000, 5000, 6000])
    place = np.array([names.index(v) for v in df[by]]) + df["jitter"].to_numpy()
    for g, name in enumerate(SPECIES):
        rows = (df["species"] == name).to_numpy()
        if rows.any():
            p.group(place[rows], df["body_mass_g"].to_numpy()[rows], g, r=2.6)
    for i, name in enumerate(names):
        mean = df[df[by] == name]["body_mass_g"].mean()
        p.line([i - 0.36, i + 0.36], [mean, mean], "ink", 3.4)
        p.text(i, 6600, name, "ink-2")
    return chart.figure([p], alt=labels["alt"], width=460, height=300)


def picture(labels: dict[str, str]) -> str:
    """The body masses grouped by species, each species with its mean."""
    return _panel(_weighed(), "species", SPECIES, labels)


def island_picture(k: int, labels: dict[str, str]) -> str:
    """The body masses grouped by island, with k of Biscoe's 123 Gentoos counted: drawn once for every stop of the
    slider (L.explore). With all of them, Biscoe's mean lies between its Adelies and its heavier Gentoos; with none,
    it is where the Adelies of the other islands are."""
    return _panel(counted(k), "island", ISLANDS, labels)


# The lesson is done when these are correct. "one-day" is the choice question in the lesson.
REQUIRED = ["per-sky", "summary", "one-day"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")


def _kind(got) -> str:
    return type(got).__name__


def _number(got) -> bool:
    return _kind(got) in ("int", "int64", "float", "float64") and not isinstance(got, bool)


def _one_column_table(got) -> bool:
    return _kind(got) == "DataFrame" and got.shape[1] == 1


def _labels(value) -> list[str]:
    return [str(k) for k in value.index]


def _same_groups(a, b) -> bool:
    """The same numbers for the same groups. In a summary per group the row labels are the groups, so here they
    count, where the course's own comparison leaves them out (a selection keeps its old labels). Only a Series is the
    same as a Series: a list of the means without their groups is not, and neither is a Series with its counts in a
    tuple, which the course's own comparison cannot compare at all."""
    from everything_ai.check import same_value
    return _kind(a) == "Series" and _kind(b) == "Series" and _labels(a) == _labels(b) and same_value(a, b)


def _bare_numbers(got) -> bool:
    """Numbers without their groups, such as `.tolist()` or `.values` of the means."""
    if _kind(got) == "ndarray":
        return got.dtype.kind in "iuf"
    return isinstance(got, list) and len(got) > 0 and all(_number(x) for x in got)


_ROUNDED = re.compile(r"\bround\(")


# --- exercise 1: the mean temperature per kind of sky -----------------------------------------------------------------

_READ = f'df = pd.read_csv("{STATION}")'
_SKY_START = "import pandas as pd\n\n" + _READ + "\nper_sky = ...\nper_sky\n"
_SKY = "import pandas as pd\n\n" + _READ + '\nper_sky = df.groupby("sky")["temp"].mean()\nper_sky'


def _sky(new: str) -> str:
    return _SKY.replace('per_sky = df.groupby("sky")["temp"].mean()', new)


_TEXT_MEAN = Mistake("text-mean", error=lambda e: isinstance(e, TypeError) and "does not support operation 'mean'" in str(e),
                     en="Your code takes the mean of a column of text, such as `sky` or `dir`, and a text has no mean. The groups come from `sky`; the mean from one column of numbers, `temp`: `df.groupby(\"sky\")[\"temp\"]`.",
                     nl="Je code neemt het gemiddelde van een kolom met tekst, zoals `sky` of `dir`, en een tekst heeft geen gemiddelde. De groepen komen uit `sky`; het gemiddelde uit één kolom met getallen, `temp`: `df.groupby(\"sky\")[\"temp\"]`.")
_NO_COLUMN = Mistake("no-column", error=lambda e: isinstance(e, KeyError),
                     en="The table has no column with that name. Its columns are `day`, `temp`, `rain`, `wind`, `dir` and `sky`, in small letters, each between quotes.",
                     nl="De tabel heeft geen kolom met die naam. De kolommen zijn `day`, `temp`, `rain`, `wind`, `dir` en `sky`, in kleine letters, elk tussen aanhalingstekens.")
_GROUPS_ONLY = Mistake("groups-only", matches=lambda got, text: _kind(got) in ("SeriesGroupBy", "DataFrameGroupBy"),
                       en="That is the groups themselves, not yet a number per group: pandas only shows what kind of thing it is. Ask for one number per group, the mean: `.mean()` at the end.",
                       nl="Dat zijn de groepen zelf, nog geen getal per groep: pandas laat alleen zien wat voor ding het is. Vraag om één getal per groep, het gemiddelde: `.mean()` aan het eind.")
_ITSELF = Mistake("itself", matches=lambda got, text: callable(got),
                  en="Your code gives back something that still needs its parentheses: `.mean` instead of `.mean()`. Without them, you get `mean` itself, not the means it gives.",
                  nl="Je code geeft iets terug dat nog zijn haakjes nodig heeft: `.mean` in plaats van `.mean()`. Zonder haakjes krijg je `mean` zelf, niet de gemiddelden die het geeft.")


EXERCISES = {
    "per-sky": CodeBox(
        name="per-sky",
        start=_SKY_START,
        solution=_SKY,
        keep=["import pandas as pd", _READ],
        # The check also runs the box with another month, with other kinds of sky and other temperatures missing.
        others=[{"df": Code(f'pd.read_csv("{STATION_2}")')}],
        first=[_DOTS],
        compare=_same_groups,
        digits=2,   # one number in a message, such as the mean of the whole month: 9.66, not 9.660714285714286
        mistakes=[
            _TEXT_MEAN, _NO_COLUMN, _GROUPS_ONLY, _ITSELF,
            Mistake("one-number", matches=lambda got, text: _number(got),
                    en="Your last line gives one number, `{got}`, and the task asks for a mean for each kind of sky. Split the rows into groups first: `df.groupby(\"sky\")`.",
                    nl="Je laatste regel geeft één getal, `{got}`, en de opdracht vraagt om een gemiddelde voor elke soort lucht. Splits de rijen eerst in groepen: `df.groupby(\"sky\")`."),
            Mistake("two-things", matches=lambda got, text: isinstance(got, tuple),
                    en="Your last line gives several things at once, with commas between them. The check looks at one: make the last line just `per_sky`.",
                    nl="Je laatste regel geeft meerdere dingen tegelijk, met komma's ertussen. De check kijkt naar één ding: maak van de laatste regel alleen `per_sky`."),
            Mistake("no-labels", matches=lambda got, text: _bare_numbers(got),
                    en="Your last line gives the numbers without the kinds of sky they belong to: a list or an array has no row labels. Leave out `.tolist()` or `.values`, so that the Series keeps each kind of sky as its row label.",
                    nl="Je laatste regel geeft de getallen zonder de soorten lucht waar ze bij horen: een lijst of een array heeft geen rijlabels. Laat `.tolist()` of `.values` weg, dan houdt de Series elke soort lucht als rijlabel."),
            Mistake("counts", wrong=_sky('per_sky = df.groupby("sky")["temp"].count()'),
                    en="Your last line gives how many temperatures each kind of sky has, not how warm it was. The task asks for the mean: `.mean()`.",
                    nl="Je laatste regel geeft hoeveel temperaturen elke soort lucht heeft, niet hoe warm het was. De opdracht vraagt om het gemiddelde: `.mean()`."),
            Mistake("sizes", wrong=_sky('per_sky = df.groupby("sky").size()'),
                    en="Your last line gives how many days each kind of sky has: `size` counts rows. The task asks for the mean temperature: take the column `temp`, then `.mean()`.",
                    nl="Je laatste regel geeft hoeveel dagen elke soort lucht heeft: `size` telt rijen. De opdracht vraagt om de gemiddelde temperatuur: neem de kolom `temp`, en dan `.mean()`."),
            Mistake("median", wrong=_sky('per_sky = df.groupby("sky")["temp"].median()'),
                    en="Your last line gives the median of each kind of sky. The task asks for the mean: `.mean()`.",
                    nl="Je laatste regel geeft de mediaan van elke soort lucht. De opdracht vraagt om het gemiddelde: `.mean()`."),
            Mistake("double-brackets", matches=lambda got, text: _one_column_table(got) and "[[" in text,
                    en="Those are the right means, but in a table with one column, because of the two pairs of brackets around `\"temp\"`. For one column on its own, use one pair: `[\"temp\"]`.",
                    nl="Dat zijn de goede gemiddelden, maar in een tabel met één kolom, door de twee paar haken om `\"temp\"`. Voor één kolom op zichzelf gebruik je één paar: `[\"temp\"]`."),
            Mistake("rounded", matches=lambda got, text: _ROUNDED.search(text) is not None,
                    en="Those may be the right means, but rounded. The check compares them with the means pandas gives, to many digits: leave out the rounding.",
                    nl="Dat kunnen de goede gemiddelden zijn, maar dan afgerond. De check vergelijkt ze met de gemiddelden die pandas geeft, tot op veel cijfers: laat het afronden weg."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the mean temperature for each kind of sky.",
                      "nl": "Vervang de `...` door de gemiddelde temperatuur voor elke soort lucht."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another month by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere maand."},
            "no_value": {"en": "The last line of the box shows nothing. Put `per_sky` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `per_sky` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives:\n\n```\n{got}\n```\n\nand it should be one mean temperature for each of the four kinds of sky. Do you group by `sky`, and take the mean of `temp`?",
                      "nl": "Je laatste regel geeft:\n\n```\n{got}\n```\n\nen dat moet één gemiddelde temperatuur zijn voor elk van de vier soorten lucht. Groepeer je op `sky`, en neem je het gemiddelde van `temp`?"},
            "other": {"en": "Right for this month. But with {where}, your last line gives:\n\n```\n{got}\n```\n\nand it should give:\n\n```\n{want}\n```\n\nDo you work it out from `df` itself?",
                      "nl": "Goed voor deze maand. Maar met {where} geeft je laatste regel:\n\n```\n{got}\n```\n\nen dat moet dit zijn:\n\n```\n{want}\n```\n\nReken je het uit met `df` zelf?"},
            "other_error": {"en": "Right for this month. But with {where}, your code stops with an error. Do you work it out from `df` itself?",
                            "nl": "Goed voor deze maand. Maar met {where} stopt je code met een fout. Reken je het uit met `df` zelf?"},
            "correct": {"en": "Four kinds of sky, each with its mean temperature, in the order of their names: pandas puts a capital letter first, so `Sunny` comes before `cloudy`. The check also tried another month, and your code gave its means too.",
                        "nl": "Vier soorten lucht, elk met zijn gemiddelde temperatuur, op volgorde van hun naam: pandas zet een hoofdletter eerst, dus `Sunny` staat vóór `cloudy`. De check probeerde ook een andere maand, en je code gaf ook daar de gemiddelden."},
        },
    ),
}


# --- exercise 2: the mean and the count per group ---------------------------------------------------------------------

def _columns(value) -> dict | None:
    """A table as {column name: its values}, for comparing tables whose columns may come in another order."""
    if _kind(value) != "DataFrame":
        return None
    return {str(c): value[c].tolist() for c in value.columns}


def _same_summary(a, b) -> bool:
    """The same table: the same groups as row labels, and the same columns, in any order, with the same values, row
    by row. Two Series are compared as in exercise 1; anything else is never the same, such as the means and the
    counts as two Series in a tuple."""
    from everything_ai.check import same_value
    ca, cb = _columns(a), _columns(b)
    if ca is None or cb is None:
        return _same_groups(a, b)
    return _labels(a) == _labels(b) and set(ca) == set(cb) and all(same_value(ca[c], cb[c]) for c in ca)


def _shown_table(value) -> str:
    return str(value)


_SUMMARY_START = "import pandas as pd\n\n\ndef summary(path, by, column):\n    df = pd.read_csv(path)\n    ...\n"
_SUMMARY = ('import pandas as pd\n\n\ndef summary(path, by, column):\n    df = pd.read_csv(path)\n'
            '    return df.groupby(by)[column].agg(["mean", "count"])')


def _summary(new: str) -> str:
    return _SUMMARY.replace('    return df.groupby(by)[column].agg(["mean", "count"])', new)


EXERCISES["summary"] = FunctionBox(
    name="summary",
    function="summary",
    start=_SUMMARY_START,
    solution=_SUMMARY,
    visible=[(PENGUINS, "species", "body_mass_g")],
    hidden=[(PENGUINS, "island", "flipper_length_mm"), (STATION, "sky", "temp"), (STATION_2, "dir", "rain")],
    compare=_same_summary,
    show=_shown_table,
    first=[_DOTS],
    mistakes=[
        Mistake("text-mean", error=lambda e: isinstance(e, TypeError) and "does not support operation 'mean'" in str(e),
                en="With `{call}`, your code takes the mean of a column of text, such as `species`, and a text has no mean. The groups come from the column `by`; the mean and the count from the column `column`: `df.groupby(by)[column]`.",
                nl="Met `{call}` neemt je code het gemiddelde van een kolom met tekst, zoals `species`, en een tekst heeft geen gemiddelde. De groepen komen uit de kolom `by`; het gemiddelde en de telling uit de kolom `column`: `df.groupby(by)[column]`."),
        Mistake("name-in-quotes", error=lambda e: isinstance(e, KeyError) and bool(e.args) and str(e.args[0]) in ("by", "column", "Column not found: column", "Column not found: by"),
                en="`\"by\"` and `\"column\"` between quotes are the words themselves, and the table has no column with such a name. Use the names `by` and `column` without quotes: they hold the names the check gives, such as `\"species\"`.",
                nl="`\"by\"` en `\"column\"` tussen aanhalingstekens zijn de woorden zelf, en de tabel heeft geen kolom met zo'n naam. Gebruik de namen `by` en `column` zonder aanhalingstekens: daarin zitten de namen die de check geeft, zoals `\"species\"`."),
        Mistake("agg-list", error=lambda e: isinstance(e, ValueError) and "numeric_only" in str(e),
                en="`agg` takes the names of what to compute in one list, between square brackets: `.agg([\"mean\", \"count\"])`.",
                nl="`agg` krijgt de namen van wat hij uitrekent in één lijst, tussen vierkante haken: `.agg([\"mean\", \"count\"])`."),
        Mistake("two-things", matches=lambda got, text: isinstance(got, tuple),
                en="`{call}` gives back two things, with a comma between them: the means and the counts, each on its own. The task asks for one table with both, in the columns `mean` and `count`: `.agg([\"mean\", \"count\"])`.",
                nl="`{call}` geeft twee dingen terug, met een komma ertussen: de gemiddelden en de tellingen, elk apart. De opdracht vraagt om één tabel met allebei, in de kolommen `mean` en `count`: `.agg([\"mean\", \"count\"])`."),
        Mistake("rounded", matches=lambda got, text: _ROUNDED.search(text) is not None,
                en="`{call}` gives\n\n```\n{got}\n```\n\nand it should give\n\n```\n{want}\n```\n\nYour function rounds. The check compares with the means pandas gives, to many digits: leave out the rounding.",
                nl="`{call}` geeft\n\n```\n{got}\n```\n\nen dat moet dit zijn:\n\n```\n{want}\n```\n\nJe functie rondt af. De check vergelijkt met de gemiddelden die pandas geeft, tot op veel cijfers: laat het afronden weg."),
        Mistake("fixed-names", state="almost",
                matches=lambda got, text: re.search(r"groupby\(\s*[\"']|\]\s*\[\s*[\"']|\)\s*\[\s*[\"']", text) is not None,
                en="Right for the example, perhaps, but `{call}` gives\n\n```\n{got}\n```\n\nand it should give\n\n```\n{want}\n```\n\nYour function uses column names of its own choice. Use the names it gets: `df.groupby(by)[column]`.",
                nl="Misschien goed voor het voorbeeld, maar `{call}` geeft\n\n```\n{got}\n```\n\nen dat moet dit zijn:\n\n```\n{want}\n```\n\nJe functie gebruikt kolomnamen naar eigen keuze. Gebruik de namen die hij krijgt: `df.groupby(by)[column]`."),
        Mistake("size", wrong=_summary('    return df.groupby(by)[column].agg(["mean", "size"])'),
                en="`{call}` gives `size` next to the mean: the number of rows in each group, also the rows where `column` is missing. The task asks for how many values the mean rests on: `count`.",
                nl="`{call}` geeft `size` naast het gemiddelde: het aantal rijen in elke groep, ook de rijen waar `column` ontbreekt. De opdracht vraagt op hoeveel waarden het gemiddelde rust: `count`."),
        Mistake("mean-only", wrong=_summary("    return df.groupby(by)[column].mean()"),
                en="`{call}` gives only the mean of each group. The task asks for the count next to it, in one table: `.agg([\"mean\", \"count\"])`.",
                nl="`{call}` geeft alleen het gemiddelde van elke groep. De opdracht vraagt om de telling ernaast, in één tabel: `.agg([\"mean\", \"count\"])`."),
        Mistake("count-only", wrong=_summary("    return df.groupby(by)[column].count()"),
                en="`{call}` gives only the count of each group. The task asks for the mean next to it, in one table: `.agg([\"mean\", \"count\"])`.",
                nl="`{call}` geeft alleen de telling van elke groep. De opdracht vraagt om het gemiddelde ernaast, in één tabel: `.agg([\"mean\", \"count\"])`."),
        Mistake("no-groups", wrong=_summary('    return df[column].agg(["mean", "count"])'),
                en="`{call}` gives one mean and one count, for the whole column. The task asks for them per group: split the rows first, with `df.groupby(by)`.",
                nl="`{call}` geeft één gemiddelde en één telling, voor de hele kolom. De opdracht vraagt om die per groep: splits de rijen eerst, met `df.groupby(by)`."),
        Mistake("groups-only", matches=lambda got, text: _kind(got) in ("SeriesGroupBy", "DataFrameGroupBy"),
                en="`{call}` gives the groups themselves, not yet numbers per group. Ask for the mean and the count with `.agg([\"mean\", \"count\"])` at the end.",
                nl="`{call}` geeft de groepen zelf, nog geen getallen per groep. Vraag om het gemiddelde en de telling met `.agg([\"mean\", \"count\"])` aan het eind."),
        Mistake("double-brackets", matches=lambda got, text: _kind(got) == "DataFrame" and got.columns.nlevels == 2,
                en="`{call}` gives the right numbers, but under two rows of column names, the name of the column above `mean` and `count`, because of the two pairs of brackets around `column`. For one column on its own, use one pair: `[column]`.",
                nl="`{call}` geeft de goede getallen, maar onder twee rijen kolomnamen, de naam van de kolom boven `mean` en `count`, door de twee paar haken om `column`. Voor één kolom op zichzelf gebruik je één paar: `[column]`."),
    ],
    texts={
        "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                    "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
        "empty": {"en": "Replace the `...` with the mean and the count of `column` for each group of `by`, given back with `return`.",
                  "nl": "Vervang de `...` door het gemiddelde en de telling van `column` voor elke groep van `by`, teruggegeven met `return`."},
        "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                    "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
        "arguments": {"en": "The check calls `{call}`, with a path and two column names, and the `def` line of your function has a different number of names between its parentheses. Keep `(path, by, column)`.",
                      "nl": "De check roept `{call}` aan, met een pad en twee kolomnamen, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(path, by, column)`."},
        "wrong": {"en": "`{call}` gives\n\n```\n{got}\n```\n\nand it should give\n\n```\n{want}\n```\n\nDo you group by `by`, take the column `column`, and ask for `mean` and `count`?",
                  "nl": "`{call}` geeft\n\n```\n{got}\n```\n\nen dat moet dit zijn:\n\n```\n{want}\n```\n\nGroepeer je op `by`, neem je de kolom `column`, en vraag je om `mean` en `count`?"},
        "almost": {"en": "Right for the example. But `{call}` gives\n\n```\n{got}\n```\n\nand it should give\n\n```\n{want}\n```\n\nDoes it work for every file and every two columns?",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft\n\n```\n{got}\n```\n\nen dat moet dit zijn:\n\n```\n{want}\n```\n\nWerkt hij voor elk bestand en elke twee kolommen?"},
        "correct": {"en": "Your function is right in all {n} cases. For the penguins, `count` is 151 for the Adelies, though there are 152: the mean is over the 151 with a body mass.",
                    "nl": "Je functie klopt in alle {n} gevallen. Bij de pinguïns is `count` 151 voor de Adelies, terwijl het er 152 zijn: het gemiddelde gaat over de 151 met een lichaamsgewicht."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_S = "import pandas as pd\n\n" + _READ + "\n"
_F = "import pandas as pd\n\n\ndef summary(path, by, column):\n    df = pd.read_csv(path)\n"
CASES = [
    ("per-sky", _SKY_START, "empty", None),
    ("per-sky", _SKY, "correct", None),
    ("per-sky", _S + 'per_sky = df.groupby(df["sky"])["temp"].mean()\nper_sky', "correct", None),
    ("per-sky", _S + 'per_sky = df.groupby("sky").mean(numeric_only=True)["temp"]\nper_sky', "correct", None),
    ("per-sky", _S + 'groups = df.groupby("sky")\nper_sky = groups["temp"].mean()\nper_sky', "correct", None),
    ("per-sky", _S + 'per_sky = df["temp"].mean()\nper_sky', "wrong", "one-number"),
    ("per-sky", _S + 'per_sky = df[df["sky"] == "sunny"]["temp"].mean()\nper_sky', "wrong", "one-number"),
    ("per-sky", _S + 'per_sky = df.groupby("sky")["temp"].count()\nper_sky', "wrong", "counts"),
    ("per-sky", _S + 'per_sky = df.groupby("sky").size()\nper_sky', "wrong", "sizes"),
    ("per-sky", _S + 'per_sky = df.groupby("sky")["temp"].median()\nper_sky', "wrong", "median"),
    ("per-sky", _S + 'per_sky = df.groupby("sky")[["temp"]].mean()\nper_sky', "wrong", "double-brackets"),
    ("per-sky", _S + 'per_sky = df.groupby("sky").agg({"temp": "mean"})\nper_sky', "wrong", None),
    ("per-sky", _S + 'per_sky = df.groupby("sky")["temp"]\nper_sky', "wrong", "groups-only"),
    ("per-sky", _S + 'per_sky = df.groupby("sky")\nper_sky', "wrong", "groups-only"),
    ("per-sky", _S + 'per_sky = df.groupby("sky")["temp"].mean\nper_sky', "wrong", "itself"),
    ("per-sky", _S + 'per_sky = df.groupby("temp")["sky"].mean()\nper_sky', "error", "text-mean"),
    ("per-sky", _S + 'per_sky = df.groupby("sky").mean()\nper_sky', "error", "text-mean"),
    ("per-sky", _S + 'per_sky = df.groupby("Sky")["temp"].mean()\nper_sky', "error", "no-column"),
    ("per-sky", _S + 'per_sky = df.groupby("sky")["Temp"].mean()\nper_sky', "error", "no-column"),
    ("per-sky", _S + 'per_sky = df.groupby("sky")["temp"].mean()\nprint(per_sky)', "wrong", "print-last"),
    ("per-sky", _S + 'per_sky = df.groupby("sky")["rain"].mean()\nper_sky', "wrong", None),
    ("per-sky", _S + 'per_sky = pd.Series([7.5, 10.785714, 9.9375, 9.0], index=["Sunny", "cloudy", "rainy", "sunny"])\nper_sky', "almost", "other-value"),
    ("per-sky", _S + 'per_sky = df.groupby("sky")["temp"].mean(...)\nper_sky', "wrong", "dots-left"),
    ("per-sky", _S + 'per_sky = df.groupby("sky")["temp"].mean()\nper_sky, df.groupby("sky")["temp"].count()', "wrong", "two-things"),
    ("per-sky", _S + 'per_sky = df.groupby("sky")["temp"].mean().tolist()\nper_sky', "wrong", "no-labels"),
    ("per-sky", _S + 'per_sky = df.groupby("sky")["temp"].mean().values\nper_sky', "wrong", "no-labels"),
    ("per-sky", _S + 'per_sky = df.groupby("sky")["temp"].mean().reset_index(drop=True)\nper_sky', "wrong", None),
    ("per-sky", _S + 'per_sky = df.groupby("sky")["temp"].mean().round(2)\nper_sky', "wrong", "rounded"),
    ("per-sky", _S + 'per_sky = df.groupby("sky")["temp"].agg("mean")\nper_sky', "correct", None),
    ("summary", _SUMMARY_START, "empty", None),
    ("summary", _SUMMARY, "correct", None),
    ("summary", _F + '    return df.groupby(by)[column].agg(["count", "mean"])', "correct", None),
    ("summary", _F + "    groups = df.groupby(by)[column]\n    return pd.DataFrame({\"mean\": groups.mean(), \"count\": groups.count()})", "correct", None),
    ("summary", _F + '    return df.groupby(by)[column].agg(["mean", "size"])', "wrong", "size"),
    ("summary", _F + "    return df.groupby(by)[column].mean()", "wrong", "mean-only"),
    ("summary", _F + "    return df.groupby(by)[column].count()", "wrong", "count-only"),
    ("summary", _F + '    return df[column].agg(["mean", "count"])', "wrong", "no-groups"),
    ("summary", _F + "    return df.groupby(by)[column]", "wrong", "groups-only"),
    ("summary", _F + '    return df.groupby(by).agg(["mean", "count"])', "error", "text-mean"),
    ("summary", _F + '    return df.groupby(column)[by].agg(["mean", "count"])', "error", "text-mean"),
    ("summary", _F + '    return df.groupby("by")[column].agg(["mean", "count"])', "error", "name-in-quotes"),
    ("summary", _F + '    return df.groupby(by)["column"].agg(["mean", "count"])', "error", "name-in-quotes"),
    ("summary", _F + '    return df.groupby(by)[column].agg("mean", "count")', "error", "agg-list"),
    ("summary", _F + '    return df.groupby("species")["body_mass_g"].agg(["mean", "count"])', "almost", "fixed-names"),
    ("summary", _F + '    return df.groupby(by)[column].agg(["mean", "count", "std"])', "wrong", None),
    ("summary", _F + "    return df.groupby(by)[column].mean(), df.groupby(by)[column].count()", "wrong", "two-things"),
    ("summary", _F + '    return df.groupby(by)[column].agg(["mean", "count"]).round(2)', "wrong", "rounded"),
    ("summary", _F + '    return df.groupby(by)[[column]].agg(["mean", "count"])', "wrong", "double-brackets"),
    ("summary", _F + '    return df.groupby(by)[column].agg(["mean", "count"]).reset_index(drop=True)', "wrong", None),
    ("summary", _F + '    return df.groupby(by)[column].agg(mean="mean", count="count")', "correct", None),
]
