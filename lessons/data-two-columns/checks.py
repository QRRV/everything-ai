# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson data-two-columns: a scatter plot of the bill length and depth of the Chinstraps among the
Palmer penguins, and the mean body mass and bill depth of each species. And the pictures: four scatter plots of the
course's own tables, up, down, none and a curve, and the play block that steps through the species, where the arrow
of all penguins points down and the arrow of each species up (Simpson's paradox)."""
import re
from functools import cache

import numpy as np
import pandas as pd   # here, so the first import of pandas never counts against the time limit of a check

from everything_ai.check import Mistake, CodeBox, PlotBox, same_value

PENGUINS = "data/penguins.csv"
STATION = "lessons/data-explore/station.csv"
STATION_2 = "lessons/data-explore/station-2.csv"
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]   # sorted by name, as groupby sorts them; group colour 1, 2 and 3
STEPS = [0, 1, 2, 3, 4]   # the stops of the play block: all penguins, each species on its own, all three together
X, Y = "bill_length_mm", "bill_depth_mm"


@cache
def penguins() -> pd.DataFrame:
    """The 342 penguins with both bill measurements (two of the 344 have none)."""
    df = pd.read_csv(PENGUINS)
    return df[df[X].isna() == False].reset_index(drop=True)


def group(species: str | None) -> pd.DataFrame:
    """The penguins of one species, or all of them for None."""
    df = penguins()
    return df if species is None else df[df["species"] == species]


def halves(df: pd.DataFrame, x: str = X, y: str = Y) -> tuple:
    """The arrow of a group: split its rows at the median of x; the arrow runs from the middle of the lower half,
    (mean x, mean y), to the middle of the upper half. It points the way the cloud goes: up when the rows with larger x
    have larger y on average, down when smaller. Also gives how much larger, the mean y of the upper half minus that
    of the lower half."""
    m = df[x].median()
    low, high = df[df[x] <= m], df[df[x] > m]
    return (low[x].mean(), low[y].mean()), (high[x].mean(), high[y].mean()), high[y].mean() - low[y].mean()


def step_numbers(step: int) -> dict:
    """What the sentence under the play block says at a stop: which penguins, how many, and by how much deeper (or,
    negative, shallower) the bills of the longer half are on average, rounded to two decimals."""
    species = None if step in (0, 4) else SPECIES[step - 1]
    df = group(species)
    out = {"species": species, "n": len(df), "gap": round(halves(df)[2], 2)}
    if step == 4:
        out["gaps"] = {s: round(halves(group(s))[2], 2) for s in SPECIES}
    return out


def _arrow(p, df, role: str, width: float):
    (x0, y0), (x1, y1), _ = halves(df)
    p.arrow(x0, y0, x1, y1, role, width)


def species_picture(step: int, labels: dict) -> str:
    """The play block at one stop. 0: all penguins as grey dots, with the arrow of all of them. 1 to 3: one species in
    its group colour and shape (style guide 1.7: three groups), the others small and light, with the arrow of that
    species. 4: all three species in their colours, each with its arrow, and the arrow of all penguins in grey behind
    them. The arrows are ink: nothing here is learned or predicted. Drawn once for every stop (L.explore); the lesson
    writes the alt text."""
    from everything_ai import chart
    df = penguins()
    p = chart.Panel((31, 61), (12.5, 22), title=labels["title"], xlabel=labels["x"], ylabel=labels["y"],
                    xticks=[35, 40, 45, 50, 55, 60], yticks=[14, 16, 18, 20, 22])
    if step == 0:
        p.points(df[X], df[Y], "data", r=3)
        _arrow(p, df, "ink", 3)
    elif step < 4:
        k = step - 1
        rest, own = df[df["species"] != SPECIES[k]], df[df["species"] == SPECIES[k]]
        p.points(rest[X], rest[Y], "ink-3", r=1.8)
        p.group(own[X], own[Y], k, r=3.4)
        _arrow(p, own, "ink", 3)
    else:
        _arrow(p, df, "ink-2", 2.2)
        for k, s in enumerate(SPECIES):
            own = df[df["species"] == s]
            p.group(own[X], own[Y], k, r=3)
        for s in SPECIES:
            _arrow(p, df[df["species"] == s], "ink", 3)
    return chart.figure([p], alt=labels["alt"], width=460, height=300)


def shapes_picture(labels: dict) -> str:
    """Four scatter plots of the course's own tables, one for each way two columns can go together: up (the flipper
    length and body mass of the penguins), down (the days of the station's month and its temperature, as the month
    cools), none (the temperature and the wind of that month, with -999 read as missing), and a curve (the days of the
    other month and its temperature, which rises and falls again). Grey dots only: it is all data."""
    from everything_ai import chart
    pen = pd.read_csv(PENGUINS)
    one, two = pd.read_csv(STATION, na_values=[-999]), pd.read_csv(STATION_2, na_values=[-999])

    def both(df, x, y):   # a day or a penguin without one of the two values is no dot
        rows = df[[x, y]].dropna()
        return rows[x], rows[y]
    panels = [
        chart.Panel((168, 235), (2500, 6500), title=labels["up"], xlabel=labels["flipper"], ylabel=labels["mass"],
                    xticks=[180, 200, 220], yticks=[3000, 4000, 5000, 6000]).points(*both(pen, "flipper_length_mm", "body_mass_g"), "data", r=2.2),
        chart.Panel((0, 31), (4, 13.5), title=labels["down"], xlabel=labels["day"], ylabel=labels["temp"],
                    xticks=[1, 10, 20, 30], yticks=[5, 7, 9, 11, 13]).points(*both(one, "day", "temp"), "data", r=3),
        chart.Panel((4, 13.5), (0, 36), title=labels["none"], xlabel=labels["temp"], ylabel=labels["wind"],
                    xticks=[5, 7, 9, 11, 13], yticks=[0, 10, 20, 30]).points(*both(one, "temp", "wind"), "data", r=3),
        chart.Panel((0, 31), (10, 17.5), title=labels["curve"], xlabel=labels["day"], ylabel=labels["temp"],
                    xticks=[1, 10, 20, 30], yticks=[10, 12, 14, 16]).points(*both(two, "day", "temp"), "data", r=3),
    ]
    # two figures of two: two rows of two at full width; in a window of 700 px the four lie one below the other
    return (chart.figure(panels[:2], alt=labels["alt_1"], width=300, height=230)
            + chart.figure(panels[2:], alt=labels["alt_2"], width=300, height=230))


# The lesson is done when these are correct. "rescues" is the choice question in the lesson.
REQUIRED = ["chinstraps", "rescues", "per-species"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_COLUMNS = ["species", "island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex", "year"]
_NOT_A_TABLE = Mistake("mask-only", error=lambda e: isinstance(e, KeyError) and bool(e.args) and e.args[0] in _COLUMNS,
                       en="The column exists, but you take it from something that is not the table: a mask such as `df[\"species\"] == \"Chinstrap\"` is on its own a column of `True` and `False`, one per penguin. Put the mask between the square brackets of `df` to keep the Chinstraps' rows, `df[df[\"species\"] == \"Chinstrap\"]`, and take both columns from that.",
                       nl="De kolom bestaat wel, maar je haalt hem uit iets dat geen tabel is: een masker zoals `df[\"species\"] == \"Chinstrap\"` is op zich een kolom met `True` en `False`, één per pinguïn. Zet het masker tussen de vierkante haken van `df` om de rijen van de Chinstraps te houden, `df[df[\"species\"] == \"Chinstrap\"]`, en neem de twee kolommen daaruit.")
_NO_COLUMN = Mistake("no-column", error=lambda e: isinstance(e, KeyError),
                     en="The table has no column with that name. Its columns are `species`, `island`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `sex` and `year`, in small letters, each between quotes.",
                     nl="De tabel heeft geen kolom met die naam. De kolommen zijn `species`, `island`, `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`, `sex` en `year`, in kleine letters, elk tussen aanhalingstekens.")


def _kind(got) -> str:
    return type(got).__name__


# --- exercise 1: the Chinstraps' scatter plot ---------------------------------------------------------------------------

def _counted(values) -> int:
    """How many points a chart has, without the ones with a missing value, which matplotlib does not draw."""
    return sum(1 for v in values if v == v)


def _points(got) -> tuple | None:
    """The x and y values of the first scatter plot the learner drew, or None."""
    charts = [c for c in (got or {}).get("charts", []) if c["kind"] == "scatter"]
    return (charts[0]["x"], charts[0]["y"]) if charts else None


def _drew(column_x: str, column_y: str, species: str | None):
    """A test on the learner's chart: it shows these two columns of these penguins."""
    def test(got, text):
        pts = _points(got)
        if pts is None:
            return False
        df = pd.read_csv(PENGUINS)
        df = df if species is None else df[df["species"] == species]
        return _same_points(pts[0], df[column_x].tolist()) and _same_points(pts[1], df[column_y].tolist())
    return test


def _same_points(a, b) -> bool:
    """The same values, without the missing ones: a penguin without a measurement is no point, whether the learner
    left it out first or matplotlib did."""
    return same_value([v for v in a if v == v], [v for v in b if v == v])


class _QuietPlotBox(PlotBox):
    """A chart check that reads back a chart of all penguins without a word from matplotlib about the two penguins
    without a bill measurement: it draws no point for them, and reading the chart back would otherwise print a
    warning under the check."""

    @staticmethod
    def describe_run(code: str, picture: bool = False):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return PlotBox.describe_run(code, picture)


_HEAD = "import matplotlib.pyplot as plt\nimport pandas as pd\n\n" + f'df = pd.read_csv("{PENGUINS}")\n'
_CHIN_START = _HEAD + "...\nplt.show()"
_CHIN = _HEAD + ('chinstrap = df[df["species"] == "Chinstrap"]\nplt.scatter(chinstrap["bill_length_mm"], chinstrap["bill_depth_mm"])\n'
                 'plt.title("Chinstraps only")\nplt.xlabel("bill length (mm)")\nplt.ylabel("bill depth (mm)")\nplt.show()')

EXERCISES = {
    "chinstraps": _QuietPlotBox(
        name="chinstraps",
        start=_CHIN_START,
        solution=_CHIN,
        compare=_same_points,
        first=[_DOTS],
        mistakes=[
            _NOT_A_TABLE,
            _NO_COLUMN,
            Mistake("no-plt", syntax=r"(?s)\A(?!.*import\s+matplotlib\.pyplot\s+as\s+plt).*\bplt\s*\.",
                    en="Python does not know `plt` yet: the box needs the line `import matplotlib.pyplot as plt` at the top. Put it back.",
                    nl="Python kent `plt` nog niet: het vak heeft de regel `import matplotlib.pyplot as plt` bovenaan nodig. Zet hem terug."),
            Mistake("lengths", error=lambda e: isinstance(e, ValueError) and re.search(r"same size|same first dimension|shape mismatch", str(e)) is not None,
                    en="Python stops while drawing: the two columns do not have the same length. Take both from the same rows: first keep the Chinstraps, then take their two columns.",
                    nl="Python stopt tijdens het tekenen: de twee kolommen zijn niet even lang. Neem ze allebei uit dezelfde rijen: houd eerst de Chinstraps, en neem dan hun twee kolommen."),
            Mistake("small-letter", matches=lambda got, text: _points(got) is None and re.search(r"==\s*[\"']chinstrap[\"']", text) is not None,
                    en="Your chart has no points: no penguin has the species `chinstrap` with a small c. The names start with a capital: `\"Chinstrap\"`.",
                    nl="Je grafiek heeft geen punten: geen pinguïn heeft de soort `chinstrap` met een kleine c. De namen beginnen met een hoofdletter: `\"Chinstrap\"`."),
            Mistake("all-penguins", matches=_drew(X, Y, None),
                    en="Your chart shows all 342 penguins with a bill measurement. The task asks for the Chinstraps only: keep their rows first, with the mask `df[\"species\"] == \"Chinstrap\"`.",
                    nl="Je grafiek laat alle 342 pinguïns met snavelmetingen zien. De opdracht vraagt alleen om de Chinstraps: houd eerst hun rijen, met het masker `df[\"species\"] == \"Chinstrap\"`."),
            Mistake("adelies", matches=_drew(X, Y, "Adelie"),
                    en="Your chart shows the Adelies, as the worked example did. The task asks for the Chinstraps: `df[\"species\"] == \"Chinstrap\"`.",
                    nl="Je grafiek laat de Adelies zien, zoals het uitgewerkte voorbeeld. De opdracht vraagt om de Chinstraps: `df[\"species\"] == \"Chinstrap\"`."),
            Mistake("gentoos", matches=_drew(X, Y, "Gentoo"),
                    en="Your chart shows the Gentoos. The task asks for the Chinstraps: `df[\"species\"] == \"Chinstrap\"`.",
                    nl="Je grafiek laat de Gentoos zien. De opdracht vraagt om de Chinstraps: `df[\"species\"] == \"Chinstrap\"`."),
            Mistake("swapped", matches=_drew(Y, X, "Chinstrap"),
                    en="Your chart has the bill depth along the bottom and the length up the side. The first column you give `plt.scatter` goes along the bottom: give `bill_length_mm` first.",
                    nl="Je grafiek heeft de snavelhoogte langs de onderkant en de lengte langs de zijkant. De eerste kolom die je aan `plt.scatter` geeft, gaat langs de onderkant: geef `bill_length_mm` eerst."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with the lines that keep the Chinstraps and draw their bill depth against their bill length, with a title and a label on each axis.",
                      "nl": "Vervang de `...` door de regels die de Chinstraps houden en hun snavelhoogte tegen hun snavellengte tekenen, met een titel en een label bij elke as."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "nothing": {"en": "Your code draws nothing yet. Draw the scatter plot with `plt.scatter`, before `plt.show()`.",
                        "nl": "Je code tekent nog niets. Teken de scatter plot met `plt.scatter`, vóór `plt.show()`."},
            "kind": {"en": "Your code draws {kind}, but this exercise asks for {want_kind}: loose points, one per penguin. Draw it with `{draw}`.",
                     "nl": "Je code tekent {kind}, maar deze oefening vraagt om {want_kind}: losse punten, één per pinguïn. Teken die met `{draw}`."},
            "values": {"en": "Your chart has other points than the bill length and bill depth of the 68 Chinstraps. Do you keep the rows whose `species` is `\"Chinstrap\"`, and draw their `bill_length_mm` along the bottom and their `bill_depth_mm` up the side?",
                       "nl": "Je grafiek heeft andere punten dan de snavellengte en snavelhoogte van de 68 Chinstraps. Houd je de rijen waarvan `species` gelijk is aan `\"Chinstrap\"`, en teken je hun `bill_length_mm` langs de onderkant en hun `bill_depth_mm` langs de zijkant?"},
            "title": {"en": "Right chart, but it has no title yet. Add `plt.title(...)` with a text that says what the chart shows.",
                      "nl": "Goede grafiek, maar hij heeft nog geen titel. Voeg `plt.title(...)` toe, met een tekst die zegt wat de grafiek laat zien."},
            "xlabel": {"en": "Right chart, but the x-axis has no label yet. Add `plt.xlabel(...)` with what is along the bottom.",
                       "nl": "Goede grafiek, maar de x-as heeft nog geen label. Voeg `plt.xlabel(...)` toe, met wat er langs de onderkant staat."},
            "ylabel": {"en": "Right chart, but the y-axis has no label yet. Add `plt.ylabel(...)` with what goes up the side.",
                       "nl": "Goede grafiek, maar de y-as heeft nog geen label. Voeg `plt.ylabel(...)` toe, met wat er langs de zijkant omhooggaat."},
            "correct": {"en": "The 68 Chinstraps, and their cloud goes up to the right, as the Adelies' did: within one species, a longer bill goes with a deeper one.",
                        "nl": "De 68 Chinstraps, en hun wolk gaat omhoog naar rechts, net als die van de Adelies: binnen één soort hoort een langere snavel bij een hogere."},
        },
    ),
}


# --- exercise 3: where each species lies ---------------------------------------------------------------------------------

_PAIR = ["body_mass_g", "bill_depth_mm"]


def _groups_as_labels(t: pd.DataFrame) -> pd.DataFrame:
    """A table with the species in a column of their own, as `reset_index()` or `groupby(..., as_index=False)` give it,
    holds the same means: with the species as row labels again, it is the table groupby gives."""
    return t.set_index("species") if "species" in map(str, t.columns) else t


def _same_means(a, b) -> bool:
    """The same table: the same groups as row labels (or in a column `species` of their own), the same columns in any
    order, and the same values. In a summary per group the row labels are the groups, so here they count. Anything else
    is never the same."""
    if _kind(a) != "DataFrame" or _kind(b) != "DataFrame":
        return False
    a, b = _groups_as_labels(a), _groups_as_labels(b)
    if [str(i) for i in a.index] != [str(i) for i in b.index] or set(map(str, a.columns)) != set(map(str, b.columns)):
        return False
    return all(same_value(a[c].tolist(), b[c].tolist()) for c in b.columns)


def _labels(got) -> list[str]:
    return [str(i) for i in got.index] if _kind(got) in ("Series", "DataFrame") else []


_READ = f'df = pd.read_csv("{PENGUINS}")'
_MEANS_START = "import pandas as pd\n\n" + _READ + "\nper_species = ...\nper_species\n"
_MEANS = "import pandas as pd\n\n" + _READ + '\nper_species = df.groupby("species")[["body_mass_g", "bill_depth_mm"]].mean()\nper_species'


def _means(new: str) -> str:
    return _MEANS.replace('per_species = df.groupby("species")[["body_mass_g", "bill_depth_mm"]].mean()', new)


EXERCISES["per-species"] = CodeBox(
    name="per-species",
    start=_MEANS_START,
    solution=_MEANS,
    keep=["import pandas as pd", _READ],
    compare=_same_means,
    first=[_DOTS],
    digits=2,
    mistakes=[
        _NO_COLUMN,
        Mistake("one-pair", error=lambda e: isinstance(e, ValueError) and "Use a list instead" in str(e),
                en="Two column names between one pair of square brackets make pandas look for one column with that whole name. For two columns, put their names in a list: two pairs of brackets, `[[\"body_mass_g\", \"bill_depth_mm\"]]`.",
                nl="Twee kolomnamen tussen één paar vierkante haken laten pandas zoeken naar één kolom met die hele naam. Voor twee kolommen zet je hun namen in een lijst: twee paar haken, `[[\"body_mass_g\", \"bill_depth_mm\"]]`."),
        Mistake("text-mean", error=lambda e: isinstance(e, TypeError) and "does not support operation 'mean'" in str(e),
                en="Your code takes the mean of every column, also of the text columns `island` and `sex`, and a text has no mean. Take the two columns first: `df.groupby(\"species\")[[\"body_mass_g\", \"bill_depth_mm\"]]`.",
                nl="Je code neemt het gemiddelde van elke kolom, ook van de tekstkolommen `island` en `sex`, en een tekst heeft geen gemiddelde. Neem eerst de twee kolommen: `df.groupby(\"species\")[[\"body_mass_g\", \"bill_depth_mm\"]]`."),
        Mistake("groups-only", matches=lambda got, text: _kind(got) in ("SeriesGroupBy", "DataFrameGroupBy"),
                en="That is the groups themselves, not yet a number per group: pandas only shows what kind of thing it is. Ask for the mean: `.mean()` at the end.",
                nl="Dat zijn de groepen zelf, nog geen getal per groep: pandas laat alleen zien wat voor ding het is. Vraag om het gemiddelde: `.mean()` aan het eind."),
        Mistake("itself", matches=lambda got, text: callable(got),
                en="Your code gives back something that still needs its parentheses: `.mean` instead of `.mean()`. Without them, you get `mean` itself, not the means it gives.",
                nl="Je code geeft iets terug dat nog zijn haakjes nodig heeft: `.mean` in plaats van `.mean()`. Zonder haakjes krijg je `mean` zelf, niet de gemiddelden die het geeft."),
        Mistake("no-groups", matches=lambda got, text: _kind(got) == "Series" and set(_labels(got)) == set(_PAIR),
                en="Your last line gives one mean for each column, over all penguins together. The task asks for the means of each species: split the rows first, with `df.groupby(\"species\")`.",
                nl="Je laatste regel geeft één gemiddelde per kolom, over alle pinguïns samen. De opdracht vraagt om de gemiddelden van elke soort: splits de rijen eerst, met `df.groupby(\"species\")`."),
        Mistake("one-column", matches=lambda got, text: _kind(got) == "Series" and _labels(got) == SPECIES,
                en="Your last line gives one number per species, for one column. The task asks for two columns, side by side: the body mass and the bill depth, in a list: `[[\"body_mass_g\", \"bill_depth_mm\"]]`.",
                nl="Je laatste regel geeft één getal per soort, voor één kolom. De opdracht vraagt om twee kolommen naast elkaar: het lichaamsgewicht en de snavelhoogte, in een lijst: `[[\"body_mass_g\", \"bill_depth_mm\"]]`."),
        Mistake("other-groups", matches=lambda got, text: _kind(got) == "DataFrame" and _labels(got) != SPECIES,
                en="Your table has other groups than the three species, such as the islands. The third column here is the species: `df.groupby(\"species\")`.",
                nl="Je tabel heeft andere groepen dan de drie soorten, zoals de eilanden. De derde kolom is hier de soort: `df.groupby(\"species\")`."),
        Mistake("more-columns", matches=lambda got, text: _kind(got) == "DataFrame" and set(_PAIR) < set(map(str, got.columns)),
                en="Your table has the means of more columns than the two the task asks for. Take only the body mass and the bill depth: `[[\"body_mass_g\", \"bill_depth_mm\"]]` before `.mean()`.",
                nl="Je tabel heeft de gemiddelden van meer kolommen dan de twee die de opdracht vraagt. Neem alleen het lichaamsgewicht en de snavelhoogte: `[[\"body_mass_g\", \"bill_depth_mm\"]]` vóór `.mean()`."),
        Mistake("median", wrong=_means('per_species = df.groupby("species")[["body_mass_g", "bill_depth_mm"]].median()'),
                en="Your last line gives the median of each species. The task asks for the mean: `.mean()`.",
                nl="Je laatste regel geeft de mediaan van elke soort. De opdracht vraagt om het gemiddelde: `.mean()`."),
        Mistake("counts", wrong=_means('per_species = df.groupby("species")[["body_mass_g", "bill_depth_mm"]].count()'),
                en="Your last line gives how many values each species has, not where it lies. The task asks for the mean: `.mean()`.",
                nl="Je laatste regel geeft hoeveel waarden elke soort heeft, niet waar hij ligt. De opdracht vraagt om het gemiddelde: `.mean()`."),
        Mistake("dropped-rows", wrong=_means('per_species = df.dropna().groupby("species")[["body_mass_g", "bill_depth_mm"]].mean()'),
                en="Your means leave out 9 penguins whose body mass and bill depth are known: `dropna()` drops every row with a gap in any column, also the rows where only `sex` is missing. `.mean()` leaves the gaps out on its own: take out the `dropna()`.",
                nl="Je gemiddelden laten 9 pinguïns weg waarvan het lichaamsgewicht en de snavelhoogte bekend zijn: `dropna()` laat elke rij met een gat in welke kolom dan ook weg, ook de rijen waarin alleen `sex` ontbreekt. `.mean()` laat de gaten zelf al weg: haal de `dropna()` weg."),
        Mistake("rounded", matches=lambda got, text: re.search(r"\bround\(", text) is not None,
                en="Those may be the right means, but rounded. The check compares them with the means pandas gives, to many digits: leave out the rounding.",
                nl="Dat kunnen de goede gemiddelden zijn, maar dan afgerond. De check vergelijkt ze met de gemiddelden die pandas geeft, tot op veel cijfers: laat het afronden weg."),
    ],
    texts={
        "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                    "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
        "empty": {"en": "Replace the `...` with the mean body mass and the mean bill depth of each species.",
                  "nl": "Vervang de `...` door het gemiddelde lichaamsgewicht en de gemiddelde snavelhoogte van elke soort."},
        "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                    "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
        "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back.",
                 "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug."},
        "no_value": {"en": "The last line of the box shows nothing. Put `per_species` on the last line, so the check can see it.",
                     "nl": "De laatste regel van het vak laat niets zien. Zet `per_species` op de laatste regel, zodat de check het kan zien."},
        "wrong": {"en": "Your last line gives:\n\n```\n{got}\n```\n\nand it should be a table with a row for each of the three species and two columns, the mean `body_mass_g` and the mean `bill_depth_mm`. Do you group by `species`, take the two columns in a list, and ask for `.mean()`?",
                  "nl": "Je laatste regel geeft:\n\n```\n{got}\n```\n\nen dat moet een tabel zijn met een rij voor elk van de drie soorten en twee kolommen, het gemiddelde van `body_mass_g` en het gemiddelde van `bill_depth_mm`. Groepeer je op `species`, neem je de twee kolommen in een lijst, en vraag je om `.mean()`?"},
        "correct": {"en": "The Gentoos weigh about 1350 grams more than the other two species, 5076 against 3701 and 3733, and their bills are about 3.4 millimeters shallower, 14.98 against 18.35 and 18.42. So over all penguins, heavier goes with shallower bills, while within each species the heavier penguins have the deeper bills: Simpson's paradox again, with the species as the confounder.",
                    "nl": "De Gentoos wegen zo'n 1350 gram meer dan de andere twee soorten, 5076 tegen 3701 en 3733, en hun snavels zijn zo'n 3,4 millimeter lager, 14,98 tegen 18,35 en 18,42. Over alle pinguïns samen hoort zwaarder dus bij lagere snavels, terwijl binnen elke soort de zwaardere pinguïns de hogere snavels hebben: weer de paradox van Simpson, met de soort als confounder."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_C = _HEAD + 'chinstrap = df[df["species"] == "Chinstrap"]\n'
_LABELS = 'plt.title("Chinstraps only")\nplt.xlabel("bill length (mm)")\nplt.ylabel("bill depth (mm)")\nplt.show()'
_P = "import pandas as pd\n\n" + _READ + "\n"
CASES = [
    ("chinstraps", _CHIN_START, "empty", None),
    ("chinstraps", _CHIN, "correct", None),
    ("chinstraps", _HEAD + 'c = df[df["species"] == "Chinstrap"].dropna()\nplt.scatter(c["bill_length_mm"], c["bill_depth_mm"])\n' + _LABELS, "correct", None),
    ("chinstraps", _C + 'plt.plot(chinstrap["bill_length_mm"], chinstrap["bill_depth_mm"], "o")\n' + _LABELS, "correct", None),
    ("chinstraps", _C + 'plt.scatter(chinstrap["bill_length_mm"], chinstrap["bill_depth_mm"])\nplt.title("Bills of the Chinstraps")\nplt.xlabel("length")\nplt.ylabel("depth")\nplt.show()', "correct", None),
    ("chinstraps", _HEAD + 'plt.scatter(df["bill_length_mm"], df["bill_depth_mm"])\n' + _LABELS, "wrong", "all-penguins"),
    ("chinstraps", _HEAD + 'a = df[df["species"] == "Adelie"]\nplt.scatter(a["bill_length_mm"], a["bill_depth_mm"])\n' + _LABELS, "wrong", "adelies"),
    ("chinstraps", _HEAD + 'g = df[df["species"] == "Gentoo"]\nplt.scatter(g["bill_length_mm"], g["bill_depth_mm"])\n' + _LABELS, "wrong", "gentoos"),
    ("chinstraps", _C + 'plt.scatter(chinstrap["bill_depth_mm"], chinstrap["bill_length_mm"])\n' + _LABELS, "wrong", "swapped"),
    ("chinstraps", _HEAD + 'c = df[df["species"] == "chinstrap"]\nplt.scatter(c["bill_length_mm"], c["bill_depth_mm"])\n' + _LABELS, "wrong", "small-letter"),
    ("chinstraps", _C + 'plt.plot(chinstrap["bill_length_mm"], chinstrap["bill_depth_mm"])\n' + _LABELS, "wrong", "kind"),
    ("chinstraps", _C + 'plt.scatter(chinstrap["bill_length_mm"], chinstrap["flipper_length_mm"])\n' + _LABELS, "wrong", "values"),
    ("chinstraps", _C + 'plt.scatter(chinstrap["bill_length_mm"], df["bill_depth_mm"])\n' + _LABELS, "error", "lengths"),
    ("chinstraps", _C + 'plt.scatter(chinstrap["bill_length"], chinstrap["bill_depth_mm"])\n' + _LABELS, "error", "no-column"),
    ("chinstraps", _HEAD + 'chinstrap = df["species"] == "Chinstrap"\nplt.scatter(chinstrap["bill_length_mm"], chinstrap["bill_depth_mm"])\n' + _LABELS, "error", "mask-only"),
    ("chinstraps", _C + 'plt.scatter(chinstrap["bill_length_mm"], chinstrap["bill_depth_mm"])\nplt.xlabel("bill length (mm)")\nplt.ylabel("bill depth (mm)")\nplt.show()', "almost", "no-title"),
    ("chinstraps", _C + 'plt.scatter(chinstrap["bill_length_mm"], chinstrap["bill_depth_mm"])\nplt.title("Chinstraps only")\nplt.ylabel("bill depth (mm)")\nplt.show()', "almost", "no-xlabel"),
    ("chinstraps", _C + 'plt.scatter(chinstrap["bill_length_mm"], chinstrap["bill_depth_mm"])\nplt.title("Chinstraps only")\nplt.xlabel("bill length (mm)")\nplt.show()', "almost", "no-ylabel"),
    ("chinstraps", _C + "plt.show()", "wrong", "nothing"),
    ("chinstraps", _CHIN.replace("import matplotlib.pyplot as plt\n", ""), "error", "no-plt"),
    ("chinstraps", _CHIN.replace('plt.title("Chinstraps only")', 'plt.title("...")\n...'), "wrong", "dots-left"),
    ("per-species", _MEANS_START, "empty", None),
    ("per-species", _MEANS, "correct", None),
    ("per-species", _P + 'per_species = df.groupby("species")[["bill_depth_mm", "body_mass_g"]].mean()\nper_species', "correct", None),
    ("per-species", _P + 'per_species = df.groupby("species").agg({"body_mass_g": "mean", "bill_depth_mm": "mean"})\nper_species', "correct", None),
    ("per-species", _P + 'per_species = df.groupby("species")[["body_mass_g", "bill_depth_mm"]].mean().reset_index()\nper_species', "correct", None),
    ("per-species", _P + 'per_species = df.groupby("species", as_index=False)[["body_mass_g", "bill_depth_mm"]].mean()\nper_species', "correct", None),
    ("per-species", _P + 'per_species = df.groupby("island")[["body_mass_g", "bill_depth_mm"]].mean().reset_index()\nper_species', "wrong", "other-groups"),
    ("per-species", _P + 'per_species = df.groupby("species").mean(numeric_only=True)[["body_mass_g", "bill_depth_mm"]]\nper_species', "correct", None),
    ("per-species", _P + 'per_species = df.groupby("species")["body_mass_g", "bill_depth_mm"].mean()\nper_species', "error", "one-pair"),
    ("per-species", _P + 'per_species = df.groupby("species").mean()\nper_species', "error", "text-mean"),
    ("per-species", _P + 'per_species = df.groupby("species").mean(numeric_only=True)\nper_species', "wrong", "more-columns"),
    ("per-species", _P + 'per_species = df.groupby("species")["body_mass_g"].mean()\nper_species', "wrong", "one-column"),
    ("per-species", _P + 'per_species = df[["body_mass_g", "bill_depth_mm"]].mean()\nper_species', "wrong", "no-groups"),
    ("per-species", _P + 'per_species = df.groupby("island")[["body_mass_g", "bill_depth_mm"]].mean()\nper_species', "wrong", "other-groups"),
    ("per-species", _P + 'per_species = df.groupby("species")[["body_mass_g", "bill_depth_mm"]].median()\nper_species', "wrong", "median"),
    ("per-species", _P + 'per_species = df.groupby("species")[["body_mass_g", "bill_depth_mm"]].count()\nper_species', "wrong", "counts"),
    ("per-species", _P + 'per_species = df.groupby("species")[["body_mass_g", "bill_depth_mm"]].mean().round(1)\nper_species', "wrong", "rounded"),
    ("per-species", _P + 'per_species = df.groupby("species")[["body_mass_g", "bill_depth_mm"]]\nper_species', "wrong", "groups-only"),
    ("per-species", _P + 'per_species = df.groupby("species")[["body_mass_g", "bill_depth_mm"]].mean\nper_species', "wrong", "itself"),
    ("per-species", _P + 'per_species = df.groupby("species")[["body_mass", "bill_depth_mm"]].mean()\nper_species', "error", "no-column"),
    ("per-species", _P + 'df = df.dropna()\nper_species = df.groupby("species")[["body_mass_g", "bill_depth_mm"]].mean()\nper_species', "wrong", "dropped-rows"),
    ("per-species", _P + 'df = df.dropna(subset=["body_mass_g", "bill_depth_mm"])\nper_species = df.groupby("species")[["body_mass_g", "bill_depth_mm"]].mean()\nper_species', "correct", None),
    ("per-species", _P + 'per_species = df.groupby("species")[["body_mass_g", "bill_length_mm"]].mean()\nper_species', "wrong", None),
    ("per-species", _P + 'per_species = df.groupby("species")[["body_mass_g", "bill_depth_mm"]].mean()\nprint(per_species)', "wrong", "print-last"),
]
