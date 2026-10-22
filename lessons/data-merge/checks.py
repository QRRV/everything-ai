# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson data-merge: a left join of the second week of weather of data-pandas with the ice creams a
made-up beach stand sold that week, and a function that counts how many rows a left join adds. And the block in *The
idea*: the week of weather next to the list of the stand where Saturday is twice, looked up day by day, with the inner
and the left join growing beside it."""
import html
import re
from functools import cache

from everything_ai.check import Code, CodeBox, FunctionBox, Mistake

WEATHER = "lessons/data-pandas/weather.csv"
WEATHER_2 = "lessons/data-pandas/weather-2.csv"
SALES = "lessons/data-merge/sales.csv"          # week 1: the stand was closed on Monday
SALES_TWICE = "lessons/data-merge/sales-twice.csv"   # the same week, with Saturday's line typed in twice
SALES_2 = "lessons/data-merge/sales-2.csv"      # week 2: closed on Monday and on Friday
PENGUINS = "data/penguins.csv"


def _read(path):
    import pandas as pd
    return pd.read_csv(path)


def rows(left: str, right: str, how: str = "inner") -> int:
    """How many rows a merge of the two files on `day` has, as the lesson's text says."""
    import pandas as pd
    return len(pd.merge(_read(left), _read(right), on="day", how=how))


# --- the block in *The idea*: a merge, day by day -----------------------------------------------------------------------

_BH, _GAP, _BOX = 23, 3, 44     # height of a row, the space between rows, the width of a box
_TOP = 46                       # where the first row begins, under the title and the column names
_MONO = "font-family:var(--ea-font-code)"


def _table(x: float, header: list[str], values: list[list[str]], inks: list[str], strong: set[int], box: float = _BOX) -> str:
    """A small table as a DataFrame shows it: the column names above, the row labels in front, the values in boxes.
    `inks[i]` is the ink of row i: "ink", or "ink-3" for a row that is not looked up yet. The rows in `strong` are
    the ones of this step: a dark edge and bold values."""
    out = []
    for j, name in enumerate(header):
        out.append(f'<text x="{x + 16 + j * (box + _GAP) + box / 2}" y="{_TOP - 6}" style="{_MONO};font-size:12px;font-weight:600;'
                   f'fill:var(--ea-ink-2);text-anchor:middle">{html.escape(name)}</text>')
    for i, row in enumerate(values):
        y = _TOP + i * (_BH + _GAP)
        edge = "stroke:var(--ea-ink);stroke-width:1.6" if i in strong else "stroke:var(--ea-line);stroke-width:1"
        weight = ";font-weight:700" if i in strong else ""
        out.append(f'<text x="{x}" y="{y + _BH / 2 + 4}" style="{_MONO};font-size:11px;fill:var(--ea-ink-3)">{i}</text>')
        for j, value in enumerate(row):
            bx = x + 16 + j * (box + _GAP)
            out.append(f'<rect x="{bx}" y="{y}" width="{box}" height="{_BH}" rx="4" style="fill:var(--ea-inset);{edge}"/>'
                       f'<text x="{bx + box / 2}" y="{y + _BH / 2 + 4.5}" style="{_MONO};font-size:12.5px;fill:var(--ea-{inks[i]}){weight};'
                       f'text-anchor:middle">{html.escape(value)}</text>')
    return "".join(out)


def _svg(width: float, height: float, parts: list[str]) -> str:
    """One picture of the block, at most 330 px wide, so that two fit side by side and stack in a narrow window."""
    return (f'<svg viewBox="0 0 {width} {height}" aria-hidden="true" style="flex:0 1 {width}px;min-width:min(100%,{width}px);'
            f'max-width:{width}px;height:auto;font-family:var(--ea-font-ui);font-variant-ligatures:none">{"".join(parts)}</svg>')


def _shown(column) -> list[str]:
    """The values of a column as pandas prints them: a whole number as 61, or as 61.0 once a NaN made it decimal."""
    if column.dtype.kind == "f":
        return ["NaN" if v != v else f"{v:.1f}" for v in column]
    return [str(v) for v in column]


@cache
def _looked_up():
    """The weather, the list where Saturday is twice, and for every number of days looked up the inner and the left
    join of those days with the list, as pandas makes them, with the columns the block shows. Read and merged once."""
    import pandas as pd
    weather, twice = _read(WEATHER), _read(SALES_TWICE)
    steps = [tuple(pd.merge(weather.head(k), twice, on="day", how=how)[["day", "temp", "sold"]] for how in ("inner", "left"))
             for k in range(len(weather) + 1)]
    return weather, twice, steps


def joins(days: int):
    """The inner and the left join of the first `days` days of the weather with the list where Saturday is twice."""
    return _looked_up()[2][days]


def lookup(days: int, labels: dict) -> str:
    """The block in *The idea* after `days` days of the weather are looked up in the list where Saturday is twice.
    The first picture: the weather (day and temperature) and the list (day and ice creams), with a line between each
    day looked up and every row of the list with the same day; the days not looked up yet are gray, and the day of
    this step is bold, with its lines. The second: the inner and the left join of the days looked up, as pandas makes
    them, with the rows of this step bold. Each picture is its own SVG, so that they go one below the other in a
    narrow window. Ink only: nothing here is learned or predicted."""
    weather, twice, steps = _looked_up()
    done = list(weather["day"][:days])
    now = done[-1] if done else None
    lx, gap = 2, 44
    rx = lx + 16 + 2 * _BOX + _GAP + gap
    parts = [f'<text x="0" y="16" style="font-size:14px;font-weight:600;fill:var(--ea-ink)">{html.escape(labels["title"])}</text>']
    for i, d in enumerate(done):
        for k, e in enumerate(twice["day"]):
            if d == e:
                y0, y1 = (_TOP + r * (_BH + _GAP) + _BH / 2 for r in (i, k))
                ink = "stroke:var(--ea-ink);stroke-width:2.2" if d == now else "stroke:var(--ea-ink-2);stroke-width:1.4"
                parts.append(f'<line x1="{lx + 16 + 2 * _BOX + _GAP + 4}" y1="{y0}" x2="{rx - 8}" y2="{y1}" style="{ink};stroke-linecap:round"/>')
    parts.append(_table(lx, ["day", "temp"], [[d, f"{t:.1f}"] for d, t in zip(weather["day"], weather["temp"])],
                        ["ink" if i < days else "ink-3" for i in range(len(weather))], {days - 1} if days else set()))
    parts.append(_table(rx, ["day", "sold"], [[d, str(n)] for d, n in zip(twice["day"], twice["sold"])],
                        ["ink" if d in done else "ink-3" for d in twice["day"]], {k for k, d in enumerate(twice["day"]) if d == now}))
    first = _svg(rx + 16 + 2 * _BOX + _GAP + 2, _TOP + len(weather) * (_BH + _GAP), parts)
    inner, left = joins(days)
    box = 42
    width = 16 + 3 * box + 2 * _GAP
    parts = []
    for x, name, table in ((2, labels["inner"], inner), (2 + width + 18, labels["left"], left)):
        n = len(table)
        title = f'{name}: {n} {labels["row"] if n == 1 else labels["rows"]}'
        parts.append(f'<text x="{x}" y="16" style="font-size:13px;font-weight:600;fill:var(--ea-ink)">{html.escape(title)}</text>')
        values = [list(r) for r in zip(table["day"], _shown(table["temp"]), _shown(table["sold"]))]
        parts.append(_table(x, ["day", "temp", "sold"], values, ["ink"] * n, {i for i, d in enumerate(table["day"]) if d == now}, box))
    # as high as the left join of the whole week, so that the block does not jump while you slide
    second = _svg(2 + 2 * width + 18 + 2, _TOP + len(steps[-1][1]) * (_BH + _GAP), parts)
    alt = labels["alt"].format(k=days, n=len(inner), m=len(left))
    return (f'<figure class="ea-figure ea-plot" role="img" aria-label="{html.escape(alt)}" style="margin:0.25rem 0 0.5rem;'
            f'display:flex;flex-wrap:wrap;gap:0.75rem 1.25rem;align-items:flex-start">{first}{second}</figure>')


# The lesson is done when these are correct. "fewer-rows" is the choice question in the lesson.
REQUIRED = ["week-2", "extra-rows", "fewer-rows"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")
_HOW_WORD = Mistake("how-word", error=lambda e: isinstance(e, ValueError) and "is not a valid Merge type" in str(e),
                    en="`how` gets one of a few words, in small letters and between quotes, such as `how=\"left\"`. And the key goes after `on=`, such as `on=\"day\"`: without `on=`, merge reads it as `how`.",
                    nl="`how` krijgt een van een paar woorden, in kleine letters en tussen aanhalingstekens, zoals `how=\"left\"`. En de key komt na `on=`, zoals `on=\"day\"`: zonder `on=` leest merge hem als `how`.")


def _kind(got) -> str:
    return type(got).__name__


def _uses_how(word: str):
    return lambda got, text: re.search(rf"how\s*=\s*[\"']{word}[\"']", text) is not None


# --- exercise 1: a left join of week 2 ---------------------------------------------------------------------------------

_READS = [f'weather = pd.read_csv("{WEATHER_2}")', f'sales = pd.read_csv("{SALES_2}")']
_WEEK_START = "import pandas as pd\n\n" + "\n".join(_READS) + "\nweek = ...\nweek\n"
_WEEK = "import pandas as pd\n\n" + "\n".join(_READS) + '\nweek = pd.merge(weather, sales, on="day", how="left")\nweek'


def _week(new: str) -> str:
    return _WEEK.replace('week = pd.merge(weather, sales, on="day", how="left")', new)


def _shown_table(value) -> str:
    return str(value)


EXERCISES = {
    "week-2": CodeBox(
        name="week-2",
        start=_WEEK_START,
        solution=_WEEK,
        keep=["import pandas as pd"] + _READS,
        # The check also runs the box with week 1, where the stand was closed on Monday only.
        others=[{"weather": Code(f'pd.read_csv("{WEATHER}")'), "sales": Code(f'pd.read_csv("{SALES}")')}],
        first=[_DOTS],
        mistakes=[
            _HOW_WORD,
            Mistake("no-column", error=lambda e: isinstance(e, KeyError),
                    en="One of the tables has no column with that name. The key is the column both tables have: `day`, in small letters, between quotes: `on=\"day\"`.",
                    nl="Een van de tabellen heeft geen kolom met die naam. De key is de kolom die beide tabellen hebben: `day`, in kleine letters, tussen aanhalingstekens: `on=\"day\"`."),
            Mistake("right-swapped", wrong=_week('week = pd.merge(sales, weather, on="day", how="right")'),
                    en="Your table has all 7 days, but its columns are in another order: `day`, `sold`, then the weather. With `sales` first, `sales` is the left table and its columns come first, and `how=\"right\"` keeps every row of the right table, `weather`. The task asks for a left join with `weather` as the left table: `pd.merge(weather, sales, on=\"day\", how=\"left\")`.",
                    nl="Je tabel heeft alle 7 dagen, maar de kolommen staan in een andere volgorde: `day`, `sold`, en dan het weer. Met `sales` eerst is `sales` de linkertabel en komen zijn kolommen eerst, en `how=\"right\"` houdt elke rij van de rechtertabel, `weather`. De opdracht vraagt om een left join met `weather` als linkertabel: `pd.merge(weather, sales, on=\"day\", how=\"left\")`."),
            Mistake("right", matches=lambda got, text: _uses_how("right")(got, text) and _kind(got) == "DataFrame" and len(got) < len(_read(WEATHER_2)),
                    en="With `how=\"right\"`, merge keeps every row of the right table, `sales`, so the days without sales are gone. The task asks for every day of `weather`, the left table: `how=\"left\"`.",
                    nl="Met `how=\"right\"` houdt merge elke rij van de rechtertabel, `sales`, dus de dagen zonder verkoop zijn weg. De opdracht vraagt om elke dag van `weather`, de linkertabel: `how=\"left\"`."),
            Mistake("outer", matches=_uses_how("outer"),
                    en="With `how=\"outer\"`, merge keeps every row of both tables, and sorts the days by name: `fri` comes first. The task asks for a left join, with every day of `weather` in its own order: `how=\"left\"`.",
                    nl="Met `how=\"outer\"` houdt merge elke rij van beide tabellen, en zet hij de dagen op alfabetische volgorde: `fri` staat vooraan. De opdracht vraagt om een left join, met elke dag van `weather` in zijn eigen volgorde: `how=\"left\"`."),
            Mistake("inner", wrong=_week('week = pd.merge(weather, sales, on="day")'),
                    en="Your table has 5 rows: Monday and Friday, the days without sales, are gone. That is an inner join, what merge does unless you say otherwise. The task asks for every day of `weather`: add `how=\"left\"`.",
                    nl="Je tabel heeft 5 rijen: maandag en vrijdag, de dagen zonder verkoop, zijn weg. Dat is een inner join, wat merge doet als je niets anders zegt. De opdracht vraagt om elke dag van `weather`: zet er `how=\"left\"` bij."),
            Mistake("swapped", wrong=_week('week = pd.merge(sales, weather, on="day", how="left")'),
                    en="Your table starts with `sales`: the left table is the first one you give merge, so it keeps the 5 days of `sales`. Put `weather` first: `pd.merge(weather, sales, ...)`.",
                    nl="Je tabel begint met `sales`: de linkertabel is de eerste die je aan merge geeft, dus hij houdt de 5 dagen van `sales`. Zet `weather` eerst: `pd.merge(weather, sales, ...)`."),
            Mistake("stacked", matches=lambda got, text: _kind(got) == "DataFrame" and len(got) == 12,
                    en="Your table has 12 rows: the 7 of `weather` and the 5 of `sales`, one under the other, with `NaN` wherever a table has no such column. That stacks the tables. The task puts them side by side, matched on the day: `pd.merge(weather, sales, on=\"day\", how=\"left\")`.",
                    nl="Je tabel heeft 12 rijen: de 7 van `weather` en de 5 van `sales`, onder elkaar, met `NaN` waar een tabel zo'n kolom niet heeft. Dat stapelt de tabellen. De opdracht zet ze naast elkaar, gekoppeld op de dag: `pd.merge(weather, sales, on=\"day\", how=\"left\")`."),
            Mistake("one-table", matches=lambda got, text: _kind(got) == "DataFrame" and "sold" not in got.columns or _kind(got) == "DataFrame" and "temp" not in got.columns,
                    en="Your table has the columns of only one of the two tables. The task asks for both side by side: merge them on `day`.",
                    nl="Je tabel heeft de kolommen van maar één van de twee tabellen. De opdracht vraagt om beide naast elkaar: voeg ze samen op `day`."),
        ],
        texts={
            "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                        "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
            "empty": {"en": "Replace the `...` with a left join of `weather` and `sales` on the day.",
                      "nl": "Vervang de `...` door een left join van `weather` en `sales` op de dag."},
            "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                        "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
            "keep": {"en": "Your code may be right, but the check needs the line `{line}` exactly as it was: put it back. The check tries another week by itself.",
                     "nl": "Je code kan goed zijn, maar de check heeft de regel `{line}` nodig zoals hij was: zet hem terug. De check probeert zelf een andere week."},
            "no_value": {"en": "The last line of the box shows nothing. Put `week` on the last line, so the check can see it.",
                         "nl": "De laatste regel van het vak laat niets zien. Zet `week` op de laatste regel, zodat de check het kan zien."},
            "wrong": {"en": "Your last line gives:\n\n```\n{got}\n```\n\nand it should be the 7 days of `weather`, in their order, with the ice creams of `sales` next to them, and `NaN` on Monday and Friday. Do you merge `weather` and `sales` on `day`, with `how=\"left\"`?",
                      "nl": "Je laatste regel geeft:\n\n```\n{got}\n```\n\nen dat moeten de 7 dagen van `weather` zijn, in hun volgorde, met de ijsjes van `sales` ernaast, en `NaN` op maandag en vrijdag. Voeg je `weather` en `sales` samen op `day`, met `how=\"left\"`?"},
            "other": {"en": "Right for this week. But with {where}, your last line gives:\n\n```\n{got}\n```\n\nand it should give:\n\n```\n{want}\n```\n\nDo you merge the tables `weather` and `sales` themselves?",
                      "nl": "Goed voor deze week. Maar met {where} geeft je laatste regel:\n\n```\n{got}\n```\n\nen dat moet dit zijn:\n\n```\n{want}\n```\n\nVoeg je de tabellen `weather` en `sales` zelf samen?"},
            "other_error": {"en": "Right for this week. But with {where}, your code stops with an error. Do you merge the tables `weather` and `sales` themselves?",
                            "nl": "Goed voor deze week. Maar met {where} stopt je code met een fout. Voeg je de tabellen `weather` en `sales` zelf samen?"},
            "correct": {"en": "7 rows, one for every day of `weather`, and `NaN` for the ice creams on Monday and Friday, when the stand was closed. The check also tried week 1, and your code kept all its days too.",
                        "nl": "7 rijen, één voor elke dag van `weather`, en `NaN` voor de ijsjes op maandag en vrijdag, toen het kraampje dicht was. De check probeerde ook week 1, en je code hield ook daar alle dagen."},
        },
    ),
}


# --- exercise 2: how many rows a left join adds ------------------------------------------------------------------------

_EXTRA_START = ("import pandas as pd\n\n\ndef extra_rows(left_path, right_path, key):\n    left = pd.read_csv(left_path)\n"
                "    right = pd.read_csv(right_path)\n    ...\n")
_EXTRA = ("import pandas as pd\n\n\ndef extra_rows(left_path, right_path, key):\n    left = pd.read_csv(left_path)\n"
          '    right = pd.read_csv(right_path)\n    merged = pd.merge(left, right, on=key, how="left")\n    return len(merged) - len(left)')


def _extra(new: str) -> str:
    return _EXTRA.replace('    merged = pd.merge(left, right, on=key, how="left")\n    return len(merged) - len(left)', new)


def _shown_value(value) -> str:
    """A number as a number; a table by its size, in words, in the lesson's language."""
    if _kind(value) == "DataFrame":
        n, cols = value.shape
        return {"en": f"a table of {n} rows and {cols} columns", "nl": f"een tabel van {n} rijen en {cols} kolommen"}
    return f"`{value}`"


EXERCISES["extra-rows"] = FunctionBox(
    name="extra-rows",
    function="extra_rows",
    start=_EXTRA_START,
    solution=_EXTRA,
    visible=[(WEATHER, SALES_TWICE, "day")],
    hidden=[(WEATHER, SALES, "day"), (SALES, WEATHER, "day"), (WEATHER_2, SALES_2, "day"), (PENGUINS, PENGUINS, "island")],
    show=_shown_value,
    first=[_DOTS],
    mistakes=[
        _HOW_WORD,
        Mistake("name-in-quotes", error=lambda e: isinstance(e, KeyError) and bool(e.args) and e.args[0] == "key",
                en="`on=\"key\"` looks for a column that is called `key`. Use the name `key` itself, without quotes: `on=key`; it holds the name the check gives, such as `\"day\"`.",
                nl="`on=\"key\"` zoekt een kolom die `key` heet. Gebruik de naam `key` zelf, zonder aanhalingstekens: `on=key`; daarin zit de naam die de check geeft, zoals `\"day\"`."),
        Mistake("fixed-key", error=lambda e: isinstance(e, KeyError) and bool(e.args) and e.args[0] == "day",
                en="With `{call}`, your function stops: it merges on `day`, and these tables have no such column. Merge on the key the function gets: `on=key`.",
                nl="Met `{call}` stopt je functie: hij voegt samen op `day`, en deze tabellen hebben die kolom niet. Voeg samen op de key die de functie krijgt: `on=key`."),
        Mistake("inner", wrong=_extra("    merged = pd.merge(left, right, on=key)\n    return len(merged) - len(left)"),
                en="`{call}` gives {got}, and it should give {want}. Your merge is an inner join, and that drops the rows without a match: here Monday drops out and the second Saturday comes in, so the count stays the same. Use `how=\"left\"`: then a row can only come in, never drop out.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn. Je merge is een inner join, en die laat de rijen zonder match weg: hier valt maandag weg en komt de tweede zaterdag erbij, zodat het aantal gelijk blijft. Gebruik `how=\"left\"`: dan kan er alleen een rij bij komen, nooit weg."),
        Mistake("sign", wrong=_extra('    merged = pd.merge(left, right, on=key, how="left")\n    return len(left) - len(merged)'),
                en="`{call}` gives {got}, and it should give {want}: your function takes the rows before minus the rows after. The task asks how many rows come in: after minus before.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie neemt de rijen ervoor min de rijen erna. De opdracht vraagt hoeveel rijen erbij komen: erna min ervoor."),
        Mistake("right-rows", state="almost", wrong=_extra('    merged = pd.merge(left, right, on=key, how="left")\n    return len(merged) - len(right)'),
                en="`{call}` gives {got}, and it should give {want}: your function compares with the rows of `right`. A left join keeps the rows of `left`: compare with `len(left)`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: je functie vergelijkt met de rijen van `right`. Een left join houdt de rijen van `left`: vergelijk met `len(left)`."),
        Mistake("how-right", matches=_uses_how("right"),
                en="`{call}` gives {got}, and it should give {want}: with `how=\"right\"`, merge keeps every row of the right table, `right`, so you count against the wrong table. A left join keeps every row of `left`: `how=\"left\"`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: met `how=\"right\"` houdt merge elke rij van de rechtertabel, `right`, dus je telt tegen de verkeerde tabel. Een left join houdt elke rij van `left`: `how=\"left\"`."),
        Mistake("outer", state="almost", matches=_uses_how("outer"),
                en="`{call}` gives {got}, and it should give {want}: with `how=\"outer\"`, merge also keeps the rows of `right` that find no match in `left`, and they count as added. A left join keeps the rows of `left` and no others: `how=\"left\"`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: met `how=\"outer\"` houdt merge ook de rijen van `right` die geen match vinden in `left`, en die tellen dan mee als erbij gekomen. Een left join houdt de rijen van `left` en geen andere: `how=\"left\"`."),
        Mistake("no-key", state="almost", wrong=_extra('    merged = pd.merge(left, right, how="left")\n    return len(merged) - len(left)'),
                en="`{call}` gives {got}, and it should give {want}: your merge does not use `key`. Without `on=`, merge matches on every column the two tables share; the penguins with themselves share every column, so each penguin finds only itself. Merge on the key the function gets: `on=key`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: je merge gebruikt `key` niet. Zonder `on=` koppelt merge op elke kolom die de twee tabellen delen; de pinguïns met zichzelf delen elke kolom, dus elke pinguïn vindt alleen zichzelf. Voeg samen op de key die de functie krijgt: `on=key`."),
        Mistake("swapped", wrong=_extra('    merged = pd.merge(right, left, on=key, how="left")\n    return len(merged) - len(left)'),
                en="`{call}` gives {got}, and it should give {want}: your merge has `right` as its left table, so it keeps the rows of `right`. Put `left` first: `pd.merge(left, right, ...)`.",
                nl="`{call}` geeft {got}, en dat moet {want} zijn: je merge heeft `right` als linkertabel, dus hij houdt de rijen van `right`. Zet `left` eerst: `pd.merge(left, right, ...)`."),
        Mistake("after-only", wrong=_extra('    merged = pd.merge(left, right, on=key, how="left")\n    return len(merged)'),
                en="`{call}` gives {got}: the number of rows after the merge. The task asks how many rows the merge adds: subtract the rows of `left`, from before the merge.",
                nl="`{call}` geeft {got}: het aantal rijen na de merge. De opdracht vraagt hoeveel rijen de merge erbij doet: trek er het aantal rijen van `left` van af, van voor de merge."),
        Mistake("table", matches=lambda got, text: _kind(got) == "DataFrame",
                en="`{call}` gives back {got}: the merged table itself. The task asks for one number: how many rows it has more than `left`, with `len`.",
                nl="`{call}` geeft {got} terug: de samengevoegde tabel zelf. De opdracht vraagt om één getal: hoeveel rijen hij meer heeft dan `left`, met `len`."),
    ],
    texts={
        "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                    "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
        "empty": {"en": "Replace the `...` with a left join of `left` and `right` on `key`, and give back how many rows it has more than `left`, with `return`.",
                  "nl": "Vervang de `...` door een left join van `left` en `right` op `key`, en geef met `return` terug hoeveel rijen die meer heeft dan `left`."},
        "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                    "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
        "arguments": {"en": "The check calls `{call}`, with two paths and a key, and the `def` line of your function has a different number of names between its parentheses. Keep `(left_path, right_path, key)`.",
                      "nl": "De check roept `{call}` aan, met twee paden en een key, en de `def`-regel van je functie heeft een ander aantal namen tussen de haakjes. Houd `(left_path, right_path, key)`."},
        "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Do you merge `left` and `right` on `key` with `how=\"left\"`, and give back the rows after minus the rows before?",
                  "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Voeg je `left` en `right` samen op `key` met `how=\"left\"`, en geef je de rijen erna min de rijen ervoor terug?"},
        "almost": {"en": "Right for the example. But `{call}` gives {got}, and it should give {want}. Does it work for every two files and every key?",
                   "nl": "Goed voor het voorbeeld. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt hij voor elke twee bestanden en elke key?"},
        "correct": {"en": "Your function is right in all {n} cases. The last one merged the penguins with themselves on `island`: every penguin matched every penguin of its island, and 344 rows became 46,304.",
                    "nl": "Je functie klopt in alle {n} gevallen. De laatste voegde de pinguïns samen met zichzelf op `island`: elke pinguïn vond elke pinguïn van zijn eiland, en 344 rijen werden er 46.304."},
    },
)

# What the tests feed each check, and what it must say: (exercise, what the learner typed, state, known mistake).
_K = "import pandas as pd\n\n" + "\n".join(_READS) + "\n"
_E = ("import pandas as pd\n\n\ndef extra_rows(left_path, right_path, key):\n    left = pd.read_csv(left_path)\n"
      "    right = pd.read_csv(right_path)\n")
CASES = [
    ("week-2", _WEEK_START, "empty", None),
    ("week-2", _WEEK, "correct", None),
    ("week-2", _K + 'week = weather.merge(sales, on="day", how="left")\nweek', "correct", None),
    ("week-2", _K + 'week = pd.merge(weather, sales, how="left")\nweek', "correct", None),
    ("week-2", _K + "week = pd.merge(weather, sales, on=[\"day\"], how='left')\nweek", "correct", None),
    ("week-2", _K + 'week = pd.merge(weather, sales, on="day")\nweek', "wrong", "inner"),
    ("week-2", _K + 'week = pd.merge(weather, sales, on="day", how="inner")\nweek', "wrong", "inner"),
    ("week-2", _K + 'week = pd.merge(weather, sales, on="day", how="right")\nweek', "wrong", "right"),
    ("week-2", _K + 'week = pd.merge(weather, sales, on="day", how="outer")\nweek', "wrong", "outer"),
    ("week-2", _K + 'week = pd.merge(sales, weather, on="day", how="left")\nweek', "wrong", "swapped"),
    ("week-2", _K + 'week = pd.merge(sales, weather, on="day", how="right")\nweek', "wrong", "right-swapped"),
    ("week-2", _K + "week = pd.concat([weather, sales])\nweek", "wrong", "stacked"),
    ("week-2", _K + "week = weather\nweek", "wrong", "one-table"),
    ("week-2", _K + 'week = pd.merge(weather, sales, on="day", how="Left")\nweek', "error", "how-word"),
    ("week-2", _K + 'week = pd.merge(weather, sales, "day")\nweek', "error", "how-word"),
    ("week-2", _K + 'week = pd.merge(weather, sales, on="sold", how="left")\nweek', "error", "no-column"),
    ("week-2", _K + 'week = pd.merge(weather, sales, on="Day", how="left")\nweek', "error", "no-column"),
    ("week-2", _K + 'week = pd.merge(weather, sales, on="day", how="left")\nprint(week)', "wrong", "print-last"),
    ("week-2", _K + 'week = pd.merge(weather, sales, on="day", how="left")\nweek["sold"]', "wrong", None),
    ("week-2", _K + 'week = pd.merge(weather, sales, on="day", how=...)\nweek', "wrong", "dots-left"),
    ("extra-rows", _EXTRA_START, "empty", None),
    ("extra-rows", _EXTRA, "correct", None),
    ("extra-rows", _E + '    return len(left.merge(right, on=key, how="left")) - len(left)', "correct", None),
    ("extra-rows", _E + '    before = left.shape[0]\n    after = pd.merge(left, right, on=key, how="left").shape[0]\n    return after - before', "correct", None),
    ("extra-rows", _E + "    merged = pd.merge(left, right, on=key)\n    return len(merged) - len(left)", "wrong", "inner"),
    ("extra-rows", _E + '    merged = pd.merge(left, right, on=key, how="left")\n    return len(left) - len(merged)', "wrong", "sign"),
    ("extra-rows", _E + '    merged = pd.merge(left, right, on=key, how="left")\n    return len(merged) - len(right)', "almost", "right-rows"),
    ("extra-rows", _E + '    merged = pd.merge(right, left, on=key, how="left")\n    return len(merged) - len(left)', "wrong", "swapped"),
    ("extra-rows", _E + '    merged = pd.merge(left, right, on=key, how="right")\n    return len(merged) - len(left)', "wrong", "how-right"),
    ("extra-rows", _E + '    merged = pd.merge(left, right, on=key, how="outer")\n    return len(merged) - len(left)', "almost", "outer"),
    ("extra-rows", _E + '    merged = pd.merge(left, right, how="left")\n    return len(merged) - len(left)', "almost", "no-key"),
    ("extra-rows", _E + '    merged = pd.merge(left, right, on=key, how="left")\n    return len(merged)', "wrong", "after-only"),
    ("extra-rows", _E + '    return pd.merge(left, right, on=key, how="left")', "wrong", "table"),
    ("extra-rows", _E + '    merged = pd.merge(left, right, on="key", how="left")\n    return len(merged) - len(left)', "error", "name-in-quotes"),
    ("extra-rows", _E + '    merged = pd.merge(left, right, on="day", how="left")\n    return len(merged) - len(left)', "error", "fixed-key"),
    ("extra-rows", _E + '    merged = pd.merge(left, right, on=key, how="Left")\n    return len(merged) - len(left)', "error", "how-word"),
    ("extra-rows", _E + "    return 1", "almost", None),
]
