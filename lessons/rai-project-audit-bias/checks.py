# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-project-audit-bias, the first step of the project audit: three functions the learner
writes and keeps in the project, report (a row per group with the yes rate, the impact ratio, the recall and the false
positive rate), calibration (per group, in one band of scores, the mean score and the share that earns more) and
threshold_for (the highest threshold that keeps the recall of a group at a target); with the pictures of the learner's
table, calibration and threshold, and what the learner's project makes of the whole table UCI Adult."""
import html
import importlib.util
import math
import numbers
import re

import pandas as pd

from everything_ai import paths, projects, yours
from everything_ai import check as checking
from everything_ai.check import FunctionBox, Mistake

# What the course's classifier gives on the test set (tests/test_rai_project_audit_bias.py checks it): per sex and
# per race the people, the yes rate with one threshold of 0.5, the impact ratio, the recall and the false positive rate.
BY_SEX = {"Female": (5421, 0.0705, 0.2843, 0.4356, 0.0259), "Male": (10860, 0.2479, 1.0, 0.5697, 0.1101)}
BY_RACE = {"Amer-Indian-Eskimo": (159, 0.0440, 0.1704, 0.2105, 0.0214),
           "Asian-Pac-Islander": (480, 0.2583, 1.0, 0.6541, 0.1066),
           "Black": (1561, 0.0705, 0.2728, 0.3855, 0.0297),
           "Other": (135, 0.0593, 0.2294, 0.2400, 0.0182),
           "White": (13946, 0.2026, 0.7841, 0.5576, 0.0841)}
RIGHT = 0.834             # how often the classifier is right on the test set, with one threshold of 0.5
RECALL = 0.55             # the recall the first fix aims at, for women and for men
THRESHOLDS = {"Female": 0.38, "Male": 0.52}   # what threshold_for gives for it on the people to train on
AFTER = {"Female": (0.1024, 0.4330, 0.5542, 0.0472), "Male": (0.2365, 1.0, 0.5507, 0.1019)}   # yes rate .. fpr
RIGHT_AFTER = 0.832       # how often it is right with the threshold per sex
# Calibration on the test set, in the band of exercise 2: (people, mean score, share that earns more) per sex.
BAND = (0.5, 0.6)
IN_BAND = {"Female": (105, 0.551, 0.476), "Male": (656, 0.550, 0.543)}

COLUMNS = ["people", "yes rate", "impact ratio", "recall", "false positive rate"]
CAL_COLUMNS = ["people", "mean score", "share rich"]
_GROUPS_OF = {"sex": ["Female", "Male"], "race": list(BY_RACE)}


def _course_model():
    """The course's own audit/model.py, as the lesson gives it to the learner: not the learner's copy."""
    spec = importlib.util.spec_from_file_location(
        "everything_ai_course_audit_model", paths.LESSONS / "rai-project-audit-bias" / "project" / "audit" / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_DATA: dict = {}


def table() -> pd.DataFrame:
    """UCI Adult with, for every person, the score of the course's classifier, rich (1 when the person earns more
    than 50,000 dollars, else 0) and yes (a score above 0.5). Made once, the first time a check needs it."""
    if "people" not in _DATA:
        people = pd.read_csv(paths.ROOT / "data" / "adult.csv.gz")
        people["score"] = _course_model().scores(people)
        people["rich"] = (people["income"] == ">50K").astype(int)
        people["yes"] = people["score"] > 0.5
        _DATA["people"] = people
    return _DATA["people"]


def test_part() -> pd.DataFrame:
    return table()[table()["split"] == "test"]


def train_part(sex: str) -> pd.DataFrame:
    people = table()
    return people[(people["split"] == "train") & (people["sex"] == sex)]


class Table:
    """A table the check calls by the name the lesson gives it, such as PEOPLE, so that a message stays short. It is
    made the first time a check needs it, and the learner's code gets its own copy."""

    def __init__(self, make, name: str):
        self.make, self.name, self._df = make, name, None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = self.make()
        return self._df

    def __repr__(self):
        return self.name

    def __deepcopy__(self, memo):
        return self.df.copy()


PEOPLE = Table(test_part, "PEOPLE")
WOMEN = Table(lambda: train_part("Female"), "WOMEN")
MEN = Table(lambda: train_part("Male"), "MEN")


def _made(name: str, rows: dict) -> Table:
    return Table(lambda: pd.DataFrame(rows), name)


# eight made-up people of two groups, with a column of another name than sex or race
EIGHT = _made("EIGHT", {"group": ["a", "a", "a", "a", "b", "b", "b", "b"],
                        "score": [0.9, 0.7, 0.4, 0.2, 0.6, 0.3, 0.2, 0.1],
                        "rich": [1, 1, 1, 0, 1, 0, 0, 0],
                        "yes": [True, True, False, False, False, True, False, False]})
# scores right on the edges of the band from 0.4 to 0.6
EDGES = _made("EDGES", {"group": ["a", "a", "a", "b", "b", "b"],
                        "score": [0.4, 0.5, 0.6, 0.45, 0.6, 0.39],
                        "rich": [0, 1, 1, 1, 0, 1],
                        "yes": [False, False, True, False, True, False]})
# four people who earn more, one at a score of exactly 0.6, and one who does not
FOUR = _made("FOUR", {"group": ["a", "a", "a", "a", "a"],
                      "score": [0.2, 0.35, 0.6, 0.8, 0.9],
                      "rich": [1, 1, 1, 1, 0],
                      "yes": [False, False, True, True, True]})


# --- the course's own functions, as the lesson writes them -----------------------------------------------------------

def report(people, column):
    rich = people[people["rich"] == 1]
    poor = people[people["rich"] == 0]
    out = pd.DataFrame()
    out["people"] = people.groupby(column).size()
    out["yes rate"] = people.groupby(column)["yes"].mean()
    out["impact ratio"] = out["yes rate"] / out["yes rate"].max()
    out["recall"] = rich.groupby(column)["yes"].mean()
    out["false positive rate"] = poor.groupby(column)["yes"].mean()
    return out


def calibration(people, column, low, high):
    band = people[people["score"] >= low]
    band = band[band["score"] < high]
    out = pd.DataFrame()
    out["people"] = band.groupby(column).size()
    out["mean score"] = band.groupby(column)["score"].mean()
    out["share rich"] = band.groupby(column)["rich"].mean()
    return out


def threshold_for(people, recall):
    rich = people[people["rich"] == 1]
    best = 0
    for t in range(1, 100):
        if (rich["score"] > t / 100).mean() >= recall:
            best = t / 100
    return best


def recall_curve(people) -> tuple[list, list]:
    """The thresholds 0.01 to 0.99 and, for each, the share of the people who earn more whose score is above it."""
    rich = people[people["rich"] == 1]["score"]
    ts = [t / 100 for t in range(1, 100)]
    return ts, [float((rich > t).mean()) for t in ts]


# --- numbers in the lesson's language --------------------------------------------------------------------------------

def number(x: float, lang: str, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", "," if lang == "nl" else ".")


def grouped(n: int, lang: str) -> str:
    """A whole number with a thousands separator: 13,946 in English, 13.946 in Dutch; 4 digits get none."""
    if abs(n) < 10000:
        return str(n)
    return f"{n:,}".replace(",", "." if lang == "nl" else ",")


def _short(x: float, decimal: str) -> str:
    """A number with at most two decimals, without zeros at the end: 0.8, 0.44, 1."""
    return f"{round(x, 2) + 0.0:g}".replace(".", decimal)


def _number(x) -> float | None:
    """A real number that is not True or False, as a float; else None."""
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return None
    x = float(x)
    return x if math.isfinite(x) else None


_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.2rem"


def markdown(df: pd.DataFrame, lang: str, flag: bool = True) -> str:
    """A table of report or calibration as a Markdown table: whole numbers grouped, shares with two decimals. With
    `flag`, a last column marks the groups whose impact ratio is below 0.8."""
    rule = {"en": "four-fifths rule", "nl": "80%-regel"}[lang]
    head = "| " + " | ".join([""] + [f"`{c}`" for c in df.columns]) + (f" | {rule} |" if flag else " |")
    line = "|---" + "|---:" * len(df.columns) + ("|---|" if flag else "|")
    out = [head, line]
    below = {"en": "below 0.8", "nl": "onder 0,8"}[lang]
    for group, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = _number(row[c])
            cells.append("–" if v is None else grouped(int(v), lang) if c == "people" else number(v, lang))
        mark = ""
        if flag:
            r = _number(row.get("impact ratio"))
            mark = f" {below} |" if r is not None and r < 0.8 else " |"
        out.append(f"| {html.escape(str(group))} | " + " | ".join(cells) + (" |" + mark if flag else " |"))
    return "\n".join(out)


def show_table(v):
    """A table in words, for the check's messages: its rows and its columns."""
    if isinstance(v, pd.DataFrame):
        rows = ", ".join(f"`{r}`" for r in list(v.index)[:6]) or "–"
        cols = ", ".join(f"`{c}`" for c in list(v.columns)[:8]) or "–"
        return {"en": f"a table with the rows {rows} and the columns {cols}",
                "nl": f"een tabel met de rijen {rows} en de kolommen {cols}"}
    if isinstance(v, pd.Series):
        rows = ", ".join(f"`{r}`" for r in list(v.index)[:6]) or "–"
        return {"en": f"one column (a Series) with the rows {rows}",
                "nl": f"één kolom (een Series) met de rijen {rows}"}
    text = f"`{v!r}`"
    if len(text) > 60:
        text = text[:58] + "…`"
    return {"en": text, "nl": text}


# --- the pictures under the checks ----------------------------------------------------------------------------------

def _cases(ex) -> list:
    return list(ex.visible) + list(ex.hidden)


def _first_different(seen) -> int:
    """The first case where the learner's function gives another value than the course's; 0 when none does."""
    ex = seen.exercise
    fn, right = seen.mine.names.get(ex.function), seen.want.names.get(ex.function)
    if not callable(fn) or not callable(right):
        return 0
    for i, case in enumerate(_cases(ex)):
        try:
            got = checking.with_time_limit(lambda: fn(*(checking._copy(a) for a in case)))
        except Exception:
            return i
        if not checking.same_value(got, right(*(checking._copy(a) for a in case))):
            return i
    return 0


def report_picture(seen) -> str:
    """The learner's table for the first case where it differs from the right one, beside the right one, with the
    cells that differ marked; the learner's table for report(PEOPLE, "sex") when it is right everywhere."""
    if seen.mine is None:
        return ""
    i = _first_different(seen)
    call = seen.exercise._call_text(_cases(seen.exercise)[i])
    title = {"en": f"Your table for {call}", "nl": f"Je tabel voor {call}"}
    return yours.table(i, title=title)(seen)


_CAL_WORDS = {
    "en": {"title": "Your calibration for women and men, in ten bands",
           "x": "mean score in the band", "y": "share that earns more",
           "key": "A dot per band of 0.1 from your calibration on PEOPLE: open for women, filled for men. On the dashed "
                  "line, the score comes true as often as it says. The band of the check, from 0.5 to 0.6, has a ring.",
           "alt": "Your calibration in ten bands. Women: {women}. Men: {men}."},
    "nl": {"title": "Jouw kalibratie voor vrouwen en mannen, in tien banden",
           "x": "gemiddelde score in de band", "y": "deel dat meer verdient",
           "key": "Een stip per band van 0,1 uit je calibration op PEOPLE: open voor vrouwen, gevuld voor mannen. Op de "
                  "stippellijn komt de score net zo vaak uit als hij zegt. De band van de check, van 0,5 tot 0,6, heeft "
                  "een ring.",
           "alt": "Jouw kalibratie in tien banden. Vrouwen: {women}. Mannen: {men}."},
}


def calibration_points(fn, people) -> dict:
    """What `fn`, a calibration function, gives per sex for the ten bands of 0.1: a list of (band, mean score, share)
    per sex, for the bands where it gives both numbers."""
    points = {"Female": [], "Male": []}
    for k in range(10):
        out = fn(people.copy(), "sex", k / 10, (k + 1) / 10)
        if not isinstance(out, pd.DataFrame) or not {"mean score", "share rich"} <= set(out.columns):
            return {}
        for sex in points:
            if sex in out.index:
                x, y = _number(out.loc[sex, "mean score"]), _number(out.loc[sex, "share rich"])
                if x is not None and y is not None and 0 <= x <= 1 and 0 <= y <= 1:
                    points[sex].append((k, x, y))
    return points


def calibration_figure(points: dict, words: dict, lang: str, ring: int | None = None) -> str:
    """Ten bands per sex as dots, open for women and filled for men, with the dashed line where a score comes true
    as often as it says. `words` gives "title", "x", "y", "key" and "alt"."""
    from everything_ai import chart
    ticks = [0, 0.25, 0.5, 0.75, 1]
    panel = chart.Panel(x=(0, 1), y=(0, 1), xlabel=words["x"], ylabel=words["y"], xticks=ticks, yticks=ticks,
                        xtick_labels=[f"{t:g}" for t in ticks])
    panel.line([0, 1], [0, 1], role="ink-3", width=1.2, dashed=True)
    for sex, hollow in (("Female", True), ("Male", False)):
        pts = points.get(sex, [])
        if pts:
            panel.points([p[1] for p in pts], [p[2] for p in pts], role="data", r=5, hollow=hollow)
            for k, x, y in pts:
                if k == ring:
                    panel.points([x], [y], role="ink", r=9, hollow=True)
    sign = "," if lang == "nl" else "."
    said = {s: ", ".join(f"{_short(x, sign)} → {_short(y, sign)}" for _, x, y in points.get(s, [])) or "–"
            for s in ("Female", "Male")}
    svg = chart.figure([panel], alt=words["alt"].format(women=said["Female"], men=said["Male"]), width=420, height=380)
    if lang == "nl":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(words["title"], svg + f'<div style="{_UI}">{html.escape(words["key"])}</div>')


def calibration_picture(seen) -> str:
    """The learner's calibration of the course's classifier on the test set, per sex, in the ten bands of 0.1."""
    mine = seen.mine
    if mine is None:
        return ""
    fn = mine.names.get(seen.exercise.function)
    if not callable(fn):
        return ""
    try:
        points = checking.with_time_limit(lambda: calibration_points(fn, PEOPLE.df))
    except Exception:
        return ""
    if not points or not (points["Female"] or points["Male"]):
        return ""
    return calibration_figure(points, _CAL_WORDS[seen.lang], seen.lang, ring=5)


_THRESHOLD_WORDS = {
    "en": {"title": "Your threshold for WOMEN, at a recall of 0.55",
           "x": "threshold", "y": "recall: share of rich with a yes", "yours": "yours {x}", "right": "right {x}",
           "outside": "Your function gives {x}, and a threshold lies from 0 to 1, so it is not in the picture.",
           "key": "The curve: the recall of the women to train on at every threshold from 0.01 to 0.99. Dashed: the "
                  "recall of 0.55. Your threshold is the solid upright line.",
           "alt": "The recall of the women to train on falls from {top} at a threshold of 0.01 to {low} at 0.99. Your "
                  "threshold: {mine}; the right one: {want}."},
    "nl": {"title": "Jouw threshold voor WOMEN, bij een recall van 0,55",
           "x": "threshold", "y": "recall: deel van rich met een ja", "yours": "jouw {x}", "right": "goed {x}",
           "outside": "Je functie geeft {x}, en een threshold ligt tussen 0 en 1, dus hij staat niet in het plaatje.",
           "key": "De lijn: de recall van de vrouwen om op te trainen bij elke threshold van 0,01 tot 0,99. "
                  "Gestippeld: de recall van 0,55. Jouw threshold is de doorgetrokken staande lijn.",
           "alt": "De recall van de vrouwen om op te trainen daalt van {top} bij een threshold van 0,01 tot {low} bij "
                  "0,99. Jouw threshold: {mine}; de goede: {want}."},
}


def threshold_picture(seen) -> str:
    """The recall of the women to train on at every threshold, the target of 0.55, the learner's threshold and,
    when it differs, the right one."""
    from everything_ai import chart
    mine = seen.mine
    if mine is None:
        return ""
    fn = mine.names.get(seen.exercise.function)
    if not callable(fn):
        return ""
    try:
        got = _number(checking.with_time_limit(lambda: fn(WOMEN.df.copy(), RECALL)))
    except Exception:
        return ""
    if got is None:
        return ""
    w = _THRESHOLD_WORDS[seen.lang]
    sign = "," if seen.lang == "nl" else "."
    want = threshold_for(WOMEN.df, RECALL)
    ts, recalls = seen.remember("curve", lambda: recall_curve(WOMEN.df))
    ticks = [0, 0.25, 0.5, 0.75, 1]
    panel = chart.Panel(x=(0, 1), y=(0, 1.08), xlabel=w["x"], ylabel=w["y"], xticks=ticks, yticks=ticks,
                        xtick_labels=[f"{t:g}" for t in ticks])
    panel.line(ts, recalls, role="data", width=2)
    panel.line([0, 1], [RECALL, RECALL], role="ink-3", width=1.2, dashed=True)
    note = w["key"]
    inside = 0 <= got <= 1
    if inside:
        panel.line([got, got], [0, 1.0], role="ink", width=2)
        panel.text(got + 0.015, 1.03, w["yours"].format(x=_short(got, sign)), "ink", anchor="start", halo=True)
    else:
        note += " " + w["outside"].format(x=_short(got, sign))
    if not (inside and abs(got - want) < 1e-9):
        panel.line([want, want], [0, 0.9], role="ink-2", width=1.2, dashed=True)
        panel.text(want - 0.015, 0.93, w["right"].format(x=_short(want, sign)), "ink-2", anchor="end", halo=True)
    alt = w["alt"].format(top=_short(recalls[0], sign), low=_short(recalls[-1], sign), mine=_short(got, sign),
                          want=_short(want, sign))
    svg = chart.figure([panel], alt=alt, width=460, height=330)
    if seen.lang == "nl":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(note)}</div>')


# --- the exercises ----------------------------------------------------------------------------------------------------

# The lesson is done when these are correct.
REQUIRED = ["report", "calibration", "threshold", "claim"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}


def _kept(name: str) -> dict:
    return {"en": f"Your `{name}` is right. You changed `{{saved}}` yourself, so the check leaves that file as it is. To save this version, delete the file and check again.",
            "nl": f"Je `{name}` klopt. Je hebt `{{saved}}` zelf veranderd, dus de check laat dat bestand zoals het is. Wil je deze versie bewaren, verwijder dan het bestand en check opnieuw."}


def _not_table(got, text) -> bool:
    return got is not None and not isinstance(got, pd.DataFrame)


def _columns_other(names: list):
    def match(got, text):
        return isinstance(got, pd.DataFrame) and list(got.columns) != names
    return match


# exercise 1: report

_R_HEAD = ("import pandas as pd\n\n\n"
           "def report(people, column):\n"
           '    rich = people[people["rich"] == 1]\n'
           '    poor = people[people["rich"] == 0]\n'
           "    table = pd.DataFrame()\n"
           '    table["people"] = people.groupby(column).size()\n')
_R_TAIL = "    return table\n"
_R_BODY = ('    table["yes rate"] = people.groupby(column)["yes"].mean()\n'
           '    table["impact ratio"] = table["yes rate"] / table["yes rate"].max()\n'
           '    table["recall"] = rich.groupby(column)["yes"].mean()\n'
           '    table["false positive rate"] = poor.groupby(column)["yes"].mean()\n')
_R_START = _R_HEAD + "    ...\n" + _R_TAIL
_R = _R_HEAD + _R_BODY + _R_TAIL


def _r(body: str) -> str:
    return _R_HEAD + body + _R_TAIL


def _rswap(old: str, new: str) -> str:
    assert _R_BODY.count(old) == 1, old
    return _r(_R_BODY.replace(old, new))


# exercise 2: calibration

_C_HEAD = ("import pandas as pd\n\n\n"
           "def calibration(people, column, low, high):\n"
           '    band = people[people["score"] >= low]\n')
_C_MIDDLE = ("    table = pd.DataFrame()\n"
             '    table["people"] = band.groupby(column).size()\n')
_C_HIGH = '    band = band[band["score"] < high]\n'
_C_TWO = ('    table["mean score"] = band.groupby(column)["score"].mean()\n'
          '    table["share rich"] = band.groupby(column)["rich"].mean()\n')


def _c(high: str = _C_HIGH, two: str = _C_TWO) -> str:
    return _C_HEAD + high + _C_MIDDLE + two + "    return table\n"


_C_START = _c("    ...\n", "    ...\n")
_C = _c()


def _ctwo(old: str, new: str) -> str:
    assert _C_TWO.count(old) == 1, old
    return _c(two=_C_TWO.replace(old, new))


# exercise 3: threshold_for

_T_HEAD = ("def threshold_for(people, recall):\n"
           '    rich = people[people["rich"] == 1]\n'
           "    best = 0\n"
           "    for t in range(1, 100):\n")
_T_IF = '        if (rich["score"] > t / 100).mean() >= recall:\n'
_T_SET = "            best = t / 100\n"


def _t(cond: str = _T_IF, body: str = _T_SET) -> str:
    return _T_HEAD + cond + body + "    return best\n"


_T_START = _T_HEAD + "        ...\n    return best\n"
_T = _t()


def _near(value: float):
    def match(got, text):
        return _number(got) is not None and abs(_number(got) - value) < 1e-9
    return match


EXERCISES = {
    "report": FunctionBox(
        name="report",
        picture=report_picture,
        function="report",
        start=_R_START,
        solution=_R,
        show=show_table,
        save_as=("audit", "report"),
        visible=[(PEOPLE, "sex")],
        hidden=[(PEOPLE, "race"), (EIGHT, "group")],
        first=[_DOTS],
        mistakes=[
            Mistake("not-table", matches=_not_table,
                    en="`{call}` gives {got}, and it should give the table `table`, with a row per group and a column per measure. Keep `return table` as the last line, and add each measure as a column of `table`.",
                    nl="`{call}` geeft {got}, en het moet de tabel `table` zijn, met een rij per groep en een kolom per maat. Houd `return table` als laatste regel, en zet elke maat als kolom in `table`."),
            Mistake("columns", matches=_columns_other(COLUMNS),
                    en="`{call}` gives {got}. The check wants these five columns, with exactly these names, in this order: `people`, `yes rate`, `impact ratio`, `recall` and `false positive rate`.",
                    nl="`{call}` geeft {got}. De check wil deze vijf kolommen, met precies deze namen, in deze volgorde: `people`, `yes rate`, `impact ratio`, `recall` en `false positive rate`."),
            Mistake("min", wrong=_rswap('/ table["yes rate"].max()', '/ table["yes rate"].min()'),
                    en="Your impact ratios are 1 or more: you divide by the lowest yes rate. The four-fifths rule compares every group with the group that gets a yes most often: divide by `table[\"yes rate\"].max()`.",
                    nl="Je impact ratio's zijn 1 of meer: je deelt door het laagste deel met een ja. De 80%-regel vergelijkt elke groep met de groep die het vaakst een ja krijgt: deel door `table[\"yes rate\"].max()`."),
            Mistake("overall", wrong=_rswap('/ table["yes rate"].max()', '/ people["yes"].mean()'),
                    en="Your impact ratios divide by how often everyone gets a yes. The four-fifths rule compares every group with the group that gets a yes most often, so the highest group gets exactly 1: divide by `table[\"yes rate\"].max()`.",
                    nl="Je impact ratio's delen door hoe vaak iedereen een ja krijgt. De 80%-regel vergelijkt elke groep met de groep die het vaakst een ja krijgt, dus de hoogste groep krijgt precies 1: deel door `table[\"yes rate\"].max()`."),
            Mistake("sum", wrong=_rswap('people.groupby(column)["yes"].mean()', 'people.groupby(column)["yes"].sum()'),
                    en="Your `yes rate` counts the people with a yes, and it should be the share of the group: `.mean()` of the column `yes`, as in the worked example.",
                    nl="Je `yes rate` telt de mensen met een ja, en het moet het deel van de groep zijn: `.mean()` van de kolom `yes`, zoals in het uitgewerkte voorbeeld."),
            Mistake("swapped", wrong=_rswap('rich.groupby(column)["yes"].mean()\n', 'poor.groupby(column)["yes"].mean()\n')
                    .replace('    table["false positive rate"] = poor.groupby', '    table["false positive rate"] = rich.groupby'),
                    en="Your `recall` and `false positive rate` are swapped. The recall counts the yes among the people who earn more, `rich`; the false positive rate among the people who do not, `poor`.",
                    nl="Je `recall` en `false positive rate` zijn omgewisseld. De recall telt de ja's onder de mensen die meer verdienen, `rich`; de false positive rate onder de mensen die dat niet doen, `poor`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with four lines, one per column: `yes rate`, `impact ratio`, `recall` and `false positive rate`.",
                      "nl": "Vervang de `...` door vier regels, één per kolom: `yes rate`, `impact ratio`, `recall` en `false positive rate`."},
            "wrong": {"en": "`{call}` gives another table than it should: the picture under this message marks the cells that differ.",
                      "nl": "`{call}` geeft een andere tabel dan hij moet: het plaatje onder dit bericht markeert de cellen die verschillen."},
            "almost": {"en": "Right for `report(PEOPLE, \"sex\")`. But `{call}` gives another table than it should: the picture under this message marks the cells that differ. Does your function work for every column and every group, with `column` everywhere a column name goes?",
                       "nl": "Goed voor `report(PEOPLE, \"sex\")`. Maar `{call}` geeft een andere tabel dan hij moet: het plaatje onder dit bericht markeert de cellen die verschillen. Werkt je functie voor elke kolom en elke groep, met `column` overal waar een kolomnaam hoort?"},
            "correct": {"en": "Your `report` is right in all {n} cases: per group the people, the yes rate, the impact ratio, the recall and the false positive rate. It is now in your project, in `{saved}`.",
                        "nl": "Je `report` klopt in alle {n} gevallen: per groep de mensen, het deel met een ja, de impact ratio, de recall en de false positive rate. Hij staat nu in je project, in `{saved}`."},
            "kept": _kept("report"),
        },
    ),
    "calibration": FunctionBox(
        name="calibration",
        picture=calibration_picture,
        function="calibration",
        start=_C_START,
        solution=_C,
        show=show_table,
        save_as=("audit", "calibration"),
        visible=[(PEOPLE, "sex", 0.5, 0.6)],
        hidden=[(PEOPLE, "race", 0.2, 0.3), (EDGES, "group", 0.4, 0.6)],
        first=[_DOTS],
        mistakes=[
            Mistake("not-table", matches=_not_table,
                    en="`{call}` gives {got}, and it should give the table `table`, with a row per group and a column per number. Keep `return table` as the last line.",
                    nl="`{call}` geeft {got}, en het moet de tabel `table` zijn, met een rij per groep en een kolom per getal. Houd `return table` als laatste regel."),
            Mistake("columns", matches=_columns_other(CAL_COLUMNS),
                    en="`{call}` gives {got}. The check wants these three columns, with exactly these names, in this order: `people`, `mean score` and `share rich`.",
                    nl="`{call}` geeft {got}. De check wil deze drie kolommen, met precies deze namen, in deze volgorde: `people`, `mean score` en `share rich`."),
            Mistake("restart", wrong=_c('    band = people[people["score"] < high]\n'),
                    en="Your second line starts again from `people`, so it throws away the first: your band holds everyone below `high`, also below `low`. Take the rows of `band` instead: `band = band[...]`.",
                    nl="Je tweede regel begint weer bij `people`, dus hij gooit de eerste weg: je band bevat iedereen onder `high`, ook onder `low`. Neem in plaats daarvan de rijen van `band`: `band = band[...]`."),
            Mistake("no-high", wrong=_c(""),
                    en="Your band has no upper end: it holds everyone with a score of `low` or more. Keep only the rows of `band` whose score is below `high`.",
                    nl="Je band heeft geen bovenkant: hij bevat iedereen met een score van `low` of meer. Houd alleen de rijen van `band` waarvan de score onder `high` ligt."),
            Mistake("edge", wrong=_c('    band = band[band["score"] <= high]\n'),
                    en="Right for most people, but a score of exactly `high` is in your band, and also in the next band, which starts at that score. Count it only in the next one: `< high`.",
                    nl="Goed voor de meeste mensen, maar een score van precies `high` zit in je band, en ook in de volgende band, die bij die score begint. Tel hem alleen in de volgende: `< high`."),
            Mistake("whole", wrong=_ctwo("band.groupby(column)[\"score\"]", "people.groupby(column)[\"score\"]")
                    .replace('band.groupby(column)["rich"]', 'people.groupby(column)["rich"]'),
                    en="Your `mean score` and `share rich` are of everyone in `people`, not of the people in the band. Group `band` instead.",
                    nl="Je `mean score` en `share rich` gaan over iedereen in `people`, niet over de mensen in de band. Groepeer `band` in plaats daarvan."),
            Mistake("yes-share", wrong=_ctwo('band.groupby(column)["rich"]', 'band.groupby(column)["yes"]'),
                    en="Your `share rich` is the share with a yes: what the classifier says, and calibration compares what it says with what is true. Take the mean of the column `rich`.",
                    nl="Je `share rich` is het deel met een ja: wat de classifier zegt, en kalibratie vergelijkt wat hij zegt met wat waar is. Neem het gemiddelde van de kolom `rich`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the two `...`: first a line that keeps the rows of `band` whose score is below `high`, then the two columns `mean score` and `share rich`.",
                      "nl": "Vervang de twee `...`: eerst een regel die de rijen van `band` houdt waarvan de score onder `high` ligt, dan de twee kolommen `mean score` en `share rich`."},
            "wrong": {"en": "`{call}` gives {got}, and its numbers differ from the right ones. Look at which people are in your band, and which column each mean is of.",
                      "nl": "`{call}` geeft {got}, en de getallen wijken af van de goede. Kijk welke mensen in je band zitten, en van welke kolom elk gemiddelde is."},
            "almost": {"en": "Right for `calibration(PEOPLE, \"sex\", 0.5, 0.6)`. But `{call}` gives other numbers than it should. Does your function use `column`, `low` and `high` everywhere, and not numbers or names of its own?",
                       "nl": "Goed voor `calibration(PEOPLE, \"sex\", 0.5, 0.6)`. Maar `{call}` geeft andere getallen dan hij moet. Gebruikt je functie overal `column`, `low` en `high`, en geen getallen of namen van zichzelf?"},
            "correct": {"en": "Your `calibration` is right in all {n} cases: per group, in the band, how many people, their mean score and the share that earns more. It is now in your project, in `{saved}`.",
                        "nl": "Je `calibration` klopt in alle {n} gevallen: per groep, in de band, hoeveel mensen, hun gemiddelde score en het deel dat meer verdient. Hij staat nu in je project, in `{saved}`."},
            "kept": _kept("calibration"),
        },
    ),
    "threshold": FunctionBox(
        name="threshold",
        picture=threshold_picture,
        function="threshold_for",
        start=_T_START,
        solution=_T,
        save_as=("audit", "fix"),
        visible=[(WOMEN, RECALL)],
        hidden=[(MEN, RECALL), (FOUR, 0.5)],
        first=[_DOTS],
        mistakes=[
            Mistake("percent", wrong=_t(body="            best = t\n"),
                    en="`{call}` gives {got}, and it should give {want}. A threshold is a probability from 0 to 1, and `t` counts from 1 to 99: keep `t / 100` in `best`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Een threshold is een kans van 0 tot 1, en `t` telt van 1 tot 99: bewaar `t / 100` in `best`."),
            Mistake("everyone", wrong=_t('        if (people["score"] > t / 100).mean() >= recall:\n'),
                    en="`{call}` gives {got}, and it should give {want}. Your `if` looks at the share of everyone with a yes, and the recall is the share of the people who earn more: take the scores of `rich`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je `if` kijkt naar het deel van iedereen met een ja, en de recall is het deel van de mensen die meer verdienen: neem de scores van `rich`."),
            Mistake("count", wrong=_t('        if (rich["score"] > t / 100).sum() >= recall:\n'),
                    en="`{call}` gives {got}, and it should give {want}. `.sum()` counts the people with a yes, and the recall is a share: take `.mean()`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. `.sum()` telt de mensen met een ja, en de recall is een deel: neem `.mean()`."),
            Mistake("strict", wrong=_t('        if (rich["score"] > t / 100).mean() > recall:\n'),
                    en="`{call}` gives {got}, and it should give {want}. At the threshold {want}, the recall is exactly {recall}, and that is enough: at least the share `recall` means `>= recall`.".replace("{recall}", "0.5"),
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Bij de threshold {want} is de recall precies {recall}, en dat is genoeg: minstens het deel `recall` betekent `>= recall`.".replace("{recall}", "0,5")),
            Mistake("at-threshold", wrong=_t('        if (rich["score"] >= t / 100).mean() >= recall:\n'),
                    en="`{call}` gives {got}, and it should give {want}. A person with a score of exactly 0.6 gets a yes in your code at the threshold 0.6, and a yes is a score above the threshold, as in `score > 0.5`: use `>` for the scores.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Iemand met een score van precies 0,6 krijgt in jouw code een ja bij de threshold 0,6, en een ja is een score boven de threshold, zoals in `score > 0.5`: gebruik `>` voor de scores."),
            Mistake("flipped", matches=_near(0.99),
                    en="`{call}` gives 0.99, the highest threshold of the list: your `if` is true even there, where almost no one gets a yes. It should hold only while at least the share `recall` of `rich` gets a yes: a yes is a score above `t / 100`, and that share must be at least `recall`.",
                    nl="`{call}` geeft 0,99, de hoogste threshold van de lijst: je `if` is zelfs daar waar, waar bijna niemand een ja krijgt. Hij moet alleen gelden zolang minstens het deel `recall` van `rich` een ja krijgt: een ja is een score boven `t / 100`, en dat deel moet minstens `recall` zijn."),
            Mistake("first", matches=_near(0.01),
                    en="`{call}` gives 0.01, the lowest threshold of the list. The function should keep going: every `t` that passes replaces `best`, so that `best` ends as the highest threshold that still passes.",
                    nl="`{call}` geeft 0,01, de laagste threshold van de lijst. De functie moet doorgaan: elke `t` die slaagt vervangt `best`, zodat `best` eindigt als de hoogste threshold die nog slaagt."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with two lines: an `if` that asks whether at least the share `recall` of `rich` has a score above `t / 100`, and under it `best = t / 100`.",
                      "nl": "Vervang de `...` door twee regels: een `if` die vraagt of minstens het deel `recall` van `rich` een score boven `t / 100` heeft, en daaronder `best = t / 100`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}: the highest threshold at which at least the share `recall` of the people who earn more gets a yes. The picture under this message shows the recall at every threshold.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn: de hoogste threshold waarbij minstens het deel `recall` van de mensen die meer verdienen een ja krijgt. Het plaatje onder dit bericht laat de recall bij elke threshold zien."},
            "almost": {"en": "Right for `threshold_for(WOMEN, 0.55)`. But `{call}` gives {got}, and it should give {want}. Does your function work for every table and every recall, with `people` and `recall` and no numbers of its own?",
                       "nl": "Goed voor `threshold_for(WOMEN, 0.55)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Werkt je functie voor elke tabel en elke recall, met `people` en `recall` en zonder eigen getallen?"},
            "correct": {"en": "Your `threshold_for` is right in all {n} cases: 0.38 for the women and 0.52 for the men, at a recall of 0.55. It is now in your project, in `{saved}`.",
                        "nl": "Je `threshold_for` klopt in alle {n} gevallen: 0,38 voor de vrouwen en 0,52 voor de mannen, bij een recall van 0,55. Hij staat nu in je project, in `{saved}`."},
            "kept": _kept("threshold_for"),
        },
    ),
}

CASES = [
    ("report", _R_START, "empty", None),
    ("report", _r("    ...\n" + _R_BODY.split("\n", 1)[1]), "wrong", "dots-left"),
    ("report", _R, "correct", None),
    ("report", _rswap('rich.groupby(column)["yes"].mean()', 'rich.groupby(column)["yes"].sum() / rich.groupby(column).size()'),
     "correct", None),
    ("report", _r('    table["yes rate"] = people.groupby(column)["yes"].mean()\n'
                  '    table["impact ratio"] = table["yes rate"] / max(table["yes rate"])\n'
                  '    table["recall"] = people[people["rich"] == 1].groupby(column)["yes"].mean()\n'
                  '    table["false positive rate"] = poor.groupby(column)["yes"].mean()\n'), "correct", None),
    ("report", _R.replace("    return table\n", '    return table["yes rate"]\n'), "wrong", "not-table"),
    ("report", _rswap('table["false positive rate"]', 'table["fpr"]'), "wrong", "columns"),
    ("report", _rswap('/ table["yes rate"].max()', '/ table["yes rate"].min()'), "wrong", "min"),
    ("report", _rswap('/ table["yes rate"].max()', '/ people["yes"].mean()'), "wrong", "overall"),
    ("report", _rswap('people.groupby(column)["yes"].mean()', 'people.groupby(column)["yes"].sum()'), "wrong", "sum"),
    ("report", EXERCISES["report"].mistakes[5].wrong, "wrong", "swapped"),
    ("report", _rswap('rich.groupby(column)["yes"].mean()', 'people.groupby(column)["yes"].mean()'), "wrong", None),
    ("report", _R.replace("groupby(column)", 'groupby("sex")'), "almost", None),
    ("calibration", _C_START, "empty", None),
    ("calibration", _c(two="    ...\n"), "wrong", "dots-left"),
    ("calibration", _C, "correct", None),
    ("calibration", _c('    band = band[high > band["score"]]\n'), "correct", None),
    ("calibration", _c('    band = people[(people["score"] >= low) * (people["score"] < high)]\n'), "correct", None),
    ("calibration", _ctwo('band.groupby(column)["rich"].mean()', 'band.groupby(column)["rich"].sum() / band.groupby(column).size()'),
     "correct", None),
    ("calibration", _c(two='    table["mean score"] = band.groupby(column)["score"].mean()\n'), "wrong", "columns"),
    ("calibration", _c(two=_C_TWO).replace("    return table\n", '    return table["share rich"]\n'), "wrong",
     "not-table"),
    ("calibration", _c('    band = people[people["score"] < high]\n'), "wrong", "restart"),
    ("calibration", _c(""), "wrong", "no-high"),
    ("calibration", _c('    band = band[band["score"] <= high]\n'), "wrong", "edge"),
    ("calibration", EXERCISES["calibration"].mistakes[5].wrong, "wrong", "whole"),
    ("calibration", _ctwo('band.groupby(column)["rich"]', 'band.groupby(column)["yes"]'), "wrong", "yes-share"),
    ("calibration", _ctwo('band.groupby(column)["score"]', 'band.groupby(column)["rich"]'), "wrong", None),
    ("calibration", _C.replace("groupby(column)", 'groupby("sex")'), "almost", None),
    ("threshold", _T_START, "empty", None),
    ("threshold", _t(body="            ...\n"), "wrong", "dots-left"),
    ("threshold", _T, "correct", None),
    ("threshold", _t("        threshold = t / 100\n        if (rich[\"score\"] > threshold).mean() >= recall:\n",
                     "            best = threshold\n"), "correct", None),
    ("threshold", _t('        if recall <= (rich["score"] > t / 100).mean():\n'), "correct", None),
    ("threshold", _t(body="            best = t\n"), "wrong", "percent"),
    ("threshold", _t('        if (people["score"] > t / 100).mean() >= recall:\n'), "wrong", "everyone"),
    ("threshold", _t('        if (rich["score"] > t / 100).sum() >= recall:\n'), "wrong", "count"),
    ("threshold", _t('        if (rich["score"] > t / 100).mean() > recall:\n'), "wrong", "strict"),
    ("threshold", _t('        if (rich["score"] >= t / 100).mean() >= recall:\n'), "wrong", "at-threshold"),
    ("threshold", _t('        if (rich["score"] > t / 100).mean() <= recall:\n'), "wrong", "flipped"),
    ("threshold", _t('        if (rich["score"] > t / 100).mean() < recall:\n'), "wrong", "flipped"),
    ("threshold", _t(body="            return t / 100\n"), "wrong", "first"),
    ("threshold", _t(body="            best = t / 10\n"), "wrong", None),
    ("threshold", _T_HEAD + "        best = 0.38\n    return best\n", "almost", None),
]


# --- what your project can do now: your audit of your own classifier ----------------------------------------------

_DEMO_RUNS: dict = {}
_MODULES = ("model", "report", "calibration", "fix")


def _project_texts() -> tuple:
    """The texts of the project files the demo uses, as they are now (None for a missing file)."""
    texts = []
    for module in _MODULES:
        path = projects.folder() / "audit" / f"{module}.py"
        texts.append(path.read_text(encoding="utf-8") if path.exists() else None)
    return tuple(texts)


def _good_report(df) -> bool:
    return isinstance(df, pd.DataFrame) and list(df.columns) == COLUMNS and len(df) >= 1


def demo() -> dict | str | None:
    """Your audit of your own classifier on the whole table: your model (audit/model.py) trained on the people to
    train on, your report per sex and per race on the test set, your calibration per sex in ten bands, and your
    threshold_for for a threshold per sex at a recall of 0.55, with your report after that fix. None before exercise 1;
    the name of the module that stops with an error or gives something else than it should."""
    texts = _project_texts()
    if texts[1] is None:
        return None
    if texts in _DEMO_RUNS:
        return _DEMO_RUNS[texts]
    projects.forget()
    out = _demo()
    _DEMO_RUNS.clear()
    _DEMO_RUNS[texts] = out
    return out


def _demo() -> dict | str:
    from everything_ai.check import run_code, with_time_limit
    have = {m: (projects.folder() / "audit" / f"{m}.py").exists() for m in _MODULES}
    if not have["model"]:
        return "model"
    run = run_code("from audit.model import scores")
    if run.error is not None:
        return "model"
    people = pd.read_csv(paths.ROOT / "data" / "adult.csv.gz")
    try:
        score = with_time_limit(lambda: run.names["scores"](people.copy()), 120)
        people["score"] = [float(s) for s in score]
    except Exception:   # also Timeout, and a length that does not fit
        return "model"
    if not all(0 <= s <= 1 for s in people["score"]):
        return "model"
    people["rich"] = (people["income"] == ">50K").astype(int)
    people["yes"] = people["score"] > 0.5
    test = people[people["split"] == "test"]
    out = {"right": float((test["yes"] == (test["rich"] == 1)).mean()), "calibration": None, "fix": None}

    run = run_code("from audit.report import report")
    if run.error is not None:
        return "report"
    try:
        out["sex"] = with_time_limit(lambda: run.names["report"](test.copy(), "sex"), 30)
        out["race"] = with_time_limit(lambda: run.names["report"](test.copy(), "race"), 30)
    except Exception:
        return "report"
    if not (_good_report(out["sex"]) and _good_report(out["race"])):
        return "report"
    report_fn = run.names["report"]

    if have["calibration"]:
        run = run_code("from audit.calibration import calibration")
        try:
            points = with_time_limit(lambda: calibration_points(run.names["calibration"], test), 30)
        except Exception:
            points = {}
        out["calibration"] = points if points and (points["Female"] or points["Male"]) else "calibration"
        if run.error is not None:
            out["calibration"] = "calibration"

    if have["fix"]:
        run = run_code("from audit.fix import threshold_for")
        try:
            thresholds = {}
            for sex in ("Female", "Male"):
                mine = people[(people["split"] == "train") & (people["sex"] == sex)]
                thresholds[sex] = _number(with_time_limit(lambda: run.names["threshold_for"](mine.copy(), RECALL), 30))
            if any(t is None or not 0 <= t <= 1 for t in thresholds.values()):
                raise ValueError("no threshold")
            fixed = test.copy()
            fixed["yes"] = fixed["score"] > fixed["sex"].map(thresholds)
            after = with_time_limit(lambda: report_fn(fixed.copy(), "sex"), 30)
            after_race = with_time_limit(lambda: report_fn(fixed.copy(), "race"), 30)
            if not (_good_report(after) and _good_report(after_race)):
                raise ValueError("no report")
            out["fix"] = {"thresholds": thresholds, "sex": after, "race": after_race,
                          "right": float((fixed["yes"] == (fixed["rich"] == 1)).mean())}
        except Exception:
            out["fix"] = "fix"
    return out


# --- the picture of the demo in The idea, and what the lesson shows of your audit --------------------------------

def picture_race(labels: dict) -> str:
    """A bar per race with its yes rate on the test set, its people beside its name, and a dashed line at 0.8 times
    the highest yes rate: the bars that end before it are below the four-fifths rule. `labels` gives "title", "xlabel",
    "line", "key", "decimal" and "alt"."""
    from everything_ai import chart
    rows = sorted(BY_RACE.items(), key=lambda kv: -kv[1][1])
    top = max(v[1] for v in BY_RACE.values())
    sign = labels["decimal"]
    n = len(rows)
    panel = chart.Panel(x=(0, 0.3), y=(0, 1.2 * n + 0.6), xlabel=labels["xlabel"], yticks=[],
                        xticks=[0, 0.1, 0.2, 0.3], xtick_labels=["0", "0.1", "0.2", "0.3"])
    line = 0.8 * top
    panel.line([line, line], [0, 1.2 * n + 0.1], role="ink-3", width=1.2, dashed=True)  # first: the texts go over it
    for i, (race, (people, rate, ratio, _, _)) in enumerate(rows):
        y = 1.2 * (n - i - 1) + 0.1
        panel.text(0.002, y + 0.8, f"{race} ({grouped(people, 'nl' if sign == ',' else 'en')})", "ink", anchor="start")
        panel.rect(0, y + 0.1, rate, y + 0.5, role="data", solid=True)
        panel.text(rate + 0.004, y + 0.3, _short(ratio, sign), "ink-2", anchor="start", halo=True)
    panel.text(line + 0.003, 1.2 * n + 0.35, labels["line"], "ink-3", anchor="start", halo=True)
    svg = chart.figure([panel], alt=labels["alt"], width=520, height=340)
    if sign == ",":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(labels["title"], svg + f'<div style="{_UI}">{html.escape(labels["key"])}</div>')


_FIX_WORDS = {
    "en": {"title": "Your fix: before and after a threshold per sex", "xlabel": "share",
           "rows": {"yes rate": "yes rate", "recall": "recall", "false positive rate": "false positive rate"},
           "sex": {"Female": "women", "Male": "men"},
           "key": "Open: one threshold of 0.5; filled: your thresholds per sex. The false positive rates in purple: "
                  "they count mistakes.",
           "alt": "Before and after your thresholds per sex: {said}."},
    "nl": {"title": "Jouw aanpassing: voor en na een threshold per geslacht", "xlabel": "deel",
           "rows": {"yes rate": "deel met een ja", "recall": "recall", "false positive rate": "false positive rate"},
           "sex": {"Female": "vrouwen", "Male": "mannen"},
           "key": "Open: één threshold van 0,5; gevuld: jouw thresholds per geslacht. De false positive rates in paars: "
                  "die tellen fouten.",
           "alt": "Voor en na jouw thresholds per geslacht: {said}."},
}


def fix_picture(before: pd.DataFrame, after: pd.DataFrame, lang: str) -> str:
    """Per sex and per measure, the value with one threshold of 0.5 (open) and with the thresholds per sex
    (filled), joined by a line; the false positive rates in the color of mistakes."""
    from everything_ai import chart
    w = _FIX_WORDS[lang]
    sign = "," if lang == "nl" else "."
    rows = []
    for measure in ("yes rate", "recall", "false positive rate"):
        for sex in ("Female", "Male"):
            if sex in before.index and sex in after.index:
                a, b = _number(before.loc[sex, measure]), _number(after.loc[sex, measure])
                if a is not None and b is not None:
                    rows.append((f'{w["rows"][measure]}, {w["sex"][sex]}', a, b, measure == "false positive rate"))
    if not rows:
        return ""
    n = len(rows)
    ticks = [0, 0.25, 0.5, 0.75, 1]
    panel = chart.Panel(x=(-0.9, 1), y=(0, n), xlabel=w["xlabel"], yticks=[], xticks=ticks,
                        xtick_labels=[f"{t:g}" for t in ticks])
    said = []
    for i, (name, a, b, mistake) in enumerate(rows):
        y = n - i - 0.5
        role = "loss" if mistake else "data"
        panel.text(-0.88, y, name, "ink", anchor="start")
        panel.line([a, b], [y, y], role=role, width=2)
        panel.points([a], [y], role=role, r=5, hollow=True)
        panel.points([b], [y], role=role, r=5)
        said.append(f"{name} {_short(a, sign)} → {_short(b, sign)}")
    svg = chart.figure([panel], alt=w["alt"].format(said="; ".join(said)), width=600, height=60 + 34 * n)
    if lang == "nl":
        svg = chart.with_decimal_comma(svg)
    return yours.figure(w["title"], svg + f'<div style="{_UI}">{html.escape(w["key"])}</div>')


_PARTS = {
    "en": {
        "sex": "Your classifier is right for **{right}** of the test set, with one threshold of 0.5. Your `report` per sex:",
        "race": "And per race:",
        "flagged": "Below 0.8, the line of the four-fifths rule: {groups}. The highest yes rate, the 1 that every group is compared with, belongs to {top}, a group of {people} people.",
        "none": "No group is below 0.8, the line of the four-fifths rule. The highest yes rate belongs to {top}, a group of {people} people.",
        "cal_wait": "*Finish exercise 2: then your calibration per band appears here.*",
        "cal": "Your calibration per sex, in ten bands of 0.1 on the test set. Where the dots of women and men lie apart at the same mean score, a score means something else for one sex than for the other. Bands with few people can lie far from the dashed line by chance.",
        "fix_wait": "*Finish exercise 3: then your thresholds per sex appear here, with your `report` after the fix.*",
        "fix": "Your `threshold_for` gives, on the people to train on, at a recall of 0.55: **{women}** for the women and **{men}** for the men. With those thresholds, your `report` on the test set:",
        "after": "The classifier is now right for **{right}** of the test set, against {before} with one threshold. The impact ratio of the women went from {ratio_before} to {ratio_after}: equal recall is not equal yes rates, because a larger share of the men than of the women in the table earn more. Per race: `White` {white_before} → {white_after}, `Black` {black_before} → {black_after}. A threshold per sex was chosen for sex; an audit measures again for every group the decision can harm.",
        "broken": {
            "model": "*Your classifier in `work/projects/audit/model.py` stops with an error, or its `scores` does not give one probability from 0 to 1 per person. Did you change the file? Undo your change, or delete the file and open the lesson again: the course then puts its own version back.*",
            "report": "*Your `report` in `work/projects/audit/report.py` stops with an error, or does not give the table with its five columns. Did you change its file? Delete it, and check exercise 1 again: its check then saves your code again.*",
            "calibration": "*Your `calibration` in `work/projects/audit/calibration.py` stops with an error, or does not give `mean score` and `share rich`. Did you change its file? Delete it, and check exercise 2 again.*",
            "fix": "*Your `threshold_for` in `work/projects/audit/fix.py` stops with an error, or does not give a threshold from 0 to 1. Did you change its file? Delete it, and check exercise 3 again.*",
        },
    },
    "nl": {
        "sex": "Je classifier heeft het goed voor **{right}** van de test set, met één threshold van 0,5. Je `report` per geslacht:",
        "race": "En per afkomst:",
        "flagged": "Onder 0,8, de lijn van de 80%-regel: {groups}. Het hoogste deel met een ja, de 1 waarmee elke groep wordt vergeleken, hoort bij {top}, een groep van {people} mensen.",
        "none": "Geen groep zit onder 0,8, de lijn van de 80%-regel. Het hoogste deel met een ja hoort bij {top}, een groep van {people} mensen.",
        "cal_wait": "*Maak eerst oefening 2: dan verschijnt hier je kalibratie per band.*",
        "cal": "Je kalibratie per geslacht, in tien banden van 0,1 op de test set. Waar de stippen van vrouwen en mannen bij dezelfde gemiddelde score uit elkaar liggen, betekent een score voor het ene geslacht iets anders dan voor het andere. Banden met weinig mensen kunnen bij toeval ver van de stippellijn liggen.",
        "fix_wait": "*Maak eerst oefening 3: dan verschijnen hier je thresholds per geslacht, met je `report` na de aanpassing.*",
        "fix": "Je `threshold_for` geeft, op de mensen om op te trainen, bij een recall van 0,55: **{women}** voor de vrouwen en **{men}** voor de mannen. Met die thresholds geeft je `report` op de test set:",
        "after": "De classifier heeft het nu goed voor **{right}** van de test set, tegen {before} met één threshold. De impact ratio van de vrouwen ging van {ratio_before} naar {ratio_after}: gelijke recall is geen gelijk deel met een ja, omdat in de tabel een groter deel van de mannen dan van de vrouwen meer verdient. Per afkomst: `White` {white_before} → {white_after}, `Black` {black_before} → {black_after}. Een threshold per geslacht is gekozen voor geslacht; een audit meet opnieuw voor elke groep die de beslissing kan schaden.",
        "broken": {
            "model": "*Je classifier in `work/projects/audit/model.py` stopt met een fout, of zijn `scores` geeft niet per persoon één kans van 0 tot 1. Heb je het bestand veranderd? Maak je wijziging ongedaan, of verwijder het bestand en open de les opnieuw: de course zet dan zijn eigen versie terug.*",
            "report": "*Je `report` in `work/projects/audit/report.py` stopt met een fout, of geeft niet de tabel met zijn vijf kolommen. Heb je het bestand veranderd? Verwijder het, en check oefening 1 opnieuw: de check bewaart je code dan weer.*",
            "calibration": "*Je `calibration` in `work/projects/audit/calibration.py` stopt met een fout, of geeft geen `mean score` en `share rich`. Heb je het bestand veranderd? Verwijder het, en check oefening 2 opnieuw.*",
            "fix": "*Je `threshold_for` in `work/projects/audit/fix.py` stopt met een fout, of geeft geen threshold van 0 tot 1. Heb je het bestand veranderd? Verwijder het, en check oefening 3 opnieuw.*",
        },
    },
}


def broken(name: str, lang: str) -> str:
    return _PARTS[lang]["broken"][name]


def _ratio(df: pd.DataFrame, group: str) -> float | None:
    return _number(df.loc[group, "impact ratio"]) if group in df.index else None


def audit_parts(audit: dict, lang: str) -> list:
    """What the lesson shows of the learner's audit, as (kind, text) pairs: "md" for Markdown, "html" for a picture."""
    w = _PARTS[lang]
    sign = "," if lang == "nl" else "."
    num = lambda x: "–" if x is None else number(x, lang)   # noqa: E731
    parts = [("md", w["sex"].format(right=number(audit["right"], lang, 3)) + "\n\n" + markdown(audit["sex"], lang)),
             ("md", w["race"] + "\n\n" + markdown(audit["race"], lang))]
    race = audit["race"]
    low = [f"`{g}` ({number(r, lang)})" for g in race.index
           if (r := _number(race.loc[g, "impact ratio"])) is not None and r < 0.8]
    rates = race["yes rate"].astype(float)
    top = str(rates.idxmax()) if rates.notna().any() else "–"
    people = grouped(int(race.loc[top, "people"]), lang) if top in race.index else "–"
    if low:
        parts.append(("md", w["flagged"].format(groups=", ".join(low), top=f"`{top}`", people=people)))
    else:
        parts.append(("md", w["none"].format(top=f"`{top}`", people=people)))
    cal = audit["calibration"]
    if cal is None:
        parts.append(("md", w["cal_wait"]))
    elif isinstance(cal, str):
        parts.append(("md", w["broken"]["calibration"]))
    else:
        parts.append(("html", calibration_figure(cal, _CAL_WORDS[lang] | {"key": w["cal"]}, lang)))
    fix = audit["fix"]
    if fix is None:
        parts.append(("md", w["fix_wait"]))
    elif isinstance(fix, str):
        parts.append(("md", w["broken"]["fix"]))
    else:
        t = fix["thresholds"]
        parts.append(("md", w["fix"].format(women=_short(t["Female"], sign), men=_short(t["Male"], sign)) + "\n\n"
                      + markdown(fix["sex"], lang)))
        parts.append(("html", fix_picture(audit["sex"], fix["sex"], lang)))
        parts.append(("md", w["after"].format(
            right=number(fix["right"], lang, 3), before=number(audit["right"], lang, 3),
            ratio_before=num(_ratio(audit["sex"], "Female")), ratio_after=num(_ratio(fix["sex"], "Female")),
            white_before=num(_ratio(race, "White")), white_after=num(_ratio(fix["race"], "White")),
            black_before=num(_ratio(race, "Black")), black_after=num(_ratio(fix["race"], "Black")))))
    return parts
