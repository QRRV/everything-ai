# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-linkage-attacks: a made-up town of 30,000 adults (data/course-town-*.csv.gz, made by
data/tools/make_course_town.py): its voter list (VOTERS), the released table of a made-up health insurer (VISITS),
the whole town as only the course knows it (PEOPLE), the columns they share (QI), and two small tables of the figure
in The idea (FEW_VISITS, FEW_VOTERS). The three functions the learner writes: share_alone (the share of the rows that
are the only one with their values in some columns), linked (how many rows of a released table match exactly one
voter) and alone (the chance that none of n - 1 others has your value); with the pictures of what the learner's
functions give, and the figure of the small tables."""
import html
import re

import pandas as pd

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake

DATA = paths.ROOT / "data"


class NamedList(list):
    """A list the check calls by its name, such as QI, so that a message stays short."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name


class Table:
    """A table the check calls by the name the lesson gives it, such as VOTERS, so that a message stays short. It is
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


def share_alone(table, columns):
    sizes = table.groupby(columns).size()
    return (sizes == 1).sum() / len(table)


def linked(released, voters, columns):
    both = pd.merge(released, voters, on=columns)
    per_visit = both.groupby("visit").size()
    return (per_visit == 1).sum()


def alone(n, values):
    miss = (values - 1) / values
    return miss ** (n - 1)


def _read(name: str):
    return lambda: pd.read_csv(DATA / f"course-town-{name}.csv.gz")


VOTERS = Table(_read("voters"), "VOTERS")
VISITS = Table(_read("visits"), "VISITS")
PEOPLE = Table(_read("people"), "PEOPLE")
QI = NamedList(["zip", "birth", "sex"], "QI")

# the two small tables of the figure in The idea: three visits, five voters
_FEW_VISITS = [("V1", 30101, "1961-04-12", "female", "asthma"), ("V2", 30102, "1975-09-03", "male", "flu"),
               ("V3", 30103, "1988-01-30", "female", "migraine")]
_FEW_VOTERS = [("Robin T. Ashford", 30101, "1961-04-12", "female"), ("Sam K. Oakley", 30102, "1975-09-03", "male"),
               ("Jo P. Millby", 30102, "1975-09-03", "male"), ("Kai L. Fernwood", 30101, "1961-04-12", "male"),
               ("Lee M. Stoneham", 30103, "1988-01-31", "female")]
FEW_VISITS = Table(lambda: pd.DataFrame(_FEW_VISITS, columns=["visit", "zip", "birth", "sex", "diagnosis"]),
                   "FEW_VISITS")
FEW_VOTERS = Table(lambda: pd.DataFrame(_FEW_VOTERS, columns=["name", "zip", "birth", "sex"]), "FEW_VOTERS")


def coarse(df: pd.DataFrame, cut: int) -> pd.DataFrame:
    """The table with each birth date cut to its first `cut` characters: 7 keeps year and month, 4 the year."""
    df = df.copy()
    df["birth"] = df["birth"].str[:cut]
    return df


_OUTCOMES: dict = {}


def outcome(columns=QI, cut: int = 10) -> dict:
    """What linking VISITS to VOTERS on `columns` gives, with the birth date cut to `cut` characters, and how many of
    the names are right according to PEOPLE."""
    key = (tuple(columns), cut)
    if key not in _OUTCOMES:
        _OUTCOMES[key] = _outcome(list(columns), cut)
    return _OUTCOMES[key]


def _outcome(columns: list, cut: int) -> dict:
    visits, voters = coarse(VISITS.df, cut), coarse(VOTERS.df, cut)
    both = pd.merge(visits, voters, on=columns)
    per_visit = both.groupby("visit").size()
    one = both[both["visit"].isin(per_visit[per_visit == 1].index)]
    truth = PEOPLE.df.dropna(subset=["visit"]).set_index("visit")["name"]
    right = int((one["name"] == one["visit"].map(truth)).sum())
    return {"rows": len(both), "matched": len(per_visit), "linked": len(one), "right": right,
            "wrong": len(one) - right, "more": int((per_visit > 1).sum()), "none": len(visits) - len(per_visit)}


# --- the pictures ---------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_BOX = "padding:0.25rem 0.5rem;margin:0.2rem 0;border-radius:4px;"
_PLAIN = "border:1px solid var(--ea-line);color:var(--ea-ink)"
_WRONG = "border:1.5px dashed var(--ea-loss);color:var(--ea-loss-text)"
_HEAVY = "border:2px solid var(--ea-ink);color:var(--ea-ink);font-weight:600"


def _code(text) -> str:
    return f'<code style="font-family:var(--ea-font-code);font-size:0.8rem;white-space:nowrap">{html.escape(str(text))}</code>'


def _row(cells: list, style: str, widths: list) -> str:
    parts = "".join(f'<div style="flex:{w};min-width:0;overflow-wrap:anywhere">{c}</div>' for c, w in zip(cells, widths))
    return f'<div style="display:flex;gap:0.6rem;align-items:baseline;{_BOX}{style}">{parts}</div>'


def _head(cells: list, widths: list) -> str:
    parts = "".join(f'<div style="flex:{w};min-width:0;overflow-wrap:anywhere">{html.escape(c)}</div>'
                    for c, w in zip(cells, widths))
    return (f'<div style="display:flex;gap:0.6rem;font-weight:600;color:var(--ea-ink);padding:0 0.5rem;'
            f'border-bottom:2px solid var(--ea-ink-3);margin-bottom:0.2rem">{parts}</div>')


def _table(head: list, rows: list, widths: list) -> str:
    return (f'<div style="font-family:var(--ea-font-ui);font-size:0.85rem;max-width:46rem">'
            f'{_head(head, widths)}{"".join(rows)}</div>')


def _grouped(n: int, lang: str) -> str:
    """A whole number as the lesson's language writes it: 19,656 in English, 19.656 in Dutch (a dot from five digits)."""
    if lang == "nl":
        return f"{n:,}".replace(",", ".") if n >= 10000 else str(n)
    return f"{n:,}"


def _percent(v: float, lang: str, digits: int = 1) -> str:
    """A share from 0 to 1 as a percentage, with a decimal comma in Dutch: 86.0% or 86,0%.

    A number outside 0 to 1 is no share, so it is shown as it is: 8805, or 172,53 in Dutch."""
    text = f"{100 * v:.{digits}f}%" if 0 <= v <= 1 else f"{v:g}".replace("-", "−")
    return text.replace(".", ",") if lang == "nl" else text


def _number(v):
    """A plain number of what a function gave back, or None."""
    try:
        if isinstance(v, bool):
            return None
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return None


def _fn(seen):
    mine = seen.mine
    if mine is None:
        return None
    fn = mine.names.get(seen.exercise.function)
    return fn if callable(fn) else None


def _call(fn, *args):
    import copy
    from everything_ai import check as checking
    return checking.with_time_limit(lambda: fn(*(copy.deepcopy(a) for a in args)))


def _same(a, b) -> bool:
    return a is not None and abs(a - b) <= 1e-9 * max(1.0, abs(b))


_LINK_WORDS = {
    "en": {"title": "Two tables, one key",
           "visits": "released: no names", "voters": "voter list: names",
           "head_visits": ["visit", "ZIP", "birth", "sex", "diagnosis"], "head_voters": ["name", "ZIP", "birth", "sex"],
           "result": "merged on ZIP code, birth date and sex",
           "one": "one voter fits: the visit gets the name {name}", "two": "two voters fit: {a} or {b}",
           "none": "no voter fits: the visit drops out of the merge",
           "key": "Three made-up visits and five made-up voters. A voter fits a visit when ZIP code, birth date and sex "
                  "are all the same. Kai L. Fernwood has the ZIP code and birth date of V1, but another sex; Lee M. "
                  "Stoneham was born one day after V3."},
    "nl": {"title": "Twee tabellen, één sleutel",
           "visits": "vrijgegeven: geen namen", "voters": "kiezerslijst: namen",
           "head_visits": ["bezoek", "postcode", "geboren", "geslacht", "diagnose"],
           "head_voters": ["naam", "postcode", "geboren", "geslacht"],
           "result": "gekoppeld op postcode, geboortedatum en geslacht",
           "one": "één kiezer past: het bezoek krijgt de naam {name}", "two": "twee kiezers passen: {a} of {b}",
           "none": "geen kiezer past: het bezoek valt weg bij de merge",
           "key": "Drie verzonnen bezoeken en vijf verzonnen kiezers. Een kiezer past bij een bezoek als postcode, "
                  "geboortedatum en geslacht allemaal gelijk zijn. Kai L. Fernwood heeft de postcode en geboortedatum "
                  "van V1, maar een ander geslacht; Lee M. Stoneham is een dag na V3 geboren."},
}


def link_picture(lang: str) -> str:
    """The figure of The idea: the small released table, the small voter list, and what merging them gives."""
    w = _LINK_WORDS[lang]
    vw, ow = [2, 3, 5, 3, 4], [5, 3, 5, 3]
    visits = _table(w["head_visits"], [_row([_code(v), str(z), _code(b), html.escape(s), html.escape(d)], _PLAIN, vw)
                                       for v, z, b, s, d in _FEW_VISITS], vw)
    voters = _table(w["head_voters"], [_row([html.escape(n), str(z), _code(b), html.escape(s)], _PLAIN, ow)
                                       for n, z, b, s in _FEW_VOTERS], ow)
    lines = []
    for v, z, b, s, d in _FEW_VISITS:
        fits = [n for n, z2, b2, s2 in _FEW_VOTERS if (z2, b2, s2) == (z, b, s)]
        if len(fits) == 1:
            text, style = w["one"].format(name=fits[0]), _HEAVY
        elif len(fits) == 2:
            text, style = w["two"].format(a=fits[0], b=fits[1]), _PLAIN
        else:
            text, style = w["none"], _PLAIN
        lines.append(_row([_code(v), html.escape(text)], style, [1, 8]))
    cols = (f'<div style="display:grid;gap:0.6rem;font-family:var(--ea-font-ui);font-size:0.83rem;max-width:34rem;'
            f'color:var(--ea-ink)">'
            f'<div><div style="font-weight:600">{html.escape(w["visits"])}</div>{visits}</div>'
            f'<div><div style="font-weight:600">{html.escape(w["voters"])}</div>{voters}</div></div>')
    result = (f'<div style="font-family:var(--ea-font-ui);font-size:0.83rem;max-width:46rem;color:var(--ea-ink);'
              f'margin-top:0.6rem"><div style="font-weight:600">↓ {html.escape(w["result"])}</div>{"".join(lines)}</div>')
    return yours.figure(w["title"], cols + result + f'<div style="{_UI}">{html.escape(w["key"])}</div>')


# the column sets the picture of share_alone goes through
_SETS = [["birth"], ["birth", "sex"], ["zip", "birth"], ["zip", "birth", "sex"]]

_SHARE_WORDS = {
    "en": {"title": "Your share_alone on the voter list and on the whole town",
           "head": ["columns", "VOTERS", "PEOPLE"],
           "key": "VOTERS holds {v} of the {p} people of the town. For each set of columns, a bar shows what your "
                  "share_alone gives on the voter list, and one on the whole town.",
           "said": "On ZIP code, birth date and sex, your share_alone gives {a} on the voter list and {b} on the whole "
                   "town.",
           "gap": "Someone who is alone on the list is not always alone in the town: the list leaves out about a third of it.",
           "wrong": "A dashed frame marks where it differs from the course's share_alone."},
    "nl": {"title": "Jouw share_alone op de kiezerslijst en op het hele stadje",
           "head": ["kolommen", "VOTERS", "PEOPLE"],
           "key": "VOTERS bevat {v} van de {p} mensen van het stadje. Voor elke set kolommen toont een balk wat jouw "
                  "share_alone geeft op de kiezerslijst, en een op het hele stadje.",
           "said": "Op postcode, geboortedatum en geslacht geeft jouw share_alone {a} op de kiezerslijst en {b} op "
                   "het hele stadje.",
           "gap": "Wie op de lijst alleen staat, staat niet altijd alleen in het stadje: de lijst laat ongeveer "
                  "een derde ervan weg.",
           "wrong": "Een gestippeld kader wijst aan waar hij verschilt van de share_alone van de course."},
}


def _bar(v, lang: str, role: str) -> str:
    if v is None:
        return "–"
    width = max(0.0, min(1.0, v)) * 100
    return (f'<div style="display:flex;align-items:center;gap:0.4rem"><div style="flex:1;height:0.7rem;'
            f'background:var(--ea-line);border-radius:2px"><div style="width:{width:.1f}%;height:100%;'
            f'background:var(--ea-{role});border-radius:2px"></div></div>'
            f'<span style="min-width:3.4rem;text-align:right">{_percent(v, lang)}</span></div>')


def share_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _SHARE_WORDS[seen.lang]
    rows, wrong_any, last = [], False, (None, None)
    for columns in _SETS:
        got = []
        for table in (VOTERS, PEOPLE):
            try:
                got.append(_number(_call(fn, table, list(columns))))
            except Exception:
                got.append(None)
        want = [share_alone(VOTERS.df, columns), share_alone(PEOPLE.df, columns)]
        wrong = not all(_same(g, x) for g, x in zip(got, want))
        wrong_any |= wrong
        rows.append(_row([_code(", ".join(columns)), _bar(got[0], seen.lang, "ink-2"), _bar(got[1], seen.lang, "ink-3")],
                         _WRONG if wrong else _PLAIN, [3, 4, 4]))
        last = got
    note = w["key"].format(v=_grouped(len(VOTERS.df), seen.lang), p=_grouped(len(PEOPLE.df), seen.lang))
    if all(x is not None for x in last):
        note += " " + w["said"].format(a=_percent(last[0], seen.lang), b=_percent(last[1], seen.lang))
    note += " " + (w["wrong"] if wrong_any else w["gap"])
    return yours.figure(w["title"], _table(w["head"], rows, [3, 4, 4]) + f'<div style="{_UI}">{html.escape(note)}</div>')


_LINKED_WORDS = {
    "en": {"title": "Your linked on VISITS and VOTERS",
           "head": ["birth", "names", "right", "wrong"],
           "cuts": {10: "full date", 7: "year and month", 4: "year"},
           "key": "Each row links the 1,000 visits to the voter list on ZIP code, sex and the birth date, cut down "
                  "more each time, in both tables. Names: what your linked gives. Right and wrong: what PEOPLE, the "
                  "whole town, says about those names.",
           "said": "With the full date, your linked gives {n} names, and {wrong} of them are wrong: for each of them, "
                   "the visit's own person is not on the voter list, and exactly one voter has the same ZIP code, "
                   "birth date and sex. With year and month, only {n7} visits get one name, {w7} of them wrong; with "
                   "only the year, none.",
           "wrong": "A dashed frame marks where it differs from the course's linked."},
    "nl": {"title": "Jouw linked op VISITS en VOTERS",
           "head": ["geboren", "namen", "goed", "fout"],
           "cuts": {10: "hele datum", 7: "jaar en maand", 4: "jaar"},
           "key": "Elke rij koppelt de 1000 bezoeken aan de kiezerslijst op postcode, geslacht en de geboortedatum, "
                  "elke keer grover, in beide tabellen. Namen: wat jouw linked geeft. Goed en fout: wat PEOPLE, het "
                  "hele stadje, over die namen zegt.",
           "said": "Met de hele datum geeft jouw linked {n} namen, en daarvan zijn er {wrong} fout: bij elk ervan "
                   "staat de persoon van het bezoek zelf niet op de kiezerslijst, en heeft precies één kiezer dezelfde "
                   "postcode, geboortedatum en hetzelfde geslacht. Met jaar en maand krijgen nog maar {n7} bezoeken "
                   "één naam, waarvan {w7} fout; met alleen het jaar geen enkel.",
           "wrong": "Een gestippeld kader wijst aan waar hij verschilt van de linked van de course."},
}


def linked_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _LINKED_WORDS[seen.lang]
    rows, wrong_any = [], False
    for cut in (10, 7, 4):
        tables = [Table(lambda t=t, c=cut: coarse(t.df, c), t.name) for t in (VISITS, VOTERS)]
        try:
            got = _number(_call(fn, tables[0], tables[1], QI))
        except Exception:
            got = None
        want = outcome(QI, cut)
        wrong = not _same(got, want["linked"])
        wrong_any |= wrong
        shown = (_grouped(int(got), seen.lang) if got is not None and got == int(got) else
                 "–" if got is None else html.escape(str(got)[:20]))
        right = "–" if wrong else _grouped(want["right"], seen.lang)
        bad = ("–" if wrong else
               f'<span style="color:var(--ea-loss-text);font-weight:600">{_grouped(want["wrong"], seen.lang)}</span>'
               if want["wrong"] else "0")
        rows.append(_row([html.escape(w["cuts"][cut]), shown, right, bad], _WRONG if wrong else _PLAIN, [4, 2, 2, 2]))
    full = outcome(QI, 10)
    note = w["key"]
    if not wrong_any:
        month, year = outcome(QI, 7), outcome(QI, 4)
        assert year["linked"] == 0   # the text says so in words
        note += " " + w["said"].format(n=full["linked"], wrong=full["wrong"], n7=month["linked"], w7=month["wrong"])
    else:
        note += " " + w["wrong"]
    return yours.figure(w["title"], _table(w["head"], rows, [4, 2, 2, 2]) + f'<div style="{_UI}">{html.escape(note)}</div>')


_ALONE_WORDS = {
    "en": {"title": "Your alone(n, 365): the chance that nobody else has your birthday",
           "x": "n: people of the same sex, age and ZIP code", "y": "chance",
           "said": "For the 174 men aged 50 in Weld's ZIP code, your alone gives {a}; for 150, the fewest that Barth-Jones finds likely, {b}. With only the "
                   "month of birth, 12 values, alone(174, 12) gives {m}."},
    "nl": {"title": "Jouw alone(n, 365): de kans dat niemand anders op jouw verjaardag jarig is",
           "x": "n: mensen met hetzelfde geslacht, dezelfde leeftijd en postcode", "y": "kans",
           "said": "Voor de 174 mannen van 50 in de postcode van Weld geeft jouw alone {a}; voor 150, het kleinste aantal dat Barth-Jones aannemelijk vindt, {b}. Met alleen "
                   "de geboortemaand, 12 waarden, geeft alone(174, 12) {m}."},
}


def _chance(v, lang: str) -> str:
    if v is None:
        return "–"
    if 0 < abs(v) < 0.001:
        text = f"{100 * v:.5f}%"
        return text.replace(".", ",") if lang == "nl" else text
    return _percent(v, lang, 0)


def alone_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _ALONE_WORDS[seen.lang]
    curve = yours.curve("alone", range(1, 401), at=lambda n: (n, 365), role="ink",
                        title=yours.Text(en=_ALONE_WORDS["en"]["title"], nl=_ALONE_WORDS["nl"]["title"]),
                        xlabel=yours.Text(en=_ALONE_WORDS["en"]["x"], nl=_ALONE_WORDS["nl"]["x"]),
                        ylabel=yours.Text(en=_ALONE_WORDS["en"]["y"], nl=_ALONE_WORDS["nl"]["y"]), y=(0, 1))(seen)
    values = []
    for args in ((174, 365), (150, 365), (174, 12)):
        try:
            values.append(_number(_call(fn, *args)))
        except Exception:
            values.append(None)
    note = w["said"].format(a=_chance(values[0], seen.lang), b=_chance(values[1], seen.lang),
                            m=_chance(values[2], seen.lang))
    return curve + f'<div style="{_UI}">{html.escape(note)}</div>'


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["share_alone", "linked", "alone", "voter-list", "estimate"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# exercise 1: share_alone
_S_HEAD = ("def share_alone(table, columns):\n"
           "    # the share of the rows that are the only one with their values in columns\n")
_S_START = _S_HEAD + "    sizes = ...\n    return ...\n"
_S_SIZES = "    sizes = table.groupby(columns).size()\n"
_S = _S_HEAD + _S_SIZES + "    return (sizes == 1).sum() / len(table)\n"
_S_COUNT = _S_HEAD + _S_SIZES + "    return (sizes == 1).sum()\n"
_S_GROUPS = _S_HEAD + _S_SIZES + "    return (sizes == 1).sum() / len(sizes)\n"
_S_DISTINCT = _S_HEAD + _S_SIZES + "    return len(sizes) / len(table)\n"
_S_MIN = _S_HEAD + _S_SIZES + "    return sizes.min()\n"
_S_PERCENT = _S_HEAD + _S_SIZES + "    return 100 * (sizes == 1).sum() / len(table)\n"

# exercise 2: linked
_K_HEAD = ("import pandas as pd\n\n\n"
           "def linked(released, voters, columns):\n"
           "    # how many rows of released match exactly one row of voters on columns\n")
_K_START = _K_HEAD + "    both = ...\n    per_visit = ...\n    return ...\n"
_K_BOTH = "    both = pd.merge(released, voters, on=columns)\n"
_K_PER = "    per_visit = both.groupby(\"visit\").size()\n"
_K = _K_HEAD + _K_BOTH + _K_PER + "    return (per_visit == 1).sum()\n"
_K_ROWS = _K_HEAD + _K_BOTH + _K_PER + "    return len(both)\n"
_K_MATCHED = _K_HEAD + _K_BOTH + _K_PER + "    return len(per_visit)\n"
_K_MORE = _K_HEAD + _K_BOTH + _K_PER + "    return (per_visit > 1).sum()\n"
_K_LEFT = (_K_HEAD + "    both = pd.merge(released, voters, on=columns, how=\"left\")\n" + _K_PER
           + "    return (per_visit == 1).sum()\n")
_K_SIZES = _K_HEAD + _K_BOTH + _K_PER + "    return per_visit\n"
_K_COMBO = _K_HEAD + _K_BOTH + "    per_visit = both.groupby(columns).size()\n    return (per_visit == 1).sum()\n"
_K_SHARE = _K_HEAD + _K_BOTH + _K_PER + "    return (per_visit == 1).sum() / len(released)\n"

# exercise 3: alone
_A_HEAD = ("def alone(n, values):\n"
           "    # the chance that none of the other n - 1 people has your value, when all values are as likely\n")
_A_START = _A_HEAD + "    miss = ...\n    return ...\n"
_A_MISS = "    miss = (values - 1) / values\n"
_A = _A_HEAD + _A_MISS + "    return miss ** (n - 1)\n"
_A_ALL = _A_HEAD + _A_MISS + "    return miss ** n\n"
_A_SHARED = _A_HEAD + _A_MISS + "    return 1 - miss ** (n - 1)\n"
_A_TIMES = _A_HEAD + _A_MISS + "    return miss * (n - 1)\n"
_A_HIT = _A_HEAD + "    miss = 1 / values\n    return miss ** (n - 1)\n"
_A_ONE = _A_HEAD + _A_MISS + "    return miss\n"
_A_COUNT = _A_HEAD + "    miss = values - 1\n    return miss ** (n - 1)\n"
_A_ORDER = _A_HEAD + _A_MISS + "    return miss ** n - 1\n"

EXERCISES = {
    "share_alone": FunctionBox(
        name="share_alone",
        picture=share_picture,
        function="share_alone",
        start=_S_START,
        solution=_S,
        visible=[(VOTERS, QI)],
        hidden=[(VOTERS, ["birth"]), (VOTERS, ["birth", "sex"]), (PEOPLE, QI), (FEW_VOTERS, QI),
                (FEW_VOTERS, ["zip"]), (FEW_VOTERS, ["birth", "sex"])],
        first=[_DOTS],
        mistakes=[
            Mistake("count", wrong=_S_COUNT,
                    en="`{call}` gives {got}: the number of rows that are alone. The task asks for their share, from 0 to 1: divide by the number of rows, `len(table)`.",
                    nl="`{call}` geeft {got}: het aantal rijen dat alleen staat. De opdracht vraagt om hun aandeel, van 0 tot 1: deel door het aantal rijen, `len(table)`."),
            Mistake("groups", wrong=_S_GROUPS,
                    en="`{call}` gives {got}, and it should give {want}. That is the share of the groups that have one row. The task asks for the share of the rows: divide the count by `len(table)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Dat is het aandeel van de groepen met één rij. De opdracht vraagt om het aandeel van de rijen: deel de telling door `len(table)`."),
            Mistake("distinct", wrong=_S_DISTINCT,
                    en="`{call}` gives {got}, and it should give {want}. That is the number of groups divided by the number of rows. Count only the groups of size 1: `(sizes == 1).sum()`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Dat is het aantal groepen gedeeld door het aantal rijen. Tel alleen de groepen met grootte 1: `(sizes == 1).sum()`."),
            Mistake("min", wrong=_S_MIN,
                    en="`{call}` gives {got}: the size of the smallest group, the k of the previous lesson. Here you count the groups of size 1, and divide by the number of rows.",
                    nl="`{call}` geeft {got}: de grootte van de kleinste groep, de k van de vorige les. Hier tel je de groepen met grootte 1, en deel je door het aantal rijen."),
            Mistake("percent", wrong=_S_PERCENT,
                    en="`{call}` gives {got}, a percentage. The task asks for a share from 0 to 1: leave out the 100.",
                    nl="`{call}` geeft {got}, een percentage. De opdracht vraagt om een aandeel van 0 tot 1: laat de 100 weg."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` after `sizes =` with the size of each group of rows with the same values in `columns`, and the `...` after `return` with the share of the rows whose group has one row.",
                      "nl": "Vervang de `...` na `sizes =` door de grootte van elke groep rijen met dezelfde waarden in `columns`, en de `...` na `return` door het aandeel van de rijen waarvan de groep één rij heeft."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Count the groups with exactly one row, and divide by the number of rows of `table`.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Tel de groepen met precies één rij, en deel door het aantal rijen van `table`."},
            "almost": {"en": "Right for `share_alone(VOTERS, QI)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `table` and the `columns` it gets?",
                       "nl": "Goed voor `share_alone(VOTERS, QI)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `table` en de `columns` die hij krijgt?"},
            "correct": {"en": "Your `share_alone` is right in all {n} cases. The picture runs it on the voter list and on the whole town.",
                        "nl": "Je `share_alone` klopt in alle {n} gevallen. Het plaatje laat hem lopen op de kiezerslijst en op het hele stadje."},
        },
    ),
    "linked": FunctionBox(
        name="linked",
        picture=linked_picture,
        function="linked",
        start=_K_START,
        solution=_K,
        visible=[(VISITS, VOTERS, QI)],
        hidden=[(VISITS, VOTERS, ["zip", "birth"]), (VISITS, VOTERS, ["birth", "sex"]), (FEW_VISITS, FEW_VOTERS, QI),
                (FEW_VISITS, FEW_VOTERS, ["zip", "birth"]), (FEW_VISITS, FEW_VOTERS, ["zip", "sex"])],
        first=[_DOTS],
        mistakes=[
            Mistake("rows", wrong=_K_ROWS,
                    en="`{call}` gives {got}, and it should give {want}. That is the number of rows of the merge: a visit that fits two voters comes out twice. Count the visits whose number of rows is exactly 1.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Dat is het aantal rijen van de merge: een bezoek waar twee kiezers bij passen, komt er twee keer uit. Tel de bezoeken met precies 1 rij."),
            Mistake("matched", wrong=_K_MATCHED,
                    en="`{call}` gives {got}, and it should give {want}. That is the number of visits that fit at least one voter. A visit gets a name only when exactly one voter fits: `(per_visit == 1).sum()`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Dat is het aantal bezoeken waar minstens één kiezer bij past. Een bezoek krijgt alleen een naam als er precies één kiezer bij past: `(per_visit == 1).sum()`."),
            Mistake("more", wrong=_K_MORE,
                    en="`{call}` gives {got}, and it should give {want}. That is the number of visits that fit more than one voter: those keep several candidates. Count the visits with exactly one.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Dat is het aantal bezoeken waar meer dan één kiezer bij past: die houden meerdere kandidaten. Tel de bezoeken met precies één."),
            Mistake("left", wrong=_K_LEFT,
                    en="`{call}` gives {got}, and it should give {want}. Your merge keeps a visit that fits no voter, as one row without a name, as `how=\"left\"` does, so that visit counts as one match. Leave `how` out: then a visit without a match drops out.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je merge houdt een bezoek waar geen kiezer bij past, als één rij zonder naam, zoals `how=\"left\"` doet, dus telt dat bezoek als één match. Laat `how` weg: dan valt een bezoek zonder match weg."),
            Mistake("combination", wrong=_K_COMBO,
                    en="`{call}` gives {got}, and it should give {want}. You count per combination of values in `columns`, so two visits with the same values fall into one group. Count the rows of each visit instead.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je telt per combinatie van waarden in `columns`, dus twee bezoeken met dezelfde waarden komen in één groep. Tel in plaats daarvan de rijen van elk bezoek."),
            Mistake("share", wrong=_K_SHARE,
                    en="`{call}` gives {got}: the share of the visits that get a name. The task asks for the number of visits: leave out the division.",
                    nl="`{call}` geeft {got}: het aandeel van de bezoeken dat een naam krijgt. De opdracht vraagt om het aantal bezoeken: laat het delen weg."),
            Mistake("sizes", wrong=_K_SIZES,
                    en="`{call}` gives the number of matching voters of every visit, one number per visit. The task asks for one number: how many of them are exactly 1.",
                    nl="`{call}` geeft het aantal passende kiezers van elk bezoek, één getal per bezoek. De opdracht vraagt om één getal: hoeveel ervan precies 1 zijn."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the three `...`: the merge of `released` and `voters` on `columns`, the number of rows per visit in that merge, and how many visits have exactly one.",
                      "nl": "Vervang de drie `...`: de merge van `released` en `voters` op `columns`, het aantal rijen per bezoek in die merge, en hoeveel bezoeken er precies één hebben."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Merge on `columns`, count the rows of each visit in the merge, and count the visits with exactly one row.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Doe een merge op `columns`, tel de rijen van elk bezoek in de merge, en tel de bezoeken met precies één rij."},
            "almost": {"en": "Right for `linked(VISITS, VOTERS, QI)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `released`, the `voters` and the `columns` it gets?",
                       "nl": "Goed voor `linked(VISITS, VOTERS, QI)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `released`, de `voters` en de `columns` die hij krijgt?"},
            "correct": {"en": "Your `linked` is right in all {n} cases. The picture checks its names against the whole town, and coarsens the birth date.",
                        "nl": "Je `linked` klopt in alle {n} gevallen. Het plaatje legt de namen naast het hele stadje, en maakt de geboortedatum grover."},
        },
    ),
    "alone": FunctionBox(
        name="alone",
        picture=alone_picture,
        function="alone",
        start=_A_START,
        solution=_A,
        visible=[(174, 365)],
        hidden=[(150, 365), (1, 365), (2, 365), (2, 2), (10, 12), (54805, 365000)],
        first=[_DOTS],
        mistakes=[
            Mistake("all", wrong=_A_ALL,
                    en="`{call}` gives {got}, and it should give {want}. Your power counts you as well: of the n people, only the other n − 1 have to miss your value, so the power is `n - 1`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je macht telt jezelf mee: van de n mensen hoeven alleen de andere n − 1 jouw waarde te missen, dus de macht is `n - 1`."),
            Mistake("shared", wrong=_A_SHARED,
                    en="`{call}` gives {got}, and it should give {want}. That is the chance that at least one other person has your value. The task asks for the chance that nobody else has it: `miss ** (n - 1)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Dat is de kans dat minstens één ander jouw waarde heeft. De opdracht vraagt om de kans dat niemand anders hem heeft: `miss ** (n - 1)`."),
            Mistake("times", wrong=_A_TIMES,
                    en="`{call}` gives {got}, and it should give {want}. Chances that must all come true are multiplied with each other: `miss` times itself, n − 1 times, is a power, `**`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Kansen die allemaal moeten uitkomen, vermenigvuldig je met elkaar: `miss` n − 1 keer met zichzelf is een macht, `**`."),
            Mistake("hit", wrong=_A_HIT,
                    en="`{call}` gives {got}, and it should give {want}. `1 / values` is the chance that one other person has your value. `miss` is the chance that they have another one: `(values - 1) / values`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. `1 / values` is de kans dat één ander jouw waarde heeft. `miss` is de kans dat die een andere heeft: `(values - 1) / values`."),
            Mistake("one", wrong=_A_ONE,
                    en="`{call}` gives {got}, and it should give {want}. That is the chance for one other person. All n − 1 others have to miss your value: `miss ** (n - 1)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Dat is de kans voor één ander. Alle n − 1 anderen moeten jouw waarde missen: `miss ** (n - 1)`."),
            Mistake("count", wrong=_A_COUNT,
                    en="Your `miss` is a count of values, not a chance: divide it by `values`.",
                    nl="Je `miss` is een aantal waarden, geen kans: deel het door `values`."),
            Mistake("order", wrong=_A_ORDER,
                    en="`{call}` gives {got}, and it should give {want}. Python works out `**` before `-`, so `miss ** n - 1` is `(miss ** n) - 1`. Put the power in brackets: `miss ** (n - 1)`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Python rekent `**` uit vóór `-`, dus `miss ** n - 1` is `(miss ** n) - 1`. Zet de macht tussen haakjes: `miss ** (n - 1)`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the first `...` with the chance that one other person has another value than yours, and the second with the chance that all n − 1 others do.",
                      "nl": "Vervang de eerste `...` door de kans dat één ander een andere waarde heeft dan jij, en de tweede door de kans dat alle n − 1 anderen dat hebben."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. One other person misses your value with chance `(values - 1) / values`; all n − 1 others: that chance to the power n − 1.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Eén ander mist jouw waarde met kans `(values - 1) / values`; alle n − 1 anderen: die kans tot de macht n − 1."},
            "almost": {"en": "Right for `alone(174, 365)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `n` and the `values` it gets?",
                       "nl": "Goed voor `alone(174, 365)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `n` en de `values` die hij krijgt?"},
            "correct": {"en": "Your `alone` is right in all {n} cases. The picture draws it for n from 1 to 400.",
                        "nl": "Je `alone` klopt in alle {n} gevallen. Het plaatje tekent hem voor n van 1 tot 400."},
        },
    ),
}

CASES = [
    ("share_alone", _S_START, "empty", None),
    ("share_alone", _S_HEAD + _S_SIZES + "    return ...\n", "wrong", "dots-left"),
    ("share_alone", _S, "correct", None),
    ("share_alone", _S_HEAD + _S_SIZES + "    alone = sizes[sizes == 1]\n    return len(alone) / len(table)\n", "correct", None),
    ("share_alone", _S_COUNT, "wrong", "count"),
    ("share_alone", _S_GROUPS, "wrong", "groups"),
    ("share_alone", _S_HEAD + _S_SIZES + "    return (sizes == 1).mean()\n", "wrong", "groups"),
    ("share_alone", _S_DISTINCT, "wrong", "distinct"),
    ("share_alone", _S_MIN, "wrong", "min"),
    ("share_alone", _S_PERCENT, "wrong", "percent"),
    ("share_alone", _S_HEAD + _S_SIZES + "    return (sizes == 2).sum() / len(table)\n", "wrong", None),
    ("linked", _K_START, "empty", None),
    ("linked", _K, "correct", None),
    ("linked", _K_HEAD + _K_BOTH + "    per_visit = both[\"visit\"].value_counts()\n    return (per_visit == 1).sum()\n",
     "correct", None),
    ("linked", _K_ROWS, "wrong", "rows"),
    ("linked", _K_MATCHED, "wrong", "matched"),
    ("linked", _K_HEAD + _K_BOTH + _K_PER + "    return both[\"visit\"].nunique()\n", "wrong", "matched"),
    ("linked", _K_MORE, "wrong", "more"),
    ("linked", _K_LEFT, "wrong", "left"),
    ("linked", _K_SIZES, "wrong", "sizes"),
    ("linked", _K_HEAD + "    both = pd.merge(released, voters, on=columns, how=\"outer\")\n" + _K_PER
     + "    return (per_visit == 1).sum()\n", "wrong", "left"),
    ("linked", _K_COMBO, "wrong", "combination"),
    ("linked", _K_SHARE, "wrong", "share"),
    ("alone", _A_START, "empty", None),
    ("alone", _A, "correct", None),
    ("alone", _A_HEAD + "    miss = 1 - 1 / values\n    return miss ** (n - 1)\n", "correct", None),
    ("alone", _A_ALL, "wrong", "all"),
    ("alone", _A_SHARED, "wrong", "shared"),
    ("alone", _A_TIMES, "wrong", "times"),
    ("alone", _A_HIT, "wrong", "hit"),
    ("alone", _A_ONE, "wrong", "one"),
    ("alone", _A_COUNT, "wrong", "count"),
    ("alone", _A_ORDER, "wrong", "order"),
    ("alone", _A_HEAD + _A_MISS + "    return miss ** (n + 1)\n", "wrong", None),
]
