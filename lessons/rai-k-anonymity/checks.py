# SPDX-FileCopyrightText: 2026 Everything AI contributors
# SPDX-License-Identifier: CC-BY-SA-4.0 AND MIT-0
"""Checks of the lesson rai-k-anonymity: UCI Adult (PEOPLE, 48,842 people from a US survey of 1994), the same table
with a column band for the ten years each age falls in (BANDED), the five quasi-identifiers of Machanavajjhala et al.
(QI), and a made-up clinic of eight patients before and after coarsening (CLINIC, RELEASED). The three functions the
learner writes: k_of (the size of the smallest group of rows with the same values in some columns), band (the ten
years an age falls in, as a text) and l_of (the smallest number of different values of a secret column in one group);
with the pictures of what the learner's functions give, and the figure of the clinic."""
import html
import re

import pandas as pd

from everything_ai import paths, yours
from everything_ai.check import FunctionBox, Mistake

DATA = paths.ROOT / "data" / "adult.csv.gz"


class NamedList(list):
    """A list the check calls by its name, such as QI, so that a message stays short."""

    def __init__(self, items, name: str):
        super().__init__(items)
        self.name = name

    def __repr__(self):
        return self.name


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


def band(age):
    """The ten years an age falls in, as a text: "30-39" for 37."""
    low = int(age / 10) * 10
    return f"{low}-{low + 9}"


def k_of(df, columns):
    return df.groupby(columns).size().min()


def l_of(df, columns, secret):
    return df.groupby(columns)[secret].nunique().min()


def _people() -> pd.DataFrame:
    return pd.read_csv(DATA)


def _banded() -> pd.DataFrame:
    people = _people()
    people["band"] = [band(age) for age in people["age"]]
    return people


# eight made-up patients of a clinic, and the same rows with the age coarsened to ten years and the ZIP code to four digits
_CLINIC = [(31, "48201", "flu"), (34, "48203", "asthma"), (36, "48201", "flu"), (38, "48207", "broken arm"),
           (52, "48262", "diabetes"), (55, "48265", "diabetes"), (57, "48262", "diabetes"), (59, "48268", "diabetes")]


def _clinic() -> pd.DataFrame:
    return pd.DataFrame(_CLINIC, columns=["age", "zip", "condition"])


def _released() -> pd.DataFrame:
    return pd.DataFrame([(band(a), z[:4] + "*", c) for a, z, c in _CLINIC], columns=["age", "zip", "condition"])


PEOPLE = Table(_people, "PEOPLE")
BANDED = Table(_banded, "BANDED")
CLINIC = Table(_clinic, "CLINIC")
RELEASED = Table(_released, "RELEASED")
QI = NamedList(["age", "sex", "race", "marital_status", "education"], "QI")


# --- the pictures ---------------------------------------------------------------------------------------------------

_UI = "font-family:var(--ea-font-ui);font-size:0.82rem;color:var(--ea-ink-3);margin-top:0.3rem"
_BOX = "padding:0.25rem 0.5rem;margin:0.2rem 0;border-radius:4px;"
_PLAIN = "border:1px solid var(--ea-line);color:var(--ea-ink)"
_WRONG = "border:1.5px dashed var(--ea-loss);color:var(--ea-loss-text)"
_HEAVY = "border:2px solid var(--ea-ink);color:var(--ea-ink);font-weight:600"


def _code(text) -> str:
    return f'<code style="font-family:var(--ea-font-code);font-size:0.8rem">{html.escape(str(text))}</code>'


def _row(cells: list, style: str, widths: list) -> str:
    parts = "".join(f'<div style="flex:{w};min-width:0;overflow-wrap:anywhere">{c}</div>' for c, w in zip(cells, widths))
    return f'<div style="display:flex;gap:0.6rem;align-items:baseline;{_BOX}{style}">{parts}</div>'


def _head(cells: list, widths: list) -> str:
    parts = "".join(f'<div style="flex:{w};min-width:0">{html.escape(c)}</div>' for c, w in zip(cells, widths))
    return (f'<div style="display:flex;gap:0.6rem;font-weight:600;color:var(--ea-ink);padding:0 0.5rem;'
            f'border-bottom:2px solid var(--ea-ink-3);margin-bottom:0.2rem">{parts}</div>')


def _table(head: list, rows: list, widths: list) -> str:
    return (f'<div style="font-family:var(--ea-font-ui);font-size:0.85rem;max-width:46rem">'
            f'{_head(head, widths)}{"".join(rows)}</div>')


def _grouped(n: int, lang: str) -> str:
    """A whole number as the lesson's language writes it: 48,842 in English, 48.842 in Dutch (a dot from five digits)."""
    if lang == "nl":
        return f"{n:,}".replace(",", ".") if n >= 10000 else str(n)
    return f"{n:,}"


def _number(v):
    """A plain whole number of what a function gave back, or None."""
    try:
        if isinstance(v, bool):
            return None
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f == int(f) else None


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


_CLINIC_WORDS = {
    "en": {"title": "Coarsen, and hide in a group",
           "before": "the clinic's own rows", "after": "released: age and ZIP code coarsened",
           "head": ["age", "ZIP code", "condition"],
           "key": "Eight made-up patients. Before: every combination of age and ZIP code occurs once, so k = 1. Released: "
                  "each age replaced by its ten years and each ZIP code cut to four digits. Now there are two groups "
                  "of four rows with the same age and ZIP code, so the table is 4-anonymous on those two columns. In "
                  "the upper group three different conditions occur; in the lower group all four rows say diabetes."},
    "nl": {"title": "Grover maken, en schuilen in een groep",
           "before": "de eigen rijen van de kliniek", "after": "vrijgegeven: leeftijd en postcode grover",
           "head": ["leeftijd", "postcode", "aandoening"],
           "key": "Acht verzonnen patiënten. Vóór: elke combinatie van leeftijd en postcode komt één keer voor, dus "
                  "k = 1. Vrijgegeven: elke leeftijd vervangen door de tien jaar waarin hij valt, en elke postcode "
                  "ingekort tot vier cijfers. Nu zijn er twee groepen van vier rijen met dezelfde leeftijd en "
                  "postcode, dus de tabel is 4-anoniem op die twee kolommen. In de bovenste groep komen drie "
                  "verschillende aandoeningen voor; in de onderste groep staat in alle vier de rijen diabetes."},
}


def clinic_picture(lang: str) -> str:
    """The figure of The idea: the made-up clinic before and after coarsening, the two groups framed."""
    w = _CLINIC_WORDS[lang]
    widths = [2, 3, 4]
    before = _table(w["head"], [_row([str(a), _code(z), html.escape(c)], _PLAIN, widths) for a, z, c in _CLINIC],
                    widths)
    released = _released()
    groups = []
    for start in (0, 4):
        part = released.iloc[start:start + 4]
        same = part["condition"].nunique() == 1
        rows = "".join(_row([html.escape(r.age), _code(r.zip),
                             f'<span style="font-weight:600">{html.escape(r.condition)}</span>' if same
                             else html.escape(r.condition)], "border:1px solid transparent;color:var(--ea-ink)", widths)
                       for r in part.itertuples())
        groups.append(f'<div style="{_BOX}{_HEAVY if same else _PLAIN};font-weight:400;padding:0.1rem">{rows}</div>')
    after = (f'<div style="font-family:var(--ea-font-ui);font-size:0.85rem;max-width:46rem">'
             f'{_head(w["head"], widths)}{"".join(groups)}</div>')
    cols = (f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));gap:0.8rem;'
            f'font-family:var(--ea-font-ui);font-size:0.83rem;max-width:46rem;color:var(--ea-ink)">'
            f'<div><div style="font-weight:600">{html.escape(w["before"])}</div>{before}</div>'
            f'<div><div style="font-weight:600">{html.escape(w["after"])} →</div>{after}</div></div>')
    return yours.figure(w["title"], cols + f'<div style="{_UI}">{html.escape(w["key"])}</div>')


# the column sets the picture of k_of goes through, one column more each time
_SETS = [["sex"], ["sex", "race"], ["sex", "race", "marital_status"], ["sex", "race", "marital_status", "education"],
         ["sex", "race", "marital_status", "education", "age"]]

_K_WORDS = {
    "en": {"title": "Your k_of on PEOPLE, one column more each time",
           "head": ["columns", "groups", "people alone", "your k_of"],
           "key": "For each set of columns: how many groups of people with the same values there are, how many people "
                  "are the only one in their group, and what your k_of gives. The last set holds the five columns of "
                  "QI.",
           "said": "With all five, your k_of gives {k}.",
           "wrong": "A dashed frame marks where it differs from the course's k_of."},
    "nl": {"title": "Jouw k_of op PEOPLE, elke keer één kolom meer",
           "head": ["kolommen", "groepen", "mensen alleen", "jouw k_of"],
           "key": "Voor elke set kolommen: hoeveel groepen mensen met dezelfde waarden er zijn, hoeveel mensen als "
                  "enige in hun groep zitten, en wat jouw k_of geeft. De laatste set bevat de vijf kolommen van QI.",
           "said": "Met alle vijf geeft jouw k_of {k}.",
           "wrong": "Een gestippeld kader wijst aan waar hij verschilt van de k_of van de course."},
}


def k_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _K_WORDS[seen.lang]
    people = PEOPLE.df
    rows, wrong_any, last = [], False, None
    for columns in _SETS:
        try:
            got = _call(fn, PEOPLE, list(columns))
        except Exception:
            got = None
        sizes = people.groupby(columns).size()
        want = int(sizes.min())
        n = _number(got)
        wrong = n != want
        wrong_any |= wrong
        shown = _grouped(n, seen.lang) if n is not None else html.escape(repr(got)[:40])
        rows.append(_row([_code(", ".join(columns)), _grouped(len(sizes), seen.lang),
                          _grouped(int((sizes == 1).sum()), seen.lang), shown],
                         _WRONG if wrong else _PLAIN, [6, 2, 2, 2]))
        last = shown
    note = w["key"] + " " + w["said"].format(k=html.unescape(last)) + (" " + w["wrong"] if wrong_any else "")
    return yours.figure(w["title"], _table(w["head"], rows, [6, 2, 2, 2]) + f'<div style="{_UI}">{html.escape(note)}</div>')


_BAND_WORDS = {
    "en": {"title": "Your band on the ages in PEOPLE",
           "head": ["your band", "ages", "people"],
           "key": "The ages in PEOPLE run from 17 to 90. Each row is one band your function gives, with the ages it "
                  "puts there and how many people have those ages.",
           "k": "Grouped by age and sex, the smallest group has {a} {pa}. Grouped by your band and sex, it has {b} "
                "{pb}.",
           "person": "person", "people": "people",
           "wrong": "A dashed frame marks a band where your function gives something other than the course's band "
                    "for those ages."},
    "nl": {"title": "Jouw band op de leeftijden in PEOPLE",
           "head": ["jouw band", "leeftijden", "mensen"],
           "key": "De leeftijden in PEOPLE lopen van 17 tot 90. Elke rij is één band die jouw functie geeft, met de "
                  "leeftijden die hij daarin zet en hoeveel mensen die leeftijden hebben.",
           "k": "Gegroepeerd op leeftijd en geslacht heeft de kleinste groep {a} {pa}. Gegroepeerd op jouw band en "
                "geslacht heeft hij er {b}.",
           "person": "persoon", "people": "mensen",
           "wrong": "Een gestippeld kader wijst een band aan waar jouw functie voor die leeftijden iets anders geeft "
                    "dan de band van de course."},
}


def band_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _BAND_WORDS[seen.lang]
    people = PEOPLE.df
    counts = people["age"].value_counts()
    labels = {}
    for age in sorted(counts.index):
        try:
            got = _call(fn, int(age))
            hash(got)
        except Exception:
            return ""
        labels[int(age)] = got
    order, ages = [], {}
    for age, label in labels.items():
        if label not in ages:
            order.append(label)
            ages[label] = []
        ages[label].append(age)
    rows, wrong_any = [], False
    for label in order:
        mine = ages[label]
        wrong = any(band(a) != label for a in mine)
        wrong_any |= wrong
        span = f"{mine[0]}" if len(mine) == 1 else f"{mine[0]}–{mine[-1]}"
        if mine != list(range(mine[0], mine[-1] + 1)):
            span = ", ".join(str(a) for a in mine[:6]) + ("…" if len(mine) > 6 else "")
        rows.append(_row([_code(repr(label) if not isinstance(label, str) else label), span,
                          _grouped(int(sum(counts[a] for a in mine)), seen.lang)],
                         _WRONG if wrong else _PLAIN, [3, 3, 2]))
    mapped = pd.DataFrame({"b": people["age"].map(labels), "sex": people["sex"]})
    a = int(people.groupby(["age", "sex"]).size().min())
    b = int(mapped.groupby(["b", "sex"]).size().min())
    note = (w["key"] + " " + w["k"].format(a=a, pa=w["person"] if a == 1 else w["people"], b=b,
                                           pb=w["person"] if b == 1 else w["people"])
            + (" " + w["wrong"] if wrong_any else ""))
    return yours.figure(w["title"], _table(w["head"], rows, [3, 3, 2]) + f'<div style="{_UI}">{html.escape(note)}</div>')


_L_WORDS = {
    "en": {"title": "Your l_of on each group of BANDED",
           "head": ["band", "women", "your l_of", "men", "your l_of"],
           "key": "For each band and sex: how many people the group has, and what your l_of gives for that group "
                  "alone, with income as the secret. A heavy frame marks a group where it gives 1.",
           "said": "On the whole of BANDED, your l_of gives {l}.",
           "one": "The table is 17-anonymous on band and sex: every group has at least 17 people. Yet the 40 women of "
                  "80-89 all earn at most 50,000 dollars, so whoever knows that a woman of that age is in the table "
                  "knows that, too. And among the 1,236 women of 10-19 both values occur, but 1,234 of them earn at "
                  "most 50,000.",
           "wrong": "A dashed frame marks where it differs from the course's l_of.",
           "whole": "Each group alone is right, but on the whole of BANDED the course's l_of gives 1: the smallest "
                    "number in the table."},
    "nl": {"title": "Jouw l_of op elke groep van BANDED",
           "head": ["band", "vrouwen", "jouw l_of", "mannen", "jouw l_of"],
           "key": "Voor elke band en elk geslacht: hoeveel mensen de groep heeft, en wat jouw l_of geeft voor alleen "
                  "die groep, met het inkomen als geheim. Een dik kader wijst een groep aan waar hij 1 geeft.",
           "said": "Op heel BANDED geeft jouw l_of {l}.",
           "one": "De tabel is 17-anoniem op band en geslacht: elke groep heeft minstens 17 mensen. Toch verdienen de "
                  "40 vrouwen van 80-89 allemaal hoogstens 50.000 dollar, dus wie weet dat een vrouw van die leeftijd "
                  "in de tabel staat, weet dat ook. En bij de 1236 vrouwen van 10-19 komen beide waarden voor, maar "
                  "1234 van hen verdienen hoogstens 50.000.",
           "wrong": "Een gestippeld kader wijst aan waar hij verschilt van de l_of van de course.",
           "whole": "Elke groep apart klopt, maar op heel BANDED geeft de l_of van de course 1: het kleinste getal "
                    "in de tabel."},
}


def l_picture(seen) -> str:
    fn = _fn(seen)
    if fn is None:
        return ""
    w = _L_WORDS[seen.lang]
    df = BANDED.df
    try:
        whole = _call(fn, BANDED, ["band", "sex"], "income")
    except Exception:
        return ""
    cells, wrong_any = {}, False
    for (b, sex), part in df.groupby(["band", "sex"]):
        try:
            got = _number(_call(fn, Table(lambda p=part: p, "PART"), ["band", "sex"], "income"))
        except Exception:
            got = None
        want = int(part["income"].nunique())
        cells[(b, sex)] = (len(part), got, got != want)
        wrong_any |= got != want
    rows = []
    for b in sorted({b for b, _ in cells}):
        parts, styles = [html.escape(b)], []
        for sex in ("Female", "Male"):
            n, got, wrong = cells.get((b, sex), (0, None, False))
            mark = (f'<span style="display:inline-block;padding:0 0.3rem;border-radius:3px;{_HEAVY}">{got}</span>'
                    if got == 1 else "–" if got is None else _grouped(got, seen.lang))
            parts += [_grouped(n, seen.lang), mark]
            styles.append(wrong)
        rows.append(_row(parts, _WRONG if any(styles) else _PLAIN, [2, 2, 2, 2, 2]))
    whole_n = _number(whole)
    note = w["key"] + " " + w["said"].format(l=whole_n if whole_n is not None else repr(whole)[:40])
    if not wrong_any and whole_n == 1:
        note += " " + w["one"]
    if wrong_any:
        note += " " + w["wrong"]
    elif whole_n != 1:
        note += " " + w["whole"]
    return yours.figure(w["title"], _table(w["head"], rows, [2, 2, 2, 2, 2]) + f'<div style="{_UI}">{html.escape(note)}</div>')


# --- the exercises ----------------------------------------------------------------------------------------------------

REQUIRED = ["k_of", "band", "l_of", "bigger-k", "skewness"]

_DOTS = Mistake("dots-left", matches=lambda got, text: re.search(r"^[^#\n]*\.\.\.", text, re.MULTILINE) is not None,
                en="There is still a `...` in your code. Replace it with your own code.",
                nl="Er staat nog een `...` in je code. Vervang die door je eigen code.")

_SAME_TEXTS = {
    "missing": {"en": "I cannot see the box. Run the cell with the box again.",
                "nl": "Ik zie het vak niet. Voer de cel met het vak opnieuw uit."},
    "cleared": {"en": "There is no code in the box any more. Undo with Ctrl+Z (on a Mac: Cmd+Z), or open the lesson again: an empty box then starts with the course's lines again.",
                "nl": "Er staat geen code meer in het vak. Maak het ongedaan met Ctrl+Z (op een Mac: Cmd+Z), of open de les opnieuw: een leeg vak begint dan weer met de regels van de course."},
}

# exercise 1: k_of
_K_HEAD = ("def k_of(df, columns):\n"
           "    # the number of rows in the smallest group of rows with the same values in columns\n")
_K_START = _K_HEAD + "    return ...\n"
_K = _K_HEAD + "    return df.groupby(columns).size().min()\n"
_K_MAX = _K_HEAD + "    return df.groupby(columns).size().max()\n"
_K_GROUPS = _K_HEAD + "    return len(df.groupby(columns).size())\n"
_K_SIZES = _K_HEAD + "    return df.groupby(columns).size()\n"
_K_COUNT = _K_HEAD + "    return df.groupby(columns).count().min()\n"

# exercise 2: band
_B_HEAD = ("def band(age):\n"
           "    # the ten years an age falls in, as a text such as \"30-39\" for 37\n")
_B_START = _B_HEAD + "    low = ...\n    return ...\n"
_B = _B_HEAD + "    low = int(age / 10) * 10\n    return f\"{low}-{low + 9}\"\n"
_B_ROUND = _B_HEAD + "    low = round(age / 10) * 10\n    return f\"{low}-{low + 9}\"\n"
_B_OVERLAP = _B_HEAD + "    low = int(age / 10) * 10\n    return f\"{low}-{low + 10}\"\n"
_B_NUMBER = _B_HEAD + "    low = int(age / 10) * 10\n    return low\n"
_B_TENS = _B_HEAD + "    low = int(age / 10)\n    return f\"{low}-{low + 9}\"\n"
_B_SPACES = _B_HEAD + "    low = int(age / 10) * 10\n    return f\"{low} - {low + 9}\"\n"

# exercise 3: l_of
_L_HEAD = ("def l_of(df, columns, secret):\n"
           "    # the smallest number of different values of the column secret in one group\n")
_L_START = _L_HEAD + "    return ...\n"
_L = _L_HEAD + "    return df.groupby(columns)[secret].nunique().min()\n"
_L_MAX = _L_HEAD + "    return df.groupby(columns)[secret].nunique().max()\n"
_L_SIZE = _L_HEAD + "    return df.groupby(columns).size().min()\n"
_L_COUNT = _L_HEAD + "    return df.groupby(columns)[secret].count().min()\n"
_L_WHOLE = _L_HEAD + "    return df[secret].nunique()\n"
_L_ALL = _L_HEAD + "    return df.groupby(columns)[secret].nunique()\n"

# a small table with a missing secret, so that counting rows and counting different values differ
_GAPS = Table(lambda: pd.DataFrame({"group": ["a", "a", "a", "b", "b", "c", "c"],
                                    "secret": ["x", None, "z", "y", "y", "x", "w"]}), "GAPS")

EXERCISES = {
    "k_of": FunctionBox(
        name="k_of",
        picture=k_picture,
        function="k_of",
        start=_K_START,
        solution=_K,
        visible=[(PEOPLE, QI)],
        hidden=[(PEOPLE, ["sex"]), (PEOPLE, ["sex", "race"]), (CLINIC, ["age", "zip"]), (RELEASED, ["age", "zip"]),
                (RELEASED, ["zip"]), (RELEASED, ["condition"])],
        first=[_DOTS],
        mistakes=[
            Mistake("max", wrong=_K_MAX,
                    en="`{call}` gives {got}, and it should give {want}. That is the size of the largest group. k is the number of rows in the smallest group: `.min()`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Dat is de grootte van de grootste groep. k is het aantal rijen in de kleinste groep: `.min()`."),
            Mistake("groups", wrong=_K_GROUPS,
                    en="`{call}` gives {got}, and it should give {want}. That is how many groups there are. k is how many rows the smallest group has.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Dat is hoeveel groepen er zijn. k is hoeveel rijen de kleinste groep heeft."),
            Mistake("sizes", wrong=_K_SIZES,
                    en="`{call}` gives the size of every group, one number per group. k is one number: the smallest of them, with `.min()`.",
                    nl="`{call}` geeft de grootte van elke groep, één getal per groep. k is één getal: de kleinste ervan, met `.min()`."),
            Mistake("count", wrong=_K_COUNT,
                    en="`{call}` gives one number for each of the other columns: `count` counts per column the values that are not missing. `.size()` counts the rows of each group.",
                    nl="`{call}` geeft één getal voor elk van de andere kolommen: `count` telt per kolom de waarden die niet ontbreken. `.size()` telt de rijen van elke groep."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with one expression: group `df` by `columns`, count the rows of each group, and take the smallest count.",
                      "nl": "Vervang de `...` door één expressie: groepeer `df` op `columns`, tel de rijen van elke groep, en neem de kleinste telling."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Group the rows by `columns`, count the rows of each group, and give back the smallest count.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Groepeer de rijen op `columns`, tel de rijen van elke groep, en geef de kleinste telling terug."},
            "almost": {"en": "Right for `k_of(PEOPLE, QI)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `df` and the `columns` it gets?",
                       "nl": "Goed voor `k_of(PEOPLE, QI)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `df` en de `columns` die hij krijgt?"},
            "correct": {"en": "Your `k_of` is right in all {n} cases. The picture adds one column at a time.",
                        "nl": "Je `k_of` klopt in alle {n} gevallen. Het plaatje voegt één kolom tegelijk toe."},
        },
    ),
    "band": FunctionBox(
        name="band",
        picture=band_picture,
        function="band",
        start=_B_START,
        solution=_B,
        visible=[(37,)],
        hidden=[(30,), (39,), (17,), (84,), (90,), (5,), (100,)],
        first=[_DOTS],
        mistakes=[
            Mistake("round", wrong=_B_ROUND,
                    en="`{call}` gives {got}, and it should give {want}. `round` rounds to the nearest whole number, so 37 lands in the band of 40. A band starts at the age rounded down: `int(...)` drops what comes after the point.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. `round` rondt af op het dichtstbijzijnde gehele getal, dus 37 komt in de band van 40. Een band begint bij de leeftijd naar beneden afgerond: `int(...)` laat weg wat na de punt komt."),
            Mistake("overlap", wrong=_B_OVERLAP,
                    en="`{call}` gives {got}, and it should give {want}. Your band ends where the next one starts, so 40 would be in two bands. A band of ten years ends at `low + 9`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je band eindigt waar de volgende begint, dus 40 zou in twee banden zitten. Een band van tien jaar eindigt bij `low + 9`."),
            Mistake("number", wrong=_B_NUMBER,
                    en="`{call}` gives the number {got}, the lowest age of the band. The task asks for the band as a text, from the lowest to the highest age: {want}.",
                    nl="`{call}` geeft het getal {got}, de laagste leeftijd van de band. De opdracht vraagt om de band als tekst, van de laagste tot de hoogste leeftijd: {want}."),
            Mistake("tens", wrong=_B_TENS,
                    en="`{call}` gives {got}, and it should give {want}. Your `low` counts the tens: 3 for 37. Multiply by 10 again to get the lowest age of the band.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je `low` telt de tientallen: 3 voor 37. Vermenigvuldig weer met 10 om de laagste leeftijd van de band te krijgen."),
            Mistake("spaces", wrong=_B_SPACES,
                    en="`{call}` gives {got}, and it should give {want}. Write the band without spaces around the hyphen, so that every band looks the same in a table.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Schrijf de band zonder spaties rond het streepje, zodat elke band er in een tabel hetzelfde uitziet."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the first `...` with the lowest age of the band, and the second with the band as a text, such as `\"30-39\"`.",
                      "nl": "Vervang de eerste `...` door de laagste leeftijd van de band, en de tweede door de band als tekst, zoals `\"30-39\"`."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. The band starts at the age rounded down to a ten, and ends nine years later.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. De band begint bij de leeftijd naar beneden afgerond op een tiental, en eindigt negen jaar later."},
            "almost": {"en": "Right for `band(37)`. But `{call}` gives {got}, and it should give {want}. Does your function use the `age` it gets?",
                       "nl": "Goed voor `band(37)`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `age` die hij krijgt?"},
            "correct": {"en": "Your `band` is right in all {n} cases. The picture puts every age in PEOPLE in its band.",
                        "nl": "Je `band` klopt in alle {n} gevallen. Het plaatje zet elke leeftijd in PEOPLE in zijn band."},
        },
    ),
    "l_of": FunctionBox(
        name="l_of",
        picture=l_picture,
        function="l_of",
        start=_L_START,
        solution=_L,
        visible=[(BANDED, ["band", "sex"], "income")],
        hidden=[(PEOPLE, ["sex", "race"], "income"), (PEOPLE, ["sex"], "occupation"), (RELEASED, ["age", "zip"], "condition"),
                (CLINIC, ["age"], "condition"), (_GAPS, ["group"], "secret")],
        first=[_DOTS],
        mistakes=[
            Mistake("max", wrong=_L_MAX,
                    en="`{call}` gives {got}, and it should give {want}. That is the number of different values in the group with the most. l is the smallest number of different values in one group: `.min()`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Dat is het aantal verschillende waarden in de groep met de meeste. l is het kleinste aantal verschillende waarden in één groep: `.min()`."),
            Mistake("size", wrong=_L_SIZE,
                    en="`{call}` gives {got}, and it should give {want}. That is k: the number of rows in the smallest group. l counts the different values of the column `secret` in each group: `[secret].nunique()`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Dat is k: het aantal rijen in de kleinste groep. l telt de verschillende waarden van de kolom `secret` in elke groep: `[secret].nunique()`."),
            Mistake("count", wrong=_L_COUNT,
                    en="`{call}` gives {got}, and it should give {want}. `count` counts the rows that have a value, also when two rows have the same value. l counts the different values: `nunique`.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. `count` telt de rijen die een waarde hebben, ook als twee rijen dezelfde waarde hebben. l telt de verschillende waarden: `nunique`."),
            Mistake("whole", wrong=_L_WHOLE,
                    en="`{call}` gives {got}, and it should give {want}. Your function counts the different values in the whole table. l looks at each group apart: group by `columns` first.",
                    nl="`{call}` geeft {got}, en dat moet {want} zijn. Je functie telt de verschillende waarden in de hele tabel. l kijkt naar elke groep apart: groepeer eerst op `columns`."),
            Mistake("all", wrong=_L_ALL,
                    en="`{call}` gives the number of different values of every group, one number per group. l is one number: the smallest of them, with `.min()`.",
                    nl="`{call}` geeft het aantal verschillende waarden van elke groep, één getal per groep. l is één getal: het kleinste ervan, met `.min()`."),
        ],
        texts=_SAME_TEXTS | {
            "empty": {"en": "Replace the `...` with one expression: group `df` by `columns`, take the column `secret`, count its different values in each group, and take the smallest.",
                      "nl": "Vervang de `...` door één expressie: groepeer `df` op `columns`, neem de kolom `secret`, tel in elke groep de verschillende waarden, en neem de kleinste."},
            "wrong": {"en": "`{call}` gives {got}, and it should give {want}. Group the rows by `columns`, count the different values of `secret` in each group, and give back the smallest count.",
                      "nl": "`{call}` geeft {got}, en dat moet {want} zijn. Groepeer de rijen op `columns`, tel in elke groep de verschillende waarden van `secret`, en geef de kleinste telling terug."},
            "almost": {"en": "Right for `l_of(BANDED, [\"band\", \"sex\"], \"income\")`. But `{call}` gives {got}, and it should give {want}. Does your function use the `df`, the `columns` and the `secret` it gets?",
                       "nl": "Goed voor `l_of(BANDED, [\"band\", \"sex\"], \"income\")`. Maar `{call}` geeft {got}, en dat moet {want} zijn. Gebruikt je functie de `df`, de `columns` en de `secret` die hij krijgt?"},
            "correct": {"en": "Your `l_of` is right in all {n} cases. The picture runs it on each group of BANDED apart.",
                        "nl": "Je `l_of` klopt in alle {n} gevallen. Het plaatje laat hem op elke groep van BANDED apart lopen."},
        },
    ),
}

CASES = [
    ("k_of", _K_START, "empty", None),
    ("k_of", _K, "correct", None),
    ("k_of", _K_HEAD + "    return df.value_counts(columns).min()\n", "correct", None),
    ("k_of", _K_HEAD + "    sizes = df.groupby(columns).size()\n    return min(sizes)\n", "correct", None),
    ("k_of", _K_MAX, "wrong", "max"),
    ("k_of", _K_GROUPS, "wrong", "groups"),
    ("k_of", _K_SIZES, "wrong", "sizes"),
    ("k_of", _K_COUNT, "wrong", "count"),
    ("k_of", _K_HEAD + "    return df.groupby(columns).size().mean()\n", "wrong", None),
    ("band", _B_START, "empty", None),
    ("band", _B_HEAD + "    low = int(age / 10) * 10\n    return ...\n", "wrong", "dots-left"),
    ("band", _B, "correct", None),
    ("band", _B_HEAD + "    low = age - age % 10\n    return str(low) + \"-\" + str(low + 9)\n", "correct", None),
    ("band", _B_ROUND, "wrong", "round"),
    ("band", _B_OVERLAP, "wrong", "overlap"),
    ("band", _B_NUMBER, "wrong", "number"),
    ("band", _B_TENS, "wrong", "tens"),
    ("band", _B_SPACES, "wrong", "spaces"),
    ("band", _B_HEAD + "    low = int(age / 10) * 10\n    return f\"{low}-{low + 19}\"\n", "wrong", None),
    ("l_of", _L_START, "empty", None),
    ("l_of", _L, "correct", None),
    ("l_of", _L_HEAD + "    different = df.groupby(columns)[secret].nunique()\n    return different.min()\n", "correct", None),
    ("l_of", _L_MAX, "wrong", "max"),
    ("l_of", _L_SIZE, "wrong", "size"),
    ("l_of", _L_COUNT, "wrong", "count"),
    ("l_of", _L_WHOLE, "wrong", "whole"),
    ("l_of", _L_ALL, "wrong", "all"),
    ("l_of", _L_HEAD + "    return df.groupby(columns)[secret].nunique().mean()\n", "wrong", None),
]
